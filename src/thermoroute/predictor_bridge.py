"""Pre-opening bridge audit for legacy and currently retrievable meteorology.

The canonical 2006--2020 panel no longer has its original HTTP responses.  This
module therefore cannot reconstruct that missing provenance.  It does provide a
strict, outcome-free compatibility gate: re-fetch 2018--2020 Daymet/gridMET with
the confirmation parser, archive those bytes, and compare the resulting predictor
values with the frozen panel on the exact site/date registry.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .historical_inputs import (
    DAYMET_FIELDS,
    DAYMET_PROVIDER,
    GRIDMET_PROVIDER,
    GRIDMET_SCHEMA_PROVIDER,
    PRELABEL_FIELDS,
    USER_AGENT,
    _acquire_cohort,
    _assert_safe_meteorology_url,
    _binding,
    _exclusive_create,
    _write_parquet,
    freeze_snapshot_index,
    load_coordinate_registry,
)
from .provenance import SnapshotStore, canonical_json_bytes
from .repro import source_tree_hash
from .usgs import (
    build_daymet_url,
    build_gridmet_wind_metadata_url,
    build_gridmet_wind_url,
    parse_gridmet_wind_metadata,
)


BRIDGE_FORMAT = "thermoroute.development-predictor-bridge.v1"
BRIDGE_START = pd.Timestamp("2018-01-01")
BRIDGE_END = pd.Timestamp("2020-12-31")
BRIDGE_FIELDS = tuple(PRELABEL_FIELDS)
_VALUE_ATOL = {
    "TEMP": 1e-9,
    "PRCP": 1e-9,
    "RHMEAN": 1e-9,
    "DH": 1e-9,
    "WDSP": 1e-9,
}


class PredictorBridgeError(RuntimeError):
    """The development-to-confirmation predictor bridge is not auditable."""


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _normalise_table(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {"site_no", "DATE", *BRIDGE_FIELDS}
    missing = required - set(frame)
    if missing:
        raise PredictorBridgeError(f"{label} lacks columns: {sorted(missing)}")
    value = frame[["site_no", "DATE", *BRIDGE_FIELDS]].copy()
    value["site_no"] = value.site_no.astype("string").str.strip()
    value["DATE"] = pd.to_datetime(value.DATE, errors="coerce").dt.normalize()
    if value.site_no.isna().any() or value.site_no.eq("").any():
        raise PredictorBridgeError(f"{label} contains an empty site identity")
    if value.DATE.isna().any() or value.duplicated(["site_no", "DATE"]).any():
        raise PredictorBridgeError(f"{label} has invalid or duplicate daily keys")
    for field in BRIDGE_FIELDS:
        value[field] = pd.to_numeric(value[field], errors="coerce")
        if np.isinf(value[field].to_numpy(float)).any():
            raise PredictorBridgeError(f"{label}/{field} contains infinity")
    return value.sort_values(["site_no", "DATE"]).reset_index(drop=True)


def frozen_bridge_slice(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    start: pd.Timestamp = BRIDGE_START,
    end: pd.Timestamp = BRIDGE_END,
) -> pd.DataFrame:
    """Map legacy panel IDs to stable site numbers without reading new outcomes."""
    required_registry = {"site_no", "legacy_site_id"}
    if required_registry - set(registry):
        raise PredictorBridgeError("station registry lacks stable/legacy identity map")
    mapping = registry[["site_no", "legacy_site_id"]].copy()
    mapping["site_no"] = mapping.site_no.astype("string").str.strip()
    mapping["legacy_site_id"] = mapping.legacy_site_id.astype("string").str.strip()
    if (
        mapping.site_no.eq("").any()
        or mapping.legacy_site_id.eq("").any()
        or mapping.site_no.duplicated().any()
        or mapping.legacy_site_id.duplicated().any()
    ):
        raise PredictorBridgeError("station registry identity map is not one-to-one")
    if {"site_id", "DATE", *BRIDGE_FIELDS} - set(panel):
        raise PredictorBridgeError("frozen panel lacks bridge fields")
    selected = panel.loc[
        pd.to_datetime(panel.DATE).between(start, end),
        ["site_id", "DATE", *BRIDGE_FIELDS],
    ].copy()
    selected["site_id"] = selected.site_id.astype("string")
    selected = selected.merge(
        mapping.rename(columns={"legacy_site_id": "site_id"}),
        on="site_id",
        how="left",
        validate="many_to_one",
    )
    if selected.site_no.isna().any():
        raise PredictorBridgeError("frozen panel contains an unmapped legacy site")
    return _normalise_table(selected.drop(columns="site_id"), label="frozen panel")


def _shift_metric(
    frozen: pd.DataFrame,
    refreshed: pd.DataFrame,
    *,
    field: str,
    shift_days: int,
) -> dict[str, Any]:
    candidate = refreshed[["site_no", "DATE", field]].copy()
    candidate["DATE"] += pd.Timedelta(days=int(shift_days))
    paired = frozen[["site_no", "DATE", field]].merge(
        candidate,
        on=["site_no", "DATE"],
        suffixes=("_frozen", "_refreshed"),
        how="inner",
        validate="one_to_one",
    ).dropna()
    left = paired[f"{field}_frozen"].to_numpy(float)
    right = paired[f"{field}_refreshed"].to_numpy(float)
    difference = right - left
    correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if len(paired) > 1 and np.std(left) > 0 and np.std(right) > 0
        else np.nan
    )
    return {
        "refreshed_date_shift_days": int(shift_days),
        "paired_finite_count": int(len(paired)),
        "rmse": _finite_or_none(float(np.sqrt(np.mean(difference ** 2))))
        if len(paired) else None,
        "correlation": _finite_or_none(correlation),
    }


def compare_predictor_bridge(
    frozen: pd.DataFrame,
    refreshed: pd.DataFrame,
    *,
    expected_site_count: int = 120,
    start: pd.Timestamp = BRIDGE_START,
    end: pd.Timestamp = BRIDGE_END,
) -> dict[str, Any]:
    """Compare exact products and return a deterministic fail-closed report."""
    frozen = _normalise_table(frozen, label="frozen panel")
    refreshed = _normalise_table(refreshed, label="refreshed predictors")
    expected_dates = pd.date_range(start, end, freq="D")
    expected_rows = int(expected_site_count) * len(expected_dates)
    for label, frame in (("frozen", frozen), ("refreshed", refreshed)):
        if (
            len(frame) != expected_rows
            or frame.site_no.nunique() != int(expected_site_count)
            or frame.groupby("site_no").DATE.nunique().ne(len(expected_dates)).any()
        ):
            raise PredictorBridgeError(f"{label} bridge key registry is incomplete")
    frozen_keys = frozen[["site_no", "DATE"]]
    refreshed_keys = refreshed[["site_no", "DATE"]]
    if not frozen_keys.equals(refreshed_keys):
        raise PredictorBridgeError("frozen and refreshed predictor keys differ")

    metrics: dict[str, Any] = {}
    failures: list[str] = []
    for field in BRIDGE_FIELDS:
        left = frozen[field].to_numpy(float)
        right = refreshed[field].to_numpy(float)
        left_missing = np.isnan(left)
        right_missing = np.isnan(right)
        paired = ~(left_missing | right_missing)
        difference = right[paired] - left[paired]
        max_abs = float(np.max(np.abs(difference))) if paired.any() else np.inf
        exact_missing = bool(np.array_equal(left_missing, right_missing))
        within = bool(
            paired.any()
            and np.all(np.abs(difference) <= _VALUE_ATOL[field])
            and exact_missing
        )
        shifts = [
            _shift_metric(frozen, refreshed, field=field, shift_days=shift)
            for shift in (-1, 0, 1)
        ]
        zero_rmse = shifts[1]["rmse"]
        alternative = [row["rmse"] for row in (shifts[0], shifts[2])]
        zero_best = bool(
            zero_rmse is not None
            and all(value is None or zero_rmse <= value for value in alternative)
        )
        if not within:
            failures.append(f"{field}: refreshed bytes do not reproduce frozen values")
        if not zero_best:
            failures.append(f"{field}: zero-day alignment is not best among -1/0/+1")
        metrics[field] = {
            "paired_finite_count": int(paired.sum()),
            "frozen_missing_count": int(left_missing.sum()),
            "refreshed_missing_count": int(right_missing.sum()),
            "missing_pattern_exact": exact_missing,
            "value_atol": _VALUE_ATOL[field],
            "max_abs_difference": _finite_or_none(max_abs),
            "bias_refreshed_minus_frozen": _finite_or_none(
                float(np.mean(difference)) if paired.any() else np.nan
            ),
            "rmse": _finite_or_none(
                float(np.sqrt(np.mean(difference ** 2))) if paired.any() else np.nan
            ),
            "exact_product_compatibility": within,
            "date_shift_sensitivity": shifts,
            "zero_day_alignment_best_or_tied": zero_best,
        }

    leap_dates = [
        date.strftime("%Y-%m-%d")
        for date in expected_dates
        if date.month == 12 and date.day == 31 and date.is_leap_year
    ]
    leap_missing = {
        field: int(
            refreshed.loc[refreshed.DATE.isin(pd.to_datetime(leap_dates)), field]
            .isna().sum()
        )
        for field in DAYMET_FIELDS
    }
    return {
        "format": BRIDGE_FORMAT,
        "status": "PASS_EXACT_PRODUCT_BRIDGE" if not failures else "NO_GO_PRODUCT_BRIDGE_MISMATCH",
        "outcome_values_requested_or_read": False,
        "interval": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
        "site_count": int(expected_site_count),
        "row_count": int(expected_rows),
        "fields": metrics,
        "daymet_calendar_attestation": {
            "provider_calendar": "365-day; leap years retain February 29 and omit December 31",
            "leap_year_omitted_dates_in_interval": leap_dates,
            "refreshed_missing_count_on_those_dates": leap_missing,
            "policy": "retain explicit missingness; do not fabricate provider values",
        },
        "interpretation_limit": (
            "This gate tests product/parser compatibility and +/-1-day value alignment. "
            "It does not prove subdaily local-day equivalence with NWIS or as-issued availability."
        ),
        "failures": failures,
    }


def _prefetch_raw_predictors(
    registry,
    *,
    daymet_store: SnapshotStore,
    gridmet_store: SnapshotStore,
    retries: int,
    workers: int,
) -> tuple[set[str], set[str]]:
    """Populate independent raw requests concurrently; publish no parsed values."""
    headers = {"User-Agent": USER_AGENT}
    jobs: list[tuple[str, str, str]] = []
    for station in registry.frame.itertuples(index=False):
        site_no = str(station.site_no)
        lat, lon = float(station.lat), float(station.lon)
        jobs.extend((
            (
                "daymet",
                site_no,
                build_daymet_url(lat, lon, "2018-01-01", "2020-12-31"),
            ),
            (
                "gridmet",
                site_no,
                build_gridmet_wind_url(lat, lon, "2018-01-01", "2020-12-31"),
            ),
        ))
    for _provider, _site, url in jobs:
        _assert_safe_meteorology_url(url)

    def fetch(job: tuple[str, str, str]) -> tuple[str, str]:
        provider, _site, url = job
        store = daymet_store if provider == "daymet" else gridmet_store
        provider_name = DAYMET_PROVIDER if provider == "daymet" else GRIDMET_PROVIDER
        _payload, record = store.fetch(
            provider=provider_name,
            url=url,
            headers=headers,
            retries=max(1, int(retries)),
        )
        return provider, str(record.request_sha256)

    requests: dict[str, set[str]] = {"daymet": set(), "gridmet": set()}
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as executor:
        futures = [executor.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            provider, request_sha256 = future.result()
            requests[provider].add(request_sha256)
    if any(len(requests[provider]) != registry.row_count for provider in requests):
        raise PredictorBridgeError("raw bridge prefetch request registry is incomplete")
    return requests["daymet"], requests["gridmet"]


def acquire_predictor_bridge(
    *,
    repo_root: str | Path,
    panel_path: str | Path,
    registry_path: str | Path,
    snapshot_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    offline: bool = False,
    retries: int = 3,
    request_interval: float = 0.1,
    expected_sites: int = 120,
    prefetch_only: bool = False,
    prefetch_workers: int = 4,
) -> dict[str, Any]:
    """Fetch immutable 2018--2020 predictor evidence and publish its audit."""
    root = Path(repo_root).resolve()
    panel_path = Path(panel_path).resolve()
    registry_path = Path(registry_path).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    for path in (panel_path, registry_path, snapshot_root, output_dir, manifest_path.parent):
        if path != root and root not in path.parents:
            raise PredictorBridgeError(f"bridge path escapes repository: {path}")
    if output_dir.exists() or manifest_path.exists():
        raise PredictorBridgeError("refusing to replace immutable bridge evidence")

    coordinates = load_coordinate_registry(
        registry_path, cohort="development_bridge", expected_count=expected_sites
    )
    daymet_store = SnapshotStore(snapshot_root / "daymet-v1", offline=offline)
    gridmet_store = SnapshotStore(snapshot_root / "gridmet-v1", offline=offline)
    schema_store = SnapshotStore(snapshot_root / "gridmet-schema-v1", offline=offline)
    headers = {"User-Agent": USER_AGENT}
    schema_url = build_gridmet_wind_metadata_url()
    _assert_safe_meteorology_url(schema_url)
    schema_payload, schema_record = schema_store.fetch(
        provider=GRIDMET_SCHEMA_PROVIDER,
        url=schema_url,
        headers=headers,
        retries=max(1, int(retries)),
    )
    gridmet_contract = parse_gridmet_wind_metadata(schema_payload)
    if prefetch_only:
        daymet_requests, gridmet_requests = _prefetch_raw_predictors(
            coordinates,
            daymet_store=daymet_store,
            gridmet_store=gridmet_store,
            retries=retries,
            workers=prefetch_workers,
        )
        daymet_index = freeze_snapshot_index(
            daymet_store, expected_request_sha256=daymet_requests
        )
        gridmet_index = freeze_snapshot_index(
            gridmet_store, expected_request_sha256=gridmet_requests
        )
        schema_index = freeze_snapshot_index(
            schema_store, expected_request_sha256={str(schema_record.request_sha256)}
        )
        return {
            "format": BRIDGE_FORMAT,
            "status": "RAW_PREDICTOR_BRIDGE_PREFETCH_COMPLETE",
            "outcome_values_requested_or_read": False,
            "request_count": int(len(daymet_requests) + len(gridmet_requests)),
            "raw_snapshot_indexes": {
                "daymet": _binding(root, daymet_index),
                "gridmet": _binding(root, gridmet_index),
                "gridmet_schema": _binding(root, schema_index),
            },
            "publication": "raw bytes only; rerun final committed parser with --offline",
        }

    registry = pd.read_csv(
        registry_path,
        usecols=["site_no", "legacy_site_id"],
        dtype={"site_no": "string", "legacy_site_id": "string"},
        keep_default_na=False,
    )
    panel = pd.read_parquet(panel_path)
    frozen = frozen_bridge_slice(panel, registry)
    refreshed, requests = _acquire_cohort(
        coordinates,
        daymet_store=daymet_store,
        gridmet_store=gridmet_store,
        history_start=BRIDGE_START,
        target_end=BRIDGE_END,
        retries=max(1, int(retries)),
        request_interval=max(0.0, float(request_interval)),
        gridmet_scale_factor=float(gridmet_contract["scale_factor"]),
        gridmet_add_offset=float(gridmet_contract["add_offset"]),
    )
    refreshed = _normalise_table(refreshed, label="refreshed predictors")
    report = compare_predictor_bridge(
        frozen, refreshed, expected_site_count=expected_sites
    )

    daymet_index = freeze_snapshot_index(
        daymet_store,
        expected_request_sha256={str(row["daymet"]["request_sha256"]) for row in requests},
    )
    gridmet_index = freeze_snapshot_index(
        gridmet_store,
        expected_request_sha256={str(row["gridmet"]["request_sha256"]) for row in requests},
    )
    schema_index = freeze_snapshot_index(
        schema_store, expected_request_sha256={str(schema_record.request_sha256)}
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        frozen_path = staging / "frozen_panel_predictors_2018_2020.parquet"
        refreshed_path = staging / "refreshed_predictors_2018_2020.parquet"
        report_path = staging / "bridge_report_v1.json"
        request_map_path = staging / "source_request_map_v1.json"
        _write_parquet(frozen_path, frozen)
        _write_parquet(refreshed_path, refreshed)
        _exclusive_create(report_path, canonical_json_bytes(report))
        request_map = {
            "format": "thermoroute.development-predictor-bridge-requests.v1",
            "outcome_values_requested_or_read": False,
            "interval": report["interval"],
            "request_count": len(requests),
            "requests": requests,
            "gridmet_provider_contract": {
                **gridmet_contract,
                "request_sha256": schema_record.request_sha256,
                "response_sha256": schema_record.response_sha256,
                "retrieved_at_utc": schema_record.retrieved_at_utc,
                "byte_count": schema_record.byte_count,
            },
        }
        _exclusive_create(request_map_path, canonical_json_bytes(request_map))
        if output_dir.exists():
            raise PredictorBridgeError("bridge output appeared during publication")
        os.rename(staging, output_dir)

    frozen_path = output_dir / frozen_path.name
    refreshed_path = output_dir / refreshed_path.name
    report_path = output_dir / report_path.name
    request_map_path = output_dir / request_map_path.name
    manifest = {
        "format": BRIDGE_FORMAT,
        "status": report["status"],
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": source_tree_hash(root),
        "panel": _binding(root, panel_path),
        "registry": _binding(root, registry_path),
        "normalized": {
            "frozen": _binding(root, frozen_path),
            "refreshed": _binding(root, refreshed_path),
        },
        "report": _binding(root, report_path),
        "request_map": _binding(root, request_map_path),
        "raw_snapshot_indexes": {
            "daymet": _binding(root, daymet_index),
            "gridmet": _binding(root, gridmet_index),
            "gridmet_schema": _binding(root, schema_index),
        },
        "gate": {
            "requires_status": "PASS_EXACT_PRODUCT_BRIDGE",
            "failure_action": "do not freeze or open Route-A models; investigate product drift",
        },
    }
    _exclusive_create(manifest_path, canonical_json_bytes(manifest))
    return manifest
