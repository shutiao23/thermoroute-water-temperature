"""Outcome-free retrospective meteorology for the sealed Route-A opening.

This module deliberately knows only stable station identity, coordinates and
the five meteorological predictors consumed by the frozen model schema.  It
does not import, read or request an NWIS daily-value endpoint.  Raw Daymet and
gridMET bytes remain content-addressed in :class:`SnapshotStore`; normalized
tables retain the complete site-by-calendar key even when a source value is
missing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config as C
from .provenance import (
    SnapshotStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .usgs import (
    build_daymet_url,
    build_gridmet_wind_metadata_url,
    build_gridmet_wind_url,
    parse_daymet_daily,
    parse_gridmet_wind_daily,
    parse_gridmet_wind_metadata,
)


INPUT_MANIFEST_FORMAT = "thermoroute.route-a-prelabel-inputs.v1"
REQUEST_MAP_FORMAT = "thermoroute.route-a-meteorology-request-map.v1"
PRELABEL_FIELDS = ("TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
DAYMET_FIELDS = ("TEMP", "PRCP", "RHMEAN", "DH")
ACTUAL_FEATURE_ORDER = (
    "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP",
)
DAYMET_PROVIDER = "ornl-daymet-single-pixel-route-a"
GRIDMET_PROVIDER = "gridmet-ncss-route-a"
GRIDMET_SCHEMA_PROVIDER = "gridmet-opendap-schema-route-a"
USER_AGENT = "ThermoRoute/1.0 Route-A pre-label meteorology"
ROUTE_A_TARGET_START = pd.Timestamp("2021-01-01")
ROUTE_A_TARGET_END = pd.Timestamp("2023-12-31")
ROUTE_A_CONTEXT_LENGTH = 32
SITE_PATTERN = re.compile(r"^[0-9]{8,15}$")


class HistoricalInputError(RuntimeError):
    """A pre-label meteorological input or its lineage is unsafe."""


@dataclass(frozen=True)
class RegistryEvidence:
    cohort: str
    path: Path
    sha256: str
    row_count: int
    frame: pd.DataFrame


def _relative(repo_root: Path, path: Path) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise HistoricalInputError(f"artifact is outside repository root: {resolved}")
    return resolved.relative_to(root).as_posix()


def _binding(repo_root: Path, path: Path) -> dict[str, str]:
    if not path.is_file():
        raise HistoricalInputError(f"cannot bind missing artifact: {path}")
    return {"path": _relative(repo_root, path), "sha256": sha256_file(path)}


def _exclusive_create(path: Path, payload: bytes) -> None:
    """Publish immutable bytes without replacement or a temporary overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise HistoricalInputError(f"refusing to replace immutable artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial immutable file is retained as evidence of an interrupted
        # attempt.  A later run must fail closed and use a new versioned path.
        raise


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write a staged Parquet artifact and fsync it before publication."""
    if path.exists():
        raise HistoricalInputError(f"refusing to replace immutable artifact: {path}")
    frame.to_parquet(path, index=False)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def load_coordinate_registry(
    path: str | Path,
    *,
    cohort: str,
    expected_count: int,
) -> RegistryEvidence:
    """Read only stable identity and coordinates from one frozen registry."""
    path = Path(path).resolve()
    if not path.is_file():
        raise HistoricalInputError(f"missing {cohort} registry: {path}")
    header = pd.read_csv(path, nrows=0)
    required = {"site_no", "lat", "lon"}
    missing = required - set(header.columns)
    if missing:
        raise HistoricalInputError(f"{cohort} registry lacks {sorted(missing)}")
    # usecols is a safety boundary: legacy development availability and every
    # other column remain uninterpreted, even though the whole frozen file is
    # byte-hashed to bind its identity.
    frame = pd.read_csv(
        path,
        usecols=["site_no", "lat", "lon"],
        dtype={"site_no": "string"},
        keep_default_na=False,
    )
    frame["site_no"] = frame.site_no.astype("string").str.strip()
    frame["lat"] = pd.to_numeric(frame.lat, errors="coerce")
    frame["lon"] = pd.to_numeric(frame.lon, errors="coerce")
    if len(frame) != expected_count:
        raise HistoricalInputError(
            f"{cohort} registry has {len(frame)} sites; expected {expected_count}"
        )
    if frame.site_no.eq("").any() or frame.site_no.isna().any():
        raise HistoricalInputError(f"{cohort} registry has an empty site_no")
    invalid_ids = [site for site in frame.site_no.astype(str) if not SITE_PATTERN.fullmatch(site)]
    if invalid_ids:
        raise HistoricalInputError(f"{cohort} registry has invalid site_no: {invalid_ids[:5]}")
    if frame.site_no.duplicated().any():
        raise HistoricalInputError(f"{cohort} registry duplicates site_no")
    coordinates_valid = (
        frame.lat.between(-90.0, 90.0)
        & frame.lon.between(-180.0, 180.0)
        & np.isfinite(frame[["lat", "lon"]]).all(axis=1)
    )
    if coordinates_valid.ne(True).any():
        bad = frame.loc[coordinates_valid.ne(True), "site_no"].astype(str).tolist()
        raise HistoricalInputError(f"{cohort} registry has invalid coordinates: {bad[:5]}")
    frame = frame.sort_values("site_no").reset_index(drop=True)
    return RegistryEvidence(
        cohort=cohort,
        path=path,
        sha256=sha256_file(path),
        row_count=len(frame),
        frame=frame,
    )


def validate_historical_protocol(protocol_path: str | Path) -> dict[str, Any]:
    """Resolve the exact label-free calendar and schema from the protocol."""
    path = Path(protocol_path).resolve()
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HistoricalInputError(f"cannot read Route-A protocol: {path}") from exc
    if protocol.get("schema_version") != 1 or protocol.get("status") not in {
        "FROZEN_NOT_ACQUIRED", "REGISTRY_FROZEN_LABELS_SEALED",
    }:
        raise HistoricalInputError("Route-A protocol is not in a sealed pre-label state")
    holdout = protocol.get("time_holdout", {})
    target_start = pd.Timestamp(str(holdout.get("primary_target_start", "")))
    target_end = pd.Timestamp(str(holdout.get("end", "")))
    if target_start != ROUTE_A_TARGET_START or target_end != ROUTE_A_TARGET_END:
        raise HistoricalInputError("Route-A target interval differs from 2021-2023")
    historical = protocol.get("primary_historical_input_contract", {})
    inference = protocol.get("primary_inference_contract", {})
    expected = {
        "horizon_specific_future_nwp_consumed": False,
        "retrospective_meteorological_inputs": list(PRELABEL_FIELDS),
    }
    wrong = {key: historical.get(key) for key, value in expected.items()
             if historical.get(key) != value}
    if wrong:
        raise HistoricalInputError(f"protocol historical-input contract changed: {wrong}")
    if tuple(inference.get("feature_order", ())) != ACTUAL_FEATURE_ORDER:
        raise HistoricalInputError("protocol feature order differs from the executable schema")
    if inference.get("wlevel_consumed") is not False:
        raise HistoricalInputError("protocol unexpectedly exposes WLEVEL to the model")
    limitation = str(historical.get("provisional_vintage_limitation", "")).strip()
    if not limitation:
        raise HistoricalInputError("protocol lacks the retrospective vintage limitation")
    if C.CONTEXT_LENGTH != ROUTE_A_CONTEXT_LENGTH:
        raise HistoricalInputError("executable context length differs from frozen Route-A")
    # Earliest h=1 issue is the day before the first target; its inclusive
    # 32-day history therefore starts exactly 32 days before target_start.
    history_start = target_start - pd.Timedelta(days=ROUTE_A_CONTEXT_LENGTH)
    return {
        "document": protocol,
        "path": path,
        "sha256": sha256_file(path),
        "target_start": target_start,
        "target_end": target_end,
        "history_start": history_start,
        "provisional_vintage_limitation": limitation,
    }


def _assert_safe_meteorology_url(url: str) -> None:
    lowered = url.lower()
    forbidden = ("/nwis/", "wtemp", "00010", "00060", "00065", "water_temperature")
    if any(token in lowered for token in forbidden):
        raise HistoricalInputError("pre-label meteorology attempted an outcome/history URL")
    allowed = (
        "https://daymet.ornl.gov/single-pixel/api/data?",
        "https://thredds.northwestknowledge.net/thredds/ncss/",
        "https://thredds.northwestknowledge.net/thredds/dodsc/",
    )
    if not lowered.startswith(allowed):
        raise HistoricalInputError(f"unapproved meteorological endpoint: {url}")


def _request_record(
    *,
    cohort: str,
    site_no: str,
    lat: float,
    lon: float,
    daymet: Any,
    gridmet: Any,
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "site_no": site_no,
        "requested_lat": lat,
        "requested_lon": lon,
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "daymet": {
            "request_sha256": daymet.request_sha256,
            "response_sha256": daymet.response_sha256,
            "retrieved_at_utc": daymet.retrieved_at_utc,
            "byte_count": daymet.byte_count,
        },
        "gridmet": {
            "request_sha256": gridmet.request_sha256,
            "response_sha256": gridmet.response_sha256,
            "retrieved_at_utc": gridmet.retrieved_at_utc,
            "byte_count": gridmet.byte_count,
        },
    }


def _acquire_cohort(
    registry: RegistryEvidence,
    *,
    daymet_store: SnapshotStore,
    gridmet_store: SnapshotStore,
    history_start: pd.Timestamp,
    target_end: pd.Timestamp,
    retries: int,
    request_interval: float,
    gridmet_scale_factor: float,
    gridmet_add_offset: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    start_text = history_start.strftime("%Y-%m-%d")
    end_text = target_end.strftime("%Y-%m-%d")
    headers = {"User-Agent": USER_AGENT}
    for station in registry.frame.itertuples(index=False):
        site_no = str(station.site_no)
        lat, lon = float(station.lat), float(station.lon)
        daymet_url = build_daymet_url(lat, lon, start_text, end_text)
        gridmet_url = build_gridmet_wind_url(lat, lon, start_text, end_text)
        _assert_safe_meteorology_url(daymet_url)
        _assert_safe_meteorology_url(gridmet_url)
        daymet_payload, daymet_record = daymet_store.fetch(
            provider=DAYMET_PROVIDER,
            url=daymet_url,
            headers=headers,
            retries=retries,
        )
        gridmet_payload, gridmet_record = gridmet_store.fetch(
            provider=GRIDMET_PROVIDER,
            url=gridmet_url,
            headers=headers,
            retries=retries,
        )
        daymet = parse_daymet_daily(daymet_payload, start=start_text, end=end_text)
        wind = parse_gridmet_wind_daily(
            gridmet_payload,
            start=start_text,
            end=end_text,
            scale_factor=gridmet_scale_factor,
            add_offset=gridmet_add_offset,
        )
        if not daymet.index.equals(wind.index):
            raise HistoricalInputError(f"source calendars disagree for site {site_no}")
        frame = daymet.copy()
        frame["WDSP"] = wind.to_numpy(float)
        frame = frame.reset_index()
        frame.insert(0, "site_no", site_no)
        frame = frame[["site_no", "DATE", *PRELABEL_FIELDS]]
        numeric = frame[list(PRELABEL_FIELDS)].to_numpy(float)
        if np.isinf(numeric).any():
            raise HistoricalInputError(f"normalized meteorology is infinite for {site_no}")
        frames.append(frame)
        records.append(_request_record(
            cohort=registry.cohort,
            site_no=site_no,
            lat=lat,
            lon=lon,
            daymet=daymet_record,
            gridmet=gridmet_record,
        ))
        if request_interval > 0.0 and not (daymet_store.offline and gridmet_store.offline):
            time.sleep(request_interval)
    combined = pd.concat(frames, ignore_index=True)
    combined["site_no"] = combined.site_no.astype("string")
    combined["DATE"] = pd.to_datetime(combined.DATE)
    expected_dates = pd.date_range(history_start, target_end, freq="D")
    expected_rows = registry.row_count * len(expected_dates)
    if len(combined) != expected_rows or combined.duplicated(["site_no", "DATE"]).any():
        raise HistoricalInputError(f"{registry.cohort} normalized key registry is incomplete")
    if set(combined.site_no.astype(str)) != set(registry.frame.site_no.astype(str)):
        raise HistoricalInputError(f"{registry.cohort} normalized sites changed")
    return combined, records


def _build_snapshot_index(
    store: SnapshotStore, *, bind_metadata_bytes: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(store.root.glob("*/*/metadata.json")):
        response_path = metadata_path.parent / "response.bin"
        try:
            metadata_bytes = metadata_path.read_bytes()
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            response = response_path.read_bytes()
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HistoricalInputError(f"incomplete raw snapshot: {metadata_path}") from exc
        request = metadata.get("request")
        if not isinstance(request, Mapping):
            raise HistoricalInputError(f"snapshot lacks a canonical request: {metadata_path}")
        request_sha = sha256_bytes(canonical_json_bytes(request))
        response_sha = sha256_bytes(response)
        expected = {
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "byte_count": len(response),
            "response_file": "response.bin",
        }
        wrong = {key: metadata.get(key) for key, value in expected.items()
                 if metadata.get(key) != value}
        if metadata_path.parent.name != request_sha or wrong:
            raise HistoricalInputError(f"raw snapshot identity mismatch: {metadata_path}")
        retrieved = pd.to_datetime(metadata.get("retrieved_at_utc"), errors="coerce", utc=True)
        if pd.isna(retrieved) or int(metadata.get("http_status", -1)) != 200:
            raise HistoricalInputError(f"raw snapshot lacks a successful retrieval: {metadata_path}")
        _assert_safe_meteorology_url(str(request.get("url", "")))
        record = {
            "provider": str(request.get("provider", "")),
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "retrieved_at_utc": str(metadata.get("retrieved_at_utc", "")),
            "byte_count": len(response),
            "request": dict(request),
            "metadata_path": str(metadata_path.relative_to(store.root)),
            "response_path": str(response_path.relative_to(store.root)),
        }
        if bind_metadata_bytes:
            record["metadata_sha256"] = sha256_bytes(metadata_bytes)
            record["metadata_byte_count"] = len(metadata_bytes)
        records.append(record)
    if not records:
        raise HistoricalInputError(f"snapshot store is empty: {store.root}")
    return {
        "schema_version": 2 if bind_metadata_bytes else 1,
        "snapshot_count": len(records),
        "records": records,
    }


def freeze_snapshot_index(
    store: SnapshotStore,
    *,
    expected_request_sha256: set[str],
) -> Path:
    """Create once, or byte-verify, a deterministic raw-snapshot index."""
    document = _build_snapshot_index(store)
    actual = {str(record["request_sha256"]) for record in document["records"]}
    if actual != expected_request_sha256:
        raise HistoricalInputError(
            "raw snapshot store contains missing or extraneous Route-A requests"
        )
    payload = canonical_json_bytes(document)
    path = store.root / "snapshot_index.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise HistoricalInputError(f"immutable snapshot index changed: {path}")
        return path
    _exclusive_create(path, payload)
    return path


def migrate_snapshot_index_metadata_v2(store: SnapshotStore) -> Path:
    """Create a v2 index beside an immutable legacy v1 snapshot index.

    The migration is offline and outcome-free: it reads only the existing HTTP
    metadata/response pairs, proves that every legacy v1 record is unchanged,
    and adds byte count/SHA-256 bindings for ``metadata.json``.  The v1 index is
    retained; publication uses the create-only ``snapshot_index_v2.json``.
    """
    legacy_path = store.root / "snapshot_index.json"
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalInputError("legacy snapshot index is absent or malformed") from exc
    if (
        not isinstance(legacy, dict)
        or set(legacy) != {"schema_version", "snapshot_count", "records"}
        or legacy.get("schema_version") != 1
        or type(legacy.get("snapshot_count")) is not int
        or not isinstance(legacy.get("records"), list)
        or legacy["snapshot_count"] != len(legacy["records"])
    ):
        raise HistoricalInputError("legacy snapshot index is not exact schema v1")
    upgraded = _build_snapshot_index(store, bind_metadata_bytes=True)
    projected_records = [
        {
            key: value for key, value in record.items()
            if key not in {"metadata_sha256", "metadata_byte_count"}
        }
        for record in upgraded["records"]
    ]
    projection = {
        "schema_version": 1,
        "snapshot_count": upgraded["snapshot_count"],
        "records": projected_records,
    }
    if projection != legacy:
        raise HistoricalInputError(
            "raw snapshots no longer project exactly to the immutable v1 index"
        )
    destination = store.root / "snapshot_index_v2.json"
    payload = canonical_json_bytes(upgraded)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise HistoricalInputError("v2 snapshot index bytes changed")
        return destination
    _exclusive_create(destination, payload)
    return destination


def _table_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": len(frame),
        "site_count": int(frame.site_no.nunique()),
        "date_count_per_site": int(frame.groupby("site_no").DATE.nunique().min()),
        "missing_value_count": {
            field: int(frame[field].isna().sum()) for field in PRELABEL_FIELDS
        },
    }


def acquire_historical_inputs(
    *,
    repo_root: str | Path,
    protocol_path: str | Path,
    temporal_registry_path: str | Path,
    external_registry_path: str | Path,
    snapshot_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    offline: bool = False,
    retries: int = 3,
    request_interval: float = 0.0,
    secondary_nwp_resolution: str = "EXPLICITLY_NOT_USED",
    expected_temporal_sites: int = 120,
    expected_external_sites: int = 30,
) -> dict[str, Any]:
    """Acquire, normalize and freeze both Route-A meteorological cohorts.

    The ``expected_*`` parameters make small offline fixture tests possible;
    the production CLI does not expose them and always requires 120+30 sites.
    """
    root = Path(repo_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    temporal_registry_path = Path(temporal_registry_path).resolve()
    external_registry_path = Path(external_registry_path).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    for path in (
        protocol_path, temporal_registry_path, external_registry_path,
        snapshot_root, output_dir, manifest_path.parent,
    ):
        if path != root and root not in path.parents:
            raise HistoricalInputError(f"Route-A artifact path escapes repository: {path}")
    if output_dir.exists():
        raise HistoricalInputError(f"refusing to replace immutable output bundle: {output_dir}")
    if manifest_path.exists():
        raise HistoricalInputError(f"refusing to replace immutable manifest: {manifest_path}")
    if secondary_nwp_resolution != "EXPLICITLY_NOT_USED":
        raise HistoricalInputError(
            "this primary-input freezer can attest only EXPLICITLY_NOT_USED; "
            "ACQUIRED_AND_FROZEN requires separate NWP-manifest verification"
        )
    protocol = validate_historical_protocol(protocol_path)
    temporal = load_coordinate_registry(
        temporal_registry_path, cohort="temporal", expected_count=expected_temporal_sites
    )
    external = load_coordinate_registry(
        external_registry_path, cohort="external", expected_count=expected_external_sites
    )
    overlap = set(temporal.frame.site_no.astype(str)) & set(external.frame.site_no.astype(str))
    if overlap:
        raise HistoricalInputError(f"external registry overlaps temporal sites: {sorted(overlap)[:5]}")

    daymet_store = SnapshotStore(snapshot_root / "daymet-v1", offline=offline)
    gridmet_store = SnapshotStore(snapshot_root / "gridmet-v1", offline=offline)
    gridmet_schema_store = SnapshotStore(
        snapshot_root / "gridmet-schema-v1", offline=offline
    )
    headers = {"User-Agent": USER_AGENT}
    schema_url = build_gridmet_wind_metadata_url()
    _assert_safe_meteorology_url(schema_url)
    schema_payload, schema_record = gridmet_schema_store.fetch(
        provider=GRIDMET_SCHEMA_PROVIDER,
        url=schema_url,
        headers=headers,
        retries=max(1, int(retries)),
    )
    gridmet_contract = parse_gridmet_wind_metadata(schema_payload)
    cohort_frames: dict[str, pd.DataFrame] = {}
    request_records: list[dict[str, Any]] = []
    for registry in (temporal, external):
        frame, records = _acquire_cohort(
            registry,
            daymet_store=daymet_store,
            gridmet_store=gridmet_store,
            history_start=protocol["history_start"],
            target_end=protocol["target_end"],
            retries=max(1, int(retries)),
            request_interval=max(0.0, float(request_interval)),
            gridmet_scale_factor=float(gridmet_contract["scale_factor"]),
            gridmet_add_offset=float(gridmet_contract["add_offset"]),
        )
        cohort_frames[registry.cohort] = frame
        request_records.extend(records)

    daymet_requests = {
        str(record["daymet"]["request_sha256"]) for record in request_records
    }
    gridmet_requests = {
        str(record["gridmet"]["request_sha256"]) for record in request_records
    }
    daymet_index = freeze_snapshot_index(
        daymet_store, expected_request_sha256=daymet_requests
    )
    gridmet_index = freeze_snapshot_index(
        gridmet_store, expected_request_sha256=gridmet_requests
    )
    gridmet_schema_index = freeze_snapshot_index(
        gridmet_schema_store,
        expected_request_sha256={str(schema_record.request_sha256)},
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        temporal_path = staging / "temporal_retrospective_meteorology_v1.parquet"
        external_path = staging / "external_retrospective_meteorology_v1.parquet"
        request_map_path = staging / "source_request_map_v1.json"
        _write_parquet(temporal_path, cohort_frames["temporal"])
        _write_parquet(external_path, cohort_frames["external"])
        request_map = {
            "format": REQUEST_MAP_FORMAT,
            "protocol_sha256": protocol["sha256"],
            "contains_outcome": False,
            "contains_outcome_labels": False,
            "labels_requested_or_read": False,
            "outcome_endpoint_called": False,
            "post_2020_wtemp_requested_or_inspected": False,
            "fields": list(PRELABEL_FIELDS),
            "history_start": protocol["history_start"].strftime("%Y-%m-%d"),
            "target_end": protocol["target_end"].strftime("%Y-%m-%d"),
            "request_count": len(request_records),
            "requests": request_records,
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
            raise HistoricalInputError(
                f"refusing to replace immutable output bundle: {output_dir}"
            )
        os.rename(staging, output_dir)

    temporal_path = output_dir / temporal_path.name
    external_path = output_dir / external_path.name
    request_map_path = output_dir / request_map_path.name
    manifest = {
        "format": INPUT_MANIFEST_FORMAT,
        "status": "FROZEN_PRELABEL_NO_OUTCOMES",
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "post_2020_wtemp_requested_or_inspected": False,
        "retrospective_provisional_vintage_reconstructable": False,
        "horizon_specific_future_nwp_consumed": False,
        "protocol_sha256": protocol["sha256"],
        "actual_feature_order": list(ACTUAL_FEATURE_ORDER),
        "prelabel_fields": list(PRELABEL_FIELDS),
        "history_start": protocol["history_start"].strftime("%Y-%m-%d"),
        "target_start": protocol["target_start"].strftime("%Y-%m-%d"),
        "target_end": protocol["target_end"].strftime("%Y-%m-%d"),
        "context_length_days": ROUTE_A_CONTEXT_LENGTH,
        "evaluation_type": "ONE_SHOT_RETROSPECTIVE_HISTORICAL_INFORMATION",
        "provisional_vintage_limitation": protocol["provisional_vintage_limitation"],
        "normalization_contract": {
            "daymet_fields": {
                "TEMP": "arithmetic mean of tmax and tmin, degrees C",
                "PRCP": "daily precipitation, mm/day; negative fill values become missing",
                "RHMEAN": "100 * vapour_pressure / Tetens_SVP(TEMP), clipped to [0,100]",
                "DH": (
                    "legacy feature name for Daymet srad: mean incident shortwave "
                    "flux over the daylight period, W/m^2; not a 24-hour mean or "
                    "daily energy total"
                ),
            },
            "gridmet_field": {
                "WDSP": (
                    "daily_mean_wind_speed packed NCSS value transformed using the "
                    "frozen OPeNDAP DAS scale_factor=0.1 and add_offset=0; m/s"
                ),
            },
            "missingness_rule": (
                "retain every site/date key; unavailable or invalid source values are NaN"
            ),
        },
        "cohort_tables": {
            "temporal": _binding(root, temporal_path),
            "external": _binding(root, external_path),
        },
        "cohort_summaries": {
            cohort: _table_summary(frame) for cohort, frame in cohort_frames.items()
        },
        "registry_inputs": {
            registry.cohort: {
                "path": _relative(root, registry.path),
                "sha256": registry.sha256,
                "columns_read": ["site_no", "lat", "lon"],
                "row_count": registry.row_count,
            }
            for registry in (temporal, external)
        },
        "source_evidence": [
            {
                "source": "ORNL Daymet single-pixel daily data",
                "evidence_type": "snapshot_index",
                "contains_outcome": False,
                "contains_outcome_labels": False,
                "fields": list(DAYMET_FIELDS),
                "artifact": _binding(root, daymet_index),
            },
            {
                "source": "gridMET daily mean wind via NWK NCSS",
                "evidence_type": "snapshot_index",
                "contains_outcome": False,
                "contains_outcome_labels": False,
                "fields": ["WDSP"],
                "artifact": _binding(root, gridmet_index),
            },
            {
                "source": "gridMET OPeNDAP dataset attributes",
                "evidence_type": "snapshot_index",
                "contains_outcome": False,
                "contains_outcome_labels": False,
                "fields": ["WDSP"],
                "contract_attributes": ["units", "scale_factor", "add_offset"],
                "artifact": _binding(root, gridmet_schema_index),
                "validated_contract": gridmet_contract,
            },
            {
                "source": "site-to-request normalization map",
                "evidence_type": "normalized_immutable_snapshot",
                "contains_outcome": False,
                "contains_outcome_labels": False,
                "fields": list(PRELABEL_FIELDS),
                "artifact": _binding(root, request_map_path),
            },
        ],
        "secondary_nwp_resolution": secondary_nwp_resolution,
    }
    _exclusive_create(manifest_path, canonical_json_bytes(manifest))
    return manifest
