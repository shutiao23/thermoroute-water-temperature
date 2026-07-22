"""Pre-opening bridge audit for legacy and currently retrievable meteorology.

The canonical 2006--2020 panel no longer has its original HTTP responses.  This
module therefore cannot reconstruct that missing provenance.  It does provide a
strict, outcome-free compatibility gate: re-fetch 2018--2020 Daymet/gridMET with
the confirmation parser, archive those bytes, and compare the resulting predictor
values with the frozen panel on the exact site/date registry.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

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
    migrate_snapshot_index_metadata_v2,
)
from .provenance import SnapshotStore, canonical_json_bytes, sha256_bytes, sha256_file
from .repro import atomic_write_bytes, source_tree_hash
from .usgs import (
    build_daymet_url,
    build_gridmet_wind_metadata_url,
    build_gridmet_wind_url,
    parse_daymet_daily,
    parse_gridmet_wind_daily,
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


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictorBridgeError(f"{label} is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise PredictorBridgeError(f"{label} is not a JSON object: {path}")
    return value


def _resolve_manifest_binding(
    root: Path, value: object, *, label: str,
) -> Path:
    """Resolve one exact, in-tree, single-link regular manifest binding."""
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PredictorBridgeError(f"bridge {label} binding is malformed")
    raw_path, digest = value.get("path"), value.get("sha256")
    if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
        raise PredictorBridgeError(f"bridge {label} path is malformed")
    relative = Path(raw_path)
    candidate = root / relative
    cursor = root
    try:
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise PredictorBridgeError(f"bridge {label} path is linked")
        status = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except PredictorBridgeError:
        raise
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise PredictorBridgeError(f"bridge {label} path is absent or invalid") from exc
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or root not in resolved.parents
        or resolved.relative_to(root).as_posix() != raw_path
        or digest != sha256_file(resolved)
    ):
        raise PredictorBridgeError(f"bridge {label} bytes/path changed")
    return resolved


def _offline_snapshot_registry(
    index_path: str | Path,
    *,
    expected_provider: str,
) -> dict[str, tuple[dict[str, Any], bytes]]:
    """Validate an archived snapshot index down to request and metadata bytes."""
    raw_index_path = Path(index_path)
    if (
        raw_index_path.is_symlink()
        or not raw_index_path.is_file()
        or not stat.S_ISREG(raw_index_path.stat().st_mode)
        or raw_index_path.stat().st_nlink != 1
    ):
        raise PredictorBridgeError("predictor snapshot index is linked or non-regular")
    index_path = raw_index_path.resolve()
    index = _read_json_object(index_path, label="predictor snapshot index")
    records = index.get("records")
    if (
        set(index) != {"schema_version", "snapshot_count", "records"}
        or index.get("schema_version") != 2
        or type(index.get("snapshot_count")) is not int
        or not isinstance(records, list)
        or index["snapshot_count"] != len(records)
        or not records
    ):
        raise PredictorBridgeError("predictor snapshot index contract changed")
    expected_record_fields = {
        "provider", "request_sha256", "response_sha256", "retrieved_at_utc",
        "byte_count", "request", "metadata_path", "response_path",
        "metadata_sha256", "metadata_byte_count",
    }
    output: dict[str, tuple[dict[str, Any], bytes]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_record_fields:
            raise PredictorBridgeError("predictor snapshot record contract changed")
        request = record.get("request")
        if not isinstance(request, Mapping):
            raise PredictorBridgeError("predictor snapshot request is malformed")
        request_document = dict(request)
        request_sha = sha256_bytes(canonical_json_bytes(request_document))
        provider = str(record.get("provider", ""))
        response_sha = str(record.get("response_sha256", ""))
        metadata_raw = record.get("metadata_path")
        response_raw = record.get("response_path")
        if (
            provider != expected_provider
            or request_document.get("provider") != expected_provider
            or record.get("request_sha256") != request_sha
            or request_sha in output
            or not isinstance(metadata_raw, str)
            or not isinstance(response_raw, str)
            or Path(metadata_raw).is_absolute()
            or Path(response_raw).is_absolute()
            or type(record.get("byte_count")) is not int
            or record["byte_count"] < 1
            or pd.isna(pd.to_datetime(record.get("retrieved_at_utc"), utc=True, errors="coerce"))
        ):
            raise PredictorBridgeError("predictor snapshot identity is malformed")
        _assert_safe_meteorology_url(str(request_document.get("url", "")))
        expected_base = Path(expected_provider) / request_sha
        if (
            Path(metadata_raw) != expected_base / "metadata.json"
            or Path(response_raw) != expected_base / "response.bin"
        ):
            raise PredictorBridgeError("predictor snapshot path is not content addressed")
        metadata_path = (index_path.parent / metadata_raw).resolve()
        response_path = (index_path.parent / response_raw).resolve()
        linked_component = False
        for relative in (Path(metadata_raw), Path(response_raw)):
            cursor = index_path.parent
            for component in relative.parts:
                cursor = cursor / component
                if cursor.is_symlink():
                    linked_component = True
                    break
        if (
            index_path.parent not in metadata_path.parents
            or index_path.parent not in response_path.parents
            or not metadata_path.is_file()
            or not response_path.is_file()
            or linked_component
            or not stat.S_ISREG(metadata_path.stat().st_mode)
            or not stat.S_ISREG(response_path.stat().st_mode)
            or metadata_path.stat().st_nlink != 1
            or response_path.stat().st_nlink != 1
        ):
            raise PredictorBridgeError("predictor snapshot path escapes or is absent")
        payload = response_path.read_bytes()
        metadata_bytes = metadata_path.read_bytes()
        metadata = _read_json_object(metadata_path, label="predictor snapshot metadata")
        metadata_fields = {
            "schema_version", "request", "request_sha256", "retrieved_at_utc",
            "http_status", "response_headers", "byte_count", "response_sha256",
            "response_file",
        }
        critical = {
            "schema_version": 1,
            "request": request_document,
            "request_sha256": request_sha,
            "retrieved_at_utc": record["retrieved_at_utc"],
            "http_status": 200,
            "byte_count": len(payload),
            "response_sha256": sha256_bytes(payload),
            "response_file": "response.bin",
        }
        if (
            set(metadata) != metadata_fields
            or not isinstance(metadata.get("response_headers"), Mapping)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata["response_headers"].items()
            )
            or any(metadata.get(key) != value for key, value in critical.items())
            or record.get("byte_count") != len(payload)
            or response_sha != critical["response_sha256"]
            or record.get("metadata_byte_count") != len(metadata_bytes)
            or record.get("metadata_sha256") != sha256_bytes(metadata_bytes)
        ):
            raise PredictorBridgeError("predictor snapshot bytes/metadata/index disagree")
        output[request_sha] = (request_document, payload)
    return output


def replay_predictor_bridge_offline(
    *,
    registry_path: str | Path,
    request_map_path: str | Path,
    daymet_index_path: str | Path,
    gridmet_index_path: str | Path,
    gridmet_schema_index_path: str | Path,
    expected_sites: int = 120,
) -> pd.DataFrame:
    """Rebuild normalized bridge predictors solely from archived raw responses.

    The current Daymet/gridMET parsers are applied to the exact indexed response
    bytes.  Every request URL, request hash, response hash, retrieval timestamp,
    byte count, provider, coordinate and request-map entry must agree before a
    value is parsed.
    """
    registry = load_coordinate_registry(
        registry_path, cohort="development_bridge", expected_count=expected_sites
    )
    request_map = _read_json_object(Path(request_map_path), label="bridge request map")
    requests = request_map.get("requests")
    provider_contract = request_map.get("gridmet_provider_contract")
    if (
        set(request_map) != {
            "format", "outcome_values_requested_or_read", "interval",
            "request_count", "requests", "gridmet_provider_contract",
        }
        or request_map.get("format")
        != "thermoroute.development-predictor-bridge-requests.v1"
        or request_map.get("outcome_values_requested_or_read") is not False
        or request_map.get("interval") != {"start": "2018-01-01", "end": "2020-12-31"}
        or type(request_map.get("request_count")) is not int
        or request_map["request_count"] != expected_sites
        or not isinstance(requests, list)
        or len(requests) != expected_sites
        or not isinstance(provider_contract, Mapping)
    ):
        raise PredictorBridgeError("bridge request map contract changed")
    daymet = _offline_snapshot_registry(
        daymet_index_path, expected_provider=DAYMET_PROVIDER
    )
    gridmet = _offline_snapshot_registry(
        gridmet_index_path, expected_provider=GRIDMET_PROVIDER
    )
    schemas = _offline_snapshot_registry(
        gridmet_schema_index_path, expected_provider=GRIDMET_SCHEMA_PROVIDER
    )
    if len(schemas) != 1:
        raise PredictorBridgeError("gridMET schema snapshot registry changed")
    schema_request_sha, (schema_request, schema_payload) = next(iter(schemas.items()))
    expected_schema_url = build_gridmet_wind_metadata_url()
    expected_schema_request = SnapshotStore.request_document(
        provider=GRIDMET_SCHEMA_PROVIDER,
        url=expected_schema_url,
        headers={"User-Agent": USER_AGENT},
    )
    schema_metadata = parse_gridmet_wind_metadata(schema_payload)
    schema_index = _read_json_object(Path(gridmet_schema_index_path), label="gridMET schema index")
    schema_record = schema_index["records"][0]
    expected_provider_contract = {
        **schema_metadata,
        "request_sha256": schema_request_sha,
        "response_sha256": schema_record["response_sha256"],
        "retrieved_at_utc": schema_record["retrieved_at_utc"],
        "byte_count": schema_record["byte_count"],
    }
    if schema_request != expected_schema_request or dict(provider_contract) != expected_provider_contract:
        raise PredictorBridgeError("gridMET provider schema lineage changed")

    request_fields = {
        "cohort", "site_no", "requested_lat", "requested_lon",
        "contains_outcome", "contains_outcome_labels", "daymet", "gridmet",
    }
    source_fields = {
        "request_sha256", "response_sha256", "retrieved_at_utc", "byte_count",
    }
    expected_sites_order = registry.frame.site_no.astype(str).tolist()
    observed_sites: list[str] = []
    frames: list[pd.DataFrame] = []
    used_daymet: set[str] = set()
    used_gridmet: set[str] = set()
    for station, row in zip(registry.frame.itertuples(index=False), requests, strict=True):
        if not isinstance(row, Mapping) or set(row) != request_fields:
            raise PredictorBridgeError("bridge per-site request record changed")
        site_no = str(station.site_no)
        lat, lon = float(station.lat), float(station.lon)
        observed_sites.append(str(row.get("site_no", "")))
        if (
            row.get("cohort") != "development_bridge"
            or row.get("site_no") != site_no
            or row.get("requested_lat") != lat
            or row.get("requested_lon") != lon
            or row.get("contains_outcome") is not False
            or row.get("contains_outcome_labels") is not False
        ):
            raise PredictorBridgeError("bridge request map does not match station registry")
        source_records: dict[str, Mapping[str, Any]] = {}
        for label in ("daymet", "gridmet"):
            source = row.get(label)
            if not isinstance(source, Mapping) or set(source) != source_fields:
                raise PredictorBridgeError("bridge source request binding changed")
            source_records[label] = source
        daymet_url = build_daymet_url(lat, lon, "2018-01-01", "2020-12-31")
        gridmet_url = build_gridmet_wind_url(lat, lon, "2018-01-01", "2020-12-31")
        expected_requests = {
            "daymet": SnapshotStore.request_document(
                provider=DAYMET_PROVIDER, url=daymet_url,
                headers={"User-Agent": USER_AGENT},
            ),
            "gridmet": SnapshotStore.request_document(
                provider=GRIDMET_PROVIDER, url=gridmet_url,
                headers={"User-Agent": USER_AGENT},
            ),
        }
        payloads: dict[str, bytes] = {}
        for label, snapshots, used in (
            ("daymet", daymet, used_daymet),
            ("gridmet", gridmet, used_gridmet),
        ):
            source = source_records[label]
            request_sha = str(source["request_sha256"])
            if request_sha not in snapshots:
                raise PredictorBridgeError("bridge request map references absent raw bytes")
            request, payload = snapshots[request_sha]
            index_path = Path(daymet_index_path if label == "daymet" else gridmet_index_path)
            index_record = next(
                item for item in _read_json_object(index_path, label="bridge source index")["records"]
                if item["request_sha256"] == request_sha
            )
            if (
                request != expected_requests[label]
                or any(source[field] != index_record[field] for field in source_fields)
            ):
                raise PredictorBridgeError("bridge request URL or raw lineage changed")
            payloads[label] = payload
            used.add(request_sha)
        daily = parse_daymet_daily(
            payloads["daymet"], start="2018-01-01", end="2020-12-31"
        )
        wind = parse_gridmet_wind_daily(
            payloads["gridmet"], start="2018-01-01", end="2020-12-31",
            scale_factor=float(provider_contract["scale_factor"]),
            add_offset=float(provider_contract["add_offset"]),
        )
        if not daily.index.equals(wind.index):
            raise PredictorBridgeError(f"raw provider calendars disagree for {site_no}")
        frame = daily.copy()
        frame["WDSP"] = wind.to_numpy(float)
        frame = frame.reset_index()
        frame.insert(0, "site_no", site_no)
        frames.append(frame[["site_no", "DATE", *BRIDGE_FIELDS]])
    if (
        observed_sites != expected_sites_order
        or used_daymet != set(daymet)
        or used_gridmet != set(gridmet)
    ):
        raise PredictorBridgeError("bridge request/raw registry is incomplete or has extras")
    return _normalise_table(pd.concat(frames, ignore_index=True), label="offline raw replay")


def migrate_development_bridge_metadata_indexes_v2(
    *, repo_root: str | Path, manifest_path: str | Path,
) -> dict[str, Any]:
    """Version legacy raw indexes and re-prove refreshed predictors offline.

    No network endpoint and no target-outcome table is accessed.  Legacy v1
    indexes remain in place; the bridge manifest is atomically rebound to new
    create-only ``snapshot_index_v2.json`` files only after current-parser replay
    exactly matches the already frozen refreshed Parquet values.
    """
    root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if root not in manifest_path.parents:
        raise PredictorBridgeError("bridge manifest escapes repository")
    manifest = _read_json_object(manifest_path, label="development bridge manifest")
    raw = manifest.get("raw_snapshot_indexes")
    if (
        set(manifest) != {
            "format", "status", "outcome_values_requested_or_read",
            "source_tree_sha256", "panel", "registry", "normalized", "report",
            "request_map", "raw_snapshot_indexes", "gate",
        }
        or manifest.get("format") != BRIDGE_FORMAT
        or manifest.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or manifest.get("outcome_values_requested_or_read") is not False
        or not isinstance(raw, Mapping)
        or set(raw) != {"daymet", "gridmet", "gridmet_schema"}
    ):
        raise PredictorBridgeError("development bridge manifest is not migratable")

    def bound_path(value: object, *, label: str) -> Path:
        return _resolve_manifest_binding(root, value, label=label)

    upgraded_bindings: dict[str, dict[str, str]] = {}
    canonical_raw_roots = {
        "daymet": root / (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/daymet-v1"
        ),
        "gridmet": root / (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/gridmet-v1"
        ),
        "gridmet_schema": root / (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-schema-v1"
        ),
    }
    for label in ("daymet", "gridmet", "gridmet_schema"):
        old_index = bound_path(raw[label], label=f"raw/{label}")
        if (
            old_index.parent != canonical_raw_roots[label].resolve()
            or old_index.name not in {"snapshot_index.json", "snapshot_index_v2.json"}
        ):
            raise PredictorBridgeError("bridge raw index path is not versioned canonically")
        upgraded = migrate_snapshot_index_metadata_v2(
            SnapshotStore(old_index.parent, offline=True)
        )
        upgraded_bindings[label] = {
            "path": upgraded.relative_to(root).as_posix(),
            "sha256": sha256_file(upgraded),
        }

    registry = bound_path(manifest.get("registry"), label="registry")
    request_map = bound_path(manifest.get("request_map"), label="request_map")
    normalized = manifest.get("normalized")
    if not isinstance(normalized, Mapping) or set(normalized) != {"frozen", "refreshed"}:
        raise PredictorBridgeError("bridge normalized bindings are malformed")
    refreshed_path = bound_path(normalized["refreshed"], label="normalized/refreshed")
    replayed = replay_predictor_bridge_offline(
        registry_path=registry,
        request_map_path=request_map,
        daymet_index_path=root / upgraded_bindings["daymet"]["path"],
        gridmet_index_path=root / upgraded_bindings["gridmet"]["path"],
        gridmet_schema_index_path=root / upgraded_bindings["gridmet_schema"]["path"],
        expected_sites=120,
    )
    assert_exact_predictor_table(
        replayed, pd.read_parquet(refreshed_path),
        label="migrated raw replay versus frozen refreshed predictors",
    )
    updated = {**manifest, "raw_snapshot_indexes": upgraded_bindings}
    atomic_write_bytes(manifest_path, canonical_json_bytes(updated))
    return updated


def validate_development_bridge_manifest_offline(
    *, repo_root: str | Path, manifest_path: str | Path, expected_sites: int = 120,
) -> dict[str, Any]:
    """Fail before model training unless raw bytes reproduce all bridge products."""
    root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json_object(manifest_path, label="development bridge manifest")
    if (
        root not in manifest_path.parents
        or set(manifest) != {
            "format", "status", "outcome_values_requested_or_read",
            "source_tree_sha256", "panel", "registry", "normalized", "report",
            "request_map", "raw_snapshot_indexes", "gate",
        }
        or manifest.get("format") != BRIDGE_FORMAT
        or manifest.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or manifest.get("outcome_values_requested_or_read") is not False
        or manifest.get("gate") != {
            "requires_status": "PASS_EXACT_PRODUCT_BRIDGE",
            "failure_action": (
                "do not freeze or open Route-A models; investigate product drift"
            ),
        }
    ):
        raise PredictorBridgeError("development bridge manifest contract changed")

    def binding(value: object, *, label: str) -> Path:
        return _resolve_manifest_binding(root, value, label=label)

    panel_path = binding(manifest.get("panel"), label="panel")
    registry_path = binding(manifest.get("registry"), label="registry")
    request_map_path = binding(manifest.get("request_map"), label="request_map")
    report_path = binding(manifest.get("report"), label="report")
    normalized = manifest.get("normalized")
    raw = manifest.get("raw_snapshot_indexes")
    if (
        not isinstance(normalized, Mapping)
        or set(normalized) != {"frozen", "refreshed"}
        or not isinstance(raw, Mapping)
        or set(raw) != {"daymet", "gridmet", "gridmet_schema"}
    ):
        raise PredictorBridgeError("development bridge artifact registries changed")
    frozen_path = binding(normalized["frozen"], label="normalized/frozen")
    refreshed_path = binding(normalized["refreshed"], label="normalized/refreshed")
    raw_paths = {
        label: binding(raw[label], label=f"raw/{label}")
        for label in ("daymet", "gridmet", "gridmet_schema")
    }
    expected_raw_paths = {
        "daymet": root / (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/daymet-v1/"
            "snapshot_index_v2.json"
        ),
        "gridmet": root / (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/gridmet-v1/"
            "snapshot_index_v2.json"
        ),
        "gridmet_schema": root / (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-schema-v1/snapshot_index_v2.json"
        ),
    }
    if any(
        raw_paths[label] != expected_raw_paths[label].resolve()
        for label in raw_paths
    ):
        raise PredictorBridgeError("bridge does not bind metadata-byte index v2")
    replayed = replay_predictor_bridge_offline(
        registry_path=registry_path,
        request_map_path=request_map_path,
        daymet_index_path=raw_paths["daymet"],
        gridmet_index_path=raw_paths["gridmet"],
        gridmet_schema_index_path=raw_paths["gridmet_schema"],
        expected_sites=expected_sites,
    )
    registry = pd.read_csv(
        registry_path,
        dtype={"site_no": "string", "legacy_site_id": "string"},
        keep_default_na=False,
    )
    expected_frozen = frozen_bridge_slice(pd.read_parquet(panel_path), registry)
    frozen = assert_exact_predictor_table(
        expected_frozen, pd.read_parquet(frozen_path),
        label="stored frozen predictor table",
    )
    refreshed = assert_exact_predictor_table(
        replayed, pd.read_parquet(refreshed_path),
        label="stored refreshed predictor table",
    )
    expected_report = compare_predictor_bridge(frozen, refreshed)
    if _read_json_object(report_path, label="development bridge report") != expected_report:
        raise PredictorBridgeError("development bridge report is not parser-replay-derived")
    return expected_report


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


def assert_exact_predictor_table(
    expected: pd.DataFrame, observed: pd.DataFrame, *, label: str,
) -> pd.DataFrame:
    """Require identical normalized keys and IEEE-754 values (NaNs included)."""
    left = _normalise_table(expected, label=f"expected {label}")
    right = _normalise_table(observed, label=f"observed {label}")
    if not left[["site_no", "DATE"]].equals(right[["site_no", "DATE"]]):
        raise PredictorBridgeError(f"{label} key registry differs from raw replay")
    for field in BRIDGE_FIELDS:
        if not np.array_equal(
            left[field].to_numpy(dtype="float64"),
            right[field].to_numpy(dtype="float64"),
            equal_nan=True,
        ):
            raise PredictorBridgeError(f"{label}/{field} differs from raw replay")
    return right


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
