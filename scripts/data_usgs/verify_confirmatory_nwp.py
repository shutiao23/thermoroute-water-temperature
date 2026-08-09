#!/usr/bin/env python3
"""Independently verify the label-free Route-A F2a acquisition.

This verifier intentionally does not import the acquisition or NWP parser.  It
reconstructs the frozen station-by-month request inventory and independently
checks the raw-response -> Parquet -> sidecar -> snapshot-index -> manifest
checksum chain.  Its only tabular input outside the acquisition is a metadata
registry read with ``usecols=["site_no", "lat", "lon"]``.  It has no argument
for, and never opens, a water-temperature, flow, panel, prediction, or score
artifact.

The Open-Meteo Previous Runs fields are fixed-valid-time-minus-lead composites.
Passing this verifier must never be described as proving a coherent forecast
initialization, an as-issued operational archive, or promotion of an F2 arm.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data_usgs" / "confirmatory_predictors" / "gfs-previous-runs-v1"
DEFAULT_SNAPSHOT_DIR = ROOT / "data_usgs" / "raw_snapshots" / "openmeteo-gfs-previous-runs-v1"
DEFAULT_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
DEFAULT_ACQUISITION_SCRIPT = ROOT / "scripts" / "data_usgs" / "fetch_confirmatory_nwp.py"
DEFAULT_NWP_MODULE = ROOT / "src" / "thermoroute" / "nwp.py"
DEFAULT_PROVENANCE_MODULE = ROOT / "src" / "thermoroute" / "provenance.py"

EXPECTED_PROTOCOL_ID = "route-a-confirmatory-v1"
EXPECTED_PROVIDER_SLUG = "open-meteo-previous-runs-gfs-global"
EXPECTED_PROVIDER = "Open-Meteo Previous Runs API"
EXPECTED_DOCUMENTATION = "https://open-meteo.com/en/docs/previous-runs-api"
EXPECTED_ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
EXPECTED_UPSTREAM_MODEL = "NOAA NCEP GFS global"
EXPECTED_OPEN_METEO_MODEL = "gfs_global"
EXPECTED_SOURCE_VARIABLE = "temperature_2m"
EXPECTED_UNIT = "degrees_C"
EXPECTED_RAW_UNIT = "°C"
EXPECTED_TIME_UNIT = "iso8601"
EXPECTED_LEADS = (1, 3, 7)
EXPECTED_ARCHIVE_START = date(2021, 3, 23)
EXPECTED_TARGET_START = date(2021, 3, 30)
EXPECTED_TARGET_END = date(2023, 12, 31)
EXPECTED_USER_AGENT = "ThermoRoute/1.0 Route-A predictor acquisition"
EXPECTED_ISSUE_SEMANTICS = (
    "rolling_fixed_valid_time_minus_lead_offset; daily composite available by "
    "23:59 UTC on issue_date; not a single model-run initialization"
)
EXPECTED_COVERAGE_THRESHOLD = 0.90

PARQUET_COLUMNS = (
    "site_no",
    "requested_lat",
    "requested_lon",
    "horizon",
    "issue_date",
    "target_date",
    "air_temp_2m_mean_c",
    "available_hour_count",
    "complete_target_day",
    "lead_field",
    "source_provider",
    "upstream_model",
    "open_meteo_model",
    "issue_semantics",
    "response_grid_lat",
    "response_grid_lon",
    "request_sha256",
    "response_sha256",
)
SIDECAR_FIELDS = {
    "schema_version",
    "artifact",
    "artifact_sha256",
    "site_no",
    "chunk_start",
    "chunk_end",
    "row_count",
    "complete_row_count",
    "request_sha256",
    "response_sha256",
    "retrieved_at_utc",
    "labels_requested_or_read",
    "protocol_sha256",
}
SNAPSHOT_METADATA_FIELDS = {
    "schema_version",
    "request",
    "request_sha256",
    "retrieved_at_utc",
    "http_status",
    "response_headers",
    "byte_count",
    "response_sha256",
    "response_file",
}
SNAPSHOT_INDEX_FIELDS = {"schema_version", "snapshot_count", "records"}
SNAPSHOT_INDEX_RECORD_FIELDS = {
    "provider",
    "request_sha256",
    "response_sha256",
    "retrieved_at_utc",
    "byte_count",
    "request",
    "metadata_path",
    "response_path",
}
MANIFEST_FIELDS = {
    "schema_version",
    "artifact_role",
    "protocol",
    "protocol_sha256",
    "source_provider",
    "source_documentation",
    "upstream_model",
    "open_meteo_model",
    "source_variable",
    "lead_days",
    "lead_fields",
    "issue_semantics",
    "timezone",
    "daily_aggregation",
    "archive_run_start",
    "common_valid_target_start",
    "target_end",
    "station_count",
    "row_count",
    "complete_row_count",
    "request_partition",
    "immutable_outputs",
    "consumed_by_primary_route_a_models",
    "primary_evaluation_dependency",
    "labels_requested_or_read",
    "outcome_endpoint_called",
    "registry_inputs",
    "raw_snapshot_index",
    "raw_snapshot_index_sha256",
    "chunks",
}
HASH_RE = re.compile(r"[0-9a-f]{64}")
CANONICAL_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00")
FORBIDDEN_RAW_KEY_FRAGMENTS = (
    "wtemp",
    "water_temperature",
    "water-temp",
    "stream_temperature",
    "flow",
    "discharge",
    "00010",
    "00060",
    "observation_label",
    "outcome_label",
)
FORBIDDEN_INPUT_PATH_FRAGMENTS = (
    "panel",
    "outcome",
    "prediction",
    "station_metrics",
    "forecast_keys",
    "paired_effects",
)


class NWPVerificationError(RuntimeError):
    """Raised when any acquisition-integrity assertion fails closed."""


@dataclass(frozen=True)
class Station:
    site_no: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Chunk:
    station: Station
    start: date
    end: date
    artifact: Path
    sidecar: Path
    url: str
    request: dict[str, object]
    request_sha256: str
    response_path: Path
    metadata_path: Path


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise NWPVerificationError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable_regular(path: Path, *, label: str) -> bytes:
    """Read one non-symlink, single-link regular file and detect concurrent drift."""
    try:
        before = path.lstat()
    except OSError as exc:
        raise NWPVerificationError(f"{label} is absent: {path}") from exc
    _fail(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {path}")
    _fail(before.st_nlink == 1, f"{label} is not a single-link file: {path}")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise NWPVerificationError(f"{label} changed while read: {path}") from exc
    identity = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    _fail(
        all(getattr(before, field) == getattr(after, field) for field in identity)
        and len(payload) == before.st_size,
        f"{label} changed while read: {path}",
    )
    return payload


def _sha256_file(path: Path, *, label: str) -> str:
    return _sha256_bytes(_read_stable_regular(path, label=label))


def _strict_json_bytes(payload: bytes, *, label: str, canonical: bool) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NWPVerificationError(f"{label} is not strict JSON") from exc
    _fail(isinstance(document, dict), f"{label} must be a JSON object")
    if canonical:
        _fail(payload == _canonical_json_bytes(document), f"{label} is not canonical JSON")
    return document


def _load_json(path: Path, *, label: str, canonical: bool = True) -> dict[str, Any]:
    return _strict_json_bytes(
        _read_stable_regular(path, label=label), label=label, canonical=canonical
    )


def _rel_repo(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT)


def _parse_date(value: object, *, label: str) -> date:
    _fail(type(value) is str, f"{label} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise NWPVerificationError(f"{label} is not an ISO date") from exc
    _fail(parsed.isoformat() == value, f"{label} is not a canonical ISO date")
    return parsed


def _canonical_utc(value: object, *, label: str) -> str:
    _fail(
        type(value) is str and CANONICAL_UTC_RE.fullmatch(value) is not None,
        f"{label} is not a canonical UTC timestamp",
    )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise NWPVerificationError(f"{label} is not a canonical UTC timestamp") from exc
    _fail(
        parsed.utcoffset() == timedelta(0) and parsed.isoformat() == value,
        f"{label} is not a canonical UTC timestamp",
    )
    return value


def _assert_hash(value: object, *, label: str) -> str:
    _fail(
        type(value) is str and HASH_RE.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _assert_no_forbidden_input_path(path: Path, *, role: str) -> None:
    lower = path.name.lower()
    _fail(
        not any(fragment in lower for fragment in FORBIDDEN_INPUT_PATH_FRAGMENTS),
        f"{role} path is forbidden by the no-label boundary: {path}",
    )


def _assert_no_label_keys(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            _fail(
                not any(fragment in lowered for fragment in FORBIDDEN_RAW_KEY_FRAGMENTS),
                f"{label} contains forbidden label-like key {key!r}",
            )
            _assert_no_label_keys(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _assert_no_label_keys(child, label=label)


def _iter_month_chunks(start: date, end: date) -> Iterable[tuple[date, date]]:
    _fail(start <= end, "target window is reversed")
    cursor = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while cursor <= final:
        month_end = cursor + pd.offsets.MonthEnd(0)
        chunk_end = min(month_end, final)
        yield cursor.date(), chunk_end.date()
        cursor = chunk_end + pd.Timedelta(days=1)


def _coordinate(value: float, *, latitude: bool) -> str:
    lower, upper = (-90.0, 90.0) if latitude else (-180.0, 180.0)
    _fail(math.isfinite(value) and lower <= value <= upper, "invalid registry coordinate")
    return f"{value:.6f}"


def _request_url(station: Station, start: date, end: date) -> str:
    fields = ",".join(f"temperature_2m_previous_day{lead}" for lead in EXPECTED_LEADS)
    params = {
        "end_date": end.isoformat(),
        "hourly": fields,
        "latitude": _coordinate(station.lat, latitude=True),
        "longitude": _coordinate(station.lon, latitude=False),
        "models": EXPECTED_OPEN_METEO_MODEL,
        "start_date": start.isoformat(),
        "temperature_unit": "celsius",
        "timezone": "GMT",
    }
    return EXPECTED_ENDPOINT + "?" + urlencode(sorted(params.items()))


def _request_document(url: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": EXPECTED_PROVIDER_SLUG,
        "method": "GET",
        "url": url,
        "headers": {"User-Agent": EXPECTED_USER_AGENT},
    }


def _protocol_contract(
    protocol_path: Path,
    *,
    expected_target_end: date,
) -> tuple[dict[str, Any], str]:
    _assert_no_forbidden_input_path(protocol_path, role="protocol")
    payload = _read_stable_regular(protocol_path, label="Route-A protocol")
    protocol = _strict_json_bytes(payload, label="Route-A protocol", canonical=False)
    protocol_sha = _sha256_bytes(payload)
    _fail(protocol.get("schema_version") == 1, "protocol schema_version changed")
    _fail(protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID, "protocol_id changed")
    secondary = protocol.get("secondary_archived_nwp_contract")
    _fail(isinstance(secondary, dict), "protocol lacks secondary NWP contract")
    expected_secondary = {
        "contains_outcome_labels": False,
        "consumed_by_primary_models": False,
        "primary_evaluation_dependency": False,
        "provider": EXPECTED_PROVIDER,
        "provider_documentation": EXPECTED_DOCUMENTATION,
        "api_endpoint": EXPECTED_ENDPOINT,
        "upstream_model": EXPECTED_UPSTREAM_MODEL,
        "open_meteo_model_parameter": EXPECTED_OPEN_METEO_MODEL,
        "variable": EXPECTED_SOURCE_VARIABLE,
        "unit": EXPECTED_UNIT,
        "lead_days": list(EXPECTED_LEADS),
        "lead_fields": [f"temperature_2m_previous_day{x}" for x in EXPECTED_LEADS],
        "timezone": "GMT_UTC_PLUS_00",
        "daily_aggregation": (
            "arithmetic mean only when all 24 valid-hour values are finite; "
            "otherwise retain an unavailable row and hour count"
        ),
        "archive_run_start": EXPECTED_ARCHIVE_START.isoformat(),
        "secondary_common_lead_target_start": EXPECTED_TARGET_START.isoformat(),
        "request_partition": (
            "one stable site_no and one UTC calendar month per immutable raw snapshot"
        ),
        "raw_snapshot_required": True,
        "derived_blocks_may_be_overwritten": False,
        "selection_or_tuning_from_predictor_availability": False,
    }
    for key, expected in expected_secondary.items():
        _fail(secondary.get(key) == expected, f"protocol NWP field {key!r} changed")
    lead_semantics = secondary.get("lead_semantics")
    _fail(
        type(lead_semantics) is str
        and "rolling fixed-lead composite" in lead_semantics
        and "not the output of one identified model initialization" in lead_semantics,
        "protocol does not preserve fixed-lead composite semantics",
    )
    time_holdout = protocol.get("time_holdout")
    _fail(isinstance(time_holdout, dict), "protocol lacks time_holdout")
    _fail(
        time_holdout.get("secondary_nwp_common_lead_target_start")
        == EXPECTED_TARGET_START.isoformat(),
        "protocol target start changed",
    )
    _fail(
        time_holdout.get("end") == expected_target_end.isoformat(),
        "protocol target end changed",
    )
    return protocol, protocol_sha


def _load_stations(
    registry_paths: Sequence[Path],
    *,
    required_station_count: int | None,
) -> tuple[list[Station], list[dict[str, object]]]:
    _fail(bool(registry_paths), "at least one metadata registry is required")
    records: dict[str, Station] = {}
    lineage: list[dict[str, object]] = []
    for path in registry_paths:
        _assert_no_forbidden_input_path(path, role="registry")
        _fail(path.is_file(), f"registry is absent: {path}")
        # This is the only CSV read in the verifier.  No other registry column is
        # selected or materialized.  Parsing and hashing use the same stable byte
        # read so registry identity cannot drift between those two operations.
        registry_payload = _read_stable_regular(path, label="input registry")
        try:
            frame = pd.read_csv(
                io.BytesIO(registry_payload),
                usecols=["site_no", "lat", "lon"],
                dtype={"site_no": "string"},
                float_precision="round_trip",
            )
        except (ValueError, OSError) as exc:
            raise NWPVerificationError(
                f"registry lacks the label-free identity/coordinate schema: {path}"
            ) from exc
        frame["site_no"] = frame["site_no"].astype("string").str.strip()
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        _fail(not frame.isna().any().any(), f"registry has missing identity/coordinates: {path}")
        _fail(not frame["site_no"].eq("").any(), f"registry has blank site_no: {path}")
        for row in frame.itertuples(index=False):
            station = Station(str(row.site_no), float(row.lat), float(row.lon))
            _coordinate(station.lat, latitude=True)
            _coordinate(station.lon, latitude=False)
            prior = records.get(station.site_no)
            if prior is not None:
                _fail(
                    prior == station,
                    f"site {station.site_no} has conflicting registry coordinates",
                )
            records[station.site_no] = station
        lineage.append(
            {
                "path": _rel_repo(path),
                "sha256": _sha256_bytes(registry_payload),
                "columns_read": ["site_no", "lat", "lon"],
                "row_count": len(frame),
            }
        )
    stations = [records[key] for key in sorted(records)]
    _fail(bool(stations), "metadata registries contain no stations")
    if required_station_count is not None:
        _fail(
            len(stations) == required_station_count,
            f"frozen registry must contain exactly {required_station_count} stations",
        )
    return stations, lineage


def _build_inventory(
    stations: Sequence[Station],
    *,
    output_dir: Path,
    snapshot_dir: Path,
    target_end: date,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for station in stations:
        for start, end in _iter_month_chunks(EXPECTED_TARGET_START, target_end):
            artifact = output_dir / station.site_no / f"{start:%Y-%m}.parquet"
            sidecar = artifact.with_suffix(".provenance.json")
            url = _request_url(station, start, end)
            request = _request_document(url)
            request_sha = _sha256_bytes(_canonical_json_bytes(request))
            raw_root = snapshot_dir / EXPECTED_PROVIDER_SLUG / request_sha
            chunks.append(
                Chunk(
                    station=station,
                    start=start,
                    end=end,
                    artifact=artifact,
                    sidecar=sidecar,
                    url=url,
                    request=request,
                    request_sha256=request_sha,
                    response_path=raw_root / "response.bin",
                    metadata_path=raw_root / "metadata.json",
                )
            )
    return chunks


def _inventory_files(root: Path) -> set[Path]:
    _fail(root.is_dir() and not root.is_symlink(), f"artifact root is absent or unsafe: {root}")
    files: set[Path] = set()
    for path in root.rglob("*"):
        _fail(not path.is_symlink(), f"symlink is forbidden in acquisition evidence: {path}")
        if path.is_file():
            files.add(path)
        else:
            _fail(path.is_dir(), f"non-file/non-directory entry in acquisition evidence: {path}")
    return files


def _validate_exact_inventory(
    chunks: Sequence[Chunk], *, output_dir: Path, snapshot_dir: Path
) -> None:
    expected_output = {output_dir / "manifest.json"}
    expected_snapshot = {snapshot_dir / "snapshot_index.json"}
    for chunk in chunks:
        expected_output.update({chunk.artifact, chunk.sidecar})
        expected_snapshot.update({chunk.response_path, chunk.metadata_path})
    actual_output = _inventory_files(output_dir)
    actual_snapshot = _inventory_files(snapshot_dir)
    missing_output = sorted(str(x) for x in expected_output - actual_output)
    extra_output = sorted(str(x) for x in actual_output - expected_output)
    missing_snapshot = sorted(str(x) for x in expected_snapshot - actual_snapshot)
    extra_snapshot = sorted(str(x) for x in actual_snapshot - expected_snapshot)
    _fail(
        not (missing_output or extra_output or missing_snapshot or extra_snapshot),
        "station-by-month inventory mismatch: "
        f"missing_output={missing_output[:5]}, extra_output={extra_output[:5]}, "
        f"missing_snapshot={missing_snapshot[:5]}, extra_snapshot={extra_snapshot[:5]}",
    )


def _parse_raw_daily(
    payload: bytes,
    *,
    chunk: Chunk,
) -> tuple[list[dict[str, object]], float, float]:
    document = _strict_json_bytes(payload, label="raw Previous Runs response", canonical=False)
    _assert_no_label_keys(document, label="raw Previous Runs response")
    _fail(document.get("timezone") in {"GMT", "UTC"}, "raw response timezone changed")
    offset = document.get("utc_offset_seconds", 0)
    _fail(
        type(offset) in {int, float, str} and not isinstance(offset, bool),
        "raw response UTC offset is invalid",
    )
    try:
        offset_number = int(offset)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NWPVerificationError("raw response UTC offset is invalid") from exc
    _fail(offset_number == 0, "raw response has a non-zero UTC offset")
    hourly = document.get("hourly")
    units = document.get("hourly_units")
    _fail(isinstance(hourly, dict) and isinstance(units, dict), "raw response lacks hourly data")
    fields = [f"temperature_2m_previous_day{x}" for x in EXPECTED_LEADS]
    _fail(set(hourly) == {"time", *fields}, "raw hourly schema changed")
    _fail(set(units) == {"time", *fields}, "raw hourly-units schema changed")
    _fail(units.get("time") == EXPECTED_TIME_UNIT, "raw time unit changed")
    for field in fields:
        _fail(units.get(field) == EXPECTED_RAW_UNIT, f"raw unit changed for {field}")
    raw_times = hourly.get("time")
    _fail(isinstance(raw_times, list), "raw response lacks hourly time list")
    expected_times = pd.date_range(
        pd.Timestamp(chunk.start), pd.Timestamp(chunk.end) + pd.Timedelta(hours=23), freq="h"
    )
    _fail(
        raw_times == expected_times.strftime("%Y-%m-%dT%H:%M").tolist(),
        "raw response does not exactly cover each requested UTC hour",
    )
    response_lat = document.get("latitude")
    response_lon = document.get("longitude")
    _fail(
        type(response_lat) in {int, float}
        and not isinstance(response_lat, bool)
        and type(response_lon) in {int, float}
        and not isinstance(response_lon, bool),
        "raw response grid coordinates are missing",
    )
    grid_lat = float(response_lat)
    grid_lon = float(response_lon)
    _fail(math.isfinite(grid_lat) and -90 <= grid_lat <= 90, "invalid response latitude")
    _fail(math.isfinite(grid_lon) and -180 <= grid_lon <= 180, "invalid response longitude")

    daily: list[dict[str, object]] = []
    days = (chunk.end - chunk.start).days + 1
    for lead in EXPECTED_LEADS:
        field = f"temperature_2m_previous_day{lead}"
        values = hourly.get(field)
        _fail(
            isinstance(values, list) and len(values) == len(raw_times),
            f"raw response array length changed for {field}",
        )
        numeric: list[float | None] = []
        for value in values:
            if value is None:
                numeric.append(None)
                continue
            _fail(
                type(value) in {int, float} and not isinstance(value, bool),
                f"raw response contains a non-numeric value in {field}",
            )
            converted = float(value)
            _fail(math.isfinite(converted), f"raw response contains non-finite {field}")
            numeric.append(converted)
        for offset_day in range(days):
            target = chunk.start + timedelta(days=offset_day)
            block = numeric[offset_day * 24 : (offset_day + 1) * 24]
            finite = [x for x in block if x is not None]
            count = len(finite)
            complete = count == 24
            daily.append(
                {
                    "site_no": chunk.station.site_no,
                    "horizon": lead,
                    "issue_date": target - timedelta(days=lead),
                    "target_date": target,
                    "air_temp_2m_mean_c": sum(finite) / 24.0 if complete else math.nan,
                    "available_hour_count": count,
                    "complete_target_day": complete,
                    "lead_field": field,
                }
            )
    return daily, grid_lat, grid_lon


def _series_exact(frame: pd.DataFrame, column: str, expected: object, *, label: str) -> None:
    _fail(frame[column].map(lambda value: value == expected).all(), f"{label} changed")


def _validate_parquet(
    chunk: Chunk,
    *,
    request_sha: str,
    response_sha: str,
    raw_daily: Sequence[dict[str, object]],
    grid_lat: float,
    grid_lon: float,
) -> tuple[int, dict[int, int], set[tuple[str, int, date, date]]]:
    try:
        frame = pd.read_parquet(chunk.artifact)
    except Exception as exc:
        raise NWPVerificationError(f"cannot read predictor Parquet: {chunk.artifact}") from exc
    _fail(tuple(frame.columns) == PARQUET_COLUMNS, f"predictor schema changed: {chunk.artifact}")
    expected_rows = len(raw_daily)
    _fail(len(frame) == expected_rows, f"predictor row count changed: {chunk.artifact}")
    _fail(pd.api.types.is_integer_dtype(frame["horizon"]), "horizon dtype is not integer")
    _fail(
        pd.api.types.is_integer_dtype(frame["available_hour_count"]),
        "available_hour_count dtype is not integer",
    )
    _fail(
        pd.api.types.is_bool_dtype(frame["complete_target_day"]),
        "complete_target_day dtype is not boolean",
    )
    _fail(
        pd.api.types.is_datetime64_ns_dtype(frame["issue_date"]),
        "issue_date dtype is not datetime64[ns]",
    )
    _fail(
        pd.api.types.is_datetime64_ns_dtype(frame["target_date"]),
        "target_date dtype is not datetime64[ns]",
    )
    _fail(
        pd.api.types.is_float_dtype(frame["air_temp_2m_mean_c"]),
        "air-temperature dtype is not floating",
    )
    _fail(pd.api.types.is_float_dtype(frame["requested_lat"]), "requested_lat dtype changed")
    _fail(pd.api.types.is_float_dtype(frame["requested_lon"]), "requested_lon dtype changed")
    _fail(
        pd.api.types.is_float_dtype(frame["response_grid_lat"]), "response_grid_lat dtype changed"
    )
    _fail(
        pd.api.types.is_float_dtype(frame["response_grid_lon"]), "response_grid_lon dtype changed"
    )
    _fail(
        not frame[["site_no", "horizon", "issue_date", "target_date"]].duplicated().any(),
        "predictor forecast key is duplicated",
    )
    _series_exact(frame, "site_no", chunk.station.site_no, label="site identity")
    _series_exact(frame, "requested_lat", chunk.station.lat, label="requested latitude")
    _series_exact(frame, "requested_lon", chunk.station.lon, label="requested longitude")
    _series_exact(frame, "response_grid_lat", grid_lat, label="response grid latitude")
    _series_exact(frame, "response_grid_lon", grid_lon, label="response grid longitude")
    _series_exact(frame, "source_provider", EXPECTED_PROVIDER, label="provider identity")
    _series_exact(frame, "upstream_model", EXPECTED_UPSTREAM_MODEL, label="upstream model")
    _series_exact(
        frame, "open_meteo_model", EXPECTED_OPEN_METEO_MODEL, label="Open-Meteo model parameter"
    )
    _series_exact(frame, "issue_semantics", EXPECTED_ISSUE_SEMANTICS, label="issue semantics")
    _series_exact(frame, "request_sha256", request_sha, label="request checksum")
    _series_exact(frame, "response_sha256", response_sha, label="response checksum")

    expected = pd.DataFrame.from_records(raw_daily)
    expected["issue_date"] = pd.to_datetime(expected["issue_date"])
    expected["target_date"] = pd.to_datetime(expected["target_date"])
    order = ["site_no", "horizon", "target_date"]
    frame = frame.sort_values(order).reset_index(drop=True)
    expected = expected.sort_values(order).reset_index(drop=True)
    for column in (
        "site_no",
        "horizon",
        "issue_date",
        "target_date",
        "available_hour_count",
        "complete_target_day",
        "lead_field",
    ):
        _fail(
            frame[column].equals(expected[column]), f"derived {column} does not match raw response"
        )
    actual_air = frame["air_temp_2m_mean_c"].to_numpy(dtype=float)
    expected_air = expected["air_temp_2m_mean_c"].to_numpy(dtype=float)
    for actual, wanted in zip(actual_air, expected_air, strict=True):
        if math.isnan(wanted):
            _fail(math.isnan(actual), "incomplete UTC day has a non-missing daily mean")
        else:
            _fail(
                math.isfinite(actual)
                and math.isclose(actual, wanted, abs_tol=1e-12, rel_tol=1e-12),
                "complete UTC-day mean does not match 24 raw hours",
            )
    chronology = frame["target_date"] - pd.to_timedelta(frame["horizon"], unit="D")
    _fail(chronology.equals(frame["issue_date"]), "issue_date != target_date - horizon")
    _fail(set(frame["horizon"].astype(int)) == set(EXPECTED_LEADS), "lead inventory changed")
    expected_fields = frame["horizon"].map(lambda lead: f"temperature_2m_previous_day{int(lead)}")
    _fail(expected_fields.equals(frame["lead_field"]), "lead-field identity changed")
    keys: set[tuple[str, int, date, date]] = set()
    coverage: dict[int, int] = {lead: 0 for lead in EXPECTED_LEADS}
    for row in frame.itertuples(index=False):
        key = (
            str(row.site_no),
            int(row.horizon),
            pd.Timestamp(row.issue_date).date(),
            pd.Timestamp(row.target_date).date(),
        )
        _fail(key not in keys, "forecast key duplicated within Parquet")
        keys.add(key)
        coverage[int(row.horizon)] += int(bool(row.complete_target_day))
    return int(frame["complete_target_day"].sum()), coverage, keys


def _validate_snapshot(
    chunk: Chunk,
) -> tuple[dict[str, Any], bytes, str]:
    metadata_payload = _read_stable_regular(chunk.metadata_path, label="raw snapshot metadata")
    metadata = _strict_json_bytes(metadata_payload, label="raw snapshot metadata", canonical=True)
    _fail(set(metadata) == SNAPSHOT_METADATA_FIELDS, "snapshot metadata schema changed")
    _fail(metadata.get("schema_version") == 1, "snapshot metadata version changed")
    _fail(metadata.get("request") == chunk.request, "snapshot request identity changed")
    _fail(
        metadata.get("request_sha256") == chunk.request_sha256, "snapshot request checksum changed"
    )
    _fail(metadata.get("response_file") == "response.bin", "snapshot response filename changed")
    _fail(
        type(metadata.get("http_status")) is int and metadata["http_status"] == 200,
        "snapshot HTTP status is not 200",
    )
    _fail(
        isinstance(metadata.get("response_headers"), dict), "snapshot response headers are invalid"
    )
    retrieved = _canonical_utc(metadata.get("retrieved_at_utc"), label="snapshot retrieval time")
    response = _read_stable_regular(chunk.response_path, label="raw snapshot response")
    response_sha = _sha256_bytes(response)
    _assert_hash(metadata.get("response_sha256"), label="snapshot response checksum")
    _fail(metadata.get("response_sha256") == response_sha, "raw response checksum mismatch")
    _fail(metadata.get("byte_count") == len(response), "raw response byte count mismatch")
    _assert_no_label_keys(metadata.get("request"), label="raw snapshot request")
    _fail(
        not any(x in chunk.url.lower() for x in FORBIDDEN_RAW_KEY_FRAGMENTS),
        "raw request URL crosses the no-label boundary",
    )
    return metadata, response, retrieved


def _validate_sidecar(
    chunk: Chunk,
    *,
    protocol_sha: str,
    artifact_sha: str,
    response_sha: str,
    retrieved: str,
    expected_rows: int,
    complete_rows: int,
) -> dict[str, Any]:
    sidecar = _load_json(chunk.sidecar, label="predictor sidecar", canonical=True)
    _fail(set(sidecar) == SIDECAR_FIELDS, "predictor sidecar schema changed")
    expected = {
        "schema_version": 1,
        "artifact": _rel_repo(chunk.artifact),
        "artifact_sha256": artifact_sha,
        "site_no": chunk.station.site_no,
        "chunk_start": chunk.start.isoformat(),
        "chunk_end": chunk.end.isoformat(),
        "row_count": expected_rows,
        "complete_row_count": complete_rows,
        "request_sha256": chunk.request_sha256,
        "response_sha256": response_sha,
        "retrieved_at_utc": retrieved,
        "labels_requested_or_read": False,
        "protocol_sha256": protocol_sha,
    }
    _fail(sidecar == expected, f"predictor sidecar identity mismatch: {chunk.sidecar}")
    return sidecar


def _validate_snapshot_index(
    path: Path,
    *,
    snapshot_dir: Path,
    snapshots: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    payload = _read_stable_regular(path, label="raw snapshot index")
    document = _strict_json_bytes(payload, label="raw snapshot index", canonical=True)
    _fail(set(document) == SNAPSHOT_INDEX_FIELDS, "snapshot index schema changed")
    _fail(document.get("schema_version") == 1, "snapshot index version changed")
    records = document.get("records")
    _fail(isinstance(records, list), "snapshot index records are invalid")
    _fail(
        document.get("snapshot_count") == len(snapshots) == len(records),
        "snapshot index count changed",
    )
    observed: dict[str, dict[str, Any]] = {}
    for record in records:
        _fail(
            isinstance(record, dict) and set(record) == SNAPSHOT_INDEX_RECORD_FIELDS,
            "snapshot index record schema changed",
        )
        request_sha = _assert_hash(record.get("request_sha256"), label="indexed request checksum")
        _fail(request_sha not in observed, "snapshot index duplicates a request")
        metadata = snapshots.get(request_sha)
        _fail(metadata is not None, "snapshot index contains an unregistered request")
        expected = {
            "provider": EXPECTED_PROVIDER_SLUG,
            "request_sha256": request_sha,
            "response_sha256": metadata["response_sha256"],
            "retrieved_at_utc": metadata["retrieved_at_utc"],
            "byte_count": metadata["byte_count"],
            "request": metadata["request"],
            "metadata_path": str(
                (snapshot_dir / EXPECTED_PROVIDER_SLUG / request_sha / "metadata.json").relative_to(
                    snapshot_dir
                )
            ),
            "response_path": str(
                (snapshot_dir / EXPECTED_PROVIDER_SLUG / request_sha / "response.bin").relative_to(
                    snapshot_dir
                )
            ),
        }
        _fail(record == expected, "snapshot index record does not match raw metadata")
        observed[request_sha] = record
    _fail(set(observed) == set(snapshots), "snapshot index request set is incomplete")
    return document, _sha256_bytes(payload)


def _validate_manifest(
    path: Path,
    *,
    protocol_path: Path,
    protocol_sha: str,
    registry_lineage: Sequence[dict[str, object]],
    snapshot_index_path: Path,
    snapshot_index_sha: str,
    chunks: Sequence[Chunk],
    sidecars: Sequence[dict[str, Any]],
    station_count: int,
    row_count: int,
    complete_row_count: int,
    target_end: date,
) -> tuple[dict[str, Any], str]:
    payload = _read_stable_regular(path, label="NWP acquisition manifest")
    manifest = _strict_json_bytes(payload, label="NWP acquisition manifest", canonical=True)
    _fail(set(manifest) == MANIFEST_FIELDS, "NWP manifest schema changed")
    expected_scalars = {
        "schema_version": 1,
        "artifact_role": "OPTIONAL_SECONDARY_NWP_AVAILABILITY_SENSITIVITY",
        "protocol": _rel_repo(protocol_path),
        "protocol_sha256": protocol_sha,
        "source_provider": EXPECTED_PROVIDER,
        "source_documentation": EXPECTED_DOCUMENTATION,
        "upstream_model": EXPECTED_UPSTREAM_MODEL,
        "open_meteo_model": EXPECTED_OPEN_METEO_MODEL,
        "source_variable": EXPECTED_SOURCE_VARIABLE,
        "lead_days": list(EXPECTED_LEADS),
        "lead_fields": [f"temperature_2m_previous_day{x}" for x in EXPECTED_LEADS],
        "issue_semantics": EXPECTED_ISSUE_SEMANTICS,
        "timezone": "GMT (UTC+00:00)",
        "daily_aggregation": (
            "arithmetic mean of exactly 24 finite valid-hour values; incomplete "
            "days retained with NaN predictor and availability count"
        ),
        "archive_run_start": EXPECTED_ARCHIVE_START.isoformat(),
        "common_valid_target_start": EXPECTED_TARGET_START.isoformat(),
        "target_end": target_end.isoformat(),
        "station_count": station_count,
        "row_count": row_count,
        "complete_row_count": complete_row_count,
        "request_partition": "one stable site_no by one UTC calendar month",
        "immutable_outputs": True,
        "consumed_by_primary_route_a_models": False,
        "primary_evaluation_dependency": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "registry_inputs": list(registry_lineage),
        "raw_snapshot_index": _rel_repo(snapshot_index_path),
        "raw_snapshot_index_sha256": snapshot_index_sha,
        "chunks": list(sidecars),
    }
    _fail(manifest == expected_scalars, "NWP manifest does not close the evidence chain")
    _fail(len(chunks) == len(sidecars), "NWP manifest chunk count changed")
    return manifest, _sha256_bytes(payload)


def verify_acquisition(
    *,
    output_dir: Path,
    snapshot_dir: Path,
    registry_paths: Sequence[Path],
    protocol_path: Path,
    acquisition_script: Path = DEFAULT_ACQUISITION_SCRIPT,
    nwp_module: Path = DEFAULT_NWP_MODULE,
    provenance_module: Path = DEFAULT_PROVENANCE_MODULE,
    verifier_script: Path | None = None,
    _expected_target_end: date = EXPECTED_TARGET_END,
    _required_station_count: int | None = 120,
) -> dict[str, Any]:
    """Verify a complete acquisition and return a non-promoting evidence report.

    The two underscored parameters are compact synthetic-test seams.  The CLI
    never exposes them and always enforces the full 120-site, 2021-03-30 through
    2023-12-31 inventory.
    """
    output_dir = output_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    protocol_path = protocol_path.resolve()
    registry_paths = [path.resolve() for path in registry_paths]
    code_paths = {
        "acquisition_script": acquisition_script.resolve(),
        "nwp_contract_module": nwp_module.resolve(),
        "snapshot_store_module": provenance_module.resolve(),
        "independent_verifier": (verifier_script or Path(__file__)).resolve(),
    }
    for role, path in code_paths.items():
        _assert_no_forbidden_input_path(path, role=role)
        _fail(path.is_file(), f"code input is absent: {path}")
    code_hashes_at_start = {
        role: _sha256_file(path, label=role) for role, path in code_paths.items()
    }
    _fail(output_dir != snapshot_dir, "output and snapshot roots must differ")
    _protocol, protocol_sha = _protocol_contract(
        protocol_path, expected_target_end=_expected_target_end
    )
    stations, registry_lineage = _load_stations(
        registry_paths, required_station_count=_required_station_count
    )
    chunks = _build_inventory(
        stations,
        output_dir=output_dir,
        snapshot_dir=snapshot_dir,
        target_end=_expected_target_end,
    )
    _validate_exact_inventory(chunks, output_dir=output_dir, snapshot_dir=snapshot_dir)

    sidecars: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    all_keys: set[tuple[str, int, date, date]] = set()
    coverage_counts: dict[tuple[str, int], int] = {
        (station.site_no, lead): 0 for station in stations for lead in EXPECTED_LEADS
    }
    total_rows = 0
    total_complete = 0
    for chunk in chunks:
        metadata, raw_payload, retrieved = _validate_snapshot(chunk)
        snapshots[chunk.request_sha256] = metadata
        raw_daily, grid_lat, grid_lon = _parse_raw_daily(raw_payload, chunk=chunk)
        artifact_sha = _sha256_file(chunk.artifact, label="predictor Parquet")
        complete, per_lead, keys = _validate_parquet(
            chunk,
            request_sha=chunk.request_sha256,
            response_sha=str(metadata["response_sha256"]),
            raw_daily=raw_daily,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
        )
        _fail(all_keys.isdisjoint(keys), "forecast key duplicated across month chunks")
        all_keys.update(keys)
        for lead, count in per_lead.items():
            coverage_counts[(chunk.station.site_no, lead)] += count
        total_rows += len(raw_daily)
        total_complete += complete
        sidecars.append(
            _validate_sidecar(
                chunk,
                protocol_sha=protocol_sha,
                artifact_sha=artifact_sha,
                response_sha=str(metadata["response_sha256"]),
                retrieved=retrieved,
                expected_rows=len(raw_daily),
                complete_rows=complete,
            )
        )

    expected_days = (_expected_target_end - EXPECTED_TARGET_START).days + 1
    expected_keys = len(stations) * len(EXPECTED_LEADS) * expected_days
    _fail(total_rows == expected_keys == len(all_keys), "global forecast-key inventory changed")
    snapshot_index_path = snapshot_dir / "snapshot_index.json"
    _snapshot_index, snapshot_index_sha = _validate_snapshot_index(
        snapshot_index_path, snapshot_dir=snapshot_dir, snapshots=snapshots
    )
    manifest_path = output_dir / "manifest.json"
    _manifest, manifest_sha = _validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
        protocol_sha=protocol_sha,
        registry_lineage=registry_lineage,
        snapshot_index_path=snapshot_index_path,
        snapshot_index_sha=snapshot_index_sha,
        chunks=chunks,
        sidecars=sidecars,
        station_count=len(stations),
        row_count=total_rows,
        complete_row_count=total_complete,
        target_end=_expected_target_end,
    )
    # A full verification can take minutes on the complete inventory.  Recheck
    # the closed-world inventory and every identity-bearing non-acquisition input
    # before returning so an edit during verification cannot be silently bound.
    _validate_exact_inventory(chunks, output_dir=output_dir, snapshot_dir=snapshot_dir)
    _fail(
        _sha256_file(protocol_path, label="Route-A protocol recheck") == protocol_sha,
        "Route-A protocol changed during verification",
    )
    for record, path in zip(registry_lineage, registry_paths, strict=True):
        _fail(
            _sha256_file(path, label="input registry recheck") == record["sha256"],
            f"input registry changed during verification: {path}",
        )
    for role, path in code_paths.items():
        _fail(
            _sha256_file(path, label=f"{role} recheck") == code_hashes_at_start[role],
            f"code input changed during verification: {path}",
        )

    coverage_rows: list[dict[str, object]] = []
    for station in stations:
        for lead in EXPECTED_LEADS:
            count = coverage_counts[(station.site_no, lead)]
            fraction = count / expected_days
            coverage_rows.append(
                {
                    "site_no": station.site_no,
                    "lead_days": lead,
                    "expected_target_days": expected_days,
                    "complete_target_days": count,
                    "coverage_fraction": fraction,
                    "passes_0_90_gate": fraction >= EXPECTED_COVERAGE_THRESHOLD,
                }
            )
    per_lead: list[dict[str, object]] = []
    for lead in EXPECTED_LEADS:
        lead_complete = sum(coverage_counts[(station.site_no, lead)] for station in stations)
        denominator = len(stations) * expected_days
        passing = sum(
            coverage_counts[(station.site_no, lead)] / expected_days >= EXPECTED_COVERAGE_THRESHOLD
            for station in stations
        )
        per_lead.append(
            {
                "lead_days": lead,
                "expected_station_days": denominator,
                "complete_station_days": lead_complete,
                "coverage_fraction": lead_complete / denominator,
                "stations_passing_0_90_gate": passing,
                "station_count": len(stations),
            }
        )
    coverage_gate_pass = all(row["passes_0_90_gate"] for row in coverage_rows)
    code_hashes = {
        role: {"path": _rel_repo(path), "sha256": code_hashes_at_start[role]}
        for role, path in sorted(code_paths.items())
    }
    report = {
        "schema_version": 1,
        "verifier_id": "route-a-f2a-independent-acquisition-verifier-v1",
        "status": (
            "VERIFIED_COMPLETE_COVERAGE_GATE_PASSED"
            if coverage_gate_pass
            else "VERIFIED_COMPLETE_COVERAGE_GATE_FAILED"
        ),
        "integrity_verification_passed": True,
        "coverage_gate_threshold": EXPECTED_COVERAGE_THRESHOLD,
        "coverage_gate_passed": coverage_gate_pass,
        "arm_promoted": False,
        "promotion_requires_separate_protocol_authorization": True,
        "contains_or_reads_outcome_labels": False,
        "input_access_boundary": {
            "registry_columns_read": ["site_no", "lat", "lon"],
            "water_temperature_outcomes_read": False,
            "flow_outcomes_read": False,
            "panel_artifacts_read": False,
        },
        "information_semantics": {
            "fixed_lead_composite": True,
            "coherent_single_initialization": False,
            "as_issued_operational_forecast_archive": False,
            "issue_semantics": EXPECTED_ISSUE_SEMANTICS,
        },
        "inventory": {
            "target_start": EXPECTED_TARGET_START.isoformat(),
            "target_end": _expected_target_end.isoformat(),
            "lead_days": list(EXPECTED_LEADS),
            "station_count": len(stations),
            "calendar_months_per_station": len(
                list(_iter_month_chunks(EXPECTED_TARGET_START, _expected_target_end))
            ),
            "station_month_chunks": len(chunks),
            "forecast_rows": total_rows,
            "complete_forecast_rows": total_complete,
        },
        "coverage_by_station_and_lead": coverage_rows,
        "coverage_by_lead": per_lead,
        "evidence_hashes": {
            "protocol": {"path": _rel_repo(protocol_path), "sha256": protocol_sha},
            "input_registries": registry_lineage,
            "code": code_hashes,
            "acquisition_manifest": {
                "path": _rel_repo(manifest_path),
                "sha256": manifest_sha,
            },
            "snapshot_index": {
                "path": _rel_repo(snapshot_index_path),
                "sha256": snapshot_index_sha,
            },
        },
    }
    return report


def partial_inventory_diagnostic(
    *,
    output_dir: Path,
    snapshot_dir: Path,
    registry_paths: Sequence[Path],
    protocol_path: Path,
    _expected_target_end: date = EXPECTED_TARGET_END,
    _required_station_count: int | None = 120,
) -> dict[str, Any]:
    """Return a read-only, explicitly non-authoritative completion diagnostic."""
    output_dir = output_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    protocol_path = protocol_path.resolve()
    _protocol, protocol_sha = _protocol_contract(
        protocol_path, expected_target_end=_expected_target_end
    )
    stations, registry_lineage = _load_stations(
        [path.resolve() for path in registry_paths],
        required_station_count=_required_station_count,
    )
    chunks = _build_inventory(
        stations,
        output_dir=output_dir,
        snapshot_dir=snapshot_dir,
        target_end=_expected_target_end,
    )
    output_complete = sum(chunk.artifact.is_file() and chunk.sidecar.is_file() for chunk in chunks)
    output_partial = sum(chunk.artifact.exists() != chunk.sidecar.exists() for chunk in chunks)
    snapshot_complete = sum(
        chunk.response_path.is_file() and chunk.metadata_path.is_file() for chunk in chunks
    )
    snapshot_partial = sum(
        chunk.response_path.exists() != chunk.metadata_path.exists() for chunk in chunks
    )
    return {
        "schema_version": 1,
        "status": "PARTIAL_INVENTORY_DIAGNOSTIC_NON_AUTHORITATIVE",
        "integrity_verification_passed": False,
        "coverage_gate_evaluated": False,
        "arm_promoted": False,
        "must_not_be_used_as_final_evidence": True,
        "expected_station_month_chunks": len(chunks),
        "complete_parquet_sidecar_pairs_observed": output_complete,
        "partial_parquet_sidecar_pairs_observed": output_partial,
        "complete_raw_snapshot_pairs_observed": snapshot_complete,
        "partial_raw_snapshot_pairs_observed": snapshot_partial,
        "manifest_present": (output_dir / "manifest.json").is_file(),
        "snapshot_index_present": (snapshot_dir / "snapshot_index.json").is_file(),
        "protocol_sha256": protocol_sha,
        "input_registries": registry_lineage,
        "information_semantics": {
            "fixed_lead_composite": True,
            "coherent_single_initialization": False,
            "as_issued_operational_forecast_archive": False,
        },
    }


def _atomic_create_report(path: Path, report: Mapping[str, object]) -> None:
    payload = _canonical_json_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise NWPVerificationError(f"refusing to overwrite verification report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise NWPVerificationError(f"refusing to overwrite verification report: {path}")
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--registry", type=Path, action="append")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--partial-diagnostic",
        action="store_true",
        help="print read-only inventory counts; never verifies integrity or coverage",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="create a canonical report outside the immutable acquisition roots",
    )
    args = parser.parse_args(argv)
    registries = args.registry or [DEFAULT_REGISTRY]
    if args.partial_diagnostic:
        if args.report is not None:
            parser.error("--partial-diagnostic cannot create --report")
        report = partial_inventory_diagnostic(
            output_dir=args.output_dir,
            snapshot_dir=args.snapshot_dir,
            registry_paths=registries,
            protocol_path=args.protocol,
        )
    else:
        report = verify_acquisition(
            output_dir=args.output_dir,
            snapshot_dir=args.snapshot_dir,
            registry_paths=registries,
            protocol_path=args.protocol,
        )
        if args.report is not None:
            if _path_is_within(args.report, args.output_dir) or _path_is_within(
                args.report, args.snapshot_dir
            ):
                parser.error("--report must remain outside both immutable acquisition roots")
            _atomic_create_report(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("coverage_gate_passed", True) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NWPVerificationError as exc:
        print(f"verification failed closed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
