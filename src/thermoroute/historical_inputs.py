"""Outcome-free retrospective meteorology for the sealed Route-A opening.

This module deliberately knows only stable station identity, coordinates and
the five meteorological predictors consumed by the frozen model schema.  It
does not import, read or request an NWIS daily-value endpoint.  Raw Daymet and
gridMET bytes remain content-addressed in :class:`SnapshotStore`; normalized
tables retain the complete site-by-calendar key even when a source value is
missing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
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


def _lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_repository_path(
    repo_root: Path,
    path: str | Path,
    *,
    label: str,
    final_kind: str | None = None,
    allow_absent_suffix: bool = False,
) -> Path:
    """Preserve a lexical repository path and reject every symlink component."""
    root = repo_root.resolve()
    lexical = _lexical(path)
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise HistoricalInputError(f"{label} escapes repository: {lexical}") from exc
    current = root
    absent = False
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            absent = True
            if not allow_absent_suffix:
                raise HistoricalInputError(f"{label} is absent: {current}")
            continue
        if absent:
            raise HistoricalInputError(f"{label} has a noncanonical path state")
        if stat.S_ISLNK(info.st_mode):
            raise HistoricalInputError(f"{label} path contains a symlink: {current}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise HistoricalInputError(f"{label} crosses a non-directory: {current}")
        if index == len(relative.parts) - 1:
            if final_kind == "file" and (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            ):
                raise HistoricalInputError(
                    f"{label} is not a single-link regular file: {current}"
                )
            if final_kind == "directory" and not stat.S_ISDIR(info.st_mode):
                raise HistoricalInputError(f"{label} is not a directory: {current}")
    if not relative.parts and final_kind == "directory":
        info = root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HistoricalInputError(f"{label} repository root is unsafe")
    return lexical


def _assert_single_link_file(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise HistoricalInputError(f"{label} is absent: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise HistoricalInputError(
            f"{label} is not a single-link regular file: {path}"
        )
    return info


def _relative(repo_root: Path, path: Path) -> str:
    root = repo_root.resolve()
    lexical = _lexical(path)
    try:
        return lexical.relative_to(root).as_posix()
    except ValueError as exc:
        raise HistoricalInputError(
            f"artifact is outside repository root: {lexical}"
        ) from exc


def _binding(repo_root: Path, path: Path) -> dict[str, str]:
    _assert_repository_path(
        repo_root, path, label="bound artifact", final_kind="file"
    )
    return {"path": _relative(repo_root, path), "sha256": sha256_file(path)}


def _exclusive_create(path: Path, payload: bytes) -> None:
    """Publish immutable bytes without replacement or a temporary overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
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
    final = _assert_single_link_file(path, label="published immutable artifact")
    if final.st_size != len(payload) or sha256_file(path) != sha256_bytes(payload):
        raise HistoricalInputError(f"immutable artifact changed during publication: {path}")
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


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
        float_precision="round_trip",
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


def _request_identity(
    *, provider: str, url: str, headers: Mapping[str, str]
) -> tuple[str, dict[str, object]]:
    _assert_safe_meteorology_url(url)
    document = SnapshotStore.request_document(
        provider=provider, url=url, headers=headers
    )
    return sha256_bytes(canonical_json_bytes(document)), document


def _expected_raw_plan(
    *,
    protocol: Mapping[str, Any],
    registries: tuple[RegistryEvidence, RegistryEvidence],
    daymet_store: SnapshotStore,
    gridmet_store: SnapshotStore,
    gridmet_schema_store: SnapshotStore,
    headers: Mapping[str, str],
) -> list[tuple[SnapshotStore, str, str, str, dict[str, object]]]:
    """Return the one canonical acquisition order without reading any outcome."""
    schema_url = build_gridmet_wind_metadata_url()
    schema_sha, schema_request = _request_identity(
        provider=GRIDMET_SCHEMA_PROVIDER, url=schema_url, headers=headers
    )
    plan = [
        (
            gridmet_schema_store,
            GRIDMET_SCHEMA_PROVIDER,
            schema_url,
            schema_sha,
            schema_request,
        )
    ]
    start_text = protocol["history_start"].strftime("%Y-%m-%d")
    end_text = protocol["target_end"].strftime("%Y-%m-%d")
    for registry in registries:
        for station in registry.frame.itertuples(index=False):
            lat, lon = float(station.lat), float(station.lon)
            for store, provider, url in (
                (
                    daymet_store,
                    DAYMET_PROVIDER,
                    build_daymet_url(lat, lon, start_text, end_text),
                ),
                (
                    gridmet_store,
                    GRIDMET_PROVIDER,
                    build_gridmet_wind_url(lat, lon, start_text, end_text),
                ),
            ):
                request_sha, request = _request_identity(
                    provider=provider, url=url, headers=headers
                )
                plan.append((store, provider, url, request_sha, request))
    return plan


def _validate_snapshot_store_namespace(
    store: SnapshotStore,
    *,
    expected_requests: Mapping[str, tuple[str, Mapping[str, object]]],
    allow_incomplete: bool,
) -> dict[str, int]:
    """Validate an exact raw-store namespace and return 0/1/2 request states."""
    states = {request_sha: 0 for request_sha in expected_requests}
    if not os.path.lexists(store.root):
        if allow_incomplete:
            return states
        raise HistoricalInputError(f"raw snapshot store is absent: {store.root}")
    root_info = store.root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise HistoricalInputError(f"raw snapshot store root is unsafe: {store.root}")
    providers = {
        SnapshotStore._provider_name(provider)
        for provider, _request in expected_requests.values()
    }
    if len(providers) != 1:
        raise HistoricalInputError("one snapshot store received multiple providers")
    provider_name = next(iter(providers))
    allowed_top = {provider_name, "snapshot_index.json", "snapshot_index_v2.json"}
    top_entries = {entry.name: entry for entry in store.root.iterdir()}
    if set(top_entries) - allowed_top:
        raise HistoricalInputError(
            f"raw snapshot store contains an extra entry: {store.root}"
        )
    for index_name in ("snapshot_index.json", "snapshot_index_v2.json"):
        if index_name in top_entries:
            _assert_single_link_file(
                top_entries[index_name], label=f"raw {index_name}"
            )
    provider_path = store.root / provider_name
    if not os.path.lexists(provider_path):
        if any(name.startswith("snapshot_index") for name in top_entries):
            raise HistoricalInputError("raw index exists without its provider namespace")
        if allow_incomplete:
            return states
        raise HistoricalInputError(f"raw provider namespace is absent: {provider_path}")
    provider_info = provider_path.lstat()
    if stat.S_ISLNK(provider_info.st_mode) or not stat.S_ISDIR(provider_info.st_mode):
        raise HistoricalInputError(f"raw provider namespace is unsafe: {provider_path}")
    observed_requests = {entry.name: entry for entry in provider_path.iterdir()}
    if set(observed_requests) - set(expected_requests):
        raise HistoricalInputError("raw snapshot store contains an extra request")
    for request_sha, (provider, expected_request) in expected_requests.items():
        request_path = provider_path / request_sha
        if request_sha not in observed_requests:
            continue
        info = request_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HistoricalInputError(f"raw request namespace is unsafe: {request_path}")
        entries = {entry.name: entry for entry in request_path.iterdir()}
        if set(entries) - {"response.bin", "metadata.json"}:
            raise HistoricalInputError(
                f"raw request namespace contains an extra entry: {request_path}"
            )
        response_present = "response.bin" in entries
        metadata_present = "metadata.json" in entries
        if metadata_present and not response_present:
            raise HistoricalInputError("raw snapshot metadata exists without response bytes")
        if response_present:
            _assert_single_link_file(
                entries["response.bin"], label="raw snapshot response"
            )
            states[request_sha] = 1
        if response_present and metadata_present:
            try:
                _payload, record = store._read_verified(
                    entries["response.bin"], entries["metadata.json"], request_sha
                )
                metadata_bytes = entries["metadata.json"].read_bytes()
                metadata = json.loads(metadata_bytes.decode("utf-8"))
            except Exception as exc:
                if isinstance(exc, HistoricalInputError):
                    raise
                raise HistoricalInputError(
                    f"raw snapshot cannot be verified: {request_path}"
                ) from exc
            if (
                metadata_bytes != canonical_json_bytes(metadata)
                or metadata.get("request") != expected_request
                or record.provider != SnapshotStore._provider_name(provider)
            ):
                raise HistoricalInputError(
                    f"raw snapshot request identity changed: {request_path}"
                )
            states[request_sha] = 2
        if states[request_sha] != 2 and not allow_incomplete:
            raise HistoricalInputError(f"raw snapshot is incomplete: {request_path}")
    if "snapshot_index_v2.json" in top_entries and "snapshot_index.json" not in top_entries:
        raise HistoricalInputError("raw v2 index exists without immutable v1 index")
    if "snapshot_index.json" in top_entries and any(state != 2 for state in states.values()):
        raise HistoricalInputError("raw snapshot index exists before request completion")
    if "snapshot_index.json" in top_entries:
        expected_v1 = canonical_json_bytes(_build_snapshot_index(store))
        if top_entries["snapshot_index.json"].read_bytes() != expected_v1:
            raise HistoricalInputError(
                f"immutable snapshot index changed: {top_entries['snapshot_index.json']}"
            )
    if "snapshot_index_v2.json" in top_entries:
        expected_v2 = canonical_json_bytes(
            _build_snapshot_index(store, bind_metadata_bytes=True)
        )
        if top_entries["snapshot_index_v2.json"].read_bytes() != expected_v2:
            raise HistoricalInputError(
                f"immutable v2 snapshot index changed: {top_entries['snapshot_index_v2.json']}"
            )
    return states


def _assert_raw_prefix(
    plan: list[tuple[SnapshotStore, str, str, str, dict[str, object]]],
    states_by_root: Mapping[Path, Mapping[str, int]],
) -> None:
    """Accept only complete* then at most response-only then absent*."""
    observed_states: list[int] = []
    seen: set[tuple[Path, str]] = set()
    for store, _provider, _url, request_sha, _request in plan:
        key = (store.root, request_sha)
        if key in seen:
            continue
        seen.add(key)
        observed_states.append(int(states_by_root[store.root][request_sha]))
    first_incomplete = next(
        (index for index, value in enumerate(observed_states) if value != 2),
        len(observed_states),
    )
    suffix = observed_states[first_incomplete:]
    if suffix:
        if suffix[0] not in {0, 1} or any(value != 0 for value in suffix[1:]):
            raise HistoricalInputError("raw snapshot cache is not a resumable prefix")


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
    resume_incomplete: bool = False,
    fault_injector: Callable[[str, Path], object] | None = None,
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
            resume_incomplete=resume_incomplete,
            _fault_injector=fault_injector,
        )
        gridmet_payload, gridmet_record = gridmet_store.fetch(
            provider=GRIDMET_PROVIDER,
            url=gridmet_url,
            headers=headers,
            retries=retries,
            resume_incomplete=resume_incomplete,
            _fault_injector=fault_injector,
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
            response, _record = store._read_verified(
                response_path, metadata_path, metadata_path.parent.name
            )
        except Exception as exc:
            raise HistoricalInputError(f"incomplete raw snapshot: {metadata_path}") from exc
        if metadata_bytes != canonical_json_bytes(metadata):
            raise HistoricalInputError(
                f"raw snapshot metadata is not canonical: {metadata_path}"
            )
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
    allow_create: bool = True,
    fault_stage: str | None = None,
    fault_injector: Callable[[str, Path], object] | None = None,
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
    if os.path.lexists(path):
        _assert_single_link_file(path, label="immutable snapshot index")
        if path.read_bytes() != payload:
            raise HistoricalInputError(f"immutable snapshot index changed: {path}")
        return path
    if not allow_create:
        raise HistoricalInputError(f"required immutable snapshot index is absent: {path}")
    _exclusive_create(path, payload)
    if fault_injector is not None and fault_stage is not None:
        fault_injector(fault_stage, path)
    return path


def _validate_optional_snapshot_index_v2(store: SnapshotStore) -> None:
    path = store.root / "snapshot_index_v2.json"
    if not os.path.lexists(path):
        return
    _assert_single_link_file(path, label="immutable v2 snapshot index")
    expected = canonical_json_bytes(
        _build_snapshot_index(store, bind_metadata_bytes=True)
    )
    if path.read_bytes() != expected:
        raise HistoricalInputError(f"immutable v2 snapshot index changed: {path}")


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


_TEMPORAL_TABLE_NAME = "temporal_retrospective_meteorology_v1.parquet"
_EXTERNAL_TABLE_NAME = "external_retrospective_meteorology_v1.parquet"
_REQUEST_MAP_NAME = "source_request_map_v1.json"
_BUNDLE_NAMES = (
    _TEMPORAL_TABLE_NAME,
    _EXTERNAL_TABLE_NAME,
    _REQUEST_MAP_NAME,
)


def _request_map_document(
    *,
    protocol: Mapping[str, Any],
    request_records: list[dict[str, Any]],
    gridmet_contract: Mapping[str, Any],
    schema_record: Any,
) -> dict[str, Any]:
    return {
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


def _bundle_payloads(
    *, cohort_frames: Mapping[str, pd.DataFrame], request_map: Mapping[str, Any]
) -> dict[str, bytes]:
    return {
        _TEMPORAL_TABLE_NAME: _parquet_bytes(cohort_frames["temporal"]),
        _EXTERNAL_TABLE_NAME: _parquet_bytes(cohort_frames["external"]),
        _REQUEST_MAP_NAME: canonical_json_bytes(request_map),
    }


def _assert_no_bundle_staging(output_dir: Path) -> None:
    parent = output_dir.parent
    if not parent.exists():
        return
    prefix = f".{output_dir.name}."
    leftovers = sorted(path.name for path in parent.iterdir() if path.name.startswith(prefix))
    if leftovers:
        raise HistoricalInputError(
            f"normalized bundle has interrupted staging evidence: {leftovers}"
        )


def _validate_bundle_structure(output_dir: Path, *, required: bool) -> bool:
    _assert_no_bundle_staging(output_dir)
    if not os.path.lexists(output_dir):
        if required:
            raise HistoricalInputError(f"normalized output bundle is absent: {output_dir}")
        return False
    info = output_dir.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise HistoricalInputError(f"normalized output bundle is unsafe: {output_dir}")
    entries = {entry.name: entry for entry in output_dir.iterdir()}
    if set(entries) != set(_BUNDLE_NAMES):
        raise HistoricalInputError(
            "normalized output bundle contains a missing or extra artifact"
        )
    for name in _BUNDLE_NAMES:
        _assert_single_link_file(entries[name], label=f"normalized bundle {name}")
    return True


def _validate_bundle_bytes(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    _validate_bundle_structure(output_dir, required=True)
    for name in _BUNDLE_NAMES:
        path = output_dir / name
        before = _assert_single_link_file(path, label=f"normalized bundle {name}")
        observed = path.read_bytes()
        after = _assert_single_link_file(path, label=f"normalized bundle {name}")
        if (
            observed != payloads[name]
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise HistoricalInputError(
                f"normalized bundle bytes differ from deterministic replay: {path}"
            )


def _publish_bundle(
    output_dir: Path,
    *,
    payloads: Mapping[str, bytes],
    fault_injector: Callable[[str, Path], object] | None,
) -> None:
    _assert_no_bundle_staging(output_dir)
    if os.path.lexists(output_dir):
        raise HistoricalInputError(f"refusing to replace immutable output bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(
        f".{output_dir.name}.thermoroute-staging.{secrets.token_hex(32)}"
    )
    try:
        staging.mkdir(mode=0o700)
    except FileExistsError as exc:  # pragma: no cover - nonce collision
        raise HistoricalInputError("cannot allocate normalized bundle staging") from exc
    for name, stage in (
        (_TEMPORAL_TABLE_NAME, "after_temporal_table_staged"),
        (_EXTERNAL_TABLE_NAME, "after_external_table_staged"),
        (_REQUEST_MAP_NAME, "after_request_map_staged"),
    ):
        path = staging / name
        _exclusive_create(path, payloads[name])
        if fault_injector is not None:
            fault_injector(stage, path)
    descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.path.lexists(output_dir):
        raise HistoricalInputError(f"refusing to replace immutable output bundle: {output_dir}")
    os.rename(staging, output_dir)
    descriptor = os.open(
        output_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if fault_injector is not None:
        fault_injector("after_normalized_bundle_publish", output_dir)


def _manifest_document(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    registries: tuple[RegistryEvidence, RegistryEvidence],
    cohort_frames: Mapping[str, pd.DataFrame],
    output_dir: Path,
    daymet_index: Path,
    gridmet_index: Path,
    gridmet_schema_index: Path,
    gridmet_contract: Mapping[str, Any],
    secondary_nwp_resolution: str,
) -> dict[str, Any]:
    temporal, external = registries
    return {
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
            "temporal": _binding(root, output_dir / _TEMPORAL_TABLE_NAME),
            "external": _binding(root, output_dir / _EXTERNAL_TABLE_NAME),
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
            for registry in registries
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
                "validated_contract": dict(gridmet_contract),
            },
            {
                "source": "site-to-request normalization map",
                "evidence_type": "normalized_immutable_snapshot",
                "contains_outcome": False,
                "contains_outcome_labels": False,
                "fields": list(PRELABEL_FIELDS),
                "artifact": _binding(root, output_dir / _REQUEST_MAP_NAME),
            },
        ],
        "secondary_nwp_resolution": secondary_nwp_resolution,
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
    check_existing: bool = False,
    _fault_injector: Callable[[str, Path], object] | None = None,
) -> dict[str, Any]:
    """Acquire, normalize and transactionally freeze both meteorological cohorts.

    The ``expected_*`` parameters make small offline fixture tests possible;
    the production CLI does not expose them and always requires 120+30 sites.
    A retry accepts only an exact completed prefix of raw responses, indexes,
    the atomic normalized bundle, and the final manifest.  Existing bytes are
    independently reconstructed and never replaced.  ``check_existing``
    requires the complete chain, forces offline reads, and publishes nothing.
    """
    root = Path(repo_root).resolve()
    _assert_repository_path(root, root, label="repository", final_kind="directory")
    protocol_path = _assert_repository_path(
        root, protocol_path, label="Route-A protocol", final_kind="file"
    )
    temporal_registry_path = _assert_repository_path(
        root, temporal_registry_path, label="temporal registry", final_kind="file"
    )
    external_registry_path = _assert_repository_path(
        root, external_registry_path, label="external registry", final_kind="file"
    )
    snapshot_root = _assert_repository_path(
        root,
        snapshot_root,
        label="raw snapshot root",
        final_kind="directory",
        allow_absent_suffix=True,
    )
    output_dir = _assert_repository_path(
        root,
        output_dir,
        label="normalized output bundle",
        final_kind="directory",
        allow_absent_suffix=True,
    )
    manifest_path = _assert_repository_path(
        root,
        manifest_path,
        label="historical-input manifest",
        final_kind="file",
        allow_absent_suffix=True,
    )
    if secondary_nwp_resolution != "EXPLICITLY_NOT_USED":
        raise HistoricalInputError(
            "this primary-input freezer can attest only EXPLICITLY_NOT_USED; "
            "ACQUIRED_AND_FROZEN requires separate NWP-manifest verification"
        )
    manifest_exists = os.path.lexists(manifest_path)
    if manifest_exists:
        _assert_single_link_file(manifest_path, label="historical-input manifest")
    bundle_exists = _validate_bundle_structure(
        output_dir, required=bool(check_existing or manifest_exists)
    )
    if manifest_exists and not bundle_exists:  # pragma: no cover - required above
        raise HistoricalInputError("manifest exists without normalized output bundle")
    if check_existing and not manifest_exists:
        raise HistoricalInputError("--check-existing requires the complete manifest")
    derived_exists = bundle_exists or manifest_exists
    effective_offline = bool(offline or check_existing or derived_exists)
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

    daymet_store = SnapshotStore(snapshot_root / "daymet-v1", offline=effective_offline)
    gridmet_store = SnapshotStore(snapshot_root / "gridmet-v1", offline=effective_offline)
    gridmet_schema_store = SnapshotStore(
        snapshot_root / "gridmet-schema-v1", offline=effective_offline
    )
    headers = {"User-Agent": USER_AGENT}
    registries = (temporal, external)
    raw_plan = _expected_raw_plan(
        protocol=protocol,
        registries=registries,
        daymet_store=daymet_store,
        gridmet_store=gridmet_store,
        gridmet_schema_store=gridmet_schema_store,
        headers=headers,
    )
    expected_by_root: dict[Path, dict[str, tuple[str, Mapping[str, object]]]] = {
        store.root: {}
        for store in (daymet_store, gridmet_store, gridmet_schema_store)
    }
    for store, provider, _url, request_sha, request in raw_plan:
        existing = expected_by_root[store.root].get(request_sha)
        candidate = (provider, request)
        if existing is not None and existing != candidate:
            raise HistoricalInputError("one raw request digest maps to multiple requests")
        expected_by_root[store.root][request_sha] = candidate
    states_by_root = {
        store.root: _validate_snapshot_store_namespace(
            store,
            expected_requests=expected_by_root[store.root],
            allow_incomplete=not derived_exists,
        )
        for store in (daymet_store, gridmet_store, gridmet_schema_store)
    }
    _assert_raw_prefix(raw_plan, states_by_root)
    index_paths = [
        daymet_store.root / "snapshot_index.json",
        gridmet_store.root / "snapshot_index.json",
        gridmet_schema_store.root / "snapshot_index.json",
    ]
    index_states = [os.path.lexists(path) for path in index_paths]
    if any(index_states[index] and not all(index_states[:index]) for index in range(3)):
        raise HistoricalInputError("raw snapshot indexes are not a resumable prefix")
    if derived_exists and not all(index_states):
        raise HistoricalInputError(
            "derived historical inputs exist before every raw index was frozen"
        )

    schema_url = build_gridmet_wind_metadata_url()
    schema_payload, schema_record = gridmet_schema_store.fetch(
        provider=GRIDMET_SCHEMA_PROVIDER,
        url=schema_url,
        headers=headers,
        retries=max(1, int(retries)),
        resume_incomplete=not effective_offline,
        _fault_injector=_fault_injector,
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
            resume_incomplete=not effective_offline,
            fault_injector=_fault_injector,
        )
        cohort_frames[registry.cohort] = frame
        request_records.extend(records)

    for store in (daymet_store, gridmet_store, gridmet_schema_store):
        _validate_snapshot_store_namespace(
            store,
            expected_requests=expected_by_root[store.root],
            allow_incomplete=False,
        )

    daymet_requests = {
        str(record["daymet"]["request_sha256"]) for record in request_records
    }
    gridmet_requests = {
        str(record["gridmet"]["request_sha256"]) for record in request_records
    }
    daymet_index = freeze_snapshot_index(
        daymet_store,
        expected_request_sha256=daymet_requests,
        allow_create=not derived_exists and not check_existing,
        fault_stage="after_daymet_index_publish",
        fault_injector=_fault_injector,
    )
    gridmet_index = freeze_snapshot_index(
        gridmet_store,
        expected_request_sha256=gridmet_requests,
        allow_create=not derived_exists and not check_existing,
        fault_stage="after_gridmet_index_publish",
        fault_injector=_fault_injector,
    )
    gridmet_schema_index = freeze_snapshot_index(
        gridmet_schema_store,
        expected_request_sha256={str(schema_record.request_sha256)},
        allow_create=not derived_exists and not check_existing,
        fault_stage="after_gridmet_schema_index_publish",
        fault_injector=_fault_injector,
    )
    for store in (daymet_store, gridmet_store, gridmet_schema_store):
        _validate_optional_snapshot_index_v2(store)

    request_map = _request_map_document(
        protocol=protocol,
        request_records=request_records,
        gridmet_contract=gridmet_contract,
        schema_record=schema_record,
    )
    payloads = _bundle_payloads(
        cohort_frames=cohort_frames,
        request_map=request_map,
    )
    if bundle_exists:
        _validate_bundle_bytes(output_dir, payloads)
    elif check_existing:  # pragma: no cover - rejected during structural preflight
        raise HistoricalInputError("--check-existing requires normalized bundle")
    else:
        _publish_bundle(
            output_dir,
            payloads=payloads,
            fault_injector=_fault_injector,
        )
        bundle_exists = True

    manifest = _manifest_document(
        root=root,
        protocol=protocol,
        registries=registries,
        cohort_frames=cohort_frames,
        output_dir=output_dir,
        daymet_index=daymet_index,
        gridmet_index=gridmet_index,
        gridmet_schema_index=gridmet_schema_index,
        gridmet_contract=gridmet_contract,
        secondary_nwp_resolution=secondary_nwp_resolution,
    )
    manifest_payload = canonical_json_bytes(manifest)
    if manifest_exists:
        _assert_single_link_file(manifest_path, label="historical-input manifest")
        if manifest_path.read_bytes() != manifest_payload:
            raise HistoricalInputError(
                "historical-input manifest differs from complete deterministic replay"
            )
    elif check_existing:
        raise HistoricalInputError("--check-existing cannot publish a missing manifest")
    else:
        _exclusive_create(manifest_path, manifest_payload)
        if _fault_injector is not None:
            _fault_injector("after_manifest_publish", manifest_path)
    return manifest
