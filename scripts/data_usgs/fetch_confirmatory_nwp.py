#!/usr/bin/env python3
"""Acquire optional, label-free archived GFS sensitivity inputs for Route A.

Requests are one station by one calendar month.  Raw JSON responses are retained
by SnapshotStore, while each derived Parquet block is create-only and checksum
sealed.  Primary Route-A models do not consume this artifact.  This script never
reads a panel or calls a USGS outcome endpoint.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.nwp import (  # noqa: E402
    GFS_ARCHIVE_RUN_START,
    ISSUE_SEMANTICS,
    OPEN_METEO_MODEL,
    NWP_COMMON_VALID_TIME_START,
    ROUTE_A_LEADS,
    build_previous_runs_url,
    iter_month_chunks,
    parse_previous_runs_daily,
)
from thermoroute.provenance import (  # noqa: E402
    SnapshotStore,
    canonical_json_bytes,
    sha256_file,
)


DEFAULT_SNAPSHOT_DIR = (
    ROOT / "data_usgs" / "raw_snapshots" / "openmeteo-gfs-previous-runs-v1"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data_usgs" / "confirmatory_predictors" / "gfs-previous-runs-v1"
)
DEFAULT_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
ROUTE_A_TARGET_END = date(2023, 12, 31)
USER_AGENT = "ThermoRoute/1.0 Route-A predictor acquisition"


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_create_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Create a Parquet block atomically without any overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    file_descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    tmp = Path(tmp_name)
    try:
        frame.to_parquet(tmp, index=False)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_station_coordinates(paths: list[Path]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Read only stable identity and coordinates from metadata registries."""
    frames = []
    lineage = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0)
        required = {"site_no", "lat", "lon"}
        missing = required - set(header.columns)
        if missing:
            raise RuntimeError(f"predictor registry {path} lacks {sorted(missing)}")
        # usecols prevents legacy development coverage columns from being loaded;
        # no panel or outcome artifact is accepted by this acquisition script.
        frame = pd.read_csv(
            path,
            usecols=["site_no", "lat", "lon"],
            dtype={"site_no": "string"},
        )
        frame["site_no"] = frame["site_no"].astype("string").str.strip()
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        if frame.isna().any().any() or frame["site_no"].eq("").any():
            raise RuntimeError(f"predictor registry {path} has missing identity/coordinates")
        frames.append(frame)
        lineage.append({
            "path": os.path.relpath(path.resolve(), ROOT),
            "sha256": sha256_file(path),
            "columns_read": ["site_no", "lat", "lon"],
            "row_count": len(frame),
        })
    stations = pd.concat(frames, ignore_index=True)
    duplicated = stations[stations["site_no"].duplicated(keep=False)]
    if not duplicated.empty:
        conflicts = duplicated.groupby("site_no")[["lat", "lon"]].nunique()
        if (conflicts > 1).any(axis=None):
            raise RuntimeError("the same site_no has conflicting registry coordinates")
        stations = stations.drop_duplicates("site_no", keep="first")
    stations = stations.sort_values("site_no").reset_index(drop=True)
    return stations, lineage


def _chunk_paths(root: Path, site_no: str, start: date) -> tuple[Path, Path]:
    artifact = root / site_no / f"{start:%Y-%m}.parquet"
    return artifact, artifact.with_suffix(".provenance.json")


def _validated_resume_record(
    artifact: Path,
    sidecar: Path,
    *,
    site_no: str,
    chunk_start: date,
    chunk_end: date,
    protocol_sha256: str,
) -> dict[str, object]:
    if not artifact.is_file() or not sidecar.is_file():
        raise FileExistsError(
            f"partial existing predictor block cannot be resumed safely: {artifact}"
        )
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid predictor sidecar: {sidecar}") from exc
    if metadata.get("artifact_sha256") != sha256_file(artifact):
        raise RuntimeError(f"predictor block checksum mismatch: {artifact}")
    expected = {
        "schema_version": 1,
        "site_no": site_no,
        "chunk_start": chunk_start.isoformat(),
        "chunk_end": chunk_end.isoformat(),
        "labels_requested_or_read": False,
        "protocol_sha256": protocol_sha256,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"predictor sidecar identity mismatch: {sidecar}")
    return metadata


def acquire(args: argparse.Namespace) -> None:
    start = pd.Timestamp(args.start_date).date()
    end = pd.Timestamp(args.end_date).date()
    if start != NWP_COMMON_VALID_TIME_START or end != ROUTE_A_TARGET_END:
        raise RuntimeError(
            "Route-A frozen target window is exactly "
            f"{NWP_COMMON_VALID_TIME_START} through {ROUTE_A_TARGET_END}"
        )
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    predictor_contract = protocol["secondary_archived_nwp_contract"]
    frozen_contract = {
        "status": "IMPLEMENTED_NOT_ACQUIRED",
        "contains_outcome_labels": False,
        "api_endpoint": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "open_meteo_model_parameter": OPEN_METEO_MODEL,
        "variable": "temperature_2m",
        "lead_days": list(ROUTE_A_LEADS),
        "archive_run_start": GFS_ARCHIVE_RUN_START.isoformat(),
        "secondary_common_lead_target_start": NWP_COMMON_VALID_TIME_START.isoformat(),
    }
    if any(predictor_contract.get(key) != value for key, value in frozen_contract.items()):
        raise RuntimeError("machine protocol and predictor acquisition contract differ")
    protocol_sha = sha256_file(args.protocol)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"completed predictor acquisition already exists: {manifest_path}")

    registry_paths = args.registry or [DEFAULT_REGISTRY]
    stations, registry_lineage = load_station_coordinates(registry_paths)
    store = SnapshotStore(args.snapshot_dir, offline=args.offline)
    chunk_records: list[dict[str, object]] = []
    row_count = 0
    complete_count = 0

    for station in stations.itertuples(index=False):
        for chunk_start, chunk_end in iter_month_chunks(start, end):
            artifact, sidecar = _chunk_paths(
                args.output_dir, str(station.site_no), chunk_start
            )
            if artifact.exists() or sidecar.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"refusing to overwrite existing predictor block: {artifact}"
                    )
                prior = _validated_resume_record(
                    artifact,
                    sidecar,
                    site_no=str(station.site_no),
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    protocol_sha256=protocol_sha,
                )
                chunk_records.append(prior)
                row_count += int(prior["row_count"])
                complete_count += int(prior["complete_row_count"])
                continue

            url = build_previous_runs_url(
                latitude=float(station.lat),
                longitude=float(station.lon),
                start_date=chunk_start,
                end_date=chunk_end,
            )
            payload, snapshot = store.fetch(
                provider="open-meteo-previous-runs-gfs-global",
                url=url,
                headers={"User-Agent": USER_AGENT},
                retries=args.retries,
            )
            frame = parse_previous_runs_daily(
                payload,
                site_no=str(station.site_no),
                requested_start=chunk_start,
                requested_end=chunk_end,
            )
            frame.insert(1, "requested_lat", float(station.lat))
            frame.insert(2, "requested_lon", float(station.lon))
            frame["request_sha256"] = snapshot.request_sha256
            frame["response_sha256"] = snapshot.response_sha256
            if artifact.exists() or sidecar.exists():
                raise FileExistsError(f"refusing to overwrite predictor block: {artifact}")
            atomic_create_parquet(artifact, frame)
            record = {
                "schema_version": 1,
                "artifact": os.path.relpath(artifact.resolve(), ROOT),
                "artifact_sha256": sha256_file(artifact),
                "site_no": str(station.site_no),
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
                "row_count": len(frame),
                "complete_row_count": int(frame["complete_target_day"].sum()),
                "request_sha256": snapshot.request_sha256,
                "response_sha256": snapshot.response_sha256,
                "retrieved_at_utc": snapshot.retrieved_at_utc,
                "labels_requested_or_read": False,
                "protocol_sha256": protocol_sha,
            }
            atomic_create(sidecar, canonical_json_bytes(record))
            chunk_records.append(record)
            row_count += len(frame)
            complete_count += int(frame["complete_target_day"].sum())
            if args.request_interval > 0:
                time.sleep(args.request_interval)

    snapshot_index = store.write_index()
    indexed = json.loads(snapshot_index.read_text(encoding="utf-8"))
    indexed_requests = {
        record["request_sha256"] for record in indexed.get("records", [])
    }
    missing_snapshots = sorted({
        str(record["request_sha256"]) for record in chunk_records
    } - indexed_requests)
    if missing_snapshots:
        raise RuntimeError(
            "derived predictor blocks lack raw snapshots: "
            f"{missing_snapshots[:10]}"
        )
    manifest = {
        "schema_version": 1,
        "artifact_role": "OPTIONAL_SECONDARY_NWP_AVAILABILITY_SENSITIVITY",
        "protocol": os.path.relpath(args.protocol.resolve(), ROOT),
        "protocol_sha256": protocol_sha,
        "source_provider": "Open-Meteo Previous Runs API",
        "source_documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "upstream_model": "NOAA NCEP GFS global",
        "open_meteo_model": OPEN_METEO_MODEL,
        "source_variable": "temperature_2m",
        "lead_days": list(ROUTE_A_LEADS),
        "lead_fields": [
            f"temperature_2m_previous_day{lead}" for lead in ROUTE_A_LEADS
        ],
        "issue_semantics": ISSUE_SEMANTICS,
        "timezone": "GMT (UTC+00:00)",
        "daily_aggregation": (
            "arithmetic mean of exactly 24 finite valid-hour values; incomplete "
            "days retained with NaN predictor and availability count"
        ),
        "archive_run_start": GFS_ARCHIVE_RUN_START.isoformat(),
        "common_valid_target_start": NWP_COMMON_VALID_TIME_START.isoformat(),
        "target_end": ROUTE_A_TARGET_END.isoformat(),
        "station_count": len(stations),
        "row_count": row_count,
        "complete_row_count": complete_count,
        "request_partition": "one stable site_no by one UTC calendar month",
        "immutable_outputs": True,
        "consumed_by_primary_route_a_models": False,
        "primary_evaluation_dependency": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "registry_inputs": registry_lineage,
        "raw_snapshot_index": os.path.relpath(snapshot_index.resolve(), ROOT),
        "raw_snapshot_index_sha256": sha256_file(snapshot_index),
        "chunks": chunk_records,
    }
    atomic_create(manifest_path, canonical_json_bytes(manifest))
    print(json.dumps({
        "manifest": str(manifest_path),
        "station_count": len(stations),
        "row_count": row_count,
        "complete_row_count": complete_count,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", type=Path, action="append",
        help=(
            "metadata registry containing only-used columns site_no/lat/lon; "
            "repeat to include frozen development and new-site cohorts"
        ),
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--start-date", default=NWP_COMMON_VALID_TIME_START.isoformat()
    )
    parser.add_argument("--end-date", default=ROUTE_A_TARGET_END.isoformat())
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be positive")
    if args.request_interval < 0:
        parser.error("--request-interval cannot be negative")
    acquire(args)


if __name__ == "__main__":
    main()
