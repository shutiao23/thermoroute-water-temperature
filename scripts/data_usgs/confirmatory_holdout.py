#!/usr/bin/env python3
"""Freeze the metadata-only registry for the untouched Route-A holdout.

This command deliberately does *not* download holdout WTEMP labels.  It creates
the evidence that must predate label acquisition: a disjoint, deterministic site
registry and a lock tying it to the raw candidate-discovery snapshot, development
panel, protocol and selection seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.evidence import (  # noqa: E402
    FrozenPanelSpec,
    load_confirmatory_protocol,
    select_confirmatory_sites,
)
from thermoroute.confirmatory import (  # noqa: E402
    CANDIDATE_COLUMNS,
    build_usgs_candidate_url,
    merge_candidate_metadata,
    parse_usgs_candidate_metadata,
)
from thermoroute.provenance import canonical_json_bytes, sha256_file  # noqa: E402
from thermoroute.repro import sha256_json  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"


def atomic_write(path: Path, payload: bytes) -> None:
    """Create one frozen artifact atomically; never replace an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite holdout lock: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite holdout lock: {path}")
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def verify_candidate_evidence(
    candidates_path: Path,
    provenance_path: Path,
    snapshot_index_path: Path,
    protocol_path: Path | None = None,
) -> pd.DataFrame:
    """Replay raw metadata responses and verify the derived candidate table."""
    for path in (candidates_path, provenance_path, snapshot_index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        snapshot_index = json.loads(snapshot_index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("candidate provenance/index is not valid JSON") from exc
    required_flags = {
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
    }
    if any(provenance.get(key) != value for key, value in required_flags.items()):
        raise RuntimeError("candidate provenance does not prove metadata-only discovery")
    if provenance.get("candidate_table_sha256") != sha256_file(candidates_path):
        raise RuntimeError("candidate table checksum differs from its provenance")
    if provenance.get("raw_snapshot_index_sha256") != sha256_file(snapshot_index_path):
        raise RuntimeError("candidate snapshot index checksum differs from provenance")
    if (
        protocol_path is not None
        and provenance.get("protocol_sha256") != sha256_file(protocol_path)
    ):
        raise RuntimeError("candidate discovery was not sealed against this protocol")

    index_by_request = {
        str(record["request_sha256"]): record
        for record in snapshot_index.get("records", [])
    }
    frames = []
    seen_states = []
    for request in provenance.get("requests", []):
        state = str(request["state"])
        request_sha = str(request["request_sha256"])
        if request_sha not in index_by_request:
            raise RuntimeError(f"candidate request {request_sha} lacks a raw snapshot")
        indexed = index_by_request[request_sha]
        if indexed.get("provider") != "usgs-nwis-confirmatory-site-metadata":
            raise RuntimeError(f"candidate request for {state} has the wrong provider")
        if indexed.get("request", {}).get("url") != build_usgs_candidate_url(state):
            raise RuntimeError(f"candidate request for {state} is not the frozen metadata URL")
        snapshot_root = snapshot_index_path.parent.resolve()
        response_path = (snapshot_root / str(indexed["response_path"])).resolve()
        if snapshot_root not in response_path.parents:
            raise RuntimeError("candidate snapshot response escapes its snapshot root")
        response_sha = sha256_file(response_path)
        if (
            response_sha != request.get("response_sha256")
            or response_sha != indexed.get("response_sha256")
        ):
            raise RuntimeError(f"candidate raw response checksum mismatch for {state}")
        frames.append(parse_usgs_candidate_metadata(response_path.read_bytes(), state=state))
        seen_states.append(state)
    if sorted(seen_states) != sorted(provenance.get("state_universe", [])):
        raise RuntimeError("candidate raw-response states differ from frozen universe")
    rebuilt = merge_candidate_metadata(frames)
    if int(provenance.get("candidate_count", -1)) != len(rebuilt):
        raise RuntimeError("candidate count differs from frozen provenance")
    provided = pd.read_csv(
        candidates_path,
        dtype={
            "site_no": "string", "station_nm": "string", "state": "string",
            "site_type": "string", "huc_cd": "string",
        },
        keep_default_na=False,
    )
    if tuple(provided.columns) != CANDIDATE_COLUMNS:
        raise RuntimeError("candidate table has a non-frozen column schema")
    for column in ("lat", "lon", "drain_area_va"):
        provided[column] = pd.to_numeric(provided[column], errors="coerce")
    provided["huc_cd"] = provided["huc_cd"].fillna("")
    try:
        pd.testing.assert_frame_equal(rebuilt, provided, check_dtype=False)
    except AssertionError as exc:
        raise RuntimeError("candidate table cannot be rebuilt from raw snapshots") from exc
    return provided


def freeze(args: argparse.Namespace) -> None:
    protocol = load_confirmatory_protocol(args.protocol)
    if protocol["new_site_external_validation"]["status"] != "PLANNED_NOT_ACQUIRED":
        raise RuntimeError("new-site candidate registry is already frozen or opened")
    planned = protocol["new_site_external_validation"]
    if args.n_sites != int(planned["planned_site_count"]):
        raise RuntimeError("site count differs from the predeclared protocol")
    if args.selection_seed != planned["selection_seed"]:
        raise RuntimeError("selection seed differs from the predeclared protocol")
    development_spec = FrozenPanelSpec.load(args.development_spec)
    development = development_spec.load_registry()
    candidates = verify_candidate_evidence(
        args.candidates,
        args.candidate_provenance,
        args.candidate_snapshot_index,
        args.protocol,
    )
    if protocol["metadata_candidate_contract"]["state_universe"] != json.loads(
        args.candidate_provenance.read_text(encoding="utf-8")
    )["state_universe"]:
        raise RuntimeError("candidate state universe differs from the protocol")
    registry = select_confirmatory_sites(
        candidates,
        set(development["site_no"].astype(str)),
        n_sites=args.n_sites,
        selection_seed=args.selection_seed,
    )
    for output in (args.out_registry, args.out_lock):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite holdout lock: {output}")
    if not args.candidate_snapshot_index.is_file():
        raise FileNotFoundError(
            "candidate discovery must have a raw snapshot index before selection")

    registry_payload = registry.to_csv(index=False, lineterminator="\n").encode("utf-8")
    atomic_write(args.out_registry, registry_payload)
    lock = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "authoritative_protocol_commit": protocol["authoritative_protocol_commit"],
        "pre_label_amendments_sha256": sha256_json(
            protocol.get("pre_label_amendments", [])
        ),
        "status": "REGISTRY_FROZEN_LABELS_SEALED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "site_count": len(registry),
        "site_primary_key": "site_no",
        "selection_seed": args.selection_seed,
        "holdout_start": protocol["time_holdout"]["start"],
        "holdout_end": protocol["time_holdout"]["end"],
        "development_panel_spec_sha256": sha256_file(args.development_spec),
        "candidate_table_sha256": sha256_file(args.candidates),
        "candidate_provenance_sha256": sha256_file(args.candidate_provenance),
        "candidate_snapshot_index_sha256": sha256_file(args.candidate_snapshot_index),
        "confirmatory_registry_sha256": sha256_file(args.out_registry),
        "frozen_artifacts": {
            "development_panel_spec": {
                "path": args.development_spec.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.development_spec),
            },
            "candidate_table": {
                "path": args.candidates.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.candidates),
            },
            "candidate_provenance": {
                "path": args.candidate_provenance.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.candidate_provenance),
            },
            "candidate_snapshot_index": {
                "path": args.candidate_snapshot_index.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.candidate_snapshot_index),
            },
        },
        "labels_state": "SEALED_NOT_ACQUIRED",
        "opening_count": 0,
    }
    atomic_write(args.out_lock, canonical_json_bytes(lock))
    print(json.dumps(lock, indent=2))


def status(args: argparse.Namespace) -> None:
    protocol = load_confirmatory_protocol(args.protocol)
    print(json.dumps(protocol, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("status")
    show.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    show.set_defaults(func=status)

    freeze_cmd = sub.add_parser("freeze-candidates")
    freeze_cmd.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze_cmd.add_argument(
        "--development-spec", type=Path,
        default=ROOT / "data_usgs" / "frozen_panel_v1.json",
    )
    freeze_cmd.add_argument("--candidates", type=Path, required=True)
    freeze_cmd.add_argument("--candidate-snapshot-index", type=Path, required=True)
    freeze_cmd.add_argument("--candidate-provenance", type=Path, required=True)
    freeze_cmd.add_argument("--out-registry", type=Path, required=True)
    freeze_cmd.add_argument("--out-lock", type=Path, required=True)
    freeze_cmd.add_argument("--n-sites", type=int, default=30)
    freeze_cmd.add_argument(
        "--selection-seed", default="route-a-confirmatory-v1-public-seed")
    freeze_cmd.set_defaults(func=freeze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
