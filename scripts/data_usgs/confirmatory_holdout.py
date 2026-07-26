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
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _isolate_project_bytecode() -> None:
    if __name__ != "__main__":
        return
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    if (
        sys.flags.isolated
        and prefix is not None
        and prefix != ROOT
        and ROOT not in prefix.parents
    ):
        return
    with tempfile.TemporaryDirectory(prefix="thermoroute-holdout-pycache-") as cache:
        result = subprocess.run(
            [sys.executable, "-I", "-X", f"pycache_prefix={cache}",
             str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()
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
from thermoroute.provenance import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from thermoroute.repro import sha256_json  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"


def _inside_root(path: Path, *, label: str) -> Path:
    root = ROOT.resolve()
    candidate = Path(os.path.abspath(root / path if not path.is_absolute() else path))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the repository root: {path}") from exc
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if not os.path.lexists(current):
            continue
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"{label} has a linked/non-directory parent: {current}")
    return candidate


def _temporary_siblings(path: Path) -> list[Path]:
    if not path.parent.is_dir():
        return []
    return sorted(path.parent.glob(f".{path.name}.*.tmp"))


def _published_bytes(path: Path, *, label: str) -> bytes:
    """Read one immutable publication without following or accepting links."""
    leftovers = _temporary_siblings(path)
    if leftovers:
        raise RuntimeError(
            f"{label} has unpublished temporary siblings: {leftovers[:3]}"
        )
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"{label} must be one regular single-link file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    def object_without_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r}")
            output[key] = value
        return output

    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    if payload != canonical_json_bytes(document):
        raise RuntimeError(f"{label} is not canonical producer JSON")
    return document


def atomic_write(path: Path, payload: bytes) -> None:
    """Create one frozen artifact once; an interrupted write stays fail-closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    leftovers = _temporary_siblings(path)
    if leftovers:
        raise RuntimeError(
            f"refusing publication beside temporary evidence: {leftovers[:3]}"
        )
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite holdout lock: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite holdout lock: {path}")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Never delete or replace a partial publication.  The next invocation
        # will identify and reject it as noncanonical evidence.
        raise


def verify_candidate_evidence(
    candidates_path: Path,
    provenance_path: Path,
    snapshot_index_path: Path,
    protocol_path: Path | None = None,
) -> pd.DataFrame:
    """Replay raw metadata responses and verify the derived candidate table."""
    candidate_payload = _published_bytes(candidates_path, label="candidate table")
    provenance = _strict_json_bytes(
        _published_bytes(provenance_path, label="candidate provenance"),
        label="candidate provenance",
    )
    snapshot_index = _strict_json_bytes(
        _published_bytes(snapshot_index_path, label="candidate snapshot index"),
        label="candidate snapshot index",
    )
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
        response_payload = _published_bytes(
            response_path, label=f"candidate raw response for {state}"
        )
        response_sha = sha256_file(response_path)
        if (
            response_sha != request.get("response_sha256")
            or response_sha != indexed.get("response_sha256")
        ):
            raise RuntimeError(f"candidate raw response checksum mismatch for {state}")
        frames.append(parse_usgs_candidate_metadata(response_payload, state=state))
        seen_states.append(state)
    if sorted(seen_states) != sorted(provenance.get("state_universe", [])):
        raise RuntimeError("candidate raw-response states differ from frozen universe")
    rebuilt = merge_candidate_metadata(frames)
    if int(provenance.get("candidate_count", -1)) != len(rebuilt):
        raise RuntimeError("candidate count differs from frozen provenance")
    provided = pd.read_csv(
        io.BytesIO(candidate_payload),
        dtype={
            "site_no": "string", "station_nm": "string", "state": "string",
            "site_type": "string", "huc_cd": "string",
        },
        keep_default_na=False,
        float_precision="round_trip",
    )
    if tuple(provided.columns) != CANDIDATE_COLUMNS:
        raise RuntimeError("candidate table has a non-frozen column schema")
    for column in ("lat", "lon", "drain_area_va"):
        provided[column] = pd.to_numeric(provided[column], errors="coerce")
    provided["huc_cd"] = provided["huc_cd"].fillna("")
    try:
        pd.testing.assert_frame_equal(
            rebuilt, provided, check_dtype=False, rtol=0.0, atol=0.0
        )
    except AssertionError as exc:
        raise RuntimeError("candidate table cannot be rebuilt from raw snapshots") from exc
    return provided


def _prepare_publication(
    args: argparse.Namespace,
) -> tuple[bytes, dict[str, Any]]:
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
    registry_payload = registry.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    lock_stable: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "authoritative_protocol_commit": protocol["authoritative_protocol_commit"],
        "pre_label_amendments_sha256": sha256_json(
            protocol.get("pre_label_amendments", [])
        ),
        "status": "REGISTRY_FROZEN_LABELS_SEALED",
        "site_count": len(registry),
        "site_primary_key": "site_no",
        "selection_seed": args.selection_seed,
        "holdout_start": protocol["time_holdout"]["start"],
        "holdout_end": protocol["time_holdout"]["end"],
        "development_panel_spec_sha256": sha256_file(args.development_spec),
        "candidate_table_sha256": sha256_file(args.candidates),
        "candidate_provenance_sha256": sha256_file(args.candidate_provenance),
        "candidate_snapshot_index_sha256": sha256_file(args.candidate_snapshot_index),
        "confirmatory_registry_sha256": sha256_bytes(registry_payload),
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
    return registry_payload, lock_stable


def _validate_lock(
    path: Path,
    *,
    expected_stable: Mapping[str, Any],
) -> dict[str, Any]:
    document = _strict_json_bytes(
        _published_bytes(path, label="external registry lock"),
        label="external registry lock",
    )
    expected_keys = set(expected_stable) | {"created_at_utc"}
    if set(document) != expected_keys:
        raise RuntimeError("external registry lock schema changed")
    created = document.get("created_at_utc")
    try:
        timestamp = datetime.fromisoformat(str(created))
    except ValueError as exc:
        raise RuntimeError("external registry lock timestamp is malformed") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise RuntimeError("external registry lock timestamp is not UTC")
    observed_stable = dict(document)
    observed_stable.pop("created_at_utc")
    if observed_stable != dict(expected_stable):
        raise RuntimeError("external registry lock differs from deterministic inputs")
    return document


def publish_or_validate(args: argparse.Namespace, *, check_only: bool) -> dict[str, Any]:
    """Resume the registry/lock suffix or validate its complete publication."""
    for attribute, label in (
        ("protocol", "Route-A protocol"),
        ("development_spec", "development panel specification"),
        ("candidates", "candidate table"),
        ("candidate_snapshot_index", "candidate snapshot index"),
        ("candidate_provenance", "candidate provenance"),
        ("out_registry", "external registry"),
        ("out_lock", "external registry lock"),
    ):
        setattr(args, attribute, _inside_root(getattr(args, attribute), label=label))
    registry_exists = os.path.lexists(args.out_registry)
    lock_exists = os.path.lexists(args.out_lock)
    if lock_exists and not registry_exists:
        raise RuntimeError("external registry publication has a lock without its registry")
    if check_only and not (registry_exists and lock_exists):
        raise RuntimeError("external registry publication is incomplete")

    registry_payload, lock_stable = _prepare_publication(args)
    if registry_exists:
        if _published_bytes(
            args.out_registry, label="external registry"
        ) != registry_payload:
            raise RuntimeError("external registry differs from deterministic selection")
    elif check_only:
        raise RuntimeError("external registry is absent")
    else:
        atomic_write(args.out_registry, registry_payload)

    if lock_exists:
        return _validate_lock(args.out_lock, expected_stable=lock_stable)
    if check_only:
        raise RuntimeError("external registry lock is absent")
    lock = {
        **lock_stable,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write(args.out_lock, canonical_json_bytes(lock))
    return lock


def freeze(args: argparse.Namespace) -> None:
    lock = publish_or_validate(args, check_only=False)
    print(json.dumps(lock, indent=2))


def check(args: argparse.Namespace) -> None:
    lock = publish_or_validate(args, check_only=True)
    print(json.dumps({
        "status": "CANDIDATE_REGISTRY_PUBLICATION_VALID",
        "registry": str(args.out_registry),
        "lock": str(args.out_lock),
        "lock_sha256": sha256_file(args.out_lock),
        "site_count": lock["site_count"],
        "network_used": False,
    }, indent=2))


def status(args: argparse.Namespace) -> None:
    protocol = load_confirmatory_protocol(args.protocol)
    print(json.dumps(protocol, indent=2))


def _add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--development-spec", type=Path,
        default=ROOT / "data_usgs" / "frozen_panel_v1.json",
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-snapshot-index", type=Path, required=True)
    parser.add_argument("--candidate-provenance", type=Path, required=True)
    parser.add_argument("--out-registry", type=Path, required=True)
    parser.add_argument("--out-lock", type=Path, required=True)
    parser.add_argument("--n-sites", type=int, default=30)
    parser.add_argument(
        "--selection-seed", default="route-a-confirmatory-v1-public-seed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("status")
    show.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    show.set_defaults(func=status)

    freeze_cmd = sub.add_parser("freeze-candidates")
    _add_candidate_arguments(freeze_cmd)
    freeze_cmd.set_defaults(func=freeze)

    check_cmd = sub.add_parser("check-candidates")
    _add_candidate_arguments(check_cmd)
    check_cmd.set_defaults(func=check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
