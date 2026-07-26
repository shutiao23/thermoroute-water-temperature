#!/usr/bin/env python3
"""Discover and optionally freeze Route-A new-site candidates from metadata only.

The USGS request is made against ``/nwis/site/``.  It advertises that a stream
site has a daily-value water-temperature data type, but requests neither values,
dates nor holdout-period coverage.  Exact RDB bytes are stored by SnapshotStore.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


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
    with tempfile.TemporaryDirectory(prefix="thermoroute-candidates-pycache-") as cache:
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

from thermoroute.confirmatory import (  # noqa: E402
    CANDIDATE_PROVIDER,
    CANDIDATE_SELECTION_RULE,
    CANDIDATE_STATE_UNIVERSE_RULE,
    CANDIDATE_USER_AGENT,
    ROUTE_A_STATE_UNIVERSE,
    audit_candidate_evidence,
    audit_candidate_snapshot_store,
    build_usgs_candidate_url,
    merge_candidate_metadata,
    normalise_states,
    parse_usgs_candidate_metadata,
)
from thermoroute.provenance import (  # noqa: E402
    AdvisoryLockHandle,
    SnapshotStore,
    candidate_publication_lock,
    canonical_json_bytes,
    create_single_link_regular,
    read_single_link_regular,
    sha256_bytes,
)


DEFAULT_SNAPSHOT_DIR = (
    ROOT / "data_usgs" / "raw_snapshots" / "confirmatory-candidates-v1"
)
DEFAULT_OUT = ROOT / "data_usgs" / "confirmatory_candidate_sites_v1.csv"
DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
USER_AGENT = CANDIDATE_USER_AGENT


def _candidate_publication_lock_path() -> Path:
    # Both CLIs lock the same tracked, immutable protocol inode.  The anchor is
    # opened read-only, so check-only validation creates no repository state.
    return ROOT / "protocols" / "route_a_confirmatory_v1.json"


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
    leftovers = _temporary_siblings(path)
    if leftovers:
        raise RuntimeError(
            f"{label} has unpublished temporary siblings: {leftovers[:3]}"
        )
    return read_single_link_regular(path, label=label, trusted_root=ROOT.resolve())


def _strict_json_bytes(
    payload: bytes,
    *,
    label: str,
    require_canonical: bool,
) -> dict[str, Any]:
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
    if require_canonical and payload != canonical_json_bytes(document):
        raise RuntimeError(f"{label} is not canonical producer JSON")
    return document


def atomic_create(path: Path, payload: bytes) -> None:
    """Create a publication once; interrupted bytes remain visible and invalid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    leftovers = _temporary_siblings(path)
    if leftovers:
        raise RuntimeError(
            f"refusing publication beside temporary evidence: {leftovers[:3]}"
        )
    try:
        create_single_link_regular(
            path, payload, trusted_root=ROOT.resolve(), mode=0o644
        )
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")


def parse_states(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ROUTE_A_STATE_UNIVERSE
    tokens = [token for value in values for token in value.split(",")]
    return normalise_states(tokens)


def holdout_freeze_command(
    *,
    candidates: Path,
    snapshot_index: Path,
    candidate_provenance: Path,
    out_registry: Path,
    out_lock: Path,
    protocol: Path,
    n_sites: int,
    selection_seed: str,
    check_existing: bool = False,
    inherited_lock_fd: int | None = None,
    inherited_lock_token: str | None = None,
) -> list[str]:
    """Return the exact invocation of the existing sealed-holdout freezer."""
    command = [
        sys.executable,
        str(ROOT / "scripts" / "data_usgs" / "confirmatory_holdout.py"),
        "check-candidates" if check_existing else "freeze-candidates",
        "--protocol", str(protocol),
        "--candidates", str(candidates),
        "--candidate-snapshot-index", str(snapshot_index),
        "--candidate-provenance", str(candidate_provenance),
        "--out-registry", str(out_registry),
        "--out-lock", str(out_lock),
        "--n-sites", str(n_sites),
        "--selection-seed", selection_seed,
    ]
    if inherited_lock_fd is not None:
        if inherited_lock_token is None:
            raise ValueError("inherited lock token is required with its descriptor")
        command.extend([
            "--inherited-publication-lock-fd", str(inherited_lock_fd),
            "--inherited-publication-lock-token", inherited_lock_token,
        ])
    return command


def _request_document(state: str) -> dict[str, object]:
    return SnapshotStore.request_document(
        provider=CANDIDATE_PROVIDER,
        url=build_usgs_candidate_url(state),
        headers={"User-Agent": USER_AGENT},
    )


def _expected_raw_paths(states: tuple[str, ...]) -> set[str]:
    output: set[str] = set()
    provider = SnapshotStore._provider_name(CANDIDATE_PROVIDER)
    for state in states:
        request_sha = sha256_bytes(canonical_json_bytes(_request_document(state)))
        base = Path(provider) / request_sha
        output |= {
            (base / "metadata.json").as_posix(),
            (base / "response.bin").as_posix(),
        }
    return output


def _audit_snapshot_tree(
    root: Path,
    *,
    expected_raw_files: set[str],
    allow_index: bool,
    require_all_raw: bool,
) -> None:
    """Reject links, partial transactions, and every unexpected snapshot node."""
    if not os.path.lexists(root):
        if require_all_raw:
            raise RuntimeError("candidate raw snapshot root is absent")
        return
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("candidate raw snapshot root must be one real directory")

    expected_files = set(expected_raw_files)
    if allow_index:
        expected_files.add("snapshot_index.json")
    expected_directories: set[str] = set()
    for relative in expected_raw_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"candidate snapshot directory is unsafe: {relative}")
            actual_directories.add(relative)
        for name in files:
            path = directory_path / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(f"candidate snapshot file is linked/unsafe: {relative}")
            actual_files.add(relative)
    unexpected_files = actual_files - expected_files
    unexpected_directories = actual_directories - expected_directories
    if unexpected_files or unexpected_directories:
        raise RuntimeError(
            "candidate snapshot store contains extra nodes: "
            f"files={sorted(unexpected_files)[:5]}, "
            f"directories={sorted(unexpected_directories)[:5]}"
        )
    for relative in expected_raw_files:
        peer = (
            relative.removesuffix("metadata.json") + "response.bin"
            if relative.endswith("metadata.json")
            else relative.removesuffix("response.bin") + "metadata.json"
        )
        if (relative in actual_files) != (peer in actual_files):
            existing = relative if relative in actual_files else peer
            if (
                not require_all_raw
                and existing.endswith("response.bin")
            ):
                continue
            raise RuntimeError("candidate raw snapshot transaction is incomplete")
    if require_all_raw and not expected_raw_files <= actual_files:
        missing = sorted(expected_raw_files - actual_files)
        raise RuntimeError(f"candidate raw snapshot requests are missing: {missing[:5]}")


def _publication_prefix(paths: list[Path], *, require_complete: bool) -> list[bool]:
    present = [os.path.lexists(path) for path in paths]
    seen_absent = False
    for exists in present:
        if not exists:
            seen_absent = True
        elif seen_absent:
            raise RuntimeError("candidate publication is not one contiguous prefix")
    if require_complete and not all(present):
        raise RuntimeError("candidate publication is incomplete")
    return present


def discover(args: argparse.Namespace) -> None:
    for attribute, label in (
        ("snapshot_dir", "candidate snapshot root"),
        ("out", "candidate table"),
        ("protocol", "Route-A protocol"),
        ("out_registry", "external registry"),
        ("out_lock", "external registry lock"),
    ):
        setattr(args, attribute, _inside_root(getattr(args, attribute), label=label))
    if args.protocol != _candidate_publication_lock_path().absolute():
        raise RuntimeError(
            "candidate publication requires the fixed canonical Route-A protocol"
        )
    lock_path = _inside_root(
        _candidate_publication_lock_path(),
        label="candidate publication transaction lock",
    )
    with candidate_publication_lock(
        lock_path,
        trusted_root=ROOT.resolve(),
        shared=bool(getattr(args, "check_existing", False)),
    ) as lock_handle:
        _discover_locked(args, lock_handle=lock_handle)


def _discover_locked(
    args: argparse.Namespace,
    *,
    lock_handle: AdvisoryLockHandle,
) -> None:
    states = parse_states(args.states)
    if args.freeze_selection and states != ROUTE_A_STATE_UNIVERSE:
        raise RuntimeError(
            "Route-A freezing requires the complete predeclared 34-state universe"
        )
    check_existing = bool(getattr(args, "check_existing", False))
    if check_existing and not args.freeze_selection:
        raise RuntimeError("--check-existing requires the complete frozen selection")
    sidecar = args.out.with_suffix(".provenance.json")
    index_path = args.snapshot_dir / "snapshot_index.json"
    publication_paths = [args.out, index_path, sidecar]
    if args.freeze_selection:
        publication_paths.extend([args.out_registry, args.out_lock])
    elif os.path.lexists(args.out_registry) or os.path.lexists(args.out_lock):
        raise RuntimeError("candidate selection exists but --freeze-selection was omitted")
    present = _publication_prefix(
        publication_paths, require_complete=check_existing
    )
    protocol_payload = _published_bytes(args.protocol, label="Route-A protocol")
    protocol = _strict_json_bytes(
        protocol_payload, label="Route-A protocol", require_canonical=False
    )

    expected_raw_files = _expected_raw_paths(states)
    derived_started = any(present)
    _audit_snapshot_tree(
        args.snapshot_dir,
        expected_raw_files=expected_raw_files,
        allow_index=present[1],
        require_all_raw=derived_started or check_existing,
    )
    store = SnapshotStore(
        args.snapshot_dir,
        offline=bool(args.offline or derived_started or check_existing),
        trusted_root=ROOT.resolve(),
    )
    state_frames = []
    request_records = []
    for state in states:
        url = build_usgs_candidate_url(state)
        payload, record = store.fetch(
            provider="usgs-nwis-confirmatory-site-metadata",
            url=url,
            headers={"User-Agent": USER_AGENT},
            retries=args.retries,
            resume_incomplete=True,
            expected_final_url=url,
        )
        frame = parse_usgs_candidate_metadata(payload, state=state)
        state_frames.append(frame)
        request_records.append({
            "state": state,
            "candidate_count": len(frame),
            "request_sha256": record.request_sha256,
            "response_sha256": record.response_sha256,
            "retrieved_at_utc": record.retrieved_at_utc,
            "byte_count": record.byte_count,
        })

    candidates = merge_candidate_metadata(state_frames)
    table_payload = candidates.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    existing_index_payload = (
        _published_bytes(index_path, label="candidate snapshot index")
        if present[1]
        else None
    )
    snapshot_audit = audit_candidate_snapshot_store(
        args.snapshot_dir,
        states,
        trusted_root=ROOT.resolve(),
        published_index_payload=existing_index_payload,
    )
    index_payload = snapshot_audit.index_payload
    provenance = {
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "protocol_sha256": sha256_bytes(protocol_payload),
        "state_universe": list(states),
        "state_universe_rule": CANDIDATE_STATE_UNIVERSE_RULE,
        "candidate_rule": CANDIDATE_SELECTION_RULE,
        "candidate_count": len(candidates),
        "site_primary_key": "site_no",
        "sort_order": ["site_no", "state"],
        "columns": list(candidates.columns),
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "raw_snapshot_index": os.path.relpath(index_path.absolute(), ROOT.resolve()),
        "raw_snapshot_index_sha256": sha256_bytes(index_payload),
        "candidate_table_sha256": sha256_bytes(table_payload),
        "requests": request_records,
    }
    provenance_payload = canonical_json_bytes(provenance)

    expected_payloads = [table_payload, index_payload, provenance_payload]
    labels = ["candidate table", "candidate snapshot index", "candidate provenance"]
    for path, expected, exists, label in zip(
        publication_paths[:3], expected_payloads, present[:3], labels, strict=True
    ):
        if exists:
            if _published_bytes(path, label=label) != expected:
                raise RuntimeError(f"{label} differs from canonical raw replay")
        elif check_existing:
            raise RuntimeError(f"{label} is absent")
        else:
            atomic_create(path, expected)

    _audit_snapshot_tree(
        args.snapshot_dir,
        expected_raw_files=expected_raw_files,
        allow_index=True,
        require_all_raw=True,
    )

    if args.freeze_selection:
        selection_seed = protocol["new_site_external_validation"]["selection_seed"]
        command = holdout_freeze_command(
            candidates=args.out,
            snapshot_index=index_path,
            candidate_provenance=sidecar,
            out_registry=args.out_registry,
            out_lock=args.out_lock,
            protocol=args.protocol,
            n_sites=args.n_sites,
            selection_seed=selection_seed,
            check_existing=check_existing,
            inherited_lock_fd=lock_handle.fd,
            inherited_lock_token=lock_handle.token,
        )
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            pass_fds=(lock_handle.fd,),
        )
    audit_candidate_evidence(
        args.out,
        sidecar,
        index_path,
        protocol_sha256=sha256_bytes(protocol_payload),
        state_universe=states,
        trusted_root=ROOT.resolve(),
    )
    if check_existing:
        print(json.dumps({
            "status": "CANDIDATE_PUBLICATION_VALID",
            "candidate_count": len(candidates),
            "state_count": len(states),
            "selection_frozen": True,
            "network_used": False,
        }, indent=2))
    else:
        print(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states", action="append",
        help=(
            "explicit two-letter state code(s), repeatable or comma-separated; "
            "default is the 34-state frozen development-support universe"
        ),
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help=(
            "perform a network-free full replay of the already published "
            "candidate table, snapshots, provenance, registry and lock"
        ),
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--freeze-selection", action="store_true",
        help="after discovery, invoke confirmatory_holdout.py to freeze 30 sites",
    )
    parser.add_argument(
        "--out-registry", type=Path,
        default=ROOT / "data_usgs" / "confirmatory_site_registry_v1.csv",
    )
    parser.add_argument(
        "--out-lock", type=Path,
        default=ROOT / "data_usgs" / "confirmatory_site_registry_v1.lock.json",
    )
    parser.add_argument("--n-sites", type=int, default=30)
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be positive")
    if args.n_sites != 30:
        parser.error("Route-A protocol requires exactly --n-sites 30")
    if args.check_existing:
        args.freeze_selection = True
        args.offline = True
    discover(args)


if __name__ == "__main__":
    main()
