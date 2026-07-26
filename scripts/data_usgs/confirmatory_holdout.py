#!/usr/bin/env python3
"""Freeze the metadata-only registry for the untouched Route-A holdout.

This command deliberately does *not* download holdout WTEMP labels.  It creates
the evidence that must predate label acquisition: a disjoint, deterministic site
registry and a lock tying it to the raw candidate-discovery snapshot, development
panel, protocol and selection seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
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


def _inherited_lock_fds_from_argv(argv: list[str]) -> tuple[int, ...]:
    """Preserve the parent's lock capability across the isolation re-exec."""
    option = "--inherited-publication-lock-fd"
    positions = [index for index, value in enumerate(argv) if value == option]
    if not positions:
        return ()
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError("inherited publication-lock descriptor is malformed")
    try:
        descriptor = int(argv[positions[0] + 1])
    except ValueError as exc:
        raise RuntimeError("inherited publication-lock descriptor is malformed") from exc
    if descriptor < 3:
        raise RuntimeError("inherited publication-lock descriptor is unsafe")
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError("inherited publication-lock descriptor is not open") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("inherited publication-lock descriptor is not regular")
    return (descriptor,)


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
            pass_fds=_inherited_lock_fds_from_argv(sys.argv[1:]),
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.evidence import (  # noqa: E402
    EvidenceError,
    FrozenPanelSpec,
    load_confirmatory_protocol,
    select_confirmatory_sites,
)
from thermoroute.confirmatory import (  # noqa: E402
    audit_candidate_evidence,
)
from thermoroute.provenance import (  # noqa: E402
    candidate_publication_lock,
    canonical_json_bytes,
    create_single_link_regular,
    read_single_link_regular,
    require_canonical_utc,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)
from thermoroute.repro import sha256_json  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"


def _candidate_publication_lock_path() -> Path:
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
    """Read one immutable publication without following or accepting links."""
    leftovers = _temporary_siblings(path)
    if leftovers:
        raise RuntimeError(
            f"{label} has unpublished temporary siblings: {leftovers[:3]}"
        )
    return read_single_link_regular(path, label=label, trusted_root=ROOT.resolve())


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
    try:
        create_single_link_regular(
            path, payload, trusted_root=ROOT.resolve(), mode=0o644
        )
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite holdout lock: {path}")


def verify_candidate_evidence(
    candidates_path: Path,
    provenance_path: Path,
    snapshot_index_path: Path,
    protocol_path: Path | None = None,
) -> pd.DataFrame:
    """Replay raw metadata responses and verify the derived candidate table."""
    input_parents = [
        Path(candidates_path).absolute().parent,
        Path(provenance_path).absolute().parent,
        Path(snapshot_index_path).absolute().parent,
    ]
    if protocol_path is not None:
        input_parents.append(Path(protocol_path).absolute().parent)
    trusted_root = Path(os.path.commonpath(input_parents))
    if protocol_path is None:
        protocol_sha = None
        states = None
    else:
        protocol_payload = read_single_link_regular(
            protocol_path,
            label="Route-A protocol",
            trusted_root=trusted_root,
        )
        try:
            protocol = json.loads(protocol_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Route-A protocol is invalid JSON") from exc
        protocol_sha = sha256_bytes(protocol_payload)
        states = protocol["metadata_candidate_contract"]["state_universe"]
    return audit_candidate_evidence(
        candidates_path,
        provenance_path,
        snapshot_index_path,
        protocol_sha256=protocol_sha,
        state_universe=states,
        trusted_root=trusted_root,
    ).candidates


def _prepare_publication(
    args: argparse.Namespace,
) -> tuple[bytes, dict[str, Any]]:
    protocol_payload = _published_bytes(args.protocol, label="Route-A protocol")
    try:
        protocol = json.loads(protocol_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Route-A protocol is invalid JSON") from exc
    if (
        not isinstance(protocol, dict)
        or protocol.get("schema_version") != 1
        or protocol.get("status") not in {
            "PLANNED_NOT_ACQUIRED", "FROZEN_NOT_ACQUIRED",
            "REGISTRY_FROZEN_LABELS_SEALED", "OPENED_ONCE",
        }
    ):
        raise RuntimeError("Route-A protocol contract changed")
    if protocol["new_site_external_validation"]["status"] != "PLANNED_NOT_ACQUIRED":
        raise RuntimeError("new-site candidate registry is already frozen or opened")
    planned = protocol["new_site_external_validation"]
    if args.n_sites != int(planned["planned_site_count"]):
        raise RuntimeError("site count differs from the predeclared protocol")
    if args.selection_seed != planned["selection_seed"]:
        raise RuntimeError("selection seed differs from the predeclared protocol")
    development_spec_payload = _published_bytes(
        args.development_spec, label="development panel specification"
    )
    development_spec_document = _strict_json_bytes(
        development_spec_payload, label="development panel specification"
    )
    if development_spec_document.get("schema_version") != 1:
        raise RuntimeError("development panel specification schema changed")
    development_spec = FrozenPanelSpec(
        spec_path=args.development_spec,
        document=development_spec_document,
    )
    development = development_spec.load_registry()
    try:
        candidate_audit = audit_candidate_evidence(
            args.candidates,
            args.candidate_provenance,
            args.candidate_snapshot_index,
            protocol_sha256=sha256_bytes(protocol_payload),
            state_universe=protocol["metadata_candidate_contract"]["state_universe"],
            trusted_root=ROOT.resolve(),
        )
    except EvidenceError as exc:
        raise RuntimeError(
            "candidate table cannot be rebuilt from exact raw snapshots"
        ) from exc
    candidates = candidate_audit.candidates
    if (
        protocol["metadata_candidate_contract"]["state_universe"]
        != candidate_audit.provenance["state_universe"]
    ):
        raise RuntimeError("candidate state universe differs from the protocol")
    retrieved_values = [
        str(record["retrieved_at_utc"])
        for record in candidate_audit.snapshot.index_document["records"]
    ]
    for value in retrieved_values:
        require_canonical_utc(value, label="candidate snapshot retrieved_at_utc")
    retrieved_min = min(retrieved_values, key=datetime.fromisoformat)
    retrieved_max = max(retrieved_values, key=datetime.fromisoformat)
    acquisition_seconds = (
        datetime.fromisoformat(retrieved_max)
        - datetime.fromisoformat(retrieved_min)
    ).total_seconds()
    if acquisition_seconds > 24 * 60 * 60:
        raise RuntimeError(
            "candidate acquisition exceeded its 24-hour session; use a new snapshot version"
        )
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
        "protocol_sha256": sha256_bytes(protocol_payload),
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
        "development_panel_spec_sha256": sha256_bytes(development_spec_payload),
        "candidate_table_sha256": sha256_bytes(candidate_audit.candidate_payload),
        "candidate_provenance_sha256": sha256_bytes(
            candidate_audit.provenance_payload
        ),
        "candidate_snapshot_index_sha256": sha256_bytes(
            candidate_audit.snapshot_index_payload
        ),
        "candidate_acquisition_session": {
            "maximum_duration_seconds": 86400,
            "retrieved_at_min_utc": retrieved_min,
            "retrieved_at_max_utc": retrieved_max,
            "clock_source": "LOCAL_SYSTEM_CLOCK_NOT_EXTERNALLY_ATTESTED",
        },
        "chronology_trust_boundary": (
            "LOCAL_HONEST_OWNER_ONLY_NO_EXTERNAL_TIMESTAMP_OR_CUSTODIAN"
        ),
        "confirmatory_registry_sha256": sha256_bytes(registry_payload),
        "frozen_artifacts": {
            "development_panel_spec": {
                "path": args.development_spec.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(development_spec_payload),
            },
            "candidate_table": {
                "path": args.candidates.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(candidate_audit.candidate_payload),
            },
            "candidate_provenance": {
                "path": args.candidate_provenance.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(candidate_audit.provenance_payload),
            },
            "candidate_snapshot_index": {
                "path": args.candidate_snapshot_index.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(candidate_audit.snapshot_index_payload),
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
    expected_keys = set(expected_stable) | {
        "registry_frozen_at_utc", "created_at_utc"
    }
    if set(document) != expected_keys:
        raise RuntimeError("external registry lock schema changed")
    session = document.get("candidate_acquisition_session")
    if (
        type(document.get("schema_version")) is not int
        or type(document.get("site_count")) is not int
        or type(document.get("opening_count")) is not int
        or type(session) is not dict
        or set(session) != {
            "maximum_duration_seconds", "retrieved_at_min_utc",
            "retrieved_at_max_utc", "clock_source",
        }
        or type(session.get("maximum_duration_seconds")) is not int
    ):
        raise RuntimeError("external registry lock scalar types changed")
    registry_frozen = require_canonical_utc(
        document.get("registry_frozen_at_utc"),
        label="external registry frozen timestamp",
    )
    created = require_canonical_utc(
        document.get("created_at_utc"),
        label="external registry lock timestamp",
    )
    raw_latest = str(
        expected_stable["candidate_acquisition_session"]["retrieved_at_max_utc"]
    )
    if not (
        datetime.fromisoformat(raw_latest)
        <= datetime.fromisoformat(registry_frozen)
        <= datetime.fromisoformat(created)
    ):
        raise RuntimeError("candidate raw/registry/lock chronology is invalid")
    observed_stable = dict(document)
    observed_stable.pop("created_at_utc")
    observed_stable.pop("registry_frozen_at_utc")
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
        shared=check_only,
        inherited_fd=getattr(args, "inherited_publication_lock_fd", None),
        inherited_token=getattr(args, "inherited_publication_lock_token", None),
    ):
        return _publish_or_validate_locked(args, check_only=check_only)


def _publish_or_validate_locked(
    args: argparse.Namespace,
    *,
    check_only: bool,
) -> dict[str, Any]:
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
    frozen_at = utc_now_iso()
    raw_latest = str(
        lock_stable["candidate_acquisition_session"]["retrieved_at_max_utc"]
    )
    if datetime.fromisoformat(frozen_at) < datetime.fromisoformat(raw_latest):
        raise RuntimeError("local clock predates the candidate acquisition evidence")
    lock = {
        **lock_stable,
        "registry_frozen_at_utc": frozen_at,
        "created_at_utc": frozen_at,
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
    parser.add_argument(
        "--inherited-publication-lock-fd",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inherited-publication-lock-token",
        default=None,
        help=argparse.SUPPRESS,
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
