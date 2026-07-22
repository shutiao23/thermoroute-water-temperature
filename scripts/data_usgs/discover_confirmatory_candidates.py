#!/usr/bin/env python3
"""Discover and optionally freeze Route-A new-site candidates from metadata only.

The USGS request is made against ``/nwis/site/``.  It advertises that a stream
site has a daily-value water-temperature data type, but requests neither values,
dates nor holdout-period coverage.  Exact RDB bytes are stored by SnapshotStore.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


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
    CANDIDATE_USER_AGENT,
    ROUTE_A_STATE_UNIVERSE,
    build_usgs_candidate_url,
    merge_candidate_metadata,
    normalise_states,
    parse_usgs_candidate_metadata,
)
from thermoroute.provenance import (  # noqa: E402
    SnapshotStore,
    canonical_json_bytes,
    sha256_file,
)


DEFAULT_SNAPSHOT_DIR = (
    ROOT / "data_usgs" / "raw_snapshots" / "confirmatory-candidates-v1"
)
DEFAULT_OUT = ROOT / "data_usgs" / "confirmatory_candidate_sites_v1.csv"
DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
USER_AGENT = CANDIDATE_USER_AGENT


def atomic_create(path: Path, payload: bytes) -> None:
    """Publish a new immutable derived file; never replace an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A race with another creator is treated as a hard failure.
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


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
) -> list[str]:
    """Return the exact invocation of the existing sealed-holdout freezer."""
    return [
        sys.executable,
        str(ROOT / "scripts" / "data_usgs" / "confirmatory_holdout.py"),
        "freeze-candidates",
        "--protocol", str(protocol),
        "--candidates", str(candidates),
        "--candidate-snapshot-index", str(snapshot_index),
        "--candidate-provenance", str(candidate_provenance),
        "--out-registry", str(out_registry),
        "--out-lock", str(out_lock),
        "--n-sites", str(n_sites),
        "--selection-seed", selection_seed,
    ]


def discover(args: argparse.Namespace) -> None:
    states = parse_states(args.states)
    if args.freeze_selection and states != ROUTE_A_STATE_UNIVERSE:
        raise RuntimeError(
            "Route-A freezing requires the complete predeclared 34-state universe"
        )
    sidecar = args.out.with_suffix(".provenance.json")
    for path in (args.out, sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)

    store = SnapshotStore(args.snapshot_dir, offline=args.offline)
    state_frames = []
    request_records = []
    for state in states:
        url = build_usgs_candidate_url(state)
        payload, record = store.fetch(
            provider="usgs-nwis-confirmatory-site-metadata",
            url=url,
            headers={"User-Agent": USER_AGENT},
            retries=args.retries,
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
    payload = candidates.to_csv(index=False, lineterminator="\n").encode("utf-8")
    atomic_create(args.out, payload)
    index_path = store.write_index()
    provenance = {
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "protocol_sha256": sha256_file(args.protocol),
        "state_universe": list(states),
        "state_universe_rule": (
            "states represented in the frozen 120-site development registry; "
            "no post-2020 outcome or coverage information"
        ),
        "candidate_rule": (
            "USGS stream sites whose site metadata advertises daily-value "
            "parameter 00010 capability; siteStatus=all"
        ),
        "candidate_count": len(candidates),
        "site_primary_key": "site_no",
        "sort_order": ["site_no", "state"],
        "columns": list(candidates.columns),
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "raw_snapshot_index": os.path.relpath(index_path.resolve(), ROOT),
        "raw_snapshot_index_sha256": sha256_file(index_path),
        "candidate_table_sha256": sha256_file(args.out),
        "requests": request_records,
    }
    atomic_create(sidecar, canonical_json_bytes(provenance))
    print(json.dumps(provenance, indent=2))

    if args.freeze_selection:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
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
        )
        subprocess.run(command, cwd=ROOT, check=True)


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
    discover(args)


if __name__ == "__main__":
    main()
