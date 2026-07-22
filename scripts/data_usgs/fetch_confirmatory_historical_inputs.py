#!/usr/bin/env python3
"""Freeze the outcome-free retrospective meteorology consumed by Route A.

Only ``site_no``, latitude and longitude are parsed from the already frozen
120-site temporal registry and 30-site external registry.  The script requests
Daymet and gridMET meteorology for the complete 32-day context through
2023-12-31, archives exact response bytes with request/retrieval/checksum
evidence, and creates complete site-by-day normalized Parquet tables.

No NWIS daily-value endpoint is called and no WTEMP/FLOW/WLEVEL value is parsed.
Use ``--offline`` to replay exclusively from an already populated raw snapshot
root.  All normalized outputs and the final manifest are create-only.
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
    with tempfile.TemporaryDirectory(prefix="thermoroute-inputs-pycache-") as cache:
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

from thermoroute.historical_inputs import (  # noqa: E402
    HistoricalInputError,
    acquire_historical_inputs,
)
from thermoroute.provenance import ProvenanceError  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
DEFAULT_TEMPORAL_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_EXTERNAL_REGISTRY = (
    ROOT / "data_usgs" / "confirmatory_site_registry_v1.csv"
)
DEFAULT_SNAPSHOT_ROOT = (
    ROOT / "data_usgs" / "raw_snapshots" / "confirmatory-historical-inputs-v1"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data_usgs" / "confirmatory_predictors" / "historical-retrospective-v1"
)
DEFAULT_MANIFEST = ROOT / "data_usgs" / "confirmatory_actual_inputs_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--temporal-registry", type=Path, default=DEFAULT_TEMPORAL_REGISTRY
    )
    parser.add_argument(
        "--external-registry", type=Path, default=DEFAULT_EXTERNAL_REGISTRY
    )
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--secondary-nwp-resolution",
        choices=("EXPLICITLY_NOT_USED",),
        default="EXPLICITLY_NOT_USED",
        help=(
            "Explicitly close the optional archived-NWP gate without using it; "
            "an acquired NWP claim requires its separate manifest verifier."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid network access and require every raw response in SnapshotStore.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.1,
        help="Polite delay after each station when network acquisition is enabled.",
    )
    args = parser.parse_args()
    try:
        manifest = acquire_historical_inputs(
            repo_root=ROOT,
            protocol_path=args.protocol,
            temporal_registry_path=args.temporal_registry,
            external_registry_path=args.external_registry,
            snapshot_root=args.snapshot_root,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            offline=args.offline,
            retries=args.retries,
            request_interval=args.request_interval,
            secondary_nwp_resolution=args.secondary_nwp_resolution,
        )
    except (HistoricalInputError, ProvenanceError, ValueError) as exc:
        parser.exit(2, f"FAIL-CLOSED: {exc}\n")
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(args.manifest),
        "history_start": manifest["history_start"],
        "target_end": manifest["target_end"],
        "temporal_sites": manifest["cohort_summaries"]["temporal"]["site_count"],
        "external_sites": manifest["cohort_summaries"]["external"]["site_count"],
        "contains_outcome_labels": False,
        "secondary_nwp_resolution": manifest["secondary_nwp_resolution"],
    }, indent=2))


if __name__ == "__main__":
    main()
