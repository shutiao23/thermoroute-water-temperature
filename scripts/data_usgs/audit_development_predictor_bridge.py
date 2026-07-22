#!/usr/bin/env python3
"""Archive and audit the 2018--2020 Daymet/gridMET product bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.historical_inputs import HistoricalInputError  # noqa: E402
from thermoroute.predictor_bridge import (  # noqa: E402
    PredictorBridgeError,
    acquire_predictor_bridge,
    migrate_development_bridge_metadata_indexes_v2,
)
from thermoroute.provenance import ProvenanceError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path,
        default=ROOT / "data_usgs" / "panel_usgs_120v2.parquet",
    )
    parser.add_argument(
        "--registry", type=Path,
        default=ROOT / "data_usgs" / "station_registry_v1.csv",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=ROOT / "data_usgs" / "raw_snapshots" / "development-predictor-bridge-v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data_usgs" / "development_predictor_bridge_v1",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data_usgs" / "development_predictor_bridge_v1.json",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--migrate-metadata-index-v2",
        action="store_true",
        help=(
            "offline-only: preserve legacy indexes, create metadata-byte-bound v2 "
            "indexes, replay current parsers, and atomically rebind the manifest"
        ),
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help=(
            "Archive and index raw responses but publish no normalized/report/manifest "
            "artifact; rerun final committed code with --offline."
        ),
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-interval", type=float, default=0.1)
    parser.add_argument("--prefetch-workers", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.migrate_metadata_index_v2:
            if args.prefetch_only:
                parser.error(
                    "--migrate-metadata-index-v2 and --prefetch-only are mutually exclusive"
                )
            manifest = migrate_development_bridge_metadata_indexes_v2(
                repo_root=ROOT, manifest_path=args.manifest,
            )
        else:
            manifest = acquire_predictor_bridge(
                repo_root=ROOT,
                panel_path=args.panel,
                registry_path=args.registry,
                snapshot_root=args.snapshot_root,
                output_dir=args.output_dir,
                manifest_path=args.manifest,
                offline=args.offline,
                retries=args.retries,
                request_interval=args.request_interval,
                prefetch_only=args.prefetch_only,
                prefetch_workers=args.prefetch_workers,
            )
    except (PredictorBridgeError, HistoricalInputError, ProvenanceError, ValueError) as exc:
        parser.exit(2, f"FAIL-CLOSED: {exc}\n")
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(args.manifest),
        "outcome_values_requested_or_read": False,
    }, indent=2))
    if manifest["status"] == "RAW_PREDICTOR_BRIDGE_PREFETCH_COMPLETE":
        return
    if manifest["status"] != "PASS_EXACT_PRODUCT_BRIDGE":
        parser.exit(2, "NO-GO: development predictor product bridge failed\n")


if __name__ == "__main__":
    main()
