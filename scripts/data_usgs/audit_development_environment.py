#!/usr/bin/env python3
"""Create the deterministic pre-opening environmental-scope audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.environmental_audit import (  # noqa: E402
    EnvironmentalAuditError,
    write_environmental_audit,
)


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
        "--rejected", type=Path,
        default=ROOT / "data_usgs" / "rejected_sites_120v2.csv",
    )
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "data_usgs" / "development_environmental_audit_v1.json",
    )
    parser.add_argument(
        "--markdown", type=Path,
        default=ROOT / "outputs" / "reports" / "development_environmental_audit_v1.md",
    )
    args = parser.parse_args()
    try:
        document = write_environmental_audit(
            panel_path=args.panel,
            registry_path=args.registry,
            rejected_path=args.rejected,
            json_path=args.json,
            markdown_path=args.markdown,
        )
    except (EnvironmentalAuditError, OSError, ValueError) as exc:
        parser.exit(2, f"FAIL-CLOSED: {exc}\n")
    print(json.dumps({
        "status": document["status"],
        "post_2020_values_read": False,
        "station_count": document["panel"]["station_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
