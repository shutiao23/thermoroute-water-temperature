#!/usr/bin/env python3
"""Fixed raw-only NWIS acquisition/ledger-resume child for Route-A."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue only missing requests from the frozen ledger",
    )
    args = parser.parse_args()
    # Argparse handles --help before any project import.  This keeps a help
    # probe read-only and avoids writing repository-local module bytecode.
    sys.path.insert(0, str(ROOT / "src"))
    from thermoroute.outcome_acquisition import acquire_from_work_order

    acquire_from_work_order(
        args.work_order,
        root=ROOT,
        entrypoint_path=__file__,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
