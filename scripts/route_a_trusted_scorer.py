#!/usr/bin/env python3
"""Fixed fresh-process scorer and receipt authority for Route-A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.opening import (  # noqa: E402
    isolated_score_and_receipt,
    isolated_verify_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--work-order", type=Path)
    mode.add_argument("--verify-release", action="store_true")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.verify_release:
        if args.authorization is None:
            parser.error("--verify-release requires --authorization")
        result = isolated_verify_release(args.authorization, root=ROOT)
        print(json.dumps(result, sort_keys=True))
        return
    if args.authorization is not None:
        parser.error("--authorization is valid only with --verify-release")
    isolated_score_and_receipt(args.work_order, root=ROOT)


if __name__ == "__main__":
    main()
