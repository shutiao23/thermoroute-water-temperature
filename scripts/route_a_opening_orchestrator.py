#!/usr/bin/env python3
"""Fixed isolated orchestrator for the one-time Route-A opening/transport resume.

This file is invoked only through ``python -I`` by the public opening API.  It
accepts an authorization path and fixed resume flag, never a callback, module
name, output path, alternate request, or command.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.opening import isolated_orchestrate_opening  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue only missing raw transactions under the existing intent",
    )
    args = parser.parse_args()
    isolated_orchestrate_opening(args.authorization, root=ROOT, resume=args.resume)


if __name__ == "__main__":
    main()
