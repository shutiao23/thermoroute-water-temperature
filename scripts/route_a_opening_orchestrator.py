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

from thermoroute.opening import (  # noqa: E402
    isolated_orchestrate_opening,
    isolated_validate_raw_preflight,
)
from thermoroute.provenance import canonical_json_bytes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--authorization", type=Path)
    mode.add_argument("--raw-preflight-work-order", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue only missing raw transactions under the existing intent",
    )
    parser.add_argument("--challenge")
    args = parser.parse_args()
    if args.raw_preflight_work_order is not None:
        if args.resume or args.challenge is None:
            parser.error(
                "raw preflight requires --challenge and prohibits --resume"
            )
        transcript = isolated_validate_raw_preflight(
            args.raw_preflight_work_order,
            root=ROOT,
            challenge=args.challenge,
        )
        sys.stdout.buffer.write(canonical_json_bytes(transcript))
        sys.stdout.buffer.flush()
        return
    if args.challenge is not None:
        parser.error("--challenge is limited to raw preflight mode")
    if args.authorization is None:  # argparse enforces the exclusive group
        parser.error("--authorization is required for opening mode")
    isolated_orchestrate_opening(
        args.authorization, root=ROOT, resume=args.resume
    )


if __name__ == "__main__":
    main()
