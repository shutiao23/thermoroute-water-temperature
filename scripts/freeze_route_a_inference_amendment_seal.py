#!/usr/bin/env python3
"""Create or verify the separate Route-A inference-amendment lineage seal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.inference_gate import (  # noqa: E402
    AMENDMENT_SEAL_RELATIVE,
    InferenceGateError,
    build_inference_amendment_seal_document,
    exclusive_create_json,
    validate_inference_amendment_seal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check"))
    parser.add_argument(
        "--seal", type=Path, default=ROOT / AMENDMENT_SEAL_RELATIVE,
    )
    parser.add_argument(
        "--final-prelabel-commit",
        help="full commit containing the amendment bytes; required by freeze",
    )
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            if not args.final_prelabel_commit:
                parser.error("freeze requires --final-prelabel-commit")
            document = build_inference_amendment_seal_document(
                root=ROOT, final_prelabel_commit=args.final_prelabel_commit,
            )
            exclusive_create_json(args.seal, document)
        else:
            document = validate_inference_amendment_seal(args.seal, root=ROOT)
    except InferenceGateError as exc:
        parser.exit(2, f"FAIL-CLOSED: {exc}\n")
    print(json.dumps({
        "status": document["status"],
        "amendment_id": document["amendment_id"],
        "seal": str(args.seal),
        "post_2020_outcomes_requested_or_inspected": False,
    }, indent=2))


if __name__ == "__main__":
    main()
