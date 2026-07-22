#!/usr/bin/env python3
"""Create or verify the outcome-free Route-A inference gate.

This entrypoint has no network operation and accepts no outcome, prediction,
model-output, or effect-vector path.  Its three evidence inputs are fixed inside
``thermoroute.inference_gate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.inference_gate import (  # noqa: E402
    DEFAULT_GATE_RELATIVE,
    InferenceGateError,
    build_inference_gate_document,
    exclusive_create_json,
    validate_inference_gate_document,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "check"),
        help="create the gate once or replay its complete validation",
    )
    parser.add_argument(
        "--gate", type=Path, default=ROOT / DEFAULT_GATE_RELATIVE,
    )
    args = parser.parse_args()
    try:
        if args.gate.resolve() != (ROOT / DEFAULT_GATE_RELATIVE).resolve():
            raise InferenceGateError(
                "gate artifact path must remain the frozen canonical path"
            )
        if args.command == "freeze":
            document = build_inference_gate_document(root=ROOT)
            exclusive_create_json(args.gate, document)
        else:
            document = validate_inference_gate_document(args.gate, root=ROOT)
    except InferenceGateError as exc:
        parser.exit(2, f"FAIL-CLOSED: {exc}\n")
    print(json.dumps({
        "status": document["status"],
        "claim_eligible": document["claim_eligible"],
        "analysis_mode": document["analysis_mode"],
        "gate": str(args.gate),
        "post_2020_outcomes_requested_or_inspected": False,
        "network_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
