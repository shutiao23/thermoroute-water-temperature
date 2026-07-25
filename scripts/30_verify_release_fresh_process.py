#!/usr/bin/env python3
"""Create one same-host fresh-process PREOPEN release-mechanics receipt.

This controller has no command override and no post-opening mode.  Its only
child command is the byte- and Git-bound ``scripts/verify_release.py`` from this
repository, launched as the exact current interpreter with ``-I -B``.  A PASS
is an engineering acceptance observation on the same host, not a fresh-machine,
container, model-readiness, scientific-result, or deployment-performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.release_acceptance import (  # noqa: E402
    RECEIPT_RELATIVE,
    ReleaseAcceptanceError,
    run_release_mechanics_acceptance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help=(
            "existing local-only PREOPEN_NOT_COMPLETE dist/*.zip; the exact "
            "sibling .sha256 sidecar is mandatory"
        ),
    )
    args = parser.parse_args()
    try:
        document = run_release_mechanics_acceptance(
            args.archive,
            root=ROOT,
            receipt_path=ROOT / RECEIPT_RELATIVE,
        )
    except (OSError, ReleaseAcceptanceError, ValueError) as exc:
        print(f"release-mechanics acceptance failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": document["status"],
                "profile": document["profile"],
                "evidence_scope": document["evidence_scope"],
                "archive_sha256": document["archive"]["sha256"],
                "receipt": RECEIPT_RELATIVE,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
