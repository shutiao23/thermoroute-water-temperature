#!/usr/bin/env python3
"""Freeze or verify Route-A's repository-internal pre-label Git chronology."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


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
    with tempfile.TemporaryDirectory(prefix="thermoroute-stage28-pycache-") as cache:
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

from thermoroute.chronology import (  # noqa: E402
    ChronologyError,
    DEFAULT_CANDIDATE_PROVENANCE,
    DEFAULT_CANDIDATE_SNAPSHOT_INDEX,
    DEFAULT_CANDIDATE_TABLE,
    DEFAULT_DEVELOPMENT_REPLAY,
    DEFAULT_EXTERNAL_LOCK,
    DEFAULT_EXTERNAL_REGISTRY,
    DEFAULT_INPUT_MANIFEST,
    DEFAULT_MODEL_SUITE,
    DEFAULT_PROTOCOL_SEAL,
    DEFAULT_RECEIPT,
    freeze_prelabel_chronology,
    validate_prelabel_chronology,
)


def _path(relative: str) -> Path:
    return ROOT / relative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=_path(DEFAULT_RECEIPT))
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--model-freeze-commit",
        help="full or unambiguous commit containing the frozen suite and replay",
    )
    parser.add_argument(
        "--input-evidence-commit",
        help="later commit containing candidate and retrospective-input evidence",
    )
    parser.add_argument("--protocol-seal", type=Path, default=_path(DEFAULT_PROTOCOL_SEAL))
    parser.add_argument("--model-suite", type=Path, default=_path(DEFAULT_MODEL_SUITE))
    parser.add_argument(
        "--development-replay", type=Path, default=_path(DEFAULT_DEVELOPMENT_REPLAY)
    )
    parser.add_argument(
        "--candidate-table", type=Path, default=_path(DEFAULT_CANDIDATE_TABLE)
    )
    parser.add_argument(
        "--candidate-provenance",
        type=Path,
        default=_path(DEFAULT_CANDIDATE_PROVENANCE),
    )
    parser.add_argument(
        "--candidate-snapshot-index",
        type=Path,
        default=_path(DEFAULT_CANDIDATE_SNAPSHOT_INDEX),
    )
    parser.add_argument(
        "--external-registry", type=Path, default=_path(DEFAULT_EXTERNAL_REGISTRY)
    )
    parser.add_argument("--external-lock", type=Path, default=_path(DEFAULT_EXTERNAL_LOCK))
    parser.add_argument("--input-manifest", type=Path, default=_path(DEFAULT_INPUT_MANIFEST))
    args = parser.parse_args()

    if args.check and (args.model_freeze_commit or args.input_evidence_commit):
        parser.error("--check reads commits from the receipt; do not pass commit options")
    if not args.check and (
        not args.model_freeze_commit or not args.input_evidence_commit
    ):
        parser.error(
            "freezing requires explicit --model-freeze-commit and "
            "--input-evidence-commit"
        )
    try:
        if args.check:
            document = validate_prelabel_chronology(args.receipt, root=ROOT)
        else:
            document = freeze_prelabel_chronology(
                args.receipt,
                root=ROOT,
                model_freeze_commit=args.model_freeze_commit,
                input_evidence_commit=args.input_evidence_commit,
                protocol_seal=args.protocol_seal,
                model_suite=args.model_suite,
                development_replay=args.development_replay,
                candidate_table=args.candidate_table,
                candidate_provenance=args.candidate_provenance,
                candidate_snapshot_index=args.candidate_snapshot_index,
                external_registry=args.external_registry,
                external_lock=args.external_lock,
                input_manifest=args.input_manifest,
            )
    except ChronologyError as exc:
        parser.exit(2, f"FAIL-CLOSED: {exc}\n")
    print(
        json.dumps(
            {
                "status": document["status"],
                "receipt": str(args.receipt),
                "model_freeze_commit": document["order"]["model_freeze_commit"],
                "input_evidence_commit": document["order"]["input_evidence_commit"],
                "external_timestamp_or_public_preregistration": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
