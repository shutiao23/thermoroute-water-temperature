#!/usr/bin/env python3
"""Freeze, audit or execute the one-time Route-A confirmation opening.

``execute`` accepts only the frozen authorization.  It launches the fixed
isolated orchestrator, raw-only acquisition child and fresh trusted scorer; no
callback, module name, command, output path or alternate transport is accepted.
``resume`` accepts that same authorization only and can continue wholly absent
entries in the already-frozen request ledger under the original opening intent.

Examples
--------
PYTHONPATH=src python scripts/24_confirmatory_opening.py preflight \
  --authorization data_usgs/confirmatory_opening_authorization_v1.json

PYTHONPATH=src python scripts/24_confirmatory_opening.py status \
  --authorization data_usgs/confirmatory_opening_authorization_v1.json

PYTHONPATH=src python scripts/24_confirmatory_opening.py execute \
  --authorization data_usgs/confirmatory_opening_authorization_v1.json

PYTHONPATH=src python scripts/24_confirmatory_opening.py resume \
  --authorization data_usgs/confirmatory_opening_authorization_v1.json
"""

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
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        # Help is parsed before project imports below.  Do not recursively
        # re-exec merely to print usage, and prohibit incidental bytecode.
        sys.dont_write_bytecode = True
        return
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    if (
        sys.flags.isolated
        and prefix is not None
        and prefix != ROOT
        and ROOT not in prefix.parents
    ):
        return
    with tempfile.TemporaryDirectory(prefix="thermoroute-opening-pycache-") as cache:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                f"pycache_prefix={cache}",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()


DEFAULT_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
DEFAULT_DEVELOPMENT_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_EXTERNAL_REGISTRY = (
    ROOT / "data_usgs" / "confirmatory_site_registry_v1.csv"
)
DEFAULT_EXTERNAL_LOCK = (
    ROOT / "data_usgs" / "confirmatory_site_registry_v1.lock.json"
)
DEFAULT_MODEL_SUITE = ROOT / "data_usgs" / "confirmatory_model_suite_v1.json"
DEFAULT_INPUT_MANIFEST = ROOT / "data_usgs" / "confirmatory_actual_inputs_v1.json"
DEFAULT_DEVELOPMENT_REPLAY = (
    ROOT / "outputs" / "model_replay" / "route_a_development_replay_v1.json"
)
DEFAULT_PRELABEL_CHRONOLOGY = (
    ROOT / "outputs" / "prelabel" / "route_a_prelabel_chronology_v1.json"
)
DEFAULT_AUTHORIZATION = (
    ROOT / "data_usgs" / "confirmatory_opening_authorization_v1.json"
)
def _authorization_state_paths(path: Path) -> tuple[Path, Path]:
    document = json.loads(path.read_text(encoding="utf-8"))
    state = document.get("state_paths", {})
    intent = (ROOT / str(state.get("intent", ""))).resolve()
    receipt = (ROOT / str(state.get("receipt", ""))).resolve()
    if ROOT.resolve() not in intent.parents or ROOT.resolve() not in receipt.parents:
        raise OpeningContractError("authorization state path escapes repository root")
    return intent, receipt


def freeze(args: argparse.Namespace) -> None:
    document = freeze_opening_authorization(
        args.authorization,
        root=ROOT,
        protocol_path=args.protocol,
        development_registry=args.development_registry,
        external_registry=args.external_registry,
        external_lock=args.external_lock,
        model_suite=args.model_suite,
        input_manifest=args.input_manifest,
        development_replay_receipt=args.development_replay_receipt,
        prelabel_chronology_receipt=args.prelabel_chronology_receipt,
    )
    print(json.dumps({
        "status": document["status"],
        "opening_id": document["opening_id"],
        "authorization": str(args.authorization),
        "labels_opened": False,
    }, indent=2))


def preflight(args: argparse.Namespace) -> None:
    result = validate_authorization(args.authorization, root=ROOT)
    print(json.dumps({
        "status": "PREFLIGHT_VALID_LABELS_STILL_SEALED",
        "opening_id": result["authorization"]["opening_id"],
        "actual_feature_order": list(result["suite"]["feature_order"]),
        "required_models": {
            cohort: list(models)
            for cohort, models in result["suite"]["required_models"].items()
        },
        "temporal_sites": len(result["registries"]["development"]),
        "external_sites": len(result["registries"]["external"]),
        "runtime_sha256": result["runtime"]["runtime_sha256"],
        "fixed_code_sha256": result["fixed_code"]["sha256"],
        "prelabel_chronology_status": result["prelabel_chronology"]["status"],
        "model_freeze_commit": result["prelabel_chronology"]["order"]
        ["model_freeze_commit"],
        "input_evidence_commit": result["prelabel_chronology"]["order"]
        ["input_evidence_commit"],
        "state_namespace": result["authorization"]["state_paths"]["namespace"],
        "labels_opened": False,
    }, indent=2))


def status(args: argparse.Namespace) -> None:
    if not args.authorization.is_file():
        print(json.dumps({
            "status": "NOT_AUTHORIZED_LABELS_SEALED",
            "authorization": str(args.authorization),
        }, indent=2))
        return
    intent, receipt = _authorization_state_paths(args.authorization)
    state = opening_status(intent_path=intent, receipt_path=receipt)
    receipt_valid = False
    raw_transport_resume_allowed = False
    transport = None
    forbidden_existing_outputs: list[str] = []
    if receipt.exists():
        try:
            validate_completed_receipt(args.authorization, root=ROOT)
            receipt_valid = True
        except OpeningContractError:
            state = "CORRUPT_OR_INCOMPLETE_OPENING_RECEIPT"
    elif state == "OPENING_INCOMPLETE_SAME_OPENING_RESUME_REQUIRES_VALIDATION":
        try:
            inspection = inspect_same_opening_transport_resume(
                args.authorization, root=ROOT
            )
        except OpeningContractError:
            state = "OPENING_INDETERMINATE_INVALID_AUTHORIZATION_NO_RAW_RESUME"
        else:
            state = inspection["status"]
            raw_transport_resume_allowed = bool(
                inspection["raw_transport_resume_allowed"]
            )
            transport = inspection["transport"]
            forbidden_existing_outputs = list(
                inspection["forbidden_existing_outputs"]
            )
    print(json.dumps({
        "status": state,
        "authorization": str(args.authorization),
        "intent_exists": intent.exists(),
        "receipt_exists": receipt.exists(),
        "receipt_valid": receipt_valid,
        "raw_transport_resume_allowed": raw_transport_resume_allowed,
        "transport": transport,
        "forbidden_existing_outputs": forbidden_existing_outputs,
    }, indent=2))


def execute(args: argparse.Namespace) -> None:
    receipt = run_opening_once(args.authorization, root=ROOT)
    print(json.dumps({
        "status": receipt["status"],
        "opening_id": receipt["opening_id"],
        "opening_count": receipt["opening_count"],
        "state_namespace": receipt["state_paths"]["namespace"],
        "receipt": receipt["state_paths"]["receipt"],
        "receipt_sha256": receipt["state_paths"]["receipt_sha256"],
    }, indent=2))


def resume(args: argparse.Namespace) -> None:
    receipt = resume_opening_once(args.authorization, root=ROOT)
    print(json.dumps({
        "status": receipt["status"],
        "opening_id": receipt["opening_id"],
        "opening_count": receipt["opening_count"],
        "transport_resume": "COMPLETED_UNDER_ORIGINAL_INTENT",
        "state_namespace": receipt["state_paths"]["namespace"],
        "receipt": receipt["state_paths"]["receipt"],
        "receipt_sha256": receipt["state_paths"]["receipt_sha256"],
    }, indent=2))


def main() -> None:
    global OpeningContractError
    global freeze_opening_authorization
    global inspect_same_opening_transport_resume
    global opening_status
    global resume_opening_once
    global run_opening_once
    global validate_authorization
    global validate_completed_receipt

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze_parser = sub.add_parser("freeze-authorization")
    freeze_parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    freeze_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze_parser.add_argument(
        "--development-registry", type=Path, default=DEFAULT_DEVELOPMENT_REGISTRY
    )
    freeze_parser.add_argument(
        "--external-registry", type=Path, default=DEFAULT_EXTERNAL_REGISTRY
    )
    freeze_parser.add_argument("--external-lock", type=Path, default=DEFAULT_EXTERNAL_LOCK)
    freeze_parser.add_argument("--model-suite", type=Path, default=DEFAULT_MODEL_SUITE)
    freeze_parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    freeze_parser.add_argument(
        "--development-replay-receipt",
        type=Path,
        default=DEFAULT_DEVELOPMENT_REPLAY,
    )
    freeze_parser.add_argument(
        "--prelabel-chronology-receipt",
        type=Path,
        default=DEFAULT_PRELABEL_CHRONOLOGY,
    )
    freeze_parser.set_defaults(func=freeze)

    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_AUTHORIZATION
    )
    preflight_parser.set_defaults(func=preflight)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    status_parser.set_defaults(func=status)

    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_AUTHORIZATION
    )
    execute_parser.set_defaults(func=execute)

    resume_parser = sub.add_parser(
        "resume",
        help="continue only missing raw requests under the existing opening intent",
    )
    resume_parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_AUTHORIZATION
    )
    resume_parser.set_defaults(func=resume)

    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from thermoroute.opening import (
        OpeningContractError,
        freeze_opening_authorization,
        inspect_same_opening_transport_resume,
        opening_status,
        resume_opening_once,
        run_opening_once,
        validate_authorization,
        validate_completed_receipt,
    )

    try:
        args.func(args)
    except OpeningContractError as exc:
        parser.exit(2, f"FAIL-CLOSED: {exc}\n")


if __name__ == "__main__":
    main()
