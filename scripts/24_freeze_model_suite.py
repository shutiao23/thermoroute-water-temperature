#!/usr/bin/env python3
"""Assemble Route-A's model suite without reading confirmation data.

The current pointer is written only when Stage 9, Stage 16 and the pooled
external training stage have all produced complete, checksum-valid components.
This command performs no fitting and has no network or post-2020 input path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage24-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE24_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE24_NONCE"


def _isolate_project_bytecode() -> None:
    if __name__ != "__main__":
        return
    worker_cache = os.environ.get(_WORKER_CACHE_ENV)
    worker_nonce = os.environ.get(_WORKER_NONCE_ENV)
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    worker_argument = len(sys.argv) > 1 and sys.argv[1] == _WORKER_ARGUMENT
    if worker_cache is not None or worker_nonce is not None or worker_argument:
        if not (worker_cache and worker_nonce and worker_argument):
            raise RuntimeError("Stage 24 formal worker handshake is incomplete")
        expected = Path(worker_cache).resolve()
        flags = (
            int(sys.flags.isolated), int(sys.flags.ignore_environment),
            int(sys.flags.no_user_site), bool(sys.flags.safe_path),
            int(sys.flags.dont_write_bytecode),
        )
        if (
            flags != (1, 1, 1, True, 0)
            or prefix != expected
            or not expected.is_dir()
            or expected == ROOT
            or ROOT in expected.parents
            or (expected / ".controller-nonce").read_text(encoding="utf-8")
            != worker_nonce
        ):
            raise RuntimeError("Stage 24 formal worker isolation contract failed")
        sys.argv.pop(1)
        return
    with tempfile.TemporaryDirectory(prefix="thermoroute-stage24-pycache-") as cache:
        cache_path = Path(cache).resolve()
        if any(cache_path.iterdir()):
            raise RuntimeError("Stage 24 controller pycache was not initially empty")
        nonce = secrets.token_hex(32)
        (cache_path / ".controller-nonce").write_text(nonce, encoding="utf-8")
        environment = os.environ.copy()
        environment[_WORKER_CACHE_ENV] = str(cache_path)
        environment[_WORKER_NONCE_ENV] = nonce
        result = subprocess.run(
            [sys.executable, "-I", "-X", f"pycache_prefix={cache}",
             str(Path(__file__).resolve()), _WORKER_ARGUMENT, *sys.argv[1:]],
            cwd=ROOT,
            env=environment,
            check=False,
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute.model_suite import (  # noqa: E402
    BUILTIN_MODELS,
    EXTERNAL_MODELS,
    MANDATORY_ABLATIONS,
    PRIMARY_MODELS,
    ModelSuiteError,
    builtin_entry,
    freeze_model_suite,
    load_component_pointer,
)
from thermoroute.repro import sha256_file, sha256_json  # noqa: E402


def _entries(pointer: dict) -> dict[str, dict]:
    return {str(entry["model_id"]): entry for entry in pointer["models"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "protocols" / "route_a_confirmatory_v1.json",
    )
    parser.add_argument(
        "--stage9", type=Path,
        default=C.MODELS / "route_a_stage9_components.json",
    )
    parser.add_argument(
        "--lstm", type=Path,
        default=C.MODELS / "route_a_lstm_components.json",
    )
    parser.add_argument(
        "--external", type=Path,
        default=C.MODELS / "route_a_external_components.json",
    )
    parser.add_argument(
        "--current", type=Path,
        default=C.MODELS / "route_a_model_suite_current.json",
    )
    parser.add_argument(
        "--destination", type=Path,
        default=ROOT / "data_usgs" / "confirmatory_model_suite_v1.json",
        help="direct frozen registry consumed by the opening preflight",
    )
    args = parser.parse_args()

    stage9 = load_component_pointer(args.stage9)
    lstm = load_component_pointer(args.lstm)
    external = load_component_pointer(args.external)
    if stage9.get("cohort") != "temporal_stage9":
        raise ModelSuiteError("Stage-9 component pointer has the wrong cohort")
    if lstm.get("cohort") != "temporal_lstm":
        raise ModelSuiteError("LSTM component pointer has the wrong cohort")
    if external.get("cohort") != "external":
        raise ModelSuiteError("external component pointer has the wrong cohort")
    feature_order = tuple(stage9["raw_feature_order"])
    if tuple(lstm["raw_feature_order"]) != feature_order:
        raise ModelSuiteError("Stage-9 and LSTM raw schemas differ")
    if tuple(external["raw_feature_order"]) != feature_order:
        raise ModelSuiteError("temporal and external raw schemas differ")
    contracts = [
        stage9.get("development_contract"),
        lstm.get("development_contract"),
        external.get("development_contract"),
    ]
    if any(value is None for value in contracts) or not all(
        value == contracts[0] for value in contracts[1:]
    ):
        raise ModelSuiteError("component pointers do not share one canonical source/data contract")

    stage9_entries = _entries(stage9)
    expected_stage9 = {"ThermoRoute", "LightGBM", *MANDATORY_ABLATIONS}
    if set(stage9_entries) != expected_stage9:
        raise ModelSuiteError("Stage-9 component pointer is incomplete")
    lstm_entries = _entries(lstm)
    if set(lstm_entries) != {"LSTM"}:
        raise ModelSuiteError("temporal LSTM component pointer is incomplete")
    external_entries = _entries(external)
    expected_external_learned = {"ThermoRoute", "LightGBM", "LSTM"}
    if set(external_entries) != expected_external_learned:
        raise ModelSuiteError("external learned component pointer is incomplete")

    builtins = [builtin_entry(model, feature_order) for model in PRIMARY_MODELS
                if model in BUILTIN_MODELS]
    temporal = [*builtins]
    for model in PRIMARY_MODELS:
        if model in stage9_entries:
            temporal.append(stage9_entries[model])
        elif model in lstm_entries:
            temporal.append(lstm_entries[model])
    temporal.extend(stage9_entries[name] for name in MANDATORY_ABLATIONS)
    external_models = [*builtins]
    external_models.extend(external_entries[model] for model in EXTERNAL_MODELS
                           if model not in BUILTIN_MODELS)

    protocol_sha = sha256_file(args.protocol)
    suite_id = sha256_json({
        "protocol_sha256": protocol_sha,
        "stage9": stage9,
        "lstm": lstm,
        "external": external,
        "features": feature_order,
    })[:20]
    versioned = C.MODELS / f"route_a_model_suite_{suite_id}.json"
    freeze_model_suite(
        versioned, args.current,
        root=ROOT, protocol_sha256=protocol_sha,
        temporal_entries=temporal, external_entries=external_models,
        actual_feature_order=feature_order,
        development_contract=contracts[0],
        registry_alias=args.destination,
    )
    print(f"frozen content-addressed Route-A model suite: {versioned}")
    print(f"frozen opening registry: {args.destination}")
    print(f"published current pointer: {args.current}")


if __name__ == "__main__":
    main()
