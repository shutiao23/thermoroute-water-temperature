#!/usr/bin/env python3
"""Assemble Route-A's model suite without reading confirmation data.

The current pointer is written only when Stage 9, Stage 16 and the pooled
external training stage have all produced complete, checksum-valid components.
Stage 9, Stage 16, the separate Stage-09b matched-control matrix and Stage 25
are accepted only with their final content-bound completion receipts.  A missing report,
incomplete 31-member matrix or external model closure, key/budget drift,
interrupted transaction or stale receipt fails closed.  All four accepted
receipt paths and checksums become part of the frozen suite identity.
This command performs no fitting and has no network or post-2020 input path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tempfile


for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
from thermoroute.development_controls_gate import (  # noqa: E402
    STAGE09B_COMPLETION_RECEIPT_PATH,
    DevelopmentControlsGateError,
    validate_stage09b_completion_receipt,
)
from thermoroute.model_suite import (  # noqa: E402
    BUILTIN_MODELS,
    EXTERNAL_MODELS,
    MANDATORY_ABLATIONS,
    PRIMARY_MODELS,
    STAGE16_COMPLETION_RECEIPT_PATH,
    STAGE9_COMPLETION_RECEIPT_PATH,
    STAGE25_COMPLETION_RECEIPT_PATH,
    ModelSuiteError,
    builtin_entry,
    file_binding,
    freeze_model_suite,
    load_component_pointer,
    validate_stage16_completion_receipt,
    validate_stage09_completion_receipt,
    validate_stage25_completion_receipt,
)
from thermoroute.repro import (  # noqa: E402
    advisory_file_lock,
    assert_formal_numerical_policy,
    configure_deterministic_runtime,
    sha256_file,
    sha256_json,
)


configure_deterministic_runtime()


def _assert_stage24_policy() -> object:
    return assert_formal_numerical_policy(require_hash_randomization=True)


def _entries(pointer: dict) -> dict[str, dict]:
    return {str(entry["model_id"]): entry for entry in pointer["models"]}


def _canonical_stage24_path(
    path: Path,
    expected: Path,
    *,
    label: str,
    require_regular_file: bool,
) -> Path:
    """Keep authority inputs/outputs at one lexical, non-aliased location."""
    raw = path if path.is_absolute() else Path.cwd() / path
    lexical = Path(os.path.abspath(raw))
    canonical = Path(os.path.abspath(expected))
    if lexical != canonical:
        raise ModelSuiteError(f"Stage 24 {label} path is not canonical")
    current = lexical
    while current != ROOT:
        if current.is_symlink():
            raise ModelSuiteError(f"Stage 24 {label} path uses a symlink")
        current = current.parent
    if require_regular_file or lexical.exists():
        try:
            metadata = lexical.lstat()
        except OSError as exc:
            raise ModelSuiteError(f"Stage 24 {label} is absent") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ModelSuiteError(
                f"Stage 24 {label} is not a single-link regular file"
            )
    return lexical


def _load_verified_stage9(
    stage9_path: Path, receipt_path: Path, *, root: Path = ROOT,
) -> tuple[dict, dict[str, str]]:
    """Require the last-transaction receipt before accepting Stage-9 pointers."""
    receipt = validate_stage09_completion_receipt(
        receipt_path,
        root=root,
        stage9_pointer=stage9_path,
        publication_guard=_assert_stage24_policy,
    )
    _assert_stage24_policy()
    stage9 = load_component_pointer(stage9_path)
    if receipt.get("run_id") != stage9.get("run_id"):
        raise ModelSuiteError("Stage-9 receipt and component pointer run ids differ")
    return stage9, file_binding(root, receipt_path)


def _load_verified_stage09b(
    receipt_path: Path, *, root: Path = ROOT,
) -> tuple[dict, dict[str, str]]:
    """Require the exact 31-member control closure before suite freezing."""
    try:
        receipt = validate_stage09b_completion_receipt(
            receipt_path,
            root=root,
            publication_guard=_assert_stage24_policy,
        )
    except DevelopmentControlsGateError as exc:
        raise ModelSuiteError("Stage-09b development-controls gate failed") from exc
    return receipt, file_binding(root, receipt_path)


def _load_verified_stage16(
    pointer_path: Path, receipt_path: Path, *, root: Path = ROOT,
) -> tuple[dict, dict[str, str]]:
    """Require Stage 16's final content-bound LSTM completion receipt."""
    receipt = validate_stage16_completion_receipt(
        receipt_path,
        root=root,
        components_pointer=pointer_path,
        replay_selection=True,
        replay_bundle=True,
        publication_guard=_assert_stage24_policy,
    )
    _assert_stage24_policy()
    components = load_component_pointer(pointer_path)
    artifacts = receipt.get("artifacts")
    if (
        receipt.get("run_id") != components.get("run_id")
        or not isinstance(artifacts, dict)
        or artifacts.get("components_pointer")
        != file_binding(root, pointer_path)
    ):
        raise ModelSuiteError(
            "Stage-16 receipt and component pointer differ after validation"
        )
    return components, file_binding(root, receipt_path)


def _load_verified_stage25(
    pointer_path: Path, receipt_path: Path, *, root: Path = ROOT,
) -> tuple[dict, dict[str, str]]:
    """Require the last atomic Stage-25 receipt before suite freezing."""
    receipt = validate_stage25_completion_receipt(
        receipt_path,
        root=root,
        components_pointer=pointer_path,
    )
    _assert_stage24_policy()
    components = load_component_pointer(pointer_path)
    artifacts = receipt.get("artifacts")
    if (
        receipt.get("run_id") != components.get("run_id")
        or not isinstance(artifacts, dict)
        or artifacts.get("components_pointer")
        != file_binding(root, pointer_path)
    ):
        raise ModelSuiteError(
            "Stage-25 receipt and component pointer differ after validation"
        )
    return components, file_binding(root, receipt_path)


def _model_suite_id(
    *,
    protocol_sha256: str,
    stage9: dict,
    stage09_completion: dict[str, str],
    stage09b_completion: dict[str, str],
    stage16_completion: dict[str, str],
    stage25_completion: dict[str, str],
    lstm: dict,
    external: dict,
    features: tuple[str, ...],
) -> str:
    """Content-address the suite, including all four completion receipts."""
    return sha256_json({
        "protocol_sha256": protocol_sha256,
        "stage9": stage9,
        "stage09_completion": stage09_completion,
        "stage09b_completion": stage09b_completion,
        "stage16_completion": stage16_completion,
        "stage25_completion": stage25_completion,
        "lstm": lstm,
        "external": external,
        "features": features,
    })[:20]


def _run() -> None:
    _assert_stage24_policy()
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
        "--stage9-receipt", type=Path,
        default=ROOT / STAGE9_COMPLETION_RECEIPT_PATH,
    )
    parser.add_argument(
        "--stage09b-receipt",
        type=Path,
        default=ROOT / STAGE09B_COMPLETION_RECEIPT_PATH,
    )
    parser.add_argument(
        "--lstm", type=Path,
        default=C.MODELS / "route_a_lstm_components.json",
    )
    parser.add_argument(
        "--lstm-receipt", type=Path,
        default=ROOT / STAGE16_COMPLETION_RECEIPT_PATH,
    )
    parser.add_argument(
        "--external", type=Path,
        default=C.MODELS / "route_a_external_components.json",
    )
    parser.add_argument(
        "--external-receipt", type=Path,
        default=ROOT / STAGE25_COMPLETION_RECEIPT_PATH,
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

    args.protocol = _canonical_stage24_path(
        args.protocol,
        ROOT / "protocols" / "route_a_confirmatory_v1.json",
        label="protocol",
        require_regular_file=True,
    )
    args.current = _canonical_stage24_path(
        args.current,
        C.MODELS / "route_a_model_suite_current.json",
        label="current pointer",
        require_regular_file=False,
    )
    args.destination = _canonical_stage24_path(
        args.destination,
        ROOT / "data_usgs" / "confirmatory_model_suite_v1.json",
        label="opening registry",
        require_regular_file=False,
    )

    stage9, stage09_completion = _load_verified_stage9(
        args.stage9, args.stage9_receipt
    )
    controls_receipt, stage09b_completion = _load_verified_stage09b(
        args.stage09b_receipt
    )
    lstm, stage16_completion = _load_verified_stage16(
        args.lstm, args.lstm_receipt
    )
    external, stage25_completion = _load_verified_stage25(
        args.external, args.external_receipt
    )
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
    controls_identity = controls_receipt["run_identity"]
    controls_config = controls_receipt["formal_configuration"]
    controls_artifacts = controls_receipt["artifacts"]
    if (
        controls_identity["panel_sha256"] != contracts[0]["panel"]["sha256"]
        or controls_identity["registry_sha256"] != contracts[0]["registry"]["sha256"]
        or controls_identity["source_sha256"] != contracts[0]["source_sha256"]
        or controls_config["development_predictor_bridge"]
        != contracts[0]["predictor_bridge"]
        or controls_artifacts["frozen_panel_spec"]
        != contracts[0]["frozen_panel_spec"]
        or controls_artifacts["panel"] != contracts[0]["panel"]
        or controls_artifacts["registry"] != contracts[0]["registry"]
        or controls_artifacts["predictor_bridge"]
        != contracts[0]["predictor_bridge"]
    ):
        raise ModelSuiteError("Stage-09b receipt differs from the canonical development contract")

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
    suite_id = _model_suite_id(
        protocol_sha256=protocol_sha,
        stage9=stage9,
        stage09_completion=stage09_completion,
        stage09b_completion=stage09b_completion,
        stage16_completion=stage16_completion,
        stage25_completion=stage25_completion,
        lstm=lstm,
        external=external,
        features=feature_order,
    )
    versioned = _canonical_stage24_path(
        C.MODELS / f"route_a_model_suite_{suite_id}.json",
        C.MODELS / f"route_a_model_suite_{suite_id}.json",
        label="versioned suite",
        require_regular_file=False,
    )
    freeze_model_suite(
        versioned, args.current,
        root=ROOT, protocol_sha256=protocol_sha,
        temporal_entries=temporal, external_entries=external_models,
        actual_feature_order=feature_order,
        development_contract=contracts[0],
        stage09_completion=stage09_completion,
        stage09b_completion=stage09b_completion,
        stage16_completion=stage16_completion,
        stage25_completion=stage25_completion,
        registry_alias=args.destination,
        publication_guard=_assert_stage24_policy,
    )
    _assert_stage24_policy()
    print(f"frozen content-addressed Route-A model suite: {versioned}")
    print(f"frozen opening registry: {args.destination}")
    print(f"published current pointer: {args.current}")


def main() -> None:
    # Hold shared Stage-16 and Stage-25 transaction snapshots while validating
    # both receipts, reading their components, and freezing one generation.
    _assert_stage24_policy()
    with advisory_file_lock(C.STAGE16_TRANSACTION_LOCK, exclusive=False):
        with advisory_file_lock(C.STAGE25_TRANSACTION_LOCK, exclusive=False):
            _run()


if __name__ == "__main__":
    main()
