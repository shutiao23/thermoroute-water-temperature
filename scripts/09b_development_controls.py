#!/usr/bin/env python3
"""Development-only matched-budget neural controls and feature ladder.

This entry point is deliberately separate from the sealed Route-A confirmatory
suite.  It reads only the frozen 2006--2020 development panel, requires the
outcome-free exact-product predictor bridge to pass, and never writes a model
suite pointer.  In particular, the 2019--2020 partition is an already-inspected
development evaluation, not a blind test.

The complete default matrix is:

* PlainMLP and PlainCausalTCN on all seven Route-A variables, all five frozen
  USGS seeds, with parameter counts closely matched to ThermoRoute; and
* a cumulative ThermoRoute feature ladder on every declared Stage-09b seed.

Each arm has a safe resumable training checkpoint and a create-only prediction
artifact.  Combined artifacts are published only after the exact arm/seed
registry and the common forecast-key registry both validate.
"""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile


_IMPORT_SAFE_THREAD_DEFAULT = os.environ.get("OMP_NUM_THREADS") or "1"
STAGE09B_MEMBER_THREADS = int(
    os.environ.get("THERMOROUTE_FORMAL_THREADS")
    or ("8" if __name__ == "__main__" else _IMPORT_SAFE_THREAD_DEFAULT)
)

if __name__ == "__main__":
    for _thread_variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_thread_variable, str(STAGE09B_MEMBER_THREADS))
    os.environ.setdefault("THERMOROUTE_FORMAL_THREADS", str(STAGE09B_MEMBER_THREADS))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage09b-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE09B_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE09B_NONCE"
_MEMBER_WORK_ORDER_OPTION = "--_thermoroute-stage09b-member-work-order"


def _formal_worker_environment(
    cache: Path, nonce: str, threads: int | None = None,
) -> dict[str, str]:
    """Return the complete allowlisted Stage-09b worker environment."""
    if threads is None:
        threads = int(
            os.environ.get("THERMOROUTE_FORMAL_THREADS")
            or STAGE09B_MEMBER_THREADS
        )
    thread_value = str(threads)
    return {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": str(cache.resolve()),
        "OMP_NUM_THREADS": thread_value,
        "MKL_NUM_THREADS": thread_value,
        "OPENBLAS_NUM_THREADS": thread_value,
        "VECLIB_MAXIMUM_THREADS": thread_value,
        "NUMEXPR_NUM_THREADS": thread_value,
        "THERMOROUTE_FORMAL_THREADS": thread_value,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        _WORKER_CACHE_ENV: str(cache.resolve()),
        _WORKER_NONCE_ENV: nonce,
    }


def _isolate_project_bytecode() -> None:
    """Re-exec a formal worker with a fresh pycache outside the repository."""
    if __name__ != "__main__":
        return
    worker_cache = os.environ.get(_WORKER_CACHE_ENV)
    worker_nonce = os.environ.get(_WORKER_NONCE_ENV)
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    worker_argument = len(sys.argv) > 1 and sys.argv[1] == _WORKER_ARGUMENT
    if worker_cache is not None or worker_nonce is not None or worker_argument:
        if not (worker_cache and worker_nonce and worker_argument):
            raise RuntimeError("Stage 09b formal worker handshake is incomplete")
        expected = Path(worker_cache).resolve()
        flags = (
            int(sys.flags.isolated),
            int(sys.flags.ignore_environment),
            int(sys.flags.no_user_site),
            bool(sys.flags.safe_path),
            int(sys.flags.dont_write_bytecode),
        )
        if (
            flags != (1, 1, 1, True, 0)
            or not bool(sys.flags.hash_randomization)
            or prefix != expected
            or not expected.is_dir()
            or expected == ROOT
            or ROOT in expected.parents
            or (expected / ".controller-nonce").read_text(encoding="utf-8")
            != worker_nonce
            or dict(os.environ)
            != _formal_worker_environment(expected, worker_nonce)
        ):
            raise RuntimeError("Stage 09b formal worker isolation contract failed")
        sys.argv.pop(1)
        return

    with tempfile.TemporaryDirectory(prefix="thermoroute-stage09b-pycache-") as cache:
        cache_path = Path(cache).resolve()
        if any(cache_path.iterdir()):
            raise RuntimeError("Stage 09b controller pycache was not initially empty")
        nonce = secrets.token_hex(32)
        (cache_path / ".controller-nonce").write_text(nonce, encoding="utf-8")
        environment = _formal_worker_environment(cache_path, nonce)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                f"pycache_prefix={cache}",
                str(Path(__file__).resolve()),
                _WORKER_ARGUMENT,
                *sys.argv[1:],
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()

import argparse  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from dataclasses import asdict  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from typing import Any, Protocol, cast  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import datasets as DS  # noqa: E402
from thermoroute import features as F  # noqa: E402
from thermoroute import results as R  # noqa: E402
from thermoroute.evidence import FrozenPanelSpec  # noqa: E402
from thermoroute.checkpoint import (  # noqa: E402
    checkpoint_sidecar_path,
    load_training_checkpoint,
)
from thermoroute.development_controls_gate import (  # noqa: E402
    STAGE09B_COMPLETION_RECEIPT_PATH,
    build_stage09b_completion_receipt,
    publish_stage09b_completion_receipt,
)
from thermoroute.development_controls import (  # noqa: E402
    ARCHITECTURE_BUDGET_FORMAT,
    ArmSpec,
    DEVELOPMENT_DISCLOSURE,
    DEVELOPMENT_SCOPE,
    FEATURE_LADDER,  # noqa: F401 - public script contract used by tests/audits
    FULL_LADDER_ARM_ID,  # noqa: F401 - public scientific-summary contract
    FULL_VARIABLES,
    METRIC_SUMMARY_FORMAT,
    MICRO_RMSE_ROLE,  # noqa: F401 - public scientific-summary contract
    MatrixAudit,
    REPORT_FORMAT,
    SEMANTIC_AUDIT_FORMAT,
    STATION_RMSE_COLUMNS,  # noqa: F401 - public scientific-summary contract
    TRAIN_CONFIG,
    architecture_budget_rows,
    architecture_configuration,
    architecture_template,
    assert_parameter_budgets,
    budget_csv_bytes,
    build_arm_model,
    declared_arms,
    expected_member_registry,
    normalise_prediction_frame,
    paired_comparison_registry,  # noqa: F401 - public paired-effect contract
    parameter_count,
    physics_count,
    prediction_content_digest,
    recompute_metric_summary,
    recompute_paired_effect_summary,
    recompute_station_rmse,
    render_report,
    scientific_summary_document,
    summary_csv_bytes,
    window_registry_digest,
    window_registry_from_windowed,
)
from thermoroute.model_suite import (  # noqa: E402
    ModelSuiteError,
    development_predictor_bridge_binding,
)
from thermoroute.input_closure import resolve_development_input_closure  # noqa: E402
from thermoroute.predictor_bridge import (  # noqa: E402
    PredictorBridgeError,
    validate_development_bridge_manifest_offline,
)
from thermoroute.registry import FORECAST_KEY, targets_match_at_model_precision  # noqa: E402
from thermoroute.repro import (  # noqa: E402
    RunIdentity,
    advisory_file_lock,
    assert_formal_numerical_policy,
    assert_role_thread_cap,
    configure_deterministic_runtime,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sha256_json,
    sidecar_path,
    validate_artifact_sidecar,
)
from thermoroute.stage09b_precompute import (  # noqa: E402
    MAX_PARALLEL_WORKERS,
    Stage09bMember,
    Stage09bPrecomputeError,
    ValidatedStage09bWorkOrder,
    allocate_create_only_artifact_temp,
    execute_stage09b_member_work_orders,
    finalize_stage09b_precompute,
    freeze_stage09b_precompute_plan,
    publish_create_only_artifact_from_temp,
    publish_stage09b_member_receipt,
    recover_create_only_artifact_state,
    stage09b_member_execution_lock,
    validate_stage09b_member_work_order,
    validate_stage09b_model_matrix_gate,
)
from thermoroute.train import (  # noqa: E402
    FitResult,
    export_predictions,
    fit_model,
)


PREDICTION_KIND = "development_control_arm_predictions"
PREDICTION_EXTRA_FORMAT = "thermoroute.development-control-arm.v2"
FINAL_PREDICTION_KIND = "development_controls_combined_predictions"
FINAL_FORMAT = "thermoroute.development-controls.v2"
SUMMARY_KIND = "development_controls_metric_summary"
SEMANTIC_AUDIT_KIND = "development_controls_semantic_audit"


class ControlExperimentError(RuntimeError):
    """The development-control registry, cache, or publication is invalid."""


class FitCallable(Protocol):
    def __call__(
        self,
        model: torch.nn.Module | Callable[[], torch.nn.Module],
        wd: DS.WindowedData,
        thresholds: dict[str, float],
        **kwargs: Any,
    ) -> FitResult: ...


_physics_count = physics_count


def _parent_bindings(
    identity: RunIdentity,
    predictor_bridge: Mapping[str, str],
) -> dict[str, str]:
    return {
        "frozen_panel": identity.panel_sha256,
        "frozen_station_registry": identity.registry_sha256,
        "development_predictor_bridge": str(predictor_bridge["sha256"]),
    }


def canonical_arm_descriptor(arm: ArmSpec) -> dict[str, Any]:
    """Return the JSON-native resolved-config descriptor for one Stage09b arm.

    ``ArmSpec.variables`` and ``ArmSpec.seeds`` are deliberately tuples; the
    formal Stage09b config contract requires Python lists so that run-manifest,
    authorization and worker work-order equality never depends on a JSON
    serialization side effect.  This helper is the single boundary where the
    tuple-to-list normalization happens, before run-identity calculation and
    plan freeze.
    """
    descriptor = asdict(arm)
    return {
        **descriptor,
        "variables": list(arm.variables),
        "seeds": list(arm.seeds),
    }


def _arm_extra_static(
    arm: ArmSpec,
    *,
    seed: int,
    parameters: int,
    n_stations: int,
    eval_batch_size: int,
) -> dict[str, Any]:
    return {
        "format": PREDICTION_EXTRA_FORMAT,
        "arm_id": arm.arm_id,
        "family": arm.family,
        "feature_set": arm.feature_set,
        "variables": list(arm.variables),
        "seed": int(seed),
        "trainable_parameters": int(parameters),
        "architecture": architecture_configuration(
            arm,
            seed=seed,
            n_stations=n_stations,
        ),
        "training_device": "cpu",
        "station_balanced": True,
        "selection_metric": "station_macro",
        "train_config": asdict(TRAIN_CONFIG),
        "context_length": C.CONTEXT_LENGTH,
        "horizons": list(C.HORIZONS),
        "development_only": True,
        "development_evaluation_interval": list(C.SPLIT.test),
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "eval_batch_size": int(eval_batch_size),
    }


def _validate_training_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "best_validation_metric",
        "selected_epoch",
        "checkpoint_final_epoch",
    }:
        raise ControlExperimentError("cached arm training summary schema is invalid")
    best = value["best_validation_metric"]
    selected = value["selected_epoch"]
    final = value["checkpoint_final_epoch"]
    if (
        type(best) not in {int, float}
        or not math.isfinite(float(best))
        or float(best) < 0.0
    ):
        raise ControlExperimentError("cached arm best validation metric is invalid")
    if type(selected) is not int or selected < 0:
        raise ControlExperimentError("cached arm selected epoch is invalid")
    if type(final) is not int or final < selected:
        raise ControlExperimentError("cached arm final checkpoint epoch is invalid")
    return value


def _validate_arm_frame(frame: pd.DataFrame, arm: ArmSpec, seed: int) -> None:
    try:
        normalise_prediction_frame(frame, arm=arm, seed=seed)
    except (TypeError, ValueError) as exc:
        raise ControlExperimentError(
            f"{arm.arm_id}/seed{seed} prediction semantics changed: {exc}"
        ) from exc


def read_arm_prediction(
    path: Path,
    *,
    identity: RunIdentity,
    arm: ArmSpec,
    seed: int,
    parameters: int,
    n_stations: int,
    eval_batch_size: int,
    parents: Mapping[str, str],
) -> pd.DataFrame | None:
    """Load an exact cache hit; reject partial, corrupt, or stale cache state."""
    artifact_exists = path.exists()
    sidecar_exists = sidecar_path(path).exists()
    if not artifact_exists and not sidecar_exists:
        return None
    if not artifact_exists or not sidecar_exists:
        raise ControlExperimentError(f"partial immutable cache state: {path}")
    try:
        metadata = validate_artifact_sidecar(
            path,
            identity=identity,
            schema=R.PREDICTION_SCHEMA_VERSION,
            kind=PREDICTION_KIND,
        )
    except (OSError, ValueError) as exc:
        raise ControlExperimentError(f"stale or corrupt immutable cache: {path}") from exc
    if metadata["parents"] != dict(sorted(parents.items())):
        raise ControlExperimentError(f"cached arm parent lineage changed: {path}")
    extra = metadata["extra"]
    expected_static = _arm_extra_static(
        arm,
        seed=seed,
        parameters=parameters,
        n_stations=n_stations,
        eval_batch_size=eval_batch_size,
    )
    if not isinstance(extra, dict) or set(extra) != {*expected_static, "training_summary"}:
        raise ControlExperimentError(f"cached arm metadata schema changed: {path}")
    if any(extra.get(key) != value for key, value in expected_static.items()):
        raise ControlExperimentError(f"cached arm metadata changed: {path}")
    _validate_training_summary(extra["training_summary"])
    try:
        frame = pd.read_parquet(path)
        _validate_arm_frame(frame, arm, seed)
    except Exception as exc:
        if isinstance(exc, ControlExperimentError):
            raise
        raise ControlExperimentError(f"cached arm prediction is malformed: {path}") from exc
    return frame


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_only_file_from_temp(
    temp_path: Path,
    destination: Path,
    *,
    publication_guard: Callable[[], object],
    _fault_injector: Callable[[str, Path, Path], object] | None = None,
) -> None:
    try:
        publish_create_only_artifact_from_temp(
            temp_path,
            destination,
            publication_guard=publication_guard,
            _fault_injector=_fault_injector,
        )
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            f"create-only artifact publication failed: {destination}"
        ) from exc


def write_arm_prediction(
    frame: pd.DataFrame,
    path: Path,
    *,
    identity: RunIdentity,
    arm: ArmSpec,
    seed: int,
    parameters: int,
    n_stations: int,
    eval_batch_size: int,
    parents: Mapping[str, str],
    training_summary: Mapping[str, Any],
    publication_guard: Callable[[], object],
) -> None:
    """Publish one prediction and sidecar without replacing existing bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        recover_create_only_artifact_state(path)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            f"arm prediction publication state is unsafe: {path}"
        ) from exc
    if path.exists() or sidecar_path(path).exists():
        raise ControlExperimentError(f"refusing to overwrite immutable arm cache: {path}")
    _validate_arm_frame(frame, arm, seed)
    _validate_training_summary(dict(training_summary))
    try:
        file_descriptor, temporary_path = allocate_create_only_artifact_temp(path)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            f"cannot stage immutable arm prediction: {path}"
        ) from exc
    os.close(file_descriptor)
    try:
        frame.loc[:, R.PRED_COLS].to_parquet(temporary_path, index=False)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _create_only_file_from_temp(
            temporary_path,
            path,
            publication_guard=publication_guard,
        )
    finally:
        temporary_path.unlink(missing_ok=True)
    extra = _arm_extra_static(
        arm,
        seed=seed,
        parameters=parameters,
        n_stations=n_stations,
        eval_batch_size=eval_batch_size,
    )
    extra["training_summary"] = dict(training_summary)
    seal_artifact(
        path,
        identity,
        kind=PREDICTION_KIND,
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents=parents,
        extra=extra,
        publication_guard=publication_guard,
    )


def recover_arm_prediction_sidecar(
    path: Path,
    replayed: pd.DataFrame,
    *,
    identity: RunIdentity,
    arm: ArmSpec,
    seed: int,
    parameters: int,
    n_stations: int,
    eval_batch_size: int,
    parents: Mapping[str, str],
    training_summary: Mapping[str, Any],
    publication_guard: Callable[[], object],
) -> None:
    """Recover only artifact-present/sidecar-absent exact crash state."""
    if not path.is_file() or sidecar_path(path).exists():
        raise ControlExperimentError("arm recovery requires only the prediction artifact")
    try:
        observed = pd.read_parquet(path, columns=R.PRED_COLS)
        _assert_exact_prediction_replay(observed, replayed, arm=arm, seed=seed)
    except Exception as exc:
        if isinstance(exc, ControlExperimentError):
            raise
        raise ControlExperimentError("orphan prediction is semantically invalid") from exc
    probe = path.with_name(f".{path.name}.sidecar-replay-probe")
    try:
        recovered_probe = recover_create_only_artifact_state(probe)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            "arm sidecar recovery-probe state is unsafe"
        ) from exc
    if recovered_probe:
        probe.unlink()
        _fsync_parent(probe)
    try:
        descriptor, temporary = allocate_create_only_artifact_temp(probe)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            "cannot stage arm sidecar recovery probe"
        ) from exc
    os.close(descriptor)
    try:
        replayed.loc[:, R.PRED_COLS].to_parquet(temporary, index=False)
        if sha256_file(temporary) != sha256_file(path):
            raise ControlExperimentError("orphan prediction bytes differ from exact replay")
    finally:
        temporary.unlink(missing_ok=True)
    extra = _arm_extra_static(
        arm, seed=seed, parameters=parameters, n_stations=n_stations,
        eval_batch_size=eval_batch_size,
    )
    extra["training_summary"] = dict(training_summary)
    seal_artifact(
        path, identity, kind=PREDICTION_KIND, schema=R.PREDICTION_SCHEMA_VERSION,
        parents=parents, extra=extra, publication_guard=publication_guard,
    )


def _checkpoint_final_epoch(path: Path) -> int:
    metadata_path = path.with_name(path.name + ".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise ControlExperimentError(f"checkpoint or sidecar is absent: {path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlExperimentError(f"checkpoint sidecar is invalid: {metadata_path}") from exc
    epoch = metadata.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise ControlExperimentError(f"checkpoint epoch is invalid: {metadata_path}")
    return epoch


def _member_parents(
    parents: Mapping[str, str], checkpoint_path: Path,
) -> dict[str, str]:
    metadata_path = checkpoint_sidecar_path(checkpoint_path)
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise ControlExperimentError("member checkpoint transaction is incomplete")
    return {
        **dict(parents),
        "training_checkpoint": sha256_file(checkpoint_path),
        "training_checkpoint_sidecar": sha256_file(metadata_path),
    }


def replay_best_model_state_prediction(
    *,
    checkpoint_path: Path,
    arm: ArmSpec,
    seed: int,
    wd: DS.WindowedData,
    thresholds: dict[str, float],
    n_stations: int,
    identity: RunIdentity,
    run_config: Mapping[str, Any],
    eval_batch_size: int,
    recover_missing_checkpoint_sidecar: bool = False,
    publication_guard: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Safely load one checkpoint best state and reproduce exported predictions."""
    parameters = parameter_count(arm, n_stations=n_stations)
    arm_config = {
        **dict(run_config),
        "arm": asdict(arm),
        "seed": int(seed),
        "trainable_parameters": parameters,
    }
    model = build_arm_model(arm, seed=seed, n_stations=n_stations).to("cpu")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=TRAIN_CONFIG.lr,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4,
    )
    try:
        resumed = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id=identity.run_id,
            expected_resolved_config=arm_config,
            map_location="cpu",
            recover_missing_sidecar=recover_missing_checkpoint_sidecar,
            publication_guard=publication_guard,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ControlExperimentError(
            f"{arm.arm_id}/seed{seed} checkpoint replay failed"
        ) from exc
    if resumed.best_model_state is None:  # defensive; v3 validation forbids this
        raise ControlExperimentError("checkpoint has no best_model_state")
    model.load_state_dict(resumed.best_model_state)
    frame = export_predictions(
        model,
        wd,
        thresholds,
        torch.device("cpu"),
        arm.arm_id,
        DEVELOPMENT_SCOPE,
        arm.feature_set,
        seed,
        batch_size=eval_batch_size,
        splits=("val", "calib", "test"),
    )
    frame["model"] = arm.arm_id
    frame["scope"] = DEVELOPMENT_SCOPE
    frame["feature_set"] = arm.feature_set
    frame["seed"] = int(seed)
    _validate_arm_frame(frame, arm, seed)
    if publication_guard is not None:
        publication_guard()
    summary = {
        "best_validation_metric": float(resumed.best_metric),
        "selected_epoch": int(resumed.best_epoch),
        "checkpoint_final_epoch": int(resumed.epoch),
    }
    _validate_training_summary(summary)
    return frame, summary


def _assert_exact_prediction_replay(
    observed: pd.DataFrame, replayed: pd.DataFrame, *, arm: ArmSpec, seed: int,
) -> None:
    left = normalise_prediction_frame(observed, arm=arm, seed=seed)
    right = normalise_prediction_frame(replayed, arm=arm, seed=seed)
    if prediction_content_digest(left) != prediction_content_digest(right):
        raise ControlExperimentError(
            f"{arm.arm_id}/seed{seed} prediction differs from checkpoint best state"
        )


def train_arm_group(
    arms: Sequence[ArmSpec],
    *,
    wd: DS.WindowedData,
    thresholds: dict[str, float],
    n_stations: int,
    identity: RunIdentity,
    run_config: Mapping[str, Any],
    run_dir: Path,
    parents: Mapping[str, str],
    eval_batch_size: int,
    verbose: bool,
    publication_guard: Callable[[], object],
    only_member: tuple[str, int] | None = None,
    fit_function: FitCallable = fit_model,  # type: ignore[assignment]
) -> list[Path]:
    """Train/cache all arms sharing one window tensor without retaining frames."""
    paths: list[Path] = []
    for arm in arms:
        parameters = parameter_count(arm, n_stations=n_stations)
        for seed in arm.seeds:
            if only_member is not None and (arm.arm_id, int(seed)) != only_member:
                continue
            prediction_path = run_dir / "arm_predictions" / arm.arm_id / f"seed{seed}.parquet"
            checkpoint_path = run_dir / "checkpoints" / arm.arm_id / f"seed{seed}.pt"
            arm_config = {
                **dict(run_config),
                "arm": asdict(arm),
                "seed": int(seed),
                "trainable_parameters": parameters,
            }

            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                recover_create_only_artifact_state(prediction_path)
            except Stage09bPrecomputeError as exc:
                raise ControlExperimentError(
                    f"prediction create-only state is unsafe: {prediction_path}"
                ) from exc

            def factory(arm: ArmSpec = arm, seed: int = seed) -> torch.nn.Module:
                return build_arm_model(arm, seed=seed, n_stations=n_stations)

            artifact_exists = prediction_path.exists()
            sidecar_exists = sidecar_path(prediction_path).exists()
            if sidecar_exists and not artifact_exists:
                raise ControlExperimentError(
                    f"prediction sidecar exists without artifact: {prediction_path}"
                )
            if artifact_exists and not sidecar_exists:
                replayed, training_summary = replay_best_model_state_prediction(
                    checkpoint_path=checkpoint_path, arm=arm, seed=seed, wd=wd,
                    thresholds=thresholds, n_stations=n_stations, identity=identity,
                    run_config=run_config, eval_batch_size=eval_batch_size,
                    recover_missing_checkpoint_sidecar=True,
                    publication_guard=publication_guard,
                )
                member_lineage = _member_parents(parents, checkpoint_path)
                recover_arm_prediction_sidecar(
                    prediction_path, replayed, identity=identity, arm=arm, seed=seed,
                    parameters=parameters, n_stations=n_stations,
                    eval_batch_size=eval_batch_size, parents=member_lineage,
                    training_summary=training_summary,
                    publication_guard=publication_guard,
                )
                paths.append(prediction_path)
                continue
            if artifact_exists and sidecar_exists:
                member_lineage = _member_parents(parents, checkpoint_path)
                cached = read_arm_prediction(
                    prediction_path, identity=identity, arm=arm, seed=seed,
                    parameters=parameters, n_stations=n_stations,
                    eval_batch_size=eval_batch_size, parents=member_lineage,
                )
                assert cached is not None
                publication_guard()
                paths.append(prediction_path)
                continue

            result = fit_function(
                factory,
                wd,
                thresholds,
                cfg=TRAIN_CONFIG,
                seed=seed,
                device="cpu",
                model_name=arm.arm_id,
                scope=DEVELOPMENT_SCOPE,
                feature_set=arm.feature_set,
                verbose=verbose,
                station_balanced=True,
                selection_metric="station_macro",
                eval_batch_size=eval_batch_size,
                checkpoint_path=checkpoint_path,
                run_id=identity.run_id,
                resolved_config=arm_config,
                resume=True,
                checkpoint_every=1,
                export_splits=("val", "calib", "test"),
                artifact_publication_guard=publication_guard,
            )
            result.pred["model"] = arm.arm_id
            result.pred["scope"] = DEVELOPMENT_SCOPE
            result.pred["feature_set"] = arm.feature_set
            result.pred["seed"] = int(seed)
            replayed, training_summary = replay_best_model_state_prediction(
                checkpoint_path=checkpoint_path, arm=arm, seed=seed, wd=wd,
                thresholds=thresholds, n_stations=n_stations, identity=identity,
                run_config=run_config, eval_batch_size=eval_batch_size,
                recover_missing_checkpoint_sidecar=True,
                publication_guard=publication_guard,
            )
            _assert_exact_prediction_replay(
                result.pred, replayed, arm=arm, seed=seed,
            )
            if (
                float(result.best_val) != training_summary["best_validation_metric"]
                or int(result.epochs) != training_summary["selected_epoch"]
                or _checkpoint_final_epoch(checkpoint_path)
                != training_summary["checkpoint_final_epoch"]
            ):
                raise ControlExperimentError("fit result differs from final checkpoint")
            member_lineage = _member_parents(parents, checkpoint_path)
            write_arm_prediction(
                replayed,
                prediction_path,
                identity=identity,
                arm=arm,
                seed=seed,
                parameters=parameters,
                n_stations=n_stations,
                eval_batch_size=eval_batch_size,
                parents=member_lineage,
                training_summary=training_summary,
                publication_guard=publication_guard,
            )
            paths.append(prediction_path)
    if only_member is not None and len(paths) != 1:
        raise ControlExperimentError(
            f"member filter did not resolve exactly once: {only_member}"
        )
    return paths


def _normalised_key_truth(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["split", *FORECAST_KEY, "y_true"]
    out = frame.loc[:, columns].copy()
    out["site_id"] = out["site_id"].astype(str)
    out["split"] = out["split"].astype(str)
    out["horizon"] = pd.to_numeric(out["horizon"], errors="raise").astype("int64")
    out["issue_date"] = pd.to_datetime(out["issue_date"])
    out["target_date"] = pd.to_datetime(out["target_date"])
    key = ["split", *FORECAST_KEY]
    if out.duplicated(key).any():
        raise ControlExperimentError("arm prediction contains a duplicate forecast key")
    return out.sort_values(key, kind="mergesort").reset_index(drop=True)


def _member_semantic_evidence(
    frame: pd.DataFrame,
    *,
    arm: ArmSpec,
    seed: int,
    canonical_registry: pd.DataFrame,
) -> dict[str, Any]:
    """Return independently reproducible receipt evidence for one prediction."""
    try:
        normalised = normalise_prediction_frame(
            frame,
            arm=arm,
            seed=seed,
            canonical_registry=canonical_registry,
        )
    except (TypeError, ValueError) as exc:
        raise ControlExperimentError(
            f"{arm.arm_id}/seed{seed} semantic evidence is invalid"
        ) from exc
    content = prediction_content_digest(normalised)
    return {
        "prediction_content_sha256": content,
        "forecast_key_truth_sha256": window_registry_digest(
            normalised[["split", *FORECAST_KEY, "y_true"]]
        ),
        "prediction_rows": len(normalised),
        "checkpoint_exact_replay_sha256": content,
    }


def validate_complete_prediction_matrix(
    frames: Mapping[tuple[str, int], pd.DataFrame],
    arms: Sequence[ArmSpec],
    *,
    allowed_sites: set[str] | None = None,
) -> MatrixAudit:
    """Require the exact matrix and identical forecast keys/truth for every member."""
    expected = expected_member_registry(arms)
    if set(frames) != set(expected) or len(frames) != len(expected):
        missing = sorted(set(expected) - set(frames))
        extra = sorted(set(frames) - set(expected))
        raise ControlExperimentError(
            f"development-control matrix is incomplete: missing={missing}, extra={extra}"
        )
    arm_by_id = {arm.arm_id: arm for arm in arms}
    reference: pd.DataFrame | None = None
    reference_member = ""
    total_rows = 0
    for member in expected:
        arm_id, seed = member
        frame = frames[member]
        try:
            normalised = normalise_prediction_frame(
                frame, arm=arm_by_id[arm_id], seed=seed,
                allowed_sites=allowed_sites,
            )
        except (TypeError, ValueError) as exc:
            raise ControlExperimentError(str(exc)) from exc
        current = _normalised_key_truth(normalised)
        total_rows += len(frame)
        if reference is None:
            reference = current
            reference_member = f"{arm_id}/seed{seed}"
            continue
        key_columns = ["split", *FORECAST_KEY]
        if not current[key_columns].equals(reference[key_columns]):
            raise ControlExperimentError(
                f"{arm_id}/seed{seed} does not share the exact forecast-key registry"
            )
        if not targets_match_at_model_precision(current["y_true"], reference["y_true"]):
            raise ControlExperimentError(
                f"{arm_id}/seed{seed} disagrees on development truth values"
            )
    assert reference is not None
    return MatrixAudit(
        expected_members=len(expected),
        prediction_rows=total_rows,
        common_forecast_keys=len(reference),
        splits=tuple(sorted(reference["split"].unique())),
        reference_member=reference_member,
    )


def validate_prediction_paths(
    paths: Sequence[Path],
    arms: Sequence[ArmSpec],
    *,
    identity: RunIdentity,
    parents: Mapping[str, str],
    n_stations: int,
    eval_batch_size: int,
    allowed_sites: set[str],
    canonical_registry: pd.DataFrame | None = None,
) -> tuple[MatrixAudit, dict[tuple[str, int], Path], list[dict[str, Any]]]:
    """Validate large member files sequentially while retaining one key registry."""
    expected = expected_member_registry(arms)
    if len(paths) != len(expected) or len({path.resolve() for path in paths}) != len(expected):
        raise ControlExperimentError("prediction path registry has missing, extra, or duplicate paths")
    expected_paths = {
        (arm.arm_id, seed): next(
            (
                path
                for path in paths
                if path.parent.name == arm.arm_id and path.stem == f"seed{seed}"
            ),
            None,
        )
        for arm in arms
        for seed in arm.seeds
    }
    if set(expected_paths) != set(expected) or any(path is None for path in expected_paths.values()):
        raise ControlExperimentError("prediction path registry is incomplete")
    arm_by_id = {arm.arm_id: arm for arm in arms}
    reference: pd.DataFrame | None = None
    reference_member = ""
    total_rows = 0
    summaries: list[dict[str, Any]] = []
    resolved_paths: dict[tuple[str, int], Path] = {}
    for arm_id, seed in expected:
        path = expected_paths[(arm_id, seed)]
        assert path is not None
        arm = arm_by_id[arm_id]
        checkpoint_path = (
            path.parents[2] / "checkpoints" / arm_id / f"seed{seed}.pt"
        )
        member_lineage = _member_parents(parents, checkpoint_path)
        frame = read_arm_prediction(
            path,
            identity=identity,
            arm=arm,
            seed=seed,
            parameters=parameter_count(arm, n_stations=n_stations),
            n_stations=n_stations,
            eval_batch_size=eval_batch_size,
            parents=member_lineage,
        )
        assert frame is not None
        try:
            normalised = normalise_prediction_frame(
                frame, arm=arm, seed=seed, allowed_sites=allowed_sites,
                canonical_registry=canonical_registry,
            )
        except (TypeError, ValueError) as exc:
            raise ControlExperimentError(
                f"{arm_id}/seed{seed} prediction contract changed: {exc}"
            ) from exc
        current = _normalised_key_truth(normalised)
        key_columns = ["split", *FORECAST_KEY]
        if reference is None:
            reference = current
            reference_member = f"{arm_id}/seed{seed}"
        elif (
            not current[key_columns].equals(reference[key_columns])
            or not targets_match_at_model_precision(current["y_true"], reference["y_true"])
        ):
            raise ControlExperimentError(
                f"{arm_id}/seed{seed} does not share exact forecast keys and truth"
            )
        total_rows += len(frame)
        summaries.extend(
            recompute_metric_summary({(arm_id, seed): normalised}).to_dict(
                orient="records"
            )
        )
        resolved_paths[(arm_id, seed)] = path
        del frame, normalised, current
    assert reference is not None
    return (
        MatrixAudit(
            expected_members=len(expected),
            prediction_rows=total_rows,
            common_forecast_keys=len(reference),
            splits=tuple(sorted(reference["split"].unique())),
            reference_member=reference_member,
        ),
        resolved_paths,
        summaries,
    )


def _create_only_bytes(
    payload: bytes,
    destination: Path,
    *,
    publication_guard: Callable[[], object],
    _fault_injector: Callable[[str, Path, Path], object] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        recover_create_only_artifact_state(destination)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            f"final byte-artifact publication state is unsafe: {destination}"
        ) from exc
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ControlExperimentError(
                f"refusing to overwrite immutable artifact: {destination}"
            )
        publication_guard()
        return
    try:
        descriptor, temporary_path = allocate_create_only_artifact_temp(destination)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            f"cannot stage immutable byte artifact: {destination}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _create_only_file_from_temp(
            temporary_path,
            destination,
            publication_guard=publication_guard,
            _fault_injector=_fault_injector,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _stream_combined_predictions(
    members: Mapping[tuple[str, int], Path],
    destination: Path,
    *,
    publication_guard: Callable[[], object],
    _fault_injector: Callable[[str, Path, Path], object] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        recover_create_only_artifact_state(destination)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            f"combined prediction publication state is unsafe: {destination}"
        ) from exc
    if destination.exists():
        raise ControlExperimentError(f"refusing to overwrite combined artifact: {destination}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        descriptor, temporary_path = allocate_create_only_artifact_temp(destination)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            f"cannot stage combined predictions: {destination}"
        ) from exc
    os.close(descriptor)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    try:
        for member in members:
            parquet = pq.ParquetFile(members[member])
            if parquet.schema_arrow.names != R.PRED_COLS:
                raise ControlExperimentError("member schema changed during streaming publish")
            for batch in parquet.iter_batches(columns=R.PRED_COLS, batch_size=65_536):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
                elif table.schema != schema:
                    raise ControlExperimentError("member Arrow schema changed")
                writer.write_table(table)
        if writer is None:
            raise ControlExperimentError("cannot publish an empty prediction matrix")
        writer.close()
        writer = None
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _create_only_file_from_temp(
            temporary_path,
            destination,
            publication_guard=publication_guard,
            _fault_injector=_fault_injector,
        )
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)


def _validate_combined_exact(
    path: Path, members: Mapping[tuple[str, int], Path],
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    combined = pq.ParquetFile(path)
    if combined.schema_arrow.names != R.PRED_COLS:
        raise ControlExperimentError("combined prediction schema changed")
    iterator = iter(combined.iter_batches(columns=R.PRED_COLS, batch_size=65_536))
    current: Any | None = None
    offset = 0
    total = 0

    def take(rows: int) -> Any:
        nonlocal current, offset
        pieces: list[Any] = []
        remaining = rows
        while remaining:
            if current is None or offset == current.num_rows:
                try:
                    current = next(iterator)
                except StopIteration as exc:
                    raise ControlExperimentError("combined predictions end early") from exc
                offset = 0
            count = min(remaining, current.num_rows - offset)
            pieces.append(current.slice(offset, count))
            offset += count
            remaining -= count
        return pa.Table.from_batches(pieces).combine_chunks()

    for member in expected_member_registry():
        source = pq.ParquetFile(members[member])
        if source.schema_arrow.names != R.PRED_COLS:
            raise ControlExperimentError("member prediction schema changed")
        for batch in source.iter_batches(columns=R.PRED_COLS, batch_size=65_536):
            expected = pa.Table.from_batches([batch]).combine_chunks()
            if not take(batch.num_rows).equals(expected, check_metadata=False):
                raise ControlExperimentError("combined predictions differ from members")
            total += batch.num_rows
    if current is not None and offset < current.num_rows:
        raise ControlExperimentError("combined predictions contain extra rows")
    try:
        next(iterator)
    except StopIteration:
        return total
    raise ControlExperimentError("combined predictions contain extra rows")


def _final_extra(audit: MatrixAudit, *, artifact_role: str) -> dict[str, Any]:
    return {
        "format": FINAL_FORMAT,
        "artifact_role": artifact_role,
        "expected_members": audit.expected_members,
        "prediction_rows": audit.prediction_rows,
        "common_forecast_keys_per_member": audit.common_forecast_keys,
        "splits": list(audit.splits),
        "reference_member": audit.reference_member,
        "development_only": True,
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
    }


def _semantic_audit_document(
    *,
    identity: RunIdentity,
    audit: MatrixAudit,
    train_examples: int,
    canonical_registry_sha256: str,
    canonical_train_registry_sha256: str,
    member_paths: Mapping[tuple[str, int], Path],
    member_digests: Mapping[tuple[str, int], str],
    paired_effects: pd.DataFrame,
    final_paths: Mapping[str, Path],
) -> dict[str, Any]:
    def descriptor(path: Path) -> dict[str, Any]:
        return {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    document: dict[str, Any] = {
        "format": SEMANTIC_AUDIT_FORMAT,
        "status": "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "run_id": identity.run_id,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
        "post_2020_outcomes_requested_or_read": False,
        "matrix_audit": asdict(audit),
        "canonical_window_registry": {
            "sha256": canonical_registry_sha256,
            "common_forecast_keys": audit.common_forecast_keys,
            "train_examples_per_epoch": train_examples,
            "train_registry_sha256": canonical_train_registry_sha256,
        },
        "scientific_summary": scientific_summary_document(paired_effects),
        "members": [
            {
                "arm_id": arm_id,
                "seed": seed,
                "checkpoint": descriptor(
                    member_paths[(arm_id, seed)].parents[2]
                    / "checkpoints" / arm_id / f"seed{seed}.pt"
                ),
                "checkpoint_sidecar": descriptor(
                    checkpoint_sidecar_path(
                        member_paths[(arm_id, seed)].parents[2]
                        / "checkpoints" / arm_id / f"seed{seed}.pt"
                    )
                ),
                "prediction": descriptor(member_paths[(arm_id, seed)]),
                "prediction_sidecar": descriptor(
                    sidecar_path(member_paths[(arm_id, seed)])
                ),
                "normalised_prediction_sha256": member_digests[(arm_id, seed)],
                "best_model_state_prediction_replay_verified": True,
            }
            for arm_id, seed in expected_member_registry()
        ],
        "derived_artifacts": {
            label: {
                "artifact": descriptor(path),
                "sidecar": descriptor(sidecar_path(path)),
            }
            for label, path in sorted(final_paths.items())
        },
    }
    document["semantic_audit_self_sha256"] = sha256_json(document)
    return document


def publish_final_artifacts(
    *,
    run_dir: Path,
    identity: RunIdentity,
    arms: Sequence[ArmSpec],
    member_paths: Mapping[tuple[str, int], Path],
    member_parents: Mapping[str, str],
    audit: MatrixAudit,
    budget: pd.DataFrame,
    summaries: Sequence[Mapping[str, Any]],
    train_examples: int,
    canonical_registry_sha256: str,
    canonical_train_registry_sha256: str,
    publication_guard: Callable[[], object],
) -> tuple[Path, Path, Path, Path, Path]:
    """Publish deterministic prediction-derived closure artifacts."""
    expected = set(expected_member_registry(arms))
    if (
        set(member_paths) != expected
        or audit.expected_members != len(expected)
        or audit.prediction_rows <= 0
        or audit.common_forecast_keys <= 0
        or audit.splits != ("calib", "test", "val")
        or set(budget["arm_id"].astype(str)) != {arm.arm_id for arm in arms}
        or len(budget) != len(arms)
    ):
        raise ControlExperimentError(
            "final publication requires the exact audited member, summary, and budget matrix"
        )
    prediction_path = run_dir / "development_controls_predictions.parquet"
    budget_path = run_dir / "development_controls_architecture_budget.csv"
    summary_path = run_dir / "development_controls_metric_summary.csv"
    report_path = run_dir / "development_controls_report.md"
    semantic_audit_path = run_dir / "development_controls_semantic_audit.json"
    final_parents = {
        **dict(member_parents),
        **{
            f"arm::{arm_id}::seed{seed}::prediction": sha256_file(path)
            for (arm_id, seed), path in member_paths.items()
        },
        **{
            f"arm::{arm_id}::seed{seed}::checkpoint": sha256_file(
                run_dir / "checkpoints" / arm_id / f"seed{seed}.pt"
            )
            for arm_id, seed in expected_member_registry(arms)
        },
        **{
            f"arm::{arm_id}::seed{seed}::checkpoint_sidecar": sha256_file(
                checkpoint_sidecar_path(
                    run_dir / "checkpoints" / arm_id / f"seed{seed}.pt"
                )
            )
            for arm_id, seed in expected_member_registry(arms)
        },
    }
    arm_by_id = {arm.arm_id: arm for arm in arms}
    member_digests: dict[tuple[str, int], str] = {}
    recomputed_parts: list[pd.DataFrame] = []
    station_parts: list[pd.DataFrame] = []
    common_registry: pd.DataFrame | None = None
    for member in expected_member_registry(arms):
        frame = pd.read_parquet(member_paths[member], columns=R.PRED_COLS)
        try:
            normalised = normalise_prediction_frame(
                frame, arm=arm_by_id[member[0]], seed=member[1]
            )
        except (TypeError, ValueError) as exc:
            raise ControlExperimentError(f"member semantic validation failed: {exc}") from exc
        member_digests[member] = prediction_content_digest(normalised)
        recomputed_parts.append(recompute_metric_summary({member: normalised}))
        station_parts.append(recompute_station_rmse({member: normalised}))
        current_registry = normalised[["split", *FORECAST_KEY, "y_true"]]
        if common_registry is None:
            common_registry = current_registry.copy()
        elif (
            not current_registry[["split", *FORECAST_KEY]].equals(
                common_registry[["split", *FORECAST_KEY]]
            )
            or not targets_match_at_model_precision(
                current_registry["y_true"], common_registry["y_true"]
            )
        ):
            raise ControlExperimentError(
                "final publication member forecast-key/truth registry changed"
            )
        del frame, normalised
    assert common_registry is not None
    recomputed_summary = pd.concat(recomputed_parts, ignore_index=True).sort_values(
        ["arm_id", "seed", "split", "horizon"], kind="mergesort"
    ).reset_index(drop=True)
    paired_effects = recompute_paired_effect_summary(
        pd.concat(station_parts, ignore_index=True),
        exact_common_forecast_keys_verified=True,
    )
    declared_summary = pd.DataFrame.from_records(summaries)
    if list(declared_summary.columns) != list(recomputed_summary.columns):
        declared_summary = declared_summary.reindex(columns=recomputed_summary.columns)
    declared_summary = declared_summary.sort_values(
        ["arm_id", "seed", "split", "horizon"], kind="mergesort"
    ).reset_index(drop=True)
    try:
        if summary_csv_bytes(declared_summary) != summary_csv_bytes(recomputed_summary):
            raise ControlExperimentError("metric summary is not prediction-derived")
    except (TypeError, ValueError) as exc:
        raise ControlExperimentError("metric summary is malformed") from exc
    report_bytes = render_report(
        run_id=identity.run_id, audit=audit, budget=budget,
        summary=recomputed_summary, paired_effects=paired_effects,
    ).encode("utf-8")
    base_specs = (
        (
            prediction_path, FINAL_PREDICTION_KIND,
            R.PREDICTION_SCHEMA_VERSION, "combined_predictions",
        ),
        (
            budget_path, "development_controls_budget",
            ARCHITECTURE_BUDGET_FORMAT, "architecture_budget",
        ),
        (summary_path, SUMMARY_KIND, METRIC_SUMMARY_FORMAT, "metric_summary"),
        (report_path, "development_controls_report", REPORT_FORMAT, "report"),
    )
    exact_bytes = {
        budget_path: budget_csv_bytes(budget),
        summary_path: summary_csv_bytes(recomputed_summary),
        report_path: report_bytes,
    }
    for artifact, metadata_path in (
        (prediction_path, sidecar_path(prediction_path)),
        *((path, sidecar_path(path)) for path in exact_bytes),
        (semantic_audit_path, sidecar_path(semantic_audit_path)),
    ):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        try:
            recover_create_only_artifact_state(artifact)
        except Stage09bPrecomputeError as exc:
            raise ControlExperimentError(
                f"final create-only artifact state is unsafe: {artifact}"
            ) from exc
        if metadata_path.exists() and not artifact.exists():
            raise ControlExperimentError(f"sidecar exists without artifact: {artifact}")

    if not prediction_path.exists():
        _stream_combined_predictions(
            member_paths,
            prediction_path,
            publication_guard=publication_guard,
        )
    elif not sidecar_path(prediction_path).exists():
        # For the one recoverable crash window, require byte identity with a
        # freshly streamed reconstruction before blessing the orphan artifact.
        probe = prediction_path.with_name(f".{prediction_path.name}.recovery-probe")
        try:
            try:
                recovered_probe = recover_create_only_artifact_state(probe)
            except Stage09bPrecomputeError as exc:
                raise ControlExperimentError(
                    "combined recovery-probe publication state is unsafe"
                ) from exc
            if recovered_probe:
                # A probe is scratch state, never evidence.  Once its own exact
                # internal hard-link crash window has been recovered, discard it
                # and reconstruct again from the current 45 bound members.
                probe.unlink()
                _fsync_parent(probe)
            _stream_combined_predictions(
                member_paths,
                probe,
                publication_guard=publication_guard,
            )
            if sha256_file(probe) != sha256_file(prediction_path):
                raise ControlExperimentError("orphan combined artifact bytes changed")
        finally:
            probe.unlink(missing_ok=True)
    if _validate_combined_exact(prediction_path, member_paths) != audit.prediction_rows:
        raise ControlExperimentError("combined prediction row count changed during publication")
    publication_guard()
    for path, payload in exact_bytes.items():
        if path.exists():
            if path.read_bytes() != payload:
                raise ControlExperimentError(f"existing final artifact changed: {path}")
        else:
            _create_only_bytes(
                payload,
                path,
                publication_guard=publication_guard,
            )
    for path, kind, schema, role in base_specs:
        if not sidecar_path(path).exists():
            seal_artifact(
                path, identity, kind=kind, schema=schema, parents=final_parents,
                extra=_final_extra(audit, artifact_role=role),
                publication_guard=publication_guard,
            )
        metadata = validate_artifact_sidecar(
            path, identity=identity, schema=schema, kind=kind,
        )
        if (
            metadata["parents"] != dict(sorted(final_parents.items()))
            or metadata["extra"] != _final_extra(audit, artifact_role=role)
        ):
            raise ControlExperimentError(f"final artifact metadata changed: {path}")
    semantic_document = _semantic_audit_document(
        identity=identity, audit=audit, train_examples=train_examples,
        canonical_registry_sha256=canonical_registry_sha256,
        canonical_train_registry_sha256=canonical_train_registry_sha256,
        member_paths=member_paths, member_digests=member_digests,
        paired_effects=paired_effects,
        final_paths={
            "architecture_budget": budget_path,
            "combined_predictions": prediction_path,
            "metric_summary": summary_path,
            "report": report_path,
        },
    )
    semantic_bytes = (
        json.dumps(semantic_document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if semantic_audit_path.exists():
        if semantic_audit_path.read_bytes() != semantic_bytes:
            raise ControlExperimentError("semantic audit changed")
    else:
        _create_only_bytes(
            semantic_bytes,
            semantic_audit_path,
            publication_guard=publication_guard,
        )
    if not sidecar_path(semantic_audit_path).exists():
        seal_artifact(
            semantic_audit_path, identity, kind=SEMANTIC_AUDIT_KIND,
            schema=SEMANTIC_AUDIT_FORMAT, parents=final_parents,
            extra=_final_extra(audit, artifact_role="semantic_audit"),
            publication_guard=publication_guard,
        )
    metadata = validate_artifact_sidecar(
        semantic_audit_path, identity=identity, schema=SEMANTIC_AUDIT_FORMAT,
        kind=SEMANTIC_AUDIT_KIND,
    )
    if (
        metadata["parents"] != dict(sorted(final_parents.items()))
        or metadata["extra"] != _final_extra(audit, artifact_role="semantic_audit")
    ):
        raise ControlExperimentError("semantic audit metadata changed")
    publication_guard()
    return prediction_path, budget_path, summary_path, report_path, semantic_audit_path


def _group_arms_by_variables(arms: Sequence[ArmSpec]) -> list[tuple[tuple[str, ...], list[ArmSpec]]]:
    groups: dict[tuple[str, ...], list[ArmSpec]] = {}
    for arm in arms:
        groups.setdefault(arm.variables, []).append(arm)
    return [(variables, groups[variables]) for variables in groups]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete CPU-only, development-only neural-control and "
            "ThermoRoute feature-ladder matrix."
        )
    )
    parser.add_argument(
        "--panel",
        default=str(ROOT / "data_usgs" / "panel_usgs_120v2.parquet"),
        help="must resolve to the canonical frozen 2006-2020 panel",
    )
    parser.add_argument(
        "--registry",
        default=str(ROOT / "data_usgs" / "station_registry_v1.csv"),
        help="must resolve to the canonical frozen 120-site registry",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4096,
        help="CPU validation/export batch size; part of the scientific run identity",
    )
    parser.add_argument(
        "--precompute-workers",
        type=int,
        default=2,
        help=(
            "execution-only member-process concurrency (excluded from the "
            "scientific configuration and run identity)"
        ),
    )
    parser.add_argument(
        _MEMBER_WORK_ORDER_OPTION,
        dest="member_work_order",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--verbose", action="store_true", help="print epoch diagnostics")
    args = parser.parse_args(argv)
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    if not 1 <= args.precompute_workers <= MAX_PARALLEL_WORKERS:
        parser.error(
            f"--precompute-workers must be between 1 and {MAX_PARALLEL_WORKERS}"
        )
    return args


def _launch_stage09b_member_process(
    work_order: Path,
    *,
    panel: str,
    registry: str,
    eval_batch_size: int,
    verbose: bool,
) -> None:
    """Launch one isolated single-native-thread member process."""
    with tempfile.TemporaryDirectory(prefix="thermoroute-stage09b-member-pycache-") as cache:
        cache_path = Path(cache).resolve()
        if any(cache_path.iterdir()):
            raise ControlExperimentError("member pycache was not initially empty")
        nonce = secrets.token_hex(32)
        (cache_path / ".controller-nonce").write_text(nonce, encoding="utf-8")
        command = [
            sys.executable,
            "-I",
            "-X",
            f"pycache_prefix={cache_path}",
            str(Path(__file__).resolve()),
            _WORKER_ARGUMENT,
            "--panel",
            panel,
            "--registry",
            registry,
            "--eval-batch-size",
            str(eval_batch_size),
            _MEMBER_WORK_ORDER_OPTION,
            str(work_order),
        ]
        if verbose:
            command.append("--verbose")
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_formal_worker_environment(
                cache_path, nonce, threads=STAGE09B_MEMBER_THREADS
            ),
            check=False,
        )
    if result.returncode:
        raise ControlExperimentError(
            f"Stage-09b member process failed ({result.returncode}): {work_order}"
        )


def _execute_authorized_member(
    *,
    validated: ValidatedStage09bWorkOrder,
    arms: Sequence[ArmSpec],
    panel_imputed: pd.DataFrame,
    masks: D.SplitMasks,
    climatology: F.HarmonicClimatology,
    thresholds: dict[str, float],
    stations: Sequence[str],
    identity: RunIdentity,
    run_config: Mapping[str, Any],
    run_dir: Path,
    parents: Mapping[str, str],
    eval_batch_size: int,
    verbose: bool,
    publication_guard: Callable[[], object],
) -> int:
    """Train/replay/commit exactly one locked Stage-09b member."""
    with stage09b_member_execution_lock(validated):
        member = validated.member
        arm = next((candidate for candidate in arms if candidate.arm_id == member.arm_id), None)
        if arm is None or member.seed not in arm.seeds:
            raise ControlExperimentError("authorized Stage-09b member is undeclared")
        wd = DS.build_windows(
            panel_imputed,
            masks,
            climatology,
            context=C.CONTEXT_LENGTH,
            horizons=C.HORIZONS,
            variables=arm.variables,
            require_observed_target=True,
        )
        paths = train_arm_group(
            (arm,),
            wd=wd,
            thresholds=thresholds,
            n_stations=len(stations),
            identity=identity,
            run_config=run_config,
            run_dir=run_dir,
            parents=parents,
            eval_batch_size=eval_batch_size,
            verbose=verbose,
            # Checkpoint/prediction/bundle files remain intermediate until the
            # member's exact replay and create-only receipt complete. Enforce
            # the live native policy at each internal publication boundary;
            # the enclosing member lock replays the stronger source/input gate
            # before and after the complete transaction.
            publication_guard=assert_formal_numerical_policy,
            only_member=(member.arm_id, member.seed),
        )
        prediction_path = paths[0]
        checkpoint_path = (
            run_dir / "checkpoints" / member.arm_id / f"seed{member.seed}.pt"
        )
        replayed, _training_summary = replay_best_model_state_prediction(
            checkpoint_path=checkpoint_path,
            arm=arm,
            seed=member.seed,
            wd=wd,
            thresholds=thresholds,
            n_stations=len(stations),
            identity=identity,
            run_config=run_config,
            eval_batch_size=eval_batch_size,
            recover_missing_checkpoint_sidecar=True,
            publication_guard=publication_guard,
        )
        observed = pd.read_parquet(prediction_path, columns=R.PRED_COLS)
        _assert_exact_prediction_replay(
            observed, replayed, arm=arm, seed=member.seed
        )
        member_registry = window_registry_from_windowed(wd, stations)
        semantic = _member_semantic_evidence(
            replayed,
            arm=arm,
            seed=member.seed,
            canonical_registry=member_registry,
        )
        validated.assert_unchanged()
        try:
            member_receipt = publish_stage09b_member_receipt(
                validated,
                semantic_evidence=semantic,
                exact_replay_verified=True,
                publication_guard=publication_guard,
            )
        except Stage09bPrecomputeError as exc:
            raise ControlExperimentError("Stage-09b member receipt failed closed") from exc
        print(json.dumps({
            "status": "COMPLETE_STAGE09B_MEMBER",
            "run_id": identity.run_id,
            "member_id": member.member_id,
            "member_receipt": str(member_receipt),
        }, sort_keys=True))
        return 0


def _run(args: argparse.Namespace) -> int:
    """Execute one already-parsed parent or authorized-member invocation."""
    configure_deterministic_runtime()
    assert_role_thread_cap(ROOT, "stage09b")
    runtime_policy = assert_formal_numerical_policy(require_hash_randomization=True)
    if torch.device("cpu").type != "cpu":  # pragma: no cover - defensive declaration
        raise ControlExperimentError("development controls require CPU execution")

    frozen_spec_path = ROOT / "data_usgs" / "frozen_panel_v1.json"
    frozen = FrozenPanelSpec.load(frozen_spec_path)
    evidence = frozen.verify()
    panel_path = Path(args.panel).resolve()
    registry_path = Path(args.registry).resolve()
    if panel_path != frozen.panel_path or registry_path != frozen.registry_path:
        raise ControlExperimentError(
            "Stage 09b accepts only the canonical frozen panel and station registry"
        )
    if str(frozen.document["panel"]["date_end"]) != "2020-12-31":
        raise ControlExperimentError("development controls must not read post-2020 outcomes")
    try:
        predictor_bridge = development_predictor_bridge_binding(
            ROOT,
            panel_sha256=sha256_file(panel_path),
            registry_sha256=sha256_file(registry_path),
        )
    except ModelSuiteError as exc:
        raise ControlExperimentError(
            "development controls require PASS_EXACT_PRODUCT_BRIDGE"
        ) from exc
    try:
        validate_development_bridge_manifest_offline(
            repo_root=ROOT,
            manifest_path=ROOT / predictor_bridge["path"],
            expected_sites=120,
        )
    except PredictorBridgeError as exc:
        raise ControlExperimentError(
            "development predictor bridge raw replay failed before training"
        ) from exc

    arms = declared_arms()
    development_input_closure = resolve_development_input_closure(ROOT)
    development_input_closure.assert_unchanged()
    try:
        model_matrix_gate = validate_stage09b_model_matrix_gate(ROOT)
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError(
            "Stage 09b requires the frozen prelabel 9-arm x 5-seed matrix"
        ) from exc

    def assert_stage09b_publication_inputs() -> None:
        assert_formal_numerical_policy(require_hash_randomization=True)
        development_input_closure.assert_unchanged()
        model_matrix_gate.assert_bytes_unchanged(ROOT)

    station_count = cast(int, evidence["station_count"])
    counts = assert_parameter_budgets(arms, n_stations=station_count)
    run_config = {
        "stage": "09b_development_controls",
        "format": FINAL_FORMAT,
        "execution_role": "prelabel_relative_to_unopened_post_2020_confirmation",
        "evidence_role": "development_only_exploratory",
        "development_disclosure": DEVELOPMENT_DISCLOSURE,
        "panel_date_range": ["2006-01-01", "2020-12-31"],
        "development_evaluation_interval": list(C.SPLIT.test),
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "training_device": "cpu",
        "variables": list(FULL_VARIABLES),
        "context_length": C.CONTEXT_LENGTH,
        "horizons": list(C.HORIZONS),
        "time_split": {
            split_name: list(split_dates)
            for split_name, split_dates in C.SPLIT.as_dict().items()
        },
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "train_config": asdict(TRAIN_CONFIG),
        "arms": [canonical_arm_descriptor(arm) for arm in arms],
        "expected_member_registry": [list(member) for member in expected_member_registry(arms)],
        "parameter_counts": counts,
        "architecture_templates": {
            arm.arm_id: architecture_template(
                arm,
                n_stations=station_count,
            )
            for arm in arms
        },
        "parameter_match_tolerance_fraction": 0.02,
        "architecture_candidates_per_arm": 1,
        "historical_tuning_budget_equalized": False,
        "development_predictor_bridge": predictor_bridge,
        "formal_numerical_policy": runtime_policy,
        "eval_batch_size": int(args.eval_batch_size),
        "input_closure_sha256": development_input_closure.binding_digest,
        "input_closure_file_count": len(development_input_closure.inventory),
    }
    identity = resolve_run_identity(
        root=ROOT,
        panel=panel_path,
        registry=registry_path,
        config=run_config,
        input_closure_sha256=development_input_closure.binding_digest,
    )
    validated_work_order: ValidatedStage09bWorkOrder | None = None
    authorization_path: Path | None = None
    if args.member_work_order is not None:
        try:
            validated_work_order = validate_stage09b_member_work_order(
                root=ROOT,
                work_order=args.member_work_order,
                expected_identity=identity,
                expected_config=run_config,
            )
        except Stage09bPrecomputeError as exc:
            raise ControlExperimentError("Stage-09b member work order failed closed") from exc
        run_dir = validated_work_order.run_directory
    else:
        run_dir = initialise_run_directory(
            ROOT / "outputs" / "runs" / "09b_development_controls",
            identity,
            run_config,
            provenance={
                "development_only": True,
                "post_2020_outcomes_requested_or_read": False,
                "suite_pointer_written": False,
                "training_device": "cpu",
            },
            publication_guard=assert_stage09b_publication_inputs,
        )
        try:
            authorization_path, work_orders = freeze_stage09b_precompute_plan(
                root=ROOT,
                run_directory=run_dir,
                identity=identity,
                resolved_config=run_config,
                matrix_gate=model_matrix_gate,
                publication_guard=assert_stage09b_publication_inputs,
            )
            execute_stage09b_member_work_orders(
                work_orders,
                workers=int(args.precompute_workers),
                launch=lambda work_order: _launch_stage09b_member_process(
                    work_order,
                    panel=str(panel_path),
                    registry=str(registry_path),
                    eval_batch_size=int(args.eval_batch_size),
                    verbose=bool(args.verbose),
                ),
            )
        except Stage09bPrecomputeError as exc:
            raise ControlExperimentError("Stage-09b member precompute failed") from exc
    parents = _parent_bindings(identity, predictor_bridge)

    bundle = D.prepare_dataset_from_panel(
        str(panel_path),
        frozen_spec=frozen_spec_path,
        stable_site_ids=True,
    )
    panel = bundle["panel_raw"]
    panel_imputed = bundle["panel"]
    masks = bundle["masks"]
    stations = tuple(str(station) for station in cast(Sequence[object], bundle["stations"]))
    if not isinstance(panel, pd.DataFrame) or not isinstance(panel_imputed, pd.DataFrame):
        raise ControlExperimentError("canonical panel preparation returned invalid tables")
    if not isinstance(masks, D.SplitMasks):
        raise ControlExperimentError("canonical panel preparation returned invalid split masks")
    if len(stations) != 120:
        raise ControlExperimentError("development controls require the exact 120-site cohort")
    climatology = F.HarmonicClimatology.fit(panel, masks.train)
    thresholds = {
        station: float(
            panel.loc[masks.train & panel["site_id"].astype(str).eq(station), "WTEMP"].quantile(
                C.EXCEEDANCE_QUANTILE
            )
        )
        for station in stations
    }
    if any(not math.isfinite(value) for value in thresholds.values()):
        raise ControlExperimentError("a station lacks a finite train-only event threshold")

    if validated_work_order is not None:
        return _execute_authorized_member(
            validated=validated_work_order,
            arms=arms,
            panel_imputed=panel_imputed,
            masks=masks,
            climatology=climatology,
            thresholds=thresholds,
            stations=stations,
            identity=identity,
            run_config=run_config,
            run_dir=run_dir,
            parents=parents,
            eval_batch_size=int(args.eval_batch_size),
            verbose=bool(args.verbose),
            publication_guard=assert_stage09b_publication_inputs,
        )

    train_examples: int | None = None
    canonical_registry: pd.DataFrame | None = None
    canonical_train_registry: pd.DataFrame | None = None
    for variables, _grouped_arms in _group_arms_by_variables(arms):
        wd = DS.build_windows(
            panel_imputed,
            masks,
            climatology,
            context=C.CONTEXT_LENGTH,
            horizons=C.HORIZONS,
            variables=variables,
            require_observed_target=True,
        )
        current_train_examples = len(wd.idx("train"))
        if train_examples is None:
            train_examples = current_train_examples
        elif current_train_examples != train_examples:
            raise ControlExperimentError("feature ladder changed the training-sample budget")
        current_registry = window_registry_from_windowed(wd, stations)
        current_train_registry = window_registry_from_windowed(
            wd, stations, splits=("train",)
        )
        if canonical_registry is None:
            canonical_registry = current_registry
            canonical_train_registry = current_train_registry
        else:
            assert canonical_train_registry is not None
            key_columns = ["split", *FORECAST_KEY]
            if (
                not current_registry[key_columns].equals(canonical_registry[key_columns])
                or not targets_match_at_model_precision(
                    current_registry["y_true"], canonical_registry["y_true"]
                )
            ):
                raise ControlExperimentError(
                    "feature ladder changed the exact forecast-key/truth registry"
                )
            if (
                not current_train_registry[key_columns].equals(
                    canonical_train_registry[key_columns]
                )
                or not targets_match_at_model_precision(
                    current_train_registry["y_true"],
                    canonical_train_registry["y_true"],
                )
            ):
                raise ControlExperimentError(
                    "feature ladder changed the exact training-window registry"
                )
        del wd, current_registry, current_train_registry
    assert (
        train_examples is not None
        and canonical_registry is not None
        and canonical_train_registry is not None
    )
    if authorization_path is None:  # pragma: no cover - parent/worker split invariant
        raise ControlExperimentError("Stage-09b parent authorization is absent")
    arm_by_id = {arm.arm_id: arm for arm in arms}

    def inspect_member_semantics(
        member: Stage09bMember,
        prediction_path: Path,
    ) -> Mapping[str, Any]:
        frame = pd.read_parquet(prediction_path, columns=R.PRED_COLS)
        return _member_semantic_evidence(
            frame,
            arm=arm_by_id[member.arm_id],
            seed=member.seed,
            canonical_registry=canonical_registry,
        )

    try:
        precomputed_members, coordinator_receipt = finalize_stage09b_precompute(
            root=ROOT,
            authorization_path=authorization_path,
            semantic_inspector=inspect_member_semantics,
            publication_guard=assert_stage09b_publication_inputs,
        )
    except Stage09bPrecomputeError as exc:
        raise ControlExperimentError("Stage-09b 45-member aggregation failed closed") from exc
    member_paths = [
        precomputed_members[member]
        for member in expected_member_registry(arms)
    ]

    audit, resolved_members, summaries = validate_prediction_paths(
        member_paths,
        arms,
        identity=identity,
        parents=parents,
        n_stations=len(stations),
        eval_batch_size=args.eval_batch_size,
        allowed_sites=set(stations),
        canonical_registry=canonical_registry,
    )
    budget = architecture_budget_rows(
        arms,
        n_stations=len(stations),
        train_examples=train_examples,
    )
    # Training-time library code may alter a native pool.  Do not publish any
    # canonical matrix unless the effective policy still holds.
    assert_stage09b_publication_inputs()
    outputs = publish_final_artifacts(
        run_dir=run_dir,
        identity=identity,
        arms=arms,
        member_paths=resolved_members,
        member_parents=parents,
        audit=audit,
        budget=budget,
        summaries=summaries,
        train_examples=train_examples,
        canonical_registry_sha256=window_registry_digest(canonical_registry),
        canonical_train_registry_sha256=window_registry_digest(
            canonical_train_registry
        ),
        # These files are not authoritative without the final completion
        # receipt. The full source/input gate surrounds this call, while each
        # internal atomic write directly rechecks the live native runtime.
        publication_guard=assert_formal_numerical_policy,
    )
    predictions, architecture_budget, metric_summary, report, semantic_audit = outputs
    receipt_path = ROOT / STAGE09B_COMPLETION_RECEIPT_PATH
    receipt = build_stage09b_completion_receipt(
        root=ROOT,
        run_id=identity.run_id,
        run_manifest=run_dir / "run.json",
        frozen_panel_spec=frozen_spec_path,
        panel=panel_path,
        registry=registry_path,
        predictor_bridge=ROOT / predictor_bridge["path"],
        member_paths=resolved_members,
        predictions=predictions,
        architecture_budget=architecture_budget,
        metric_summary=metric_summary,
        report=report,
        semantic_audit=semantic_audit,
        matrix_audit=asdict(audit),
    )
    # This is deliberately the final write in the transaction.  Any missing
    # member, budget/report failure, sidecar drift, or common-key mismatch raises
    # before the stable receipt can be replaced.
    assert_stage09b_publication_inputs()
    assert_formal_numerical_policy()
    publish_stage09b_completion_receipt(
        receipt_path,
        receipt,
        root=ROOT,
        publication_guard=assert_formal_numerical_policy,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_DEVELOPMENT_ONLY",
                "run_id": identity.run_id,
                "run_dir": str(run_dir),
                "members": audit.expected_members,
                "common_forecast_keys": audit.common_forecast_keys,
                "precompute_coordinator_receipt": str(coordinator_receipt),
                "completion_receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.member_work_order is not None:
        # Member workers use their own exact per-member locks. Taking the parent
        # lock here would deadlock against the controller that launched them.
        return _run(args)
    # The create-only publisher deliberately assumes one enclosing owner for
    # plan/coordinator namespaces. Hold that ownership across plan freezing,
    # all member launches, independent aggregation, final artifacts, and the
    # completion receipt. A second controller may resume only after the first
    # complete parent transaction releases this lock.
    manager = advisory_file_lock(
        C.OUTPUTS / ".stage09b-parent.lock",
        exclusive=True,
    )
    try:
        manager.__enter__()
    except RuntimeError as exc:
        raise ControlExperimentError(
            "cannot acquire the Stage-09b parent transaction lock"
        ) from exc
    try:
        return _run(args)
    finally:
        manager.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
