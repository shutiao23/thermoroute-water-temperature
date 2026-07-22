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
* a cumulative ThermoRoute feature ladder on three declared seeds per rung.

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


for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage09b-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE09B_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE09B_NONCE"


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
            or prefix != expected
            or not expected.is_dir()
            or expected == ROOT
            or ROOT in expected.parents
            or (expected / ".controller-nonce").read_text(encoding="utf-8")
            != worker_nonce
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
        environment = os.environ.copy()
        environment[_WORKER_CACHE_ENV] = str(cache_path)
        environment[_WORKER_NONCE_ENV] = nonce
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
from thermoroute.development_controls_gate import (  # noqa: E402
    STAGE09B_COMPLETION_RECEIPT_PATH,
    build_stage09b_completion_receipt,
    publish_stage09b_completion_receipt,
)
from thermoroute.development_controls import (  # noqa: E402
    ArmSpec,
    DEVELOPMENT_DISCLOSURE,
    DEVELOPMENT_SCOPE,
    FEATURE_LADDER,  # noqa: F401 - public script contract used by tests/audits
    FULL_VARIABLES,
    MatrixAudit,
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
    parameter_count,
    physics_count,
    prediction_content_digest,
    recompute_metric_summary,
    render_report,
    summary_csv_bytes,
    window_registry_digest,
    window_registry_from_windowed,
)
from thermoroute.model_suite import (  # noqa: E402
    ModelSuiteError,
    development_predictor_bridge_binding,
)
from thermoroute.registry import FORECAST_KEY, targets_match_at_model_precision  # noqa: E402
from thermoroute.repro import (  # noqa: E402
    RunIdentity,
    assert_formal_numerical_policy,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sha256_json,
    sidecar_path,
    validate_artifact_sidecar,
)
from thermoroute.train import FitResult, configure_deterministic_runtime, fit_model  # noqa: E402


PREDICTION_KIND = "development_control_arm_predictions"
PREDICTION_EXTRA_FORMAT = "thermoroute.development-control-arm.v1"
FINAL_PREDICTION_KIND = "development_controls_combined_predictions"
FINAL_FORMAT = "thermoroute.development-controls.v2"
SUMMARY_KIND = "development_controls_metric_summary"
SEMANTIC_AUDIT_KIND = "development_controls_semantic_audit"
SEMANTIC_AUDIT_FORMAT = "thermoroute.development-controls-semantic-audit.v1"


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


def _arm_extra_static(
    arm: ArmSpec,
    *,
    seed: int,
    parameters: int,
    n_stations: int,
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
    if not isinstance(best, (int, float)) or not math.isfinite(float(best)):
        raise ControlExperimentError("cached arm best validation metric is invalid")
    if type(selected) is not int or selected < 0:
        raise ControlExperimentError("cached arm selected epoch is invalid")
    if final is not None and (type(final) is not int or final < selected):
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


def _create_only_file_from_temp(temp_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise ControlExperimentError(f"refusing to overwrite immutable artifact: {destination}") from exc
    _fsync_parent(destination)


def write_arm_prediction(
    frame: pd.DataFrame,
    path: Path,
    *,
    identity: RunIdentity,
    arm: ArmSpec,
    seed: int,
    parameters: int,
    n_stations: int,
    parents: Mapping[str, str],
    training_summary: Mapping[str, Any],
) -> None:
    """Publish one prediction and sidecar without replacing existing bytes."""
    if path.exists() or sidecar_path(path).exists():
        raise ControlExperimentError(f"refusing to overwrite immutable arm cache: {path}")
    _validate_arm_frame(frame, arm, seed)
    _validate_training_summary(dict(training_summary))
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.loc[:, R.PRED_COLS].to_parquet(temporary_path, index=False)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _create_only_file_from_temp(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    extra = _arm_extra_static(
        arm,
        seed=seed,
        parameters=parameters,
        n_stations=n_stations,
    )
    extra["training_summary"] = dict(training_summary)
    seal_artifact(
        path,
        identity,
        kind=PREDICTION_KIND,
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents=parents,
        extra=extra,
    )


def _checkpoint_final_epoch(path: Path) -> int | None:
    metadata_path = path.with_name(path.name + ".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlExperimentError(f"checkpoint sidecar is invalid: {metadata_path}") from exc
    epoch = metadata.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise ControlExperimentError(f"checkpoint epoch is invalid: {metadata_path}")
    return epoch


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
    fit_function: FitCallable = fit_model,  # type: ignore[assignment]
) -> list[Path]:
    """Train/cache all arms sharing one window tensor without retaining frames."""
    paths: list[Path] = []
    for arm in arms:
        parameters = parameter_count(arm, n_stations=n_stations)
        for seed in arm.seeds:
            prediction_path = run_dir / "arm_predictions" / arm.arm_id / f"seed{seed}.parquet"
            cached = read_arm_prediction(
                prediction_path,
                identity=identity,
                arm=arm,
                seed=seed,
                parameters=parameters,
                n_stations=n_stations,
                parents=parents,
            )
            if cached is not None:
                paths.append(prediction_path)
                continue
            checkpoint_path = run_dir / "checkpoints" / arm.arm_id / f"seed{seed}.pt"
            arm_config = {
                **dict(run_config),
                "arm": asdict(arm),
                "seed": int(seed),
                "trainable_parameters": parameters,
            }

            def factory(arm: ArmSpec = arm, seed: int = seed) -> torch.nn.Module:
                return build_arm_model(arm, seed=seed, n_stations=n_stations)

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
            )
            result.pred["model"] = arm.arm_id
            result.pred["scope"] = DEVELOPMENT_SCOPE
            result.pred["feature_set"] = arm.feature_set
            result.pred["seed"] = int(seed)
            training_summary = {
                "best_validation_metric": float(result.best_val),
                "selected_epoch": int(result.epochs),
                "checkpoint_final_epoch": _checkpoint_final_epoch(checkpoint_path),
            }
            write_arm_prediction(
                result.pred,
                prediction_path,
                identity=identity,
                arm=arm,
                seed=seed,
                parameters=parameters,
                n_stations=n_stations,
                parents=parents,
                training_summary=training_summary,
            )
            paths.append(prediction_path)
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
        frame = read_arm_prediction(
            path,
            identity=identity,
            arm=arm,
            seed=seed,
            parameters=parameter_count(arm, n_stations=n_stations),
            n_stations=n_stations,
            parents=parents,
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


def _create_only_bytes(payload: bytes, destination: Path) -> None:
    if destination.exists():
        raise ControlExperimentError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _create_only_file_from_temp(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _stream_combined_predictions(
    members: Mapping[tuple[str, int], Path],
    destination: Path,
) -> None:
    if destination.exists():
        raise ControlExperimentError(f"refusing to overwrite combined artifact: {destination}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    try:
        for member in members:
            frame = pd.read_parquet(members[member], columns=R.PRED_COLS)
            table = pa.Table.from_pandas(frame, preserve_index=False, schema=schema, safe=True)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise ControlExperimentError("cannot publish an empty prediction matrix")
        writer.close()
        writer = None
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _create_only_file_from_temp(temporary_path, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)


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
        "evidence_scope": "prediction_artifact_closure",
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
    final_paths: Mapping[str, Path],
) -> dict[str, Any]:
    def descriptor(path: Path) -> dict[str, Any]:
        return {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    document: dict[str, Any] = {
        "format": SEMANTIC_AUDIT_FORMAT,
        "status": "PASS_PREDICTION_ARTIFACT_CLOSURE",
        "run_id": identity.run_id,
        "evidence_scope": "prediction_artifact_closure",
        "training_replay_verified": False,
        "post_2020_outcomes_requested_or_read": False,
        "matrix_audit": asdict(audit),
        "canonical_window_registry": {
            "sha256": canonical_registry_sha256,
            "common_forecast_keys": audit.common_forecast_keys,
            "train_examples_per_epoch": train_examples,
            "train_registry_sha256": canonical_train_registry_sha256,
        },
        "members": [
            {
                "arm_id": arm_id,
                "seed": seed,
                "prediction": descriptor(member_paths[(arm_id, seed)]),
                "prediction_sidecar": descriptor(
                    sidecar_path(member_paths[(arm_id, seed)])
                ),
                "normalised_prediction_sha256": member_digests[(arm_id, seed)],
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
            f"arm::{arm_id}::seed{seed}": sha256_file(path)
            for (arm_id, seed), path in member_paths.items()
        },
    }
    arm_by_id = {arm.arm_id: arm for arm in arms}
    normalised_frames: dict[tuple[str, int], pd.DataFrame] = {}
    member_digests: dict[tuple[str, int], str] = {}
    for member in expected_member_registry(arms):
        frame = pd.read_parquet(member_paths[member], columns=R.PRED_COLS)
        try:
            normalised = normalise_prediction_frame(
                frame, arm=arm_by_id[member[0]], seed=member[1]
            )
        except (TypeError, ValueError) as exc:
            raise ControlExperimentError(f"member semantic validation failed: {exc}") from exc
        normalised_frames[member] = normalised
        member_digests[member] = prediction_content_digest(normalised)
    recomputed_summary = recompute_metric_summary(normalised_frames)
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
    del normalised_frames
    report_bytes = render_report(
        run_id=identity.run_id, audit=audit, budget=budget,
        summary=recomputed_summary,
    ).encode("utf-8")
    base_specs = (
        (
            prediction_path, FINAL_PREDICTION_KIND,
            R.PREDICTION_SCHEMA_VERSION, "combined_predictions",
        ),
        (budget_path, "development_controls_budget", "text/csv", "architecture_budget"),
        (summary_path, SUMMARY_KIND, "text/csv", "metric_summary"),
        (report_path, "development_controls_report", "text/markdown", "report"),
    )
    existing = [
        path.exists() or sidecar_path(path).exists()
        for path in (
            prediction_path, budget_path, summary_path, report_path,
            semantic_audit_path,
        )
    ]
    if any(existing):
        if not all(existing):
            raise ControlExperimentError("partial final publication state exists")
        for path, kind, schema, role in (*base_specs, (
            semantic_audit_path, SEMANTIC_AUDIT_KIND,
            "application/json", "semantic_audit",
        )):
            metadata = validate_artifact_sidecar(
                path, identity=identity, schema=schema, kind=kind
            )
            if metadata["parents"] != dict(sorted(final_parents.items())):
                raise ControlExperimentError(f"final artifact parent lineage changed: {path}")
            if metadata["extra"] != _final_extra(audit, artifact_role=role):
                raise ControlExperimentError(f"final artifact metadata changed: {path}")
        expected_semantic = _semantic_audit_document(
            identity=identity, audit=audit, train_examples=train_examples,
            canonical_registry_sha256=canonical_registry_sha256,
            canonical_train_registry_sha256=canonical_train_registry_sha256,
            member_paths=member_paths, member_digests=member_digests,
            final_paths={
                "architecture_budget": budget_path,
                "combined_predictions": prediction_path,
                "metric_summary": summary_path,
                "report": report_path,
            },
        )
        if semantic_audit_path.read_bytes() != (
            json.dumps(expected_semantic, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"):
            raise ControlExperimentError("semantic audit changed")
        return prediction_path, budget_path, summary_path, report_path, semantic_audit_path

    _stream_combined_predictions(member_paths, prediction_path)
    import pyarrow.parquet as pq

    if pq.ParquetFile(prediction_path).metadata.num_rows != audit.prediction_rows:
        raise ControlExperimentError("combined prediction row count changed during publication")
    _create_only_bytes(budget_csv_bytes(budget), budget_path)
    _create_only_bytes(summary_csv_bytes(recomputed_summary), summary_path)
    _create_only_bytes(report_bytes, report_path)
    for path, kind, schema, role in base_specs:
        seal_artifact(
            path,
            identity,
            kind=kind,
            schema=schema,
            parents=final_parents,
            extra=_final_extra(audit, artifact_role=role),
        )
    semantic_document = _semantic_audit_document(
        identity=identity, audit=audit, train_examples=train_examples,
        canonical_registry_sha256=canonical_registry_sha256,
        canonical_train_registry_sha256=canonical_train_registry_sha256,
        member_paths=member_paths, member_digests=member_digests,
        final_paths={
            "architecture_budget": budget_path,
            "combined_predictions": prediction_path,
            "metric_summary": summary_path,
            "report": report_path,
        },
    )
    _create_only_bytes(
        (json.dumps(semantic_document, indent=2, sort_keys=True, allow_nan=False) + "\n")
        .encode("utf-8"),
        semantic_audit_path,
    )
    seal_artifact(
        semantic_audit_path, identity, kind=SEMANTIC_AUDIT_KIND,
        schema="application/json", parents=final_parents,
        extra=_final_extra(audit, artifact_role="semantic_audit"),
    )
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
        help="CPU validation/export batch size; does not change the train budget",
    )
    parser.add_argument("--verbose", action="store_true", help="print epoch diagnostics")
    args = parser.parse_args(argv)
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_deterministic_runtime()
    runtime_policy = assert_formal_numerical_policy()
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

    arms = declared_arms()
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
        "time_split": C.SPLIT.as_dict(),
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "train_config": asdict(TRAIN_CONFIG),
        "arms": [asdict(arm) for arm in arms],
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
    }
    identity = resolve_run_identity(
        root=ROOT,
        panel=panel_path,
        registry=registry_path,
        config=run_config,
    )
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
    )
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

    member_paths: list[Path] = []
    train_examples: int | None = None
    canonical_registry: pd.DataFrame | None = None
    canonical_train_registry: pd.DataFrame | None = None
    for variables, grouped_arms in _group_arms_by_variables(arms):
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
        member_paths.extend(
            train_arm_group(
                grouped_arms,
                wd=wd,
                thresholds=thresholds,
                n_stations=len(stations),
                identity=identity,
                run_config=run_config,
                run_dir=run_dir,
                parents=parents,
                eval_batch_size=args.eval_batch_size,
                verbose=args.verbose,
            )
        )
        del wd, current_registry, current_train_registry
    assert (
        train_examples is not None
        and canonical_registry is not None
        and canonical_train_registry is not None
    )

    audit, resolved_members, summaries = validate_prediction_paths(
        member_paths,
        arms,
        identity=identity,
        parents=parents,
        n_stations=len(stations),
        allowed_sites=set(stations),
        canonical_registry=canonical_registry,
    )
    budget = architecture_budget_rows(
        arms,
        n_stations=len(stations),
        train_examples=train_examples,
    )
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
    publish_stage09b_completion_receipt(receipt_path, receipt, root=ROOT)
    print(
        json.dumps(
            {
                "status": "COMPLETE_DEVELOPMENT_ONLY",
                "run_id": identity.run_id,
                "run_dir": str(run_dir),
                "members": audit.expected_members,
                "common_forecast_keys": audit.common_forecast_keys,
                "completion_receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
