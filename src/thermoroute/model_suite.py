"""Safe, executable model artifacts for the sealed Route-A model suite.

This module deliberately contains no acquisition code.  It serialises models
that were fitted on the development panel and validates them without reading a
confirmation table.  Torch models use :mod:`thermoroute.checkpoint`'s
weights-only format; LightGBM boosters use their native textual representation
instead of pickle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence, cast

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import config as C
from . import data as D
from . import datasets as DS
from . import features as F
from . import probability as P
from . import results as R
from .checkpoint import (
    CHECKPOINT_METADATA_VERSION,
    CHECKPOINT_VERSION,
    _CHECKPOINT_FIELDS,
    _CHECKPOINT_METADATA_FIELDS,
    _validate_checkpoint_payload,
    checkpoint_sidecar_path,
    instantiate_inference_ensemble,
    load_inference_bundle,
    neural_output_head_schema,
)
from .conformal import (
    CQRContractError,
    cqr_offsets_with_audit,
    cqr_policy_contract,
    validate_cqr_offset_bundle,
)
from .chronology import STAGE09_ARTIFACT_PATHS
from .development_controls_gate import (
    DevelopmentControlsGateError,
    validate_stage09b_completion_receipt,
)
from .input_closure import (
    InputClosure,
    InputClosureError,
    compose_input_closure_digest,
    resolve_development_input_closure,
)
from .model_matrix_amendment import (
    AMENDMENT_FORMAT as MODEL_MATRIX_AMENDMENT_FORMAT,
    AMENDMENT_ID as MODEL_MATRIX_AMENDMENT_ID,
    AMENDMENT_RELATIVE as MODEL_MATRIX_AMENDMENT_PATH,
    AMENDMENT_SEAL_FORMAT as MODEL_MATRIX_AMENDMENT_SEAL_FORMAT,
    AMENDMENT_SEAL_RELATIVE as MODEL_MATRIX_AMENDMENT_SEAL_PATH,
    AMENDMENT_SEAL_STATUS as MODEL_MATRIX_AMENDMENT_SEAL_STATUS,
    AMENDMENT_STATUS as MODEL_MATRIX_AMENDMENT_STATUS,
    MODEL_MATRIX_SUITE_BINDING_FORMAT,
    ModelMatrixAmendmentError,
    model_matrix_contract_id,
    validate_model_matrix_amendment,
    validate_model_matrix_amendment_seal,
)
from .provenance import sha256_file
from .quantiles import (
    LIGHTGBM_QUANTILE_REPAIR_METHOD,
    RAW_QUANTILE_CROSSING_AUDIT_FORMAT,
    QuantileIdentityError,
    lightgbm_quantile_repair_contract,
    raw_quantile_crossing_summary,
    repair_lightgbm_quantiles,
    validate_raw_quantile_crossing_summary,
)
from .repro import (
    ARTIFACT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    formal_policy_document,
    numerical_runtime_contract,
    sha256_json,
    sidecar_path,
    source_tree_hash,
    validate_artifact_sidecar,
)
from .registry import (
    FORECAST_KEY,
    ROUTE_A_PRIMARY_MODELS,
    enforce_common_forecast_keys,
    targets_match_at_model_precision,
)
from .weighting import (
    ROW_EQUAL_WEIGHTING,
    STATION_EQUAL_WEIGHTING,
    STATION_SUMMARY_EQUAL_WEIGHTING,
)


LIGHTGBM_BUNDLE_FORMAT = "thermoroute.lightgbm-bundle.v2"
MODEL_SUITE_FORMAT = "thermoroute.route-a-model-suite.v1"
MODEL_SUITE_POINTER_FORMAT = "thermoroute.route-a-model-suite-pointer.v1"
MODEL_SUITE_DOCUMENT_FIELDS = frozenset({
    "format",
    "status",
    "training_device",
    "numerical_runtime_sha256",
    "protocol_sha256",
    "actual_feature_order",
    "development_contract",
    "model_matrix_amendment",
    "preopening_gates",
    "cohorts",
})
MODEL_MATRIX_SUITE_BINDING_FIELDS = frozenset({
    "format", "document", "seal", "contract_id",
})
MODEL_MATRIX_SUITE_DOCUMENT_BINDING_FIELDS = frozenset({
    "path", "sha256", "format", "status", "amendment_id",
    "amendment_document_commit",
})
MODEL_MATRIX_SUITE_SEAL_BINDING_FIELDS = frozenset({
    "path", "sha256", "format", "status",
})
COMPONENT_POINTER_FORMAT = "thermoroute.route-a-model-components.v1"
STAGE9_COMPLETION_FORMAT = "thermoroute.stage09-completion-receipt.v1"
STAGE9_COMPLETION_STATUS = "PASS_FORMAL_STAGE09_COMPLETE"
STAGE9_COMPLETION_RECEIPT_PATH = (
    "outputs/models/route_a_stage09_completion.json"
)
STAGE9_COMPONENT_POINTER_PATH = (
    "outputs/models/route_a_stage9_components.json"
)
STAGE16_COMPLETION_FORMAT = "thermoroute.stage16-completion-receipt.v1"
STAGE16_COMPLETION_STATUS = "PASS_FORMAL_STAGE16_COMPLETE"
STAGE16_COMPLETION_RECEIPT_PATH = (
    "outputs/models/route_a_stage16_completion.json"
)
STAGE16_COMPONENT_POINTER_PATH = (
    "outputs/models/route_a_lstm_components.json"
)
STAGE16_SHORTCUT_POINTER_PATH = "outputs/models/lstm_usgs_bundle.json"
STAGE16_SELECTION_PATH = "outputs/tables/lstm_validation_selection.csv"
STAGE16_PARENT_PREDICTION_PATH = (
    "outputs/predictions/usgs_predictions_stage9_v2.parquet"
)
STAGE16_DEVELOPMENT_PREDICTION_PATH = (
    "outputs/predictions/usgs_predictions_v2.parquet"
)
STAGE16_PARITY_AUDIT_FORMAT = "thermoroute.stage16-bundle-parity.v1"
STAGE16_SELECTION_AUDIT_FORMAT = "thermoroute.stage16-selection-audit.v1"
STAGE16_SELECTION_METRIC_ATOL = 1e-5
STAGE25_COMPLETION_FORMAT = "thermoroute.stage25-completion-receipt.v1"
STAGE25_COMPLETION_STATUS = "COMPLETE"
STAGE25_COMPLETION_RECEIPT_PATH = (
    "outputs/models/route_a_stage25_completion.json"
)
STAGE25_COMPONENT_POINTER_PATH = (
    "outputs/models/route_a_external_components.json"
)
STAGE25_REQUIRED_MODELS = ("ThermoRoute", "LSTM", "LightGBM")
STAGE9_COMPLETION_ARTIFACTS = (
    "run_manifest",
    "predictions",
    "prediction_sidecar",
    "scores",
    "report",
    "lightgbm_selection",
    "thermoroute_pointer",
    "lightgbm_pointer",
    "components_pointer",
)
STAGE9_AIR2STREAM_DISPLAY_NAME = (
    "Air2stream-style a4/a8 (unofficial, non-primary)"
)
STAGE9_AIR2STREAM_MODELS = ("Air2stream-a4", "Air2stream-a8")
STAGE9_LGO_MODEL = "ThermoRoute-LGO-WarmStart"
STAGE9_KNOWN_SPLITS = frozenset({"train", "val", "calib", "test", "none"})
STAGE9_POINT_KEY = (*FORECAST_KEY, "split")
DEVELOPMENT_PREDICTOR_BRIDGE_FORMAT = (
    "thermoroute.development-predictor-bridge.v1"
)
DEVELOPMENT_PREDICTOR_BRIDGE_PATH = (
    "data_usgs/development_predictor_bridge_v1.json"
)

LIGHTGBM_HEADS = ("point", "q05", "q50", "q95", "event")
LIGHTGBM_QUANTILE_AUDIT_KEY_COLUMNS = (
    "site_id", "horizon", "split", "issue_date", "target_date",
)
DEVELOPMENT_CALIBRATED_HEAD_GATE_FORMAT = (
    "thermoroute.development-calibrated-head-gate.v1"
)
CALIBRATION_FIT_CONTRACT_FORMAT = "thermoroute.route-a-calibration-fit.v1"
CALIBRATION_REPLAY_ATOL = 1e-12


def route_a_calibration_fit_contract(*, external: bool) -> dict[str, Any]:
    """Return the exact development-only fit interval for CQR and Platt."""
    return {
        "format": CALIBRATION_FIT_CONTRACT_FORMAT,
        "model_training_interval_inclusive": list(C.SPLIT.train),
        "hyperparameter_selection_interval_inclusive": list(C.SPLIT.val),
        "source_split": "calib",
        "issue_date_interval_inclusive": list(C.SPLIT.calib),
        "post_2018_rows_allowed": False,
        "member_aggregation_before_fit": "equal_weight_member_mean",
        "event_reference_fit_interval_inclusive": [
            C.SPLIT.train[0], C.SPLIT.calib[1]
        ],
        "cqr": {
            "alpha": 0.10,
            "grouping": "pooled_by_horizon" if external else "site_by_horizon",
            "target_date_boundary_purge": C.SPLIT.calib[1],
            "deployed_offset": "qhat_plus=max(raw_qhat,0)",
        },
        "event_probability": {
            "method": "Platt_logistic_by_horizon",
            "grouping": "pooled_by_horizon",
            "fit_interval": list(C.SPLIT.calib),
            "fit_weighting": STATION_EQUAL_WEIGHTING,
        },
    }
LSTM_VALIDATION_GRID = (
    {"d": 64, "layers": 1, "dropout": 0.0, "station_embed_dim": 8,
     "use_derived_context": False, "anchor": "persistence"},
    {"d": 64, "layers": 1, "dropout": 0.0, "station_embed_dim": 8,
     "use_derived_context": True, "anchor": "damped"},
    {"d": 64, "layers": 2, "dropout": 0.10, "station_embed_dim": 8,
     "use_derived_context": True, "anchor": "damped"},
)
STAGE16_LSTM_SELECTION_COLUMNS = (
    "candidate_id", "d", "layers", "dropout", "station_embed_dim",
    "use_derived_context", "anchor", "val_station_macro_rmse", "selected",
    "selection_split",
)


def stage16_validation_winner(metrics: Sequence[float]) -> int:
    """Choose the lowest-ID candidate within the predeclared metric tolerance."""
    values = [float(value) for value in metrics]
    if len(values) != len(LSTM_VALIDATION_GRID) or any(
        not np.isfinite(value) or value < 0.0 for value in values
    ):
        raise ModelSuiteError("Stage-16 validation metrics are malformed")
    best = min(values)
    return min(
        candidate_id
        for candidate_id, value in enumerate(values)
        if value <= best + STAGE16_SELECTION_METRIC_ATOL
    )


def _stage16_consistent_validation_winner(
    reported: Sequence[float],
    recomputed: Sequence[float],
    checkpoint: Sequence[float],
) -> int:
    """Require all three tolerated metric views to select the same candidate."""
    winners = {
        stage16_validation_winner(reported),
        stage16_validation_winner(recomputed),
        stage16_validation_winner(checkpoint),
    }
    if len(winners) != 1:
        raise ModelSuiteError(
            "Stage-16 reported, recomputed, and checkpoint validation winners differ"
        )
    return winners.pop()


STAGE9_LIGHTGBM_SELECTION_COLUMNS = (
    "horizon", "candidate_id", "num_leaves", "min_child_samples",
    "learning_rate", "val_station_macro_rmse", "best_iteration", "selected",
    "selection_split",
)
STAGE9_LIGHTGBM_VALIDATION_GRID = (
    {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.03},
    {"num_leaves": 31, "min_child_samples": 40, "learning_rate": 0.03},
    {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
    {"num_leaves": 31, "min_child_samples": 80, "learning_rate": 0.05},
)
STAGE9_USGS_VARIABLES = (
    "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP",
)
STAGE9_FORMAL_PROTOCOL = (
    f"route_a_strict_v1_balanced_delta{C.DELTA_SCALE:g}"
)
STAGE9_FORMAL_TRAIN_CONFIG = asdict(C.TrainConfig(batch_size=1536))
STAGE9_FORMAL_CONFIG_FIELDS = frozenset({
    "stage", "protocol", "panel", "station_registry", "variables",
    "horizons", "context_length", "seeds", "time_split", "train_config",
    "thermoroute_seeds", "lightgbm_seeds", "ablation_seeds", "delta_scale",
    "station_sampling", "selection_metric", "ablations", "air2stream",
    "device", "training_device", "execution_role",
    "development_predictor_bridge", "eval_batch_size",
    "lightgbm_validation_grid", "event_reference_fit_interval",
    "formal_numerical_policy",
    "input_closure_sha256", "input_closure_file_count",
})
PRIMARY_MODELS = (
    "Persistence", "DampedPersistence", "Climatology",
    "LightGBM", "LSTM", "ThermoRoute",
)
MANDATORY_ABLATIONS = (
    "DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
    "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded",
)
STAGE9_ABLATION_SEEDS = tuple(int(seed) for seed in C.USGS_SEEDS)
ABLATION_INTERVENTIONS: dict[str, dict[str, Any]] = {
    "DampedPriorOnly": {"use_prior": False, "residual_model": False},
    "TR-noDynamicPrior": {"use_prior": False},
    "TR-fixedKappa": {"fixed_kappa": True},
    "TR-noRouter": {"use_router": False},
    "TR-noMoE": {"use_moe": False},
    "TR-noTCN": {"use_tcn": False},
    "TR-unbounded": {"delta_scale": None},
}
STAGE9_PREDICTION_MODELS = (
    "Persistence", "DampedPersistence", "Climatology",
    "LightGBM", "ThermoRoute", *MANDATORY_ABLATIONS, STAGE9_LGO_MODEL,
)
TEMPORAL_MODELS = PRIMARY_MODELS + MANDATORY_ABLATIONS
EXTERNAL_MODELS = PRIMARY_MODELS
BUILTIN_MODELS = frozenset({"Persistence", "DampedPersistence", "Climatology"})
DEVELOPMENT_REPLAY_MODEL_CONTRACTS = {
    "temporal": {
        "LightGBM": ("lightgbm_bundle", 5, 1e-12),
        "LSTM": ("lstm_bundle", 5, 1e-5),
        "ThermoRoute": ("thermoroute_bundle", 5, 1e-5),
        **{
            model: ("thermoroute_bundle", len(STAGE9_ABLATION_SEEDS), 1e-5)
            for model in MANDATORY_ABLATIONS
        },
    },
    "external": {
        "LightGBM": ("lightgbm_bundle", 5, 1e-12),
        "LSTM": ("lstm_bundle", 5, 1e-5),
        "ThermoRoute": ("thermoroute_bundle", 5, 1e-5),
    },
}


class ModelSuiteError(RuntimeError):
    """A supposedly frozen artifact is missing, inconsistent, or unsafe."""


def _relative(root: Path, path: str | Path) -> str:
    root = root.resolve()
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ModelSuiteError(f"model artifact is outside repository: {resolved}")
    return resolved.relative_to(root).as_posix()


def _resolve_inside(root: Path, relative: object, *, directory: bool = False) -> Path:
    root = root.resolve()
    raw = Path(str(relative))
    if raw.is_absolute():
        raise ModelSuiteError(f"artifact path must be repository-relative: {raw}")
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ModelSuiteError(f"artifact path escapes repository: {raw}")
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise ModelSuiteError(f"artifact is missing: {raw}")
    return path


def file_binding(root: str | Path, path: str | Path) -> dict[str, str]:
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(root, path), "sha256": sha256_file(path)}


def canonical_stage09_artifact_paths(run_id: str) -> dict[str, str]:
    """Return the only paths that may participate in a formal Stage-9 PASS."""
    if not isinstance(run_id, str) or not run_id:
        raise ModelSuiteError("Stage-9 canonical paths require a run id")
    return {
        "run_manifest": f"outputs/runs/09_usgs_experiment/{run_id}/run.json",
        **STAGE09_ARTIFACT_PATHS,
    }


def stage09_outputs_are_canonical(
    *,
    root: str | Path,
    predictions: str | Path,
    scores: str | Path,
    report: str | Path,
) -> bool:
    """Whether the three user-selectable Stage-9 outputs use formal paths."""
    root = Path(root).resolve()
    requested = {
        "predictions": Path(predictions).resolve(),
        "scores": Path(scores).resolve(),
        "report": Path(report).resolve(),
    }
    return all(
        path == (root / STAGE09_ARTIFACT_PATHS[label]).resolve()
        for label, path in requested.items()
    )


def _require_canonical_stage09_paths(
    root: Path,
    run_id: str,
    paths: Mapping[str, Path],
) -> None:
    expected = canonical_stage09_artifact_paths(run_id)
    for label, path in paths.items():
        expected_relative = expected.get(label)
        if expected_relative is None or _relative(root, path) != expected_relative:
            raise ModelSuiteError(
                f"Stage-9 {label} is not at its exact canonical path"
            )


def development_predictor_bridge_binding(
    root: str | Path,
    *,
    panel_sha256: str,
    registry_sha256: str,
    path: str | Path | None = None,
) -> dict[str, str]:
    """Validate and bind the outcome-free development predictor bridge gate."""
    root = Path(root).resolve()
    manifest_path = (
        (root / DEVELOPMENT_PREDICTOR_BRIDGE_PATH).resolve()
        if path is None else Path(path).resolve()
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelSuiteError(
            "formal training requires the development predictor bridge manifest"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("format") != DEVELOPMENT_PREDICTOR_BRIDGE_FORMAT
        or manifest.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or manifest.get("outcome_values_requested_or_read") is not False
    ):
        raise ModelSuiteError("development predictor bridge is not an exact-product PASS")
    panel = manifest.get("panel")
    registry = manifest.get("registry")
    if (
        not isinstance(panel, Mapping)
        or not isinstance(registry, Mapping)
        or panel.get("sha256") != str(panel_sha256)
        or registry.get("sha256") != str(registry_sha256)
    ):
        raise ModelSuiteError(
            "development predictor bridge binds another panel or station registry"
        )
    return file_binding(root, manifest_path)


def canonical_development_contract(
    root: str | Path,
    frozen_spec_path: str | Path,
    *,
    panel_sha256: str,
    registry_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Resolve the formal development inputs through ``FrozenPanelSpec`` only."""
    from .evidence import FrozenPanelSpec

    root = Path(root).resolve()
    spec_path = Path(frozen_spec_path).resolve()
    spec = FrozenPanelSpec.load(spec_path)
    spec.verify()
    if sha256_file(spec.panel_path) != str(panel_sha256):
        raise ModelSuiteError("run panel differs from canonical FrozenPanelSpec")
    if sha256_file(spec.registry_path) != str(registry_sha256):
        raise ModelSuiteError("run registry differs from canonical FrozenPanelSpec")
    source = str(source_sha256)
    if len(source) != 64:
        raise ModelSuiteError("run source tree lacks SHA-256 identity")
    predictor_bridge = development_predictor_bridge_binding(
        root,
        panel_sha256=panel_sha256,
        registry_sha256=registry_sha256,
    )
    return {
        "frozen_panel_spec": file_binding(root, spec_path),
        "panel": file_binding(root, spec.panel_path),
        "registry": file_binding(root, spec.registry_path),
        "predictor_bridge": predictor_bridge,
        "source_sha256": source,
    }


def directory_binding(root: str | Path, directory: str | Path) -> dict[str, str]:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    metadata, weights = directory / "metadata.json", directory / "weights.pt"
    if not metadata.is_file() or not weights.is_file():
        raise ModelSuiteError(f"torch bundle is incomplete: {directory}")
    return {
        "path": _relative(root, directory),
        "metadata_sha256": sha256_file(metadata),
        "weights_sha256": sha256_file(weights),
    }


def _booster(model: Any) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    booster = getattr(model, "booster_", None)
    if not isinstance(booster, lgb.Booster):
        raise TypeError("LightGBM bundle members must expose a fitted Booster")
    return booster


def _prediction_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_frame_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash a sorted tabular registry with explicit date/float normalisation."""
    missing = set(columns) - set(frame)
    if missing:
        raise ModelSuiteError(f"prediction frame lacks digest columns: {sorted(missing)}")
    normalised = frame.loc[:, list(columns)].copy()
    for column in columns:
        if column.endswith("date"):
            normalised[column] = pd.to_datetime(normalised[column]).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )
        elif pd.api.types.is_float_dtype(normalised[column]):
            normalised[column] = normalised[column].map(
                lambda value: "NA" if pd.isna(value) else format(float(value), ".17g")
            )
        else:
            normalised[column] = normalised[column].astype(str)
    normalised = normalised.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    payload = normalised.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def development_prediction_binding(
    root: str | Path,
    artifact: str | Path,
    predictions: pd.DataFrame,
    *,
    max_abs_difference: float,
    atol: float,
) -> dict[str, Any]:
    """Bind one bundle to its immutable development rows and parity replay."""
    artifact = Path(artifact)
    sidecar = sidecar_path(artifact)
    if not artifact.is_file() or not sidecar.is_file():
        raise ModelSuiteError("development prediction and sealed sidecar are required")
    try:
        validate_artifact_sidecar(
            artifact, schema=R.PREDICTION_SCHEMA_VERSION
        )
    except ValueError as exc:
        raise ModelSuiteError(
            "development prediction sidecar fails strict lineage validation"
        ) from exc
    key_columns = ("site_id", "horizon", "issue_date", "target_date")
    value_columns = tuple(R.PRED_COLS)
    missing_prediction_columns = set(value_columns) - set(predictions)
    if missing_prediction_columns:
        raise ModelSuiteError(
            f"development predictions lack canonical columns: "
            f"{sorted(missing_prediction_columns)}"
        )
    if len(predictions) == 0:
        raise ModelSuiteError("development prediction parity frame is empty")
    models = tuple(sorted(predictions["model"].astype(str).unique()))
    if len(models) != 1:
        raise ModelSuiteError("one bundle prediction binding must select exactly one model")
    seeds = tuple(sorted(int(value) for value in predictions["seed"].unique()))
    difference = float(max_abs_difference)
    tolerance = float(atol)
    if not np.isfinite(difference) or difference < 0 or tolerance < 0 or difference > tolerance:
        raise ModelSuiteError("development prediction parity exceeds its tolerance")
    return {
        "artifact": {
            **file_binding(root, artifact),
            "sidecar": file_binding(root, sidecar),
        },
        "rows": int(len(predictions)),
        "selection": {"model": models[0], "seeds": list(seeds)},
        "forecast_key_columns": list(key_columns),
        "prediction_columns": list(value_columns),
        "forecast_key_registry_sha256": canonical_frame_digest(
            predictions.drop_duplicates(list(key_columns)), key_columns
        ),
        "prediction_sha256": canonical_frame_digest(predictions, value_columns),
        "max_abs_difference": difference,
        "atol": tolerance,
    }


def verify_sequence_prediction_parity(
    directory: str | Path,
    *,
    wd: Any,
    expected: pd.DataFrame,
    model_factory: Any,
    member_seeds: Mapping[str, int],
    atol: float = 1e-5,
    batch_size: int = 4096,
    splits: tuple[str, ...] = ("val", "calib", "test"),
    publication_guard: Callable[[], object] | None = None,
) -> float:
    """Replay every sequence member and compare all five prediction heads."""
    models, metadata = instantiate_inference_ensemble(
        directory,
        model_factory=lambda member, bundle: model_factory(member, bundle),
        expected_member_count=len(member_seeds),
        device="cpu",
        publication_guard=publication_guard,
    )
    if set(models) != set(member_seeds):
        raise ModelSuiteError("sequence parity member registry differs from bundle")
    from .train import _export_predictions

    keys = ["seed", "site_id", "horizon", "split", "issue_date", "target_date"]
    values = ["y_true", "y_pred", "q05", "q50", "q95", "p_exceed"]
    reference = expected.copy()
    for column in ("model", "scope", "feature_set"):
        unique_values = reference[column].astype(str).unique()
        if len(unique_values) != 1 or not str(unique_values[0]):
            raise ModelSuiteError(
                f"sequence parity expected rows mix {column} values"
            )
    reference["issue_date"] = pd.to_datetime(reference["issue_date"])
    reference["target_date"] = pd.to_datetime(reference["target_date"])
    maximum = 0.0
    model_name = str(reference["model"].iloc[0])
    scope = str(reference["scope"].iloc[0])
    feature_set = str(reference["feature_set"].iloc[0])
    for member, model in models.items():
        seed = int(member_seeds[member])
        replay = _export_predictions(
            model, wd, {}, torch.device("cpu"), model_name, scope, feature_set,
            seed, batch_size=batch_size, splits=splits,
        )
        if publication_guard is not None:
            publication_guard()
        expected_member = reference[reference["seed"].astype(int).eq(seed)]
        paired = expected_member[keys + values].merge(
            replay[keys + values], on=keys, how="outer", suffixes=("_reference", "_bundle"),
            indicator=True, validate="one_to_one",
        )
        if not paired["_merge"].eq("both").all():
            raise ModelSuiteError(f"sequence parity keys differ for {member}")
        for value in values:
            left = paired[f"{value}_reference"].to_numpy(float)
            right = paired[f"{value}_bundle"].to_numpy(float)
            difference = np.abs(left - right)
            if np.any(~np.isfinite(difference)):
                raise ModelSuiteError(f"sequence parity has non-finite {value} values")
            maximum = max(maximum, float(difference.max(initial=0.0)))
    if maximum > float(atol):
        raise ModelSuiteError(
            f"sequence development prediction parity failed: {maximum} > {atol}"
        )
    if publication_guard is not None:
        publication_guard()
    return maximum


def update_torch_development_prediction(
    directory: str | Path,
    binding: Mapping[str, Any],
) -> None:
    """Verify measured parity without mutating an already published bundle.

    The bundle freezes an upper bound before the round-trip replay.  The replay
    may prove a smaller observed difference, but that observation is not allowed
    to rewrite the content-addressed model object.
    """
    directory = Path(directory)
    _, metadata = load_inference_bundle(directory)
    _require_compatible_measured_binding(
        metadata.get("development_prediction"), binding, label="sequence"
    )
    load_inference_bundle(directory, expected_member_count=int(metadata["member_count"]))


def _require_compatible_measured_binding(
    frozen: object,
    measured: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(frozen, Mapping):
        raise ModelSuiteError(f"{label} bundle lacks a frozen prediction binding")
    left, right = dict(frozen), dict(measured)
    frozen_bound = float(left.pop("max_abs_difference", np.inf))
    measured_value = float(right.pop("max_abs_difference", np.inf))
    if left != right:
        raise ModelSuiteError(f"{label} measured parity refers to another prediction artifact")
    tolerance = float(frozen.get("atol", -1.0))
    if (
        not np.isfinite(frozen_bound)
        or not np.isfinite(measured_value)
        or tolerance < 0.0
        or frozen_bound > tolerance
        or measured_value > tolerance
    ):
        raise ModelSuiteError(f"{label} measured parity exceeds its frozen tolerance")


@dataclass(frozen=True)
class _DevelopmentPredictionSnapshot:
    """One immutable read of a prediction artifact and its lineage sidecar."""

    frame: pd.DataFrame
    selected: pd.DataFrame
    artifact_path: Path
    artifact_sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    sidecar: Mapping[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_prediction_sidecar_snapshot(
    metadata: object,
    *,
    artifact: Path,
    artifact_payload: bytes,
    artifact_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    """Validate lineage against already-read artifact bytes, never the path."""
    expected_keys = {
        "schema_version", "kind", "artifact", "artifact_sha256",
        "artifact_bytes", "content_schema", "run", "parents", "extra",
        "created_utc",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected_keys:
        raise ModelSuiteError(
            f"{label} development prediction sidecar schema is not exact"
        )
    try:
        created = datetime.fromisoformat(str(metadata["created_utc"]))
    except ValueError as exc:
        raise ModelSuiteError(
            f"{label} development prediction sidecar timestamp is invalid"
        ) from exc
    artifact_bytes = metadata.get("artifact_bytes")
    if (
        created.tzinfo is None
        or created.utcoffset() is None
        or metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or metadata.get("artifact") != artifact.name
        or isinstance(artifact_bytes, (bool, np.bool_))
        or not isinstance(artifact_bytes, (int, np.integer))
        or int(artifact_bytes) != len(artifact_payload)
        or metadata.get("artifact_sha256") != artifact_sha256
        or metadata.get("content_schema") != R.PREDICTION_SCHEMA_VERSION
        or not isinstance(metadata.get("kind"), str)
        or not metadata.get("kind")
        or not isinstance(metadata.get("parents"), Mapping)
        or not isinstance(metadata.get("extra"), Mapping)
    ):
        raise ModelSuiteError(
            f"{label} development prediction bytes or sidecar fields changed"
        )
    parents = metadata["parents"]
    assert isinstance(parents, Mapping)
    if any(
        not isinstance(name, str)
        or not name
        or not _is_sha256(digest)
        for name, digest in parents.items()
    ):
        raise ModelSuiteError(
            f"{label} development prediction parent registry is malformed"
        )
    run = metadata.get("run")
    run_keys = {
        "run_id", "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "input_closure_sha256",
        "schema_version",
    }
    if (
        not isinstance(run, Mapping)
        or set(run) != run_keys
        or run.get("schema_version") != RUN_SCHEMA_VERSION
        or not isinstance(run.get("run_id"), str)
        or not run.get("run_id")
        or any(
            not _is_sha256(run.get(field))
            for field in (
                "panel_sha256", "registry_sha256", "config_sha256",
                "source_sha256", "runtime_sha256", "input_closure_sha256",
            )
        )
    ):
        raise ModelSuiteError(
            f"{label} development prediction run identity is malformed"
        )
    return dict(metadata)


def _read_development_prediction_snapshot(
    root: str | Path,
    value: object,
    *,
    label: str,
) -> _DevelopmentPredictionSnapshot:
    """Read, bind and parse development predictions from one byte snapshot."""
    if not isinstance(value, Mapping):
        raise ModelSuiteError(f"{label} lacks development prediction binding")
    required = {
        "artifact", "rows", "forecast_key_registry_sha256", "prediction_sha256",
        "max_abs_difference", "atol", "selection", "forecast_key_columns",
        "prediction_columns",
    }
    if set(value) != required:
        raise ModelSuiteError(f"{label} prediction binding schema is not exact")
    artifact = value["artifact"]
    artifact_keys = {"path", "sha256", "sidecar"}
    if not isinstance(artifact, Mapping) or set(artifact) != artifact_keys:
        raise ModelSuiteError(f"{label} prediction artifact binding is malformed")
    sidecar_binding = artifact.get("sidecar")
    if (
        not isinstance(sidecar_binding, Mapping)
        or set(sidecar_binding) != {"path", "sha256"}
    ):
        raise ModelSuiteError(
            f"{label} prediction sidecar binding is malformed"
        )
    path = _resolve_inside(Path(root), artifact.get("path"))
    try:
        artifact_payload = path.read_bytes()
    except OSError as exc:
        raise ModelSuiteError(
            f"{label} development prediction cannot be read"
        ) from exc
    artifact_digest = _sha256_bytes(artifact_payload)
    if artifact_digest != artifact.get("sha256"):
        raise ModelSuiteError(f"{label} development prediction checksum mismatch")
    sidecar = _resolve_inside(Path(root), sidecar_binding.get("path"))
    if sidecar != sidecar_path(path).resolve():
        raise ModelSuiteError(f"{label} binds a non-canonical prediction sidecar")
    try:
        sidecar_payload = sidecar.read_bytes()
    except OSError as exc:
        raise ModelSuiteError(
            f"{label} development prediction sidecar cannot be read"
        ) from exc
    sidecar_digest = _sha256_bytes(sidecar_payload)
    if sidecar_digest != sidecar_binding.get("sha256"):
        raise ModelSuiteError(
            f"{label} development prediction sidecar checksum mismatch"
        )
    try:
        sidecar_document = json.loads(sidecar_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteError(
            f"{label} development prediction sidecar is invalid JSON"
        ) from exc
    validated_sidecar = _validate_prediction_sidecar_snapshot(
        sidecar_document,
        artifact=path,
        artifact_payload=artifact_payload,
        artifact_sha256=artifact_digest,
        label=label,
    )
    rows = value.get("rows")
    if (
        isinstance(rows, (bool, np.bool_))
        or not isinstance(rows, (int, np.integer))
        or int(rows) < 1
    ):
        raise ModelSuiteError(f"{label} development prediction row count is empty")
    for field in ("forecast_key_registry_sha256", "prediction_sha256"):
        if not _is_sha256(value.get(field)):
            raise ModelSuiteError(f"{label} {field} is not SHA-256")
    try:
        difference = float(value["max_abs_difference"])
        tolerance = float(value["atol"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelSuiteError(
            f"{label} development prediction parity is malformed"
        ) from exc
    if (
        isinstance(value["max_abs_difference"], (bool, np.bool_))
        or isinstance(value["atol"], (bool, np.bool_))
        or not np.isfinite(difference)
        or difference < 0
        or not np.isfinite(tolerance)
        or tolerance < 0
        or difference > tolerance
    ):
        raise ModelSuiteError(f"{label} development prediction parity failed")
    selection = value["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {"model", "seeds"}:
        raise ModelSuiteError(f"{label} prediction selection is malformed")
    model = selection.get("model")
    raw_seeds = selection.get("seeds")
    if (
        not isinstance(model, str)
        or not model
        or isinstance(raw_seeds, (str, bytes))
        or not isinstance(raw_seeds, Sequence)
        or not raw_seeds
        or any(
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            for seed in raw_seeds
        )
    ):
        raise ModelSuiteError(f"{label} prediction selection is malformed")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise ModelSuiteError(f"{label} prediction seed registry is duplicated")
    if tuple(value["forecast_key_columns"]) != (
        "site_id", "horizon", "issue_date", "target_date"
    ):
        raise ModelSuiteError(f"{label} forecast-key schema changed")
    if tuple(value["prediction_columns"]) != tuple(R.PRED_COLS):
        raise ModelSuiteError(f"{label} prediction value schema changed")
    try:
        frame = pd.read_parquet(BytesIO(artifact_payload))
    except Exception as exc:
        raise ModelSuiteError(
            f"{label} development prediction cannot be parsed"
        ) from exc
    missing = set(R.PRED_COLS) - set(frame)
    if missing:
        raise ModelSuiteError(
            f"{label} development prediction columns are incomplete: {sorted(missing)}"
        )
    numeric_seed = pd.to_numeric(frame["seed"], errors="coerce")
    if (
        numeric_seed.isna().any()
        or not np.isfinite(numeric_seed.to_numpy(float)).all()
        or not np.equal(
            numeric_seed.to_numpy(float), numeric_seed.to_numpy(float).astype(int)
        ).all()
    ):
        raise ModelSuiteError(f"{label} development prediction seed is not integral")
    selected = frame[
        frame["model"].astype(str).eq(model)
        & numeric_seed.astype(int).isin(seeds)
    ].copy()
    if len(selected) != int(rows):
        raise ModelSuiteError(f"{label} development prediction row count changed")
    key_columns = ("site_id", "horizon", "issue_date", "target_date")
    key_digest = canonical_frame_digest(
        selected.drop_duplicates(list(key_columns)), key_columns
    )
    prediction_digest = canonical_frame_digest(selected, R.PRED_COLS)
    if key_digest != value["forecast_key_registry_sha256"]:
        raise ModelSuiteError(f"{label} development forecast-key digest mismatch")
    if prediction_digest != value["prediction_sha256"]:
        raise ModelSuiteError(f"{label} development prediction digest mismatch")
    return _DevelopmentPredictionSnapshot(
        frame=frame,
        selected=selected,
        artifact_path=path,
        artifact_sha256=artifact_digest,
        sidecar_path=sidecar,
        sidecar_sha256=sidecar_digest,
        sidecar=validated_sidecar,
    )


def _assert_prediction_snapshot_unchanged(
    snapshot: _DevelopmentPredictionSnapshot, *, label: str
) -> None:
    """Close the validation window before accepting a replay result."""
    try:
        artifact_digest = sha256_file(snapshot.artifact_path)
        sidecar_digest = sha256_file(snapshot.sidecar_path)
    except OSError as exc:
        raise ModelSuiteError(
            f"{label} development prediction changed during validation"
        ) from exc
    if (
        artifact_digest != snapshot.artifact_sha256
        or sidecar_digest != snapshot.sidecar_sha256
    ):
        raise ModelSuiteError(
            f"{label} development prediction changed during validation"
        )


def validate_development_prediction_binding(
    root: str | Path, value: object, *, label: str,
) -> None:
    snapshot = _read_development_prediction_snapshot(root, value, label=label)
    _assert_prediction_snapshot_unchanged(snapshot, label=label)


def _normalise_lightgbm_models(
    models: Mapping[str, Mapping[int | str, Mapping[str, Any]]],
    horizons: Sequence[int],
) -> dict[str, dict[int, dict[str, Any]]]:
    expected = tuple(int(value) for value in horizons)
    normalised: dict[str, dict[int, dict[str, Any]]] = {}
    for raw_member, horizon_models in models.items():
        member = str(raw_member)
        if not member or member in normalised:
            raise ModelSuiteError("LightGBM member names are empty or duplicated")
        normalised[member] = {}
        for raw_horizon, heads in horizon_models.items():
            horizon = int(raw_horizon)
            if horizon in normalised[member]:
                raise ModelSuiteError(f"duplicate LightGBM {member} horizon {horizon}")
            if set(heads) != set(LIGHTGBM_HEADS):
                raise ModelSuiteError(
                    f"LightGBM {member}/h{horizon} heads differ from {LIGHTGBM_HEADS}"
                )
            normalised[member][horizon] = dict(heads)
        if set(normalised[member]) != set(expected) or len(normalised[member]) != len(expected):
            raise ModelSuiteError(
                f"LightGBM {member} does not contain every declared horizon"
            )
    if not normalised:
        raise ModelSuiteError("LightGBM bundle has no ensemble members")
    return normalised


def _normalise_lightgbm_quantile_audit_inputs(
    values: Mapping[Any, tuple[pd.DataFrame, Any]],
    horizons: Sequence[int],
) -> dict[int, tuple[pd.DataFrame, Any]]:
    expected = tuple(int(value) for value in horizons)
    if not isinstance(values, Mapping):
        raise ModelSuiteError("LightGBM quantile audit inputs are not a mapping")
    provided: dict[int, tuple[pd.DataFrame, Any]] = {}
    for raw_horizon, value in values.items():
        if isinstance(raw_horizon, bool):
            raise ModelSuiteError("LightGBM quantile audit horizon is boolean")
        horizon = int(raw_horizon)
        if horizon in provided:
            raise ModelSuiteError("LightGBM quantile audit duplicates a horizon")
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ModelSuiteError("LightGBM quantile audit input is malformed")
        registry, design = value
        if not isinstance(registry, pd.DataFrame):
            raise ModelSuiteError("LightGBM quantile audit registry is not tabular")
        registry = registry.reset_index(drop=True).copy()
        required = {"site_id", "split", "issue_date", "target_date"}
        missing = required - set(registry)
        if missing:
            raise ModelSuiteError(
                f"LightGBM quantile audit registry lacks {sorted(missing)}"
            )
        if len(registry) < 1 or len(registry) != len(design):
            raise ModelSuiteError(
                "LightGBM quantile audit registry and design lengths differ"
            )
        if registry[["site_id", "split"]].isna().any().any():
            raise ModelSuiteError("LightGBM quantile audit keys contain nulls")
        if "horizon" in registry and not registry["horizon"].astype(int).eq(
            int(horizon)
        ).all():
            raise ModelSuiteError("LightGBM quantile audit horizon column changed")
        registry["site_id"] = registry["site_id"].astype(str)
        registry["split"] = registry["split"].astype(str)
        registry["horizon"] = int(horizon)
        for column in ("issue_date", "target_date"):
            registry[column] = pd.to_datetime(registry[column], errors="raise")
        key_columns = list(LIGHTGBM_QUANTILE_AUDIT_KEY_COLUMNS)
        if (
            registry["site_id"].str.strip().eq("").any()
            or registry["split"].str.strip().eq("").any()
            or registry[key_columns].isna().any().any()
            or registry.duplicated(key_columns).any()
        ):
            raise ModelSuiteError("LightGBM quantile audit keys are invalid")
        provided[horizon] = (registry, design)
    if set(provided) != set(expected) or len(provided) != len(expected):
        raise ModelSuiteError(
            "LightGBM quantile audit inputs do not cover every horizon"
        )
    return provided


def _build_lightgbm_raw_crossing_audit(
    models: Mapping[str, Mapping[int, Mapping[str, lgb.Booster]]],
    values: Mapping[Any, tuple[pd.DataFrame, Any]],
    *,
    members: Sequence[str],
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Evaluate every nominal quantile head before the frozen repair."""
    inputs = _normalise_lightgbm_quantile_audit_inputs(values, horizons)
    audits: dict[str, dict[str, dict[str, Any]]] = {}
    for member in members:
        audits[str(member)] = {}
        for horizon in horizons:
            registry, design = inputs[int(horizon)]
            key_digest = canonical_frame_digest(
                registry, LIGHTGBM_QUANTILE_AUDIT_KEY_COLUMNS
            )
            heads = models[str(member)][int(horizon)]
            raw = {
                name: np.asarray(
                    heads[name].predict(design, num_threads=1), dtype=float
                )
                for name in ("q05", "q50", "q95")
            }
            if any(len(prediction) != len(registry) for prediction in raw.values()):
                raise ModelSuiteError(
                    f"LightGBM {member}/h{horizon} raw quantile rows changed"
                )
            try:
                audits[str(member)][str(int(horizon))] = (
                    raw_quantile_crossing_summary(
                        raw["q05"], raw["q50"], raw["q95"],
                        forecast_key_sha256=key_digest,
                    )
                )
            except QuantileIdentityError as exc:
                raise ModelSuiteError(
                    f"LightGBM {member}/h{horizon} raw quantile audit failed"
                ) from exc
    document = {
        "format": RAW_QUANTILE_CROSSING_AUDIT_FORMAT,
        "scope": "development_export_rows_before_repair",
        "key_columns": list(LIGHTGBM_QUANTILE_AUDIT_KEY_COLUMNS),
        "repair_method": LIGHTGBM_QUANTILE_REPAIR_METHOD,
        "members": audits,
    }
    return {**document, "audit_sha256": sha256_json(document)}


def _validate_lightgbm_quantile_metadata(manifest: Mapping[str, Any]) -> None:
    if manifest.get("quantile_repair") != lightgbm_quantile_repair_contract():
        raise ModelSuiteError("LightGBM quantile repair contract changed")
    members = tuple(str(value) for value in manifest.get("members", ()))
    try:
        horizons = tuple(int(value) for value in manifest.get("horizons", ()))
    except (TypeError, ValueError) as exc:
        raise ModelSuiteError("LightGBM raw quantile audit horizons are invalid") from exc
    if (
        not members
        or len(members) != len(set(members))
        or not horizons
        or len(horizons) != len(set(horizons))
    ):
        raise ModelSuiteError("LightGBM raw quantile audit registry is invalid")
    audit = manifest.get("raw_quantile_crossing_audit")
    expected_fields = {
        "format", "scope", "key_columns", "repair_method", "members",
        "audit_sha256",
    }
    if not isinstance(audit, Mapping) or set(audit) != expected_fields:
        raise ModelSuiteError("LightGBM raw quantile audit schema is not exact")
    if (
        audit.get("format") != RAW_QUANTILE_CROSSING_AUDIT_FORMAT
        or audit.get("scope") != "development_export_rows_before_repair"
        or tuple(audit.get("key_columns", ()))
        != LIGHTGBM_QUANTILE_AUDIT_KEY_COLUMNS
        or audit.get("repair_method") != LIGHTGBM_QUANTILE_REPAIR_METHOD
    ):
        raise ModelSuiteError("LightGBM raw quantile audit contract changed")
    stable_audit = {
        key: value for key, value in audit.items() if key != "audit_sha256"
    }
    if audit.get("audit_sha256") != sha256_json(stable_audit):
        raise ModelSuiteError("LightGBM raw quantile audit self hash changed")
    member_audits = audit.get("members")
    if not isinstance(member_audits, Mapping) or set(member_audits) != set(members):
        raise ModelSuiteError("LightGBM raw quantile audit member registry changed")
    expected_horizons = {str(horizon) for horizon in horizons}
    horizon_keys: dict[str, tuple[int, str]] = {}
    for member in members:
        values = member_audits[member]
        if not isinstance(values, Mapping) or set(values) != expected_horizons:
            raise ModelSuiteError(
                f"LightGBM {member} raw quantile audit horizons changed"
            )
        for horizon in expected_horizons:
            try:
                validate_raw_quantile_crossing_summary(values[horizon])
            except QuantileIdentityError as exc:
                raise ModelSuiteError(
                    f"LightGBM {member}/h{horizon} raw crossing audit is invalid"
                ) from exc
            rows_and_keys = (
                int(values[horizon]["rows"]),
                str(values[horizon]["forecast_key_sha256"]),
            )
            if horizon in horizon_keys and horizon_keys[horizon] != rows_and_keys:
                raise ModelSuiteError(
                    f"LightGBM h{horizon} raw audit keys differ across members"
                )
            horizon_keys[horizon] = rows_and_keys


def _frozen_lightgbm_n_jobs() -> int:
    """Return the frozen LightGBM training concurrency from the policy.

    The numerical-policy document (route_a_numerical_policy_v2.json) freezes
    one uniform role cap for every stage; the bundle metadata must declare
    exactly that cap.  The per-process cap itself is enforced by
    ``assert_role_thread_cap`` before training.
    """
    policy = formal_policy_document(Path(__file__).resolve().parents[2])
    caps = policy.get("role_thread_caps")
    if not isinstance(caps, Mapping) or not caps:
        raise ModelSuiteError("numerical policy role caps are invalid")
    values = {caps[role] for role in caps}
    if len(values) != 1 or next(iter(values)) < 1:
        raise ModelSuiteError("numerical policy role caps are not uniform")
    return next(iter(values))


def save_lightgbm_bundle(
    directory: str | Path,
    *,
    models: Mapping[str, Mapping[int | str, Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    quantile_audit_inputs: Mapping[
        int | str, tuple[pd.DataFrame, Any]
    ],
    parity_inputs: Mapping[int | str, Any] | None = None,
    parity_atol: float = 1e-12,
    publication_guard: Callable[[], object] | None = None,
) -> Path:
    """Save point/quantile/event boosters and prove native-text round-trip parity.

    ``metadata`` must describe both the raw seven-variable information set and
    the exact engineered design columns.  ``quantile_audit_inputs`` is the full
    exported development registry/design used to record every raw crossing
    before repair.  No Python object is pickled.  When ``parity_inputs`` is
    supplied, every head is reconstructed from disk and compared with its
    in-memory booster before the manifest is finalised.
    """
    required = {
        "run_id", "raw_feature_order", "design_feature_order", "horizons",
        "station_agnostic", "uses_station_categorical", "preprocessing",
        "station_categories",
        "training_weighting", "deterministic_training",
        "event_thresholds", "event_calibrators", "conformal_offsets",
        "conformal_policy", "conformal_offset_audit",
        "calibration_fit_contract",
        "source_sha256", "panel_sha256", "registry_sha256", "config_sha256",
        "runtime_sha256", "input_closure_sha256", "training_device",
        "development_prediction",
    }
    missing = required - set(metadata)
    if missing:
        raise ModelSuiteError(f"LightGBM metadata missing: {sorted(missing)}")
    reserved = {"quantile_repair", "raw_quantile_crossing_audit"} & set(metadata)
    if reserved:
        raise ModelSuiteError(
            "LightGBM caller cannot override generated quantile metadata: "
            f"{sorted(reserved)}"
        )
    raw_order = tuple(str(value) for value in metadata["raw_feature_order"])
    design_order = tuple(str(value) for value in metadata["design_feature_order"])
    horizons = tuple(int(value) for value in metadata["horizons"])
    if not raw_order or "WTEMP" not in raw_order or len(raw_order) != len(set(raw_order)):
        raise ModelSuiteError("invalid LightGBM raw feature order")
    if not design_order or len(design_order) != len(set(design_order)):
        raise ModelSuiteError("invalid LightGBM design feature order")
    station_agnostic = bool(metadata["station_agnostic"])
    uses_category = bool(metadata["uses_station_categorical"])
    if station_agnostic == uses_category:
        raise ModelSuiteError(
            "station-agnostic LightGBM must omit site category; same-station must use it"
        )
    if metadata.get("training_weighting") != "equal_total_weight_per_station":
        raise ModelSuiteError("LightGBM training is not station-balanced")
    if metadata.get("deterministic_training") != {
        "deterministic": True, "force_col_wise": True,
        "n_jobs": _frozen_lightgbm_n_jobs(),
    }:
        raise ModelSuiteError("LightGBM deterministic training contract changed")
    if metadata.get("training_device") != "cpu":
        raise ModelSuiteError("formal LightGBM bundles must be trained on CPU")
    _validate_cqr_metadata(metadata, label="LightGBM bundle")
    _validate_calibration_fit_metadata(
        metadata, label="LightGBM bundle", external=station_agnostic
    )
    categories = tuple(str(value) for value in metadata["station_categories"])
    if uses_category:
        if not categories or len(categories) != len(set(categories)):
            raise ModelSuiteError("same-station LightGBM lacks stable station categories")
        if design_order[-1] != "station_code":
            raise ModelSuiteError("same-station LightGBM design must end in station_code")
    elif categories or "station_code" in design_order:
        raise ModelSuiteError("station-agnostic LightGBM must omit station categories")
    normalised = _normalise_lightgbm_models(models, horizons)
    members = tuple(sorted(normalised))
    declared_members = tuple(str(value) for value in metadata.get("members", members))
    declared_count = int(metadata.get("member_count", len(members)))
    if declared_members != members or declared_count != len(members):
        raise ModelSuiteError("LightGBM member registry is inconsistent")
    original_boosters: dict[str, dict[int, dict[str, lgb.Booster]]] = {
        member: {
            horizon: {
                head: _booster(normalised[member][horizon][head])
                for head in LIGHTGBM_HEADS
            }
            for horizon in horizons
        }
        for member in members
    }
    raw_crossing_audit = _build_lightgbm_raw_crossing_audit(
        original_boosters,
        quantile_audit_inputs,
        members=members,
        horizons=horizons,
    )
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", suffix=".staging",
        dir=destination.parent,
    ))

    bindings: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for member in members:
        bindings[member] = {}
        for horizon in horizons:
            bindings[member][str(horizon)] = {}
            for head in LIGHTGBM_HEADS:
                booster = original_boosters[member][horizon][head]
                path = directory / f"{member}_h{horizon}_{head}.txt"
                atomic_write_bytes(path, booster.model_to_string().encode("utf-8"))
                bindings[member][str(horizon)][head] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                }

    manifest = {
        **dict(metadata),
        "format": LIGHTGBM_BUNDLE_FORMAT,
        "horizons": list(horizons),
        "raw_feature_order": list(raw_order),
        "design_feature_order": list(design_order),
        "heads": list(LIGHTGBM_HEADS),
        "members": list(members),
        "member_count": len(members),
        "quantile_repair": lightgbm_quantile_repair_contract(),
        "raw_quantile_crossing_audit": raw_crossing_audit,
        "models": bindings,
        # Kept as a first-class index for the opening preflight contract.
        "point_models": {
            str(horizon): {
                member: bindings[member][str(horizon)]["point"] for member in members
            }
            for horizon in horizons
        },
    }
    manifest_path = directory / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    parity: dict[str, dict[str, dict[str, Any]]] = {}
    loaded, _ = load_lightgbm_bundle(manifest_path)
    if parity_inputs is not None:
        provided = {int(key): value for key, value in parity_inputs.items()}
        if set(provided) != set(horizons):
            raise ModelSuiteError("parity inputs do not cover every LightGBM horizon")
        for member in members:
            parity[member] = {}
            for horizon in horizons:
                X = provided[horizon]
                if len(X) == 0:
                    raise ModelSuiteError(f"empty parity input for horizon {horizon}")
                parity[member][str(horizon)] = {}
                for head in LIGHTGBM_HEADS:
                    before = np.asarray(
                        original_boosters[member][horizon][head].predict(
                            X, num_threads=1
                        ), dtype=float
                    )
                    after = np.asarray(
                        loaded[member][horizon][head].predict(X, num_threads=1),
                        dtype=float,
                    )
                    difference = float(np.max(np.abs(before - after)))
                    if not np.allclose(before, after, rtol=0.0, atol=parity_atol):
                        raise ModelSuiteError(
                            f"LightGBM {member}/h{horizon}/{head} parity failed: {difference}"
                        )
                    parity[member][str(horizon)][head] = {
                        "rows": int(len(before)),
                        "max_abs_difference": difference,
                        "prediction_sha256": _prediction_digest(after),
                    }
    manifest["roundtrip_parity"] = parity
    atomic_write_json(manifest_path, manifest)
    # Re-open after the final manifest write, so a malformed index never escapes.
    load_lightgbm_bundle(manifest_path)
    try:
        if destination.exists():
            if not destination.is_dir() or not _directory_bytes_equal(
                directory, destination
            ):
                raise FileExistsError(
                    f"refusing to replace non-identical LightGBM bundle: {destination}"
                )
            if publication_guard is not None:
                publication_guard()
            shutil.rmtree(directory)
            return destination / "manifest.json"
        if publication_guard is not None:
            publication_guard()
        os.rename(directory, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return destination / "manifest.json"


def _directory_bytes_equal(left: Path, right: Path) -> bool:
    """Compare two flat model objects without trusting names alone."""
    left_files = {path.relative_to(left) for path in left.rglob("*") if path.is_file()}
    right_files = {path.relative_to(right) for path in right.rglob("*") if path.is_file()}
    return left_files == right_files and all(
        sha256_file(left / relative) == sha256_file(right / relative)
        for relative in left_files
    )


def load_lightgbm_bundle(
    manifest_or_directory: str | Path,
    *,
    publication_guard: Callable[[], object] | None = None,
) -> tuple[dict[str, dict[int, dict[str, lgb.Booster]]], dict[str, Any]]:
    """Reconstruct every native-text booster after strict checksum validation.

    Formal authority-bearing callers pass ``publication_guard`` so a live
    native-thread drift cannot be hidden inside an otherwise valid bundle
    acceptance.
    """
    if publication_guard is not None:
        publication_guard()
    value = Path(manifest_or_directory)
    manifest_path = value / "manifest.json" if value.is_dir() else value
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelSuiteError(f"cannot read LightGBM manifest: {manifest_path}") from exc
    if manifest.get("format") != LIGHTGBM_BUNDLE_FORMAT:
        raise ModelSuiteError("unsupported LightGBM bundle format")
    raw_order = tuple(str(value) for value in manifest.get("raw_feature_order", ()))
    design_order = tuple(str(value) for value in manifest.get("design_feature_order", ()))
    if not raw_order or "WTEMP" not in raw_order or not design_order:
        raise ModelSuiteError("LightGBM feature registry is malformed")
    station_agnostic = bool(manifest.get("station_agnostic"))
    uses_category = bool(manifest.get("uses_station_categorical"))
    categories = tuple(str(value) for value in manifest.get("station_categories", ()))
    if station_agnostic == uses_category:
        raise ModelSuiteError("LightGBM station-identity flags are inconsistent")
    if manifest.get("training_weighting") != "equal_total_weight_per_station":
        raise ModelSuiteError("LightGBM training weighting is not frozen")
    if manifest.get("deterministic_training") != {
        "deterministic": True, "force_col_wise": True,
        "n_jobs": _frozen_lightgbm_n_jobs(),
    }:
        raise ModelSuiteError("LightGBM deterministic contract is not frozen")
    if uses_category:
        if not categories or len(categories) != len(set(categories)):
            raise ModelSuiteError("LightGBM station categories are malformed")
        if design_order[-1:] != ("station_code",):
            raise ModelSuiteError("same-station LightGBM design lacks station_code")
    elif categories or "station_code" in design_order:
        raise ModelSuiteError("station-agnostic LightGBM retains a station category")
    horizons = tuple(int(value) for value in manifest.get("horizons", ()))
    if tuple(manifest.get("heads", ())) != LIGHTGBM_HEADS:
        raise ModelSuiteError("LightGBM head registry is incomplete or reordered")
    bindings = manifest.get("models")
    members = tuple(str(value) for value in manifest.get("members", ()))
    if (not members or len(members) != len(set(members))
            or int(manifest.get("member_count", -1)) != len(members)):
        raise ModelSuiteError("LightGBM member registry is incomplete")
    _validate_lightgbm_quantile_metadata(manifest)
    if not isinstance(bindings, Mapping) or set(bindings) != set(members):
        raise ModelSuiteError("LightGBM model registry differs from its members")
    output: dict[str, dict[int, dict[str, lgb.Booster]]] = {}
    root = manifest_path.parent.resolve()
    for member in members:
        horizon_bindings = bindings[member]
        if (not isinstance(horizon_bindings, Mapping)
                or set(horizon_bindings) != {str(h) for h in horizons}):
            raise ModelSuiteError(f"LightGBM {member} horizon registry is incomplete")
        output[member] = {}
        for horizon in horizons:
            heads = horizon_bindings[str(horizon)]
            if not isinstance(heads, Mapping) or set(heads) != set(LIGHTGBM_HEADS):
                raise ModelSuiteError(
                    f"LightGBM {member}/h{horizon} head registry is incomplete"
                )
            output[member][horizon] = {}
            for head in LIGHTGBM_HEADS:
                binding = heads[head]
                if not isinstance(binding, Mapping) or set(binding) < {"path", "sha256"}:
                    raise ModelSuiteError(
                        f"LightGBM {member}/h{horizon}/{head} binding is malformed"
                    )
                raw = Path(str(binding["path"]))
                if raw.is_absolute():
                    raise ModelSuiteError("LightGBM model path must be bundle-relative")
                path = (root / raw).resolve()
                if root not in path.parents or not path.is_file():
                    raise ModelSuiteError("LightGBM model path escapes or is missing")
                if sha256_file(path) != binding["sha256"]:
                    raise ModelSuiteError(
                        f"LightGBM {member}/h{horizon}/{head} checksum mismatch"
                    )
                output[member][horizon][head] = lgb.Booster(model_file=str(path))
                if publication_guard is not None:
                    publication_guard()
    points = manifest.get("point_models")
    if not isinstance(points, Mapping) or set(points) != {str(h) for h in horizons}:
        raise ModelSuiteError("LightGBM point-model index is incomplete")
    for horizon in horizons:
        expected = {
            member: bindings[member][str(horizon)]["point"] for member in members
        }
        if points[str(horizon)] != expected:
            raise ModelSuiteError("LightGBM point-model index disagrees with head registry")
    if publication_guard is not None:
        publication_guard()
    return output, manifest


def update_lightgbm_development_prediction(
    manifest_path: str | Path,
    binding: Mapping[str, Any],
) -> None:
    """Verify measured parity without rewriting the native-text bundle."""
    _, manifest = load_lightgbm_bundle(manifest_path)
    _require_compatible_measured_binding(
        manifest.get("development_prediction"), binding, label="LightGBM"
    )


def verify_lightgbm_prediction_parity(
    manifest_path: str | Path,
    *,
    evaluation_design: Mapping[int, tuple[pd.DataFrame, Any]],
    expected: pd.DataFrame,
    member_seeds: Mapping[str, int],
    atol: float = 1e-12,
    publication_guard: Callable[[], object] | None = None,
) -> float:
    """Replay all five native boosters against the Stage-9 development rows."""
    models, metadata = load_lightgbm_bundle(
        manifest_path,
        publication_guard=publication_guard,
    )
    if set(models) != set(member_seeds):
        raise ModelSuiteError("LightGBM parity member registry differs from bundle")
    measured_crossing_audit = _build_lightgbm_raw_crossing_audit(
        models,
        evaluation_design,
        members=tuple(str(value) for value in metadata["members"]),
        horizons=tuple(int(value) for value in metadata["horizons"]),
    )
    if measured_crossing_audit != metadata.get("raw_quantile_crossing_audit"):
        raise ModelSuiteError(
            "LightGBM raw crossing audit differs from native-model replay"
        )
    if publication_guard is not None:
        publication_guard()
    keys = ["seed", "site_id", "horizon", "split", "issue_date", "target_date"]
    values = ["y_true", "y_pred", "q05", "q50", "q95", "p_exceed"]
    reference = expected.copy()
    reference["issue_date"] = pd.to_datetime(reference["issue_date"])
    reference["target_date"] = pd.to_datetime(reference["target_date"])
    maximum = 0.0
    for member, seed in member_seeds.items():
        for horizon in tuple(int(value) for value in metadata["horizons"]):
            if horizon not in evaluation_design:
                raise ModelSuiteError(f"LightGBM parity lacks h{horizon} design")
            registry, X = evaluation_design[horizon]
            registry = registry.reset_index(drop=True).copy()
            if len(registry) != len(X):
                raise ModelSuiteError("LightGBM parity design and registry lengths differ")
            heads = models[member][horizon]
            try:
                q05, q50, q95 = repair_lightgbm_quantiles(
                    heads["q05"].predict(X, num_threads=1),
                    heads["q50"].predict(X, num_threads=1),
                    heads["q95"].predict(X, num_threads=1),
                )
            except QuantileIdentityError as exc:
                raise ModelSuiteError(
                    f"LightGBM parity quantiles are invalid for {member}/h{horizon}"
                ) from exc
            replay = pd.DataFrame({
                "seed": int(seed),
                "site_id": registry["site_id"].astype(str).to_numpy(),
                "horizon": int(horizon),
                "split": registry["split"].astype(str).to_numpy(),
                "issue_date": pd.to_datetime(registry["issue_date"]).to_numpy(),
                "target_date": pd.to_datetime(registry["target_date"]).to_numpy(),
                "y_true": registry["y"].to_numpy(float),
                "y_pred": heads["point"].predict(X, num_threads=1),
                "q05": q05, "q50": q50, "q95": q95,
                "p_exceed": heads["event"].predict(X, num_threads=1),
            })
            ref = reference[
                reference["seed"].astype(int).eq(int(seed))
                & reference["horizon"].astype(int).eq(int(horizon))
            ]
            paired = ref[keys + values].merge(
                replay[keys + values], on=keys, how="outer",
                suffixes=("_reference", "_bundle"), indicator=True,
                validate="one_to_one",
            )
            if not paired["_merge"].eq("both").all():
                raise ModelSuiteError(f"LightGBM parity keys differ for {member}/h{horizon}")
            for value in values:
                left = paired[f"{value}_reference"].to_numpy(float)
                right = paired[f"{value}_bundle"].to_numpy(float)
                if value == "y_true":
                    if not targets_match_at_model_precision(left, right):
                        raise ModelSuiteError(
                            "LightGBM parity target labels differ at frozen "
                            "model precision"
                        )
                    continue
                difference = np.abs(left - right)
                if np.any(~np.isfinite(difference)):
                    raise ModelSuiteError(f"LightGBM parity has non-finite {value}")
                maximum = max(maximum, float(difference.max(initial=0.0)))
            if publication_guard is not None:
                publication_guard()
    if maximum > float(atol):
        raise ModelSuiteError(
            f"LightGBM development prediction parity failed: {maximum} > {atol}"
        )
    if publication_guard is not None:
        publication_guard()
    return maximum


def fit_pooled_imputer(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    fit_stations: Sequence[str],
) -> D.Imputer:
    """Fit one station-balanced development-only imputer.

    Each station is first reduced to one median per day-of-year (and one global
    median), then those station summaries receive equal weight through a second
    median.  A station with a longer or more complete record therefore cannot
    dominate the pooled fill value.  Replication makes the station-agnostic
    contract machine-checkable and allows expansion to new site identifiers
    without observing their confirmation data.
    """
    sites = tuple(str(site) for site in fit_stations)
    if not sites or len(sites) != len(set(sites)):
        raise ValueError("pooled imputer fit stations are empty or duplicated")
    selected = np.asarray(train_mask, dtype=bool) & panel.site_id.astype(str).isin(sites).to_numpy()
    training = panel.loc[selected].copy()
    if training.empty:
        raise ValueError("pooled imputer training partition is empty")
    represented = set(training.site_id.astype(str))
    missing_sites = sorted(set(sites) - represented)
    if missing_sites:
        raise ValueError(
            "pooled imputer lacks train rows for fit stations: "
            f"{missing_sites[:5]}"
        )
    training["doy"] = pd.to_datetime(training["DATE"]).dt.dayofyear
    medians: dict[tuple[str, str], pd.Series] = {}
    global_median: dict[tuple[str, str], float] = {}
    for variable in C.ALL_VARS:
        station_seasonal = (
            training.groupby(["site_id", "doy"], sort=True)[variable]
            .median()
            .dropna()
            .rename("station_median")
            .reset_index()
        )
        seasonal = station_seasonal.groupby("doy", sort=True)[
            "station_median"
        ].median()
        station_global = training.groupby("site_id", sort=True)[variable].median()
        fallback = float(station_global.median())
        if not np.isfinite(fallback):
            raise ValueError(
                f"pooled imputer cannot fit a finite station-summary {variable} median"
            )
        for site in sites:
            medians[(site, variable)] = seasonal.copy()
            global_median[(site, variable)] = fallback
    return D.Imputer(
        medians=medians,
        global_median=global_median,
        fit_stations=sites,
        pooled=True,
    )


def _finite(value: object) -> float | None:
    number = float(value)  # type: ignore[arg-type]
    return number if np.isfinite(number) else None


def _tuple_map(values: Mapping[tuple[str, str], object], variables: set[str]) -> dict[str, Any]:
    return {
        f"{station}|{variable}": _finite(value)
        for (station, variable), value in sorted(values.items())
        if variable in variables
    }


def serialise_preprocessing(wd: Any, climatology: Any, imputer: D.Imputer) -> dict[str, Any]:
    """Serialise all train-fit transformations shared by TR/LSTM/LightGBM."""
    variables = set(wd.var_names)
    seasonal: dict[str, dict[str, float]] = {}
    for (station, variable), series in sorted(imputer.medians.items()):
        if variable in variables:
            seasonal[f"{station}|{variable}"] = {
                str(int(day)): float(value)
                for day, value in series.items() if np.isfinite(value)
            }
    imputer_pooled = bool(getattr(imputer, "pooled", False))
    anchor = wd.damped_anchor
    anchor_fit_stations = tuple(str(value) for value in anchor.fit_stations)
    eligible_fit_stations = tuple(
        str(value) for value in anchor.eligible_fit_stations
    )
    pair_counts = {
        str(station): int(value) for station, value in anchor.pair_counts.items()
    }
    mean_squares = {
        str(station): (
            None if value is None else float(value)
        )
        for station, value in anchor.lagged_anomaly_mean_squares.items()
    }
    if (
        anchor.eligibility_rule != F.DAMPED_ELIGIBILITY_RULE
        or not anchor_fit_stations
        or len(set(anchor_fit_stations)) != len(anchor_fit_stations)
        or set(pair_counts) != set(anchor_fit_stations)
        or set(mean_squares) != set(anchor_fit_stations)
        or any(type(value) is not int or value < 0 for value in pair_counts.values())
        or any(
            value is not None and (not np.isfinite(value) or value < 0.0)
            for value in mean_squares.values()
        )
    ):
        raise ModelSuiteError("damped-anchor eligibility evidence is malformed")
    def is_anchor_eligible(station: str) -> bool:
        mean_square = mean_squares[station]
        return (
            pair_counts[station] >= int(anchor.min_pairs)
            and mean_square is not None
            and mean_square >= float(anchor.min_mean_square)
        )

    derived_eligible = tuple(filter(is_anchor_eligible, anchor_fit_stations))
    if eligible_fit_stations != derived_eligible:
        raise ModelSuiteError("damped-anchor eligible-station registry is stale")
    if bool(anchor.pooled) and eligible_fit_stations != anchor_fit_stations:
        raise ModelSuiteError(
            "pooled damped anchor silently excludes a declared fit station"
        )
    return {
        "input_schema": {
            "variables": list(wd.var_names),
            "physics_forcings": list(wd.phys_vars),
            "context_length": int(wd.X.shape[1]),
            "transforms": {
                variable: ("signed_log1p" if variable == "FLOW" else "log1p_nonnegative")
                for variable in C.LOG1P_VARS if variable in variables
            },
            "missingness_mask": True,
        },
        "imputer": {
            "method": D.POOLED_STATION_BALANCED_IMPUTER_METHOD if imputer_pooled
                      else D.PER_STATION_IMPUTER_METHOD,
            "pooled": imputer_pooled,
            "pool_weighting": (
                STATION_SUMMARY_EQUAL_WEIGHTING if imputer_pooled else None
            ),
            "fit_stations": list(getattr(imputer, "fit_stations", C.STATIONS)),
            "seasonal_medians": seasonal,
            "global_medians": _tuple_map(imputer.global_median, variables),
        },
        "scaler": {
            "method": D.POOLED_SCALER_METHOD if wd.scaler.pooled
                      else D.PER_STATION_SCALER_METHOD,
            "pool_weighting": (
                STATION_EQUAL_WEIGHTING if wd.scaler.pooled else None
            ),
            "variance": (
                D.POOLED_SCALER_VARIANCE if wd.scaler.pooled
                else "within_station_sample_variance_ddof_1"
            ),
            "mean": _tuple_map(wd.scaler.mean, variables),
            "std": _tuple_map(wd.scaler.std, variables),
            "fit_stations": list(wd.scaler.fit_stations),
            "pooled": bool(wd.scaler.pooled),
        },
        "climatology": {
            "method": F.POOLED_HARMONIC_METHOD if climatology.pooled
                      else F.PER_STATION_HARMONIC_METHOD,
            "pool_weighting": (
                STATION_EQUAL_WEIGHTING if climatology.pooled else None
            ),
            "harmonics": int(climatology.k),
            "coefficients": {
                str(station): [float(value) for value in coefficients]
                for station, coefficients in sorted(climatology.coef.items())
            },
            "fit_stations": list(climatology.fit_stations),
            "pooled": bool(climatology.pooled),
        },
        "damped_anchor": {
            "method": F.POOLED_DAMPED_AR_METHOD if anchor.pooled
                      else F.DAMPED_AR_METHOD,
            "phi": {str(station): float(value)
                    for station, value in sorted(anchor.phi.items())},
            "fit_stations": list(anchor_fit_stations),
            "pooled": bool(anchor.pooled),
            "fallback": float(anchor.fallback),
            "min_pairs": int(anchor.min_pairs),
            "coefficient_bounds": [
                float(anchor.lower_bound),
                float(anchor.upper_bound),
            ],
            "minimum_lagged_anomaly_mean_square": float(
                anchor.min_mean_square
            ),
            "pair_rule": F.DAMPED_PAIR_RULE,
            "pool_weighting": STATION_EQUAL_WEIGHTING,
            "eligibility_rule": F.DAMPED_ELIGIBILITY_RULE,
            "eligible_fit_stations": list(eligible_fit_stations),
            "pair_counts": dict(sorted(pair_counts.items())),
            "lagged_anomaly_mean_squares": dict(sorted(mean_squares.items())),
        },
    }


def serialise_offsets(offsets: Mapping[tuple[str, int], object]) -> dict[str, float]:
    """Serialise only finite, nonnegative deployed CQR offsets.

    Signed raw order statistics belong in ``conformal_offset_audit``.  Allowing
    this final-deployment serializer to turn infinity into JSON ``null`` or to
    preserve a negative value would bypass the Route-A widens-only contract.
    """
    output: dict[str, float] = {}
    for (station, horizon), raw_value in sorted(offsets.items()):
        if isinstance(raw_value, (bool, np.bool_)):
            raise ModelSuiteError("deployed CQR offset cannot be boolean")
        try:
            value = float(cast(Any, raw_value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelSuiteError("deployed CQR offset is not numeric") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ModelSuiteError("deployed CQR offsets must be finite and nonnegative")
        key = f"{str(station)}|{int(horizon)}"
        if key in output:
            raise ModelSuiteError("deployed CQR registry has a duplicate key")
        output[key] = value
    if not output:
        raise ModelSuiteError("deployed CQR registry is empty")
    return output


def _validate_cqr_metadata(
    metadata: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Require an exact nonnegative offset registry with raw signed provenance."""
    try:
        return validate_cqr_offset_bundle(
            metadata.get("conformal_offsets", {}),
            metadata.get("conformal_policy"),
            metadata.get("conformal_offset_audit"),
        )
    except CQRContractError as exc:
        raise ModelSuiteError(f"{label} CQR deployment contract is invalid") from exc


def _validate_calibration_fit_metadata(
    metadata: Mapping[str, Any], *, label: str, external: bool
) -> None:
    if metadata.get("calibration_fit_contract") != (
        route_a_calibration_fit_contract(external=external)
    ):
        raise ModelSuiteError(
            f"{label} CQR/Platt development fit interval contract changed"
        )


def _strict_calibration_replay_match(
    expected: object,
    observed: object,
    *,
    label: str,
    field: str,
) -> None:
    """Compare replayed metadata recursively with a fixed absolute tolerance."""
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ModelSuiteError(
                f"{label} replayed {field} field registry differs from metadata"
            )
        for key in sorted(expected, key=str):
            _strict_calibration_replay_match(
                expected[key], observed[key], label=label,
                field=f"{field}.{key}",
            )
        return
    if isinstance(expected, (list, tuple)):
        if (
            not isinstance(observed, (list, tuple))
            or len(observed) != len(expected)
        ):
            raise ModelSuiteError(
                f"{label} replayed {field} sequence differs from metadata"
            )
        for index, (left, right) in enumerate(zip(expected, observed)):
            _strict_calibration_replay_match(
                left, right, label=label, field=f"{field}[{index}]"
            )
        return
    if isinstance(expected, (bool, np.bool_)):
        if not isinstance(observed, (bool, np.bool_)) or bool(observed) != bool(expected):
            raise ModelSuiteError(
                f"{label} replayed {field} differs from metadata"
            )
        return
    if expected is None:
        if observed is not None:
            raise ModelSuiteError(
                f"{label} replayed {field} differs from metadata"
            )
        return
    if isinstance(expected, (int, np.integer)):
        if (
            isinstance(observed, (bool, np.bool_))
            or not isinstance(observed, (int, np.integer))
            or int(observed) != int(expected)
        ):
            raise ModelSuiteError(
                f"{label} replayed {field} differs from metadata"
            )
        return
    if isinstance(expected, (float, np.floating)):
        if isinstance(observed, (bool, np.bool_)):
            raise ModelSuiteError(
                f"{label} replayed {field} differs from metadata"
            )
        try:
            left, right = float(expected), float(cast(Any, observed))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelSuiteError(
                f"{label} replayed {field} is not numeric"
            ) from exc
        if (
            not np.isfinite(left)
            or not np.isfinite(right)
            or not np.isclose(
                left, right, rtol=0.0, atol=CALIBRATION_REPLAY_ATOL
            )
        ):
            raise ModelSuiteError(
                f"{label} replayed {field} differs from metadata"
            )
        return
    if type(observed) is not type(expected) or observed != expected:
        raise ModelSuiteError(
            f"{label} replayed {field} differs from metadata"
        )


def _recompute_event_definition(
    panel: pd.DataFrame,
    *,
    selected_sites: set[str],
    external: bool,
    label: str,
) -> tuple[dict[str, float], dict[str, object]]:
    """Recompute q90 thresholds and the frozen reference from panel outcomes."""
    required = {"DATE", "site_id", "WTEMP"}
    missing = required - set(panel)
    if missing:
        raise ModelSuiteError(
            f"{label} canonical development panel lacks: {sorted(missing)}"
        )
    working = panel[["DATE", "site_id", "WTEMP"]].copy()
    working["DATE"] = pd.to_datetime(working["DATE"], errors="coerce")
    working["site_id"] = working["site_id"].astype(str)
    working["WTEMP"] = pd.to_numeric(working["WTEMP"], errors="coerce")
    if (
        working["DATE"].isna().any()
        or working["site_id"].eq("").any()
        or working.duplicated(["site_id", "DATE"]).any()
    ):
        raise ModelSuiteError(
            f"{label} canonical development panel keys are invalid"
        )
    panel_sites = set(working["site_id"])
    if panel_sites != selected_sites:
        raise ModelSuiteError(
            f"{label} development prediction station registry differs from the panel"
        )
    finite = np.isfinite(working["WTEMP"].to_numpy(float))
    train = working[
        working["DATE"].between(*map(pd.Timestamp, C.SPLIT.train)) & finite
    ].copy()
    if train.empty:
        raise ModelSuiteError(
            f"{label} canonical development training interval has no outcomes"
        )
    if external:
        threshold = float(train["WTEMP"].quantile(0.90))
        thresholds = {"__pooled__": threshold}
    else:
        thresholds = {
            str(site): float(group["WTEMP"].quantile(0.90))
            for site, group in train.groupby("site_id", sort=True)
        }
        if set(thresholds) != panel_sites:
            raise ModelSuiteError(
                f"{label} one or more stations lack finite training outcomes"
            )
    if not thresholds or not np.isfinite(
        np.asarray(list(thresholds.values()), dtype=float)
    ).all():
        raise ModelSuiteError(f"{label} replayed event thresholds are invalid")
    try:
        event_reference = P.fit_frozen_seasonal_event_reference(
            working,
            thresholds,
            pooled=external,
            fit_interval=(C.SPLIT.train[0], C.SPLIT.calib[1]),
        )
    except ValueError as exc:
        raise ModelSuiteError(
            f"{label} frozen seasonal event reference cannot be replayed"
        ) from exc
    return thresholds, event_reference


def _validate_development_truth_against_frozen_panel(
    panel: pd.DataFrame,
    prediction_truth: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Bind every selected development target back to canonical panel truth.

    ``panel`` must already use the stable ``site_no`` identifiers produced by
    :class:`FrozenPanelSpec`.  Forecasts at different horizons can share a
    target date, so the comparison registry is deliberately the outcome key
    ``(site_id, target_date)`` rather than the issue-time forecast key.
    """
    panel_required = {"DATE", "site_id", "WTEMP"}
    prediction_required = {"site_id", "target_date", "y_true"}
    panel_missing = panel_required - set(panel)
    prediction_missing = prediction_required - set(prediction_truth)
    if panel_missing:
        raise ModelSuiteError(
            f"{label} canonical development panel lacks truth columns: "
            f"{sorted(panel_missing)}"
        )
    if prediction_missing:
        raise ModelSuiteError(
            f"{label} development prediction lacks truth columns: "
            f"{sorted(prediction_missing)}"
        )

    canonical = panel[["DATE", "site_id", "WTEMP"]].copy()
    panel_site_missing = canonical["site_id"].isna().any()
    canonical["DATE"] = pd.to_datetime(canonical["DATE"], errors="coerce")
    canonical["site_id"] = canonical["site_id"].astype(str)
    canonical["WTEMP"] = pd.to_numeric(canonical["WTEMP"], errors="coerce")
    if (
        canonical.empty
        or panel_site_missing
        or canonical["DATE"].isna().any()
        or not canonical["DATE"].eq(canonical["DATE"].dt.normalize()).all()
        or canonical["site_id"].eq("").any()
        or canonical["site_id"].str.strip().ne(canonical["site_id"]).any()
        or canonical.duplicated(["site_id", "DATE"]).any()
    ):
        raise ModelSuiteError(
            f"{label} canonical development panel truth keys are invalid or duplicated"
        )

    observed = prediction_truth[["site_id", "target_date", "y_true"]].copy()
    prediction_site_missing = observed["site_id"].isna().any()
    observed["site_id"] = observed["site_id"].astype(str)
    observed["target_date"] = pd.to_datetime(
        observed["target_date"], errors="coerce"
    )
    observed["y_true"] = pd.to_numeric(observed["y_true"], errors="coerce")
    if (
        observed.empty
        or prediction_site_missing
        or observed["target_date"].isna().any()
        or not observed["target_date"].eq(
            observed["target_date"].dt.normalize()
        ).all()
        or observed["site_id"].eq("").any()
        or observed["site_id"].str.strip().ne(observed["site_id"]).any()
        or not np.isfinite(observed["y_true"].to_numpy(float)).all()
    ):
        raise ModelSuiteError(
            f"{label} development prediction truth registry is invalid"
        )

    truth_spread = observed.groupby(
        ["site_id", "target_date"], sort=True, dropna=False
    )["y_true"].agg(["min", "max"])
    if truth_spread.empty or not truth_spread["min"].eq(
        truth_spread["max"]
    ).all():
        raise ModelSuiteError(
            f"{label} development predictions disagree on an outcome key"
        )
    unique_truth = (
        observed.drop_duplicates(["site_id", "target_date"])
        .rename(columns={"target_date": "DATE", "y_true": "prediction_y_true"})
        .sort_values(["site_id", "DATE"], kind="mergesort")
        .reset_index(drop=True)
    )
    matched = unique_truth.merge(
        canonical.rename(columns={"WTEMP": "panel_y_true"}),
        on=["site_id", "DATE"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not matched["_merge"].eq("both").all():
        raise ModelSuiteError(
            f"{label} development prediction truth key is absent from frozen panel"
        )
    panel_truth = matched["panel_y_true"].to_numpy(float)
    prediction_values = matched["prediction_y_true"].to_numpy(float)
    if not np.isfinite(panel_truth).all():
        raise ModelSuiteError(
            f"{label} frozen panel has a missing or non-finite selected truth"
        )
    # Stage 09 canonicalises prediction targets to model precision (float32)
    # before serialising them as float64.  Compare in that same precision so
    # a legitimate round trip is accepted while any model-visible change is
    # rejected exactly.
    with np.errstate(over="ignore", invalid="ignore"):
        panel_model_precision = np.asarray(panel_truth, dtype=np.float32)
        prediction_model_precision = np.asarray(
            prediction_values, dtype=np.float32
        )
    if (
        not np.isfinite(panel_model_precision).all()
        or not np.isfinite(prediction_model_precision).all()
    ):
        raise ModelSuiteError(
            f"{label} development truth is not finite at model precision"
        )
    if not np.array_equal(panel_model_precision, prediction_model_precision):
        raise ModelSuiteError(
            f"{label} development prediction y_true differs from frozen panel WTEMP"
        )


def _recompute_event_definition_from_frozen_panel(
    root: str | Path,
    metadata: Mapping[str, Any],
    *,
    selected_sites: set[str],
    external: bool,
    label: str,
    prediction_truth: pd.DataFrame | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    """Load only the canonical frozen panel and replay the event definition."""
    from .evidence import EvidenceError, FrozenPanelSpec

    root_path = Path(root).resolve()
    spec_path = (root_path / "data_usgs" / "frozen_panel_v1.json").resolve()
    try:
        spec = FrozenPanelSpec.load(spec_path)
    except EvidenceError as exc:
        raise ModelSuiteError(
            f"{label} canonical frozen development panel spec is invalid"
        ) from exc
    expected_panel = (root_path / "data_usgs" / "panel_usgs_120v2.parquet").resolve()
    expected_registry = (root_path / "data_usgs" / "station_registry_v1.csv").resolve()
    if spec.panel_path != expected_panel or spec.registry_path != expected_registry:
        raise ModelSuiteError(
            f"{label} frozen panel spec resolves non-canonical artifacts"
        )
    panel_sha256 = metadata.get("panel_sha256")
    registry_sha256 = metadata.get("registry_sha256")
    if (
        not _is_sha256(panel_sha256)
        or not _is_sha256(registry_sha256)
        or spec.document.get("panel", {}).get("sha256") != panel_sha256
        or spec.document.get("station_registry", {}).get("sha256")
        != registry_sha256
    ):
        raise ModelSuiteError(
            f"{label} metadata is bound to another development panel/registry"
        )
    try:
        panel = spec.load_panel(stable_site_ids=True)
    except EvidenceError as exc:
        raise ModelSuiteError(
            f"{label} canonical frozen development panel is invalid"
        ) from exc
    if prediction_truth is not None:
        _validate_development_truth_against_frozen_panel(
            panel, prediction_truth, label=label
        )
    return _recompute_event_definition(
        panel,
        selected_sites=selected_sites,
        external=external,
        label=label,
    )


def _validate_event_calibrator_metadata(
    value: object, *, label: str
) -> Mapping[str, Any]:
    expected_horizons = {str(int(horizon)) for horizon in C.HORIZONS}
    if not isinstance(value, Mapping) or set(value) != expected_horizons:
        raise ModelSuiteError(
            f"{label} event calibrator horizon registry is not exact"
        )
    for horizon in sorted(expected_horizons, key=int):
        calibrator = value[horizon]
        if (
            not isinstance(calibrator, Mapping)
            or set(calibrator) != {"intercept", "slope", "constant"}
        ):
            raise ModelSuiteError(
                f"{label} h{horizon} event calibrator schema is not exact"
            )
        if isinstance(calibrator["intercept"], (bool, np.bool_)) or isinstance(
            calibrator["slope"], (bool, np.bool_)
        ):
            raise ModelSuiteError(
                f"{label} h{horizon} event calibrator is nonnumeric"
            )
        try:
            intercept = float(calibrator["intercept"])
            slope = float(calibrator["slope"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelSuiteError(
                f"{label} h{horizon} event calibrator is nonnumeric"
            ) from exc
        if not np.isfinite(intercept) or not np.isfinite(slope):
            raise ModelSuiteError(
                f"{label} h{horizon} event calibrator is non-finite"
            )
        constant = calibrator["constant"]
        if constant is None:
            continue
        if isinstance(constant, (bool, np.bool_)):
            raise ModelSuiteError(
                f"{label} h{horizon} constant calibrator is malformed"
            )
        try:
            constant_value = float(constant)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelSuiteError(
                f"{label} h{horizon} constant calibrator is malformed"
            ) from exc
        expected_intercept = float(P.logit(np.asarray([constant_value]))[0])
        if (
            not np.isfinite(constant_value)
            or not 0.0 < constant_value < 1.0
            or slope != 0.0
            or not np.isclose(
                intercept,
                expected_intercept,
                rtol=0.0,
                atol=CALIBRATION_REPLAY_ATOL,
            )
        ):
            raise ModelSuiteError(
                f"{label} h{horizon} constant calibrator invariant failed"
            )
    return value


def _validate_development_split_registry(
    selected: pd.DataFrame, *, label: str
) -> None:
    """Require the exact frozen val/calib/test forecast interval registry."""
    expected_splits = {"val", "calib", "test"}
    splits = set(selected["split"].astype(str))
    if splits != expected_splits:
        raise ModelSuiteError(
            f"{label} development split registry must be exactly val/calib/test"
        )
    horizons = pd.to_numeric(selected["horizon"], errors="coerce")
    if (
        horizons.isna().any()
        or set(horizons.astype(int)) != set(C.HORIZONS)
        or not np.equal(horizons.to_numpy(float), horizons.astype(int)).all()
    ):
        raise ModelSuiteError(
            f"{label} development horizon registry is not exact"
        )
    issue_date = pd.to_datetime(selected["issue_date"], errors="coerce")
    target_date = pd.to_datetime(selected["target_date"], errors="coerce")
    if (
        issue_date.isna().any()
        or target_date.isna().any()
        or not issue_date.eq(issue_date.dt.normalize()).all()
        or not target_date.eq(target_date.dt.normalize()).all()
    ):
        raise ModelSuiteError(
            f"{label} development forecast dates are invalid"
        )
    for split_name in sorted(expected_splits):
        mask = selected["split"].astype(str).eq(split_name)
        lower, upper = (
            pd.Timestamp(value)
            for value in getattr(C.SPLIT, split_name)
        )
        if (
            not mask.any()
            or not issue_date[mask].between(lower, upper).all()
            or not target_date[mask].between(lower, upper).all()
        ):
            raise ModelSuiteError(
                f"{label} {split_name} rows escape the frozen split interval"
            )
    calibration_target = target_date[selected["split"].astype(str).eq("calib")]
    if (calibration_target > pd.Timestamp(C.SPLIT.calib[1])).any():
        raise ModelSuiteError(
            f"{label} calibration rows use post-2018 targets"
        )


def validate_development_calibrated_head_gate(
    root: str | Path,
    metadata: Mapping[str, Any],
    *,
    label: str,
    external: bool,
) -> dict[str, Any]:
    """Replay CQR/Platt from frozen development evidence, then audit heads."""
    _validate_cqr_metadata(metadata, label=label)
    _validate_calibration_fit_metadata(
        metadata, label=label, external=external
    )
    binding = metadata.get("development_prediction")
    snapshot = _read_development_prediction_snapshot(root, binding, label=label)
    if not isinstance(binding, Mapping):  # authoritative validator above
        raise ModelSuiteError(f"{label} development prediction binding is malformed")
    selection = binding.get("selection")
    if not isinstance(selection, Mapping):
        raise ModelSuiteError(f"{label} development prediction selection is malformed")
    expected_run = {
        "run_id": metadata.get("run_id"),
        "panel_sha256": metadata.get("panel_sha256"),
        "registry_sha256": metadata.get("registry_sha256"),
        "config_sha256": metadata.get("config_sha256"),
        "source_sha256": metadata.get("source_sha256"),
        "runtime_sha256": metadata.get("runtime_sha256"),
        "input_closure_sha256": metadata.get("input_closure_sha256"),
        "schema_version": RUN_SCHEMA_VERSION,
    }
    if snapshot.sidecar.get("run") != expected_run:
        raise ModelSuiteError(
            f"{label} prediction sidecar belongs to another model run"
        )
    try:
        R.validate_predictions(snapshot.frame)
    except Exception as exc:
        raise ModelSuiteError(f"{label} development predictions are invalid") from exc
    declared_seeds = tuple(int(value) for value in selection.get("seeds", ()))
    seeds = set(declared_seeds)
    if not seeds or len(seeds) != len(declared_seeds):
        raise ModelSuiteError(f"{label} CQR gate seed registry is empty or duplicated")
    member_count = metadata.get("member_count")
    if (
        isinstance(member_count, (bool, np.bool_))
        or not isinstance(member_count, (int, np.integer))
        or int(member_count) != len(declared_seeds)
    ):
        raise ModelSuiteError(
            f"{label} prediction seed registry differs from bundle members"
        )
    selected = snapshot.selected.copy()
    if len(selected) != int(binding.get("rows", -1)) or selected.empty:
        raise ModelSuiteError(f"{label} CQR gate selection row count changed")
    _validate_development_split_registry(selected, label=label)
    selected_sites = set(selected["site_id"].astype(str))
    observed_combinations = set(
        selected[["site_id", "horizon", "split"]]
        .assign(
            site_id=lambda value: value["site_id"].astype(str),
            horizon=lambda value: value["horizon"].astype(int),
            split=lambda value: value["split"].astype(str),
        )
        .itertuples(index=False, name=None)
    )
    expected_combinations = {
        (site, int(horizon), split)
        for site in selected_sites
        for horizon in C.HORIZONS
        for split in ("val", "calib", "test")
    }
    if observed_combinations != expected_combinations:
        raise ModelSuiteError(
            f"{label} development site×horizon×split registry is incomplete"
        )
    group_columns = [
        "model", "scope", "feature_set", "site_id", "horizon", "split",
        "issue_date", "target_date",
    ]
    seed_key = [*group_columns, "seed"]
    if selected.duplicated(seed_key).any():
        raise ModelSuiteError(
            f"{label} CQR gate has a duplicate forecast-key×seed row"
        )
    coverage = selected.groupby(group_columns, dropna=False, sort=True)["seed"].agg(
        ["size", "nunique"]
    )
    expected_members = len(seeds)
    if (
        coverage.empty
        or not coverage["size"].eq(expected_members).all()
        or not coverage["nunique"].eq(expected_members).all()
    ):
        raise ModelSuiteError(
            f"{label} CQR gate does not contain every declared seed exactly once "
            "for every forecast key"
        )
    if (
        selected["scope"].astype(str).nunique(dropna=False) != 1
        or selected["feature_set"].astype(str).nunique(dropna=False) != 1
    ):
        raise ModelSuiteError(
            f"{label} calibration selection mixes scopes or feature sets"
        )
    if set(selected["seed"].astype(int)) != seeds:
        raise ModelSuiteError(f"{label} CQR gate seed registry is incomplete")
    truth_coverage = selected.groupby(
        group_columns, dropna=False, sort=True
    )["y_true"].agg(["min", "max"])
    if truth_coverage.empty or not truth_coverage["min"].eq(
        truth_coverage["max"]
    ).all():
        raise ModelSuiteError(
            f"{label} ensemble members disagree on development outcomes"
        )
    ensemble = selected.groupby(
        group_columns, as_index=False, dropna=False, sort=True
    ).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        q05=("q05", "mean"),
        q50=("q50", "mean"),
        q95=("q95", "mean"),
        p_exceed=("p_exceed", "mean"),
    )
    replay_values = ensemble[
        ["y_true", "y_pred", "q05", "q50", "q95", "p_exceed"]
    ].to_numpy(dtype=float)
    heads = ensemble[["q05", "q50", "q95"]].to_numpy(dtype=float)
    if (
        not len(heads)
        or not np.isfinite(replay_values).all()
        or (heads[:, 0] > heads[:, 1]).any()
        or (heads[:, 1] > heads[:, 2]).any()
        or (heads[:, 0] >= heads[:, 2]).any()
        or (ensemble["p_exceed"].to_numpy(float) < 0.0).any()
        or (ensemble["p_exceed"].to_numpy(float) > 1.0).any()
    ):
        raise ModelSuiteError(f"{label} nominal development heads are invalid")

    if external:
        expected_threshold_estimator = {
            "method": "pooled_training_empirical_quantile_v1",
            "quantile": 0.90,
            "pool_weighting": ROW_EQUAL_WEIGHTING,
            "station_balanced": False,
        }
        if metadata.get("event_threshold_estimator") != expected_threshold_estimator:
            raise ModelSuiteError(
                f"{label} pooled threshold estimator contract changed"
            )
    thresholds, event_reference = _recompute_event_definition_from_frozen_panel(
        root,
        metadata,
        selected_sites=selected_sites,
        external=external,
        label=label,
        prediction_truth=ensemble[["site_id", "target_date", "y_true"]],
    )
    _strict_calibration_replay_match(
        thresholds,
        metadata.get("event_thresholds"),
        label=label,
        field="event_thresholds",
    )
    _strict_calibration_replay_match(
        event_reference,
        metadata.get("event_reference_climatology"),
        label=label,
        field="event_reference_climatology",
    )

    calibration = ensemble[ensemble["split"].astype(str).eq("calib")].copy()
    if calibration.empty:
        raise ModelSuiteError(f"{label} calibration replay is empty")
    y_true = calibration["y_true"].to_numpy(float)
    finite = np.isfinite(y_true)
    if not finite.any():
        raise ModelSuiteError(f"{label} calibration replay lacks finite observed targets")
    if not finite.all():
        calibration = calibration.loc[finite].copy()
    if (
        pd.to_datetime(calibration["target_date"])
        > pd.Timestamp(C.SPLIT.calib[1])
    ).any():
        raise ModelSuiteError(
            f"{label} calibration replay is empty or uses post-2018 targets"
        )
    if external:
        threshold = thresholds["__pooled__"]
        calibration["event"] = (
            calibration["y_true"].to_numpy(float) > threshold
        ).astype(int)
        cqr_calibration = calibration.copy()
        cqr_calibration["site_id"] = "__pooled__"
    else:
        threshold_values = calibration["site_id"].astype(str).map(thresholds)
        if threshold_values.isna().any():
            raise ModelSuiteError(
                f"{label} calibration contains a site without a replayed threshold"
            )
        calibration["event"] = (
            calibration["y_true"].to_numpy(float)
            > threshold_values.to_numpy(float)
        ).astype(int)
        cqr_calibration = calibration
    try:
        replayed_offsets, replayed_offset_audit = cqr_offsets_with_audit(
            cqr_calibration,
            alpha=0.10,
            purge_boundary=True,
        )
    except (CQRContractError, ValueError) as exc:
        raise ModelSuiteError(f"{label} CQR offsets cannot be replayed") from exc
    serialised_replayed_offsets = serialise_offsets(replayed_offsets)
    expected_offset_keys = (
        {f"__pooled__|{int(horizon)}" for horizon in C.HORIZONS}
        if external
        else {
            f"{site}|{int(horizon)}"
            for site in selected_sites
            for horizon in C.HORIZONS
        }
    )
    if set(serialised_replayed_offsets) != expected_offset_keys:
        raise ModelSuiteError(
            f"{label} replayed CQR offset registry is incomplete"
        )
    _strict_calibration_replay_match(
        serialised_replayed_offsets,
        metadata.get("conformal_offsets"),
        label=label,
        field="conformal_offsets",
    )
    _strict_calibration_replay_match(
        replayed_offset_audit,
        metadata.get("conformal_offset_audit"),
        label=label,
        field="conformal_offset_audit",
    )

    declared_calibrators = _validate_event_calibrator_metadata(
        metadata.get("event_calibrators"), label=label
    )
    try:
        replayed_calibrators = P.fit_horizon_calibrators(
            calibration,
            probability_col="p_exceed",
            outcome_col="event",
            min_samples=100,
            weighting=STATION_EQUAL_WEIGHTING,
        )
    except ValueError as exc:
        raise ModelSuiteError(
            f"{label} event calibrators cannot be replayed"
        ) from exc
    if set(replayed_calibrators) != set(C.HORIZONS):
        raise ModelSuiteError(
            f"{label} replayed event calibrator horizon registry is incomplete"
        )
    serialised_replayed_calibrators = {
        str(int(horizon)): calibrator.as_dict()
        for horizon, calibrator in sorted(replayed_calibrators.items())
    }
    _strict_calibration_replay_match(
        serialised_replayed_calibrators,
        declared_calibrators,
        label=label,
        field="event_calibrators",
    )

    offsets = metadata["conformal_offsets"]
    assert isinstance(offsets, Mapping)
    keys = [
        (
            f"__pooled__|{int(horizon)}"
            if external else f"{str(site)}|{int(horizon)}"
        )
        for site, horizon in ensemble[["site_id", "horizon"]].itertuples(
            index=False, name=None
        )
    ]
    missing = sorted(set(keys) - set(offsets))
    if missing:
        raise ModelSuiteError(f"{label} CQR gate lacks offsets: {missing[:5]}")
    delta = np.asarray([float(offsets[key]) for key in keys], dtype=float)
    calibrated = heads.copy()
    calibrated[:, 0] -= delta
    calibrated[:, 2] += delta
    nominal_width = heads[:, 2] - heads[:, 0]
    calibrated_width = calibrated[:, 2] - calibrated[:, 0]
    if (
        not np.isfinite(calibrated).all()
        or (calibrated[:, 0] > calibrated[:, 1]).any()
        or (calibrated[:, 1] > calibrated[:, 2]).any()
        or (calibrated_width <= 0.0).any()
        or (calibrated_width < nominal_width).any()
    ):
        raise ModelSuiteError(f"{label} final calibrated development heads are unsafe")
    digest_frame = ensemble[
        ["site_id", "horizon", "split", "issue_date", "target_date"]
    ].copy()
    digest_frame[["q05", "q50", "q95"]] = calibrated
    digest_columns = (
        "site_id", "horizon", "split", "issue_date", "target_date",
        "q05", "q50", "q95",
    )
    result = {
        "format": DEVELOPMENT_CALIBRATED_HEAD_GATE_FORMAT,
        "status": "PASS_NONNEGATIVE_CQR_WIDENS_ONLY",
        "rows": int(len(ensemble)),
        "offset_min": float(delta.min()),
        "offset_max": float(delta.max()),
        "nominal_interval_min_width": float(nominal_width.min()),
        "calibrated_interval_min_width": float(calibrated_width.min()),
        "minimum_width_increase": float((calibrated_width - nominal_width).min()),
        "calibrated_heads_sha256": canonical_frame_digest(
            digest_frame, digest_columns
        ),
        "all_final_heads_finite_ordered_nonempty": True,
        "every_interval_weakly_widened": True,
    }
    _assert_prediction_snapshot_unchanged(snapshot, label=label)
    return result


def sequence_bundle_metadata(
    *,
    run_id: str,
    architecture_class: str,
    architecture_kwargs: Mapping[str, Any],
    train_config: Any,
    wd: Any,
    climatology: Any,
    imputer: D.Imputer,
    thresholds: Mapping[str, float],
    event_reference_climatology: Mapping[str, object],
    conformal_offsets: Mapping[tuple[str, int], object],
    conformal_offset_audit: Mapping[str, Any],
    event_calibrators: Mapping[int, Any],
    source_sha256: str,
    panel_sha256: str,
    registry_sha256: str,
    config_sha256: str,
    runtime_sha256: str,
    input_closure_sha256: str,
    training_device: str,
    development_prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete metadata required by a weights-only sequence bundle."""
    config_value = asdict(train_config) if hasattr(train_config, "__dataclass_fields__") \
        else dict(train_config)
    if str(training_device) != "cpu":
        raise ModelSuiteError("formal sequence bundles must be trained on CPU")
    station_agnostic = bool(architecture_kwargs.get("station_agnostic", False))
    return {
        "run_id": str(run_id),
        "architecture": {
            "class": str(architecture_class),
            "kwargs": dict(architecture_kwargs),
            "train_config": config_value,
        },
        "feature_order": list(wd.var_names),
        "horizons": [int(value) for value in wd.horizons],
        "station_to_index": {str(site): index for index, site in enumerate(C.STATIONS)},
        "preprocessing": serialise_preprocessing(wd, climatology, imputer),
        "event_thresholds": {str(site): float(value)
                             for site, value in sorted(thresholds.items())},
        "event_reference_climatology": dict(event_reference_climatology),
        "event_calibrators": {
            str(horizon): (calibrator.as_dict() if hasattr(calibrator, "as_dict")
                           else dict(calibrator))
            for horizon, calibrator in sorted(event_calibrators.items())
        },
        "conformal_offsets": serialise_offsets(conformal_offsets),
        "conformal_policy": cqr_policy_contract(),
        "conformal_offset_audit": dict(conformal_offset_audit),
        "calibration_fit_contract": route_a_calibration_fit_contract(
            external=station_agnostic
        ),
        "source_sha256": str(source_sha256),
        "panel_sha256": str(panel_sha256),
        "registry_sha256": str(registry_sha256),
        "config_sha256": str(config_sha256),
        "runtime_sha256": str(runtime_sha256),
        "input_closure_sha256": str(input_closure_sha256),
        "training_device": "cpu",
        "output_head_schema": neural_output_head_schema(),
        "development_prediction": dict(development_prediction),
    }


def builtin_entry(model_id: str, raw_feature_order: Sequence[str]) -> dict[str, Any]:
    if model_id not in BUILTIN_MODELS:
        raise ModelSuiteError(f"not a frozen builtin: {model_id}")
    return {
        "model_id": model_id,
        "executor": "builtin",
        "raw_feature_order": list(raw_feature_order),
    }


def torch_entry(
    root: str | Path,
    *,
    model_id: str,
    executor: str,
    directory: str | Path,
    member_count: int,
    raw_feature_order: Sequence[str],
    intervention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if executor not in {"thermoroute_bundle", "lstm_bundle"}:
        raise ModelSuiteError(f"unsafe torch executor: {executor}")
    _, metadata = load_inference_bundle(directory, expected_member_count=member_count)
    if tuple(metadata.get("feature_order", ())) != tuple(raw_feature_order):
        raise ModelSuiteError(f"{model_id} bundle has another feature schema")
    entry = {
        "model_id": model_id,
        "executor": executor,
        "raw_feature_order": list(raw_feature_order),
        "member_count": int(member_count),
        "artifact": directory_binding(root, directory),
    }
    if intervention is not None:
        entry["intervention"] = dict(intervention)
    return entry


def lightgbm_entry(
    root: str | Path,
    *,
    manifest: str | Path,
    raw_feature_order: Sequence[str],
) -> dict[str, Any]:
    _, metadata = load_lightgbm_bundle(manifest)
    if tuple(metadata.get("raw_feature_order", ())) != tuple(raw_feature_order):
        raise ModelSuiteError("LightGBM bundle has another feature schema")
    return {
        "model_id": "LightGBM",
        "executor": "lightgbm_bundle",
        "raw_feature_order": list(raw_feature_order),
        "member_count": int(metadata["member_count"]),
        "artifact": file_binding(root, manifest),
    }


def write_component_pointer(
    destination: str | Path,
    *,
    run_id: str,
    cohort: str,
    entries: Sequence[Mapping[str, Any]],
    raw_feature_order: Sequence[str],
    development_contract: Mapping[str, Any] | None = None,
    development_prediction_artifact: Mapping[str, Any] | None = None,
    publication_guard: Callable[[], object] | None = None,
) -> Path:
    """Publish a component pointer only after every referenced artifact verifies."""
    if cohort not in {"temporal_stage9", "temporal_lstm", "external"}:
        raise ModelSuiteError(f"unknown component cohort: {cohort}")
    ids = [str(entry.get("model_id")) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ModelSuiteError("component pointer contains duplicate model ids")
    document: dict[str, Any] = {
        "format": COMPONENT_POINTER_FORMAT,
        "status": "COMPLETE",
        "training_device": "cpu",
        "run_id": str(run_id),
        "cohort": cohort,
        "raw_feature_order": list(raw_feature_order),
        "models": [dict(entry) for entry in entries],
    }
    if development_contract is not None:
        document["development_contract"] = dict(development_contract)
    if development_prediction_artifact is not None:
        document["development_prediction_artifact"] = dict(
            development_prediction_artifact
        )
    atomic_write_json(
        destination,
        document,
        publication_guard=publication_guard,
    )
    return Path(destination)


def load_component_pointer(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelSuiteError(f"cannot read component pointer: {path}") from exc
    if document.get("format") != COMPONENT_POINTER_FORMAT or document.get("status") != "COMPLETE":
        raise ModelSuiteError("component pointer is not complete")
    if document.get("training_device") != "cpu":
        raise ModelSuiteError("formal component pointer is not CPU-trained")
    entries = document.get("models")
    if not isinstance(entries, list) or not entries:
        raise ModelSuiteError("component pointer has no models")
    ids = [str(value.get("model_id")) for value in entries if isinstance(value, Mapping)]
    if len(ids) != len(entries) or len(ids) != len(set(ids)):
        raise ModelSuiteError("component model registry is malformed")
    return document


def _validate_stage09_suite_alignment(
    stage9: Mapping[str, Any],
    temporal_entries: Sequence[Mapping[str, Any]],
    development_contract: Mapping[str, Any],
) -> None:
    """Require the frozen suite to contain the receipt-admitted Stage-9 closure.

    Validating the receipt and the suite entries independently is insufficient:
    a concurrent Stage-9 publication could otherwise pair entries read from one
    generation with a receipt read from the next.  Compare the complete Stage-9
    entries and development contract, not only their run ids.
    """
    expected_ids = {"ThermoRoute", "LightGBM", *MANDATORY_ABLATIONS}
    stage_entries = stage9.get("models")
    if not isinstance(stage_entries, list):
        raise ModelSuiteError("Stage-9 completion model registry is malformed")
    stage_by_id = {
        str(entry.get("model_id")): entry
        for entry in stage_entries
        if isinstance(entry, Mapping)
    }
    suite_by_id = {
        str(entry.get("model_id")): entry
        for entry in temporal_entries
        if isinstance(entry, Mapping)
    }
    if (
        set(stage_by_id) != expected_ids
        or not expected_ids <= set(suite_by_id)
        or len(stage_by_id) != len(stage_entries)
        or stage9.get("development_contract") != dict(development_contract)
        or any(
            dict(stage_by_id[model_id]) != dict(suite_by_id[model_id])
            for model_id in expected_ids
        )
    ):
        raise ModelSuiteError(
            "frozen suite Stage-9 entries differ from its completion receipt"
        )


def _stage09_formal_configuration(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    resolved = run_manifest.get("resolved_config")
    if (
        not isinstance(resolved, dict)
        or set(resolved) != STAGE9_FORMAL_CONFIG_FIELDS
    ):
        raise ModelSuiteError("Stage-9 run manifest lacks resolved configuration")
    bridge = resolved["development_predictor_bridge"]
    numerical_policy = resolved["formal_numerical_policy"]
    input_closure_sha256 = resolved["input_closure_sha256"]
    input_closure_file_count = resolved["input_closure_file_count"]
    if (
        resolved["stage"] != "09_usgs_experiment"
        or resolved["protocol"] != STAGE9_FORMAL_PROTOCOL
        or resolved["panel"] != "panel_usgs_120v2.parquet"
        or resolved["station_registry"] != "station_registry_v1.csv"
        or resolved["variables"] != list(STAGE9_USGS_VARIABLES)
        or resolved["horizons"] != list(C.HORIZONS)
        or type(resolved["context_length"]) is not int
        or resolved["context_length"] != C.CONTEXT_LENGTH
        or type(resolved["seeds"]) is not int
        or resolved["seeds"] != len(C.USGS_SEEDS)
        or resolved["time_split"] != {
            name: list(interval) for name, interval in C.SPLIT.as_dict().items()
        }
        or resolved["train_config"] != STAGE9_FORMAL_TRAIN_CONFIG
        or resolved["thermoroute_seeds"] != list(C.USGS_SEEDS)
        or resolved["lightgbm_seeds"] != list(C.USGS_SEEDS)
        or resolved["ablation_seeds"] != list(STAGE9_ABLATION_SEEDS)
        or type(resolved["delta_scale"]) is not float
        or resolved["delta_scale"] != C.DELTA_SCALE
        or resolved["station_sampling"] != "balanced"
        or resolved["selection_metric"] != "station_macro"
        or resolved["ablations"] is not True
        or resolved["air2stream"] is not False
        or resolved["device"] != "cpu"
        or resolved["training_device"] != "cpu"
        or resolved["execution_role"] != "route_a_formal_candidate"
        or not isinstance(bridge, Mapping)
        or set(bridge) != {"path", "sha256"}
        or type(resolved["eval_batch_size"]) is not int
        or resolved["eval_batch_size"] < 1
        or resolved["lightgbm_validation_grid"] != [
            dict(params) for params in STAGE9_LIGHTGBM_VALIDATION_GRID
        ]
        or resolved["event_reference_fit_interval"]
        != ["2006-01-01", "2018-12-31"]
        or not isinstance(numerical_policy, Mapping)
        or not numerical_policy
        or not isinstance(input_closure_sha256, str)
        or len(input_closure_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in input_closure_sha256
        )
        or type(input_closure_file_count) is not int
        or input_closure_file_count < 1
    ):
        raise ModelSuiteError(
            "Stage-9 run manifest has malformed formal configuration"
        )
    return json.loads(json.dumps(resolved, sort_keys=True))


def _validated_file_binding(
    root: Path, value: object, *, label: str,
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ModelSuiteError(f"{label} binding is malformed")
    raw = value.get("path")
    if not isinstance(raw, str) or Path(raw).is_absolute():
        raise ModelSuiteError(f"{label} binding is malformed")
    root = root.resolve()
    path = Path(os.path.abspath(root / raw))
    if path != root and root not in path.parents:
        raise ModelSuiteError(f"{label} path escapes repository")
    current = path
    while current != root:
        if current.is_symlink():
            raise ModelSuiteError(f"{label} path uses a symlink")
        current = current.parent
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ModelSuiteError(f"{label} artifact is absent") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ModelSuiteError(f"{label} is not a single-link regular file")
    if dict(value) != file_binding(root, path):
        raise ModelSuiteError(f"{label} checksum or canonical path changed")
    return path


def _canonical_completion_receipt_path(
    root: Path,
    receipt_path: str | Path,
    *,
    relative: str,
    label: str,
    document_supplied: bool,
) -> Path:
    """Preserve a completion receipt's lexical identity before dereferencing."""
    raw = Path(receipt_path)
    if not raw.is_absolute():
        raw = root / raw
    lexical = Path(os.path.abspath(raw))
    expected = root / relative
    if lexical != expected:
        raise ModelSuiteError(f"{label} is not at its exact canonical path")
    current = lexical
    while current != root:
        if current.is_symlink():
            raise ModelSuiteError(f"{label} path uses a symlink")
        current = current.parent
    if not document_supplied or lexical.exists() or lexical.is_symlink():
        try:
            metadata = lexical.lstat()
        except OSError as exc:
            raise ModelSuiteError(f"{label} is absent or invalid") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ModelSuiteError(
                f"{label} is not a single-link regular file"
            )
    return lexical


def _validated_content_addressed_run_identity(
    run_manifest: Mapping[str, Any], *, label: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Require a run manifest identity to be internally content-addressed."""
    resolved = run_manifest.get("resolved_config")
    identity = run_manifest.get("identity")
    identity_keys = {
        "run_id", "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "input_closure_sha256",
        "schema_version",
    }
    digest_fields = (
        "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "input_closure_sha256",
    )
    if (
        not isinstance(resolved, Mapping)
        or not isinstance(identity, dict)
        or set(identity) != identity_keys
        or identity.get("schema_version") != RUN_SCHEMA_VERSION
        or not isinstance(identity.get("run_id"), str)
        or not identity["run_id"]
        or any(
            not isinstance(identity.get(field), str)
            or len(identity[field]) != 64
            or any(character not in "0123456789abcdef" for character in identity[field])
            for field in digest_fields
        )
        or identity.get("config_sha256") != sha256_json(resolved)
    ):
        raise ModelSuiteError(f"{label} run manifest identity is malformed")
    identity_parts = {
        "schema_version": identity["schema_version"],
        **{field: identity[field] for field in digest_fields},
    }
    if identity["run_id"] != sha256_json(identity_parts)[:20]:
        raise ModelSuiteError(f"{label} run id is not derived from its identity")
    return dict(identity), resolved


def _verified_development_input_closure(
    root: Path,
    *,
    identity: Mapping[str, Any],
    configuration: Mapping[str, Any],
    label: str,
    composed: bool,
) -> InputClosure:
    """Resolve the fixed bytes independently of a receipt's own declarations."""
    try:
        closure = resolve_development_input_closure(root)
        closure.assert_unchanged()
        expected_digest = (
            compose_input_closure_digest({
                "development": closure.binding_digest,
            })
            if composed else closure.binding_digest
        )
    except InputClosureError as exc:
        raise ModelSuiteError(
            f"{label} fixed development input closure cannot be replayed"
        ) from exc
    if (
        identity.get("input_closure_sha256") != expected_digest
        or configuration.get("input_closure_sha256") != expected_digest
        or configuration.get("input_closure_file_count")
        != len(closure.inventory)
    ):
        raise ModelSuiteError(
            f"{label} run identity differs from the fixed development input closure"
        )
    return closure


def _validated_stage09_run_identity(
    run_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Require the Stage-9 identity while preserving its public diagnostics."""
    return _validated_content_addressed_run_identity(
        run_manifest, label="Stage-9"
    )


def _load_formal_stage09_manifest(
    path: Path, *, root: Path, run_id: str,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], InputClosure
]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteError("Stage-9 receipt binds a malformed run manifest") from exc
    if not isinstance(manifest, dict):
        raise ModelSuiteError("Stage-9 receipt binds a malformed run manifest")
    identity, resolved_config = _validated_stage09_run_identity(manifest)
    if (
        identity["run_id"] != run_id
        or identity["source_sha256"] != source_tree_hash(root)
        or identity["config_sha256"] != sha256_json(resolved_config)
    ):
        raise ModelSuiteError(
            "Stage-9 completion receipt is stale for the run or current source"
        )
    configuration = _stage09_formal_configuration(manifest)
    if configuration["input_closure_sha256"] != identity[
        "input_closure_sha256"
    ]:
        raise ModelSuiteError(
            "Stage-9 configuration and run identity bind different input closures"
        )
    panel_path = root / "data_usgs" / "panel_usgs_120v2.parquet"
    registry_path = root / "data_usgs" / "station_registry_v1.csv"
    try:
        panel_sha256 = sha256_file(panel_path)
        registry_sha256 = sha256_file(registry_path)
    except OSError as exc:
        raise ModelSuiteError(
            "Stage-9 canonical panel or station registry is absent"
        ) from exc
    if (
        panel_sha256 != identity["panel_sha256"]
        or registry_sha256 != identity["registry_sha256"]
    ):
        raise ModelSuiteError(
            "Stage-9 identity differs from the canonical panel or station registry"
        )
    expected_bridge = development_predictor_bridge_binding(
        root,
        panel_sha256=panel_sha256,
        registry_sha256=registry_sha256,
    )
    if configuration["development_predictor_bridge"] != expected_bridge:
        raise ModelSuiteError(
            "Stage-9 configuration binds another development predictor bridge"
        )
    input_closure = _verified_development_input_closure(
        root,
        identity=identity,
        configuration=configuration,
        label="Stage-9",
        composed=False,
    )
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("evidence_role")
        != "prelabel_route_a_model_build_development_only"
        or provenance.get("training_device") != "cpu"
    ):
        raise ModelSuiteError("Stage-9 receipt lacks development-only provenance")
    return manifest, identity, configuration, input_closure


def _validate_stage09_prediction_outputs(
    predictions: Path,
    prediction_sidecar: Path,
    *,
    identity: Mapping[str, Any],
    run_id: str,
    configuration: Mapping[str, Any],
    root: Path,
) -> pd.DataFrame:
    try:
        prediction_meta = validate_artifact_sidecar(predictions)
    except ValueError as exc:
        raise ModelSuiteError("Stage-9 prediction sidecar is invalid") from exc
    if (
        sidecar_path(predictions).resolve() != prediction_sidecar
        or prediction_meta.get("kind") != "canonical_stage9_usgs_predictions"
        or prediction_meta.get("content_schema") != R.PREDICTION_SCHEMA_VERSION
        or prediction_meta.get("run") != identity
        or identity.get("run_id") != run_id
    ):
        raise ModelSuiteError("Stage-9 prediction binding differs from the run")
    try:
        frame = pd.read_parquet(predictions)
    except (OSError, ValueError, ImportError) as exc:
        raise ModelSuiteError("Stage-9 prediction parquet is unreadable") from exc
    registry_path = root / "data_usgs" / "station_registry_v1.csv"
    try:
        registry = pd.read_csv(registry_path, dtype={"site_no": "string"})
    except (
        OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError,
    ) as exc:
        raise ModelSuiteError("Stage-9 canonical station registry is unreadable") from exc
    if (
        sha256_file(registry_path) != identity.get("registry_sha256")
        or "site_no" not in registry
        or registry["site_no"].isna().any()
        or registry["site_no"].duplicated().any()
    ):
        raise ModelSuiteError("Stage-9 canonical station registry binding is invalid")
    expected_sites = tuple(registry["site_no"].astype(str).tolist())
    if not expected_sites or any(not site or site != site.strip() for site in expected_sites):
        raise ModelSuiteError("Stage-9 canonical station registry sites are malformed")
    return _validate_stage09_prediction_frame(
        frame,
        configuration=configuration,
        expected_sites=expected_sites,
    )


def _string_column_is_canonical(frame: pd.DataFrame, column: str) -> bool:
    values = frame[column].tolist()
    return bool(values) and all(
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        for value in values
    )


def _key_truth_registry(frame: pd.DataFrame) -> dict[tuple[Any, ...], float]:
    keys = list(STAGE9_POINT_KEY)
    return {
        tuple(row): float(truth)
        for row, truth in zip(
            frame[keys].itertuples(index=False, name=None),
            frame["y_true"].to_numpy(dtype=float),
            strict=True,
        )
    }


def _require_exact_seed_grid(
    frame: pd.DataFrame,
    model: str,
    expected_seeds: tuple[int, ...],
) -> None:
    selected = frame[frame["model"].eq(model)]
    if selected.empty or set(selected["seed"].tolist()) != set(expected_seeds):
        raise ModelSuiteError(
            f"Stage-9 {model} does not have the exact seed registry "
            f"{expected_seeds}"
        )
    counts = selected.groupby(
        list(STAGE9_POINT_KEY), observed=True, dropna=False
    ).size()
    if counts.empty or not counts.eq(len(expected_seeds)).all():
        raise ModelSuiteError(
            f"Stage-9 {model} does not contain every seed on every exact key"
        )


def _validate_stage09_prediction_frame(
    frame: pd.DataFrame,
    *,
    configuration: Mapping[str, Any],
    expected_sites: Sequence[str],
) -> pd.DataFrame:
    """Validate the serialized Stage-9 numerical truth without dtype coercion."""
    if list(frame.columns) != list(R.PRED_COLS) or frame.empty:
        raise ModelSuiteError("Stage-9 prediction schema is not exact")
    for column in ("model", "scope", "feature_set", "site_id", "split"):
        if not _string_column_is_canonical(frame, column):
            raise ModelSuiteError(
                f"Stage-9 prediction column {column} is not canonical text"
            )
    for column in ("seed", "horizon"):
        dtype = frame[column].dtype
        if (
            pd.api.types.is_bool_dtype(dtype)
            or not pd.api.types.is_integer_dtype(dtype)
            or frame[column].isna().any()
        ):
            raise ModelSuiteError(
                f"Stage-9 prediction column {column} must be a non-null integer dtype"
            )
    for column in ("issue_date", "target_date"):
        if str(frame[column].dtype) != "datetime64[ns]" or frame[column].isna().any():
            raise ModelSuiteError(
                f"Stage-9 prediction column {column} must be timezone-naive datetime64[ns]"
            )
        if not frame[column].dt.normalize().equals(frame[column]):
            raise ModelSuiteError(
                f"Stage-9 prediction column {column} must contain midnight dates only"
            )
    for column in ("y_true", "y_pred", "q05", "q50", "q95", "p_exceed"):
        if (
            pd.api.types.is_bool_dtype(frame[column].dtype)
            or not pd.api.types.is_float_dtype(frame[column].dtype)
        ):
            raise ModelSuiteError(
                f"Stage-9 prediction column {column} must use floating dtype"
            )
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ModelSuiteError("Stage-9 point predictions contain non-finite values")
    if not set(frame["split"].tolist()) <= STAGE9_KNOWN_SPLITS:
        raise ModelSuiteError("Stage-9 prediction contains an unknown split")
    if set(frame["horizon"].tolist()) != set(C.HORIZONS):
        raise ModelSuiteError("Stage-9 prediction horizons are not exact")

    air2stream = configuration.get("air2stream")
    if not isinstance(air2stream, bool):
        raise ModelSuiteError("Stage-9 Air2stream configuration is malformed")
    expected_models = set(STAGE9_PREDICTION_MODELS)
    if air2stream:
        expected_models.update(STAGE9_AIR2STREAM_MODELS)
    if set(frame["model"].tolist()) != expected_models:
        raise ModelSuiteError(
            "Stage-9 prediction model rows disagree with the frozen configuration"
        )

    time_split = configuration.get("time_split")
    if not isinstance(time_split, Mapping):
        raise ModelSuiteError("Stage-9 split configuration is malformed")
    expected_split = np.full(len(frame), "none", dtype=object)
    issue = frame["issue_date"]
    target = frame["target_date"]
    for split_name in ("train", "val", "calib", "test"):
        interval = time_split.get(split_name)
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ModelSuiteError("Stage-9 split configuration is malformed")
        lower, upper = pd.Timestamp(interval[0]), pd.Timestamp(interval[1])
        mask = issue.between(lower, upper) & target.between(lower, upper)
        expected_split[mask.to_numpy()] = split_name
    if not np.array_equal(frame["split"].to_numpy(object), expected_split):
        raise ModelSuiteError(
            "Stage-9 prediction split labels disagree with frozen date windows"
        )
    none_rows = frame["split"].eq("none")
    if none_rows.any() and not frame.loc[
        none_rows, "model"
    ].isin(STAGE9_AIR2STREAM_MODELS).all():
        raise ModelSuiteError(
            "Stage-9 split=none is reserved for Air2stream boundary rows"
        )

    duplicate_key = ["model", "seed", *STAGE9_POINT_KEY]
    if frame.duplicated(duplicate_key).any():
        raise ModelSuiteError("Stage-9 prediction has duplicate model×seed×key rows")
    try:
        R.validate_predictions(frame)
    except (TypeError, ValueError) as exc:
        raise ModelSuiteError("Stage-9 prediction values are not canonical") from exc

    truth_counts = frame.groupby(
        ["model", *STAGE9_POINT_KEY], observed=True, dropna=False
    )["y_true"].nunique(dropna=False)
    if truth_counts.empty or not truth_counts.eq(1).all():
        raise ModelSuiteError("Stage-9 seeds disagree on exact-key y_true")

    five_seeds = tuple(int(seed) for seed in C.USGS_SEEDS)
    _require_exact_seed_grid(frame, "ThermoRoute", five_seeds)
    _require_exact_seed_grid(frame, "LightGBM", five_seeds)
    for model in (
        "Persistence", "DampedPersistence", "Climatology", STAGE9_LGO_MODEL,
        *(STAGE9_AIR2STREAM_MODELS if air2stream else ()),
    ):
        _require_exact_seed_grid(frame, model, (0,))

    for model in MANDATORY_ABLATIONS:
        _require_exact_seed_grid(frame, model, STAGE9_ABLATION_SEEDS)

    full_all_seeds = frame[frame["model"].eq("ThermoRoute")]
    full_registry = _key_truth_registry(full_all_seeds)
    for control in MANDATORY_ABLATIONS:
        registry = _key_truth_registry(frame[frame["model"].eq(control)])
        if registry != full_registry:
            raise ModelSuiteError(
                f"Stage-9 {control} five-seed keys/y_true differ from ThermoRoute"
            )

    full_seed0 = frame[
        frame["model"].eq("ThermoRoute") & frame["seed"].eq(0)
    ]
    full_test_registry = _key_truth_registry(
        full_seed0[full_seed0["split"].eq("test")]
    )
    if not full_test_registry:
        raise ModelSuiteError("Stage-9 ThermoRoute seed0 has no test rows")
    full_test = full_seed0[full_seed0["split"].eq("test")]
    if set(full_test["site_id"].tolist()) != set(expected_sites):
        raise ModelSuiteError(
            "Stage-9 test predictions do not cover the canonical station registry"
        )
    if any(
        set(full_test.loc[full_test["horizon"].eq(horizon), "site_id"].tolist())
        != set(expected_sites)
        for horizon in C.HORIZONS
    ):
        raise ModelSuiteError(
            "Stage-9 test predictions omit a station at a declared horizon"
        )
    for model in ("Persistence", "DampedPersistence", "Climatology", "LightGBM"):
        selected = frame[
            frame["model"].eq(model)
            & frame["seed"].eq(0)
            & frame["split"].eq("test")
        ]
        if _key_truth_registry(selected) != full_test_registry:
            raise ModelSuiteError(
                f"Stage-9 {model} test keys/y_true differ from ThermoRoute seed0"
            )

    if air2stream:
        for model in STAGE9_AIR2STREAM_MODELS:
            selected = frame[
                frame["model"].eq(model) & frame["split"].eq("test")
            ]
            if _key_truth_registry(selected) != full_test_registry:
                raise ModelSuiteError(
                    f"Stage-9 {model} test keys/y_true differ from the headline registry"
                )

    lgo_test = frame[
        frame["model"].eq(STAGE9_LGO_MODEL) & frame["split"].eq("test")
    ]
    lgo_all = frame[frame["model"].eq(STAGE9_LGO_MODEL)]
    held_sites = set(lgo_test["site_id"].tolist())
    permutation = np.random.default_rng(0).permutation(list(expected_sites))
    expected_held_sites = {
        str(site) for site in permutation[:max(1, len(expected_sites) // 4)]
    }
    persistence_held = frame[
        frame["model"].eq("Persistence")
        & frame["split"].eq("test")
        & frame["site_id"].isin(held_sites)
    ]
    if (
        held_sites != expected_held_sites
        or set(lgo_all["site_id"].tolist()) != expected_held_sites
        or any(
            set(lgo_test.loc[
                lgo_test["horizon"].eq(horizon), "site_id"
            ].tolist()) != expected_held_sites
            for horizon in C.HORIZONS
        )
        or _key_truth_registry(lgo_test) != _key_truth_registry(persistence_held)
    ):
        raise ModelSuiteError(
            "Stage-9 warm-start rows differ from the frozen held-site registry"
        )

    sort_columns = [
        "model", "seed", "site_id", "horizon", "split",
        "issue_date", "target_date",
    ]
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _station_rmse_from_predictions(
    frame: pd.DataFrame,
    model: str,
    *,
    seed: int | None = None,
) -> dict[tuple[int, str], float]:
    selected = frame[
        frame["model"].eq(model) & frame["split"].eq("test")
    ]
    if seed is not None:
        selected = selected[selected["seed"].eq(seed)]
    collapsed = selected.groupby(
        list(FORECAST_KEY), as_index=False, observed=True, sort=True
    ).agg(y_pred=("y_pred", "mean"), y_true=("y_true", "first"))
    result: dict[tuple[int, str], float] = {}
    for (horizon, site), group in collapsed.groupby(
        ["horizon", "site_id"], observed=True, sort=True
    ):
        error = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        result[(int(horizon), str(site))] = float(np.sqrt(np.mean(error ** 2)))
    return result


def _expected_stage09_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    persistence = _station_rmse_from_predictions(frame, "Persistence", seed=0)
    damped = _station_rmse_from_predictions(frame, "DampedPersistence", seed=0)
    thermoroute = _station_rmse_from_predictions(frame, "ThermoRoute")
    if not persistence or set(persistence) != set(damped) or set(persistence) != set(thermoroute):
        raise ModelSuiteError("Stage-9 headline station registries differ")
    rows = [
        {
            "horizon": horizon,
            "site": site,
            "rmse_persist": persistence[(horizon, site)],
            "rmse_damped": damped[(horizon, site)],
            "rmse_thermo": thermoroute[(horizon, site)],
        }
        for horizon, site in sorted(persistence)
    ]
    return pd.DataFrame(rows, columns=[
        "horizon", "site", "rmse_persist", "rmse_damped", "rmse_thermo",
    ])


def _expected_stage09_headline_rows(
    frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    *,
    air2stream: bool,
) -> dict[str, tuple[str, ...]]:
    lightgbm = _station_rmse_from_predictions(frame, "LightGBM")
    air = {
        model: _station_rmse_from_predictions(frame, model, seed=0)
        for model in STAGE9_AIR2STREAM_MODELS
    } if air2stream else {}
    rows: dict[str, tuple[str, ...]] = {}
    for horizon in C.HORIZONS:
        current = score_frame[score_frame["horizon"].eq(horizon)]
        sites = current["site"].astype(str).tolist()
        if not sites or any((horizon, site) not in lightgbm for site in sites):
            raise ModelSuiteError("Stage-9 LightGBM headline registry is incomplete")
        median_lgb = float(np.median([lightgbm[(horizon, site)] for site in sites]))
        if air2stream:
            air_values = []
            for model in STAGE9_AIR2STREAM_MODELS:
                values = [
                    value for (model_horizon, _site), value in air[model].items()
                    if model_horizon == horizon
                ]
                if not values:
                    raise ModelSuiteError(f"Stage-9 {model} headline rows are absent")
                air_values.append(float(np.median(values)))
            air_cell = f"{air_values[0]:.3f} / {air_values[1]:.3f}"
        else:
            air_cell = "NOT_RUN / NA"
        median_persist = float(current["rmse_persist"].median())
        median_damped = float(current["rmse_damped"].median())
        median_thermo = float(current["rmse_thermo"].median())
        skill_persist = 1.0 - float(
            (current["rmse_thermo"] / current["rmse_persist"]).median()
        )
        skill_damped = 1.0 - float(
            (current["rmse_thermo"] / current["rmse_damped"]).median()
        )
        win_rate = float(
            (current["rmse_thermo"] < current["rmse_damped"]).mean()
        )
        rows[str(horizon)] = (
            f"{median_persist:.3f}", f"{median_damped:.3f}", air_cell,
            f"{median_lgb:.3f}", f"{median_thermo:.3f}",
            f"{skill_persist:+.3f}", f"{skill_damped:+.3f}",
            f"{win_rate:.2f}",
        )
    return rows


def _expected_stage09_module_rows(
    frame: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for model in ("ThermoRoute", *MANDATORY_ABLATIONS):
        selected = frame[frame["model"].eq(model) & frame["split"].eq("test")]
        ensemble = selected.groupby(
            list(FORECAST_KEY), as_index=False, sort=True
        ).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        cells = []
        for horizon in C.HORIZONS:
            current = ensemble[ensemble["horizon"].eq(horizon)]
            horizon_values = [
                float(np.sqrt(np.mean(np.square(
                    group["y_pred"].to_numpy(float)
                    - group["y_true"].to_numpy(float)
                ))))
                for _site, group in current.groupby("site_id")
            ]
            if not horizon_values:
                raise ModelSuiteError(f"Stage-9 {model} module rows are absent")
            cells.append(f"{float(np.median(horizon_values)):.3f}")
        rows[model] = tuple(cells)
    return rows


def _expected_stage09_lgo_rows(
    frame: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    lgo = frame[
        frame["model"].eq(STAGE9_LGO_MODEL) & frame["split"].eq("test")
    ]
    held_sites = set(lgo["site_id"].tolist())
    persistence = frame[
        frame["model"].eq("Persistence")
        & frame["split"].eq("test")
        & frame["site_id"].isin(held_sites)
    ]
    rows: dict[str, tuple[str, ...]] = {}
    for horizon in C.HORIZONS:
        current_lgo = lgo[lgo["horizon"].eq(horizon)]
        current_persist = persistence[persistence["horizon"].eq(horizon)]
        if current_lgo.empty or current_persist.empty:
            raise ModelSuiteError("Stage-9 warm-start report rows are absent")
        lgo_error = (
            current_lgo["y_pred"].to_numpy(float)
            - current_lgo["y_true"].to_numpy(float)
        )
        persistence_error = (
            current_persist["y_pred"].to_numpy(float)
            - current_persist["y_true"].to_numpy(float)
        )
        lgo_rmse = float(np.sqrt(np.mean(lgo_error ** 2)))
        persistence_rmse = float(np.sqrt(np.mean(persistence_error ** 2)))
        rows[str(horizon)] = (
            f"{lgo_rmse:.3f}",
            f"{persistence_rmse:.3f}",
            f"{1.0 - lgo_rmse / persistence_rmse:+.3f}",
        )
    return rows


def stage09_air2stream_report_status(enabled: bool) -> str:
    if enabled:
        return (
            f"{STAGE9_AIR2STREAM_DISPLAY_NAME}: RUN; both a4 and a8 were executed."
        )
    return (
        f"{STAGE9_AIR2STREAM_DISPLAY_NAME}: NOT_RUN; headline entry is "
        "NOT_RUN / NA."
    )


def _parse_markdown_table(
    report_text: str,
    header: tuple[str, ...],
    *,
    label: str,
) -> dict[str, tuple[str, ...]]:
    lines = report_text.splitlines()
    expected_header = "| " + " | ".join(header) + " |"
    positions = [index for index, line in enumerate(lines) if line == expected_header]
    if len(positions) != 1:
        raise ModelSuiteError(f"Stage-9 {label} table header is not exact")
    start = positions[0]
    if start + 1 >= len(lines):
        raise ModelSuiteError(f"Stage-9 {label} table is incomplete")
    separator = [cell.strip() for cell in lines[start + 1].strip("|").split("|")]
    if len(separator) != len(header) or any(cell != "---" for cell in separator):
        raise ModelSuiteError(f"Stage-9 {label} table separator is not exact")
    rows: dict[str, tuple[str, ...]] = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        if len(cells) != len(header) or not cells[0] or cells[0] in rows:
            raise ModelSuiteError(f"Stage-9 {label} table rows are malformed")
        rows[cells[0]] = cells[1:]
    if not rows:
        raise ModelSuiteError(f"Stage-9 {label} table has no rows")
    return rows


def _validate_stage09_score_frame(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    columns = ["horizon", "site", "rmse_persist", "rmse_damped", "rmse_thermo"]
    if list(actual.columns) != columns or actual.empty:
        raise ModelSuiteError("Stage-9 score table schema is not exact")
    if (
        pd.api.types.is_bool_dtype(actual["horizon"].dtype)
        or not pd.api.types.is_integer_dtype(actual["horizon"].dtype)
        or not _string_column_is_canonical(actual, "site")
        or actual.duplicated(["horizon", "site"]).any()
    ):
        raise ModelSuiteError("Stage-9 score table keys are not canonical")
    metric_columns = columns[2:]
    if any(
        pd.api.types.is_bool_dtype(actual[column].dtype)
        or not pd.api.types.is_float_dtype(actual[column].dtype)
        for column in metric_columns
    ) or not np.isfinite(actual[metric_columns].to_numpy(float)).all():
        raise ModelSuiteError("Stage-9 score table metrics are not finite floats")
    actual_sorted = actual.sort_values(["horizon", "site"]).reset_index(drop=True)
    expected_sorted = expected.sort_values(["horizon", "site"]).reset_index(drop=True)
    if actual_sorted[["horizon", "site"]].to_dict(
        orient="records"
    ) != expected_sorted[["horizon", "site"]].to_dict(orient="records"):
        raise ModelSuiteError(
            "Stage-9 score table values differ from the bound predictions"
        )
    try:
        # The dedicated round-trip reader preserves the producer's binary64
        # values exactly, so even an adjacent-float substitution is evidence
        # drift rather than a serialization allowance.
        np.testing.assert_array_max_ulp(
            actual_sorted[metric_columns].to_numpy(float),
            expected_sorted[metric_columns].to_numpy(float),
            maxulp=0,
        )
    except AssertionError as exc:
        raise ModelSuiteError(
            "Stage-9 score table values differ from the bound predictions"
        ) from exc


def _read_stage09_score_frame(path: str | Path) -> pd.DataFrame:
    """Read shortest-decimal scores without the fast parser's multi-ULP drift."""
    return pd.read_csv(
        path,
        dtype={"site": "string"},
        float_precision="round_trip",
    )


def _read_stage09_lightgbm_selection_frame(path: str | Path) -> pd.DataFrame:
    """Preserve the exact validation metrics bound into the booster manifest."""
    return pd.read_csv(path, float_precision="round_trip")


def _validate_stage09_lightgbm_selection_frame(
    selection_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Validate the frozen Stage-9 grid, row order, and selection rule."""
    if (
        tuple(selection_frame.columns) != STAGE9_LIGHTGBM_SELECTION_COLUMNS
        or selection_frame.empty
    ):
        raise ModelSuiteError("Stage-9 LightGBM selection table schema is not exact")
    if (
        pd.api.types.is_bool_dtype(selection_frame["horizon"].dtype)
        or not pd.api.types.is_integer_dtype(selection_frame["horizon"].dtype)
        or pd.api.types.is_bool_dtype(selection_frame["candidate_id"].dtype)
        or not pd.api.types.is_integer_dtype(selection_frame["candidate_id"].dtype)
        or pd.api.types.is_bool_dtype(selection_frame["num_leaves"].dtype)
        or not pd.api.types.is_integer_dtype(selection_frame["num_leaves"].dtype)
        or pd.api.types.is_bool_dtype(selection_frame["min_child_samples"].dtype)
        or not pd.api.types.is_integer_dtype(
            selection_frame["min_child_samples"].dtype
        )
        or pd.api.types.is_bool_dtype(selection_frame["best_iteration"].dtype)
        or not pd.api.types.is_integer_dtype(
            selection_frame["best_iteration"].dtype
        )
        or pd.api.types.is_bool_dtype(selection_frame["learning_rate"].dtype)
        or not pd.api.types.is_float_dtype(selection_frame["learning_rate"].dtype)
        or not pd.api.types.is_bool_dtype(selection_frame["selected"].dtype)
        or pd.api.types.is_bool_dtype(
            selection_frame["val_station_macro_rmse"].dtype
        )
        or not pd.api.types.is_float_dtype(
            selection_frame["val_station_macro_rmse"].dtype
        )
        or not _string_column_is_canonical(selection_frame, "selection_split")
    ):
        raise ModelSuiteError("Stage-9 LightGBM selection dtypes are not canonical")

    expected_order = [
        (horizon, candidate_id)
        for horizon in C.HORIZONS
        for candidate_id in range(len(STAGE9_LIGHTGBM_VALIDATION_GRID))
    ]
    actual_order = list(selection_frame[["horizon", "candidate_id"]].itertuples(
        index=False, name=None,
    ))
    if actual_order != expected_order:
        raise ModelSuiteError(
            "Stage-9 LightGBM selection row order is not the frozen "
            "horizon-by-candidate order"
        )

    selection_metrics = selection_frame["val_station_macro_rmse"].to_numpy(float)
    if (
        not np.isfinite(selection_metrics).all()
        or (selection_metrics < 0.0).any()
        or not np.isfinite(selection_frame["learning_rate"].to_numpy(float)).all()
        or not selection_frame["best_iteration"].gt(0).all()
        or not selection_frame["selection_split"].eq(
            "2016-2017 validation"
        ).all()
    ):
        raise ModelSuiteError("Stage-9 LightGBM selection values are invalid")

    for row in selection_frame.itertuples(index=False):
        expected_params = STAGE9_LIGHTGBM_VALIDATION_GRID[int(row.candidate_id)]
        actual_params = {
            "num_leaves": int(row.num_leaves),
            "min_child_samples": int(row.min_child_samples),
            "learning_rate": float(row.learning_rate),
        }
        if actual_params != expected_params:
            raise ModelSuiteError(
                "Stage-9 LightGBM selection parameters differ from the frozen grid"
            )

    for horizon in C.HORIZONS:
        current = selection_frame[selection_frame["horizon"].eq(horizon)]
        expected_candidate = min(
            current.itertuples(index=False),
            key=lambda row: (
                float(row.val_station_macro_rmse), int(row.candidate_id)
            ),
        ).candidate_id
        selected_candidates = current.loc[
            current["selected"], "candidate_id"
        ].tolist()
        if selected_candidates != [expected_candidate]:
            raise ModelSuiteError(
                "Stage-9 LightGBM selection is not the deterministic validation "
                "argmin"
            )

    return selection_frame.loc[
        :, STAGE9_LIGHTGBM_SELECTION_COLUMNS
    ].to_dict(orient="records")


def _validate_stage09_report_outputs(
    report: Path,
    scores: Path,
    lightgbm_selection: Path,
    *,
    prediction_frame: pd.DataFrame,
    configuration: Mapping[str, Any],
) -> None:
    try:
        report_text = report.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelSuiteError("Stage-9 report is unreadable") from exc
    required_report_fragments = {
        "# USGS large-sample experiment",
        "## Random held-station warm-start diagnostic",
        "## Module ablations",
        STAGE9_AIR2STREAM_DISPLAY_NAME,
        "| ThermoRoute |",
        *(f"| {name} |" for name in MANDATORY_ABLATIONS),
    }
    if any(fragment not in report_text for fragment in required_report_fragments):
        raise ModelSuiteError("Stage-9 report is incomplete")
    module_section = report_text.partition("## Module ablations")[2].split(
        "\n## ", maxsplit=1
    )[0]
    required_ablation_contract = {
        "five-seed deletion/intervention sensitivity",
        "same five seeds",
        "every mandatory control contains seeds 0--4",
        "identical forecast keys and exact y_true",
        (
            "not evidence of module necessity, causal mechanism, or "
            "capacity-matched attribution"
        ),
    }
    if any(
        fragment not in module_section
        for fragment in required_ablation_contract
    ):
        raise ModelSuiteError(
            "Stage-9 report lacks the mandatory five-seed ablation contract"
        )
    try:
        score_frame = _read_stage09_score_frame(scores)
        selection_frame = _read_stage09_lightgbm_selection_frame(
            lightgbm_selection
        )
    except (
        OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError,
    ) as exc:
        raise ModelSuiteError("Stage-9 table output is unreadable") from exc
    expected_scores = _expected_stage09_score_frame(prediction_frame)
    _validate_stage09_score_frame(score_frame, expected_scores)
    air2stream = configuration.get("air2stream")
    if not isinstance(air2stream, bool):
        raise ModelSuiteError("Stage-9 Air2stream configuration is malformed")
    if stage09_air2stream_report_status(air2stream) not in report_text:
        raise ModelSuiteError("Stage-9 Air2stream report status is inconsistent")
    expected_module_heading = (
        "## Module ablations (five-seed deletion/intervention sensitivity; "
        "ensemble-mean median per-station RMSE, "
        f"delta_scale={configuration['delta_scale']})"
    )
    if expected_module_heading not in report_text:
        raise ModelSuiteError(
            "Stage-9 report lacks the mandatory five-seed ablation contract"
        )
    expected_sites = prediction_frame.loc[
        prediction_frame["model"].eq("ThermoRoute")
        & prediction_frame["seed"].eq(0)
        & prediction_frame["split"].eq("test"),
        "site_id",
    ].nunique()
    if (
        f"# USGS large-sample experiment ({expected_sites} stations, 5 seeds)"
        not in report_text
        or "ThermoRoute = 5-seed mean" not in report_text
    ):
        raise ModelSuiteError("Stage-9 headline ensemble description is inconsistent")
    headline_header = (
        "horizon", "persist", "damped", STAGE9_AIR2STREAM_DISPLAY_NAME,
        "LightGBM", "ThermoRoute", "skill vs persist", "skill vs damped",
        "win-rate vs damped",
    )
    actual_headline = _parse_markdown_table(
        report_text, headline_header, label="headline"
    )
    expected_headline = _expected_stage09_headline_rows(
        prediction_frame, expected_scores, air2stream=air2stream
    )
    if actual_headline != expected_headline:
        raise ModelSuiteError(
            "Stage-9 headline report values differ from the bound predictions"
        )
    held_sites = prediction_frame.loc[
        prediction_frame["model"].eq(STAGE9_LGO_MODEL)
        & prediction_frame["split"].eq("test"),
        "site_id",
    ].nunique()
    expected_lgo_heading = (
        "## Random held-station warm-start diagnostic "
        f"({expected_sites - held_sites}→{held_sites})"
    )
    if expected_lgo_heading not in report_text:
        raise ModelSuiteError("Stage-9 warm-start report heading is inconsistent")
    actual_lgo = _parse_markdown_table(
        report_text,
        ("horizon", "warm-start RMSE", "persistence RMSE", "warm-start skill"),
        label="warm-start",
    )
    if actual_lgo != _expected_stage09_lgo_rows(prediction_frame):
        raise ModelSuiteError(
            "Stage-9 warm-start report values differ from the bound predictions"
        )
    actual_modules = _parse_markdown_table(
        report_text,
        ("variant", *(f"h{horizon}" for horizon in C.HORIZONS)),
        label="module ablation",
    )
    if actual_modules != _expected_stage09_module_rows(prediction_frame):
        raise ModelSuiteError(
            "Stage-9 module report values differ from the bound predictions"
        )
    _validate_stage09_lightgbm_selection_frame(selection_frame)


def validate_stage09_prepublication_outputs(
    *,
    root: str | Path,
    run_id: str,
    run_manifest: str | Path,
    predictions: str | Path,
    scores: str | Path,
    report: str | Path,
    lightgbm_selection: str | Path,
) -> None:
    """Validate every non-pointer output before any formal pointer can move."""
    root = Path(root).resolve()
    paths = {
        "run_manifest": Path(run_manifest).resolve(),
        "predictions": Path(predictions).resolve(),
        "scores": Path(scores).resolve(),
        "report": Path(report).resolve(),
        "lightgbm_selection": Path(lightgbm_selection).resolve(),
    }
    paths["prediction_sidecar"] = sidecar_path(paths["predictions"]).resolve()
    _require_canonical_stage09_paths(root, str(run_id), paths)
    for label, path in paths.items():
        file_binding(root, path)
    _, identity, configuration, input_closure = _load_formal_stage09_manifest(
        paths["run_manifest"], root=root, run_id=str(run_id)
    )
    prediction_frame = _validate_stage09_prediction_outputs(
        paths["predictions"], paths["prediction_sidecar"],
        identity=identity, run_id=str(run_id), configuration=configuration,
        root=root,
    )
    _validate_stage09_report_outputs(
        paths["report"], paths["scores"], paths["lightgbm_selection"],
        prediction_frame=prediction_frame,
        configuration=configuration,
    )
    input_closure.assert_unchanged()


def build_stage09_completion_receipt(
    *,
    root: str | Path,
    run_id: str,
    run_manifest: str | Path,
    predictions: str | Path,
    scores: str | Path,
    report: str | Path,
    lightgbm_selection: str | Path,
    thermoroute_pointer: str | Path,
    lightgbm_pointer: str | Path,
    components_pointer: str | Path,
) -> dict[str, Any]:
    """Bind every formal Stage-9 output after its report and pointers exist."""
    root = Path(root).resolve()
    paths = {
        "run_manifest": Path(run_manifest).resolve(),
        "predictions": Path(predictions).resolve(),
        "scores": Path(scores).resolve(),
        "report": Path(report).resolve(),
        "lightgbm_selection": Path(lightgbm_selection).resolve(),
        "thermoroute_pointer": Path(thermoroute_pointer).resolve(),
        "lightgbm_pointer": Path(lightgbm_pointer).resolve(),
        "components_pointer": Path(components_pointer).resolve(),
    }
    paths["prediction_sidecar"] = sidecar_path(paths["predictions"]).resolve()
    _require_canonical_stage09_paths(root, str(run_id), paths)
    _manifest, identity, configuration, input_closure = (
        _load_formal_stage09_manifest(
            paths["run_manifest"], root=root, run_id=str(run_id)
        )
    )
    document: dict[str, Any] = {
        "format": STAGE9_COMPLETION_FORMAT,
        "status": STAGE9_COMPLETION_STATUS,
        "stage": "09_usgs_experiment",
        "run_id": str(run_id),
        "run_identity": identity,
        "formal_configuration": configuration,
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": {
            label: file_binding(root, paths[label])
            for label in STAGE9_COMPLETION_ARTIFACTS
        },
    }
    document["receipt_self_sha256"] = sha256_json(document)
    input_closure.assert_unchanged()
    return document


def write_stage09_completion_receipt(
    path: str | Path,
    document: Mapping[str, Any],
    *,
    publication_guard: Callable[[], object] | None = None,
) -> Path:
    """Atomically publish the Stage-9 receipt as the transaction's last write."""
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    if document.get("receipt_self_sha256") != sha256_json(stable):
        raise ModelSuiteError("Stage-9 completion receipt self hash is invalid")
    destination = Path(path)
    atomic_write_json(
        destination,
        dict(document),
        publication_guard=publication_guard,
    )
    return destination


def validate_stage09_completion_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path,
    stage9_pointer: str | Path,
    document: Mapping[str, Any] | None = None,
    publication_guard: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Validate a candidate document or the canonically published receipt."""
    if publication_guard is not None:
        publication_guard()
    root = Path(root).resolve()
    receipt_path = _canonical_completion_receipt_path(
        root,
        receipt_path,
        relative=STAGE9_COMPLETION_RECEIPT_PATH,
        label="Stage-9 completion receipt",
        document_supplied=document is not None,
    )
    if document is None:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelSuiteError(
                "Stage-9 completion receipt is absent or invalid"
            ) from exc
    else:
        receipt = dict(document)
    expected_keys = {
        "format", "status", "stage", "run_id", "run_identity",
        "formal_configuration", "confirmation_outcomes_requested_or_read",
        "artifacts", "receipt_self_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ModelSuiteError("Stage-9 completion receipt schema is not exact")
    stable = {
        key: value for key, value in receipt.items()
        if key != "receipt_self_sha256"
    }
    if receipt.get("receipt_self_sha256") != sha256_json(stable):
        raise ModelSuiteError("Stage-9 completion receipt self hash changed")
    if (
        receipt.get("format") != STAGE9_COMPLETION_FORMAT
        or receipt.get("status") != STAGE9_COMPLETION_STATUS
        or receipt.get("stage") != "09_usgs_experiment"
        or receipt.get("confirmation_outcomes_requested_or_read") is not False
    ):
        raise ModelSuiteError("Stage-9 completion receipt is not a formal PASS")
    run_id = str(receipt.get("run_id", ""))
    if not run_id:
        raise ModelSuiteError("Stage-9 completion receipt lacks a run id")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        STAGE9_COMPLETION_ARTIFACTS
    ):
        raise ModelSuiteError("Stage-9 completion artifact registry is incomplete")
    expected_artifacts = canonical_stage09_artifact_paths(run_id)
    if any(
        not isinstance(artifacts[label], Mapping)
        or artifacts[label].get("path") != expected_artifacts[label]
        for label in STAGE9_COMPLETION_ARTIFACTS
    ):
        raise ModelSuiteError("Stage-9 completion artifact path is not canonical")
    paths = {
        label: _validated_file_binding(root, artifacts[label], label=label)
        for label in STAGE9_COMPLETION_ARTIFACTS
    }

    _, identity, configuration, input_closure = _load_formal_stage09_manifest(
        paths["run_manifest"], root=root, run_id=run_id
    )
    if identity != receipt.get("run_identity"):
        raise ModelSuiteError("Stage-9 completion run identity changed")
    if configuration != receipt.get("formal_configuration"):
        raise ModelSuiteError("Stage-9 completion configuration changed")
    prediction_frame = _validate_stage09_prediction_outputs(
        paths["predictions"], paths["prediction_sidecar"],
        identity=identity, run_id=run_id, configuration=configuration,
        root=root,
    )
    _validate_stage09_report_outputs(
        paths["report"], paths["scores"], paths["lightgbm_selection"],
        prediction_frame=prediction_frame,
        configuration=configuration,
    )

    expected_pointer = Path(stage9_pointer).resolve()
    if paths["components_pointer"] != expected_pointer:
        raise ModelSuiteError("Stage-9 receipt binds another component pointer")
    components = load_component_pointer(expected_pointer)
    entries = components["models"]
    by_id = {str(entry["model_id"]): entry for entry in entries}
    expected_models = {"ThermoRoute", "LightGBM", *MANDATORY_ABLATIONS}
    if (
        components.get("cohort") != "temporal_stage9"
        or components.get("run_id") != run_id
        or set(by_id) != expected_models
        or len(entries) != len(expected_models)
        or int(by_id["ThermoRoute"].get("member_count", 0)) != 5
        or int(by_id["LightGBM"].get("member_count", 0)) != 5
        or any(
            int(by_id[name].get("member_count", 0))
            != len(STAGE9_ABLATION_SEEDS)
            or by_id[name].get("intervention") != ABLATION_INTERVENTIONS[name]
            for name in MANDATORY_ABLATIONS
        )
    ):
        raise ModelSuiteError("Stage-9 completion component registry changed")
    prediction_binding = components.get("development_prediction_artifact")
    if prediction_binding != {
        **dict(artifacts["predictions"]),
        "sidecar": dict(artifacts["prediction_sidecar"]),
    }:
        raise ModelSuiteError("Stage-9 component pointer binds other predictions")
    development = components.get("development_contract")
    if (
        not isinstance(development, Mapping)
        or development.get("source_sha256") != identity.get("source_sha256")
        or development.get("panel", {}).get("sha256")
        != identity.get("panel_sha256")
        or development.get("registry", {}).get("sha256")
        != identity.get("registry_sha256")
    ):
        raise ModelSuiteError("Stage-9 component pointer has different lineage")

    try:
        thermoroute_pointer = json.loads(
            paths["thermoroute_pointer"].read_text(encoding="utf-8")
        )
        lightgbm_pointer = json.loads(
            paths["lightgbm_pointer"].read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ModelSuiteError("Stage-9 shortcut pointer is malformed") from exc
    primary_artifact = by_id["ThermoRoute"]["artifact"]
    if thermoroute_pointer != {
        "run_id": run_id,
        "bundle_path": primary_artifact["path"],
        "member_count": 5,
        "metadata_sha256": primary_artifact["metadata_sha256"],
        "weights_sha256": primary_artifact["weights_sha256"],
    }:
        raise ModelSuiteError("Stage-9 ThermoRoute shortcut pointer changed")
    if lightgbm_pointer != {
        "run_id": run_id,
        "manifest": by_id["LightGBM"]["artifact"],
        "member_count": 5,
    }:
        raise ModelSuiteError("Stage-9 LightGBM shortcut pointer changed")
    lightgbm_manifest_path = _validated_file_binding(
        root,
        by_id["LightGBM"].get("artifact"),
        label="Stage-9 LightGBM manifest",
    )
    try:
        lightgbm_manifest = json.loads(
            lightgbm_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteError("Stage-9 LightGBM manifest is malformed") from exc
    if not isinstance(lightgbm_manifest, Mapping):
        raise ModelSuiteError("Stage-9 LightGBM manifest is malformed")
    try:
        selection_frame = _read_stage09_lightgbm_selection_frame(
            paths["lightgbm_selection"]
        )
    except (
        OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError,
    ) as exc:
        raise ModelSuiteError("Stage-9 LightGBM selection table is unreadable") from exc
    selection_records = _validate_stage09_lightgbm_selection_frame(selection_frame)
    manifest_selection = lightgbm_manifest.get("validation_selection")
    if not isinstance(manifest_selection, list) or any(
        not isinstance(record, Mapping)
        or set(record) != set(STAGE9_LIGHTGBM_SELECTION_COLUMNS)
        for record in manifest_selection
    ):
        raise ModelSuiteError(
            "Stage-9 LightGBM selection CSV differs from the bound bundle manifest"
        )
    normalised_manifest_selection = [
        {
            column: record[column]
            for column in STAGE9_LIGHTGBM_SELECTION_COLUMNS
        }
        for record in manifest_selection
    ]
    if normalised_manifest_selection != selection_records:
        raise ModelSuiteError(
            "Stage-9 LightGBM selection CSV differs from the bound bundle manifest"
        )
    if (
        tuple(lightgbm_manifest.get("members", ()))
        != tuple(f"seed{seed}" for seed in C.USGS_SEEDS)
        or tuple(lightgbm_manifest.get("horizons", ())) != tuple(C.HORIZONS)
    ):
        raise ModelSuiteError(
            "Stage-9 LightGBM quantile audit registry is incomplete"
        )
    _validate_lightgbm_quantile_metadata(lightgbm_manifest)
    input_closure.assert_unchanged()
    if publication_guard is not None:
        publication_guard()
    return receipt


def stage09_completion_gate_binding(
    receipt_path: str | Path,
    *,
    root: str | Path,
    stage9_pointer: str | Path,
    publication_guard: Callable[[], object] | None = None,
) -> dict[str, str]:
    """Validate Stage 9 and return the exact receipt binding frozen downstream."""
    validate_stage09_completion_receipt(
        receipt_path,
        root=root,
        stage9_pointer=stage9_pointer,
        publication_guard=publication_guard,
    )
    return file_binding(root, receipt_path)


def publish_stage09_completion_receipt(
    receipt_path: str | Path,
    document: Mapping[str, Any],
    *,
    root: str | Path,
    stage9_pointer: str | Path,
    publication_guard: Callable[[], object],
) -> Path:
    """Validate the full closure before atomically publishing its PASS marker."""
    validate_stage09_completion_receipt(
        receipt_path,
        root=root,
        stage9_pointer=stage9_pointer,
        document=document,
        publication_guard=publication_guard,
    )
    publication_guard()
    destination = write_stage09_completion_receipt(
        receipt_path,
        document,
        publication_guard=publication_guard,
    )
    validate_stage09_completion_receipt(
        destination,
        root=root,
        stage9_pointer=stage9_pointer,
        publication_guard=publication_guard,
    )
    publication_guard()
    return destination


def canonical_stage16_artifact_paths(run_id: str) -> dict[str, str]:
    """Return the only paths admitted to a formal Stage-16 completion."""
    if not isinstance(run_id, str) or not run_id:
        raise ModelSuiteError("Stage-16 canonical paths require a run id")
    bundle = f"outputs/models/lstm_usgs_bundle_{run_id}"
    return {
        "run_manifest": f"outputs/runs/16_lstm_baseline/{run_id}/run.json",
        "stage09_completion_receipt": STAGE9_COMPLETION_RECEIPT_PATH,
        "stage09_parent_predictions": STAGE16_PARENT_PREDICTION_PATH,
        "stage09_parent_prediction_sidecar": (
            f"{STAGE16_PARENT_PREDICTION_PATH}.meta.json"
        ),
        "lstm_validation_selection": STAGE16_SELECTION_PATH,
        "development_predictions": STAGE16_DEVELOPMENT_PREDICTION_PATH,
        "development_prediction_sidecar": (
            f"{STAGE16_DEVELOPMENT_PREDICTION_PATH}.meta.json"
        ),
        "bundle": bundle,
        "bundle_metadata": f"{bundle}/metadata.json",
        "bundle_weights": f"{bundle}/weights.pt",
        "shortcut_pointer": STAGE16_SHORTCUT_POINTER_PATH,
        "components_pointer": STAGE16_COMPONENT_POINTER_PATH,
        **{
            f"seed{seed}_predictions": (
                f"outputs/runs/16_lstm_baseline/{run_id}/predictions/"
                f"seed{seed}.parquet"
            )
            for seed in C.USGS_SEEDS
        },
        **{
            f"seed{seed}_prediction_sidecar": (
                f"outputs/runs/16_lstm_baseline/{run_id}/predictions/"
                f"seed{seed}.parquet.meta.json"
            )
            for seed in C.USGS_SEEDS
        },
        **{
            f"candidate{candidate_id}_predictions": (
                f"outputs/runs/16_lstm_baseline/{run_id}/selection/"
                f"candidate{candidate_id}.parquet"
            )
            for candidate_id in range(len(LSTM_VALIDATION_GRID))
        },
        **{
            f"candidate{candidate_id}_prediction_sidecar": (
                f"outputs/runs/16_lstm_baseline/{run_id}/selection/"
                f"candidate{candidate_id}.parquet.meta.json"
            )
            for candidate_id in range(len(LSTM_VALIDATION_GRID))
        },
        **{
            f"candidate{candidate_id}_checkpoint": (
                f"outputs/runs/16_lstm_baseline/{run_id}/selection/"
                f"candidate{candidate_id}.pt"
            )
            for candidate_id in range(len(LSTM_VALIDATION_GRID))
        },
        **{
            f"candidate{candidate_id}_checkpoint_sidecar": (
                f"outputs/runs/16_lstm_baseline/{run_id}/selection/"
                f"candidate{candidate_id}.pt.meta.json"
            )
            for candidate_id in range(len(LSTM_VALIDATION_GRID))
        },
    }


def _stage16_exact_file(root: Path, relative: str, *, label: str) -> Path:
    """Resolve one exact, single-link regular file without accepting aliases."""
    lexical = root / relative
    current = lexical
    while current != root:
        if current.is_symlink():
            raise ModelSuiteError(f"Stage-16 {label} is a symlink")
        current = current.parent
    try:
        lexical_stat = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ModelSuiteError(f"Stage-16 {label} is absent or unreadable") from exc
    if (
        not stat.S_ISREG(lexical_stat.st_mode)
        or lexical_stat.st_nlink != 1
        or not resolved.is_file()
        or _relative(root, resolved) != relative
        or resolved.is_symlink()
    ):
        raise ModelSuiteError(
            f"Stage-16 {label} path is not a canonical single-link file"
        )
    return resolved


def _stage16_exact_directory(root: Path, relative: str, *, label: str) -> Path:
    """Resolve one exact directory closure without accepting symlink aliases."""
    lexical = root / relative
    current = lexical
    while current != root:
        if current.is_symlink():
            raise ModelSuiteError(f"Stage-16 {label} is a symlink")
        current = current.parent
    resolved = lexical.resolve()
    if not resolved.is_dir() or _relative(root, resolved) != relative:
        raise ModelSuiteError(f"Stage-16 {label} path is not canonical")
    return resolved


@dataclass(frozen=True)
class _Stage16FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


def _stage16_file_snapshot(path: Path, *, label: str) -> _Stage16FileSnapshot:
    """Read one no-follow file descriptor and bind its exact inode and bytes."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ModelSuiteError(f"Stage-16 {label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ModelSuiteError(
                f"Stage-16 {label} is not a single-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_nlink,
    )
    try:
        lexical = path.lstat()
    except OSError as exc:
        raise ModelSuiteError(f"Stage-16 {label} changed while read") from exc
    identity_lexical = (
        lexical.st_dev, lexical.st_ino, lexical.st_size, lexical.st_mtime_ns,
        lexical.st_nlink,
    )
    if identity_before != identity_after or identity_after != identity_lexical:
        raise ModelSuiteError(f"Stage-16 {label} changed while read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise ModelSuiteError(f"Stage-16 {label} byte count changed while read")
    return _Stage16FileSnapshot(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
    )


def _stage16_snapshot_binding(
    root: Path, snapshot: _Stage16FileSnapshot,
) -> dict[str, str]:
    return {"path": _relative(root, snapshot.path), "sha256": snapshot.sha256}


def _stage16_assert_snapshot_current(
    snapshot: _Stage16FileSnapshot, *, label: str,
) -> None:
    observed = _stage16_file_snapshot(snapshot.path, label=label)
    if (
        observed.device,
        observed.inode,
        observed.size,
        observed.mtime_ns,
        observed.sha256,
    ) != (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.sha256,
    ):
        raise ModelSuiteError(f"Stage-16 {label} changed during validation")


def _stage16_read_prediction_frame(
    path: Path, *, label: str, payload: bytes | None = None,
) -> pd.DataFrame:
    """Read one prediction Parquet without admitting physical-type aliases."""
    source: Path | BytesIO = path if payload is None else BytesIO(payload)
    try:
        schema = pq.ParquetFile(source).schema_arrow
    except (OSError, ValueError, TypeError) as exc:
        raise ModelSuiteError(f"Stage-16 {label} Parquet schema is unreadable") from exc
    if schema.names != list(R.PRED_COLS):
        raise ModelSuiteError(f"Stage-16 {label} column order changed")
    text_columns = {"model", "scope", "feature_set", "site_id", "split"}
    integer_columns = {"seed", "horizon"}
    date_columns = {"issue_date", "target_date"}
    float_columns = {"y_true", "y_pred", "q05", "q50", "q95", "p_exceed"}
    for field in schema:
        if field.name in text_columns and not pa.types.is_string(field.type):
            raise ModelSuiteError(
                f"Stage-16 {label} {field.name} Arrow type changed"
            )
        if field.name in integer_columns and not pa.types.is_integer(field.type):
            raise ModelSuiteError(
                f"Stage-16 {label} {field.name} Arrow type changed"
            )
        if field.name in date_columns and not (
            pa.types.is_timestamp(field.type)
            and field.type.unit == "ns"
            and field.type.tz is None
        ):
            raise ModelSuiteError(
                f"Stage-16 {label} {field.name} Arrow type changed"
            )
        if field.name in float_columns and not pa.types.is_floating(field.type):
            raise ModelSuiteError(
                f"Stage-16 {label} {field.name} Arrow type changed"
            )
    try:
        frame = pd.read_parquet(path if payload is None else BytesIO(payload))
    except (OSError, ValueError, ImportError) as exc:
        raise ModelSuiteError(f"Stage-16 {label} Parquet is unreadable") from exc
    if tuple(frame.columns) != tuple(R.PRED_COLS) or frame.empty:
        raise ModelSuiteError(f"Stage-16 {label} logical schema changed")
    for column in text_columns:
        if not _string_column_is_canonical(frame, column):
            raise ModelSuiteError(
                f"Stage-16 {label} {column} is not canonical text"
            )
    for column in integer_columns:
        values = frame[column]
        if (
            pd.api.types.is_bool_dtype(values.dtype)
            or not pd.api.types.is_integer_dtype(values.dtype)
            or values.isna().any()
        ):
            raise ModelSuiteError(
                f"Stage-16 {label} {column} is not a non-null integer"
            )
    for column in date_columns:
        values = frame[column]
        if (
            str(values.dtype) != "datetime64[ns]"
            or values.isna().any()
            or not values.equals(values.dt.normalize())
        ):
            raise ModelSuiteError(
                f"Stage-16 {label} {column} is not a canonical day"
            )
    for column in float_columns:
        values = frame[column]
        if (
            pd.api.types.is_bool_dtype(values.dtype)
            or not pd.api.types.is_float_dtype(values.dtype)
        ):
            raise ModelSuiteError(
                f"Stage-16 {label} {column} is not floating point"
            )
    try:
        R.validate_predictions(frame)
    except (ValueError, RuntimeError, TypeError) as exc:
        raise ModelSuiteError(f"Stage-16 {label} values are invalid") from exc
    return frame


def _stage16_formal_configuration(
    resolved: Mapping[str, Any], *, parent_sha256: str,
) -> dict[str, Any]:
    """Validate the exact same-station LSTM configuration frozen by Stage 16."""
    expected_fields = {
        "stage", "role", "parent_sha256", "models", "seeds", "variables",
        "horizons", "context_length", "station_embedding",
        "station_balanced", "selection_metric", "validation_grid",
        "validation_selection_seed", "validation_selection_split",
        "event_reference_fit_interval", "train_config", "training_device",
        "formal_numerical_policy", "input_closure_sha256",
        "input_closure_file_count",
    }
    numerical_policy = resolved.get("formal_numerical_policy")
    input_closure_sha256 = resolved.get("input_closure_sha256")
    input_closure_file_count = resolved.get("input_closure_file_count")
    if (
        set(resolved) != expected_fields
        or resolved.get("stage") != "16_lstm_baseline_insample"
        or resolved.get("role") != "final_route_a_development_predictions"
        or resolved.get("parent_sha256") != parent_sha256
        or tuple(resolved.get("models", ())) != tuple(ROUTE_A_PRIMARY_MODELS)
        or tuple(resolved.get("seeds", ())) != tuple(C.USGS_SEEDS)
        or tuple(resolved.get("variables", ())) != STAGE9_USGS_VARIABLES
        or tuple(resolved.get("horizons", ())) != tuple(C.HORIZONS)
        or resolved.get("context_length") != C.CONTEXT_LENGTH
        or resolved.get("station_embedding") is not True
        or resolved.get("station_balanced") is not True
        or resolved.get("selection_metric") != "station_macro"
        or resolved.get("validation_grid")
        != [dict(value) for value in LSTM_VALIDATION_GRID]
        or resolved.get("validation_selection_seed") != C.USGS_SEEDS[0]
        or resolved.get("validation_selection_split") != "2016-2017 only"
        or resolved.get("event_reference_fit_interval")
        != ["2006-01-01", "2018-12-31"]
        or resolved.get("train_config")
        != asdict(C.TrainConfig(batch_size=1536))
        or resolved.get("training_device") != "cpu"
        or not isinstance(numerical_policy, Mapping)
        or not numerical_policy
        or not isinstance(input_closure_sha256, str)
        or len(input_closure_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in input_closure_sha256
        )
        or type(input_closure_file_count) is not int
        or input_closure_file_count < 5
    ):
        raise ModelSuiteError(
            "Stage-16 run manifest has malformed formal configuration"
        )
    return json.loads(json.dumps(resolved, sort_keys=True))


def _load_formal_stage16_manifest(
    path: Path,
    *,
    root: Path,
    run_id: str,
    parent_sha256: str,
    enforce_current_runtime: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load one content-addressed, CPU-only, development-only Stage-16 run."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteError(
            "Stage-16 receipt binds a malformed run manifest"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {
            "schema_version", "identity", "resolved_config", "created_utc",
            "environment", "git", "provenance",
        }
        or manifest.get("schema_version") != RUN_SCHEMA_VERSION
    ):
        raise ModelSuiteError("Stage-16 run manifest schema is not exact")
    identity, resolved = _validated_content_addressed_run_identity(
        manifest, label="Stage-16"
    )
    if (
        identity["run_id"] != run_id
        or identity["source_sha256"] != source_tree_hash(root)
        or identity["config_sha256"] != sha256_json(resolved)
    ):
        raise ModelSuiteError(
            "Stage-16 completion receipt is stale for the run or current source"
        )
    panel = root / "data_usgs" / "panel_usgs_120v2.parquet"
    registry = root / "data_usgs" / "station_registry_v1.csv"
    try:
        panel_sha256 = sha256_file(panel)
        registry_sha256 = sha256_file(registry)
    except OSError as exc:
        raise ModelSuiteError(
            "Stage-16 canonical panel or station registry is absent"
        ) from exc
    if (
        identity["panel_sha256"] != panel_sha256
        or identity["registry_sha256"] != registry_sha256
    ):
        raise ModelSuiteError(
            "Stage-16 identity differs from the canonical development data"
        )
    if enforce_current_runtime and identity["runtime_sha256"] != sha256_json(
        numerical_runtime_contract()
    ):
        raise ModelSuiteError(
            "Stage-16 identity differs from the current numerical runtime"
        )
    configuration = _stage16_formal_configuration(
        resolved, parent_sha256=parent_sha256
    )
    if configuration["input_closure_sha256"] != identity[
        "input_closure_sha256"
    ]:
        raise ModelSuiteError(
            "Stage-16 configuration and run identity bind different input closures"
        )
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != {"evidence_role", "training_device"}
        or provenance.get("evidence_role")
        != "prelabel_route_a_model_build_development_only"
        or provenance.get("training_device") != "cpu"
    ):
        raise ModelSuiteError(
            "Stage-16 run manifest lacks development-only provenance"
        )
    return manifest, identity, configuration


def _read_stage16_selection(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the frozen grid and return its deterministic validation winner."""
    try:
        frame = pd.read_csv(path)
    except (
        OSError, UnicodeDecodeError, pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise ModelSuiteError(
            "Stage-16 LSTM validation selection is unreadable"
        ) from exc
    if tuple(frame.columns) != STAGE16_LSTM_SELECTION_COLUMNS:
        raise ModelSuiteError(
            "Stage-16 LSTM validation selection schema changed"
        )
    if len(frame) != len(LSTM_VALIDATION_GRID):
        raise ModelSuiteError(
            "Stage-16 LSTM validation grid is incomplete"
        )
    records: list[dict[str, Any]] = []
    for expected_id, expected in enumerate(LSTM_VALIDATION_GRID):
        row = frame.iloc[expected_id]
        raw_id = row["candidate_id"]
        raw_selected = row["selected"]
        raw_metric = row["val_station_macro_rmse"]
        integer_fields = ("d", "layers", "station_embed_dim")
        if (
            isinstance(raw_id, (bool, np.bool_))
            or not isinstance(raw_id, (int, np.integer))
            or int(raw_id) != expected_id
            or not isinstance(raw_selected, (bool, np.bool_))
            or isinstance(raw_metric, (bool, np.bool_))
            or not isinstance(raw_metric, (float, np.floating))
            or any(
                isinstance(row[field], (bool, np.bool_))
                or not isinstance(row[field], (int, np.integer))
                for field in integer_fields
            )
            or isinstance(row["dropout"], (bool, np.bool_))
            or not isinstance(row["dropout"], (float, np.floating))
            or not isinstance(row["use_derived_context"], (bool, np.bool_))
            or not isinstance(row["anchor"], str)
            or not isinstance(row["selection_split"], str)
        ):
            raise ModelSuiteError(
                "Stage-16 LSTM validation selection row is malformed"
            )
        try:
            metric = float(raw_metric)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelSuiteError(
                "Stage-16 LSTM validation metric is malformed"
            ) from exc
        candidate = {
            "d": int(row["d"]),
            "layers": int(row["layers"]),
            "dropout": float(row["dropout"]),
            "station_embed_dim": int(row["station_embed_dim"]),
            "use_derived_context": bool(row["use_derived_context"]),
            "anchor": str(row["anchor"]),
        }
        if (
            candidate != dict(expected)
            or not np.isfinite(metric)
            or metric < 0.0
            or str(row["selection_split"])
            != "2016-2017 validation"
        ):
            raise ModelSuiteError(
                "Stage-16 LSTM validation selection differs from the frozen grid"
            )
        records.append({
            "candidate_id": expected_id,
            **candidate,
            "val_station_macro_rmse": metric,
            "selected": bool(raw_selected),
            "selection_split": "2016-2017 validation",
        })
    selected = [record for record in records if record["selected"]]
    winner = records[stage16_validation_winner([
        float(record["val_station_macro_rmse"]) for record in records
    ])]
    if len(selected) != 1 or selected[0] != winner:
        raise ModelSuiteError(
            "Stage-16 LSTM validation winner is not deterministic"
        )
    return records, {
        key: winner[key]
        for key in LSTM_VALIDATION_GRID[int(winner["candidate_id"])]
    }


def _stage16_replay_inputs(
    root: Path, *, build_windows: bool,
) -> tuple[Any | None, dict[str, float], dict[str, Any]]:
    """Rebuild canonical windows and the exact train-q90 threshold registry."""
    prepared = D.prepare_dataset_from_panel(
        str(root / "data_usgs" / "panel_usgs_120v2.parquet"),
        frozen_spec=root / "data_usgs" / "frozen_panel_v1.json",
    )
    panel_raw = cast(pd.DataFrame, prepared["panel_raw"])
    panel = cast(pd.DataFrame, prepared["panel"])
    masks = cast(D.SplitMasks, prepared["masks"])
    prepared_stations = cast(Sequence[object], prepared["stations"])
    stations = tuple(str(value) for value in prepared_stations)
    thresholds = {
        site: float(
            panel_raw.loc[
                masks.train & panel_raw["site_id"].astype(str).eq(site),
                "WTEMP",
            ].quantile(0.90)
        )
        for site in stations
    }
    if (
        stations != tuple(C.STATIONS)
        or set(thresholds) != set(C.STATIONS)
        or not np.isfinite(np.asarray(list(thresholds.values()), dtype=float)).all()
    ):
        raise ModelSuiteError("Stage-16 train-q90 threshold registry changed")
    sorted_thresholds = {
        site: thresholds[site] for site in sorted(thresholds)
    }
    threshold_contract = {
        "target": "WTEMP",
        "fit_split": "canonical development train mask",
        "scope": "station-specific",
        "estimator": "pandas Series.quantile(q=0.90, interpolation=linear)",
        "quantile": 0.90,
        "registry": sorted_thresholds,
        "registry_sha256": sha256_json(sorted_thresholds),
    }
    if not build_windows:
        return None, thresholds, threshold_contract
    climatology = F.HarmonicClimatology.fit(panel_raw, masks.train)
    return (
        DS.build_windows(
            panel,
            masks,
            climatology,
            variables=STAGE9_USGS_VARIABLES,
            require_observed_target=True,
        ),
        thresholds,
        threshold_contract,
    )


def _validate_stage16_candidate_checkpoint_payload(
    *,
    candidate_id: int,
    candidate_config: Mapping[str, Any],
    checkpoint_payload: object,
    run_id: str,
    expected_config_json: str,
) -> None:
    """Apply the production checkpoint schema to one frozen selection member."""
    from .train import LSTMForecaster

    train_config = C.TrainConfig(batch_size=1536)
    try:
        model = LSTMForecaster(
            n_vars=len(STAGE9_USGS_VARIABLES),
            n_stations=len(C.STATIONS),
            context=C.CONTEXT_LENGTH,
            station_agnostic=False,
            **dict(candidate_config),
        ).to("cpu")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=4
        )
        _, extra = _validate_checkpoint_payload(
            checkpoint_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id=run_id,
            expected_config_json=expected_config_json,
        )
        if not isinstance(checkpoint_payload, Mapping):  # pragma: no cover
            raise TypeError("checkpoint payload is not a mapping")
        bad_epochs = extra.get("bad_epochs")
        train_rng_state = extra.get("train_rng_state")
        epoch = checkpoint_payload.get("epoch")
        if (
            set(extra) != {"bad_epochs", "train_rng_state"}
            or type(bad_epochs) is not int
            or type(epoch) is not int
            or not isinstance(train_rng_state, Mapping)
            or bad_epochs < 0
            or bad_epochs > epoch + 1
            or not (
                epoch == train_config.max_epochs - 1
                or bad_epochs >= train_config.patience
            )
        ):
            raise ValueError("checkpoint does not prove terminal formal training")
        # Assignment applies NumPy's full bit-generator schema validation
        # without mutating any process-global RNG state.
        probe_rng = np.random.default_rng(C.USGS_SEEDS[0])
        probe_rng.bit_generator.state = dict(train_rng_state)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelSuiteError(
            f"Stage-16 candidate{candidate_id} checkpoint payload is invalid"
        ) from exc


def _verify_stage16_candidate_checkpoint(
    *,
    candidate_id: int,
    candidate_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    expected: pd.DataFrame,
    wd: Any,
    thresholds: Mapping[str, float],
    publication_guard: Callable[[], object] | None,
) -> float:
    """Replay one validation candidate from its frozen best model state."""
    from .train import LSTMForecaster, export_predictions

    if publication_guard is not None:
        publication_guard()
    state = checkpoint_payload.get("best_model_state")
    if not isinstance(state, Mapping) or not state:
        raise ModelSuiteError(
            f"Stage-16 candidate{candidate_id} lacks a best model state"
        )
    try:
        model = LSTMForecaster(
            n_vars=len(STAGE9_USGS_VARIABLES),
            n_stations=len(C.STATIONS),
            context=C.CONTEXT_LENGTH,
            station_agnostic=False,
            **dict(candidate_config),
        ).to("cpu")
        model.load_state_dict(state, strict=True)
        replay = export_predictions(
            model,
            wd,
            dict(thresholds),
            torch.device("cpu"),
            f"LSTM-grid-{candidate_id}",
            "validation_selection",
            "USGS",
            C.USGS_SEEDS[0],
            batch_size=max(C.TrainConfig(batch_size=1536).batch_size, 2048),
            splits=("val",),
        )
        R.validate_predictions(replay)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelSuiteError(
            f"Stage-16 candidate{candidate_id} best-state replay failed"
        ) from exc
    keys = [
        "model", "scope", "feature_set", "seed", "site_id", "horizon",
        "split", "issue_date", "target_date",
    ]
    values = ["y_true", "y_pred", "q05", "q50", "q95", "p_exceed"]
    left = expected.loc[:, keys + values].copy()
    right = replay.loc[:, keys + values].copy()
    for frame in (left, right):
        frame["issue_date"] = pd.to_datetime(frame["issue_date"])
        frame["target_date"] = pd.to_datetime(frame["target_date"])
    try:
        paired = left.merge(
            right,
            on=keys,
            how="outer",
            suffixes=("_reference", "_checkpoint"),
            indicator=True,
            validate="one_to_one",
        )
    except pd.errors.MergeError as exc:
        raise ModelSuiteError(
            f"Stage-16 candidate{candidate_id} replay keys are not unique"
        ) from exc
    if not paired["_merge"].eq("both").all():
        raise ModelSuiteError(
            f"Stage-16 candidate{candidate_id} replay keys changed"
        )
    maximum = 0.0
    for column in values:
        difference = np.abs(
            paired[f"{column}_reference"].to_numpy(float)
            - paired[f"{column}_checkpoint"].to_numpy(float)
        )
        if np.any(~np.isfinite(difference)):
            raise ModelSuiteError(
                f"Stage-16 candidate{candidate_id} replay has non-finite values"
            )
        maximum = max(maximum, float(difference.max(initial=0.0)))
    if maximum > STAGE16_SELECTION_METRIC_ATOL:
        raise ModelSuiteError(
            f"Stage-16 candidate{candidate_id} best-state replay differs: "
            f"{maximum} > {STAGE16_SELECTION_METRIC_ATOL}"
        )
    if publication_guard is not None:
        publication_guard()
    return maximum


def _stage16_selection_input_closure(
    *,
    root: Path,
    run_manifest: Path,
    selection_path: Path,
    candidate_files: Sequence[Mapping[str, str]],
    threshold_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind selection replay to every data, grid, and checkpoint byte."""
    return {
        "run_manifest": file_binding(root, run_manifest),
        "panel": file_binding(
            root, root / "data_usgs" / "panel_usgs_120v2.parquet"
        ),
        "frozen_panel_spec": file_binding(
            root, root / "data_usgs" / "frozen_panel_v1.json"
        ),
        "station_registry": file_binding(
            root, root / "data_usgs" / "station_registry_v1.csv"
        ),
        "selection": file_binding(root, selection_path),
        "candidate_files": [dict(value) for value in candidate_files],
        "event_threshold_contract": dict(threshold_contract),
    }


def _make_stage16_selection_audit(
    *,
    selection_records: Sequence[Mapping[str, Any]],
    recomputed_metrics: Sequence[float],
    checkpoint_metrics: Sequence[float],
    replay_differences: Sequence[float],
    winner_candidate_id: int,
    input_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the content-bound three-candidate best-state replay audit."""
    return {
        "format": STAGE16_SELECTION_AUDIT_FORMAT,
        "status": "PASS_BEST_STATE_REPLAY_AND_VALIDATION_SELECTION_PARITY",
        "metric": "mean_station_rmse_across_all_horizons",
        "selection_split": "2016-2017 validation",
        "metric_atol": STAGE16_SELECTION_METRIC_ATOL,
        "replay_atol": STAGE16_SELECTION_METRIC_ATOL,
        "winner_candidate_id": winner_candidate_id,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "recomputed_val_station_macro_rmse": float(
                    recomputed_metrics[candidate_id]
                ),
                "reported_val_station_macro_rmse": float(
                    selection_records[candidate_id]["val_station_macro_rmse"]
                ),
                "checkpoint_best_metric": float(
                    checkpoint_metrics[candidate_id]
                ),
                "best_state_max_abs_difference": float(
                    replay_differences[candidate_id]
                ),
                "selected": candidate_id == winner_candidate_id,
            }
            for candidate_id in range(len(recomputed_metrics))
        ],
        "input_closure": dict(input_closure),
        "input_closure_sha256": sha256_json(input_closure),
    }


def _validate_stage16_selection_audit(
    value: object,
    *,
    selection_records: Sequence[Mapping[str, Any]],
    recomputed_metrics: Sequence[float],
    checkpoint_metrics: Sequence[float],
    winner_candidate_id: int,
    input_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Statically verify a content-bound candidate best-state replay audit."""
    expected_keys = {
        "format", "status", "metric", "selection_split", "metric_atol",
        "replay_atol", "winner_candidate_id", "candidates",
        "input_closure", "input_closure_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ModelSuiteError("Stage-16 validation-selection audit schema changed")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(
        LSTM_VALIDATION_GRID
    ):
        raise ModelSuiteError("Stage-16 validation-selection audit is incomplete")
    if (
        value.get("format") != STAGE16_SELECTION_AUDIT_FORMAT
        or value.get("status")
        != "PASS_BEST_STATE_REPLAY_AND_VALIDATION_SELECTION_PARITY"
        or value.get("metric") != "mean_station_rmse_across_all_horizons"
        or value.get("selection_split") != "2016-2017 validation"
        or value.get("metric_atol") != STAGE16_SELECTION_METRIC_ATOL
        or value.get("replay_atol") != STAGE16_SELECTION_METRIC_ATOL
        or value.get("winner_candidate_id") != winner_candidate_id
        or value.get("input_closure") != dict(input_closure)
        or value.get("input_closure_sha256") != sha256_json(input_closure)
    ):
        raise ModelSuiteError("Stage-16 validation-selection audit changed")
    candidate_keys = {
        "candidate_id", "recomputed_val_station_macro_rmse",
        "reported_val_station_macro_rmse", "checkpoint_best_metric",
        "best_state_max_abs_difference", "selected",
    }
    for candidate_id, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != candidate_keys:
            raise ModelSuiteError(
                "Stage-16 validation-selection candidate audit changed"
            )
        try:
            recomputed = float(candidate["recomputed_val_station_macro_rmse"])
            reported = float(candidate["reported_val_station_macro_rmse"])
            checkpoint_metric = float(candidate["checkpoint_best_metric"])
            difference = float(candidate["best_state_max_abs_difference"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelSuiteError(
                "Stage-16 validation-selection candidate audit is malformed"
            ) from exc
        if (
            candidate.get("candidate_id") != candidate_id
            or recomputed != float(recomputed_metrics[candidate_id])
            or reported
            != float(selection_records[candidate_id]["val_station_macro_rmse"])
            or checkpoint_metric != float(checkpoint_metrics[candidate_id])
            or candidate.get("selected") is not (
                candidate_id == winner_candidate_id
            )
            or not np.isfinite(difference)
            or difference < 0.0
            or difference > STAGE16_SELECTION_METRIC_ATOL
        ):
            raise ModelSuiteError(
                "Stage-16 validation-selection candidate audit changed"
            )
    return dict(value)


def _verify_stage16_final_bundle_parity(
    *,
    root: Path,
    bundle: Path,
    expected: pd.DataFrame,
    metadata: Mapping[str, Any],
    wd: Any | None = None,
    publication_guard: Callable[[], object] | None,
) -> dict[str, Any]:
    """Actually replay all five final members on val/calib/test windows."""
    binding = metadata.get("development_prediction")
    if not isinstance(binding, Mapping):
        raise ModelSuiteError("Stage-16 LSTM lacks a prediction parity binding")
    raw_atol = binding.get("atol")
    if (
        isinstance(raw_atol, (bool, np.bool_))
        or not isinstance(raw_atol, (float, np.floating))
        or float(raw_atol) != 1e-5
    ):
        raise ModelSuiteError("Stage-16 LSTM parity tolerance changed")
    if publication_guard is not None:
        publication_guard()
    if wd is None:
        wd, _, _ = _stage16_replay_inputs(root, build_windows=True)
    if wd is None:  # pragma: no cover - build_windows=True is authoritative
        raise ModelSuiteError("Stage-16 bundle replay lacks canonical windows")
    from .frozen_inference import lstm_factory_from_metadata

    difference = verify_sequence_prediction_parity(
        bundle,
        wd=wd,
        expected=expected,
        model_factory=lambda _member, value: lstm_factory_from_metadata(value),
        member_seeds={f"seed{seed}": seed for seed in C.USGS_SEEDS},
        atol=float(raw_atol),
        splits=("val", "calib", "test"),
        publication_guard=publication_guard,
    )
    input_closure = _stage16_parity_input_closure(
        root=root, bundle=bundle, metadata=metadata
    )
    audit = {
        "format": STAGE16_PARITY_AUDIT_FORMAT,
        "status": "PASS_FIVE_MEMBER_VAL_CALIB_TEST_REPLAY",
        "members": [f"seed{seed}" for seed in C.USGS_SEEDS],
        "splits": ["val", "calib", "test"],
        "atol": float(raw_atol),
        "max_abs_difference": float(difference),
        "input_closure": input_closure,
        "input_closure_sha256": sha256_json(input_closure),
    }
    if publication_guard is not None:
        publication_guard()
    return audit


def _stage16_parity_input_closure(
    *, root: Path, bundle: Path, metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a replay result to every byte that can affect its conclusion."""
    prediction = metadata.get("development_prediction")
    if not isinstance(prediction, Mapping):
        raise ModelSuiteError("Stage-16 parity lacks a prediction binding")
    return {
        "panel": file_binding(
            root, root / "data_usgs" / "panel_usgs_120v2.parquet"
        ),
        "frozen_panel_spec": file_binding(
            root, root / "data_usgs" / "frozen_panel_v1.json"
        ),
        "station_registry": file_binding(
            root, root / "data_usgs" / "station_registry_v1.csv"
        ),
        "bundle_metadata": file_binding(root, bundle / "metadata.json"),
        "bundle_weights": file_binding(root, bundle / "weights.pt"),
        "development_prediction": dict(prediction),
    }


def _validate_stage16_parity_audit(
    value: object,
    *,
    input_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Statically verify a previously executed, content-bound replay audit."""
    expected_keys = {
        "format", "status", "members", "splits", "atol",
        "max_abs_difference", "input_closure", "input_closure_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ModelSuiteError("Stage-16 five-member parity audit schema changed")
    try:
        tolerance = float(value["atol"])
        difference = float(value["max_abs_difference"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelSuiteError("Stage-16 five-member parity audit is malformed") from exc
    if (
        value.get("format") != STAGE16_PARITY_AUDIT_FORMAT
        or value.get("status") != "PASS_FIVE_MEMBER_VAL_CALIB_TEST_REPLAY"
        or tuple(value.get("members", ()))
        != tuple(f"seed{seed}" for seed in C.USGS_SEEDS)
        or tuple(value.get("splits", ())) != ("val", "calib", "test")
        or tolerance != 1e-5
        or not np.isfinite(difference)
        or difference < 0.0
        or difference > tolerance
        or value.get("input_closure") != dict(input_closure)
        or value.get("input_closure_sha256") != sha256_json(input_closure)
    ):
        raise ModelSuiteError("Stage-16 five-member parity audit changed")
    return dict(value)


def _stage16_expected_artifacts(
    *,
    root: Path,
    run_id: str,
    run_manifest: Path,
    stage09_receipt_path: Path,
    selection_path: Path,
    components_pointer: Path,
    identity: Mapping[str, Any],
    configuration: Mapping[str, Any],
    replay_selection: bool,
    replay_bundle: bool,
    publication_guard: Callable[[], object] | None,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any] | None,
    dict[str, Any] | None, dict[str, Any], dict[str, Any],
]:
    """Validate and bind Stage 16's complete authoritative file closure."""
    canonical = canonical_stage16_artifact_paths(run_id)
    for label, path in {
        "run_manifest": run_manifest,
        "stage09_completion_receipt": stage09_receipt_path,
        "lstm_validation_selection": selection_path,
        "components_pointer": components_pointer,
    }.items():
        exact = _stage16_exact_file(root, canonical[label], label=label)
        if _relative(root, path) != canonical[label] or path.resolve() != exact:
            raise ModelSuiteError(
                f"Stage-16 {label} path is not canonical"
            )
    if publication_guard is not None:
        publication_guard()
    stage09_pointer = root / STAGE9_COMPONENT_POINTER_PATH
    stage09_receipt = validate_stage09_completion_receipt(
        stage09_receipt_path,
        root=root,
        stage9_pointer=stage09_pointer,
        publication_guard=publication_guard,
    )
    stage09_artifacts = stage09_receipt.get("artifacts")
    if not isinstance(stage09_artifacts, Mapping):
        raise ModelSuiteError("Stage-16 Stage-9 parent receipt is malformed")
    parent_path = _stage16_exact_file(
        root,
        canonical["stage09_parent_predictions"],
        label="Stage-9 parent predictions",
    )
    parent_sidecar = _stage16_exact_file(
        root,
        canonical["stage09_parent_prediction_sidecar"],
        label="Stage-9 parent prediction sidecar",
    )
    if (
        stage09_artifacts.get("predictions") != file_binding(root, parent_path)
        or stage09_artifacts.get("prediction_sidecar")
        != file_binding(root, parent_sidecar)
        or parent_sidecar != sidecar_path(parent_path).resolve()
    ):
        raise ModelSuiteError(
            "Stage-16 parent is not the receipt-validated Stage-9 prediction"
        )
    parent_sha256 = sha256_file(parent_path)
    stage09_identity = stage09_receipt.get("run_identity")
    stage09_configuration = stage09_receipt.get("formal_configuration")
    if (
        not isinstance(stage09_identity, Mapping)
        or not isinstance(stage09_configuration, Mapping)
        or not isinstance(
            stage09_identity.get("input_closure_sha256"), str
        )
        or type(stage09_configuration.get("input_closure_file_count")) is not int
    ):
        raise ModelSuiteError(
            "Stage-16 Stage-9 parent lacks an input-closure contract"
        )
    expected_input_closure_sha256 = compose_input_closure_digest({
        "development": str(stage09_identity["input_closure_sha256"]),
        "stage09_parent_prediction": parent_sha256,
        "stage09_parent_sidecar": sha256_file(parent_sidecar),
        "stage09_completion_receipt": sha256_file(stage09_receipt_path),
        "stage09_components": sha256_file(stage09_pointer),
    })
    if (
        identity.get("input_closure_sha256")
        != expected_input_closure_sha256
        or configuration.get("input_closure_sha256")
        != expected_input_closure_sha256
        or configuration.get("input_closure_file_count")
        != int(stage09_configuration["input_closure_file_count"]) + 4
    ):
        raise ModelSuiteError(
            "Stage-16 run identity does not bind its complete Stage-9 input closure"
        )
    if publication_guard is not None:
        publication_guard()
    selection_records, selected_candidate = _read_stage16_selection(selection_path)
    replay_wd, replay_thresholds, threshold_contract = _stage16_replay_inputs(
        root,
        build_windows=replay_selection or replay_bundle,
    )

    selection_file_bindings: list[dict[str, str]] = []
    recomputed_metrics: list[float] = []
    checkpoint_metrics: list[float] = []
    replay_differences: list[float] = []
    for candidate_id, candidate_config in enumerate(LSTM_VALIDATION_GRID):
        candidate_path = _stage16_exact_file(
            root,
            canonical[f"candidate{candidate_id}_predictions"],
            label=f"candidate{candidate_id} predictions",
        )
        candidate_sidecar = _stage16_exact_file(
            root,
            canonical[f"candidate{candidate_id}_prediction_sidecar"],
            label=f"candidate{candidate_id} prediction sidecar",
        )
        checkpoint = _stage16_exact_file(
            root,
            canonical[f"candidate{candidate_id}_checkpoint"],
            label=f"candidate{candidate_id} checkpoint",
        )
        checkpoint_sidecar = _stage16_exact_file(
            root,
            canonical[f"candidate{candidate_id}_checkpoint_sidecar"],
            label=f"candidate{candidate_id} checkpoint sidecar",
        )
        expected_checkpoint_config = {
            **dict(configuration),
            "candidate_id": candidate_id,
            "candidate": dict(candidate_config),
        }
        expected_config_json = canonical_json(expected_checkpoint_config)
        expected_config_sha256 = sha256_json(expected_checkpoint_config)
        candidate_snapshot = _stage16_file_snapshot(
            candidate_path, label=f"candidate{candidate_id} predictions"
        )
        candidate_sidecar_snapshot = _stage16_file_snapshot(
            candidate_sidecar,
            label=f"candidate{candidate_id} prediction sidecar",
        )
        checkpoint_snapshot = _stage16_file_snapshot(
            checkpoint, label=f"candidate{candidate_id} checkpoint"
        )
        checkpoint_sidecar_snapshot = _stage16_file_snapshot(
            checkpoint_sidecar,
            label=f"candidate{candidate_id} checkpoint sidecar",
        )
        try:
            candidate_lineage = validate_artifact_sidecar(
                candidate_path,
                schema=R.PREDICTION_SCHEMA_VERSION,
                kind="lstm_validation_candidate_predictions",
            )
            candidate_snapshot_lineage = json.loads(
                candidate_sidecar_snapshot.payload.decode("utf-8")
            )
            candidate_frame = _stage16_read_prediction_frame(
                candidate_path,
                label=f"candidate{candidate_id} predictions",
                payload=candidate_snapshot.payload,
            )
            checkpoint_payload = torch.load(
                BytesIO(checkpoint_snapshot.payload),
                map_location="cpu",
                weights_only=True,
            )
            checkpoint_metadata = json.loads(
                checkpoint_sidecar_snapshot.payload.decode("utf-8")
            )
        except ModelSuiteError:
            raise
        except (
            OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError,
            RuntimeError, TypeError,
        ) as exc:
            raise ModelSuiteError(
                f"Stage-16 candidate{candidate_id} evidence is invalid"
            ) from exc
        _validate_stage16_candidate_checkpoint_payload(
            candidate_id=candidate_id,
            candidate_config=candidate_config,
            checkpoint_payload=checkpoint_payload,
            run_id=str(identity["run_id"]),
            expected_config_json=expected_config_json,
        )
        candidate_extra = candidate_lineage.get("extra")
        checkpoint_extra_json = (
            checkpoint_payload.get("extra_json")
            if isinstance(checkpoint_payload, Mapping) else None
        )
        if (
            candidate_sidecar != sidecar_path(candidate_path).resolve()
            or candidate_lineage != candidate_snapshot_lineage
            or candidate_lineage.get("artifact_sha256")
            != candidate_snapshot.sha256
            or candidate_lineage.get("artifact_bytes")
            != candidate_snapshot.size
            or checkpoint_sidecar != checkpoint_sidecar_path(checkpoint).resolve()
            or candidate_lineage.get("run") != dict(identity)
            or candidate_lineage.get("parents") != {}
            or candidate_extra != {
                "candidate_id": candidate_id,
                "candidate": dict(candidate_config),
                "selection_split": "2016-2017 validation",
            }
            or tuple(candidate_frame.columns) != tuple(R.PRED_COLS)
            or set(candidate_frame["model"].astype(str))
            != {f"LSTM-grid-{candidate_id}"}
            or set(candidate_frame["scope"].astype(str))
            != {"validation_selection"}
            or set(candidate_frame["feature_set"].astype(str)) != {"USGS"}
            or set(candidate_frame["seed"].astype(int))
            != {C.USGS_SEEDS[0]}
            or set(candidate_frame["split"].astype(str)) != {"val"}
            or not isinstance(checkpoint_payload, Mapping)
            or set(checkpoint_payload) != _CHECKPOINT_FIELDS
            or checkpoint_payload.get("format") != CHECKPOINT_VERSION
            or checkpoint_payload.get("run_id") != identity.get("run_id")
            or checkpoint_payload.get("resolved_config_json")
            != expected_config_json
            or checkpoint_payload.get("resolved_config_sha256")
            != expected_config_sha256
            or hashlib.sha256(expected_config_json.encode("utf-8")).hexdigest()
            != expected_config_sha256
            or not isinstance(checkpoint_extra_json, str)
            or checkpoint_payload.get("extra_sha256")
            != hashlib.sha256(checkpoint_extra_json.encode("utf-8")).hexdigest()
            or not isinstance(checkpoint_metadata, Mapping)
            or set(checkpoint_metadata) != _CHECKPOINT_METADATA_FIELDS
            or checkpoint_metadata.get("format") != CHECKPOINT_METADATA_VERSION
            or checkpoint_metadata.get("checkpoint_format")
            != CHECKPOINT_VERSION
            or checkpoint_metadata.get("run_id") != identity.get("run_id")
            or checkpoint_metadata.get("checkpoint_sha256")
            != checkpoint_snapshot.sha256
            or checkpoint_metadata.get("checkpoint_bytes")
            != checkpoint_snapshot.size
            or checkpoint_metadata.get("resolved_config_sha256")
            != expected_config_sha256
            or checkpoint_metadata.get("extra_sha256")
            != checkpoint_payload.get("extra_sha256")
            or checkpoint_metadata.get("epoch")
            != checkpoint_payload.get("epoch")
            or any(
                checkpoint_metadata.get(field) != checkpoint_payload.get(field)
                for field in (
                    "model_class", "optimizer_class", "scheduler_class",
                    "scheduler_present",
                )
            )
        ):
            raise ModelSuiteError(
                f"Stage-16 candidate{candidate_id} evidence has another lineage"
            )
        raw_best_metric = checkpoint_payload.get("best_metric")
        if (
            isinstance(raw_best_metric, (bool, np.bool_))
            or not isinstance(raw_best_metric, (float, np.floating))
            or not np.isfinite(float(raw_best_metric))
        ):
            raise ModelSuiteError(
                f"Stage-16 candidate{candidate_id} checkpoint metric is invalid"
            )
        station_rmse = (
            candidate_frame.assign(
                __squared_error=(
                    candidate_frame["y_pred"].to_numpy(float)
                    - candidate_frame["y_true"].to_numpy(float)
                ) ** 2
            )
            .groupby("site_id", sort=True)["__squared_error"]
            .mean()
            .pow(0.5)
        )
        recomputed = float(station_rmse.mean())
        reported = float(selection_records[candidate_id]["val_station_macro_rmse"])
        if (
            not np.isfinite(recomputed)
            or abs(recomputed - reported) > STAGE16_SELECTION_METRIC_ATOL
            or abs(recomputed - float(raw_best_metric))
            > STAGE16_SELECTION_METRIC_ATOL
        ):
            raise ModelSuiteError(
                f"Stage-16 candidate{candidate_id} validation metric changed"
            )
        recomputed_metrics.append(recomputed)
        checkpoint_metrics.append(float(raw_best_metric))
        if replay_selection:
            if replay_wd is None:  # pragma: no cover - guarded by construction
                raise ModelSuiteError("Stage-16 selection replay lacks windows")
            replay_differences.append(
                _verify_stage16_candidate_checkpoint(
                    candidate_id=candidate_id,
                    candidate_config=candidate_config,
                    checkpoint_payload=checkpoint_payload,
                    expected=candidate_frame,
                    wd=replay_wd,
                    thresholds=replay_thresholds,
                    publication_guard=publication_guard,
                )
            )
        selection_file_bindings.extend((
            _stage16_snapshot_binding(root, candidate_snapshot),
            _stage16_snapshot_binding(root, candidate_sidecar_snapshot),
            _stage16_snapshot_binding(root, checkpoint_snapshot),
            _stage16_snapshot_binding(root, checkpoint_sidecar_snapshot),
        ))
        for snapshot, label in (
            (candidate_snapshot, f"candidate{candidate_id} predictions"),
            (
                candidate_sidecar_snapshot,
                f"candidate{candidate_id} prediction sidecar",
            ),
            (checkpoint_snapshot, f"candidate{candidate_id} checkpoint"),
            (
                checkpoint_sidecar_snapshot,
                f"candidate{candidate_id} checkpoint sidecar",
            ),
        ):
            _stage16_assert_snapshot_current(snapshot, label=label)
        if publication_guard is not None:
            publication_guard()
    selected_ids = [
        int(record["candidate_id"])
        for record in selection_records if bool(record["selected"])
    ]
    reported_metrics = [
        float(record["val_station_macro_rmse"])
        for record in selection_records
    ]
    reported_winner = _stage16_consistent_validation_winner(
        reported_metrics, recomputed_metrics, checkpoint_metrics
    )
    if selected_ids != [reported_winner]:
        raise ModelSuiteError(
            "Stage-16 selected architecture is not the tolerance-aware "
            "reported validation winner"
        )
    selection_input_closure = _stage16_selection_input_closure(
        root=root,
        run_manifest=run_manifest,
        selection_path=selection_path,
        candidate_files=selection_file_bindings,
        threshold_contract=threshold_contract,
    )
    selection_audit = (
        _make_stage16_selection_audit(
            selection_records=selection_records,
            recomputed_metrics=recomputed_metrics,
            checkpoint_metrics=checkpoint_metrics,
            replay_differences=replay_differences,
            winner_candidate_id=reported_winner,
            input_closure=selection_input_closure,
        )
        if replay_selection else None
    )
    selection_context = {
        "selection_records": selection_records,
        "recomputed_metrics": recomputed_metrics,
        "checkpoint_metrics": checkpoint_metrics,
        "winner_candidate_id": reported_winner,
        "input_closure": selection_input_closure,
    }

    seed_frames: list[pd.DataFrame] = []
    seed_file_bindings: list[dict[str, str]] = []
    for seed in C.USGS_SEEDS:
        seed_path = _stage16_exact_file(
            root,
            canonical[f"seed{seed}_predictions"],
            label=f"seed{seed} predictions",
        )
        seed_sidecar = _stage16_exact_file(
            root,
            canonical[f"seed{seed}_prediction_sidecar"],
            label=f"seed{seed} prediction sidecar",
        )
        try:
            seed_lineage = validate_artifact_sidecar(
                seed_path,
                schema=R.PREDICTION_SCHEMA_VERSION,
                kind="lstm_seed_predictions",
            )
            seed_frame = _stage16_read_prediction_frame(
                seed_path, label=f"seed{seed} predictions"
            )
        except ModelSuiteError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelSuiteError(
                f"Stage-16 seed{seed} prediction cache is invalid"
            ) from exc
        if (
            seed_sidecar != sidecar_path(seed_path).resolve()
            or seed_lineage.get("run") != dict(identity)
            or seed_lineage.get("parents") != {}
            or seed_lineage.get("extra") != {}
            or set(seed_frame["model"].astype(str)) != {"LSTM"}
            or set(seed_frame["scope"].astype(str)) != {"joint_usgs"}
            or set(seed_frame["feature_set"].astype(str)) != {"USGS"}
            or set(seed_frame["seed"].astype(int)) != {seed}
            or set(seed_frame["split"].astype(str))
            != {"val", "calib", "test"}
        ):
            raise ModelSuiteError(
                f"Stage-16 seed{seed} prediction cache has another closure"
            )
        seed_frames.append(seed_frame.loc[:, R.PRED_COLS].copy())
        seed_file_bindings.extend((
            file_binding(root, seed_path),
            file_binding(root, seed_sidecar),
        ))
        if publication_guard is not None:
            publication_guard()

    prediction_path = _stage16_exact_file(
        root,
        canonical["development_predictions"],
        label="development predictions",
    )
    prediction_sidecar = _stage16_exact_file(
        root,
        canonical["development_prediction_sidecar"],
        label="development prediction sidecar",
    )
    if prediction_sidecar != sidecar_path(prediction_path).resolve():
        raise ModelSuiteError(
            "Stage-16 development prediction sidecar path is not canonical"
        )
    try:
        prediction_lineage = validate_artifact_sidecar(
            prediction_path,
            schema=R.PREDICTION_SCHEMA_VERSION,
            kind="final_route_a_development_predictions",
        )
    except (OSError, ValueError) as exc:
        raise ModelSuiteError(
            "Stage-16 development prediction sidecar is invalid"
        ) from exc
    extra = prediction_lineage.get("extra")
    primary_models = extra.get("primary_models") if isinstance(extra, Mapping) else None
    count_fields = (
        "primary_common_test_keys", "dropped_primary_rows",
        "lstm_validation_rows", "lstm_calibration_rows",
    )
    if (
        prediction_lineage.get("run") != dict(identity)
        or prediction_lineage.get("parents")
        != {Path(STAGE16_PARENT_PREDICTION_PATH).name: parent_sha256}
        or not isinstance(extra, Mapping)
        or set(extra) != {
            "parent_run_id", "primary_models", "primary_common_test_keys",
            "dropped_primary_rows", "lstm_validation_rows",
            "lstm_calibration_rows",
        }
        or extra.get("parent_run_id") != stage09_receipt.get("run_id")
        or isinstance(primary_models, (str, bytes))
        or not isinstance(primary_models, Sequence)
        or tuple(primary_models)
        != tuple(ROUTE_A_PRIMARY_MODELS)
        or any(
            isinstance(extra.get(field), (bool, np.bool_))
            or not isinstance(extra.get(field), (int, np.integer))
            or int(extra[field]) < 0
            for field in count_fields
        )
        or int(extra["primary_common_test_keys"]) < 1
        or int(extra["lstm_validation_rows"]) < 1
        or int(extra["lstm_calibration_rows"]) < 1
    ):
        raise ModelSuiteError(
            "Stage-16 development prediction lineage differs from its run"
        )
    if publication_guard is not None:
        publication_guard()

    parent_frame = _stage16_read_prediction_frame(
        parent_path, label="Stage-9 parent predictions"
    )
    prediction_frame = _stage16_read_prediction_frame(
        prediction_path, label="development predictions"
    )
    raw_lstm = pd.concat(seed_frames, ignore_index=True)
    retained_lstm = raw_lstm[
        raw_lstm["split"].isin(("val", "calib", "test"))
    ].copy()
    candidate = pd.concat(
        [
            parent_frame[parent_frame["model"].astype(str).ne("LSTM")],
            retained_lstm,
        ],
        ignore_index=True,
    )
    try:
        expected_prediction, common_audit = enforce_common_forecast_keys(
            candidate,
            ROUTE_A_PRIMARY_MODELS,
            split="test",
        )
    except (AssertionError, ValueError) as exc:
        raise ModelSuiteError(
            "Stage-16 parent and LSTM rows cannot reproduce the common-key V2"
        ) from exc
    if (
        len(expected_prediction) != len(prediction_frame)
        or canonical_frame_digest(expected_prediction, R.PRED_COLS)
        != canonical_frame_digest(prediction_frame, R.PRED_COLS)
        or int(extra["primary_common_test_keys"]) != common_audit.common_unique
        or int(extra["dropped_primary_rows"]) != common_audit.dropped_rows
        or int(extra["lstm_validation_rows"])
        != int(retained_lstm["split"].eq("val").sum())
        or int(extra["lstm_calibration_rows"])
        != int(retained_lstm["split"].eq("calib").sum())
    ):
        raise ModelSuiteError(
            "Stage-16 V2 is not the exact Stage-9-plus-LSTM derivation"
        )
    parent_non_lstm = parent_frame[
        parent_frame["model"].astype(str).ne("LSTM")
    ]
    final_non_lstm = prediction_frame[
        prediction_frame["model"].astype(str).ne("LSTM")
    ]
    if (
        common_audit.dropped_rows != 0
        or int(extra["dropped_primary_rows"]) != 0
        or len(parent_non_lstm) != len(final_non_lstm)
        or canonical_frame_digest(parent_non_lstm, R.PRED_COLS)
        != canonical_frame_digest(final_non_lstm, R.PRED_COLS)
    ):
        raise ModelSuiteError(
            "Stage-16 changed or deleted a receipt-frozen non-LSTM row"
        )
    if publication_guard is not None:
        publication_guard()

    components = load_component_pointer(components_pointer)
    expected_pointer_fields = {
        "format", "status", "training_device", "run_id", "cohort",
        "raw_feature_order", "models", "development_contract",
        "development_prediction_artifact",
    }
    entries = components.get("models")
    if (
        set(components) != expected_pointer_fields
        or components.get("cohort") != "temporal_lstm"
        or components.get("run_id") != run_id
        or components.get("training_device") != "cpu"
        or not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(entries[0], Mapping)
    ):
        raise ModelSuiteError(
            "Stage-16 component pointer is not an exact LSTM closure"
        )
    entry = dict(entries[0])
    feature_order = tuple(components.get("raw_feature_order", ()))
    if feature_order != STAGE9_USGS_VARIABLES:
        raise ModelSuiteError("Stage-16 raw feature order changed")
    expected_prediction_binding = {
        **file_binding(root, prediction_path),
        "sidecar": file_binding(root, prediction_sidecar),
    }
    if components.get("development_prediction_artifact") != expected_prediction_binding:
        raise ModelSuiteError(
            "Stage-16 component pointer binds another development prediction"
        )
    expected_development = canonical_development_contract(
        root,
        root / "data_usgs" / "frozen_panel_v1.json",
        panel_sha256=str(identity["panel_sha256"]),
        registry_sha256=str(identity["registry_sha256"]),
        source_sha256=str(identity["source_sha256"]),
    )
    if components.get("development_contract") != expected_development:
        raise ModelSuiteError(
            "Stage-16 component pointer binds another source/data contract"
        )
    if (
        entry.get("model_id") != "LSTM"
        or entry.get("executor") != "lstm_bundle"
        or entry.get("member_count") != 5
        or tuple(entry.get("raw_feature_order", ())) != feature_order
    ):
        raise ModelSuiteError("Stage-16 LSTM component entry is malformed")
    artifact = entry.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ModelSuiteError("Stage-16 LSTM bundle binding is malformed")
    bundle = _stage16_exact_directory(
        root, canonical["bundle"], label="LSTM bundle"
    )
    if (
        _relative(root, bundle) != canonical["bundle"]
        or bundle.is_symlink()
        or dict(artifact) != directory_binding(root, bundle)
    ):
        raise ModelSuiteError("Stage-16 LSTM bundle path or checksum changed")
    try:
        bundle_entries = list(bundle.iterdir())
    except OSError as exc:
        raise ModelSuiteError("Stage-16 LSTM bundle is unreadable") from exc
    files = {path.name: path for path in bundle_entries}
    if (
        len(files) != len(bundle_entries)
        or set(files) != {"metadata.json", "weights.pt"}
        or any(
            path.is_symlink() or not path.is_file()
            for path in bundle_entries
        )
    ):
        raise ModelSuiteError("Stage-16 LSTM bundle file closure changed")
    for name in ("metadata.json", "weights.pt"):
        files[name] = _stage16_exact_file(
            root,
            f"{canonical['bundle']}/{name}",
            label=f"LSTM bundle {name}",
        )
    metadata = _entry_artifact_valid(
        root,
        entry,
        feature_order,
        external=False,
        publication_guard=publication_guard,
    )
    if not isinstance(metadata, Mapping):
        raise ModelSuiteError("Stage-16 LSTM metadata is absent")
    architecture = metadata.get("architecture")
    kwargs = architecture.get("kwargs") if isinstance(architecture, Mapping) else None
    station_map = metadata.get("station_to_index")
    expected_station_map = {
        site: index for index, site in enumerate(C.STATIONS)
    }
    expected_kwargs = {
        "n_vars": len(feature_order),
        "n_stations": len(expected_station_map),
        "context": C.CONTEXT_LENGTH,
        "station_agnostic": False,
        **selected_candidate,
    }
    if (
        metadata.get("run_id") != run_id
        or metadata.get("source_sha256") != identity.get("source_sha256")
        or metadata.get("panel_sha256") != identity.get("panel_sha256")
        or metadata.get("registry_sha256") != identity.get("registry_sha256")
        or metadata.get("config_sha256") != identity.get("config_sha256")
        or metadata.get("runtime_sha256") != identity.get("runtime_sha256")
        or metadata.get("input_closure_sha256")
        != identity.get("input_closure_sha256")
        or metadata.get("training_device") != "cpu"
        or tuple(metadata.get("members", ()))
        != tuple(f"seed{seed}" for seed in C.USGS_SEEDS)
        or not isinstance(architecture, Mapping)
        or architecture.get("class")
        != "thermoroute.train.LSTMForecaster"
        or architecture.get("train_config")
        != asdict(C.TrainConfig(batch_size=1536))
        or station_map != expected_station_map
        or kwargs != expected_kwargs
    ):
        raise ModelSuiteError(
            "Stage-16 LSTM bundle metadata differs from its run or winner"
        )
    development_snapshot = _read_development_prediction_snapshot(
        root, metadata.get("development_prediction"), label="Stage-16 LSTM"
    )
    if (
        development_snapshot.sidecar.get("run") != dict(identity)
        or set(development_snapshot.selected["split"].astype(str))
        != {"val", "calib", "test"}
        or set(development_snapshot.selected["seed"].astype(int))
        != set(C.USGS_SEEDS)
        or any(
            set(group["split"].astype(str)) != {"val", "calib", "test"}
            for _, group in development_snapshot.selected.groupby("seed")
        )
    ):
        raise ModelSuiteError(
            "Stage-16 LSTM prediction binding lacks val/calib/test closure"
        )
    shortcut_path = _stage16_exact_file(
        root, canonical["shortcut_pointer"], label="shortcut pointer"
    )
    try:
        shortcut = json.loads(shortcut_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteError("Stage-16 LSTM shortcut pointer is malformed") from exc
    if shortcut != {
        "run_id": run_id,
        "bundle_path": canonical["bundle"],
        "member_count": 5,
        "metadata_sha256": sha256_file(files["metadata.json"]),
        "weights_sha256": sha256_file(files["weights.pt"]),
    }:
        raise ModelSuiteError("Stage-16 LSTM shortcut pointer changed")
    if publication_guard is not None:
        publication_guard()
    artifacts: dict[str, Any] = {
        "run_manifest": file_binding(root, run_manifest),
        "stage09_completion_receipt": file_binding(root, stage09_receipt_path),
        "stage09_parent_predictions": file_binding(root, parent_path),
        "stage09_parent_prediction_sidecar": file_binding(root, parent_sidecar),
        "lstm_validation_selection": file_binding(root, selection_path),
        "development_predictions": file_binding(root, prediction_path),
        "development_prediction_sidecar": file_binding(root, prediction_sidecar),
        "model_files": [
            file_binding(root, files[name])
            for name in ("metadata.json", "weights.pt")
        ],
        "lstm_seed_prediction_files": seed_file_bindings,
        "selection_candidate_files": selection_file_bindings,
        "shortcut_pointer": file_binding(root, shortcut_path),
        "components_pointer": file_binding(root, components_pointer),
    }
    parity_input_closure = _stage16_parity_input_closure(
        root=root, bundle=bundle, metadata=metadata
    )
    parity = (
        _verify_stage16_final_bundle_parity(
            root=root,
            bundle=bundle,
            expected=development_snapshot.selected,
            metadata=metadata,
            wd=replay_wd,
            publication_guard=publication_guard,
        )
        if replay_bundle else None
    )
    return (
        artifacts,
        entry,
        parity,
        selection_audit,
        parity_input_closure,
        selection_context,
    )


def build_stage16_completion_receipt(
    *,
    root: str | Path,
    run_id: str,
    run_manifest: str | Path,
    stage09_receipt: str | Path,
    selection: str | Path,
    components_pointer: str | Path,
    enforce_current_runtime: bool = True,
    publication_guard: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Build a deterministic receipt over the exact formal Stage-16 closure."""
    if publication_guard is not None:
        publication_guard()
    root = Path(root).resolve()
    run_manifest = Path(run_manifest).resolve()
    stage09_receipt = Path(stage09_receipt).resolve()
    selection = Path(selection).resolve()
    components_pointer = Path(components_pointer).resolve()
    parent = (root / STAGE16_PARENT_PREDICTION_PATH).resolve()
    parent_sha256 = sha256_file(parent)
    _, identity, configuration = _load_formal_stage16_manifest(
        run_manifest,
        root=root,
        run_id=str(run_id),
        parent_sha256=parent_sha256,
        enforce_current_runtime=enforce_current_runtime,
    )
    (
        artifacts, _, parity, selection_audit, _parity_inputs,
        _selection_context,
    ) = _stage16_expected_artifacts(
        root=root,
        run_id=str(run_id),
        run_manifest=run_manifest,
        stage09_receipt_path=stage09_receipt,
        selection_path=selection,
        components_pointer=components_pointer,
        identity=identity,
        configuration=configuration,
        replay_selection=True,
        replay_bundle=True,
        publication_guard=publication_guard,
    )
    if parity is None or selection_audit is None:  # pragma: no cover
        raise ModelSuiteError("Stage-16 build did not execute required replays")
    document: dict[str, Any] = {
        "format": STAGE16_COMPLETION_FORMAT,
        "status": STAGE16_COMPLETION_STATUS,
        "stage": "16_lstm_baseline_insample",
        "run_id": str(run_id),
        "parent_stage09_run_id": str(
            json.loads(stage09_receipt.read_text(encoding="utf-8"))["run_id"]
        ),
        "run_identity": identity,
        "formal_configuration": configuration,
        "training_device": "cpu",
        "confirmation_outcomes_requested_or_read": False,
        "selection_audit": selection_audit,
        "bundle_prediction_parity": parity,
        "artifacts": artifacts,
        "artifact_closure_sha256": sha256_json(artifacts),
    }
    document["receipt_self_sha256"] = sha256_json(document)
    if publication_guard is not None:
        publication_guard()
    return document


def write_stage16_completion_receipt(
    path: str | Path,
    document: Mapping[str, Any],
    *,
    publication_guard: Callable[[], object] | None = None,
) -> Path:
    """Atomically publish Stage 16's receipt as its final filesystem write."""
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    if document.get("receipt_self_sha256") != sha256_json(stable):
        raise ModelSuiteError("Stage-16 completion receipt self hash is invalid")
    destination = Path(path)
    atomic_write_json(
        destination,
        dict(document),
        publication_guard=publication_guard,
    )
    return destination


def validate_stage16_completion_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path,
    components_pointer: str | Path | None = None,
    document: Mapping[str, Any] | None = None,
    enforce_current_runtime: bool = True,
    replay_selection: bool = False,
    replay_bundle: bool = False,
    publication_guard: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Fail closed unless the standalone Stage-16 completion is exact.

    Formal callers hold the shared Stage-16 advisory lock, which defines the
    atomicity boundary for cooperating project writers.  Candidate evidence is
    additionally loaded and hashed from the same no-follow byte snapshots.
    Deliberate concurrent replacement by another process with the same OS user
    remains outside that cooperative-writer transaction model.
    """
    if publication_guard is not None:
        publication_guard()
    root = Path(root).resolve()
    receipt_path = _canonical_completion_receipt_path(
        root,
        receipt_path,
        relative=STAGE16_COMPLETION_RECEIPT_PATH,
        label="Stage-16 completion receipt",
        document_supplied=document is not None,
    )
    if document is None:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelSuiteError(
                "Stage-16 completion receipt is absent or invalid"
            ) from exc
    else:
        receipt = dict(document)
    expected_keys = {
        "format", "status", "stage", "run_id", "parent_stage09_run_id",
        "run_identity", "formal_configuration", "training_device",
        "confirmation_outcomes_requested_or_read", "selection_audit",
        "bundle_prediction_parity", "artifacts",
        "artifact_closure_sha256", "receipt_self_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ModelSuiteError("Stage-16 completion receipt schema is not exact")
    stable = {
        key: value for key, value in receipt.items()
        if key != "receipt_self_sha256"
    }
    if receipt.get("receipt_self_sha256") != sha256_json(stable):
        raise ModelSuiteError("Stage-16 completion receipt self hash changed")
    if (
        receipt.get("format") != STAGE16_COMPLETION_FORMAT
        or receipt.get("status") != STAGE16_COMPLETION_STATUS
        or receipt.get("stage") != "16_lstm_baseline_insample"
        or receipt.get("training_device") != "cpu"
        or receipt.get("confirmation_outcomes_requested_or_read") is not False
    ):
        raise ModelSuiteError("Stage-16 completion receipt is not a formal PASS")
    run_id = str(receipt.get("run_id", ""))
    if not run_id:
        raise ModelSuiteError("Stage-16 completion receipt lacks a run id")
    artifacts = receipt.get("artifacts")
    artifact_keys = {
        "run_manifest", "stage09_completion_receipt",
        "stage09_parent_predictions", "stage09_parent_prediction_sidecar",
        "lstm_validation_selection", "development_predictions",
        "development_prediction_sidecar", "model_files",
        "lstm_seed_prediction_files",
        "selection_candidate_files",
        "shortcut_pointer", "components_pointer",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_keys:
        raise ModelSuiteError("Stage-16 artifact closure schema is not exact")
    if receipt.get("artifact_closure_sha256") != sha256_json(artifacts):
        raise ModelSuiteError("Stage-16 artifact closure hash changed")
    canonical = canonical_stage16_artifact_paths(run_id)
    run_manifest = _validated_file_binding(
        root, artifacts["run_manifest"], label="Stage-16 run manifest"
    )
    stage09_receipt = _validated_file_binding(
        root,
        artifacts["stage09_completion_receipt"],
        label="Stage-16 Stage-9 completion receipt",
    )
    selection = _validated_file_binding(
        root,
        artifacts["lstm_validation_selection"],
        label="Stage-16 LSTM validation selection",
    )
    pointer = _validated_file_binding(
        root, artifacts["components_pointer"], label="Stage-16 component pointer"
    )
    if (
        _relative(root, run_manifest) != canonical["run_manifest"]
        or _relative(root, stage09_receipt)
        != canonical["stage09_completion_receipt"]
        or _relative(root, selection) != canonical["lstm_validation_selection"]
        or _relative(root, pointer) != canonical["components_pointer"]
    ):
        raise ModelSuiteError("Stage-16 receipt binds noncanonical artifacts")
    if components_pointer is not None and pointer != Path(
        components_pointer
    ).resolve():
        raise ModelSuiteError("Stage-16 receipt binds another component pointer")
    parent_sha256 = sha256_file(root / STAGE16_PARENT_PREDICTION_PATH)
    _, identity, configuration = _load_formal_stage16_manifest(
        run_manifest,
        root=root,
        run_id=run_id,
        parent_sha256=parent_sha256,
        enforce_current_runtime=enforce_current_runtime,
    )
    if identity != receipt.get("run_identity"):
        raise ModelSuiteError("Stage-16 completion run identity changed")
    if configuration != receipt.get("formal_configuration"):
        raise ModelSuiteError("Stage-16 completion configuration changed")
    (
        expected_artifacts, _, parity, selection_audit, parity_inputs,
        selection_context,
    ) = _stage16_expected_artifacts(
        root=root,
        run_id=run_id,
        run_manifest=run_manifest,
        stage09_receipt_path=stage09_receipt,
        selection_path=selection,
        components_pointer=pointer,
        identity=identity,
        configuration=configuration,
        replay_selection=replay_selection,
        replay_bundle=replay_bundle,
        publication_guard=publication_guard,
    )
    if dict(artifacts) != expected_artifacts:
        raise ModelSuiteError("Stage-16 exact artifact closure changed")
    validated_parity = _validate_stage16_parity_audit(
        receipt.get("bundle_prediction_parity"),
        input_closure=parity_inputs,
    )
    if parity is not None and validated_parity != parity:
        raise ModelSuiteError("Stage-16 five-member parity replay changed")
    validated_selection = _validate_stage16_selection_audit(
        receipt.get("selection_audit"),
        selection_records=selection_context["selection_records"],
        recomputed_metrics=selection_context["recomputed_metrics"],
        checkpoint_metrics=selection_context["checkpoint_metrics"],
        winner_candidate_id=selection_context["winner_candidate_id"],
        input_closure=selection_context["input_closure"],
    )
    if selection_audit is not None and validated_selection != selection_audit:
        raise ModelSuiteError("Stage-16 validation-selection replay changed")
    try:
        stage09_document = json.loads(stage09_receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteError("Stage-16 Stage-9 receipt changed") from exc
    if receipt.get("parent_stage09_run_id") != stage09_document.get("run_id"):
        raise ModelSuiteError("Stage-16 parent Stage-9 run changed")
    if publication_guard is not None:
        publication_guard()
    return receipt


def publish_stage16_completion_receipt(
    receipt_path: str | Path,
    document: Mapping[str, Any],
    *,
    root: str | Path,
    components_pointer: str | Path,
    publication_guard: Callable[[], object],
) -> Path:
    """Preflight Stage 16, atomically publish, then re-open its exact closure."""
    validate_stage16_completion_receipt(
        receipt_path,
        root=root,
        components_pointer=components_pointer,
        document=document,
        publication_guard=publication_guard,
    )
    publication_guard()
    destination = write_stage16_completion_receipt(
        receipt_path,
        document,
        publication_guard=publication_guard,
    )
    validate_stage16_completion_receipt(
        destination,
        root=root,
        components_pointer=components_pointer,
        publication_guard=publication_guard,
    )
    publication_guard()
    return destination


def stage16_completion_gate_binding(
    receipt_path: str | Path,
    *,
    root: str | Path,
    components_pointer: str | Path,
    enforce_current_runtime: bool = True,
    replay_selection: bool = False,
    replay_bundle: bool = False,
    publication_guard: Callable[[], object] | None = None,
) -> dict[str, str]:
    """Validate Stage 16 and return the exact binding frozen downstream."""
    validate_stage16_completion_receipt(
        receipt_path,
        root=root,
        components_pointer=components_pointer,
        enforce_current_runtime=enforce_current_runtime,
        replay_selection=replay_selection,
        replay_bundle=replay_bundle,
        publication_guard=publication_guard,
    )
    return file_binding(root, receipt_path)


def canonical_stage25_artifact_paths(run_id: str) -> dict[str, str]:
    """Return the only top-level paths admitted to a Stage-25 completion."""
    if not isinstance(run_id, str) or not run_id:
        raise ModelSuiteError("Stage-25 canonical paths require a run id")
    prediction = (
        "outputs/predictions/"
        f"external_pooled_development_{run_id}.parquet"
    )
    return {
        "run_manifest": (
            f"outputs/runs/25_external_pooled/{run_id}/run.json"
        ),
        "predictions": prediction,
        "prediction_sidecar": f"{prediction}.meta.json",
        "components_pointer": STAGE25_COMPONENT_POINTER_PATH,
        "thermoroute_bundle": (
            "outputs/models/"
            f"external_thermoroute_bundle_{run_id}"
        ),
        "lstm_bundle": (
            f"outputs/models/external_lstm_bundle_{run_id}"
        ),
        "lightgbm_manifest": (
            "outputs/models/"
            f"external_lightgbm_bundle_{run_id}/manifest.json"
        ),
    }


def _stage25_formal_configuration(
    resolved: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    """Validate the complete development-only Stage-25 run configuration."""
    expected_fields = {
        "stage", "role", "panel", "registry", "variables", "horizons",
        "seeds", "train_config", "preprocessing", "station_agnostic",
        "lstm_validation_grid", "lightgbm_validation_grid",
        "event_reference_fit_interval", "event_threshold_estimator",
        "post_2020_data_read", "training_device",
        "development_predictor_bridge", "formal_numerical_policy",
        "input_closure_sha256", "input_closure_file_count",
        "input_closure_component_count",
    }
    bridge = resolved.get("development_predictor_bridge")
    numerical_policy = resolved.get("formal_numerical_policy")
    input_closure_sha256 = resolved.get("input_closure_sha256")
    expected_threshold = {
        "method": "pooled_training_empirical_quantile_v1",
        "quantile": 0.90,
        "pool_weighting": "equal_weight_per_finite_training_row",
        "station_balanced": False,
    }
    if (
        set(resolved) != expected_fields
        or resolved.get("stage") != "25_train_external_pooled_suite"
        or resolved.get("role")
        != "prelabel_station_agnostic_development_training"
        or resolved.get("panel") != "panel_usgs_120v2.parquet"
        or resolved.get("registry") != "station_registry_v1.csv"
        or tuple(resolved.get("variables", ())) != STAGE9_USGS_VARIABLES
        or tuple(resolved.get("horizons", ())) != tuple(C.HORIZONS)
        or tuple(resolved.get("seeds", ())) != tuple(C.USGS_SEEDS)
        or resolved.get("train_config")
        != asdict(C.TrainConfig(batch_size=1536))
        or resolved.get("preprocessing")
        != "pooled_development_train_only"
        or resolved.get("station_agnostic") is not True
        or resolved.get("lstm_validation_grid")
        != [dict(value) for value in LSTM_VALIDATION_GRID]
        or resolved.get("lightgbm_validation_grid")
        != [dict(value) for value in STAGE9_LIGHTGBM_VALIDATION_GRID]
        or resolved.get("event_reference_fit_interval")
        != ["2006-01-01", "2018-12-31"]
        or resolved.get("event_threshold_estimator") != expected_threshold
        or resolved.get("post_2020_data_read") is not False
        or resolved.get("training_device") != "cpu"
        or not isinstance(bridge, Mapping)
        or set(bridge) != {"path", "sha256"}
        or not isinstance(numerical_policy, Mapping)
        or not numerical_policy
        or not isinstance(input_closure_sha256, str)
        or len(input_closure_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in input_closure_sha256
        )
        or type(resolved.get("input_closure_file_count")) is not int
        or resolved["input_closure_file_count"] < 1
        or resolved.get("input_closure_component_count") != 1
    ):
        raise ModelSuiteError(
            "Stage-25 run manifest has malformed formal configuration"
        )
    expected_bridge = development_predictor_bridge_binding(
        root,
        panel_sha256=sha256_file(
            root / "data_usgs" / "panel_usgs_120v2.parquet"
        ),
        registry_sha256=sha256_file(
            root / "data_usgs" / "station_registry_v1.csv"
        ),
    )
    if dict(bridge) != expected_bridge:
        raise ModelSuiteError(
            "Stage-25 configuration binds another development predictor bridge"
        )
    return json.loads(json.dumps(resolved, sort_keys=True))


def _load_formal_stage25_manifest(
    path: Path,
    *,
    root: Path,
    run_id: str,
    enforce_current_runtime: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], InputClosure]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteError(
            "Stage-25 receipt binds a malformed run manifest"
        ) from exc
    if not isinstance(manifest, dict):
        raise ModelSuiteError("Stage-25 receipt binds a malformed run manifest")
    if (
        set(manifest) != {
            "schema_version", "identity", "resolved_config", "created_utc",
            "environment", "git", "provenance",
        }
        or manifest.get("schema_version") != RUN_SCHEMA_VERSION
    ):
        raise ModelSuiteError(
            "Stage-25 run manifest schema is not exact"
        )
    identity, resolved = _validated_content_addressed_run_identity(
        manifest, label="Stage-25"
    )
    if (
        identity["run_id"] != run_id
        or identity["source_sha256"] != source_tree_hash(root)
        or identity["config_sha256"] != sha256_json(resolved)
    ):
        raise ModelSuiteError(
            "Stage-25 completion receipt is stale for the run or current source"
        )
    panel_path = root / "data_usgs" / "panel_usgs_120v2.parquet"
    registry_path = root / "data_usgs" / "station_registry_v1.csv"
    try:
        panel_sha256 = sha256_file(panel_path)
        registry_sha256 = sha256_file(registry_path)
    except OSError as exc:
        raise ModelSuiteError(
            "Stage-25 canonical panel or station registry is absent"
        ) from exc
    if (
        identity["panel_sha256"] != panel_sha256
        or identity["registry_sha256"] != registry_sha256
    ):
        raise ModelSuiteError(
            "Stage-25 identity differs from the canonical development data"
        )
    if enforce_current_runtime and identity["runtime_sha256"] != sha256_json(
        numerical_runtime_contract()
    ):
        raise ModelSuiteError(
            "Stage-25 identity differs from the current numerical runtime"
        )
    configuration = _stage25_formal_configuration(resolved, root=root)
    if configuration["input_closure_sha256"] != identity[
        "input_closure_sha256"
    ]:
        raise ModelSuiteError(
            "Stage-25 configuration and run identity bind different input closures"
        )
    input_closure = _verified_development_input_closure(
        root,
        identity=identity,
        configuration=configuration,
        label="Stage-25",
        composed=True,
    )
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != {"outcome_status", "training_device"}
        or provenance.get("outcome_status") != "NO_POST_2020_DATA_READ"
        or provenance.get("training_device") != "cpu"
    ):
        raise ModelSuiteError(
            "Stage-25 run manifest lacks development-only provenance"
        )
    return manifest, identity, configuration, input_closure


def _stage25_model_file_closure(
    root: Path,
    components: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    prediction_binding: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Validate every learned component and return its exact file closure."""
    expected_pointer_fields = {
        "format", "status", "training_device", "run_id", "cohort",
        "raw_feature_order", "models", "development_contract",
        "development_prediction_artifact",
    }
    if set(components) != expected_pointer_fields:
        raise ModelSuiteError("Stage-25 component pointer schema is not exact")
    if (
        components.get("format") != COMPONENT_POINTER_FORMAT
        or components.get("status") != "COMPLETE"
        or components.get("training_device") != "cpu"
        or components.get("cohort") != "external"
        or components.get("run_id") != identity.get("run_id")
    ):
        raise ModelSuiteError("Stage-25 component pointer is not complete")
    entries = components.get("models")
    if not isinstance(entries, list):
        raise ModelSuiteError("Stage-25 component registry is malformed")
    by_id = {
        str(entry.get("model_id")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if (
        len(by_id) != len(entries)
        or set(by_id) != set(STAGE25_REQUIRED_MODELS)
    ):
        raise ModelSuiteError("Stage-25 component registry is incomplete")
    feature_order = tuple(str(value) for value in components["raw_feature_order"])
    if feature_order != STAGE9_USGS_VARIABLES:
        raise ModelSuiteError("Stage-25 raw feature order changed")
    canonical = canonical_stage25_artifact_paths(str(identity["run_id"]))
    expected_components = {
        "ThermoRoute": ("thermoroute_bundle", canonical["thermoroute_bundle"]),
        "LSTM": ("lstm_bundle", canonical["lstm_bundle"]),
        "LightGBM": ("lightgbm_bundle", canonical["lightgbm_manifest"]),
    }
    closure: dict[str, dict[str, str]] = {}
    for model_id in STAGE25_REQUIRED_MODELS:
        entry = by_id[model_id]
        executor, expected_path = expected_components[model_id]
        if (
            entry.get("executor") != executor
            or int(entry.get("member_count", 0)) != 5
            or tuple(entry.get("raw_feature_order", ())) != feature_order
        ):
            raise ModelSuiteError(
                f"Stage-25 {model_id} registry entry is malformed"
            )
        metadata = _entry_artifact_valid(
            root, entry, feature_order, external=True
        )
        if not isinstance(metadata, Mapping):
            raise ModelSuiteError(f"Stage-25 {model_id} metadata is absent")
        if (
            metadata.get("run_id") != identity.get("run_id")
            or metadata.get("source_sha256") != identity.get("source_sha256")
            or metadata.get("panel_sha256") != identity.get("panel_sha256")
            or metadata.get("registry_sha256") != identity.get("registry_sha256")
            or metadata.get("config_sha256") != identity.get("config_sha256")
            or metadata.get("runtime_sha256") != identity.get("runtime_sha256")
            or metadata.get("input_closure_sha256")
            != identity.get("input_closure_sha256")
            or metadata.get("training_device") != "cpu"
        ):
            raise ModelSuiteError(
                f"Stage-25 {model_id} lineage differs from its run identity"
            )
        development = metadata.get("development_prediction")
        if (
            not isinstance(development, Mapping)
            or development.get("artifact") != prediction_binding
        ):
            raise ModelSuiteError(
                f"Stage-25 {model_id} binds another development prediction"
            )
        artifact = entry.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ModelSuiteError(f"Stage-25 {model_id} artifact is malformed")
        if executor in {"thermoroute_bundle", "lstm_bundle"}:
            if set(artifact) != {"path", "metadata_sha256", "weights_sha256"}:
                raise ModelSuiteError(
                    f"Stage-25 {model_id} bundle binding is malformed"
                )
            directory = _resolve_inside(root, artifact.get("path"), directory=True)
            if _relative(root, directory) != expected_path or directory.is_symlink():
                raise ModelSuiteError(
                    f"Stage-25 {model_id} bundle path is not canonical"
                )
            files = {
                path.relative_to(directory).as_posix(): path
                for path in directory.rglob("*")
                if path.is_file()
            }
            if set(files) != {"metadata.json", "weights.pt"} or any(
                path.is_symlink() for path in files.values()
            ):
                raise ModelSuiteError(
                    f"Stage-25 {model_id} bundle file closure changed"
                )
            expected_binding = directory_binding(root, directory)
            if dict(artifact) != expected_binding:
                raise ModelSuiteError(
                    f"Stage-25 {model_id} bundle checksum changed"
                )
            for path in files.values():
                binding = file_binding(root, path)
                closure[binding["path"]] = binding
            continue

        if set(artifact) != {"path", "sha256"}:
            raise ModelSuiteError("Stage-25 LightGBM binding is malformed")
        manifest_path = _validated_file_binding(
            root, artifact, label="Stage-25 LightGBM manifest"
        )
        if _relative(root, manifest_path) != expected_path:
            raise ModelSuiteError(
                "Stage-25 LightGBM manifest path is not canonical"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelSuiteError("Stage-25 LightGBM manifest is malformed") from exc
        model_registry = manifest.get("models") if isinstance(manifest, Mapping) else None
        expected_members = {f"seed{seed}" for seed in C.USGS_SEEDS}
        expected_horizons = {str(value) for value in C.HORIZONS}
        if not isinstance(model_registry, Mapping) or set(model_registry) != expected_members:
            raise ModelSuiteError("Stage-25 LightGBM member closure changed")
        expected_files = {manifest_path.resolve()}
        for member in expected_members:
            horizons = model_registry[member]
            if not isinstance(horizons, Mapping) or set(horizons) != expected_horizons:
                raise ModelSuiteError("Stage-25 LightGBM horizon closure changed")
            for horizon in expected_horizons:
                heads = horizons[horizon]
                if not isinstance(heads, Mapping) or set(heads) != set(LIGHTGBM_HEADS):
                    raise ModelSuiteError("Stage-25 LightGBM head closure changed")
                for head in LIGHTGBM_HEADS:
                    binding = heads[head]
                    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
                        raise ModelSuiteError(
                            "Stage-25 LightGBM model binding is malformed"
                        )
                    raw_path = Path(str(binding["path"]))
                    if raw_path.is_absolute() or len(raw_path.parts) != 1:
                        raise ModelSuiteError(
                            "Stage-25 LightGBM model path is not flat"
                        )
                    path = (manifest_path.parent / raw_path).resolve()
                    if (
                        path.parent != manifest_path.parent.resolve()
                        or not path.is_file()
                        or path.is_symlink()
                    ):
                        raise ModelSuiteError(
                            "Stage-25 LightGBM model path is not canonical"
                        )
                    if dict(binding) != {
                        "path": raw_path.as_posix(), "sha256": sha256_file(path)
                    }:
                        raise ModelSuiteError(
                            "Stage-25 LightGBM model checksum changed"
                        )
                    if path in expected_files:
                        raise ModelSuiteError(
                            "Stage-25 LightGBM file registry is duplicated"
                        )
                    expected_files.add(path)
        actual_files = {
            path.resolve()
            for path in manifest_path.parent.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files or any(
            path.is_symlink() for path in actual_files
        ):
            raise ModelSuiteError("Stage-25 LightGBM file closure changed")
        for path in actual_files:
            binding = file_binding(root, path)
            closure[binding["path"]] = binding
    if len(closure) != 80:
        raise ModelSuiteError(
            f"Stage-25 model file closure has {len(closure)} files, expected 80"
        )
    return [closure[path] for path in sorted(closure)]


def _stage25_expected_artifacts(
    *,
    root: Path,
    run_id: str,
    run_manifest: Path,
    components_pointer: Path,
    components: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = canonical_stage25_artifact_paths(run_id)
    prediction = components.get("development_prediction_artifact")
    if (
        not isinstance(prediction, Mapping)
        or set(prediction) != {"path", "sha256", "sidecar"}
        or not isinstance(prediction.get("sidecar"), Mapping)
    ):
        raise ModelSuiteError(
            "Stage-25 development prediction binding is malformed"
        )
    prediction_path = _validated_file_binding(
        root,
        {"path": prediction.get("path"), "sha256": prediction.get("sha256")},
        label="Stage-25 development predictions",
    )
    sidecar_path_value = _validated_file_binding(
        root, prediction["sidecar"], label="Stage-25 prediction sidecar"
    )
    if (
        _relative(root, prediction_path) != canonical["predictions"]
        or _relative(root, sidecar_path_value) != canonical["prediction_sidecar"]
        or sidecar_path_value != sidecar_path(prediction_path).resolve()
    ):
        raise ModelSuiteError(
            "Stage-25 development prediction path is not canonical"
        )
    try:
        prediction_lineage = validate_artifact_sidecar(
            prediction_path,
            schema=R.PREDICTION_SCHEMA_VERSION,
            kind="external_pooled_development_predictions",
        )
    except (OSError, ValueError) as exc:
        raise ModelSuiteError(
            "Stage-25 development prediction sidecar is invalid"
        ) from exc
    extra = prediction_lineage.get("extra")
    common_test_keys = extra.get("common_test_keys") if isinstance(extra, Mapping) else None
    if (
        prediction_lineage.get("run") != dict(identity)
        or prediction_lineage.get("parents") != {}
        or not isinstance(extra, Mapping)
        or set(extra) != {"common_test_keys", "post_2020_data_read"}
        or isinstance(common_test_keys, (bool, np.bool_))
        or not isinstance(common_test_keys, (int, np.integer))
        or int(common_test_keys) < 1
        or extra.get("post_2020_data_read") is not False
    ):
        raise ModelSuiteError(
            "Stage-25 development prediction lineage differs from its run"
        )
    development = components.get("development_contract")
    if not isinstance(development, Mapping):
        raise ModelSuiteError("Stage-25 development contract is malformed")
    expected_development = canonical_development_contract(
        root,
        root / "data_usgs" / "frozen_panel_v1.json",
        panel_sha256=str(identity["panel_sha256"]),
        registry_sha256=str(identity["registry_sha256"]),
        source_sha256=str(identity["source_sha256"]),
    )
    if dict(development) != expected_development:
        raise ModelSuiteError(
            "Stage-25 pointer binds another source/data contract"
        )
    model_files = _stage25_model_file_closure(
        root,
        components,
        identity=identity,
        prediction_binding=dict(prediction),
    )
    return {
        "run_manifest": file_binding(root, run_manifest),
        "components_pointer": file_binding(root, components_pointer),
        "development_predictions": file_binding(root, prediction_path),
        "development_prediction_sidecar": file_binding(root, sidecar_path_value),
        "frozen_panel_spec": dict(development["frozen_panel_spec"]),
        "development_panel": dict(development["panel"]),
        "station_registry": dict(development["registry"]),
        "development_predictor_bridge": dict(development["predictor_bridge"]),
        "model_files": model_files,
    }


def build_stage25_completion_receipt(
    *,
    root: str | Path,
    run_id: str,
    run_manifest: str | Path,
    components_pointer: str | Path,
    enforce_current_runtime: bool = True,
) -> dict[str, Any]:
    """Build a deterministic receipt over the exact Stage-25 file closure."""
    root = Path(root).resolve()
    run_manifest = Path(run_manifest).resolve()
    components_pointer = Path(components_pointer).resolve()
    canonical = canonical_stage25_artifact_paths(str(run_id))
    if (
        _relative(root, run_manifest) != canonical["run_manifest"]
        or _relative(root, components_pointer) != canonical["components_pointer"]
    ):
        raise ModelSuiteError("Stage-25 top-level artifact path is not canonical")
    _, identity, configuration, input_closure = _load_formal_stage25_manifest(
        run_manifest,
        root=root,
        run_id=str(run_id),
        enforce_current_runtime=enforce_current_runtime,
    )
    components = load_component_pointer(components_pointer)
    artifacts = _stage25_expected_artifacts(
        root=root,
        run_id=str(run_id),
        run_manifest=run_manifest,
        components_pointer=components_pointer,
        components=components,
        identity=identity,
    )
    document: dict[str, Any] = {
        "format": STAGE25_COMPLETION_FORMAT,
        "status": STAGE25_COMPLETION_STATUS,
        "stage": "25_train_external_pooled_suite",
        "run_id": str(run_id),
        "run_identity": identity,
        "formal_configuration": configuration,
        "training_device": "cpu",
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": artifacts,
        "artifact_closure_sha256": sha256_json(artifacts),
    }
    document["receipt_self_sha256"] = sha256_json(document)
    input_closure.assert_unchanged()
    return document


def write_stage25_completion_receipt(
    path: str | Path,
    document: Mapping[str, Any],
    *,
    publication_guard: Callable[[], object] | None = None,
) -> Path:
    """Atomically publish the Stage-25 receipt as the transaction's last write."""
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    if document.get("receipt_self_sha256") != sha256_json(stable):
        raise ModelSuiteError("Stage-25 completion receipt self hash is invalid")
    destination = Path(path)
    atomic_write_json(
        destination,
        dict(document),
        publication_guard=publication_guard,
    )
    return destination


def validate_stage25_completion_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path,
    components_pointer: str | Path | None = None,
    document: Mapping[str, Any] | None = None,
    enforce_current_runtime: bool = True,
) -> dict[str, Any]:
    """Fail closed unless the standalone Stage-25 completion is exact."""
    root = Path(root).resolve()
    receipt_path = _canonical_completion_receipt_path(
        root,
        receipt_path,
        relative=STAGE25_COMPLETION_RECEIPT_PATH,
        label="Stage-25 completion receipt",
        document_supplied=document is not None,
    )
    if document is None:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelSuiteError(
                "Stage-25 completion receipt is absent or invalid"
            ) from exc
    else:
        receipt = dict(document)
    expected_keys = {
        "format", "status", "stage", "run_id", "run_identity",
        "formal_configuration", "training_device",
        "confirmation_outcomes_requested_or_read", "artifacts",
        "artifact_closure_sha256", "receipt_self_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ModelSuiteError("Stage-25 completion receipt schema is not exact")
    stable = {
        key: value for key, value in receipt.items()
        if key != "receipt_self_sha256"
    }
    if receipt.get("receipt_self_sha256") != sha256_json(stable):
        raise ModelSuiteError("Stage-25 completion receipt self hash changed")
    if (
        receipt.get("format") != STAGE25_COMPLETION_FORMAT
        or receipt.get("status") != STAGE25_COMPLETION_STATUS
        or receipt.get("stage") != "25_train_external_pooled_suite"
        or receipt.get("training_device") != "cpu"
        or receipt.get("confirmation_outcomes_requested_or_read") is not False
    ):
        raise ModelSuiteError("Stage-25 completion receipt is not COMPLETE")
    run_id = str(receipt.get("run_id", ""))
    if not run_id:
        raise ModelSuiteError("Stage-25 completion receipt lacks a run id")
    artifacts = receipt.get("artifacts")
    artifact_keys = {
        "run_manifest", "components_pointer", "development_predictions",
        "development_prediction_sidecar", "frozen_panel_spec",
        "development_panel", "station_registry",
        "development_predictor_bridge", "model_files",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_keys:
        raise ModelSuiteError("Stage-25 artifact closure schema is not exact")
    if receipt.get("artifact_closure_sha256") != sha256_json(artifacts):
        raise ModelSuiteError("Stage-25 artifact closure hash changed")
    canonical = canonical_stage25_artifact_paths(run_id)
    run_manifest = _validated_file_binding(
        root, artifacts["run_manifest"], label="Stage-25 run manifest"
    )
    pointer = _validated_file_binding(
        root, artifacts["components_pointer"], label="Stage-25 component pointer"
    )
    if (
        _relative(root, run_manifest) != canonical["run_manifest"]
        or _relative(root, pointer) != canonical["components_pointer"]
    ):
        raise ModelSuiteError("Stage-25 receipt binds noncanonical artifacts")
    if components_pointer is not None and pointer != Path(
        components_pointer
    ).resolve():
        raise ModelSuiteError("Stage-25 receipt binds another component pointer")
    _, identity, configuration, input_closure = _load_formal_stage25_manifest(
        run_manifest,
        root=root,
        run_id=run_id,
        enforce_current_runtime=enforce_current_runtime,
    )
    if identity != receipt.get("run_identity"):
        raise ModelSuiteError("Stage-25 completion run identity changed")
    if configuration != receipt.get("formal_configuration"):
        raise ModelSuiteError("Stage-25 completion configuration changed")
    components = load_component_pointer(pointer)
    expected_artifacts = _stage25_expected_artifacts(
        root=root,
        run_id=run_id,
        run_manifest=run_manifest,
        components_pointer=pointer,
        components=components,
        identity=identity,
    )
    if dict(artifacts) != expected_artifacts:
        raise ModelSuiteError("Stage-25 exact artifact closure changed")
    input_closure.assert_unchanged()
    return receipt


def publish_stage25_completion_receipt(
    receipt_path: str | Path,
    document: Mapping[str, Any],
    *,
    root: str | Path,
    components_pointer: str | Path,
    publication_guard: Callable[[], object],
) -> Path:
    """Preflight the whole closure, atomically publish, then re-open it."""
    validate_stage25_completion_receipt(
        receipt_path,
        root=root,
        components_pointer=components_pointer,
        document=document,
    )
    publication_guard()
    destination = write_stage25_completion_receipt(
        receipt_path,
        document,
        publication_guard=publication_guard,
    )
    validate_stage25_completion_receipt(
        destination,
        root=root,
        components_pointer=components_pointer,
    )
    publication_guard()
    return destination


def stage25_completion_gate_binding(
    receipt_path: str | Path,
    *,
    root: str | Path,
    components_pointer: str | Path,
    enforce_current_runtime: bool = True,
) -> dict[str, str]:
    """Validate Stage 25 and return the exact binding frozen downstream."""
    validate_stage25_completion_receipt(
        receipt_path,
        root=root,
        components_pointer=components_pointer,
        enforce_current_runtime=enforce_current_runtime,
    )
    return file_binding(root, receipt_path)


def _entry_artifact_valid(
    root: Path,
    entry: Mapping[str, Any],
    feature_order: tuple[str, ...],
    *,
    external: bool,
    publication_guard: Callable[[], object] | None = None,
) -> Mapping[str, Any] | None:
    model_id, executor = str(entry.get("model_id")), str(entry.get("executor"))
    if tuple(entry.get("raw_feature_order", ())) != feature_order:
        raise ModelSuiteError(f"{model_id} raw feature order differs from suite")
    if model_id in BUILTIN_MODELS:
        if executor != "builtin" or "artifact" in entry:
            raise ModelSuiteError(f"{model_id} builtin entry is malformed")
        return None
    artifact = entry.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ModelSuiteError(f"{model_id} lacks an artifact binding")
    if executor == "lightgbm_bundle":
        path = _resolve_inside(root, artifact.get("path"))
        if sha256_file(path) != artifact.get("sha256"):
            raise ModelSuiteError("LightGBM manifest checksum mismatch")
        _, metadata = load_lightgbm_bundle(
            path,
            publication_guard=publication_guard,
        )
        if bool(metadata.get("station_agnostic")) != external:
            raise ModelSuiteError("LightGBM station-identity contract differs from cohort")
        if int(entry.get("member_count", 0)) != int(metadata.get("member_count", -1)):
            raise ModelSuiteError("LightGBM member count differs from suite entry")
        for field in (
            "source_sha256", "panel_sha256", "registry_sha256", "config_sha256",
            "runtime_sha256", "input_closure_sha256", "training_device",
        ):
            if not metadata.get(field):
                raise ModelSuiteError(f"LightGBM bundle lacks {field}")
        if metadata.get("training_device") != "cpu":
            raise ModelSuiteError("formal LightGBM bundle is not CPU-trained")
        _validate_cqr_metadata(metadata, label=model_id)
        _validate_calibration_fit_metadata(
            metadata, label=model_id, external=external
        )
        validate_development_prediction_binding(
            root, metadata.get("development_prediction"), label="LightGBM"
        )
        parity = metadata.get("roundtrip_parity")
        if not isinstance(parity, Mapping) or set(parity) != set(metadata.get("members", ())):
            raise ModelSuiteError("LightGBM native-text roundtrip parity is incomplete")
        for member in metadata["members"]:
            for horizon in metadata["horizons"]:
                heads = parity.get(member, {}).get(str(horizon), {})
                if set(heads) != set(LIGHTGBM_HEADS):
                    raise ModelSuiteError("LightGBM roundtrip parity lacks a model head")
                if any(float(value.get("max_abs_difference", np.inf)) > 1e-12
                       for value in heads.values()):
                    raise ModelSuiteError("LightGBM native-text roundtrip parity failed")
        return metadata
    if executor not in {"thermoroute_bundle", "lstm_bundle"}:
        raise ModelSuiteError(f"unsupported model executor: {executor}")
    directory = _resolve_inside(root, artifact.get("path"), directory=True)
    if sha256_file(directory / "metadata.json") != artifact.get("metadata_sha256"):
        raise ModelSuiteError(f"{model_id} metadata checksum mismatch")
    if sha256_file(directory / "weights.pt") != artifact.get("weights_sha256"):
        raise ModelSuiteError(f"{model_id} weights checksum mismatch")
    count = int(entry.get("member_count", 0))
    _, metadata = load_inference_bundle(
        directory,
        expected_member_count=count,
        publication_guard=publication_guard,
    )
    if tuple(metadata.get("feature_order", ())) != feature_order:
        raise ModelSuiteError(f"{model_id} sequence schema differs from suite")
    kwargs = metadata.get("architecture", {}).get("kwargs", {})
    if bool(kwargs.get("station_agnostic", False)) != external:
        raise ModelSuiteError(f"{model_id} station-identity contract differs from cohort")
    required_lineage = {
        "source_sha256", "panel_sha256", "registry_sha256", "config_sha256",
        "runtime_sha256", "input_closure_sha256", "training_device",
        "development_prediction",
    }
    missing = required_lineage - set(metadata)
    if missing:
        raise ModelSuiteError(f"{model_id} bundle lacks lineage: {sorted(missing)}")
    if metadata.get("training_device") != "cpu":
        raise ModelSuiteError(f"formal {model_id} bundle is not CPU-trained")
    _validate_cqr_metadata(metadata, label=model_id)
    _validate_calibration_fit_metadata(
        metadata, label=model_id, external=external
    )
    validate_development_prediction_binding(
        root, metadata.get("development_prediction"), label=model_id
    )
    # Loading a tensor dictionary is insufficient: instantiate the declared
    # executable class and demand a strict state-dict load for every member.
    from .frozen_inference import sequence_factory_from_metadata
    try:
        instantiate_inference_ensemble(
            directory,
            model_factory=lambda _member, bundle: sequence_factory_from_metadata(bundle),
            expected_member_count=count,
            device="cpu",
            publication_guard=publication_guard,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ModelSuiteError(f"{model_id} cannot be strictly reconstructed") from exc
    return metadata


def model_matrix_amendment_suite_binding(
    root: str | Path,
    *,
    live_git_tip_commit: str | None = None,
    publication_guard: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Rebuild the exact suite binding from the two frozen local JSON files.

    With a local Git repository, the amendment validator replays the complete
    document/seal lineage.  ``live_git_tip_commit`` additionally requires the
    seal creation to be a strict ancestor of that already-resolved commit.  In
    a Gitless archive this helper proves only exact local bytes and semantics;
    it deliberately makes no ancestry claim.
    """
    repository = Path(root).resolve()
    amendment_file = repository / MODEL_MATRIX_AMENDMENT_PATH
    seal_file = repository / MODEL_MATRIX_AMENDMENT_SEAL_PATH
    if publication_guard is not None:
        publication_guard()
    try:
        amendment_sha256 = sha256_file(amendment_file)
        seal_sha256 = sha256_file(seal_file)
        amendment = validate_model_matrix_amendment(
            MODEL_MATRIX_AMENDMENT_PATH,
            root=repository,
        )
        seal = validate_model_matrix_amendment_seal(
            MODEL_MATRIX_AMENDMENT_SEAL_PATH,
            root=repository,
            allow_gitless_archive=True,
            expected_amendment_sha256=amendment_sha256,
            expected_seal_sha256=seal_sha256,
            # The model-matrix validator treats each supplied descendant as a
            # strict step.  At Stage 24 this is the current pre-freeze HEAD;
            # chronology later binds the future model-freeze commit itself.
            model_freeze_commit=live_git_tip_commit,
        )
    except (FileNotFoundError, OSError, ModelMatrixAmendmentError) as exc:
        raise ModelSuiteError(
            "model suite cannot validate its frozen model-matrix amendment"
        ) from exc
    if publication_guard is not None:
        publication_guard()
    if (
        sha256_file(amendment_file) != amendment_sha256
        or sha256_file(seal_file) != seal_sha256
    ):
        raise ModelSuiteError(
            "model-matrix amendment bytes changed during suite validation"
        )
    stage09_matrix = amendment.get("stage09_architecture_control_matrix")
    stage09b_matrix = amendment.get("stage09b_development_control_matrix")
    if not isinstance(stage09_matrix, Mapping) or not isinstance(
        stage09b_matrix, Mapping
    ):
        raise ModelSuiteError("model-matrix amendment lacks its two matrix objects")
    return {
        "format": MODEL_MATRIX_SUITE_BINDING_FORMAT,
        "document": {
            "path": MODEL_MATRIX_AMENDMENT_PATH,
            "sha256": amendment_sha256,
            "format": MODEL_MATRIX_AMENDMENT_FORMAT,
            "status": MODEL_MATRIX_AMENDMENT_STATUS,
            "amendment_id": MODEL_MATRIX_AMENDMENT_ID,
            "amendment_document_commit": str(
                seal["amendment_document_commit"]
            ),
        },
        "seal": {
            "path": MODEL_MATRIX_AMENDMENT_SEAL_PATH,
            "sha256": seal_sha256,
            "format": MODEL_MATRIX_AMENDMENT_SEAL_FORMAT,
            "status": MODEL_MATRIX_AMENDMENT_SEAL_STATUS,
        },
        "contract_id": model_matrix_contract_id(
            stage09_matrix,
            stage09b_matrix,
        ),
    }


def validate_model_matrix_suite_binding(
    value: object,
    *,
    root: str | Path,
    publication_guard: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Validate an exact suite binding by independently rebuilding it."""
    if not isinstance(value, Mapping) or set(value) != set(
        MODEL_MATRIX_SUITE_BINDING_FIELDS
    ):
        raise ModelSuiteError("model suite model-matrix binding schema changed")
    document = value.get("document")
    seal = value.get("seal")
    if (
        not isinstance(document, Mapping)
        or set(document) != set(MODEL_MATRIX_SUITE_DOCUMENT_BINDING_FIELDS)
        or not isinstance(seal, Mapping)
        or set(seal) != set(MODEL_MATRIX_SUITE_SEAL_BINDING_FIELDS)
    ):
        raise ModelSuiteError("model suite model-matrix nested binding changed")
    expected = model_matrix_amendment_suite_binding(
        root,
        publication_guard=publication_guard,
    )
    if dict(value) != expected:
        raise ModelSuiteError(
            "model suite model-matrix bytes, identity, seal, commit, or contract changed"
        )
    return expected


def validate_model_suite_document(
    document: Mapping[str, Any],
    *,
    root: str | Path,
    publication_guard: Callable[[], object] | None = None,
) -> None:
    """Fail closed on a missing member, stale checksum, or wrong cohort contract."""
    if publication_guard is not None:
        publication_guard()
    root = Path(root).resolve()
    fields = set(document)
    if (
        fields != set(MODEL_SUITE_DOCUMENT_FIELDS)
        and fields != set(MODEL_SUITE_DOCUMENT_FIELDS) | {"versioned_suite"}
    ):
        raise ModelSuiteError("model suite top-level schema changed")
    if document.get("format") != MODEL_SUITE_FORMAT:
        raise ModelSuiteError("unsupported model suite format")
    if document.get("status") != "FROZEN_BEFORE_LABEL_OPENING":
        raise ModelSuiteError("model suite is not frozen")
    validate_model_matrix_suite_binding(
        document.get("model_matrix_amendment"),
        root=root,
        publication_guard=publication_guard,
    )
    versioned_binding = document.get("versioned_suite")
    if versioned_binding is not None:
        versioned_path = _validated_file_binding(
            root,
            versioned_binding,
            label="versioned model suite",
        )
        try:
            versioned_document = json.loads(
                versioned_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelSuiteError("versioned model suite is malformed") from exc
        alias_document = dict(document)
        alias_document.pop("versioned_suite")
        if (
            not isinstance(versioned_document, Mapping)
            or set(versioned_document) != set(MODEL_SUITE_DOCUMENT_FIELDS)
            or dict(versioned_document) != alias_document
        ):
            raise ModelSuiteError(
                "opening registry differs from its versioned model suite"
            )
    features = tuple(str(value) for value in document.get("actual_feature_order", ()))
    if not features or "WTEMP" not in features or len(features) != len(set(features)):
        raise ModelSuiteError("model suite feature order is invalid")
    development = document.get("development_contract")
    if not isinstance(development, Mapping):
        raise ModelSuiteError("model suite lacks canonical development contract")
    for label in ("frozen_panel_spec", "panel", "registry"):
        binding = development.get(label)
        if not isinstance(binding, Mapping):
            raise ModelSuiteError(f"development contract lacks {label} binding")
        path = _resolve_inside(root, binding.get("path"))
        if sha256_file(path) != binding.get("sha256"):
            raise ModelSuiteError(f"development {label} checksum mismatch")
    bridge = development.get("predictor_bridge")
    if not isinstance(bridge, Mapping):
        raise ModelSuiteError("development contract lacks predictor bridge binding")
    bridge_path = _resolve_inside(root, bridge.get("path"))
    expected_bridge = development_predictor_bridge_binding(
        root,
        panel_sha256=str(development["panel"]["sha256"]),
        registry_sha256=str(development["registry"]["sha256"]),
        path=bridge_path,
    )
    if dict(bridge) != expected_bridge:
        raise ModelSuiteError("development predictor bridge binding changed")
    source_digest = str(development.get("source_sha256", ""))
    if len(source_digest) != 64:
        raise ModelSuiteError("development contract lacks a source-tree SHA-256")
    if source_digest != source_tree_hash(root):
        raise ModelSuiteError(
            "development contract source-tree SHA-256 differs from current source"
        )
    cohorts = document.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {"temporal", "external"}:
        raise ModelSuiteError("model suite must contain temporal and external cohorts")
    suite_runtime_digests: set[str] = set()
    for name, required, is_external in (
        ("temporal", TEMPORAL_MODELS, False),
        ("external", EXTERNAL_MODELS, True),
    ):
        cohort = cohorts[name]
        entries = cohort.get("models") if isinstance(cohort, Mapping) else None
        if not isinstance(entries, list):
            raise ModelSuiteError(f"{name} model registry is malformed")
        ids = [str(entry.get("model_id")) for entry in entries
               if isinstance(entry, Mapping)]
        if len(ids) != len(entries) or set(ids) != set(required) or len(ids) != len(required):
            raise ModelSuiteError(
                f"{name} suite is incomplete: required={list(required)}, found={ids}"
            )
        by_id = {str(entry["model_id"]): entry for entry in entries}
        if int(by_id["ThermoRoute"].get("member_count", 0)) != 5:
            raise ModelSuiteError(f"{name} ThermoRoute must contain five members")
        if int(by_id["LSTM"].get("member_count", 0)) != 5:
            raise ModelSuiteError(f"{name} LSTM must contain five members")
        if int(by_id["LightGBM"].get("member_count", 0)) != 5:
            raise ModelSuiteError(f"{name} LightGBM must contain five members")
        loaded_metadata: dict[str, Mapping[str, Any]] = {}
        for model_id in required:
            expected_executor = (
                "builtin" if model_id in BUILTIN_MODELS else
                "lightgbm_bundle" if model_id == "LightGBM" else
                "lstm_bundle" if model_id == "LSTM" else "thermoroute_bundle"
            )
            if by_id[model_id].get("executor") != expected_executor:
                raise ModelSuiteError(f"{name}/{model_id} has wrong executor")
            if (
                model_id in MANDATORY_ABLATIONS
                and int(by_id[model_id].get("member_count", 0))
                != len(STAGE9_ABLATION_SEEDS)
            ):
                raise ModelSuiteError(
                    f"{model_id} must contain the complete five-seed control ensemble"
                )
            if model_id in MANDATORY_ABLATIONS:
                intervention = by_id[model_id].get("intervention")
                if intervention != ABLATION_INTERVENTIONS[model_id]:
                    raise ModelSuiteError(f"{model_id} intervention is not the frozen control")
            metadata = _entry_artifact_valid(
                root,
                by_id[model_id],
                features,
                external=is_external,
                publication_guard=publication_guard,
            )
            if publication_guard is not None:
                publication_guard()
            if metadata is not None:
                loaded_metadata[model_id] = metadata
        if not is_external:
            primary_kwargs = dict(
                loaded_metadata["ThermoRoute"].get("architecture", {}).get("kwargs", {})
            )
            for model_id in MANDATORY_ABLATIONS:
                control_kwargs = dict(
                    loaded_metadata[model_id].get("architecture", {}).get("kwargs", {})
                )
                expected = {**primary_kwargs, **ABLATION_INTERVENTIONS[model_id]}
                if control_kwargs != expected:
                    raise ModelSuiteError(
                        f"{model_id} architecture differs by more than its intervention"
                    )
        primary_metadata = loaded_metadata["ThermoRoute"]
        primary_preprocessing = primary_metadata.get("preprocessing")
        if not isinstance(primary_preprocessing, Mapping):
            raise ModelSuiteError(f"{name} ThermoRoute preprocessing is malformed")
        primary_event_reference = primary_metadata.get(
            "event_reference_climatology"
        )
        if not isinstance(primary_event_reference, Mapping):
            raise ModelSuiteError(
                f"{name} ThermoRoute event reference is malformed"
            )
        for model_id, metadata in loaded_metadata.items():
            if metadata.get("preprocessing") != primary_preprocessing:
                raise ModelSuiteError(
                    f"{name}/{model_id} preprocessing differs from primary ThermoRoute"
                )
            if metadata.get("event_reference_climatology") != primary_event_reference:
                raise ModelSuiteError(
                    f"{name}/{model_id} event reference differs from primary ThermoRoute"
                )
            calibrators = metadata.get("event_calibrators")
            if not isinstance(calibrators, Mapping) or set(calibrators) != {"1", "3", "7"}:
                raise ModelSuiteError(f"{name}/{model_id} event calibrators are incomplete")
            thresholds = metadata.get("event_thresholds")
            offsets = metadata.get("conformal_offsets")
            if not isinstance(thresholds, Mapping) or not isinstance(offsets, Mapping):
                raise ModelSuiteError(f"{name}/{model_id} calibration registry is malformed")
            _validate_cqr_metadata(metadata, label=f"{name}/{model_id}")
            if is_external:
                if set(thresholds) != {"__pooled__"}:
                    raise ModelSuiteError(f"external/{model_id} threshold is not pooled")
                if set(offsets) != {"__pooled__|1", "__pooled__|3", "__pooled__|7"}:
                    raise ModelSuiteError(f"external/{model_id} CQR offsets are not pooled")
                try:
                    P.validate_frozen_seasonal_event_reference(
                        primary_event_reference, pooled=True
                    )
                except ValueError as exc:
                    raise ModelSuiteError(
                        "external seasonal event reference is invalid"
                    ) from exc
            else:
                station_map = primary_metadata.get("station_to_index")
                if not isinstance(station_map, Mapping):
                    raise ModelSuiteError("temporal station registry is malformed")
                sites = set(str(site) for site in station_map)
                if set(thresholds) != sites:
                    raise ModelSuiteError(f"temporal/{model_id} threshold registry changed")
                expected_offsets = {
                    f"{site}|{horizon}" for site in sites for horizon in (1, 3, 7)
                }
                if set(offsets) != expected_offsets:
                    raise ModelSuiteError(f"temporal/{model_id} CQR registry is incomplete")
                try:
                    P.validate_frozen_seasonal_event_reference(
                        primary_event_reference,
                        expected_sites=sites,
                        pooled=False,
                    )
                except ValueError as exc:
                    raise ModelSuiteError(
                        "temporal seasonal event reference is invalid"
                    ) from exc
            validate_development_calibrated_head_gate(
                root,
                metadata,
                label=f"{name}/{model_id}",
                external=is_external,
            )
        lgb_metadata = loaded_metadata["LightGBM"]
        if is_external:
            if lgb_metadata.get("station_categories") != []:
                raise ModelSuiteError("external LightGBM retains station categories")
            if "station_code" in tuple(lgb_metadata.get("design_feature_order", ())):
                raise ModelSuiteError("external LightGBM design retains station_code")
        else:
            ordered_sites = [
                site for site, _index in sorted(
                    primary_metadata["station_to_index"].items(),
                    key=lambda value: int(value[1]),
                )
            ]
            if list(lgb_metadata.get("station_categories", ())) != ordered_sites:
                raise ModelSuiteError("temporal LightGBM categorical order changed")
            if tuple(lgb_metadata.get("design_feature_order", ()))[-1:] != ("station_code",):
                raise ModelSuiteError("temporal LightGBM design lacks station_code")
        for model_id, metadata in loaded_metadata.items():
            if metadata.get("panel_sha256") != development["panel"]["sha256"]:
                raise ModelSuiteError(f"{name}/{model_id} is bound to another panel")
            if metadata.get("registry_sha256") != development["registry"]["sha256"]:
                raise ModelSuiteError(f"{name}/{model_id} is bound to another registry")
            if metadata.get("source_sha256") != source_digest:
                raise ModelSuiteError(f"{name}/{model_id} is bound to another source tree")
            config_digest = str(metadata.get("config_sha256", ""))
            if len(config_digest) != 64:
                raise ModelSuiteError(f"{name}/{model_id} lacks a config SHA-256")
            runtime_digest = str(metadata.get("runtime_sha256", ""))
            if len(runtime_digest) != 64:
                raise ModelSuiteError(f"{name}/{model_id} lacks a runtime SHA-256")
            input_closure_digest = str(
                metadata.get("input_closure_sha256", "")
            )
            if not _is_sha256(input_closure_digest):
                raise ModelSuiteError(
                    f"{name}/{model_id} lacks an input-closure SHA-256"
                )
            if metadata.get("training_device") != "cpu":
                raise ModelSuiteError(f"{name}/{model_id} is not CPU-trained")
            suite_runtime_digests.add(runtime_digest)
    if len(suite_runtime_digests) != 1:
        raise ModelSuiteError(
            "model suite components were produced by different numerical runtimes"
        )
    runtime_digest = next(iter(suite_runtime_digests))
    if document.get("numerical_runtime_sha256") != runtime_digest:
        raise ModelSuiteError(
            "model suite numerical runtime is not the exact learned-metadata value"
        )
    if document.get("training_device") != "cpu":
        raise ModelSuiteError("formal model suite is not CPU-trained")
    gates = document.get("preopening_gates")
    # Security-tightening policy for the existing suite envelope: historical
    # two-gate documents are not grandfathered.  They must be rebuilt so the
    # exact external-training transaction is admitted by a Stage-25 receipt.
    required_gates = {
        "stage09_completion",
        "stage09b_development_controls",
        "stage16_lstm_completion",
        "stage25_external_completion",
    }
    if not isinstance(gates, Mapping) or set(gates) != required_gates:
        raise ModelSuiteError(
            "model suite lacks the Stage-9/09b/16/25 completion gates"
        )
    receipt_path = _validated_file_binding(
        root, gates["stage09_completion"], label="Stage-9 completion gate"
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        pointer_binding = receipt["artifacts"]["components_pointer"]
    except (
        OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError,
    ) as exc:
        raise ModelSuiteError("model suite Stage-9 completion gate is malformed") from exc
    if not isinstance(pointer_binding, Mapping):
        raise ModelSuiteError("model suite Stage-9 component binding is malformed")
    stage9_pointer = _resolve_inside(root, pointer_binding.get("path"))
    expected_gate = stage09_completion_gate_binding(
        receipt_path,
        root=root,
        stage9_pointer=stage9_pointer,
        publication_guard=publication_guard,
    )
    if publication_guard is not None:
        publication_guard()
    if dict(gates["stage09_completion"]) != expected_gate:
        raise ModelSuiteError("model suite Stage-9 completion binding changed")
    stage9 = load_component_pointer(stage9_pointer)
    temporal = cohorts["temporal"]
    temporal_entries = temporal.get("models") if isinstance(temporal, Mapping) else None
    if not isinstance(temporal_entries, list):
        raise ModelSuiteError("temporal model registry is malformed")
    _validate_stage09_suite_alignment(stage9, temporal_entries, development)

    controls_receipt_path = _validated_file_binding(
        root,
        gates["stage09b_development_controls"],
        label="Stage-09b development-controls completion gate",
    )
    try:
        controls_receipt = validate_stage09b_completion_receipt(
            controls_receipt_path,
            root=root,
            publication_guard=publication_guard,
        )
    except DevelopmentControlsGateError as exc:
        raise ModelSuiteError(
            "model suite Stage-09b development-controls gate failed"
        ) from exc
    if dict(gates["stage09b_development_controls"]) != file_binding(
        root, controls_receipt_path
    ):
        raise ModelSuiteError("model suite Stage-09b completion binding changed")
    controls_identity = controls_receipt.get("run_identity")
    controls_config = controls_receipt.get("formal_configuration")
    controls_artifacts = controls_receipt.get("artifacts")
    if (
        not isinstance(controls_identity, Mapping)
        or not isinstance(controls_config, Mapping)
        or not isinstance(controls_artifacts, Mapping)
        or controls_identity.get("panel_sha256") != development["panel"]["sha256"]
        or controls_identity.get("registry_sha256") != development["registry"]["sha256"]
        or controls_identity.get("source_sha256") != development["source_sha256"]
        or controls_identity.get("runtime_sha256") != runtime_digest
        or controls_config.get("development_predictor_bridge")
        != development["predictor_bridge"]
        or controls_artifacts.get("frozen_panel_spec")
        != development["frozen_panel_spec"]
        or controls_artifacts.get("panel") != development["panel"]
        or controls_artifacts.get("registry") != development["registry"]
        or controls_artifacts.get("predictor_bridge")
        != development["predictor_bridge"]
    ):
        raise ModelSuiteError(
            "model suite Stage-09b gate differs from its Stage-9 development closure"
        )

    stage16_receipt_path = _validated_file_binding(
        root,
        gates["stage16_lstm_completion"],
        label="Stage-16 LSTM completion gate",
    )
    try:
        stage16_candidate = json.loads(
            stage16_receipt_path.read_text(encoding="utf-8")
        )
        stage16_artifacts = stage16_candidate["artifacts"]
        stage16_pointer_binding = stage16_artifacts["components_pointer"]
    except (
        OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError,
    ) as exc:
        raise ModelSuiteError(
            "model suite Stage-16 completion gate is malformed"
        ) from exc
    if not isinstance(stage16_pointer_binding, Mapping):
        raise ModelSuiteError(
            "model suite Stage-16 component binding is malformed"
        )
    stage16_pointer = _resolve_inside(
        root, stage16_pointer_binding.get("path")
    )
    expected_stage16_gate = stage16_completion_gate_binding(
        stage16_receipt_path,
        root=root,
        components_pointer=stage16_pointer,
        enforce_current_runtime=False,
        publication_guard=publication_guard,
    )
    if dict(gates["stage16_lstm_completion"]) != expected_stage16_gate:
        raise ModelSuiteError("model suite Stage-16 completion binding changed")
    if (
        not isinstance(stage16_artifacts, Mapping)
        or stage16_artifacts.get("stage09_completion_receipt")
        != dict(gates["stage09_completion"])
    ):
        raise ModelSuiteError(
            "model suite Stage-16 gate binds another Stage-9 completion"
        )
    stage16 = load_component_pointer(stage16_pointer)
    stage16_entries = stage16.get("models")
    frozen_lstm = [
        entry for entry in temporal_entries
        if isinstance(entry, Mapping) and entry.get("model_id") == "LSTM"
    ]
    stage16_identity = stage16_candidate.get("run_identity")
    if (
        dict(stage16_pointer_binding) != file_binding(root, stage16_pointer)
        or not isinstance(stage16_entries, list)
        or len(stage16_entries) != 1
        or len(frozen_lstm) != 1
        or stage16_entries[0] != frozen_lstm[0]
        or stage16.get("development_contract") != development
        or not isinstance(stage16_identity, Mapping)
        or stage16_identity.get("panel_sha256")
        != development["panel"]["sha256"]
        or stage16_identity.get("registry_sha256")
        != development["registry"]["sha256"]
        or stage16_identity.get("source_sha256")
        != development["source_sha256"]
        or stage16_identity.get("runtime_sha256") != runtime_digest
        or stage16_candidate.get("parent_stage09_run_id")
        != receipt.get("run_id")
    ):
        raise ModelSuiteError(
            "model suite LSTM cohort differs from its Stage-16 completion"
        )
    if publication_guard is not None:
        publication_guard()

    stage25_receipt_path = _validated_file_binding(
        root,
        gates["stage25_external_completion"],
        label="Stage-25 external completion gate",
    )
    try:
        stage25_candidate = json.loads(
            stage25_receipt_path.read_text(encoding="utf-8")
        )
        stage25_pointer_binding = stage25_candidate["artifacts"][
            "components_pointer"
        ]
    except (
        OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError,
    ) as exc:
        raise ModelSuiteError(
            "model suite Stage-25 completion gate is malformed"
        ) from exc
    if not isinstance(stage25_pointer_binding, Mapping):
        raise ModelSuiteError(
            "model suite Stage-25 component binding is malformed"
        )
    stage25_pointer = _resolve_inside(
        root, stage25_pointer_binding.get("path")
    )
    stage25_receipt = validate_stage25_completion_receipt(
        stage25_receipt_path,
        root=root,
        components_pointer=stage25_pointer,
        # A self-contained suite must remain verifiable on another machine.
        # The receipt still binds the original runtime exactly; only --check
        # and the Stage-24 build require that runtime to be the current one.
        enforce_current_runtime=False,
    )
    if publication_guard is not None:
        publication_guard()
    if dict(gates["stage25_external_completion"]) != file_binding(
        root, stage25_receipt_path
    ):
        raise ModelSuiteError("model suite Stage-25 completion binding changed")
    stage25 = load_component_pointer(stage25_pointer)
    external_cohort = cohorts["external"]
    external_entries = (
        external_cohort.get("models")
        if isinstance(external_cohort, Mapping)
        else None
    )
    if not isinstance(external_entries, list):
        raise ModelSuiteError("external model registry is malformed")
    frozen_learned = {
        str(entry.get("model_id")): entry
        for entry in external_entries
        if isinstance(entry, Mapping)
        and str(entry.get("model_id")) not in BUILTIN_MODELS
    }
    stage25_learned = {
        str(entry.get("model_id")): entry
        for entry in stage25.get("models", ())
        if isinstance(entry, Mapping)
    }
    stage25_identity = stage25_receipt.get("run_identity")
    if (
        dict(stage25_pointer_binding) != file_binding(root, stage25_pointer)
        or frozen_learned != stage25_learned
        or stage25.get("development_contract") != development
        or not isinstance(stage25_identity, Mapping)
        or stage25_identity.get("panel_sha256")
        != development["panel"]["sha256"]
        or stage25_identity.get("registry_sha256")
        != development["registry"]["sha256"]
        or stage25_identity.get("source_sha256")
        != development["source_sha256"]
        or stage25_identity.get("runtime_sha256") != runtime_digest
    ):
        raise ModelSuiteError(
            "model suite external cohort differs from its Stage-25 completion"
        )
    if publication_guard is not None:
        publication_guard()


def _learned_metadata_runtime_sha256(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    publication_guard: Callable[[], object] | None = None,
) -> str:
    """Derive one runtime digest from every learned artifact's own metadata."""
    digests: set[str] = set()
    learned = 0
    for entry in entries:
        if str(entry.get("model_id")) in BUILTIN_MODELS:
            continue
        artifact = entry.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ModelSuiteError("learned suite entry lacks an artifact binding")
        executor = str(entry.get("executor"))
        if executor == "lightgbm_bundle":
            path = _resolve_inside(root, artifact.get("path"))
            _, metadata = load_lightgbm_bundle(
                path,
                publication_guard=publication_guard,
            )
        elif executor in {"thermoroute_bundle", "lstm_bundle"}:
            directory = _resolve_inside(root, artifact.get("path"), directory=True)
            _, metadata = load_inference_bundle(
                directory,
                expected_member_count=int(entry.get("member_count", 0)),
                publication_guard=publication_guard,
            )
        else:
            raise ModelSuiteError(f"unsupported learned executor: {executor}")
        if publication_guard is not None:
            publication_guard()
        learned += 1
        digest = str(metadata.get("runtime_sha256", ""))
        if len(digest) != 64:
            raise ModelSuiteError("learned model metadata lacks a runtime SHA-256")
        if metadata.get("training_device") != "cpu":
            raise ModelSuiteError("formal suite contains a non-CPU learned artifact")
        digests.add(digest)
    if learned < 1 or len(digests) != 1:
        raise ModelSuiteError(
            "model suite is incomplete or learned metadata does not identify "
            "one numerical runtime"
        )
    return next(iter(digests))


def _authority_output_path(
    root: Path, path: str | Path, *, label: str,
) -> Path:
    """Return an in-root lexical output path without dereferencing aliases."""
    raw = Path(path)
    if not raw.is_absolute():
        raw = root / raw
    lexical = Path(os.path.abspath(raw))
    if lexical == root or root not in lexical.parents:
        raise ModelSuiteError(f"{label} path escapes repository")
    current = lexical
    while current != root:
        if current.is_symlink():
            raise ModelSuiteError(f"{label} path uses a symlink")
        current = current.parent
    if lexical.exists() or lexical.is_symlink():
        try:
            metadata = lexical.lstat()
        except OSError as exc:  # pragma: no cover - exists/lstat race
            raise ModelSuiteError(f"{label} is unreadable") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ModelSuiteError(f"{label} is not a single-link regular file")
    return lexical


def freeze_model_suite(
    destination: str | Path,
    current_pointer: str | Path,
    *,
    root: str | Path,
    protocol_sha256: str,
    temporal_entries: Sequence[Mapping[str, Any]],
    external_entries: Sequence[Mapping[str, Any]],
    actual_feature_order: Sequence[str],
    development_contract: Mapping[str, Any],
    model_matrix_amendment: Mapping[str, Any] | None = None,
    stage09_completion: Mapping[str, Any] | None = None,
    stage09b_completion: Mapping[str, Any] | None = None,
    stage16_completion: Mapping[str, Any] | None = None,
    stage25_completion: Mapping[str, Any] | None = None,
    registry_alias: str | Path | None = None,
    publication_guard: Callable[[], object],
) -> Path:
    """Write the versioned suite, then (and only then) publish its current pointer."""
    publication_guard()
    root = Path(root).resolve()
    destination = _authority_output_path(
        root, destination, label="versioned model suite"
    )
    current_pointer = _authority_output_path(
        root, current_pointer, label="current model-suite pointer"
    )
    learned_runtime_sha256 = _learned_metadata_runtime_sha256(
        root,
        [*temporal_entries, *external_entries],
        publication_guard=publication_guard,
    )
    document = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": learned_runtime_sha256,
        "protocol_sha256": str(protocol_sha256),
        "actual_feature_order": list(actual_feature_order),
        "development_contract": dict(development_contract),
        "model_matrix_amendment": (
            dict(model_matrix_amendment)
            if model_matrix_amendment is not None
            else None
        ),
        **(
            {"preopening_gates": {
                "stage09_completion": dict(stage09_completion),
                "stage09b_development_controls": dict(stage09b_completion),
                "stage16_lstm_completion": dict(stage16_completion),
                "stage25_external_completion": dict(stage25_completion),
            }}
            if (
                stage09_completion is not None
                and stage09b_completion is not None
                and stage16_completion is not None
                and stage25_completion is not None
            )
            else {}
        ),
        "cohorts": {
            "temporal": {
                "site_mode": "same_station",
                "models": [dict(value) for value in temporal_entries],
            },
            "external": {
                "site_mode": "station_agnostic_history_dependent_new_site",
                "models": [dict(value) for value in external_entries],
            },
        },
    }
    # Validate before any current pointer can exist.
    validate_model_suite_document(
        document,
        root=root,
        publication_guard=publication_guard,
    )
    publication_guard()
    _create_json_or_require_identical(
        destination,
        document,
        publication_guard=publication_guard,
    )
    validate_model_suite_document(
        json.loads(destination.read_text(encoding="utf-8")),
        root=root,
        publication_guard=publication_guard,
    )
    publication_guard()
    current_suite = destination
    if registry_alias is not None:
        alias = _authority_output_path(
            root, registry_alias, label="opening model-suite registry"
        )
        alias_document = {
            **document,
            "versioned_suite": file_binding(root, destination),
        }
        validate_model_suite_document(
            alias_document,
            root=root,
            publication_guard=publication_guard,
        )
        publication_guard()
        _create_json_or_require_identical(
            alias,
            alias_document,
            publication_guard=publication_guard,
        )
        validate_model_suite_document(
            json.loads(alias.read_text(encoding="utf-8")),
            root=root,
            publication_guard=publication_guard,
        )
        publication_guard()
        current_suite = alias
    pointer = {
        "format": MODEL_SUITE_POINTER_FORMAT,
        "status": "CURRENT_COMPLETE",
        "suite_path": _relative(root, current_suite),
        "suite_sha256": sha256_file(current_suite),
    }
    atomic_write_json(
        current_pointer,
        pointer,
        publication_guard=publication_guard,
    )
    resolved = resolve_model_suite_pointer(
        current_pointer,
        root=root,
        publication_guard=publication_guard,
    )
    if resolved != current_suite:
        raise ModelSuiteError("published model-suite pointer resolves elsewhere")
    publication_guard()
    return destination


def _create_json_or_require_identical(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    publication_guard: Callable[[], object] | None = None,
) -> None:
    """Stage and create-only publish JSON; retries must be byte-identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.read_bytes() != payload
        ):
            raise FileExistsError(f"refusing to replace frozen model registry: {path}")
        if publication_guard is not None:
            publication_guard()
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        if publication_guard is not None:
            publication_guard()
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_nlink != 1
                or path.read_bytes() != payload
            ):
                raise FileExistsError(
                    f"refusing to replace frozen model registry: {path}"
                ) from None
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_nlink != 1
        or path.read_bytes() != payload
    ):
        raise FileExistsError(f"frozen model registry publication failed: {path}")


def resolve_model_suite_pointer(
    path: str | Path,
    *,
    root: str | Path,
    publication_guard: Callable[[], object] | None = None,
) -> Path:
    if publication_guard is not None:
        publication_guard()
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != MODEL_SUITE_POINTER_FORMAT:
        raise ModelSuiteError("unsupported model-suite pointer format")
    suite = _resolve_inside(Path(root), document.get("suite_path"))
    if sha256_file(suite) != document.get("suite_sha256"):
        raise ModelSuiteError("model-suite pointer checksum mismatch")
    validate_model_suite_document(
        json.loads(suite.read_text(encoding="utf-8")),
        root=root,
        publication_guard=publication_guard,
    )
    if publication_guard is not None:
        publication_guard()
    return suite
