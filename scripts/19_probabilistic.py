#!/usr/bin/env python3
"""Lineage-closed Route-A development probabilistic verification.

This stage scores the previously inspected 2019--2020 development evaluation.
It never fits a calibrator, a conformal offset, an event threshold, or an event
reference.  Those objects are loaded from the formal model bundles and must be
bound to the same source tree, panel, registry, numerical runtime and prediction
lineage as this evaluation.

The metric contract is deliberately explicit:

* q05/q50/q95 pinball losses use the unmodified ensemble quantiles;
* the reported three-quantile score is their unscaled arithmetic mean, not CRPS;
* coverage, width and Winkler use the frozen, widens-only CQR endpoints;
* probabilities use the frozen 2018 Platt calibrators;
* Brier skill uses the bundle-frozen 2006--2018 seasonal event reference;
* every retained station contributes the same total weight; and
* all compared models share exact forecast keys and byte-exact ``y_true`` values.

Outputs are useful only together with their lineage sidecars and the self-hashed
receipt written last by this script.  A same-named file from an older run is not
accepted as evidence merely because it exists.
"""
# ruff: noqa: E402
from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence, cast
import warnings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from thermoroute import config as C
from thermoroute import conformal as CF
from thermoroute import results as R
from thermoroute.checkpoint import load_inference_bundle
from thermoroute.input_closure import compose_input_closure_digest
from thermoroute.model_suite import (
    ModelSuiteError,
    STAGE16_COMPLETION_RECEIPT_PATH,
    file_binding,
    load_component_pointer,
    load_lightgbm_bundle,
    route_a_calibration_fit_contract,
    validate_stage16_completion_receipt,
)
from thermoroute.probability import (
    PlattCalibrator,
    logit,
    predict_frozen_seasonal_event_reference,
    validate_frozen_seasonal_event_reference,
)
from thermoroute.registry import ROUTE_A_PRIMARY_MODELS
from thermoroute.repro import (
    advisory_file_lock,
    assert_formal_numerical_policy,
    atomic_write_bytes,
    atomic_write_json,
    configure_deterministic_runtime,
    numerical_runtime_contract,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sha256_json,
    sidecar_path,
    source_tree_hash,
    validate_artifact_sidecar,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry


configure_deterministic_runtime()


PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
PREDICTIONS = C.PREDICTIONS / "usgs_predictions_v2.parquet"
STAGE16_RECEIPT = ROOT / STAGE16_COMPLETION_RECEIPT_PATH
STAGE9_COMPONENTS = C.MODELS / "route_a_stage9_components.json"
LSTM_COMPONENTS = C.MODELS / "route_a_lstm_components.json"
PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"

PROBABILITY_SCORES = C.TABLES / "probabilistic_scores.csv"
POINT_SCORES = C.TABLES / "multi_metric.csv"
CALIBRATION_AUDIT = C.TABLES / "event_calibrators.json"
RELIABILITY_FIGURE = C.FIGURES / "fig_reliability.png"
REPORT = C.REPORTS / "probabilistic.md"
RECEIPT = C.TABLES / "probabilistic_evaluation_v2.json"

PROB_MODELS = ("ThermoRoute", "LightGBM", "LSTM")
POINT_MODELS = (
    "Persistence", "DampedPersistence", "LightGBM", "LSTM", "ThermoRoute"
)
COLORS = {"ThermoRoute": "#B3132B", "LightGBM": "#185FA5", "LSTM": "#6A4C93"}
IDENTITY = ("site_id", "horizon", "issue_date", "target_date")
HORIZONS = (1, 3, 7)
MINIMUM_SITE_HORIZON_TARGETS = 100

RECEIPT_FORMAT = "thermoroute.development-probabilistic-evaluation.v2"
CALIBRATION_AUDIT_FORMAT = "thermoroute.frozen-probability-audit.v2"
SCORE_CONTENT_SCHEMA = "thermoroute.development-probabilistic-scores.v2"
POINT_CONTENT_SCHEMA = "thermoroute.development-point-scores.v2"
CALIBRATION_CONTENT_SCHEMA = "thermoroute.frozen-probability-audit.v2"
REPORT_CONTENT_SCHEMA = "thermoroute.development-probabilistic-report.v2"
FIGURE_CONTENT_SCHEMA = "thermoroute.development-reliability-figure.v2"

OUTPUT_CONTENT_SCHEMAS: Mapping[str, str] = {
    "probability_scores": SCORE_CONTENT_SCHEMA,
    "point_scores": POINT_CONTENT_SCHEMA,
    "calibration_audit": CALIBRATION_CONTENT_SCHEMA,
    "reliability_figure": FIGURE_CONTENT_SCHEMA,
    "report": REPORT_CONTENT_SCHEMA,
}

CANONICAL_RECEIPT_PATH = "outputs/tables/probabilistic_evaluation_v2.json"
CANONICAL_INPUT_PATHS: Mapping[tuple[str, ...], str] = {
    ("prediction", "artifact"): "outputs/predictions/usgs_predictions_v2.parquet",
    ("prediction", "lineage_sidecar"): (
        "outputs/predictions/usgs_predictions_v2.parquet.meta.json"
    ),
    ("stage16_completion_receipt",): STAGE16_COMPLETION_RECEIPT_PATH,
    ("panel",): "data_usgs/panel_usgs_120v2.parquet",
    ("registry",): "data_usgs/station_registry_v1.csv",
    ("protocol",): "protocols/route_a_confirmatory_v1.json",
    ("component_pointers", "stage9"): (
        "outputs/models/route_a_stage9_components.json"
    ),
    ("component_pointers", "lstm"): (
        "outputs/models/route_a_lstm_components.json"
    ),
}
CANONICAL_OUTPUT_PATHS: Mapping[str, str] = {
    "probability_scores": "outputs/tables/probabilistic_scores.csv",
    "point_scores": "outputs/tables/multi_metric.csv",
    "calibration_audit": "outputs/tables/event_calibrators.json",
    "reliability_figure": "outputs/figures/fig_reliability.png",
    "report": "outputs/reports/probabilistic.md",
}

RECEIPT_TOP_LEVEL_FIELDS = {
    "format",
    "status",
    "scientific_role",
    "inference_computed",
    "run_identity",
    "lineage",
    "inputs",
    "contract",
    "contract_sha256",
    "event_reference_sha256",
    "probability_registry_audit",
    "point_registry_audit",
    "artifacts",
    "receipt_self_sha256",
}

class ProbabilityContractError(RuntimeError):
    """A frozen probability input or derived evaluation is inconsistent."""


@dataclass(frozen=True)
class FrozenModelContract:
    """Verified probability metadata and its immutable artifact bindings."""

    model_id: str
    executor: str
    member_count: int
    prediction_seeds: tuple[int, ...]
    metadata: Mapping[str, Any]
    pointer: Mapping[str, str]
    metadata_file: Mapping[str, str]
    model_files: tuple[Mapping[str, str], ...]


def _inside(root: Path, value: object, *, directory: bool = False) -> Path:
    """Resolve one repository-relative path without permitting path escape."""
    raw = Path(str(value))
    if raw.is_absolute():
        raise ProbabilityContractError(f"artifact path must be relative: {raw}")
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ProbabilityContractError(f"artifact path escapes repository: {raw}")
    if directory and not path.is_dir():
        raise ProbabilityContractError(f"bundle directory is absent: {raw}")
    if not directory and not path.is_file():
        raise ProbabilityContractError(f"artifact file is absent: {raw}")
    return path


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilityContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ProbabilityContractError(f"{label} must be a JSON object")
    return value


def _model_file_bindings(root: Path, directory: Path) -> tuple[Mapping[str, str], ...]:
    """Bind every regular file in a flat/content-addressed bundle directory."""
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ProbabilityContractError(f"model bundle is empty: {directory}")
    return tuple(file_binding(root, path) for path in files)


def _prediction_seed_registry(
    metadata: Mapping[str, Any], *, model_id: str, member_count: int
) -> tuple[int, ...]:
    """Map exact frozen ``seedN`` bundle members to prediction seed integers."""
    members = metadata.get("members")
    if not isinstance(members, list) or len(members) != member_count:
        raise ProbabilityContractError(f"{model_id} member registry is malformed")
    seeds: list[int] = []
    for member in members:
        match = re.fullmatch(r"seed(0|[1-9][0-9]*)", str(member))
        if match is None:
            raise ProbabilityContractError(
                f"{model_id} member {member!r} cannot bind a prediction seed"
            )
        seeds.append(int(match.group(1)))
    registry = tuple(sorted(seeds))
    if len(registry) != len(set(registry)) or registry != tuple(C.USGS_SEEDS):
        raise ProbabilityContractError(
            f"{model_id} frozen seed registry differs from {tuple(C.USGS_SEEDS)}"
        )
    return registry


def _load_pointer_contracts(
    root: Path,
    pointer_path: Path,
    requested_models: Sequence[str],
) -> dict[str, FrozenModelContract]:
    """Load bundle metadata only through a checksum-valid component pointer."""
    try:
        pointer = load_component_pointer(pointer_path)
    except ModelSuiteError as exc:
        raise ProbabilityContractError(
            f"formal component pointer is invalid: {pointer_path}"
        ) from exc
    entries = pointer.get("models")
    if not isinstance(entries, list):
        raise ProbabilityContractError("component pointer model registry is malformed")
    by_id = {
        str(entry.get("model_id")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if not set(requested_models) <= set(by_id):
        missing = sorted(set(requested_models) - set(by_id))
        raise ProbabilityContractError(f"component pointer lacks models: {missing}")
    pointer_binding = file_binding(root, pointer_path)
    output: dict[str, FrozenModelContract] = {}
    for model_id in requested_models:
        entry = by_id[model_id]
        executor = str(entry.get("executor"))
        artifact = entry.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ProbabilityContractError(f"{model_id} has no artifact binding")
        member_count = int(entry.get("member_count", 0))
        if member_count < 1:
            raise ProbabilityContractError(f"{model_id} member count is invalid")
        if executor == "lightgbm_bundle":
            manifest_path = _inside(root, artifact.get("path"))
            if sha256_file(manifest_path) != artifact.get("sha256"):
                raise ProbabilityContractError("LightGBM manifest checksum changed")
            try:
                _members, metadata = load_lightgbm_bundle(manifest_path)
            except (ModelSuiteError, ValueError) as exc:
                raise ProbabilityContractError("LightGBM bundle cannot be verified") from exc
            if int(metadata.get("member_count", -1)) != member_count:
                raise ProbabilityContractError("LightGBM member count differs from pointer")
            model_files = _model_file_bindings(root, manifest_path.parent)
            metadata_file = file_binding(root, manifest_path)
        elif executor in {"thermoroute_bundle", "lstm_bundle"}:
            directory = _inside(root, artifact.get("path"), directory=True)
            metadata_path = directory / "metadata.json"
            weights_path = directory / "weights.pt"
            if (
                sha256_file(metadata_path) != artifact.get("metadata_sha256")
                or sha256_file(weights_path) != artifact.get("weights_sha256")
            ):
                raise ProbabilityContractError(f"{model_id} bundle checksum changed")
            try:
                _weights, metadata = load_inference_bundle(
                    directory, expected_member_count=member_count
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ProbabilityContractError(
                    f"{model_id} bundle cannot be verified"
                ) from exc
            model_files = _model_file_bindings(root, directory)
            metadata_file = file_binding(root, metadata_path)
        else:
            raise ProbabilityContractError(
                f"{model_id} uses unsupported executor {executor!r}"
            )
        prediction_seeds = _prediction_seed_registry(
            metadata, model_id=model_id, member_count=member_count
        )
        output[model_id] = FrozenModelContract(
            model_id=model_id,
            executor=executor,
            member_count=member_count,
            prediction_seeds=prediction_seeds,
            metadata=metadata,
            pointer=pointer_binding,
            metadata_file=metadata_file,
            model_files=model_files,
        )
    return output


def load_frozen_probability_contracts(
    *,
    root: Path,
    stage9_components: Path,
    lstm_components: Path,
) -> dict[str, FrozenModelContract]:
    """Resolve exactly the three frozen bundle-backed development models."""
    contracts = _load_pointer_contracts(
        root, stage9_components, ("ThermoRoute", "LightGBM")
    )
    contracts.update(_load_pointer_contracts(root, lstm_components, ("LSTM",)))
    if set(contracts) != set(PROB_MODELS):
        raise ProbabilityContractError("frozen probability model registry changed")
    return contracts


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ProbabilityContractError(f"{label} cannot be boolean")
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProbabilityContractError(f"{label} is not numeric") from exc
    if not np.isfinite(parsed):
        raise ProbabilityContractError(f"{label} is non-finite")
    return parsed


def _parse_platt_registry(
    value: object, horizons: Sequence[int], *, label: str
) -> dict[int, PlattCalibrator]:
    if not isinstance(value, Mapping) or set(value) != {
        str(int(horizon)) for horizon in horizons
    }:
        raise ProbabilityContractError(f"{label} Platt registry changed")
    output: dict[int, PlattCalibrator] = {}
    for horizon in horizons:
        raw = value[str(int(horizon))]
        if not isinstance(raw, Mapping) or set(raw) != {
            "intercept", "slope", "constant"
        }:
            raise ProbabilityContractError(f"{label} h={horizon} Platt schema changed")
        intercept = _finite_float(raw["intercept"], label="Platt intercept")
        slope = _finite_float(raw["slope"], label="Platt slope")
        constant_raw = raw["constant"]
        constant = (
            None
            if constant_raw is None
            else _finite_float(constant_raw, label="Platt constant")
        )
        if constant is not None and not 0.0 < constant < 1.0:
            raise ProbabilityContractError("Platt constant must lie strictly inside (0,1)")
        if constant is not None and (
            slope != 0.0
            or not np.isclose(
                intercept,
                float(logit(np.asarray([constant]))[0]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ProbabilityContractError(
                "constant Platt calibrator must have zero slope and logit intercept"
            )
        output[int(horizon)] = PlattCalibrator(intercept, slope, constant)
    return output


def _validate_calibration_fit_contract(value: object, *, label: str) -> Mapping[str, Any]:
    """Require machine-readable proof that Platt and CQR were frozen on 2018.

    ``calibration_fit_contract`` is a v2 bundle field.  The exact schema is
    validated by the model-suite gate; Stage 19 independently checks the facts it
    consumes and fails closed on legacy bundles that only implied them in prose.
    """
    if not isinstance(value, Mapping):
        raise ProbabilityContractError(f"{label} lacks calibration_fit_contract")
    if dict(value) != route_a_calibration_fit_contract(external=False):
        raise ProbabilityContractError(
            f"{label} calibration_fit_contract differs from the exact temporal contract"
        )
    return value


def validate_probability_metadata(
    metadata_by_model: Mapping[str, Mapping[str, Any]],
    *,
    sites: Sequence[str],
    horizons: Sequence[int] = HORIZONS,
    expected_lineage: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate frozen event/CQR/Platt metadata without fitting anything."""
    if set(metadata_by_model) != set(PROB_MODELS):
        raise ProbabilityContractError("probability metadata model registry changed")
    site_registry = tuple(str(site) for site in sites)
    if (
        not site_registry
        or len(site_registry) != len(set(site_registry))
        or any(not site or site != site.strip() for site in site_registry)
    ):
        raise ProbabilityContractError("probability station registry is malformed")
    expected_offsets = {
        f"{site}|{int(horizon)}" for site in site_registry for horizon in horizons
    }
    expected_thresholds = set(site_registry)
    first_thresholds: Mapping[str, Any] | None = None
    first_reference: Mapping[str, Any] | None = None
    model_audit: dict[str, Any] = {}
    for model_id in PROB_MODELS:
        metadata = metadata_by_model[model_id]
        if expected_lineage is not None:
            for field in (
                "source_sha256", "panel_sha256", "registry_sha256", "runtime_sha256"
            ):
                if metadata.get(field) != expected_lineage.get(field):
                    raise ProbabilityContractError(
                        f"{model_id} {field} differs from prediction lineage"
                    )
        if tuple(int(value) for value in metadata.get("horizons", ())) != tuple(horizons):
            raise ProbabilityContractError(f"{model_id} horizon registry changed")
        thresholds = metadata.get("event_thresholds")
        reference = metadata.get("event_reference_climatology")
        offsets = metadata.get("conformal_offsets")
        if not isinstance(thresholds, Mapping) or set(map(str, thresholds)) != expected_thresholds:
            raise ProbabilityContractError(f"{model_id} event threshold registry changed")
        threshold_values = np.asarray(
            [_finite_float(value, label="event threshold") for value in thresholds.values()]
        )
        if not len(threshold_values):
            raise ProbabilityContractError(f"{model_id} event threshold registry is empty")
        if not isinstance(reference, Mapping):
            raise ProbabilityContractError(f"{model_id} lacks frozen event reference")
        try:
            validate_frozen_seasonal_event_reference(
                reference, expected_sites=set(site_registry), pooled=False
            )
        except ValueError as exc:
            raise ProbabilityContractError(
                f"{model_id} frozen event reference is invalid"
            ) from exc
        if reference.get("fit_interval") != ["2006-01-01", "2018-12-31"]:
            raise ProbabilityContractError(
                f"{model_id} event reference was not frozen on 2006--2018"
            )
        if not isinstance(offsets, Mapping) or set(map(str, offsets)) != expected_offsets:
            raise ProbabilityContractError(f"{model_id} CQR registry changed")
        try:
            cqr_audit = CF.validate_cqr_offset_bundle(
                offsets,
                metadata.get("conformal_policy"),
                metadata.get("conformal_offset_audit"),
            )
        except (CF.CQRContractError, TypeError, ValueError) as exc:
            raise ProbabilityContractError(f"{model_id} CQR audit failed") from exc
        calibrators = _parse_platt_registry(
            metadata.get("event_calibrators"), horizons, label=model_id
        )
        fit_contract = _validate_calibration_fit_contract(
            metadata.get("calibration_fit_contract"), label=model_id
        )
        if first_thresholds is None:
            first_thresholds = thresholds
            first_reference = reference
        else:
            assert first_reference is not None
            if dict(thresholds) != dict(first_thresholds) or dict(reference) != dict(
                first_reference
            ):
                raise ProbabilityContractError(
                    f"{model_id} event definition/reference differs from ThermoRoute"
                )
        deployed = np.asarray([float(value) for value in offsets.values()], dtype=float)
        model_audit[model_id] = {
            "event_calibrators": {
                str(horizon): calibrator.as_dict()
                for horizon, calibrator in sorted(calibrators.items())
            },
            "event_calibrators_sha256": sha256_json(metadata["event_calibrators"]),
            "calibration_fit_contract": dict(fit_contract),
            "calibration_fit_contract_sha256": sha256_json(fit_contract),
            "conformal_policy": metadata.get("conformal_policy"),
            "conformal_offset_audit": dict(cqr_audit),
            "conformal_offsets_sha256": sha256_json(offsets),
            "deployed_offset_min_c": float(deployed.min()),
            "deployed_offset_max_c": float(deployed.max()),
            "quantile_repair_contract": metadata.get(
                "quantile_repair",
                "NOT_APPLICABLE_ORDERED_NEURAL_OUTPUT_HEADS",
            ),
            "bundle_raw_quantile_crossing_audit": metadata.get(
                "raw_quantile_crossing_audit"
            ),
        }
    assert first_thresholds is not None and first_reference is not None
    return {
        "format": CALIBRATION_AUDIT_FORMAT,
        "event_threshold_scope": "station_development_train_2006_2015_q90",
        "event_thresholds": {str(key): float(value) for key, value in first_thresholds.items()},
        "event_thresholds_sha256": sha256_json(first_thresholds),
        "event_reference": dict(first_reference),
        "event_reference_sha256": sha256_json(first_reference),
        "event_reference_fit_interval": ["2006-01-01", "2018-12-31"],
        "platt_and_cqr_fit_interval": ["2018-01-01", "2018-12-31"],
        "fitting_performed_by_stage19": False,
        "models": model_audit,
    }


def _normalise_prediction_identity(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["site_id"] = output.site_id.astype(str)
    output["horizon"] = pd.to_numeric(output.horizon, errors="raise").astype(int)
    output["issue_date"] = pd.to_datetime(output.issue_date, errors="raise")
    output["target_date"] = pd.to_datetime(output.target_date, errors="raise")
    return output


def strict_ensemble_frames(
    predictions: pd.DataFrame,
    models: Sequence[str],
    *,
    split: str = "test",
    require_probability_heads: bool,
    expected_seeds: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Average members only after exact key, head, member and truth checks."""
    requested = tuple(dict.fromkeys(str(model) for model in models))
    subset = _normalise_prediction_identity(
        predictions[
            predictions.model.astype(str).isin(requested)
            & predictions.split.astype(str).eq(split)
        ]
    )
    absent = sorted(set(requested) - set(subset.model.astype(str)))
    if absent:
        raise ProbabilityContractError(f"prediction artifact lacks models: {absent}")
    if split == "test":
        lower, upper = (pd.Timestamp(value) for value in C.SPLIT.test)
        if not (
            subset.issue_date.between(lower, upper).all()
            and subset.target_date.between(lower, upper).all()
        ):
            raise ProbabilityContractError(
                "development evaluation rows are not confined to 2019--2020"
            )
    duplicate = ["model", "seed", *IDENTITY]
    if subset.duplicated(duplicate).any():
        raise ProbabilityContractError(
            "model/seed prediction identity is duplicated across scope or feature rows"
        )
    frames: dict[str, pd.DataFrame] = {}
    for model in requested:
        selected = subset[subset.model.astype(str).eq(model)].copy()
        groups = selected.groupby(list(IDENTITY), sort=True, observed=True, dropna=False)
        truth_min = groups.y_true.min()
        truth_max = groups.y_true.max()
        if not np.array_equal(
            truth_min.to_numpy(float), truth_max.to_numpy(float), equal_nan=False
        ):
            raise ProbabilityContractError(f"{model} members disagree on exact y_true")
        sizes = groups.size()
        expected = None if expected_seeds is None else expected_seeds.get(model)
        if expected is not None:
            frozen_seed_registry = tuple(sorted(int(seed) for seed in expected))
            if len(frozen_seed_registry) != len(set(frozen_seed_registry)):
                raise ProbabilityContractError(f"{model} expected seed registry is duplicated")
            if pd.api.types.is_bool_dtype(selected.seed.dtype):
                raise ProbabilityContractError(f"{model} prediction seed is boolean")
            numeric_seed = pd.to_numeric(selected.seed, errors="raise")
            if not np.equal(numeric_seed, np.floor(numeric_seed)).all():
                raise ProbabilityContractError(f"{model} prediction seed is not integral")
            actual_seed_registry = tuple(sorted(set(numeric_seed.astype(int))))
            if actual_seed_registry != frozen_seed_registry:
                raise ProbabilityContractError(
                    f"{model} prediction seed registry {actual_seed_registry} differs "
                    f"from frozen {frozen_seed_registry}"
                )
            if sizes.empty or not sizes.eq(len(frozen_seed_registry)).all():
                raise ProbabilityContractError(
                    f"{model} does not contain every frozen seed on every key"
                )
        aggregation: dict[str, tuple[str, str]] = {
            "y_true": ("y_true", "first"),
            "y_pred": ("y_pred", "mean"),
        }
        if require_probability_heads:
            head_columns = ("q05", "q50", "q95", "p_exceed")
            missing_counts = {
                column: int(selected[column].isna().sum()) for column in head_columns
            }
            if any(missing_counts.values()):
                raise ProbabilityContractError(
                    f"{model} has missing probabilistic heads: {missing_counts}"
                )
            values = selected[list(head_columns)].to_numpy(float)
            if not np.isfinite(values).all():
                raise ProbabilityContractError(f"{model} has non-finite probability heads")
            if not (
                (values[:, 0] <= values[:, 1])
                & (values[:, 1] <= values[:, 2])
                & (values[:, 0] < values[:, 2])
            ).all():
                raise ProbabilityContractError(
                    f"{model} exported quantiles cross or form an empty interval; "
                    "Stage19 never repairs them"
                )
            if not ((0.0 <= values[:, 3]) & (values[:, 3] <= 1.0)).all():
                raise ProbabilityContractError(
                    f"{model} exported event probability is outside [0, 1]"
                )
            for column in head_columns:
                if not groups[column].count().equals(sizes):
                    raise ProbabilityContractError(
                        f"{model} member-level {column} registry is incomplete"
                    )
                aggregation[column] = (column, "mean")
        ensemble = groups.agg(**aggregation).reset_index().sort_values(list(IDENTITY))
        if ensemble.duplicated(list(IDENTITY)).any():
            raise AssertionError("ensemble identity unexpectedly duplicated")
        if require_probability_heads:
            heads = ensemble[["q05", "q50", "q95", "p_exceed"]].to_numpy(float)
            if not (
                np.isfinite(heads).all()
                and (heads[:, 0] <= heads[:, 1]).all()
                and (heads[:, 1] <= heads[:, 2]).all()
                and (heads[:, 0] < heads[:, 2]).all()
                and ((0.0 <= heads[:, 3]) & (heads[:, 3] <= 1.0)).all()
            ):
                raise ProbabilityContractError(
                    f"{model} ensemble has an invalid interval or event probability"
                )
        frames[model] = ensemble.reset_index(drop=True)

    reference = frames[requested[0]].set_index(list(IDENTITY)).sort_index()
    for model in requested[1:]:
        candidate = frames[model].set_index(list(IDENTITY)).sort_index()
        if not reference.index.equals(candidate.index):
            raise ProbabilityContractError(
                f"{model} keys are not exactly equal to {requested[0]} keys"
            )
        if not np.array_equal(
            reference.y_true.to_numpy(float),
            candidate.y_true.to_numpy(float),
            equal_nan=False,
        ):
            raise ProbabilityContractError(
                f"{model} y_true differs on the exact common key registry"
            )
    return frames


def _retained_sites(
    reference: pd.DataFrame,
    horizon: int,
    *,
    minimum: int = MINIMUM_SITE_HORIZON_TARGETS,
) -> tuple[tuple[str, ...], dict[str, int]]:
    selected = reference[reference.horizon.eq(int(horizon))]
    counts = selected.site_id.astype(str).value_counts().sort_index()
    retained = tuple(str(site) for site in counts[counts.ge(minimum)].index)
    if not retained:
        raise ProbabilityContractError(
            f"no station has at least {minimum} exact common targets at h={horizon}"
        )
    return retained, {str(site): int(count) for site, count in counts.items()}


def station_balanced_weights(sites: Sequence[str]) -> np.ndarray:
    """Give each station exactly 1/n_sites total mass, irrespective of row count."""
    values = pd.Series(np.asarray(sites, dtype=str), dtype="string")
    if values.empty or values.isna().any() or values.eq("").any():
        raise ProbabilityContractError("cannot weight an empty/malformed station registry")
    counts = values.value_counts().sort_index()
    n_sites = len(counts)
    mapping = {str(site): 1.0 / (n_sites * int(count)) for site, count in counts.items()}
    weights = values.astype(str).map(mapping).to_numpy(float)
    totals = pd.DataFrame({"site": values.astype(str), "weight": weights}).groupby(
        "site", sort=True
    ).weight.sum().to_numpy(float)
    if not (
        np.isfinite(weights).all()
        and (weights > 0.0).all()
        and np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)
        and np.allclose(
            totals,
            np.full(n_sites, 1.0 / n_sites),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ProbabilityContractError("station-balanced weights failed postconditions")
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape or not (
        np.isfinite(values).all()
        and np.isfinite(weights).all()
        and (weights > 0).all()
        and np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ProbabilityContractError("weighted metric inputs are invalid")
    return float(np.sum(values * weights))


def _pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    residual = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def three_quantile_pinball_mean(
    y: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Return the unscaled arithmetic mean of three weighted pinball losses."""
    values = {
        "q05": _weighted_mean(_pinball_loss(y, q05, 0.05), weights),
        "q50": _weighted_mean(_pinball_loss(y, q50, 0.50), weights),
        "q95": _weighted_mean(_pinball_loss(y, q95, 0.95), weights),
    }
    return float(np.mean(list(values.values()))), values


def _weighted_ece(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> float:
    probability = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    event = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.digitize(probability, edges[1:-1]), 0, 9)
    value = 0.0
    for index in range(10):
        selected = bins == index
        if selected.any():
            mass = float(weights[selected].sum())
            value += mass * abs(
                float(np.average(event[selected], weights=weights[selected]))
                - float(np.average(probability[selected], weights=weights[selected]))
            )
    return float(value)


def _weighted_calibration_intercept_slope(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    event = np.asarray(y, dtype=int)
    if np.unique(event).size < 2:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(
                logit(np.asarray(p, dtype=float)).reshape(-1, 1),
                event,
                sample_weight=np.asarray(weights, dtype=float),
            )
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            return float("nan"), float("nan")
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    values = np.asarray([model.intercept_[0], model.coef_[0, 0]], dtype=float)
    return (float(values[0]), float(values[1])) if np.isfinite(values).all() else (
        float("nan"), float("nan")
    )


def _frozen_model_frame(
    raw: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only bundle-frozen CQR and Platt objects to an ensemble frame."""
    offsets = metadata["conformal_offsets"]
    calibrators = _parse_platt_registry(
        metadata["event_calibrators"], HORIZONS, label="bundle"
    )
    output = raw.copy()
    heads = output[["q05", "q50", "q95", "p_exceed"]].to_numpy(float)
    if not np.isfinite(heads).all():
        raise ProbabilityContractError("ensemble has an incomplete probability head")
    raw_crossing = (
        (heads[:, 0] > heads[:, 1])
        | (heads[:, 1] > heads[:, 2])
        | (heads[:, 0] >= heads[:, 2])
    )
    if raw_crossing.any():
        raise ProbabilityContractError(
            "exported ensemble quantiles cross or form an empty interval; "
            "evaluation-time repair is prohibited"
        )
    if not ((0.0 <= heads[:, 3]) & (heads[:, 3] <= 1.0)).all():
        raise ProbabilityContractError(
            "exported ensemble event probability is outside [0, 1]"
        )
    deltas = np.asarray(
        [float(offsets[f"{site}|{int(horizon)}"])
         for site, horizon in output[["site_id", "horizon"]].itertuples(
             index=False, name=None
         )],
        dtype=float,
    )
    if not np.isfinite(deltas).all() or (deltas < 0.0).any():
        raise ProbabilityContractError("deployed CQR offset is invalid")
    output["cqr_q05"] = output.q05.to_numpy(float) - deltas
    output["cqr_q95"] = output.q95.to_numpy(float) + deltas
    output["p_exceed_calibrated"] = np.nan
    for horizon in HORIZONS:
        selected = output.horizon.eq(horizon)
        output.loc[selected, "p_exceed_calibrated"] = calibrators[horizon].predict(
            output.loc[selected, "p_exceed"].to_numpy(float)
        )
    calibrated = output[
        ["cqr_q05", "q50", "cqr_q95", "p_exceed_calibrated"]
    ].to_numpy(float)
    if not (
        np.isfinite(calibrated).all()
        and (calibrated[:, 0] <= calibrated[:, 1]).all()
        and (calibrated[:, 1] <= calibrated[:, 2]).all()
        and (calibrated[:, 0] < calibrated[:, 2]).all()
        and ((0.0 <= calibrated[:, 3]) & (calibrated[:, 3] <= 1.0)).all()
    ):
        raise ProbabilityContractError("frozen calibration produced invalid heads")
    nominal_width = output.q95.to_numpy(float) - output.q05.to_numpy(float)
    final_width = output.cqr_q95.to_numpy(float) - output.cqr_q05.to_numpy(float)
    if (final_width < nominal_width).any():
        raise ProbabilityContractError("CQR shrank an interval")
    return output, {
        "rows": int(len(output)),
        "missing_head_rows": 0,
        "exported_ensemble_crossing_rows": int(raw_crossing.sum()),
        "evaluation_time_quantile_repair_rows": 0,
        "evaluation_time_quantile_repair_applied": False,
        "cqr_offset_min_c": float(deltas.min()),
        "cqr_offset_max_c": float(deltas.max()),
        "minimum_interval_width_increase_c": float((final_width - nominal_width).min()),
        "q50_unchanged_by_cqr": True,
    }


def _weighted_event_metrics(
    event: np.ndarray,
    probability: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    event = np.asarray(event, dtype=int)
    p = np.asarray(probability, dtype=float)
    ref = np.asarray(reference, dtype=float)
    brier = _weighted_mean((p - event) ** 2, weights)
    reference_brier = _weighted_mean((ref - event) ** 2, weights)
    if reference_brier <= 0.0:
        raise ProbabilityContractError("frozen event reference has zero Brier score")
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    output = {
        "BRIER": brier,
        "BRIER_SKILL": float(1.0 - brier / reference_brier),
        "REFERENCE_BRIER": reference_brier,
        "LOG_LOSS": _weighted_mean(
            -(event * np.log(clipped) + (1 - event) * np.log(1.0 - clipped)),
            weights,
        ),
        "BASE_RATE": _weighted_mean(event.astype(float), weights),
    }
    if np.unique(event).size < 2:
        output["AUROC"] = float("nan")
        output["AUPRC"] = float("nan")
    else:
        output["AUROC"] = float(
            roc_auc_score(event, p, sample_weight=weights)
        )
        output["AUPRC"] = float(
            average_precision_score(event, p, sample_weight=weights)
        )
    return output


def evaluate_probability_models(
    frames: Mapping[str, pd.DataFrame],
    metadata_by_model: Mapping[str, Mapping[str, Any]],
    probability_audit: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Score frozen heads with exact common keys and station-balanced weights."""
    calibrated: dict[str, pd.DataFrame] = {}
    application_audit: dict[str, Any] = {}
    for model in PROB_MODELS:
        calibrated[model], application_audit[model] = _frozen_model_frame(
            frames[model], metadata_by_model[model]
        )
    thresholds = {
        str(site): float(value)
        for site, value in probability_audit["event_thresholds"].items()
    }
    reference = probability_audit["event_reference"]
    rows: list[dict[str, Any]] = []
    reliability: dict[str, list[pd.DataFrame]] = {model: [] for model in PROB_MODELS}
    retention: dict[str, Any] = {}
    reference_frame = calibrated[PROB_MODELS[0]]
    for horizon in HORIZONS:
        retained, counts = _retained_sites(reference_frame, horizon)
        retention[str(horizon)] = {
            "minimum_targets": MINIMUM_SITE_HORIZON_TARGETS,
            "retained_sites": list(retained),
            "excluded_site_counts": {
                site: count for site, count in counts.items() if site not in retained
            },
            "all_site_counts": counts,
        }
        expected_index: pd.MultiIndex | None = None
        expected_truth: np.ndarray | None = None
        for model in PROB_MODELS:
            selected = calibrated[model][
                calibrated[model].horizon.eq(horizon)
                & calibrated[model].site_id.astype(str).isin(retained)
            ].copy().sort_values(list(IDENTITY))
            index = pd.MultiIndex.from_frame(selected[list(IDENTITY)])
            truth = selected.y_true.to_numpy(float)
            if expected_index is None:
                expected_index, expected_truth = index, truth
            else:
                assert expected_truth is not None
                if not expected_index.equals(index) or not np.array_equal(
                    expected_truth, truth, equal_nan=False
                ):
                    raise ProbabilityContractError(
                        f"{model} retained keys/y_true differ at h={horizon}"
                    )
            weights = station_balanced_weights(selected.site_id.astype(str).to_numpy())
            sites = selected.site_id.astype(str).to_numpy()
            y = selected.y_true.to_numpy(float)
            raw_q05 = selected.q05.to_numpy(float)
            raw_q50 = selected.q50.to_numpy(float)
            raw_q95 = selected.q95.to_numpy(float)
            cqr_q05 = selected.cqr_q05.to_numpy(float)
            cqr_q95 = selected.cqr_q95.to_numpy(float)
            raw_probability = selected.p_exceed.to_numpy(float)
            probability = selected.p_exceed_calibrated.to_numpy(float)
            event = (y > np.asarray([thresholds[site] for site in sites])).astype(int)
            reference_probability = predict_frozen_seasonal_event_reference(
                reference,
                sites,
                pd.to_datetime(selected.target_date).to_numpy(),
            )
            three_q, pinballs = three_quantile_pinball_mean(
                y, raw_q05, raw_q50, raw_q95, weights
            )
            raw_event = _weighted_event_metrics(
                event, raw_probability, reference_probability, weights
            )
            calibrated_event = _weighted_event_metrics(
                event, probability, reference_probability, weights
            )
            intercept, slope = _weighted_calibration_intercept_slope(
                event, probability, weights
            )
            width = cqr_q95 - cqr_q05
            alpha = 0.10
            winkler = width + np.where(
                y < cqr_q05, (2.0 / alpha) * (cqr_q05 - y), 0.0
            ) + np.where(y > cqr_q95, (2.0 / alpha) * (y - cqr_q95), 0.0)
            site_totals = pd.DataFrame({"site": sites, "weight": weights}).groupby(
                "site", sort=True
            ).weight.sum()
            rows.append({
                "model": model,
                "horizon": horizon,
                "n_common": int(len(selected)),
                "n_sites": int(len(retained)),
                "minimum_site_targets": int(min(counts[site] for site in retained)),
                "metric_weighting": "equal_total_weight_per_retained_station",
                "minimum_site_total_weight": float(site_totals.min()),
                "maximum_site_total_weight": float(site_totals.max()),
                "PICP": _weighted_mean(
                    ((y >= cqr_q05) & (y <= cqr_q95)).astype(float), weights
                ),
                "MPIW": _weighted_mean(width, weights),
                "WINKLER": _weighted_mean(winkler, weights),
                "RAW_PINBALL_Q05": pinballs["q05"],
                "RAW_PINBALL_Q50": pinballs["q50"],
                "RAW_PINBALL_Q95": pinballs["q95"],
                "RAW_THREE_QUANTILE_PINBALL_MEAN": three_q,
                "THREE_QUANTILE_SCORE": three_q,
                "THREE_QUANTILE_SCORE_MULTIPLIER": 1.0,
                "THREE_QUANTILE_SCORE_IS_CRPS": False,
                "interval_endpoint_source": "bundle_frozen_2018_cqr",
                "pinball_quantile_source": (
                    "canonical_nominal_pre_cqr_ensemble_q05_q50_q95;"
                    "LightGBM_members_include_bundle_declared_median_preserving_repair"
                ),
                "Brier_raw": raw_event["BRIER"],
                "Brier_calibrated": calibrated_event["BRIER"],
                "BrierSkill_calibrated": calibrated_event["BRIER_SKILL"],
                "LogLoss_calibrated": calibrated_event["LOG_LOSS"],
                "ECE_raw": _weighted_ece(event, raw_probability, weights),
                "ECE_calibrated": _weighted_ece(event, probability, weights),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "AUPRC": calibrated_event["AUPRC"],
                "AUROC": calibrated_event["AUROC"],
                "base_rate_evaluation": calibrated_event["BASE_RATE"],
                "reference_brier": calibrated_event["REFERENCE_BRIER"],
                "event_reference_fit_start": "2006-01-01",
                "event_reference_fit_end": "2018-12-31",
                "platt_fit_split": "calib_2018_only",
                "cqr_fit_split": "calib_2018_only",
                "raw_crossing_rows": 0,
                "evaluation_time_repair_rows": 0,
                "offset_min_c": float(
                    selected.q05.sub(selected.cqr_q05).min()
                ),
                "offset_max_c": float(
                    selected.q05.sub(selected.cqr_q05).max()
                ),
            })
            selected["event"] = event
            selected["p_reference"] = reference_probability
            selected["station_balanced_weight"] = weights
            reliability[model].append(selected)
    reliability_frames = {
        model: pd.concat(parts, ignore_index=True) for model, parts in reliability.items()
    }
    return pd.DataFrame(rows), reliability_frames, {
        "exact_common_key_identity": list(IDENTITY),
        "exact_y_true_required": True,
        "minimum_valid_targets_per_station_horizon": MINIMUM_SITE_HORIZON_TARGETS,
        "retention": retention,
        "calibration_application": application_audit,
    }


def _weighted_point_scores(
    y: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - y
    mean_y = _weighted_mean(y, weights)
    mean_prediction = _weighted_mean(prediction, weights)
    var_y = _weighted_mean((y - mean_y) ** 2, weights)
    var_prediction = _weighted_mean((prediction - mean_prediction) ** 2, weights)
    covariance = _weighted_mean(
        (y - mean_y) * (prediction - mean_prediction), weights
    )
    correlation = (
        float("nan")
        if var_y <= 0.0 or var_prediction <= 0.0
        else covariance / np.sqrt(var_y * var_prediction)
    )
    alpha = float("nan") if var_y <= 0.0 else np.sqrt(var_prediction / var_y)
    beta = float("nan") if abs(mean_y) <= 1e-12 else mean_prediction / mean_y
    kge = (
        float("nan")
        if not np.isfinite([correlation, alpha, beta]).all()
        else float(1.0 - np.sqrt((correlation - 1.0) ** 2
                                 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    )
    return {
        "RMSE": float(np.sqrt(_weighted_mean(error ** 2, weights))),
        "MAE": _weighted_mean(np.abs(error), weights),
        "BIAS": _weighted_mean(error, weights),
        "NSE": float(1.0 - _weighted_mean(error ** 2, weights) / (var_y + 1e-12)),
        "KGE": kge,
        "PBIAS": float(100.0 * _weighted_mean(error, weights) / (mean_y + 1e-12)),
    }


def evaluate_point_models(
    frames: Mapping[str, pd.DataFrame],
    huc: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    retention: dict[str, Any] = {}
    reference = frames[POINT_MODELS[0]]
    for horizon in HORIZONS:
        retained, counts = _retained_sites(reference, horizon)
        retention[str(horizon)] = {
            "retained_sites": list(retained),
            "excluded_site_counts": {
                site: count for site, count in counts.items() if site not in retained
            },
        }
        for model in POINT_MODELS:
            group = frames[model][
                frames[model].horizon.eq(horizon)
                & frames[model].site_id.astype(str).isin(retained)
            ].copy().sort_values(list(IDENTITY))
            weights = station_balanced_weights(group.site_id.astype(str).to_numpy())
            scores = _weighted_point_scores(
                group.y_true.to_numpy(float), group.y_pred.to_numpy(float), weights
            )
            station_rmse = {
                str(site): float(np.sqrt(np.mean(
                    (part.y_pred.to_numpy(float) - part.y_true.to_numpy(float)) ** 2
                )))
                for site, part in group.groupby("site_id", sort=True)
            }
            regions: dict[str, list[float]] = {}
            for site, value in station_rmse.items():
                regions.setdefault(huc.get(site, "unmapped"), []).append(value)
            rows.append({
                "model": model,
                "horizon": horizon,
                "n_common": int(len(group)),
                "n_sites": int(len(retained)),
                "minimum_site_targets": int(min(counts[site] for site in retained)),
                "metric_weighting": "equal_total_weight_per_retained_station",
                **scores,
                "RMSE_region_wtd": float(
                    np.mean([np.median(values) for values in regions.values()])
                ),
            })
    return pd.DataFrame(rows), {
        "exact_common_key_identity": list(IDENTITY),
        "exact_y_true_required": True,
        "retention": retention,
    }


def _reliability_figure(reliability: Mapping[str, pd.DataFrame]) -> bytes:
    figure, axes = plt.subplots(1, len(HORIZONS), figsize=(12, 4), sharex=True, sharey=True)
    edges = np.linspace(0.0, 1.0, 11)
    for axis, horizon in zip(axes, HORIZONS, strict=True):
        axis.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1)
        for model, frame in reliability.items():
            group = frame[frame.horizon.eq(horizon)]
            y = group.event.to_numpy(int)
            weights = group.station_balanced_weight.to_numpy(float)
            for column, style, suffix in (
                ("p_exceed", ":", " raw"),
                ("p_exceed_calibrated", "-", " frozen Platt"),
            ):
                probability = group[column].to_numpy(float)
                membership = np.clip(np.digitize(probability, edges[1:-1]), 0, 9)
                points = []
                for index in range(10):
                    selected = membership == index
                    if selected.any():
                        points.append((
                            float(np.average(probability[selected], weights=weights[selected])),
                            float(np.average(y[selected], weights=weights[selected])),
                        ))
                if points:
                    x, observed = zip(*points, strict=True)
                    axis.plot(
                        x, observed, style, color=COLORS[model], linewidth=1.5,
                        label=model + suffix,
                    )
        axis.set_title(f"h={horizon} d")
        axis.grid(alpha=0.25)
        axis.set_xlabel("forecast probability")
    axes[0].set_ylabel("station-balanced observed frequency")
    axes[-1].legend(fontsize=7, loc="lower right")
    figure.suptitle("Development event reliability: raw and bundle-frozen Platt")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _render_report(scores: pd.DataFrame) -> bytes:
    lines = [
        "# Route-A probabilistic verification",
        "",
        (
            "2019--2020 is a previously inspected development evaluation, not a "
            "blind test. Stage 19 performs no fitting: event thresholds, the "
            "2006--2018 seasonal reference, 2018 Platt calibrators, and 2018 CQR "
            "offsets all come from checksum-bound formal bundles."
        ),
        "",
        (
            "Every metric retains only stations with at least 100 exact common "
            "targets per horizon and gives every retained station equal total "
            "weight. Coverage/width use frozen conformal endpoints. The 3Q value "
            "uses raw q05/q50/q95 and is the unscaled pinball arithmetic mean; "
            "it is not CRPS."
        ),
        "",
        (
            "| model | h | sites | n | CQR PICP | CQR width | raw 3Q pinball | "
            "Brier raw | Brier Platt | BSS frozen-ref | ECE Platt |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scores.itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.horizon)} | {int(row.n_sites)} | "
            f"{int(row.n_common)} | {row.PICP:.3f} | {row.MPIW:.3f} | "
            f"{row.THREE_QUANTILE_SCORE:.3f} | {row.Brier_raw:.3f} | "
            f"{row.Brier_calibrated:.3f} | {row.BrierSkill_calibrated:+.3f} | "
            f"{row.ECE_calibrated:.3f} |"
        )
    lines.extend([
        "",
        (
            "CQR is reported as empirical marginal development coverage only. "
            "No conditional-coverage, exchangeability, ecological-threshold, "
            "regulatory, or confirmatory claim is made."
        ),
        "",
        "![station-balanced reliability](../figures/fig_reliability.png)",
        "",
    ])
    return "\n".join(lines).encode("utf-8")


def _binding(root: Path, path: Path) -> dict[str, str]:
    return file_binding(root, path)


def _iter_file_bindings(value: object) -> Iterable[Mapping[str, str]]:
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256"}:
            yield value  # type: ignore[misc]
        else:
            for nested in value.values():
                yield from _iter_file_bindings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_file_bindings(nested)


def _write_self_hashed_receipt(path: Path, document: Mapping[str, Any]) -> None:
    if "receipt_self_sha256" in document:
        raise ProbabilityContractError("receipt input already contains a self-hash")
    payload = dict(document)
    payload["receipt_self_sha256"] = sha256_json(payload)
    atomic_write_json(
        path,
        payload,
        publication_guard=assert_formal_numerical_policy,
    )


def _publish_stage19_outputs(
    *,
    receipt_path: Path,
    artifacts: Mapping[str, Path],
    probability_scores_payload: bytes,
    point_scores_payload: bytes,
    calibration_audit: Mapping[str, Any],
    reliability_figure_payload: bytes,
    report_payload: bytes,
) -> None:
    """Revoke PASS, then guard every durable Stage-19 replacement boundary."""
    atomic_write_json(
        receipt_path,
        {
            "format": RECEIPT_FORMAT,
            "status": "INCOMPLETE",
            "scientific_role": (
                "previously_inspected_development_evaluation_2019_2020"
            ),
            "inference_computed": False,
        },
        publication_guard=assert_formal_numerical_policy,
    )
    atomic_write_bytes(
        artifacts["probability_scores"],
        probability_scores_payload,
        publication_guard=assert_formal_numerical_policy,
    )
    atomic_write_bytes(
        artifacts["point_scores"],
        point_scores_payload,
        publication_guard=assert_formal_numerical_policy,
    )
    atomic_write_json(
        artifacts["calibration_audit"],
        calibration_audit,
        publication_guard=assert_formal_numerical_policy,
    )
    atomic_write_bytes(
        artifacts["reliability_figure"],
        reliability_figure_payload,
        publication_guard=assert_formal_numerical_policy,
    )
    atomic_write_bytes(
        artifacts["report"],
        report_payload,
        publication_guard=assert_formal_numerical_policy,
    )


def validate_probability_receipt(
    path: Path,
    *,
    root: Path,
    enforce_current_source_and_runtime: bool = True,
    enforce_canonical_paths: bool = True,
    transaction_locks_held: bool = False,
) -> Mapping[str, Any]:
    """Validate receipt self-hash plus every bound input/output byte artifact."""
    assert_formal_numerical_policy()
    resolved_receipt = path.resolve()
    resolved_root = root.resolve()
    if resolved_receipt == resolved_root or resolved_root not in resolved_receipt.parents:
        raise ProbabilityContractError("probability receipt is outside repository")
    if (
        enforce_canonical_paths
        and resolved_receipt.relative_to(resolved_root).as_posix()
        != CANONICAL_RECEIPT_PATH
    ):
        raise ProbabilityContractError(
            "probability receipt is not at its exact canonical path"
        )
    if (
        enforce_canonical_paths
        and resolved_root == ROOT.resolve()
        and not transaction_locks_held
    ):
        # All formal callers use the global Stage16 -> Stage19 lock order.  The
        # recursive call performs only validation while both shared locks stay
        # held, closing receipt/artifact replacement races for direct callers.
        with advisory_file_lock(C.STAGE16_TRANSACTION_LOCK, exclusive=False):
            with advisory_file_lock(C.STAGE19_TRANSACTION_LOCK, exclusive=False):
                return validate_probability_receipt(
                    path,
                    root=root,
                    enforce_current_source_and_runtime=(
                        enforce_current_source_and_runtime
                    ),
                    enforce_canonical_paths=enforce_canonical_paths,
                    transaction_locks_held=True,
                )
    document = _load_json(path, label="probability receipt")
    if set(document) != RECEIPT_TOP_LEVEL_FIELDS:
        raise ProbabilityContractError("probability receipt top-level schema changed")
    if (
        document.get("format") != RECEIPT_FORMAT
        or document.get("status") != "PASS"
        or document.get("scientific_role")
        != "previously_inspected_development_evaluation_2019_2020"
        or document.get("inference_computed") is not False
    ):
        raise ProbabilityContractError("probability receipt format/status changed")
    payload = dict(document)
    self_hash = payload.pop("receipt_self_sha256", None)
    if not isinstance(self_hash, str) or self_hash != sha256_json(payload):
        raise ProbabilityContractError("probability receipt self-hash changed")

    inputs = document.get("inputs")
    artifacts = document.get("artifacts")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "prediction",
        "stage16_completion_receipt",
        "panel",
        "registry",
        "protocol",
        "component_pointers",
        "bundle_metadata",
        "bundle_files",
    }:
        raise ProbabilityContractError("probability receipt input closure changed")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        OUTPUT_CONTENT_SCHEMAS
    ):
        raise ProbabilityContractError("probability receipt artifact closure changed")
    prediction = inputs.get("prediction")
    component_pointers = inputs.get("component_pointers")
    bundle_metadata = inputs.get("bundle_metadata")
    bundle_files = inputs.get("bundle_files")
    if (
        not isinstance(prediction, Mapping)
        or set(prediction) != {"artifact", "lineage_sidecar"}
        or not isinstance(component_pointers, Mapping)
        or set(component_pointers) != {"stage9", "lstm"}
        or not isinstance(bundle_metadata, Mapping)
        or set(bundle_metadata) != set(PROB_MODELS)
        or not isinstance(bundle_files, Mapping)
        or set(bundle_files) != set(PROB_MODELS)
        or any(not isinstance(bundle_files[model], list) or not bundle_files[model]
               for model in PROB_MODELS)
    ):
        raise ProbabilityContractError("probability receipt nested input schema changed")

    if enforce_canonical_paths:
        for key_path, expected_path in CANONICAL_INPUT_PATHS.items():
            value: object = inputs
            for key in key_path:
                if not isinstance(value, Mapping):
                    raise ProbabilityContractError(
                        "probability receipt canonical input registry is malformed"
                    )
                value = value.get(key)
            if not isinstance(value, Mapping) or value.get("path") != expected_path:
                raise ProbabilityContractError(
                    f"probability receipt input {'.'.join(key_path)} is noncanonical"
                )
        for label, expected_path in CANONICAL_OUTPUT_PATHS.items():
            output = artifacts[label]
            if (
                not isinstance(output, Mapping)
                or not isinstance(output.get("artifact"), Mapping)
                or output["artifact"].get("path") != expected_path
                or not isinstance(output.get("lineage_sidecar"), Mapping)
                or output["lineage_sidecar"].get("path")
                != f"{expected_path}.meta.json"
            ):
                raise ProbabilityContractError(
                    f"probability receipt output {label} is noncanonical"
                )

    def is_file_binding(value: object) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == {"path", "sha256"}
            and isinstance(value.get("path"), str)
            and bool(value["path"])
            and isinstance(value.get("sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])))
        )

    required_bindings: list[object] = [
        prediction["artifact"],
        prediction["lineage_sidecar"],
        inputs["stage16_completion_receipt"],
        inputs["panel"],
        inputs["registry"],
        inputs["protocol"],
        component_pointers["stage9"],
        component_pointers["lstm"],
        *bundle_metadata.values(),
        *(binding for model in PROB_MODELS for binding in bundle_files[model]),
    ]
    if not all(is_file_binding(binding) for binding in required_bindings):
        raise ProbabilityContractError("probability receipt file binding is malformed")
    for model in PROB_MODELS:
        metadata_binding = bundle_metadata[model]
        if (
            not isinstance(metadata_binding, Mapping)
            or metadata_binding not in bundle_files[model]
        ):
            raise ProbabilityContractError(
                f"probability receipt {model} metadata is outside its bundle closure"
            )

    for binding in _iter_file_bindings({
        "inputs": inputs,
        "artifacts": artifacts,
    }):
        resolved = _inside(root, binding.get("path"))
        if sha256_file(resolved) != binding.get("sha256"):
            raise ProbabilityContractError(
                f"probability receipt binding changed: {binding.get('path')}"
            )
    if enforce_canonical_paths:
        stage16_binding = inputs["stage16_completion_receipt"]
        assert isinstance(stage16_binding, Mapping)
        stage16_path = _inside(root, stage16_binding.get("path"))
        try:
            stage16_document = validate_stage16_completion_receipt(
                stage16_path,
                root=root,
                components_pointer=_inside(
                    root, component_pointers["lstm"].get("path")
                ),
                enforce_current_runtime=enforce_current_source_and_runtime,
                replay_bundle=False,
            )
        except ModelSuiteError as exc:
            raise ProbabilityContractError(
                "Stage-16 completion gate is absent, stale, or malformed"
            ) from exc
        stage16_artifacts = stage16_document.get("artifacts")
        if (
            dict(stage16_binding) != file_binding(root, stage16_path)
            or not isinstance(stage16_artifacts, Mapping)
            or stage16_artifacts.get("development_predictions")
            != prediction["artifact"]
            or stage16_artifacts.get("development_prediction_sidecar")
            != prediction["lineage_sidecar"]
            or stage16_artifacts.get("components_pointer")
            != component_pointers["lstm"]
        ):
            raise ProbabilityContractError(
                "probability receipt disagrees with its Stage-16 completion gate"
            )
    lineage = document.get("lineage")
    run_identity = document.get("run_identity")
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != {
            "source_sha256",
            "panel_sha256",
            "registry_sha256",
            "execution_runtime_sha256",
            "frozen_bundle_runtime_sha256",
        }
        or not isinstance(run_identity, Mapping)
        or set(run_identity) != {
            "run_id",
            "panel_sha256",
            "registry_sha256",
            "config_sha256",
            "source_sha256",
            "runtime_sha256",
            "input_closure_sha256",
            "schema_version",
        }
    ):
        raise ProbabilityContractError("probability receipt lineage is malformed")
    input_components = {
        "panel": str(inputs["panel"]["sha256"]),
        "registry": str(inputs["registry"]["sha256"]),
        "prediction": str(prediction["artifact"]["sha256"]),
        "prediction_lineage": str(prediction["lineage_sidecar"]["sha256"]),
        "stage16_completion_receipt": str(
            inputs["stage16_completion_receipt"]["sha256"]
        ),
        "stage9_components": str(component_pointers["stage9"]["sha256"]),
        "lstm_components": str(component_pointers["lstm"]["sha256"]),
        "protocol": str(inputs["protocol"]["sha256"]),
        **{
            f"model_file:{binding['path']}": str(binding["sha256"])
            for model in PROB_MODELS
            for binding in bundle_files[model]
        },
    }
    if (
        lineage["source_sha256"] != run_identity["source_sha256"]
        or lineage["panel_sha256"] != run_identity["panel_sha256"]
        or lineage["registry_sha256"] != run_identity["registry_sha256"]
        or lineage["execution_runtime_sha256"] != run_identity["runtime_sha256"]
        or lineage["frozen_bundle_runtime_sha256"] != run_identity["runtime_sha256"]
        or inputs["panel"].get("sha256") != run_identity["panel_sha256"]
        or inputs["registry"].get("sha256") != run_identity["registry_sha256"]
        or compose_input_closure_digest(input_components)
        != run_identity["input_closure_sha256"]
    ):
        raise ProbabilityContractError("probability receipt lineage bindings disagree")
    contract = document.get("contract")
    if not isinstance(contract, Mapping) or sha256_json(contract) != document.get(
        "contract_sha256"
    ):
        raise ProbabilityContractError("probability receipt contract hash changed")

    prediction_path = _inside(root, prediction["artifact"].get("path"))
    prediction_sidecar = _inside(root, prediction["lineage_sidecar"].get("path"))
    if prediction_sidecar != sidecar_path(prediction_path).resolve():
        raise ProbabilityContractError("prediction lineage sidecar path changed")
    try:
        prediction_metadata = validate_artifact_sidecar(
            prediction_path,
            schema=R.PREDICTION_SCHEMA_VERSION,
            kind="final_route_a_development_predictions",
        )
    except ValueError as exc:
        raise ProbabilityContractError("prediction lineage sidecar is invalid") from exc
    prediction_run = prediction_metadata.get("run")
    if not isinstance(prediction_run, Mapping) or any(
        prediction_run.get(field) != lineage[receipt_field]
        for field, receipt_field in (
            ("source_sha256", "source_sha256"),
            ("panel_sha256", "panel_sha256"),
            ("registry_sha256", "registry_sha256"),
            ("runtime_sha256", "frozen_bundle_runtime_sha256"),
        )
    ):
        raise ProbabilityContractError("prediction run differs from probability receipt")

    expected_parents = {
        "prediction": str(prediction["artifact"]["sha256"]),
        "prediction_lineage": str(prediction["lineage_sidecar"]["sha256"]),
        "stage16_completion_receipt": str(
            inputs["stage16_completion_receipt"]["sha256"]
        ),
        "stage9_components": str(component_pointers["stage9"]["sha256"]),
        "lstm_components": str(component_pointers["lstm"]["sha256"]),
        "protocol": str(inputs["protocol"]["sha256"]),
        **{
            f"{model}_bundle_metadata": str(bundle_metadata[model]["sha256"])
            for model in PROB_MODELS
        },
    }
    receipt_relative = resolved_receipt.relative_to(resolved_root).as_posix()
    for label, schema in OUTPUT_CONTENT_SCHEMAS.items():
        output = artifacts[label]
        if not isinstance(output, Mapping) or set(output) != {
            "artifact", "lineage_sidecar"
        }:
            raise ProbabilityContractError(f"{label} receipt binding schema changed")
        if not is_file_binding(output["artifact"]) or not is_file_binding(
            output["lineage_sidecar"]
        ):
            raise ProbabilityContractError(f"{label} file binding is malformed")
        artifact_path = _inside(root, output["artifact"].get("path"))
        lineage_path = _inside(root, output["lineage_sidecar"].get("path"))
        if lineage_path != sidecar_path(artifact_path).resolve():
            raise ProbabilityContractError(f"{label} sidecar path changed")
        try:
            metadata = validate_artifact_sidecar(
                artifact_path,
                schema=schema,
                kind=f"route_a_development_{label}",
            )
        except ValueError as exc:
            raise ProbabilityContractError(f"{label} lineage sidecar is invalid") from exc
        if (
            metadata.get("run") != dict(run_identity)
            or metadata.get("parents") != expected_parents
            or metadata.get("extra") != {
                "development_only": True,
                "receipt_path": receipt_relative,
            }
        ):
            raise ProbabilityContractError(f"{label} lineage closure changed")

    if enforce_current_source_and_runtime:
        if lineage.get("source_sha256") != source_tree_hash(root):
            raise ProbabilityContractError("probability receipt is stale for current source")
        if lineage.get("execution_runtime_sha256") != sha256_json(
            numerical_runtime_contract()
        ):
            raise ProbabilityContractError("probability receipt is stale for current runtime")
    assert_formal_numerical_policy()
    return document


def _output_artifact_bindings(root: Path, artifacts: Mapping[str, Path]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, artifact in artifacts.items():
        output[label] = {
            "artifact": _binding(root, artifact),
            "lineage_sidecar": _binding(root, sidecar_path(artifact)),
        }
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing self-hashed receipt and every bound artifact",
    )
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--registry", type=Path, default=STATION_REGISTRY)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--stage9-components", type=Path, default=STAGE9_COMPONENTS)
    parser.add_argument("--lstm-components", type=Path, default=LSTM_COMPONENTS)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--probability-scores", type=Path, default=PROBABILITY_SCORES)
    parser.add_argument("--point-scores", type=Path, default=POINT_SCORES)
    parser.add_argument("--calibration-audit", type=Path, default=CALIBRATION_AUDIT)
    parser.add_argument("--reliability-figure", type=Path, default=RELIABILITY_FIGURE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    return parser.parse_args()


def _validated_stage16_gate_under_lock() -> tuple[Mapping[str, Any], dict[str, str]]:
    """Validate and bind Stage 16 while its shared transaction lock is held."""
    assert_formal_numerical_policy()
    try:
        document = validate_stage16_completion_receipt(
            STAGE16_RECEIPT,
            root=ROOT,
            components_pointer=LSTM_COMPONENTS,
            replay_bundle=False,
        )
    except ModelSuiteError as exc:
        raise ProbabilityContractError(
            "Stage-16 completion gate is absent, stale, or malformed"
        ) from exc
    assert_formal_numerical_policy()
    return document, _binding(ROOT, STAGE16_RECEIPT)


def _run(
    args: argparse.Namespace,
    *,
    stage16_document: Mapping[str, Any],
    stage16_binding: Mapping[str, str],
) -> None:
    assert_formal_numerical_policy()
    formal_paths = {
        "panel": (args.panel, PANEL),
        "registry": (args.registry, STATION_REGISTRY),
        "predictions": (args.predictions, PREDICTIONS),
        "stage9_components": (args.stage9_components, STAGE9_COMPONENTS),
        "lstm_components": (args.lstm_components, LSTM_COMPONENTS),
        "protocol": (args.protocol, PROTOCOL),
        "probability_scores": (args.probability_scores, PROBABILITY_SCORES),
        "point_scores": (args.point_scores, POINT_SCORES),
        "calibration_audit": (args.calibration_audit, CALIBRATION_AUDIT),
        "reliability_figure": (args.reliability_figure, RELIABILITY_FIGURE),
        "report": (args.report, REPORT),
        "receipt": (args.receipt, RECEIPT),
    }
    changed = [
        label for label, (actual, expected) in formal_paths.items()
        if actual.resolve() != expected.resolve()
    ]
    if changed:
        raise ProbabilityContractError(
            "formal Stage19 paths are immutable; noncanonical arguments: "
            f"{changed}"
        )
    panel = args.panel.resolve()
    registry = args.registry.resolve()
    predictions_path = args.predictions.resolve()
    stage9_components = args.stage9_components.resolve()
    lstm_components = args.lstm_components.resolve()
    protocol_path = args.protocol.resolve()
    stage16_artifacts = stage16_document.get("artifacts")
    if (
        not isinstance(stage16_artifacts, Mapping)
        or stage16_artifacts.get("development_predictions")
        != _binding(ROOT, predictions_path)
        or stage16_artifacts.get("development_prediction_sidecar")
        != _binding(ROOT, sidecar_path(predictions_path))
        or stage16_artifacts.get("components_pointer")
        != _binding(ROOT, lstm_components)
    ):
        raise ProbabilityContractError(
            "Stage-19 inputs disagree with the validated Stage-16 completion gate"
        )

    prediction_lineage = validate_artifact_sidecar(
        predictions_path,
        schema=R.PREDICTION_SCHEMA_VERSION,
        kind="final_route_a_development_predictions",
    )
    predictions = R.load_route_a_predictions(
        predictions_path,
        root=ROOT,
        panel_path=panel,
        registry_path=registry,
    )
    if set(ROUTE_A_PRIMARY_MODELS) - set(predictions.model.astype(str)):
        raise ProbabilityContractError("final prediction artifact is not primary-model complete")
    contracts = load_frozen_probability_contracts(
        root=ROOT,
        stage9_components=stage9_components,
        lstm_components=lstm_components,
    )
    # Bundle loaders may reuse validated in-process state.  Acceptance is still
    # conditional on the live native/Torch limiter, not just a runtime hash.
    assert_formal_numerical_policy()
    run_lineage = prediction_lineage["run"]
    actual_runtime_sha256 = sha256_json(numerical_runtime_contract())
    expected_lineage = {
        "source_sha256": source_tree_hash(ROOT),
        "panel_sha256": sha256_file(panel),
        "registry_sha256": sha256_file(registry),
        "runtime_sha256": actual_runtime_sha256,
    }
    for field, expected in expected_lineage.items():
        if run_lineage.get(field) != expected:
            raise ProbabilityContractError(
                f"prediction lineage {field} differs from current formal input"
            )
    metadata_by_model = {
        model: contract.metadata for model, contract in contracts.items()
    }
    probability_audit = validate_probability_metadata(
        metadata_by_model,
        sites=load_station_registry(registry).site_no.astype(str).tolist(),
        expected_lineage=expected_lineage,
    )
    expected_seeds = {
        model: contract.prediction_seeds for model, contract in contracts.items()
    }
    probability_frames = strict_ensemble_frames(
        predictions,
        PROB_MODELS,
        require_probability_heads=True,
        expected_seeds=expected_seeds,
    )
    probability_scores, reliability, probability_registry_audit = (
        evaluate_probability_models(
            probability_frames, metadata_by_model, probability_audit
        )
    )
    reliability_png = _reliability_figure(reliability)
    del probability_frames, reliability
    point_frames = strict_ensemble_frames(
        predictions,
        POINT_MODELS,
        require_probability_heads=False,
        expected_seeds={
            "Persistence": (0,),
            "DampedPersistence": (0,),
            **expected_seeds,
        },
    )
    huc = huc2_cluster_map(load_station_registry(registry))
    point_scores, point_registry_audit = evaluate_point_models(point_frames, huc)
    del point_frames

    protocol = _load_json(protocol_path, label="Route-A protocol")
    probability_contract = (
        protocol.get("primary_inference_contract", {})
        .get("probabilistic_event_contract")
    )
    if not isinstance(probability_contract, Mapping):
        raise ProbabilityContractError("Route-A protocol lacks probability contract")
    probability_audit = {
        **probability_audit,
        "protocol_probability_contract": dict(probability_contract),
        "protocol_probability_contract_sha256": sha256_json(probability_contract),
        "prediction_head_audit": probability_registry_audit,
    }

    artifacts = {
        "probability_scores": args.probability_scores.resolve(),
        "point_scores": args.point_scores.resolve(),
        "calibration_audit": args.calibration_audit.resolve(),
        "reliability_figure": args.reliability_figure.resolve(),
        "report": args.report.resolve(),
    }
    for path in (*artifacts.values(), args.receipt.resolve()):
        if path != ROOT and ROOT not in path.parents:
            raise ProbabilityContractError(f"output path is outside repository: {path}")

    # Revoke any previous PASS before the first output byte can move.  The
    # exclusive transaction lock prevents Stage10 from validating the old
    # receipt and then reading a partly replaced artifact set.
    report_payload = _render_report(probability_scores)
    _publish_stage19_outputs(
        receipt_path=args.receipt.resolve(),
        artifacts=artifacts,
        probability_scores_payload=probability_scores.to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ).encode("utf-8"),
        point_scores_payload=point_scores.to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ).encode("utf-8"),
        calibration_audit=probability_audit,
        reliability_figure_payload=reliability_png,
        report_payload=report_payload,
    )

    stage19_config = {
        "stage": "19_probabilistic_v2",
        "split": "development_evaluation_2019_2020",
        "models": list(PROB_MODELS),
        "point_models": list(POINT_MODELS),
        "horizons": list(HORIZONS),
        "minimum_site_horizon_targets": MINIMUM_SITE_HORIZON_TARGETS,
        "station_weighting": "equal_total_weight_per_retained_station",
        "three_quantile_score": "unscaled_arithmetic_mean_of_raw_pinball",
        "interval_endpoints": "bundle_frozen_2018_cqr",
        "event_calibration": "bundle_frozen_2018_platt",
        "event_reference": "bundle_frozen_2006_2018_seasonal",
        "component_pointer_sha256": {
            "stage9": sha256_file(stage9_components),
            "lstm": sha256_file(lstm_components),
        },
        "stage16_completion_receipt_sha256": str(stage16_binding["sha256"]),
        "protocol_sha256": sha256_file(protocol_path),
    }
    input_components = {
        "panel": sha256_file(panel),
        "registry": sha256_file(registry),
        "prediction": sha256_file(predictions_path),
        "prediction_lineage": sha256_file(sidecar_path(predictions_path)),
        "stage16_completion_receipt": str(stage16_binding["sha256"]),
        "stage9_components": sha256_file(stage9_components),
        "lstm_components": sha256_file(lstm_components),
        "protocol": sha256_file(protocol_path),
    }
    for contract in contracts.values():
        for binding in contract.model_files:
            input_components[f"model_file:{binding['path']}"] = str(
                binding["sha256"]
            )
    input_closure_sha256 = compose_input_closure_digest(input_components)
    stage19_config.update({
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": len(input_components),
    })
    identity = resolve_run_identity(
        root=ROOT,
        panel=panel,
        registry=registry,
        config=stage19_config,
        input_closure_sha256=input_closure_sha256,
    )
    if identity.runtime_sha256 != actual_runtime_sha256:
        raise ProbabilityContractError("Stage19 runtime identity changed during evaluation")
    parent_bindings = {
        "prediction": sha256_file(predictions_path),
        "prediction_lineage": sha256_file(sidecar_path(predictions_path)),
        "stage16_completion_receipt": str(stage16_binding["sha256"]),
        "stage9_components": sha256_file(stage9_components),
        "lstm_components": sha256_file(lstm_components),
        "protocol": sha256_file(protocol_path),
        **{
            f"{model}_bundle_metadata": str(contract.metadata_file["sha256"])
            for model, contract in contracts.items()
        },
    }
    for label, artifact in artifacts.items():
        seal_artifact(
            artifact,
            identity,
            kind=f"route_a_development_{label}",
            schema=OUTPUT_CONTENT_SCHEMAS[label],
            parents=parent_bindings,
            extra={
                "development_only": True,
                "receipt_path": args.receipt.resolve().relative_to(ROOT).as_posix(),
            },
            publication_guard=assert_formal_numerical_policy,
        )

    inputs = {
        "prediction": {
            "artifact": _binding(ROOT, predictions_path),
            "lineage_sidecar": _binding(ROOT, sidecar_path(predictions_path)),
        },
        "stage16_completion_receipt": dict(stage16_binding),
        "panel": _binding(ROOT, panel),
        "registry": _binding(ROOT, registry),
        "protocol": _binding(ROOT, protocol_path),
        "component_pointers": {
            "stage9": _binding(ROOT, stage9_components),
            "lstm": _binding(ROOT, lstm_components),
        },
        "bundle_metadata": {
            model: contract.metadata_file for model, contract in contracts.items()
        },
        "bundle_files": {
            model: list(contract.model_files) for model, contract in contracts.items()
        },
    }
    receipt_document = {
        "format": RECEIPT_FORMAT,
        "status": "PASS",
        "scientific_role": "previously_inspected_development_evaluation_2019_2020",
        "inference_computed": False,
        "run_identity": identity.as_dict(),
        "lineage": {
            "source_sha256": identity.source_sha256,
            "panel_sha256": identity.panel_sha256,
            "registry_sha256": identity.registry_sha256,
            "execution_runtime_sha256": identity.runtime_sha256,
            "frozen_bundle_runtime_sha256": expected_lineage["runtime_sha256"],
        },
        "inputs": inputs,
        "contract": stage19_config,
        "contract_sha256": sha256_json(stage19_config),
        "event_reference_sha256": probability_audit["event_reference_sha256"],
        "probability_registry_audit": probability_registry_audit,
        "point_registry_audit": point_registry_audit,
        "artifacts": _output_artifact_bindings(ROOT, artifacts),
    }
    receipt_path = args.receipt.resolve()
    _write_self_hashed_receipt(receipt_path, receipt_document)
    validate_probability_receipt(
        receipt_path,
        root=ROOT,
        transaction_locks_held=True,
    )
    print(report_payload.decode("utf-8"))
    print(f"validated self-hashed receipt: {receipt_path.relative_to(ROOT)}")


def main() -> None:
    args = _parse_args()
    # One global order is used everywhere: Stage16 first, Stage19 second.  The
    # Stage16 lock stays shared through receipt validation and all reads/writes,
    # so a V2 generation can never be replaced halfway through this stage.
    with advisory_file_lock(C.STAGE16_TRANSACTION_LOCK, exclusive=False):
        stage16_document, stage16_binding = (
            _validated_stage16_gate_under_lock()
        )
        if args.check:
            receipt_path = args.receipt.resolve()
            with advisory_file_lock(C.STAGE19_TRANSACTION_LOCK, exclusive=False):
                validate_probability_receipt(
                    receipt_path,
                    root=ROOT,
                    transaction_locks_held=True,
                )
            print(
                f"validated self-hashed receipt: {receipt_path.relative_to(ROOT)}"
            )
            return
        with advisory_file_lock(C.STAGE19_TRANSACTION_LOCK, exclusive=True):
            _run(
                args,
                stage16_document=stage16_document,
                stage16_binding=stage16_binding,
            )


if __name__ == "__main__":
    main()
