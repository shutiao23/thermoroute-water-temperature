"""Safe, executable model artifacts for the sealed Route-A model suite.

This module deliberately contains no acquisition code.  It serialises models
that were fitted on the development panel and validates them without reading a
confirmation table.  Torch models use :mod:`thermoroute.checkpoint`'s
weights-only format; LightGBM boosters use their native textual representation
instead of pickle.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import probability as P
from . import results as R
from .checkpoint import (
    instantiate_inference_ensemble,
    load_inference_bundle,
    neural_output_head_schema,
)
from .provenance import sha256_file
from .repro import (
    atomic_write_bytes,
    atomic_write_json,
    sidecar_path,
    source_tree_hash,
    validate_artifact_sidecar,
)


LIGHTGBM_BUNDLE_FORMAT = "thermoroute.lightgbm-bundle.v1"
MODEL_SUITE_FORMAT = "thermoroute.route-a-model-suite.v1"
MODEL_SUITE_POINTER_FORMAT = "thermoroute.route-a-model-suite-pointer.v1"
COMPONENT_POINTER_FORMAT = "thermoroute.route-a-model-components.v1"
DEVELOPMENT_PREDICTOR_BRIDGE_FORMAT = (
    "thermoroute.development-predictor-bridge.v1"
)
DEVELOPMENT_PREDICTOR_BRIDGE_PATH = (
    "data_usgs/development_predictor_bridge_v1.json"
)

LIGHTGBM_HEADS = ("point", "q05", "q50", "q95", "event")
LSTM_VALIDATION_GRID = (
    {"d": 64, "layers": 1, "dropout": 0.0, "station_embed_dim": 8,
     "use_derived_context": False, "anchor": "persistence"},
    {"d": 64, "layers": 1, "dropout": 0.0, "station_embed_dim": 8,
     "use_derived_context": True, "anchor": "damped"},
    {"d": 64, "layers": 2, "dropout": 0.10, "station_embed_dim": 8,
     "use_derived_context": True, "anchor": "damped"},
)
PRIMARY_MODELS = (
    "Persistence", "DampedPersistence", "Climatology",
    "LightGBM", "LSTM", "ThermoRoute",
)
MANDATORY_ABLATIONS = (
    "DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
    "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded",
)
ABLATION_INTERVENTIONS: dict[str, dict[str, Any]] = {
    "DampedPriorOnly": {"use_prior": False, "residual_model": False},
    "TR-noDynamicPrior": {"use_prior": False},
    "TR-fixedKappa": {"fixed_kappa": True},
    "TR-noRouter": {"use_router": False},
    "TR-noMoE": {"use_moe": False},
    "TR-noTCN": {"use_tcn": False},
    "TR-unbounded": {"delta_scale": None},
}
TEMPORAL_MODELS = PRIMARY_MODELS + MANDATORY_ABLATIONS
EXTERNAL_MODELS = PRIMARY_MODELS
BUILTIN_MODELS = frozenset({"Persistence", "DampedPersistence", "Climatology"})


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
) -> float:
    """Replay every sequence member and compare all five prediction heads."""
    models, metadata = instantiate_inference_ensemble(
        directory,
        model_factory=lambda member, bundle: model_factory(member, bundle),
        expected_member_count=len(member_seeds),
        device="cpu",
    )
    if set(models) != set(member_seeds):
        raise ModelSuiteError("sequence parity member registry differs from bundle")
    from .train import _export_predictions

    keys = ["seed", "site_id", "horizon", "split", "issue_date", "target_date"]
    values = ["y_true", "y_pred", "q05", "q50", "q95", "p_exceed"]
    reference = expected.copy()
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


def validate_development_prediction_binding(
    root: str | Path, value: object, *, label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ModelSuiteError(f"{label} lacks development prediction binding")
    required = {
        "artifact", "rows", "forecast_key_registry_sha256", "prediction_sha256",
        "max_abs_difference", "atol", "selection", "forecast_key_columns",
        "prediction_columns",
    }
    missing = required - set(value)
    if missing:
        raise ModelSuiteError(f"{label} prediction binding missing: {sorted(missing)}")
    artifact = value["artifact"]
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("sidecar"), Mapping):
        raise ModelSuiteError(f"{label} prediction artifact/sidecar binding is malformed")
    path = _resolve_inside(Path(root), artifact.get("path"))
    if sha256_file(path) != artifact.get("sha256"):
        raise ModelSuiteError(f"{label} development prediction checksum mismatch")
    sidecar = _resolve_inside(Path(root), artifact["sidecar"].get("path"))
    if sidecar != sidecar_path(path).resolve():
        raise ModelSuiteError(f"{label} binds a non-canonical prediction sidecar")
    if sha256_file(sidecar) != artifact["sidecar"].get("sha256"):
        raise ModelSuiteError(f"{label} development prediction sidecar checksum mismatch")
    try:
        validate_artifact_sidecar(
            path, schema=R.PREDICTION_SCHEMA_VERSION
        )
    except ValueError as exc:
        raise ModelSuiteError(
            f"{label} development prediction sidecar is malformed"
        ) from exc
    if int(value.get("rows", 0)) < 1:
        raise ModelSuiteError(f"{label} development prediction row count is empty")
    for field in ("forecast_key_registry_sha256", "prediction_sha256"):
        digest = str(value.get(field, ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ModelSuiteError(f"{label} {field} is not SHA-256")
    difference, tolerance = float(value["max_abs_difference"]), float(value["atol"])
    if not np.isfinite(difference) or difference < 0 or tolerance < 0 or difference > tolerance:
        raise ModelSuiteError(f"{label} development prediction parity failed")
    selection = value["selection"]
    if not isinstance(selection, Mapping) or set(selection) < {"model", "seeds"}:
        raise ModelSuiteError(f"{label} prediction selection is malformed")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ModelSuiteError(f"{label} development prediction cannot be read") from exc
    selected = frame[
        frame["model"].astype(str).eq(str(selection["model"]))
        & frame["seed"].astype(int).isin([int(seed) for seed in selection["seeds"]])
    ].copy()
    if len(selected) != int(value["rows"]):
        raise ModelSuiteError(f"{label} development prediction row count changed")
    key_columns = tuple(str(column) for column in value["forecast_key_columns"])
    prediction_columns = tuple(str(column) for column in value["prediction_columns"])
    key_digest = canonical_frame_digest(
        selected.drop_duplicates(list(key_columns)), key_columns
    )
    prediction_digest = canonical_frame_digest(selected, prediction_columns)
    if key_digest != value["forecast_key_registry_sha256"]:
        raise ModelSuiteError(f"{label} development forecast-key digest mismatch")
    if prediction_digest != value["prediction_sha256"]:
        raise ModelSuiteError(f"{label} development prediction digest mismatch")


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


def save_lightgbm_bundle(
    directory: str | Path,
    *,
    models: Mapping[str, Mapping[int | str, Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    parity_inputs: Mapping[int | str, Any] | None = None,
    parity_atol: float = 1e-12,
) -> Path:
    """Save point/quantile/event boosters and prove native-text round-trip parity.

    ``metadata`` must describe both the raw seven-variable information set and
    the exact engineered design columns.  No Python object is pickled.  When
    ``parity_inputs`` is supplied, every head is reconstructed from disk and
    compared with its in-memory booster before the manifest is finalised.
    """
    required = {
        "run_id", "raw_feature_order", "design_feature_order", "horizons",
        "station_agnostic", "uses_station_categorical", "preprocessing",
        "station_categories",
        "training_weighting", "deterministic_training",
        "event_thresholds", "event_calibrators", "conformal_offsets",
        "source_sha256", "panel_sha256", "registry_sha256", "config_sha256",
        "runtime_sha256", "training_device",
        "development_prediction",
    }
    missing = required - set(metadata)
    if missing:
        raise ModelSuiteError(f"LightGBM metadata missing: {sorted(missing)}")
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
        "deterministic": True, "force_col_wise": True, "n_jobs": 1,
    }:
        raise ModelSuiteError("LightGBM deterministic training contract changed")
    if metadata.get("training_device") != "cpu":
        raise ModelSuiteError("formal LightGBM bundles must be trained on CPU")
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
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", suffix=".staging",
        dir=destination.parent,
    ))

    bindings: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    original_boosters: dict[str, dict[int, dict[str, lgb.Booster]]] = {}
    for member in members:
        bindings[member] = {}
        original_boosters[member] = {}
        for horizon in horizons:
            bindings[member][str(horizon)] = {}
            original_boosters[member][horizon] = {}
            for head in LIGHTGBM_HEADS:
                booster = _booster(normalised[member][horizon][head])
                original_boosters[member][horizon][head] = booster
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
            shutil.rmtree(directory)
            return destination / "manifest.json"
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
) -> tuple[dict[str, dict[int, dict[str, lgb.Booster]]], dict[str, Any]]:
    """Reconstruct every native-text booster after strict checksum validation."""
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
        "deterministic": True, "force_col_wise": True, "n_jobs": 1,
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
    points = manifest.get("point_models")
    if not isinstance(points, Mapping) or set(points) != {str(h) for h in horizons}:
        raise ModelSuiteError("LightGBM point-model index is incomplete")
    for horizon in horizons:
        expected = {
            member: bindings[member][str(horizon)]["point"] for member in members
        }
        if points[str(horizon)] != expected:
            raise ModelSuiteError("LightGBM point-model index disagrees with head registry")
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
) -> float:
    """Replay all five native boosters against the Stage-9 development rows."""
    models, metadata = load_lightgbm_bundle(manifest_path)
    if set(models) != set(member_seeds):
        raise ModelSuiteError("LightGBM parity member registry differs from bundle")
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
            quantiles = np.sort(np.vstack([
                heads["q05"].predict(X, num_threads=1),
                heads["q50"].predict(X, num_threads=1),
                heads["q95"].predict(X, num_threads=1),
            ]), axis=0)
            replay = pd.DataFrame({
                "seed": int(seed),
                "site_id": registry["site_id"].astype(str).to_numpy(),
                "horizon": int(horizon),
                "split": registry["split"].astype(str).to_numpy(),
                "issue_date": pd.to_datetime(registry["issue_date"]).to_numpy(),
                "target_date": pd.to_datetime(registry["target_date"]).to_numpy(),
                "y_true": registry["y"].to_numpy(float),
                "y_pred": heads["point"].predict(X, num_threads=1),
                "q05": quantiles[0], "q50": quantiles[1], "q95": quantiles[2],
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
                difference = np.abs(
                    paired[f"{value}_reference"].to_numpy(float)
                    - paired[f"{value}_bundle"].to_numpy(float)
                )
                if np.any(~np.isfinite(difference)):
                    raise ModelSuiteError(f"LightGBM parity has non-finite {value}")
                maximum = max(maximum, float(difference.max(initial=0.0)))
    if maximum > float(atol):
        raise ModelSuiteError(
            f"LightGBM development prediction parity failed: {maximum} > {atol}"
        )
    return maximum


def fit_pooled_imputer(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    fit_stations: Sequence[str],
) -> D.Imputer:
    """Fit one development-only imputer and replicate it across training sites.

    Replication makes the station-agnostic contract machine-checkable: every
    stored station/variable statistic is byte-for-byte the same and can later be
    expanded to new site identifiers without observing their confirmation data.
    """
    sites = tuple(str(site) for site in fit_stations)
    selected = np.asarray(train_mask, dtype=bool) & panel.site_id.astype(str).isin(sites).to_numpy()
    training = panel.loc[selected].copy()
    if training.empty:
        raise ValueError("pooled imputer training partition is empty")
    training["doy"] = pd.to_datetime(training["DATE"]).dt.dayofyear
    medians: dict[tuple[str, str], pd.Series] = {}
    global_median: dict[tuple[str, str], float] = {}
    for variable in C.ALL_VARS:
        seasonal = training.groupby("doy")[variable].median()
        fallback = float(training[variable].median())
        if not np.isfinite(fallback):
            raise ValueError(f"pooled imputer cannot fit finite {variable} median")
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
            "method": "pooled_day_of_year_median_fit_on_train" if imputer_pooled
                      else "station_day_of_year_median_fit_on_train",
            "pooled": imputer_pooled,
            "fit_stations": list(getattr(imputer, "fit_stations", C.STATIONS)),
            "seasonal_medians": seasonal,
            "global_medians": _tuple_map(imputer.global_median, variables),
        },
        "scaler": {
            "method": "pooled_train_only_standardization" if wd.scaler.pooled
                      else "per_station_train_only_standardization",
            "mean": _tuple_map(wd.scaler.mean, variables),
            "std": _tuple_map(wd.scaler.std, variables),
            "fit_stations": list(wd.scaler.fit_stations),
            "pooled": bool(wd.scaler.pooled),
        },
        "climatology": {
            "method": "pooled_harmonic_least_squares_fit_on_train"
                      if climatology.pooled else "harmonic_least_squares_fit_on_train",
            "harmonics": int(climatology.k),
            "coefficients": {
                str(station): [float(value) for value in coefficients]
                for station, coefficients in sorted(climatology.coef.items())
            },
            "fit_stations": list(climatology.fit_stations),
            "pooled": bool(climatology.pooled),
        },
        "damped_anchor": {
            "method": "pooled_train_fit_ar1_anomaly" if wd.damped_anchor.pooled
                      else "train_fit_ar1_anomaly",
            "phi": {str(station): float(value)
                    for station, value in sorted(wd.damped_anchor.phi.items())},
            "fit_stations": list(wd.damped_anchor.fit_stations),
            "pooled": bool(wd.damped_anchor.pooled),
            "fallback": float(wd.damped_anchor.fallback),
        },
    }


def serialise_offsets(offsets: Mapping[tuple[str, int], object]) -> dict[str, float | None]:
    return {
        f"{station}|{int(horizon)}": _finite(value)
        for (station, horizon), value in sorted(offsets.items())
    }


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
    event_calibrators: Mapping[int, Any],
    source_sha256: str,
    panel_sha256: str,
    registry_sha256: str,
    config_sha256: str,
    runtime_sha256: str,
    training_device: str,
    development_prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete metadata required by a weights-only sequence bundle."""
    config_value = asdict(train_config) if hasattr(train_config, "__dataclass_fields__") \
        else dict(train_config)
    if str(training_device) != "cpu":
        raise ModelSuiteError("formal sequence bundles must be trained on CPU")
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
        "source_sha256": str(source_sha256),
        "panel_sha256": str(panel_sha256),
        "registry_sha256": str(registry_sha256),
        "config_sha256": str(config_sha256),
        "runtime_sha256": str(runtime_sha256),
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
    atomic_write_json(destination, document)
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


def _entry_artifact_valid(root: Path, entry: Mapping[str, Any], feature_order: tuple[str, ...],
                          *, external: bool) -> Mapping[str, Any] | None:
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
        _, metadata = load_lightgbm_bundle(path)
        if bool(metadata.get("station_agnostic")) != external:
            raise ModelSuiteError("LightGBM station-identity contract differs from cohort")
        if int(entry.get("member_count", 0)) != int(metadata.get("member_count", -1)):
            raise ModelSuiteError("LightGBM member count differs from suite entry")
        for field in (
            "source_sha256", "panel_sha256", "registry_sha256", "config_sha256",
            "runtime_sha256", "training_device",
        ):
            if not metadata.get(field):
                raise ModelSuiteError(f"LightGBM bundle lacks {field}")
        if metadata.get("training_device") != "cpu":
            raise ModelSuiteError("formal LightGBM bundle is not CPU-trained")
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
    _, metadata = load_inference_bundle(directory, expected_member_count=count)
    if tuple(metadata.get("feature_order", ())) != feature_order:
        raise ModelSuiteError(f"{model_id} sequence schema differs from suite")
    kwargs = metadata.get("architecture", {}).get("kwargs", {})
    if bool(kwargs.get("station_agnostic", False)) != external:
        raise ModelSuiteError(f"{model_id} station-identity contract differs from cohort")
    required_lineage = {
        "source_sha256", "panel_sha256", "registry_sha256", "config_sha256",
        "runtime_sha256", "training_device",
        "development_prediction",
    }
    missing = required_lineage - set(metadata)
    if missing:
        raise ModelSuiteError(f"{model_id} bundle lacks lineage: {sorted(missing)}")
    if metadata.get("training_device") != "cpu":
        raise ModelSuiteError(f"formal {model_id} bundle is not CPU-trained")
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
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ModelSuiteError(f"{model_id} cannot be strictly reconstructed") from exc
    return metadata


def validate_model_suite_document(document: Mapping[str, Any], *, root: str | Path) -> None:
    """Fail closed on a missing member, stale checksum, or wrong cohort contract."""
    root = Path(root).resolve()
    if document.get("format") != MODEL_SUITE_FORMAT:
        raise ModelSuiteError("unsupported model suite format")
    if document.get("status") != "FROZEN_BEFORE_LABEL_OPENING":
        raise ModelSuiteError("model suite is not frozen")
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
    for name, required, external in (
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
            if model_id in MANDATORY_ABLATIONS and int(by_id[model_id].get("member_count", 0)) != 1:
                raise ModelSuiteError(f"{model_id} must be a one-member exploratory control")
            if model_id in MANDATORY_ABLATIONS:
                intervention = by_id[model_id].get("intervention")
                if intervention != ABLATION_INTERVENTIONS[model_id]:
                    raise ModelSuiteError(f"{model_id} intervention is not the frozen control")
            metadata = _entry_artifact_valid(
                root, by_id[model_id], features, external=external
            )
            if metadata is not None:
                loaded_metadata[model_id] = metadata
        if not external:
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
            if external:
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
        lgb_metadata = loaded_metadata["LightGBM"]
        if external:
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


def _learned_metadata_runtime_sha256(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
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
            _, metadata = load_lightgbm_bundle(path)
        elif executor in {"thermoroute_bundle", "lstm_bundle"}:
            directory = _resolve_inside(root, artifact.get("path"), directory=True)
            _, metadata = load_inference_bundle(
                directory,
                expected_member_count=int(entry.get("member_count", 0)),
            )
        else:
            raise ModelSuiteError(f"unsupported learned executor: {executor}")
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
    registry_alias: str | Path | None = None,
) -> Path:
    """Write the versioned suite, then (and only then) publish its current pointer."""
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    learned_runtime_sha256 = _learned_metadata_runtime_sha256(
        root, [*temporal_entries, *external_entries]
    )
    document = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": learned_runtime_sha256,
        "protocol_sha256": str(protocol_sha256),
        "actual_feature_order": list(actual_feature_order),
        "development_contract": dict(development_contract),
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
    validate_model_suite_document(document, root=root)
    _create_json_or_require_identical(destination, document)
    validate_model_suite_document(
        json.loads(destination.read_text(encoding="utf-8")), root=root
    )
    current_suite = destination
    if registry_alias is not None:
        alias = Path(registry_alias).resolve()
        alias_document = {
            **document,
            "versioned_suite": file_binding(root, destination),
        }
        validate_model_suite_document(alias_document, root=root)
        _create_json_or_require_identical(alias, alias_document)
        validate_model_suite_document(
            json.loads(alias.read_text(encoding="utf-8")), root=root
        )
        current_suite = alias
    pointer = {
        "format": MODEL_SUITE_POINTER_FORMAT,
        "status": "CURRENT_COMPLETE",
        "suite_path": _relative(root, current_suite),
        "suite_sha256": sha256_file(current_suite),
    }
    atomic_write_json(current_pointer, pointer)
    return destination


def _create_json_or_require_identical(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    """Create a frozen JSON artifact once; retries must be byte-identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace frozen model registry: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Preserve any partial create as evidence; never silently replace it.
        raise


def resolve_model_suite_pointer(path: str | Path, *, root: str | Path) -> Path:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != MODEL_SUITE_POINTER_FORMAT:
        raise ModelSuiteError("unsupported model-suite pointer format")
    suite = _resolve_inside(Path(root), document.get("suite_path"))
    if sha256_file(suite) != document.get("suite_sha256"):
        raise ModelSuiteError("model-suite pointer checksum mismatch")
    validate_model_suite_document(json.loads(suite.read_text(encoding="utf-8")), root=root)
    return suite
