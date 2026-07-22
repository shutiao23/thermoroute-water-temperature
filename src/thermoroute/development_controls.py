"""Authoritative scientific contract for the Stage-09b development controls.

Both the training entry point and its independent completion gate import this
module.  Keeping architecture construction, window-registry reconstruction,
prediction normalisation, metric derivation, and report rendering here avoids
the unsafe pattern where a producer and verifier silently implement different
experiments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import datasets as DS
from . import features as F
from . import results as R
from .neural_baselines import PlainCausalTCNForecaster, PlainMLPForecaster
from .registry import FORECAST_KEY, targets_match_at_model_precision
from .thermoroute import ThermoRoute


FULL_VARIABLES: tuple[str, ...] = (
    "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP",
)
FEATURE_LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("01_WTEMP", ("WTEMP",)),
    ("02_plus_FLOW", ("WTEMP", "FLOW")),
    ("03_plus_TEMP", ("WTEMP", "FLOW", "TEMP")),
    ("04_plus_PRCP", ("WTEMP", "FLOW", "TEMP", "PRCP")),
    ("05_plus_RHMEAN", ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN")),
    ("06_plus_DH", ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH")),
    ("07_plus_WDSP", FULL_VARIABLES),
)
CONTROL_SEEDS: tuple[int, ...] = C.USGS_SEEDS
LADDER_SEEDS: tuple[int, ...] = C.USGS_SEEDS[:3]
TRAIN_CONFIG = C.TrainConfig(batch_size=1536)
MLP_HIDDEN_DIM = 70
TCN_CHANNELS = 54
THERMOROUTE_REFERENCE_PARAMETERS = 38_505
MLP_EXPECTED_PARAMETERS = 38_545
TCN_EXPECTED_PARAMETERS = 38_031
DEVELOPMENT_SCOPE = "development_only_2006_2020"
DEVELOPMENT_DISCLOSURE = (
    "2019-2020 outcomes were already inspected during development; this is "
    "exploratory development evidence, not a blind or confirmatory test."
)
METRIC_SUMMARY_FORMAT = "thermoroute.development-controls-metric-summary.v2"
ARCHITECTURE_BUDGET_FORMAT = (
    "thermoroute.development-controls-architecture-budget.v1"
)
REPORT_FORMAT = "thermoroute.development-controls-report.v2"
SEMANTIC_AUDIT_FORMAT = "thermoroute.development-controls-semantic-audit.v3"
SCIENTIFIC_SUMMARY_FORMAT = "thermoroute.development-controls-scientific-summary.v1"
PRIMARY_ESTIMAND = "median_across_stations_of_within_station_rmse_c"
MICRO_RMSE_ROLE = "secondary_not_primary_estimand"
PAIRED_EFFECT_ESTIMAND = (
    "median_across_stations_of_candidate_rmse_minus_reference_rmse_c"
)
FULL_LADDER_ARM_ID = "ThermoRoute-ladder-07_plus_WDSP"
SUMMARY_COLUMNS = (
    "arm_id", "seed", "split", "horizon", "forecast_keys", "stations",
    "median_station_rmse_c", "micro_rmse_c", "micro_mae_c",
)
STATION_RMSE_COLUMNS = (
    "arm_id", "seed", "split", "horizon", "site_id", "forecast_keys",
    "station_rmse_c",
)
PAIRED_EFFECT_COLUMNS = (
    "comparison_family", "comparison_id", "candidate_arm_id",
    "reference_arm_id", "seed", "split", "horizon", "common_forecast_keys",
    "stations", "median_paired_station_rmse_difference_c",
)
CANONICAL_REGISTRY_COLUMNS = ("split", *FORECAST_KEY, "y_true")
PREDICTION_DIGEST_FORMAT = "thermoroute.prediction-content-digest.v1"
WINDOW_REGISTRY_DIGEST_FORMAT = "thermoroute.window-registry-digest.v1"


class DevelopmentControlsContractError(ValueError):
    """The declared Stage-09b scientific contract was violated."""


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    family: str
    feature_set: str
    variables: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class PairedComparison:
    comparison_family: str
    comparison_id: str
    candidate_arm_id: str
    reference_arm_id: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class MatrixAudit:
    expected_members: int
    prediction_rows: int
    common_forecast_keys: int
    splits: tuple[str, ...]
    reference_member: str


@dataclass(frozen=True)
class CanonicalWindowContract:
    registry: pd.DataFrame
    train_examples: int
    stations: tuple[str, ...]
    registry_sha256: str
    train_registry_sha256: str


def declared_arms() -> tuple[ArmSpec, ...]:
    controls = (
        ArmSpec(
            "PlainMLP-7var", "PlainMLP", "all_7_variables",
            FULL_VARIABLES, CONTROL_SEEDS,
        ),
        ArmSpec(
            "PlainCausalTCN-7var", "PlainCausalTCN", "all_7_variables",
            FULL_VARIABLES, CONTROL_SEEDS,
        ),
    )
    ladder = tuple(
        ArmSpec(
            f"ThermoRoute-ladder-{rung}",
            "ThermoRoute",
            f"feature_ladder_{rung}",
            variables,
            LADDER_SEEDS,
        )
        for rung, variables in FEATURE_LADDER
    )
    return controls + ladder


def expected_member_registry(
    arms: Sequence[ArmSpec] | None = None,
) -> tuple[tuple[str, int], ...]:
    selected = declared_arms() if arms is None else tuple(arms)
    members = tuple((arm.arm_id, int(seed)) for arm in selected for seed in arm.seeds)
    if len(members) != len(set(members)):
        raise DevelopmentControlsContractError("control member registry is not unique")
    return members


def paired_comparison_registry() -> tuple[PairedComparison, ...]:
    """Return the frozen same-seed descriptive-comparison registry.

    The full ThermoRoute arm is compared with both matched neural controls on
    seeds 0--2.  Each feature-ladder rung is compared only with its immediately
    preceding rung on those same seeds.  The latter comparisons are therefore
    fixed-order, path-dependent contrasts, not independent feature importance
    or causal effects.
    """
    controls = tuple(
        PairedComparison(
            comparison_family="full_vs_control",
            comparison_id=f"{FULL_LADDER_ARM_ID}-minus-{reference}",
            candidate_arm_id=FULL_LADDER_ARM_ID,
            reference_arm_id=reference,
            seeds=LADDER_SEEDS,
        )
        for reference in ("PlainMLP-7var", "PlainCausalTCN-7var")
    )
    ladder_arm_ids = tuple(
        f"ThermoRoute-ladder-{rung}" for rung, _variables in FEATURE_LADDER
    )
    adjacent = tuple(
        PairedComparison(
            comparison_family="adjacent_feature_ladder",
            comparison_id=f"{candidate}-minus-{reference}",
            candidate_arm_id=candidate,
            reference_arm_id=reference,
            seeds=LADDER_SEEDS,
        )
        for reference, candidate in zip(
            ladder_arm_ids[:-1], ladder_arm_ids[1:], strict=True,
        )
    )
    return controls + adjacent


def physics_count(variables: Sequence[str]) -> int:
    return sum(variable in DS.PHYS_FORCINGS for variable in variables)


def build_arm_model(arm: ArmSpec, *, seed: int, n_stations: int) -> torch.nn.Module:
    if arm.family == "PlainMLP":
        return PlainMLPForecaster(
            n_vars=len(arm.variables), context_length=C.CONTEXT_LENGTH,
            horizons=C.HORIZONS, n_stations=n_stations,
            station_agnostic=False, init_seed=int(seed),
            hidden_dim=MLP_HIDDEN_DIM, depth=2, dropout=TRAIN_CONFIG.dropout,
        )
    if arm.family == "PlainCausalTCN":
        return PlainCausalTCNForecaster(
            n_vars=len(arm.variables), context_length=C.CONTEXT_LENGTH,
            horizons=C.HORIZONS, n_stations=n_stations,
            station_agnostic=False, init_seed=int(seed),
            channels=TCN_CHANNELS, blocks=4, kernel_size=3,
            dropout=TRAIN_CONFIG.dropout,
        )
    if arm.family == "ThermoRoute":
        return ThermoRoute(
            n_vars=len(arm.variables), n_stations=n_stations,
            horizons=C.HORIZONS, cfg=TRAIN_CONFIG,
            n_phys=physics_count(arm.variables), station_agnostic=False,
            delta_scale=C.DELTA_SCALE, safety_anchor="damped",
        )
    raise DevelopmentControlsContractError(f"unknown control family: {arm.family}")


def parameter_count(arm: ArmSpec, *, n_stations: int = 120) -> int:
    model = build_arm_model(arm, seed=arm.seeds[0], n_stations=n_stations)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_configuration(
    arm: ArmSpec, *, seed: int, n_stations: int,
) -> dict[str, Any]:
    model = build_arm_model(arm, seed=seed, n_stations=n_stations)
    if isinstance(model, (PlainMLPForecaster, PlainCausalTCNForecaster)):
        metadata = model.architecture_metadata()
    elif isinstance(model, ThermoRoute):
        metadata = {
            "format_version": 2,
            "architecture_id": "thermoroute_full_v2",
            "module": model.__class__.__module__,
            "class_name": model.__class__.__name__,
            "constructor_kwargs": {
                "n_vars": len(arm.variables),
                "n_stations": n_stations,
                "horizons": list(C.HORIZONS),
                "train_config": asdict(TRAIN_CONFIG),
                "n_phys": physics_count(arm.variables),
                "station_agnostic": False,
                "delta_scale": C.DELTA_SCALE,
                "safety_anchor": "damped",
            },
            "initialization_seed": int(seed),
            "trainable_parameters": model.n_params(),
            "input_variables": list(arm.variables),
        }
    else:  # pragma: no cover
        raise DevelopmentControlsContractError("unsupported control architecture")
    return json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))


def architecture_template(arm: ArmSpec, *, n_stations: int) -> dict[str, Any]:
    template = architecture_configuration(
        arm, seed=arm.seeds[0], n_stations=n_stations,
    )
    constructor = template.get("constructor_kwargs")
    if isinstance(constructor, dict) and "init_seed" in constructor:
        constructor["init_seed"] = "member_seed"
    if "initialization_seed" in template:
        template["initialization_seed"] = "member_seed"
    template["initialization_seed_policy"] = "exact declared member seed"
    return template


def assert_parameter_budgets(
    arms: Sequence[ArmSpec] | None = None, *, n_stations: int,
) -> dict[str, int]:
    selected = declared_arms() if arms is None else tuple(arms)
    counts = {arm.arm_id: parameter_count(arm, n_stations=n_stations) for arm in selected}
    expected = {
        "PlainMLP-7var": MLP_EXPECTED_PARAMETERS,
        "PlainCausalTCN-7var": TCN_EXPECTED_PARAMETERS,
        "ThermoRoute-ladder-07_plus_WDSP": THERMOROUTE_REFERENCE_PARAMETERS,
    }
    wrong = {key: (counts.get(key), value) for key, value in expected.items()
             if counts.get(key) != value}
    if wrong:
        raise DevelopmentControlsContractError(f"architecture parameter drift: {wrong}")
    for arm_id in ("PlainMLP-7var", "PlainCausalTCN-7var"):
        ratio = abs(counts[arm_id] - THERMOROUTE_REFERENCE_PARAMETERS) / (
            THERMOROUTE_REFERENCE_PARAMETERS
        )
        if ratio > 0.02:
            raise DevelopmentControlsContractError(f"{arm_id} exceeds the 2% budget")
    return counts


def architecture_budget_rows(
    arms: Sequence[ArmSpec] | None = None,
    *,
    n_stations: int,
    train_examples: int,
) -> pd.DataFrame:
    selected = declared_arms() if arms is None else tuple(arms)
    if type(train_examples) is not int or train_examples < 1:
        raise DevelopmentControlsContractError("train-example budget must be positive")
    counts = assert_parameter_budgets(selected, n_stations=n_stations)
    steps_per_epoch = math.ceil(train_examples / TRAIN_CONFIG.batch_size)
    rows: list[dict[str, Any]] = []
    for arm in selected:
        count = counts[arm.arm_id]
        rows.append({
            "arm_id": arm.arm_id,
            "family": arm.family,
            "feature_set": arm.feature_set,
            "variables": "+".join(arm.variables),
            "variable_count": len(arm.variables),
            "seed_count": len(arm.seeds),
            "seeds": ",".join(str(seed) for seed in arm.seeds),
            "trainable_parameters": count,
            "thermoroute_full_reference_parameters": THERMOROUTE_REFERENCE_PARAMETERS,
            "parameter_difference_from_full_thermoroute": (
                count - THERMOROUTE_REFERENCE_PARAMETERS
            ),
            "parameter_ratio_to_full_thermoroute": (
                count / THERMOROUTE_REFERENCE_PARAMETERS
            ),
            "matched_within_2pct_of_full_thermoroute": (
                abs(count - THERMOROUTE_REFERENCE_PARAMETERS)
                / THERMOROUTE_REFERENCE_PARAMETERS <= 0.02
            ),
            "context_length": C.CONTEXT_LENGTH,
            "horizons": ",".join(str(horizon) for horizon in C.HORIZONS),
            "optimizer": "torch.optim.AdamW",
            "learning_rate": TRAIN_CONFIG.lr,
            "weight_decay": TRAIN_CONFIG.weight_decay,
            "batch_size": TRAIN_CONFIG.batch_size,
            "max_epochs": TRAIN_CONFIG.max_epochs,
            "early_stopping_patience": TRAIN_CONFIG.patience,
            "selection_metric": "station_macro_rmse",
            "station_sampling": "equal_station_fixed_size_bootstrap",
            "train_examples_per_epoch": train_examples,
            "maximum_optimizer_steps_per_seed": steps_per_epoch * TRAIN_CONFIG.max_epochs,
            "architecture_candidates_in_this_entrypoint": 1,
            "architecture_configuration": json.dumps(
                architecture_template(arm, n_stations=n_stations),
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ),
            "mlp_hidden_dim": MLP_HIDDEN_DIM if arm.family == "PlainMLP" else None,
            "mlp_depth": 2 if arm.family == "PlainMLP" else None,
            "tcn_channels": TCN_CHANNELS if arm.family == "PlainCausalTCN" else None,
            "tcn_blocks": 4 if arm.family == "PlainCausalTCN" else None,
            "tcn_kernel_size": 3 if arm.family == "PlainCausalTCN" else None,
            "thermoroute_d_model": TRAIN_CONFIG.d_model if arm.family == "ThermoRoute" else None,
            "historical_tuning_budget_equalized": False,
            "training_device": "cpu",
            "evidence_role": "development_only_exploratory",
        })
    return pd.DataFrame(rows)


def budget_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def _update_strings(digest: Any, values: pd.Series) -> None:
    for value in values.astype(str):
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)


def _content_digest(
    frame: pd.DataFrame, *, columns: Sequence[str], format_name: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(format_name.encode("ascii"))
    digest.update(struct.pack("<Q", len(frame)))
    integer_columns = {"seed", "horizon"}
    date_columns = {"issue_date", "target_date"}
    float_columns = {"y_true", "y_pred", "q05", "q50", "q95", "p_exceed"}
    for column in columns:
        encoded = column.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        if column in integer_columns:
            values = np.asarray(frame[column], dtype="<i8")
            digest.update(values.tobytes(order="C"))
        elif column in date_columns:
            values = pd.to_datetime(frame[column], errors="raise").to_numpy(
                dtype="datetime64[ns]"
            ).astype("<i8", copy=False)
            digest.update(values.tobytes(order="C"))
        elif column in float_columns:
            values = np.asarray(frame[column], dtype="<f8").copy()
            values[values == 0.0] = 0.0  # canonicalise negative zero
            digest.update(values.tobytes(order="C"))
        else:
            _update_strings(digest, frame[column])
    return digest.hexdigest()


def normalise_window_registry(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(CANONICAL_REGISTRY_COLUMNS):
        raise DevelopmentControlsContractError("window registry schema changed")
    out = frame.loc[:, CANONICAL_REGISTRY_COLUMNS].copy()
    out["split"] = out["split"].astype(str)
    out["site_id"] = out["site_id"].astype(str)
    horizon = pd.to_numeric(out["horizon"], errors="raise")
    if not np.equal(horizon, np.floor(horizon)).all():
        raise DevelopmentControlsContractError("window registry horizon is non-integral")
    out["horizon"] = horizon.astype("int64")
    out["issue_date"] = pd.to_datetime(out["issue_date"], errors="raise")
    out["target_date"] = pd.to_datetime(out["target_date"], errors="raise")
    if out[["issue_date", "target_date"]].isna().any().any():
        raise DevelopmentControlsContractError("window registry dates contain nulls")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype("float64")
    key = ["split", *FORECAST_KEY]
    if out.duplicated(key).any() or not np.isfinite(out["y_true"]).all():
        raise DevelopmentControlsContractError("window registry has duplicate/non-finite labels")
    return out.sort_values(key, kind="mergesort").reset_index(drop=True)


def window_registry_digest(frame: pd.DataFrame) -> str:
    normalised = normalise_window_registry(frame)
    return _content_digest(
        normalised, columns=CANONICAL_REGISTRY_COLUMNS,
        format_name=WINDOW_REGISTRY_DIGEST_FORMAT,
    )


def window_registry_from_windowed(
    wd: DS.WindowedData,
    stations: Sequence[str],
    *,
    splits: Sequence[str] = ("val", "calib", "test"),
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    station_ids = tuple(str(site) for site in stations)
    selected_splits = tuple(str(value) for value in splits)
    if not selected_splits or len(selected_splits) != len(set(selected_splits)):
        raise DevelopmentControlsContractError("window registry split set is invalid")
    for row_index in np.where(np.isin(wd.split, selected_splits))[0]:
        site_index = int(wd.station[row_index])
        if not 0 <= site_index < len(station_ids):
            raise DevelopmentControlsContractError("window station index is out of range")
        for column, horizon in enumerate(wd.horizons):
            if not bool(wd.target_valid[row_index, column]):
                raise DevelopmentControlsContractError("development window has a masked target")
            records.append({
                "split": str(wd.split[row_index]),
                "site_id": station_ids[site_index],
                "horizon": int(horizon),
                "issue_date": wd.issue_date[row_index],
                "target_date": wd.target_date[row_index, column],
                "y_true": float(wd.y[row_index, column]),
            })
    return normalise_window_registry(pd.DataFrame.from_records(records))


def rebuild_canonical_window_contract(
    *, panel_path: str | Path, frozen_spec_path: str | Path,
) -> CanonicalWindowContract:
    """Re-run the exact producer preparation/window rules from frozen inputs."""
    bundle = D.prepare_dataset_from_panel(
        str(Path(panel_path).resolve()), frozen_spec=Path(frozen_spec_path).resolve(),
        stable_site_ids=True,
    )
    panel_raw = bundle["panel_raw"]
    panel = bundle["panel"]
    masks = bundle["masks"]
    stations = tuple(str(value) for value in cast(Sequence[object], bundle["stations"]))
    if not isinstance(panel_raw, pd.DataFrame) or not isinstance(panel, pd.DataFrame):
        raise DevelopmentControlsContractError("canonical panel preparation failed")
    if not isinstance(masks, D.SplitMasks):
        raise DevelopmentControlsContractError("canonical split masks are invalid")
    climatology = F.HarmonicClimatology.fit(panel_raw, masks.train)
    reference: pd.DataFrame | None = None
    train_reference: pd.DataFrame | None = None
    train_examples: int | None = None
    unique_variables = tuple(dict.fromkeys(arm.variables for arm in declared_arms()))
    for variables in unique_variables:
        wd = DS.build_windows(
            panel, masks, climatology, context=C.CONTEXT_LENGTH,
            horizons=C.HORIZONS, variables=variables,
            require_observed_target=True,
        )
        current_train = len(wd.idx("train"))
        current_registry = window_registry_from_windowed(wd, stations)
        current_train_registry = window_registry_from_windowed(
            wd, stations, splits=("train",)
        )
        if train_examples is None:
            train_examples = current_train
            reference = current_registry
            train_reference = current_train_registry
        elif current_train != train_examples:
            raise DevelopmentControlsContractError(
                "feature ladder changed the canonical training-example registry"
            )
        else:
            assert reference is not None and train_reference is not None
            keys = ["split", *FORECAST_KEY]
            if (
                not current_registry[keys].equals(reference[keys])
                or not targets_match_at_model_precision(
                    current_registry["y_true"], reference["y_true"]
                )
            ):
                raise DevelopmentControlsContractError(
                    "feature ladder changed the canonical evaluation registry"
                )
            if (
                not current_train_registry[keys].equals(train_reference[keys])
                or not targets_match_at_model_precision(
                    current_train_registry["y_true"], train_reference["y_true"]
                )
            ):
                raise DevelopmentControlsContractError(
                    "feature ladder changed the canonical training registry"
                )
        del wd, current_registry, current_train_registry
    assert (
        reference is not None
        and train_reference is not None
        and train_examples is not None
    )
    return CanonicalWindowContract(
        registry=reference,
        train_examples=train_examples,
        stations=stations,
        registry_sha256=window_registry_digest(reference),
        train_registry_sha256=window_registry_digest(train_reference),
    )


def _is_true_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _assert_prediction_physical_schema(frame: pd.DataFrame) -> None:
    text_columns = ("model", "scope", "feature_set", "site_id", "split")
    for column in text_columns:
        values = frame[column]
        if not (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
        ) or not all(
            isinstance(value, str) and bool(value.strip()) for value in values
        ):
            raise DevelopmentControlsContractError(
                f"prediction {column} values must be non-empty strings"
            )
    for column in ("seed", "horizon"):
        values = frame[column]
        if (
            not pd.api.types.is_integer_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or not all(_is_true_integer(value) for value in values)
        ):
            raise DevelopmentControlsContractError(
                f"prediction {column} values must be true integers"
            )
    for column in ("issue_date", "target_date"):
        values = frame[column]
        if str(values.dtype) != "datetime64[ns]":
            raise DevelopmentControlsContractError(
                f"prediction {column} must have naive datetime64[ns] dtype"
            )
        if values.isna().any():
            raise DevelopmentControlsContractError("prediction dates contain nulls")
        if not values.equals(values.dt.normalize()):
            raise DevelopmentControlsContractError(
                "prediction dates must be timezone-naive normalized days"
            )
    for column in ("y_true", "y_pred", "q05", "q50", "q95", "p_exceed"):
        values = frame[column]
        if not pd.api.types.is_float_dtype(values.dtype) or not all(
            isinstance(value, (float, np.floating)) for value in values
        ):
            raise DevelopmentControlsContractError(
                f"prediction {column} values must have floating dtype"
            )


def normalise_prediction_frame(
    frame: pd.DataFrame,
    *,
    arm: ArmSpec,
    seed: int,
    allowed_sites: set[str] | None = None,
    canonical_registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if list(frame.columns) != R.PRED_COLS:
        raise DevelopmentControlsContractError("prediction columns/order changed")
    _assert_prediction_physical_schema(frame)
    out = frame.loc[:, R.PRED_COLS].copy()
    for column in ("model", "scope", "feature_set", "site_id", "split"):
        if out[column].isna().any():
            raise DevelopmentControlsContractError(f"prediction {column} contains nulls")
        out[column] = out[column].astype(str)
    expected_static = {
        "model": arm.arm_id,
        "scope": DEVELOPMENT_SCOPE,
        "feature_set": arm.feature_set,
        "seed": int(seed),
    }
    for column in ("model", "scope", "feature_set"):
        if set(out[column]) != {str(expected_static[column])}:
            raise DevelopmentControlsContractError(f"prediction {column} changed")
    numeric_seed = out["seed"].astype("int64")
    if set(numeric_seed) != {int(seed)}:
        raise DevelopmentControlsContractError("prediction seed changed")
    out["seed"] = numeric_seed
    out["horizon"] = out["horizon"].astype("int64")
    numeric = ("y_true", "y_pred", "q05", "q50", "q95", "p_exceed")
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="raise").astype("float64")
    if not np.isfinite(out.loc[:, numeric].to_numpy(dtype=float)).all():
        raise DevelopmentControlsContractError("prediction numeric outputs must all be finite")
    if not (
        (out["q05"] <= out["q50"]) & (out["q50"] <= out["q95"])
    ).all():
        raise DevelopmentControlsContractError("prediction quantiles are not ordered")
    if not out["p_exceed"].between(0.0, 1.0, inclusive="both").all():
        raise DevelopmentControlsContractError("prediction probability is outside [0,1]")
    if set(out["split"]) != {"val", "calib", "test"}:
        raise DevelopmentControlsContractError("prediction split registry changed")
    if set(out["horizon"]) != set(C.HORIZONS):
        raise DevelopmentControlsContractError("prediction horizon registry changed")
    sites = set(out["site_id"])
    if allowed_sites is not None and sites != allowed_sites:
        raise DevelopmentControlsContractError("prediction stable-site registry changed")
    if any(not site.isdigit() or not 8 <= len(site) <= 15 for site in sites):
        raise DevelopmentControlsContractError("prediction does not use stable USGS site_no")
    expected_target = out["issue_date"] + pd.to_timedelta(out["horizon"], unit="D")
    if not expected_target.equals(out["target_date"]):
        raise DevelopmentControlsContractError("prediction target date/horizon changed")
    key = ["split", *FORECAST_KEY]
    if out.duplicated(key).any():
        raise DevelopmentControlsContractError("prediction has duplicate forecast keys")
    out = out.sort_values(key, kind="mergesort").reset_index(drop=True)
    if canonical_registry is not None:
        expected = normalise_window_registry(canonical_registry)
        if (
            not out[key].equals(expected[key])
            or not targets_match_at_model_precision(out["y_true"], expected["y_true"])
        ):
            raise DevelopmentControlsContractError(
                "prediction does not equal the rebuilt canonical window registry"
            )
        # Downstream metrics and digests must use the rebuilt target bytes, not
        # an attacker-controlled float64 value that merely rounds to the same
        # model-precision target.
        out["y_true"] = expected["y_true"].to_numpy(dtype="float64")
    return out


def prediction_content_digest(frame: pd.DataFrame) -> str:
    if list(frame.columns) != R.PRED_COLS:
        raise DevelopmentControlsContractError("prediction digest requires canonical columns")
    return _content_digest(
        frame, columns=R.PRED_COLS, format_name=PREDICTION_DIGEST_FORMAT,
    )


def _stable_error_metrics(
    error: np.ndarray, *, context: str,
) -> tuple[float, float]:
    values = np.asarray(error, dtype="float64")
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise DevelopmentControlsContractError(
            f"{context} overflows finite metric arithmetic"
        )
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 0.0, 0.0
    scaled = values / scale
    rmse = scale * float(np.sqrt(np.mean(scaled ** 2)))
    mae = scale * float(np.mean(np.abs(scaled)))
    if not math.isfinite(rmse) or not math.isfinite(mae):
        raise DevelopmentControlsContractError(
            f"{context} metrics are non-finite"
        )
    return rmse, mae


def recompute_station_rmse(
    frames: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Derive the station-level RMSE units used by every primary summary."""
    rows: list[dict[str, Any]] = []
    for (arm_id, seed), frame in frames.items():
        required = {"split", "horizon", "site_id", "y_pred", "y_true"}
        if not required.issubset(frame.columns):
            raise DevelopmentControlsContractError(
                "station RMSE input lacks required prediction columns"
            )
        for (split, horizon, site_id), group in frame.groupby(
            ["split", "horizon", "site_id"], sort=True,
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                error = (
                    group["y_pred"].to_numpy(dtype="float64")
                    - group["y_true"].to_numpy(dtype="float64")
                )
            station_rmse, _station_mae = _stable_error_metrics(
                error,
                context=(
                    f"{arm_id}/seed{seed}/{split}/h{horizon}/{site_id} "
                    "station error"
                ),
            )
            rows.append({
                "arm_id": str(arm_id),
                "seed": int(seed),
                "split": str(split),
                "horizon": int(horizon),
                "site_id": str(site_id),
                "forecast_keys": int(len(group)),
                "station_rmse_c": station_rmse,
            })
    station = pd.DataFrame.from_records(rows, columns=STATION_RMSE_COLUMNS)
    if station.empty:
        raise DevelopmentControlsContractError("station RMSE registry is empty")
    return station.sort_values(
        ["arm_id", "seed", "split", "horizon", "site_id"], kind="mergesort"
    ).reset_index(drop=True)


def recompute_metric_summary(
    frames: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    station_metrics = recompute_station_rmse(frames)
    rows: list[dict[str, Any]] = []
    for (arm_id, seed), frame in frames.items():
        for (split, horizon), group in frame.groupby(["split", "horizon"], sort=True):
            with np.errstate(over="ignore", invalid="ignore"):
                error = (
                    group["y_pred"].to_numpy(dtype="float64")
                    - group["y_true"].to_numpy(dtype="float64")
                )
            micro_rmse, micro_mae = _stable_error_metrics(
                error, context="prediction error",
            )
            station_group = station_metrics.loc[
                station_metrics["arm_id"].eq(str(arm_id))
                & station_metrics["seed"].eq(int(seed))
                & station_metrics["split"].eq(str(split))
                & station_metrics["horizon"].eq(int(horizon))
            ]
            if station_group.empty:
                raise DevelopmentControlsContractError(
                    "member metric has no station-level RMSE units"
                )
            median_station_rmse = float(
                np.median(station_group["station_rmse_c"].to_numpy(dtype="float64"))
            )
            if not math.isfinite(median_station_rmse):
                raise DevelopmentControlsContractError(
                    "station-median RMSE is non-finite"
                )
            rows.append({
                "arm_id": str(arm_id), "seed": int(seed), "split": str(split),
                "horizon": int(horizon), "forecast_keys": int(len(group)),
                "stations": int(len(station_group)),
                "median_station_rmse_c": median_station_rmse,
                "micro_rmse_c": micro_rmse,
                "micro_mae_c": micro_mae,
            })
    summary = pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)
    return summary.sort_values(
        ["arm_id", "seed", "split", "horizon"], kind="mergesort"
    ).reset_index(drop=True)


def recompute_paired_effect_summary(
    station_metrics: pd.DataFrame,
    *,
    exact_common_forecast_keys_verified: bool,
) -> pd.DataFrame:
    """Derive frozen same-seed station-paired descriptive RMSE effects.

    The caller must first prove that every member shares the same exact
    forecast-key/truth registry.  This function then verifies the derived
    station registry and per-station key counts for every pair before taking
    the median of station-level RMSE differences.  Negative values favour the
    candidate arm named in the record.
    """
    if exact_common_forecast_keys_verified is not True:
        raise DevelopmentControlsContractError(
            "paired effects require verified exact common forecast keys"
        )
    if list(station_metrics.columns) != list(STATION_RMSE_COLUMNS):
        raise DevelopmentControlsContractError("station RMSE schema changed")
    if station_metrics.empty or station_metrics.duplicated(
        ["arm_id", "seed", "split", "horizon", "site_id"]
    ).any():
        raise DevelopmentControlsContractError("station RMSE registry is incomplete")
    observed_members = set(zip(
        station_metrics["arm_id"].astype(str),
        station_metrics["seed"].astype(int),
        strict=True,
    ))
    if observed_members != set(expected_member_registry()):
        raise DevelopmentControlsContractError(
            "paired effects require the exact 31-member registry"
        )
    if set(station_metrics["split"].astype(str)) != {"val", "calib", "test"}:
        raise DevelopmentControlsContractError("paired-effect split registry changed")
    if set(station_metrics["horizon"].astype(int)) != set(C.HORIZONS):
        raise DevelopmentControlsContractError("paired-effect horizon registry changed")
    numeric = station_metrics[["forecast_keys", "station_rmse_c"]].to_numpy(
        dtype="float64"
    )
    if (
        not np.isfinite(numeric).all()
        or (station_metrics["forecast_keys"] < 1).any()
        or (station_metrics["station_rmse_c"] < 0.0).any()
    ):
        raise DevelopmentControlsContractError("station RMSE values are invalid")

    rows: list[dict[str, Any]] = []
    pair_keys = ["split", "horizon", "site_id"]
    for comparison in paired_comparison_registry():
        for seed in comparison.seeds:
            candidate = station_metrics.loc[
                station_metrics["arm_id"].eq(comparison.candidate_arm_id)
                & station_metrics["seed"].eq(seed)
            ].sort_values(pair_keys, kind="mergesort").reset_index(drop=True)
            reference = station_metrics.loc[
                station_metrics["arm_id"].eq(comparison.reference_arm_id)
                & station_metrics["seed"].eq(seed)
            ].sort_values(pair_keys, kind="mergesort").reset_index(drop=True)
            if (
                candidate.empty
                or reference.empty
                or not candidate[pair_keys].equals(reference[pair_keys])
                or not np.array_equal(
                    candidate["forecast_keys"].to_numpy(dtype="int64"),
                    reference["forecast_keys"].to_numpy(dtype="int64"),
                )
            ):
                raise DevelopmentControlsContractError(
                    f"{comparison.comparison_id}/seed{seed} station registry changed"
                )
            paired = candidate[pair_keys + ["forecast_keys"]].copy()
            paired["effect_c"] = (
                candidate["station_rmse_c"].to_numpy(dtype="float64")
                - reference["station_rmse_c"].to_numpy(dtype="float64")
            )
            if not np.isfinite(paired["effect_c"]).all():
                raise DevelopmentControlsContractError(
                    "paired station RMSE effect is non-finite"
                )
            for (split, horizon), group in paired.groupby(
                ["split", "horizon"], sort=True,
            ):
                effect = float(np.median(group["effect_c"].to_numpy(dtype="float64")))
                if not math.isfinite(effect):
                    raise DevelopmentControlsContractError(
                        "paired station RMSE summary is non-finite"
                    )
                rows.append({
                    "comparison_family": comparison.comparison_family,
                    "comparison_id": comparison.comparison_id,
                    "candidate_arm_id": comparison.candidate_arm_id,
                    "reference_arm_id": comparison.reference_arm_id,
                    "seed": int(seed),
                    "split": str(split),
                    "horizon": int(horizon),
                    "common_forecast_keys": int(group["forecast_keys"].sum()),
                    "stations": int(len(group)),
                    "median_paired_station_rmse_difference_c": effect,
                })
    paired_summary = pd.DataFrame.from_records(rows, columns=PAIRED_EFFECT_COLUMNS)
    expected_identities = [
        (
            comparison.comparison_family,
            comparison.comparison_id,
            comparison.candidate_arm_id,
            comparison.reference_arm_id,
            seed,
            split,
            horizon,
        )
        for comparison in paired_comparison_registry()
        for seed in LADDER_SEEDS
        for split in ("calib", "test", "val")
        for horizon in C.HORIZONS
    ]
    observed_identities = list(paired_summary[[
        "comparison_family", "comparison_id", "candidate_arm_id",
        "reference_arm_id", "seed", "split", "horizon",
    ]].itertuples(index=False, name=None))
    if observed_identities != expected_identities:
        raise DevelopmentControlsContractError("paired-effect registry is incomplete")
    return paired_summary.reset_index(drop=True)


def scientific_summary_document(paired_effects: pd.DataFrame) -> dict[str, Any]:
    """Build the machine-readable estimand and paired-effect contract."""
    if list(paired_effects.columns) != list(PAIRED_EFFECT_COLUMNS):
        raise DevelopmentControlsContractError("paired-effect schema changed")
    records: list[dict[str, Any]] = [
        {
            "comparison_family": str(row.comparison_family),
            "comparison_id": str(row.comparison_id),
            "candidate_arm_id": str(row.candidate_arm_id),
            "reference_arm_id": str(row.reference_arm_id),
            "seed": int(row.seed),
            "split": str(row.split),
            "horizon": int(row.horizon),
            "common_forecast_keys": int(row.common_forecast_keys),
            "stations": int(row.stations),
            "median_paired_station_rmse_difference_c": float(
                row.median_paired_station_rmse_difference_c
            ),
        }
        for row in paired_effects.itertuples(index=False)
    ]
    if any(
        not math.isfinite(record["median_paired_station_rmse_difference_c"])
        for record in records
    ):
        raise DevelopmentControlsContractError("paired-effect record is non-finite")
    records_bytes = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return {
        "format": SCIENTIFIC_SUMMARY_FORMAT,
        "metric_summary_format": METRIC_SUMMARY_FORMAT,
        "primary_member_estimand": {
            "name": PRIMARY_ESTIMAND,
            "column": "median_station_rmse_c",
            "unit": "degree_Celsius",
            "aggregation": "median_of_within_station_RMSE",
            "station_weighting": "one_station_one_value",
        },
        "secondary_member_estimands": {
            "micro_rmse_c": {
                "role": MICRO_RMSE_ROLE,
                "aggregation": "RMSE_over_all_forecast_keys",
            },
            "micro_mae_c": {
                "role": "secondary_not_primary_estimand",
                "aggregation": "MAE_over_all_forecast_keys",
            },
        },
        "paired_descriptive_effects": {
            "estimand": PAIRED_EFFECT_ESTIMAND,
            "effect_convention": "candidate_minus_reference",
            "negative_favours": "candidate",
            "same_seed": True,
            "exact_common_forecast_keys_verified": True,
            "comparison_registry": [
                {
                    "comparison_family": comparison.comparison_family,
                    "comparison_id": comparison.comparison_id,
                    "candidate_arm_id": comparison.candidate_arm_id,
                    "reference_arm_id": comparison.reference_arm_id,
                    "seeds": list(comparison.seeds),
                }
                for comparison in paired_comparison_registry()
            ],
            "feature_ladder_order": [
                {"rung": rung, "variables": list(variables)}
                for rung, variables in FEATURE_LADDER
            ],
            "feature_ladder_fixed_order_path_dependent": True,
            "independent_feature_contribution_claimed": False,
            "causal_effect_claimed": False,
            "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
            "records": records,
        },
    }


def summary_csv_bytes(frame: pd.DataFrame) -> bytes:
    if list(frame.columns) != list(SUMMARY_COLUMNS):
        raise DevelopmentControlsContractError("metric summary schema changed")
    return frame.to_csv(
        index=False, float_format="%.17g", lineterminator="\n"
    ).encode("utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not math.isfinite(float(value)) else f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def render_report(
    *, run_id: str, audit: MatrixAudit | Mapping[str, Any],
    budget: pd.DataFrame, summary: pd.DataFrame, paired_effects: pd.DataFrame,
) -> str:
    values = asdict(audit) if isinstance(audit, MatrixAudit) else dict(audit)
    development = summary.loc[summary["split"].eq("test")]
    aggregate = (
        development.groupby(["arm_id", "horizon"], as_index=False)
        .agg(
            primary_median_station_rmse_seed_mean_c=(
                "median_station_rmse_c", "mean",
            ),
            primary_median_station_rmse_seed_sd_c=(
                "median_station_rmse_c", "std",
            ),
            secondary_micro_rmse_seed_mean_c=("micro_rmse_c", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(
            ["horizon", "primary_median_station_rmse_seed_mean_c", "arm_id"],
            kind="mergesort",
        )
    )
    result_table = _markdown_table(aggregate)
    development_pairs = paired_effects.loc[paired_effects["split"].eq("test")]
    pair_columns = [
        "candidate_arm_id", "reference_arm_id", "seed", "horizon", "stations",
        "median_paired_station_rmse_difference_c",
    ]
    control_effects = _markdown_table(
        development_pairs.loc[
            development_pairs["comparison_family"].eq("full_vs_control"),
            pair_columns,
        ]
    )
    ladder_effects = _markdown_table(
        development_pairs.loc[
            development_pairs["comparison_family"].eq(
                "adjacent_feature_ladder"
            ),
            pair_columns,
        ]
    )
    budget_view = _markdown_table(budget[[
        "arm_id", "variables", "seed_count", "trainable_parameters",
        "parameter_ratio_to_full_thermoroute", "maximum_optimizer_steps_per_seed",
    ]])
    splits = values["splits"]
    return f"""# Development-only neural controls and feature ladder

Run ID: `{run_id}`

Status: **COMPLETE BEST-MODEL-STATE PREDICTION REPLAY**. Every stored prediction
member is reproduced from the safely loaded checkpoint `best_model_state` and
derived artifacts are regenerated. This is not optimiser-step/trajectory replay
and is not part of the sealed confirmatory model suite.

> {DEVELOPMENT_DISCLOSURE}

## Design

All models use the frozen 120-site 2006--2020 panel, 32 days of history,
horizons 1/3/7 days, CPU-only deterministic execution, equal-station fixed-size
bootstrap sampling, AdamW, the same declared maximum optimisation budget, and
early-stopping rule. PlainMLP and PlainCausalTCN receive the seven declared
history variables and masks. ThermoRoute additionally receives its declared
train-fit/calendar-derived physical-anchor inputs. The feature ladder adds one
declared variable at a time in the fixed order WTEMP, FLOW, TEMP, PRCP, RHMEAN,
DH, WDSP.

The two pure-neural controls are parameter-matched within 2% of the full
ThermoRoute architecture. Each architecture has one fixed candidate here.
This does not equalise ThermoRoute's historical tuning advantage, so
`historical_tuning_budget_equalized` remains false.

Exact member count: {values['expected_members']}. Common forecast keys per member:
{values['common_forecast_keys']}. Total prediction rows: {values['prediction_rows']}.
Validated splits: {', '.join(splits)}.

## Architecture and declared maximum optimisation budget

{budget_view}

## 2019--2020 development-evaluation results

These values are deterministically derived from the machine-readable summary,
which is itself recomputed from every stored prediction row. `test` means the
already-inspected 2019--2020 development partition, never a blind test.

The primary member-level estimand is median station RMSE: RMSE is first computed
within each station on the exact common forecast keys and then the station RMSEs
are aggregated by their median. Micro RMSE is retained only as a secondary,
non-primary estimand; it weights stations according to their available row count.

{result_table}

## Same-seed station-paired descriptive effects

Every effect below is the median across stations of candidate RMSE minus
reference RMSE on the same seed and exact common forecast keys. Negative values
favour the candidate. These are descriptive development effects, not hypothesis
tests.

### Full ThermoRoute versus matched neural controls

{control_effects}

### Adjacent cumulative feature-ladder rungs

{ladder_effects}

The ladder order is fixed as WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP. Each
adjacent contrast is path-dependent on every preceding rung. It is not an
independent feature contribution, feature importance score, or causal effect.

## Interpretation boundary

These artifacts describe architecture comparisons and fixed-path adjacent
ladder contrasts on historical development data. They verify best-state
prediction replay, not the
full training trajectory, and cannot establish prospective, operational, causal,
safety, or confirmatory performance. They do not modify the frozen Route-A suite
pointer.
"""
