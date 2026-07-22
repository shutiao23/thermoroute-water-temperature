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
SUMMARY_COLUMNS = ("arm_id", "seed", "split", "horizon", "n", "rmse", "mae")
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
    numeric_seed = pd.to_numeric(out["seed"], errors="raise")
    if not np.equal(numeric_seed, np.floor(numeric_seed)).all() or set(
        numeric_seed.astype("int64")
    ) != {int(seed)}:
        raise DevelopmentControlsContractError("prediction seed changed")
    out["seed"] = numeric_seed.astype("int64")
    horizon = pd.to_numeric(out["horizon"], errors="raise")
    if not np.equal(horizon, np.floor(horizon)).all():
        raise DevelopmentControlsContractError("prediction horizon is non-integral")
    out["horizon"] = horizon.astype("int64")
    out["issue_date"] = pd.to_datetime(out["issue_date"], errors="raise")
    out["target_date"] = pd.to_datetime(out["target_date"], errors="raise")
    if out[["issue_date", "target_date"]].isna().any().any():
        raise DevelopmentControlsContractError("prediction dates contain nulls")
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


def recompute_metric_summary(
    frames: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm_id, seed), frame in frames.items():
        for (split, horizon), group in frame.groupby(["split", "horizon"], sort=True):
            with np.errstate(over="ignore", invalid="ignore"):
                error = (
                    group["y_pred"].to_numpy(float)
                    - group["y_true"].to_numpy(float)
                )
            if not np.isfinite(error).all():
                raise DevelopmentControlsContractError(
                    "prediction error overflows finite metric arithmetic"
                )
            scale = float(np.max(np.abs(error), initial=0.0))
            rmse = (
                0.0
                if scale == 0.0
                else scale * float(np.sqrt(np.mean((error / scale) ** 2)))
            )
            mae = float(np.mean(np.abs(error)))
            if not math.isfinite(rmse) or not math.isfinite(mae):
                raise DevelopmentControlsContractError(
                    "prediction-derived metrics are non-finite"
                )
            rows.append({
                "arm_id": str(arm_id), "seed": int(seed), "split": str(split),
                "horizon": int(horizon), "n": int(len(group)),
                "rmse": rmse,
                "mae": mae,
            })
    summary = pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)
    return summary.sort_values(
        ["arm_id", "seed", "split", "horizon"], kind="mergesort"
    ).reset_index(drop=True)


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
    budget: pd.DataFrame, summary: pd.DataFrame,
) -> str:
    values = asdict(audit) if isinstance(audit, MatrixAudit) else dict(audit)
    development = summary.loc[summary["split"].eq("test")]
    aggregate = (
        development.groupby(["arm_id", "horizon"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), seeds=("seed", "nunique"))
        .sort_values(["horizon", "rmse_mean", "arm_id"], kind="mergesort")
    )
    result_table = _markdown_table(aggregate)
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

{result_table}

## Interpretation boundary

These artifacts diagnose architecture and cumulative feature contribution on
historical development data. They verify best-state prediction replay, not the
full training trajectory, and cannot establish prospective, operational, causal,
safety, or confirmatory performance. They do not modify the frozen Route-A suite
pointer.
"""
