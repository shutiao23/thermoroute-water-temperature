"""Reconstruct Route-A preprocessing and sequence models from frozen bundles.

The confirmatory evaluator must never call a ``fit`` method after labels are
opened.  This module turns the JSON metadata written by Stage 9 into immutable
transform objects and builds confirmation windows with those objects.  It also
defines the stricter contract for the external new-site arm: every learned
station effect and every preprocessing statistic must be pooled, and the model
must have been trained with ``station_agnostic=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import datasets as DS
from . import features as F
from .checkpoint import neural_output_head_schema
from .thermoroute import ThermoRoute
from .train import LSTMForecaster
from .weighting import ROW_EQUAL_WEIGHTING, STATION_EQUAL_WEIGHTING


class FrozenInferenceError(RuntimeError):
    """Raised when a bundle cannot prove label-free frozen inference."""


def _pair_key(key: object) -> tuple[str, str]:
    values = str(key).split("|", 1)
    if len(values) != 2 or not all(values):
        raise FrozenInferenceError(f"invalid station/variable key: {key!r}")
    return values[0], values[1]


def _float(value: object, *, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FrozenInferenceError(f"{label} is not numeric") from exc
    if not np.isfinite(number):
        raise FrozenInferenceError(f"{label} is not finite")
    return number


def _same_value(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise FrozenInferenceError(f"pooled {label} is empty")
    first = float(values[0])
    if not np.allclose(np.asarray(values, dtype=float), first, rtol=0.0, atol=1e-12):
        raise FrozenInferenceError(
            f"external bundle declares pooled {label}, but stored values differ"
        )
    return first


def _architecture_kwargs(
    metadata: Mapping[str, Any], *, thermoroute_only: bool = False
) -> dict[str, Any]:
    if metadata.get("output_head_schema") != neural_output_head_schema():
        raise FrozenInferenceError(
            "bundle lacks the independent point/q50 output-head schema"
        )
    architecture = metadata.get("architecture")
    if not isinstance(architecture, Mapping):
        raise FrozenInferenceError("bundle lacks an architecture mapping")
    if thermoroute_only and (
        architecture.get("class") != "thermoroute.thermoroute.ThermoRoute"
    ):
        raise FrozenInferenceError("bundle is not a reconstructable ThermoRoute model")
    kwargs = architecture.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise FrozenInferenceError("bundle architecture lacks constructor kwargs")
    return dict(kwargs)


def _ordered_training_stations(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    mapping = metadata.get("station_to_index")
    if not isinstance(mapping, Mapping) or not mapping:
        raise FrozenInferenceError("bundle station_to_index is empty")
    try:
        ordered = sorted(((int(index), str(site)) for site, index in mapping.items()))
    except (TypeError, ValueError) as exc:
        raise FrozenInferenceError("station_to_index contains a non-integer index") from exc
    if [index for index, _ in ordered] != list(range(len(ordered))):
        raise FrozenInferenceError("station_to_index is not contiguous from zero")
    sites = tuple(site for _, site in ordered)
    if len(sites) != len(set(sites)):
        raise FrozenInferenceError("station_to_index contains duplicate sites")
    return sites


@dataclass(frozen=True)
class FrozenTransforms:
    """All train-fit transforms needed to build one inference tensor registry."""

    feature_order: tuple[str, ...]
    imputer_seasonal: Mapping[tuple[str, str], Mapping[int, float]]
    imputer_global: Mapping[tuple[str, str], float]
    scaler: D.StandardScalerPerStation
    climatology: F.HarmonicClimatology
    damped_anchor: F.DampedPersistenceAnchor
    station_ids: tuple[str, ...]
    station_agnostic: bool

    def impute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply only frozen medians; no statistic is estimated from ``panel``."""
        required = {"DATE", "site_id", *self.feature_order}
        missing = required - set(panel)
        if missing:
            raise FrozenInferenceError(
                f"confirmation panel lacks bundle fields: {sorted(missing)}"
            )
        out = panel.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        sites = set(out["site_id"].astype(str))
        if sites != set(self.station_ids):
            raise FrozenInferenceError(
                "confirmation panel station registry differs from the frozen cohort"
            )
        doy = out["DATE"].dt.dayofyear.to_numpy()
        for variable in self.feature_order:
            out[f"{variable}_observed"] = pd.to_numeric(
                out[variable], errors="coerce"
            ).notna()
        for station in self.station_ids:
            station_mask = out["site_id"].astype(str).eq(station).to_numpy()
            for variable in self.feature_order:
                key = (station, variable)
                if key not in self.imputer_seasonal or key not in self.imputer_global:
                    raise FrozenInferenceError(
                        f"frozen imputer lacks {station}|{variable}"
                    )
                values = pd.to_numeric(out[variable], errors="coerce").to_numpy(
                    dtype=float, copy=True
                )
                missing_mask = station_mask & ~np.isfinite(values)
                if not missing_mask.any():
                    continue
                seasonal = self.imputer_seasonal[key]
                fill = np.asarray(
                    [seasonal.get(int(day), np.nan) for day in doy[missing_mask]],
                    dtype=float,
                )
                fill = np.where(
                    np.isfinite(fill), fill, self.imputer_global[key]
                )
                values[missing_mask] = fill
                out[variable] = values
        if not np.isfinite(out[list(self.feature_order)].to_numpy(float)).all():
            raise FrozenInferenceError("frozen imputer left non-finite model inputs")
        return out


def _normalise_seasonal_map(
    raw: Mapping[str, Any],
) -> dict[tuple[str, str], dict[int, float]]:
    output: dict[tuple[str, str], dict[int, float]] = {}
    for raw_key, values in raw.items():
        key = _pair_key(raw_key)
        if not isinstance(values, Mapping) or not values:
            raise FrozenInferenceError(f"seasonal imputer {raw_key!r} is empty")
        days: dict[int, float] = {}
        for day, value in values.items():
            try:
                integer_day = int(day)
            except (TypeError, ValueError) as exc:
                raise FrozenInferenceError(
                    f"seasonal imputer {raw_key!r} has an invalid day"
                ) from exc
            if not 1 <= integer_day <= 366:
                raise FrozenInferenceError(
                    f"seasonal imputer {raw_key!r} day is outside 1..366"
                )
            days[integer_day] = _float(value, label=f"imputer {raw_key}|{day}")
        output[key] = days
    return output


def _normalise_numeric_pair_map(
    raw: Mapping[str, Any], *, label: str
) -> dict[tuple[str, str], float]:
    return {
        _pair_key(key): _float(value, label=f"{label} {key}")
        for key, value in raw.items()
    }


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenInferenceError(f"bundle preprocessing lacks {label}")
    return value


def _require_method(
    block: Mapping[str, Any], expected: str, *, label: str
) -> None:
    if block.get("method") != expected:
        raise FrozenInferenceError(f"bundle {label} method differs from frozen code")


def _pooled_seasonal(
    stored: Mapping[tuple[str, str], Mapping[int, float]],
    variable: str,
) -> dict[int, float]:
    explicit = stored.get(("__pooled__", variable))
    if explicit is not None:
        return dict(explicit)
    candidates = [dict(values) for (site, name), values in stored.items()
                  if name == variable and site != "__pooled__"]
    if not candidates:
        raise FrozenInferenceError(f"external imputer lacks pooled {variable} medians")
    first = candidates[0]
    if any(values != first for values in candidates[1:]):
        raise FrozenInferenceError(
            f"external imputer is not pooled for {variable}; new-site labels could leak"
        )
    return first


def _pooled_numeric(
    stored: Mapping[tuple[str, str], float],
    variable: str,
    *,
    label: str,
) -> float:
    explicit = stored.get(("__pooled__", variable))
    if explicit is not None:
        return float(explicit)
    candidates = [value for (site, name), value in stored.items()
                  if name == variable and site != "__pooled__"]
    return _same_value(candidates, label=f"{label} {variable}")


def reconstruct_frozen_transforms(
    metadata: Mapping[str, Any],
    station_ids: Sequence[str],
    *,
    external: bool,
) -> FrozenTransforms:
    """Deserialize transforms, rejecting any need to fit on confirmation data."""
    station_ids = tuple(str(site) for site in station_ids)
    if not station_ids or len(station_ids) != len(set(station_ids)):
        raise FrozenInferenceError("inference station registry is empty or duplicated")
    feature_order = tuple(str(value) for value in metadata.get("feature_order", ()))
    if not feature_order or "WTEMP" not in feature_order:
        raise FrozenInferenceError("bundle feature_order must contain WTEMP")
    if len(feature_order) != len(set(feature_order)):
        raise FrozenInferenceError("bundle feature_order contains duplicates")

    preprocessing = _require_mapping(metadata.get("preprocessing"), label="preprocessing")
    schema = _require_mapping(preprocessing.get("input_schema"), label="input_schema")
    if tuple(schema.get("variables", ())) != feature_order:
        raise FrozenInferenceError("input schema and bundle feature_order disagree")
    expected_transforms = {
        variable: ("signed_log1p" if variable == "FLOW" else "log1p_nonnegative")
        for variable in C.LOG1P_VARS if variable in feature_order
    }
    if schema.get("transforms") != expected_transforms:
        raise FrozenInferenceError(
            "bundle stabilising-transform map differs from the executing code"
        )
    if "log1p_variables" in schema:
        raise FrozenInferenceError(
            "legacy unsigned log1p bundle is not valid for Route-A confirmation"
        )
    if int(schema.get("context_length", -1)) != C.CONTEXT_LENGTH:
        raise FrozenInferenceError("bundle context length differs from Route-A")
    if schema.get("missingness_mask") is not True:
        raise FrozenInferenceError("bundle does not freeze the missingness-mask input")

    imputer = _require_mapping(preprocessing.get("imputer"), label="imputer")
    seasonal = _normalise_seasonal_map(
        _require_mapping(imputer.get("seasonal_medians"), label="seasonal medians")
    )
    global_medians = _normalise_numeric_pair_map(
        _require_mapping(imputer.get("global_medians"), label="global medians"),
        label="imputer global median",
    )
    scaler_block = _require_mapping(preprocessing.get("scaler"), label="scaler")
    means = _normalise_numeric_pair_map(
        _require_mapping(scaler_block.get("mean"), label="scaler mean"),
        label="scaler mean",
    )
    stds = _normalise_numeric_pair_map(
        _require_mapping(scaler_block.get("std"), label="scaler std"),
        label="scaler std",
    )
    climate_block = _require_mapping(
        preprocessing.get("climatology"), label="climatology"
    )
    coefficient_raw = _require_mapping(
        climate_block.get("coefficients"), label="climatology coefficients"
    )
    coefficients = {
        str(site): np.asarray(values, dtype=float)
        for site, values in coefficient_raw.items()
    }
    if any(array.ndim != 1 or not np.isfinite(array).all()
           for array in coefficients.values()):
        raise FrozenInferenceError("climatology coefficients are malformed")
    anchor_block = _require_mapping(
        preprocessing.get("damped_anchor"), label="damped anchor"
    )
    raw_phi = _require_mapping(anchor_block.get("phi"), label="damped phi")
    phi = {str(site): _float(value, label=f"damped phi {site}")
           for site, value in raw_phi.items()}
    if any(not 0.0 <= value <= 0.999 for value in phi.values()):
        raise FrozenInferenceError("damped phi falls outside [0, 0.999]")

    training_stations = _ordered_training_stations(metadata)
    blocks = {
        "imputer": imputer,
        "scaler": scaler_block,
        "climatology": climate_block,
        "damped anchor": anchor_block,
    }
    for label, block in blocks.items():
        if type(block.get("pooled")) is not bool:
            raise FrozenInferenceError(f"bundle {label} lacks an exact pooled flag")
        if tuple(str(site) for site in block.get("fit_stations", ())) != training_stations:
            raise FrozenInferenceError(
                f"bundle {label} fit-station registry differs from training stations"
            )
    imputer_pooled = bool(imputer["pooled"])
    scaler_pooled = bool(scaler_block["pooled"])
    climate_pooled = bool(climate_block["pooled"])
    anchor_pooled = bool(anchor_block["pooled"])
    _require_method(
        imputer,
        D.POOLED_ROW_IMPUTER_METHOD if imputer_pooled
        else D.PER_STATION_IMPUTER_METHOD,
        label="imputer",
    )
    expected_imputer_weighting = ROW_EQUAL_WEIGHTING if imputer_pooled else None
    if imputer.get("pool_weighting") != expected_imputer_weighting:
        raise FrozenInferenceError("bundle imputer pool weighting is not explicit")
    _require_method(
        scaler_block,
        D.POOLED_SCALER_METHOD if scaler_pooled else D.PER_STATION_SCALER_METHOD,
        label="scaler",
    )
    if scaler_block.get("pool_weighting") != (
        STATION_EQUAL_WEIGHTING if scaler_pooled else None
    ):
        raise FrozenInferenceError("bundle scaler pool weighting differs from frozen code")
    if scaler_block.get("variance") != (
        D.POOLED_SCALER_VARIANCE if scaler_pooled
        else "within_station_sample_variance_ddof_1"
    ):
        raise FrozenInferenceError("bundle scaler variance definition differs from frozen code")
    _require_method(
        climate_block,
        F.POOLED_HARMONIC_METHOD if climate_pooled
        else F.PER_STATION_HARMONIC_METHOD,
        label="climatology",
    )
    if climate_block.get("pool_weighting") != (
        STATION_EQUAL_WEIGHTING if climate_pooled else None
    ):
        raise FrozenInferenceError(
            "bundle climatology pool weighting differs from frozen code"
        )
    _require_method(
        anchor_block,
        F.POOLED_DAMPED_AR_METHOD if anchor_pooled else F.DAMPED_AR_METHOD,
        label="damped anchor",
    )
    if anchor_block.get("pair_rule") != F.DAMPED_PAIR_RULE:
        raise FrozenInferenceError("bundle damped pair rule differs from frozen code")
    if anchor_block.get("pool_weighting") != STATION_EQUAL_WEIGHTING:
        raise FrozenInferenceError("bundle damped pool weighting differs from frozen code")
    min_pairs = anchor_block.get("min_pairs")
    if type(min_pairs) is not int or min_pairs < 1:
        raise FrozenInferenceError("bundle damped min_pairs is invalid")
    raw_bounds = anchor_block.get("coefficient_bounds")
    if not isinstance(raw_bounds, list) or len(raw_bounds) != 2:
        raise FrozenInferenceError("bundle damped coefficient bounds are malformed")
    lower_bound = _float(raw_bounds[0], label="damped lower bound")
    upper_bound = _float(raw_bounds[1], label="damped upper bound")
    fallback = _float(anchor_block.get("fallback"), label="damped fallback")
    min_mean_square = _float(
        anchor_block.get("minimum_lagged_anomaly_mean_square"),
        label="damped minimum lagged anomaly mean square",
    )
    if not lower_bound <= fallback <= upper_bound or lower_bound >= upper_bound:
        raise FrozenInferenceError("bundle damped fallback/bounds are inconsistent")
    if min_mean_square <= 0.0:
        raise FrozenInferenceError("bundle damped minimum mean square is not positive")
    if any(not lower_bound <= value <= upper_bound for value in phi.values()):
        raise FrozenInferenceError("damped phi falls outside its frozen bounds")

    kwargs = _architecture_kwargs(metadata)
    station_agnostic = bool(kwargs.get("station_agnostic", False))
    if external:
        if not station_agnostic:
            raise FrozenInferenceError(
                "external new-site inference requires station_agnostic=True"
            )
        for label, block in (
            ("imputer", imputer),
            ("scaler", scaler_block),
            ("climatology", climate_block),
            ("damped anchor", anchor_block),
        ):
            if block.get("pooled") is not True:
                raise FrozenInferenceError(
                    f"external new-site {label} must declare pooled=true"
                )
        expanded_seasonal: dict[tuple[str, str], Mapping[int, float]] = {}
        expanded_global: dict[tuple[str, str], float] = {}
        expanded_mean: dict[tuple[str, str], float] = {}
        expanded_std: dict[tuple[str, str], float] = {}
        for variable in feature_order:
            pooled_seasonal = _pooled_seasonal(seasonal, variable)
            pooled_global = _pooled_numeric(
                global_medians, variable, label="imputer global median"
            )
            pooled_mean = _pooled_numeric(means, variable, label="scaler mean")
            pooled_std = _pooled_numeric(stds, variable, label="scaler std")
            if pooled_std <= 0:
                raise FrozenInferenceError(f"pooled scaler std is non-positive for {variable}")
            for site in station_ids:
                key = (site, variable)
                expanded_seasonal[key] = pooled_seasonal
                expanded_global[key] = pooled_global
                expanded_mean[key] = pooled_mean
                expanded_std[key] = pooled_std
        if "__pooled__" in coefficients:
            pooled_coef = coefficients["__pooled__"]
        else:
            arrays = [coefficients[site] for site in training_stations
                      if site in coefficients]
            if not arrays:
                raise FrozenInferenceError("external climatology has no pooled coefficient")
            pooled_coef = arrays[0]
            if any(not np.allclose(array, pooled_coef, rtol=0.0, atol=1e-12)
                   for array in arrays[1:]):
                raise FrozenInferenceError("external climatology coefficients are not pooled")
        if "__pooled__" in phi:
            pooled_phi = phi["__pooled__"]
        else:
            pooled_phi = _same_value(
                [phi[site] for site in training_stations if site in phi],
                label="damped phi",
            )
        climate_coef = {site: pooled_coef.copy() for site in station_ids}
        anchor_phi = {site: pooled_phi for site in station_ids}
        fit_stations = training_stations
        pooled = True
    else:
        if any(bool(block["pooled"]) for block in blocks.values()):
            raise FrozenInferenceError(
                "same-station inference requires station-specific preprocessing"
            )
        if set(station_ids) != set(training_stations):
            raise FrozenInferenceError(
                "same-station bundle cannot score a different site registry"
            )
        expanded_seasonal = {}
        expanded_global = {}
        expanded_mean = {}
        expanded_std = {}
        for site in station_ids:
            for variable in feature_order:
                key = (site, variable)
                try:
                    expanded_seasonal[key] = seasonal[key]
                    expanded_global[key] = global_medians[key]
                    expanded_mean[key] = means[key]
                    expanded_std[key] = stds[key]
                except KeyError as exc:
                    raise FrozenInferenceError(
                        f"same-station preprocessing lacks {site}|{variable}"
                    ) from exc
                if expanded_std[key] <= 0:
                    raise FrozenInferenceError(
                        f"scaler std is non-positive for {site}|{variable}"
                    )
        try:
            climate_coef = {site: coefficients[site].copy() for site in station_ids}
            anchor_phi = {site: phi[site] for site in station_ids}
        except KeyError as exc:
            raise FrozenInferenceError(
                f"same-station climate/anchor lacks {exc.args[0]}"
            ) from exc
        fit_stations = training_stations
        pooled = False

    harmonics = int(climate_block.get("harmonics", C.SEASONAL_HARMONICS))
    expected_coefficients = 1 + 2 * harmonics
    if any(len(values) != expected_coefficients for values in climate_coef.values()):
        raise FrozenInferenceError("climatology coefficient length disagrees with harmonics")
    scaler = D.StandardScalerPerStation(
        mean=dict(expanded_mean), std=dict(expanded_std),
        fit_stations=fit_stations, pooled=pooled,
    )
    climatology = F.HarmonicClimatology(
        coef=climate_coef, k=harmonics, fit_stations=fit_stations, pooled=pooled
    )
    damped = F.DampedPersistenceAnchor(
        phi=anchor_phi,
        fit_stations=fit_stations,
        pooled=pooled,
        fallback=fallback,
        min_pairs=min_pairs,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        min_mean_square=min_mean_square,
        pool_weighting=STATION_EQUAL_WEIGHTING,
    )
    return FrozenTransforms(
        feature_order=feature_order,
        imputer_seasonal=expanded_seasonal,
        imputer_global=expanded_global,
        scaler=scaler,
        climatology=climatology,
        damped_anchor=damped,
        station_ids=station_ids,
        station_agnostic=station_agnostic,
    )


def build_frozen_confirmation_windows(
    panel: pd.DataFrame,
    metadata: Mapping[str, Any],
    station_ids: Sequence[str],
    *,
    interval: tuple[str, str],
    external: bool,
) -> tuple[DS.WindowedData, FrozenTransforms, pd.DataFrame]:
    """Build outcome-bearing windows without estimating a single parameter."""
    transforms = reconstruct_frozen_transforms(
        metadata, station_ids, external=external
    )
    raw = panel.copy()
    raw["site_id"] = raw["site_id"].astype(str)
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    imputed = transforms.impute(raw)

    # ``build_windows`` historically indexes embeddings through this explicit
    # registry.  Same-station inference must preserve the training index order;
    # external inference is station-agnostic and uses a deterministic order.
    if external:
        ordered_sites = tuple(sorted(transforms.station_ids))
    else:
        training_order = _ordered_training_stations(metadata)
        ordered_sites = tuple(site for site in training_order if site in transforms.station_ids)
    C.STATIONS = ordered_sites
    C.UPSTREAM = {site: None for site in ordered_sites}

    # Masks are shape-compatible only; no field is fitted through them because
    # scaler, climatology, imputer and anchor are all explicitly injected.
    empty = np.zeros(len(imputed), dtype=bool)
    masks = D.SplitMasks(train=empty, val=empty, calib=empty, test=empty)
    wd = DS.build_windows(
        imputed,
        masks,
        transforms.climatology,
        variables=transforms.feature_order,
        require_observed_target=True,
        scaler=transforms.scaler,
        damped_anchor=transforms.damped_anchor,
        evaluation_interval=interval,
        evaluation_split="confirm",
        independent_horizon_targets=True,
    )
    return wd, transforms, imputed


def thermoroute_factory_from_metadata(metadata: Mapping[str, Any]) -> ThermoRoute:
    """Instantiate exactly the architecture described by a weights-only bundle."""
    kwargs = _architecture_kwargs(metadata, thermoroute_only=True)
    feature_order = tuple(metadata.get("feature_order", ()))
    horizons = tuple(int(value) for value in metadata.get("horizons", ()))
    if not horizons:
        raise FrozenInferenceError("bundle has no horizons")
    if int(kwargs.get("n_vars", -1)) != len(feature_order):
        raise FrozenInferenceError("architecture n_vars disagrees with feature_order")
    kwargs["horizons"] = horizons
    train_config = metadata["architecture"].get("train_config")
    if train_config is not None:
        if not isinstance(train_config, Mapping):
            raise FrozenInferenceError("architecture train_config is malformed")
        allowed = C.TrainConfig.__dataclass_fields__
        unknown = set(train_config) - set(allowed)
        if unknown:
            raise FrozenInferenceError(
                f"architecture train_config has unknown fields: {sorted(unknown)}"
            )
        kwargs["cfg"] = C.TrainConfig(**dict(train_config))
    try:
        return ThermoRoute(**kwargs)
    except (TypeError, ValueError) as exc:
        raise FrozenInferenceError("cannot reconstruct ThermoRoute architecture") from exc


def lstm_factory_from_metadata(metadata: Mapping[str, Any]) -> LSTMForecaster:
    """Instantiate the exact weights-only global-LSTM architecture in a bundle."""
    architecture = metadata.get("architecture")
    if not isinstance(architecture, Mapping) or (
        architecture.get("class") != "thermoroute.train.LSTMForecaster"
    ):
        raise FrozenInferenceError("bundle is not a reconstructable LSTM model")
    kwargs = _architecture_kwargs(metadata)
    feature_order = tuple(str(value) for value in metadata.get("feature_order", ()))
    horizons = tuple(int(value) for value in metadata.get("horizons", ()))
    if not horizons:
        raise FrozenInferenceError("bundle has no horizons")
    if int(kwargs.get("n_vars", -1)) != len(feature_order):
        raise FrozenInferenceError("LSTM architecture n_vars disagrees with feature_order")
    kwargs["horizons"] = horizons
    try:
        return LSTMForecaster(**kwargs)
    except (TypeError, ValueError) as exc:
        raise FrozenInferenceError("cannot reconstruct LSTM architecture") from exc


def sequence_factory_from_metadata(metadata: Mapping[str, Any]) -> torch.nn.Module:
    """Dispatch only to the two explicitly supported sequence architectures."""
    architecture = metadata.get("architecture")
    class_name = architecture.get("class") if isinstance(architecture, Mapping) else None
    if class_name == "thermoroute.thermoroute.ThermoRoute":
        return thermoroute_factory_from_metadata(metadata)
    if class_name == "thermoroute.train.LSTMForecaster":
        return lstm_factory_from_metadata(metadata)
    raise FrozenInferenceError(f"unsupported sequence architecture: {class_name!r}")
