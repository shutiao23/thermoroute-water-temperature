"""Auditable input-stress and out-of-distribution evaluation utilities.

The functions in this module deliberately separate three questions:

* *Input robustness* changes only information available at or before the issue
  time (missing forcings, recent outages, sensor noise, and covariate shifts).
* *Outcome stratification* evaluates the unchanged forecasts/labels on heat,
  low-flow, and high-flow subsets.  It is not presented as a causal experiment.
* *Uncertainty* is reported for the paired RMSE degradation, ``stress-clean``,
  by resampling complete stations or complete HUC2 groups.

The target ``y``, target dates, seasonal climatology, and forecast-key registry
are immutable under every stress.  Random perturbations are stateless functions
of ``(site, observation date, variable)``, so the same historical sensor cell is
changed identically in overlapping windows and results do not depend on batch size.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from .registry import FORECAST_KEY


ROBUSTNESS_SCHEMA_VERSION = "thermoroute.robustness.v1"
CONDITION_COLUMNS = ("condition_id", "scenario", "severity", "severity_unit")
IMMUTABLE_BATCH_FIELDS = ("y", "clim_t", "clim_tgt", "season", "station")


@dataclass(frozen=True)
class PerturbationSpec:
    """One fully resolved, machine-readable input stress.

    Severity semantics are fixed by scenario:

    ``clean``
        No perturbation, severity 0 (unit ``none``).
    ``missing_rate``
        Independent forcing-cell dropout probability in ``[0, 1]``.
    ``missing_block``
        Number of most-recent consecutive issue/history days removed.
    ``sensor_noise``
        Gaussian noise standard deviation in train-standardised units.
    ``air_temperature_shift``
        Additive TEMP shift in train-standard-deviation units.
    ``flow_shift``
        Multiplicative shift to FLOW in its original signed physical units.

    By default missingness affects forcing channels but preserves the mandatory
    issue-time WTEMP safety anchor.  Sensor noise includes WTEMP and synchronises
    the damped anchor to the noisy issue observation; it never changes future y.
    """

    scenario: str
    severity: float
    severity_unit: str
    variables: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        allowed = {
            "clean": "none",
            "missing_rate": "fraction",
            "missing_block": "days",
            "sensor_noise": "train_sd",
            "air_temperature_shift": "train_sd",
            "flow_shift": "multiplier",
        }
        if self.scenario not in allowed:
            raise ValueError(f"unknown robustness scenario: {self.scenario!r}")
        if self.severity_unit != allowed[self.scenario]:
            raise ValueError(
                f"{self.scenario} severity unit must be {allowed[self.scenario]!r}")
        if not np.isfinite(self.severity):
            raise ValueError("severity must be finite")
        if self.scenario == "clean" and self.severity != 0:
            raise ValueError("clean severity must be zero")
        if self.scenario == "missing_rate" and not 0 <= self.severity <= 1:
            raise ValueError("missing_rate severity must lie in [0, 1]")
        if self.scenario == "missing_block" and (
                self.severity < 1 or not float(self.severity).is_integer()):
            raise ValueError("missing_block severity must be a positive integer")
        if self.scenario == "sensor_noise" and self.severity < 0:
            raise ValueError("sensor_noise severity must be non-negative")
        if self.scenario == "flow_shift" and self.severity <= 0:
            raise ValueError("flow_shift multiplier must be positive")

    @property
    def condition_id(self) -> str:
        payload = {
            "scenario": self.scenario,
            "severity": float(self.severity),
            "severity_unit": self.severity_unit,
            "variables": list(self.variables) if self.variables is not None else None,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"{self.scenario}:{self.severity:g}:{digest}"


def route_a_perturbation_ladder() -> tuple[PerturbationSpec, ...]:
    """Predeclared Route-A severities; no result-dependent scenario selection."""
    return (
        PerturbationSpec("clean", 0.0, "none"),
        *(PerturbationSpec("missing_rate", p, "fraction")
          for p in (0.10, 0.30, 0.50)),
        *(PerturbationSpec("missing_block", days, "days")
          for days in (3, 7, 14)),
        *(PerturbationSpec("sensor_noise", sd, "train_sd")
          for sd in (0.10, 0.25, 0.50)),
        *(PerturbationSpec("air_temperature_shift", sd, "train_sd")
          for sd in (-2.0, -1.0, 1.0, 2.0)),
        *(PerturbationSpec("flow_shift", factor, "multiplier")
          for factor in (0.50, 0.75, 1.25, 2.0)),
    )


def _default_variables(spec: PerturbationSpec, var_names: Sequence[str]) -> tuple[str, ...]:
    if spec.variables is not None:
        selected = tuple(spec.variables)
    elif spec.scenario in {"missing_rate", "missing_block"}:
        # WTEMP_t is the declared safety anchor, not an optional forcing sensor.
        selected = tuple(v for v in var_names if v != "WTEMP")
    elif spec.scenario == "sensor_noise":
        selected = tuple(var_names)
    elif spec.scenario == "air_temperature_shift":
        selected = ("TEMP",)
    elif spec.scenario == "flow_shift":
        selected = ("FLOW",)
    else:
        selected = ()
    missing = sorted(set(selected) - set(var_names))
    if missing:
        raise ValueError(f"perturbation variables absent from feature schema: {missing}")
    if len(selected) != len(set(selected)):
        raise ValueError("perturbation variables must be unique")
    return selected


def _seed64(spec: PerturbationSpec, base_seed: int, salt: str) -> np.uint64:
    value = f"{ROBUSTNESS_SCHEMA_VERSION}|{base_seed}|{spec.condition_id}|{salt}"
    return np.frombuffer(hashlib.sha256(value.encode()).digest()[:8], dtype="<u8")[0]


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64, used only as a deterministic counter RNG."""
    x = np.asarray(x, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x += np.uint64(0x9E3779B97F4A7C15)
        z = x.copy()
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _uniform_cells(wd, global_indices: np.ndarray, length: int,
                   variable_positions: Sequence[int], seed: np.uint64) -> np.ndarray:
    """U(0,1) keyed by (station, calendar observation day, feature variable)."""
    index = np.asarray(global_indices, dtype=int)
    station = np.asarray(wd.station[index], dtype=np.uint64)[:, None, None]
    issue_day = np.asarray(wd.issue_date[index], dtype="datetime64[D]").astype(
        np.int64)[:, None, None]
    relative_day = np.arange(-length + 1, 1, dtype=np.int64)[None, :, None]
    observation_day = (issue_day + relative_day).astype(np.uint64)
    var = np.asarray(variable_positions, dtype=np.uint64)[None, None, :]
    with np.errstate(over="ignore"):
        counter = (station * np.uint64(0xD2B74407B1CE6E93)
                   ^ observation_day * np.uint64(0xCA5A826395121157)
                   ^ var * np.uint64(0x9E3779B185EBCA87)
                   ^ seed)
    bits = _splitmix64(counter)
    # Open interval avoids log(0) in Box-Muller.
    return ((bits >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def _normal_cells(wd, global_indices: np.ndarray, length: int,
                  variable_positions: Sequence[int],
                  seed1: np.uint64, seed2: np.uint64) -> np.ndarray:
    u1 = _uniform_cells(wd, global_indices, length, variable_positions, seed1)
    u2 = _uniform_cells(wd, global_indices, length, variable_positions, seed2)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _station_scaler_vectors(wd, global_indices: np.ndarray, variable: str,
                            device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stations = [C.STATIONS[int(wd.station[i])] for i in global_indices]
    mean = torch.as_tensor(
        [wd.scaler.mean[(station, variable)] for station in stations],
        dtype=torch.float32, device=device)
    std = torch.as_tensor(
        [wd.scaler.std[(station, variable)] for station in stations],
        dtype=torch.float32, device=device)
    return mean, std


def _synchronise_issue_paths(batch: dict[str, torch.Tensor], wd,
                             global_indices: np.ndarray,
                             touched: set[str]) -> None:
    """Keep every issue-time side input consistent with the perturbed X tail."""
    names = tuple(wd.var_names)
    tail = batch["X"][:, -1, :]
    device = tail.device

    if "TEMP" in touched and "TEMP" in names:
        value = tail[:, names.index("TEMP")]
        batch["gate"][:, 2] = value
    if "FLOW" in touched and "FLOW" in names:
        value = tail[:, names.index("FLOW")]
        batch["logflowz"] = value
        batch["gate"][:, 3] = value
    if "PRCP" in touched and "PRCP" in names:
        batch["gate"][:, 4] = tail[:, names.index("PRCP")]
    if "WLEVEL" in touched and "WLEVEL" in names:
        batch["wlevelz"] = tail[:, names.index("WLEVEL")]
    if "WTEMP" in touched and "WTEMP" in names:
        wt_index = names.index("WTEMP")
        mean, std = _station_scaler_vectors(wd, global_indices, "WTEMP", device)
        batch["wtemp_t"] = tail[:, wt_index] * std + mean
        if batch["X"].shape[1] >= 2:
            batch["gate"][:, 5] = (
                batch["X"][:, -1, wt_index] - batch["X"][:, -2, wt_index])
        stations = [C.STATIONS[int(wd.station[i])] for i in global_indices]
        phi = torch.as_tensor(
            [wd.damped_anchor.phi[station] for station in stations],
            dtype=torch.float32, device=device)
        horizons = torch.as_tensor(wd.horizons, dtype=torch.float32, device=device)
        batch["damped_prior"] = (batch["clim_tgt"]
                                  + phi[:, None] ** horizons[None, :]
                                  * (batch["wtemp_t"] - batch["clim_t"])[:, None])

    # The physics array declares its own exact variable order.
    for position, variable in enumerate(wd.phys_vars):
        if variable in touched and variable in names:
            batch["phys_std"][:, position] = tail[:, names.index(variable)]


def perturb_batch(batch: Mapping[str, torch.Tensor], wd,
                  global_indices: np.ndarray, spec: PerturbationSpec,
                  *, base_seed: int = 0) -> dict[str, torch.Tensor]:
    """Return a perturbed batch without mutating ``batch`` or ``WindowedData``.

    ``global_indices`` locate the stable site and issue date in ``wd``.  Random
    draws are keyed to each underlying calendar sensor cell, making overlapping
    windows coherent and inference invariant to batch size/order.  An equality
    assertion closes the function if y or another immutable field changes.
    """
    index = np.asarray(global_indices, dtype=int)
    if len(index) != len(batch["X"]):
        raise ValueError("global_indices length must match batch size")
    out = {name: value.clone() for name, value in batch.items()}
    immutable_before = {name: out[name].clone() for name in IMMUTABLE_BATCH_FIELDS
                        if name in out}
    if spec.scenario == "clean":
        return out

    names = tuple(wd.var_names)
    variables = _default_variables(spec, names)
    positions = [names.index(variable) for variable in variables]
    if not positions:
        raise ValueError(f"{spec.scenario} selected no input variables")
    length = int(out["X"].shape[1])
    if spec.scenario == "missing_block" and int(spec.severity) > length:
        raise ValueError("missing block exceeds the available context length")

    if spec.scenario == "missing_rate":
        random = _uniform_cells(wd, index, length, positions,
                                _seed64(spec, base_seed, "missing"))
        remove = torch.as_tensor(random < spec.severity, device=out["X"].device)
        for local, position in enumerate(positions):
            drop = remove[:, :, local]
            out["X"][:, :, position] = torch.where(
                drop, torch.zeros_like(out["X"][:, :, position]),
                out["X"][:, :, position])
            out["Mask"][:, :, position] = torch.where(
                drop, torch.zeros_like(out["Mask"][:, :, position]),
                out["Mask"][:, :, position])
    elif spec.scenario == "missing_block":
        start = length - int(spec.severity)
        out["X"][:, start:, positions] = 0.0
        out["Mask"][:, start:, positions] = 0.0
    elif spec.scenario == "sensor_noise":
        noise = _normal_cells(
            wd, index, length, positions,
            _seed64(spec, base_seed, "noise-u1"),
            _seed64(spec, base_seed, "noise-u2"),
        ) * spec.severity
        noise_t = torch.as_tensor(noise, dtype=out["X"].dtype,
                                  device=out["X"].device)
        for local, position in enumerate(positions):
            observed = out["Mask"][:, :, position] > 0
            out["X"][:, :, position] = torch.where(
                observed, out["X"][:, :, position] + noise_t[:, :, local],
                out["X"][:, :, position])
    elif spec.scenario == "air_temperature_shift":
        position = names.index("TEMP")
        observed = out["Mask"][:, :, position] > 0
        shifted = out["X"][:, :, position] + float(spec.severity)
        out["X"][:, :, position] = torch.where(
            observed, shifted, out["X"][:, :, position])
    elif spec.scenario == "flow_shift":
        position = names.index("FLOW")
        mean, std = _station_scaler_vectors(wd, index, "FLOW", out["X"].device)
        z = out["X"][:, :, position]
        # FLOW uses a signed-log1p transform so reverse-flow observations remain
        # distinct from zero.  Invert it, multiply in physical units, then pass
        # through the exact same frozen transform and scaler.
        transformed = z * std[:, None] + mean[:, None]
        raw = torch.sign(transformed) * torch.expm1(torch.abs(transformed))
        shifted_raw = raw * float(spec.severity)
        shifted_transformed = torch.sign(shifted_raw) * torch.log1p(torch.abs(shifted_raw))
        shifted = (shifted_transformed - mean[:, None]) / std[:, None]
        observed = out["Mask"][:, :, position] > 0
        out["X"][:, :, position] = torch.where(observed, shifted, z)
    else:  # pragma: no cover - guarded by PerturbationSpec
        raise AssertionError(spec.scenario)

    _synchronise_issue_paths(out, wd, index, set(variables))
    for name, expected in immutable_before.items():
        if not torch.equal(out[name], expected):
            raise AssertionError(f"robustness stress illegally changed immutable field {name}")
    return out


@torch.inference_mode()
def predict_perturbation(model: torch.nn.Module, wd, indices: np.ndarray,
                         spec: PerturbationSpec, *, device: str = "cpu",
                         batch_size: int = 2048, base_seed: int = 0,
                         model_name: str = "ThermoRoute",
                         model_seed: int = 0) -> pd.DataFrame:
    """Chunked inference for one stress condition in the canonical key space."""
    index = np.asarray(indices, dtype=int)
    if len(index) == 0:
        raise ValueError("no windows selected for robustness evaluation")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = model.to(device)
    model.eval()
    median_parts: list[np.ndarray] = []
    for start in range(0, len(index), batch_size):
        selected = index[start:start + batch_size]
        batch = wd.batch(selected, device)
        stressed = perturb_batch(batch, wd, selected, spec, base_seed=base_seed)
        output = model(stressed)
        median_parts.append(output.point.detach().cpu().numpy())
    median = np.concatenate(median_parts, axis=0)
    if median.shape != wd.y[index].shape or not np.isfinite(median).all():
        raise ValueError("model emitted invalid robustness predictions")

    station = np.asarray([C.STATIONS[int(i)] for i in wd.station[index]], dtype=str)
    frames = []
    for hi, horizon in enumerate(wd.horizons):
        frames.append(pd.DataFrame({
            "condition_id": spec.condition_id,
            "scenario": spec.scenario,
            "severity": float(spec.severity),
            "severity_unit": spec.severity_unit,
            "model": model_name,
            "seed": int(model_seed),
            "site_id": station,
            "horizon": int(horizon),
            "issue_date": pd.to_datetime(wd.issue_date[index]),
            "target_date": pd.to_datetime(wd.target_date[index, hi]),
            "y_true": wd.y[index, hi].astype(float),
            "y_pred": median[:, hi].astype(float),
        }))
    result = pd.concat(frames, ignore_index=True)
    expected = (pd.to_datetime(result.issue_date)
                + pd.to_timedelta(result.horizon, unit="D"))
    if not expected.equals(pd.to_datetime(result.target_date)):
        raise AssertionError("robustness prediction target_date violates the horizon")
    return result


@dataclass(frozen=True)
class CommonKeyAudit:
    conditions: tuple[str, ...]
    n_common: int
    rows_per_condition: int


def enforce_common_robustness_keys(predictions: pd.DataFrame) -> CommonKeyAudit:
    """Fail closed unless every condition has identical keys and ground truth."""
    required = set(CONDITION_COLUMNS) | set(FORECAST_KEY) | {
        "model", "seed", "y_true", "y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"robustness predictions missing columns: {missing}")
    frame = predictions.copy()
    frame["issue_date"] = pd.to_datetime(frame.issue_date)
    frame["target_date"] = pd.to_datetime(frame.target_date)
    identity = ["model", "seed", *FORECAST_KEY]
    conditions = tuple(sorted(frame.condition_id.astype(str).unique()))
    if not any(frame.loc[frame.condition_id == condition, "scenario"].eq("clean").all()
               for condition in conditions):
        raise ValueError("robustness registry requires one clean condition")

    reference_keys: pd.MultiIndex | None = None
    reference_truth: pd.Series | None = None
    rows_per_condition: int | None = None
    for condition in conditions:
        part = frame[frame.condition_id.astype(str) == condition].copy()
        metadata_rows = part[["scenario", "severity", "severity_unit"]].drop_duplicates()
        if len(metadata_rows) != 1:
            raise ValueError(f"condition {condition} has inconsistent severity metadata")
        if part.duplicated(identity).any():
            raise ValueError(f"duplicate forecast keys in condition {condition}")
        indexed = part.set_index(identity).sort_index()
        if reference_keys is None:
            reference_keys = indexed.index
            reference_truth = indexed.y_true
            rows_per_condition = len(indexed)
        elif not indexed.index.equals(reference_keys):
            raise ValueError(f"condition {condition} does not share exact forecast keys")
        else:
            assert reference_truth is not None
            if not np.allclose(
                indexed.y_true.to_numpy(float),
                reference_truth.to_numpy(float),
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError(f"condition {condition} disagrees on y_true")
    assert reference_keys is not None and rows_per_condition is not None
    return CommonKeyAudit(conditions, len(reference_keys), rows_per_condition)


def build_outcome_strata(wd, indices: np.ndarray, *,
                         heat_thresholds: Mapping[str, float],
                         low_flow_thresholds: Mapping[str, float],
                         high_flow_thresholds: Mapping[str, float]) -> pd.DataFrame:
    """Return frozen evaluation subsets; thresholds must be fitted on train only.

    ``heat_event`` is an outcome-defined target WTEMP exceedance and is used only
    for conditional scoring.  ``low_flow`` and ``high_flow`` use clean issue-time
    FLOW.  None of these masks is fed into the model or used to tune severities.
    """
    index = np.asarray(indices, dtype=int)
    stations = np.asarray([C.STATIONS[int(i)] for i in wd.station[index]], dtype=str)
    required = set(stations)
    for name, mapping in (
        ("heat", heat_thresholds), ("low-flow", low_flow_thresholds),
        ("high-flow", high_flow_thresholds),
    ):
        absent = sorted(required - set(map(str, mapping.keys())))
        if absent:
            raise ValueError(f"{name} thresholds missing stations: {absent[:5]}")
    if "FLOW" not in wd.var_names:
        raise ValueError("FLOW is required for flow-regime strata")
    flow_position = wd.var_names.index("FLOW")
    z = wd.X[index, -1, flow_position].astype(float)
    flow_observed = wd.Mask[index, -1, flow_position].astype(bool)
    flow = np.empty(len(index), dtype=float)
    for row, station in enumerate(stations):
        value = z[row] * wd.scaler.std[(station, "FLOW")] + wd.scaler.mean[(station, "FLOW")]
        flow[row] = float(D.inverse_stabilising_transform("FLOW", np.array([value]))[0])
    low = flow_observed & (
        flow <= np.asarray([low_flow_thresholds[s] for s in stations], dtype=float)
    )
    high = flow_observed & (
        flow >= np.asarray([high_flow_thresholds[s] for s in stations], dtype=float)
    )

    frames: list[pd.DataFrame] = []
    for hi, horizon in enumerate(wd.horizons):
        base = pd.DataFrame({
            "site_id": stations,
            "horizon": int(horizon),
            "issue_date": pd.to_datetime(wd.issue_date[index]),
            "target_date": pd.to_datetime(wd.target_date[index, hi]),
        })
        heat = wd.y[index, hi] > np.asarray([heat_thresholds[s] for s in stations])
        for label, selected in (
            ("all", np.ones(len(index), dtype=bool)),
            ("heat_event", heat),
            ("low_flow", low),
            ("high_flow", high),
        ):
            part = base.loc[selected].copy()
            part["stratum"] = label
            frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated([*FORECAST_KEY, "stratum"]).any():
        raise AssertionError("outcome stratum registry contains duplicate keys")
    return result


def _bootstrap_station_effects(effects: pd.DataFrame, cluster_col: str, *,
                               n_boot: int, seed: int) -> dict[str, float]:
    if n_boot < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    columns = list(dict.fromkeys(["site_id", "delta_rmse", cluster_col]))
    values = effects[columns].dropna()
    clusters = values[cluster_col].astype(str).unique()
    if len(clusters) < 2:
        return {"ci_low": np.nan, "ci_high": np.nan, "n_clusters": len(clusters)}
    groups = {cluster: values.loc[values[cluster_col].astype(str) == cluster,
                                  "delta_rmse"].to_numpy(float)
              for cluster in clusters}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        draws[draw] = np.median(np.concatenate([groups[cluster] for cluster in sampled]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high),
            "n_clusters": int(len(clusters))}


def summarise_degradation(predictions: pd.DataFrame, strata: pd.DataFrame, *,
                          huc2_by_site: Mapping[str, str], n_boot: int = 2000,
                          seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise paired station RMSE degradation with station/HUC2 CIs.

    Returns ``(summary, station_effects)``.  Positive ``delta_rmse`` means the
    stressed input is worse; negative values are retained rather than relabelled
    as a robustness benefit.  No scenario is dropped based on its result.
    """
    enforce_common_robustness_keys(predictions)
    if predictions["model"].nunique() != 1 or predictions["seed"].nunique() != 1:
        raise ValueError(
            "summarise_degradation expects one resolved model/ensemble; collapse "
            "members on each forecast key before inference")
    pred = predictions.copy()
    for column in ("issue_date", "target_date"):
        pred[column] = pd.to_datetime(pred[column])
    strata = strata.copy()
    for column in ("issue_date", "target_date"):
        strata[column] = pd.to_datetime(strata[column])
    clean_conditions = pred.loc[pred.scenario.eq("clean"), "condition_id"].unique()
    if len(clean_conditions) != 1:
        raise ValueError("exactly one clean condition is required")
    clean = pred[pred.condition_id == clean_conditions[0]]
    identity = ["model", "seed", *FORECAST_KEY]
    clean = clean[identity + ["y_true", "y_pred"]].rename(
        columns={"y_true": "y_clean", "y_pred": "pred_clean"})

    effect_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    condition_fields = ["condition_id", "scenario", "severity", "severity_unit"]
    stressed = pred[~pred.scenario.eq("clean")]
    for condition_values, condition in stressed.groupby(condition_fields, sort=False):
        paired = condition.merge(clean, on=identity, validate="one_to_one")
        if len(paired) != len(clean):
            raise AssertionError("condition lost common forecast keys during pairing")
        if not np.allclose(paired.y_true, paired.y_clean, atol=1e-8, rtol=0):
            raise AssertionError("condition pairing changed y_true")
        for stratum, registry in strata.groupby("stratum", sort=False):
            selected = paired.merge(registry[[*FORECAST_KEY]], on=list(FORECAST_KEY),
                                    validate="one_to_one")
            if selected.empty:
                continue
            for horizon, group in selected.groupby("horizon"):
                losses = group.assign(
                    clean_squared=(group.pred_clean.to_numpy(float)
                                   - group.y_true.to_numpy(float)) ** 2,
                    stressed_squared=(group.y_pred.to_numpy(float)
                                      - group.y_true.to_numpy(float)) ** 2,
                )
                per_station = losses.groupby("site_id", as_index=False).agg(
                    clean_mse=("clean_squared", "mean"),
                    stressed_mse=("stressed_squared", "mean"),
                    n_forecasts=("y_true", "size"),
                )
                per_station["clean_rmse"] = np.sqrt(per_station.pop("clean_mse"))
                per_station["stressed_rmse"] = np.sqrt(
                    per_station.pop("stressed_mse"))
                per_station["delta_rmse"] = (
                    per_station.stressed_rmse - per_station.clean_rmse)
                per_station["relative_delta"] = np.where(
                    per_station.clean_rmse > 0,
                    per_station.delta_rmse / per_station.clean_rmse, np.nan)
                per_station["huc2"] = per_station.site_id.astype(str).map(
                    {str(k): str(v) for k, v in huc2_by_site.items()})
                missing_huc = per_station.huc2.isna() | per_station.huc2.str.lower().eq("nan")
                per_station.loc[missing_huc, "huc2"] = (
                    "UNMAPPED:" + per_station.loc[missing_huc, "site_id"].astype(str))
                for column, value in zip(condition_fields, condition_values):
                    per_station[column] = value
                per_station["stratum"] = stratum
                per_station["horizon"] = int(horizon)
                effect_frames.append(per_station)

                for cluster_level, cluster_col in (
                    ("station", "site_id"), ("huc2", "huc2")):
                    seed_material = (
                        f"{ROBUSTNESS_SCHEMA_VERSION}|{seed}|{condition_values[0]}|"
                        f"{stratum}|h{horizon}|{cluster_level}"
                    ).encode()
                    boot_seed = int.from_bytes(
                        hashlib.sha256(seed_material).digest()[:4], "little")
                    inference = _bootstrap_station_effects(
                        per_station, cluster_col, n_boot=n_boot, seed=boot_seed)
                    summary_rows.append({
                        **dict(zip(condition_fields, condition_values)),
                        "stratum": stratum,
                        "horizon": int(horizon),
                        "cluster_level": cluster_level,
                        "delta_rmse": float(np.median(per_station.delta_rmse)),
                        "ci_low": inference["ci_low"],
                        "ci_high": inference["ci_high"],
                        "clean_rmse": float(np.median(per_station.clean_rmse)),
                        "stressed_rmse": float(np.median(per_station.stressed_rmse)),
                        "relative_delta": float(np.nanmedian(per_station.relative_delta)),
                        "n_forecasts": int(per_station.n_forecasts.sum()),
                        "n_stations": int(len(per_station)),
                        "n_clusters": inference["n_clusters"],
                    })
    if not summary_rows:
        raise ValueError("no stressed conditions could be summarised")
    return (pd.DataFrame(summary_rows),
            pd.concat(effect_frames, ignore_index=True))
