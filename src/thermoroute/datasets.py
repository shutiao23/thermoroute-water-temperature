"""Windowed tensors for the sequence models, with a built-in leakage guard.

A sample for station *s* issued on day *t* carries a ``CONTEXT_LENGTH`` history
ending at *t* (features), the raw water temperature and forcings at *t* (physics
anchor), the deterministic climatology at *t* and at every target *t+h*, and the
targets ``WTEMP_{t+h}``.  ``feature_max_time == issue_time == t`` is asserted, so
no future observation can ever enter the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import features as F

PHYS_FORCINGS = ("TEMP", "RHMEAN", "WDSP", "DH")     # drive the equilibrium T^eq


@dataclass(frozen=True)
class FeatureSchema:
    """One auditable declaration of every input path exposed to ThermoRoute.

    Sequence channels, physics forcings, κ modulators and regime-gate channels
    are all derived from ``variables``.  A V1 model can therefore no longer
    receive FLOW/TEMP/PRCP through a hidden side input while claiming to use
    WTEMP only.
    """

    variables: tuple[str, ...]
    physics_forcings: tuple[str, ...]
    gate_channels: tuple[str, ...] = (
        "sin_doy", "cos_doy", "TEMP", "FLOW", "PRCP", "WTEMP_tendency",
    )

    @classmethod
    def from_variables(cls, variables: tuple[str, ...]) -> "FeatureSchema":
        values = tuple(variables)
        unknown = sorted(set(values) - set(C.ALL_VARS))
        if unknown:
            raise ValueError(f"unknown feature variables: {unknown}")
        if len(values) != len(set(values)):
            raise ValueError("feature variables must be unique")
        if "WTEMP" not in values:
            raise ValueError("WTEMP is required as the issue-time forecast anchor")
        return cls(values, tuple(v for v in PHYS_FORCINGS if v in values))

    def includes(self, variable: str) -> bool:
        return variable in self.variables

    def gate_enabled(self, channel: str) -> bool:
        # Calendar and target tendency are valid whenever WTEMP is present;
        # forcing-specific channels require explicit schema membership.
        return channel in {"sin_doy", "cos_doy", "WTEMP_tendency"} or self.includes(channel)


@dataclass
class WindowedData:
    """Numpy arrays for all samples plus split tags; ``batch`` makes tensors."""
    X: np.ndarray            # [N, L, V]  standardised history
    Mask: np.ndarray         # [N, L, V]  observed indicator
    wtemp_t: np.ndarray      # [N]        raw WTEMP_t  (relaxation anchor)
    clim_t: np.ndarray       # [N]        raw climatology at t
    clim_tgt: np.ndarray     # [N, H]     raw climatology at t+h
    phys_std: np.ndarray     # [N, P]     standardised PHYS_FORCINGS at t
    logflowz: np.ndarray     # [N]        z(log1p FLOW_t)
    wlevelz: np.ndarray      # [N]        z(WLEVEL_t)
    season: np.ndarray       # [N, 2]     sin/cos DOY_t
    gate: np.ndarray         # [N, G]     regime-gate features
    station: np.ndarray      # [N]        station index
    y: np.ndarray            # [N, H]     raw targets
    split: np.ndarray        # [N]        split tag
    issue_date: np.ndarray   # [N]        datetime64
    target_date: np.ndarray  # [N, H]     datetime64; each remains in split
    damped_prior: np.ndarray # [N, H]     fixed train-fit safety anchor
    var_names: tuple[str, ...]
    horizons: tuple[int, ...]
    scaler: D.StandardScalerPerStation
    feature_schema: FeatureSchema
    damped_anchor: F.DampedPersistenceAnchor
    phys_vars: tuple[str, ...] = PHYS_FORCINGS

    @property
    def n_phys(self) -> int:
        return self.phys_std.shape[1]

    def idx(self, split: str) -> np.ndarray:
        return np.where(self.split == split)[0]

    def batch(self, index: np.ndarray, device: str = "cpu") -> dict[str, torch.Tensor]:
        t = lambda a, dt=torch.float32: torch.as_tensor(a[index], dtype=dt, device=device)
        b = {
            "X": t(self.X), "Mask": t(self.Mask),
            "wtemp_t": t(self.wtemp_t), "clim_t": t(self.clim_t),
            "clim_tgt": t(self.clim_tgt), "damped_prior": t(self.damped_prior),
            "phys_std": t(self.phys_std),
            "logflowz": t(self.logflowz), "wlevelz": t(self.wlevelz),
            "season": t(self.season), "gate": t(self.gate),
            "station": t(self.station, torch.long), "y": t(self.y),
        }
        return b


def build_windows(panel: pd.DataFrame, masks: D.SplitMasks,
                  clim: F.HarmonicClimatology,
                  context: int = C.CONTEXT_LENGTH,
                  horizons: tuple[int, ...] = C.HORIZONS,
                  variables: tuple[str, ...] = C.ALL_VARS,
                  require_observed_target: bool = False,
                  feature_schema: FeatureSchema | None = None,
                  scaler_fit_stations: tuple[str, ...] | None = None,
                  pooled_scaler: bool = False,
                  damped_fit_stations: tuple[str, ...] | None = None,
                  pooled_damped: bool = False,
                  damped_anchor: F.DampedPersistenceAnchor | None = None,
                  scaler: D.StandardScalerPerStation | None = None,
                  evaluation_interval: tuple[str, str] | None = None,
                  evaluation_split: str = "confirm") -> WindowedData:
    """Build windowed tensors. With ``require_observed_target`` a sample is kept
    only if the issue-day and every target WTEMP are genuinely observed (used for
    gappy large-sample panels, where history may be imputed but labels must be
    real). Missing WLEVEL (all-NaN channel) is handled by zeroing its z-score."""
    # Guard the global C.STATIONS aliasing hazard: every station↔index decode
    # downstream assumes C.STATIONS holds exactly this panel's stations (order is
    # cascade for 3-station, sorted for USGS — so we check SET membership, not
    # order).
    assert set(panel.site_id.unique()) == set(C.STATIONS), (
        "C.STATIONS is out of sync with the panel passed to build_windows — call "
        "data.prepare_dataset_from_panel(...) first (it sets C.STATIONS). "
        f"C.STATIONS has {len(C.STATIONS)} sites, panel has {panel.site_id.nunique()}.")
    schema = feature_schema or FeatureSchema.from_variables(tuple(variables))
    if feature_schema is not None and tuple(variables) not in (C.ALL_VARS, schema.variables):
        raise ValueError("variables and feature_schema.variables disagree")
    variables = schema.variables
    scaler = scaler or D.StandardScalerPerStation.fit(
        panel, masks.train, variables=variables,
        fit_stations=scaler_fit_stations, pooled=pooled_scaler)
    anchor_fit = damped_anchor or F.DampedPersistenceAnchor.fit(
        panel, masks.train, clim, fit_stations=damped_fit_stations,
        pooled=pooled_damped)
    max_h = max(horizons)
    V = len(variables)
    # Every auxiliary path is derived from the same declared schema.
    phys_vars = schema.physics_forcings
    P = len(phys_vars)
    st_index = {s: i for i, s in enumerate(C.STATIONS)}

    rows = {k: [] for k in
            ("X", "Mask", "wtemp_t", "clim_t", "clim_tgt", "phys_std",
             "logflowz", "wlevelz", "season", "gate", "station", "y",
             "split", "issue_date", "target_date", "damped_prior")}

    for st in C.STATIONS:
        sub = panel[panel.site_id == st].sort_values("DATE").reset_index(drop=True)
        dates = sub["DATE"].to_numpy()
        doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
        raw = {v: sub[v].to_numpy(float) for v in variables}
        obs = {v: sub[f"{v}_observed"].to_numpy(float) for v in variables}
        # standardised matrix [T, V]
        # Defensive: any residual NaN (e.g. a doy with no training samples for a
        # given site/variable and global_median itself NaN) is set to 0 (the
        # standardised mean), so the encoder never sees NaN. The Mask column
        # records whether each cell was originally observed.
        Xstd = np.stack([np.nan_to_num(scaler.transform_value(st, v, raw[v]), nan=0.0)
                         for v in variables], axis=1)
        Mstd = np.stack([obs[v] for v in variables], axis=1)
        clim_series = clim.predict(st, doy)
        logflowz = (np.nan_to_num(scaler.transform_value(st, "FLOW", raw["FLOW"]), nan=0.0)
                    if schema.includes("FLOW") else np.zeros(len(sub), dtype=float))
        wlevelz = (np.nan_to_num(scaler.transform_value(st, "WLEVEL", raw["WLEVEL"]), nan=0.0)
                   if schema.includes("WLEVEL") else np.zeros(len(sub), dtype=float))
        obs_wt = (sub["WTEMP_observed"].to_numpy() if "WTEMP_observed" in sub.columns
                  else np.ones(len(sub), dtype=bool))
        if P > 0:
            phys = np.stack([np.nan_to_num(scaler.transform_value(st, v, raw[v]), nan=0.0)
                             for v in phys_vars], axis=1)
        else:
            phys = np.zeros((len(sub), 0), dtype=float)
        wtemp_std = np.nan_to_num(scaler.transform_value(st, "WTEMP", raw["WTEMP"]), nan=0.0)
        prcpz = (np.nan_to_num(scaler.transform_value(st, "PRCP", raw["PRCP"]), nan=0.0)
                 if schema.includes("PRCP") else np.zeros(len(sub), dtype=float))
        tempz = (np.nan_to_num(scaler.transform_value(st, "TEMP", raw["TEMP"]), nan=0.0)
                 if schema.includes("TEMP") else np.zeros(len(sub), dtype=float))
        dwt = np.concatenate([[0.0], np.diff(wtemp_std)])  # standardised tendency
        sin_d = np.sin(2 * np.pi * doy / C.SEASONAL_PERIOD)
        cos_d = np.cos(2 * np.pi * doy / C.SEASONAL_PERIOD)

        n = len(sub)
        for t in range(context - 1, n - max_h):
            d = dates[t]
            target_dates = np.asarray([dates[t + h] for h in horizons])
            if evaluation_interval is None:
                sp = D.split_for_forecast_interval(d, target_dates)
            else:
                lower, upper = map(np.datetime64, evaluation_interval)
                inside = lower <= d <= upper and np.all(
                    (target_dates >= lower) & (target_dates <= upper)
                )
                sp = evaluation_split if inside else "none"
            if sp == "none":
                continue
            if require_observed_target and (
                    not obs_wt[t] or not all(obs_wt[t + h] for h in horizons)):
                continue          # labels must be real, not imputed
            # availability guard: history strictly up to t
            rows["X"].append(Xstd[t - context + 1: t + 1])
            rows["Mask"].append(Mstd[t - context + 1: t + 1])
            rows["wtemp_t"].append(raw["WTEMP"][t])
            rows["clim_t"].append(clim_series[t])
            clim_target = np.asarray([clim_series[t + h] for h in horizons], dtype=float)
            rows["clim_tgt"].append(clim_target)
            rows["damped_prior"].append(anchor_fit.predict(
                st, horizons, raw["WTEMP"][t], clim_series[t], clim_target))
            rows["phys_std"].append(phys[t])
            rows["logflowz"].append(logflowz[t])
            rows["wlevelz"].append(wlevelz[t])
            rows["season"].append([sin_d[t], cos_d[t]])
            rows["gate"].append([sin_d[t], cos_d[t], tempz[t], logflowz[t], prcpz[t], dwt[t]])
            rows["station"].append(st_index[st])
            rows["y"].append([raw["WTEMP"][t + h] for h in horizons])
            rows["split"].append(sp)
            rows["issue_date"].append(d)
            rows["target_date"].append(target_dates)

    wd = WindowedData(
        X=np.asarray(rows["X"], np.float32),
        Mask=np.asarray(rows["Mask"], np.float32),
        wtemp_t=np.asarray(rows["wtemp_t"], np.float32),
        clim_t=np.asarray(rows["clim_t"], np.float32),
        clim_tgt=np.asarray(rows["clim_tgt"], np.float32),
        phys_std=np.asarray(rows["phys_std"], np.float32),
        logflowz=np.asarray(rows["logflowz"], np.float32),
        wlevelz=np.asarray(rows["wlevelz"], np.float32),
        season=np.asarray(rows["season"], np.float32),
        gate=np.asarray(rows["gate"], np.float32),
        station=np.asarray(rows["station"], np.int64),
        y=np.asarray(rows["y"], np.float32),
        split=np.asarray(rows["split"], object),
        issue_date=np.asarray(rows["issue_date"], "datetime64[ns]"),
        target_date=np.asarray(rows["target_date"], "datetime64[ns]"),
        damped_prior=np.asarray(rows["damped_prior"], np.float32),
        var_names=variables, horizons=horizons, scaler=scaler,
        feature_schema=schema, damped_anchor=anchor_fit, phys_vars=phys_vars,
    )
    _assert_no_leakage(
        wd, panel,
        evaluation_interval=evaluation_interval,
        evaluation_split=evaluation_split,
    )
    return wd


def _assert_no_leakage(
    wd: WindowedData,
    panel: pd.DataFrame,
    *,
    evaluation_interval: tuple[str, str] | None = None,
    evaluation_split: str = "confirm",
) -> None:
    """Spot-check that the last history step equals WTEMP_t (no future bleed)."""
    if len(wd.X) == 0:
        raise RuntimeError("no windows built — check context/horizon vs series length")
    wt_col = wd.var_names.index("WTEMP")
    expected_targets = wd.issue_date[:, None] + np.asarray(wd.horizons)[None, :] * np.timedelta64(1, "D")
    if not np.array_equal(wd.target_date, expected_targets):
        raise AssertionError("stored target dates do not equal issue_date + horizon")
    if evaluation_interval is None:
        for split_name in C.SPLIT.as_dict():
            selected = wd.split == split_name
            if not selected.any():
                continue
            for i in np.where(selected)[0]:
                if D.split_for_forecast_interval(
                        wd.issue_date[i], wd.target_date[i]) != split_name:
                    raise AssertionError("forecast target crosses a split boundary")
    else:
        lower, upper = map(np.datetime64, evaluation_interval)
        if set(wd.split) != {evaluation_split}:
            raise AssertionError("confirmation windows contain a non-confirmation split")
        if np.any(wd.issue_date < lower) or np.any(wd.issue_date > upper):
            raise AssertionError("confirmation issue date falls outside the frozen interval")
        if np.any(wd.target_date < lower) or np.any(wd.target_date > upper):
            raise AssertionError("confirmation target date falls outside the frozen interval")
    last_hist_std = wd.X[:, -1, wt_col]
    # invert the standardisation of WTEMP at the issue day and compare
    for st_i, st in enumerate(C.STATIONS):
        sel = wd.station == st_i
        if not sel.any():
            continue
        mean = wd.scaler.mean[(st, "WTEMP")]
        std = wd.scaler.std[(st, "WTEMP")]
        recon = last_hist_std[sel] * std + mean
        if not np.allclose(recon, wd.wtemp_t[sel], atol=1e-3):
            raise AssertionError("window tail does not match WTEMP_t — leakage risk")
