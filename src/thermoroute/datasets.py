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
    var_names: tuple[str, ...]
    horizons: tuple[int, ...]
    scaler: D.StandardScalerPerStation
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
            "clim_tgt": t(self.clim_tgt), "phys_std": t(self.phys_std),
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
                  require_observed_target: bool = False) -> WindowedData:
    """Build windowed tensors. With ``require_observed_target`` a sample is kept
    only if the issue-day and every target WTEMP are genuinely observed (used for
    gappy large-sample panels, where history may be imputed but labels must be
    real). Missing WLEVEL (all-NaN channel) is handled by zeroing its z-score."""
    scaler = D.StandardScalerPerStation.fit(panel, masks.train, variables=C.ALL_VARS)
    max_h = max(horizons)
    V = len(variables)
    # physics forcings available in this feature set (V1 ⇒ none ⇒ relax to clim)
    phys_vars = tuple(v for v in PHYS_FORCINGS if v in variables)
    P = len(phys_vars)
    st_index = {s: i for i, s in enumerate(C.STATIONS)}

    # split lookup by date
    split_of = {}
    for name, (lo, hi) in C.SPLIT.as_dict().items():
        split_of[name] = (np.datetime64(lo), np.datetime64(hi))

    def which_split(d: np.datetime64) -> str:
        for name, (lo, hi) in split_of.items():
            if lo <= d <= hi:
                return name
        return "none"

    rows = {k: [] for k in
            ("X", "Mask", "wtemp_t", "clim_t", "clim_tgt", "phys_std",
             "logflowz", "wlevelz", "season", "gate", "station", "y",
             "split", "issue_date")}

    for st in C.STATIONS:
        sub = panel[panel.site_id == st].sort_values("DATE").reset_index(drop=True)
        dates = sub["DATE"].to_numpy()
        doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
        raw = {v: sub[v].to_numpy(float) for v in C.ALL_VARS}
        obs = {v: sub[f"{v}_observed"].to_numpy(float) for v in C.ALL_VARS}
        # standardised matrix [T, V]
        Xstd = np.stack([scaler.transform_value(st, v, raw[v]) for v in variables], axis=1)
        Mstd = np.stack([obs[v] for v in variables], axis=1)
        clim_series = clim.predict(st, doy)
        logflowz = np.nan_to_num(scaler.transform_value(st, "FLOW", raw["FLOW"]), nan=0.0)
        wlevelz = np.nan_to_num(scaler.transform_value(st, "WLEVEL", raw["WLEVEL"]), nan=0.0)
        obs_wt = (sub["WTEMP_observed"].to_numpy() if "WTEMP_observed" in sub.columns
                  else np.ones(len(sub), dtype=bool))
        if P > 0:
            phys = np.stack([scaler.transform_value(st, v, raw[v]) for v in phys_vars], axis=1)
        else:
            phys = np.zeros((len(sub), 0), dtype=float)
        wtemp_std = scaler.transform_value(st, "WTEMP", raw["WTEMP"])
        prcpz = scaler.transform_value(st, "PRCP", raw["PRCP"])
        tempz = scaler.transform_value(st, "TEMP", raw["TEMP"])
        dwt = np.concatenate([[0.0], np.diff(wtemp_std)])  # standardised tendency
        sin_d = np.sin(2 * np.pi * doy / C.SEASONAL_PERIOD)
        cos_d = np.cos(2 * np.pi * doy / C.SEASONAL_PERIOD)

        n = len(sub)
        for t in range(context - 1, n - max_h):
            d = dates[t]
            sp = which_split(d)
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
            rows["clim_tgt"].append([clim_series[t + h] for h in horizons])
            rows["phys_std"].append(phys[t])
            rows["logflowz"].append(logflowz[t])
            rows["wlevelz"].append(wlevelz[t])
            rows["season"].append([sin_d[t], cos_d[t]])
            rows["gate"].append([sin_d[t], cos_d[t], tempz[t], logflowz[t], prcpz[t], dwt[t]])
            rows["station"].append(st_index[st])
            rows["y"].append([raw["WTEMP"][t + h] for h in horizons])
            rows["split"].append(sp)
            rows["issue_date"].append(d)

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
        var_names=variables, horizons=horizons, scaler=scaler, phys_vars=phys_vars,
    )
    _assert_no_leakage(wd, panel)
    return wd


def _assert_no_leakage(wd: WindowedData, panel: pd.DataFrame) -> None:
    """Spot-check that the last history step equals WTEMP_t (no future bleed)."""
    if len(wd.X) == 0:
        raise RuntimeError("no windows built — check context/horizon vs series length")
    wt_col = wd.var_names.index("WTEMP")
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
