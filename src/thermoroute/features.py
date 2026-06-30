"""Feature engineering: harmonic climatology, tabular lag features, DOY terms.

All fit-style objects take a ``train_mask`` and learn parameters on the training
fold only.  Day-of-year is a deterministic calendar fact, so seasonal terms
evaluated at the *target* time ``t+h`` are NOT leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Harmonic day-of-year terms
# --------------------------------------------------------------------------- #
def doy_harmonics(doy: np.ndarray, k: int = C.SEASONAL_HARMONICS) -> np.ndarray:
    """Return ``[sin1, cos1, ..., sinK, cosK]`` for an array of day-of-year."""
    phase = 2.0 * np.pi * doy[:, None] * np.arange(1, k + 1)[None, :] / C.SEASONAL_PERIOD
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)


@dataclass
class HarmonicClimatology:
    """Per-station seasonal baseline ``C_s(doy)`` via harmonic least squares.

    Fit on the training fold; used both as a stand-alone baseline and as the
    anomaly-decomposition backbone inside ThermoRoute.
    """
    coef: dict[str, np.ndarray]      # station -> regression coefficients
    k: int = C.SEASONAL_HARMONICS

    @classmethod
    def fit(cls, panel: pd.DataFrame, train_mask: np.ndarray,
            target: str = C.TARGET, k: int = C.SEASONAL_HARMONICS) -> "HarmonicClimatology":
        tr = panel.loc[train_mask]
        coef: dict[str, np.ndarray] = {}
        for st in C.STATIONS:
            sub = tr[tr.site_id == st]
            doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
            X = np.concatenate([np.ones((len(doy), 1)), doy_harmonics(doy, k)], axis=1)
            y = sub[target].to_numpy(dtype=float)
            m = ~np.isnan(y)                       # NaN-safe for gappy large-sample data
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            coef[st] = beta
        return cls(coef=coef, k=k)

    def predict(self, station: str, doy: np.ndarray) -> np.ndarray:
        X = np.concatenate([np.ones((len(doy), 1)), doy_harmonics(doy, self.k)], axis=1)
        return X @ self.coef[station]

    def predict_dates(self, station: str, dates: pd.Series) -> np.ndarray:
        return self.predict(station, pd.to_datetime(dates).dt.dayofyear.to_numpy())


# --------------------------------------------------------------------------- #
# Tabular lag features for tree / linear models
# --------------------------------------------------------------------------- #
def build_tabular(
    panel: pd.DataFrame,
    horizon: int,
    variables: tuple[str, ...],
    clim: HarmonicClimatology,
    drop_feature_nans: bool = True,
    require_observed_target: bool = True,
) -> pd.DataFrame:
    """Build a leakage-safe tabular design for one horizon.

    For issue day ``t`` the target is ``WTEMP_{t+h}``.  Features use only
    information available at ``t`` (lags / rolling stats of the chosen variables)
    plus the deterministic seasonal expectation at the target time ``t+h``.
    Returns one row per (station, issue_date) with a ``split`` tag attached later.
    """
    out_frames = []
    for st in C.STATIONS:
        sub = panel[panel.site_id == st].sort_values("DATE").reset_index(drop=True)
        target_date = sub["DATE"] + pd.to_timedelta(horizon, unit="D")
        cols: dict[str, np.ndarray] = {}

        for var in variables:
            s = sub[var].astype(float)
            for lag in C.SHORT_LAGS:
                cols[f"{var}_lag{lag}"] = s.shift(lag).to_numpy()
            for w in C.ROLLING_WINDOWS:
                cols[f"{var}_rollmean{w}"] = s.rolling(w).mean().to_numpy()
                if var == C.TARGET:
                    cols[f"{var}_rollstd{w}"] = s.rolling(w).std().to_numpy()
            cols[f"{var}_delta1"] = (s - s.shift(1)).to_numpy()
            cols[f"{var}_delta3"] = (s - s.shift(3)).to_numpy()

        # deterministic seasonal context (no leakage: calendar only)
        doy_t = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
        doy_th = pd.to_datetime(target_date).dt.dayofyear.to_numpy()
        H = doy_harmonics(doy_t)
        for j in range(H.shape[1]):
            cols[f"doy_h{j}"] = H[:, j]
        cols["clim_t"] = clim.predict(st, doy_t)
        cols["clim_target"] = clim.predict(st, doy_th)
        cols["clim_anom"] = sub[C.TARGET].to_numpy() - cols["clim_t"]
        # persistence reference exposed as a feature (helps trees beat it cleanly)
        cols["persistence"] = sub[C.TARGET].astype(float).to_numpy()

        meta = pd.DataFrame({"site_id": st, "issue_date": sub["DATE"].to_numpy(),
                             "target_date": target_date.to_numpy()})
        feat = pd.concat([meta, pd.DataFrame(cols)], axis=1)
        feat["y"] = sub[C.TARGET].shift(-horizon).astype(float).to_numpy()
        # Capture whether the target was REALLY observed (or imputed) so
        # require_observed_target can drop imputed-target rows correctly even
        # when the input panel has been imputed.
        if f"{C.TARGET}_observed" in sub.columns:
            target_obs = sub[f"{C.TARGET}_observed"].astype(bool).shift(-horizon)
            feat["y_observed"] = target_obs.fillna(False).to_numpy()
        else:
            feat["y_observed"] = feat["y"].notna().to_numpy()
        out_frames.append(feat)

    tab = pd.concat(out_frames, ignore_index=True)
    # Target must be REALLY observed (never trained nor evaluated on imputed
    # labels). When an imputed panel is passed, the y column is non-NaN by
    # construction; the y_observed boolean preserves the truth.
    if require_observed_target:
        tab = tab[tab["y_observed"].astype(bool)]
    tab = tab.drop(columns=["y_observed"])
    # Feature NaNs: by default drop rows that still have any (mirrors the legacy
    # behaviour used by 3-station baselines, which fed clean lag features into
    # LightGBM). For sample-consistency with ThermoRoute windowed inputs (which
    # use imputed features behind a mask), pass ``drop_feature_nans=False``;
    # caller must then impute or zero-fill before fitting tree models.
    if drop_feature_nans:
        tab = tab.dropna()
    return tab.reset_index(drop=True)


def attach_split(tab: pd.DataFrame, split: C.TimeSplit = C.SPLIT) -> pd.DataFrame:
    """Tag each row with its split partition using the *issue_date*."""
    d = pd.to_datetime(tab["issue_date"]).to_numpy()
    tag = np.full(len(tab), "none", dtype=object)
    for name, (lo, hi) in split.as_dict().items():
        m = (d >= np.datetime64(lo)) & (d <= np.datetime64(hi))
        tag[m] = name
    out = tab.copy()
    out["split"] = tag
    return out


def feature_columns(tab: pd.DataFrame) -> list[str]:
    drop = {"site_id", "issue_date", "target_date", "y", "split"}
    return [c for c in tab.columns if c not in drop]
