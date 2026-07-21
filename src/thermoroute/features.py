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
    fit_stations: tuple[str, ...] = ()
    pooled: bool = False

    @classmethod
    def fit(cls, panel: pd.DataFrame, train_mask: np.ndarray,
            target: str = C.TARGET, k: int = C.SEASONAL_HARMONICS,
            fit_stations: tuple[str, ...] | None = None,
            pooled: bool = False) -> "HarmonicClimatology":
        """Fit train-only seasonal coefficients.

        For zero-shot spatial transfer, pass the in-fold ``fit_stations`` and
        ``pooled=True``.  The same pooled coefficient vector is then used for
        both training and held-out stations, preventing held-out WTEMP history
        from entering through a nominally "deterministic" climatology.
        """
        fitted = tuple(C.STATIONS if fit_stations is None else fit_stations)
        allowed = np.asarray(train_mask, dtype=bool) & panel["site_id"].isin(fitted).to_numpy()
        tr = panel.loc[allowed]
        coef: dict[str, np.ndarray] = {}

        def regress(sub: pd.DataFrame) -> np.ndarray | None:
            doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
            X = np.concatenate([np.ones((len(doy), 1)), doy_harmonics(doy, k)], axis=1)
            y = pd.to_numeric(sub[target], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(y)
            if m.sum() < X.shape[1]:
                return None
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            return beta

        pooled_beta = regress(tr)
        if pooled_beta is None:
            raise ValueError("not enough finite train-station data to fit climatology")
        for st in C.STATIONS:
            sub = tr[tr.site_id == st]
            station_beta = None if pooled or st not in fitted else regress(sub)
            coef[st] = pooled_beta.copy() if station_beta is None else station_beta
        return cls(coef=coef, k=k, fit_stations=fitted, pooled=pooled)

    def predict(self, station: str, doy: np.ndarray) -> np.ndarray:
        X = np.concatenate([np.ones((len(doy), 1)), doy_harmonics(doy, self.k)], axis=1)
        return X @ self.coef[station]

    def predict_dates(self, station: str, dates: pd.Series) -> np.ndarray:
        return self.predict(station, pd.to_datetime(dates).dt.dayofyear.to_numpy())


@dataclass(frozen=True)
class DampedPersistenceAnchor:
    """Frozen, train-fit AR(1) anomaly anchor used by the safety contract."""

    phi: dict[str, float]
    fit_stations: tuple[str, ...]
    pooled: bool = False
    fallback: float = 0.9

    @classmethod
    def fit(
        cls,
        panel: pd.DataFrame,
        train_mask: np.ndarray,
        clim: HarmonicClimatology,
        *,
        fit_stations: tuple[str, ...] | None = None,
        pooled: bool = False,
        fallback: float = 0.9,
        min_pairs: int = 30,
    ) -> "DampedPersistenceAnchor":
        """Estimate anomaly lag-1 correlation using observed train pairs only."""
        fitted = tuple(C.STATIONS if fit_stations is None else fit_stations)
        allowed = np.asarray(train_mask, dtype=bool) & panel["site_id"].isin(fitted).to_numpy()
        tr = panel.loc[allowed]

        def pairs(station: str) -> tuple[np.ndarray, np.ndarray]:
            sub = tr[tr.site_id == station].sort_values("DATE")
            if len(sub) < 2:
                return np.empty(0), np.empty(0)
            anomaly = (pd.to_numeric(sub["WTEMP"], errors="coerce").to_numpy(float)
                       - clim.predict_dates(station, sub["DATE"]))
            observed = (sub["WTEMP_observed"].to_numpy(bool)
                        if "WTEMP_observed" in sub else np.isfinite(anomaly))
            dates = pd.to_datetime(sub["DATE"]).to_numpy(dtype="datetime64[D]")
            valid = (observed[1:] & observed[:-1]
                     & np.isfinite(anomaly[1:]) & np.isfinite(anomaly[:-1])
                     & ((dates[1:] - dates[:-1]) == np.timedelta64(1, "D")))
            return anomaly[:-1][valid], anomaly[1:][valid]

        all_x, all_y = [], []
        per_station: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for station in fitted:
            x, y = pairs(station)
            per_station[station] = (x, y)
            all_x.append(x)
            all_y.append(y)
        px = np.concatenate(all_x) if all_x else np.empty(0)
        py = np.concatenate(all_y) if all_y else np.empty(0)

        def estimate(x: np.ndarray, y: np.ndarray, default: float) -> float:
            if len(x) < min_pairs or np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return float(default)
            value = float(np.corrcoef(x, y)[0, 1])
            return float(np.clip(value, 0.0, 0.999)) if np.isfinite(value) else float(default)

        pooled_phi = estimate(px, py, fallback)
        phi: dict[str, float] = {}
        for station in C.STATIONS:
            if pooled or station not in fitted:
                phi[station] = pooled_phi
            else:
                x, y = per_station[station]
                phi[station] = estimate(x, y, pooled_phi)
        return cls(phi=phi, fit_stations=fitted, pooled=pooled, fallback=fallback)

    def predict(
        self,
        station: str,
        horizons: tuple[int, ...],
        wtemp_t: float,
        clim_t: float,
        clim_tgt: np.ndarray,
    ) -> np.ndarray:
        h = np.asarray(horizons, dtype=float)
        return np.asarray(clim_tgt, dtype=float) + self.phi[station] ** h * (wtemp_t - clim_t)


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
            feat["y_observed"] = target_obs.eq(True).to_numpy()
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
    """Tag rows only when both issue and target stay in one partition.

    Rows whose target crosses a boundary receive ``none`` and are excluded by
    training/evaluation callers.  This is a horizon-sized temporal embargo.
    """
    issue = pd.to_datetime(tab["issue_date"]).to_numpy()
    target = pd.to_datetime(tab["target_date"]).to_numpy()
    tag = np.full(len(tab), "none", dtype=object)
    for name, (lo, hi) in split.as_dict().items():
        lower, upper = np.datetime64(lo), np.datetime64(hi)
        m = ((issue >= lower) & (issue <= upper)
             & (target >= lower) & (target <= upper))
        tag[m] = name
    out = tab.copy()
    out["split"] = tag
    return out


def feature_columns(tab: pd.DataFrame) -> list[str]:
    drop = {"site_id", "issue_date", "target_date", "y", "split"}
    return [c for c in tab.columns if c not in drop]
