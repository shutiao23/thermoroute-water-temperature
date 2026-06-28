"""Air2stream — canonical Toffolon & Piccolroaz (2015) hybrid water-temperature model.

The original *air2stream-lite* in ``baselines.py`` was a one-parameter relaxation
prior. This module implements the full canonical model in two standard variants:

* **4-parameter (a1..a4)** — minimal version (no seasonal forcing, no discharge
  modulation of the relaxation time-constant); a fair physical baseline when
  discharge is absent or noisy.
* **8-parameter (a1..a8)** — the complete formulation with a discharge-dependent
  thermal capacity (1/(θ^a4)), a sinusoidal seasonal forcing (a5 amplitude, a6
  phase), and a discharge-modulated daily lower-bound (a7,a8) that prevents
  unphysical drift in cold seasons. This is the *standard hydrology baseline* in
  the stream-temperature literature.

Daily discrete-time form (e.g. Piccolroaz et al. 2016, eq. 5; cf. Toffolon &
Piccolroaz 2015):

    θ_t = Q_t / Q̄                                            (normalised discharge)
    1/τ_t = θ_t ^ a4                                          (thermal "speed")
    T_{t+1} = T_t + (1/τ_t) [ a1 + a2·Ta_t − a3·T_t
                              + a5·cos( 2π·(t/365 − a6) ) ]   (8-param)
              − a7·(T_t − a8·θ_t)·1{T_t < some_threshold}      (low-T correction)

The 4-parameter version sets a5=a6=a7=a8=0. Calibration is per-station, on the
training-fold day-of-year–air-temperature–discharge–water-temperature record,
minimising 1-step-ahead squared error with bounded least-squares.

For multi-step forecasts under the Track-H (no future observed weather)
protocol, we roll the recursion forward using the *climatological* air
temperature and a flat discharge persistence (Q_{t+h} ≈ Q_t), which is what an
operationally-fair air2stream comparison requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C
from . import features as F
from . import results as R


# --------------------------------------------------------------------------- #
# Core recursion
# --------------------------------------------------------------------------- #
def _step(T: float, Ta: float, theta: float, doy: int, params: np.ndarray,
          variant: str) -> float:
    a1, a2, a3, a4 = params[:4]
    # discharge-dependent inverse thermal capacity, clipped against 0/inf
    th = max(theta, 1e-3)
    inv_tau = th ** a4
    drive = a1 + a2 * Ta - a3 * T
    if variant == "a4":
        return T + inv_tau * drive
    # 8-parameter: + seasonal + low-T correction
    a5, a6, a7, a8 = params[4:8]
    seasonal = a5 * np.cos(2.0 * np.pi * (doy / 365.2425 - a6))
    T_next = T + inv_tau * (drive + seasonal)
    if T_next < a8 * th:
        T_next = T_next - a7 * (T_next - a8 * th)
    return T_next


def _residual(params, Ta, Q, T, doy, Qbar, variant):
    th = Q / Qbar
    pred = np.empty_like(T)
    pred[0] = T[0]
    for t in range(len(T) - 1):
        pred[t + 1] = _step(T[t], Ta[t], th[t], doy[t], params, variant)
    # mask away the implausibly far excursions to keep the optimiser stable
    e = pred[1:] - T[1:]
    return np.clip(e, -25.0, 25.0)


@dataclass
class Air2streamFit:
    params: np.ndarray
    Qbar: float
    variant: str


def fit(Ta: np.ndarray, Q: np.ndarray, T: np.ndarray, doy: np.ndarray,
        variant: str = "a8") -> Air2streamFit:
    """Calibrate a4 (4-param) or a8 (8-param) by bounded least squares."""
    m = (~np.isnan(Ta)) & (~np.isnan(Q)) & (~np.isnan(T))
    Ta, Q, T, doy = Ta[m], Q[m], T[m], doy[m]
    Qbar = float(np.nanmean(Q)) if np.nanmean(Q) > 0 else 1.0
    if variant == "a4":
        p0 = np.array([2.0, 0.3, 0.3, 0.5])
        lb = np.array([-50.0, 0.0, 0.0, 0.0])
        ub = np.array([+50.0, 2.0, 2.0, 3.0])
    else:
        p0 = np.array([2.0, 0.3, 0.3, 0.5, 1.0, 0.0, 0.05, 0.0])
        lb = np.array([-50.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -10.0])
        ub = np.array([+50.0, 2.0, 2.0, 3.0, 30.0, 1.0, 5.0, 30.0])
    sol = least_squares(_residual, p0, bounds=(lb, ub), max_nfev=6000,
                        args=(Ta, Q, T, doy, Qbar, variant))
    return Air2streamFit(params=sol.x, Qbar=Qbar, variant=variant)


def forecast_horizon(fit_obj: Air2streamFit, T0: float, Q0: float, doy0: int,
                     Ta_future: Sequence[float], doy_future: Sequence[int]) -> float:
    """Roll one step `len(Ta_future)` times to produce a horizon-h forecast.

    ``Ta_future`` and ``doy_future`` are length-h sequences; we keep discharge
    flat at Q0 (a persistence proxy under Track-H, since the model has no future
    discharge information either).
    """
    T = T0
    theta = max(Q0 / fit_obj.Qbar, 1e-3)
    for ta, doy in zip(Ta_future, doy_future):
        T = _step(T, ta, theta, int(doy), fit_obj.params, fit_obj.variant)
    return float(T)


# --------------------------------------------------------------------------- #
# Run on a panel — returns canonical predictions
# --------------------------------------------------------------------------- #
def run_air2stream(panel: pd.DataFrame, masks, clim_air: F.HarmonicClimatology,
                   stations: tuple[str, ...] = None,
                   variant: str = "a8") -> pd.DataFrame:
    """Calibrate per-station on train, then forecast 1/3/7 days using
    climatological air temperature and persisted discharge (Track-H)."""
    if stations is None:
        stations = C.STATIONS
    out_frames = []
    for st in stations:
        sub = panel[panel.site_id == st].sort_values("DATE").reset_index(drop=True)
        Ta = sub["TEMP"].to_numpy(float)
        Q = sub["FLOW"].to_numpy(float)
        T = sub["WTEMP"].to_numpy(float)
        doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
        # train mask aligned to this site's rows
        train_rows = masks.train[(panel.site_id == st).to_numpy()]
        try:
            fit_obj = fit(Ta[train_rows], Q[train_rows], T[train_rows],
                          doy[train_rows], variant=variant)
        except Exception:
            continue

        # climatological air temperature (used to roll forward)
        Ta_clim = clim_air.predict(st, doy)
        n = len(sub)
        for h in C.HORIZONS:
            yhat = np.full(n, np.nan)
            for t in range(n - h):
                Ta_future = Ta_clim[t + 1: t + 1 + h]
                doy_future = doy[t + 1: t + 1 + h]
                yhat[t] = forecast_horizon(fit_obj, T[t], Q[t], int(doy[t]),
                                           Ta_future, doy_future)
            valid = ~np.isnan(yhat[: n - h]) & ~np.isnan(T[h:])
            issue_dates = sub["DATE"].to_numpy()[: n - h][valid]
            target_dates = sub["DATE"].to_numpy()[h:][valid]
            y_true = T[h:][valid]
            y_pred = yhat[: n - h][valid]
            # mark train/val/calib/test by issue_date
            splits = np.full(len(issue_dates), "none", dtype=object)
            for name, (lo, hi) in C.SPLIT.as_dict().items():
                mask = (issue_dates >= np.datetime64(lo)) & (issue_dates <= np.datetime64(hi))
                splits[mask] = name
            out_frames.append(R.make_pred_frame(
                model=f"Air2stream-{variant}", scope="per_station", feature_set="phys",
                seed=0, site_id=np.full(len(issue_dates), st),
                horizon=np.full(len(issue_dates), h), split=splits,
                issue_date=issue_dates, target_date=target_dates,
                y_true=y_true, y_pred=y_pred,
            ))
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
