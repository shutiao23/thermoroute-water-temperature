"""Air2stream-style hybrid water-temperature baseline (a *variant* of Toffolon &
Piccolroaz, 2015 — NOT the official implementation; see the caveat below).

The original *air2stream-lite* in ``baselines.py`` was a one-parameter relaxation
prior. This module implements a fuller discrete-time thermal model in two variants:

* **4-parameter (a1..a4)** — minimal version (no seasonal forcing, no discharge
  modulation of the relaxation time-constant).
* **8-parameter (a1..a8)** — adds a discharge-dependent thermal capacity
  (θ^a4), a sinusoidal seasonal forcing (a5 amplitude, a6 phase), and a
  discharge-modulated daily lower-bound (a7,a8) that limits cold-season drift.

**Caveat (fair-baseline disclosure).** This is a *variant*, not the published
air2stream: the parameter semantics differ from Toffolon & Piccolroaz (2015)
(θ enters as θ^{+a4}; a7/a8 act as a low-temperature clamp not present in the
original), and calibration is a single-start bounded least-squares rather than
the official particle-swarm global search. It is calibrated on observed-target
training days only and forecast under Track-H with the observed air temperature
driving the first step. It is offered as a *physical reference point*, and any
comparison against it is stated as such — not as a claim against the official
air2stream. To claim superiority over the published model one must run the
official code (a documented future-work item).

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

For multi-step forecasts under Track-H, the FIRST step is driven by the air
temperature observed at issue time (Ta_t, available under Track-H); steps t+2…t+h
fall back to climatology, and discharge is held flat (Q_{t+h} ≈ Q_t). This gives
the physical baseline the same observed-at-issue forcing the learned baselines
receive, rather than starving its first step of information.
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


def _residual(params, Ta, Q, T, doy, Qbar, variant, obs=None):
    th = Q / Qbar
    pred = np.empty_like(T)
    pred[0] = T[0]
    for t in range(len(T) - 1):
        pred[t + 1] = _step(T[t], Ta[t], th[t], doy[t], params, variant)
    e = np.clip(pred[1:] - T[1:], -25.0, 25.0)
    # only penalise days whose target WTEMP was actually observed (never fit the
    # loss to imputed labels — matches ThermoRoute's require_observed_target)
    if obs is not None:
        e = e * obs[1:]
    return e


@dataclass
class Air2streamFit:
    params: np.ndarray
    Qbar: float
    variant: str


def fit(Ta: np.ndarray, Q: np.ndarray, T: np.ndarray, doy: np.ndarray,
        variant: str = "a8", obs: np.ndarray = None) -> Air2streamFit:
    """Calibrate a4 (4-param) or a8 (8-param) by bounded least squares.

    ``obs`` (optional bool mask over the training days) restricts the calibration
    LOSS to days whose target WTEMP was genuinely observed; the recursion still
    uses the full consecutive-day state so the ODE stepping stays valid.
    """
    m = (~np.isnan(Ta)) & (~np.isnan(Q)) & (~np.isnan(T))
    obs = (np.ones(len(T), bool) if obs is None else obs.astype(bool))[m]
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
                        args=(Ta, Q, T, doy, Qbar, variant, obs))
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
# Run on a panel — returns predictions in the canonical schema
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
        # observed-target mask (never calibrate on imputed WTEMP)
        wt_obs = (sub["WTEMP_observed"].to_numpy(bool) if "WTEMP_observed" in sub
                  else ~np.isnan(T))
        # train mask aligned to this site's rows
        train_rows = masks.train[(panel.site_id == st).to_numpy()]
        try:
            fit_obj = fit(Ta[train_rows], Q[train_rows], T[train_rows],
                          doy[train_rows], variant=variant, obs=wt_obs[train_rows])
        except Exception:
            continue

        # climatological air temperature for FUTURE (unobserved) days
        Ta_clim = clim_air.predict(st, doy)
        n = len(sub)
        for h in C.HORIZONS:
            yhat = np.full(n, np.nan)
            for t in range(n - h):
                # Track-H forcing: the FIRST step (t->t+1) is driven by the
                # air temperature OBSERVED at issue time (Ta_t, available under
                # Track-H); subsequent steps fall back to climatology. Previously
                # every step used climatology, which starved air2stream of the
                # observed forcing the learned baselines get and pushed its 1-day
                # RMSE to raw persistence.
                ta_step1 = Ta[t] if np.isfinite(Ta[t]) else Ta_clim[t + 1]
                Ta_future = np.concatenate([[ta_step1], Ta_clim[t + 2: t + 1 + h]])
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
