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
original), and calibration uses deterministic multi-start bounded least-squares
rather than the official particle-swarm global search. It is calibrated on
observed-target training days only and forecast under Track-H with the observed
air temperature driving the first step. It is offered as a *physical reference
point*, and any comparison against it is stated as such — not as a claim against
the official air2stream. To claim superiority over the published model one must
run the official code (a documented future-work item).

The discharge ratio in this variant is defined only for positive FLOW. Signed
tidal/backwater values are preserved by the main data pipeline, but Air2stream
does not reinterpret them: those issue/transition rows are excluded from this
development-only physical-reference baseline.

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
from . import data as D
from . import features as F
from . import results as R


_A4_DEFAULT_STARTS: tuple[tuple[float, ...], ...] = (
    (2.0, 0.3, 0.3, 0.5),
    (0.0, 0.1, 0.1, 0.1),
    (5.0, 0.5, 0.5, 1.0),
    (-5.0, 0.8, 0.2, 2.0),
    (10.0, 1.0, 0.8, 0.25),
    (-20.0, 1.5, 1.5, 2.5),
)

_A8_DEFAULT_STARTS: tuple[tuple[float, ...], ...] = (
    (2.0, 0.3, 0.3, 0.5, 1.0, 0.0, 0.05, 0.0),
    (0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0),
    (-5.0, 0.5, 0.2, 1.0, 5.0, -0.5, 0.1, 5.0),
    (5.0, 0.8, 0.6, 2.0, 10.0, 0.25, 0.5, 10.0),
    (15.0, 1.2, 1.0, 0.25, 20.0, 0.75, 1.0, 20.0),
    (-20.0, 1.5, 1.5, 2.5, 28.0, -0.75, 3.0, -5.0),
)


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
    next_day = (doy[1:] == doy[:-1] + 1) | (
        np.isin(doy[:-1], (365.0, 366.0)) & (doy[1:] == 1)
    )
    eligible = (
        np.isfinite(T[:-1])
        & np.isfinite(T[1:])
        & np.isfinite(Ta[:-1])
        & np.isfinite(Q[:-1])
        & (Q[:-1] > 0.0)
        & next_day
    )
    if obs is not None:
        # Both the recursive state at t and the target at t+1 must be measured.
        # Requiring only obs[1:] would allow an imputed water temperature to
        # seed a nominally observation-calibrated physical transition.
        eligible &= obs[:-1] & obs[1:]
    indices = np.flatnonzero(eligible)
    predictions = np.asarray([
        _step(
            float(T[index]),
            float(Ta[index]),
            float(Q[index]) / Qbar,
            int(doy[index]),
            params,
            variant,
        )
        for index in indices
    ])
    return np.clip(predictions - T[indices + 1], -25.0, 25.0)


@dataclass(frozen=True)
class Air2streamStartDiagnostic:
    """Auditable outcome for one deterministic optimisation start."""

    initial_params: tuple[float, ...]
    success: bool
    objective: float | None
    status: int | None
    message: str
    nfev: int | None
    fitted_params: tuple[float, ...] | None


class Air2streamOptimizationError(RuntimeError):
    """Raised only when every bounded optimisation start fails."""

    def __init__(self, diagnostics: tuple[Air2streamStartDiagnostic, ...]):
        super().__init__("all deterministic Air2stream optimisation starts failed")
        self.diagnostics = diagnostics


@dataclass
class Air2streamFit:
    params: np.ndarray
    Qbar: float
    variant: str
    diagnostics: tuple[Air2streamStartDiagnostic, ...] = ()
    training_objective: float | None = None
    selected_initial_params: tuple[float, ...] | None = None


def _variant_spec(variant: str) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, ...], ...]]:
    if variant == "a4":
        return (
            np.array([-50.0, 0.0, 0.0, 0.0]),
            np.array([+50.0, 2.0, 2.0, 3.0]),
            _A4_DEFAULT_STARTS,
        )
    if variant == "a8":
        return (
            np.array([-50.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -10.0]),
            np.array([+50.0, 2.0, 2.0, 3.0, 30.0, 1.0, 5.0, 30.0]),
            _A8_DEFAULT_STARTS,
        )
    raise ValueError("variant must be 'a4' or 'a8'")


def _one_dimensional_float(name: str, values: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values")
    return array


def _prepare_starts(
    variant: str,
    starts: Sequence[Sequence[float] | np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, ...], ...]]:
    lb, ub, defaults = _variant_spec(variant)
    supplied = defaults if starts is None else starts
    normalised: set[tuple[float, ...]] = set()
    for index, start in enumerate(supplied):
        try:
            vector = np.asarray(start, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"initial_starts[{index}] must be numeric") from exc
        if vector.shape != lb.shape:
            raise ValueError(
                f"initial_starts[{index}] must contain exactly {len(lb)} parameters"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"initial_starts[{index}] must contain only finite parameters")
        if np.any(vector < lb) or np.any(vector > ub):
            raise ValueError(f"initial_starts[{index}] lies outside the parameter bounds")
        normalised.add(tuple(float(value) for value in vector))
    if not normalised:
        raise ValueError("initial_starts must contain at least one start")
    # Sorting and de-duplicating makes both optimisation order and exact-tie
    # behaviour independent of the caller's start ordering.
    return lb, ub, tuple(sorted(normalised))


def _optional_int(value: object) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def fit(
    Ta: np.ndarray,
    Q: np.ndarray,
    T: np.ndarray,
    doy: np.ndarray,
    variant: str = "a8",
    obs: np.ndarray | None = None,
    *,
    initial_starts: Sequence[Sequence[float] | np.ndarray] | None = None,
    max_nfev: int = 6000,
) -> Air2streamFit:
    """Calibrate the a4 or a8 variant by deterministic multi-start fitting.

    Every start is fitted against the same training-only one-step objective. The
    successful solution with the lowest recomputed sum of squared residuals is
    selected; parameter values and then the initial point break exact ties.
    ``obs`` restricts the loss to genuinely observed target-water-temperature
    days. NaNs are treated as missing rows for backwards compatibility, while
    infinities and malformed inputs fail explicitly.
    """
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)):
        raise ValueError("max_nfev must be a positive integer")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")

    Ta_array = _one_dimensional_float("Ta", Ta)
    Q_array = _one_dimensional_float("Q", Q)
    T_array = _one_dimensional_float("T", T)
    doy_array = _one_dimensional_float("doy", doy)
    lengths = {len(Ta_array), len(Q_array), len(T_array), len(doy_array)}
    if len(lengths) != 1:
        raise ValueError("Ta, Q, T, and doy must have identical lengths")

    if obs is None:
        obs_array = np.ones(len(T_array), dtype=bool)
    else:
        raw_obs = np.asarray(obs)
        if raw_obs.ndim != 1 or len(raw_obs) != len(T_array):
            raise ValueError("obs must be one-dimensional and match the training arrays")
        if np.issubdtype(raw_obs.dtype, np.number) and not np.isfinite(raw_obs).all():
            raise ValueError("obs must not contain non-finite values")
        obs_array = raw_obs.astype(bool)

    if len(T_array) < 2:
        raise ValueError("at least two training rows are required")
    if not np.isfinite(doy_array).all() or not np.equal(
        doy_array, np.floor(doy_array)
    ).all() or not (
        (doy_array >= 1) & (doy_array <= 366)
    ).all():
        raise ValueError("doy values must be integer days in [1, 366]")

    finite_flow = Q_array[np.isfinite(Q_array) & (Q_array > 0.0)]
    Qbar = float(np.mean(finite_flow)) if len(finite_flow) else float("nan")
    if not np.isfinite(Qbar) or Qbar <= 0:
        raise ValueError("positive training discharge is required")
    eligibility_probe = _residual(
        np.zeros(4 if variant == "a4" else 8, dtype=float),
        Ta_array,
        Q_array,
        T_array,
        doy_array,
        Qbar,
        variant,
        obs_array,
    )
    if len(eligibility_probe) == 0:
        raise ValueError(
            "at least one consecutive observed target pair is required"
        )
    lb, ub, starts = _prepare_starts(variant, initial_starts)

    diagnostics: list[Air2streamStartDiagnostic] = []
    candidates: list[tuple[float, tuple[float, ...], tuple[float, ...], np.ndarray]] = []
    residual_args = (
        Ta_array,
        Q_array,
        T_array,
        doy_array,
        Qbar,
        variant,
        obs_array,
    )
    for initial in starts:
        try:
            solution = least_squares(
                _residual,
                np.asarray(initial, dtype=float),
                bounds=(lb, ub),
                max_nfev=int(max_nfev),
                args=residual_args,
            )
            fitted = np.asarray(solution.x, dtype=float)
            valid_params = (
                fitted.shape == lb.shape
                and np.isfinite(fitted).all()
                and np.all(fitted >= lb)
                and np.all(fitted <= ub)
            )
            residual = (
                np.asarray(_residual(fitted, *residual_args), dtype=float)
                if valid_params
                else np.asarray([np.nan])
            )
            objective = float(np.dot(residual, residual))
            success = bool(getattr(solution, "success", False))
            success = success and np.isfinite(residual).all() and np.isfinite(objective)
            fitted_tuple = (
                tuple(float(value) for value in fitted) if valid_params else None
            )
            status = _optional_int(getattr(solution, "status", None))
            nfev = _optional_int(getattr(solution, "nfev", None))
            message = str(getattr(solution, "message", "optimizer returned no message"))
            if not valid_params:
                success = False
                objective = float("nan")
                message = f"invalid fitted parameters: {message}"
            diagnostic = Air2streamStartDiagnostic(
                initial_params=initial,
                success=success,
                objective=objective if np.isfinite(objective) else None,
                status=status,
                message=message,
                nfev=nfev,
                fitted_params=fitted_tuple,
            )
            diagnostics.append(diagnostic)
            if success and fitted_tuple is not None:
                candidates.append((objective, fitted_tuple, initial, fitted.copy()))
        except Exception as exc:
            diagnostics.append(Air2streamStartDiagnostic(
                initial_params=initial,
                success=False,
                objective=None,
                status=None,
                message=f"{type(exc).__name__}: {exc}",
                nfev=None,
                fitted_params=None,
            ))

    diagnostics_tuple = tuple(diagnostics)
    if not candidates:
        raise Air2streamOptimizationError(diagnostics_tuple)
    objective, _, selected_initial, fitted = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return Air2streamFit(
        params=fitted,
        Qbar=Qbar,
        variant=variant,
        diagnostics=diagnostics_tuple,
        training_objective=objective,
        selected_initial_params=selected_initial,
    )


def forecast_horizon(
    fit_obj: Air2streamFit,
    T0: float,
    Q0: float,
    doy0: int,
    Ta_future: Sequence[float] | np.ndarray,
    doy_future: Sequence[int] | np.ndarray,
) -> float:
    """Roll one step `len(Ta_future)` times to produce a horizon-h forecast.

    ``Ta_future`` and ``doy_future`` are length-h sequences; we keep discharge
    flat at Q0 (a persistence proxy under Track-H, since the model has no future
    discharge information either).
    """
    lb, ub, _ = _variant_spec(fit_obj.variant)
    params = np.asarray(fit_obj.params, dtype=float)
    if params.shape != lb.shape or not np.isfinite(params).all():
        raise ValueError("fit_obj.params has the wrong shape or non-finite values")
    if np.any(params < lb) or np.any(params > ub):
        raise ValueError("fit_obj.params lies outside the fitted parameter bounds")
    scalars = np.asarray([T0, Q0, doy0, fit_obj.Qbar], dtype=float)
    if not np.isfinite(scalars).all():
        raise ValueError("T0, Q0, doy0, and Qbar must be finite")
    if fit_obj.Qbar <= 0 or Q0 <= 0:
        raise ValueError("fit_obj.Qbar and issue-time discharge must be positive")
    if float(doy0) != np.floor(float(doy0)) or not 1 <= int(doy0) <= 366:
        raise ValueError("doy0 must be an integer day in [1, 366]")

    ta_array = _one_dimensional_float("Ta_future", np.asarray(Ta_future))
    future_doy = _one_dimensional_float("doy_future", np.asarray(doy_future))
    if len(ta_array) != len(future_doy):
        raise ValueError("Ta_future and doy_future must have identical lengths")
    if not np.isfinite(ta_array).all() or not np.isfinite(future_doy).all():
        raise ValueError("future forcing values must be finite")
    if not np.equal(future_doy, np.floor(future_doy)).all() or not (
        (future_doy >= 1) & (future_doy <= 366)
    ).all():
        raise ValueError("doy_future values must be integer days in [1, 366]")

    T = float(T0)
    theta = max(float(Q0) / fit_obj.Qbar, 1e-3)
    for ta, doy in zip(ta_array, future_doy):
        T = _step(T, float(ta), theta, int(doy), params, fit_obj.variant)
        if not np.isfinite(T):
            raise ValueError("Air2stream recursion produced a non-finite forecast")
    return float(T)


# --------------------------------------------------------------------------- #
# Run on a panel — returns predictions in the canonical schema
# --------------------------------------------------------------------------- #
def run_air2stream(panel: pd.DataFrame, masks, clim_air: F.HarmonicClimatology,
                   stations: tuple[str, ...] | None = None,
                   variant: str = "a8") -> pd.DataFrame:
    """Calibrate per-station on train, then forecast 1/3/7 days using
    climatological air temperature and persisted discharge (Track-H)."""
    if stations is None:
        stations = C.STATIONS
    out_frames = []
    train_mask = np.asarray(masks.train, dtype=bool)
    if train_mask.ndim != 1 or len(train_mask) != len(panel):
        raise ValueError("masks.train must align one-to-one with panel rows")
    for st in stations:
        station_rows = panel.site_id.eq(st).to_numpy()
        sub = panel.loc[station_rows].copy()
        sub["__train_row"] = train_mask[station_rows]
        sub = sub.sort_values("DATE").reset_index(drop=True)
        if sub.empty or pd.to_datetime(sub["DATE"]).duplicated().any():
            raise ValueError(f"Air2stream-{variant} requires unique rows for station {st}")
        Ta = sub["TEMP"].to_numpy(float)
        Q = sub["FLOW"].to_numpy(float)
        T = sub["WTEMP"].to_numpy(float)
        doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
        # The shared panel is fold-safely imputed for learned models.  A
        # process-style reference must not silently relabel those replacements
        # as measured physical drivers, so restore missing TEMP/FLOW to NaN for
        # calibration and use the explicit observation flags at issue time.
        wt_obs = (sub["WTEMP_observed"].to_numpy(bool) if "WTEMP_observed" in sub
                  else ~np.isnan(T))
        ta_obs = (sub["TEMP_observed"].to_numpy(bool) if "TEMP_observed" in sub
                  else ~np.isnan(Ta))
        flow_obs = (sub["FLOW_observed"].to_numpy(bool) if "FLOW_observed" in sub
                    else ~np.isnan(Q))
        Ta_measured = np.where(ta_obs, Ta, np.nan)
        Q_measured = np.where(flow_obs, Q, np.nan)
        # The mask was attached before sorting, so an otherwise-valid unsorted
        # caller cannot shift a development row into the calibration fit.
        train_rows = sub.pop("__train_row").to_numpy(bool)
        try:
            fit_obj = fit(Ta_measured[train_rows], Q_measured[train_rows], T[train_rows],
                          doy[train_rows], variant=variant, obs=wt_obs[train_rows])
        except Exception as exc:
            raise RuntimeError(
                f"Air2stream-{variant} calibration failed for station {st}"
            ) from exc

        # climatological air temperature for FUTURE (unobserved) days
        Ta_clim = clim_air.predict(st, doy)
        n = len(sub)
        for h in C.HORIZONS:
            yhat = np.full(n, np.nan)
            for t in range(n - h):
                if not (
                    wt_obs[t]
                    and flow_obs[t]
                    and np.isfinite(T[t])
                    and np.isfinite(Q[t])
                    and Q[t] > 0.0
                ):
                    continue
                # Track-H forcing: the FIRST step (t->t+1) is driven by the
                # air temperature OBSERVED at issue time (Ta_t, available under
                # Track-H); subsequent steps fall back to climatology. Previously
                # every step used climatology, which starved air2stream of the
                # observed forcing the learned baselines get and pushed its 1-day
                # RMSE to raw persistence.
                ta_step1 = (
                    Ta[t]
                    if ta_obs[t] and np.isfinite(Ta[t])
                    else Ta_clim[t + 1]
                )
                Ta_future = np.concatenate([[ta_step1], Ta_clim[t + 2: t + 1 + h]])
                doy_future = doy[t + 1: t + 1 + h]
                yhat[t] = forecast_horizon(fit_obj, T[t], Q[t], int(doy[t]),
                                           Ta_future, doy_future)
            valid = (~np.isnan(yhat[: n - h]) & ~np.isnan(T[h:])
                     & wt_obs[: n - h] & wt_obs[h:])
            issue_dates = sub["DATE"].to_numpy()[: n - h][valid]
            target_dates = sub["DATE"].to_numpy()[h:][valid]
            y_true = T[h:][valid]
            y_pred = yhat[: n - h][valid]
            # Both issue and target must remain inside one partition; otherwise
            # the horizon crosses a split boundary and the row is embargoed.
            splits = np.asarray([
                D.split_for_forecast_interval(issue, [target])
                for issue, target in zip(issue_dates, target_dates)
            ], dtype=object)
            out_frames.append(R.make_pred_frame(
                model=f"Air2stream-{variant}", scope="per_station", feature_set="phys",
                seed=0, site_id=np.full(len(issue_dates), st),
                horizon=np.full(len(issue_dates), h), split=splits,
                issue_date=issue_dates, target_date=target_dates,
                y_true=y_true, y_pred=y_pred,
            ))
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
