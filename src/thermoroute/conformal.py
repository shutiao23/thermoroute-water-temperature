"""Conformalised quantile regression (Romano et al., 2019).

The model's raw quantiles are under-calibrated (the baselines showed PICP≈0.80
for a nominal 90% band). CQR widens the interval using calibration-set
conformity scores, done per (station × horizon) — a Mondrian split that respects
the heteroscedasticity we expect across stations and lead times.

Note on the "guarantee": split-CQR's formal finite-sample (1−α) coverage holds
under exchangeability between calibration and test points. In this study
calibration (2018) and test (2019–2020) are disjoint future years and stations
are not i.i.d., so the assumption is **not satisfied strictly**. The intervals
should therefore be read as *empirically* near-nominal — and we report the
achieved PICP on the development years rather than claim a formal guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Exact split-conformal order statistic, or +inf when unattainable."""
    scores = np.sort(np.asarray(scores, dtype=float))
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(scores[k - 1]) if k <= n else float("inf")


def cqr_offsets(cal: pd.DataFrame, alpha: float = 0.10,
                purge_boundary: bool = True) -> dict[tuple, float]:
    """Per (site_id, horizon) interval offset q̂ from the calibration split.

    Uses the exact split-conformal order statistic — the ⌈(n+1)(1−α)⌉-th smallest
    conformity score (no off-by-one), returning +∞ when that index exceeds n.

    ``purge_boundary`` drops calibration samples whose *target* day falls after the
    calibration window (e.g. a 7-day-ahead forecast issued in late-Dec 2018 whose
    label lands in the 2019 test year), removing a small calib/test label leak.
    """
    cal = cal.copy()
    if purge_boundary and "target_date" in cal.columns:
        cal_end = np.datetime64(C.SPLIT.calib[1])
        cal = cal[pd.to_datetime(cal["target_date"]).to_numpy() <= cal_end]
    off = {}
    for (st, h), g in cal.groupby(["site_id", "horizon"]):
        y = g["y_true"].to_numpy(float)
        lo, hi = g["q05"].to_numpy(float), g["q95"].to_numpy(float)
        scores = np.maximum(lo - y, y - hi)
        off[(st, h)] = conformal_quantile(scores, alpha)
    return off


def block_cqr_offsets(cal: pd.DataFrame, alpha: float = 0.10,
                      block_days: int = 7) -> dict[tuple, float]:
    """Temporal-block CQR using the maximum score in consecutive time blocks.

    The resulting intervals target empirical robustness to short-range dependence;
    they are not advertised as a finite-sample guarantee for arbitrary time series.
    """
    if block_days < 1:
        raise ValueError("block_days must be positive")
    required = {"site_id", "horizon", "issue_date", "y_true", "q05", "q95"}
    missing = required - set(cal)
    if missing:
        raise ValueError(f"calibration frame missing columns: {sorted(missing)}")
    output = {}
    for (site, horizon), group in cal.groupby(["site_id", "horizon"]):
        group = group.sort_values("issue_date").reset_index(drop=True)
        score = np.maximum(group.q05 - group.y_true, group.y_true - group.q95).to_numpy(float)
        block = np.arange(len(group)) // block_days
        block_maxima = np.array([
            np.nanmax(score[block == index]) for index in np.unique(block)
        ])
        output[(site, horizon)] = conformal_quantile(block_maxima, alpha)
    return output


def hierarchical_cqr_offsets(cal: pd.DataFrame, group_col: str,
                             alpha: float = 0.10, min_group: int = 100
                             ) -> dict[tuple, float]:
    """Offsets pooled by deployment-visible group and horizon.

    Keys are ``(group, horizon)`` plus ``("__global__", horizon)`` fallbacks.
    This supports held-group deployment without reading held-site calibration labels.
    """
    required = {group_col, "horizon", "y_true", "q05", "q95"}
    missing = required - set(cal)
    if missing:
        raise ValueError(f"calibration frame missing columns: {sorted(missing)}")
    result = {}
    for horizon, group in cal.groupby("horizon"):
        score = np.maximum(group.q05 - group.y_true, group.y_true - group.q95)
        result[("__global__", int(horizon))] = conformal_quantile(score, alpha)
    for (label, horizon), group in cal.groupby([group_col, "horizon"]):
        if len(group) < min_group:
            continue
        score = np.maximum(group.q05 - group.y_true, group.y_true - group.q95)
        result[(str(label), int(horizon))] = conformal_quantile(score, alpha)
    return result


def apply_hierarchical_cqr(pred: pd.DataFrame, offsets: dict[tuple, float],
                           group_col: str) -> pd.DataFrame:
    output = pred.copy()
    values = []
    for label, horizon in zip(output[group_col], output.horizon):
        values.append(offsets.get(
            (str(label), int(horizon)), offsets[("__global__", int(horizon))]
        ))
    delta = np.asarray(values, dtype=float)
    output["q05"] = output.q05.to_numpy(float) - delta
    output["q95"] = output.q95.to_numpy(float) + delta
    return output


def apply_cqr(pred: pd.DataFrame, offsets: dict[tuple, float]) -> pd.DataFrame:
    """Return a copy with q05/q95 conformalised; q50 and MSE point stay untouched."""
    out = pred.copy()
    key = list(zip(out["site_id"], out["horizon"]))
    missing = sorted(
        set(key) - set(offsets),
        key=lambda value: (str(value[0]), int(value[1])),
    )
    if missing:
        raise KeyError(
            "CQR offsets are missing prediction site×horizon keys: "
            f"{missing[:5]}"
        )
    delta = np.array([offsets[k] for k in key], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("CQR offsets must be finite for every prediction key")
    out["q05"] = out["q05"].to_numpy(float) - delta
    out["q95"] = out["q95"].to_numpy(float) + delta
    return out
