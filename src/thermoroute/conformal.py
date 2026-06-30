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
achieved PICP on the blind years rather than claim a formal guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


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
        scores = np.sort(np.maximum(lo - y, y - hi))
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - alpha)))      # 1-indexed order statistic
        # When the calibration set is too small for an exact (1−α) guarantee
        # (k>n), fall back to the largest observed conformity score — a finite,
        # conservative offset — rather than +∞.
        off[(st, h)] = float(scores[min(k, n) - 1])
    return off


def apply_cqr(pred: pd.DataFrame, offsets: dict[tuple, float]) -> pd.DataFrame:
    """Return a copy with conformalised q05/q95 (q50/median untouched)."""
    out = pred.copy()
    key = list(zip(out["site_id"], out["horizon"]))
    delta = np.array([offsets.get(k, 0.0) for k in key])
    out["q05"] = out["q05"].to_numpy(float) - delta
    out["q95"] = out["q95"].to_numpy(float) + delta
    return out
