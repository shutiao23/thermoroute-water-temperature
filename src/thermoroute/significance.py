"""Significance tools that respect temporal autocorrelation.

Daily water temperature is strongly autocorrelated, so i.i.d. bootstrap and
naive paired t-tests understate uncertainty.  We use a moving-block bootstrap
for confidence intervals and a Diebold-Mariano test with an autocorrelation
(HAC) variance for pairwise model comparison.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def moving_block_bootstrap_ci(
    errors_sq: np.ndarray, block: int = 30, n_boot: int = 2000,
    seed: int = 0, stat: str = "rmse",
) -> tuple[float, float, float]:
    """Bootstrap CI for RMSE/MAE from a series of per-day squared/abs errors.

    ``errors_sq`` should be squared errors for RMSE or absolute errors for MAE.
    Returns ``(point, lo95, hi95)``.
    """
    rng = np.random.default_rng(seed)
    n = len(errors_sq)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    point = float(np.sqrt(errors_sq.mean())) if stat == "rmse" else float(errors_sq.mean())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, starts_max + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        samp = errors_sq[idx]
        boots[b] = np.sqrt(samp.mean()) if stat == "rmse" else samp.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def block_bootstrap_station_avg_rmse(
    err2_by_station: dict[str, np.ndarray], block: int = 30, n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the *station-averaged* RMSE (the headline aggregation).

    Each station's per-day squared errors are block-resampled independently;
    per-station RMSE is computed, then averaged over stations. This reproduces
    the reported point estimate (mean over stations of per-station RMSE) and,
    crucially, does NOT average predictions across stations (which would cancel
    independent errors and deflate the RMSE).
    """
    rng = np.random.default_rng(seed)
    stations = list(err2_by_station)
    point = float(np.mean([np.sqrt(err2_by_station[s].mean()) for s in stations]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        per_station = []
        for s in stations:
            e = err2_by_station[s]
            n = len(e)
            n_blocks = int(np.ceil(n / block))
            starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
            idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
            per_station.append(np.sqrt(e[idx].mean()))
        boots[b] = np.mean(per_station)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def diebold_mariano(
    err_a: np.ndarray, err_b: np.ndarray, h: int = 1, loss: str = "sq",
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy (model A vs B).

    Negative DM statistic ⇒ model A has the smaller loss.  Uses a Newey-West HAC
    variance with bandwidth ``h-1`` (small-sample Harvey correction applied).
    Returns ``(dm_stat, two_sided_p)``.
    """
    if loss == "sq":
        d = err_a ** 2 - err_b ** 2
    else:
        d = np.abs(err_a) - np.abs(err_b)
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * cov
    var = var / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var)
    # Harvey, Leybourne & Newbold small-sample correction
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= adj
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)
