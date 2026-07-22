"""Evaluation metrics: point, probabilistic and high-temperature event scores.

Everything operates on 1-D numpy arrays of *real units* (degrees Celsius).  No
metric is computed on standardised values — predictions are always inverse
transformed first.
"""

from __future__ import annotations

import numpy as np

try:  # sklearn is available; keep a graceful fallback for the AUC scores
    from sklearn.metrics import average_precision_score, roc_auc_score
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


# --------------------------------------------------------------------------- #
# Point forecasts
# --------------------------------------------------------------------------- #
def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(yhat - y))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def nse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency (identical formula to R^2 vs observed mean)."""
    return r2(y, yhat)


def kge(y: np.ndarray, yhat: np.ndarray) -> float:
    """Kling-Gupta efficiency (2009)."""
    if y.std() < 1e-9 or yhat.std() < 1e-9:
        return float("nan")
    r = np.corrcoef(y, yhat)[0, 1]
    alpha = yhat.std() / y.std()
    beta = yhat.mean() / (y.mean() + 1e-12)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(100.0 * np.sum(yhat - y) / (np.sum(y) + 1e-12))


def skill_score(y: np.ndarray, yhat: np.ndarray, ref: np.ndarray) -> float:
    """RMSE skill vs a reference forecast (e.g. persistence): 1 - RMSE/RMSE_ref."""
    return float(1.0 - rmse(y, yhat) / (rmse(y, ref) + 1e-12))


def point_scores(y: np.ndarray, yhat: np.ndarray,
                 ref: np.ndarray | None = None) -> dict[str, float]:
    out = {"RMSE": rmse(y, yhat), "MAE": mae(y, yhat), "BIAS": bias(y, yhat),
           "R2": r2(y, yhat), "NSE": nse(y, yhat), "KGE": kge(y, yhat),
           "PBIAS": pbias(y, yhat)}
    if ref is not None:
        out["SKILL_RMSE"] = skill_score(y, yhat, ref)
    return out


# --------------------------------------------------------------------------- #
# Probabilistic forecasts (quantile triples q_lo < q_med < q_hi)
# --------------------------------------------------------------------------- #
def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Winkler interval score for a central (1-alpha) interval."""
    width = hi - lo
    below = (y < lo)
    above = (y > hi)
    pen = np.where(below, (2.0 / alpha) * (lo - y), 0.0) \
        + np.where(above, (2.0 / alpha) * (y - hi), 0.0)
    return float(np.mean(width + pen))


def three_quantile_score(y: np.ndarray, quants: dict[float, np.ndarray]) -> float:
    """Twice the mean pinball loss over the supplied quantiles.

    With only q05/q50/q95 this is a proper discrete-quantile score, not CRPS.
    Calling it CRPS would imply integration over a predictive distribution that
    the model does not provide.
    """
    taus = sorted(quants)
    return float(np.mean([pinball(y, quants[t], t) for t in taus]) * 2.0)


def probabilistic_scores(y: np.ndarray, quants: dict[float, np.ndarray],
                         central: tuple[float, float] = (0.05, 0.95)) -> dict[str, float]:
    lo, hi = quants[central[0]], quants[central[1]]
    alpha = central[0] + (1.0 - central[1])
    out = {
        "PINBALL": float(np.mean([pinball(y, quants[t], t) for t in quants])),
        "THREE_QUANTILE_SCORE": three_quantile_score(y, quants),
        "PICP": coverage(y, lo, hi),
        "MPIW": mean_interval_width(lo, hi),
        "WINKLER": winkler(y, lo, hi, alpha),
    }
    return out


# --------------------------------------------------------------------------- #
# High-temperature exceedance (binary warning)
# --------------------------------------------------------------------------- #
def brier(y_bin: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_bin) ** 2))


def brier_skill(y_bin: np.ndarray, p: np.ndarray, reference_p: np.ndarray) -> float:
    """Brier skill against probabilities fixed outside the evaluation sample."""
    reference_p = np.asarray(reference_p, dtype=float)
    if reference_p.shape != np.asarray(y_bin).shape:
        raise ValueError("reference probability must align one-to-one with outcomes")
    bs_ref = brier(y_bin, reference_p)
    return float(1.0 - brier(y_bin, p) / (bs_ref + 1e-12))


def event_scores(y_bin: np.ndarray, p: np.ndarray,
                 reference_p: np.ndarray | None = None) -> dict[str, float]:
    """Event verification without an oracle evaluation-period climatology."""
    y_bin = np.asarray(y_bin, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {"BRIER": brier(y_bin, p), "BASE_RATE": float(y_bin.mean()),
           "LOG_LOSS": float(-np.mean(y_bin * np.log(p) + (1 - y_bin) * np.log(1 - p)))}
    if reference_p is not None:
        out["BRIER_SKILL"] = brier_skill(y_bin, p, np.asarray(reference_p, dtype=float))
    if _HAS_SK and 0 < y_bin.sum() < len(y_bin):
        out["AUROC"] = float(roc_auc_score(y_bin, p))
        out["AUPRC"] = float(average_precision_score(y_bin, p))
    else:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
    return out
