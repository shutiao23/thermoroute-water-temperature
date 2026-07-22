"""Forecast value via the cost–loss decision model (Relative Economic Value).

A decision-maker can take a protective action against a high-temperature
exceedance event at cost C; an unprotected event costs L. With cost–loss ratio
α = C/L, the optimal action given a *calibrated* probability p is to act iff
p > α. Relative Economic Value (Richardson 2000; Wilks) measures the fraction of
the perfect-forecast value a forecast captures, relative to a climatological
default:

    REV(α) = (E_clim − E_forecast) / (E_clim − E_perfect)

expressed per unit L: ``E_f/L = α·P(act) + P(miss)`` and
``E_perfect/L = s·α``, with base rate *s*.  Here the reference expense is
computed from an out-of-evaluation, seasonally varying probability supplied for
every row.  It is therefore not the simpler constant-base-rate expression
``min(α, s)``.

This turns the calibrated exceedance probabilities into a management-relevant
metric and is exactly why calibration matters: the rule p>α is only optimal when
p is reliable.
"""

from __future__ import annotations

import numpy as np


def rev_curve(events: np.ndarray, score: np.ndarray, alphas: np.ndarray,
              probabilistic: bool = True,
              reference_probability: np.ndarray | None = None) -> np.ndarray:
    """Relative Economic Value over a grid of cost–loss ratios.

    ``events``  binary outcome (1 = exceedance occurred).
    ``score``   calibrated probability (probabilistic=True) or a fixed binary
                warning (probabilistic=False, e.g. persistence > threshold).
    """
    events = np.asarray(events, dtype=int)
    score = np.asarray(score, dtype=float)
    if reference_probability is None:
        raise ValueError(
            "reference_probability must be fitted outside the evaluation sample"
        )
    reference_probability = np.asarray(reference_probability, dtype=float)
    if reference_probability.shape != events.shape:
        raise ValueError("reference_probability must align one-to-one with events")
    s = float(events.mean())
    rev = np.empty_like(alphas, dtype=float)
    for i, a in enumerate(alphas):
        act = (score > a) if probabilistic else (score > 0.5)
        p_act = float(act.mean())
        miss = float(((~act) & (events == 1)).mean())
        e_f = a * p_act + miss
        ref_act = reference_probability > a
        e_clim = a * float(ref_act.mean()) + float(((~ref_act) & (events == 1)).mean())
        e_perf = s * a
        denom = e_clim - e_perf
        rev[i] = (e_clim - e_f) / denom if denom > 1e-12 else np.nan
    return rev


def value_summary(events: np.ndarray, score: np.ndarray,
                  probabilistic: bool = True,
                  reference_probability: np.ndarray | None = None,
                  report_alphas=(0.05, 0.10, 0.20, 0.50)) -> dict:
    """Peak REV (and the α achieving it) plus REV at representative α values."""
    grid = np.linspace(0.01, 0.99, 99)
    rev = rev_curve(events, score, grid, probabilistic, reference_probability)
    out = {"base_rate": float(events.mean()),
           "REV_max": float(np.nanmax(rev)),
           "alpha_at_max": float(grid[int(np.nanargmax(rev))])}
    for a in report_alphas:
        r = rev_curve(events, score, np.array([a]), probabilistic,
                      reference_probability)[0]
        out[f"REV@{a:g}"] = float(r)
    return out


def cluster_bootstrap_rev(events: np.ndarray, score: np.ndarray,
                          reference_probability: np.ndarray,
                          clusters: np.ndarray, alpha: float, *,
                          probabilistic: bool = True, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float]:
    """Percentile interval resampling whole spatial/hydrologic clusters."""
    events = np.asarray(events)
    score = np.asarray(score)
    reference_probability = np.asarray(reference_probability)
    clusters = np.asarray(clusters)
    if not (len(events) == len(score) == len(reference_probability) == len(clusters)):
        raise ValueError("events, scores, references, and clusters must align")
    unique = np.unique(clusters)
    if len(unique) < 2:
        return float("nan"), float("nan")
    by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_cluster[cluster] for cluster in sampled])
        value = rev_curve(
            events[index], score[index], np.array([alpha]), probabilistic,
            reference_probability[index],
        )[0]
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
