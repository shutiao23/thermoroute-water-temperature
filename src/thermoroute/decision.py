"""Forecast value via the cost–loss decision model (Relative Economic Value).

A decision-maker can take a protective action against a high-temperature
exceedance event at cost C; an unprotected event costs L. With cost–loss ratio
α = C/L, the optimal action given a *calibrated* probability p is to act iff
p > α. Relative Economic Value (Richardson 2000; Wilks) measures the fraction of
the perfect-forecast value a forecast captures, relative to a climatological
default:

    REV(α) = (E_clim − E_forecast) / (E_clim − E_perfect)

expressed per unit L:  E_f/L = α·P(act) + P(miss),
E_clim/L = min(α, s),  E_perfect/L = s·α,  with base rate s.

This turns the calibrated exceedance probabilities into a management-relevant
metric and is exactly why calibration matters: the rule p>α is only optimal when
p is reliable.
"""

from __future__ import annotations

import numpy as np


def rev_curve(events: np.ndarray, score: np.ndarray, alphas: np.ndarray,
              probabilistic: bool = True) -> np.ndarray:
    """Relative Economic Value over a grid of cost–loss ratios.

    ``events``  binary outcome (1 = exceedance occurred).
    ``score``   calibrated probability (probabilistic=True) or a fixed binary
                warning (probabilistic=False, e.g. persistence > threshold).
    """
    s = float(events.mean())
    rev = np.empty_like(alphas, dtype=float)
    for i, a in enumerate(alphas):
        act = (score > a) if probabilistic else (score > 0.5)
        p_act = float(act.mean())
        miss = float(((~act) & (events == 1)).mean())
        e_f = a * p_act + miss
        e_clim = min(a, s)
        e_perf = s * a
        denom = e_clim - e_perf
        rev[i] = (e_clim - e_f) / denom if denom > 1e-12 else np.nan
    return rev


def value_summary(events: np.ndarray, score: np.ndarray,
                  probabilistic: bool = True,
                  report_alphas=(0.05, 0.10, 0.20, 0.50)) -> dict:
    """Peak REV (and the α achieving it) plus REV at representative α values."""
    grid = np.linspace(0.01, 0.99, 99)
    rev = rev_curve(events, score, grid, probabilistic)
    out = {"base_rate": float(events.mean()),
           "REV_max": float(np.nanmax(rev)),
           "alpha_at_max": float(grid[int(np.nanargmax(rev))])}
    for a in report_alphas:
        r = rev_curve(events, score, np.array([a]), probabilistic)[0]
        out[f"REV@{a:g}"] = float(r)
    return out
