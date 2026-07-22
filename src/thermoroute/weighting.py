"""Shared weighting contracts for station-level estimands.

Rows are not the scientific sampling unit in the spatial-transfer analyses.
This helper gives every represented station the same total weight while keeping
the mean row weight equal to one for estimators whose optimisers are sensitive
to the absolute weight scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


STATION_EQUAL_WEIGHTING = "equal_total_weight_per_station"
ROW_EQUAL_WEIGHTING = "equal_weight_per_finite_training_row"


def station_equal_sample_weight(site_ids: object) -> np.ndarray:
    """Return positive finite weights with equal total mass per station."""
    sites = pd.Series(site_ids, dtype="string")
    if sites.empty or sites.isna().any() or sites.eq("").any():
        raise ValueError("station-balanced weights require nonempty station ids")
    counts = sites.value_counts(dropna=False)
    weights = sites.map((1.0 / counts).to_dict()).to_numpy(float)
    weights *= len(weights) / weights.sum()
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("station-balanced weights are invalid")
    return weights
