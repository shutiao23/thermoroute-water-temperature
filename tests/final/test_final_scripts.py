"""Tests for the final-stage analysis scripts (protocol v1).

Synthetic fixtures only.  Covers the mechanism state classification
(issue-time vs outcome-conditioned separation, signed warming/cooling), the
spatial-factorial cell contract, and the claim resolver's new estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thermoroute import final_results as FR


class _Row:
    """Lightweight row-like object with attributes for state_of()."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_state_of_signed_warming_cooling():
    import scripts.final.run_mechanism_analysis as M
    sq = {
        "anomaly": {"q10": -2.0, "q90": 2.0},
        "trend7": {"q10": -0.5, "q90": 0.5},
        "disequilibrium": {"q10": -3.0, "q90": 3.0},
        "flow_change": {"q10": -0.4, "q90": 0.4},
        "month_flow": {6: {"q10": 10.0, "q90": 100.0}},
    }
    # rapid recent warming (positive trend above q90)
    row = _Row(anomaly=1.0, trend7=1.2, diseq=0.0, dlogq3=0.0, month=6, FLOW=50.0)
    states = M.state_of(row, sq)
    assert "issue_rapid_recent_warming" in states
    assert "issue_rapid_recent_cooling" not in states
    # rapid recent cooling (negative trend below q10)
    row = _Row(anomaly=1.0, trend7=-1.2, diseq=0.0, dlogq3=0.0, month=6, FLOW=50.0)
    states = M.state_of(row, sq)
    assert "issue_rapid_recent_cooling" in states
    # signed disequilibrium is directional
    row = _Row(anomaly=0.0, trend7=0.0, diseq=5.0, dlogq3=0.0, month=6, FLOW=50.0)
    states = M.state_of(row, sq)
    assert "issue_strong_warming_pressure" in states
    assert "issue_strong_cooling_pressure" not in states
    # station-relative flow uses station-month quantiles, not global cfs
    row = _Row(anomaly=0.0, trend7=0.0, diseq=0.0, dlogq3=0.0, month=6, FLOW=200.0)
    states = M.state_of(row, sq)
    assert "issue_high_flow" in states
    row = _Row(anomaly=0.0, trend7=0.0, diseq=0.0, dlogq3=0.0, month=6, FLOW=5.0)
    states = M.state_of(row, sq)
    assert "issue_low_flow" in states


def test_outcome_state_separates_warming_and_cooling():
    import scripts.final.run_mechanism_analysis as M
    sq = {"dT": {7: {"q10": -4.0, "q90": 4.0}},
          "target_wt": {7: {"q10": 5.0, "q90": 25.0}},
          "month_flow": {6: {"q10": 10.0, "q90": 100.0}}}
    # actual rapid warming: positive change above q90
    row = _Row(horizon=7, dT_actual=6.0, WT_t=20.0, FLOW_t=50.0, month=6)
    states = M.outcome_state_of(row, sq)
    assert "actual_rapid_warming" in states
    assert "actual_rapid_cooling" not in states
    # actual rapid cooling: negative change below q10
    row = _Row(horizon=7, dT_actual=-6.0, WT_t=20.0, FLOW_t=50.0, month=6)
    states = M.outcome_state_of(row, sq)
    assert "actual_rapid_cooling" in states
    assert "actual_rapid_warming" not in states
    # |dT| below the thresholds is neither
    row = _Row(horizon=7, dT_actual=2.0, WT_t=20.0, FLOW_t=50.0, month=6)
    states = M.outcome_state_of(row, sq)
    assert "actual_rapid_warming" not in states
    assert "actual_rapid_cooling" not in states


def test_claim_resolver_spatial_estimators():
    effects = pd.DataFrame({
        "site_id": ["a", "a", "b", "b", "a", "a", "b", "b"],
        "seed": [0, 1, 0, 1, 0, 1, 0, 1],
        "geometry": ["region", "region", "region", "region",
                     "random", "random", "random", "random"],
        "rmse": [1.0, 1.1, 1.5, 1.4, 0.9, 0.95, 1.3, 1.35],
    })
    claim = {"source_table": "spatial_effects.parquet",
             "estimator": "median_station_region_minus_random_rmse",
             "filter": {"model": "LightGBM", "horizon": 3, "adaptation": "local"}}
    # the resolver pivots site x geometry and takes the median of
    # region - random per site
    frame = effects.assign(model="LightGBM", horizon=3, adaptation="local")
    value = FR._resolve_estimator(frame, claim)
    pivot = frame.pivot_table(index="site_id", columns="geometry",
                              values="rmse", aggfunc="mean")
    expected = float((pivot["region"] - pivot["random"]).median())
    assert abs(value - expected) < 1e-12
    assert abs(value - 0.125) < 1e-9


def test_claim_resolver_nearest_km():
    frame = pd.DataFrame({
        "geometry": ["random", "random", "region"],
        "nearest_km": [60.0, 62.0, 268.0],
    })
    value = FR._resolve_estimator(
        frame, {"estimator": "median_nearest_km", "filter": {"geometry": "random"}})
    assert value == 61.0


def test_half_life_terminology():
    # t_1/2 = ln(0.5)/ln(phi); e-folding time is -1/ln(phi); they differ.
    phi = 0.9
    half_life = np.log(0.5) / np.log(phi)
    e_folding = -1.0 / np.log(phi)
    assert abs(half_life - 6.58) < 0.01
    assert abs(e_folding - 9.49) < 0.01
    assert not np.isclose(half_life, e_folding)
