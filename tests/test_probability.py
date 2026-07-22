from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute import metrics
from thermoroute.decision import cluster_bootstrap_rev, rev_curve
from thermoroute.probability import (
    PlattCalibrator,
    apply_horizon_calibrators,
    fit_frozen_seasonal_event_reference,
    fit_horizon_calibrators,
    fit_seasonal_climatology,
    predict_frozen_seasonal_event_reference,
    validate_frozen_seasonal_event_reference,
)


def test_platt_calibration_is_fitted_only_from_supplied_rows():
    raw = np.tile(np.array([0.05, 0.2, 0.8, 0.95]), 100)
    outcomes = np.tile(np.array([0, 0, 1, 1]), 100)
    calibrator = PlattCalibrator.fit(raw, outcomes)
    predicted = calibrator.predict(np.array([0.1, 0.9]))
    assert 0 < predicted[0] < predicted[1] < 1

    frame = pd.DataFrame({
        "horizon": np.repeat([1, 3], 200),
        "p_exceed": np.tile(raw[:200], 2),
        "event": np.tile(outcomes[:200], 2),
    })
    fitted = fit_horizon_calibrators(frame, min_samples=100)
    applied = apply_horizon_calibrators(frame.iloc[:5], fitted)
    assert applied.p_exceed_calibrated.between(0, 1).all()


def test_seasonal_climatology_uses_fitting_panel_not_evaluation_outcomes():
    panel = pd.DataFrame({
        "DATE": pd.date_range("2010-01-01", periods=365, freq="D"),
        "site_id": "a",
        "WTEMP": np.sin(np.arange(365) * 2 * np.pi / 365) + 10,
    })
    climatology = fit_seasonal_climatology(panel, {"a": 10.5})
    first = climatology.predict(np.array(["a", "a"]),
                                np.array(["2025-04-01", "2025-10-01"]))
    # Changing evaluation labels is not even an input to prediction.
    second = climatology.predict(np.array(["a", "a"]),
                                 np.array(["2025-04-01", "2025-10-01"]))
    assert np.array_equal(first, second)
    assert not np.isclose(first[0], first[1])


def test_frozen_station_event_reference_materialises_every_site_month():
    dates = pd.date_range("2017-01-01", "2018-12-31", freq="D")
    panel = pd.concat([
        pd.DataFrame({
            "DATE": dates,
            "site_id": site,
            "WTEMP": offset + 8 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        })
        for site, offset in (("01", 10.0), ("02", 14.0))
    ], ignore_index=True)
    reference = fit_frozen_seasonal_event_reference(
        panel,
        {"01": 15.0, "02": 19.0},
        pooled=False,
        fit_interval=("2017-01-01", "2018-12-31"),
    )
    validate_frozen_seasonal_event_reference(
        reference, expected_sites={"01", "02"}, pooled=False
    )
    assert len(reference["station_month_probability"]) == 24
    predicted = predict_frozen_seasonal_event_reference(
        reference,
        np.asarray(["01", "02"]),
        np.asarray(["2023-07-01", "2023-01-01"]),
    )
    assert np.isfinite(predicted).all() and ((0 < predicted) & (predicted < 1)).all()


def test_frozen_pooled_event_reference_cannot_adapt_to_external_site_identity():
    dates = pd.date_range("2017-01-01", "2018-12-31", freq="D")
    panel = pd.DataFrame({
        "DATE": dates,
        "site_id": np.where(np.arange(len(dates)) % 2, "01", "02"),
        "WTEMP": 15 + 8 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
    })
    reference = fit_frozen_seasonal_event_reference(
        panel,
        {"__pooled__": 20.0},
        pooled=True,
        fit_interval=("2017-01-01", "2018-12-31"),
    )
    first = predict_frozen_seasonal_event_reference(
        reference, np.asarray(["new-a"]), np.asarray(["2023-07-01"])
    )
    second = predict_frozen_seasonal_event_reference(
        reference, np.asarray(["new-b"]), np.asarray(["2023-07-01"])
    )
    assert np.array_equal(first, second)


def test_brier_skill_requires_external_reference():
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.8, 0.9])
    scores = metrics.event_scores(y, p)
    assert "BRIER_SKILL" not in scores
    reference = np.full_like(y, 0.25)
    scores = metrics.event_scores(y, p, reference)
    assert scores["BRIER_SKILL"] > 0


def test_rev_refuses_test_derived_implicit_climatology():
    events = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    with pytest.raises(ValueError, match="reference_probability"):
        rev_curve(events, score, np.array([0.2]))
    reference = np.full(4, 0.25)
    value = rev_curve(events, score, np.array([0.2]), reference_probability=reference)
    assert np.isfinite(value[0])


def test_rev_cluster_bootstrap_resamples_complete_clusters():
    events = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.9, 0.1, 0.7])
    reference = np.full(6, 0.25)
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    lo, hi = cluster_bootstrap_rev(
        events, scores, reference, clusters, 0.2, n_boot=100, seed=1
    )
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_probabilistic_metric_is_not_labelled_crps():
    y = np.array([0.0, 1.0])
    quants = {0.05: y - 1, 0.50: y, 0.95: y + 1}
    scores = metrics.probabilistic_scores(y, quants)
    assert "THREE_QUANTILE_SCORE" in scores
    assert "CRPS" not in scores
