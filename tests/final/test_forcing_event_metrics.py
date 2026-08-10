"""Tests for the forcing-arm event and decision metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_forcing_event_metrics as E

SUMMARY = E.FINAL / "forcing_event_metrics_summary.parquet"
pytestmark = pytest.mark.skipif(
    not SUMMARY.exists(), reason="event metrics have not been built"
)


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_parquet(SUMMARY)


def test_contingency_table_is_the_standard_definition() -> None:
    observed = np.array([1, 1, 1, 0, 0], dtype=bool)
    predicted = np.array([1, 1, 0, 1, 0], dtype=bool)
    stats = E._contingency(observed, predicted)
    assert stats["pod"] == pytest.approx(2 / 3)       # 2 hits, 1 miss
    assert stats["far"] == pytest.approx(1 / 3)       # 1 false alarm of 3 forecasts
    assert stats["csi"] == pytest.approx(2 / 4)       # hits / (hits+miss+fa)
    assert stats["n_events"] == 3


def test_contingency_handles_a_cell_with_no_events() -> None:
    stats = E._contingency(np.zeros(5, bool), np.zeros(5, bool))
    assert np.isnan(stats["pod"])
    assert stats["n_events"] == 0


def test_brier_on_a_deterministic_forecast_is_the_error_rate() -> None:
    observed = np.array([1, 0, 1, 0], dtype=bool)
    predicted = np.array([1, 1, 1, 0], dtype=bool)
    assert E._contingency(observed, predicted)["brier"] == pytest.approx(0.25)


def test_every_event_class_and_metric_is_covered(summary: pd.DataFrame) -> None:
    assert set(summary["event"]) == set(E.EVENTS)
    assert set(summary["metric"]) == {"pod", "far", "csi", "brier"}
    assert set(summary["horizon"]) == {1, 3, 7}


def test_the_oracle_arm_detects_more_events(summary: pd.DataFrame) -> None:
    pod = summary[(summary["metric"] == "pod") & (summary["horizon"] == 7)]
    assert (pod["median_delta_F3_minus_F0"] > 0).all()
    assert (pod["ci_low"] > 0).all()


def test_the_gain_is_not_a_base_rate_trade(summary: pd.DataFrame) -> None:
    """More detections must not come with more false alarms."""
    far = summary[(summary["metric"] == "far") & (summary["horizon"] == 7)]
    assert (far["median_delta_F3_minus_F0"] < 0).all()
    assert (far["ci_high"] < 0).all()
    # CSI accounts for both and must improve
    csi = summary[(summary["metric"] == "csi") & (summary["horizon"] == 7)]
    assert (csi["median_delta_F3_minus_F0"] > 0).all()


def test_the_event_gain_follows_the_rmse_lead_structure(summary: pd.DataFrame) -> None:
    """Small at one day, large at three and seven, as the RMSE result is."""
    csi = summary[(summary["metric"] == "csi") & (summary["model"] == "LightGBM")]
    by_lead = csi.groupby("horizon")["median_delta_F3_minus_F0"].median()
    assert by_lead[1] < by_lead[3]
    assert by_lead[1] < by_lead[7]


def test_higher_is_better_flag_matches_the_metric(summary: pd.DataFrame) -> None:
    for row in summary.itertuples():
        assert row.higher_is_better == (row.metric in ("pod", "csi"))


def test_thresholds_are_training_period_only() -> None:
    assert E.TRAIN_END == pd.Timestamp("2015-12-31")
    assert E.TRAIN_START == pd.Timestamp("2006-01-01")
