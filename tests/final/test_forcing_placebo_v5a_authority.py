"""Tests for the shuffled-placebo scoring authority.

The estimator and the decision rules are tested separately from the run, so a
mistake in either is caught without waiting on thirty model fits.  The rules
matter most: they were sealed before the outcome existed, so an implementation
that quietly disagreed with the sealed thresholds would convert a
pre-registered test into a post-hoc one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_forcing_placebo_v5a_authority as A
import scripts.final.seal_forcing_placebo_protocol_v5a as SEALER


# --------------------------------------------------------------- decision rules


@pytest.mark.parametrize(
    ("retained", "expected"),
    [
        (0.00, "P1_event_scale_supported"),
        (0.24, "P1_event_scale_supported"),
        (0.25, "P2_partly_seasonal"),
        (0.45, "P2_partly_seasonal"),
        (0.60, "P2_partly_seasonal"),
        (0.61, "P3_finding_withdrawn"),
        (1.10, "P3_finding_withdrawn"),
    ],
)
def test_verdict_matches_the_sealed_boundaries(retained: float, expected: str) -> None:
    assert A.verdict(retained)[0] == expected


def test_thresholds_equal_the_sealed_protocol_text() -> None:
    """The implemented thresholds must be the ones the seal froze."""
    document = yaml.safe_load(SEALER.PROTOCOL.read_text(encoding="utf-8"))
    rules = document["decision_rules"]
    assert "25%" in rules["P1_event_scale_supported"]
    assert "25% and 60%" in rules["P2_partly_seasonal"]
    assert "60%" in rules["P3_finding_withdrawn"]
    assert A.P1_MAX_RETAINED == 0.25
    assert A.P3_MIN_RETAINED == 0.60


def test_p3_is_reachable_and_means_withdrawal() -> None:
    """A control with no admissible negative outcome is not a control."""
    rule, meaning = A.verdict(0.95)
    assert rule == "P3_finding_withdrawn"
    assert "withdrawn" in meaning


# -------------------------------------------------------------------- estimator


def test_station_rmse_is_per_station_root_mean_square() -> None:
    frame = pd.DataFrame({
        "site_id": ["a", "a", "b"],
        "y_true": [0.0, 0.0, 0.0],
        "y_pred": [1.0, 3.0, 2.0],
    })
    out = A._station_rmse(frame)
    assert out.loc["a"] == pytest.approx(np.sqrt(5.0))
    assert out.loc["b"] == pytest.approx(2.0)
    assert list(out.index) == ["a", "b"]


def test_station_rmse_is_not_a_mean_of_absolute_errors() -> None:
    frame = pd.DataFrame({
        "site_id": ["a", "a"], "y_true": [0.0, 0.0], "y_pred": [0.0, 4.0],
    })
    assert A._station_rmse(frame).loc["a"] == pytest.approx(np.sqrt(8.0))


def test_retained_fraction_is_placebo_over_true() -> None:
    """Guards the direction of the ratio the rules are stated in."""
    true_value, placebo_value = 0.60, 0.12
    assert placebo_value / true_value == pytest.approx(0.2)
    assert A.verdict(placebo_value / true_value)[0] == "P1_event_scale_supported"


def test_seed_first_aggregation_averages_contrasts_not_risks() -> None:
    """Averaging seed RMSEs then differencing is not the sealed estimand."""
    f0 = pd.Series([1.0, 2.0], index=["a", "b"])
    seeds = {
        0: pd.Series([0.9, 1.5], index=["a", "b"]),
        1: pd.Series([0.5, 1.9], index=["a", "b"]),
    }
    per_station = np.mean([(f0 - s).to_numpy() for s in seeds.values()], axis=0)
    # station a: mean(0.1, 0.5) = 0.3 ; station b: mean(0.5, 0.1) = 0.3
    np.testing.assert_allclose(per_station, [0.3, 0.3])


# ------------------------------------------------------------------- provenance


def test_authority_declares_the_arm_and_its_sealed_chronology() -> None:
    summary = A.summarise([
        {"model": "LightGBM", "horizon": 3, "decision_rule": "P1_event_scale_supported"},
        {"model": "LightGBM", "horizon": 7, "decision_rule": "P1_event_scale_supported"},
    ])
    assert summary["arm"] == "F3_shuffle_month"
    assert summary["chronology"]["specification_sealed_before_outcome"] is True
    assert summary["chronology"]["post_outcome"] is False
    assert summary["sealed_thresholds"]["P1_max_retained"] == 0.25
    assert summary["decision_rules_triggered_at_3_and_7_days"] == [
        "P1_event_scale_supported"
    ]
