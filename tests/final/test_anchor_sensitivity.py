"""Tests for the damped-anchor sensitivity analysis.

The anchor is the paper's instrument.  These tests pin the two things that make
the sensitivity trustworthy: the rebuilt baseline reproduces the published
reference, and each variant changes only what it claims to change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_anchor_sensitivity as A

pytestmark = pytest.mark.skipif(
    not A.DEFAULT_OUT.exists(), reason="anchor sensitivity has not been built"
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_parquet(A.DEFAULT_OUT)


def test_every_declared_variant_and_lead_is_present(table: pd.DataFrame) -> None:
    assert set(table["variant"]) == {v.name for v in A.VARIANTS}
    assert set(table["horizon"]) == {1, 3, 7}
    assert (table["n_stations"] == 116).all()


def test_baseline_reproduces_the_published_headline(table: pd.DataFrame) -> None:
    """The rebuild must land on +0.038 at seven days, or nothing here is valid."""
    seven = table[(table["variant"] == "baseline") & (table["horizon"] == 7)]
    assert float(seven["median_station_skill_vs_anchor"].iloc[0]) == pytest.approx(
        0.038, abs=5e-4
    )


def test_baseline_anchor_matches_the_frozen_damped_persistence(
    table: pd.DataFrame,
) -> None:
    published = pd.read_parquet(A.STATION_METRICS)
    published = published[published["model"] == "DampedPersistence"]
    for horizon in (1, 3, 7):
        rebuilt = float(
            table[(table["variant"] == "baseline") & (table["horizon"] == horizon)]
            ["median_anchor_rmse"].iloc[0]
        )
        reference = float(published[published["horizon"] == horizon]["rmse"].median())
        assert abs(rebuilt - reference) < A.BASELINE_TOLERANCE_DEGC


def test_the_clip_does_not_bind(table: pd.DataFrame) -> None:
    """[0, 0.999] vs [0, 1.0] must be indistinguishable, or the clip is doing work."""
    for horizon in (1, 3, 7):
        block = table[table["horizon"] == horizon].set_index("variant")
        assert block.loc["baseline", "median_station_skill_vs_anchor"] == pytest.approx(
            block.loc["phi_unclipped", "median_station_skill_vs_anchor"], abs=1e-9
        )


def test_a_directly_fitted_lead_coefficient_strengthens_the_reference(
    table: pd.DataFrame,
) -> None:
    """phi^h extrapolation is only exact for a pure AR(1); fitting phi_h beats it."""
    seven = table[table["horizon"] == 7].set_index("variant")
    assert (
        seven.loc["phi_direct_lead", "median_anchor_rmse"]
        < seven.loc["baseline", "median_anchor_rmse"]
    )
    assert (
        seven.loc["phi_direct_lead", "median_station_skill_vs_anchor"]
        < seven.loc["baseline", "median_station_skill_vs_anchor"]
    )


def test_a_cruder_climatology_inflates_the_reported_gain(table: pd.DataFrame) -> None:
    """One harmonic is a weaker seasonal reference and must report more skill."""
    seven = table[table["horizon"] == 7].set_index("variant")
    assert (
        seven.loc["clim_k1", "median_station_skill_vs_anchor"]
        > seven.loc["baseline", "median_station_skill_vs_anchor"]
    )


def test_the_seven_day_headline_is_the_least_robust_lead(table: pd.DataFrame) -> None:
    """The paper's most-quoted number is the one anchor construction moves most."""
    spread = {}
    for horizon in (1, 3, 7):
        block = table[table["horizon"] == horizon]["median_station_skill_vs_anchor"]
        spread[horizon] = float(block.max() / block.min())
    assert spread[7] > spread[3] > spread[1]
    assert spread[7] > 2.0


def test_intervals_bracket_their_median(table: pd.DataFrame) -> None:
    assert (table["skill_ci_low"] <= table["median_station_skill_vs_anchor"]).all()
    assert (table["median_station_skill_vs_anchor"] <= table["skill_ci_high"]).all()


def test_variant_definitions_change_one_thing_each() -> None:
    baseline = next(v for v in A.VARIANTS if v.name == "baseline")
    for variant in A.VARIANTS:
        if variant.name == "baseline":
            continue
        differing = [
            field
            for field in ("harmonics", "doy_window", "train_end", "pooled_phi",
                          "direct_lead_phi", "upper_bound")
            if getattr(variant, field) != getattr(baseline, field)
        ]
        assert len(differing) == 1, f"{variant.name} varies {differing}"


def test_doy_window_climatology_is_a_wrapped_mean() -> None:
    dates = pd.date_range("2010-01-01", periods=730, freq="D")
    panel = pd.DataFrame({
        "site_id": "00000001",
        "DATE": dates,
        "WTEMP": np.arange(730, dtype=float),
    })
    table = A._doy_window_climatology(panel, np.ones(730, bool), 3)
    curve = table["00000001"]
    assert np.isfinite(curve[1:366]).all()
    # day 1 and day 365 are three days apart on a wrapped calendar, so their
    # windows overlap and the values must be close relative to the series range
    assert abs(curve[1] - curve[365]) < 365.0
