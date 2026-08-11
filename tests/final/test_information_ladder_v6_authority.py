"""Tests for the corrected L-ladder result authority."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_information_ladder_v6_authority as A

SUMMARY = A.DEFAULT_OUT / "information_ladder_v6_summary.json"
pytestmark = pytest.mark.skipif(
    not SUMMARY.exists(), reason="ladder authority has not been built"
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_parquet(A.DEFAULT_OUT / "information_ladder_v6_contrasts.parquet")


def test_every_contrast_model_and_lead_is_present(table: pd.DataFrame) -> None:
    assert set(table["contrast"]) == {f"{h}-{low}" for h, low, _ in A.CONTRASTS}
    assert len(table) == len(A.CONTRASTS) * 2 * 3
    assert (table["n_stations"] == 116).all()


def test_losing_local_thermal_state_is_costly_everywhere(table: pd.DataFrame) -> None:
    total = table[table["contrast"] == "L2-L0"]
    assert (total["median_value_degC"] > 1.0).all()
    assert (total["ci_low"] > 0).all()
    assert (total["station_fraction_positive"] == 1.0).all()
    assert total["loco_sign_stable"].all()


def test_the_recent_sequence_dominates_the_long_term_statistics(
    table: pd.DataFrame,
) -> None:
    """Pooling a station's climatology costs little; losing its readings costs everything."""
    sequence = table[table["contrast"] == "L2-L1"].set_index(["model", "horizon"])
    statistics = table[table["contrast"] == "L1-L0"].set_index(["model", "horizon"])
    for key in sequence.index:
        assert (
            sequence.loc[key, "median_value_degC"]
            > 5.0 * statistics.loc[key, "median_value_degC"]
        )


def test_discharge_adds_nothing_once_thermal_history_is_gone(
    table: pd.DataFrame,
) -> None:
    flow = table[table["contrast"] == "L2_U2-L2"]
    # every interval must cover zero, and stations must split near evenly
    assert ((flow["ci_low"] <= 0) & (flow["ci_high"] >= 0)).all()
    assert flow["station_fraction_positive"].between(0.4, 0.65).all()


def test_the_nested_rungs_are_ordered(table: pd.DataFrame) -> None:
    """Each withheld rung must be at least as bad as the one above it."""
    for _, row in table[table["contrast"].isin(["L2-L1", "L2-L0"])].iterrows():
        assert row["median_rmse_high"] > row["median_rmse_low"]


def test_the_decomposition_is_consistent(table: pd.DataFrame) -> None:
    """L2-L0 must sit near (L1-L0) + (L2-L1); medians do not add exactly."""
    for model in ("LightGBM", "ResidualLightGBM"):
        for horizon in (1, 3, 7):
            cell = table[(table["model"] == model)
                         & (table["horizon"] == horizon)].set_index("contrast")
            parts = (cell.loc["L1-L0", "median_value_degC"]
                     + cell.loc["L2-L1", "median_value_degC"])
            assert abs(cell.loc["L2-L0", "median_value_degC"] - parts) < 0.25


def test_intervals_bracket_their_median(table: pd.DataFrame) -> None:
    assert (table["ci_low"] <= table["median_value_degC"]).all()
    assert (table["median_value_degC"] <= table["ci_high"]).all()


def test_the_authority_forbids_adding_the_axes() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert "never" in summary["non_additivity"]
    assert summary["chronology"]["confirmatory"] is False
    assert "432-cell" in summary["chronology"]["replaces"]
