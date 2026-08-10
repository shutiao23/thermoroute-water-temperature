"""Tests for the shift-arm scoring authority and sealed rule P4."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_forcing_shift_authority as A
import scripts.final.run_forcing_placebo_v5a_shift as SH

SUMMARY = A.DEFAULT_OUT / "forcing_shift_summary.json"
pytestmark = pytest.mark.skipif(
    not SUMMARY.exists(), reason="shift authority has not been built"
)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_parquet(A.DEFAULT_OUT / "forcing_shift_rows.parquet")


def test_both_arms_at_every_model_and_lead(table: pd.DataFrame) -> None:
    assert set(table["arm"]) == set(SH.SHIFT_ARMS)
    assert len(table) == 2 * 2 * 3


def test_only_the_plus_arm_is_the_primary_timing_control(table: pd.DataFrame) -> None:
    primary = table[table["is_primary_timing_control"]]
    assert set(primary["arm"]) == {"F3_shift_plus7"}
    assert A.PRIMARY_ARM == "F3_shift_plus7"


def test_the_common_shift_registry_is_smaller_than_the_primary_one(
    table: pd.DataFrame,
) -> None:
    """The boundary rule must actually drop keys, or it was not applied."""
    assert (table["n_common_shift_keys"] > 0).all()
    assert (table["n_common_shift_keys"] < 358_765).all()


def test_displacing_the_future_by_a_week_destroys_the_forcing_value(
    table: pd.DataFrame,
) -> None:
    assert (table["retained_fraction"] < A.P4_MAX_RETAINED).all()
    assert (table["station_win_fraction_shift_worse"] > 0.85).all()


def test_p4_is_satisfied_and_recorded(summary: dict) -> None:
    decision = summary["decision"]
    assert decision["rule"] == "P4_timing_specificity"
    assert decision["satisfied"] is True
    assert decision["threshold"] == 0.25
    assert "event timing" in decision["meaning"]


def test_p4_would_have_been_reachable_in_the_negative() -> None:
    """A control whose failure is unreachable is not a control."""
    rows = [{"is_primary_timing_control": True, "horizon": 7,
             "retained_fraction": 0.80}]
    assert A.verdict(rows)["satisfied"] is False
    assert "withdrawn" in A.verdict(rows)["meaning"]


def test_the_two_arms_are_never_averaged(summary: dict) -> None:
    assert "weaker_control" in summary
    arms = {row["arm"] for row in summary["rows"]}
    assert arms == set(SH.SHIFT_ARMS)


def test_chronology_declares_the_pre_outcome_seal(summary: dict) -> None:
    assert summary["chronology"]["specification_sealed_before_outcome"] is True
    assert summary["chronology"]["post_outcome"] is False
