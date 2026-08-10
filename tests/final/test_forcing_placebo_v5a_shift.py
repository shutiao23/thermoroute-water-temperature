"""Tests for the +/-7-day time-shifted forcing arms (protocol v5a, rule P4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from test_forcing_placebo_v5a_shuffle import _panel

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shift as SH

MET = list(V5.METEOROLOGY_VARIABLES)


@pytest.mark.parametrize("days", [7, -7])
def test_each_row_receives_the_meteorology_of_the_displaced_day(days: int) -> None:
    true = _panel(n_sites=2, days=120)
    out = SH.shift_meteorology(true, days=days)
    a = true[true["site_id"] == "00000001"].reset_index(drop=True)
    b = out[out["site_id"] == "00000001"].reset_index(drop=True)
    if days > 0:
        np.testing.assert_allclose(b.loc[0, MET].to_numpy(float),
                                   a.loc[days, MET].to_numpy(float))
        assert b[MET].tail(days).isna().all().all()
    else:
        np.testing.assert_allclose(b.loc[-days, MET].to_numpy(float),
                                   a.loc[0, MET].to_numpy(float))
        assert b[MET].head(-days).isna().all().all()


@pytest.mark.parametrize("days", [7, -7])
def test_identity_water_temperature_and_flow_are_untouched(days: int) -> None:
    true = _panel()
    out = SH.shift_meteorology(true, days=days)
    for column in ("site_id", "DATE", "WTEMP", "FLOW"):
        np.testing.assert_array_equal(true[column].to_numpy(), out[column].to_numpy())


@pytest.mark.parametrize("days", [7, -7])
def test_observedness_follows_the_shifted_values(days: int) -> None:
    out = SH.shift_meteorology(_panel(), days=days)
    for variable in MET:
        np.testing.assert_array_equal(
            out[f"{variable}_observed"].to_numpy(),
            np.isfinite(out[variable].to_numpy()),
        )
    V5._validate_raw_frame(out)


def test_a_zero_shift_is_refused() -> None:
    with pytest.raises(SH.ShiftError, match="not a control"):
        SH.shift_meteorology(_panel(), days=0)


def test_a_broken_daily_calendar_is_refused() -> None:
    """A positional shift silently mis-dates the donor if a day is missing."""
    true = _panel(n_sites=1, days=60).drop(index=10).reset_index(drop=True)
    with pytest.raises(SH.ShiftError, match="strict daily calendar"):
        SH.shift_meteorology(true, days=7)


@pytest.mark.parametrize("days", [7, -7])
def test_the_ledger_accounts_for_every_row(days: int) -> None:
    true = _panel(n_sites=3, days=200)
    out = SH.shift_meteorology(true, days=days)
    ledger = out.attrs["shift_ledger"]
    assert ledger["rows_with_a_donor"] + ledger["rows_left_without_a_donor"] == len(true)
    assert ledger["rows_left_without_a_donor"] == 3 * abs(days)


def test_both_arms_are_declared_with_the_primary_named() -> None:
    assert set(SH.SHIFT_ARMS) == {"F3_shift_plus7", "F3_shift_minus7"}
    assert SH.SHIFT_ARMS["F3_shift_plus7"] == 7
    assert SH.SHIFT_ARMS["F3_shift_minus7"] == -7


def test_runner_passes_the_required_precommit_check() -> None:
    source = Path(SH.__file__).read_text(encoding="utf-8")
    assert "precommit_check=" in source
