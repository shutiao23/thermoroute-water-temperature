"""Tests for the forcing component arms (protocol v5a)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_components_v5a as CP
import scripts.final.run_forcing_ladder_v5_observed as V5


class _Clim:
    """Constant climatology, so a substituted value is unmistakable."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_dates(self, station: str, dates: pd.Series) -> np.ndarray:
        return np.full(len(dates), self.value, dtype=float)


def _frame() -> pd.DataFrame:
    rows = 6
    frame = pd.DataFrame({
        "site_id": ["A"] * rows,
        "DATE": pd.date_range("2010-01-01", periods=rows),
        "WTEMP": np.arange(rows, dtype=float),
        "FLOW": np.arange(rows, dtype=float) + 100.0,
    })
    for index, variable in enumerate(V5.METEOROLOGY_VARIABLES):
        frame[variable] = np.arange(rows, dtype=float) + 10.0 * index
    for variable in ("WTEMP", "FLOW", *V5.METEOROLOGY_VARIABLES):
        frame[f"{variable}_observed"] = np.isfinite(frame[variable])
    return frame


def _climatologies() -> dict[str, _Clim]:
    return {v: _Clim(-999.0) for v in V5.METEOROLOGY_VARIABLES}


def test_both_arm_families_cover_every_group() -> None:
    names = [name for name, _ in CP.arms()]
    assert len(names) == 8
    for group in CP.GROUPS:
        assert f"only_{group}" in names
        assert f"without_{group}" in names


def test_only_and_without_are_complements() -> None:
    by_name = dict(CP.arms())
    for group, members in CP.GROUPS.items():
        assert set(by_name[f"only_{group}"]) == set(members)
        assert set(by_name[f"without_{group}"]) == set(V5.METEOROLOGY_VARIABLES) - set(members)


def test_humidity_and_wind_stay_one_group() -> None:
    assert set(CP.GROUPS["humidity_wind"]) == {"RHMEAN", "WDSP"}


def test_selected_variables_keep_their_realized_values() -> None:
    frame = _frame()
    out = CP.climatological_panel(frame, _climatologies(), ("TEMP",))
    np.testing.assert_array_equal(out["TEMP"].to_numpy(), frame["TEMP"].to_numpy())


def test_unselected_variables_become_climatological() -> None:
    frame = _frame()
    out = CP.climatological_panel(frame, _climatologies(), ("TEMP",))
    for variable in V5.METEOROLOGY_VARIABLES:
        if variable == "TEMP":
            continue
        assert (out[variable].to_numpy() == -999.0).all(), variable


def test_water_temperature_discharge_and_identity_are_untouched() -> None:
    frame = _frame()
    out = CP.climatological_panel(frame, _climatologies(), ("DH",))
    for column in ("site_id", "DATE", "WTEMP", "FLOW"):
        np.testing.assert_array_equal(out[column].to_numpy(), frame[column].to_numpy())


def test_observedness_follows_the_substituted_values() -> None:
    frame = _frame()
    out = CP.climatological_panel(frame, _climatologies(), ("TEMP",))
    for variable in V5.METEOROLOGY_VARIABLES:
        np.testing.assert_array_equal(
            out[f"{variable}_observed"].to_numpy(),
            np.isfinite(out[variable].to_numpy()),
        )


def test_the_column_namespace_never_changes_between_arms() -> None:
    """A component arm must not confound 'which variable' with 'how many columns'."""
    frame = _frame()
    a = CP.climatological_panel(frame, _climatologies(), ("TEMP",))
    b = CP.climatological_panel(frame, _climatologies(), ("PRCP",))
    assert list(a.columns) == list(b.columns) == list(frame.columns)


def test_an_unknown_variable_is_refused() -> None:
    with pytest.raises(CP.ComponentError, match="unknown meteorological"):
        CP.climatological_panel(_frame(), _climatologies(), ("NOPE",))
