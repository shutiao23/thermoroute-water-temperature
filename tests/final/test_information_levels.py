"""Tests for the local-information level contract.

The previous ladder was destroyed by a feature-path change nobody checked
against the actual training rows. These tests check the level definition
against the real 210-column frozen namespace, not a fixture, because the
failure mode is a column nobody remembered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5
from thermoroute.information_levels import (
    LEVELS,
    admissible_columns,
    level,
    prohibited_columns,
)

COLUMNS = V5.FROZEN_BASE_FEATURE_COLUMNS


def test_the_frozen_namespace_is_the_one_we_reason_about() -> None:
    assert len(COLUMNS) == 210


def test_l0_forbids_nothing() -> None:
    assert prohibited_columns("L0", COLUMNS) == ()
    assert admissible_columns("L0", COLUMNS) == COLUMNS


def test_l1_forbids_no_column_but_pools_the_climatology() -> None:
    """L1 is a statistics change, not a column change."""
    assert prohibited_columns("L1", COLUMNS) == ()
    assert level("L1").pooled_climatology is True
    assert level("L0").pooled_climatology is False


def test_l2_removes_every_water_temperature_derived_column() -> None:
    kept = admissible_columns("L2", COLUMNS)
    assert not [c for c in kept if c.startswith("WTEMP_")]
    # the two that are water-temperature-derived without saying so
    assert "clim_anom" not in kept
    assert "persistence" not in kept


def test_l2_keeps_discharge_and_meteorology() -> None:
    kept = admissible_columns("L2", COLUMNS)
    assert [c for c in kept if c.startswith("FLOW_")]
    for variable in ("TEMP_", "PRCP_", "RHMEAN_", "DH_", "WDSP_"):
        assert [c for c in kept if c.startswith(variable)], variable
    # the seasonal calendar and the pooled climatology remain legal
    assert "clim_t" in kept and "clim_target" in kept
    assert "doy_h0" in kept


def test_l2_u2_additionally_removes_discharge() -> None:
    kept = admissible_columns("L2_U2", COLUMNS)
    assert not [c for c in kept if c.startswith("FLOW_")]
    assert not [c for c in kept if c.startswith("WTEMP_")]
    # meteorology is not a target-site observation and survives
    assert [c for c in kept if c.startswith("TEMP_")]


def test_the_levels_are_strictly_nested() -> None:
    """Each rung must admit a subset of the rung above it."""
    l0 = set(admissible_columns("L0", COLUMNS))
    l1 = set(admissible_columns("L1", COLUMNS))
    l2 = set(admissible_columns("L2", COLUMNS))
    u2 = set(admissible_columns("L2_U2", COLUMNS))
    assert u2 < l2 < l1
    assert l1 == l0


def test_no_observedness_flag_of_a_prohibited_variable_survives() -> None:
    """A mask flag leaks the very thing the level is meant to hide."""
    for name in ("L2", "L2_U2"):
        kept = admissible_columns(name, COLUMNS)
        leaked = [
            c for c in kept
            if "observed" in c and (
                c.startswith("WTEMP_")
                or (name == "L2_U2" and c.startswith("FLOW_"))
            )
        ]
        assert leaked == [], leaked


def test_prohibited_and_admissible_partition_the_namespace() -> None:
    for name in LEVELS:
        kept = admissible_columns(name, COLUMNS)
        dropped = prohibited_columns(name, COLUMNS)
        assert len(kept) + len(dropped) == len(COLUMNS)
        assert not set(kept) & set(dropped)


def test_column_order_is_preserved() -> None:
    kept = admissible_columns("L2", COLUMNS)
    assert list(kept) == [c for c in COLUMNS if c in set(kept)]


def test_an_unknown_level_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown information level"):
        level("L9")


def test_counts_are_what_the_namespace_implies() -> None:
    """Regression guard: a namespace change must move these numbers visibly."""
    assert len(admissible_columns("L0", COLUMNS)) == 210
    assert len(admissible_columns("L2", COLUMNS)) == 210 - 34
    assert len(admissible_columns("L2_U2", COLUMNS)) == 210 - 34 - 28
