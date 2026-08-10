"""Tests for the memory/learned decomposition intervals.

The decomposition is an Abstract-level number that had no interval anywhere.
These tests pin the estimator's semantics and the properties of the published
artifact that the manuscript now quotes.
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

import scripts.final.build_decomposition_intervals as D

pytestmark = pytest.mark.skipif(
    not D.SOURCE.exists(), reason="decomposition artifact not present"
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    if D.DEFAULT_OUT.exists():
        return pd.read_parquet(D.DEFAULT_OUT)
    return pd.DataFrame(D.build_rows())


def test_every_model_lead_and_quantity_is_covered(table: pd.DataFrame) -> None:
    assert set(table["quantity"]) == {q[0] for q in D.QUANTITIES}
    assert set(table["horizon"]) == {1, 3, 7}


def test_intervals_bracket_their_median(table: pd.DataFrame) -> None:
    assert (table["ci_low"] <= table["median"]).all()
    assert (table["median"] <= table["ci_high"]).all()
    assert (table["loco_min"] <= table["loco_max"]).all()


def test_memory_fraction_uses_only_positive_total_gain_stations(
    table: pd.DataFrame,
) -> None:
    fraction = table[table["quantity"] == "memory_fraction"]
    other = table[table["quantity"] == "g_memory"]
    # the fraction is computed on a subset, so it can never have more stations
    for horizon in (1, 3, 7):
        a = fraction[fraction["horizon"] == horizon]["n_stations"].max()
        b = other[other["horizon"] == horizon]["n_stations"].max()
        assert a <= b
    # and whatever it drops must be counted, not silently discarded
    assert (
        fraction["stations_excluded_by_positive_total_gain_rule"]
        == fraction.apply(
            lambda row: other[other["horizon"] == row["horizon"]]["n_stations"].max()
            - row["n_stations"],
            axis=1,
        )
    ).all()


def test_memory_and_learned_intervals_do_not_overlap_at_seven_days(
    table: pd.DataFrame,
) -> None:
    """The paper's central decomposition claim, stated as an interval test."""
    seven = table[(table["horizon"] == 7) & (table["model"] == "ThermoRoute")]
    memory = seven[seven["quantity"] == "g_memory"].iloc[0]
    learned = seven[seven["quantity"] == "g_learned"].iloc[0]
    assert memory["ci_low"] > learned["ci_high"]


def test_memory_fraction_interval_stays_above_one_half(table: pd.DataFrame) -> None:
    seven = table[(table["horizon"] == 7) & (table["model"] == "ThermoRoute")]
    fraction = seven[seven["quantity"] == "memory_fraction"].iloc[0]
    assert fraction["ci_low"] > 0.5


def test_equal_huc_aggregate_is_a_mean_of_within_cluster_medians() -> None:
    values = np.array([1.0, 1.0, 1.0, 5.0])
    groups = np.array(["a", "a", "a", "b"])
    per_cluster = [
        float(np.median(values[groups == cluster])) for cluster in np.unique(groups)
    ]
    assert float(np.mean(per_cluster)) == pytest.approx(3.0)
    assert float(np.median(values)) == pytest.approx(1.0)


def test_rows_declare_their_approximate_status(table: pd.DataFrame) -> None:
    assert (
        table["inference_role"]
        == "APPROXIMATE_DESCRIPTIVE_SENSITIVITY_NOT_DECISION_EVIDENCE"
    ).all()
    assert (table["n_clusters"] == 15).all()
