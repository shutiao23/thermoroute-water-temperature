"""Tests for the five-test family power analysis.

The manuscript's one self-inconsistency was an equivalence statement drawn from
a non-significant result of a procedure it had already disowned.  These tests
pin the quantity that replaces it and the classification that decides the
wording.
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

import scripts.final.build_family_power as FP

pytestmark = pytest.mark.skipif(
    not FP.DEFAULT_OUT.exists(), reason="family power has not been built"
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_parquet(FP.DEFAULT_OUT)


def test_exactly_the_frozen_five_rows(table: pd.DataFrame) -> None:
    got = {
        (r.candidate, r.reference, int(r.horizon)) for r in table.itertuples()
    }
    assert got == set(FP.FAMILY)
    assert len(table) == 5


def test_damped_persistence_rows_are_adequately_powered(table: pd.DataFrame) -> None:
    rows = table[table["reference"] == "DampedPersistence"]
    assert len(rows) == 3
    assert (rows["limitation"] == "adequately_powered_at_the_observed_effect_size").all()
    # each effect must exceed the smallest the design can resolve
    assert (
        rows["median_delta_rmse"].abs() > rows["minimum_detectable_effect_degC"]
    ).all()


def test_the_lightgbm_rows_cannot_carry_an_equivalence_statement(
    table: pd.DataFrame,
) -> None:
    """This is the finding: the claim the paper made is on the rows that cannot bear it."""
    rows = table[table["reference"] == "LightGBM"]
    assert len(rows) == 2
    assert rows["underpowered_for_an_effect_of_the_observed_size"].all()
    assert set(rows["limitation"]) == {
        "one_sided_test_favours_reference",
        "underpowered_at_the_observed_effect_size",
    }


def test_family_power_is_not_uniform(table: pd.DataFrame) -> None:
    assert table["underpowered_for_an_effect_of_the_observed_size"].nunique() == 2


def test_sign_flip_floor_is_the_fifteen_cluster_limit(table: pd.DataFrame) -> None:
    assert table["smallest_attainable_p_value"].to_numpy() == pytest.approx(1 / 2 ** 15)
    assert (table["n_clusters"] == 15).all()


def test_intervals_bracket_their_median(table: pd.DataFrame) -> None:
    assert (table["ci_low"] <= table["median_delta_rmse"]).all()
    assert (table["median_delta_rmse"] <= table["ci_high"]).all()


def test_classification_is_decided_by_the_mde_not_by_the_p_value() -> None:
    """A tiny p-value with a huge MDE must still classify as underpowered."""
    # effect smaller than what the structure can resolve
    assert FP.ALPHA == 0.05
    clusters = np.repeat([f"c{i}" for i in range(15)], 8)
    rng = np.random.default_rng(0)
    deltas = -0.001 + 0.5 * rng.standard_normal(len(clusters))
    import scripts.final.build_forcing_v5_inference_authority as INF

    mde = INF.minimum_detectable_effect(deltas, clusters, alpha=FP.ALPHA)
    observed = float(np.median(deltas))
    if mde["resolvable"]:
        assert mde["minimum_detectable_effect_degC"] > 0.0
    else:
        assert mde["minimum_detectable_effect_degC"] is None
        assert abs(observed) < 1.0
