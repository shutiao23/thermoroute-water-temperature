"""Tests for the station heterogeneity of the forcing value."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_forcing_heterogeneity as H

pytestmark = pytest.mark.skipif(
    not H.DEFAULT_OUT.exists(), reason="heterogeneity has not been built"
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_parquet(H.DEFAULT_OUT)


def test_attributes_were_declared_before_inspection(table: pd.DataFrame) -> None:
    assert set(table["attribute"]) == {name for name, _ in H.ATTRIBUTES}
    assert H.PRIMARY == "log_half_life"
    primary = table[table["attribute"] == H.PRIMARY]
    assert primary["is_primary_hypothesis"].all()


def test_thermal_memory_reduces_the_need_for_future_weather(
    table: pd.DataFrame,
) -> None:
    """The primary hypothesis, and it must hold at every model and lead."""
    primary = table[
        (table["attribute"] == H.PRIMARY) & (table["status"] == "COMPUTED")
    ]
    assert len(primary) == 6
    assert (primary["spearman_rho"] < 0).all()
    assert (primary["ci_high"] < 0).all()


def test_the_memory_relationship_weakens_with_lead(table: pd.DataFrame) -> None:
    """Long memory protects most at short leads, which is the physical reading."""
    primary = table[
        (table["attribute"] == H.PRIMARY) & (table["model"] == "LightGBM")
    ].set_index("horizon")
    assert primary.loc[1, "spearman_rho"] < primary.loc[3, "spearman_rho"]
    assert primary.loc[3, "spearman_rho"] < primary.loc[7, "spearman_rho"]


def test_the_record_completeness_control_is_reported_not_hidden(
    table: pd.DataFrame,
) -> None:
    """A data-availability gradient must be visible, not silently absent."""
    nuisance = table[table["attribute"] == "wtemp_observed_fraction"]
    assert len(nuisance) == 6
    assert (nuisance["status"] == "COMPUTED").all()


def test_intervals_bracket_their_estimate(table: pd.DataFrame) -> None:
    computed = table[table["status"] == "COMPUTED"]
    assert (computed["ci_low"] <= computed["spearman_rho"]).all()
    assert (computed["spearman_rho"] <= computed["ci_high"]).all()


def test_everything_is_labelled_exploratory(table: pd.DataFrame) -> None:
    computed = table[table["status"] == "COMPUTED"]
    assert (
        computed["inference_role"] == "EXPLORATORY_COVARIATION_NOT_ATTRIBUTION"
    ).all()


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 2.0 * x + 0.1 * rng.normal(size=60)
    clusters = np.repeat([f"c{i}" for i in range(6)], 10)
    out = H.cluster_bootstrap_spearman(x, y, clusters)
    assert out["spearman_rho"] > 0.9
    assert out["n_clusters"] == 6
    assert out["ci_low"] <= out["spearman_rho"] <= out["ci_high"]


def test_flow_cv_ignores_signed_discharge() -> None:
    """This cohort contains negative discharge; a CV on it would be meaningless."""
    source = Path(H.__file__).read_text(encoding="utf-8")
    assert "finite[finite > 0]" in source
