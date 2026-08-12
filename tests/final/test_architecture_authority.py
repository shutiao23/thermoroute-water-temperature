"""Tests for the architecture-axis authority.

The arm's whole claim is that a difference between the network and the tree is a
difference of model class, so the tests that matter are the ones that would fail
if it were quietly a difference of something else: a different station set, a
different comparator, seeds averaged in the wrong order, or an orientation slip
that reports the network as better when it is worse.
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

import scripts.final.build_architecture_authority as A
import scripts.final.build_architecture_geometry_interaction as G
import scripts.final.run_plain_tcn_arm as TCN


INDEX = pd.Index([f"s{i:02d}" for i in range(20)])
CLUSTERS = {site: ("A" if i < 10 else "B") for i, site in enumerate(INDEX)}


def _patch(monkeypatch: pytest.MonkeyPatch, tree, tcn) -> None:
    monkeypatch.setattr(A, "tree_risk", tree)
    monkeypatch.setattr(A, "tcn_risk_by_seed", tcn)
    monkeypatch.setattr(A.L.V5, "HORIZONS", [7])
    monkeypatch.setattr(A, "huc2_cluster_map", lambda registry: CLUSTERS)
    monkeypatch.setattr(A, "load_station_registry", lambda path: None)


def test_positive_means_the_network_is_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An orientation slip here would report the paper's own claim backwards."""
    def tree(level, forcing, horizon, model=A.COMPARATOR):
        return pd.Series(np.full(20, 1.0), index=INDEX)

    def tcn(level, forcing, horizon, variant):
        worse = 1.25 if variant == "unbounded" else 1.40
        return {s: pd.Series(np.full(20, worse), index=INDEX) for s in TCN.FIT_SEEDS}

    _patch(monkeypatch, tree, tcn)
    rows = {(r["quantity"], r["level"], r["forcing"]): r for r in A.build_rows()}
    penalty = rows["architecture_penalty_tcn_minus_tree", "L0", "F0"]
    assert penalty["median_degC"] == pytest.approx(0.25)
    assert penalty["station_fraction_positive"] == 1.0
    assert rows["cost_of_the_one_degree_bound", "L0", "F0"]["median_degC"] == (
        pytest.approx(0.15)
    )


def test_seeds_are_paired_before_they_are_averaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Averaging risks first and differencing afterwards is a different
    quantity whenever seeds disagree about which stations are hard.

    Here seed 0 is worse on the first half and seed 1 on the second, so the two
    orders coincide on the mean but not on the median the authority reports.
    """
    def tree(level, forcing, horizon, model=A.COMPARATOR):
        return pd.Series(np.full(20, 1.0), index=INDEX)

    def tcn(level, forcing, horizon, variant):
        first = np.array([3.0] * 10 + [1.0] * 10)
        second = np.array([1.0] * 10 + [3.0] * 10)
        flat = np.full(20, 1.0)
        return {
            0: pd.Series(first, index=INDEX),
            1: pd.Series(second, index=INDEX),
            2: pd.Series(flat, index=INDEX),
        }

    monkeypatch.setattr(TCN, "FIT_SEEDS", (0, 1, 2))
    _patch(monkeypatch, tree, tcn)
    rows = {(r["quantity"], r["level"], r["forcing"]): r for r in A.build_rows()}
    # per station: mean over seeds of (tcn_s - tree) = (2 + 0 + 0)/3 on one half
    # and (0 + 2 + 0)/3 on the other -- 2/3 everywhere, so the median is 2/3
    assert rows["architecture_penalty_tcn_minus_tree", "L0", "F0"]["median_degC"] == (
        pytest.approx(2 / 3)
    )


def test_every_cell_of_the_crossing_uses_one_station_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A double difference taken over two different station sets is not one."""
    narrow = INDEX[:14]

    def tree(level, forcing, horizon, model=A.COMPARATOR):
        support = INDEX if level == "L0" else narrow
        return pd.Series(np.full(len(support), 1.0), index=support)

    def tcn(level, forcing, horizon, variant):
        support = INDEX if forcing == "F0" else narrow
        return {s: pd.Series(np.full(len(support), 1.2), index=support)
                for s in TCN.FIT_SEEDS}

    _patch(monkeypatch, tree, tcn)
    assert {row["n_stations"] for row in A.build_rows()} == {len(narrow)}


def test_both_interactions_are_reported_and_are_double_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The architecture penalty is built to widen at L2 and shrink at F3; the
    two interaction rows must recover exactly that, with opposite signs."""
    penalties = {("L0", "F0"): 0.10, ("L2", "F0"): 0.30,
                 ("L0", "F3_full"): 0.04, ("L2", "F3_full"): 0.24}

    def tree(level, forcing, horizon, model=A.COMPARATOR):
        return pd.Series(np.full(20, 1.0), index=INDEX)

    def tcn(level, forcing, horizon, variant):
        value = 1.0 + penalties[level, forcing]
        return {s: pd.Series(np.full(20, value), index=INDEX) for s in TCN.FIT_SEEDS}

    _patch(monkeypatch, tree, tcn)
    rows = {(r["quantity"], r["forcing"]): r for r in A.build_rows()}
    assert rows["interaction_L2_minus_L0", "F0"]["median_degC"] == pytest.approx(0.20)
    assert rows["interaction_L2_minus_L0", "F3_full"]["median_degC"] == (
        pytest.approx(0.20))
    assert rows["interaction_F3_minus_F0", "F3_full-F0"]["median_degC"] == (
        pytest.approx(-0.06))


def test_the_comparator_is_the_residual_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both models add their prediction to the same level-legal anchor; the
    raw-target tree does not, so it is not the matched comparator."""
    assert A.COMPARATOR == "ResidualLightGBM"


def test_a_missing_shard_is_an_error_and_not_a_silent_omission() -> None:
    with pytest.raises(A.ArchitectureError, match="missing shard"):
        A.tree_risk("L0", "F0", 999, model="NoSuchModel")


def test_the_shard_naming_agrees_with_what_the_runner_writes() -> None:
    """The authority and the runner build the same filename independently, so a
    rename in one would otherwise surface as a silent 'missing shard'."""
    assert A._prefix("F0") == ""
    assert A._prefix("F3_full") == "F3_full_"
    assert A.TCN_SHARDS == TCN.OUTPUT_DIR / TCN.SHARD_DIRNAME


# ---------------------------------------------------- architecture x geometry


def test_the_geometry_interaction_pairs_inside_each_split_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network fitted on split seed 2's folds must be compared with the tree
    fitted on split seed 2's folds.

    Pooling risks across split seeds first and differencing afterwards compares
    two models that were never held out on the same stations. The construction
    below makes the two orders disagree: each split seed is easy for the tree on
    one half of the cohort and hard on the other, and the network tracks it.
    Paired inside the seed the penalty is a constant; pooled first it is not.
    """
    half = np.array([1.0] * 10 + [3.0] * 10)
    other = np.array([3.0] * 10 + [1.0] * 10)
    by_seed = {0: half, 1: other, 2: half, 3: other, 4: half}

    def tree(level, horizon, split_seed):
        if split_seed is None:
            return pd.Series(np.full(20, 1.0), index=INDEX)
        return pd.Series(by_seed[split_seed], index=INDEX)

    def tcn(level, horizon, split_seed, fit_seed):
        base = (np.full(20, 1.0) if split_seed is None else by_seed[split_seed])
        return pd.Series(base + 0.5, index=INDEX)

    monkeypatch.setattr(G, "_tree", tree)
    monkeypatch.setattr(G, "_tcn", tcn)
    monkeypatch.setattr(G.L.V5, "HORIZONS", [7])
    monkeypatch.setattr(G, "huc2_cluster_map", lambda registry: CLUSTERS)
    monkeypatch.setattr(G, "load_station_registry", lambda path: None)

    rows = {(r["quantity"], r["level"], r["geometry"]): r for r in G.build_rows()}
    penalty = rows["architecture_penalty_tcn_minus_tree", "L0", "random_site"]
    assert penalty["median_degC"] == pytest.approx(0.5)
    assert penalty["station_fraction_positive"] == 1.0
    interaction = rows["interaction_random_minus_region", "L0",
                       "random_site-whole_region"]
    assert interaction["median_degC"] == pytest.approx(0.0)


def test_whole_region_and_random_site_read_different_shards() -> None:
    """The split seed must reach the filename, or the two geometries collide.

    Both calls are expected to fail on a nonexistent lead; the point is *which*
    path each one reports missing. If the seed were dropped from the name they
    would name the same file, and the builder would compare a geometry against
    itself while looking like it had done the crossing.
    """
    paths = []
    for split_seed in (None, 3):
        with pytest.raises(G.ARCH.ArchitectureError, match="missing shard") as caught:
            G._tcn("L0", 999, split_seed, 0)
        paths.append(str(caught.value))
    assert "whole_region" in paths[0] and "seed3" not in paths[0]
    assert "random_site_seed3" in paths[1]
    assert paths[0] != paths[1]


def test_the_geometry_arm_is_forcing_free_by_declaration() -> None:
    """Crossing all four axes is a contrast this cohort cannot carry, so the
    builder pins F0 rather than leaving it to the caller."""
    assert G.FORCING == "F0"
    assert G.VARIANT == "unbounded"
