from __future__ import annotations

from pathlib import Path
import sys
from itertools import product

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
    cluster_sign_flip_pvalue,
    equivalence_decision,
    holm_adjust,
    noninferiority_decision,
)


def _brute_force_cluster_sign_flip(
    effects: np.ndarray,
    clusters: np.ndarray,
    *,
    null_margin: float,
    statistic: str,
) -> float:
    centered = effects.astype(float) - null_margin
    unique = np.unique(clusters)
    cluster_index = np.searchsorted(unique, clusters)
    reduce = np.median if statistic == "median" else np.mean
    observed = float(reduce(centered))
    simulated = []
    for signs in product((-1.0, 1.0), repeat=len(unique)):
        simulated.append(float(reduce(np.asarray(signs)[cluster_index] * centered)))
    return float(np.mean(np.asarray(simulated) <= observed + 1e-15))


def test_cluster_bootstrap_preserves_negative_improvement_effect():
    effects = np.array([-0.2, -0.1, -0.3, -0.2, -0.1, -0.25])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    result = cluster_bootstrap_paired_effect(effects, clusters, n_boot=500, seed=3)
    assert result["effect"] < 0 and result["ci_high"] < 0
    assert result["n_clusters"] == 3


def test_non_significance_does_not_imply_equivalence():
    assert not equivalence_decision(-0.2, 0.2, margin=0.05)
    assert equivalence_decision(-0.02, 0.03, margin=0.05)
    assert noninferiority_decision(0.04, margin=0.05)
    assert not noninferiority_decision(0.06, margin=0.05)


def test_holm_adjustment_is_monotone_in_ranked_order():
    raw = np.array([0.01, 0.04, 0.02])
    adjusted = holm_adjust(raw)
    assert (adjusted >= raw).all()
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_cluster_sign_flip_detects_consistent_improvement_and_margin() -> None:
    effects = np.array([-0.20, -0.15, -0.10, -0.25, -0.12, -0.18])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    p_superiority = cluster_sign_flip_pvalue(
        effects, clusters, n_randomisations=5000, seed=9)
    p_noninferiority = cluster_sign_flip_pvalue(
        effects + 0.10, clusters, null_margin=0.05,
        n_randomisations=5000, seed=9)
    assert p_superiority < 0.15  # only three independent clusters: coarse test
    assert p_noninferiority < 0.15


def test_exact_cluster_sign_flip_matches_brute_force_for_unbalanced_clusters() -> None:
    effects = np.array([-0.42, -0.31, -0.27, 0.08, 0.11, -0.04, 0.19])
    clusters = np.array(["large", "large", "large", "large", "b", "c", "d"])
    for statistic in ("median", "mean"):
        actual = cluster_sign_flip_pvalue(
            effects,
            clusters,
            null_margin=0.05,
            statistic=statistic,
            exact_max_clusters=20,
        )
        expected = _brute_force_cluster_sign_flip(
            effects, clusters, null_margin=0.05, statistic=statistic
        )
        assert actual == expected


def test_cluster_sensitivity_flags_small_and_dominant_cluster_design() -> None:
    effects = np.array([-0.20] * 8 + [0.08, 0.12])
    clusters = np.array(["dominant"] * 8 + ["b", "c"])
    result = cluster_inference_sensitivity(effects, clusters)
    warnings = set(result["warning_codes"])

    assert result["role"] == (
        "EXPLORATORY_SENSITIVITY_NOT_A_CONFIRMATORY_TEST_OR_DECISION"
    )
    assert result["formal_five_test_decision_rule_modified"] is False
    assert result["estimand_weighting"] == "equal_station"
    assert result["largest_cluster_share"] == 0.8
    assert result["effective_cluster_count_inverse_herfindahl"] < 2.0
    assert "SMALL_CLUSTER_COUNT_LT_30" in warnings
    assert "DOMINANT_CLUSTER_SHARE_AT_OR_ABOVE_THRESHOLD" in warnings
    assert "LOW_EFFECTIVE_CLUSTER_FRACTION" in warnings
    assert result["inference_strength"] == "NO_STRONG_INFERENCE"


def test_cluster_sensitivity_single_cluster_is_not_randomisation_estimable() -> None:
    result = cluster_inference_sensitivity(
        np.array([-0.2, -0.1, 0.1]), np.array(["only", "only", "only"])
    )

    assert result["n_clusters"] == 1
    assert result["randomisation_method"] == "NOT_ESTIMABLE_SINGLE_CLUSTER"
    assert result["sign_flip_p_one_sided_sensitivity"] is None
    assert result["loco_direction"] == "NOT_ESTIMABLE_SINGLE_CLUSTER"
    assert result["inference_strength"] == "NO_STRONG_INFERENCE"
    assert "SINGLE_CLUSTER_RANDOMIZATION_NOT_ESTIMABLE" in result["warning_codes"]


def test_cluster_sensitivity_seeded_monte_carlo_fallback_is_deterministic() -> None:
    effects = np.linspace(-0.8, 1.2, 21)
    clusters = np.array([f"huc-{index:02d}" for index in range(21)])
    kwargs = {
        "exact_max_clusters": 20,
        "n_randomisations": 1000,
        "seed": 8675309,
    }
    first = cluster_inference_sensitivity(effects, clusters, **kwargs)
    second = cluster_inference_sensitivity(effects, clusters, **kwargs)

    assert first["exact_enumeration"] is False
    assert first["randomisation_method"] == (
        "SEEDED_MONTE_CARLO_WHOLE_CLUSTER_SIGN_FLIP"
    )
    assert first["randomisation_draws_evaluated"] == 1000
    assert first["sign_flip_p_one_sided_sensitivity"] == (
        second["sign_flip_p_one_sided_sensitivity"]
    )
    assert first["monte_carlo_standard_error"] == (
        second["monte_carlo_standard_error"]
    )


def test_fifteen_cluster_sensitivity_uses_all_32768_configurations_but_warns() -> None:
    effects = np.linspace(-0.4, 0.1, 15)
    clusters = np.array([f"huc-{index:02d}" for index in range(15)])
    result = cluster_inference_sensitivity(effects, clusters, seed=999)

    assert result["exact_enumeration"] is True
    assert result["randomisation_draws_evaluated"] == 2 ** 15
    assert result["randomisation_seed"] is None
    assert result["monte_carlo_standard_error"] == 0.0
    assert result["effective_cluster_count_inverse_herfindahl"] == 15.0
    assert "SMALL_CLUSTER_COUNT_LT_30" in result["warning_codes"]
    assert result["inference_strength"] == "NO_STRONG_INFERENCE"


def test_cluster_sensitivity_loco_reports_stability_and_crossing() -> None:
    stable = cluster_inference_sensitivity(
        np.array([0.01, 0.02, 0.03]),
        np.array(["a", "b", "c"]),
        null_margin=0.05,
    )
    assert stable["point_effect_minus_null_margin"] < 0.0
    assert stable["loco_effect_minus_null_margin_max"] < 0.0
    assert stable["loco_direction"] == "ALL_LOCO_EFFECTS_BELOW_NULL_MARGIN"
    assert stable["loco_direction_stable"] is True
    assert len(stable["leave_one_cluster_out"]) == 3

    crossing = cluster_inference_sensitivity(
        np.array([-2.0, 0.1, 0.2]), np.array(["a", "b", "c"])
    )
    assert crossing["loco_effect_minus_null_margin_min"] < 0.0
    assert crossing["loco_effect_minus_null_margin_max"] > 0.0
    assert crossing["loco_direction"] == "LOCO_EFFECTS_CROSS_OR_TOUCH_NULL_MARGIN"
    assert crossing["loco_direction_stable"] is False
    assert "LOCO_DIRECTION_CROSSES_OR_TOUCHES_NULL_MARGIN" in crossing["warning_codes"]
