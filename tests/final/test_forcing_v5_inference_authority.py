"""Tests for the forcing-v5 inference authority.

These cover the estimator semantics (sign conventions, station-first double
differences, equal-cluster aggregation, minimum detectable effect) against
hand-computable fixtures, and the published-artifact invariants that a reader
of the manuscript depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_forcing_v5_inference_authority as INF

AUTHORITY_DIR = INF.DEFAULT_OUT_DIR
SUMMARY = AUTHORITY_DIR / "forcing_v5_inference_summary.json"

pytestmark = pytest.mark.skipif(
    not SUMMARY.exists(),
    reason="forcing-v5 inference authority has not been built in this checkout",
)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def strata() -> pd.DataFrame:
    return pd.read_parquet(AUTHORITY_DIR / "forcing_v5_inference_strata.parquet")


@pytest.fixture(scope="module")
def contrasts() -> pd.DataFrame:
    return pd.read_parquet(AUTHORITY_DIR / "forcing_v5_inference_contrasts.parquet")


# ---------------------------------------------------------------- estimators


def test_rmse_matches_definition() -> None:
    assert INF._rmse(np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))


def test_equal_cluster_effect_is_unweighted_over_clusters() -> None:
    # cluster "a" holds three stations, "b" holds one; the equal-cluster mean
    # must not let "a" dominate the way a station-weighted median would.
    deltas = np.array([-1.0, -1.0, -1.0, -5.0])
    clusters = np.array(["a", "a", "a", "b"])
    assert INF.equal_cluster_effect(deltas, clusters) == pytest.approx(-3.0)
    assert float(np.median(deltas)) == pytest.approx(-1.0)


def test_station_effects_from_pairs_is_per_station_rmse() -> None:
    frame = pd.DataFrame({
        "site_id": ["A", "A", "B"],
        "y_true": [0.0, 0.0, 0.0],
        "y_pred_F0": [1.0, 1.0, 2.0],
        "y_pred_F3": [0.0, 2.0, 1.0],
    })
    out = INF.station_effects_from_pairs(frame).set_index("site_id")
    assert out.loc["A", "rmse_F0"] == pytest.approx(1.0)
    # RMSE over (0, 2) is sqrt(2), not the mean of the two absolute errors
    assert out.loc["A", "rmse_F3_full"] == pytest.approx(np.sqrt(2.0))
    assert out.loc["A", "forcing_value_rmse_F0_minus_F3"] == pytest.approx(
        1.0 - np.sqrt(2.0)
    )
    assert out.loc["B", "n_keys"] == 1


def test_forcing_value_is_exactly_the_negated_delta() -> None:
    deltas = np.array([-0.4, -0.2, -0.3, 0.1, -0.5, -0.25])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    cell = INF.infer_cell(deltas, clusters, with_mde=False)
    assert cell["median_forcing_value_rmse_F0_minus_F3"] == pytest.approx(
        -cell["median_delta_rmse_F3_minus_F0"]
    )
    # the interval must be negated *and* swapped, never negated in place
    assert cell["cluster_bootstrap_ci_low_forcing_value"] == pytest.approx(
        -cell["cluster_bootstrap_ci_high_delta"]
    )
    assert (
        cell["cluster_bootstrap_ci_low_forcing_value"]
        <= cell["cluster_bootstrap_ci_high_forcing_value"]
    )
    assert cell["loco_forcing_value_min"] <= cell["loco_forcing_value_max"]


def test_win_fraction_counts_strictly_better_stations() -> None:
    deltas = np.array([-1.0, 0.0, 1.0, -1.0])
    clusters = np.array(["a", "a", "b", "b"])
    cell = INF.infer_cell(deltas, clusters, with_mde=False)
    assert cell["station_win_fraction_F3_better"] == pytest.approx(0.5)


# ------------------------------------------------- minimum detectable effect


def test_mde_is_smaller_than_a_clearly_resolved_effect() -> None:
    rng = np.random.default_rng(0)
    clusters = np.repeat([f"c{i}" for i in range(12)], 8)
    deltas = -0.5 + 0.02 * rng.standard_normal(len(clusters))
    mde = INF.minimum_detectable_effect(deltas, clusters)
    assert mde["resolvable"] is True
    assert 0.0 < mde["minimum_detectable_effect_degC"] < 0.5
    assert mde["smallest_attainable_p_value"] == pytest.approx(1.0 / 2 ** 12)


def test_mde_refuses_when_the_effect_favours_the_reference() -> None:
    clusters = np.repeat([f"c{i}" for i in range(6)], 4)
    deltas = np.full(len(clusters), 0.3)
    mde = INF.minimum_detectable_effect(deltas, clusters)
    assert mde["resolvable"] is False
    assert mde["minimum_detectable_effect_degC"] is None


def test_mde_refuses_when_the_observed_effect_does_not_clear_alpha() -> None:
    # three clusters give a floor tail of 1/8 = 0.125 > 0.05, so nothing is
    # detectable however large the effect is
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    deltas = np.full(6, -10.0)
    mde = INF.minimum_detectable_effect(deltas, clusters)
    assert mde["resolvable"] is False
    assert mde["smallest_attainable_p_value"] == pytest.approx(0.125)


# ------------------------------------------------------------- published rows


def test_primary_covers_every_model_and_horizon(summary: dict) -> None:
    seen = {(row["model"], row["horizon"]) for row in summary["primary"]}
    assert seen == {
        (model, horizon)
        for model in ("LightGBM", "ResidualLightGBM")
        for horizon in (1, 3, 7)
    }


def test_primary_uses_the_full_reportable_cohort(summary: dict) -> None:
    for row in summary["primary"]:
        assert row["n_stations"] == 116
        assert row["n_clusters"] == 15
        assert row["sign_flip_is_exact"] is True


def test_forcing_value_grows_with_lead(summary: dict) -> None:
    by_model: dict[str, dict[int, float]] = {}
    for row in summary["primary"]:
        by_model.setdefault(row["model"], {})[row["horizon"]] = row[
            "median_forcing_value_rmse_F0_minus_F3"
        ]
    for values in by_model.values():
        assert values[1] < values[3] < values[7]


def test_every_primary_interval_excludes_zero_and_loco_is_stable(summary: dict) -> None:
    for row in summary["primary"]:
        assert row["cluster_bootstrap_ci_low_forcing_value"] > 0.0
        assert row["loco_forcing_value_min"] > 0.0
        assert row["loco_sign_stable"] is True


def test_effect_exceeds_its_minimum_detectable_size(summary: dict) -> None:
    for row in summary["primary"]:
        mde = row["minimum_detectable_effect"]
        assert mde["resolvable"] is True
        assert (
            row["median_forcing_value_rmse_F0_minus_F3"]
            > mde["minimum_detectable_effect_degC"]
        )


def test_authority_declares_its_post_outcome_status(summary: dict) -> None:
    assert summary["authority_status"] == INF.AUTHORITY_STATUS
    assert summary["chronology"]["post_outcome"] is True
    assert summary["chronology"]["confirmatory"] is False
    assert summary["chronology"]["prospectively_registered"] is False
    assert summary["interpretation_limits"]["additive_information_budget"] is False
    assert summary["interpretation_limits"]["cross_experiment_ratio_allowed"] is False
    assert "oracle" in summary["interpretation_limits"]["f3_is"]


def test_strata_are_complete_and_consistently_signed(strata: pd.DataFrame) -> None:
    years = strata[strata["stratum_kind"] == "year"]
    seasons = strata[strata["stratum_kind"] == "season"]
    assert set(years["stratum"]) == {"2021", "2022", "2023"}
    assert set(seasons["stratum"]) == {"DJF", "MAM", "JJA", "SON"}
    # 2 models x 3 horizons x (3 years + 4 seasons)
    assert len(strata) == 2 * 3 * 7
    assert (strata["median_forcing_value_rmse_F0_minus_F3"] > 0.0).all()


def test_no_stratum_was_dropped_without_a_ledger_entry(strata: pd.DataFrame) -> None:
    ledger = pd.read_parquet(
        AUTHORITY_DIR / "forcing_v5_inference_excluded_subgroups.parquet"
    )
    populated = ledger.dropna(subset=["reason"])
    declared = len(strata[strata["interval_status"] != "COMPUTED"])
    # every uncomputed stratum must appear in the ledger; a fully computed run
    # leaves the ledger empty
    assert len(populated) >= declared


def test_lead_interaction_shows_a_plateau_after_three_days(
    contrasts: pd.DataFrame,
) -> None:
    leads = contrasts[contrasts["contrast_kind"] == "lead_interaction"]
    for model in ("LightGBM", "ResidualLightGBM"):
        block = leads[leads["model"] == model].set_index("label")
        early = block.loc["V_F(3d) - V_F(1d)", "median_forcing_value_rmse_F0_minus_F3"]
        late = block.loc["V_F(7d) - V_F(3d)", "median_forcing_value_rmse_F0_minus_F3"]
        assert early > 0.0 and late > 0.0
        assert early > 3.0 * late


def test_model_contrast_is_small_next_to_the_forcing_value(
    contrasts: pd.DataFrame, summary: dict
) -> None:
    model_rows = contrasts[contrasts["contrast_kind"] == "model_contrast"]
    assert len(model_rows) == 3
    largest = model_rows["median_forcing_value_rmse_F0_minus_F3"].abs().max()
    smallest_forcing = min(
        row["median_forcing_value_rmse_F0_minus_F3"] for row in summary["primary"]
    )
    assert largest < 0.25 * smallest_forcing


def test_manifest_binds_every_published_file() -> None:
    manifest = json.loads(
        (AUTHORITY_DIR / "forcing_v5_inference_authority_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    published = {
        path.name
        for path in AUTHORITY_DIR.iterdir()
        if path.name != "forcing_v5_inference_authority_manifest.json"
    }
    assert published == set(manifest["outputs_sha256"])
    for name, digest in manifest["outputs_sha256"].items():
        assert INF._sha256_file(AUTHORITY_DIR / name) == digest
    assert len(manifest["inputs"]["shard_sha256"]) == 12
