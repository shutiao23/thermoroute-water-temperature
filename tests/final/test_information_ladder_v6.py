"""Tests for the corrected L-ladder runner (Phase 3).

The previous ladder died from an unchecked feature-path change. These tests
cover the three things that make this one different: the folds hold whole
regions, every statistic a level forbids is refitted without the held stations,
and the mask-invariance proof can actually fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_information_ladder_v6_observed as L
from thermoroute.information_levels import level


def _registry() -> pd.DataFrame:
    rows = []
    for region, count in (("01", 21), ("02", 18), ("03", 15), ("11", 14),
                          ("17", 16), ("05", 12), ("14", 9), ("18", 15)):
        for index in range(count):
            rows.append({"site_no": f"{region}{index:06d}", "huc2": region})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- folds


def test_folds_partition_the_cohort_exactly() -> None:
    registry = _registry()
    folds = L.whole_region_folds(registry)
    assert len(folds) == L.N_FOLDS
    held = [s for fold in folds for s in fold.held]
    assert sorted(held) == sorted({f"{r:s}" for r in registry["site_no"]})
    assert len(held) == len(set(held)), "a station is held out twice"


def test_a_held_region_is_never_split_across_folds() -> None:
    """Whole-region holdout means whole regions, or it means nothing."""
    registry = _registry()
    folds = L.whole_region_folds(registry)
    owner: dict[str, int] = {}
    for fold in folds:
        for site in fold.held:
            region = site[:2]
            assert owner.setdefault(region, fold.index) == fold.index


def test_in_fold_and_held_are_disjoint_and_complete() -> None:
    registry = _registry()
    everyone = {str(s) for s in registry["site_no"]}
    for fold in L.whole_region_folds(registry):
        assert not set(fold.in_fold) & set(fold.held)
        assert set(fold.in_fold) | set(fold.held) == everyone


def test_the_packing_is_a_pure_function_of_the_registry() -> None:
    registry = _registry()
    first = L.whole_region_folds(registry)
    second = L.whole_region_folds(registry.sample(frac=1.0, random_state=3))
    assert [f.held for f in first] == [f.held for f in second]


def test_folds_are_roughly_balanced() -> None:
    sizes = [len(f.held) for f in L.whole_region_folds(_registry())]
    assert max(sizes) - min(sizes) <= max(sizes) * 0.6


# ------------------------------------------------------------------ levels


def test_only_l0_may_use_the_target_site_s_own_statistics() -> None:
    assert level("L0").pooled_climatology is False
    for name in ("L1", "L2", "L2_U2"):
        assert level(name).pooled_climatology is True


def test_levels_without_water_temperature_have_no_damped_anchor() -> None:
    """A damped anchor needs the issue-date reading, which L2 forbids."""
    for name in ("L0", "L1"):
        assert level(name).water_temperature_visible is True
    for name in ("L2", "L2_U2"):
        assert level(name).water_temperature_visible is False
        assert "climatology_only" in level(name).anchor


def test_perturbation_touches_only_held_stations_and_only_hidden_variables() -> None:
    frame = pd.DataFrame({
        "site_id": ["A"] * 4 + ["B"] * 4,
        "DATE": list(pd.date_range("2010-01-01", periods=4)) * 2,
        "WTEMP": [1.0, 2.0, np.nan, 4.0] * 2,
        "FLOW": [10.0, 20.0, 30.0, 40.0] * 2,
    })
    frame["WTEMP_observed"] = np.isfinite(frame["WTEMP"])
    frame["FLOW_observed"] = np.isfinite(frame["FLOW"])
    fold = L.Fold(index=0, in_fold=("A",), held=("B",), regions_held=("99",))

    out = L.perturb_prohibited_inputs(frame, fold, "L2", seed=1)
    a = out[out["site_id"] == "A"]
    b = out[out["site_id"] == "B"]
    # in-fold station untouched
    np.testing.assert_array_equal(
        a["WTEMP"].to_numpy(), frame[frame.site_id == "A"]["WTEMP"].to_numpy()
    )
    # held station's water temperature moved, missingness preserved
    assert not np.allclose(
        np.nan_to_num(b["WTEMP"].to_numpy()),
        np.nan_to_num(frame[frame.site_id == "B"]["WTEMP"].to_numpy()),
    )
    np.testing.assert_array_equal(
        np.isnan(b["WTEMP"].to_numpy()),
        np.isnan(frame[frame.site_id == "B"]["WTEMP"].to_numpy()),
    )
    # L2 keeps discharge, so discharge must not move anywhere
    np.testing.assert_array_equal(out["FLOW"].to_numpy(), frame["FLOW"].to_numpy())


def test_l2_u2_perturbs_discharge_as_well() -> None:
    frame = pd.DataFrame({
        "site_id": ["A", "A", "B", "B"],
        "DATE": list(pd.date_range("2010-01-01", periods=2)) * 2,
        "WTEMP": [1.0, 2.0, 3.0, 4.0],
        "FLOW": [10.0, 20.0, 30.0, 40.0],
    })
    frame["WTEMP_observed"] = True
    frame["FLOW_observed"] = True
    fold = L.Fold(index=0, in_fold=("A",), held=("B",), regions_held=("99",))
    out = L.perturb_prohibited_inputs(frame, fold, "L2_U2", seed=1)
    held = out[out["site_id"] == "B"]
    assert not np.allclose(held["FLOW"].to_numpy(), [30.0, 40.0])
    np.testing.assert_array_equal(
        out[out.site_id == "A"]["FLOW"].to_numpy(), [10.0, 20.0]
    )


def test_a_level_that_hides_nothing_needs_no_proof() -> None:
    frame = pd.DataFrame({"site_id": ["A"], "DATE": [pd.Timestamp("2010-01-01")]})
    fold = L.Fold(index=0, in_fold=("A",), held=(), regions_held=())
    assert L.perturb_prohibited_inputs(frame, fold, "L0", seed=1) is frame


def test_perturbation_is_deterministic_for_a_seed() -> None:
    frame = pd.DataFrame({
        "site_id": ["B"] * 5,
        "DATE": pd.date_range("2010-01-01", periods=5),
        "WTEMP": [1.0, 2.0, 3.0, 4.0, 5.0],
        "FLOW": [1.0] * 5,
    })
    frame["WTEMP_observed"] = True
    frame["FLOW_observed"] = True
    fold = L.Fold(index=0, in_fold=(), held=("B",), regions_held=())
    a = L.perturb_prohibited_inputs(frame, fold, "L2", seed=7)
    b = L.perturb_prohibited_inputs(frame, fold, "L2", seed=7)
    np.testing.assert_array_equal(a["WTEMP"].to_numpy(), b["WTEMP"].to_numpy())
