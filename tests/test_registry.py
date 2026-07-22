"""Fast contract tests for the canonical forecast-key registry."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np
import pytest

from thermoroute.registry import (
    FORECAST_KEY,
    STAGE9_PRIMARY_MODELS,
    enforce_common_forecast_keys,
    restrict_tabular_to_window_registry,
    targets_match_at_model_precision,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry


def _row(model, issue, target, y, *, seed=0, split="test"):
    return {
        "model": model, "split": split, "site_id": "s1", "horizon": 1,
        "issue_date": issue, "target_date": target, "y_true": y,
        "y_pred": y, "seed": seed,
    }


def test_registry_intersects_exact_keys_and_preserves_seed_rows():
    rows = [
        _row("A", "2020-01-01", "2020-01-02", 1.0, seed=0),
        _row("A", "2020-01-02", "2020-01-03", 2.0, seed=0),
        _row("A", "2020-01-02", "2020-01-03", 2.0, seed=1),
        _row("B", "2020-01-02", "2020-01-03", 2.0),
        _row("B", "2020-01-03", "2020-01-04", 3.0),
        # Non-comparable rows and other splits must survive untouched.
        _row("diagnostic", "2020-02-01", "2020-02-02", 9.0),
        _row("A", "2019-01-01", "2019-01-02", 4.0, split="val"),
    ]
    aligned, audit = enforce_common_forecast_keys(pd.DataFrame(rows), ("A", "B"))

    kept = aligned[(aligned["split"] == "test") & aligned["model"].isin(["A", "B"])]
    assert len(kept[kept["model"] == "A"]) == 2  # both seeds retained
    assert len(kept[kept["model"] == "B"]) == 1
    assert kept["issue_date"].nunique() == 1
    assert kept["target_date"].nunique() == 1
    assert audit.common_unique == 1
    assert audit.before_unique == {"A": 2, "B": 2}
    assert len(aligned[aligned["model"] == "diagnostic"]) == 1
    assert len(aligned[aligned["split"] == "val"]) == 1


def test_target_date_is_part_of_the_key():
    df = pd.DataFrame([
        _row("A", "2020-01-01", "2020-01-02", 1.0),
        _row("B", "2020-01-01", "2020-01-03", 1.0),
    ])
    with pytest.raises(ValueError, match="no common forecast keys"):
        enforce_common_forecast_keys(df, ("A", "B"))
    assert "target_date" in FORECAST_KEY


def test_registry_rejects_disagreeing_truth_for_same_key():
    df = pd.DataFrame([
        _row("A", "2020-01-01", "2020-01-02", 1.0),
        _row("B", "2020-01-01", "2020-01-02", 1.1),
    ])
    with pytest.raises(ValueError, match="disagree on y_true"):
        enforce_common_forecast_keys(df, ("A", "B"), truth_atol=1e-6)


def test_canonical_hot_target_has_identical_float32_truth_semantics():
    """Regression: canonical >32 C labels legitimately quantise by >1e-6."""
    panel_path = Path(__file__).resolve().parents[1] / "data_usgs" / "panel_usgs_120v2.parquet"
    panel = pd.read_parquet(panel_path, columns=["WTEMP"])
    values = panel.WTEMP.dropna().to_numpy(float)
    roundtrip = values.astype(np.float32).astype(float)
    positions = np.flatnonzero(np.abs(values - roundtrip) > 1e-6)
    assert len(positions) > 0, "canonical regression fixture no longer exercises float32 drift"
    truth64 = float(values[positions[0]])
    truth32 = np.float32(truth64)

    frame = pd.DataFrame([
        _row("A", "2020-01-01", "2020-01-02", truth64),
        _row("B", "2020-01-01", "2020-01-02", truth32),
    ])
    aligned, audit = enforce_common_forecast_keys(frame, ("A", "B"))
    assert len(aligned) == 2
    assert audit.common_unique == 1


def test_derived_parent_truth_uses_exact_float32_semantics():
    truth64 = np.array([32.1], dtype=np.float64)
    truth32 = truth64.astype(np.float32)
    assert targets_match_at_model_precision(truth64, truth32)
    different = np.nextafter(truth32, np.float32(np.inf), dtype=np.float32)
    assert not targets_match_at_model_precision(truth64, different)
    assert not targets_match_at_model_precision(truth64, np.array([np.nan]))


@pytest.mark.parametrize("value", [0.25, 32.1])
def test_registry_rejects_one_float32_ulp_truth_difference(value):
    truth = np.float32(value)
    different = np.nextafter(truth, np.float32(np.inf), dtype=np.float32)
    assert different != truth
    frame = pd.DataFrame([
        _row("A", "2020-01-01", "2020-01-02", truth),
        _row("B", "2020-01-01", "2020-01-02", different),
    ])
    with pytest.raises(ValueError, match="disagree on y_true"):
        enforce_common_forecast_keys(frame, ("A", "B"))


def test_registry_fails_when_a_required_model_is_absent():
    df = pd.DataFrame([_row("A", "2020-01-01", "2020-01-02", 1.0)])
    with pytest.raises(ValueError, match="models absent"):
        enforce_common_forecast_keys(df, ("A", "B"))


def test_optional_model_cannot_shrink_fixed_primary_registry():
    rows = []
    for model in STAGE9_PRIMARY_MODELS:
        rows.extend([
            _row(model, "2020-01-01", "2020-01-02", 1.0),
            _row(model, "2020-01-02", "2020-01-03", 2.0),
        ])
    # This optional diagnostic covers only one key.  Inferring the registry
    # from every present model would wrongly discard the second primary key.
    rows.append(_row("Air2stream-a8", "2020-01-01", "2020-01-02", 1.0))
    aligned, audit = enforce_common_forecast_keys(
        pd.DataFrame(rows), STAGE9_PRIMARY_MODELS
    )
    assert audit.common_unique == 2
    primary = aligned[
        aligned["split"].eq("test") & aligned["model"].isin(STAGE9_PRIMARY_MODELS)
    ]
    assert primary.groupby("model").size().eq(2).all()
    assert len(aligned[aligned["model"].eq("Air2stream-a8")]) == 1


def test_tabular_training_rows_are_restricted_to_window_registry():
    class WD:
        horizons = (1,)
        station = np.array([0, 0])
        issue_date = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
        target_date = np.array([["2020-01-02"], ["2020-01-03"]], dtype="datetime64[ns]")
        split = np.array(["train", "val"], dtype=object)
        y = np.array([[1.0], [2.0]])

    tab = pd.DataFrame([
        {"site_id": "s1", "issue_date": "2019-12-31", "target_date": "2020-01-01",
         "split": "none", "y": 0.0, "x": 4.0},
        {"site_id": "s1", "issue_date": "2020-01-01", "target_date": "2020-01-02",
         "split": "train", "y": 1.0, "x": 5.0},
        {"site_id": "s1", "issue_date": "2020-01-02", "target_date": "2020-01-03",
         "split": "val", "y": 2.0, "x": 6.0},
    ])
    aligned = restrict_tabular_to_window_registry(tab, WD(), ("s1",), 1)
    assert aligned["x"].tolist() == [5.0, 6.0]


@pytest.mark.parametrize("truth64", [0.25, 32.1])
def test_tabular_window_truth_comparison_uses_float32_and_rejects_next_ulp(
    truth64,
):

    class WD:
        horizons = (1,)
        station = np.array([0])
        issue_date = np.array(["2020-01-01"], dtype="datetime64[ns]")
        target_date = np.array([["2020-01-02"]], dtype="datetime64[ns]")
        split = np.array(["train"], dtype=object)
        y = np.array([[truth64]], dtype=np.float32)

    tab = pd.DataFrame([{
        "site_id": "s1", "issue_date": "2020-01-01",
        "target_date": "2020-01-02", "split": "train", "y": truth64,
    }])
    aligned = restrict_tabular_to_window_registry(tab, WD(), ("s1",), 1)
    assert len(aligned) == 1

    tab.loc[0, "y"] = np.nextafter(
        np.float32(truth64), np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ValueError, match="disagree on target labels"):
        restrict_tabular_to_window_registry(tab, WD(), ("s1",), 1)


def test_canonical_registry_retains_leading_zeroes_and_exactly_15_huc2_clusters():
    registry = load_station_registry()
    assert len(registry) == 120
    assert registry.site_no.str.fullmatch(r"\d{8,15}").all()
    assert registry.huc2.str.fullmatch(r"\d{2}").all()
    assert registry.huc_cd.str[:2].eq(registry.huc2).all()
    clusters = huc2_cluster_map(registry)
    assert len(clusters) == 120
    assert len(set(clusters.values())) == 15
    assert "HUC2:01" in set(clusters.values())
    assert "HUC2:09" in set(clusters.values())
