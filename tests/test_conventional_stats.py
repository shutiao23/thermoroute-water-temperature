"""Tests for the pure-statistics derivation layer (conventional_stats.py)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from thermoroute import conventional_stats as CS

REPO = Path(__file__).resolve().parents[1]


def _prediction_table(*, n_stations=6, n_days=120, seed=0, n_baselines=True):
    rng = np.random.default_rng(seed)
    rows = []
    models = ["ThermoRoute", "LightGBM"]
    if n_baselines:
        models += list(CS.BASELINE_MODELS)
    for model in models:
        for site in range(n_stations):
            for day in range(n_days):
                y_true = 10.0 + site * 0.5 + rng.normal(0, 1)
                noise = rng.normal(0, 1)
                y_pred = y_true - 0.3 * (day % 3 == 0) + noise * 0.2
                rows.append({
                    "model": model, "horizon": 1, "seed": 0,
                    "site_id": f"0100000{site}", "split": "test",
                    "issue_date": pd.Timestamp(f"2021-01-{day % 28 + 1:02d}"),
                    "target_date": pd.Timestamp(f"2021-01-{day % 28 + 2:02d}"),
                    "y_true": y_true, "y_pred": y_pred,
                    "q05": np.nan, "q50": np.nan, "q95": np.nan,
                    "p_exceed": np.nan, "calibration_state": "NOT_APPLICABLE_POINT_ONLY",
                    "event_observed": np.nan, "event_threshold_c": np.nan,
                    "huc2": str(site % 3), "n_members": 1,
                    "bundle_sha256": "b" * 64, "cohort": "temporal",
                })
    return pd.DataFrame(rows)


def test_minimum_targets_boundary():
    pred = _prediction_table(n_days=105)
    station = CS.station_metrics(pred, minimum_targets=100)
    assert station.n.min() >= 100
    assert len(station) == len(pred.model.unique()) * pred.site_id.nunique()
    station_strict = CS.station_metrics(pred, minimum_targets=106)
    assert station_strict.empty


def test_station_median_is_unweighted_across_unequal_n():
    pred = _prediction_table(n_stations=4, n_days=120, seed=2)
    station = CS.station_metrics(pred)
    # give one station a much larger n by duplicating its rows
    extra = pred[pred.site_id == "s0"].copy()
    extra["site_id"] = "s0x"
    pred2 = pd.concat([pred, extra], ignore_index=True)
    station2 = CS.station_metrics(pred2)
    med = CS._station_median_rmse(station2)
    # median must not shift towards the duplicated station's value
    med_without = CS._station_median_rmse(station)
    assert np.isclose(med[("ThermoRoute", 1)], med_without[("ThermoRoute", 1)], rtol=0.05)


def test_skill_damped_persistence_present_for_every_model_horizon():
    pred = _prediction_table(n_days=120, seed=3)
    station = CS.station_metrics(pred)
    pooled = CS.pooled_metrics(pred)
    skill = CS.skill_table(station, pooled)
    for model in ("ThermoRoute", "LightGBM"):
        sub = skill[(skill.model == model) & (skill.baseline == "DampedPersistence")]
        assert set(sub.horizon) == {1}
        assert (sub.skill.notna()).all()


def test_effect_sign_convention_and_win_rate():
    pred = _prediction_table(n_days=120, seed=4)
    # make LightGBM strictly better (negative delta RMSE vs ThermoRoute)
    mask = pred.model == "LightGBM"
    pred.loc[mask, "y_pred"] = pred.loc[mask, "y_true"]  # perfect
    pred.loc[pred.model == "ThermoRoute", "y_pred"] = pred.loc[
        pred.model == "ThermoRoute", "y_true"
    ] + 0.25  # deterministic degradation
    station = CS.station_metrics(pred)
    effects = CS.paired_effects(
        station, contrasts=[("LightGBM", "ThermoRoute", 1)]
    )
    assert (effects.effect < 0).all()
    registry = pd.DataFrame({
        "site_no": [f"0100000{i}" for i in range(6)], "lat": 0.0, "lon": 0.0,
        "huc2": [f"0{i % 3}" for i in range(6)],
        "huc_metadata_status": ["USGS_SNAPSHOT_SITE_NO_MATCH"] * 6,
    })
    inference = CS.cluster_inference(effects, registry, holm_family=5)
    record = inference["LightGBM|ThermoRoute|1"]
    assert record["win_rate"] == 1.0
    assert record["effect"] < 0.0


def test_cluster_inference_holm_applied_to_exactly_five():
    rng = np.random.default_rng(7)
    rows = []
    for contrast in range(5):
        for site in range(30):
            rows.append({
                "candidate": "C", "reference": "R",
                "horizon": (1, 3, 7, 14, 21)[contrast],
                "site_id": f"0100000{site}",
                "effect": float(rng.normal(-0.1, 0.3)),
            })
    effects = pd.DataFrame(rows)
    registry = pd.DataFrame({
        "site_no": [f"0100000{i}" for i in range(30)], "lat": 0.0, "lon": 0.0,
        "huc2": [f"0{i % 6}" for i in range(30)],
        "huc_metadata_status": ["USGS_SNAPSHOT_SITE_NO_MATCH"] * 30,
    })
    inference = CS.cluster_inference(effects, registry, holm_family=5, n_boot=200)
    records = list(inference.values())
    assert len(records) == 5
    assert all(record["holm_p"] is not None and not np.isnan(record["holm_p"]) for record in records)


def test_load_registry_preserves_huc2():
    from scripts.conventional_derive_statistics import load_registry
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "registry.csv"
        path.write_text("site_no,lat,lon,huc2,huc_metadata_status\n01073319,43.0,-71.0,1,USGS_SNAPSHOT_SITE_NO_MATCH\n", encoding="utf-8")
        registry = load_registry(path)
    assert list(registry.columns) == ["site_no", "lat", "lon", "huc2", "huc_metadata_status"]
    assert registry.huc2.iloc[0] == 1


def test_probability_metrics_skips_uncalibrated_models():
    pred = _prediction_table(n_days=60, seed=8)
    assert CS.probability_metrics(pred).empty  # all NOT_APPLICABLE_POINT_ONLY
    calibrated = pred[pred.model.isin(("ThermoRoute", "LightGBM"))].copy()
    calibrated["calibration_state"] = "FROZEN_CQR_PLATT_APPLIED"
    calibrated["q05"] = 8.0
    calibrated["q50"] = 10.0
    calibrated["q95"] = 12.0
    calibrated["p_exceed"] = 0.3
    calibrated["event_observed"] = 0
    out = CS.probability_metrics(calibrated)
    assert not out.empty
    assert set(out.model) == {"ThermoRoute", "LightGBM"}


def test_reliability_bins_cover_all_probabilities_and_are_monotone():
    pred = _prediction_table(n_days=300, seed=9)
    pred["calibration_state"] = "FROZEN_CQR_PLATT_APPLIED"
    pred["event_observed"] = 0
    pred["p_exceed"] = np.linspace(0.01, 0.99, len(pred))
    pred["q05"], pred["q50"], pred["q95"] = 8.0, 10.0, 12.0
    bins = CS.reliability_bins(pred, n_bins=10)
    assert not bins.empty
    grouped = bins.groupby(["model", "horizon"])
    for (model, horizon), group in grouped:
        group = group.sort_values("bin_low")
        assert (group.bin_high.diff().dropna() > 0).all()
        assert (group.mean_forecast.diff().dropna() >= 0).all()


def test_derive_module_imports_no_torch():
    source = (REPO / "scripts/conventional_derive_statistics.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        name.split(".")[0] in {"torch", "lightgbm"}
        for name in imports
    ), f"derivation script must not import model frameworks: {imports}"


def test_stats_module_imports_no_torch():
    source = (REPO / "src/thermoroute/conventional_stats.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        name.split(".")[0] in {"torch", "lightgbm"}
        for name in imports
    ), f"stats module must not import model frameworks: {imports}"
