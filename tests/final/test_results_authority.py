"""Tests for the single results authority (protocol v1).

Synthetic fixtures only — no bundles, no network.  These tests pin the
estimand rules that the external review found violated:

1. station-first RMSE, never pooled arithmetic for a headline;
2. skill = median of per-station ratios, never a ratio of medians;
3. exact station-level additive decomposition (G_total = G_memory + G_learned);
4. common station sets per horizon across models;
5. stable key ids, target = issue + horizon;
6. paper_values macros resolve only from the claim ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thermoroute import final_results as FR


def make_predictions(n_stations: int = 5, n_days: int = 200,
                     horizons=(1, 3, 7)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    models = ("Persistence", "DampedPersistence", "Climatology",
              "ThermoRoute", "LightGBM")
    rows = []
    for station in range(n_stations):
        site = f"{10000000 + station:08d}"
        for day in range(n_days):
            issue = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
            for horizon in horizons:
                target = issue + pd.Timedelta(days=horizon)
                y_true = 10.0 + 5.0 * np.sin(2 * np.pi * day / 365) + rng.normal(0, 1)
                for model in models:
                    rows.append({
                        "model": model, "site_id": site, "horizon": horizon,
                        "issue_date": issue, "target_date": target,
                        "y_true": y_true,
                        "y_pred": y_true + rng.normal(0, 0.2),
                    })
    return pd.DataFrame(rows)


def test_common_key_registry_is_unique_and_dated():
    pred = make_predictions()
    keys = FR.build_key_registry(pred)
    assert keys.duplicated("key_id").sum() == 0
    assert keys.duplicated(["site_id", "horizon", "issue_date"]).sum() == 0
    expected = keys["issue_date"] + pd.to_timedelta(keys["horizon"], unit="D")
    pd.testing.assert_series_equal(keys["target_date"], expected.rename("target_date"))
    # every model must be present on every key
    coverage = pred.merge(keys, on=["site_id", "horizon", "issue_date",
                                    "target_date", "y_true"])
    per_model = coverage.groupby("model")["key_id"].nunique()
    assert (per_model == len(keys)).all()


def test_station_first_rmse_and_common_station_sets():
    pred = make_predictions()
    station = FR.station_metrics(pred)
    # every station-model-horizon cell has the full key count
    assert (station["n"] == 200).all()
    FR.assert_common_station_sets(station)
    per_model_h = station.groupby(["model", "horizon"])["site_id"].nunique()
    assert (per_model_h == 5).all()


def test_skill_is_median_of_station_ratios_not_ratio_of_medians():
    pred = make_predictions()
    station = FR.station_metrics(pred)
    effects = FR.paired_effects(
        station, contrasts=[("ThermoRoute", "DampedPersistence", 7)])
    skill = effects["station_skill"]
    assert skill.between(-1, 1).all()
    computed = effects["station_skill"].median()
    ratio_of_medians = (
        1.0 - station[(station.model == "ThermoRoute") & (station.horizon == 7)]["rmse"].median()
        / station[(station.model == "DampedPersistence") & (station.horizon == 7)]["rmse"].median()
    )
    # these two quantities coincide only by accident; the estimator must be
    # the median of station ratios, which is what paired_effects computes
    assert abs(computed - skill_brute(station, 7)) < 1e-12
    assert not np.isclose(computed, ratio_of_medians)


def skill_brute(station: pd.DataFrame, horizon: int) -> float:
    a = station[(station.model == "ThermoRoute") & (station.horizon == horizon)] \
        .set_index("site_id")["rmse"]
    b = station[(station.model == "DampedPersistence") & (station.horizon == horizon)] \
        .set_index("site_id")["rmse"]
    both = a.index.intersection(b.index)
    return float((1.0 - a.loc[both] / b.loc[both]).median())


def test_decomposition_is_exactly_additive_per_station():
    pred = make_predictions()
    station = FR.station_metrics(pred)
    decomp = FR.decomposition_effects(station)
    assert (np.abs(decomp["g_total"] - decomp["g_memory"] - decomp["g_learned"])
            < 1e-12).all()
    # fractions are well-defined only where g_total > 0
    positive = decomp[decomp["g_total_positive"]]
    assert positive["memory_fraction"].notna().all()
    assert positive["learned_fraction"].notna().all()
    assert (np.abs(positive["memory_fraction"] + positive["learned_fraction"] - 1.0)
            < 1e-12).all()
    # medians of gains do NOT have to sum to the median total; means do
    med = decomp.groupby("horizon")[["g_total", "g_memory", "g_learned"]].median()
    assert not np.allclose(
        med["g_memory"] + med["g_learned"], med["g_total"])
    for horizon in (1, 3, 7):
        waterfall = FR.mean_additive_waterfall(decomp, horizon)
        assert abs(waterfall["mean_g_memory"] + waterfall["mean_g_learned"]
                   - waterfall["mean_g_total"]) < 1e-12


def test_key_id_is_stable_hash():
    issue = pd.Timestamp("2021-06-15")
    target = pd.Timestamp("2021-06-22")
    first = FR.key_id_for("temporal", "known_site", "01234567", issue, target, 7)
    again = FR.key_id_for("temporal", "known_site", "01234567", issue, target, 7)
    assert first == again
    assert len(first) == 64
    other = FR.key_id_for("temporal", "known_site", "01234567", issue, target, 3)
    assert first != other


def test_missingness_against_brute_force():
    rng = np.random.default_rng(1)
    n_days = 120
    dates = pd.date_range("2020-11-01", periods=n_days, freq="D")
    site = "01000001"
    panel = pd.DataFrame({
        "site_id": site,
        "DATE": dates,
        "WTEMP": np.where(rng.random(n_days) < 0.2, np.nan,
                          10 + rng.normal(0, 1, n_days)),
    })
    registry = pd.DataFrame({
        "site_id": [site], "horizon": [7],
        "issue_date": pd.to_datetime(["2021-02-10"]),
        "target_date": pd.to_datetime(["2021-02-17"]),
        "y_true": [12.0],
    })
    keys = FR.attach_context_missingness(registry, panel)
    row = keys.iloc[0]
    issue = pd.Timestamp("2021-02-10")
    for days in (7, 14, 32):
        window = panel[(panel["DATE"] > issue - pd.Timedelta(days=days))
                       & (panel["DATE"] <= issue)]
        expected = int(window["WTEMP"].notna().sum())
        assert row[f"n_observed_wtemp_{days}d"] == expected, days


def test_issue_observedness_and_recency_are_as_of_each_issue_date():
    site = "01000001"
    dates = pd.date_range("2021-01-01", periods=8, freq="D")
    panel = pd.DataFrame({
        "site_id": site,
        "DATE": dates,
        # The final observation must not leak backwards into earlier issues.
        "WTEMP": [np.nan, 5.0, np.nan, np.nan, 6.0, np.nan, np.nan, 99.0],
    })
    issue_dates = pd.to_datetime([
        "2021-01-01",
        "2021-01-02",
        "2021-01-04",
        "2021-01-05",
        "2021-01-07",
    ])
    registry = pd.DataFrame({
        "site_id": [site] * len(issue_dates),
        "horizon": [1] * len(issue_dates),
        "issue_date": issue_dates,
        "target_date": issue_dates + pd.Timedelta(days=1),
        "y_true": np.arange(len(issue_dates), dtype=float),
    })

    keys = FR.attach_context_missingness(registry, panel, context_days=(2,))

    assert keys["issue_wtemp_observed"].tolist() == [False, True, False, True, False]
    assert keys["days_since_last_observed_wtemp"].tolist() == [-1, 0, 2, 0, 2]
    assert (keys["days_since_last_observed_wtemp"] >= -1).all()


def test_claim_ledger_resolution():
    pred = make_predictions()
    station = FR.station_metrics(pred)
    effects = FR.paired_effects(station)
    decomp = FR.decomposition_effects(station)
    ledger = [
        {
            "claim_id": "C1",
            "used_in": ["abstract"],
            "source_table": "paired_effects.parquet",
            "filter": {"candidate": "ThermoRoute", "reference": "DampedPersistence",
                       "horizon": 7},
            "estimator": "median_station_skill",
            "latex_macro": "SevenDaySkillDamped",
        },
        {
            "claim_id": "C2",
            "used_in": ["abstract"],
            "source_table": "decomposition_effects.parquet",
            "filter": {"horizon": 7},
            "estimator": "median_station_g_memory",
            "latex_macro": "SevenDayMemoryGain",
        },
        {
            "claim_id": "C3",
            "used_in": ["abstract"],
            "source_table": "station_metrics.parquet",
            "filter": {"model": "ThermoRoute", "horizon": 7},
            "estimator": "reportable_station_count",
            "latex_macro": "NHeldoutStationsSevenDay",
        },
    ]
    resolved = FR.resolve_claim_ledger(
        {"claims": ledger},
        {"paired_effects.parquet": effects,
         "decomposition_effects.parquet": decomp,
         "station_metrics.parquet": station})
    assert (resolved["status"] == "RESOLVED").all()
    c3 = resolved[resolved.claim_id == "C3"].iloc[0]
    assert int(c3.value) == 5
