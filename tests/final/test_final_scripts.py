"""Tests for the final-stage analysis scripts (protocol v1).

Synthetic fixtures only.  Covers the mechanism state classification
(issue-time vs outcome-conditioned separation, signed warming/cooling), the
spatial-factorial cell contract, and the claim resolver's new estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thermoroute import final_results as FR


class _Row:
    """Lightweight row-like object with attributes for state_of()."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_state_of_signed_warming_cooling():
    import scripts.final.run_mechanism_analysis as M
    sq = {
        "anomaly": {"q10": -2.0, "q90": 2.0},
        "trend7": {"q10": -0.5, "q90": 0.5},
        "disequilibrium": {"q10": -3.0, "q90": 3.0},
        "flow_change": {"q10": -0.4, "q90": 0.4},
        "month_flow": {6: {"q10": 10.0, "q90": 100.0}},
    }
    # rapid recent warming (positive trend above q90)
    row = _Row(anomaly=1.0, trend7=1.2, diseq=0.0, dlogq3=0.0, month=6, FLOW=50.0)
    states = M.state_of(row, sq)
    assert "issue_rapid_recent_warming" in states
    assert "issue_rapid_recent_cooling" not in states
    # rapid recent cooling (negative trend below q10)
    row = _Row(anomaly=1.0, trend7=-1.2, diseq=0.0, dlogq3=0.0, month=6, FLOW=50.0)
    states = M.state_of(row, sq)
    assert "issue_rapid_recent_cooling" in states
    # signed disequilibrium is directional
    row = _Row(anomaly=0.0, trend7=0.0, diseq=5.0, dlogq3=0.0, month=6, FLOW=50.0)
    states = M.state_of(row, sq)
    assert "issue_strong_warming_pressure" in states
    assert "issue_strong_cooling_pressure" not in states
    # station-relative flow uses station-month quantiles, not global cfs
    row = _Row(anomaly=0.0, trend7=0.0, diseq=0.0, dlogq3=0.0, month=6, FLOW=200.0)
    states = M.state_of(row, sq)
    assert "issue_high_flow" in states
    row = _Row(anomaly=0.0, trend7=0.0, diseq=0.0, dlogq3=0.0, month=6, FLOW=5.0)
    states = M.state_of(row, sq)
    assert "issue_low_flow" in states


def test_outcome_state_separates_warming_and_cooling():
    import scripts.final.run_mechanism_analysis as M
    sq = {"dT": {7: {"q10": -4.0, "q90": 4.0}},
          "target_wt": {7: {"q10": 5.0, "q90": 25.0}},
          "month_flow": {6: {"q10": 10.0, "q90": 100.0}}}
    # actual rapid warming: positive change above q90
    row = _Row(horizon=7, dT_actual=6.0, WT_t=20.0, FLOW_t=50.0, month=6)
    states = M.outcome_state_of(row, sq)
    assert "actual_rapid_warming" in states
    assert "actual_rapid_cooling" not in states
    # actual rapid cooling: negative change below q10
    row = _Row(horizon=7, dT_actual=-6.0, WT_t=20.0, FLOW_t=50.0, month=6)
    states = M.outcome_state_of(row, sq)
    assert "actual_rapid_cooling" in states
    assert "actual_rapid_warming" not in states
    # |dT| below the thresholds is neither
    row = _Row(horizon=7, dT_actual=2.0, WT_t=20.0, FLOW_t=50.0, month=6)
    states = M.outcome_state_of(row, sq)
    assert "actual_rapid_warming" not in states
    assert "actual_rapid_cooling" not in states


def test_claim_resolver_spatial_estimators():
    effects = pd.DataFrame({
        "site_id": ["a", "a", "b", "b", "a", "a", "b", "b"],
        "seed": [0, 1, 0, 1, 0, 1, 0, 1],
        "geometry": ["region", "region", "region", "region",
                     "random", "random", "random", "random"],
        "rmse": [1.0, 1.1, 1.5, 1.4, 0.9, 0.95, 1.3, 1.35],
    })
    claim = {"source_table": "spatial_effects.parquet",
             "estimator": "median_station_region_minus_random_rmse",
             "filter": {"model": "LightGBM", "horizon": 3, "adaptation": "local"}}
    # the resolver pivots site x geometry and takes the median of
    # region - random per site
    frame = effects.assign(model="LightGBM", horizon=3, adaptation="local")
    value = FR._resolve_estimator(frame, claim)
    pivot = frame.pivot_table(index="site_id", columns="geometry",
                              values="rmse", aggfunc="mean")
    expected = float((pivot["region"] - pivot["random"]).median())
    assert abs(value - expected) < 1e-12
    assert abs(value - 0.125) < 1e-9


def test_claim_resolver_nearest_km():
    frame = pd.DataFrame({
        "geometry": ["random", "random", "region"],
        "nearest_km": [60.0, 62.0, 268.0],
    })
    value = FR._resolve_estimator(
        frame, {"estimator": "median_nearest_km", "filter": {"geometry": "random"}})
    assert value == 61.0


def test_half_life_terminology():
    # t_1/2 = ln(0.5)/ln(phi); e-folding time is -1/ln(phi); they differ.
    phi = 0.9
    half_life = np.log(0.5) / np.log(phi)
    e_folding = -1.0 / np.log(phi)
    assert abs(half_life - 6.58) < 0.01
    assert abs(e_folding - 9.49) < 0.01
    assert not np.isclose(half_life, e_folding)


# --------------------------------------------------------------------------- #
# Consistency-gate regression tests (review defects F1/F2 and P0.6)
# --------------------------------------------------------------------------- #

def _resolved_row(claim_id="T1", value="0.250", used_in=None,
                  macro="SevenDaySkillPersistence"):
    return {
        "claim_id": claim_id, "status": "RESOLVED", "value": value,
        "latex_macro": macro,
        "used_in": ";".join(used_in or ["abstract"]),
        "print_precision": None,
    }


_MAN = """## Key Points
- a claim value of 0.250 appears here

## Abstract
Abstract prose carries 0.250 and 0.589 as well.

### 4.6 Held-out 2021-2023 evaluation
Section prose.

## 7. Conclusions
Conclusions prose.
"""


def test_numbers_check_rejects_absent_claim_value():
    import scripts.final.check_manuscript_consistency as G
    resolved = pd.DataFrame([_resolved_row(value="0.250")])
    assert G.check_manuscript_numbers(_MAN, resolved) == []
    bad = _MAN.replace("0.250", "9.999")
    problems = G.check_manuscript_numbers(bad, resolved)
    assert any("not printed" in p for p in problems)


def test_numbers_check_rejects_wrong_precision_substring():
    import scripts.final.check_manuscript_consistency as G
    # "0.25" must not be satisfied by "0.251"
    resolved = pd.DataFrame([_resolved_row(value="0.250")])
    man = _MAN.replace("0.250", "0.251")
    problems = G.check_manuscript_numbers(man, resolved)
    assert any("not printed" in p for p in problems)


def test_numbers_check_tex_drift_detected():
    import scripts.final.check_manuscript_consistency as G
    resolved = pd.DataFrame([_resolved_row(value="0.250", macro="NoSuchMacroXX")])
    # macro missing -> flagged; value present in md -> only macro/tex issues
    problems = G.check_manuscript_numbers(_MAN, resolved)
    assert any("macro" in p for p in problems)


def test_generated_table_full_block_check(tmp_path):
    import scripts.final.check_manuscript_consistency as G
    gen = tmp_path / "generated.md"
    man = tmp_path / "manuscript.md"
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
    gen.write_text(table)
    G.GENERATED = gen
    G.MANUSCRIPT = man
    # full block present -> clean
    man.write_text(table)
    assert G.check_generated_tables() == []
    # only the LAST row present: the first row must still be checked
    man.write_text("| 3 | 4 |\n")
    problems = G.check_generated_tables()
    assert any("generated table block not in manuscript" in p for p in problems)
    # only the FIRST row present: the last row must also be checked
    man.write_text("| a | b |\n| --- | --- |\n| 1 | 2 |\n")
    problems = G.check_generated_tables()
    assert any("generated table block not in manuscript" in p for p in problems)
    G.GENERATED = G.ROOT / "paper" / "tables_final.md"
    G.MANUSCRIPT = G.ROOT / "paper" / "ThermoRoute_paper.md"


def test_printable_forms_no_coarse_false_positives():
    import scripts.final.check_manuscript_consistency as G
    forms = G._printable_forms("6.9", rendered="6.9")
    assert "7" not in forms
    forms = G._printable_forms("0.250", rendered="0.250")
    assert "0.25" not in forms


def test_table_4_12_all_keys_matches_paired_effects():
    import scripts.final.generate_manuscript_tables as T
    state = pd.read_parquet("outputs/final/hydrologic_state_effects.parquet")
    effects = pd.read_parquet("outputs/final/paired_effects.parquet")
    table = T.table_4_12(state)
    assert "| All keys | -0.069 | 116 | 1,060 |" in table
    assert "1,023" not in table
    assert "1,060" in table
    # the All-keys delta equals Table 4.6 row 3's delta, computed two ways
    all_keys = effects[(effects.candidate == "ThermoRoute")
                       & (effects.reference == "DampedPersistence")
                       & (effects.horizon == 7)]
    assert T.f3(all_keys.delta_rmse.median()) == "-0.069"
    keys = pd.read_parquet("outputs/final/forecast_keys.parquet")
    median_keys = keys[keys.horizon == 7].groupby("site_id").size().median()
    assert f"{int(median_keys):,}" == "1,060"


def test_table_4_6_p_value_columns():
    import json

    import scripts.final.generate_manuscript_tables as T
    station = pd.read_parquet("outputs/final/station_metrics.parquet")
    effects = pd.read_parquet("outputs/final/paired_effects.parquet")
    table = T.table_4_6(station, effects)
    assert "p (sign flip)" in table and "Holm p" in table
    assert "| 3 | ThermoRoute vs. DampedPersistence | 7 d | -0.069 | -0.086 | -0.057 | 0.95 | 116 | 6.1e-05 | 1.8e-04 |" in table
    with open("outputs/final/cluster_inference_2021_2023.json", encoding="utf-8") as fh:
        inference = json.load(fh)
    rec = inference["ThermoRoute|DampedPersistence|7"]
    assert abs(rec["p_cluster_sign_flip"] - 6.103515625e-05) < 1e-9
    assert abs(rec["holm_p"] - 0.00018310546875) < 1e-9
