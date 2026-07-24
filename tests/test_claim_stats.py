from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_claim_stats():
    module_name = "thermoroute_test_claim_stats"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / "12_claim_stats.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


CLAIM_STATS = _load_claim_stats()


def _station_table(counts: dict[tuple[str, int, str], int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "horizon": horizon,
                "site_id": site,
                "RMSE": float(index + 1),
                "n": count,
            }
            for index, ((site, horizon, model), count) in enumerate(counts.items())
        ]
    )


def test_station_statistics_keep_only_cells_with_at_least_100_common_targets():
    counts = {
        (site, 1, model): count
        for site, count in (("eligible", 100), ("too_short", 99))
        for model in CLAIM_STATS.MODELS
    }
    retained = CLAIM_STATS._retain_reportable_station_rmse(_station_table(counts))
    assert set(retained.site_id) == {"eligible"}
    assert len(retained) == len(CLAIM_STATS.MODELS)
    assert retained.n.eq(CLAIM_STATS.MINIMUM_SITE_HORIZON_TARGETS).all()


def test_station_statistics_fail_closed_when_model_counts_disagree():
    counts = {
        ("site", 3, "ThermoRoute"): 100,
        ("site", 3, "DampedPersistence"): 100,
        ("site", 3, "LightGBM"): 101,
    }
    with pytest.raises(RuntimeError, match="target counts disagree across models"):
        CLAIM_STATS._retain_reportable_station_rmse(_station_table(counts))


def test_station_statistics_fail_closed_when_model_cell_is_missing():
    counts = {
        ("site", 7, "ThermoRoute"): 100,
        ("site", 7, "DampedPersistence"): 100,
    }
    with pytest.raises(RuntimeError, match="model coverage is incomplete"):
        CLAIM_STATS._retain_reportable_station_rmse(_station_table(counts))


def test_station_statistics_reject_duplicate_model_station_horizon_cell():
    counts = {
        ("site", 1, model): 100
        for model in CLAIM_STATS.MODELS
    }
    table = _station_table(counts)
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicates a model/station/horizon cell"):
        CLAIM_STATS._retain_reportable_station_rmse(table)


def _development_identities() -> list[dict[str, object]]:
    return [
        CLAIM_STATS._development_row_identity(
            horizon=horizon,
            reference=reference,
            source_prediction_sha256="a" * 64,
            source_tree_sha256="b" * 64,
        )
        for horizon in (1, 3, 7)
        for reference in ("DampedPersistence", "LightGBM")
    ]


def test_development_mirror_rows_have_exact_nonformal_identity_schema():
    rows = _development_identities()
    expected = set(CLAIM_STATS.DEVELOPMENT_IDENTITY_COLUMNS)
    assert all(set(row) == expected for row in rows)
    assert all(
        row["evidence_role"]
        == "DEVELOPMENT_EXPLORATORY_PREVIOUSLY_INSPECTED"
        for row in rows
    )
    assert all(row["previously_inspected"] is True for row in rows)
    assert all(row["formal_confirmatory_result"] is False for row in rows)
    assert all(row["decision_eligible"] is False for row in rows)
    assert all(
        row["p_value_role"]
        == "ASSUMPTION_CONDITIONAL_DEVELOPMENT_SENSITIVITY"
        for row in rows
    )
    assert all(
        row["equivalence_role"] == "EXPLORATORY_DIAGNOSTIC_NO_CLAIM"
        for row in rows
    )
    assert all(
        row["minimum_common_targets"]
        == CLAIM_STATS.MINIMUM_SITE_HORIZON_TARGETS
        for row in rows
    )
    assert all(
        row["canonical_registry_models"]
        == "|".join(CLAIM_STATS.ROUTE_A_PRIMARY_MODELS)
        for row in rows
    )
    assert all(
        row["summarized_models"] == "|".join(CLAIM_STATS.MODELS)
        for row in rows
    )


def test_development_mirror_has_exactly_five_family_rows_and_excludes_h1_lightgbm():
    rows = _development_identities()
    family = [row for row in rows if row["in_frozen_five_test_family"]]
    assert len(family) == 5
    assert {row["mirror_test_id"] for row in family} == set(
        CLAIM_STATS.FROZEN_FAMILY_TEST_IDS.values()
    )
    h1_lightgbm = rows[1]
    assert h1_lightgbm["mirror_test_id"] == ""
    assert h1_lightgbm["in_frozen_five_test_family"] is False
    assert h1_lightgbm["comparison_role"] == (
        "EXPLORATORY_H1_LIGHTGBM_OUTSIDE_FROZEN_FAMILY"
    )


def test_development_mirror_uses_only_explicitly_exploratory_output_names():
    source = (ROOT / "scripts" / "12_claim_stats.py").read_text(encoding="utf-8")
    for name in (
        "development_route_a_estimand_mirror.csv",
        "development_route_a_estimand_mirror.md",
        "development_route_a_estimand_mirror_cluster_sensitivity.csv",
        "development_route_a_estimand_mirror_cluster_loco.csv",
        "development_route_a_estimand_mirror.png",
    ):
        assert name in source
    for obsolete in (
        "claim1_significance",
        "claim1_cluster_sensitivity",
        "claim1_cluster_loco",
        "p_holm_confirmatory_family",
    ):
        assert obsolete not in source
