from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from thermoroute.coverage_audit import (
    POLICY_RELATIVE,
    CoverageAuditError,
    build_temporal_coverage_audit,
    validate_temporal_coverage_audit,
    validate_temporal_coverage_policy,
)
from thermoroute.repro import sha256_json


ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2001-01-01")
END = pd.Timestamp("2003-12-31")
HORIZONS = (1, 3, 7)
SEASONS = ("DJF", "MAM", "JJA", "SON")
MODELS = ("Candidate", "Baseline", "Other")
SITES = {"temporal": ("t1", "t2"), "external": ("e1",)}


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _formal_tests() -> list[dict[str, Any]]:
    return [
        {
            "test_id": "H1-h1-vs-baseline",
            "candidate": "Candidate",
            "reference": "Baseline",
            "horizon": 1,
            "margin_c": 0.0,
        },
        {
            "test_id": "H1-h3-vs-baseline",
            "candidate": "Candidate",
            "reference": "Baseline",
            "horizon": 3,
            "margin_c": 0.0,
        },
        {
            "test_id": "H1-h7-vs-baseline",
            "candidate": "Candidate",
            "reference": "Baseline",
            "horizon": 7,
            "margin_c": 0.0,
        },
        {
            "test_id": "H2-h3-vs-other",
            "candidate": "Candidate",
            "reference": "Other",
            "horizon": 3,
            "margin_c": 0.05,
        },
        {
            "test_id": "H2-h7-vs-other",
            "candidate": "Candidate",
            "reference": "Other",
            "horizon": 7,
            "margin_c": 0.05,
        },
    ]


def _observability(
    *,
    observed_rule: Callable[[str, str, pd.Timestamp], bool] | None = None,
) -> dict[str, pd.DataFrame]:
    dates = pd.date_range(START - pd.Timedelta(days=31), END, freq="D")
    output: dict[str, pd.DataFrame] = {}
    for cohort, sites in SITES.items():
        rows = []
        for site in sites:
            for date in dates:
                observed = (
                    True
                    if observed_rule is None
                    else observed_rule(cohort, site, pd.Timestamp(date))
                )
                rows.append(
                    {
                        "site_id": site,
                        "date": date,
                        "wtemp_observed": observed,
                    }
                )
        output[cohort] = pd.DataFrame(rows)
    return output


def _prediction_error(model: str, target_date: pd.Timestamp) -> float:
    if model == "Baseline":
        return 0.65
    if model == "Other":
        return 0.50
    by_season = {"DJF": 0.20, "MAM": 0.35, "JJA": 0.90, "SON": 0.30}
    return by_season[_season(target_date.month)] + 0.04 * (target_date.year - 2001)


def _predictions(
    observability: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for cohort, sites in SITES.items():
        observed = observability[cohort].set_index(["site_id", "date"])[
            "wtemp_observed"
        ]
        rows = []
        for site_index, site in enumerate(sites):
            values = observed.loc[site]
            for horizon in HORIZONS:
                for issue_date in pd.date_range(
                    START, END - pd.Timedelta(days=horizon), freq="D"
                ):
                    target_date = issue_date + pd.Timedelta(days=horizon)
                    if not bool(values.loc[issue_date]) or not bool(
                        values.loc[target_date]
                    ):
                        continue
                    y_true = 10.0 + site_index + 0.01 * target_date.month
                    for model in MODELS:
                        rows.append(
                            {
                                "model": model,
                                "site_id": site,
                                "horizon": horizon,
                                "issue_date": issue_date,
                                "target_date": target_date,
                                "y_true": y_true,
                                "y_pred": y_true + _prediction_error(model, target_date),
                                "ignored_extra_column": "not-consumed",
                            }
                        )
        output[cohort] = pd.DataFrame(rows)
    return output


def _availability(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort, sites in SITES.items():
        one_model = predictions[cohort][predictions[cohort].model.eq("Candidate")]
        counts = one_model.groupby(["site_id", "horizon"]).size().to_dict()
        for site in sites:
            for horizon in HORIZONS:
                count = int(counts.get((site, horizon), 0))
                rows.append(
                    {
                        "cohort": cohort,
                        "site_no": site,
                        "horizon": horizon,
                        "n_valid_targets": count,
                        "reportable": count >= 100,
                    }
                )
    return pd.DataFrame(rows)


def _rmse(group: pd.DataFrame) -> float:
    ordered = group.sort_values(["issue_date", "target_date"], kind="mergesort")
    error = ordered.y_pred.to_numpy(float) - ordered.y_true.to_numpy(float)
    return float(np.sqrt(np.mean(np.square(error))))


def _primary_statistics(
    predictions: dict[str, pd.DataFrame], availability: pd.DataFrame
) -> list[dict[str, Any]]:
    temporal = predictions["temporal"]
    rows = []
    for test in _formal_tests():
        horizon = int(test["horizon"])
        reportable_sites = sorted(
            availability.loc[
                availability.cohort.eq("temporal")
                & availability.horizon.eq(horizon)
                & availability.reportable,
                "site_no",
            ].astype(str)
        )
        effects = []
        for site in reportable_sites:
            selected = temporal[
                temporal.site_id.eq(site) & temporal.horizon.eq(horizon)
            ]
            candidate = selected[selected.model.eq(test["candidate"])]
            reference = selected[selected.model.eq(test["reference"])]
            effects.append(_rmse(candidate) - _rmse(reference))
        rows.append(
            {
                "test_id": test["test_id"],
                "median_effect_c": (
                    None if not effects else float(np.median(np.asarray(effects)))
                ),
                "n_stations": len(effects),
                "ignored_inferential_field": "not-consumed",
            }
        )
    return rows


def _fixture(
    *,
    observed_rule: Callable[[str, str, pd.Timestamp], bool] | None = None,
) -> dict[str, Any]:
    observations = _observability(observed_rule=observed_rule)
    predictions = _predictions(observations)
    availability = _availability(predictions)
    return {
        "policy": validate_temporal_coverage_policy(ROOT / POLICY_RELATIVE),
        "target_start": START,
        "target_end": END,
        "sites_by_cohort": SITES,
        "observability_by_cohort": observations,
        "predictions_by_cohort": predictions,
        "availability": availability,
        "formal_tests": _formal_tests(),
        "primary_statistics": _primary_statistics(predictions, availability),
    }


def _build(arguments: dict[str, Any]) -> dict[str, Any]:
    return build_temporal_coverage_audit(**arguments)


def test_frozen_policy_is_exact_outcome_free_and_nonfiltering() -> None:
    policy = validate_temporal_coverage_policy(ROOT / POLICY_RELATIVE)
    assert policy["post_2020_wtemp_requested_or_inspected"] is False
    assert policy["coverage_contract"][
        "coverage_balance_changes_primary_reportability"
    ] is False
    assert policy["scope"]["not_a_missing_at_random_assessment"] is True
    assert policy["sensitivity_contract"][
        "primary_station_set_or_statistic_changed"
    ] is False
    assert policy["prohibited_outputs"] == [
        "p_value",
        "confidence_interval",
        "Holm_adjustment",
        "pass_fail_decision",
    ]


def test_policy_semantic_or_self_hash_tamper_is_rejected(tmp_path: Path) -> None:
    policy = validate_temporal_coverage_policy(ROOT / POLICY_RELATIVE)
    attacked = deepcopy(policy)
    attacked["coverage_contract"][
        "coverage_balance_changes_primary_reportability"
    ] = True
    path = tmp_path / "policy.json"
    path.write_text(__import__("json").dumps(attacked), encoding="utf-8")
    with pytest.raises(CoverageAuditError, match="semantics changed"):
        validate_temporal_coverage_policy(path)


def test_complete_zero_inclusive_cells_and_target_date_partition() -> None:
    def observed(cohort: str, site: str, date: pd.Timestamp) -> bool:
        return not (
            cohort == "temporal"
            and site == "t2"
            and date.year == 2002
            and date.month in (9, 10, 11)
        )

    document = _build(_fixture(observed_rule=observed))
    cells = pd.DataFrame(document["coverage_cells"])
    assert len(cells) == 3 * 3 * 3 * 4
    assert not cells.duplicated(
        ["cohort", "site_no", "horizon", "target_year", "target_season"]
    ).any()
    zero = cells[
        cells.cohort.eq("temporal")
        & cells.site_no.eq("t2")
        & cells.horizon.eq(7)
        & cells.target_year.eq(2002)
        & cells.target_season.eq("SON")
    ].iloc[0]
    assert zero.n_calendar_opportunities > 0
    assert zero.n_valid_keys == 0

    # Classification is by target date.  In the first non-leap DJF, the first
    # possible target is Jan 2 for h=1 and Jan 8 for h=7.
    t1 = cells[cells.cohort.eq("temporal") & cells.site_no.eq("t1")]
    h1_djf = t1[
        t1.horizon.eq(1)
        & t1.target_year.eq(2001)
        & t1.target_season.eq("DJF")
    ].iloc[0]
    h7_djf = t1[
        t1.horizon.eq(7)
        & t1.target_year.eq(2001)
        & t1.target_season.eq("DJF")
    ].iloc[0]
    assert h1_djf.n_calendar_opportunities == 89
    assert h7_djf.n_calendar_opportunities == 83


def test_counts_reconstruct_availability_and_primary_is_unchanged() -> None:
    arguments = _fixture()
    document = _build(arguments)
    cells = pd.DataFrame(document["coverage_cells"])
    reconstructed = cells.groupby(["cohort", "site_no", "horizon"])[
        "n_valid_keys"
    ].sum()
    for row in arguments["availability"].itertuples(index=False):
        assert reconstructed[(row.cohort, row.site_no, row.horizon)] == (
            row.n_valid_targets
        )
    expected = {
        row["test_id"]: row for row in arguments["primary_statistics"]
    }
    for row in document["comparison_sensitivities"]:
        assert row["primary_median_effect_c"] == expected[row["test_id"]][
            "median_effect_c"
        ]
        assert row["n_primary_reportable_stations"] == expected[row["test_id"]][
            "n_stations"
        ]
    assert document["primary_statistics_unchanged"] is True
    assert document["primary_station_set_unchanged"] is True


@pytest.mark.parametrize(
    ("run_length", "expected_h1_reportable"), [(100, False), (101, True)]
)
def test_reportability_boundary_is_exactly_100(
    run_length: int, expected_h1_reportable: bool
) -> None:
    last = START + pd.Timedelta(days=run_length - 1)

    def observed(_cohort: str, _site: str, date: pd.Timestamp) -> bool:
        return START <= date <= last

    arguments = _fixture(observed_rule=observed)
    document = _build(arguments)
    availability = arguments["availability"]
    h1 = availability[availability.horizon.eq(1)]
    assert set(h1.n_valid_targets) == {run_length - 1}
    assert h1.reportable.eq(expected_h1_reportable).all()
    cells = pd.DataFrame(document["coverage_cells"])
    assert cells.groupby(["cohort", "site_no", "horizon"]).n_valid_keys.sum()[
        ("temporal", "t1", 1)
    ] == run_length - 1


def _equal_cell_effect(
    frame: pd.DataFrame, *, candidate: str, reference: str, horizon: int
) -> float:
    effects = []
    for site, site_rows in frame[frame.horizon.eq(horizon)].groupby("site_id"):
        model_rmse = {}
        for model in (candidate, reference):
            selected = site_rows[site_rows.model.eq(model)].copy()
            selected["year"] = selected.target_date.dt.year
            selected["season"] = selected.target_date.dt.month.map(_season)
            selected["se"] = np.square(selected.y_pred - selected.y_true)
            cell_mse = selected.groupby(["year", "season"]).se.mean()
            assert len(cell_mse) == 12, site
            model_rmse[model] = float(np.sqrt(cell_mse.mean()))
        effects.append(model_rmse[candidate] - model_rmse[reference])
    return float(np.median(np.asarray(effects)))


def test_equal_cell_and_leave_one_sensitivities_use_fixed_complete_subset() -> None:
    arguments = _fixture()
    document = _build(arguments)
    row = document["comparison_sensitivities"][0]
    assert row["status"] == "ESTIMABLE_DESCRIPTIVE"
    assert row["n_complete_12cell_stations"] == 2
    expected = _equal_cell_effect(
        arguments["predictions_by_cohort"]["temporal"],
        candidate="Candidate",
        reference="Baseline",
        horizon=1,
    )
    assert row["equal_12cell_median_effect_c"] == pytest.approx(expected, abs=1e-15)
    assert row["equal_12cell_median_effect_c"] != row[
        "primary_median_effect_complete_support_c"
    ]
    assert [item["omitted_year"] for item in row["leave_one_year_equal_cell"]] == [
        2001,
        2002,
        2003,
    ]
    assert [
        item["omitted_season"] for item in row["leave_one_season_equal_cell"]
    ] == list(SEASONS)


def test_missing_cell_excludes_only_sensitivity_not_primary() -> None:
    def observed(cohort: str, site: str, date: pd.Timestamp) -> bool:
        return not (
            cohort == "temporal"
            and site == "t2"
            and date.year == 2002
            and date.month in (9, 10, 11)
        )

    document = _build(_fixture(observed_rule=observed))
    for row in document["comparison_sensitivities"]:
        assert row["n_primary_reportable_stations"] == 2
        assert row["n_complete_12cell_stations"] == 1
        assert row["primary_median_effect_c"] is not None
        assert row["equal_12cell_median_effect_c"] is not None


def test_no_inferential_or_decision_fields_are_emitted() -> None:
    document = _build(_fixture())
    prohibited = {
        "p_value",
        "confidence_interval",
        "Holm_adjustment",
        "pass_fail_decision",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & prohibited)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document["comparison_sensitivities"])
    assert document["inference_computed"] is False


def test_semantic_input_order_is_irrelevant_and_exact_replay_validates() -> None:
    arguments = _fixture()
    document = _build(arguments)
    shuffled = deepcopy(arguments)
    for frame in shuffled["observability_by_cohort"].values():
        frame[:] = frame.sample(frac=1.0, random_state=17).to_numpy()
    for cohort, frame in shuffled["predictions_by_cohort"].items():
        shuffled["predictions_by_cohort"][cohort] = frame.sample(
            frac=1.0, random_state=19
        ).reset_index(drop=True)
    shuffled["availability"] = shuffled["availability"].sample(
        frac=1.0, random_state=23
    ).reset_index(drop=True)
    assert _build(shuffled) == document
    assert validate_temporal_coverage_audit(document, **arguments) == document


def test_self_hash_and_semantic_tamper_fail_closed() -> None:
    arguments = _fixture()
    document = _build(arguments)
    attacked = deepcopy(document)
    attacked["coverage_cells"][0]["n_valid_keys"] -= 1
    with pytest.raises(CoverageAuditError, match="self-hash changed"):
        validate_temporal_coverage_audit(attacked, **arguments)

    # Re-hashing an altered document cannot defeat exact semantic replay.
    stable = deepcopy(attacked)
    stable.pop("audit_self_sha256")
    attacked["audit_self_sha256"] = sha256_json(stable)
    with pytest.raises(CoverageAuditError, match="stale or tampered"):
        validate_temporal_coverage_audit(attacked, **arguments)


def test_changed_prediction_semantics_cannot_reuse_primary_statistics() -> None:
    arguments = _fixture()
    document = _build(arguments)
    changed = deepcopy(arguments)
    temporal = changed["predictions_by_cohort"]["temporal"]
    mask = temporal.model.eq("Candidate") & temporal.site_id.eq("t1")
    temporal.loc[mask, "y_pred"] += 0.25
    with pytest.raises(CoverageAuditError, match="primary effect exact crosscheck"):
        validate_temporal_coverage_audit(document, **changed)
