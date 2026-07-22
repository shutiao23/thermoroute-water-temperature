from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from thermoroute.coverage_audit import (
    COMPARISON_MODELS,
    FORECAST_KEY_DIGEST_DOMAIN,
    MODEL_REGISTRY,
    POLICY_FILE_SHA256,
    POLICY_RELATIVE,
    PREDICTION_COLUMNS,
    ROUTE_A_FORMAL_TESTS,
    SOURCE_BINDING_KEYS,
    TARGET_END,
    TARGET_START,
    Y_TRUE_DIGEST_DOMAIN,
    CoverageAuditError,
    build_temporal_coverage_audit,
    validate_temporal_coverage_audit,
    validate_temporal_coverage_policy,
)
from thermoroute.repro import canonical_json, sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp(TARGET_START)
END = pd.Timestamp(TARGET_END)
HORIZONS = (1, 3, 7)
SEASONS = ("DJF", "MAM", "JJA", "SON")
DEFAULT_SITES = {"temporal": ("t1", "t2"), "external": ("e1",)}


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _digest_rows(domain: str, rows: Sequence[Sequence[object]]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\n")
    for row in rows:
        digest.update(canonical_json(list(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _formal_tests() -> list[dict[str, Any]]:
    return [dict(row) for row in ROUTE_A_FORMAL_TESTS]


def _source_bindings() -> dict[str, dict[str, str]]:
    output = {
        key: {
            "path": f"evidence/coverage/{key}.json",
            "sha256": hashlib.sha256(key.encode("ascii")).hexdigest(),
        }
        for key in SOURCE_BINDING_KEYS
    }
    output["policy"] = {
        "path": POLICY_RELATIVE,
        "sha256": POLICY_FILE_SHA256,
    }
    return output


def _observability(
    sites: Mapping[str, Sequence[str]],
    *,
    observed_rule: Callable[[str, str, pd.Timestamp], bool] | None = None,
) -> dict[str, pd.DataFrame]:
    dates = pd.date_range(START - pd.Timedelta(days=31), END, freq="D")
    output: dict[str, pd.DataFrame] = {}
    for cohort in ("temporal", "external"):
        rows: list[dict[str, object]] = []
        for site in sites[cohort]:
            for date in dates:
                rows.append(
                    {
                        "site_id": site,
                        "date": date,
                        "wtemp_observed": (
                            True
                            if observed_rule is None
                            else observed_rule(cohort, site, pd.Timestamp(date))
                        ),
                    }
                )
        output[cohort] = pd.DataFrame(
            rows, columns=["site_id", "date", "wtemp_observed"]
        )
    return output


def _valid_keys(
    observations: pd.DataFrame, sites: Sequence[str]
) -> list[tuple[str, int, pd.Timestamp, pd.Timestamp]]:
    indexed = observations.set_index(["site_id", "date"])["wtemp_observed"]
    rows: list[tuple[str, int, pd.Timestamp, pd.Timestamp]] = []
    for site in sorted(sites):
        values = indexed.loc[site]
        for horizon in HORIZONS:
            for issue in pd.date_range(
                START, END - pd.Timedelta(days=horizon), freq="D"
            ):
                target = issue + pd.Timedelta(days=horizon)
                if bool(values.loc[issue]) and bool(values.loc[target]):
                    rows.append((site, horizon, pd.Timestamp(issue), pd.Timestamp(target)))
    return rows


def _truth(site: str, target: pd.Timestamp) -> float:
    site_offset = sum(ord(value) for value in site) % 7
    return 10.0 + site_offset + 0.01 * target.month


def _error(model: str, target: pd.Timestamp) -> float:
    if model == "DampedPersistence":
        return 0.65
    if model == "LightGBM":
        return 0.50
    by_season = {"DJF": 0.20, "MAM": 0.35, "JJA": 0.90, "SON": 0.30}
    return by_season[_season(target.month)] + 0.04 * (target.year - 2021)


def _comparison_predictions(
    observations: Mapping[str, pd.DataFrame], sites: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site, horizon, issue, target in _valid_keys(
        observations["temporal"], sites["temporal"]
    ):
        y_true = _truth(site, target)
        for model in COMPARISON_MODELS:
            rows.append(
                {
                    "model": model,
                    "site_id": site,
                    "horizon": horizon,
                    "issue_date": issue,
                    "target_date": target,
                    "y_true": y_true,
                    "y_pred": y_true + _error(model, target),
                    "upstream_extra_column": "bound-but-not-consumed",
                }
            )
    return pd.DataFrame(rows, columns=[*PREDICTION_COLUMNS, "upstream_extra_column"])


def _model_summary_from_keys(
    keys: Sequence[tuple[str, int, pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    key_rows: list[list[object]] = []
    y_rows: list[list[object]] = []
    for site, horizon, issue, target in keys:
        key = [
            site,
            horizon,
            issue.strftime("%Y-%m-%d"),
            target.strftime("%Y-%m-%d"),
        ]
        key_rows.append(key)
        y_rows.append([*key, format(_truth(site, target), ".17g")])
    return {
        "row_count": len(keys),
        "forecast_key_sha256": _digest_rows(FORECAST_KEY_DIGEST_DOMAIN, key_rows),
        "y_true_sha256": _digest_rows(Y_TRUE_DIGEST_DOMAIN, y_rows),
    }


def _model_key_audits(
    observations: Mapping[str, pd.DataFrame], sites: Mapping[str, Sequence[str]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for cohort in ("temporal", "external"):
        summary = _model_summary_from_keys(
            _valid_keys(observations[cohort], sites[cohort])
        )
        output[cohort] = [
            {"model": model, **summary} for model in MODEL_REGISTRY[cohort]
        ]
    return output


def _availability(
    observations: Mapping[str, pd.DataFrame], sites: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in ("temporal", "external"):
        counts: dict[tuple[str, int], int] = {}
        for site, horizon, _issue, _target in _valid_keys(
            observations[cohort], sites[cohort]
        ):
            counts[(site, horizon)] = counts.get((site, horizon), 0) + 1
        for site in sites[cohort]:
            for horizon in HORIZONS:
                count = counts.get((site, horizon), 0)
                rows.append(
                    {
                        "cohort": cohort,
                        "site_no": site,
                        "horizon": horizon,
                        "n_valid_targets": count,
                        "reportable": count >= 100,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "cohort",
            "site_no",
            "horizon",
            "n_valid_targets",
            "reportable",
        ],
    )


def _rmse(group: pd.DataFrame) -> float:
    ordered = group.sort_values(["issue_date", "target_date"], kind="mergesort")
    error = ordered.y_pred.to_numpy(float) - ordered.y_true.to_numpy(float)
    return float(np.sqrt(np.mean(np.square(error))))


def _statistics(
    predictions: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    force_one_cluster: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test in ROUTE_A_FORMAL_TESTS:
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
            selected = predictions[
                predictions.site_id.eq(site) & predictions.horizon.eq(horizon)
            ]
            effects.append(
                _rmse(selected[selected.model.eq(test["candidate"])])
                - _rmse(selected[selected.model.eq(test["reference"])])
            )
        effect = None if not effects else float(np.median(np.asarray(effects)))
        clusters = 1 if effects and force_one_cluster else min(len(effects), 2)
        estimable = bool(effects) and clusters >= 2
        rows.append(
            {
                "test_id": test["test_id"],
                "status": (
                    "ESTIMABLE"
                    if estimable
                    else "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS"
                ),
                "median_effect_c": effect if estimable else None,
                "n_stations": len(effects),
                "n_clusters": clusters,
                "upstream_inferential_field": "bound-but-not-consumed",
            }
        )
    return rows


def _fixture(
    *,
    sites: Mapping[str, Sequence[str]] | None = None,
    observed_rule: Callable[[str, str, pd.Timestamp], bool] | None = None,
    force_one_cluster: bool = False,
) -> dict[str, Any]:
    site_registry = dict(DEFAULT_SITES if sites is None else sites)
    observations = _observability(site_registry, observed_rule=observed_rule)
    predictions = _comparison_predictions(observations, site_registry)
    availability = _availability(observations, site_registry)
    return {
        "policy": validate_temporal_coverage_policy(ROOT / POLICY_RELATIVE),
        "source_bindings": _source_bindings(),
        "target_start": START,
        "target_end": END,
        "sites_by_cohort": site_registry,
        "model_registry_by_cohort": MODEL_REGISTRY,
        "model_key_audits_by_cohort": _model_key_audits(
            observations, site_registry
        ),
        "observability_by_cohort": observations,
        "temporal_comparison_predictions": predictions,
        "availability": availability,
        "formal_tests": _formal_tests(),
        "primary_statistics": _statistics(
            predictions, availability, force_one_cluster=force_one_cluster
        ),
    }


def _build(arguments: dict[str, Any]) -> dict[str, Any]:
    return build_temporal_coverage_audit(**arguments)


def test_policy_freezes_real_family_models_interval_and_file_digest() -> None:
    policy = validate_temporal_coverage_policy(ROOT / POLICY_RELATIVE)
    assert policy["post_2020_wtemp_requested_or_inspected"] is False
    assert policy["formal_comparisons"] == _formal_tests()
    assert policy["model_registry_by_cohort"] == {
        cohort: list(models) for cohort, models in MODEL_REGISTRY.items()
    }
    assert policy["route_a_target_interval"] == {
        "start": TARGET_START,
        "end": TARGET_END,
        "inclusive": True,
    }
    assert policy["sensitivity_contract"][
        "unfavorable_sensitivity_must_be_reported"
    ] is True
    assert policy["sensitivity_contract"][
        "sensitivity_changes_primary_result_or_decision"
    ] is False
    assert sha256_file(ROOT / POLICY_RELATIVE) == POLICY_FILE_SHA256


@pytest.mark.parametrize("rewrite", ["whitespace", "reserialize"])
def test_policy_semantically_identical_physical_rewrite_fails_closed(
    tmp_path: Path, rewrite: str
) -> None:
    source = ROOT / POLICY_RELATIVE
    if rewrite == "whitespace":
        payload = source.read_bytes() + b"\n"
    else:
        import json

        payload = json.dumps(
            json.loads(source.read_text(encoding="utf-8")),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    attacked = tmp_path / "policy.json"
    attacked.write_bytes(payload)
    with pytest.raises(CoverageAuditError, match="physical bytes"):
        validate_temporal_coverage_policy(attacked)


@pytest.mark.parametrize(
    "attack", ["self", "margin", "id", "order", "duplicate", "bool_h", "bool_m"]
)
def test_formal_family_identity_attacks_fail_closed(attack: str) -> None:
    arguments = _fixture()
    tests = deepcopy(arguments["formal_tests"])
    if attack == "self":
        tests[0]["reference"] = tests[0]["candidate"]
    elif attack == "margin":
        tests[3]["margin_c"] = 0.051
    elif attack == "id":
        tests[0]["test_id"] = "H1-renamed"
    elif attack == "order":
        tests[0], tests[1] = tests[1], tests[0]
    elif attack == "duplicate":
        tests[1].update(
            {
                "candidate": tests[0]["candidate"],
                "reference": tests[0]["reference"],
                "horizon": tests[0]["horizon"],
            }
        )
    elif attack == "bool_h":
        tests[0]["horizon"] = True
    else:
        tests[0]["margin_c"] = False
    arguments["formal_tests"] = tests
    with pytest.raises(CoverageAuditError, match="formal"):
        _build(arguments)


@pytest.mark.parametrize(
    "bad_path",
    [
        "/absolute/file",
        "C:/absolute/file",
        "../escape",
        "a/../b",
        "a\\b",
        "a//b",
        "a/./b",
        "a/",
    ],
)
def test_source_binding_path_attacks_fail_closed(bad_path: str) -> None:
    arguments = _fixture()
    arguments["source_bindings"]["protocol"]["path"] = bad_path
    with pytest.raises(CoverageAuditError, match="path"):
        _build(arguments)


def test_source_binding_closure_and_shape_are_exact() -> None:
    arguments = _fixture()
    missing = deepcopy(arguments)
    missing["source_bindings"].pop("statistics")
    with pytest.raises(CoverageAuditError, match="closure"):
        _build(missing)
    extra_field = deepcopy(arguments)
    extra_field["source_bindings"]["statistics"]["bytes"] = "1"
    with pytest.raises(CoverageAuditError, match="malformed"):
        _build(extra_field)
    upper = deepcopy(arguments)
    upper["source_bindings"]["statistics"]["sha256"] = "A" * 64
    with pytest.raises(CoverageAuditError, match="SHA-256"):
        _build(upper)
    wrong_policy = deepcopy(arguments)
    wrong_policy["source_bindings"]["policy"]["sha256"] = "0" * 64
    with pytest.raises(CoverageAuditError, match="policy source binding"):
        _build(wrong_policy)


def test_full_build_binds_sources_models_cells_and_descriptive_support() -> None:
    arguments = _fixture()
    document = _build(arguments)
    assert document["status"] == "DERIVED_CORE_REQUIRES_RECEIPT_BINDING"
    assert document["source_bindings"] == arguments["source_bindings"]
    assert len(document["coverage_cells"]) == 3 * 3 * 3 * 4
    assert document["primary_statistics_unchanged"] is True
    assert document["sensitivity_changes_primary_result_or_decision"] is False
    assert document["inference_computed"] is False
    for row in document["comparison_sensitivities"]:
        assert row["formal_statistics_status"] == "ESTIMABLE"
        assert row["formal_median_effect_c"] == row[
            "prediction_derived_descriptive_median_effect_c"
        ]
        assert row["n_all_12_cells_nonempty_stations"] == 2
        assert len(row["all_12_cells_nonempty_station_support"]) == 2
        for station in row["all_12_cells_nonempty_station_support"]:
            assert (
                0
                < station["min_valid_keys_per_cell"]
                <= station["median_valid_keys_per_cell"]
                <= station["max_valid_keys_per_cell"]
            )
        assert row["does_not_establish_year_or_season_stability"] is True
        candidates = row["frozen_sensitivity_candidates"]
        assert [candidate["source"] for candidate in candidates] == [
            "equal_12cell",
            "leave_one_year_2021",
            "leave_one_year_2022",
            "leave_one_year_2023",
            "leave_one_season_DJF",
            "leave_one_season_MAM",
            "leave_one_season_JJA",
            "leave_one_season_SON",
        ]
        assert len(candidates) == 8
        effects = [
            candidate["descriptive_median_effect_c"] for candidate in candidates
        ]
        worst = row["frozen_worst_unfavorable_sensitivity"]
        expected_index = next(
            index for index, value in enumerate(effects) if value == max(effects)
        )
        assert worst == {
            "status": "DESCRIPTIVE_ESTIMABLE",
            "direction": "LARGER_CANDIDATE_MINUS_REFERENCE_IS_MORE_UNFAVORABLE",
            "tie_rule": "FIRST_IN_FROZEN_CANDIDATE_ORDER",
            "frozen_order_index": expected_index,
            "source": candidates[expected_index]["source"],
            "descriptive_median_effect_c": effects[expected_index],
        }


def test_preaggregated_temporal_sensitivities_match_slow_reference() -> None:
    arguments = _fixture()
    document = _build(arguments)
    predictions = arguments["temporal_comparison_predictions"].copy()
    predictions["target_date"] = pd.to_datetime(predictions.target_date)

    def equal_cell_rmse(
        frame: pd.DataFrame, *, years: Sequence[int], seasons: Sequence[str]
    ) -> float:
        values: list[float] = []
        for year in years:
            for season in seasons:
                months = {
                    "DJF": (12, 1, 2),
                    "MAM": (3, 4, 5),
                    "JJA": (6, 7, 8),
                    "SON": (9, 10, 11),
                }[season]
                cell = frame[
                    frame.target_date.dt.year.eq(year)
                    & frame.target_date.dt.month.isin(months)
                ].sort_values(["issue_date", "target_date"], kind="mergesort")
                error = cell.y_pred.to_numpy(float) - cell.y_true.to_numpy(float)
                values.append(float(np.mean(np.square(error))))
        return float(np.sqrt(np.mean(np.asarray(values, dtype=float))))

    for row in document["comparison_sensitivities"]:
        sites = [
            value["site_no"]
            for value in row["all_12_cells_nonempty_station_support"]
        ]

        def effects(*, years: Sequence[int], seasons: Sequence[str]) -> float:
            values = []
            for site in sites:
                selected = predictions[
                    predictions.site_id.eq(site)
                    & predictions.horizon.eq(row["horizon"])
                ]
                values.append(
                    equal_cell_rmse(
                        selected[selected.model.eq(row["candidate"])],
                        years=years,
                        seasons=seasons,
                    )
                    - equal_cell_rmse(
                        selected[selected.model.eq(row["reference"])],
                        years=years,
                        seasons=seasons,
                    )
                )
            return float(np.median(np.asarray(values, dtype=float)))

        years = (2021, 2022, 2023)
        seasons = ("DJF", "MAM", "JJA", "SON")
        assert row["equal_12cell_descriptive_median_effect_c"] == effects(
            years=years, seasons=seasons
        )
        for value in row["leave_one_year_equal_cell_descriptive"]:
            assert value["descriptive_median_effect_c"] == effects(
                years=[year for year in years if year != value["omitted_year"]],
                seasons=seasons,
            )
        for value in row["leave_one_season_equal_cell_descriptive"]:
            assert value["descriptive_median_effect_c"] == effects(
                years=years,
                seasons=[
                    season
                    for season in seasons
                    if season != value["omitted_season"]
                ],
            )


def test_model_registry_delete_or_rename_is_rejected() -> None:
    deleted = _fixture()
    deleted["model_registry_by_cohort"] = deepcopy(MODEL_REGISTRY)
    deleted["model_registry_by_cohort"]["temporal"] = MODEL_REGISTRY["temporal"][:-1]
    with pytest.raises(CoverageAuditError, match="model registry"):
        _build(deleted)
    renamed = _fixture()
    renamed["model_registry_by_cohort"] = deepcopy(MODEL_REGISTRY)
    values = list(MODEL_REGISTRY["external"])
    values[-1] = "ThermoRoute-renamed"
    renamed["model_registry_by_cohort"]["external"] = values
    with pytest.raises(CoverageAuditError, match="model registry"):
        _build(renamed)


def test_model_key_audit_delete_or_summary_replacement_is_rejected() -> None:
    deleted = _fixture()
    deleted["model_key_audits_by_cohort"]["temporal"].pop()
    with pytest.raises(CoverageAuditError, match="registry"):
        _build(deleted)
    one_changed = _fixture()
    one_changed["model_key_audits_by_cohort"]["temporal"][0][
        "forecast_key_sha256"
    ] = "a" * 64
    with pytest.raises(CoverageAuditError, match="share exact"):
        _build(one_changed)
    all_changed = _fixture()
    for row in all_changed["model_key_audits_by_cohort"]["external"]:
        row["forecast_key_sha256"] = "b" * 64
    with pytest.raises(CoverageAuditError, match="differs from observability"):
        _build(all_changed)
    all_y_changed = _fixture()
    for row in all_y_changed["model_key_audits_by_cohort"]["temporal"]:
        row["y_true_sha256"] = "d" * 64
    with pytest.raises(CoverageAuditError, match="comparison projection differs"):
        _build(all_y_changed)


@pytest.mark.parametrize("value", [True, complex(1, 0), 1 << 70])
def test_model_key_audit_row_count_requires_strict_int64(value: object) -> None:
    arguments = _fixture()
    arguments["model_key_audits_by_cohort"]["temporal"][0]["row_count"] = value
    with pytest.raises(CoverageAuditError, match="model-key row count"):
        _build(arguments)


def test_zero_key_horizon_and_empty_external_cohort_are_legal() -> None:
    sites = {"temporal": ("t1", "t2"), "external": ()}

    def observed(_cohort: str, _site: str, date: pd.Timestamp) -> bool:
        return 1 <= date.day <= 5

    arguments = _fixture(sites=sites, observed_rule=observed)
    document = _build(arguments)
    availability = arguments["availability"]
    assert availability[
        availability.cohort.eq("temporal") & availability.horizon.eq(7)
    ].n_valid_targets.eq(0).all()
    assert not any(
        row["cohort"] == "external" for row in document["coverage_cells"]
    )
    external_audits = arguments["model_key_audits_by_cohort"]["external"]
    assert external_audits and all(row["row_count"] == 0 for row in external_audits)
    h7 = [
        row
        for row in document["comparison_sensitivities"]
        if row["horizon"] == 7
    ]
    assert h7 and all(row["n_primary_reportable_stations"] == 0 for row in h7)
    for row in h7:
        assert len(row["frozen_sensitivity_candidates"]) == 8
        assert all(
            candidate["descriptive_median_effect_c"] is None
            for candidate in row["frozen_sensitivity_candidates"]
        )
        assert row["frozen_worst_unfavorable_sensitivity"] == {
            "status": "DESCRIPTIVE_NOT_ESTIMABLE_NO_ELIGIBLE_STATION",
            "direction": "LARGER_CANDIDATE_MINUS_REFERENCE_IS_MORE_UNFAVORABLE",
            "tie_rule": "FIRST_IN_FROZEN_CANDIDATE_ORDER",
            "frozen_order_index": None,
            "source": None,
            "descriptive_median_effect_c": None,
        }


def test_empty_cohort_requires_canonical_empty_y_true_digest() -> None:
    arguments = _fixture(sites={"temporal": ("t1", "t2"), "external": ()})
    for row in arguments["model_key_audits_by_cohort"]["external"]:
        row["y_true_sha256"] = "f" * 64
    with pytest.raises(CoverageAuditError, match="noncanonical digest"):
        _build(arguments)


def test_empty_temporal_cohort_is_explicit_and_legal() -> None:
    arguments = _fixture(sites={"temporal": (), "external": ("e1",)})
    document = _build(arguments)
    assert arguments["temporal_comparison_predictions"].empty
    assert all(
        row["formal_statistics_status"]
        == "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS"
        and row["prediction_derived_descriptive_median_effect_c"] is None
        and row["temporal_reweighting_status"]
        == "DESCRIPTIVE_NOT_ESTIMABLE_NO_STATION_WITH_ALL_12_CELLS_NONEMPTY"
        for row in document["comparison_sensitivities"]
    )


def test_not_estimable_formal_row_reports_separate_descriptive_effect() -> None:
    arguments = _fixture(force_one_cluster=True)
    document = _build(arguments)
    for row in document["comparison_sensitivities"]:
        assert row["formal_statistics_status"] == (
            "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS"
        )
        assert row["formal_median_effect_c"] is None
        assert row["prediction_derived_descriptive_median_effect_c"] is not None
        assert row[
            "prediction_derived_descriptive_effect_does_not_upgrade_inference"
        ] is True


def test_not_estimable_formal_row_cannot_carry_a_formal_effect() -> None:
    arguments = _fixture(force_one_cluster=True)
    arguments["primary_statistics"][0]["median_effect_c"] = 0.0
    with pytest.raises(CoverageAuditError, match="not-estimable"):
        _build(arguments)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("horizon", True),
        ("n_valid_targets", complex(1, 0)),
        ("n_valid_targets", 1 << 70),
        ("reportable", complex(1, 0)),
    ],
)
def test_availability_integer_and_boolean_attacks_fail(
    column: str, value: object
) -> None:
    arguments = _fixture()
    arguments["availability"][column] = arguments["availability"][column].astype(
        object
    )
    arguments["availability"].loc[0, column] = value
    with pytest.raises(CoverageAuditError, match="availability"):
        _build(arguments)


def test_observability_complex_flag_is_not_coerced_to_boolean() -> None:
    arguments = _fixture()
    frame = arguments["observability_by_cohort"]["temporal"]
    frame["wtemp_observed"] = frame["wtemp_observed"].astype(object)
    frame.loc[0, "wtemp_observed"] = complex(1, 0)
    with pytest.raises(CoverageAuditError, match="observability flag"):
        _build(arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_stations", True),
        ("n_clusters", complex(1, 0)),
        ("n_stations", 1 << 70),
        ("median_effect_c", np.inf),
        ("status", "UNKNOWN"),
    ],
)
def test_primary_statistics_nonfinite_type_and_status_attacks_fail(
    field: str, value: object
) -> None:
    arguments = _fixture()
    arguments["primary_statistics"][0][field] = value
    with pytest.raises(CoverageAuditError, match="primary"):
        _build(arguments)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("horizon", True),
        ("horizon", 1.5),
        ("horizon", complex(1, 0)),
        ("horizon", 1 << 70),
        ("y_true", np.inf),
        ("y_pred", complex(1, 0)),
    ],
)
def test_prediction_nonfinite_and_type_attacks_fail(
    column: str, value: object
) -> None:
    arguments = _fixture()
    arguments["temporal_comparison_predictions"][column] = arguments[
        "temporal_comparison_predictions"
    ][column].astype(object)
    arguments["temporal_comparison_predictions"].loc[0, column] = value
    with pytest.raises(CoverageAuditError, match="comparison prediction"):
        _build(arguments)


def test_exact_y_true_digest_rejects_cross_model_change() -> None:
    arguments = _fixture()
    predictions = arguments["temporal_comparison_predictions"]
    mask = predictions.model.eq("LightGBM")
    predictions.loc[mask & (predictions.index == predictions[mask].index[0]), "y_true"] += 1e-9
    with pytest.raises(CoverageAuditError, match="comparison projection differs"):
        _build(arguments)


def test_declared_projection_ignores_extra_columns_but_binding_is_retained() -> None:
    arguments = _fixture()
    first = _build(arguments)
    projected = deepcopy(arguments)
    projected["temporal_comparison_predictions"]["another_unconsumed_column"] = (
        "anything"
    )
    for row in projected["formal_tests"]:
        row["upstream_protocol_field"] = "bound-by-protocol-sha"
    for row in projected["primary_statistics"]:
        row["upstream_ci_field"] = "bound-by-statistics-sha"
    assert _build(projected) == first
    assert first["source_bindings"] == arguments["source_bindings"]


def test_input_order_is_irrelevant_and_exact_replay_validates() -> None:
    arguments = _fixture()
    document = _build(arguments)
    shuffled = deepcopy(arguments)
    for cohort, frame in shuffled["observability_by_cohort"].items():
        shuffled["observability_by_cohort"][cohort] = frame.sample(
            frac=1.0, random_state=11
        ).reset_index(drop=True)
    shuffled["temporal_comparison_predictions"] = shuffled[
        "temporal_comparison_predictions"
    ].sample(frac=1.0, random_state=13).reset_index(drop=True)
    shuffled["availability"] = shuffled["availability"].sample(
        frac=1.0, random_state=17
    ).reset_index(drop=True)
    assert _build(shuffled) == document
    assert validate_temporal_coverage_audit(document, **arguments) == document


def test_self_hash_rewrite_cannot_validate_changed_evidence() -> None:
    arguments = _fixture()
    document = _build(arguments)
    attacked = deepcopy(document)
    attacked["coverage_cells"][0]["n_valid_keys"] -= 1
    with pytest.raises(CoverageAuditError, match="self-hash changed"):
        validate_temporal_coverage_audit(attacked, **arguments)
    stable = deepcopy(attacked)
    stable.pop("audit_self_sha256")
    attacked["audit_self_sha256"] = sha256_json(stable)
    with pytest.raises(CoverageAuditError, match="stale or tampered"):
        validate_temporal_coverage_audit(attacked, **arguments)


def test_changed_source_binding_requires_exact_replay_input() -> None:
    arguments = _fixture()
    document = _build(arguments)
    attacked = deepcopy(document)
    attacked["source_bindings"]["statistics"]["sha256"] = "c" * 64
    stable = deepcopy(attacked)
    stable.pop("audit_self_sha256")
    attacked["audit_self_sha256"] = sha256_json(stable)
    with pytest.raises(CoverageAuditError, match="stale or tampered"):
        validate_temporal_coverage_audit(attacked, **arguments)


def test_estimable_effect_tamper_is_rejected_exactly() -> None:
    arguments = _fixture()
    arguments["primary_statistics"][0]["median_effect_c"] += 1e-15
    with pytest.raises(CoverageAuditError, match="exact crosscheck"):
        _build(arguments)
