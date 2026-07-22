"""Frozen temporal-coverage audit for Route A.

This module contains no data discovery, file-system traversal, network access, or
production outcome paths.  A caller must pass explicit in-memory observability,
prediction, availability, formal-test, and primary-statistic tables.  The design
is frozen before outcome opening; the deterministic document is built only after
an authorised caller supplies those tables.

The audit is deliberately non-filtering.  It reconstructs every
cohort-by-site-by-horizon-by-year-by-season cell, checks the existing >=100-key
reportability registry, and reports temporal-composition sensitivities for the
same five formal comparisons.  It never changes the primary station set or
emits a p-value, confidence interval, multiplicity adjustment, or decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .repro import sha256_json


POLICY_FORMAT = "thermoroute.route-a-temporal-coverage-policy.v1"
AUDIT_FORMAT = "thermoroute.route-a-temporal-coverage-audit.v1"
POLICY_RELATIVE = "protocols/route_a_temporal_coverage_policy_v1.json"
POLICY_ID = "route-a-temporal-coverage-audit-001"

COHORT_ORDER = ("temporal", "external")
HORIZONS = (1, 3, 7)
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
CONTEXT_LENGTH_DAYS = 32
MINIMUM_VALID_TARGETS = 100
FORMAL_TEST_COUNT = 5

FORECAST_KEY = ("site_id", "horizon", "issue_date", "target_date")
PREDICTION_COLUMNS = (
    "model",
    "site_id",
    "horizon",
    "issue_date",
    "target_date",
    "y_true",
    "y_pred",
)
OBSERVABILITY_COLUMNS = ("site_id", "date", "wtemp_observed")
AVAILABILITY_COLUMNS = (
    "cohort",
    "site_no",
    "horizon",
    "n_valid_targets",
    "reportable",
)
COVERAGE_CELL_COLUMNS = (
    "cohort",
    "site_no",
    "horizon",
    "target_year",
    "target_season",
    "n_calendar_opportunities",
    "n_issue_wtemp_observed",
    "n_target_wtemp_observed",
    "n_valid_keys",
)


class CoverageAuditError(RuntimeError):
    """The frozen coverage policy, inputs, or deterministic result is invalid."""


def _expected_policy_stable() -> dict[str, Any]:
    return {
        "format": POLICY_FORMAT,
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "policy_id": POLICY_ID,
        "recorded_date": "2026-07-22",
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent_design": True,
        "scope": {
            "cohorts": list(COHORT_ORDER),
            "coverage_scope": "both_frozen_cohorts",
            "comparison_sensitivity_scope": "temporal_primary_five_tests_only",
            "not_a_missing_at_random_assessment": True,
            "not_all_calendar_day_performance": True,
        },
        "calendar_contract": {
            "classification_date": "target_date",
            "year_rule": "Gregorian_calendar_year_of_target_date",
            "interval_rule": (
                "caller_supplies_one_frozen_inclusive_interval_covering_exactly_"
                "three_complete_Gregorian_years"
            ),
            "required_year_count": 3,
            "season_order": list(SEASON_ORDER),
            "season_months": {
                key: list(SEASON_MONTHS[key]) for key in SEASON_ORDER
            },
            "season_role": (
                "deterministic_meteorological_reporting_partition_not_ecological_"
                "life_stage_or_watershed_specific_season"
            ),
            "horizons_days": list(HORIZONS),
            "context_length_days": CONTEXT_LENGTH_DAYS,
            "calendar_opportunity": (
                "issue_and_target_inside_interval_and_exact_daily_context_rows_present"
            ),
            "valid_key": (
                "calendar_opportunity_and_issue_WTEMP_observed_and_target_WTEMP_observed"
            ),
        },
        "coverage_contract": {
            "minimum_valid_targets_per_station_horizon": MINIMUM_VALID_TARGETS,
            "all_site_horizon_year_season_cells_required_including_zero": True,
            "availability_registry_must_reconstruct_exactly": True,
            "all_models_must_share_exact_forecast_keys": True,
            "availability_may_not_replace_or_filter_a_frozen_site": True,
            "coverage_balance_changes_primary_reportability": False,
        },
        "sensitivity_contract": {
            "formal_comparison_count": FORMAL_TEST_COUNT,
            "role": "DESCRIPTIVE_NOT_IN_CONFIRMATORY_FAMILY",
            "primary_effect_crosscheck": (
                "exact_median_of_reportable_station_RMSE_candidate_minus_reference"
            ),
            "complete_support": (
                "primary_reportable_station_has_at_least_one_valid_key_in_each_of_"
                "three_years_by_four_seasons"
            ),
            "equal_cell_station_rmse": (
                "sqrt_of_unweighted_mean_of_the_twelve_within_cell_MSE_values"
            ),
            "leave_one_year": (
                "same_complete_support_stations_and_equal_weight_over_remaining_"
                "eight_cells"
            ),
            "leave_one_season": (
                "same_complete_support_stations_and_equal_weight_over_remaining_"
                "nine_cells"
            ),
            "primary_station_set_or_statistic_changed": False,
            "retraining_or_recalibration_performed": False,
            "favorable_sensitivity_may_rescue_primary_result": False,
            "unfavorable_sensitivity_may_hide_primary_result": False,
        },
        "prohibited_outputs": [
            "p_value",
            "confidence_interval",
            "Holm_adjustment",
            "pass_fail_decision",
        ],
        "integrity_contract": {
            "canonical_semantic_digest_for_every_consumed_input": True,
            "canonical_document_self_hash": True,
            "exact_recomputation_required": True,
            "missing_unknown_duplicate_or_tampered_evidence_fails_closed": True,
        },
        "output_artifact": "trusted/temporal_coverage_audit_v1.json",
    }


def _validated_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(dict(policy))
    self_hash = document.pop("policy_self_sha256", None)
    expected = _expected_policy_stable()
    if document != expected:
        raise CoverageAuditError("temporal-coverage policy semantics changed")
    if (
        not isinstance(self_hash, str)
        or len(self_hash) != 64
        or sha256_json(document) != self_hash
    ):
        raise CoverageAuditError("temporal-coverage policy self-hash changed")
    return {**document, "policy_self_sha256": self_hash}


def validate_temporal_coverage_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate the exact frozen, outcome-free coverage policy."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoverageAuditError(
            "temporal-coverage policy is absent or malformed"
        ) from exc
    if not isinstance(value, dict):
        raise CoverageAuditError("temporal-coverage policy is not a JSON object")
    return _validated_policy(value)


def _parse_date(value: object, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise CoverageAuditError(f"{label} is not a valid date") from exc
    if parsed.tzinfo is not None or parsed != parsed.normalize():
        raise CoverageAuditError(f"{label} must be a timezone-free calendar date")
    return parsed


def _normalize_date_column(values: pd.Series, *, label: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise CoverageAuditError(f"{label} contains an invalid date") from exc
    if not isinstance(parsed, pd.Series):  # pragma: no cover - Series in public API
        parsed = pd.Series(parsed, index=values.index)
    if parsed.dt.tz is not None or not parsed.eq(parsed.dt.normalize()).all():
        raise CoverageAuditError(
            f"{label} must contain timezone-free calendar dates"
        )
    return parsed.astype("datetime64[ns]")


def _normalize_bool(values: pd.Series, *, label: str) -> pd.Series:
    mapped = values.map(
        {
            True: True,
            False: False,
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
        }
    )
    if mapped.isna().any():
        raise CoverageAuditError(f"{label} contains a non-boolean value")
    return mapped.astype(bool)


def _normalize_integer(values: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise CoverageAuditError(f"{label} contains a non-numeric value") from exc
    array = numeric.to_numpy(float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise CoverageAuditError(f"{label} contains a non-integral value")
    return numeric.astype("int64")


def _semantic_frame_digest(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    date_columns: frozenset[str] = frozenset(),
    float_columns: frozenset[str] = frozenset(),
    boolean_columns: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    canonical = frame.loc[:, list(columns)].copy()
    for column in columns:
        if column in date_columns:
            canonical[column] = pd.to_datetime(canonical[column]).dt.strftime(
                "%Y-%m-%d"
            )
        elif column in float_columns:
            canonical[column] = canonical[column].map(
                lambda value: format(float(value), ".17g")
            )
        elif column in boolean_columns:
            canonical[column] = canonical[column].map(
                lambda value: "true" if bool(value) else "false"
            )
        else:
            canonical[column] = canonical[column].astype(str)
    canonical = canonical.sort_values(list(columns), kind="mergesort").reset_index(
        drop=True
    )
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "canonical_columns": list(columns),
        "row_count": int(len(canonical)),
        "canonical_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _normalize_sites(
    sites_by_cohort: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(sites_by_cohort) != set(COHORT_ORDER):
        raise CoverageAuditError("coverage audit requires both frozen cohorts")
    output: dict[str, tuple[str, ...]] = {}
    for cohort in COHORT_ORDER:
        sites = tuple(sorted(str(site).strip() for site in sites_by_cohort[cohort]))
        if not sites or any(not site for site in sites) or len(sites) != len(set(sites)):
            raise CoverageAuditError(f"{cohort} frozen site registry is invalid")
        output[cohort] = sites
    if set(output["temporal"]) & set(output["external"]):
        raise CoverageAuditError("temporal and external frozen sites must be disjoint")
    return output


def _normalize_observability(
    frame: pd.DataFrame,
    *,
    cohort: str,
    sites: tuple[str, ...],
    history_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.DataFrame:
    if set(frame) != set(OBSERVABILITY_COLUMNS):
        raise CoverageAuditError(
            f"{cohort} observability has a non-frozen schema"
        )
    output = frame.loc[:, list(OBSERVABILITY_COLUMNS)].copy()
    output["site_id"] = output.site_id.astype(str).str.strip()
    output["date"] = _normalize_date_column(
        output.date, label=f"{cohort} observability"
    )
    output["wtemp_observed"] = _normalize_bool(
        output.wtemp_observed, label=f"{cohort} observability"
    )
    if output.duplicated(["site_id", "date"]).any():
        raise CoverageAuditError(f"{cohort} observability duplicates a site/date")
    expected_dates = pd.date_range(history_start, target_end, freq="D")
    expected = pd.MultiIndex.from_product(
        [sites, expected_dates], names=["site_id", "date"]
    )
    actual = pd.MultiIndex.from_frame(output[["site_id", "date"]])
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise CoverageAuditError(
            f"{cohort} observability does not contain the exact daily calendar"
        )
    return output.sort_values(["site_id", "date"], kind="mergesort").reset_index(
        drop=True
    )


def _normalize_predictions(
    frame: pd.DataFrame,
    *,
    cohort: str,
    known_sites: set[str],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(frame)
    if missing:
        raise CoverageAuditError(
            f"{cohort} predictions lack required columns: {sorted(missing)}"
        )
    output = frame.loc[:, list(PREDICTION_COLUMNS)].copy()
    output["model"] = output.model.astype(str).str.strip()
    output["site_id"] = output.site_id.astype(str).str.strip()
    output["horizon"] = _normalize_integer(
        output.horizon, label=f"{cohort} prediction horizon"
    )
    output["issue_date"] = _normalize_date_column(
        output.issue_date, label=f"{cohort} prediction issue_date"
    )
    output["target_date"] = _normalize_date_column(
        output.target_date, label=f"{cohort} prediction target_date"
    )
    for column in ("y_true", "y_pred"):
        try:
            output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise CoverageAuditError(
                f"{cohort} predictions contain a non-numeric {column}"
            ) from exc
        if not np.isfinite(output[column].to_numpy(float)).all():
            raise CoverageAuditError(
                f"{cohort} predictions contain a non-finite {column}"
            )
    if output.empty or (output.model.eq("") | output.site_id.eq("")).any():
        raise CoverageAuditError(f"{cohort} predictions are empty or unnamed")
    if not set(output.site_id) <= known_sites:
        raise CoverageAuditError(f"{cohort} predictions contain an unknown site")
    if set(output.horizon) != set(HORIZONS):
        raise CoverageAuditError(f"{cohort} predictions omit or add a horizon")
    if (
        output.issue_date.lt(target_start).any()
        or output.target_date.gt(target_end).any()
        or not (
            output.target_date
            - output.issue_date
            == pd.to_timedelta(output.horizon, unit="D")
        ).all()
    ):
        raise CoverageAuditError(f"{cohort} prediction key leaves the interval")
    identity = ["model", *FORECAST_KEY]
    if output.duplicated(identity).any():
        raise CoverageAuditError(f"{cohort} predictions duplicate a model/key")
    target_counts = output.groupby(list(FORECAST_KEY), sort=False).y_true.nunique(
        dropna=False
    )
    if not target_counts.eq(1).all():
        raise CoverageAuditError(f"{cohort} models disagree on y_true")
    return output.sort_values(identity, kind="mergesort").reset_index(drop=True)


def _normalize_availability(
    frame: pd.DataFrame,
    *,
    sites: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    if set(frame) != set(AVAILABILITY_COLUMNS):
        raise CoverageAuditError("availability registry has a non-frozen schema")
    output = frame.loc[:, list(AVAILABILITY_COLUMNS)].copy()
    output["cohort"] = output.cohort.astype(str).str.strip()
    output["site_no"] = output.site_no.astype(str).str.strip()
    output["horizon"] = _normalize_integer(
        output.horizon, label="availability horizon"
    )
    output["n_valid_targets"] = _normalize_integer(
        output.n_valid_targets, label="availability target count"
    )
    output["reportable"] = _normalize_bool(
        output.reportable, label="availability reportable flag"
    )
    expected = {
        (cohort, site, horizon)
        for cohort in COHORT_ORDER
        for site in sites[cohort]
        for horizon in HORIZONS
    }
    actual = set(
        output[["cohort", "site_no", "horizon"]].itertuples(index=False, name=None)
    )
    if actual != expected or output.duplicated(
        ["cohort", "site_no", "horizon"]
    ).any():
        raise CoverageAuditError("availability omits or duplicates a frozen cell")
    if output.n_valid_targets.lt(0).any() or not np.array_equal(
        output.reportable.to_numpy(bool),
        output.n_valid_targets.ge(MINIMUM_VALID_TARGETS).to_numpy(bool),
    ):
        raise CoverageAuditError("availability reportability disagrees with policy")
    return output.sort_values(
        ["cohort", "site_no", "horizon"], kind="mergesort"
    ).reset_index(drop=True)


def _normalize_formal_tests(
    formal_tests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(formal_tests) != FORMAL_TEST_COUNT:
        raise CoverageAuditError("coverage sensitivity requires exactly five tests")
    output: list[dict[str, Any]] = []
    for item in formal_tests:
        if not isinstance(item, Mapping):
            raise CoverageAuditError("formal comparison is not an object")
        missing = {"test_id", "candidate", "reference", "horizon", "margin_c"} - set(
            item
        )
        if missing:
            raise CoverageAuditError(
                f"formal comparison lacks fields: {sorted(missing)}"
            )
        try:
            horizon_float = float(item["horizon"])
            margin = float(item["margin_c"])
        except (TypeError, ValueError) as exc:
            raise CoverageAuditError("formal comparison has a non-numeric value") from exc
        if (
            not np.isfinite(horizon_float)
            or horizon_float != int(horizon_float)
            or int(horizon_float) not in HORIZONS
            or not np.isfinite(margin)
        ):
            raise CoverageAuditError("formal comparison has an invalid horizon/margin")
        row = {
            "test_id": str(item["test_id"]).strip(),
            "candidate": str(item["candidate"]).strip(),
            "reference": str(item["reference"]).strip(),
            "horizon": int(horizon_float),
            "margin_c": margin,
        }
        if not row["test_id"] or not row["candidate"] or not row["reference"]:
            raise CoverageAuditError("formal comparison contains an empty identifier")
        output.append(row)
    if len({row["test_id"] for row in output}) != FORMAL_TEST_COUNT:
        raise CoverageAuditError("formal comparison test IDs are not unique")
    return output


def _normalize_statistics_rows(
    statistics_rows: Sequence[Mapping[str, Any]],
    *,
    tests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(statistics_rows) != FORMAL_TEST_COUNT:
        raise CoverageAuditError("primary statistics do not contain five rows")
    by_test: dict[str, dict[str, Any]] = {}
    for item in statistics_rows:
        if not isinstance(item, Mapping) or not {
            "test_id",
            "median_effect_c",
            "n_stations",
        } <= set(item):
            raise CoverageAuditError("primary statistic row has a non-frozen schema")
        test_id = str(item["test_id"])
        if test_id in by_test:
            raise CoverageAuditError("primary statistics duplicate a test ID")
        count_value = item["n_stations"]
        if isinstance(count_value, bool):
            raise CoverageAuditError("primary statistic station count is invalid")
        try:
            count_float = float(count_value)
        except (TypeError, ValueError) as exc:
            raise CoverageAuditError("primary statistic station count is invalid") from exc
        if (
            not np.isfinite(count_float)
            or count_float < 0
            or count_float != int(count_float)
        ):
            raise CoverageAuditError("primary statistic station count is invalid")
        effect_value = item["median_effect_c"]
        if effect_value is None:
            effect = None
        else:
            try:
                effect = float(effect_value)
            except (TypeError, ValueError) as exc:
                raise CoverageAuditError("primary effect is non-numeric") from exc
            if not np.isfinite(effect):
                raise CoverageAuditError("primary effect is non-finite")
        by_test[test_id] = {
            "test_id": test_id,
            "median_effect_c": effect,
            "n_stations": int(count_float),
        }
    expected_ids = [str(test["test_id"]) for test in tests]
    if set(by_test) != set(expected_ids):
        raise CoverageAuditError("primary statistics test registry changed")
    return [by_test[test_id] for test_id in expected_ids]


def _season_for_month(month: int) -> str:
    for season in SEASON_ORDER:
        if month in SEASON_MONTHS[season]:
            return season
    raise CoverageAuditError("target month is outside the Gregorian calendar")


def _coverage_cells_and_keys(
    observability: Mapping[str, pd.DataFrame],
    *,
    sites: Mapping[str, tuple[str, ...]],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    years: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[Any, ...]]]]:
    rows: list[dict[str, Any]] = []
    valid_keys: dict[str, set[tuple[Any, ...]]] = {
        cohort: set() for cohort in COHORT_ORDER
    }
    for cohort in COHORT_ORDER:
        indexed = observability[cohort].set_index(["site_id", "date"])[
            "wtemp_observed"
        ]
        for site in sites[cohort]:
            site_values = indexed.loc[site]
            for horizon in HORIZONS:
                issue_dates = pd.date_range(
                    target_start, target_end - pd.Timedelta(days=horizon), freq="D"
                )
                target_dates = issue_dates + pd.Timedelta(days=horizon)
                issue_observed = site_values.reindex(issue_dates).to_numpy(bool)
                target_observed = site_values.reindex(target_dates).to_numpy(bool)
                valid = issue_observed & target_observed
                for issue_date, target_date in zip(
                    issue_dates[valid], target_dates[valid], strict=True
                ):
                    valid_keys[cohort].add(
                        (site, horizon, pd.Timestamp(issue_date), pd.Timestamp(target_date))
                    )
                cell_frame = pd.DataFrame(
                    {
                        "target_year": target_dates.year,
                        "target_season": [
                            _season_for_month(int(month)) for month in target_dates.month
                        ],
                        "issue_observed": issue_observed,
                        "target_observed": target_observed,
                        "valid": valid,
                    }
                )
                for year in years:
                    for season in SEASON_ORDER:
                        selected = cell_frame[
                            cell_frame.target_year.eq(year)
                            & cell_frame.target_season.eq(season)
                        ]
                        rows.append(
                            {
                                "cohort": cohort,
                                "site_no": site,
                                "horizon": horizon,
                                "target_year": year,
                                "target_season": season,
                                "n_calendar_opportunities": int(len(selected)),
                                "n_issue_wtemp_observed": int(
                                    selected.issue_observed.sum()
                                ),
                                "n_target_wtemp_observed": int(
                                    selected.target_observed.sum()
                                ),
                                "n_valid_keys": int(selected.valid.sum()),
                            }
                        )
    return rows, valid_keys


def _prediction_key_set(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
    return set(frame.loc[:, list(FORECAST_KEY)].itertuples(index=False, name=None))


def _validate_common_prediction_keys(
    predictions: Mapping[str, pd.DataFrame],
    *,
    expected_keys: Mapping[str, set[tuple[Any, ...]]],
) -> None:
    for cohort in COHORT_ORDER:
        frame = predictions[cohort]
        key_sets = {
            str(model): _prediction_key_set(group)
            for model, group in frame.groupby("model", sort=True)
        }
        if not key_sets:
            raise CoverageAuditError(f"{cohort} has no prediction models")
        first = next(iter(key_sets.values()))
        if any(keys != first for keys in key_sets.values()):
            raise CoverageAuditError(f"{cohort} models do not share exact keys")
        if first != expected_keys[cohort]:
            raise CoverageAuditError(
                f"{cohort} prediction keys differ from frozen observability"
            )


def _validate_coverage_counts(
    coverage_rows: Sequence[Mapping[str, Any]],
    *,
    availability: pd.DataFrame,
) -> dict[tuple[str, str, int], bool]:
    totals: dict[tuple[str, str, int], int] = {}
    for row in coverage_rows:
        possible = int(row["n_calendar_opportunities"])
        issue = int(row["n_issue_wtemp_observed"])
        target = int(row["n_target_wtemp_observed"])
        valid = int(row["n_valid_keys"])
        neither = possible - issue - target + valid
        if not (
            0 <= valid <= issue <= possible
            and 0 <= valid <= target <= possible
            and neither >= 0
        ):
            raise CoverageAuditError("coverage cell has impossible observability counts")
        key = (str(row["cohort"]), str(row["site_no"]), int(row["horizon"]))
        totals[key] = totals.get(key, 0) + valid
    reportable: dict[tuple[str, str, int], bool] = {}
    for cohort, site, horizon, count, is_reportable in availability.itertuples(
        index=False, name=None
    ):
        key = (str(cohort), str(site), int(horizon))
        if totals.get(key) != int(count):
            raise CoverageAuditError(
                "coverage cells do not reconstruct the availability registry"
            )
        reportable[key] = bool(is_reportable)
    if set(totals) != set(reportable):
        raise CoverageAuditError("coverage totals omit an availability cell")
    return reportable


def _rmse(group: pd.DataFrame) -> float:
    ordered = group.sort_values(["issue_date", "target_date"], kind="mergesort")
    error = ordered.y_pred.to_numpy(float) - ordered.y_true.to_numpy(float)
    return float(np.sqrt(np.mean(np.square(error))))


def _station_effects(
    predictions: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    horizon: int,
    sites: Sequence[str],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for site in sorted(sites):
        selected = predictions[
            predictions.site_id.eq(site) & predictions.horizon.eq(horizon)
        ]
        candidate_rows = selected[selected.model.eq(candidate)]
        reference_rows = selected[selected.model.eq(reference)]
        if candidate_rows.empty or reference_rows.empty:
            raise CoverageAuditError("formal comparison lacks a reportable station")
        output[site] = _rmse(candidate_rows) - _rmse(reference_rows)
    return output


def _median_effect(effects: Mapping[str, float]) -> float | None:
    if not effects:
        return None
    return float(np.median(np.asarray([effects[site] for site in sorted(effects)])))


def _equal_cell_rmse(
    group: pd.DataFrame,
    *,
    years: Sequence[int],
    seasons: Sequence[str],
) -> float:
    working = group.copy()
    working["target_year"] = working.target_date.dt.year
    working["target_season"] = working.target_date.dt.month.map(_season_for_month)
    error = working.y_pred.to_numpy(float) - working.y_true.to_numpy(float)
    working["squared_error"] = np.square(error)
    values: list[float] = []
    for year in years:
        for season in seasons:
            selected = working[
                working.target_year.eq(year)
                & working.target_season.eq(season)
            ]
            if selected.empty:
                raise CoverageAuditError(
                    "complete-support sensitivity unexpectedly lacks a cell"
                )
            values.append(float(selected.squared_error.mean()))
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float))))


def _equal_cell_station_effects(
    predictions: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    horizon: int,
    sites: Sequence[str],
    years: Sequence[int],
    seasons: Sequence[str],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for site in sorted(sites):
        selected = predictions[
            predictions.site_id.eq(site) & predictions.horizon.eq(horizon)
        ]
        candidate_rows = selected[selected.model.eq(candidate)]
        reference_rows = selected[selected.model.eq(reference)]
        output[site] = _equal_cell_rmse(
            candidate_rows, years=years, seasons=seasons
        ) - _equal_cell_rmse(reference_rows, years=years, seasons=seasons)
    return output


def _comparison_sensitivities(
    *,
    predictions: pd.DataFrame,
    coverage_rows: Sequence[Mapping[str, Any]],
    reportable: Mapping[tuple[str, str, int], bool],
    tests: Sequence[Mapping[str, Any]],
    statistics: Sequence[Mapping[str, Any]],
    years: tuple[int, ...],
) -> list[dict[str, Any]]:
    coverage = pd.DataFrame(coverage_rows)
    temporal_models = set(predictions.model.astype(str))
    statistics_by_test = {str(row["test_id"]): row for row in statistics}
    output: list[dict[str, Any]] = []
    for test in tests:
        test_id = str(test["test_id"])
        candidate = str(test["candidate"])
        reference = str(test["reference"])
        horizon = int(test["horizon"])
        if candidate not in temporal_models or reference not in temporal_models:
            raise CoverageAuditError("formal comparison model is absent")
        primary_sites = sorted(
            site
            for cohort, site, value_horizon in reportable
            if cohort == "temporal"
            and value_horizon == horizon
            and reportable[(cohort, site, value_horizon)]
        )
        primary_effects = _station_effects(
            predictions,
            candidate=candidate,
            reference=reference,
            horizon=horizon,
            sites=primary_sites,
        )
        primary_effect = _median_effect(primary_effects)
        expected = statistics_by_test[test_id]
        if int(expected["n_stations"]) != len(primary_sites):
            raise CoverageAuditError("primary station count crosscheck failed")
        if expected["median_effect_c"] != primary_effect:
            raise CoverageAuditError("primary effect exact crosscheck failed")

        complete_sites: list[str] = []
        for site in primary_sites:
            selected = coverage[
                coverage.cohort.eq("temporal")
                & coverage.site_no.eq(site)
                & coverage.horizon.eq(horizon)
            ]
            if len(selected) != len(years) * len(SEASON_ORDER):
                raise CoverageAuditError("coverage sensitivity cell registry changed")
            if selected.n_valid_keys.gt(0).all():
                complete_sites.append(site)

        primary_complete = _median_effect(
            _station_effects(
                predictions,
                candidate=candidate,
                reference=reference,
                horizon=horizon,
                sites=complete_sites,
            )
        )
        if complete_sites:
            equal_effect = _median_effect(
                _equal_cell_station_effects(
                    predictions,
                    candidate=candidate,
                    reference=reference,
                    horizon=horizon,
                    sites=complete_sites,
                    years=years,
                    seasons=SEASON_ORDER,
                )
            )
            leave_year = [
                {
                    "omitted_year": year,
                    "median_effect_c": _median_effect(
                        _equal_cell_station_effects(
                            predictions,
                            candidate=candidate,
                            reference=reference,
                            horizon=horizon,
                            sites=complete_sites,
                            years=[value for value in years if value != year],
                            seasons=SEASON_ORDER,
                        )
                    ),
                }
                for year in years
            ]
            leave_season = [
                {
                    "omitted_season": season,
                    "median_effect_c": _median_effect(
                        _equal_cell_station_effects(
                            predictions,
                            candidate=candidate,
                            reference=reference,
                            horizon=horizon,
                            sites=complete_sites,
                            years=years,
                            seasons=[value for value in SEASON_ORDER if value != season],
                        )
                    ),
                }
                for season in SEASON_ORDER
            ]
            status = "ESTIMABLE_DESCRIPTIVE"
        else:
            equal_effect = None
            leave_year = [
                {"omitted_year": year, "median_effect_c": None} for year in years
            ]
            leave_season = [
                {"omitted_season": season, "median_effect_c": None}
                for season in SEASON_ORDER
            ]
            status = "NOT_ESTIMABLE_NO_COMPLETE_12CELL_SUPPORT"
        output.append(
            {
                "test_id": test_id,
                "candidate": candidate,
                "reference": reference,
                "horizon": horizon,
                "margin_c": float(test["margin_c"]),
                "status": status,
                "n_primary_reportable_stations": len(primary_sites),
                "primary_median_effect_c": primary_effect,
                "n_complete_12cell_stations": len(complete_sites),
                "primary_median_effect_complete_support_c": primary_complete,
                "equal_12cell_median_effect_c": equal_effect,
                "leave_one_year_equal_cell": leave_year,
                "leave_one_season_equal_cell": leave_season,
            }
        )
    return output


def _assert_no_prohibited_sensitivity_fields(
    comparisons: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> None:
    prohibited = set(cast(Sequence[str], policy["prohibited_outputs"]))

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            overlap = set(str(key) for key in value) & prohibited
            if overlap:
                raise CoverageAuditError(
                    f"coverage sensitivity contains prohibited outputs: {sorted(overlap)}"
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(list(comparisons))


def build_temporal_coverage_audit(
    *,
    policy: Mapping[str, Any],
    target_start: str | pd.Timestamp,
    target_end: str | pd.Timestamp,
    sites_by_cohort: Mapping[str, Sequence[str]],
    observability_by_cohort: Mapping[str, pd.DataFrame],
    predictions_by_cohort: Mapping[str, pd.DataFrame],
    availability: pd.DataFrame,
    formal_tests: Sequence[Mapping[str, Any]],
    primary_statistics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the complete, deterministic, non-filtering temporal coverage audit.

    ``observability_by_cohort`` contains only ``site_id``, calendar ``date``, and
    a boolean ``wtemp_observed``.  It must cover the 31 context days before the
    interval and every day through ``target_end`` for every frozen site.
    ``predictions_by_cohort`` is an in-memory trusted prediction table; this
    function has no mechanism for finding or opening a production artifact.
    """
    frozen_policy = _validated_policy(policy)
    start = _parse_date(target_start, label="coverage target_start")
    end = _parse_date(target_end, label="coverage target_end")
    if (
        start > end
        or (start.month, start.day) != (1, 1)
        or (end.month, end.day) != (12, 31)
        or end.year - start.year != 2
    ):
        raise CoverageAuditError(
            "coverage interval must contain exactly three complete Gregorian years"
        )
    years = tuple(range(start.year, end.year + 1))
    sites = _normalize_sites(sites_by_cohort)
    if set(observability_by_cohort) != set(COHORT_ORDER):
        raise CoverageAuditError("observability tables omit or add a cohort")
    if set(predictions_by_cohort) != set(COHORT_ORDER):
        raise CoverageAuditError("prediction tables omit or add a cohort")
    history_start = start - pd.Timedelta(days=CONTEXT_LENGTH_DAYS - 1)
    observations = {
        cohort: _normalize_observability(
            observability_by_cohort[cohort],
            cohort=cohort,
            sites=sites[cohort],
            history_start=history_start,
            target_end=end,
        )
        for cohort in COHORT_ORDER
    }
    predictions = {
        cohort: _normalize_predictions(
            predictions_by_cohort[cohort],
            cohort=cohort,
            known_sites=set(sites[cohort]),
            target_start=start,
            target_end=end,
        )
        for cohort in COHORT_ORDER
    }
    availability_frame = _normalize_availability(availability, sites=sites)
    tests = _normalize_formal_tests(formal_tests)
    statistics = _normalize_statistics_rows(primary_statistics, tests=tests)
    coverage_rows, expected_keys = _coverage_cells_and_keys(
        observations,
        sites=sites,
        target_start=start,
        target_end=end,
        years=years,
    )
    _validate_common_prediction_keys(predictions, expected_keys=expected_keys)
    reportable = _validate_coverage_counts(
        coverage_rows, availability=availability_frame
    )
    comparisons = _comparison_sensitivities(
        predictions=predictions["temporal"],
        coverage_rows=coverage_rows,
        reportable=reportable,
        tests=tests,
        statistics=statistics,
        years=years,
    )
    _assert_no_prohibited_sensitivity_fields(comparisons, frozen_policy)

    semantic_inputs = {
        "observability": {
            cohort: _semantic_frame_digest(
                observations[cohort],
                columns=OBSERVABILITY_COLUMNS,
                date_columns=frozenset({"date"}),
                boolean_columns=frozenset({"wtemp_observed"}),
            )
            for cohort in COHORT_ORDER
        },
        "predictions": {
            cohort: _semantic_frame_digest(
                predictions[cohort],
                columns=PREDICTION_COLUMNS,
                date_columns=frozenset({"issue_date", "target_date"}),
                float_columns=frozenset({"y_true", "y_pred"}),
            )
            for cohort in COHORT_ORDER
        },
        "availability": _semantic_frame_digest(
            availability_frame,
            columns=AVAILABILITY_COLUMNS,
            boolean_columns=frozenset({"reportable"}),
        ),
        "formal_tests": {
            "row_count": len(tests),
            "canonical_sha256": sha256_json(tests),
        },
        "primary_statistics": {
            "row_count": len(statistics),
            "canonical_sha256": sha256_json(statistics),
        },
    }
    stable: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "status": "POSTOPEN_TRUSTED_RECOMPUTED",
        "role": "DESCRIPTIVE_COVERAGE_AND_TEMPORAL_COMPOSITION_SENSITIVITY",
        "policy_self_sha256": frozen_policy["policy_self_sha256"],
        "target_interval": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "years": list(years),
        },
        "target_date_partition": {
            "classification_date": "target_date",
            "year_rule": "Gregorian_calendar_year_of_target_date",
            "season_order": list(SEASON_ORDER),
            "season_months": {
                key: list(SEASON_MONTHS[key]) for key in SEASON_ORDER
            },
        },
        "reportability_rule": {
            "minimum_valid_targets_per_station_horizon": MINIMUM_VALID_TARGETS,
            "coverage_audit_changes_primary_reportability": False,
        },
        "semantic_inputs": semantic_inputs,
        "input_semantic_sha256": sha256_json(semantic_inputs),
        "coverage_cells": coverage_rows,
        "comparison_sensitivities": comparisons,
        "primary_statistics_unchanged": True,
        "primary_station_set_unchanged": True,
        "inference_computed": False,
    }
    return {**stable, "audit_self_sha256": sha256_json(stable)}


def validate_temporal_coverage_audit(
    document: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    target_start: str | pd.Timestamp,
    target_end: str | pd.Timestamp,
    sites_by_cohort: Mapping[str, Sequence[str]],
    observability_by_cohort: Mapping[str, pd.DataFrame],
    predictions_by_cohort: Mapping[str, pd.DataFrame],
    availability: pd.DataFrame,
    formal_tests: Sequence[Mapping[str, Any]],
    primary_statistics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute every field and reject a stale, partial, or tampered audit."""
    actual = deepcopy(dict(document))
    self_hash = actual.pop("audit_self_sha256", None)
    if (
        not isinstance(self_hash, str)
        or len(self_hash) != 64
        or sha256_json(actual) != self_hash
    ):
        raise CoverageAuditError("temporal-coverage audit self-hash changed")
    expected = build_temporal_coverage_audit(
        policy=policy,
        target_start=target_start,
        target_end=target_end,
        sites_by_cohort=sites_by_cohort,
        observability_by_cohort=observability_by_cohort,
        predictions_by_cohort=predictions_by_cohort,
        availability=availability,
        formal_tests=formal_tests,
        primary_statistics=primary_statistics,
    )
    if dict(document) != expected:
        raise CoverageAuditError("temporal-coverage audit is stale or tampered")
    return expected
