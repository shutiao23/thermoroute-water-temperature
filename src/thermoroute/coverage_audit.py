"""Outcome-free design and deterministic core for Route-A coverage auditing.

The core has no data discovery, production paths, or network access.  It accepts
explicit in-memory projections plus exact upstream file bindings.  Full upstream
artifacts are closed by those SHA-256 bindings; this module semantically digests
only the declared columns it consumes.

The returned document is not a trusted receipt.  It remains a derived core until
the opening/release wrapper verifies every bound physical file and incorporates
the document into the signed receipt.  Coverage and temporal reweighting never
change the primary station set, formal statistic, or formal result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, cast

import numpy as np
import pandas as pd

from .repro import canonical_json, sha256_json


POLICY_FORMAT = "thermoroute.route-a-temporal-coverage-policy.v1"
AUDIT_FORMAT = "thermoroute.route-a-temporal-coverage-audit.v1"
POLICY_RELATIVE = "protocols/route_a_temporal_coverage_policy_v1.json"
POLICY_ID = "route-a-temporal-coverage-audit-001"
# Filled from the final frozen policy bytes.  It is intentionally a file digest,
# distinct from the semantic self-hash stored inside the policy.
POLICY_FILE_SHA256 = "3146965e3909a41ecf59db5754f84c6f28253d00ccdaca13b87ef9d7b20b5ae3"

COHORT_ORDER = ("temporal", "external")
HORIZONS = (1, 3, 7)
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
TARGET_START = "2021-01-01"
TARGET_END = "2023-12-31"
CONTEXT_LENGTH_DAYS = 32
MINIMUM_VALID_TARGETS = 100

ROUTE_A_FORMAL_TESTS: tuple[dict[str, Any], ...] = (
    {
        "test_id": "H1-h1-vs-damped",
        "candidate": "ThermoRoute",
        "reference": "DampedPersistence",
        "horizon": 1,
        "margin_c": 0.0,
    },
    {
        "test_id": "H1-h3-vs-damped",
        "candidate": "ThermoRoute",
        "reference": "DampedPersistence",
        "horizon": 3,
        "margin_c": 0.0,
    },
    {
        "test_id": "H1-h7-vs-damped",
        "candidate": "ThermoRoute",
        "reference": "DampedPersistence",
        "horizon": 7,
        "margin_c": 0.0,
    },
    {
        "test_id": "H2-h3-vs-lightgbm",
        "candidate": "ThermoRoute",
        "reference": "LightGBM",
        "horizon": 3,
        "margin_c": 0.05,
    },
    {
        "test_id": "H2-h7-vs-lightgbm",
        "candidate": "ThermoRoute",
        "reference": "LightGBM",
        "horizon": 7,
        "margin_c": 0.05,
    },
)

MODEL_REGISTRY: dict[str, tuple[str, ...]] = {
    "temporal": (
        "Persistence",
        "DampedPersistence",
        "Climatology",
        "LightGBM",
        "LSTM",
        "ThermoRoute",
        "DampedPriorOnly",
        "TR-noDynamicPrior",
        "TR-fixedKappa",
        "TR-noRouter",
        "TR-noMoE",
        "TR-noTCN",
        "TR-unbounded",
    ),
    "external": (
        "Persistence",
        "DampedPersistence",
        "Climatology",
        "LightGBM",
        "LSTM",
        "ThermoRoute",
    ),
}
COMPARISON_MODELS = ("ThermoRoute", "DampedPersistence", "LightGBM")

SOURCE_BINDING_KEYS = (
    "policy",
    "protocol",
    "acquisition_manifest",
    "temporal_normalized_outcomes",
    "external_normalized_outcomes",
    "temporal_site_registry",
    "external_site_registry",
    "temporal_full_predictions",
    "external_full_predictions",
    "availability_registry",
    "statistics",
)

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
MODEL_KEY_AUDIT_COLUMNS = (
    "model",
    "row_count",
    "forecast_key_sha256",
    "y_true_sha256",
)

FORECAST_KEY_DIGEST_DOMAIN = "thermoroute.coverage.forecast-key.v1"
Y_TRUE_DIGEST_DOMAIN = "thermoroute.coverage.forecast-key-y-true.v1"
PROJECTION_DIGEST_DOMAIN = "thermoroute.coverage.declared-projection.v1"
HASH_CHUNK_ROWS = 8192
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_INTEGER_TEXT_RE = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z")
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


class CoverageAuditError(RuntimeError):
    """The frozen policy, declared projection, or derived audit is invalid."""


def _expected_policy_stable() -> dict[str, Any]:
    return {
        "format": POLICY_FORMAT,
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "policy_id": POLICY_ID,
        "recorded_date": "2026-07-22",
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent_design": True,
        "route_a_target_interval": {
            "start": TARGET_START,
            "end": TARGET_END,
            "inclusive": True,
        },
        "scope": {
            "cohorts": list(COHORT_ORDER),
            "coverage_scope": "both_frozen_cohorts",
            "comparison_sensitivity_scope": "temporal_primary_five_tests_only",
            "not_a_missing_at_random_assessment": True,
            "not_all_calendar_day_performance": True,
            "core_status_requires_receipt_binding": True,
        },
        "formal_comparisons": [dict(row) for row in ROUTE_A_FORMAL_TESTS],
        "model_registry_by_cohort": {
            cohort: list(MODEL_REGISTRY[cohort]) for cohort in COHORT_ORDER
        },
        "calendar_contract": {
            "classification_date": "target_date",
            "year_rule": "Gregorian_calendar_year_of_target_date",
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
            "all_frozen_models_must_share_exact_key_and_y_true_summaries": True,
            "zero_key_horizon_or_cohort_is_legal_when_explicitly_audited": True,
            "availability_may_not_replace_or_filter_a_frozen_site": True,
            "coverage_balance_changes_primary_reportability": False,
        },
        "sensitivity_contract": {
            "role": "DESCRIPTIVE_NOT_IN_CONFIRMATORY_FAMILY",
            "formal_comparison_count": len(ROUTE_A_FORMAL_TESTS),
            "comparison_models": list(COMPARISON_MODELS),
            "all_12_cells_nonempty": (
                "primary_reportable_station_has_at_least_one_valid_key_in_each_of_"
                "three_years_by_four_seasons"
            ),
            "equal_cell_station_rmse": (
                "sqrt_of_unweighted_mean_of_the_twelve_within_cell_MSE_values"
            ),
            "leave_one_year": (
                "same_all_12_cells_nonempty_stations_and_equal_weight_over_"
                "remaining_eight_cells"
            ),
            "leave_one_season": (
                "same_all_12_cells_nonempty_stations_and_equal_weight_over_"
                "remaining_nine_cells"
            ),
            "does_not_establish_year_or_season_stability": True,
            "retraining_or_recalibration_performed": False,
            "favorable_sensitivity_may_rescue_primary_result": False,
            "unfavorable_sensitivity_must_be_reported": True,
            "sensitivity_changes_primary_result_or_decision": False,
        },
        "formal_statistics_contract": {
            "allowed_status": [
                "ESTIMABLE",
                "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS",
            ],
            "estimable_effect_must_exactly_match_prediction_derived_effect": True,
            "not_estimable_formal_effect_must_be_null": True,
            "prediction_derived_descriptive_effect_never_upgrades_inference": True,
        },
        "source_binding_contract": {
            "required_keys": list(SOURCE_BINDING_KEYS),
            "binding_fields": ["path", "sha256"],
            "canonical_repository_relative_posix_path_required": True,
            "physical_file_verification_delegated_to_opening_release_wrapper": True,
            "full_upstream_files_are_closed_by_sha256": True,
            "core_semantic_digests_cover_declared_projection_only": True,
            "unconsumed_upstream_fields_are_not_silently_claimed_as_validated": True,
        },
        "declared_projection_contract": {
            "missing_unknown_duplicate_or_tampered_declared_values_fail_closed": True,
            "extra_upstream_fields_are_not_consumed": True,
            "extra_upstream_fields_remain_covered_by_full_file_source_binding": True,
        },
        "prohibited_sensitivity_outputs": [
            "p_value",
            "confidence_interval",
            "Holm_adjustment",
            "pass_fail_decision",
        ],
        "integrity_contract": {
            "canonical_chunked_semantic_digest_for_consumed_rows": True,
            "canonical_document_self_hash": True,
            "exact_recomputation_required": True,
            "rewriting_only_the_self_hash_cannot_validate_changed_evidence": True,
        },
        "output_artifact": "trusted/temporal_coverage_audit_v1.json",
    }


def _validated_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(dict(policy))
    self_hash = document.pop("policy_self_sha256", None)
    expected = _expected_policy_stable()
    if document != expected:
        raise CoverageAuditError("temporal-coverage policy semantics changed")
    if not isinstance(self_hash, str) or not _SHA256_RE.fullmatch(self_hash):
        raise CoverageAuditError("temporal-coverage policy self-hash is malformed")
    if sha256_json(document) != self_hash:
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
    if not isinstance(parsed, pd.Series):  # pragma: no cover - Series public input
        parsed = pd.Series(parsed, index=values.index)
    if parsed.dt.tz is not None or not parsed.eq(parsed.dt.normalize()).all():
        raise CoverageAuditError(
            f"{label} must contain timezone-free calendar dates"
        )
    return parsed.astype("datetime64[ns]")


def _strict_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise CoverageAuditError(f"{label} contains a boolean or complex value")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, (float, np.floating)):
        number = float(cast(Any, value))
        if not np.isfinite(number) or not number.is_integer():
            raise CoverageAuditError(f"{label} contains a non-integral value")
        result = int(number)
        if float(result) != number:
            raise CoverageAuditError(f"{label} fails exact integer round-trip")
    elif isinstance(value, str) and _INTEGER_TEXT_RE.fullmatch(value):
        result = int(value)
    else:
        raise CoverageAuditError(f"{label} contains a non-integral value")
    if not _INT64_MIN <= result <= _INT64_MAX:
        raise CoverageAuditError(f"{label} exceeds signed int64")
    return result


def _normalize_integer(values: pd.Series, *, label: str) -> pd.Series:
    return pd.Series(
        [_strict_integer(value, label=label) for value in values.tolist()],
        index=values.index,
        dtype="int64",
    )


def _strict_float(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise CoverageAuditError(f"{label} contains a boolean or complex value")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CoverageAuditError(f"{label} contains a non-numeric value") from exc
    if not np.isfinite(number):
        raise CoverageAuditError(f"{label} contains a non-finite value")
    return number


def _normalize_float(values: pd.Series, *, label: str) -> pd.Series:
    return pd.Series(
        [_strict_float(value, label=label) for value in values.tolist()],
        index=values.index,
        dtype=float,
    )


def _normalize_bool(values: pd.Series, *, label: str) -> pd.Series:
    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return bool(int(value))
        if isinstance(value, str) and value in {
            "1",
            "0",
            "true",
            "false",
            "True",
            "False",
        }:
            return value in {"1", "true", "True"}
        raise CoverageAuditError(f"{label} contains a non-boolean value")

    return pd.Series(
        [parse(value) for value in values.tolist()],
        index=values.index,
        dtype=bool,
    )


def _canonical_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CoverageAuditError(f"{label} path is not canonical repository-relative")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CoverageAuditError(f"{label} path has an empty or traversal segment")
    path = PurePosixPath(value)
    if re.match(r"[A-Za-z]:/", value) or path.is_absolute() or path.as_posix() != value:
        raise CoverageAuditError(f"{label} path is not canonical repository-relative")
    return value


def _normalize_source_bindings(
    source_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if set(source_bindings) != set(SOURCE_BINDING_KEYS):
        raise CoverageAuditError("coverage source-binding closure is incomplete")
    output: dict[str, dict[str, str]] = {}
    for key in SOURCE_BINDING_KEYS:
        binding = source_bindings[key]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise CoverageAuditError(f"coverage source binding is malformed: {key}")
        path = _canonical_relative_path(binding["path"], label=key)
        digest = binding["sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise CoverageAuditError(f"coverage source binding SHA-256 is invalid: {key}")
        output[key] = {"path": path, "sha256": digest}
    if output["policy"] != {
        "path": POLICY_RELATIVE,
        "sha256": POLICY_FILE_SHA256,
    }:
        raise CoverageAuditError("coverage policy source binding is not exact")
    return output


def _update_digest(hasher: Any, row: Sequence[object]) -> None:
    hasher.update(canonical_json(list(row)).encode("utf-8"))
    hasher.update(b"\n")


def _new_hasher(domain: str) -> Any:
    hasher = hashlib.sha256()
    hasher.update(domain.encode("ascii"))
    hasher.update(b"\n")
    return hasher


def _chunked_projection_digest(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    kinds: Mapping[str, str],
    sort_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    ordered = frame.sort_values(
        list(sort_columns or columns), kind="mergesort"
    ).reset_index(drop=True)
    hasher = _new_hasher(PROJECTION_DIGEST_DOMAIN)
    for offset in range(0, len(ordered), HASH_CHUNK_ROWS):
        chunk = ordered.iloc[offset : offset + HASH_CHUNK_ROWS]
        for values in chunk.loc[:, list(columns)].itertuples(index=False, name=None):
            encoded: list[object] = []
            for column, value in zip(columns, values, strict=True):
                kind = kinds.get(column, "string")
                if kind == "date":
                    encoded.append(pd.Timestamp(value).strftime("%Y-%m-%d"))
                elif kind == "float":
                    encoded.append(format(float(value), ".17g"))
                elif kind == "integer":
                    encoded.append(int(value))
                elif kind == "boolean":
                    encoded.append(bool(value))
                else:
                    encoded.append(str(value))
            _update_digest(hasher, encoded)
    return {
        "digest_format": PROJECTION_DIGEST_DOMAIN,
        "canonical_columns": list(columns),
        "row_count": int(len(ordered)),
        "canonical_sha256": hasher.hexdigest(),
    }


def _normalize_sites(
    sites_by_cohort: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(sites_by_cohort) != set(COHORT_ORDER):
        raise CoverageAuditError("coverage core requires both cohort registries")
    output: dict[str, tuple[str, ...]] = {}
    for cohort in COHORT_ORDER:
        sites = tuple(sorted(str(site).strip() for site in sites_by_cohort[cohort]))
        if any(not site for site in sites) or len(sites) != len(set(sites)):
            raise CoverageAuditError(f"{cohort} site registry is invalid")
        output[cohort] = sites
    if set(output["temporal"]) & set(output["external"]):
        raise CoverageAuditError("temporal and external site registries overlap")
    return output


def _normalize_model_registry(
    registry: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(registry) != set(COHORT_ORDER):
        raise CoverageAuditError("model registry omits or adds a cohort")
    output = {
        cohort: tuple(str(model) for model in registry[cohort])
        for cohort in COHORT_ORDER
    }
    if output != MODEL_REGISTRY:
        raise CoverageAuditError("model registry differs from the frozen policy")
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
        raise CoverageAuditError(f"{cohort} observability projection schema changed")
    output = frame.loc[:, list(OBSERVABILITY_COLUMNS)].copy()
    output["site_id"] = output.site_id.astype(str).str.strip()
    output["date"] = _normalize_date_column(
        output.date, label=f"{cohort} observability date"
    )
    output["wtemp_observed"] = _normalize_bool(
        output.wtemp_observed, label=f"{cohort} observability flag"
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


def _normalize_availability(
    frame: pd.DataFrame,
    *,
    sites: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    if set(frame) != set(AVAILABILITY_COLUMNS):
        raise CoverageAuditError("availability projection schema changed")
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
    output: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for item in formal_tests:
        if not isinstance(item, Mapping) or not {
            "test_id",
            "candidate",
            "reference",
            "horizon",
            "margin_c",
        } <= set(item):
            raise CoverageAuditError("formal comparison projection is malformed")
        candidate = str(item["candidate"])
        reference = str(item["reference"])
        if candidate == reference:
            raise CoverageAuditError("formal comparison candidate equals reference")
        horizon = _strict_integer(item["horizon"], label="formal horizon")
        margin = _strict_float(item["margin_c"], label="formal margin")
        comparison_key = (candidate, reference, horizon)
        if comparison_key in seen_keys:
            raise CoverageAuditError("formal comparison key is duplicated")
        seen_keys.add(comparison_key)
        output.append(
            {
                "test_id": str(item["test_id"]),
                "candidate": candidate,
                "reference": reference,
                "horizon": horizon,
                "margin_c": margin,
            }
        )
    expected = [dict(row) for row in ROUTE_A_FORMAL_TESTS]
    if output != expected:
        raise CoverageAuditError("formal comparison family differs from frozen policy")
    return output


def _normalize_statistics_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    allowed_status = {
        "ESTIMABLE",
        "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS",
    }
    output: list[dict[str, Any]] = []
    for expected, item in zip(ROUTE_A_FORMAL_TESTS, rows, strict=False):
        if not isinstance(item, Mapping) or not {
            "test_id",
            "status",
            "median_effect_c",
            "n_stations",
            "n_clusters",
        } <= set(item):
            raise CoverageAuditError("primary statistic projection is malformed")
        test_id = str(item["test_id"])
        status = str(item["status"])
        n_stations = _strict_integer(
            item["n_stations"], label="primary statistic station count"
        )
        n_clusters = _strict_integer(
            item["n_clusters"], label="primary statistic cluster count"
        )
        if (
            test_id != expected["test_id"]
            or status not in allowed_status
            or n_stations < 0
            or n_clusters < 0
            or n_clusters > n_stations
        ):
            raise CoverageAuditError("primary statistic identity/status is invalid")
        raw_effect = item["median_effect_c"]
        if status == "ESTIMABLE":
            effect = _strict_float(raw_effect, label="primary formal effect")
            if n_stations == 0 or n_clusters < 2:
                raise CoverageAuditError("estimable primary statistic has too few units")
        else:
            if raw_effect is not None or n_clusters >= 2:
                raise CoverageAuditError("not-estimable primary statistic is inconsistent")
            effect = None
        output.append(
            {
                "test_id": test_id,
                "status": status,
                "median_effect_c": effect,
                "n_stations": n_stations,
                "n_clusters": n_clusters,
            }
        )
    if len(rows) != len(ROUTE_A_FORMAL_TESTS) or len(output) != len(
        ROUTE_A_FORMAL_TESTS
    ):
        raise CoverageAuditError("primary statistics do not contain exact five rows")
    return output


def _normalize_model_key_audits(
    audits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    model_registry: Mapping[str, tuple[str, ...]],
) -> dict[str, list[dict[str, Any]]]:
    if set(audits) != set(COHORT_ORDER):
        raise CoverageAuditError("model-key audits omit or add a cohort")
    output: dict[str, list[dict[str, Any]]] = {}
    for cohort in COHORT_ORDER:
        rows: list[dict[str, Any]] = []
        for item in audits[cohort]:
            if not isinstance(item, Mapping) or set(item) != set(
                MODEL_KEY_AUDIT_COLUMNS
            ):
                raise CoverageAuditError(f"{cohort} model-key audit schema changed")
            count = _strict_integer(
                item["row_count"], label=f"{cohort} model-key row count"
            )
            key_digest = item["forecast_key_sha256"]
            y_digest = item["y_true_sha256"]
            if (
                count < 0
                or not isinstance(key_digest, str)
                or not _SHA256_RE.fullmatch(key_digest)
                or not isinstance(y_digest, str)
                or not _SHA256_RE.fullmatch(y_digest)
            ):
                raise CoverageAuditError(f"{cohort} model-key audit is invalid")
            rows.append(
                {
                    "model": str(item["model"]),
                    "row_count": count,
                    "forecast_key_sha256": key_digest,
                    "y_true_sha256": y_digest,
                }
            )
        if [row["model"] for row in rows] != list(model_registry[cohort]):
            raise CoverageAuditError(f"{cohort} model-key registry changed")
        summaries = {
            (
                row["row_count"],
                row["forecast_key_sha256"],
                row["y_true_sha256"],
            )
            for row in rows
        }
        if len(summaries) != 1:
            raise CoverageAuditError(
                f"{cohort} frozen models do not share exact key/y_true summaries"
            )
        output[cohort] = rows
    return output


def _season_for_month(month: int) -> str:
    for season in SEASON_ORDER:
        if month in SEASON_MONTHS[season]:
            return season
    raise CoverageAuditError("target month is outside the Gregorian calendar")


def _coverage_cells_and_key_summaries(
    observability: Mapping[str, pd.DataFrame],
    *,
    sites: Mapping[str, tuple[str, ...]],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    years: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    coverage_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for cohort in COHORT_ORDER:
        key_hasher = _new_hasher(FORECAST_KEY_DIGEST_DOMAIN)
        row_count = 0
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
                    _update_digest(
                        key_hasher,
                        (
                            site,
                            horizon,
                            pd.Timestamp(issue_date).strftime("%Y-%m-%d"),
                            pd.Timestamp(target_date).strftime("%Y-%m-%d"),
                        ),
                    )
                    row_count += 1
                cells = pd.DataFrame(
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
                        selected = cells[
                            cells.target_year.eq(year)
                            & cells.target_season.eq(season)
                        ]
                        coverage_rows.append(
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
        summaries[cohort] = {
            "row_count": row_count,
            "forecast_key_sha256": key_hasher.hexdigest(),
        }
    return coverage_rows, summaries


def _validate_model_key_audits_against_coverage(
    audits: Mapping[str, Sequence[Mapping[str, Any]]],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    for cohort in COHORT_ORDER:
        for row in audits[cohort]:
            if (
                row["row_count"] != expected[cohort]["row_count"]
                or row["forecast_key_sha256"]
                != expected[cohort]["forecast_key_sha256"]
            ):
                raise CoverageAuditError(
                    f"{cohort} model-key audit differs from observability"
                )


def _normalize_temporal_comparison_predictions(
    frame: pd.DataFrame,
    *,
    known_sites: set[str],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(frame)
    if missing:
        raise CoverageAuditError(
            f"temporal comparison predictions lack columns: {sorted(missing)}"
        )
    # Filter before copying the declared projection.  Callers may pass a full
    # temporal frame; non-comparison columns/models remain closed by the full-file
    # source binding but are not semantically consumed here.
    model_projection = frame["model"].astype(str).str.strip()
    selected = frame.loc[
        model_projection.isin(COMPARISON_MODELS), list(PREDICTION_COLUMNS)
    ].copy()
    selected["model"] = selected.model.astype(str).str.strip()
    selected["site_id"] = selected.site_id.astype(str).str.strip()
    selected["horizon"] = _normalize_integer(
        selected.horizon, label="comparison prediction horizon"
    )
    selected["issue_date"] = _normalize_date_column(
        selected.issue_date, label="comparison prediction issue_date"
    )
    selected["target_date"] = _normalize_date_column(
        selected.target_date, label="comparison prediction target_date"
    )
    selected["y_true"] = _normalize_float(
        selected.y_true, label="comparison prediction y_true"
    )
    selected["y_pred"] = _normalize_float(
        selected.y_pred, label="comparison prediction y_pred"
    )
    if not set(selected.site_id) <= known_sites:
        raise CoverageAuditError("comparison predictions contain an unknown site")
    if not set(selected.horizon) <= set(HORIZONS):
        raise CoverageAuditError("comparison predictions contain an unknown horizon")
    if not selected.empty and (
        selected.issue_date.lt(target_start).any()
        or selected.target_date.gt(target_end).any()
        or not (
            selected.target_date
            - selected.issue_date
            == pd.to_timedelta(selected.horizon, unit="D")
        ).all()
    ):
        raise CoverageAuditError("comparison prediction key leaves the interval")
    if selected.duplicated(["model", *FORECAST_KEY]).any():
        raise CoverageAuditError("comparison predictions duplicate a model/key")
    return selected.sort_values(
        ["model", *FORECAST_KEY], kind="mergesort"
    ).reset_index(drop=True)


def _prediction_summary(frame: pd.DataFrame) -> dict[str, Any]:
    key_hasher = _new_hasher(FORECAST_KEY_DIGEST_DOMAIN)
    y_hasher = _new_hasher(Y_TRUE_DIGEST_DOMAIN)
    ordered = frame.sort_values(list(FORECAST_KEY), kind="mergesort")
    for offset in range(0, len(ordered), HASH_CHUNK_ROWS):
        chunk = ordered.iloc[offset : offset + HASH_CHUNK_ROWS]
        for site, horizon, issue, target, y_true in chunk.loc[
            :, [*FORECAST_KEY, "y_true"]
        ].itertuples(index=False, name=None):
            key = (
                str(site),
                int(horizon),
                pd.Timestamp(issue).strftime("%Y-%m-%d"),
                pd.Timestamp(target).strftime("%Y-%m-%d"),
            )
            _update_digest(key_hasher, key)
            _update_digest(y_hasher, (*key, format(float(y_true), ".17g")))
    return {
        "row_count": int(len(ordered)),
        "forecast_key_sha256": key_hasher.hexdigest(),
        "y_true_sha256": y_hasher.hexdigest(),
    }


def _validate_comparison_summaries(
    predictions: pd.DataFrame,
    *,
    audits: Sequence[Mapping[str, Any]],
) -> None:
    audit_by_model = {str(row["model"]): row for row in audits}
    for model in COMPARISON_MODELS:
        actual = _prediction_summary(predictions[predictions.model.eq(model)])
        expected = audit_by_model[model]
        if any(actual[key] != expected[key] for key in actual):
            raise CoverageAuditError(
                f"comparison projection differs from model-key audit: {model}"
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
    for cohort, site, horizon, count, flag in availability.itertuples(
        index=False, name=None
    ):
        key = (str(cohort), str(site), int(horizon))
        if totals.get(key) != int(count):
            raise CoverageAuditError(
                "coverage cells do not reconstruct the availability registry"
            )
        reportable[key] = bool(flag)
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
            raise CoverageAuditError("comparison lacks a reportable station")
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
    working["squared_error"] = np.square(
        working.y_pred.to_numpy(float) - working.y_true.to_numpy(float)
    )
    values: list[float] = []
    for year in years:
        for season in seasons:
            selected = working[
                working.target_year.eq(year)
                & working.target_season.eq(season)
            ]
            if selected.empty:
                raise CoverageAuditError(
                    "all-12-cells-nonempty station unexpectedly lacks a cell"
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
        output[site] = _equal_cell_rmse(
            selected[selected.model.eq(candidate)], years=years, seasons=seasons
        ) - _equal_cell_rmse(
            selected[selected.model.eq(reference)], years=years, seasons=seasons
        )
    return output


def _comparison_sensitivities(
    *,
    predictions: pd.DataFrame,
    coverage_rows: Sequence[Mapping[str, Any]],
    reportable: Mapping[tuple[str, str, int], bool],
    statistics: Sequence[Mapping[str, Any]],
    years: tuple[int, ...],
) -> list[dict[str, Any]]:
    coverage = pd.DataFrame(coverage_rows)
    output: list[dict[str, Any]] = []
    for test, formal in zip(ROUTE_A_FORMAL_TESTS, statistics, strict=True):
        candidate = str(test["candidate"])
        reference = str(test["reference"])
        horizon = int(test["horizon"])
        primary_sites = sorted(
            site
            for cohort, site, value_horizon in reportable
            if cohort == "temporal"
            and value_horizon == horizon
            and reportable[(cohort, site, value_horizon)]
        )
        primary_effect = _median_effect(
            _station_effects(
                predictions,
                candidate=candidate,
                reference=reference,
                horizon=horizon,
                sites=primary_sites,
            )
        )
        if int(formal["n_stations"]) != len(primary_sites):
            raise CoverageAuditError("formal station count crosscheck failed")
        if formal["status"] == "ESTIMABLE" and formal["median_effect_c"] != primary_effect:
            raise CoverageAuditError("estimable formal effect exact crosscheck failed")

        selected_sites: list[str] = []
        station_support: list[dict[str, Any]] = []
        for site in primary_sites:
            cells = coverage[
                coverage.cohort.eq("temporal")
                & coverage.site_no.eq(site)
                & coverage.horizon.eq(horizon)
            ]
            if len(cells) != len(years) * len(SEASON_ORDER):
                raise CoverageAuditError("coverage cell registry changed")
            counts = cells.n_valid_keys.to_numpy(int)
            if (counts > 0).all():
                selected_sites.append(site)
                station_support.append(
                    {
                        "site_no": site,
                        "min_valid_keys_per_cell": int(np.min(counts)),
                        "median_valid_keys_per_cell": float(np.median(counts)),
                        "max_valid_keys_per_cell": int(np.max(counts)),
                    }
                )

        primary_selected = _median_effect(
            _station_effects(
                predictions,
                candidate=candidate,
                reference=reference,
                horizon=horizon,
                sites=selected_sites,
            )
        )
        if selected_sites:
            equal_effect = _median_effect(
                _equal_cell_station_effects(
                    predictions,
                    candidate=candidate,
                    reference=reference,
                    horizon=horizon,
                    sites=selected_sites,
                    years=years,
                    seasons=SEASON_ORDER,
                )
            )
            leave_year = [
                {
                    "omitted_year": year,
                    "descriptive_median_effect_c": _median_effect(
                        _equal_cell_station_effects(
                            predictions,
                            candidate=candidate,
                            reference=reference,
                            horizon=horizon,
                            sites=selected_sites,
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
                    "descriptive_median_effect_c": _median_effect(
                        _equal_cell_station_effects(
                            predictions,
                            candidate=candidate,
                            reference=reference,
                            horizon=horizon,
                            sites=selected_sites,
                            years=years,
                            seasons=[value for value in SEASON_ORDER if value != season],
                        )
                    ),
                }
                for season in SEASON_ORDER
            ]
            sensitivity_status = "DESCRIPTIVE_ESTIMABLE_ALL_12_CELLS_NONEMPTY"
        else:
            equal_effect = None
            leave_year = [
                {"omitted_year": year, "descriptive_median_effect_c": None}
                for year in years
            ]
            leave_season = [
                {"omitted_season": season, "descriptive_median_effect_c": None}
                for season in SEASON_ORDER
            ]
            sensitivity_status = (
                "DESCRIPTIVE_NOT_ESTIMABLE_NO_STATION_WITH_ALL_12_CELLS_NONEMPTY"
            )
        output.append(
            {
                "test_id": test["test_id"],
                "candidate": candidate,
                "reference": reference,
                "horizon": horizon,
                "margin_c": float(test["margin_c"]),
                "formal_statistics_status": formal["status"],
                "formal_n_clusters": formal["n_clusters"],
                "formal_median_effect_c": formal["median_effect_c"],
                "n_primary_reportable_stations": len(primary_sites),
                "prediction_derived_descriptive_median_effect_c": primary_effect,
                "prediction_derived_descriptive_effect_does_not_upgrade_inference": True,
                "temporal_reweighting_status": sensitivity_status,
                "n_all_12_cells_nonempty_stations": len(selected_sites),
                "all_12_cells_nonempty_station_support": station_support,
                "primary_weight_descriptive_median_effect_all_12_cells_nonempty_c": (
                    primary_selected
                ),
                "equal_12cell_descriptive_median_effect_c": equal_effect,
                "leave_one_year_equal_cell_descriptive": leave_year,
                "leave_one_season_equal_cell_descriptive": leave_season,
                "sensitivity_changes_primary_result_or_decision": False,
                "does_not_establish_year_or_season_stability": True,
            }
        )
    return output


def _assert_no_prohibited_sensitivity_fields(
    comparisons: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> None:
    prohibited = set(policy["prohibited_sensitivity_outputs"])

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            overlap = {str(key) for key in value} & prohibited
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
    source_bindings: Mapping[str, Mapping[str, str]],
    target_start: str | pd.Timestamp,
    target_end: str | pd.Timestamp,
    sites_by_cohort: Mapping[str, Sequence[str]],
    model_registry_by_cohort: Mapping[str, Sequence[str]],
    model_key_audits_by_cohort: Mapping[str, Sequence[Mapping[str, Any]]],
    observability_by_cohort: Mapping[str, pd.DataFrame],
    temporal_comparison_predictions: pd.DataFrame,
    availability: pd.DataFrame,
    formal_tests: Sequence[Mapping[str, Any]],
    primary_statistics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic, non-filtering coverage/reweighting core.

    The full prediction artifacts are never copied here.  The caller supplies a
    compact per-model key/y_true audit for every frozen model and, separately,
    only the temporal comparison rows needed for the five descriptive
    reweightings.  Empty cohorts and zero-key horizons remain explicit and legal.
    """
    frozen_policy = _validated_policy(policy)
    bindings = _normalize_source_bindings(source_bindings)
    start = _parse_date(target_start, label="coverage target_start")
    end = _parse_date(target_end, label="coverage target_end")
    if (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")) != (
        TARGET_START,
        TARGET_END,
    ):
        raise CoverageAuditError("coverage interval differs from frozen Route A")
    years = tuple(range(start.year, end.year + 1))
    sites = _normalize_sites(sites_by_cohort)
    model_registry = _normalize_model_registry(model_registry_by_cohort)
    if set(observability_by_cohort) != set(COHORT_ORDER):
        raise CoverageAuditError("observability projections omit or add a cohort")
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
    availability_frame = _normalize_availability(availability, sites=sites)
    tests = _normalize_formal_tests(formal_tests)
    statistics = _normalize_statistics_rows(primary_statistics)
    audits = _normalize_model_key_audits(
        model_key_audits_by_cohort, model_registry=model_registry
    )
    coverage_rows, expected_key_summaries = _coverage_cells_and_key_summaries(
        observations,
        sites=sites,
        target_start=start,
        target_end=end,
        years=years,
    )
    _validate_model_key_audits_against_coverage(audits, expected_key_summaries)
    comparison_predictions = _normalize_temporal_comparison_predictions(
        temporal_comparison_predictions,
        known_sites=set(sites["temporal"]),
        target_start=start,
        target_end=end,
    )
    _validate_comparison_summaries(
        comparison_predictions, audits=audits["temporal"]
    )
    reportable = _validate_coverage_counts(
        coverage_rows, availability=availability_frame
    )
    comparisons = _comparison_sensitivities(
        predictions=comparison_predictions,
        coverage_rows=coverage_rows,
        reportable=reportable,
        statistics=statistics,
        years=years,
    )
    _assert_no_prohibited_sensitivity_fields(comparisons, frozen_policy)

    semantic_inputs = {
        "source_bindings": {
            "row_count": len(bindings),
            "canonical_sha256": sha256_json(bindings),
        },
        "sites_by_cohort": {
            "canonical_sha256": sha256_json(
                {cohort: list(sites[cohort]) for cohort in COHORT_ORDER}
            )
        },
        "model_registry_by_cohort": {
            "canonical_sha256": sha256_json(
                {cohort: list(model_registry[cohort]) for cohort in COHORT_ORDER}
            )
        },
        "model_key_audits_by_cohort": {
            "canonical_sha256": sha256_json(audits)
        },
        "observability": {
            cohort: _chunked_projection_digest(
                observations[cohort],
                columns=OBSERVABILITY_COLUMNS,
                kinds={"date": "date", "wtemp_observed": "boolean"},
            )
            for cohort in COHORT_ORDER
        },
        "temporal_comparison_predictions": _chunked_projection_digest(
            comparison_predictions,
            columns=PREDICTION_COLUMNS,
            kinds={
                "horizon": "integer",
                "issue_date": "date",
                "target_date": "date",
                "y_true": "float",
                "y_pred": "float",
            },
        ),
        "availability": _chunked_projection_digest(
            availability_frame,
            columns=AVAILABILITY_COLUMNS,
            kinds={
                "horizon": "integer",
                "n_valid_targets": "integer",
                "reportable": "boolean",
            },
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
        "status": "DERIVED_CORE_REQUIRES_RECEIPT_BINDING",
        "role": "DESCRIPTIVE_COVERAGE_AND_TEMPORAL_REWEIGHTING_CORE",
        "policy_self_sha256": frozen_policy["policy_self_sha256"],
        "source_bindings": bindings,
        "target_interval": {
            "start": TARGET_START,
            "end": TARGET_END,
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
        "sensitivity_changes_primary_result_or_decision": False,
        "inference_computed": False,
    }
    return {**stable, "audit_self_sha256": sha256_json(stable)}


def validate_temporal_coverage_audit(
    document: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    source_bindings: Mapping[str, Mapping[str, str]],
    target_start: str | pd.Timestamp,
    target_end: str | pd.Timestamp,
    sites_by_cohort: Mapping[str, Sequence[str]],
    model_registry_by_cohort: Mapping[str, Sequence[str]],
    model_key_audits_by_cohort: Mapping[str, Sequence[Mapping[str, Any]]],
    observability_by_cohort: Mapping[str, pd.DataFrame],
    temporal_comparison_predictions: pd.DataFrame,
    availability: pd.DataFrame,
    formal_tests: Sequence[Mapping[str, Any]],
    primary_statistics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute every declared field and reject stale or re-hashed evidence."""
    actual = deepcopy(dict(document))
    self_hash = actual.pop("audit_self_sha256", None)
    if not isinstance(self_hash, str) or not _SHA256_RE.fullmatch(self_hash):
        raise CoverageAuditError("temporal-coverage audit self-hash is malformed")
    if sha256_json(actual) != self_hash:
        raise CoverageAuditError("temporal-coverage audit self-hash changed")
    expected = build_temporal_coverage_audit(
        policy=policy,
        source_bindings=source_bindings,
        target_start=target_start,
        target_end=target_end,
        sites_by_cohort=sites_by_cohort,
        model_registry_by_cohort=model_registry_by_cohort,
        model_key_audits_by_cohort=model_key_audits_by_cohort,
        observability_by_cohort=observability_by_cohort,
        temporal_comparison_predictions=temporal_comparison_predictions,
        availability=availability,
        formal_tests=formal_tests,
        primary_statistics=primary_statistics,
    )
    if dict(document) != expected:
        raise CoverageAuditError("temporal-coverage audit is stale or tampered")
    return expected
