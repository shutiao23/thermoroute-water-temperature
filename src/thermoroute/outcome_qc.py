"""Predeclared, non-filtering outcome-QC and influence gate for Route A.

The policy is frozen before labels are opened.  The post-opening computation
never removes a row from the primary analysis, changes the cohort, or refits a
model.  It only asks whether directional wording remains defensible after a
fixed plausibility audit and two fixed influence diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .repro import sha256_json


POLICY_FORMAT = "thermoroute.route-a-outcome-qc-policy.v1"
GATE_FORMAT = "thermoroute.route-a-outcome-qc-gate.v1"
POLICY_RELATIVE = "protocols/route_a_outcome_qc_policy_v1.json"
BASE_PROTOCOL_RELATIVE = "protocols/route_a_confirmatory_v1.json"
POLICY_ID = "route-a-outcome-qc-and-influence-001"
MAX_EFFECT_CHANGE_C = 0.05
TARGET_LOWER_C = -2.0
TARGET_UPPER_C = 50.0

FORECAST_KEY = ("site_id", "issue_date", "target_date", "horizon")


class OutcomeQCGateError(RuntimeError):
    """The frozen policy or its deterministic post-opening result is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise OutcomeQCGateError(f"{label} path is not repository-relative")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise OutcomeQCGateError(f"{label} path escapes repository")
    if not path.is_file():
        raise OutcomeQCGateError(f"{label} is absent: {relative}")
    return path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OutcomeQCGateError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise OutcomeQCGateError(f"{label} is not a JSON object")
    return value


def file_binding(root: str | Path, path: str | Path) -> dict[str, str]:
    root_path = Path(root).resolve()
    resolved = Path(path).resolve()
    if resolved != root_path and root_path not in resolved.parents:
        raise OutcomeQCGateError("outcome-QC binding escapes repository")
    if not resolved.is_file():
        raise OutcomeQCGateError("outcome-QC binding target is absent")
    return {
        "path": resolved.relative_to(root_path).as_posix(),
        "sha256": _sha256_file(resolved),
    }


def _formal_family(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    inference = protocol.get("primary_inference_contract")
    family = inference.get("confirmatory_family") if isinstance(
        inference, Mapping
    ) else None
    if not isinstance(family, list) or len(family) != 5:
        raise OutcomeQCGateError("base protocol lacks the exact five-test family")
    required = {
        "test_id", "candidate", "reference", "horizon", "margin_c",
        "alternative", "bootstrap_seed", "sign_flip_seed", "description",
    }
    output: list[dict[str, Any]] = []
    for item in family:
        if not isinstance(item, Mapping) or set(item) != required:
            raise OutcomeQCGateError("base protocol five-test schema changed")
        output.append(dict(item))
    return output


def _expected_policy(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol_path = _inside(
        root, BASE_PROTOCOL_RELATIVE, label="outcome-QC base protocol"
    )
    family = _formal_family(protocol)
    return {
        "format": POLICY_FORMAT,
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "policy_id": POLICY_ID,
        "recorded_date": "2026-07-22",
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent": True,
        "base_protocol": file_binding(root, protocol_path),
        "confirmatory_family_sha256": sha256_json(family),
        "application_contract": {
            "primary_statistics_remain_unfiltered": True,
            "outcome_based_site_or_key_selection_forbidden": True,
            "outcome_based_retraining_or_recalibration_forbidden": True,
            "pair_symmetric_operations_only": True,
            "all_five_comparisons_must_be_audited": True,
            "missing_unknown_or_nonestimable_is_failure": True,
        },
        "target_plausibility_gate": {
            "variable": "WTEMP",
            "units": "degrees_C",
            "confirmation_interval_only": True,
            "lower_inclusive_c": TARGET_LOWER_C,
            "upper_inclusive_c": TARGET_UPPER_C,
            "outside_range_action": (
                "FLAG_RETAIN_AND_WITHHOLD_DIRECTIONAL_CLAIMS"
            ),
            "censor_or_replace_values": False,
        },
        "single_extreme_influence_gate": {
            "scope": "each_formal_comparison_each_reportable_station",
            "forecast_key": list(FORECAST_KEY),
            "selection_score": (
                "candidate_squared_error_plus_reference_squared_error"
            ),
            "selection_count_per_station": 1,
            "tie_break": [
                "issue_date_ascending", "target_date_ascending",
            ],
            "deletion_rule": (
                "delete_the_same_selected_forecast_key_from_candidate_and_reference"
            ),
            "station_effect": "candidate_rmse_minus_reference_rmse",
            "cohort_effect": "median_station_effect",
            "maximum_absolute_cohort_effect_change_c": MAX_EFFECT_CHANGE_C,
            "margin_direction_must_remain_unchanged": True,
            "minimum_valid_targets_still_required_after_deletion": True,
            "max_combined_sse_share_is_reported_not_thresholded": True,
        },
        "leave_one_huc_gate": {
            "required": True,
            "effect": (
                "leave_one_huc_station_weighted_median_effect_minus_margin"
            ),
            "all_estimable_huc_deletions_must_match_full_effect_margin_direction": True,
            "touching_margin_or_nonestimable_is_failure": True,
        },
        "decision": {
            "all_components_must_pass": True,
            "pass_status": "PASS_DIRECTIONAL_REPORTING_QC",
            "failure_status": "FAIL_WITHHOLD_DIRECTIONAL_CLAIMS",
            "failure_action": (
                "REPORT_UNFILTERED_EFFECTS_WITH_QC_WARNINGS_NO_DIRECTIONAL_CLAIM"
            ),
            "strong_p_value_or_favorable_interval_cannot_override_failure": True,
            "output_artifact": "trusted/outcome_qc_gate_v1.json",
        },
    }


def validate_outcome_qc_policy(
    policy_path: str | Path,
    *,
    root: str | Path,
    protocol_path: str | Path = BASE_PROTOCOL_RELATIVE,
) -> dict[str, Any]:
    """Require the one canonical outcome-free policy and exact semantics."""
    root_path = Path(root).resolve()
    requested = Path(policy_path)
    if not requested.is_absolute():
        requested = root_path / requested
    requested = requested.resolve()
    canonical = (root_path / POLICY_RELATIVE).resolve()
    if requested != canonical:
        raise OutcomeQCGateError("outcome-QC policy path is not canonical")
    protocol_requested = Path(protocol_path)
    if not protocol_requested.is_absolute():
        protocol_requested = root_path / protocol_requested
    if protocol_requested.resolve() != (root_path / BASE_PROTOCOL_RELATIVE).resolve():
        raise OutcomeQCGateError("outcome-QC base protocol path is not canonical")
    policy = _load_json(requested, label="outcome-QC policy")
    protocol = _load_json(protocol_requested.resolve(), label="base protocol")
    if policy != _expected_policy(root_path, protocol):
        raise OutcomeQCGateError("outcome-QC policy is stale or changed")
    return policy


def _direction(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NOT_ESTIMABLE"
    if value < 0.0:
        return "BELOW_MARGIN"
    if value > 0.0:
        return "ABOVE_MARGIN"
    return "TOUCHES_MARGIN"


def _pair_comparison(
    predictions: pd.DataFrame,
    *,
    test: Mapping[str, Any],
) -> pd.DataFrame:
    horizon = int(test["horizon"])
    candidate = predictions[
        predictions["model"].astype(str).eq(str(test["candidate"]))
        & pd.to_numeric(predictions["horizon"], errors="coerce").eq(horizon)
    ]
    reference = predictions[
        predictions["model"].astype(str).eq(str(test["reference"]))
        & pd.to_numeric(predictions["horizon"], errors="coerce").eq(horizon)
    ]
    keys = list(FORECAST_KEY)
    columns = [*keys, "y_true", "y_pred"]
    if candidate.duplicated(keys).any() or reference.duplicated(keys).any():
        raise OutcomeQCGateError("outcome-QC comparison has duplicate forecast keys")
    paired = candidate[columns].merge(
        reference[columns],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_candidate", "_reference"),
    )
    numeric = paired[
        ["y_true_candidate", "y_true_reference", "y_pred_candidate", "y_pred_reference"]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        paired.empty
        or not paired["_merge"].eq("both").all()
        or not np.isfinite(numeric.to_numpy(float)).all()
        or not np.allclose(
            numeric["y_true_candidate"], numeric["y_true_reference"],
            rtol=0.0, atol=0.0,
        )
    ):
        raise OutcomeQCGateError(
            "outcome-QC comparison lacks exact common finite predictions/truth"
        )
    paired = paired.drop(columns="_merge")
    paired[[
        "y_true_candidate", "y_true_reference", "y_pred_candidate",
        "y_pred_reference",
    ]] = numeric
    return paired.sort_values(
        ["site_id", "issue_date", "target_date"], kind="mergesort"
    ).reset_index(drop=True)


def _single_extreme_comparison(
    paired: pd.DataFrame,
    *,
    test: Mapping[str, Any],
    minimum_targets: int,
) -> dict[str, Any]:
    primary_effects: list[float] = []
    deleted_effects: list[float] = []
    station_rows: list[dict[str, Any]] = []
    nonestimable: list[str] = []
    for site, raw_group in paired.groupby("site_id", sort=True):
        group = raw_group.sort_values(
            ["issue_date", "target_date"], kind="mergesort"
        ).reset_index(drop=True)
        if len(group) < minimum_targets:
            continue
        candidate_error = (
            group["y_pred_candidate"].to_numpy(float)
            - group["y_true_candidate"].to_numpy(float)
        )
        reference_error = (
            group["y_pred_reference"].to_numpy(float)
            - group["y_true_reference"].to_numpy(float)
        )
        combined = candidate_error ** 2 + reference_error ** 2
        selected = int(np.argmax(combined))
        total_sse = float(np.sum(combined))
        share = 0.0 if total_sse == 0.0 else float(combined[selected] / total_sse)
        primary = float(
            np.sqrt(np.mean(candidate_error ** 2))
            - np.sqrt(np.mean(reference_error ** 2))
        )
        primary_effects.append(primary)
        remaining = np.ones(len(group), dtype=bool)
        remaining[selected] = False
        chosen = group.iloc[selected]
        station_row: dict[str, Any] = {
            "site_no": str(site),
            "n_common_keys": int(len(group)),
            "selected_issue_date": pd.Timestamp(chosen["issue_date"]).strftime(
                "%Y-%m-%d"
            ),
            "selected_target_date": pd.Timestamp(chosen["target_date"]).strftime(
                "%Y-%m-%d"
            ),
            "selected_combined_squared_error": float(combined[selected]),
            "selected_combined_sse_share": share,
            "primary_station_effect_c": primary,
            "deleted_station_effect_c": None,
        }
        if int(remaining.sum()) < minimum_targets:
            nonestimable.append(str(site))
            station_rows.append(station_row)
            continue
        deleted = float(
            np.sqrt(np.mean(candidate_error[remaining] ** 2))
            - np.sqrt(np.mean(reference_error[remaining] ** 2))
        )
        deleted_effects.append(deleted)
        station_row["deleted_station_effect_c"] = deleted
        station_rows.append(station_row)
    primary_effect = (
        None if not primary_effects else float(np.median(primary_effects))
    )
    if nonestimable or not deleted_effects:
        deleted_effect = None
        effect_change = None
    else:
        deleted_effect = float(np.median(deleted_effects))
        if primary_effect is None:  # defensive; both lists advance together
            raise OutcomeQCGateError("outcome-QC primary influence effect is absent")
        effect_change = abs(deleted_effect - primary_effect)
    margin = float(test["margin_c"])
    primary_direction = _direction(
        None if primary_effect is None else primary_effect - margin
    )
    deleted_direction = _direction(
        None if deleted_effect is None else deleted_effect - margin
    )
    stable = (
        primary_direction in {"BELOW_MARGIN", "ABOVE_MARGIN"}
        and deleted_direction == primary_direction
    )
    passed = (
        effect_change is not None
        and effect_change <= MAX_EFFECT_CHANGE_C
        and stable
        and not nonestimable
    )
    maximum_share_row = max(
        station_rows,
        key=lambda row: (float(row["selected_combined_sse_share"]), row["site_no"]),
        default=None,
    )
    return {
        "test_id": str(test["test_id"]),
        "candidate": str(test["candidate"]),
        "reference": str(test["reference"]),
        "horizon": int(test["horizon"]),
        "margin_c": margin,
        "n_reportable_stations": len(primary_effects),
        "nonestimable_after_deletion_sites": sorted(nonestimable),
        "primary_unfiltered_effect_c": primary_effect,
        "one_extreme_per_station_deleted_effect_c": deleted_effect,
        "absolute_effect_change_c": effect_change,
        "maximum_allowed_absolute_effect_change_c": MAX_EFFECT_CHANGE_C,
        "primary_margin_direction": primary_direction,
        "deleted_margin_direction": deleted_direction,
        "margin_direction_stable": stable,
        "maximum_selected_combined_sse_share": (
            None if maximum_share_row is None
            else maximum_share_row["selected_combined_sse_share"]
        ),
        "maximum_share_site_no": (
            None if maximum_share_row is None else maximum_share_row["site_no"]
        ),
        "station_audit": station_rows,
        "pass": passed,
    }


def _leave_one_huc_comparison(
    spatial: Mapping[str, Any], *, test: Mapping[str, Any]
) -> dict[str, Any]:
    comparisons = spatial.get("comparisons")
    if not isinstance(comparisons, list):
        raise OutcomeQCGateError("spatial sensitivity lacks comparison rows")
    matches = [
        row for row in comparisons
        if isinstance(row, Mapping) and row.get("test_id") == test["test_id"]
    ]
    if len(matches) != 1:
        raise OutcomeQCGateError("spatial sensitivity does not match five-test registry")
    row = matches[0]
    full = row.get("station_weighted_median_effect_c")
    margin = float(test["margin_c"])
    full_minus_margin = None if full is None else float(full) - margin
    full_direction = _direction(full_minus_margin)
    leave_one = row.get("leave_one_huc")
    if not isinstance(leave_one, list) or not leave_one:
        leave_one = []
    observed: list[dict[str, Any]] = []
    for item in leave_one:
        if not isinstance(item, Mapping):
            raise OutcomeQCGateError("leave-one-HUC row is malformed")
        value = item.get("effect_minus_margin_c")
        numeric = None if value is None else float(value)
        observed.append({
            "held_out_huc2": str(item.get("held_out_huc2", "")),
            "effect_minus_margin_c": numeric,
            "margin_direction": _direction(numeric),
        })
    stable = (
        full_direction in {"BELOW_MARGIN", "ABOVE_MARGIN"}
        and bool(observed)
        and all(item["margin_direction"] == full_direction for item in observed)
    )
    return {
        "test_id": str(test["test_id"]),
        "full_effect_minus_margin_c": full_minus_margin,
        "full_margin_direction": full_direction,
        "leave_one_huc": observed,
        "all_huc_deletions_match_full_margin_direction": stable,
        "pass": stable,
    }


def build_outcome_qc_gate_document(
    *,
    root: str | Path,
    policy_path: str | Path,
    protocol: Mapping[str, Any],
    temporal_predictions: pd.DataFrame,
    normalized_temporal: pd.DataFrame,
    spatial_sensitivity: Mapping[str, Any],
    minimum_targets: int,
) -> dict[str, Any]:
    """Execute the frozen post-opening audit without altering primary rows."""
    root_path = Path(root).resolve()
    policy = validate_outcome_qc_policy(policy_path, root=root_path)
    family = _formal_family(protocol)
    if type(minimum_targets) is not int or minimum_targets < 2:
        raise OutcomeQCGateError("outcome-QC minimum-target contract is invalid")
    required_prediction_columns = {
        "model", "site_id", "horizon", "issue_date", "target_date",
        "y_true", "y_pred",
    }
    if required_prediction_columns - set(temporal_predictions):
        raise OutcomeQCGateError("outcome-QC predictions lack required columns")
    required_outcome_columns = {"site_no", "DATE", "WTEMP"}
    if required_outcome_columns - set(normalized_temporal):
        raise OutcomeQCGateError("outcome-QC normalized outcomes lack required columns")

    interval = protocol.get("time_holdout")
    if not isinstance(interval, Mapping):
        raise OutcomeQCGateError("base protocol lacks confirmation interval")
    start = pd.Timestamp(str(interval.get("primary_target_start", "")))
    end = pd.Timestamp(str(interval.get("end", "")))
    dates = pd.to_datetime(normalized_temporal["DATE"], errors="raise")
    values = pd.to_numeric(normalized_temporal["WTEMP"], errors="coerce")
    finite = np.isfinite(values.to_numpy(float))
    in_interval = dates.between(start, end, inclusive="both").to_numpy()
    outside = finite & in_interval & (
        (values.to_numpy(float) < TARGET_LOWER_C)
        | (values.to_numpy(float) > TARGET_UPPER_C)
    )
    outside_rows = normalized_temporal.loc[outside, ["site_no", "DATE", "WTEMP"]]
    outside_records = [
        {
            "site_no": str(row.site_no),
            "date": pd.Timestamp(row.DATE).strftime("%Y-%m-%d"),
            "wtemp_c": float(row.WTEMP),
        }
        for row in outside_rows.sort_values(
            ["site_no", "DATE"], kind="mergesort"
        ).itertuples(index=False)
    ]
    plausibility = {
        "lower_inclusive_c": TARGET_LOWER_C,
        "upper_inclusive_c": TARGET_UPPER_C,
        "finite_confirmation_values_checked": int(np.sum(finite & in_interval)),
        "outside_range_count": len(outside_records),
        "outside_range_values_retained_in_primary_analysis": True,
        "outside_range_records": outside_records,
        "pass": not outside_records,
    }

    single_extreme = [
        _single_extreme_comparison(
            _pair_comparison(temporal_predictions, test=test),
            test=test,
            minimum_targets=minimum_targets,
        )
        for test in family
    ]
    leave_one_huc = [
        _leave_one_huc_comparison(spatial_sensitivity, test=test)
        for test in family
    ]
    components = {
        "target_plausibility_pass": bool(plausibility["pass"]),
        "single_extreme_influence_pass": all(
            bool(row["pass"]) for row in single_extreme
        ),
        "leave_one_huc_direction_pass": all(
            bool(row["pass"]) for row in leave_one_huc
        ),
    }
    passed = all(components.values())
    stable: dict[str, Any] = {
        "format": GATE_FORMAT,
        "status": (
            policy["decision"]["pass_status"]
            if passed else policy["decision"]["failure_status"]
        ),
        "policy": {
            **file_binding(root_path, Path(policy_path).resolve()),
            "policy_id": policy["policy_id"],
        },
        "confirmatory_family_sha256": sha256_json(family),
        "minimum_valid_targets_per_station_horizon": minimum_targets,
        "primary_statistics_filtered_or_recomputed_on_selected_rows": False,
        "models_retrained_or_recalibrated": False,
        "sites_or_primary_keys_removed_by_qc": False,
        "target_plausibility": plausibility,
        "single_extreme_influence": single_extreme,
        "leave_one_huc_direction": leave_one_huc,
        "components": components,
        "pass": passed,
        "directional_claims_allowed_by_outcome_qc": passed,
        "failure_action": policy["decision"]["failure_action"],
    }
    stable["gate_self_sha256"] = sha256_json(stable)
    return stable


def validate_outcome_qc_gate_document(
    document: Mapping[str, Any],
    *,
    root: str | Path,
    policy_path: str | Path,
    protocol: Mapping[str, Any],
    temporal_predictions: pd.DataFrame,
    normalized_temporal: pd.DataFrame,
    spatial_sensitivity: Mapping[str, Any],
    minimum_targets: int,
) -> dict[str, Any]:
    """Recompute the complete gate and require exact semantic equality."""
    expected = build_outcome_qc_gate_document(
        root=root,
        policy_path=policy_path,
        protocol=protocol,
        temporal_predictions=temporal_predictions,
        normalized_temporal=normalized_temporal,
        spatial_sensitivity=spatial_sensitivity,
        minimum_targets=minimum_targets,
    )
    if dict(document) != expected:
        raise OutcomeQCGateError("outcome-QC gate result is stale or tampered")
    return expected
