from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermoroute.outcome_qc import (
    POLICY_RELATIVE,
    OutcomeQCGateError,
    build_outcome_qc_gate_document,
    validate_outcome_qc_gate_document,
    validate_outcome_qc_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads(
        (ROOT / "protocols" / "route_a_confirmatory_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _predictions(*, rows_per_site: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    starts = pd.date_range("2021-01-01", periods=rows_per_site, freq="D")
    model_error = {
        "ThermoRoute": 0.20,
        "DampedPersistence": 0.50,
        "LightGBM": 0.35,
    }
    for site_index, site in enumerate(("a", "b")):
        for horizon in (1, 3, 7):
            for issue_date in starts:
                target_date = issue_date + pd.Timedelta(days=horizon)
                for model, error in model_error.items():
                    rows.append({
                        "model": model,
                        "site_id": site,
                        "horizon": horizon,
                        "issue_date": issue_date,
                        "target_date": target_date,
                        "y_true": 10.0 + site_index,
                        "y_pred": 10.0 + site_index + error,
                    })
    return pd.DataFrame(rows)


def _normalized(*, outside: bool = False) -> pd.DataFrame:
    return pd.DataFrame({
        "site_no": ["a", "b"],
        "DATE": pd.to_datetime(["2021-01-02", "2021-01-02"]),
        "WTEMP": [51.0 if outside else 10.0, 11.0],
    })


def _spatial(protocol: dict, *, unstable: bool = False) -> dict:
    comparisons = []
    for test in protocol["primary_inference_contract"]["confirmatory_family"]:
        margin = float(test["margin_c"])
        full = -0.20
        comparisons.append({
            "test_id": test["test_id"],
            "station_weighted_median_effect_c": full,
            "margin_c": margin,
            "leave_one_huc": [
                {
                    "held_out_huc2": "01",
                    "effect_minus_margin_c": (
                        0.01 if unstable else full - margin
                    ),
                },
                {
                    "held_out_huc2": "02",
                    "effect_minus_margin_c": full - margin,
                },
            ],
        })
    return {"comparisons": comparisons}


def _build(
    *,
    rows_per_site: int = 4,
    outside: bool = False,
    unstable_huc: bool = False,
) -> dict:
    protocol = _protocol()
    return build_outcome_qc_gate_document(
        root=ROOT,
        policy_path=ROOT / POLICY_RELATIVE,
        protocol=protocol,
        temporal_predictions=_predictions(rows_per_site=rows_per_site),
        normalized_temporal=_normalized(outside=outside),
        spatial_sensitivity=_spatial(protocol, unstable=unstable_huc),
        minimum_targets=3,
    )


def test_policy_is_exact_outcome_free_and_nonfiltering() -> None:
    policy = validate_outcome_qc_policy(ROOT / POLICY_RELATIVE, root=ROOT)
    assert policy["post_2020_wtemp_requested_or_inspected"] is False
    assert policy["application_contract"][
        "primary_statistics_remain_unfiltered"
    ] is True
    assert policy["target_plausibility_gate"]["censor_or_replace_values"] is False


def test_policy_rejects_alternative_path_and_semantic_tamper(tmp_path: Path) -> None:
    alternative = tmp_path / "policy.json"
    alternative.write_bytes((ROOT / POLICY_RELATIVE).read_bytes())
    with pytest.raises(OutcomeQCGateError, match="not canonical"):
        validate_outcome_qc_policy(alternative, root=ROOT)


def test_gate_passes_stable_synthetic_family_and_recomputes_exactly() -> None:
    document = _build()
    assert document["status"] == "PASS_DIRECTIONAL_REPORTING_QC"
    assert document["pass"] is True
    assert document["directional_claims_allowed_by_outcome_qc"] is True
    assert len(document["single_extreme_influence"]) == 5
    assert all(row["pass"] for row in document["single_extreme_influence"])
    protocol = _protocol()
    assert validate_outcome_qc_gate_document(
        document,
        root=ROOT,
        policy_path=ROOT / POLICY_RELATIVE,
        protocol=protocol,
        temporal_predictions=_predictions(),
        normalized_temporal=_normalized(),
        spatial_sensitivity=_spatial(protocol),
        minimum_targets=3,
    ) == document


def test_plausibility_flags_but_never_filters_primary_statistics() -> None:
    document = _build(outside=True)
    assert document["status"] == "FAIL_WITHHOLD_DIRECTIONAL_CLAIMS"
    assert document["target_plausibility"]["outside_range_count"] == 1
    assert document["target_plausibility"][
        "outside_range_values_retained_in_primary_analysis"
    ] is True
    assert document["primary_statistics_filtered_or_recomputed_on_selected_rows"] is False


def test_nonestimable_deletion_and_unstable_huc_fail_closed() -> None:
    too_short = _build(rows_per_site=3)
    assert too_short["components"]["single_extreme_influence_pass"] is False
    assert all(
        row["nonestimable_after_deletion_sites"] == ["a", "b"]
        for row in too_short["single_extreme_influence"]
    )
    assert all(
        row["n_reportable_stations"] == 2
        and row["primary_unfiltered_effect_c"] is not None
        and all(
            station["deleted_station_effect_c"] is None
            for station in row["station_audit"]
        )
        for row in too_short["single_extreme_influence"]
    )
    unstable = _build(unstable_huc=True)
    assert unstable["components"]["leave_one_huc_direction_pass"] is False
    assert unstable["directional_claims_allowed_by_outcome_qc"] is False


def test_gate_tamper_is_rejected_by_exact_recomputation() -> None:
    document = _build()
    attacked = deepcopy(document)
    attacked["directional_claims_allowed_by_outcome_qc"] = False
    protocol = _protocol()
    with pytest.raises(OutcomeQCGateError, match="stale or tampered"):
        validate_outcome_qc_gate_document(
            attacked,
            root=ROOT,
            policy_path=ROOT / POLICY_RELATIVE,
            protocol=protocol,
            temporal_predictions=_predictions(),
            normalized_temporal=_normalized(),
            spatial_sensitivity=_spatial(protocol),
            minimum_targets=3,
        )


def test_single_extreme_audit_reports_pair_symmetric_selected_key() -> None:
    predictions = _predictions()
    mask = (
        predictions.model.eq("ThermoRoute")
        & predictions.site_id.eq("a")
        & predictions.horizon.eq(1)
        & predictions.issue_date.eq(pd.Timestamp("2021-01-01"))
    )
    predictions.loc[mask, "y_pred"] = 12.0
    protocol = _protocol()
    document = build_outcome_qc_gate_document(
        root=ROOT,
        policy_path=ROOT / POLICY_RELATIVE,
        protocol=protocol,
        temporal_predictions=predictions,
        normalized_temporal=_normalized(),
        spatial_sensitivity=_spatial(protocol),
        minimum_targets=3,
    )
    comparison = next(
        row for row in document["single_extreme_influence"]
        if row["test_id"] == "H1-h1-vs-damped"
    )
    station = next(row for row in comparison["station_audit"] if row["site_no"] == "a")
    assert station["selected_issue_date"] == "2021-01-01"
    assert station["selected_target_date"] == "2021-01-02"
    assert np.isfinite(station["selected_combined_sse_share"])
