"""Focused synthetic contracts for the forcing-ladder v4 runner.

These tests exercise validation and scoring helpers only.  They deliberately do
not fit LightGBM or Air2stream models and do not read production data/artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.final.run_forcing_ladder_v4 as V4
from thermoroute import air2stream
from thermoroute import config as C

SITE = "00000001"


@dataclass(frozen=True)
class _ConstantClimatology:
    value: float

    def predict(self, _station: str, doy: np.ndarray) -> np.ndarray:
        return np.full(len(doy), self.value, dtype=float)


def _evaluation_rows(
    *,
    site: str,
    key_count: int,
    prediction: float,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    issue_dates = pd.date_range(start, periods=key_count, freq="D")
    frame = pd.DataFrame(
        {
            "key_id": [f"{site}-k{index}" for index in range(key_count)],
            "site_id": site,
            "issue_date": issue_dates,
            "target_date": issue_dates + pd.Timedelta(days=1),
            "y_true": 0.0,
            **{column: 0 for column in V4.SUBSTITUTION_COLUMNS},
            "forcing_substitution_count": 0,
        }
    )
    frame["_prediction"] = float(prediction)
    return frame


def _key_frame(
    evaluation: pd.DataFrame,
    arm: str,
    *,
    model: str = "LightGBM",
) -> pd.DataFrame:
    prediction = evaluation.pop("_prediction").to_numpy(float)
    return V4.make_key_level_frame(
        evaluation,
        prediction,
        prediction,
        arm=arm,
        model=model,
        horizon=1,
    )


def _protocol_text(*, status: str, authorized: bool, seal_path: Path | None) -> str:
    rendered_seal = "null" if seal_path is None else json.dumps(str(seal_path))
    return f"""\
protocol_id: {V4.EXPECTED_PROTOCOL_ID}
version: {V4.EXPECTED_PROTOCOL_VERSION}
status: {status}
execution_authorized: {str(authorized).lower()}
seal_path: {rendered_seal}
chronology_and_evidence_status:
  already_viewed_domains:
    - existing F0 tree-model result
    - existing F3_full tree-model result
  normalization_only_domains:
    - station-paired re-estimation of already-viewed F3_full effects
"""


def test_f1_fits_one_train_only_climatology_per_meteorological_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", (SITE,))
    dates = pd.date_range("2006-01-01", periods=730, freq="D")
    expected = {
        "TEMP": 11.0,
        "PRCP": 0.25,
        "DH": 181.0,
        "RHMEAN": 67.0,
        "WDSP": 4.5,
    }
    panel = pd.DataFrame({"site_id": SITE, "DATE": dates})
    training_mask = dates <= pd.Timestamp("2006-12-31")
    for variable, value in expected.items():
        panel[variable] = np.where(training_mask, value, value + 10_000.0)

    climatologies = V4.fit_meteorological_climatologies(
        panel,
        np.asarray(training_mask, dtype=bool),
        (SITE,),
    )
    future = V4.future_met_columns(
        panel,
        climatologies,
        "F1",
        max_future_days=1,
    )

    assert future is not None
    assert set(climatologies) == set(V4.META_VARS)
    for variable, value in expected.items():
        np.testing.assert_allclose(future[f"{variable}_fut1"], value, atol=1e-8)
        assert not future[f"{variable}_fut1_substituted"].any()
    assert future["DH_fut1"].iloc[0] != future["TEMP_fut1"].iloc[0]
    assert future["RHMEAN_fut1"].iloc[0] != future["WDSP_fut1"].iloc[0]


@pytest.mark.parametrize(
    ("arms", "models", "leads", "message"),
    [
        ("F0,F2", "LightGBM", "1", "unsupported forcing arms"),
        ("F0", "TCN", "1", "unsupported models"),
        ("F0", "PlainCausalTCN", "1", "unsupported models"),
        ("F0", "LightGBM", "2", "unsupported leads"),
        ("F0", "LightGBM", "1,01", "duplicate integer values"),
        ("F0,F1", "air2stream", "1", "never silently dropped"),
        ("F3", "LightGBM", "1", "legacy engineering arm"),
        ("F3_temperature_only", "LightGBM", "1", "not implemented"),
        ("F3_full", "LightGBM", "1", "must include the F0 reference arm"),
    ],
)
def test_request_enums_and_model_arm_compatibility_fail_closed(
    arms: str,
    models: str,
    leads: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        V4.validate_request(arms, models, leads)


def test_legacy_f3_alias_is_explicit_and_rows_are_canonical_protocol_arm() -> None:
    arms, _models, _leads, cells = V4.validate_request(
        "F0,F3",
        "LightGBM",
        "1",
        allow_legacy_f3_alias=True,
    )
    assert arms == ["F0", "F3_full"]
    assert cells[-1] == V4.Cell("F3_full", "LightGBM", 1)

    evaluation = _evaluation_rows(site=SITE, key_count=1, prediction=1.0)
    frame = _key_frame(evaluation, "F3_full")
    assert frame.loc[0, "arm"] == "F3_full"
    assert frame.loc[0, "protocol_arm"] == "F3_full"

    mislabeled = frame.copy()
    mislabeled["protocol_arm"] = "F3_temperature_only"
    with pytest.raises(AssertionError, match="protocol_arm"):
        V4.validate_key_level_frame(mislabeled)


def test_normalization_scope_rejects_f1_and_air2stream_new_outcomes() -> None:
    f1_cell = [V4.Cell("F1", "LightGBM", 1)]
    hybrid_cell = [V4.Cell("F3_full", "air2stream", 1)]
    with pytest.raises(ValueError, match="new-outcome cells are forbidden"):
        V4.validate_normalization_scope(f1_cell)
    with pytest.raises(ValueError, match="new-outcome cells are forbidden"):
        V4.validate_normalization_scope(hybrid_cell)


def test_air2stream_worker_budget_is_explicit_and_formally_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THERMOROUTE_FORMAL_THREADS", "4")
    assert V4.resolve_air2stream_worker_budget(None) == 1
    assert V4.resolve_air2stream_worker_budget(4) == 4
    with pytest.raises(ValueError, match="exceeds"):
        V4.resolve_air2stream_worker_budget(5)


def test_hybrid_uses_wtemp_and_flow_as_named_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "site_id": [SITE, SITE],
            "DATE": pd.to_datetime(["2018-01-01", "2018-01-02"]),
            "TEMP": [11.0, 12.0],
            "WTEMP": [22.0, 23.0],
            "FLOW": [33.0, 34.0],
            "TEMP_observed": [1, 1],
            "WTEMP_observed": [1, 1],
            "FLOW_observed": [1, 1],
        }
    )
    state_maps = V4._build_station_state_maps(panel)
    state = state_maps[SITE][pd.Timestamp("2018-01-01")]
    assert state.air_temp == 11.0
    assert state.water_temp == 22.0
    assert state.flow == 33.0
    assert state.water_temp_available
    assert state.flow_available

    captured: dict[str, object] = {}

    def fake_forecast(
        _fit: object,
        water_temp: float,
        flow: float,
        doy: int,
        future_air: np.ndarray,
        future_doy: np.ndarray,
    ) -> float:
        captured.update(
            water_temp=water_temp,
            flow=flow,
            doy=doy,
            future_air=future_air.copy(),
            future_doy=future_doy.copy(),
        )
        return 24.0

    monkeypatch.setattr(air2stream, "forecast_horizon", fake_forecast)
    reference = pd.DataFrame(
        {
            "key_id": ["key-1"],
            "site_id": [SITE],
            "horizon": [1],
            "issue_date": pd.to_datetime(["2018-01-01"]),
            "target_date": pd.to_datetime(["2018-01-02"]),
            "y_true": [23.0],
        }
    )
    result = V4.run_air2stream_cell(
        {SITE: object()},
        state_maps,
        _ConstantClimatology(99.0),
        reference,
        arm="F3_full",
        horizon=1,
    )

    assert captured["water_temp"] == 22.0
    assert captured["flow"] == 33.0
    np.testing.assert_array_equal(captured["future_air"], [12.0])
    assert result.loc[0, "y_pred"] == 24.0
    assert result.loc[0, "forcing_substitution_count"] == 0
    assert result.loc[0, "protocol_arm"] == "F3_full"

    unavailable_flow = panel.copy()
    unavailable_flow.loc[0, "FLOW_observed"] = 0
    with pytest.raises(RuntimeError, match="produced no predictions"):
        V4.run_air2stream_cell(
            {SITE: object()},
            V4._build_station_state_maps(unavailable_flow),
            _ConstantClimatology(99.0),
            reference,
            arm="F3_full",
            horizon=1,
        )


def test_f3_uses_matching_variable_climatology_and_accounts_per_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", (SITE,))
    dates = pd.date_range("2018-01-01", periods=3, freq="D")
    panel = pd.DataFrame(
        {
            "site_id": SITE,
            "DATE": dates,
            "TEMP": [10.0, 20.0, 30.0],
            "PRCP": [1.0, np.nan, 3.0],
            "DH": [100.0, 101.0, 102.0],
            "RHMEAN": [60.0, 61.0, 62.0],
            "WDSP": [4.0, 5.0, 6.0],
        }
    )
    for variable in V4.META_VARS:
        panel[f"{variable}_observed"] = 1
    panel.loc[1, "PRCP_observed"] = 0
    values = {
        "TEMP": 111.0,
        "PRCP": 222.0,
        "DH": 333.0,
        "RHMEAN": 444.0,
        "WDSP": 555.0,
    }
    climatologies = {variable: _ConstantClimatology(value) for variable, value in values.items()}

    future = V4.future_met_columns(
        panel,
        climatologies,
        "F3_full",
        max_future_days=1,
    )
    assert future is not None
    first = future.loc[future.DATE.eq(dates[0])].iloc[0]
    assert first["TEMP_fut1"] == 20.0
    assert not bool(first["TEMP_fut1_substituted"])
    assert first["PRCP_fut1"] == 222.0
    assert bool(first["PRCP_fut1_substituted"])

    tab = pd.DataFrame({"site_id": [SITE], "issue_date": [dates[0]]})
    attached, added = V4.attach_future_features(tab, future, horizon=1)
    assert set(added) == {
        f"{variable}_{suffix}" for variable in V4.META_VARS for suffix in ("fut1", "futmean1")
    }
    assert attached.loc[0, "forcing_substitutions_PRCP"] == 1
    assert attached.loc[0, "forcing_substitutions_TEMP"] == 0
    assert attached.loc[0, "forcing_substitution_count"] == 1

    evaluation = attached.assign(
        key_id="key-1",
        target_date=dates[0] + pd.Timedelta(days=1),
        y_true=12.0,
    )
    key_frame = V4.make_key_level_frame(
        evaluation,
        [12.5],
        [12.25],
        arm="F3_full",
        model="LightGBM",
        horizon=1,
    )
    checked = V4.validate_key_level_frame(
        key_frame,
        cell=V4.Cell("F3_full", "LightGBM", 1),
    )
    assert tuple(checked.columns) == V4.KEY_LEVEL_COLUMNS
    assert checked.loc[0, "forcing_substitution_count"] == 1


def test_target_binding_uses_absolute_only_tolerance_and_exact_registry_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_target = float(np.float32(32.4))
    issue_dates = pd.to_datetime(["2017-12-31", "2018-01-01"])
    tab = pd.DataFrame(
        {
            "site_id": [SITE, SITE],
            "issue_date": issue_dates,
            "target_date": issue_dates + pd.Timedelta(days=1),
            "split": ["train", "confirm"],
            "issue_wtemp_observed": [1, 1],
            "clim_target": [10.0, 10.0],
            "persistence": [10.0, 10.0],
            "clim_t": [10.0, 10.0],
            "y": [10.0, 32.4],
            "feature": [0.0, 1.0],
            **{column: 0 for column in V4.SUBSTITUTION_COLUMNS},
            "forcing_substitution_count": [0, 0],
        }
    )
    reference = pd.DataFrame(
        {
            "key_id": ["registry-key"],
            "site_id": [SITE],
            "horizon": [1],
            "issue_date": [issue_dates[1]],
            "target_date": [issue_dates[1] + pd.Timedelta(days=1)],
            "y_true": [registry_target],
        }
    )

    class _Fitted:
        def predict(self, rows: pd.DataFrame, *, num_threads: int) -> np.ndarray:
            assert num_threads == 1
            return np.zeros(len(rows), dtype=float)

    monkeypatch.setattr(V4, "_lgb_fit", lambda *args, **kwargs: _Fitted())
    result = V4._run_tree_cell(
        tab,
        ["feature"],
        {SITE: 0.9},
        reference,
        arm="F0",
        model="LightGBM",
        horizon=1,
    )
    assert result.loc[0, "y_true"] == registry_target
    assert result.loc[0, "y_true"] != tab.loc[1, "y"]
    assert V4.target_values_equal([registry_target], [32.4])

    outside = tab.copy()
    outside.loc[1, "y"] = registry_target + V4.TARGET_ABSOLUTE_TOLERANCE + 1e-7
    with pytest.raises(AssertionError, match="beyond absolute tolerance"):
        V4._run_tree_cell(
            outside,
            ["feature"],
            {SITE: 0.9},
            reference,
            arm="F0",
            model="LightGBM",
            horizon=1,
        )

    audit = V4.target_value_binding_audit()
    assert audit["absolute_tolerance"] == 2e-6
    assert audit["relative_tolerance"] == 0.0
    assert audit["identity_fields_remain_exact"] is True
    assert audit["persisted_value_source"] == "forecast_key_registry"


def test_station_paired_estimand_and_reportability_are_not_marginal_medians() -> None:
    reference_rmse = {"site-a": 0.0, "site-b": 100.0, "site-c": 200.0}
    candidate_rmse = {"site-a": 100.0, "site-b": 200.0, "site-c": 100.0}
    frames: list[pd.DataFrame] = []
    for site, prediction in reference_rmse.items():
        frames.append(
            _key_frame(
                _evaluation_rows(site=site, key_count=2, prediction=prediction),
                "F0",
            )
        )
    for site, prediction in candidate_rmse.items():
        frames.append(
            _key_frame(
                _evaluation_rows(site=site, key_count=2, prediction=prediction),
                "F3_full",
            )
        )
    # One underpowered station must be retained for audit but excluded from the
    # reportable aggregate.
    frames.append(
        _key_frame(
            _evaluation_rows(site="site-d", key_count=1, prediction=1.0),
            "F0",
        )
    )
    frames.append(
        _key_frame(
            _evaluation_rows(site="site-d", key_count=1, prediction=1001.0),
            "F3_full",
        )
    )
    predictions = pd.concat(frames, ignore_index=True)

    effects = V4.station_metrics_from_predictions(predictions, min_keys=2)
    contrasts = V4.paired_arm_contrasts(predictions, min_keys=2)
    summary = V4.summarize_paired_contrasts(contrasts)

    assert set(effects["protocol_arm"]) == {"F0", "F3_full"}
    assert set(effects["model_status"]) == {"frozen_tree_model"}
    assert set(contrasts["contrast_role"]) == {"primary_tree_F3_full_minus_F0"}
    assert set(contrasts["reference_protocol_arm"]) == {"F0"}
    assert not effects.loc[effects.site_id.eq("site-d"), "reportable"].any()
    assert not contrasts.loc[contrasts.site_id.eq("site-d"), "reportable"].any()
    reportable = effects[effects.reportable]
    marginal_difference = (
        reportable.loc[reportable.arm.eq("F3_full"), "rmse"].median()
        - reportable.loc[reportable.arm.eq("F0"), "rmse"].median()
    )
    result = summary["LightGBM/h1/F3_full_minus_F0"]
    assert marginal_difference == 0.0
    assert result["median_station_delta_rmse"] == 100.0
    assert result["median_forcing_value_rmse"] == -100.0
    assert result["estimand"] == "median_i[RMSE_i(F3_full)-RMSE_i(F0)]"
    assert result["forcing_value_estimand"] == "median_i[RMSE_i(F0)-RMSE_i(F3_full)]"
    assert "positive forcing value means improvement" in result["forcing_value_sign_relation"]
    assert result["n_reportable_stations"] == 3

    governance = {
        "analysis_mode": "POST_OUTCOME_NORMALIZATION_ONLY",
        "post_outcome_normalization": True,
    }
    document = V4.build_summary(
        predictions,
        effects,
        contrasts,
        [V4.Cell("F0", "LightGBM", 1), V4.Cell("F3_full", "LightGBM", 1)],
        governance=governance,
        legacy_f3_alias_used=False,
        air2stream_worker_budget=1,
    )
    assert document["analysis_mode"] == "POST_OUTCOME_NORMALIZATION_ONLY"
    assert document["post_outcome_normalization"] is True
    assert document["governance"] == governance
    assert document["arm_identity"]["legacy_F3_alias_used"] is False
    assert document["arm_identity"]["F3_temperature_only_implemented"] is False
    assert document["cells"][1]["protocol_arm"] == "F3_full"
    assert document["target_value_binding"]["absolute_tolerance"] == 2e-6
    paired_anchor = document["paired_learned_vs_damped"]["LightGBM/h1/F3_full"]
    assert paired_anchor["estimand"] == "median_i[RMSE_i(model)-RMSE_i(damped)]"
    assert paired_anchor["median_station_learned_minus_damped_rmse"] == 0.0
    semantics = document["F3_full_input_semantics"]
    assert "gridded meteorological estimates" in semantics["description"]
    assert "not direct station observations" in semantics["description"]
    assert "proxy" in semantics["field_caveats"]["RHMEAN"]
    assert "daylight-period mean" in semantics["field_caveats"]["DH"]
    assert "gridMET" in semantics["field_caveats"]["WDSP"]


def test_unofficial_hybrid_contrast_is_f3_full_minus_f1_and_never_silent() -> None:
    f1 = _key_frame(
        _evaluation_rows(site=SITE, key_count=2, prediction=1.0),
        "F1",
        model="air2stream",
    )
    f3_full = _key_frame(
        _evaluation_rows(site=SITE, key_count=2, prediction=3.0),
        "F3_full",
        model="air2stream",
    )
    contrasts = V4.paired_arm_contrasts(
        pd.concat([f1, f3_full], ignore_index=True),
        min_keys=2,
    )
    assert len(contrasts) == 1
    row = contrasts.iloc[0]
    assert row["protocol_arm"] == "F3_full"
    assert row["reference_protocol_arm"] == "F1"
    assert row["contrast_role"] == "secondary_unofficial_hybrid_F3_full_minus_F1"
    assert row["model_status"] == "unofficial_unvalidated_hybrid"

    summary = V4.summarize_paired_contrasts(contrasts)
    result = summary["air2stream/h1/F3_full_minus_F1"]
    assert result["primary_forcing_value"] is False
    assert result["validated_hybrid"] is False
    assert result["estimand"] == "median_i[RMSE_i(F3_full)-RMSE_i(F1)]"
    assert "median_forcing_value_rmse" not in result
    assert "forcing_value_estimand" not in result

    with pytest.raises(AssertionError, match="requires exactly F3_full minus F1"):
        V4.paired_arm_contrasts(f1, min_keys=2)


def test_tree_f1_contrast_is_labeled_diagnostic_not_primary() -> None:
    f0 = _key_frame(
        _evaluation_rows(site=SITE, key_count=2, prediction=1.0),
        "F0",
    )
    f1 = _key_frame(
        _evaluation_rows(site=SITE, key_count=2, prediction=2.0),
        "F1",
    )
    contrasts = V4.paired_arm_contrasts(
        pd.concat([f0, f1], ignore_index=True),
        min_keys=2,
    )
    assert set(contrasts["contrast_role"]) == {"diagnostic_tree_F1_minus_F0"}
    result = V4.summarize_paired_contrasts(contrasts)["LightGBM/h1/F1_minus_F0"]
    assert result["primary_forcing_value"] is False
    assert result["median_forcing_value_rmse"] == -1.0


def test_p5_components_remain_independent_and_emit_no_automatic_verdict() -> None:
    effect_rows: list[dict[str, object]] = []
    for arm, learned, damped in (
        ("F0", [1.60, 1.80], [1.64, 1.84]),
        ("F3_full", [1.16, 1.24], [1.51, 1.59]),
    ):
        for index, (rmse, rmse_damped) in enumerate(zip(learned, damped, strict=True)):
            effect_rows.append(
                {
                    "arm": arm,
                    "protocol_arm": arm,
                    "model": "LightGBM",
                    "model_status": "frozen_tree_model",
                    "site_id": f"site-{index}",
                    "horizon": 7,
                    "n": 100,
                    "rmse": rmse,
                    "rmse_damped": rmse_damped,
                    **{column: 0 for column in V4.SUBSTITUTION_COLUMNS},
                    "forcing_substitution_count": 0,
                    "reportable": True,
                }
            )
    effects = pd.DataFrame(effect_rows, columns=V4.EFFECT_COLUMNS)
    predictions = pd.DataFrame(
        [
            {
                "arm": arm,
                "protocol_arm": arm,
                "model": "LightGBM",
                "horizon": 7,
                **{column: 0 for column in V4.SUBSTITUTION_COLUMNS},
                "forcing_substitution_count": 0,
            }
            for arm in ("F0", "F3_full")
        ]
    )
    summary = V4.build_summary(
        predictions,
        effects,
        pd.DataFrame(columns=V4.CONTRAST_COLUMNS),
        [V4.Cell("F0", "LightGBM", 7), V4.Cell("F3_full", "LightGBM", 7)],
        governance={
            "analysis_mode": "POST_OUTCOME_NORMALIZATION_ONLY",
            "post_outcome_normalization": True,
        },
        legacy_f3_alias_used=False,
        air2stream_worker_budget=1,
    )

    components = summary["P5_independent_components"]["LightGBM"]
    rmse_range = components["seven_day_F3_full_rmse_range"]
    learned_threshold = components["seven_day_learned_advantage_threshold"]
    assert rmse_range["observed_median_station_rmse"] == 1.2
    assert rmse_range["within_registered_range"] is True
    assert learned_threshold["observed_median_station_learned_minus_damped_rmse"] == pytest.approx(
        -0.35
    )
    assert learned_threshold["meets_registered_threshold"] is True
    assert components["adjudication_status"] == "COMPONENT_VALUES_ONLY_NO_AUTOMATIC_VERDICT"
    assert summary["P5_joint_verdict"] == "NOT_ADJUDICATED_BY_RUNNER"


def test_draft_protocol_blocks_f1_before_any_output_or_training(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        _protocol_text(
            status="DRAFT_NOT_SEALED",
            authorized=False,
            seal_path=None,
        ),
        encoding="utf-8",
    )
    args = V4.build_parser().parse_args(
        [
            "--arms",
            "F0,F1",
            "--models",
            "LightGBM",
            "--leads",
            "1",
            "--protocol",
            str(protocol),
            "--output-dir",
            str(tmp_path / "forbidden-output"),
        ]
    )

    with pytest.raises(ValueError, match="governance-blocked"):
        V4.run(args)
    assert not (tmp_path / "forbidden-output").exists()

    governance = V4.validate_execution_governance(
        protocol,
        normalization_only=True,
    )
    assert governance["analysis_mode"] == "POST_OUTCOME_NORMALIZATION_ONLY"
    assert governance["post_outcome_normalization"] is True
    assert governance["execution_authorized_by_seal"] is False


def test_sealed_execution_requires_exact_protocol_byte_binding(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    seal_path = tmp_path / "protocol_seal.json"
    protocol.write_text(
        _protocol_text(
            status="SEALED",
            authorized=True,
            seal_path=seal_path,
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    seal = {
        "format": V4.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED",
        "execution_authorized": True,
        "protocol": {
            "path": str(protocol),
            "protocol_id": V4.EXPECTED_PROTOCOL_ID,
            "version": V4.EXPECTED_PROTOCOL_VERSION,
            "sha256": digest,
        },
    }
    seal_path.write_text(json.dumps(seal), encoding="utf-8")

    governance = V4.validate_execution_governance(
        protocol,
        normalization_only=False,
    )
    assert governance["analysis_mode"] == "SEALED_AUTHORIZED_EXTENSION_EXECUTION"
    assert governance["execution_authorized_by_seal"] is True

    protocol.write_text(protocol.read_text(encoding="utf-8") + "# changed\n")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        V4.validate_execution_governance(protocol, normalization_only=False)


def test_atomic_v4_shard_and_completeness_gate(tmp_path: Path) -> None:
    cell = V4.Cell("F0", "LightGBM", 1)
    evaluation = _evaluation_rows(site=SITE, key_count=1, prediction=1.0)
    frame = _key_frame(evaluation, "F0")
    reference = frame[
        ["key_id", "site_id", "horizon", "issue_date", "target_date", "y_true"]
    ].copy()

    with pytest.raises(AssertionError, match="completeness gate"):
        V4.validate_complete_shards(tmp_path, [cell], reference)

    path = V4.shard_path(tmp_path, cell)
    V4.atomic_write_parquet(frame, path, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        V4.atomic_write_parquet(frame, path, overwrite=False)
    combined = V4.validate_complete_shards(tmp_path, [cell], reference)
    assert list(combined.key_id) == list(frame.key_id)
    assert path.parent.name == "forcing_shards_v4"

    wrong_binding = reference.assign(site_id="wrong-site")
    with pytest.raises(AssertionError, match="identity/chronology differs"):
        V4.validate_key_level_frame(
            frame,
            cell=cell,
            reference=wrong_binding,
            require_full_registry=True,
        )
