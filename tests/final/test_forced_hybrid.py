"""Contracts for the transparent forced thermal-response reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.final.run_forced_hybrid_reference as RUNNER
from thermoroute.forced_hybrid import (
    ALPHA_MAX,
    ALPHA_MIN,
    ATMOSPHERIC_VARIABLES,
    CHANNEL_F0,
    CHANNEL_F2A_TEMPERATURE_ONLY,
    CHANNEL_F3_FULL,
    CHANNEL_F3_TEMPERATURE_ONLY,
    COEFFICIENT_NAMES,
    FORCING_TERMS,
    FORCING_UNITS,
    FUTURE_DISCHARGE_ENABLED_BY_DEFAULT,
    LEGAL_CHANNELS,
    LEGAL_HORIZONS,
    PREDICTION_COLUMNS,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    ForcedHybridContractError,
    ForcedHybridModel,
    bind_formal_keys,
    forecast_key_id,
)

SITE_A = "01000001"
SITE_B = "01000002"
TRUE_COEFFICIENTS = {
    "intercept": 1.0,
    "TEMP": 0.70,
    "DH": 0.50,
    "RHMEAN": 0.10,
    "WDSP": -0.05,
    "PRCP": -0.10,
}
TRUE_ALPHA = {SITE_A: 0.15, SITE_B: 0.25}


def _known_model() -> ForcedHybridModel:
    return ForcedHybridModel(
        coefficients={
            "intercept": 1.0,
            "TEMP": 0.80,
            "DH": 1.20,
            "RHMEAN": 0.30,
            "WDSP": -0.40,
            "PRCP": -0.50,
        },
        alpha_by_station={SITE_A: 0.25},
        training_pair_count_by_station={SITE_A: 100},
        iterations=4,
        converged=True,
    )


def _issue_forcing() -> dict[str, float]:
    return {
        "TEMP": 10.0,
        "DH": 200.0,
        "RHMEAN": 60.0,
        "WDSP": 3.0,
        "PRCP": 2.0,
    }


def _temperature_future(issue: str = "2016-06-01") -> pd.DataFrame:
    issue_date = pd.Timestamp(issue)
    return pd.DataFrame(
        {
            "step": np.asarray([3], dtype=np.int16),
            "valid_date": np.asarray([issue_date + pd.Timedelta(days=3)]),
            "TEMP": np.asarray([12.0], dtype=np.float64),
        }
    )


def _full_future(issue: str = "2016-06-01") -> pd.DataFrame:
    issue_date = pd.Timestamp(issue)
    return pd.DataFrame(
        {
            "step": np.asarray([1, 2, 3], dtype=np.int16),
            "valid_date": pd.date_range(issue_date + pd.Timedelta(days=1), periods=3),
            "TEMP": np.asarray([11.0, 13.0, 12.0], dtype=np.float64),
            "DH": np.asarray([220.0, 250.0, 180.0], dtype=np.float64),
            "RHMEAN": np.asarray([62.0, 55.0, 70.0], dtype=np.float64),
            "WDSP": np.asarray([3.5, 2.0, 4.0], dtype=np.float64),
            "PRCP": np.asarray([0.0, 5.0, 20.0], dtype=np.float64),
        },
        columns=("step", "valid_date", *ATMOSPHERIC_VARIABLES),
    )


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(27101)
    dates = pd.date_range("2006-01-01", "2017-12-31", freq="D")
    frames: list[pd.DataFrame] = []
    coefficients = np.asarray(
        [TRUE_COEFFICIENTS[name] for name in COEFFICIENT_NAMES], dtype=np.float64
    )
    for station in (SITE_A, SITE_B):
        phase = np.arange(len(dates), dtype=np.float64)
        air = 10.0 + 8.0 * np.sin(2.0 * np.pi * phase / 365.0) + rng.normal(0.0, 0.5, len(dates))
        shortwave = np.clip(
            200.0
            + 100.0 * np.sin(2.0 * np.pi * (phase - 80.0) / 365.0)
            + rng.normal(0.0, 5.0, len(dates)),
            0.0,
            None,
        )
        humidity = np.clip(
            65.0 - 10.0 * np.sin(2.0 * np.pi * phase / 365.0) + rng.normal(0.0, 2.0, len(dates)),
            0.0,
            100.0,
        )
        wind = np.clip(3.0 + rng.normal(0.0, 0.2, len(dates)), 0.0, None)
        precipitation = np.clip(rng.gamma(1.0, 2.0, len(dates)) - 1.0, 0.0, None)
        design = np.column_stack(
            [
                np.ones(len(dates)),
                air,
                shortwave / 100.0,
                humidity / 10.0,
                wind,
                precipitation / 10.0,
            ]
        )
        equilibrium = design @ coefficients
        water = np.empty(len(dates), dtype=np.float64)
        water[0] = equilibrium[0]
        for index in range(1, len(dates)):
            water[index] = water[index - 1] + TRUE_ALPHA[station] * (
                equilibrium[index] - water[index - 1]
            )
        frames.append(
            pd.DataFrame(
                {
                    "DATE": dates,
                    "site_id": np.full(len(dates), station, dtype=object),
                    "WTEMP": water,
                    "TEMP": air,
                    "DH": shortwave,
                    "RHMEAN": humidity,
                    "WDSP": wind,
                    "PRCP": precipitation,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_equation_terms_units_bounds_and_physical_directions_are_explicit() -> None:
    assert tuple(term.variable for term in FORCING_TERMS) == ATMOSPHERIC_VARIABLES
    assert {term.variable: term.input_unit for term in FORCING_TERMS} == dict(FORCING_UNITS)
    assert ALPHA_MIN > 0.0
    assert ALPHA_MAX == 1.0
    assert FUTURE_DISCHARGE_ENABLED_BY_DEFAULT is False

    model = _known_model()
    baseline = _issue_forcing()
    base_equilibrium = model.equilibrium_temperature(baseline, forcing_units=FORCING_UNITS)[0]
    for term in FORCING_TERMS:
        perturbed = dict(baseline)
        perturbed[term.variable] += term.reference_increment
        response = (
            model.equilibrium_temperature(perturbed, forcing_units=FORCING_UNITS)[0]
            - base_equilibrium
        )
        assert response == pytest.approx(model.coefficients[term.variable])
        if term.physical_direction == "nonnegative":
            assert response > 0.0
        else:
            assert term.physical_direction == "nonpositive"
            assert response < 0.0

    wrong_units = dict(FORCING_UNITS)
    wrong_units["DH"] = "MJ m-2 day-1"
    with pytest.raises(ForcedHybridContractError, match="forcing units"):
        model.equilibrium_temperature(baseline, forcing_units=wrong_units)

    for bad_alpha in (0.0, 1.01):
        with pytest.raises(ForcedHybridContractError, match="alpha"):
            ForcedHybridModel(
                coefficients=TRUE_COEFFICIENTS,
                alpha_by_station={SITE_A: bad_alpha},
                training_pair_count_by_station={SITE_A: 100},
                iterations=1,
                converged=True,
            )
    tiny_positive = ForcedHybridModel(
        coefficients=TRUE_COEFFICIENTS,
        alpha_by_station={SITE_A: ALPHA_MIN / 2.0},
        training_pair_count_by_station={SITE_A: 100},
        iterations=1,
        converged=True,
    )
    assert 0.0 < tiny_positive.alpha_by_station[SITE_A] < ALPHA_MIN


def test_fit_is_training_only_and_validation_mutation_invariant(
    synthetic_panel: pd.DataFrame,
) -> None:
    fitted = ForcedHybridModel.fit(synthetic_panel, forcing_units=FORCING_UNITS)
    assert fitted.converged
    assert fitted.to_payload()["training_period"] == ["2006-01-01", "2015-12-31"]
    assert fitted.to_payload()["validation_period"] == ["2016-01-01", "2017-12-31"]
    assert TRAIN_START == pd.Timestamp("2006-01-01")
    assert TRAIN_END == pd.Timestamp("2015-12-31")
    assert VALIDATION_START == pd.Timestamp("2016-01-01")
    assert VALIDATION_END == pd.Timestamp("2017-12-31")
    assert fitted.training_pair_count_by_station == {SITE_A: 3_651, SITE_B: 3_651}
    for name, expected in TRUE_COEFFICIENTS.items():
        assert fitted.coefficients[name] == pytest.approx(expected, abs=2.0e-7)
    for station, expected in TRUE_ALPHA.items():
        assert fitted.alpha_by_station[station] == pytest.approx(expected, abs=2.0e-7)

    attacked = synthetic_panel.copy(deep=True)
    validation = attacked["DATE"].between(VALIDATION_START, VALIDATION_END)
    attacked.loc[validation, "WTEMP"] += 20.0
    attacked.loc[validation, "TEMP"] -= 30.0
    attacked.loc[validation, "DH"] += 500.0
    attacked.loc[validation, "RHMEAN"] = 1.0
    attacked.loc[validation, "WDSP"] += 10.0
    attacked.loc[validation, "PRCP"] += 100.0
    refitted = ForcedHybridModel.fit(attacked, forcing_units=FORCING_UNITS)
    assert refitted.to_json() == fitted.to_json()
    assert refitted.sha256 == fitted.sha256
    with pytest.raises(ForcedHybridContractError, match="did not converge"):
        ForcedHybridModel.fit(
            synthetic_panel,
            forcing_units=FORCING_UNITS,
            max_iterations=1,
        )
    unusable_station = synthetic_panel.loc[
        synthetic_panel["site_id"].eq(SITE_A)
        & synthetic_panel["DATE"].between("2006-01-01", "2006-02-28")
    ].copy()
    unusable_station["site_id"] = "01000003"
    unusable_station["TEMP"] = np.nan
    with pytest.raises(ForcedHybridContractError, match="minimum complete training pairs"):
        ForcedHybridModel.fit(
            pd.concat([synthetic_panel, unusable_station], ignore_index=True),
            forcing_units=FORCING_UNITS,
        )


def test_legal_forcing_channels_are_exact_and_discharge_is_separate() -> None:
    model = _known_model()
    issue_date = pd.Timestamp("2016-06-01")
    common = {
        "station": SITE_A,
        "issue_date": issue_date,
        "temperature_current": 9.0,
        "issue_forcing": _issue_forcing(),
        "horizon": 3,
        "forcing_units": FORCING_UNITS,
    }
    f0 = model.rollout(**common, future_forcing=None, channel=CHANNEL_F0)
    f2a = model.rollout(
        **common,
        future_forcing=_temperature_future(),
        channel=CHANNEL_F2A_TEMPERATURE_ONLY,
    )
    f3_temperature = model.rollout(
        **common,
        future_forcing=_temperature_future(),
        channel=CHANNEL_F3_TEMPERATURE_ONLY,
    )
    f3_full = model.rollout(
        **common,
        future_forcing=_full_future(),
        channel=CHANNEL_F3_FULL,
    )

    assert LEGAL_CHANNELS == (
        "F0",
        "F2a_temperature_only",
        "F3_temperature_only",
        "F3_full",
    )
    assert LEGAL_HORIZONS == (1, 3, 7)
    assert tuple(f0.columns) == PREDICTION_COLUMNS
    assert f0["temperature_equilibrium"].nunique() == 1
    np.testing.assert_array_equal(f2a["y_pred"], f3_temperature["y_pred"])
    np.testing.assert_array_equal(
        f2a.loc[:1, "temperature_equilibrium"],
        f0.loc[:1, "temperature_equilibrium"],
    )
    assert f2a.loc[2, "temperature_equilibrium"] != f0.loc[2, "temperature_equilibrium"]
    assert not np.array_equal(f3_full["y_pred"], f3_temperature["y_pred"])
    assert set(f0["channel"]) == {CHANNEL_F0}
    assert set(f2a["channel"]) == {CHANNEL_F2A_TEMPERATURE_ONLY}

    with pytest.raises(ForcedHybridContractError, match="F0 forbids"):
        model.rollout(
            **common,
            future_forcing=_temperature_future(),
            channel=CHANNEL_F0,
        )
    with pytest.raises(ForcedHybridContractError, match="future schema"):
        model.rollout(
            **common,
            future_forcing=_full_future(),
            channel=CHANNEL_F2A_TEMPERATURE_ONLY,
        )
    reconstructed_trajectory = _full_future()[["step", "valid_date", "TEMP"]]
    with pytest.raises(ForcedHybridContractError, match="one target-day value"):
        model.rollout(
            **common,
            future_forcing=reconstructed_trajectory,
            channel=CHANNEL_F2A_TEMPERATURE_ONLY,
        )
    bundled_discharge = _full_future().assign(FLOW=100.0)
    with pytest.raises(ForcedHybridContractError, match="future schema"):
        model.rollout(
            **common,
            future_forcing=bundled_discharge,
            channel=CHANNEL_F3_FULL,
        )
    with pytest.raises(ForcedHybridContractError, match="future discharge"):
        model.rollout(
            **common,
            future_forcing=_full_future(),
            channel=CHANNEL_F3_FULL,
            future_discharge=pd.DataFrame({"FLOW": [100.0, 100.0, 100.0]}),
        )
    invalid_horizon = dict(common)
    invalid_horizon["horizon"] = 2
    with pytest.raises(ForcedHybridContractError, match="frozen value"):
        model.rollout(
            **invalid_horizon,
            future_forcing=None,
            channel=CHANNEL_F0,
        )


def test_rollout_is_recurrent_daily_and_valid_date_bound() -> None:
    model = _known_model()
    future = _full_future()
    initial = 7.5
    predictions = model.rollout(
        station=SITE_A,
        issue_date="2016-06-01",
        temperature_current=initial,
        issue_forcing=_issue_forcing(),
        future_forcing=future,
        horizon=3,
        channel=CHANNEL_F3_FULL,
        forcing_units=FORCING_UNITS,
    )
    manual_current = initial
    manual: list[float] = []
    for forcing_row in future[list(ATMOSPHERIC_VARIABLES)].to_dict(orient="records"):
        manual_current = model.step(
            SITE_A,
            manual_current,
            forcing_row,
            forcing_units=FORCING_UNITS,
        )
        manual.append(manual_current)
    np.testing.assert_allclose(predictions["y_pred"], manual, rtol=0.0, atol=1.0e-14)

    second_equilibrium = predictions.loc[1, "temperature_equilibrium"]
    non_recurrent_second = initial + model.alpha_by_station[SITE_A] * (second_equilibrium - initial)
    assert predictions.loc[1, "y_pred"] != pytest.approx(non_recurrent_second)
    assert list(predictions["target_date"]) == list(pd.date_range("2016-06-02", periods=3))
    endpoint = model.forecast_endpoint(
        station=SITE_A,
        issue_date="2016-06-01",
        temperature_current=initial,
        issue_forcing=_issue_forcing(),
        future_forcing=future,
        horizon=3,
        channel=CHANNEL_F3_FULL,
        forcing_units=FORCING_UNITS,
    )
    assert len(endpoint) == 1
    assert endpoint.loc[0, "lead_days"] == 3
    assert endpoint.loc[0, "y_pred"] == predictions.loc[2, "y_pred"]

    bad_dates = future.copy()
    bad_dates.loc[1, "valid_date"] += pd.Timedelta(days=1)
    with pytest.raises(ForcedHybridContractError, match="valid_date"):
        model.rollout(
            station=SITE_A,
            issue_date="2016-06-01",
            temperature_current=initial,
            issue_forcing=_issue_forcing(),
            future_forcing=bad_dates,
            horizon=3,
            channel=CHANNEL_F3_FULL,
            forcing_units=FORCING_UNITS,
        )


def test_formal_key_binding_is_canonical_two_sided_and_exact() -> None:
    predictions = _known_model().forecast_endpoint(
        station=SITE_A,
        issue_date="2016-06-01",
        temperature_current=8.0,
        issue_forcing=_issue_forcing(),
        future_forcing=_temperature_future(),
        horizon=3,
        channel=CHANNEL_F3_TEMPERATURE_ONLY,
        forcing_units=FORCING_UNITS,
    )
    keys = predictions[["site_id", "issue_date", "target_date", "lead_days"]].copy()
    keys.insert(
        0,
        "key_id",
        [
            forecast_key_id(row.site_id, row.issue_date, row.target_date, int(row.lead_days))
            for row in keys.itertuples(index=False)
        ],
    )
    keys["reportable_primary"] = True
    bound = bind_formal_keys(predictions, keys)
    assert list(bound["key_id"]) == list(keys["key_id"])
    np.testing.assert_array_equal(bound["y_pred"], predictions["y_pred"])

    with pytest.raises(ForcedHybridContractError, match="two-sided exact"):
        bind_formal_keys(predictions, keys.iloc[:-1])
    attacked = keys.copy()
    attacked.loc[0, "key_id"] = "0" * 64
    with pytest.raises(ForcedHybridContractError, match="canonical identity"):
        bind_formal_keys(predictions, attacked)
    attacked = predictions.copy()
    attacked.loc[0, "target_date"] += pd.Timedelta(days=1)
    with pytest.raises(ForcedHybridContractError, match="target boundary"):
        bind_formal_keys(attacked, keys)
    attacked = predictions.copy()
    attacked["channel"] = "F4_undeclared"
    with pytest.raises(ForcedHybridContractError, match="unsupported forcing channel"):
        bind_formal_keys(attacked, keys)
    attacked_predictions = predictions.copy()
    attacked_predictions["lead_days"] = np.int16(2)
    attacked_predictions["target_date"] = attacked_predictions["issue_date"] + pd.Timedelta(days=2)
    attacked_keys = keys.copy()
    attacked_keys["lead_days"] = np.int16(2)
    attacked_keys["target_date"] = attacked_keys["issue_date"] + pd.Timedelta(days=2)
    attacked_keys["key_id"] = forecast_key_id(
        SITE_A,
        attacked_keys.loc[0, "issue_date"],
        attacked_keys.loc[0, "target_date"],
        2,
    )
    with pytest.raises(ForcedHybridContractError, match="frozen horizon set"):
        bind_formal_keys(attacked_predictions, attacked_keys)


def test_serialization_is_canonical_complete_and_rejects_contract_mutation() -> None:
    model = _known_model()
    document = model.to_json()
    restored = ForcedHybridModel.from_json(document)
    assert restored.to_json() == document
    assert restored.sha256 == model.sha256
    assert restored.coefficients == model.coefficients
    assert restored.alpha_by_station == model.alpha_by_station
    assert document.endswith("\n")

    attacked = json.loads(document)
    attacked["forcing_terms"][1]["input_unit"] = "MJ m-2 day-1"
    with pytest.raises(ForcedHybridContractError, match="forcing_terms"):
        ForcedHybridModel.from_payload(attacked)
    attacked = json.loads(document)
    attacked["coefficients"]["WDSP"] = 0.1
    with pytest.raises(ForcedHybridContractError, match="coefficient WDSP"):
        ForcedHybridModel.from_payload(attacked)
    attacked = json.loads(document)
    attacked["alpha_by_station"][SITE_A] = 0.0
    with pytest.raises(ForcedHybridContractError, match="alpha"):
        ForcedHybridModel.from_payload(attacked)
    attacked = json.loads(document)
    attacked["converged"] = False
    with pytest.raises(ForcedHybridContractError, match="converged calibration"):
        ForcedHybridModel.from_payload(attacked)
    with pytest.raises(ForcedHybridContractError, match="repeats JSON key"):
        ForcedHybridModel.from_json('{"schema_version":"x","schema_version":"y"}')


def test_runner_defaults_to_data_free_dry_run_and_execute_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("locked runner crossed a data, fit, or output boundary")

    monkeypatch.setattr(RUNNER, "_sha256_file", forbidden)
    monkeypatch.setattr(RUNNER.pd, "read_parquet", forbidden)
    monkeypatch.setattr(RUNNER.pd, "read_csv", forbidden)
    monkeypatch.setattr(RUNNER.ForcedHybridModel, "fit", forbidden)
    monkeypatch.setattr(RUNNER, "_publish_create_only", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)

    assert RUNNER.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["execution_performed"] is False
    assert plan["equation"] == ("T_next=T_current+alpha_i*(T_equilibrium(F_next)-T_current)")
    assert plan["periods"] == {
        "training": ["2006-01-01", "2015-12-31"],
        "validation": ["2016-01-01", "2017-12-31"],
    }
    assert plan["calibration"]["weighting"] == (
        "equal_station_total_weight_within_training_complete_pairs"
    )
    assert plan["calibration"]["selection_or_tuning_on_validation"] is False
    assert plan["legal_channels"] == list(LEGAL_CHANNELS)
    assert plan["legal_horizons"] == [1, 3, 7]
    assert plan["future_discharge"] == {
        "enabled": False,
        "bundled_with_atmospheric_forcing": False,
        "status": "SEPARATE_EXTENSION_NOT_AUTHORIZED",
    }
    assert plan["inputs"]["holdout_2021_2023_read"] is False
    assert plan["inputs"]["score_artifacts_read"] is False
    assert plan["execution_gate"]["expected_sha256"] is None
    assert plan["execution_gate"]["state"] == ("LOCKED_PENDING_REVIEWED_EXECUTION_SEAL_SHA256")

    assert RUNNER.EXPECTED_EXECUTION_SEAL_SHA256 is None
    with pytest.raises(RuntimeError, match="fail-closed"):
        RUNNER.main(["--execute"])


def test_runner_maps_legacy_development_aliases_to_canonical_station_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = [f"n{index:03d}" for index in range(120)]
    canonical = [f"{index + 1:08d}" for index in range(120)]
    panel = pd.DataFrame(
        {
            "DATE": pd.to_datetime(np.full(120, "2006-01-01")),
            "site_id": legacy,
            "WTEMP": np.ones(120, dtype=np.float64),
        }
    )
    registry = pd.DataFrame(
        {
            "site_no": canonical,
            "legacy_site_id": legacy,
        }
    )
    monkeypatch.setattr(RUNNER.pd, "read_parquet", lambda *_args, **_kwargs: panel.copy())
    monkeypatch.setattr(RUNNER.pd, "read_csv", lambda *_args, **_kwargs: registry.copy())
    mapped = RUNNER._load_calibration_panel()
    assert list(mapped["site_id"]) == canonical
    assert list(panel["site_id"]) == legacy
