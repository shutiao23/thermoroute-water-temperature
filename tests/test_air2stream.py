from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute import air2stream


def _training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(4),
        np.ones(4),
        np.arange(4, dtype=float),
        np.arange(1, 5, dtype=float),
    )


def _successful_solution(params: np.ndarray, *, success: bool = True):
    return SimpleNamespace(
        x=np.asarray(params, dtype=float),
        success=success,
        status=1 if success else 0,
        message="converged" if success else "evaluation budget exhausted",
        nfev=7,
    )


def test_multistart_selects_lowest_recomputed_training_objective(monkeypatch):
    def fake_least_squares(fun, start, *, bounds, max_nfev, args):
        del fun, bounds, max_nfev, args
        return _successful_solution(start)

    monkeypatch.setattr(air2stream, "least_squares", fake_least_squares)
    fitted = air2stream.fit(
        *_training_arrays(),
        variant="a4",
        initial_starts=[(3.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
    )

    assert np.array_equal(fitted.params, np.array([1.0, 0.0, 0.0, 0.0]))
    assert fitted.training_objective == pytest.approx(0.0)
    assert fitted.selected_initial_params == (1.0, 0.0, 0.0, 0.0)
    assert len(fitted.diagnostics) == 2
    assert all(item.success for item in fitted.diagnostics)


def test_start_order_and_exact_objective_ties_have_deterministic_resolution(monkeypatch):
    def fake_least_squares(fun, start, *, bounds, max_nfev, args):
        del fun, bounds, max_nfev, args
        return _successful_solution(start)

    monkeypatch.setattr(air2stream, "least_squares", fake_least_squares)
    starts = [(2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
    forward = air2stream.fit(
        *_training_arrays(), variant="a4", initial_starts=starts
    )
    reverse = air2stream.fit(
        *_training_arrays(), variant="a4", initial_starts=list(reversed(starts))
    )

    # a1=0 and a1=2 have identical SSE around the true daily increment of one;
    # the lexicographically smaller fitted vector is the documented tie-break.
    assert np.array_equal(forward.params, np.array([0.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(reverse.params, forward.params)
    assert forward.selected_initial_params == reverse.selected_initial_params
    assert forward.diagnostics == reverse.diagnostics


def test_real_bounded_optimizer_is_reproducible_when_starts_are_reordered():
    n_rows = 40
    air = np.linspace(0.0, 10.0, n_rows)
    flow = np.ones(n_rows)
    water = np.empty(n_rows)
    water[0] = 4.0
    for index in range(n_rows - 1):
        water[index + 1] = (
            water[index] + 1.0 + 0.2 * air[index] - 0.3 * water[index]
        )
    doy = np.arange(1, n_rows + 1, dtype=float)
    starts = [(2.0, 0.3, 0.3, 0.5), (0.0, 0.1, 0.1, 0.1)]

    forward = air2stream.fit(
        air, flow, water, doy, variant="a4", initial_starts=starts, max_nfev=500
    )
    reverse = air2stream.fit(
        air,
        flow,
        water,
        doy,
        variant="a4",
        initial_starts=list(reversed(starts)),
        max_nfev=500,
    )

    assert np.array_equal(forward.params, reverse.params)
    assert forward.training_objective == reverse.training_objective
    assert forward.diagnostics == reverse.diagnostics


def test_failed_starts_are_reported_and_do_not_discard_a_converged_start(monkeypatch):
    def fake_least_squares(fun, start, *, bounds, max_nfev, args):
        del fun, bounds, max_nfev, args
        if start[0] < 0:
            raise RuntimeError("synthetic numerical failure")
        if start[0] == 0:
            return _successful_solution(start, success=False)
        return _successful_solution(start)

    monkeypatch.setattr(air2stream, "least_squares", fake_least_squares)
    fitted = air2stream.fit(
        *_training_arrays(),
        variant="a4",
        initial_starts=[
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
    )

    assert np.array_equal(fitted.params, np.array([1.0, 0.0, 0.0, 0.0]))
    assert [item.success for item in fitted.diagnostics] == [False, False, True]
    assert "RuntimeError" in fitted.diagnostics[0].message
    assert fitted.diagnostics[1].status == 0
    assert fitted.diagnostics[1].nfev == 7


def test_all_failed_starts_raise_with_complete_diagnostics(monkeypatch):
    def failed_least_squares(fun, start, *, bounds, max_nfev, args):
        del fun, bounds, max_nfev, args
        return _successful_solution(start, success=False)

    monkeypatch.setattr(air2stream, "least_squares", failed_least_squares)
    with pytest.raises(air2stream.Air2streamOptimizationError) as caught:
        air2stream.fit(
            *_training_arrays(),
            variant="a4",
            initial_starts=[(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )

    assert len(caught.value.diagnostics) == 2
    assert all(not item.success for item in caught.value.diagnostics)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ((np.array([0.0, np.inf, 0.0, 0.0]), 0), "infinite"),
        ((np.ones(3), 1), "identical lengths"),
        ((np.zeros(4), 1), "positive"),
        ((np.array([1.0, 1.5, 3.0, 4.0]), 3), "integer days"),
    ],
)
def test_fit_rejects_invalid_training_data(replacement, message):
    arrays = list(_training_arrays())
    value, index = replacement
    arrays[index] = value
    with pytest.raises(ValueError, match=message):
        air2stream.fit(*arrays, variant="a4")


def test_fit_rejects_invalid_variant_start_and_observation_contracts():
    arrays = _training_arrays()
    with pytest.raises(ValueError, match="variant"):
        air2stream.fit(*arrays, variant="official")
    with pytest.raises(ValueError, match="outside"):
        air2stream.fit(
            *arrays,
            variant="a4",
            initial_starts=[(51.0, 0.0, 0.0, 0.0)],
        )
    with pytest.raises(ValueError, match="observed target"):
        air2stream.fit(*arrays, variant="a4", obs=np.zeros(4, dtype=bool))


def test_missing_rows_are_not_compressed_into_false_one_day_transitions(monkeypatch):
    def fake_least_squares(fun, start, *, bounds, max_nfev, args):
        del fun, bounds, max_nfev, args
        return _successful_solution(start)

    monkeypatch.setattr(air2stream, "least_squares", fake_least_squares)
    fitted = air2stream.fit(
        np.zeros(5),
        np.ones(5),
        np.array([0.0, 1.0, np.nan, 100.0, 101.0]),
        np.arange(1, 6, dtype=float),
        variant="a4",
        initial_starts=[(1.0, 0.0, 0.0, 0.0)],
    )

    # Only the genuinely consecutive pairs 0->1 and 100->101 contribute.
    # Compressing away the NaN would fabricate a 1->100 daily transition.
    assert fitted.training_objective == pytest.approx(0.0)


def test_observation_mask_excludes_imputed_issue_state_and_target(monkeypatch):
    def fake_least_squares(fun, start, *, bounds, max_nfev, args):
        del fun, bounds, max_nfev, args
        return _successful_solution(start)

    monkeypatch.setattr(air2stream, "least_squares", fake_least_squares)
    fitted = air2stream.fit(
        np.zeros(5),
        np.ones(5),
        # The middle value is finite to mimic imputation; the observation mask
        # must prevent both transitions touching it from entering the loss.
        np.array([0.0, 1.0, 100.0, 3.0, 4.0]),
        np.arange(1, 6, dtype=float),
        variant="a4",
        obs=np.array([True, True, False, True, True]),
        initial_starts=[(1.0, 0.0, 0.0, 0.0)],
    )

    assert fitted.training_objective == pytest.approx(0.0)


def test_fit_rejects_data_without_a_consecutive_observed_pair():
    with pytest.raises(ValueError, match="consecutive observed target pair"):
        air2stream.fit(
            np.zeros(4),
            np.ones(4),
            np.array([0.0, np.nan, 2.0, np.nan]),
            np.arange(1, 5, dtype=float),
            variant="a4",
        )


def test_forecast_rejects_nonfinite_or_out_of_bounds_fit_parameters():
    malformed = air2stream.Air2streamFit(
        params=np.array([np.nan, 0.0, 0.0, 0.0]),
        Qbar=1.0,
        variant="a4",
    )
    with pytest.raises(ValueError, match="non-finite"):
        air2stream.forecast_horizon(malformed, 1.0, 1.0, 1, [1.0], [2])

    outside = air2stream.Air2streamFit(
        params=np.array([0.0, 3.0, 0.0, 0.0]),
        Qbar=1.0,
        variant="a4",
    )
    with pytest.raises(ValueError, match="outside"):
        air2stream.forecast_horizon(outside, 1.0, 1.0, 1, [1.0], [2])

    valid = air2stream.Air2streamFit(
        params=np.array([0.0, 0.3, 0.3, 0.5]),
        Qbar=1.0,
        variant="a4",
    )
    with pytest.raises(ValueError, match="issue-time discharge must be positive"):
        air2stream.forecast_horizon(valid, 1.0, -0.1, 1, [1.0], [2])


def test_panel_runner_never_relabels_imputed_physical_drivers_as_observed(
    monkeypatch,
):
    dates = pd.date_range("2006-01-01", periods=10, freq="D")
    panel = pd.DataFrame({
        "site_id": "s1",
        "DATE": dates,
        "TEMP": np.arange(10, dtype=float),
        "FLOW": np.ones(10, dtype=float),
        "WTEMP": np.arange(10, 20, dtype=float),
        "TEMP_observed": True,
        "FLOW_observed": True,
        "WTEMP_observed": True,
    })
    # These finite values mimic a fold-safe imputer's output.  Their flags are
    # the authoritative statement that no physical measurement was available.
    panel.loc[1, "FLOW"] = 999.0
    panel.loc[1, "FLOW_observed"] = False
    panel.loc[2, "TEMP"] = 999.0
    panel.loc[2, "TEMP_observed"] = False

    fitted_inputs = {}
    calls = []

    def fake_fit(Ta, Q, T, doy, **kwargs):
        fitted_inputs["Ta"] = np.asarray(Ta)
        fitted_inputs["Q"] = np.asarray(Q)
        return air2stream.Air2streamFit(
            params=np.array([0.0, 0.0, 0.0, 0.0]),
            Qbar=1.0,
            variant=kwargs["variant"],
        )

    def fake_forecast(fit_obj, T0, Q0, doy0, Ta_future, doy_future):
        del fit_obj, Q0, doy0, doy_future
        calls.append((float(T0), np.asarray(Ta_future, dtype=float)))
        return float(T0)

    monkeypatch.setattr(air2stream, "fit", fake_fit)
    monkeypatch.setattr(air2stream, "forecast_horizon", fake_forecast)
    climatology = SimpleNamespace(
        predict=lambda station, doy: np.zeros(len(doy), dtype=float)
    )
    result = air2stream.run_air2stream(
        panel,
        SimpleNamespace(train=np.ones(len(panel), dtype=bool)),
        climatology,
        stations=("s1",),
        variant="a4",
    )

    assert np.isnan(fitted_inputs["Q"][1])
    assert np.isnan(fitted_inputs["Ta"][2])
    # Missing issue-time FLOW suppresses the physical forecast altogether.
    assert dates[1] not in set(pd.to_datetime(result.issue_date))
    # Missing issue-time TEMP uses the declared train-fit climatology, not the
    # finite imputed value 999.0.
    missing_temperature_call = next(values for T0, values in calls if T0 == 12.0)
    assert missing_temperature_call[0] == 0.0


def test_panel_runner_carries_train_mask_through_date_sort(monkeypatch):
    dates = pd.date_range("2006-01-01", periods=10, freq="D")
    ordered = pd.DataFrame({
        "site_id": "s1",
        "DATE": dates,
        "TEMP": np.arange(10, dtype=float),
        "FLOW": np.ones(10, dtype=float),
        "WTEMP": np.arange(10, 20, dtype=float),
    })
    panel = ordered.iloc[::-1].reset_index(drop=True)
    train = panel.DATE.le(pd.Timestamp("2006-01-04")).to_numpy()
    captured = {}

    def fake_fit(Ta, Q, T, doy, **kwargs):
        del Ta, Q, doy
        captured["T"] = np.asarray(T)
        return air2stream.Air2streamFit(
            params=np.array([0.0, 0.0, 0.0, 0.0]),
            Qbar=1.0,
            variant=kwargs["variant"],
        )

    monkeypatch.setattr(air2stream, "fit", fake_fit)
    monkeypatch.setattr(
        air2stream,
        "forecast_horizon",
        lambda fit_obj, T0, Q0, doy0, Ta_future, doy_future: float(T0),
    )
    air2stream.run_air2stream(
        panel,
        SimpleNamespace(train=train),
        SimpleNamespace(predict=lambda station, doy: np.zeros(len(doy))),
        stations=("s1",),
        variant="a4",
    )

    assert np.array_equal(captured["T"], np.array([10.0, 11.0, 12.0, 13.0]))
