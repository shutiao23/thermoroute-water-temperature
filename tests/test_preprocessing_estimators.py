from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute.checkpoint import neural_output_head_schema
from thermoroute.frozen_inference import (
    FrozenInferenceError,
    reconstruct_frozen_transforms,
)
from thermoroute.model_suite import serialise_preprocessing
from thermoroute.weighting import ROW_EQUAL_WEIGHTING, STATION_EQUAL_WEIGHTING


def _zero_climatology(stations: tuple[str, ...]) -> F.HarmonicClimatology:
    return F.HarmonicClimatology(
        coef={station: np.zeros(3) for station in stations},
        k=1,
        fit_stations=stations,
        pooled=False,
    )


def _pair_panel(
    slopes: dict[str, float], *, repeats: dict[str, int] | None = None
) -> pd.DataFrame:
    repeats = repeats or {}
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2006-01-01")
    x_values = np.linspace(1.0, 5.0, 40)
    for station_index, (station, slope) in enumerate(slopes.items()):
        pair_index = 0
        for _repeat in range(repeats.get(station, 1)):
            for x in x_values:
                start = base + pd.Timedelta(days=station_index * 1000 + 3 * pair_index)
                rows.extend([
                    {
                        "site_id": station,
                        "DATE": start,
                        "WTEMP": x,
                        "WTEMP_observed": True,
                    },
                    {
                        "site_id": station,
                        "DATE": start + pd.Timedelta(days=1),
                        "WTEMP": slope * x,
                        "WTEMP_observed": True,
                    },
                ])
                pair_index += 1
    return pd.DataFrame(rows)


def _fit_anchor(
    monkeypatch: pytest.MonkeyPatch,
    panel: pd.DataFrame,
    *,
    pooled: bool = False,
    fallback: float = 0.9,
) -> F.DampedPersistenceAnchor:
    stations = tuple(dict.fromkeys(panel["site_id"].astype(str)))
    monkeypatch.setattr(C, "STATIONS", stations)
    return F.DampedPersistenceAnchor.fit(
        panel,
        np.ones(len(panel), dtype=bool),
        _zero_climatology(stations),
        fit_stations=stations,
        pooled=pooled,
        fallback=fallback,
    )


def test_damped_anchor_fits_ar_slope_not_correlation_and_minimizes_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _pair_panel({"a": 0.4})
    anchor = _fit_anchor(monkeypatch, panel)
    assert anchor.phi["a"] == pytest.approx(0.4, abs=1e-12)

    x = panel.WTEMP.to_numpy(float)[0::2]
    y = panel.WTEMP.to_numpy(float)[1::2]
    correlation_phi = float(np.clip(np.corrcoef(x, y)[0, 1], 0.0, 0.999))
    fitted_sse = float(np.sum(np.square(y - anchor.phi["a"] * x)))
    correlation_sse = float(np.sum(np.square(y - correlation_phi * x)))
    assert fitted_sse < correlation_sse


def test_damped_anchor_requires_consecutive_observed_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _pair_panel({"a": 0.4})
    origin = pd.Timestamp(panel["DATE"].min())
    panel["DATE"] = origin + 2 * (pd.to_datetime(panel["DATE"]) - origin)
    anchor = _fit_anchor(monkeypatch, panel, fallback=0.73)
    assert anchor.phi["a"] == 0.73

    too_short = _pair_panel({"a": 0.4}).iloc[:20].copy()
    anchor = _fit_anchor(monkeypatch, too_short, fallback=0.73)
    assert anchor.phi["a"] == 0.73

    panel = _pair_panel({"a": 0.4})
    panel.loc[panel.index[1::2], "WTEMP_observed"] = False
    anchor = _fit_anchor(monkeypatch, panel, fallback=0.73)
    assert anchor.phi["a"] == 0.73


def test_damped_anchor_clips_negative_and_explosive_slopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _pair_panel({"negative": -0.4, "explosive": 1.5})
    anchor = _fit_anchor(monkeypatch, panel)
    assert anchor.phi["negative"] == F.DAMPED_LOWER_BOUND
    assert anchor.phi["explosive"] == F.DAMPED_UPPER_BOUND


def test_pooled_damped_anchor_is_station_balanced_and_duplication_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fit_anchor(
        monkeypatch, _pair_panel({"a": 0.2, "b": 0.8}), pooled=True
    )
    duplicated = _fit_anchor(
        monkeypatch,
        _pair_panel({"a": 0.2, "b": 0.8}, repeats={"a": 4}),
        pooled=True,
    )
    assert original.phi["a"] == pytest.approx(0.5, abs=1e-12)
    assert original.phi == pytest.approx(duplicated.phi, abs=1e-12)


def _harmonic_panel(*, repeat_a: int = 1, missing_b: bool = False) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray
]:
    dates = pd.date_range("2007-01-01", periods=80, freq="4D")
    doy = dates.dayofyear.to_numpy()
    design = np.concatenate(
        [np.ones((len(dates), 1)), F.doy_harmonics(doy, 1)], axis=1
    )
    beta_a = np.array([2.0, 1.5, -0.25])
    beta_b = np.array([10.0, -0.5, 0.75])
    frames = []
    for station, beta, repeats in (
        ("a", beta_a, repeat_a), ("b", beta_b, 1)
    ):
        frame = pd.DataFrame({
            "site_id": station,
            "DATE": dates,
            "WTEMP": design @ beta,
        })
        if station == "b" and missing_b:
            frame["WTEMP"] = np.nan
        frames.extend([frame.copy() for _ in range(repeats)])
    return pd.concat(frames, ignore_index=True), beta_a, beta_b


def test_pooled_harmonic_is_station_balanced_and_duplication_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", ("a", "b"))
    panel, beta_a, beta_b = _harmonic_panel()
    original = F.HarmonicClimatology.fit(
        panel, np.ones(len(panel), dtype=bool), k=1, pooled=True
    )
    duplicated_panel, _, _ = _harmonic_panel(repeat_a=7)
    duplicated = F.HarmonicClimatology.fit(
        duplicated_panel,
        np.ones(len(duplicated_panel), dtype=bool),
        k=1,
        pooled=True,
    )
    expected = (beta_a + beta_b) / 2.0
    assert np.allclose(original.coef["a"], expected, rtol=0.0, atol=1e-12)
    assert np.allclose(original.coef["a"], duplicated.coef["a"], rtol=0.0, atol=1e-12)


def test_pooled_harmonic_fails_closed_on_missing_fit_station(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", ("a", "b"))
    panel, _, _ = _harmonic_panel(missing_b=True)
    with pytest.raises(ValueError, match="no finite train target"):
        F.HarmonicClimatology.fit(
            panel, np.ones(len(panel), dtype=bool), k=1, pooled=True
        )


def test_nonpooled_harmonic_remains_per_station_ordinary_least_squares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", ("a", "b"))
    panel, beta_a, beta_b = _harmonic_panel(repeat_a=5)
    fitted = F.HarmonicClimatology.fit(
        panel, np.ones(len(panel), dtype=bool), k=1, pooled=False
    )
    assert np.allclose(fitted.coef["a"], beta_a, rtol=0.0, atol=1e-12)
    assert np.allclose(fitted.coef["b"], beta_b, rtol=0.0, atol=1e-12)


def _scaler_panel(*, repeat_a: int = 1, missing_b: bool = False) -> pd.DataFrame:
    a = pd.DataFrame({"site_id": "a", "WTEMP": [0.0, 2.0, 4.0]})
    b = pd.DataFrame({"site_id": "b", "WTEMP": [10.0, 14.0, 18.0]})
    if missing_b:
        b["WTEMP"] = np.nan
    return pd.concat([*[a.copy() for _ in range(repeat_a)], b], ignore_index=True)


def test_pooled_scaler_matches_station_macro_closed_form_and_duplication_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", ("a", "b"))
    panel = _scaler_panel()
    original = D.StandardScalerPerStation.fit(
        panel,
        np.ones(len(panel), dtype=bool),
        variables=("WTEMP",),
        pooled=True,
    )
    duplicated_panel = _scaler_panel(repeat_a=9)
    duplicated = D.StandardScalerPerStation.fit(
        duplicated_panel,
        np.ones(len(duplicated_panel), dtype=bool),
        variables=("WTEMP",),
        pooled=True,
    )
    a, b = np.array([0.0, 2.0, 4.0]), np.array([10.0, 14.0, 18.0])
    expected_mean = (a.mean() + b.mean()) / 2.0
    expected_variance = (
        np.mean(np.square(a - expected_mean))
        + np.mean(np.square(b - expected_mean))
    ) / 2.0
    key = ("a", "WTEMP")
    assert original.mean[key] == pytest.approx(expected_mean, abs=1e-12)
    assert original.std[key] == pytest.approx(np.sqrt(expected_variance), abs=1e-12)
    assert original.mean == pytest.approx(duplicated.mean, abs=1e-12)
    assert original.std == pytest.approx(duplicated.std, abs=1e-12)


def test_pooled_scaler_fails_closed_on_missing_fit_station(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", ("a", "b"))
    panel = _scaler_panel(missing_b=True)
    with pytest.raises(ValueError, match="no finite train WTEMP"):
        D.StandardScalerPerStation.fit(
            panel,
            np.ones(len(panel), dtype=bool),
            variables=("WTEMP",),
            pooled=True,
        )


def test_nonpooled_scaler_retains_station_sample_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "STATIONS", ("a", "b"))
    panel = _scaler_panel(repeat_a=5)
    fitted = D.StandardScalerPerStation.fit(
        panel,
        np.ones(len(panel), dtype=bool),
        variables=("WTEMP",),
        pooled=False,
    )
    assert fitted.mean[("a", "WTEMP")] == pytest.approx(2.0)
    assert fitted.std[("a", "WTEMP")] == pytest.approx(
        panel.loc[panel.site_id.eq("a"), "WTEMP"].std(ddof=1)
    )
    assert fitted.mean[("b", "WTEMP")] == pytest.approx(14.0)
    assert fitted.std[("b", "WTEMP")] == pytest.approx(4.0)


def test_serialised_pooled_transforms_replay_exactly_and_freeze_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_sites = ("a", "b")
    monkeypatch.setattr(C, "STATIONS", training_sites)
    panel, _, _ = _harmonic_panel()
    climatology = F.HarmonicClimatology.fit(
        panel, np.ones(len(panel), dtype=bool), k=1, pooled=True
    )
    scaler = D.StandardScalerPerStation.fit(
        panel,
        np.ones(len(panel), dtype=bool),
        variables=("WTEMP",),
        pooled=True,
    )
    damped = F.DampedPersistenceAnchor(
        phi={site: 0.4 for site in training_sites},
        fit_stations=training_sites,
        pooled=True,
    )
    wd = SimpleNamespace(
        var_names=("WTEMP",),
        phys_vars=(),
        X=np.zeros((1, C.CONTEXT_LENGTH, 1)),
        scaler=scaler,
        damped_anchor=damped,
    )
    seasonal = pd.Series({day: 6.0 for day in range(1, 367)})
    imputer = D.Imputer(
        medians={(site, "WTEMP"): seasonal.copy() for site in training_sites},
        global_median={(site, "WTEMP"): 6.0 for site in training_sites},
        fit_stations=training_sites,
        pooled=True,
    )
    preprocessing = serialise_preprocessing(wd, climatology, imputer)
    assert preprocessing["imputer"]["pool_weighting"] == ROW_EQUAL_WEIGHTING
    assert preprocessing["scaler"]["pool_weighting"] == STATION_EQUAL_WEIGHTING
    assert preprocessing["climatology"]["pool_weighting"] == STATION_EQUAL_WEIGHTING
    assert preprocessing["damped_anchor"]["pool_weighting"] == STATION_EQUAL_WEIGHTING

    metadata = {
        "feature_order": ["WTEMP"],
        "station_to_index": {site: index for index, site in enumerate(training_sites)},
        "output_head_schema": neural_output_head_schema(),
        "architecture": {
            "class": "thermoroute.thermoroute.ThermoRoute",
            "kwargs": {"station_agnostic": True},
        },
        "preprocessing": preprocessing,
    }
    replayed = reconstruct_frozen_transforms(
        metadata, ("new-a", "new-b"), external=True
    )
    assert replayed.scaler.mean[("new-a", "WTEMP")] == scaler.mean[("a", "WTEMP")]
    assert np.array_equal(replayed.climatology.coef["new-a"], climatology.coef["a"])
    assert replayed.damped_anchor.phi["new-b"] == 0.4
    assert replayed.damped_anchor.min_pairs == F.DAMPED_MIN_PAIRS

    preprocessing["scaler"]["method"] = "legacy_row_pooled_scaler"
    with pytest.raises(FrozenInferenceError, match="scaler method"):
        reconstruct_frozen_transforms(metadata, ("new-a",), external=True)
