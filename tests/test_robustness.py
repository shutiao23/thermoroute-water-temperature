"""Contracts for Route-A OOD/input-robustness evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute import config as C
from thermoroute.robustness import (
    PerturbationSpec,
    build_outcome_strata,
    enforce_common_robustness_keys,
    perturb_batch,
    predict_perturbation,
    route_a_perturbation_ladder,
    summarise_degradation,
)


class TinyWindows:
    def __init__(self):
        self.var_names = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN")
        self.phys_vars = ("TEMP", "RHMEAN")
        self.horizons = (1, 3)
        self.station = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        n, length, n_vars = 8, 4, len(self.var_names)
        self.X = np.zeros((n, length, n_vars), np.float32)
        # Give every cell a deterministic nonzero value.  FLOW is a standardised
        # log1p value and WTEMP tail reconstructs to 10 C below.
        self.X += np.linspace(-0.3, 0.4, n)[:, None, None]
        self.Mask = np.ones_like(self.X)
        self.issue_date = np.asarray(
            pd.date_range("2020-07-01", periods=n), dtype="datetime64[ns]")
        self.target_date = self.issue_date[:, None] + (
            np.asarray(self.horizons)[None, :] * np.timedelta64(1, "D"))
        self.y = np.column_stack([np.linspace(9, 16, n), np.linspace(10, 17, n)]).astype(
            np.float32)
        self.split = np.full(n, "test", dtype=object)
        self.scaler = SimpleNamespace(mean={}, std={})
        for station in C.STATIONS:
            for variable in self.var_names:
                self.scaler.mean[(station, variable)] = (
                    np.log1p(10.0) if variable == "FLOW" else
                    10.0 if variable == "WTEMP" else 0.0)
                self.scaler.std[(station, variable)] = 2.0 if variable == "WTEMP" else 1.0
        self.damped_anchor = SimpleNamespace(phi={station: 0.8 for station in C.STATIONS})

        wt = self.X[:, -1, 0] * 2.0 + 10.0
        self.wtemp_t = wt.astype(np.float32)
        self.clim_t = np.full(n, 9.0, np.float32)
        self.clim_tgt = np.full((n, 2), 9.5, np.float32)
        self.damped_prior = (self.clim_tgt + np.asarray([0.8, 0.8**3])[None, :]
                             * (self.wtemp_t - self.clim_t)[:, None]).astype(np.float32)
        self.phys_std = self.X[:, -1][:, [2, 4]].copy()
        self.logflowz = self.X[:, -1, 1].copy()
        self.wlevelz = np.zeros(n, np.float32)
        self.season = np.tile(np.array([[1.0, 0.0]], np.float32), (n, 1))
        self.gate = np.column_stack([
            self.season,
            self.X[:, -1, 2], self.X[:, -1, 1], self.X[:, -1, 3],
            self.X[:, -1, 0] - self.X[:, -2, 0],
        ]).astype(np.float32)

    def batch(self, index, device="cpu"):
        selected = np.asarray(index, dtype=int)
        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(x[selected], dtype=dtype, device=device)
        return {
            "X": tensor(self.X), "Mask": tensor(self.Mask),
            "wtemp_t": tensor(self.wtemp_t), "clim_t": tensor(self.clim_t),
            "clim_tgt": tensor(self.clim_tgt),
            "damped_prior": tensor(self.damped_prior),
            "phys_std": tensor(self.phys_std), "logflowz": tensor(self.logflowz),
            "wlevelz": tensor(self.wlevelz), "season": tensor(self.season),
            "gate": tensor(self.gate),
            "station": tensor(self.station, torch.long), "y": tensor(self.y),
        }


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(C, "STATIONS", ("001", "002", "003", "004"))
    return TinyWindows()


def test_route_a_ladder_has_declared_severity_units():
    ladder = route_a_perturbation_ladder()
    assert ladder[0] == PerturbationSpec("clean", 0, "none")
    assert {s.scenario for s in ladder} == {
        "clean", "missing_rate", "missing_block", "sensor_noise",
        "air_temperature_shift", "flow_shift",
    }
    assert len({s.condition_id for s in ladder}) == len(ladder)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PerturbationSpec("missing_rate", 1.1, "fraction")


def test_missingness_changes_only_issue_or_history_forcings(windows):
    index = np.array([0, 1])
    clean = windows.batch(index)
    stressed = perturb_batch(
        clean, windows, index, PerturbationSpec("missing_rate", 1.0, "fraction"))

    # The mandatory issue-time WTEMP anchor is not an optional forcing dropout.
    assert torch.equal(stressed["X"][:, :, 0], clean["X"][:, :, 0])
    assert torch.equal(stressed["wtemp_t"], clean["wtemp_t"])
    assert torch.equal(stressed["damped_prior"], clean["damped_prior"])
    assert torch.count_nonzero(stressed["X"][:, :, 1:]) == 0
    assert torch.count_nonzero(stressed["Mask"][:, :, 1:]) == 0
    assert torch.count_nonzero(stressed["phys_std"]) == 0
    assert torch.count_nonzero(stressed["logflowz"]) == 0
    assert torch.equal(stressed["y"], clean["y"])
    # Source tensors are never modified through torch.as_tensor views.
    assert torch.count_nonzero(clean["X"][:, :, 1:]) > 0


def test_recent_missing_block_is_contiguous_and_bounded_by_context(windows):
    index = np.array([0, 2])
    clean = windows.batch(index)
    spec = PerturbationSpec("missing_block", 2, "days", variables=("TEMP",))
    stressed = perturb_batch(clean, windows, index, spec)
    assert torch.equal(stressed["X"][:, :2, 2], clean["X"][:, :2, 2])
    assert torch.count_nonzero(stressed["X"][:, 2:, 2]) == 0
    assert torch.count_nonzero(stressed["Mask"][:, 2:, 2]) == 0
    with pytest.raises(ValueError, match="context"):
        perturb_batch(clean, windows, index,
                      PerturbationSpec("missing_block", 5, "days"))


def test_random_noise_is_deterministic_and_batch_size_invariant(windows):
    spec = PerturbationSpec("sensor_noise", 0.5, "train_sd")
    index = np.array([0, 1, 2, 3])
    whole = perturb_batch(windows.batch(index), windows, index, spec, base_seed=19)
    halves = [
        perturb_batch(windows.batch(part), windows, part, spec, base_seed=19)
        for part in (index[:2], index[2:])
    ]
    for name in ("X", "Mask", "wtemp_t", "damped_prior", "phys_std", "gate", "y"):
        combined = torch.cat([part[name] for part in halves])
        assert torch.equal(whole[name], combined), name
    assert not torch.equal(whole["X"], windows.batch(index)["X"])
    assert torch.equal(whole["y"], windows.batch(index)["y"])
    # Rows 0 and 1 are consecutive issues at the same station.  Their three
    # overlapping calendar observations receive exactly the same sensor error.
    clean = windows.batch(index)
    increment = whole["X"] - clean["X"]
    assert torch.allclose(increment[0, 1:, :], increment[1, :-1, :], atol=1e-7)


def test_temperature_and_flow_shifts_use_declared_units_and_sync_side_paths(windows):
    index = np.array([0, 3])
    clean = windows.batch(index)
    warm = perturb_batch(
        clean, windows, index,
        PerturbationSpec("air_temperature_shift", 2.0, "train_sd"))
    assert torch.allclose(warm["X"][:, :, 2], clean["X"][:, :, 2] + 2.0)
    assert torch.equal(warm["phys_std"][:, 0], warm["X"][:, -1, 2])
    assert torch.equal(warm["gate"][:, 2], warm["X"][:, -1, 2])

    doubled = perturb_batch(
        clean, windows, index, PerturbationSpec("flow_shift", 2.0, "multiplier"))
    mean = np.log1p(10.0)
    original_raw = torch.expm1(clean["X"][:, :, 1] + mean)
    shifted_raw = torch.expm1(doubled["X"][:, :, 1] + mean)
    assert torch.allclose(shifted_raw, 2.0 * original_raw, atol=1e-5)
    assert torch.equal(doubled["logflowz"], doubled["X"][:, -1, 1])
    assert torch.equal(doubled["gate"][:, 3], doubled["X"][:, -1, 1])
    assert torch.equal(doubled["y"], clean["y"])


def test_noisy_issue_wtemp_updates_anchor_but_never_target(windows):
    index = np.array([0, 1])
    clean = windows.batch(index)
    noisy = perturb_batch(
        clean, windows, index,
        PerturbationSpec("sensor_noise", 0.5, "train_sd", variables=("WTEMP",)),
        base_seed=2)
    assert not torch.equal(noisy["wtemp_t"], clean["wtemp_t"])
    expected = (noisy["clim_tgt"] + torch.tensor([0.8, 0.8**3])[None, :]
                * (noisy["wtemp_t"] - noisy["clim_t"])[:, None])
    assert torch.allclose(noisy["damped_prior"], expected)
    assert torch.equal(noisy["y"], clean["y"])


class TinyModel(torch.nn.Module):
    def forward(self, batch):
        signal = 0.01 * batch["X"].sum(dim=(1, 2))
        median = batch["damped_prior"] + signal[:, None]
        return SimpleNamespace(point=median)


def test_prediction_keys_remain_common_across_inference_batch_sizes(windows):
    index = np.arange(8)
    spec = PerturbationSpec("sensor_noise", 0.25, "train_sd")
    one = predict_perturbation(TinyModel(), windows, index, spec, batch_size=1, base_seed=5)
    many = predict_perturbation(TinyModel(), windows, index, spec, batch_size=7, base_seed=5)
    pd.testing.assert_frame_equal(one, many)

    clean = predict_perturbation(
        TinyModel(), windows, index, PerturbationSpec("clean", 0, "none"))
    audit = enforce_common_robustness_keys(pd.concat([clean, one], ignore_index=True))
    assert audit.n_common == len(clean)
    assert audit.rows_per_condition == 16


def test_strata_and_cluster_bootstrap_report_positive_degradation(windows):
    index = np.arange(8)
    stations = C.STATIONS
    strata = build_outcome_strata(
        windows, index,
        heat_thresholds={s: 12.0 for s in stations},
        low_flow_thresholds={s: 8.0 for s in stations},
        high_flow_thresholds={s: 12.0 for s in stations},
    )
    assert set(strata.stratum) == {"all", "heat_event", "low_flow", "high_flow"}
    assert not strata.duplicated(["site_id", "horizon", "issue_date",
                                   "target_date", "stratum"]).any()

    clean = predict_perturbation(
        TinyModel(), windows, index, PerturbationSpec("clean", 0, "none"))
    stress = clean.copy()
    spec = PerturbationSpec("sensor_noise", 0.5, "train_sd")
    stress["condition_id"] = spec.condition_id
    stress["scenario"] = spec.scenario
    stress["severity"] = spec.severity
    stress["severity_unit"] = spec.severity_unit
    # Construct a controlled paired degradation without touching y_true.
    clean["y_pred"] = clean.y_true + 0.5
    stress["y_pred"] = stress.y_true + 1.5
    predictions = pd.concat([clean, stress], ignore_index=True)
    summary, station_effects = summarise_degradation(
        predictions, strata[strata.stratum == "all"],
        huc2_by_site={"001": "01", "002": "01", "003": "02", "004": "02"},
        n_boot=100, seed=4)
    assert set(summary.cluster_level) == {"station", "huc2"}
    assert np.allclose(summary.delta_rmse, 1.0)
    assert np.allclose(summary.ci_low, 1.0)
    assert np.allclose(summary.ci_high, 1.0)
    assert np.allclose(station_effects.delta_rmse, 1.0)


def test_flow_strata_exclude_imputed_issue_flow(windows):
    index = np.arange(len(windows.X))
    flow_position = windows.var_names.index("FLOW")
    windows.Mask[index[0], -1, flow_position] = 0.0
    station = C.STATIONS[int(windows.station[index[0]])]
    issue = pd.Timestamp(windows.issue_date[index[0]])
    strata = build_outcome_strata(
        windows,
        index,
        heat_thresholds={site: 11.0 for site in C.STATIONS},
        low_flow_thresholds={site: 1e9 for site in C.STATIONS},
        high_flow_thresholds={site: -1e9 for site in C.STATIONS},
    )
    flow_rows = strata[
        strata.stratum.isin(["low_flow", "high_flow"])
        & strata.site_id.eq(station)
        & strata.issue_date.eq(issue)
    ]
    assert flow_rows.empty


def test_common_key_audit_rejects_missing_or_changed_forecasts(windows):
    index = np.arange(8)
    clean = predict_perturbation(
        TinyModel(), windows, index, PerturbationSpec("clean", 0, "none"))
    stress = predict_perturbation(
        TinyModel(), windows, index,
        PerturbationSpec("air_temperature_shift", 1, "train_sd"))
    with pytest.raises(ValueError, match="exact forecast keys"):
        enforce_common_robustness_keys(pd.concat([clean, stress.iloc[:-1]], ignore_index=True))
    stress.loc[stress.index[0], "y_true"] += 0.1
    with pytest.raises(ValueError, match="disagrees on y_true"):
        enforce_common_robustness_keys(pd.concat([clean, stress], ignore_index=True))
