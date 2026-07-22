from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute import config as C
from thermoroute.thermoroute import ThermoRouteOutputs
from thermoroute.train import fit_model


class _TinyWindows:
    """Window fixture that records the largest device transfer requested."""

    horizons = (1, 3, 7)

    def __init__(self):
        rng = np.random.default_rng(4)
        n = 22
        self.X = rng.normal(size=(n, 3, 2)).astype(np.float32)
        self.y = np.stack([
            0.4 * self.X[:, -1, 0] + 0.1,
            0.3 * self.X[:, -1, 0] - 0.2,
            0.2 * self.X[:, -1, 1] + 0.3,
        ], axis=1).astype(np.float32)
        self.station = np.asarray([0, 1] * (n // 2), dtype=np.int64)
        self.split = np.asarray(
            ["train"] * 10 + ["val"] * 4 + ["calib"] * 4 + ["test"] * 4,
            dtype=object,
        )
        self.issue_date = np.arange(
            np.datetime64("2015-01-01"), np.datetime64("2015-01-23")
        ).astype("datetime64[ns]")
        self.target_date = self.issue_date[:, None] + np.asarray(self.horizons)[None, :] * np.timedelta64(1, "D")
        self.max_batch_requested = 0

    def idx(self, split):
        return np.flatnonzero(self.split == split)

    def batch(self, index, device="cpu"):
        index = np.asarray(index)
        self.max_batch_requested = max(self.max_batch_requested, len(index))
        return {
            "X": torch.as_tensor(self.X[index], dtype=torch.float32, device=device),
            "y": torch.as_tensor(self.y[index], dtype=torch.float32, device=device),
        }


class _TinyForecaster(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(6, 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(8, 3),
        )
        self.spread = torch.nn.Parameter(torch.ones(3))
        self.event = torch.nn.Linear(6, 3)

    def forward(self, batch):
        flat = batch["X"].flatten(1)
        point = self.net(batch["X"])
        spread = torch.nn.functional.softplus(self.spread)[None, :]
        zeros = point.new_zeros(point.shape[0])
        return ThermoRouteOutputs(
            point=point,
            q05=point - spread,
            q50=point,
            q95=point + spread,
            exceed_logit=self.event(flat),
            prior=point.new_zeros(point.shape),
            kappa=zeros,
            teq=zeros,
            lag_weights=point.new_zeros(point.shape[0], 3, 1, 1),
            pi=point.new_zeros(point.shape[0], 1),
        )


def test_interrupted_training_resume_is_bitwise_equivalent_and_batched(tmp_path):
    previous_stations = C.STATIONS
    C.STATIONS = ("s0", "s1")
    try:
        cfg = replace(
            C.TRAIN,
            batch_size=3,
            max_epochs=5,
            patience=20,
            dropout=0.0,
        )
        thresholds = {"s0": 0.0, "s1": 0.0}
        resolved = {"fixture": "resume-equivalence", "epochs": cfg.max_epochs}

        uninterrupted_windows = _TinyWindows()
        uninterrupted = fit_model(
            _TinyForecaster,
            uninterrupted_windows,
            thresholds,
            cfg=cfg,
            seed=13,
            eval_batch_size=2,
            model_name="fixture",
        )

        interrupted_windows = _TinyWindows()
        checkpoint = tmp_path / "epoch.pt"
        fit_model(
            _TinyForecaster,
            interrupted_windows,
            thresholds,
            cfg=cfg,
            seed=13,
            eval_batch_size=2,
            model_name="fixture",
            checkpoint_path=checkpoint,
            run_id="fixture-run",
            resolved_config=resolved,
            stop_after_epoch=1,
        )
        resumed = fit_model(
            _TinyForecaster,
            interrupted_windows,
            thresholds,
            cfg=cfg,
            seed=13,
            eval_batch_size=2,
            model_name="fixture",
            checkpoint_path=checkpoint,
            run_id="fixture-run",
            resolved_config=resolved,
        )

        assert uninterrupted_windows.max_batch_requested <= 3
        assert interrupted_windows.max_batch_requested <= 3
        assert uninterrupted.best_val == resumed.best_val
        for name, value in uninterrupted.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[name]), name
        left = uninterrupted.pred.sort_values(
            ["split", "site_id", "horizon", "issue_date"]
        ).reset_index(drop=True)
        right = resumed.pred.sort_values(
            ["split", "site_id", "horizon", "issue_date"]
        ).reset_index(drop=True)
        assert left.equals(right)
    finally:
        C.STATIONS = previous_stations


def test_export_splits_can_exclude_calibration_and_evaluation_rows():
    previous_stations = C.STATIONS
    C.STATIONS = ("s0", "s1")
    try:
        windows = _TinyWindows()
        result = fit_model(
            _TinyForecaster,
            windows,
            {"s0": 0.0, "s1": 0.0},
            cfg=replace(C.TRAIN, batch_size=4, max_epochs=1, patience=2),
            seed=5,
            eval_batch_size=2,
            export_splits=("val",),
        )
        assert set(result.pred["split"]) == {"val"}
    finally:
        C.STATIONS = previous_stations


def test_patience_complete_checkpoint_resumes_without_an_extra_training_epoch(tmp_path):
    previous_stations = C.STATIONS
    C.STATIONS = ("s0", "s1")
    try:
        cfg = replace(
            C.TRAIN,
            batch_size=3,
            lr=0.0,
            max_epochs=6,
            patience=1,
            dropout=0.0,
        )
        thresholds = {"s0": 0.0, "s1": 0.0}
        resolved = {"fixture": "patience-complete", "epochs": cfg.max_epochs}
        checkpoint = tmp_path / "patience.pt"

        completed = fit_model(
            _TinyForecaster,
            _TinyWindows(),
            thresholds,
            cfg=cfg,
            seed=23,
            eval_batch_size=2,
            model_name="fixture",
            checkpoint_path=checkpoint,
            run_id="patience-run",
            resolved_config=resolved,
        )
        checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        sidecar = checkpoint.with_name(checkpoint.name + ".meta.json")
        sidecar_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()

        resumed = fit_model(
            _TinyForecaster,
            _TinyWindows(),
            thresholds,
            cfg=cfg,
            seed=23,
            eval_batch_size=2,
            model_name="fixture",
            checkpoint_path=checkpoint,
            run_id="patience-run",
            resolved_config=resolved,
        )

        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest
        assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == sidecar_digest
        assert completed.best_val == resumed.best_val
        for name, value in completed.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[name]), name
        assert completed.pred.equals(resumed.pred)
    finally:
        C.STATIONS = previous_stations
