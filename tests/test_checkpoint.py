from __future__ import annotations

from pathlib import Path
import hashlib
import json
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.checkpoint import (
    checkpoint_sidecar_path,
    instantiate_inference_ensemble,
    load_inference_bundle,
    load_training_checkpoint,
    neural_output_head_schema,
    save_inference_bundle,
    save_training_checkpoint,
)


def _write_attack_marker(path: str) -> None:
    Path(path).write_text("unsafe checkpoint loader executed a pickle global")


class _MaliciousCheckpointValue:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _write_attack_marker, (str(self.marker),)


def _model_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return model, optimizer, scheduler


def test_checkpoint_restores_model_optimizer_and_rng(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer, scheduler = _model_optimizer()
    x = torch.tensor([[1.0, 2.0]])
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step(0.4)
    checkpoint = tmp_path / "epoch.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        best_epoch=2,
        best_metric=0.4,
        best_model_state=model.state_dict(),
        run_id="run-a",
        resolved_config={"seed": 7},
    )
    expected_random = (random.random(), np.random.rand(), torch.rand(1))

    restored_model, restored_optimizer, restored_scheduler = _model_optimizer()
    state = load_training_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_run_id="run-a",
        expected_resolved_config={"seed": 7},
    )
    actual_random = (random.random(), np.random.rand(), torch.rand(1))
    assert state.epoch == 3 and state.best_epoch == 2
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    assert torch.equal(actual_random[2], expected_random[2])
    for left, right in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(left, right)

    with pytest.raises(ValueError, match="run_id"):
        load_training_checkpoint(
            checkpoint,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            expected_run_id="different-run",
            expected_resolved_config={"seed": 7},
        )


def _save_valid_checkpoint(path: Path, *, with_scheduler: bool = True):
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    model, optimizer, scheduler = _model_optimizer()
    loss = model(torch.tensor([[0.25, -0.5]])).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step(0.3)
    selected_scheduler = scheduler if with_scheduler else None
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=selected_scheduler,
        epoch=1,
        best_epoch=0,
        best_metric=0.3,
        best_model_state=model.state_dict(),
        run_id="safe-run",
        resolved_config={"seed": 19, "scheduler": with_scheduler},
        extra={"bad_epochs": 1},
    )
    return model, optimizer, selected_scheduler


def _refresh_checkpoint_sidecar(path: Path, **updates) -> None:
    sidecar = checkpoint_sidecar_path(path)
    metadata = json.loads(sidecar.read_text())
    metadata.update({
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **updates,
    })
    sidecar.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")


def test_checkpoint_digest_rejects_truncation_and_same_length_tampering(tmp_path):
    checkpoint = tmp_path / "durable.pt"
    _save_valid_checkpoint(checkpoint)
    original = checkpoint.read_bytes()

    checkpoint.write_bytes(original[:-13])
    model, optimizer, scheduler = _model_optimizer()
    with pytest.raises(ValueError, match="byte length"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
        )

    checkpoint.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="content digest"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
        )


def test_legacy_checkpoint_schema_fails_before_state_load(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    _save_valid_checkpoint(checkpoint)
    sidecar = checkpoint_sidecar_path(checkpoint)
    metadata = json.loads(sidecar.read_text())
    metadata["checkpoint_format"] = "thermoroute.training-checkpoint.v2"
    sidecar.write_text(json.dumps(metadata))

    model, optimizer, scheduler = _model_optimizer()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="unsupported payload format"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
        )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())

def test_checkpoint_rejects_missing_or_extra_scheduler_before_state_load(tmp_path):
    with_scheduler = tmp_path / "with-scheduler.pt"
    _save_valid_checkpoint(with_scheduler, with_scheduler=True)
    model, optimizer, _ = _model_optimizer()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="scheduler"):
        load_training_checkpoint(
            with_scheduler,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
        )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())

    without_scheduler = tmp_path / "without-scheduler.pt"
    _save_valid_checkpoint(without_scheduler, with_scheduler=False)
    model, optimizer, scheduler = _model_optimizer()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="scheduler"):
        load_training_checkpoint(
            without_scheduler,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": False},
        )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_checkpoint_weights_only_loader_never_executes_malicious_pickle_global(tmp_path):
    checkpoint = tmp_path / "malicious.pt"
    _save_valid_checkpoint(checkpoint)
    marker = tmp_path / "executed.txt"
    torch.save({"evil": _MaliciousCheckpointValue(marker)}, checkpoint)
    _refresh_checkpoint_sidecar(checkpoint)

    model, optimizer, scheduler = _model_optimizer()
    with pytest.raises(ValueError, match="safely deserialized"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
        )
    assert not marker.exists()


def test_checkpoint_rejects_nonfinite_payload_before_mutating_model(tmp_path):
    checkpoint = tmp_path / "nonfinite.pt"
    _save_valid_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["best_metric"] = float("nan")
    torch.save(payload, checkpoint)
    _refresh_checkpoint_sidecar(checkpoint)

    model, optimizer, scheduler = _model_optimizer()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="best_metric"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
        )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_weights_only_inference_bundle_round_trip_and_checksum(tmp_path):
    torch.manual_seed(1)
    model, _, _ = _model_optimizer()
    metadata = {
        "run_id": "abc",
        "architecture": {"name": "fixture"},
        "feature_order": ["WTEMP", "FLOW"],
        "horizons": [1, 3, 7],
        "station_to_index": {"01234567": 0},
        "preprocessing": {"scaler": "embedded"},
        "event_thresholds": {"01234567": 20.0},
        "event_calibrators": {"1": {"intercept": 0.0, "slope": 1.0, "constant": None}},
        "conformal_offsets": {"01234567:1": 0.2},
        "source_sha256": "s",
        "panel_sha256": "p",
        "registry_sha256": "r",
        "runtime_sha256": "t",
        "output_head_schema": neural_output_head_schema(),
    }
    directory = save_inference_bundle(tmp_path / "bundle", members={"seed0": model},
                                      metadata=metadata)
    weights, loaded = load_inference_bundle(directory)
    assert loaded["members"] == ["seed0"]
    assert set(weights["seed0"]) == set(model.state_dict())

    weights_path = directory / "weights.pt"
    weights_path.write_bytes(weights_path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_inference_bundle(directory)


def test_five_member_bundle_reconstructs_models_with_prediction_parity(tmp_path):
    metadata = {
        "run_id": "route-a",
        "architecture": {"name": "linear-2x1"},
        "feature_order": ["WTEMP", "FLOW"],
        "horizons": [1, 3, 7],
        "station_to_index": {"01234567": 0},
        "preprocessing": {"scaler": "fixture"},
        "event_thresholds": {"01234567": 20.0},
        "event_calibrators": {"1": {"intercept": 0.0, "slope": 1.0, "constant": None}},
        "conformal_offsets": {},
        "source_sha256": "s",
        "panel_sha256": "p",
        "registry_sha256": "r",
        "runtime_sha256": "t",
        "output_head_schema": neural_output_head_schema(),
    }
    members = {}
    x = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
    expected = {}
    for seed in range(5):
        torch.manual_seed(seed)
        model = torch.nn.Linear(2, 1)
        model.eval()
        members[f"seed{seed}"] = model
        expected[f"seed{seed}"] = model(x).detach().clone()

    directory = save_inference_bundle(
        tmp_path / "ensemble",
        members=members,
        metadata=metadata,
        expected_member_count=5,
    )
    restored, loaded = instantiate_inference_ensemble(
        directory,
        model_factory=lambda _name, _metadata: torch.nn.Linear(2, 1),
        expected_member_count=5,
    )
    assert loaded["member_count"] == 5
    assert list(restored) == [f"seed{i}" for i in range(5)]
    for name, model in restored.items():
        assert torch.equal(model(x), expected[name])

    with pytest.raises(ValueError, match="expected 4"):
        load_inference_bundle(directory, expected_member_count=4)


def test_inference_bundle_is_create_only_and_idempotent(tmp_path):
    torch.manual_seed(4)
    model, _, _ = _model_optimizer()
    metadata = {
        "run_id": "immutable", "architecture": {"name": "fixture"},
        "feature_order": ["WTEMP", "FLOW"], "horizons": [1, 3, 7],
        "station_to_index": {"01234567": 0}, "preprocessing": {},
        "event_thresholds": {}, "event_calibrators": {},
        "conformal_offsets": {}, "source_sha256": "s",
        "panel_sha256": "p", "registry_sha256": "r",
        "runtime_sha256": "t",
        "output_head_schema": neural_output_head_schema(),
    }
    target = tmp_path / "immutable"
    save_inference_bundle(target, members={"seed0": model}, metadata=metadata)
    before = hashlib.sha256((target / "weights.pt").read_bytes()).hexdigest()
    save_inference_bundle(target, members={"seed0": model}, metadata=metadata)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(FileExistsError, match="non-identical"):
        save_inference_bundle(target, members={"seed0": model}, metadata=metadata)
    after = hashlib.sha256((target / "weights.pt").read_bytes()).hexdigest()
    assert after == before


def test_legacy_or_semantically_ambiguous_inference_bundle_fails_closed(tmp_path):
    model, _, _ = _model_optimizer()
    metadata = {
        "run_id": "schema-v2", "architecture": {"name": "fixture"},
        "feature_order": ["WTEMP"], "horizons": [1],
        "station_to_index": {"01234567": 0}, "preprocessing": {},
        "event_thresholds": {}, "event_calibrators": {},
        "conformal_offsets": {}, "source_sha256": "s",
        "panel_sha256": "p", "registry_sha256": "r", "runtime_sha256": "t",
        "output_head_schema": neural_output_head_schema(),
    }
    legacy = save_inference_bundle(
        tmp_path / "legacy", members={"seed0": model}, metadata=metadata
    )
    legacy_document = json.loads((legacy / "metadata.json").read_text())
    legacy_document["format"] = "thermoroute.inference-bundle.v1"
    (legacy / "metadata.json").write_text(json.dumps(legacy_document))
    with pytest.raises(ValueError, match="unsupported bundle format"):
        load_inference_bundle(legacy)

    ambiguous = save_inference_bundle(
        tmp_path / "ambiguous", members={"seed0": model}, metadata=metadata
    )
    ambiguous_document = json.loads((ambiguous / "metadata.json").read_text())
    ambiguous_document.pop("output_head_schema")
    (ambiguous / "metadata.json").write_text(json.dumps(ambiguous_document))
    with pytest.raises(ValueError, match="metadata missing"):
        load_inference_bundle(ambiguous)
