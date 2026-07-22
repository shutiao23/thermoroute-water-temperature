from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import textwrap

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


_CRASH_CHECKPOINT_SCRIPT = textwrap.dedent(
    r"""
    import os
    from pathlib import Path
    import random
    import signal
    import sys

    import numpy as np
    import torch
    import thermoroute.checkpoint as checkpoint_module

    destination = Path(sys.argv[1])
    mode = sys.argv[2]
    random.seed(431)
    np.random.seed(431)
    torch.manual_seed(431)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    config = {"seed": 431, "transaction": "subprocess-sigkill"}

    def step(values, metric):
        optimizer.zero_grad()
        loss = model(torch.tensor([values], dtype=torch.float32)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step(metric)

    def save(epoch):
        checkpoint_module.save_training_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_epoch=epoch,
            best_metric=float(0.4 - epoch * 0.1),
            best_model_state=model.state_dict(),
            run_id="subprocess-run",
            resolved_config=config,
            extra={"completed_epoch": epoch},
        )

    step((0.25, -0.5), 0.4)
    if mode in {"stale", "complete1", "payload_temp", "sidecar_temp"}:
        save(0)
    if mode in {"missing", "stale"}:
        def kill_after_payload(*_args, **_kwargs):
            os.kill(os.getpid(), signal.SIGKILL)
        checkpoint_module.atomic_write_json = kill_after_payload
    if mode in {"payload_temp", "sidecar_temp"}:
        real_replace = os.replace
        def kill_before_selected_replace(source, destination):
            destination = str(destination)
            should_kill = (
                mode == "payload_temp" and destination.endswith(".pt")
            ) or (
                mode == "sidecar_temp" and destination.endswith(".meta.json")
            )
            if should_kill:
                os.kill(os.getpid(), signal.SIGKILL)
            return real_replace(source, destination)
        checkpoint_module.os.replace = kill_before_selected_replace
    if mode in {"missing", "complete0"}:
        save(0)
    else:
        random.random()
        np.random.rand()
        torch.rand(3)
        step((-0.75, 1.25), 0.3)
        save(1)
    """
)


def _run_checkpoint_subprocess(path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "-c", _CRASH_CHECKPOINT_SCRIPT, str(path), mode],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_state_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right)
        for key in left:
            _assert_state_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_tree_equal(left_item, right_item)
    else:
        assert left == right


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="requires POSIX SIGKILL")
@pytest.mark.parametrize(
    ("crash_mode", "reference_mode", "expected_epoch"),
    [
        ("missing", "complete0", 0),
        ("stale", "complete1", 1),
        ("payload_temp", "complete0", 0),
        ("sidecar_temp", "complete1", 1),
    ],
)
def test_checkpoint_recovers_exact_payload_to_sidecar_sigkill_windows(
    tmp_path: Path, crash_mode: str, reference_mode: str, expected_epoch: int,
) -> None:
    crashed = tmp_path / f"{crash_mode}.pt"
    reference = tmp_path / f"{reference_mode}.pt"
    killed = _run_checkpoint_subprocess(crashed, crash_mode)
    assert killed.returncode == -signal.SIGKILL, (killed.stdout, killed.stderr)
    completed = _run_checkpoint_subprocess(reference, reference_mode)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)

    crashed_sidecar = checkpoint_sidecar_path(crashed)
    orphan_temps = list(crashed.parent.glob(f".{crashed.name}.*.tmp")) + list(
        crashed.parent.glob(f".{crashed.name}.meta.json.*.tmp")
    )
    if crash_mode == "missing":
        assert not crashed_sidecar.exists()
    elif crash_mode in {"stale", "sidecar_temp"}:
        stale = json.loads(crashed_sidecar.read_text())
        assert stale["epoch"] == 0
        assert stale["checkpoint_sha256"] != hashlib.sha256(crashed.read_bytes()).hexdigest()
    else:
        current = json.loads(crashed_sidecar.read_text())
        assert current["epoch"] == 0
        assert current["checkpoint_sha256"] == hashlib.sha256(crashed.read_bytes()).hexdigest()
    if crash_mode in {"payload_temp", "sidecar_temp"}:
        assert orphan_temps
        model, optimizer, scheduler = _model_optimizer()
        with pytest.raises(ValueError, match="orphan temp"):
            load_training_checkpoint(
                crashed, model=model, optimizer=optimizer, scheduler=scheduler,
                expected_run_id="subprocess-run",
                expected_resolved_config={
                    "seed": 431, "transaction": "subprocess-sigkill",
                },
                recover_missing_sidecar=False,
            )

    def restore(path: Path, *, recover: bool):
        model, optimizer, scheduler = _model_optimizer()
        state = load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="subprocess-run",
            expected_resolved_config={
                "seed": 431, "transaction": "subprocess-sigkill",
            },
            recover_missing_sidecar=recover,
        )
        next_rng = (random.random(), float(np.random.rand()), torch.rand(4).clone())
        return state, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), next_rng

    recovered = restore(crashed, recover=True)
    expected = restore(reference, recover=False)
    assert recovered[0].epoch == expected_epoch == expected[0].epoch
    assert recovered[0].best_epoch == expected[0].best_epoch
    assert recovered[0].best_metric == expected[0].best_metric
    assert recovered[0].extra == expected[0].extra
    for left, right in zip(recovered[1:4], expected[1:4], strict=True):
        _assert_state_tree_equal(left, right)
    assert recovered[4][0:2] == expected[4][0:2]
    assert torch.equal(recovered[4][2], expected[4][2])

    repaired = json.loads(crashed_sidecar.read_text())
    assert repaired["epoch"] == expected_epoch
    assert repaired["checkpoint_bytes"] == crashed.stat().st_size
    assert repaired["checkpoint_sha256"] == hashlib.sha256(crashed.read_bytes()).hexdigest()
    assert not list(crashed.parent.glob(f".{crashed.name}.*.tmp"))
    assert not list(crashed.parent.glob(f".{crashed.name}.meta.json.*.tmp"))


def test_checkpoint_recovery_does_not_launder_run_or_config_sidecar_attacks(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "attacked.pt"
    _save_valid_checkpoint(checkpoint)
    sidecar = checkpoint_sidecar_path(checkpoint)
    original = json.loads(sidecar.read_text())

    for field, forged in (
        ("run_id", "another-run"),
        ("resolved_config_sha256", "0" * 64),
    ):
        attacked = {**original, field: forged}
        sidecar.write_text(json.dumps(attacked, sort_keys=True) + "\n")
        model, optimizer, scheduler = _model_optimizer()
        with pytest.raises(ValueError, match="run_id|resolved_config"):
            load_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_run_id="safe-run",
                expected_resolved_config={"seed": 19, "scheduler": True},
                recover_missing_sidecar=True,
            )
        assert json.loads(sidecar.read_text()) == attacked

    sidecar.unlink()
    model, optimizer, scheduler = _model_optimizer()
    with pytest.raises(ValueError, match="resolved_config"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 999, "scheduler": True},
            recover_missing_sidecar=True,
        )
    assert not sidecar.exists()


def test_checkpoint_recovery_rejects_linked_orphan_temp_attacks(tmp_path: Path) -> None:
    checkpoint = tmp_path / "linked.pt"
    _save_valid_checkpoint(checkpoint)
    target = tmp_path / "attacker-owned-content"
    target.write_bytes(b"not a checkpoint transaction")
    orphan = tmp_path / f".{checkpoint.name}.attack.tmp"
    orphan.symlink_to(target.name)
    model, optimizer, scheduler = _model_optimizer()
    with pytest.raises(ValueError, match="linked|non-regular|foreign"):
        load_training_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
            recover_missing_sidecar=True,
        )
    assert orphan.is_symlink()

    orphan.unlink()
    os.link(target, orphan)
    model, optimizer, scheduler = _model_optimizer()
    with pytest.raises(ValueError, match="linked|non-regular|foreign"):
        load_training_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
            expected_run_id="safe-run",
            expected_resolved_config={"seed": 19, "scheduler": True},
            recover_missing_sidecar=True,
        )
    assert orphan.exists() and target.stat().st_nlink == 2


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
