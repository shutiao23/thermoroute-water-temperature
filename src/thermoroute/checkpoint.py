"""Recoverable training checkpoints and self-describing inference bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .repro import atomic_write_json, canonical_json, sha256_file


CHECKPOINT_VERSION = "thermoroute.training-checkpoint.v3"
CHECKPOINT_METADATA_VERSION = "thermoroute.training-checkpoint-metadata.v2"
BUNDLE_VERSION = "thermoroute.inference-bundle.v2"
NEURAL_OUTPUT_HEAD_SCHEMA_FORMAT = "thermoroute.neural-output-heads.v3"


def neural_output_head_schema() -> dict[str, Any]:
    """Return the exact statistical meaning of every frozen neural head.

    This is deliberately metadata, not an inference-time guess from tensor
    names.  Legacy bundles made ``q50`` an alias of the MSE point forecast; v2
    artifacts must prove that the two heads are distinct before weights load.
    """
    return {
        "format": NEURAL_OUTPUT_HEAD_SCHEMA_FORMAT,
        "point": {
            "field": "point",
            "objective": "mse_conditional_mean",
            "is_quantile": False,
        },
        "quantiles": {"q05": 0.05, "q50": 0.50, "q95": 0.95},
        "quantile_objective": "pinball",
        "quantile_ordering": "q05<=q50<=q95_by_construction",
        "crossing_loss_role": "identically_zero_serialization_compatibility_only",
        "point_relationship": "independent_not_sorted_with_quantiles",
    }


def _validate_neural_output_head_schema(value: object) -> None:
    if value != neural_output_head_schema():
        raise ValueError(
            "inference bundle lacks the independent point/q50 output-head schema"
        )


_CHECKPOINT_FIELDS = {
    "format",
    "run_id",
    "resolved_config_json",
    "resolved_config_sha256",
    "extra_json",
    "extra_sha256",
    "epoch",
    "best_epoch",
    "best_metric",
    "model_class",
    "optimizer_class",
    "scheduler_class",
    "model_state",
    "best_model_state",
    "optimizer_state",
    "scheduler_present",
    "scheduler_state",
    "rng_state",
}
_CHECKPOINT_METADATA_FIELDS = {
    "format",
    "checkpoint_format",
    "run_id",
    "epoch",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "resolved_config_sha256",
    "extra_sha256",
    "model_class",
    "optimizer_class",
    "scheduler_class",
    "scheduler_present",
}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def _type_identifier(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def checkpoint_sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_tree(value: Any, *, label: str, key: str | int | None = None) -> Any:
    """Copy a state tree into the primitives accepted by weights-only loading."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().clone()
        return tensor.contiguous() if tensor.layout == torch.strided else tensor
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value) and key != "mode_worse":
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _safe_tree(value.item(), label=label, key=key)
    if isinstance(value, (list, tuple)):
        return [_safe_tree(item, label=label) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str | int, Any] = {}
        for raw_key, item in value.items():
            if type(raw_key) not in {str, int}:
                raise TypeError(
                    f"{label} contains an unsupported mapping key: "
                    f"{type(raw_key).__name__}"
                )
            copied[raw_key] = _safe_tree(item, label=label, key=raw_key)
        return copied
    raise TypeError(f"{label} contains unsupported type {type(value).__name__}")


def _assert_safe_tree(value: Any, *, label: str,
                      key: str | int | None = None) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{label} tensors must deserialize on CPU")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{label} contains a non-finite tensor")
        return
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value) and key != "mode_worse":
            raise ValueError(f"{label} contains a non-finite float")
        return
    if type(value) is list:
        for item in value:
            _assert_safe_tree(item, label=label)
        return
    if type(value) is dict:
        for raw_key, item in value.items():
            if type(raw_key) not in {str, int}:
                raise TypeError(
                    f"{label} contains an unsupported mapping key: "
                    f"{type(raw_key).__name__}"
                )
            _assert_safe_tree(item, label=label, key=raw_key)
        return
    raise TypeError(f"{label} contains unsupported type {type(value).__name__}")


def _canonical_mapping(value: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    encoded = canonical_json(dict(value))
    decoded = json.loads(encoded)
    if type(decoded) is not dict:
        raise TypeError(f"{label} must encode a JSON object")
    return encoded, _sha256_text(encoded)


def _decode_canonical_mapping(value: Any, digest: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not str or type(digest) is not str or not _HEX_SHA256.fullmatch(digest):
        raise ValueError(f"checkpoint {label} encoding is invalid")
    if _sha256_text(value) != digest:
        raise ValueError(f"checkpoint {label} digest mismatch")
    try:
        decoded = json.loads(value)
        if type(decoded) is not dict or canonical_json(decoded) != value:
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint {label} is not canonical JSON") from exc
    return decoded


def _assert_exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"checkpoint {label} must be an integer >= {minimum}")
    return value


def _capture_python_rng_state() -> dict[str, Any]:
    version, internal, gauss_next = random.getstate()
    return {
        "version": int(version),
        "state": torch.tensor(internal, dtype=torch.int64),
        "gauss_next": None if gauss_next is None else float(gauss_next),
    }


def _capture_numpy_rng_state() -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "algorithm": str(algorithm),
        "keys": torch.as_tensor(np.asarray(keys, dtype=np.int64).copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": _capture_python_rng_state(),
        "numpy": _capture_numpy_rng_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state().cpu().clone()
    return state


def _validate_rng_state(state: Any) -> None:
    if type(state) is not dict:
        raise TypeError("checkpoint rng_state must be a dictionary")
    required = {"python", "numpy", "torch_cpu"}
    allowed = required | {"torch_cuda", "torch_mps"}
    if not required.issubset(state) or not set(state).issubset(allowed):
        raise ValueError("checkpoint rng_state fields are invalid")

    python_state = state["python"]
    if type(python_state) is not dict or set(python_state) != {
        "version", "state", "gauss_next"
    }:
        raise ValueError("checkpoint Python RNG state fields are invalid")
    current_python = random.getstate()
    if type(python_state["version"]) is not int or python_state["version"] != current_python[0]:
        raise ValueError("checkpoint Python RNG version is incompatible")
    python_tensor = python_state["state"]
    if (
        not isinstance(python_tensor, torch.Tensor)
        or python_tensor.device.type != "cpu"
        or python_tensor.dtype != torch.int64
        or python_tensor.ndim != 1
        or python_tensor.numel() != len(current_python[1])
    ):
        raise ValueError("checkpoint Python RNG state tensor is invalid")
    python_values = python_tensor.tolist()
    if any(type(item) is not int or item < 0 or item > 0xFFFFFFFF for item in python_values[:-1]):
        raise ValueError("checkpoint Python RNG state values are invalid")
    if not 0 <= python_values[-1] <= len(python_values) - 1:
        raise ValueError("checkpoint Python RNG index is invalid")
    gauss_next = python_state["gauss_next"]
    if gauss_next is not None and (type(gauss_next) is not float or not math.isfinite(gauss_next)):
        raise ValueError("checkpoint Python Gaussian cache is invalid")

    numpy_state = state["numpy"]
    if type(numpy_state) is not dict or set(numpy_state) != {
        "algorithm", "keys", "position", "has_gauss", "cached_gaussian"
    }:
        raise ValueError("checkpoint NumPy RNG state fields are invalid")
    keys = numpy_state["keys"]
    if (
        numpy_state["algorithm"] != "MT19937"
        or not isinstance(keys, torch.Tensor)
        or keys.device.type != "cpu"
        or keys.dtype != torch.int64
        or tuple(keys.shape) != (624,)
        or bool(((keys < 0) | (keys > 0xFFFFFFFF)).any().item())
    ):
        raise ValueError("checkpoint NumPy RNG state tensor is invalid")
    _assert_exact_int(numpy_state["position"], label="NumPy RNG position")
    if numpy_state["position"] > 624:
        raise ValueError("checkpoint NumPy RNG position is invalid")
    if type(numpy_state["has_gauss"]) is not int or numpy_state["has_gauss"] not in {0, 1}:
        raise ValueError("checkpoint NumPy Gaussian-cache flag is invalid")
    if (
        type(numpy_state["cached_gaussian"]) is not float
        or not math.isfinite(numpy_state["cached_gaussian"])
    ):
        raise ValueError("checkpoint NumPy Gaussian cache is invalid")

    cpu_state = state["torch_cpu"]
    expected_cpu = torch.get_rng_state()
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.device.type != "cpu"
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() != expected_cpu.numel()
    ):
        raise ValueError("checkpoint Torch CPU RNG state is invalid")
    if "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if not torch.cuda.is_available() or type(cuda_states) is not list:
            raise ValueError("checkpoint CUDA RNG state is incompatible with this runtime")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count is incompatible")
        for value, current in zip(cuda_states, torch.cuda.get_rng_state_all()):
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.uint8
                or value.ndim != 1
                or value.numel() != current.numel()
            ):
                raise ValueError("checkpoint CUDA RNG state tensor is invalid")
    if "torch_mps" in state:
        if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
            raise ValueError("checkpoint MPS RNG state is incompatible with this runtime")
        mps_state = state["torch_mps"]
        current_mps = torch.mps.get_rng_state()
        if (
            not isinstance(mps_state, torch.Tensor)
            or mps_state.device.type != "cpu"
            or mps_state.dtype != torch.uint8
            or mps_state.ndim != 1
            or mps_state.numel() != current_mps.numel()
        ):
            raise ValueError("checkpoint MPS RNG state tensor is invalid")


def restore_rng_state(state: Mapping[str, Any]) -> None:
    _validate_rng_state(state)
    python_state = state["python"]
    random.setstate((
        python_state["version"],
        tuple(int(item) for item in python_state["state"].tolist()),
        python_state["gauss_next"],
    ))
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        numpy_state["keys"].numpy().astype(np.uint32, copy=True),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])


def _move_optimizer_state(optimizer: torch.optim.Optimizer,
                          device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_torch_save(value: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        torch.save(value, tmp_name)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    best_epoch: int
    best_metric: float
    best_model_state: Mapping[str, Any] | None
    resolved_config: Mapping[str, Any]
    extra: Mapping[str, Any]


def save_training_checkpoint(path: str | Path, *, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             scheduler: Any | None, epoch: int, best_epoch: int,
                             best_metric: float, best_model_state: Mapping[str, Any] | None,
                             run_id: str, resolved_config: Mapping[str, Any],
                             extra: Mapping[str, Any] | None = None) -> None:
    """Save a digest-bound, weights-only-safe checkpoint for the next epoch."""
    path = Path(path)
    if type(run_id) is not str or not run_id:
        raise ValueError("checkpoint run_id must be a non-empty string")
    _assert_exact_int(epoch, label="epoch")
    _assert_exact_int(best_epoch, label="best_epoch")
    if best_epoch > epoch:
        raise ValueError("checkpoint best_epoch cannot exceed epoch")
    if type(best_metric) is not float or not math.isfinite(best_metric):
        raise ValueError("checkpoint best_metric must be a finite float")
    if best_model_state is None:
        raise ValueError("checkpoint requires a finite best model state")

    config_json, config_sha256 = _canonical_mapping(
        resolved_config, label="resolved_config"
    )
    extra_json, extra_sha256 = _canonical_mapping(extra or {}, label="extra")
    scheduler_present = scheduler is not None
    model_class = _type_identifier(model)
    optimizer_class = _type_identifier(optimizer)
    scheduler_class = _type_identifier(scheduler) if scheduler_present else None
    payload = {
        "format": CHECKPOINT_VERSION,
        "run_id": run_id,
        "resolved_config_json": config_json,
        "resolved_config_sha256": config_sha256,
        "extra_json": extra_json,
        "extra_sha256": extra_sha256,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "model_class": model_class,
        "optimizer_class": optimizer_class,
        "scheduler_class": scheduler_class,
        "model_state": _safe_tree(model.state_dict(), label="model_state"),
        "best_model_state": _safe_tree(best_model_state, label="best_model_state"),
        "optimizer_state": _safe_tree(optimizer.state_dict(), label="optimizer_state"),
        "scheduler_present": scheduler_present,
        "scheduler_state": (
            _safe_tree(scheduler.state_dict(), label="scheduler_state")
            if scheduler is not None else None
        ),
        "rng_state": capture_rng_state(),
    }
    _validate_checkpoint_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_run_id=run_id,
        expected_config_json=config_json,
    )
    _atomic_torch_save(payload, path)
    metadata = {
        "format": CHECKPOINT_METADATA_VERSION,
        "checkpoint_format": CHECKPOINT_VERSION,
        "run_id": run_id,
        "epoch": epoch,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256_file(path),
        "resolved_config_sha256": config_sha256,
        "extra_sha256": extra_sha256,
        "model_class": model_class,
        "optimizer_class": optimizer_class,
        "scheduler_class": scheduler_class,
        "scheduler_present": scheduler_present,
    }
    atomic_write_json(checkpoint_sidecar_path(path), metadata)


def _assert_state_dict(payload: Any, current: Mapping[str, Any], *, label: str) -> None:
    if type(payload) is not dict or any(type(key) is not str for key in payload):
        raise TypeError(f"checkpoint {label} must be a string-keyed dictionary")
    if set(payload) != set(current):
        raise ValueError(f"checkpoint {label} keys do not match the current model")
    for name, expected in current.items():
        value = payload[name]
        if not isinstance(expected, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint {label}[{name!r}] must be a tensor")
        if value.device.type != "cpu":
            raise ValueError(f"checkpoint {label}[{name!r}] must be on CPU")
        if value.dtype != expected.dtype or tuple(value.shape) != tuple(expected.shape):
            raise ValueError(f"checkpoint {label}[{name!r}] tensor schema mismatch")
        _assert_safe_tree(value, label=f"{label}[{name!r}]")


def _assert_optimizer_state(payload: Any, optimizer: torch.optim.Optimizer) -> None:
    if type(payload) is not dict or set(payload) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer_state fields are invalid")
    state = payload["state"]
    groups = payload["param_groups"]
    current = optimizer.state_dict()
    if type(state) is not dict or type(groups) is not list:
        raise TypeError("checkpoint optimizer_state containers are invalid")
    if len(groups) != len(current["param_groups"]):
        raise ValueError("checkpoint optimizer parameter-group count mismatch")
    known_parameter_ids: set[int] = set()
    for saved_group, current_group in zip(groups, current["param_groups"]):
        if type(saved_group) is not dict or set(saved_group) != set(current_group):
            raise ValueError("checkpoint optimizer parameter-group schema mismatch")
        if type(saved_group.get("params")) is not list or saved_group["params"] != current_group["params"]:
            raise ValueError("checkpoint optimizer parameter registry mismatch")
        if any(type(item) is not int for item in saved_group["params"]):
            raise TypeError("checkpoint optimizer parameter identifiers must be integers")
        known_parameter_ids.update(saved_group["params"])
    if any(type(key) is not int or key not in known_parameter_ids for key in state):
        raise ValueError("checkpoint optimizer state references an unknown parameter")
    _assert_safe_tree(payload, label="optimizer_state")


def _assert_scheduler_schema(saved: Any, current: Any, *, path: str = "scheduler_state") -> None:
    if type(current) is dict:
        if type(saved) is not dict or set(saved) != set(current):
            raise ValueError(f"checkpoint {path} fields are invalid")
        for key in current:
            _assert_scheduler_schema(saved[key], current[key], path=f"{path}.{key}")
        return
    if type(current) in {list, tuple}:
        if type(saved) is not list or len(saved) != len(current):
            raise ValueError(f"checkpoint {path} list schema is invalid")
        for index, (saved_item, current_item) in enumerate(zip(saved, current)):
            _assert_scheduler_schema(saved_item, current_item, path=f"{path}[{index}]")
        return
    if isinstance(current, torch.Tensor):
        if (
            not isinstance(saved, torch.Tensor)
            or saved.dtype != current.dtype
            or tuple(saved.shape) != tuple(current.shape)
        ):
            raise ValueError(f"checkpoint {path} tensor schema is invalid")
        _assert_safe_tree(saved, label=path)
        return
    if current is None:
        if saved is not None:
            raise ValueError(f"checkpoint {path} must be null")
        return
    if type(current) is float:
        if type(saved) is not float:
            raise TypeError(f"checkpoint {path} must be a float")
        if path.endswith(".mode_worse"):
            if saved != current:
                raise ValueError("checkpoint scheduler mode_worse sentinel is invalid")
        elif not math.isfinite(saved):
            raise ValueError(f"checkpoint {path} must be finite")
        return
    if type(saved) is not type(current) or type(current) not in {bool, int, str}:
        raise TypeError(f"checkpoint {path} primitive schema is invalid")


def _validate_checkpoint_payload(
    payload: Any,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    expected_run_id: str,
    expected_config_json: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(payload) is not dict or set(payload) != _CHECKPOINT_FIELDS:
        raise ValueError("checkpoint payload fields are invalid")
    if payload["format"] != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {payload['format']!r}")
    if type(payload["run_id"]) is not str or payload["run_id"] != expected_run_id:
        raise ValueError(
            f"checkpoint run_id {payload['run_id']!r} != {expected_run_id!r}"
        )
    config = _decode_canonical_mapping(
        payload["resolved_config_json"],
        payload["resolved_config_sha256"],
        label="resolved_config",
    )
    if expected_config_json is not None and payload["resolved_config_json"] != expected_config_json:
        raise ValueError("checkpoint resolved_config does not match the requested run")
    extra = _decode_canonical_mapping(
        payload["extra_json"], payload["extra_sha256"], label="extra"
    )
    epoch = _assert_exact_int(payload["epoch"], label="epoch")
    best_epoch = _assert_exact_int(payload["best_epoch"], label="best_epoch")
    if best_epoch > epoch:
        raise ValueError("checkpoint best_epoch cannot exceed epoch")
    if type(payload["best_metric"]) is not float or not math.isfinite(payload["best_metric"]):
        raise ValueError("checkpoint best_metric must be a finite float")

    expected_model_class = _type_identifier(model)
    expected_optimizer_class = _type_identifier(optimizer)
    expected_scheduler_class = _type_identifier(scheduler) if scheduler is not None else None
    if payload["model_class"] != expected_model_class:
        raise ValueError("checkpoint model class does not match")
    if payload["optimizer_class"] != expected_optimizer_class:
        raise ValueError("checkpoint optimizer class does not match")
    if payload["scheduler_class"] != expected_scheduler_class:
        raise ValueError("checkpoint scheduler class does not match")
    if type(payload["scheduler_present"]) is not bool:
        raise TypeError("checkpoint scheduler_present must be a boolean")
    if payload["scheduler_present"] != (scheduler is not None):
        raise ValueError("checkpoint scheduler presence does not match")
    if payload["scheduler_present"] != (payload["scheduler_state"] is not None):
        raise ValueError("checkpoint scheduler state presence is inconsistent")

    current_model_state = model.state_dict()
    _assert_state_dict(payload["model_state"], current_model_state, label="model_state")
    _assert_state_dict(
        payload["best_model_state"], current_model_state, label="best_model_state"
    )
    _assert_optimizer_state(payload["optimizer_state"], optimizer)
    if scheduler is not None:
        _assert_scheduler_schema(payload["scheduler_state"], scheduler.state_dict())
    _validate_rng_state(payload["rng_state"])
    return config, extra


def _load_checkpoint_metadata(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    expected_run_id: str,
    expected_config_sha256: str | None,
) -> dict[str, Any]:
    sidecar = checkpoint_sidecar_path(path)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint sidecar is missing or invalid") from exc
    if type(metadata) is not dict or set(metadata) != _CHECKPOINT_METADATA_FIELDS:
        raise ValueError("checkpoint sidecar fields are invalid")
    if metadata["format"] != CHECKPOINT_METADATA_VERSION:
        raise ValueError("unsupported checkpoint sidecar format")
    if metadata["checkpoint_format"] != CHECKPOINT_VERSION:
        raise ValueError("checkpoint sidecar declares an unsupported payload format")
    if type(metadata["run_id"]) is not str or metadata["run_id"] != expected_run_id:
        raise ValueError("checkpoint sidecar run_id does not match")
    _assert_exact_int(metadata["epoch"], label="sidecar epoch")
    if (
        type(metadata["checkpoint_bytes"]) is not int
        or metadata["checkpoint_bytes"] <= 0
        or type(metadata["checkpoint_sha256"]) is not str
        or not _HEX_SHA256.fullmatch(metadata["checkpoint_sha256"])
    ):
        raise ValueError("checkpoint sidecar content digest is invalid")
    for name in ("resolved_config_sha256", "extra_sha256"):
        if type(metadata[name]) is not str or not _HEX_SHA256.fullmatch(metadata[name]):
            raise ValueError(f"checkpoint sidecar {name} is invalid")
    if (
        expected_config_sha256 is not None
        and metadata["resolved_config_sha256"] != expected_config_sha256
    ):
        raise ValueError("checkpoint sidecar resolved_config does not match")
    if type(metadata["scheduler_present"]) is not bool:
        raise TypeError("checkpoint sidecar scheduler_present must be a boolean")
    if metadata["scheduler_present"] != (scheduler is not None):
        raise ValueError("checkpoint sidecar scheduler presence does not match")
    expected_classes = {
        "model_class": _type_identifier(model),
        "optimizer_class": _type_identifier(optimizer),
        "scheduler_class": _type_identifier(scheduler) if scheduler is not None else None,
    }
    if any(metadata[name] != value for name, value in expected_classes.items()):
        raise ValueError("checkpoint sidecar training-component lineage does not match")
    return metadata


def _assert_single_regular_file(path: Path, *, label: str) -> None:
    """Reject symlinks, hard links, devices, directories, and absent files."""
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        status = path.stat()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise ValueError(f"{label} must be one unlinked regular file")


def _handle_checkpoint_transaction_temps(path: Path, *, recover: bool) -> None:
    """Reject or remove only this checkpoint transaction's orphan temp files.

    A complete temp is never promoted: canonical publication is exclusively the
    atomic rename in the original writer.  Recovery may delete an orphan only
    when it is a same-owner, single-link regular file opened no-follow from the
    exact parent directory.  This models an honest owner process crash; it is
    not a defence against the same UID deliberately replacing both evidence
    files with a forged, internally valid higher-epoch transaction.
    """
    parent = path.parent
    if not parent.exists():
        return
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("checkpoint parent directory is linked or non-directory")
    sidecar = checkpoint_sidecar_path(path)
    prefixes = (f".{path.name}.", f".{sidecar.name}.")
    try:
        with os.scandir(parent) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if entry.name.endswith(".tmp")
                and any(entry.name.startswith(prefix) for prefix in prefixes)
            )
    except OSError as exc:
        raise ValueError("checkpoint transaction directory cannot be inspected") from exc
    if not names:
        return
    if not recover:
        raise ValueError("checkpoint transaction has an unhandled orphan temp file")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as exc:
        raise ValueError("checkpoint transaction directory cannot be opened safely") from exc
    try:
        for name in names:
            try:
                status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError("checkpoint orphan temp cannot be inspected") from exc
            owner_matches = not hasattr(os, "getuid") or status.st_uid == os.getuid()
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or not owner_matches
            ):
                raise ValueError("checkpoint orphan temp is linked, non-regular, or foreign")
            file_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            try:
                file_fd = os.open(name, file_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ValueError("checkpoint orphan temp cannot be opened safely") from exc
            try:
                opened = os.fstat(file_fd)
                if (
                    opened.st_dev != status.st_dev
                    or opened.st_ino != status.st_ino
                    or opened.st_nlink != 1
                    or not stat.S_ISREG(opened.st_mode)
                ):
                    raise ValueError("checkpoint orphan temp changed during inspection")
            finally:
                os.close(file_fd)
            os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_checkpoint_payload(
    path: Path, *, map_location: str | torch.device,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[Any, int, str]:
    """Hash and weights-only-load the same no-follow file descriptor."""
    _assert_single_regular_file(path, label="checkpoint payload")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("checkpoint payload cannot be opened safely") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            status = os.fstat(handle.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise ValueError("checkpoint payload changed into a linked/non-regular file")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
            observed_sha256 = digest.hexdigest()
            if expected_bytes is not None and status.st_size != expected_bytes:
                raise ValueError("checkpoint byte length does not match its sidecar")
            if expected_sha256 is not None and observed_sha256 != expected_sha256:
                raise ValueError("checkpoint content digest does not match its sidecar")
            handle.seek(0)
            try:
                payload = torch.load(
                    handle, map_location=map_location, weights_only=True,
                )
            except Exception as exc:
                raise ValueError("checkpoint cannot be safely deserialized") from exc
    except BaseException:
        # ``fdopen`` owns and closes the descriptor once entered.  It may fail
        # before ownership transfers only in exceptional interpreter states.
        raise
    return payload, int(status.st_size), observed_sha256


def _checkpoint_metadata_for_payload(
    payload: Mapping[str, Any], *, checkpoint_bytes: int, checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_METADATA_VERSION,
        "checkpoint_format": CHECKPOINT_VERSION,
        "run_id": str(payload["run_id"]),
        "epoch": int(payload["epoch"]),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": str(payload["resolved_config_sha256"]),
        "extra_sha256": str(payload["extra_sha256"]),
        "model_class": str(payload["model_class"]),
        "optimizer_class": str(payload["optimizer_class"]),
        "scheduler_class": payload["scheduler_class"],
        "scheduler_present": bool(payload["scheduler_present"]),
    }


def load_training_checkpoint(path: str | Path, *, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             scheduler: Any | None, expected_run_id: str,
                             expected_resolved_config: Mapping[str, Any],
                             map_location: str | torch.device = "cpu",
                             recover_missing_sidecar: bool = False) -> ResumeState:
    """Restore a validated checkpoint without allowing arbitrary pickle globals."""
    path = Path(path)
    if type(expected_run_id) is not str or not expected_run_id:
        raise ValueError("expected_run_id must be a non-empty string")
    expected_config_json, expected_config_sha256 = _canonical_mapping(
        expected_resolved_config, label="expected_resolved_config"
    )
    _handle_checkpoint_transaction_temps(
        path, recover=recover_missing_sidecar,
    )
    sidecar = checkpoint_sidecar_path(path)
    if sidecar.is_symlink():
        raise ValueError("checkpoint sidecar must not be a symbolic link")
    metadata: dict[str, Any] | None = None
    if sidecar.exists():
        _assert_single_regular_file(sidecar, label="checkpoint sidecar")
        metadata = _load_checkpoint_metadata(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_id=expected_run_id,
            expected_config_sha256=expected_config_sha256,
        )
    elif not recover_missing_sidecar:
        raise ValueError("checkpoint sidecar is missing or invalid")

    payload, payload_bytes, payload_sha256 = _read_checkpoint_payload(
        path,
        map_location=map_location,
        expected_bytes=(
            int(metadata["checkpoint_bytes"])
            if metadata is not None and not recover_missing_sidecar else None
        ),
        expected_sha256=(
            str(metadata["checkpoint_sha256"])
            if metadata is not None and not recover_missing_sidecar else None
        ),
    )
    config, extra = _validate_checkpoint_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_run_id=expected_run_id,
        expected_config_json=expected_config_json,
    )
    expected_metadata = _checkpoint_metadata_for_payload(
        payload,
        checkpoint_bytes=payload_bytes,
        checkpoint_sha256=payload_sha256,
    )
    if metadata is None:
        # The only missing-sidecar recovery state is a complete validated
        # payload.  A broken link is rejected above rather than treated absent.
        atomic_write_json(sidecar, expected_metadata)
        _assert_single_regular_file(sidecar, label="recovered checkpoint sidecar")
        metadata = _load_checkpoint_metadata(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            expected_run_id=expected_run_id,
            expected_config_sha256=expected_config_sha256,
        )
    elif metadata != expected_metadata:
        if not recover_missing_sidecar:
            if metadata["checkpoint_bytes"] != payload_bytes:
                raise ValueError("checkpoint byte length does not match its sidecar")
            if metadata["checkpoint_sha256"] != payload_sha256:
                raise ValueError("checkpoint content digest does not match its sidecar")
        # Repeated atomic checkpoint publication has one legitimate two-file
        # kill window: the new, higher-epoch payload replaced the old payload,
        # while the old, otherwise-valid sidecar remains.  Require both a
        # strictly older epoch and a different payload digest.  This does not
        # launder malformed sidecars, wrong run/config/component lineage, or a
        # same-epoch metadata edit.
        recoverable_stale_pair = (
            recover_missing_sidecar
            and metadata["epoch"] < expected_metadata["epoch"]
            and metadata["checkpoint_sha256"]
            != expected_metadata["checkpoint_sha256"]
        )
        if not recoverable_stale_pair:
            raise ValueError("checkpoint payload/sidecar transaction is inconsistent")
        atomic_write_json(sidecar, expected_metadata)
        _assert_single_regular_file(sidecar, label="recovered checkpoint sidecar")
        metadata = _load_checkpoint_metadata(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            expected_run_id=expected_run_id,
            expected_config_sha256=expected_config_sha256,
        )
    if metadata != expected_metadata:
        raise ValueError("checkpoint recovery sidecar does not match the payload")
    # Every field, tensor schema, lineage binding, and RNG state has been checked
    # above.  No mutable training object is touched before this point.
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    try:
        model_device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - optimizers cannot normally be empty
        model_device = torch.device("cpu")
    _move_optimizer_state(optimizer, model_device)
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"])
    return ResumeState(
        epoch=int(payload["epoch"]),
        best_epoch=int(payload["best_epoch"]),
        best_metric=float(payload["best_metric"]),
        best_model_state=payload["best_model_state"],
        resolved_config=config,
        extra=extra,
    )


REQUIRED_BUNDLE_METADATA = {
    "run_id",
    "architecture",
    "feature_order",
    "horizons",
    "station_to_index",
    "preprocessing",
    "event_thresholds",
    "event_calibrators",
    "conformal_offsets",
    "source_sha256",
    "panel_sha256",
    "registry_sha256",
    "runtime_sha256",
    "output_head_schema",
}


def save_inference_bundle(directory: str | Path, *,
                          members: Mapping[str, torch.nn.Module | Mapping[str, torch.Tensor]],
                          metadata: Mapping[str, Any],
                          expected_member_count: int | None = None) -> Path:
    """Save all ensemble members as weights-only tensors plus explicit metadata."""
    missing = REQUIRED_BUNDLE_METADATA - set(metadata)
    if missing:
        raise ValueError(f"bundle metadata missing: {sorted(missing)}")
    _validate_neural_output_head_schema(metadata.get("output_head_schema"))
    if not members:
        raise ValueError("inference bundle must contain at least one member")
    if expected_member_count is not None and len(members) != expected_member_count:
        raise ValueError(
            f"inference bundle has {len(members)} members; expected {expected_member_count}"
        )
    directory = Path(directory)
    weights: dict[str, dict[str, torch.Tensor]] = {}
    for name, member in members.items():
        state = member.state_dict() if isinstance(member, torch.nn.Module) else member
        weights[str(name)] = {
            str(k): value.detach().cpu().contiguous() for k, value in state.items()
        }
    if directory.exists():
        if not directory.is_dir():
            raise FileExistsError(f"inference bundle target is not a directory: {directory}")
        try:
            existing_weights, existing_metadata = load_inference_bundle(
                directory, expected_member_count=expected_member_count
            )
        except Exception as exc:
            raise FileExistsError(
                f"refusing to replace incomplete or invalid inference bundle: {directory}"
            ) from exc
        existing_declared = {
            key: existing_metadata.get(key) for key in metadata
        }
        if existing_declared != dict(metadata) or set(existing_weights) != set(weights):
            raise FileExistsError(
                f"refusing to replace non-identical inference bundle: {directory}"
            )
        for member_name, state in weights.items():
            old_state = existing_weights[member_name]
            if set(old_state) != set(state) or any(
                not torch.equal(old_state[name].cpu(), tensor)
                for name, tensor in state.items()
            ):
                raise FileExistsError(
                    f"refusing to replace non-identical inference bundle: {directory}"
                )
        return directory

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{directory.name}.", suffix=".staging", dir=directory.parent
    ))
    weights_path = staging / "weights.pt"
    _atomic_torch_save(weights, weights_path)
    bundle_metadata = dict(metadata)
    bundle_metadata.update({
        "format": BUNDLE_VERSION,
        "members": sorted(weights),
        "member_count": len(weights),
        "weights_sha256": sha256_file(weights_path),
    })
    atomic_write_json(staging / "metadata.json", bundle_metadata)
    try:
        # Validate the complete staged object before a single atomic directory
        # rename publishes it.  A retry may read an identical object, but can
        # never overwrite an earlier content address.
        load_inference_bundle(staging, expected_member_count=expected_member_count)
        os.rename(staging, directory)
        descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


def load_inference_bundle(directory: str | Path, *,
                          expected_member_count: int | None = None,
                          map_location: str | torch.device = "cpu"
                          ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Load a bundle without allowing arbitrary pickle globals."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format") != BUNDLE_VERSION:
        raise ValueError(f"unsupported bundle format: {metadata.get('format')!r}")
    missing = REQUIRED_BUNDLE_METADATA - set(metadata)
    if missing:
        raise ValueError(f"inference bundle metadata missing: {sorted(missing)}")
    _validate_neural_output_head_schema(metadata.get("output_head_schema"))
    weights_path = directory / "weights.pt"
    if sha256_file(weights_path) != metadata.get("weights_sha256"):
        raise ValueError("inference bundle weights checksum mismatch")
    weights = torch.load(weights_path, map_location=map_location, weights_only=True)
    if sorted(weights) != metadata.get("members"):
        raise ValueError("inference bundle member registry mismatch")
    if metadata.get("member_count") != len(weights):
        raise ValueError("inference bundle member count is inconsistent")
    if expected_member_count is not None and len(weights) != expected_member_count:
        raise ValueError(
            f"inference bundle has {len(weights)} members; expected {expected_member_count}"
        )
    return weights, metadata


def instantiate_inference_ensemble(
    directory: str | Path,
    *,
    model_factory: Any,
    expected_member_count: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    """Restore a weights-only ensemble and put every member in inference mode.

    ``model_factory`` is called as ``model_factory(member_name, metadata)``.  The
    architecture is therefore executable code supplied by the caller, while the
    bundle remains free of arbitrary pickled Python objects.
    """
    weights, metadata = load_inference_bundle(
        directory,
        expected_member_count=expected_member_count,
        map_location=device,
    )
    models: dict[str, torch.nn.Module] = {}
    for member_name in metadata["members"]:
        model = model_factory(member_name, metadata)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model_factory must return torch.nn.Module")
        model.load_state_dict(weights[member_name], strict=True)
        model.to(device)
        model.eval()
        models[member_name] = model
    return models, metadata
