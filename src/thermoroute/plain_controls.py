"""Standalone plain neural controls for the conventional holdout scorer.

The two development-only plain controls (``PlainMLP-7var`` and
``PlainCausalTCN-7var``) were trained under Stage 09b as architecture
comparisons.  They publish *training checkpoints* (``seed{0..4}.pt``,
``thermoroute.training-checkpoint.v3``) but, unlike the deployed ThermoRoute /
LightGBM / LSTM models, they have **no inference bundle**.  The conventional
holdout scorer therefore loads their ``best_model_state`` directly and borrows
the frozen stage-09 preprocessing from a same-variable ThermoRoute bundle (see
``conventional_score.plain_control_ensemble`` and the G15 verification).

This module is deliberately STANDALONE: it imports only the bare model classes
(``neural_baselines``), the project config, the dataset schema, and the
checkpoint safe-load helpers.  It never imports the pre-registration apparatus
(``development_controls`` re-exports ``model_suite`` / ``predictor_bridge`` /
``input_closure`` transitively), so it survives the apparatus deletion.  The
architecture constructor kwargs below are copied verbatim from
``development_controls.build_arm_model`` / ``architecture_configuration``; they
are the frozen scientific contract, not a re-derivation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from . import checkpoint
from . import config as C
from .neural_baselines import (
    PlainCausalTCNForecaster,
    PlainMLPForecaster,
)
from .datasets import PHYS_FORCINGS


# --------------------------------------------------------------------------- #
# Frozen scientific contract (mirrors development_controls.py verbatim)
# --------------------------------------------------------------------------- #
# These constants are the Stage-09b frozen arm contract.  They are duplicated
# here rather than imported from ``development_controls`` because that module
# transitively imports the to-be-deleted apparatus; the values must never drift.
FULL_VARIABLES: tuple[str, ...] = (
    "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP",
)
MLP_HIDDEN_DIM: int = 70
TCN_CHANNELS: int = 54
INFORMATION_MATCHED_GATE_DIM: int = 6
CONTROL_SEEDS: tuple[int, ...] = C.USGS_SEEDS
DEVELOPMENT_SCOPE: str = "development_only_2006_2020"
PLAIN_FEATURE_SET: str = "all_7_variables"
DEFAULT_N_STATIONS: int = 120

# Expected trainable parameter counts (stage09b_precompute.py:91-95).
EXPECTED_PARAMETER_COUNTS: dict[str, int] = {
    "PlainMLP-7var": 38_860,
    "PlainCausalTCN-7var": 38_346,
}

PLAIN_CONTROL_ARMS: tuple[str, ...] = ("PlainMLP-7var", "PlainCausalTCN-7var")


class PlainControlError(RuntimeError):
    """A plain-control checkpoint could not be safely loaded or reconstructed."""


def _physics_count(variables: tuple[str, ...]) -> int:
    return sum(variable in PHYS_FORCINGS for variable in variables)


def build_plain_control_model(
    arm_id: str,
    *,
    seed: int,
    n_stations: int = DEFAULT_N_STATIONS,
) -> torch.nn.Module:
    """Reconstruct one plain control architecture from its frozen contract.

    The constructor kwargs are copied verbatim from
    ``development_controls.build_arm_model`` so the rebuilt architecture is
    byte-for-byte load-compatible with the stored ``best_model_state``.  Both
    controls are station-aware (``station_agnostic=False``) and use the
    Stage-09b information-matched context (``damped_prior`` anchor).
    """
    n_phys = _physics_count(FULL_VARIABLES)
    if arm_id == "PlainMLP-7var":
        return PlainMLPForecaster(
            n_vars=len(FULL_VARIABLES),
            context_length=C.CONTEXT_LENGTH,
            horizons=C.HORIZONS,
            n_stations=n_stations,
            station_agnostic=False,
            init_seed=int(seed),
            hidden_dim=MLP_HIDDEN_DIM,
            depth=2,
            dropout=C.TRAIN.dropout,
            use_information_matched_context=True,
            n_phys=n_phys,
            gate_dim=INFORMATION_MATCHED_GATE_DIM,
        )
    if arm_id == "PlainCausalTCN-7var":
        return PlainCausalTCNForecaster(
            n_vars=len(FULL_VARIABLES),
            context_length=C.CONTEXT_LENGTH,
            horizons=C.HORIZONS,
            n_stations=n_stations,
            station_agnostic=False,
            init_seed=int(seed),
            channels=TCN_CHANNELS,
            blocks=4,
            kernel_size=3,
            dropout=C.TRAIN.dropout,
            use_information_matched_context=True,
            n_phys=n_phys,
            gate_dim=INFORMATION_MATCHED_GATE_DIM,
        )
    raise PlainControlError(f"unknown plain control arm: {arm_id!r}")


def plain_control_parameter_count(
    arm_id: str, *, n_stations: int = DEFAULT_N_STATIONS,
) -> int:
    """Trainable parameter count for one arm, rebuilt from the frozen contract."""
    model = build_plain_control_model(arm_id, seed=CONTROL_SEEDS[0], n_stations=n_stations)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _validate_sidecar(sidecar: Mapping[str, Any], *, arm_id: str) -> dict[str, Any]:
    """Validate the training-checkpoint metadata sidecar (v2) fields."""
    required = {
        "format", "checkpoint_format", "run_id", "epoch",
        "checkpoint_bytes", "checkpoint_sha256",
        "resolved_config_sha256", "extra_sha256",
        "model_class", "optimizer_class", "scheduler_class", "scheduler_present",
    }
    if not isinstance(sidecar, Mapping) or set(sidecar) != required:
        raise PlainControlError(f"{arm_id}: checkpoint sidecar fields are invalid")
    if sidecar["format"] != checkpoint.CHECKPOINT_METADATA_VERSION:
        raise PlainControlError(f"{arm_id}: unsupported checkpoint sidecar format")
    if sidecar["checkpoint_format"] != checkpoint.CHECKPOINT_VERSION:
        raise PlainControlError(f"{arm_id}: unsupported checkpoint payload format")
    if not isinstance(sidecar["run_id"], str) or not sidecar["run_id"]:
        raise PlainControlError(f"{arm_id}: checkpoint run_id is invalid")
    if not isinstance(sidecar["model_class"], str):
        raise PlainControlError(f"{arm_id}: checkpoint model_class is invalid")
    return dict(sidecar)


def load_plain_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    arm_id: str,
    seed: int,
    n_stations: int = DEFAULT_N_STATIONS,
    expected_run_id: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Safely load a plain control ``best_model_state`` without the apparatus.

    This is the standalone equivalent of
    ``09b_development_controls.replay_best_model_state_prediction``'s checkpoint
    load: it reuses ``checkpoint._read_checkpoint_payload`` (no-follow open,
    byte/digest check against the sidecar, ``weights_only=True`` torch.load) so
    no arbitrary pickle globals are ever accepted, then validates the payload
    lineage and loads ``best_model_state`` strictly.  Unlike
    ``checkpoint.load_training_checkpoint`` it does **not** reconstruct or
    validate an optimiser/scheduler/RNG state, because the conventional scorer
    only ever runs inference from the best model state.

    Returns ``(model, metadata)`` where ``model`` has its ``best_model_state``
    loaded and is in eval mode, and ``metadata`` carries the self-describing
    architecture contract (constructor kwargs + trainable parameter count).
    """
    path = Path(checkpoint_path)
    sidecar_path = checkpoint.checkpoint_sidecar_path(path)
    if not path.is_file() or not sidecar_path.is_file():
        raise PlainControlError(f"{arm_id}/seed{seed}: checkpoint or sidecar is absent: {path}")
    try:
        sidecar_raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlainControlError(f"{arm_id}/seed{seed}: checkpoint sidecar is invalid") from exc
    sidecar = _validate_sidecar(sidecar_raw, arm_id=arm_id)
    if expected_run_id is not None and sidecar["run_id"] != expected_run_id:
        raise PlainControlError(
            f"{arm_id}/seed{seed}: checkpoint run_id {sidecar['run_id']!r} "
            f"!= expected {expected_run_id!r}"
        )

    expected_model_class = f"thermoroute.neural_baselines.{'PlainMLPForecaster' if arm_id == 'PlainMLP-7var' else 'PlainCausalTCNForecaster'}"
    if sidecar["model_class"] != expected_model_class:
        raise PlainControlError(
            f"{arm_id}/seed{seed}: checkpoint model_class {sidecar['model_class']!r} "
            f"!= expected {expected_model_class!r}"
        )

    # Hardened no-follow open + hash check + weights-only torch.load.
    payload, _payload_bytes, _payload_sha = checkpoint._read_checkpoint_payload(
        path,
        map_location="cpu",
        expected_bytes=int(sidecar["checkpoint_bytes"]),
        expected_sha256=str(sidecar["checkpoint_sha256"]),
    )
    if not isinstance(payload, Mapping) or payload.get("format") != checkpoint.CHECKPOINT_VERSION:
        raise PlainControlError(f"{arm_id}/seed{seed}: checkpoint payload format is invalid")
    if payload.get("run_id") != sidecar["run_id"]:
        raise PlainControlError(f"{arm_id}/seed{seed}: checkpoint payload run_id disagrees with sidecar")
    if payload.get("model_class") != sidecar["model_class"]:
        raise PlainControlError(f"{arm_id}/seed{seed}: checkpoint payload model_class disagrees with sidecar")
    best_state = payload.get("best_model_state")
    if not isinstance(best_state, Mapping) or not best_state:
        raise PlainControlError(f"{arm_id}/seed{seed}: checkpoint has no best_model_state")

    model = build_plain_control_model(arm_id, seed=int(seed), n_stations=n_stations)
    reconstructed_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_params = EXPECTED_PARAMETER_COUNTS.get(arm_id)
    if expected_params is not None and reconstructed_params != expected_params:
        raise PlainControlError(
            f"{arm_id}/seed{seed}: reconstructed parameter count {reconstructed_params} "
            f"!= frozen contract {expected_params}"
        )
    state_numel = sum(int(v.numel()) for v in best_state.values() if isinstance(v, torch.Tensor))
    if state_numel != reconstructed_params:
        raise PlainControlError(
            f"{arm_id}/seed{seed}: best_model_state numel {state_numel} "
            f"!= reconstructed parameters {reconstructed_params}"
        )
    try:
        model.load_state_dict(best_state, strict=True)
    except (TypeError, ValueError) as exc:
        raise PlainControlError(
            f"{arm_id}/seed{seed}: best_model_state is not load-compatible with the rebuilt architecture"
        ) from exc
    model.eval()

    architecture = model.architecture_metadata()  # type: ignore[attr-defined]
    metadata: dict[str, Any] = {
        "model_class": sidecar["model_class"],
        "run_id": sidecar["run_id"],
        "epoch": int(sidecar["epoch"]),
        "best_epoch": int(payload.get("best_epoch", -1)),
        "best_metric": float(payload.get("best_metric", float("nan"))),
        "arm_id": arm_id,
        "seed": int(seed),
        "trainable_parameters": reconstructed_params,
        "architecture": json.loads(json.dumps(architecture, sort_keys=True, allow_nan=False)),
        "calibration_state": "NO_FROZEN_CALIBRATION",
        "has_bundle": False,
        "checkpoint_path": str(path),
    }
    return model, metadata


__all__ = [
    "CONTROL_SEEDS",
    "DEFAULT_N_STATIONS",
    "DEVELOPMENT_SCOPE",
    "EXPECTED_PARAMETER_COUNTS",
    "FULL_VARIABLES",
    "PLAIN_CONTROL_ARMS",
    "PLAIN_FEATURE_SET",
    "PlainControlError",
    "build_plain_control_model",
    "load_plain_control_checkpoint",
    "plain_control_parameter_count",
]
