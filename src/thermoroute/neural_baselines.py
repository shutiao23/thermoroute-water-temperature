"""Plain neural controls for development-only architecture comparisons.

The models in this module deliberately contain none of ThermoRoute's physics
prior, dynamic lag router, mixture of experts, or bounded residual.  They use
only the observed history tensors ``X`` and ``Mask`` and, when explicitly
enabled, the issue site's integer identity.  A batch may contain the remaining
``WindowedData.batch`` fields, including ``y``; those fields are never read.

These controls make an architecture comparison possible, but their default
parameter counts are not a fairness guarantee.  A study must match parameter
budget, optimisation budget, input schema, and tuning budget externally and
report the resulting counts.  ``architecture_metadata`` exposes the exact
constructor kwargs and trainable parameter count needed for that audit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


_BUDGET_MATCHING_NOTE = (
    "Match trainable parameters, optimiser steps, input schema, early-stopping rule, "
    "and tuning budget externally; the constructor defaults do not establish fairness."
)
_FUTURE_KEYS_NEVER_READ = ("y", "clim_tgt", "damped_prior", "target_date")


@dataclass(frozen=True)
class NeuralBaselineOutputs:
    """Pure-neural forecasts plus aliases used by the existing training loop.

    ``point`` is the MSE conditional-mean forecast and is independently
    parameterised from the pinball q50 head.  ``q_lo`` and ``q_hi`` use separate
    positive-spread heads around q50, guaranteeing q05 <= q50 <= q95 without
    sorting the point forecast.  Compatibility fields are intentionally
    empty/NaN.
    """

    point: Tensor
    q_lo: Tensor
    q_med: Tensor
    q_hi: Tensor
    event_logit: Tensor
    prior: Tensor
    kappa: Tensor
    teq: Tensor
    lag_weights: Tensor
    pi: Tensor
    internal_prior: Tensor | None = None

    @property
    def q05(self) -> Tensor:
        return self.q_lo

    @property
    def q50(self) -> Tensor:
        return self.q_med

    @property
    def q95(self) -> Tensor:
        return self.q_hi

    @property
    def exceed_logit(self) -> Tensor:
        return self.event_logit


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    values = tuple(horizons)
    if not values:
        raise ValueError("horizons must not be empty")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in values):
        raise ValueError("horizons must contain positive integers")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("horizons must be strictly increasing")
    return values


def _validated_probability(value: float, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not math.isfinite(number) or not lower_ok or number >= 1.0:
        interval = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return number


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("init_seed must be an integer in [0, 2**63)")
    return seed


def _validate_common_constructor(
    *,
    n_vars: int,
    context_length: int,
    horizons: Sequence[int],
    n_stations: int,
    station_agnostic: bool,
    station_embed_dim: int,
    dropout: float,
    min_spread: float,
    init_seed: int,
) -> tuple[int, int, tuple[int, ...], int, bool, int, float, float, int]:
    if not isinstance(station_agnostic, bool):
        raise ValueError("station_agnostic must be explicitly True or False")
    spread = _validated_probability(min_spread, "min_spread", allow_zero=False)
    return (
        _positive_int(n_vars, "n_vars"),
        _positive_int(context_length, "context_length"),
        _validated_horizons(horizons),
        _positive_int(n_stations, "n_stations"),
        station_agnostic,
        _positive_int(station_embed_dim, "station_embed_dim"),
        _validated_probability(dropout, "dropout", allow_zero=True),
        spread,
        _validated_seed(init_seed),
    )


def _validate_history(
    batch: Mapping[str, Tensor],
    *,
    n_vars: int,
    context_length: int,
) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping of WindowedData tensor fields")
    missing = sorted({"X", "Mask"} - set(batch))
    if missing:
        raise KeyError(f"batch is missing required history fields: {missing}")
    X, mask = batch["X"], batch["Mask"]
    if not isinstance(X, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("batch['X'] and batch['Mask'] must be torch tensors")
    expected_tail = (context_length, n_vars)
    if X.ndim != 3 or tuple(X.shape[1:]) != expected_tail:
        raise ValueError(
            f"batch['X'] must have shape [B, {context_length}, {n_vars}], "
            f"got {tuple(X.shape)}"
        )
    if X.shape[0] <= 0:
        raise ValueError("batch size must be positive")
    if tuple(mask.shape) != tuple(X.shape):
        raise ValueError("batch['Mask'] must have exactly the same shape as batch['X']")
    if not X.is_floating_point():
        raise TypeError("batch['X'] must have a floating dtype")
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise TypeError("batch['Mask'] must have a bool or floating dtype")
    if X.device != mask.device:
        raise ValueError("batch['X'] and batch['Mask'] must be on the same device")
    if not bool(torch.isfinite(X).all().item()):
        raise ValueError("batch['X'] must be finite; missingness belongs in batch['Mask']")
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all().item()):
        raise ValueError("batch['Mask'] must be finite")
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("batch['Mask'] must be binary")
    return X, mask.to(dtype=torch.bool)


def _masked_history_features(X: Tensor, observed: Tensor) -> Tensor:
    values = torch.where(observed, X, torch.zeros((), dtype=X.dtype, device=X.device))
    return torch.cat([values, observed.to(dtype=X.dtype)], dim=-1)


def _validated_station(
    batch: Mapping[str, Tensor],
    *,
    batch_size: int,
    n_stations: int,
    device: torch.device,
) -> Tensor:
    if "station" not in batch:
        raise KeyError("station-aware mode requires batch['station']")
    station = batch["station"]
    if not isinstance(station, Tensor):
        raise TypeError("batch['station'] must be a torch tensor")
    if station.dtype != torch.long:
        raise TypeError("batch['station'] must have dtype torch.long")
    if tuple(station.shape) != (batch_size,):
        raise ValueError(f"batch['station'] must have shape [{batch_size}]")
    if station.device != device:
        raise ValueError("batch['station'] must be on the same device as batch['X']")
    if bool(((station < 0) | (station >= n_stations)).any().item()):
        raise ValueError(f"batch['station'] must lie in [0, {n_stations})")
    return station


def _outputs_from_raw(raw: Tensor, *, horizons: int, min_spread: float) -> NeuralBaselineOutputs:
    batch_size = raw.shape[0]
    raw = raw.reshape(batch_size, horizons, 5)
    point = raw[..., 0]
    q_med = raw[..., 1]
    q_lo = q_med - (nn.functional.softplus(raw[..., 2]) + min_spread)
    q_hi = q_med + (nn.functional.softplus(raw[..., 3]) + min_spread)
    event_logit = raw[..., 4]
    nan_forecast = torch.full_like(q_med, float("nan"))
    nan_sample = torch.full(
        (batch_size,), float("nan"), dtype=q_med.dtype, device=q_med.device
    )
    return NeuralBaselineOutputs(
        point=point,
        q_lo=q_lo,
        q_med=q_med,
        q_hi=q_hi,
        event_logit=event_logit,
        prior=nan_forecast,
        kappa=nan_sample,
        teq=nan_sample.clone(),
        lag_weights=q_med.new_zeros((batch_size, horizons, 0, 0)),
        pi=q_med.new_zeros((batch_size, 0)),
    )


class _PlainBaseline(nn.Module):
    """Shared strict batch and metadata contract; not a public architecture."""

    architecture_id: str
    n_vars: int
    context_length: int
    horizons: tuple[int, ...]
    n_stations: int
    station_agnostic: bool
    station_embed_dim: int
    dropout_probability: float
    min_spread: float
    init_seed: int
    station_embedding: nn.Embedding | None

    def _history(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        return _validate_history(
            batch, n_vars=self.n_vars, context_length=self.context_length
        )

    def _append_station(
        self,
        representation: Tensor,
        batch: Mapping[str, Tensor],
    ) -> Tensor:
        if self.station_embedding is None:
            return representation
        station = _validated_station(
            batch,
            batch_size=representation.shape[0],
            n_stations=self.n_stations,
            device=representation.device,
        )
        return torch.cat([representation, self.station_embedding(station)], dim=-1)

    def n_params(self) -> int:
        """Number of trainable parameters; use it when matching control budgets."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture_kwargs(self) -> dict[str, object]:
        raise NotImplementedError

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "format_version": 2,
            "architecture_id": self.architecture_id,
            "module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
            "constructor_kwargs": self.architecture_kwargs(),
            "input_keys_read": (
                ("X", "Mask") if self.station_agnostic else ("X", "Mask", "station")
            ),
            "future_keys_never_read": _FUTURE_KEYS_NEVER_READ,
            "output_keys": ("point", "q_lo", "q_med", "q_hi", "event_logit"),
            "point_objective": "mse_conditional_mean",
            "q50_is_independent_from_point": True,
            "quantile_levels": (0.05, 0.50, 0.95),
            "trainable_parameters": self.n_params(),
            "budget_matching_note": _BUDGET_MATCHING_NOTE,
        }


class PlainMLPForecaster(_PlainBaseline):
    """A history-flattened MLP with no physical or temporal inductive prior."""

    architecture_id = "plain_history_mlp_v2"

    def __init__(
        self,
        *,
        n_vars: int,
        context_length: int,
        horizons: Sequence[int],
        station_agnostic: bool,
        init_seed: int,
        n_stations: int = 1,
        station_embed_dim: int = 8,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.10,
        min_spread: float = 1e-4,
    ) -> None:
        super().__init__()
        (
            self.n_vars,
            self.context_length,
            self.horizons,
            self.n_stations,
            self.station_agnostic,
            self.station_embed_dim,
            self.dropout_probability,
            self.min_spread,
            self.init_seed,
        ) = _validate_common_constructor(
            n_vars=n_vars,
            context_length=context_length,
            horizons=horizons,
            n_stations=n_stations,
            station_agnostic=station_agnostic,
            station_embed_dim=station_embed_dim,
            dropout=dropout,
            min_spread=min_spread,
            init_seed=init_seed,
        )
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.depth = _positive_int(depth, "depth")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            layers: list[nn.Module] = []
            input_dim = 2 * self.context_length * self.n_vars
            for layer_index in range(self.depth):
                layers.extend(
                    [
                        nn.Linear(input_dim if layer_index == 0 else self.hidden_dim,
                                  self.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.dropout_probability),
                    ]
                )
            self.encoder = nn.Sequential(*layers)
            self.station_embedding = (
                None
                if self.station_agnostic
                else nn.Embedding(self.n_stations, self.station_embed_dim)
            )
            head_dim = self.hidden_dim + (0 if self.station_agnostic else self.station_embed_dim)
            self.head = nn.Linear(head_dim, 5 * len(self.horizons))

    def forward(self, batch: Mapping[str, Tensor]) -> NeuralBaselineOutputs:
        X, observed = self._history(batch)
        flattened = _masked_history_features(X, observed).flatten(start_dim=1)
        representation = self._append_station(self.encoder(flattened), batch)
        return _outputs_from_raw(
            self.head(representation),
            horizons=len(self.horizons),
            min_spread=self.min_spread,
        )

    def architecture_kwargs(self) -> dict[str, object]:
        return {
            "n_vars": self.n_vars,
            "context_length": self.context_length,
            "horizons": self.horizons,
            "station_agnostic": self.station_agnostic,
            "init_seed": self.init_seed,
            "n_stations": self.n_stations,
            "station_embed_dim": self.station_embed_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout_probability,
            "min_spread": self.min_spread,
        }


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        padded = nn.functional.pad(sequence, (self.left_padding, 0))
        return sequence + self.dropout(self.activation(self.conv(padded)))


class PlainCausalTCNForecaster(_PlainBaseline):
    """A residual causal TCN without a router, MoE, prior, or output bound."""

    architecture_id = "plain_causal_tcn_v2"

    def __init__(
        self,
        *,
        n_vars: int,
        context_length: int,
        horizons: Sequence[int],
        station_agnostic: bool,
        init_seed: int,
        n_stations: int = 1,
        station_embed_dim: int = 8,
        channels: int = 48,
        blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.10,
        min_spread: float = 1e-4,
    ) -> None:
        super().__init__()
        (
            self.n_vars,
            self.context_length,
            self.horizons,
            self.n_stations,
            self.station_agnostic,
            self.station_embed_dim,
            self.dropout_probability,
            self.min_spread,
            self.init_seed,
        ) = _validate_common_constructor(
            n_vars=n_vars,
            context_length=context_length,
            horizons=horizons,
            n_stations=n_stations,
            station_agnostic=station_agnostic,
            station_embed_dim=station_embed_dim,
            dropout=dropout,
            min_spread=min_spread,
            init_seed=init_seed,
        )
        self.channels = _positive_int(channels, "channels")
        self.blocks = _positive_int(blocks, "blocks")
        self.kernel_size = _positive_int(kernel_size, "kernel_size")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2 for a temporal convolution")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.input_projection = nn.Conv1d(2 * self.n_vars, self.channels, 1)
            self.temporal_blocks = nn.ModuleList(
                [
                    _CausalResidualBlock(
                        self.channels,
                        self.kernel_size,
                        dilation=2**block_index,
                        dropout=self.dropout_probability,
                    )
                    for block_index in range(self.blocks)
                ]
            )
            # LayerNorm acts independently at each time position, so it cannot
            # leak suffix information into a prefix representation.
            self.output_norm = nn.LayerNorm(self.channels)
            self.station_embedding = (
                None
                if self.station_agnostic
                else nn.Embedding(self.n_stations, self.station_embed_dim)
            )
            head_dim = self.channels + (0 if self.station_agnostic else self.station_embed_dim)
            self.head = nn.Linear(head_dim, 5 * len(self.horizons))

    def _encode_validated(self, X: Tensor, observed: Tensor) -> Tensor:
        sequence = _masked_history_features(X, observed).transpose(1, 2)
        hidden = self.input_projection(sequence)
        for block in self.temporal_blocks:
            hidden = block(hidden)
        return self.output_norm(hidden.transpose(1, 2))

    def encode_sequence(self, X: Tensor, Mask: Tensor) -> Tensor:
        """Return every causal hidden state for an explicit prefix audit.

        In evaluation mode, changing positions after index ``t`` leaves all
        returned states through ``t`` exactly unchanged.  The method enforces
        the same fixed context schema as :meth:`forward`.
        """
        validated_X, observed = _validate_history(
            {"X": X, "Mask": Mask},
            n_vars=self.n_vars,
            context_length=self.context_length,
        )
        return self._encode_validated(validated_X, observed)

    def forward(self, batch: Mapping[str, Tensor]) -> NeuralBaselineOutputs:
        X, observed = self._history(batch)
        final_state = self._encode_validated(X, observed)[:, -1, :]
        representation = self._append_station(final_state, batch)
        return _outputs_from_raw(
            self.head(representation),
            horizons=len(self.horizons),
            min_spread=self.min_spread,
        )

    def architecture_kwargs(self) -> dict[str, object]:
        return {
            "n_vars": self.n_vars,
            "context_length": self.context_length,
            "horizons": self.horizons,
            "station_agnostic": self.station_agnostic,
            "init_seed": self.init_seed,
            "n_stations": self.n_stations,
            "station_embed_dim": self.station_embed_dim,
            "channels": self.channels,
            "blocks": self.blocks,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_probability,
            "min_spread": self.min_spread,
        }


__all__ = [
    "NeuralBaselineOutputs",
    "PlainCausalTCNForecaster",
    "PlainMLPForecaster",
]
