"""Training loop, composite loss, and prediction export for sequence models.

The RMSE point head is trained by MSE.  Separate q05/q50/q95 heads are trained
by pinball loss, with q05 and q95 constructed around q50 so they cannot cross.
The remaining terms are exceedance BCE and an L1 leash from the point forecast
to the frozen physical safety anchor.  All weights are fixed in
``config.TrainConfig`` and selected on the validation years only.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from . import config as C
from . import results as R
from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .datasets import WindowedData


def configure_deterministic_runtime(*, threads: int = 1) -> None:
    """Apply the formal Torch determinism and thread contract.

    Formal entry points set BLAS/OpenMP environment variables before importing
    numerical libraries.  This function closes the Torch side of that contract;
    callers must invoke it before resolving a content-addressed run identity.
    """
    if threads != 1:
        raise ValueError("formal numerical runtime requires one Torch thread")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        if torch.get_num_interop_threads() != threads:
            raise
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


def set_seed(seed: int) -> None:
    """Seed every RNG used here and enforce deterministic Torch kernels.

    Model construction must happen *after* this call.  ``fit_model`` therefore
    accepts a factory; passing an already-created module remains supported for
    legacy scripts, but cannot retroactively make its initialisation seeded.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # A formal run must fail when an accelerator has no deterministic kernel;
    # a warning is easy to suppress and cannot support a replay claim.
    configure_deterministic_runtime()


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def pinball_loss(y: Tensor, q: Tensor, tau: float) -> Tensor:
    d = y - q
    return torch.mean(torch.maximum(tau * d, (tau - 1.0) * d))


def composite_loss(out, y: Tensor, ybin: Tensor, cfg: C.TrainConfig) -> Tensor:
    # Point head trained on MSE so it targets the (RMSE-optimal) conditional mean;
    # q50 has its own pinball-trained parameters and is never an alias/sorted
    # version of that point forecast.
    point = torch.mean((y - out.point) ** 2)
    lq = (pinball_loss(y, out.q05, 0.05)
          + pinball_loss(y, out.q50, 0.50)
          + pinball_loss(y, out.q95, 0.95))
    # Every supported neural forecaster constructs q05=q50-softplus(.) and
    # q95=q50+softplus(.).  This compatibility term is therefore identically
    # zero; it is retained only so historical TrainConfig fields remain
    # explicit, not because it supplies an effective non-crossing penalty.
    cross = (torch.relu(out.q05 - out.q50) + torch.relu(out.q50 - out.q95)).mean()
    evt = nn.functional.binary_cross_entropy_with_logits(out.exceed_logit, ybin)
    finite_prior = torch.isfinite(out.prior)
    resid = ((out.point[finite_prior] - out.prior[finite_prior]).abs().mean()
             if finite_prior.any() else out.point.new_zeros(()))
    return point + lq + cfg.lambda_event * evt + cfg.lambda_crossing * cross \
        + cfg.lambda_residual * resid


# --------------------------------------------------------------------------- #
# GRU reference deep model (same heads/prior-free) for an apples-to-apples DL row
# --------------------------------------------------------------------------- #
class GRUForecaster(nn.Module):
    def __init__(self, n_vars: int, horizons=C.HORIZONS, d: int = 64, layers: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.H = len(horizons)
        self.rnn = nn.GRU(n_vars, d, layers, batch_first=True, dropout=dropout)
        self.head_point = nn.Linear(d, self.H)
        self.head_q50 = nn.Linear(d, self.H)
        self.head_lo = nn.Linear(d, self.H)
        self.head_hi = nn.Linear(d, self.H)
        self.head_evt = nn.Linear(d, self.H)

    def forward(self, batch):
        h, _ = self.rnn(batch["X"])
        z = h[:, -1, :]
        point = self.head_point(z) + batch["clim_tgt"]  # MSE conditional mean
        q50 = self.head_q50(z) + batch["clim_tgt"]      # pinball median
        lo = torch.nn.functional.softplus(self.head_lo(z))
        hi = torch.nn.functional.softplus(self.head_hi(z))
        from .thermoroute import ThermoRouteOutputs
        return ThermoRouteOutputs(point, q50 - lo, q50, q50 + hi, self.head_evt(z),
                                  batch["clim_tgt"], point.new_zeros(point.shape[0]),
                                  batch["clim_t"],
                                  point.new_zeros(point.shape[0], self.H, 1, 1),
                                  point.new_zeros(point.shape[0], 1))


# --------------------------------------------------------------------------- #
# LSTM reference deep model — the field-standard "top-down global LSTM" foil.
# The same-station arm can use a station embedding, matching ThermoRoute's access
# to site identity.  Held-region transfer disables that embedding.  It has the
# exact same heads / composite loss as ThermoRoute, so differences are not caused
# by an arbitrary history-length or site-identity information disadvantage. This is
# the deep baseline the
# stream-temperature-ML literature (Rahmani 2021; Willard 2024) benchmarks against.
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    # 1 layer, hidden 64, over the exact same 32-day context exposed to
    # ThermoRoute.  A shorter context would silently give the baseline less
    # information even if both models shared the same sample registry.
    def __init__(self, n_vars: int, horizons=C.HORIZONS, d: int = 64, layers: int = 1,
                 dropout: float = 0.0, context: int = C.CONTEXT_LENGTH,
                 n_stations: int = len(C.STATIONS), station_agnostic: bool = False,
                 station_embed_dim: int = 8, use_derived_context: bool = False,
                 anchor: str = "persistence"):
        super().__init__()
        if anchor not in {"persistence", "damped"}:
            raise ValueError("LSTM anchor must be 'persistence' or 'damped'")
        self.H = len(horizons)
        self.context = context
        self.station_agnostic = station_agnostic
        self.use_derived_context = use_derived_context
        self.anchor = anchor
        # The main ThermoRoute encoder receives an explicit observed-value mask.
        # Concatenating the same mask here prevents missingness patterns from
        # becoming an undeclared information advantage for either architecture.
        self.rnn = nn.LSTM(2 * n_vars, d, layers, batch_first=True, dropout=dropout)
        self.station_embedding = (
            None if station_agnostic else nn.Embedding(n_stations, station_embed_dim)
        )
        head_dim = d if station_agnostic else d + station_embed_dim
        # These are train-fitted/calendar-derived quantities already available
        # to ThermoRoute.  They contain no future observation.  Their inclusion
        # is selected only on the validation split as a declared fairness arm.
        if use_derived_context:
            head_dim += 2 * self.H + 2  # clim_tgt + damped_prior + sin/cos season
        self.head_point = nn.Linear(head_dim, self.H)
        self.head_q50 = nn.Linear(head_dim, self.H)
        self.head_lo = nn.Linear(head_dim, self.H)
        self.head_hi = nn.Linear(head_dim, self.H)
        self.head_evt = nn.Linear(head_dim, self.H)

    def forward(self, batch):
        sequence = torch.cat([batch["X"], batch["Mask"]], dim=-1)
        h, _ = self.rnn(sequence[:, -self.context:, :])
        z = h[:, -1, :]
        if self.station_embedding is not None:
            z = torch.cat([z, self.station_embedding(batch["station"])], dim=-1)
        if self.use_derived_context:
            z = torch.cat([
                z, batch["clim_tgt"], batch["damped_prior"], batch["season"]
            ], dim=-1)
        # The conventional arm predicts an increment from persistence.  A
        # predeclared fairness arm instead predicts an unrestricted increment
        # from the same train-fit damped anchor exposed to ThermoRoute.  Which arm
        # is primary is selected on validation only; neither is structurally
        # bounded like ThermoRoute.
        anchor = (
            batch["damped_prior"] if self.anchor == "damped"
            else batch["wtemp_t"][:, None].expand(-1, self.H)
        )
        point = self.head_point(z) + anchor
        q50 = self.head_q50(z) + anchor
        lo = torch.nn.functional.softplus(self.head_lo(z))
        hi = torch.nn.functional.softplus(self.head_hi(z))
        reference = anchor
        from .thermoroute import ThermoRouteOutputs
        return ThermoRouteOutputs(point, q50 - lo, q50, q50 + hi, self.head_evt(z),
                                  reference,
                                  torch.zeros(point.shape[0], device=point.device),
                                  batch["clim_t"],
                                  point.new_zeros(point.shape[0], self.H, 1, 1),
                                  point.new_zeros(point.shape[0], 1))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    model: nn.Module
    pred: pd.DataFrame
    best_val: float
    epochs: int


def _station_threshold_tensor(thresholds, station_idx: np.ndarray, device) -> Tensor:
    arr = np.array([thresholds[C.STATIONS[i]] for i in station_idx])
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _balanced_epoch_order(stations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return an equal-station, fixed-size bootstrap for one training epoch.

    The epoch contains exactly ``len(stations)`` optimisation examples, so the
    station-balanced sensitivity changes sampling weights without silently
    changing the number of gradient updates.
    """
    station_ids = np.unique(stations)
    if len(station_ids) == 0:
        return np.empty(0, dtype=int)
    quota, remainder = divmod(len(stations), len(station_ids))
    order: list[np.ndarray] = []
    remainder_order = rng.permutation(station_ids)
    extras = set(remainder_order[:remainder].tolist())
    for station in station_ids:
        candidates = np.flatnonzero(stations == station)
        size = quota + int(station in extras)
        order.append(rng.choice(candidates, size=size, replace=(size > len(candidates))))
    merged = np.concatenate(order)
    return merged[rng.permutation(len(merged))]


def _station_macro_rmse(pred: Tensor, y: Tensor, stations: np.ndarray) -> float:
    """Mean of per-station RMSEs, giving every station equal validation weight."""
    values = []
    for station in np.unique(stations):
        idx = torch.as_tensor(np.flatnonzero(stations == station),
                              dtype=torch.long, device=pred.device)
        values.append(torch.sqrt(torch.mean((pred[idx] - y[idx]) ** 2)))
    return float(torch.stack(values).mean().item())


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit CPU/MPS/CUDA target (or choose the best available)."""
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported training device: {resolved.type}")
    return resolved


def _index_chunks(index: np.ndarray, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(index), batch_size):
        yield index[start:start + batch_size]


def _validate_batched(model: nn.Module, wd: WindowedData, index: np.ndarray,
                      thresholds: Mapping[str, float], cfg: C.TrainConfig,
                      device: torch.device, batch_size: int
                      ) -> tuple[float, float, float, float]:
    """Validation without materialising the complete fold on the accelerator."""
    if len(index) == 0:
        raise ValueError("validation split is empty")
    weighted_loss = 0.0
    examples = 0
    sq_error = 0.0
    target_count = 0
    prior_sq_error = 0.0
    prior_count = 0
    station_sse: dict[int, float] = {}
    station_count: dict[int, int] = {}
    model.eval()
    with torch.no_grad():
        for chunk in _index_chunks(index, batch_size):
            batch = wd.batch(chunk, device)
            y = batch["y"]
            threshold = _station_threshold_tensor(thresholds, wd.station[chunk], device)
            ybin = (y > threshold[:, None]).float()
            out = model(batch)
            loss = composite_loss(out, y, ybin, cfg)
            weighted_loss += float(loss.item()) * len(chunk)
            examples += len(chunk)
            error = (out.point - y).square()
            sq_error += float(error.sum().item())
            target_count += error.numel()
            finite_prior = torch.isfinite(out.prior)
            if finite_prior.any():
                prior_error = (out.prior[finite_prior] - y[finite_prior]).square()
                prior_sq_error += float(prior_error.sum().item())
                prior_count += prior_error.numel()
            station_cpu = wd.station[chunk]
            per_example_sse = error.sum(dim=1).detach().cpu().numpy()
            for station in np.unique(station_cpu):
                selected = station_cpu == station
                station_sse[int(station)] = station_sse.get(int(station), 0.0) + float(
                    per_example_sse[selected].sum()
                )
                station_count[int(station)] = station_count.get(int(station), 0) + int(
                    selected.sum() * error.shape[1]
                )
    macro = float(np.mean([
        np.sqrt(station_sse[station] / station_count[station])
        for station in sorted(station_sse)
    ]))
    return (
        weighted_loss / examples,
        float(np.sqrt(sq_error / target_count)),
        macro,
        float(np.sqrt(prior_sq_error / prior_count)) if prior_count else float("nan"),
    )


def fit_model(model: nn.Module | Callable[[], nn.Module], wd: WindowedData,
              thresholds: dict[str, float],
              cfg: C.TrainConfig = C.TRAIN, seed: int = 0, device: str = "cpu",
              model_name: str = "ThermoRoute", scope: str = "joint",
              feature_set: str = "V3", verbose: bool = False,
              train_stations: tuple[str, ...] | None = None,
              station_balanced: bool = False,
              selection_metric: Literal["micro", "station_macro"] = "micro",
              eval_batch_size: int | None = None,
              checkpoint_path: str | Path | None = None,
              run_id: str | None = None,
              resolved_config: Mapping[str, Any] | None = None,
              resume: bool = True,
              checkpoint_every: int = 1,
              stop_after_epoch: int | None = None,
              export_splits: tuple[str, ...] = ("val", "calib", "test")) -> FitResult:
    """Fit a model under a reproducible and optionally station-balanced recipe.

    Prefer passing ``model`` as a zero-argument factory.  This lets the function
    seed RNGs before parameter initialisation, eliminating the old situation in
    which nominally different/repeated seeds depended on ambient process state.
    """
    if selection_metric not in {"micro", "station_macro"}:
        raise ValueError("selection_metric must be 'micro' or 'station_macro'")
    unknown_export_splits = set(export_splits) - {"val", "calib", "test"}
    if unknown_export_splits:
        raise ValueError(f"unsupported export splits: {sorted(unknown_export_splits)}")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    set_seed(seed)
    device_obj = resolve_device(device)
    if not isinstance(model, nn.Module):
        model = model()
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    model.to(device_obj)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    tr_idx, va_idx = wd.idx("train"), wd.idx("val")
    if train_stations is not None:        # LOSO: train only on the in-sample stations
        keep = np.array([C.STATIONS[i] in train_stations for i in wd.station])
        tr_idx = tr_idx[keep[tr_idx]]
        va_idx = va_idx[keep[va_idx]]
    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    n = len(tr_idx)
    if n == 0:
        raise ValueError("training split is empty")
    if len(va_idx) == 0:
        raise ValueError("validation split is empty")
    eval_batch_size = int(eval_batch_size or max(cfg.batch_size, 2048))
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and not run_id:
        raise ValueError("run_id is required when checkpoint_path is set")
    checkpoint_config = dict(resolved_config or {
        "train": asdict(cfg),
        "seed": seed,
        "model_name": model_name,
        "scope": scope,
        "feature_set": feature_set,
        "station_balanced": station_balanced,
        "selection_metric": selection_metric,
        "export_splits": export_splits,
    })
    rng = np.random.default_rng(seed)
    start_epoch = 0
    if checkpoint is not None and checkpoint.exists() and resume:
        resumed = load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=opt,
            scheduler=sched,
            expected_run_id=str(run_id),
            expected_resolved_config=checkpoint_config,
            # Always deserialize the trusted training checkpoint onto CPU.  The
            # model loader copies parameters and the checkpoint helper moves
            # optimizer tensors to the model device; RNG byte tensors must stay
            # on CPU for torch.set_rng_state.
            map_location="cpu",
        )
        start_epoch = resumed.epoch + 1
        best_epoch = resumed.best_epoch
        best_val = resumed.best_metric
        best_state = resumed.best_model_state
        if set(resumed.extra) != {"bad_epochs", "train_rng_state"}:
            raise ValueError("checkpoint training-extra fields are invalid")
        raw_bad_epochs = resumed.extra["bad_epochs"]
        if (
            type(raw_bad_epochs) is not int
            or raw_bad_epochs < 0
            or raw_bad_epochs > resumed.epoch + 1
        ):
            raise ValueError("checkpoint bad_epochs is invalid")
        bad = raw_bad_epochs
        try:
            rng.bit_generator.state = resumed.extra["train_rng_state"]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("checkpoint training RNG state is invalid") from exc
        # A patience-triggered checkpoint is already complete.  Re-load its
        # best state below without taking one extra optimisation epoch.
        if bad >= cfg.patience:
            start_epoch = cfg.max_epochs
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        perm = (_balanced_epoch_order(wd.station[tr_idx], rng)
                if station_balanced else rng.permutation(n))
        for s in range(0, n, cfg.batch_size):
            j = perm[s:s + cfg.batch_size]
            global_index = tr_idx[j]
            sub = wd.batch(global_index, device_obj)
            y = sub["y"]
            threshold = _station_threshold_tensor(
                thresholds, wd.station[global_index], device_obj
            )
            ybin = (y > threshold[:, None]).float()
            out = model(sub)
            loss = composite_loss(out, y, ybin, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        vloss, vrmse, vmacro, vprior = _validate_batched(
            model, wd, va_idx, thresholds, cfg, device_obj, eval_batch_size
        )
        # Select on the declared point metric; conformal fixes coverage.  Macro
        # selection is paired with station-balanced training in the strict main
        # protocol, while micro remains available as an explicit sensitivity.
        select = vrmse if selection_metric == "micro" else vmacro
        sched.step(select)
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d} val_loss {vloss:.4f} val_rmse {vrmse:.4f} "
                  f"val_macro_rmse {vmacro:.4f} "
                  f"prior_rmse {vprior:.4f}")
        if select < best_val - 1e-5:
            best_val, best_state, best_epoch, bad = select, \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            bad += 1
        if checkpoint is not None and (
            (epoch + 1) % checkpoint_every == 0
            or bad >= cfg.patience
            or epoch == cfg.max_epochs - 1
            or (stop_after_epoch is not None and epoch >= stop_after_epoch)
        ):
            save_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=opt,
                scheduler=sched,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_val,
                best_model_state=best_state,
                run_id=str(run_id),
                resolved_config=checkpoint_config,
                extra={
                    "bad_epochs": bad,
                    "train_rng_state": rng.bit_generator.state,
                },
            )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    pred = _export_predictions(
        model, wd, thresholds, device_obj, model_name, scope, feature_set, seed,
        batch_size=eval_batch_size, splits=export_splits,
    )
    return FitResult(model, pred, best_val, best_epoch)


def _export_predictions(model, wd, thresholds, device, model_name, scope,
                        feature_set, seed, batch_size: int = 4096,
                        splits: tuple[str, ...] = ("val", "calib", "test")) -> pd.DataFrame:
    model.eval()
    frames = []
    for split in splits:
        idx = wd.idx(split)
        if len(idx) == 0:
            continue
        for chunk in _index_chunks(idx, batch_size):
            with torch.no_grad():
                out = model(wd.batch(chunk, device))
            n = len(chunk)
            point = out.point.detach().cpu().numpy()
            q05 = out.q05.detach().cpu().numpy()
            q50 = out.q50.detach().cpu().numpy()
            q95 = out.q95.detach().cpu().numpy()
            pexc = torch.sigmoid(out.exceed_logit).detach().cpu().numpy()
            for hi, h in enumerate(wd.horizons):
                site = np.array([C.STATIONS[i] for i in wd.station[chunk]])
                issue = wd.issue_date[chunk]
                tdate = (wd.target_date[chunk][:, hi] if hasattr(wd, "target_date")
                         else issue + np.timedelta64(h, "D"))
                frames.append(R.make_pred_frame(
                    model=model_name, scope=scope, feature_set=feature_set, seed=seed,
                    site_id=site, horizon=np.full(n, h), split=np.full(n, split),
                    issue_date=issue, target_date=tdate,
                    y_true=wd.y[chunk][:, hi], y_pred=point[:, hi],
                    q05=q05[:, hi], q50=q50[:, hi], q95=q95[:, hi],
                    p_exceed=pexc[:, hi]))
    return pd.concat(frames, ignore_index=True) if frames else R.empty_predictions()
