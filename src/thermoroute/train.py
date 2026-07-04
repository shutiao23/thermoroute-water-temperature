"""Training loop, composite loss, and prediction export for ThermoRoute / GRU.

Loss = multi-horizon pinball (drives calibrated quantiles and a median point
forecast) + exceedance BCE + an L1 leash keeping the neural residual close to the
physics prior + a quantile non-crossing penalty.  All weights are fixed in
``config.TrainConfig`` and selected on the validation years only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from . import config as C
from . import results as R
from .datasets import WindowedData
from .thermoroute import ThermoRoute


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def pinball_loss(y: Tensor, q: Tensor, tau: float) -> Tensor:
    d = y - q
    return torch.mean(torch.maximum(tau * d, (tau - 1.0) * d))


def composite_loss(out, y: Tensor, ybin: Tensor, cfg: C.TrainConfig) -> Tensor:
    # Point head trained on MSE so it targets the (RMSE-optimal) conditional mean;
    # pinball only on the outer quantiles for calibrated, sharp intervals.
    point = torch.mean((y - out.median) ** 2)
    lq = pinball_loss(y, out.q05, 0.05) + pinball_loss(y, out.q95, 0.95)
    cross = (torch.relu(out.q05 - out.q50) + torch.relu(out.q50 - out.q95)).mean()
    evt = nn.functional.binary_cross_entropy_with_logits(out.exceed_logit, ybin)
    resid = (out.median - out.prior).abs().mean()
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
        self.head_med = nn.Linear(d, self.H)
        self.head_lo = nn.Linear(d, self.H)
        self.head_hi = nn.Linear(d, self.H)
        self.head_evt = nn.Linear(d, self.H)

    def forward(self, batch):
        h, _ = self.rnn(batch["X"])
        z = h[:, -1, :]
        med = self.head_med(z) + batch["clim_tgt"]      # anchor on climatology
        lo = torch.nn.functional.softplus(self.head_lo(z))
        hi = torch.nn.functional.softplus(self.head_hi(z))
        from .thermoroute import ThermoRouteOutputs
        return ThermoRouteOutputs(med, med - lo, med, med + hi, self.head_evt(z),
                                  batch["clim_tgt"], torch.zeros(med.shape[0]),
                                  batch["clim_t"], torch.zeros(med.shape[0], self.H, 1, 1),
                                  torch.zeros(med.shape[0], 1))


# --------------------------------------------------------------------------- #
# LSTM reference deep model — the field-standard "top-down global LSTM" foil.
# Station-agnostic (no learned per-station embedding) so the SAME network can be
# trained in-sample AND transferred to held-out HUC2 regions (the entity-aware
# variant cannot embed an unseen gage). It has the exact same heads / climatology
# anchor / composite loss as ThermoRoute, so any difference is the physics prior +
# bounded residual, not the training recipe. This is the deep baseline the
# stream-temperature-ML literature (Rahmani 2021; Willard 2024) benchmarks against.
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    # 1 layer, hidden 64, over a 14-day context window — the standard modest LSTM
    # sizing for daily stream-temperature (Feigl 2021; Rahmani 2021), and what
    # keeps the sequential CPU cost tractable on a 527k-window panel. The 14-day
    # window is sliced from the shared 32-day windows so every model sees an
    # identical sample set / split.
    def __init__(self, n_vars: int, horizons=C.HORIZONS, d: int = 64, layers: int = 1,
                 dropout: float = 0.0, context: int = 14):
        super().__init__()
        self.H = len(horizons)
        self.context = context
        self.rnn = nn.LSTM(n_vars, d, layers, batch_first=True, dropout=dropout)
        self.head_med = nn.Linear(d, self.H)
        self.head_lo = nn.Linear(d, self.H)
        self.head_hi = nn.Linear(d, self.H)
        self.head_evt = nn.Linear(d, self.H)

    def forward(self, batch):
        h, _ = self.rnn(batch["X"][:, -self.context:, :])
        z = h[:, -1, :]
        # Anchor on PERSISTENCE (last observed WTEMP), i.e. predict the multi-day
        # increment. This is the standard strong short-horizon parameterisation and
        # makes the LSTM persistence-competitive at h1; climatology is a poor h1
        # anchor. Persistence is the trivial baseline available to every model, so
        # this gives the LSTM no physics prior — the contrast with ThermoRoute's
        # damped-relaxation prior + bounded residual + calibration is preserved.
        med = self.head_med(z) + batch["wtemp_t"][:, None]
        lo = torch.nn.functional.softplus(self.head_lo(z))
        hi = torch.nn.functional.softplus(self.head_hi(z))
        from .thermoroute import ThermoRouteOutputs
        return ThermoRouteOutputs(med, med - lo, med, med + hi, self.head_evt(z),
                                  batch["clim_tgt"], torch.zeros(med.shape[0]),
                                  batch["clim_t"], torch.zeros(med.shape[0], self.H, 1, 1),
                                  torch.zeros(med.shape[0], 1))


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


def fit_model(model: nn.Module, wd: WindowedData, thresholds: dict[str, float],
              cfg: C.TrainConfig = C.TRAIN, seed: int = 0, device: str = "cpu",
              model_name: str = "ThermoRoute", scope: str = "joint",
              feature_set: str = "V3", verbose: bool = False,
              train_stations: tuple[str, ...] | None = None) -> FitResult:
    set_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    tr_idx, va_idx = wd.idx("train"), wd.idx("val")
    if train_stations is not None:        # LOSO: train only on the in-sample stations
        keep = np.array([C.STATIONS[i] in train_stations for i in wd.station])
        tr_idx = tr_idx[keep[tr_idx]]
        va_idx = va_idx[keep[va_idx]]
    thr_tr = _station_threshold_tensor(thresholds, wd.station[tr_idx], device)
    thr_va = _station_threshold_tensor(thresholds, wd.station[va_idx], device)
    y_tr = torch.as_tensor(wd.y[tr_idx], dtype=torch.float32, device=device)
    y_va = torch.as_tensor(wd.y[va_idx], dtype=torch.float32, device=device)
    ybin_tr = (y_tr > thr_tr[:, None]).float()
    ybin_va = (y_va > thr_va[:, None]).float()
    batch_tr = wd.batch(tr_idx, device)
    batch_va = wd.batch(va_idx, device)

    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    n = len(tr_idx)
    rng = np.random.default_rng(seed)
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, cfg.batch_size):
            j = perm[s:s + cfg.batch_size]
            sub = {k: v[j] for k, v in batch_tr.items()}
            out = model(sub)
            loss = composite_loss(out, y_tr[j], ybin_tr[j], cfg)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(batch_va)
            vloss = composite_loss(vout, y_va, ybin_va, cfg).item()
            vrmse = torch.sqrt(((vout.median - y_va) ** 2).mean()).item()
            vprior = torch.sqrt(((vout.prior - y_va) ** 2).mean()).item()
        sched.step(vrmse)
        # Select on point accuracy (the headline metric); conformal fixes coverage.
        select = vrmse
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d} val_loss {vloss:.4f} val_rmse {vrmse:.4f} "
                  f"prior_rmse {vprior:.4f}")
        if select < best_val - 1e-5:
            best_val, best_state, best_epoch, bad = select, \
                {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    pred = _export_predictions(model, wd, thresholds, device, model_name, scope, feature_set, seed)
    return FitResult(model, pred, best_val, best_epoch)


def _export_predictions(model, wd, thresholds, device, model_name, scope,
                        feature_set, seed) -> pd.DataFrame:
    model.eval()
    frames = []
    for split in ("val", "calib", "test"):
        idx = wd.idx(split)
        if len(idx) == 0:
            continue
        with torch.no_grad():
            out = model(wd.batch(idx, device))
        H = len(wd.horizons)
        n = len(idx)
        med = out.median.cpu().numpy()
        q05, q50, q95 = out.q05.cpu().numpy(), out.q50.cpu().numpy(), out.q95.cpu().numpy()
        pexc = torch.sigmoid(out.exceed_logit).cpu().numpy()
        for hi, h in enumerate(wd.horizons):
            site = np.array([C.STATIONS[i] for i in wd.station[idx]])
            issue = wd.issue_date[idx]
            tdate = issue + np.timedelta64(h, "D")
            frames.append(R.make_pred_frame(
                model=model_name, scope=scope, feature_set=feature_set, seed=seed,
                site_id=site, horizon=np.full(n, h), split=np.full(n, split),
                issue_date=issue, target_date=tdate,
                y_true=wd.y[idx][:, hi], y_pred=med[:, hi],
                q05=q05[:, hi], q50=q50[:, hi], q95=q95[:, hi], p_exceed=pexc[:, hi]))
    return pd.concat(frames, ignore_index=True)
