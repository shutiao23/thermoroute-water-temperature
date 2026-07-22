"""The ThermoRoute model.

The strict model separates two objects that must not be conflated:

* ``damped_prior`` is fitted on the training partition and then frozen.  It is
  the auditable safety anchor supplied by :mod:`thermoroute.datasets`.
* ``DynamicThermalRelaxationPrior`` is a learned proposal.  It may improve the
  forecast and remains interpretable, but it is *not* used as the reference in
  the bounded-deviation guarantee.

With a finite ``delta_scale`` the final point forecast is therefore guaranteed
to lie within ``±delta_scale`` of the fixed damped-persistence forecast.  Setting
``delta_scale=None`` gives the otherwise identical unbounded sensitivity model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from . import config as C


# --------------------------------------------------------------------------- #
# Sparse normalisation (sparsemax) — yields interpretable, sparse lag weights
# --------------------------------------------------------------------------- #
def sparsemax(z: Tensor, dim: int = -1) -> Tensor:
    """Martins & Astudillo (2016) sparsemax along ``dim``."""
    z = z.transpose(dim, -1)
    z_sorted, _ = torch.sort(z, descending=True, dim=-1)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    z_cumsum = z_sorted.cumsum(dim=-1)
    support = 1 + k * z_sorted > z_cumsum
    k_z = support.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(-1, k_z - 1) - 1) / k_z
    out = torch.clamp(z - tau, min=0.0)
    return out.transpose(dim, -1)


# --------------------------------------------------------------------------- #
# Module 1: dynamic thermal-relaxation physics prior
# --------------------------------------------------------------------------- #
class DynamicThermalRelaxationPrior(nn.Module):
    """Anomaly relaxation around the horizon-shifted climatology:

        a_t   = W_t − C_t                      (today's thermal anomaly)
        e_t   = g(weather)                     (weather-driven equilibrium anomaly)
        a_h   = e_t + (1−κ)^h (a_t − e_t)      (anomaly relaxes toward e_t)
        Ŵ_{t+h} = C_{t+h} + a_h

    κ is the *daily relaxation rate* — small κ ⇒ long thermal memory.  With e_t=0
    and κ=1−φ this reduces **exactly** to damped persistence toward climatology,
    so the strong baseline is a special case.  Route A lets κ depend on FLOW and
    season.  A separately declared ``use_wlevel`` mode exists for legacy feature
    schemas, but Route A fixes it off because gage height is provenance-only.
    """

    def __init__(self, n_phys: int, n_stations: int, horizons,
                 station_agnostic: bool, use_wlevel: bool = False):
        super().__init__()
        self.horizons = torch.tensor(list(horizons), dtype=torch.float32)
        self.station_agnostic = station_agnostic
        self.use_wlevel = bool(use_wlevel)
        self.eq_lin = nn.Linear(n_phys, 1)               # weather → equilibrium anomaly
        self.eq_station = nn.Embedding(n_stations, 1)
        # κ logit components
        self.k_station = nn.Embedding(n_stations, 1)
        self.k_flow = nn.Parameter(torch.zeros(1))
        self.k_level = nn.Parameter(torch.zeros(1))
        self.k_season = nn.Linear(2, 1)
        # warm-start κ≈0.05 ⇒ the prior begins essentially at damped persistence
        self.k_bias = nn.Parameter(torch.tensor(-2.94))
        nn.init.zeros_(self.eq_lin.weight)
        nn.init.zeros_(self.eq_lin.bias)
        nn.init.zeros_(self.eq_station.weight)
        nn.init.zeros_(self.k_station.weight)
        if not self.use_wlevel:
            self.k_level.requires_grad_(False)

    def forward(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        st = batch["station"]
        s_eq = 0.0 if self.station_agnostic else self.eq_station(st).squeeze(-1)
        s_k = 0.0 if self.station_agnostic else self.k_station(st).squeeze(-1)
        e_t = self.eq_lin(batch["phys_std"]).squeeze(-1) + s_eq      # [B] eq. anomaly
        a_t = batch["wtemp_t"] - batch["clim_t"]                     # [B] today anomaly
        level_term = (
            self.k_level * batch["wlevelz"] if self.use_wlevel else 0.0
        )
        k_logit = (self.k_bias + s_k
                   + self.k_flow * batch["logflowz"]
                   + level_term
                   + self.k_season(batch["season"]).squeeze(-1))
        kappa = torch.sigmoid(k_logit).clamp(1e-3, 0.999)            # [B]
        h = self.horizons.to(kappa.device)
        decay = (1.0 - kappa).unsqueeze(-1) ** h.unsqueeze(0)        # [B,H]
        a_h = e_t.unsqueeze(-1) + decay * (a_t - e_t).unsqueeze(-1)  # [B,H] anomaly
        prior = batch["clim_tgt"] + a_h                             # [B,H]
        teq = batch["clim_t"] + e_t                                 # interpretable T^eq
        return prior, kappa, teq


# --------------------------------------------------------------------------- #
# Module 2: horizon-conditioned sparse variable–lag router
# --------------------------------------------------------------------------- #
class DynamicLagRouter(nn.Module):
    def __init__(self, n_vars: int, max_lag: int, n_horizons: int, n_stations: int,
                 d: int, gate_dim: int, station_agnostic: bool, sparse: bool = True):
        super().__init__()
        self.V, self.Lr1, self.H, self.d = n_vars, max_lag + 1, n_horizons, d
        self.sparse = sparse
        self.station_agnostic = station_agnostic
        self.e_var = nn.Embedding(n_vars, d)
        self.e_lag = nn.Embedding(max_lag + 1, d)
        self.e_h = nn.Embedding(n_horizons, d)
        self.e_station = nn.Embedding(n_stations, d)
        self.gate_proj = nn.Linear(gate_dim, d)
        self.w_val = nn.Parameter(torch.randn(d) * 0.1)
        self.var_proj = nn.Embedding(n_vars, d)      # maps a variable's value -> d

    def forward(self, batch) -> tuple[Tensor, Tensor]:
        X = batch["X"]                                # [B, L, V]
        B = X.shape[0]
        Lr1 = self.Lr1
        # lag block: position ℓ -> value at t-ℓ  (lag 0 == most recent)
        tail = X[:, -Lr1:, :]                         # [B, Lr1, V]
        val = torch.flip(tail, dims=[1]).transpose(1, 2)   # [B, V, Lr1]

        dev = X.device
        ev = self.e_var(torch.arange(self.V, device=dev))            # [V,d]
        el = self.e_lag(torch.arange(Lr1, device=dev))              # [Lr1,d]
        eh = self.e_h(torch.arange(self.H, device=dev))             # [H,d]
        base = ev[:, None, :] + el[None, :, :]                      # [V,Lr1,d]
        ve = val.unsqueeze(-1) * self.w_val                         # [B,V,Lr1,d]
        key = base[None] + ve                                       # [B,V,Lr1,d]

        ctx = self.gate_proj(batch["gate"])                        # [B,d]
        if not self.station_agnostic:
            ctx = ctx + self.e_station(batch["station"])
        query = eh[None] + ctx[:, None, :]                         # [B,H,d]

        score = torch.einsum("bhd,bvld->bhvl", query, key) / (self.d ** 0.5)
        flat = score.reshape(B, self.H, self.V * Lr1)
        w = sparsemax(flat, dim=-1) if self.sparse else torch.softmax(flat, dim=-1)
        weights = w.reshape(B, self.H, self.V, Lr1)                # [B,H,V,Lr1]

        proj = self.var_proj(torch.arange(self.V, device=dev))     # [V,d]
        contrib = (val.unsqueeze(-1) * proj[None, :, None, :])     # [B,V,Lr1,d]
        routed = torch.einsum("bhvl,bvld->bhd", weights, contrib)  # [B,H,d]
        return routed, weights


# --------------------------------------------------------------------------- #
# Module 3: causal TCN encoder (global context)
# --------------------------------------------------------------------------- #
class CausalTCN(nn.Module):
    def __init__(self, n_vars: int, d: int, blocks: int, kernel: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(n_vars, d, 1)
        layers = []
        for i in range(blocks):
            dil = 2 ** i
            layers.append(nn.Sequential(
                nn.ConstantPad1d(((kernel - 1) * dil, 0), 0.0),     # causal pad
                nn.Conv1d(d, d, kernel, dilation=dil),
                nn.GELU(), nn.Dropout(dropout)))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, X: Tensor) -> Tensor:
        h = self.inp(X.transpose(1, 2))               # [B,d,L]
        for layer in self.layers:
            h = h + layer(h)
        return self.norm(h[:, :, -1])                 # [B,d] last (causal) step


# --------------------------------------------------------------------------- #
# Module 4: regime mixture-of-experts residual
# --------------------------------------------------------------------------- #
class RegimeMoE(nn.Module):
    def __init__(self, d: int, gate_dim: int, n_experts: int, n_horizons: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(gate_dim + d, d), nn.GELU(),
                                  nn.Linear(d, n_experts))
        self.e_h = nn.Embedding(n_horizons, d)
        in_dim = d + d + d
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(n_experts)])

    def forward(self, routed: Tensor, latent: Tensor, gate_feats: Tensor):
        B, H, d = routed.shape
        pi = torch.softmax(self.gate(torch.cat([gate_feats, latent], -1)), -1)  # [B,K]
        eh = self.e_h(torch.arange(H, device=routed.device))[None].expand(B, H, d)
        lat = latent[:, None, :].expand(B, H, d)
        x = torch.cat([routed, lat, eh], dim=-1)                  # [B,H,3d]
        outs = torch.stack([e(x) for e in self.experts], dim=1)   # [B,K,H,d]
        rep = torch.einsum("bk,bkhd->bhd", pi, outs)
        return rep, pi


class SingleResidualExpert(nn.Module):
    """Non-mixture control with the same inputs and depth as one MoE expert.

    The former ``noMoE`` path returned ``routed`` directly, accidentally
    deleting the TCN and horizon embedding together with the mixture.  This
    control removes only expert routing/gating.
    """

    def __init__(self, d: int, n_horizons: int):
        super().__init__()
        self.e_h = nn.Embedding(n_horizons, d)
        self.net = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, routed: Tensor, latent: Tensor) -> Tensor:
        B, H, d = routed.shape
        eh = self.e_h(torch.arange(H, device=routed.device))[None].expand(B, H, d)
        lat = latent[:, None, :].expand(B, H, d)
        return self.net(torch.cat([routed, lat, eh], dim=-1))


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
@dataclass
class ThermoRouteOutputs:
    # ``point`` is the conditional-mean forecast trained by MSE.  It is not a
    # quantile and is deliberately kept outside the quantile-ordering
    # construction below.
    point: Tensor
    q05: Tensor
    q50: Tensor
    q95: Tensor
    exceed_logit: Tensor
    prior: Tensor
    kappa: Tensor
    teq: Tensor
    lag_weights: Tensor
    pi: Tensor
    # Learned proposal, exposed separately so analyses cannot accidentally call
    # it the safety reference.  Non-ThermoRoute baselines leave this as ``None``.
    internal_prior: Tensor | None = None


class ThermoRoute(nn.Module):
    def __init__(self, n_vars: int, n_stations: int = len(C.STATIONS),
                 horizons=C.HORIZONS, cfg: C.TrainConfig = C.TRAIN,
                 station_agnostic: bool = False, n_phys: int | None = None,
                 use_prior: bool = True, use_router: bool = True,
                 use_moe: bool = True, sparse_router: bool = True,
                 fixed_kappa: bool = False, delta_scale: float | None = 0.4,
                 use_tcn: bool = True, residual_model: bool = True,
                 safety_anchor: str = "damped", use_wlevel: bool = False):
        super().__init__()
        if safety_anchor not in {"internal", "damped", "none"}:
            raise ValueError("safety_anchor must be 'internal', 'damped', or 'none'")
        if delta_scale is not None and delta_scale <= 0:
            raise ValueError("delta_scale must be positive or None for an unbounded model")
        if safety_anchor == "none" and delta_scale is not None:
            raise ValueError("a finite residual bound requires a named safety anchor")
        d = cfg.d_model
        self.horizons = horizons
        self.H = len(horizons)
        self.n_vars = n_vars
        self.use_prior = use_prior
        self.use_router = use_router
        self.use_moe = use_moe
        self.use_tcn = use_tcn
        self.residual_model = residual_model
        self.fixed_kappa = fixed_kappa
        self.use_wlevel = bool(use_wlevel)
        self.safety_anchor = safety_anchor

        from .datasets import PHYS_FORCINGS
        if n_phys is None:
            n_phys = len(PHYS_FORCINGS)
        self.prior = DynamicThermalRelaxationPrior(
            n_phys, n_stations, horizons, station_agnostic,
            use_wlevel=self.use_wlevel,
        )
        if fixed_kappa:
            # Constant-in-time κ control.  Freezing randomly initialised season
            # weights (the old behaviour) did *not* remove seasonality.  Zero
            # every time-varying coefficient before freezing; station offsets
            # remain trainable, so the control changes only κ dynamics.
            with torch.no_grad():
                self.prior.k_flow.zero_()
                self.prior.k_level.zero_()
                self.prior.k_season.weight.zero_()
                self.prior.k_season.bias.zero_()
            self.prior.k_flow.requires_grad_(False)
            self.prior.k_level.requires_grad_(False)
            for p in self.prior.k_season.parameters():
                p.requires_grad_(False)

        gate_dim = 6
        self.router = (DynamicLagRouter(
            n_vars, C.MAX_ROUTER_LAG, self.H, n_stations, d, gate_dim,
            station_agnostic, sparse=sparse_router) if use_router and residual_model else None)
        self.encoder = (CausalTCN(
            n_vars, d, cfg.encoder_blocks, cfg.kernel_size, cfg.dropout)
            if use_tcn and residual_model else None)
        self.moe = (RegimeMoE(d, gate_dim, cfg.n_experts, self.H)
                    if use_moe and residual_model else None)
        self.single_expert = (SingleResidualExpert(d, self.H)
                              if not use_moe and residual_model else None)

        # Missingness is part of the model input rather than silently discarded.
        # Each absent standardised value is replaced by a learned per-variable
        # token before either router or TCN sees it.
        self.missing_token = (nn.Parameter(torch.zeros(n_vars))
                              if residual_model else None)

        self.head_delta = nn.Linear(d, 1) if residual_model else None
        # The RMSE point and probabilistic median are different statistical
        # functionals.  They therefore have distinct parameters.  The q50 head
        # shares upstream representations, but pinball gradients cannot update
        # ``head_delta`` through an accidental q50=point alias.
        self.head_q50 = nn.Linear(d, 1)
        self.head_lo = nn.Linear(d, 1)
        self.head_hi = nn.Linear(d, 1)
        self.head_evt = nn.Linear(d, 1)
        if self.head_delta is not None:
            nn.init.zeros_(self.head_delta.weight)
            nn.init.zeros_(self.head_delta.bias)
        nn.init.zeros_(self.head_q50.weight)
        nn.init.zeros_(self.head_q50.bias)
        self.delta_scale = delta_scale

    def _mask_aware_batch(self, batch) -> dict[str, Tensor]:
        """Return a shallow batch copy whose sequence explicitly encodes gaps."""
        if self.missing_token is None:
            return batch
        observed = batch["Mask"].to(dtype=torch.bool)
        token = self.missing_token.view(1, 1, -1)
        out = dict(batch)
        out["X"] = torch.where(observed, batch["X"], token)
        return out

    def forward(self, batch) -> ThermoRouteOutputs:
        B = batch["X"].shape[0]
        if self.use_prior:
            internal_prior, kappa, teq = self.prior(batch)
        else:
            internal_prior = None
            kappa = torch.full((B,), float("nan"), device=batch["X"].device)
            teq = batch["clim_t"]

        if self.safety_anchor == "damped":
            if "damped_prior" not in batch:
                raise KeyError(
                    "strict safety_anchor='damped' requires batch['damped_prior']; "
                    "build batches through datasets.build_windows")
            anchor = batch["damped_prior"]
        elif self.safety_anchor == "internal":
            if internal_prior is None:
                raise ValueError("internal safety anchor requires use_prior=True")
            anchor = internal_prior
        else:
            anchor = None

        mb = self._mask_aware_batch(batch)
        d = self.head_lo.in_features
        latent = (self.encoder(mb["X"]) if self.encoder is not None
                  else torch.zeros(B, d, device=batch["X"].device,
                                   dtype=batch["X"].dtype))
        if self.router is not None:
            routed, lag_w = self.router(mb)
        else:
            # A true no-router control: no duplicated TCN signal is smuggled into
            # the routed slot.  The unchanged TCN path still reaches the expert.
            routed = torch.zeros(B, self.H, d, device=batch["X"].device,
                                 dtype=batch["X"].dtype)
            lag_w = torch.zeros(B, self.H, self.n_vars, C.MAX_ROUTER_LAG + 1,
                                device=batch["X"].device, dtype=batch["X"].dtype)
        if not self.residual_model:
            rep = torch.zeros_like(routed)
            pi = torch.zeros(B, 0, device=batch["X"].device)
            neural_proposal = torch.zeros(B, self.H, device=batch["X"].device,
                                          dtype=batch["X"].dtype)
        elif self.moe is not None:
            rep, pi = self.moe(routed, latent, batch["gate"])
            assert self.head_delta is not None
            neural_proposal = self.head_delta(rep).squeeze(-1)
        else:
            assert self.single_expert is not None and self.head_delta is not None
            rep = self.single_expert(routed, latent)
            pi = torch.ones(B, 1, device=batch["X"].device,
                            dtype=batch["X"].dtype)
            neural_proposal = self.head_delta(rep).squeeze(-1)

        # This proposal is independently parameterised from the MSE point
        # proposal.  Both may use the same frozen physical anchor, but neither
        # head is derived from or sorted with the other.
        q50_proposal = self.head_q50(rep).squeeze(-1)

        # In strict mode the learned dynamic prior is a proposal only.  Even an
        # arbitrarily drifting internal prior cannot move the final prediction
        # outside the fixed damped anchor's certified band.
        point_proposal = neural_proposal
        if internal_prior is not None:
            if anchor is None:
                point_proposal = point_proposal + internal_prior
                q50_proposal = q50_proposal + internal_prior
            elif anchor is not internal_prior:
                prior_displacement = internal_prior - anchor
                point_proposal = point_proposal + prior_displacement
                q50_proposal = q50_proposal + prior_displacement
        if anchor is None:
            point = point_proposal                # pure-neural, no safety claim
            q50 = q50_proposal
            prior_out = torch.full_like(point, float("nan"))
        else:
            point_correction = (
                point_proposal if self.delta_scale is None else
                self.delta_scale * torch.tanh(point_proposal / self.delta_scale)
            )
            q50_correction = (
                q50_proposal if self.delta_scale is None else
                self.delta_scale * torch.tanh(q50_proposal / self.delta_scale)
            )
            point = anchor + point_correction
            q50 = anchor + q50_correction
            prior_out = anchor
        lo = torch.nn.functional.softplus(self.head_lo(rep)).squeeze(-1)
        hi = torch.nn.functional.softplus(self.head_hi(rep)).squeeze(-1)
        q05 = q50 - lo
        q95 = q50 + hi
        evt = self.head_evt(rep).squeeze(-1)               # [B,H] logit
        return ThermoRouteOutputs(point, q05, q50, q95, evt,
                                  prior_out, kappa, teq, lag_w, pi, internal_prior)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
