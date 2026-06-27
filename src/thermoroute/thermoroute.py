"""The ThermoRoute model.

Forecast = dynamic thermal-relaxation **physics prior** (a learnable, flow- and
season-modulated generalisation of damped persistence) + a **neural residual**
read out from a horizon-conditioned *sparse* variable–lag router, a causal TCN
encoder and a regime mixture-of-experts.  Outputs are monotone quantiles plus a
high-temperature exceedance probability.

The prior makes the strong baseline (damped persistence) a special case, so the
network only has to learn what the physics leaves unexplained — which is exactly
the scientific claim the paper tests.
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
    so the strong baseline is a special case; letting κ depend on FLOW, WLEVEL and
    season is the dynamic-thermal-memory hypothesis the paper tests.
    """

    def __init__(self, n_phys: int, n_stations: int, horizons, station_agnostic: bool):
        super().__init__()
        self.horizons = torch.tensor(list(horizons), dtype=torch.float32)
        self.station_agnostic = station_agnostic
        self.eq_lin = nn.Linear(n_phys, 1)               # weather → equilibrium anomaly
        self.eq_station = nn.Embedding(n_stations, 1)
        # κ logit components
        self.k_station = nn.Embedding(n_stations, 1)
        self.k_flow = nn.Parameter(torch.zeros(1))
        self.k_level = nn.Parameter(torch.zeros(1))
        self.k_season = nn.Linear(2, 1)
        # warm-start κ≈0.05 ⇒ the prior begins essentially at damped persistence
        self.k_bias = nn.Parameter(torch.tensor(-2.94))
        nn.init.zeros_(self.eq_lin.weight); nn.init.zeros_(self.eq_lin.bias)
        nn.init.zeros_(self.eq_station.weight); nn.init.zeros_(self.k_station.weight)

    def forward(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        st = batch["station"]
        s_eq = 0.0 if self.station_agnostic else self.eq_station(st).squeeze(-1)
        s_k = 0.0 if self.station_agnostic else self.k_station(st).squeeze(-1)
        e_t = self.eq_lin(batch["phys_std"]).squeeze(-1) + s_eq      # [B] eq. anomaly
        a_t = batch["wtemp_t"] - batch["clim_t"]                     # [B] today anomaly
        k_logit = (self.k_bias + s_k
                   + self.k_flow * batch["logflowz"]
                   + self.k_level * batch["wlevelz"]
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


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
@dataclass
class ThermoRouteOutputs:
    median: Tensor
    q05: Tensor
    q50: Tensor
    q95: Tensor
    exceed_logit: Tensor
    prior: Tensor
    kappa: Tensor
    teq: Tensor
    lag_weights: Tensor
    pi: Tensor


class ThermoRoute(nn.Module):
    def __init__(self, n_vars: int, n_stations: int = len(C.STATIONS),
                 horizons=C.HORIZONS, cfg: C.TrainConfig = C.TRAIN,
                 station_agnostic: bool = False, n_phys: int | None = None,
                 use_prior: bool = True, use_router: bool = True,
                 use_moe: bool = True, sparse_router: bool = True,
                 fixed_kappa: bool = False, delta_scale: float = 0.4):
        super().__init__()
        d = cfg.d_model
        self.horizons = horizons
        self.H = len(horizons)
        self.use_prior = use_prior
        self.use_router = use_router
        self.use_moe = use_moe
        self.fixed_kappa = fixed_kappa

        from .datasets import PHYS_FORCINGS
        if n_phys is None:
            n_phys = len(PHYS_FORCINGS)
        self.prior = DynamicThermalRelaxationPrior(
            n_phys, n_stations, horizons, station_agnostic)
        if fixed_kappa:   # ablation: freeze κ's dynamic modulators
            self.prior.k_flow.requires_grad_(False)
            self.prior.k_level.requires_grad_(False)
            for p in self.prior.k_season.parameters():
                p.requires_grad_(False)

        gate_dim = 6
        self.router = DynamicLagRouter(n_vars, C.MAX_ROUTER_LAG, self.H, n_stations,
                                       d, gate_dim, station_agnostic, sparse=sparse_router)
        self.encoder = CausalTCN(n_vars, d, cfg.encoder_blocks, cfg.kernel_size, cfg.dropout)
        self.moe = RegimeMoE(d, gate_dim, cfg.n_experts, self.H)

        self.head_delta = nn.Linear(d, 1)
        self.head_lo = nn.Linear(d, 1)
        self.head_hi = nn.Linear(d, 1)
        self.head_evt = nn.Linear(d, 1)
        nn.init.zeros_(self.head_delta.weight); nn.init.zeros_(self.head_delta.bias)
        # The neural residual is bounded to ±delta_scale °C around the physics
        # prior. On the small, strongly-damped 3-station data the prior is the
        # ceiling, so a tight bound (0.4) keeps ThermoRoute stable; on the large
        # sample with real headroom a looser bound lets the residual add skill at
        # 3–7 days (selected in scripts/11_retune.py).
        self.delta_scale = delta_scale

    def forward(self, batch) -> ThermoRouteOutputs:
        B = batch["X"].shape[0]
        if self.use_prior:
            prior, kappa, teq = self.prior(batch)
        else:                                   # ablation: no physics prior
            prior = batch["clim_tgt"]
            kappa = torch.full((B,), float("nan"), device=batch["X"].device)
            teq = batch["clim_t"]

        latent = self.encoder(batch["X"])
        if self.use_router:
            routed, lag_w = self.router(batch)
        else:
            routed = latent[:, None, :].expand(B, self.H, latent.shape[-1])
            lag_w = torch.zeros(B, self.H, self.router.V, self.router.Lr1,
                                device=batch["X"].device)
        if self.use_moe:
            rep, pi = self.moe(routed, latent, batch["gate"])
        else:
            rep, pi = routed, torch.zeros(B, 1, device=batch["X"].device)

        delta = self.delta_scale * torch.tanh(self.head_delta(rep).squeeze(-1))  # [B,H]
        median = prior + delta
        lo = torch.nn.functional.softplus(self.head_lo(rep)).squeeze(-1)
        hi = torch.nn.functional.softplus(self.head_hi(rep)).squeeze(-1)
        q05 = median - lo
        q95 = median + hi
        evt = self.head_evt(rep).squeeze(-1)               # [B,H] logit
        return ThermoRouteOutputs(median, q05, median, q95, evt,
                                  prior, kappa, teq, lag_w, pi)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
