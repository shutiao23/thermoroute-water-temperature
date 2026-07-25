#!/usr/bin/env python3
"""Stage 5 — exploratory latent-component diagnostics for ThermoRoute.

Trains one ThermoRoute (seed 0, V3, joint), then extracts the inspectable
internal allocation summaries used only for software and sensitivity checks:

* horizon-conditioned variable×lag allocation maps (overall, by season, by flow
  stratum) from the sparse router;
* the learned decay coefficient κ and its descriptive association with flow
  stratum / level / season;
* the mixture-of-experts gate occupancy.

The legacy identifiers b1, s2, and p3 are ordinary monitoring stations.  This
script encodes no river network, reservoir cascade, hydraulic ordering, or travel
time.  Its learned coefficients and allocation weights are descriptive latent
quantities, not physical parameters, feature importance, or causal effects.

All arrays are saved to ``outputs/tables/explain.npz`` for the figure stage, and
the model state to ``outputs/models/thermoroute_explain.pt``.

Run:  PYTHONPATH=src python3 scripts/05_explain.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute import results as R
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

from _legacy_site_semantics import (
    EPISTEMIC_TOPOLOGY_SENTENCE,
    ORDINARY_MONITORING_SENTENCE,
    find_legacy_semantic_violations,
    load_legacy_semantic_policy,
    retire_legacy_report_output,
)


def season_of(month: np.ndarray) -> np.ndarray:
    lut = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    return np.array([lut[m] for m in month])


def main() -> None:
    # Retire the misleading old filename before any model/data operation, so a
    # failed or partial rerun cannot leave a stale physical-mechanism claim.
    legacy_output = retire_legacy_report_output(C.REPORTS)
    semantic_policy = load_legacy_semantic_policy(ROOT)
    bundle = D.prepare_dataset()
    panel, masks = bundle["panel"], bundle["masks"]
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = R.exceedance_thresholds(panel, masks)
    wd = DS.build_windows(panel, masks, clim, variables=C.FEATURE_SETS["V3"])

    factory = lambda: ThermoRoute(
        n_vars=len(wd.var_names), n_stations=len(C.STATIONS), n_phys=wd.n_phys,
        delta_scale=C.DELTA_SCALE, safety_anchor="damped",
    )
    res = fit_model(factory, wd, thr, seed=0, model_name="ThermoRoute", verbose=True)
    model = res.model
    torch.save(model.state_dict(), C.MODELS / "thermoroute_explain.pt")
    print("trained explain model:", res.epochs + 1, "epochs, val_rmse",
          round(res.best_val, 4), flush=True)

    # Run on non-training splits as a development-only diagnostic sample.
    idx = np.concatenate([wd.idx("val"), wd.idx("calib"), wd.idx("test")])
    model.eval()
    with torch.no_grad():
        out = model(wd.batch(idx))
    lag_w = out.lag_weights.numpy()        # [N,H,V,Lr1]
    kappa = out.kappa.numpy()              # [N]
    teq = out.teq.numpy()
    pi = out.pi.numpy()                    # [N,K]
    N, H, V, Lr1 = lag_w.shape

    months = pd.to_datetime(wd.issue_date[idx]).month.to_numpy()
    seasons = season_of(months)
    station = wd.station[idx]
    logflowz = wd.logflowz[idx]
    wlevelz = wd.wlevelz[idx]

    # overall and stratified lag maps
    overall = lag_w.mean(axis=0)           # [H,V,Lr1]
    by_season = {s: lag_w[seasons == s].mean(axis=0) for s in ("DJF", "MAM", "JJA", "SON")}
    # flow regime tertiles
    q1, q2 = np.quantile(logflowz, [1 / 3, 2 / 3])
    regime = np.where(logflowz <= q1, "low", np.where(logflowz <= q2, "mid", "high"))
    by_flow = {r: lag_w[regime == r].mean(axis=0) for r in ("low", "mid", "high")}

    np.savez(
        C.TABLES / "explain.npz",
        var_names=np.array(C.FEATURE_SETS["V3"]),
        horizons=np.array(C.HORIZONS), max_lag=C.MAX_ROUTER_LAG,
        overall=overall,
        season_keys=np.array(["DJF", "MAM", "JJA", "SON"]),
        season_maps=np.stack([by_season[s] for s in ("DJF", "MAM", "JJA", "SON")]),
        flow_keys=np.array(["low", "mid", "high"]),
        flow_maps=np.stack([by_flow[r] for r in ("low", "mid", "high")]),
        kappa=kappa, logflowz=logflowz, wlevelz=wlevelz, teq=teq,
        station=station, months=months, pi=pi,
        semantic_contract_version=np.array(
            "thermoroute.legacy-monitoring-latent-diagnostics.v1"
        ),
        site_ids=np.array(C.STATIONS),
        site_classification=np.array(
            "ORDINARY_MONITORING_STATIONS_NOT_RESERVOIRS"
        ),
        verified_network_metadata=np.array(False),
        topology_inference_allowed=np.array(False),
        physical_interpretation_allowed=np.array(False),
        causal_interpretation_allowed=np.array(False),
        analysis_role=np.array("SINGLE_SEED_DESCRIPTIVE_DIAGNOSTIC"),
        diagnostic_seed=np.array(0, dtype=np.int64),
    )
    print("saved explain.npz", flush=True)

    # Quick exploratory summary.  Keep the interpretation boundary inside the
    # generated artifact so it cannot be separated from the diagnostic numbers.
    L = ["# Latent-component diagnostic summary (ThermoRoute, seed 0)\n"]
    L.append(
        f"**Interpretation boundary.** {ORDINARY_MONITORING_SENTENCE} "
        f"{EPISTEMIC_TOPOLOGY_SENTENCE} This exploratory single-seed diagnostic "
        "does not identify physical parameters, feature importance, or causal "
        "effects.\n"
    )
    L.append(f"- Trained {res.epochs + 1} epochs, params={model.n_params()}, "
             f"val median-RMSE={res.best_val:.4f} °C\n")
    L.append("## Learned decay coefficient κ (latent diagnostic)\n")
    L.append("| station | mean κ | κ low-flow stratum | κ high-flow stratum |")
    L.append("|---|---|---|---|")
    for i, st in enumerate(C.STATIONS):
        sel = station == i
        kl = kappa[sel & (regime == "low")].mean()
        kh = kappa[sel & (regime == "high")].mean()
        km = kappa[sel].mean()
        L.append(f"| {st} | {km:.3f} | {kl:.3f} | {kh:.3f} |")
    L.append("\n## Largest variable×lag allocations by horizon (router weight share)\n")
    vn = list(C.FEATURE_SETS["V3"])
    for hi, h in enumerate(C.HORIZONS):
        var_imp = overall[hi].sum(axis=1)        # sum over lags -> [V]
        order = np.argsort(var_imp)[::-1][:3]
        top = ", ".join(f"{vn[v]} ({var_imp[v]*100:.0f}%)" for v in order)
        # dominant lag for WTEMP
        wlag = overall[hi, vn.index("WTEMP")].argmax()
        L.append(
            f"- **h={h}d**: {top}; largest allocated WTEMP lag index = {wlag} "
            "(diagnostic index, not propagation time)"
        )
    output = C.REPORTS / "latent_component_diagnostics.md"
    report_text = "\n".join(L)
    violations = find_legacy_semantic_violations(report_text, semantic_policy)
    if violations:
        details = "; ".join(
            f"{violation.lint_id}: {violation.excerpt}"
            for violation in violations
        )
        raise RuntimeError(f"legacy semantic guard refused diagnostic report: {details}")
    output.write_text(report_text, encoding="utf-8")
    print(f"wrote {output.name} and withdrawal tombstone {legacy_output.name}", flush=True)


if __name__ == "__main__":
    main()
