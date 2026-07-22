#!/usr/bin/env python3
"""Stage 5 — mechanism analysis of a trained ThermoRoute model.

Trains one ThermoRoute (seed 0, V3, joint), then extracts the interpretable
internals the paper relies on:

* horizon-conditioned variable×lag importance maps (overall, by season, by flow
  regime) from the sparse router;
* the dynamic relaxation rate κ and its dependence on flow / level / season;
* the mixture-of-experts gate occupancy.

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


def season_of(month: np.ndarray) -> np.ndarray:
    lut = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    return np.array([lut[m] for m in month])


def main() -> None:
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

    # run on all non-train splits for a rich, leak-free interpretation sample
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
    )
    print("saved explain.npz", flush=True)

    # quick text summary for the report
    L = ["# Mechanism summary (ThermoRoute, seed 0)\n"]
    L.append(f"- Trained {res.epochs + 1} epochs, params={model.n_params()}, "
             f"val median-RMSE={res.best_val:.4f} °C\n")
    L.append("## Dynamic relaxation rate κ (per-day memory)\n")
    L.append("| station | mean κ | κ low-flow | κ high-flow | implied memory 1/κ (d) |")
    L.append("|---|---|---|---|---|")
    for i, st in enumerate(C.STATIONS):
        sel = station == i
        kl = kappa[sel & (regime == "low")].mean()
        kh = kappa[sel & (regime == "high")].mean()
        km = kappa[sel].mean()
        L.append(f"| {st} | {km:.3f} | {kl:.3f} | {kh:.3f} | {1 / km:.1f} |")
    L.append("\n## Top variable×lag drivers by horizon (router weight share)\n")
    vn = list(C.FEATURE_SETS["V3"])
    for hi, h in enumerate(C.HORIZONS):
        var_imp = overall[hi].sum(axis=1)        # sum over lags -> [V]
        order = np.argsort(var_imp)[::-1][:3]
        top = ", ".join(f"{vn[v]} ({var_imp[v]*100:.0f}%)" for v in order)
        # dominant lag for WTEMP
        wlag = overall[hi, vn.index("WTEMP")].argmax()
        L.append(f"- **h={h}d**: {top}; dominant WTEMP lag = {wlag} d")
    (C.REPORTS / "mechanism_summary.md").write_text("\n".join(L))
    print("wrote mechanism_summary.md", flush=True)


if __name__ == "__main__":
    main()
