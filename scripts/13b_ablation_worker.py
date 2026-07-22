#!/usr/bin/env python3
"""Parallel ablation-seed worker for stage 13.

Trains a subset of the (ablation × seed) grid and writes each result to the
exact checkpoint path scripts/13_rigor.py expects
(``outputs/predictions/rigor_ckpt/{name}_seed{sd}.parquet``). Launch several of
these concurrently (each pinned to a few threads) to use all CPU cores, then run
scripts/13_rigor.py once — it loads every cached fold + ablation seed instantly
and writes rigor.md.

Usage:
  PYTHONPATH=src python3 scripts/13b_ablation_worker.py TR-noDynamicPrior:1 TR-noMoE:1
"""
from __future__ import annotations

import os
# Each worker uses a small thread pool so N workers × threads ≈ physical cores.
os.environ["OMP_NUM_THREADS"] = os.environ.get("WORKER_THREADS", "2")

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

torch.set_num_threads(int(os.environ.get("WORKER_THREADS", "2")))

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
CFG = C.TrainConfig(batch_size=1536)
DELTA = C.DELTA_SCALE   # single source (config.py)
CKPT = C.PREDICTIONS / "rigor_ckpt_route_a_strict_v1"
CKPT.mkdir(exist_ok=True)

ABLS = {
    "ThermoRoute": {},
    "TR-noDynamicPrior": {"use_prior": False},
    "TR-fixedKappa": {"fixed_kappa": True},
    "TR-noRouter": {"use_router": False},
    "TR-noMoE": {"use_moe": False},
    "TR-noTCN": {"use_tcn": False},
    "TR-unbounded": {"delta_scale": None},
    "DampedPriorOnly": {"use_prior": False, "residual_model": False},
    "TR-naturalSampling": {},
}
_t0 = time.time()


def log(m):
    print(f"[{time.time()-_t0:6.0f}s pid{os.getpid()}] {m}", flush=True)


def prep():
    panel_path = Path(os.environ.get(
        "USGS_PANEL", str(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")))
    bundle = D.prepare_dataset_from_panel(str(panel_path))
    panel, pi, masks = bundle["panel_raw"], bundle["panel"], bundle["masks"]
    stations = bundle["stations"]
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    wd = DS.build_windows(pi, masks, clim, variables=USGS_VARS, require_observed_target=True)
    return wd, thr, stations


def train_one(wd, thr, stations, name, sd):
    f = CKPT / f"{name}_seed{sd}.parquet"
    if f.exists():
        log(f"{name} seed{sd}: already done, skip")
        return
    te = time.time()
    model_kw = dict(ABLS[name])
    model_kw.setdefault("delta_scale", DELTA)
    factory = lambda: ThermoRoute(
        n_vars=len(wd.var_names), n_stations=len(stations), n_phys=wd.n_phys,
        safety_anchor="damped", **model_kw)
    balanced = name != "TR-naturalSampling"
    r = fit_model(factory, wd, thr, cfg=CFG, seed=sd, model_name=name,
                  scope="ablation_usgs", feature_set="USGS",
                  station_balanced=balanced,
                  selection_metric="station_macro" if balanced else "micro")
    r.pred["seed"] = sd
    sub = r.pred[r.pred.split == "test"]
    sub.to_parquet(f)
    log(f"{name} seed{sd}: DONE {time.time()-te:.0f}s val={r.best_val:.4f} -> {f.name}")


def main():
    jobs = []
    for arg in sys.argv[1:]:
        name, sd = arg.split(":")
        assert name in ABLS, f"unknown ablation {name}"
        jobs.append((name, int(sd)))
    log(f"worker jobs: {jobs}")
    # skip prep entirely if everything is already checkpointed
    if all((CKPT / f"{n}_seed{s}.parquet").exists() for n, s in jobs):
        log("all assigned jobs already done"); return
    wd, thr, stations = prep()
    log(f"prepared {len(stations)} stations, N={len(wd.X)}")
    for name, sd in jobs:
        train_one(wd, thr, stations, name, sd)
    log("worker complete")


if __name__ == "__main__":
    main()
