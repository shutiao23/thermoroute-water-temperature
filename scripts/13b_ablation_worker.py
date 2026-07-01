#!/usr/bin/env python3
"""Parallel ablation-seed worker for stage 13.

Trains a subset of the (ablation × seed) grid and writes each result to the
exact checkpoint path scripts/13_rigor.py expects
(``outputs/predictions/rigor_ckpt/{name}_seed{sd}.parquet``). Launch several of
these concurrently (each pinned to a few threads) to use all CPU cores, then run
scripts/13_rigor.py once — it loads every cached fold + ablation seed instantly
and writes rigor.md.

Usage:
  PYTHONPATH=src python3 scripts/13b_ablation_worker.py TR-noPrior:1 TR-noMoE:1
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Each worker uses a small thread pool so N workers × threads ≈ physical cores.
os.environ["OMP_NUM_THREADS"] = os.environ.get("WORKER_THREADS", "2")

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
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
DELTA = 1.5
CKPT = C.PREDICTIONS / "rigor_ckpt"
CKPT.mkdir(exist_ok=True)

ABLS = {"TR-noPrior": dict(use_prior=False), "TR-fixedKappa": dict(fixed_kappa=True),
        "TR-noRouter": dict(use_router=False), "TR-noMoE": dict(use_moe=False)}
_t0 = time.time()


def log(m):
    print(f"[{time.time()-_t0:6.0f}s pid{os.getpid()}] {m}", flush=True)


def prep():
    _p100 = ROOT / "data_usgs" / "panel_usgs_100.parquet"
    _pwind = ROOT / "data_usgs" / "panel_usgs_wind.parquet"
    panel_path = _p100 if _p100.exists() else _pwind
    panel = pd.read_parquet(panel_path)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    stations = tuple(sorted(panel.site_id.unique()))
    C.STATIONS = stations
    C.UPSTREAM = {s: None for s in stations}
    for v in C.ALL_VARS:
        panel[f"{v}_observed"] = panel[v].notna()
    masks = D.split_masks(panel["DATE"])
    pi = D.Imputer.fit(panel, masks.train).transform(panel)
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
    m = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                    n_phys=wd.n_phys, delta_scale=DELTA, **ABLS[name])
    r = fit_model(m, wd, thr, cfg=CFG, seed=sd, model_name=name,
                  scope="ablation_usgs", feature_set="USGS")
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
