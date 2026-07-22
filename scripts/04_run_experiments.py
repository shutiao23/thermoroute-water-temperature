#!/usr/bin/env python3
"""Stage 4 — the full experiment matrix.

Runs every model, applies conformal calibration per run, saves all per-day
predictions and a tidy scores table.  Designed to be re-runnable and to
checkpoint predictions so a long run is never lost.

Matrix
------
* Baselines (per-station): persistence, damped persistence, climatology, ridge,
  air2stream-lite, LightGBM (V1/V2/V3, with quantiles + exceedance).
* Deep, joint 3-station: GRU (V3) and ThermoRoute (V3), multiple seeds.
* ThermoRoute feature ladder: V1, V2, V3.
* ThermoRoute leave-one-station-out warm-start diagnostic (not zero-shot).
* ThermoRoute module ablations: no-prior, fixed-κ, softmax router.

Run:  PYTHONPATH=src python3 scripts/04_run_experiments.py
"""
from __future__ import annotations

import os
# Must be set before torch / lightgbm import: avoids an OpenMP duplicate-runtime
# crash when both libraries are loaded in one process (macOS/anaconda).
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute import baselines as B
from thermoroute import results as R
from thermoroute.conformal import cqr_offsets, apply_cqr
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model, GRUForecaster

DEEP_SEEDS = (0, 1, 2)
PRED_PATH = C.PREDICTIONS / "predictions.parquet"
_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _t0:6.0f}s] {msg}", flush=True)


def calibrate(pred: pd.DataFrame) -> pd.DataFrame:
    """Apply CQR using the run's own calibration split (only if quantiles exist)."""
    cal = pred[pred.split == "calib"]
    if cal["q05"].notna().all() and len(cal) > 0:
        return apply_cqr(pred, cqr_offsets(cal))
    return pred


def main() -> None:
    C.ensure_output_directories()
    bundle = D.prepare_dataset()
    panel, masks = bundle["panel"], bundle["masks"]
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    clim_air = F.HarmonicClimatology.fit(panel, masks.train, target="TEMP")
    thr = R.exceedance_thresholds(panel, masks)
    log(f"data ready; thresholds={ {k: round(v, 1) for k, v in thr.items()} }")

    chunks: list[pd.DataFrame] = []

    # ---- baselines -------------------------------------------------------- #
    tabs_v3 = B._tab_by_horizon(panel, clim, C.FEATURE_SETS["V3"])
    chunks.append(B.run_persistence(tabs_v3))
    chunks.append(B.run_climatology(tabs_v3))
    dp, phi = B.run_damped_persistence(panel, masks, tabs_v3, clim)
    chunks.append(dp)
    chunks.append(B.run_ridge(tabs_v3))
    chunks.append(B.run_air2stream(panel, masks, clim_air))
    log("trivial + ridge + air2stream done")
    for fs in ("V1", "V2", "V3"):
        tabs = tabs_v3 if fs == "V3" else B._tab_by_horizon(panel, clim, C.FEATURE_SETS[fs])
        lg = calibrate(B.run_lightgbm(tabs, thr, feature_set=fs))
        chunks.append(lg)
        log(f"LightGBM {fs} done")
    R.write_predictions(pd.concat(chunks, ignore_index=True), PRED_PATH)

    # ---- deep models (joint) --------------------------------------------- #
    windows = {fs: DS.build_windows(panel, masks, clim, variables=C.FEATURE_SETS[fs])
               for fs in ("V1", "V2", "V3")}
    nvars = {fs: len(windows[fs].var_names) for fs in windows}
    log(f"windows built V1/V2/V3 (N={len(windows['V3'].X)})")

    def run_deep(make_model, name, fs, seeds, scope="joint", **fit_kw):
        wd = windows[fs]
        for sd in seeds:
            te = time.time()
            # The training loop sets the seed before invoking this factory, so
            # parameter initialisation is part of the declared seed contract.
            res = fit_model(lambda: make_model(wd), wd, thr, seed=sd, model_name=name,
                            scope=scope, feature_set=fs, **fit_kw)
            chunks.append(calibrate(res.pred))
            log(f"{name} {scope} {fs} seed{sd}: {res.epochs + 1}ep "
                f"{time.time() - te:.0f}s val_rmse={res.best_val:.4f}")
            R.write_predictions(pd.concat(chunks, ignore_index=True), PRED_PATH)

    # GRU reference
    run_deep(lambda wd: GRUForecaster(nvars["V3"]), "GRU", "V3", DEEP_SEEDS[:2])

    # ThermoRoute main + feature ladder
    run_deep(lambda wd: ThermoRoute(
        n_vars=nvars["V3"], n_phys=wd.n_phys, delta_scale=C.DELTA_SCALE,
        safety_anchor="damped"), "ThermoRoute", "V3", DEEP_SEEDS)
    run_deep(lambda wd: ThermoRoute(
        n_vars=nvars["V1"], n_phys=windows["V1"].n_phys,
        delta_scale=C.DELTA_SCALE, safety_anchor="damped"),
             "ThermoRoute", "V1", (0,))
    run_deep(lambda wd: ThermoRoute(
        n_vars=nvars["V2"], n_phys=windows["V2"].n_phys,
        delta_scale=C.DELTA_SCALE, safety_anchor="damped"),
             "ThermoRoute", "V2", (0,))

    # ---- LOSO warm-start diagnostic -------------------------------------- #
    wd = windows["V3"]
    for held in C.STATIONS:
        train_st = tuple(s for s in C.STATIONS if s != held)
        te = time.time()
        res = fit_model(
            lambda: ThermoRoute(
                n_vars=nvars["V3"], n_phys=wd.n_phys, station_agnostic=True,
                delta_scale=C.DELTA_SCALE, safety_anchor="damped"),
            wd, thr, seed=0, model_name="ThermoRoute-LOSO-WarmStart",
            scope=f"loso_warm_start_{held}", feature_set="V3",
            train_stations=train_st,
        )
        sub = res.pred[res.pred.site_id == held].copy()
        chunks.append(calibrate(sub))
        log(f"LOSO warm-start hold {held}: {time.time() - te:.0f}s")
        R.write_predictions(pd.concat(chunks, ignore_index=True), PRED_PATH)

    # ---- module ablations (V3 joint, single seed) ------------------------ #
    ablations = {
        "TR-noPrior": dict(use_prior=False),
        "TR-fixedKappa": dict(fixed_kappa=True),
        "TR-softmax": dict(sparse_router=False),
        "TR-noMoE": dict(use_moe=False),
        "TR-noRouter": dict(use_router=False),
    }
    for name, kw in ablations.items():
        run_deep(lambda wd, kw=kw: ThermoRoute(
            n_vars=nvars["V3"], n_phys=wd.n_phys,
            delta_scale=C.DELTA_SCALE, safety_anchor="damped", **kw),
                 name, "V3", (0,), scope="ablation")

    allp = pd.concat(chunks, ignore_index=True)
    R.write_predictions(allp, PRED_PATH)
    log(f"ALL DONE: {len(allp)} prediction rows -> {PRED_PATH}")

    # ---- score and save tables ------------------------------------------- #
    scores = R.evaluate(allp, thr, splits=("test", "calib", "val"))
    scores.to_csv(C.TABLES / "scores_all.csv", index=False)
    log(f"scores -> {C.TABLES / 'scores_all.csv'} ({len(scores)} rows)")


if __name__ == "__main__":
    main()
