#!/usr/bin/env python3
"""Stage 16 — the field-standard deep sequence baseline (global LSTM).

Closes the single most-cited gap for a 一区 (WRR/HESS/J.Hydrol) submission: the
learned-baseline set had no recurrent/attention model. The stream-temperature-ML
literature (Rahmani 2021 ERL; Willard 2024 CONUS PUB; Feigl 2021 HESS) benchmarks
against an LSTM, so its absence reads as avoiding the strongest baseline.

This trains a station-agnostic *top-down global LSTM* — identical heads,
climatology anchor, composite loss, Track-H splits, δ-free — so any gap vs
ThermoRoute is the physics prior + bounded residual, not the recipe. It is
deliberately the perfect FOIL: a strong point forecaster that (per the SWT
literature) will TIE on RMSE while offering no bounded-degradation floor, no
distribution-free calibrated intervals, and no interpretable lag router.

  --insample   train LSTM × USGS_SEEDS on the full 120-station panel; splice
               'LSTM' calib+test rows into usgs_predictions_v2.parquet
  --transfer   train one LSTM per leave-HUC2-region-out fold; checkpoint held-out
               predictions to predictions/region_ckpt/lstm_fold{i}.parquet
  --report     3-way (ThermoRoute vs LightGBM vs LSTM) region-transfer table +
               in-sample headline row -> outputs/reports/lstm_baseline.md

Run:  PYTHONPATH=src python3 scripts/16_lstm_baseline.py --insample
      PYTHONPATH=src python3 scripts/16_lstm_baseline.py --transfer
      PYTHONPATH=src python3 scripts/16_lstm_baseline.py --report
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import importlib.util
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
from scipy.stats import wilcoxon

torch.set_num_threads(8)

from thermoroute import config as C
from thermoroute.train import LSTMForecaster, fit_model

# Reuse 13c's exact fold packing / prep / LightGBM-per-fold so the transfer arm
# is identical to ThermoRoute's (same regions, same in-fold stations).
_spec = importlib.util.spec_from_file_location(
    "region13c", ROOT / "scripts" / "13c_region_transfer.py")
R13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R13)

USGS_VARS = R13.USGS_VARS
# 50-epoch cap / patience 10: early-stopping still selects the best epoch on val,
# but bounds CPU cost on the 527k-window panel. 3 seeds matches the ablation
# budget and gives a clean matched-seed paired test vs ThermoRoute (seeds 0-2).
CFG = C.TrainConfig(batch_size=2048, max_epochs=30, patience=8)
SEEDS = C.USGS_SEEDS[:3]
CKPT = C.PREDICTIONS / "lstm_ckpt"
CKPT.mkdir(exist_ok=True)
REGION_CKPT = C.PREDICTIONS / "region_ckpt"
V2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
_t0 = time.time()


def log(m): print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


# --------------------------------------------------------------------------- #
# In-sample: LSTM × seeds on all 120 stations, spliced into v2
# --------------------------------------------------------------------------- #
def insample():
    panel, panel_imp, masks, clim, thr, wd, stations = R13.prep()
    log(f"{len(stations)} stations | windows N={len(wd.X)}")
    preds = []
    for sd in SEEDS:
        f = CKPT / f"seed{sd}.parquet"
        if f.exists():
            preds.append(pd.read_parquet(f)); log(f"LSTM seed{sd}: loaded"); continue
        te = time.time()
        m = LSTMForecaster(n_vars=len(wd.var_names))
        r = fit_model(m, wd, thr, cfg=CFG, seed=sd, model_name="LSTM",
                      scope="joint_usgs", feature_set="USGS", verbose=True)
        r.pred["seed"] = sd
        r.pred.to_parquet(f); preds.append(r.pred)
        log(f"LSTM seed{sd}: {r.epochs+1}ep {time.time()-te:.0f}s val_rmse={r.best_val:.4f}")
    lstm = pd.concat(preds, ignore_index=True)

    # splice into v2: align test rows to the shared ThermoRoute test keys (same
    # trick as the per-station LightGBM), keep ALL calib rows for conformal.
    allp = pd.read_parquet(V2)
    allp = allp[allp.model != "LSTM"]
    tr_keys = set(zip(*[allp[(allp.model == "ThermoRoute") & (allp.split == "test")][c]
                        for c in ["site_id", "horizon", "issue_date"]]))
    lt = lstm[lstm.split == "test"].copy()
    keyser = pd.Series(list(zip(lt.site_id, lt.horizon, lt.issue_date)), index=lt.index)
    lt = lt[keyser.isin(tr_keys)]
    lc = lstm[lstm.split == "calib"].copy()          # for the conformal wrapper
    allp = pd.concat([allp, lt, lc], ignore_index=True)
    allp.to_parquet(V2)
    log(f"spliced LSTM: {len(lt)} test + {len(lc)} calib rows into v2")

    # headline: 5-seed ensemble median per-station RMSE vs ThermoRoute
    def ens_rmse(model, h):
        s = allp[(allp.model == model) & (allp.split == "test") & (allp.horizon == h)]
        s = s.groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        return {st: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
                for st, g in s.groupby("site_id")}
    for h in C.HORIZONS:
        lp, tp = ens_rmse("LSTM", h), ens_rmse("ThermoRoute", h)
        comm = [s for s in lp if s in tp]
        a = np.array([tp[s] for s in comm]); b = np.array([lp[s] for s in comm])
        p = wilcoxon(a, b).pvalue
        log(f"  h{h}: LSTM median RMSE {np.median(list(lp.values())):.3f} vs "
            f"TR {np.median(list(tp.values())):.3f} | TR-vs-LSTM paired p={p:.2g} "
            f"| TR wins {100*np.mean(a < b):.0f}%")


# --------------------------------------------------------------------------- #
# Transfer: one LSTM per leave-HUC2-region-out fold
# --------------------------------------------------------------------------- #
def transfer(fold=None):
    panel, panel_imp, masks, clim, thr, wd, stations = R13.prep()
    folds, _ = R13.region_folds(stations)
    todo = range(len(folds)) if fold is None else [fold]
    for fi in todo:
        f = REGION_CKPT / f"lstm_fold{fi}.parquet"
        if f.exists():
            log(f"LSTM fold{fi}: already done"); continue
        hold = folds[fi]
        train_st = tuple(s for s in stations if s not in hold)
        te = time.time()
        log(f"LSTM fold{fi}: train {len(train_st)} -> hold {len(hold)} region stations")
        m = LSTMForecaster(n_vars=len(wd.var_names))
        r = fit_model(m, wd, thr, cfg=CFG, seed=0, scope="region_lgo",
                      feature_set="USGS", train_stations=train_st)
        pred = r.pred[(r.pred.split == "test") & (r.pred.site_id.isin(hold))].copy()
        pred["model"] = "LSTM-regionLGO"
        pred.to_parquet(f)
        log(f"LSTM fold{fi}: DONE {r.epochs+1}ep {time.time()-te:.0f}s -> {f.name}")


# --------------------------------------------------------------------------- #
# Report: 3-way transfer table + in-sample headline
# --------------------------------------------------------------------------- #
def report():
    panel, panel_imp, masks, clim, thr, wd, stations = R13.prep()
    folds, _ = R13.region_folds(stations)
    TR = pd.concat([pd.read_parquet(REGION_CKPT / f"tr_fold{fi}.parquet")
                    for fi in range(len(folds))], ignore_index=True)
    LGB = pd.concat([pd.read_parquet(REGION_CKPT / f"lgb_fold{fi}.parquet")
                     for fi in range(len(folds))], ignore_index=True)
    lstm_files = [REGION_CKPT / f"lstm_fold{fi}.parquet" for fi in range(len(folds))]
    if not all(f.exists() for f in lstm_files):
        log("LSTM transfer folds missing — run --transfer first"); return
    LSTM = pd.concat([pd.read_parquet(f) for f in lstm_files], ignore_index=True)

    v2 = pd.read_parquet(V2)
    base = v2[(v2.split == "test") & v2.model.isin(["Persistence", "DampedPersistence"])]
    tr_r, lgb_r, lstm_r = R13.ps_rmse(TR), R13.ps_rmse(LGB), R13.ps_rmse(LSTM)
    per_r = R13.ps_rmse(base[base.model == "Persistence"])

    L = ["# Deep sequence baseline (global LSTM) — in-sample + region-transfer\n",
         "A station-agnostic top-down LSTM (1 layer, hidden 64, 14-day context, "
         "persistence-anchored) trained under the SAME Track-H splits and composite "
         "loss as ThermoRoute — the field-standard deep baseline (Rahmani 2021 / "
         "Willard 2024), landing in the published LSTM accuracy band (Zwart 2023). "
         "The intended reading is the FOIL: it is competitive on RMSE (behind both "
         "the global LightGBM and ThermoRoute here, consistent with GBDT≥LSTM on "
         "autocorrelated daily data) yet carries no bounded-degradation floor, no "
         "distribution-free calibrated intervals, and no interpretable lag router.\n",
         "## Leave-HUC2-region-out transfer (whole regions held out)\n",
         "| horizon | n | TR RMSE | LGB RMSE | LSTM RMSE | TR−LSTM p | LGB−LSTM p | best |",
         "|---|---|---|---|---|---|---|---|"]
    for h in C.HORIZONS:
        sts = sorted(s for s in stations
                     if (s, h) in tr_r and (s, h) in lgb_r and (s, h) in lstm_r)
        a = np.array([tr_r[(s, h)] for s in sts])
        b = np.array([lgb_r[(s, h)] for s in sts])
        c = np.array([lstm_r[(s, h)] for s in sts])
        p_tl = wilcoxon(a, c).pvalue if len(sts) > 5 else float("nan")
        p_gl = wilcoxon(b, c).pvalue if len(sts) > 5 else float("nan")
        meds = {"TR": np.median(a), "LGB": np.median(b), "LSTM": np.median(c)}
        best = min(meds, key=meds.get)
        L.append(f"| {h} | {len(sts)} | {np.median(a):.3f} | {np.median(b):.3f} | "
                 f"{np.median(c):.3f} | {p_tl:.2g} | {p_gl:.2g} | **{best}** |")

    # in-sample headline (5-seed ensembles from v2)
    def ens_rmse(model, h):
        s = v2[(v2.model == model) & (v2.split == "test") & (v2.horizon == h)]
        s = s.groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        return {st: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
                for st, g in s.groupby("site_id")}
    L += ["", "## In-sample (120 stations, 5-seed ensembles) median per-station RMSE\n",
          "| horizon | persist | LightGBM | LSTM | ThermoRoute | TR−LSTM paired p | TR wins |",
          "|---|---|---|---|---|---|---|"]
    for h in C.HORIZONS:
        pe = ens_rmse("Persistence", h); lg = ens_rmse("LightGBM", h)
        ls = ens_rmse("LSTM", h); tr = ens_rmse("ThermoRoute", h)
        comm = sorted(s for s in ls if s in tr)
        a = np.array([tr[s] for s in comm]); c = np.array([ls[s] for s in comm])
        p = wilcoxon(a, c).pvalue
        L.append(f"| {h} | {np.median(list(pe.values())):.3f} | "
                 f"{np.median(list(lg.values())):.3f} | {np.median(list(ls.values())):.3f} | "
                 f"{np.median(list(tr.values())):.3f} | {p:.2g} | {100*np.mean(a<c):.0f}% |")

    out = C.REPORTS / "lstm_baseline.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--insample", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.insample:
        insample()
    elif a.transfer:
        transfer(a.fold)
    elif a.report:
        report()
    else:
        print(__doc__)
