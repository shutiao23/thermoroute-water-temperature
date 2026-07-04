#!/usr/bin/env python3
"""Leave-HUC2-region-out transfer, benchmarked vs the strong learned baseline.

Fixes the two blockers of the strong-accept review:
  B1 — transfer was never benchmarked vs LightGBM (only persistence/damped).
  B2 — "unseen basins" was a RANDOM station split; here whole HUC2 regions are
       held out so no gage on a held-out river/region is in training.

Design: 16 HUC2 regions greedily packed into 4 folds (~30 stations each), each
fold holds out whole regions. Under this protocol we train, per fold, a
station-agnostic ThermoRoute AND a global LightGBM on the in-fold regions and
forecast the held-out region stations. Persistence / damped persistence are
training-free so their per-station test RMSE is read from the v2 predictions.
Then a per-station paired Wilcoxon (TR vs LightGBM) at each horizon decides the
central claim, plus a skill-vs-distance-to-nearest-training-gage gradient.

Usage:
  PYTHONPATH=src python3 scripts/13c_region_transfer.py --fold N   # train fold N's TR
  PYTHONPATH=src python3 scripts/13c_region_transfer.py --assemble # LGB + stats + report
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("WORKER_THREADS", "8"))

import argparse
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

torch.set_num_threads(int(os.environ.get("WORKER_THREADS", "8")))

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute import baselines as B
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
CFG = C.TrainConfig(batch_size=1536)
DELTA = 1.0                                   # val-selected, matches 09/13
PANEL = ROOT / "data_usgs" / "panel_usgs_100.parquet"
HUC = ROOT / "outputs" / "tables" / "usgs_stations_with_huc.csv"
CKPT = C.PREDICTIONS / "region_ckpt"
CKPT.mkdir(exist_ok=True)
_t0 = time.time()


def log(m): print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


def region_folds(stations):
    """Greedy pack whole HUC2 regions into 4 balanced folds (deterministic)."""
    huc = pd.read_csv(HUC).set_index("site_id")["huc2"].to_dict()
    by_reg = {}
    for s in stations:
        by_reg.setdefault(int(huc.get(s, -1)), []).append(s)
    regions = sorted(by_reg.items(), key=lambda kv: -len(kv[1]))
    folds, load = [[], [], [], []], [0, 0, 0, 0]
    reg_of_fold = [[], [], [], []]
    for reg, sts in regions:
        i = int(np.argmin(load))
        folds[i] += sts; load[i] += len(sts); reg_of_fold[i].append(reg)
    return [set(f) for f in folds], reg_of_fold


def prep():
    b = D.prepare_dataset_from_panel(str(PANEL))
    panel, panel_imp, masks = b["panel_raw"], b["panel"], b["masks"]
    stations = b["stations"]
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    wd = DS.build_windows(panel_imp, masks, clim, variables=USGS_VARS,
                          require_observed_target=True)
    return panel, panel_imp, masks, clim, thr, wd, stations


def train_fold_tr(fold_i):
    panel, panel_imp, masks, clim, thr, wd, stations = prep()
    folds, _ = region_folds(stations)
    hold = folds[fold_i]
    f = CKPT / f"tr_fold{fold_i}.parquet"
    if f.exists():
        log(f"TR fold{fold_i}: already done"); return
    train_st = tuple(s for s in stations if s not in hold)
    log(f"TR fold{fold_i}: train {len(train_st)} -> hold {len(hold)} region stations")
    m = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                    n_phys=wd.n_phys, station_agnostic=True, delta_scale=DELTA)
    r = fit_model(m, wd, thr, cfg=CFG, seed=0, scope="region_lgo",
                  feature_set="USGS", train_stations=train_st)
    pred = r.pred[(r.pred.split == "test") & (r.pred.site_id.isin(hold))].copy()
    pred["model"] = "ThermoRoute-regionLGO"
    pred.to_parquet(f)
    log(f"TR fold{fold_i}: DONE {r.epochs+1}ep val={r.best_val:.4f} -> {f.name}")


def lgb_fold(panel_imp, clim, train_st, hold):
    """Global LightGBM trained on in-fold stations, predicting held-out stations."""
    frames = []
    for h in C.HORIZONS:
        tab = F.attach_split(F.build_tabular(panel_imp, h, USGS_VARS, clim,
                             drop_feature_nans=False, require_observed_target=True))
        cols = F.feature_columns(tab)
        for c in cols:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
        tr = tab[(tab.split == "train") & (tab.site_id.isin(train_st))]
        va = tab[(tab.split == "val") & (tab.site_id.isin(train_st))]
        ev = tab[(tab.split == "test") & (tab.site_id.isin(hold))]
        mp = B._lgb_fit(tr[cols].to_numpy(float), tr["y"].to_numpy(float),
                        va[cols].to_numpy(float), va["y"].to_numpy(float), "regression")
        frames.append(pd.DataFrame({
            "model": "LightGBM-regionLGO", "site_id": ev["site_id"].to_numpy(),
            "horizon": h, "issue_date": ev["issue_date"].to_numpy(),
            "y_true": ev["y"].to_numpy(float), "y_pred": mp.predict(ev[cols].to_numpy(float))}))
    return pd.concat(frames, ignore_index=True)


def ps_rmse(df, model=None):
    d = df if model is None else df[df.model == model]
    return {(s, h): float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
            for (s, h), g in d.groupby(["site_id", "horizon"])}


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    d = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(d))


def assemble():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panel, panel_imp, masks, clim, thr, wd, stations = prep()
    folds, reg_of_fold = region_folds(stations)
    meta = pd.read_csv(HUC).set_index("site_id")

    # LightGBM per fold + collect TR fold predictions
    tr_all, lgb_all = [], []
    for fi, hold in enumerate(folds):
        tr_all.append(pd.read_parquet(CKPT / f"tr_fold{fi}.parquet"))
        train_st = tuple(s for s in stations if s not in hold)
        lf = CKPT / f"lgb_fold{fi}.parquet"
        if lf.exists():
            lgb_all.append(pd.read_parquet(lf))
        else:
            g = lgb_fold(panel_imp, clim, train_st, hold)
            g.to_parquet(lf); lgb_all.append(g)
            log(f"LGB fold{fi}: {len(g)} rows")
    TR = pd.concat(tr_all, ignore_index=True)
    LGB = pd.concat(lgb_all, ignore_index=True)

    # training-free baselines from the v2 predictions (identical regardless of LGO)
    v2 = pd.read_parquet(C.PREDICTIONS / "usgs_predictions_v2.parquet")
    base = v2[(v2.split == "test") & v2.model.isin(["Persistence", "DampedPersistence"])]
    tr_r = ps_rmse(TR); lgb_r = ps_rmse(LGB)
    per_r = ps_rmse(base[base.model == "Persistence"])
    dmp_r = ps_rmse(base[base.model == "DampedPersistence"])

    # nearest-training-gage distance per held-out station
    dist = {}
    for fi, hold in enumerate(folds):
        train_st = [s for s in stations if s not in hold]
        tr_ll = [(meta.loc[s, "lat"], meta.loc[s, "lon"]) for s in train_st if s in meta.index]
        for s in hold:
            if s not in meta.index:
                continue
            p = (meta.loc[s, "lat"], meta.loc[s, "lon"])
            dist[s] = min(haversine(p, q) for q in tr_ll)

    L = ["# Leave-HUC2-region-out transfer — ThermoRoute vs the strong learned baseline\n",
         f"16 HUC2 regions packed into 4 folds ({[len(f) for f in folds]} stations); "
         "each fold holds out **whole regions** so no held-out-region gage is in "
         "training (fixes the random-split spatial leak). Station-agnostic "
         "ThermoRoute and a global LightGBM are each trained on the in-fold regions "
         "and forecast the held-out region stations. Persistence/damped are "
         "training-free (read from v2).\n",
         "| horizon | n | TR RMSE | LGB RMSE | persist | damped | TR skill/persist | "
         "LGB skill/persist | TR−LGB paired Wilcoxon p | winner |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for h in C.HORIZONS:
        sts = sorted(s for s in stations
                     if (s, h) in tr_r and (s, h) in lgb_r
                     and (s, h) in per_r and (s, h) in dmp_r)
        a = np.array([tr_r[(s, h)] for s in sts])       # TR
        b = np.array([lgb_r[(s, h)] for s in sts])      # LGB
        pr = np.array([per_r[(s, h)] for s in sts])
        dm = np.array([dmp_r[(s, h)] for s in sts])
        p = wilcoxon(a, b).pvalue if len(sts) > 5 else float("nan")
        tr_win = float((a < b).mean())
        win = ("TR" if (np.median(a) < np.median(b) and p < 0.05)
               else "LGB" if (np.median(b) < np.median(a) and p < 0.05) else "tie")
        verdict[h] = {"n": len(sts), "tr": float(np.median(a)), "lgb": float(np.median(b)),
                      "p": p, "tr_win_rate": tr_win, "winner": win,
                      "skill_tr_persist": float(np.median(1 - a/pr)),
                      "skill_lgb_persist": float(np.median(1 - b/pr)),
                      "skill_tr_damped": float(np.median(1 - a/dm))}
        L.append(f"| {h} | {len(sts)} | {np.median(a):.3f} | {np.median(b):.3f} | "
                 f"{np.median(pr):.3f} | {np.median(dm):.3f} | "
                 f"{np.median(1-a/pr):+.3f} | {np.median(1-b/pr):+.3f} | "
                 f"{p:.2g} | **{win}** |")

    # skill-vs-distance figure (h=7)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    h = 7
    sts = sorted(s for s in stations if (s, h) in tr_r and s in dist and (s, h) in per_r)
    dvals = np.array([dist[s] for s in sts])
    tskill = np.array([1 - tr_r[(s, h)]/per_r[(s, h)] for s in sts])
    ax.scatter(dvals, tskill, s=20, alpha=0.7, color="#185FA5")
    ax.axhline(0, color="#993C1D", ls="--", lw=1)
    ax.set_xlabel("distance to nearest training gage (km)")
    ax.set_ylabel("TR transfer skill vs persistence (h=7d)")
    ax.set_title("Region-holdout transfer skill vs spatial extrapolation distance")
    ax.grid(alpha=0.25)
    fig.savefig(C.FIGURES / "fig_region_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    L += ["", "## Verdict\n"]
    winners = [verdict[h]["winner"] for h in C.HORIZONS]
    if winners.count("TR") >= 2:
        L.append("**GO** — ThermoRoute significantly beats the strong learned baseline "
                 "(LightGBM) out-of-region at a majority of horizons. Transfer is a "
                 "defensible central claim; pursue the 一区/CCF-A framing.")
    elif winners.count("LGB") >= 2:
        L.append("**NO-GO (LGB)** — LightGBM transfers better; drop the transfer-superiority "
                 "claim and pivot to the calibrated-forecaster + honest-delineation framing.")
    else:
        L.append("**TIE** — ThermoRoute and LightGBM transfer comparably out-of-region. "
                 "Report parity honestly; the contribution is calibration + physics-vs-GBDT "
                 "delineation, not a transfer-superiority claim (二区-strong framing).")
    L.append(f"\nPer-horizon winners: {dict(zip(C.HORIZONS, winners))}")
    L.append(f"Mean nearest-training-gage distance for held-out stations: "
             f"{np.mean(list(dist.values())):.0f} km "
             f"(vs a random split where it would be near 0).")

    out = C.REPORTS / "region_transfer.md"
    out.write_text("\n".join(L))
    pd.DataFrame(verdict).T.to_csv(C.TABLES / "region_transfer.csv")
    print("\n".join(L))
    log(f"wrote {out} + fig_region_transfer.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    if a.fold is not None:
        train_fold_tr(a.fold)
    elif a.assemble:
        assemble()
    else:
        _, _, _, _, _, _, stations = prep()
        folds, reg = region_folds(stations)
        for i, (f, r) in enumerate(zip(folds, reg)):
            print(f"fold{i}: {len(f)} stations, regions {r}")
