#!/usr/bin/env python3
"""Leave-HUC2-region-out transfer, benchmarked vs the strong learned baseline.

Fixes the two blockers of the strong-accept review:
  B1 — transfer was never benchmarked vs LightGBM (only persistence/damped).
  B2 — "unseen basins" was a RANDOM station split; here whole HUC2 regions are
       held out so no gage on a held-out river/region is in training.

Design: verified HUC2 regions are greedily packed into 4 folds (~30 stations each), each
fold holds out whole regions. Under this protocol we train, per fold, a
station-agnostic ThermoRoute AND a global LightGBM on the in-fold regions and
forecast the held-out region stations. Persistence / damped persistence are
training-free so their per-station test RMSE is read from the v2 predictions.
Then descriptive per-station paired effects with whole-HUC2 bootstrap intervals
compare TR with LightGBM, alongside a skill-versus-distance diagnostic.  This arm
is exploratory and never auto-declares parity or a publication go/no-go.

Usage:
  PYTHONPATH=src python3 scripts/13c_region_transfer.py --fold N   # train fold N's TR
  PYTHONPATH=src python3 scripts/13c_region_transfer.py --assemble # LGB + stats + report
"""
from __future__ import annotations

import os
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

torch.set_num_threads(int(os.environ.get("WORKER_THREADS", "8")))

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute import baselines as B
from thermoroute import results as R
from thermoroute.registry import (
    enforce_common_forecast_keys,
    restrict_tabular_to_window_registry,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
CFG = C.TrainConfig(batch_size=1536)
DELTA = C.DELTA_SCALE                         # val-selected, matches 09/13
PANEL = Path(os.environ.get(
    "USGS_PANEL", str(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")))
STATION_REGISTRY = Path(os.environ.get(
    "USGS_STATION_REGISTRY",
    str(ROOT / "data_usgs" / "station_registry_v1.csv"),
))
CKPT = C.PREDICTIONS / "region_ckpt_route_a_gauged_transfer_v2"
_t0 = time.time()


def log(m): print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


def spatial_metadata():
    """Read stable site_no metadata without falling back to legacy nXX aliases."""
    return load_station_registry(STATION_REGISTRY).set_index("site_no")


def region_folds(stations):
    """Greedy pack whole HUC2 regions into 4 balanced folds (deterministic)."""
    meta = spatial_metadata()
    huc = huc2_cluster_map(meta.reset_index())
    by_reg = {}
    for s in stations:
        by_reg.setdefault(huc.get(s, f"UNMAPPED:{s}"), []).append(s)
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


def prep_fold(fold_i):
    """Build a held-region fold without held-station fitted statistics.

    Held regions contribute neither climatology coefficients, scaling moments,
    damped-AR phi nor imputation statistics.  Recent observed WTEMP remains an
    issue-time input, matching the deployed sensor-forecast task.
    """
    b = D.prepare_dataset_from_panel(str(PANEL))
    panel, masks, stations = b["panel_raw"], b["masks"], b["stations"]
    folds, _ = region_folds(stations)
    hold = folds[fold_i]
    train_st = tuple(s for s in stations if s not in hold)
    clim = F.HarmonicClimatology.fit(
        panel, masks.train, fit_stations=train_st, pooled=True)
    wd = DS.build_windows(
        panel, masks, clim, variables=USGS_VARS, require_observed_target=True,
        scaler_fit_stations=train_st, pooled_scaler=True,
        damped_fit_stations=train_st, pooled_damped=True)

    train_rows = panel.loc[masks.train & panel.site_id.isin(train_st).to_numpy()]
    global_thr = float(train_rows.WTEMP.quantile(0.9))
    thr = {}
    for station in stations:
        local = train_rows[train_rows.site_id == station].WTEMP
        thr[station] = float(local.quantile(0.9)) if local.notna().any() else global_thr
    return panel, masks, clim, thr, wd, stations, train_st, hold


def train_fold_tr(fold_i):
    CKPT.mkdir(parents=True, exist_ok=True)
    panel, masks, clim, thr, wd, stations, train_st, hold = prep_fold(fold_i)
    f = CKPT / f"tr_fold{fold_i}.parquet"
    if f.exists():
        log(f"TR fold{fold_i}: already done"); return
    log(f"TR fold{fold_i}: train {len(train_st)} -> hold {len(hold)} region stations")
    factory = lambda: ThermoRoute(
        n_vars=len(wd.var_names), n_stations=len(stations), n_phys=wd.n_phys,
        station_agnostic=True, delta_scale=DELTA, safety_anchor="damped")
    r = fit_model(factory, wd, thr, cfg=CFG, seed=0, scope="region_lgo",
                  feature_set="USGS", train_stations=train_st,
                  station_balanced=True, selection_metric="station_macro")
    pred = r.pred[(r.pred.split == "test") & (r.pred.site_id.isin(hold))].copy()
    pred["model"] = "ThermoRoute-regionLGO"
    pred.to_parquet(f)
    log(f"TR fold{fold_i}: DONE {r.epochs+1}ep val={r.best_val:.4f} -> {f.name}")


def lgb_fold(panel, clim, train_st, hold, wd):
    """Global LightGBM trained on in-fold stations, predicting held-out stations."""
    frames = []
    for h in C.HORIZONS:
        tab = F.attach_split(F.build_tabular(panel, h, USGS_VARS, clim,
                             drop_feature_nans=False, require_observed_target=True,
                             include_missingness=True))
        tab = restrict_tabular_to_window_registry(tab, wd, C.STATIONS, h)
        cols = F.feature_columns(tab)
        for c in cols:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
        tr = tab[(tab.split == "train") & (tab.site_id.isin(train_st))]
        va = tab[(tab.split == "val") & (tab.site_id.isin(train_st))]
        ev = tab[(tab.split == "test") & (tab.site_id.isin(hold))]
        mp = B._lgb_fit(tr[cols].to_numpy(float), tr["y"].to_numpy(float),
                        va[cols].to_numpy(float), va["y"].to_numpy(float), "regression")
        frames.append(R.make_pred_frame(
            model="LightGBM-regionLGO", scope="region_lgo_gauged",
            feature_set="USGS", seed=0, site_id=ev["site_id"].to_numpy(),
            horizon=np.full(len(ev), h), split=ev["split"].to_numpy(),
            issue_date=ev["issue_date"].to_numpy(),
            target_date=ev["target_date"].to_numpy(),
            y_true=ev["y"].to_numpy(float),
            y_pred=mp.predict(ev[cols].to_numpy(float))))
    return pd.concat(frames, ignore_index=True)


def baseline_fold(wd, hold):
    """Persistence and the exact pooled damped anchor on held samples."""
    idx = wd.idx("test")
    site = np.asarray([C.STATIONS[i] for i in wd.station[idx]])
    selected = np.isin(site, list(hold))
    idx, site = idx[selected], site[selected]
    frames = []
    for hi, horizon in enumerate(wd.horizons):
        common = dict(
            scope="region_lgo_gauged", feature_set="USGS", seed=0,
            site_id=site, horizon=np.full(len(idx), horizon),
            split=np.full(len(idx), "test"), issue_date=wd.issue_date[idx],
            target_date=wd.target_date[idx, hi], y_true=wd.y[idx, hi])
        frames.append(R.make_pred_frame(
            model="Persistence-regionLGO", y_pred=wd.wtemp_t[idx], **common))
        frames.append(R.make_pred_frame(
            model="DampedPersistence-regionLGO",
            y_pred=wd.damped_prior[idx, hi], **common))
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
    CKPT.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panel, panel_imp, masks, clim, thr, wd, stations = prep()
    folds, reg_of_fold = region_folds(stations)
    meta = spatial_metadata()

    # Fold-specific pooled preprocessing for every learned/reference model.
    tr_all, lgb_all, base_all = [], [], []
    for fi, hold in enumerate(folds):
        fold_panel, _, fold_clim, _, fold_wd, _, train_st, strict_hold = prep_fold(fi)
        if strict_hold != hold:
            raise AssertionError("region fold construction changed within one run")
        tr_all.append(pd.read_parquet(CKPT / f"tr_fold{fi}.parquet"))
        lf = CKPT / f"lgb_fold{fi}.parquet"
        if lf.exists():
            lgb_all.append(pd.read_parquet(lf))
        else:
            g = lgb_fold(fold_panel, fold_clim, train_st, hold, fold_wd)
            g.to_parquet(lf); lgb_all.append(g)
            log(f"LGB fold{fi}: {len(g)} rows")
        base_all.append(baseline_fold(fold_wd, hold))
    TR = pd.concat(tr_all, ignore_index=True)
    LGB = pd.concat(lgb_all, ignore_index=True)
    base = pd.concat(base_all, ignore_index=True)
    combined, audit = enforce_common_forecast_keys(
        pd.concat([TR, LGB, base], ignore_index=True),
        ("ThermoRoute-regionLGO", "LightGBM-regionLGO",
         "Persistence-regionLGO", "DampedPersistence-regionLGO"), split="test")
    log(f"region registry: {audit.common_unique} exact shared keys; "
        f"dropped={audit.dropped_rows}; before={audit.before_unique}")
    TR = combined[combined.model == "ThermoRoute-regionLGO"]
    LGB = combined[combined.model == "LightGBM-regionLGO"]
    base = combined[combined.model.isin(
        ["Persistence-regionLGO", "DampedPersistence-regionLGO"])]
    tr_r = ps_rmse(TR); lgb_r = ps_rmse(LGB)
    per_r = ps_rmse(base[base.model == "Persistence-regionLGO"])
    dmp_r = ps_rmse(base[base.model == "DampedPersistence-regionLGO"])
    huc_by_site = huc2_cluster_map(meta.reset_index())

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
         f"{sum(len(r) for r in reg_of_fold)} HUC2/unknown groups packed into 4 folds "
         f"({[len(f) for f in folds]} stations); "
         "each fold holds out **whole regions** so no held-out-region gage is in "
         "training (fixes the random-split spatial leak). Station-agnostic "
         "ThermoRoute and a global LightGBM are each trained on the in-fold regions "
         "and forecast the held-out region stations. All climatology, scaling and "
         "damped-phi parameters are pooled from in-fold stations only; held-site "
         "WTEMP enters only as observed issue/history input.\n",
         "| horizon | n | TR RMSE | LGB RMSE | persist | damped | TR skill/persist | "
         "LGB skill/persist | median TR−LGB [HUC2 95% CI] | TR win rate |",
         "|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for h in C.HORIZONS:
        sts = sorted(s for s in stations
                     if (s, h) in tr_r and (s, h) in lgb_r
                     and (s, h) in per_r and (s, h) in dmp_r)
        a = np.array([tr_r[(s, h)] for s in sts])       # TR
        b = np.array([lgb_r[(s, h)] for s in sts])      # LGB
        pr = np.array([per_r[(s, h)] for s in sts])
        dm = np.array([dmp_r[(s, h)] for s in sts])
        effects = a - b
        clusters = np.asarray([huc_by_site.get(s, f"UNMAPPED:{s}") for s in sts])
        inference = cluster_bootstrap_paired_effect(
            effects, clusters, n_boot=10000, seed=1300 + h,
        )
        tr_win = float((a < b).mean())
        verdict[h] = {"n": len(sts), "tr": float(np.median(a)), "lgb": float(np.median(b)),
                      "median_tr_minus_lgb": inference["effect"],
                      "ci_low": inference["ci_low"], "ci_high": inference["ci_high"],
                      "tr_win_rate": tr_win,
                      "skill_tr_persist": float(np.median(1 - a/pr)),
                      "skill_lgb_persist": float(np.median(1 - b/pr)),
                      "skill_tr_damped": float(np.median(1 - a/dm))}
        L.append(f"| {h} | {len(sts)} | {np.median(a):.3f} | {np.median(b):.3f} | "
                 f"{np.median(pr):.3f} | {np.median(dm):.3f} | "
                 f"{np.median(1-a/pr):+.3f} | {np.median(1-b/pr):+.3f} | "
                 f"{inference['effect']:+.3f} [{inference['ci_low']:+.3f},"
                 f"{inference['ci_high']:+.3f}] | {tr_win:.2f} |")

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

    L += ["", "## Interpretation guard\n"]
    L.append(
        "This held-region analysis is exploratory and does not auto-declare a "
        "winner, tie, parity, or publication go/no-go. Intervals resample complete "
        "HUC2 groups and quantify cross-region sampling variation only. Issue-time "
        "WTEMP history is required, so this is not ungauged prediction."
    )
    L.append(f"Mean nearest-training-gage distance for held-out stations: "
             f"{np.mean(list(dist.values())):.0f} km.")

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
