#!/usr/bin/env python3
"""Stage 9 — comprehensive large-sample experiment on the USGS station set.

Produces the headline result for the (re-)paper: does ThermoRoute beat
persistence AND damped persistence on rivers with real forecast headroom, with
calibrated uncertainty, and does it transfer to unseen basins?

Outputs (all in the canonical predictions schema so the analysis stage reuses
conformal/metrics/decision code unchanged):
  * outputs/predictions/usgs_predictions.parquet  (baselines + LightGBM + ThermoRoute seeds + LGO)
  * outputs/models/thermoroute_usgs.pt            (seed-0 model for mechanism analysis)
  * outputs/tables/usgs_scores.csv, outputs/reports/usgs_experiment.md

Run:  PYTHONPATH=src python3 scripts/09_usgs_experiment.py --seeds 5
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

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
import lightgbm as lgb

torch.set_num_threads(8)

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute import results as R
from thermoroute import significance as S
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")  # +gridMET wind
CFG = C.TrainConfig(batch_size=1536)         # larger batch ⇒ fewer steps on 100k+ samples
DELTA_SCALE = 1.5                            # loosened bound (selected by scripts/11_retune.py)
_t0 = time.time()


def log(m):
    print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


def prep(panel_path: str):
    panel = pd.read_parquet(panel_path)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    stations = tuple(sorted(panel.site_id.unique()))
    C.STATIONS = stations
    C.UPSTREAM = {s: None for s in stations}
    for v in C.ALL_VARS:
        panel[f"{v}_observed"] = panel[v].notna()
    masks = D.split_masks(panel["DATE"])
    panel_imp = D.Imputer.fit(panel, masks.train).transform(panel)
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    return panel, panel_imp, masks, clim, stations


def phi_per_station(panel, clim, masks):
    phi = {}
    tr = panel.loc[masks.train]
    for st in C.STATIONS:
        sub = tr[tr.site_id == st].sort_values("DATE")
        a = (sub.WTEMP - clim.predict_dates(st, sub.DATE)).to_numpy(float)
        m = ~np.isnan(a[1:]) & ~np.isnan(a[:-1])
        phi[st] = float(np.clip(np.corrcoef(a[1:][m], a[:-1][m])[0, 1], 0, 0.999)) if m.sum() > 30 else 0.9
    return phi


def canon(wd, idx, model_name, preds_by_h, scope="joint_usgs"):
    site = np.array([C.STATIONS[i] for i in wd.station[idx]])
    issue = wd.issue_date[idx]
    frames = []
    for hi, h in enumerate(wd.horizons):
        frames.append(R.make_pred_frame(
            model=model_name, scope=scope, feature_set="USGS", seed=0,
            site_id=site, horizon=np.full(len(idx), h), split=np.full(len(idx), "test"),
            issue_date=issue, target_date=issue + np.timedelta64(h, "D"),
            y_true=wd.y[idx][:, hi], y_pred=preds_by_h[h]))
    return pd.concat(frames, ignore_index=True)


def lightgbm_joint(panel_imp, panel_raw, clim, masks, thr):
    """One LightGBM across all stations per horizon (point + quantiles + exceedance)."""
    from thermoroute import baselines as B
    frames = []
    for h in C.HORIZONS:
        tab = F.attach_split(F.build_tabular(panel_raw, h, USGS_VARS, clim))
        cols = F.feature_columns(tab)
        tr, va = tab[tab.split == "train"], tab[tab.split == "val"]
        ev = tab[tab.split.isin(["calib", "test"])]
        Xtr, ytr, Xva, yva = tr[cols].to_numpy(float), tr["y"].to_numpy(float), \
            va[cols].to_numpy(float), va["y"].to_numpy(float)
        Xev = ev[cols].to_numpy(float)
        mp = B._lgb_fit(Xtr, ytr, Xva, yva, "regression")
        q = {}
        for a in C.QUANTILES:
            q[a] = B._lgb_fit(Xtr, ytr, Xva, yva, "quantile", alpha=a).predict(Xev)
        st_arr = ev["site_id"].to_numpy()
        thr_arr = np.array([thr[s] for s in st_arr])
        thr_tr = np.array([thr[s] for s in tr["site_id"]])
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=50, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1,
                                 seed=0, n_jobs=1)
        clf.fit(Xtr, (ytr > thr_tr).astype(int))
        stacked = np.sort(np.vstack([q[0.05], q[0.5], q[0.95]]), axis=0)
        frames.append(R.make_pred_frame(
            model="LightGBM", scope="joint_usgs", feature_set="USGS", seed=0,
            site_id=st_arr, horizon=np.full(len(ev), h), split=ev["split"].to_numpy(),
            issue_date=ev["issue_date"].to_numpy(), target_date=ev["target_date"].to_numpy(),
            y_true=ev["y"].to_numpy(float), y_pred=mp.predict(Xev),
            q05=stacked[0], q50=stacked[1], q95=stacked[2],
            p_exceed=clf.predict_proba(Xev)[:, 1]))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "data_usgs" / "panel_usgs_wind.parquet"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--delta_scale", type=float, default=DELTA_SCALE)
    ap.add_argument("--ablations", action="store_true", default=True)
    args = ap.parse_args()

    panel, panel_imp, masks, clim, stations = prep(args.panel)
    phi = phi_per_station(panel, clim, masks)
    wd = DS.build_windows(panel_imp, masks, clim, variables=USGS_VARS,
                          require_observed_target=True)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    test_idx = wd.idx("test")
    log(f"{len(stations)} stations | windows N={len(wd.X)} test={len(test_idx)}")

    chunks = []
    # ---- baselines from identical windowed samples ---------------------- #
    site = np.array([C.STATIONS[i] for i in wd.station[test_idx]])
    for name, fn in [("Persistence", lambda hi, h: wd.wtemp_t[test_idx]),
                     ("Climatology", lambda hi, h: wd.clim_tgt[test_idx][:, hi]),
                     ("DampedPersistence", lambda hi, h: wd.clim_tgt[test_idx][:, hi]
                      + np.array([phi[s] for s in site]) ** h
                      * (wd.wtemp_t[test_idx] - wd.clim_t[test_idx]))]:
        preds = {h: fn(hi, h) for hi, h in enumerate(wd.horizons)}
        chunks.append(canon(wd, test_idx, name, preds))
    log("baselines done")

    # ---- joint LightGBM ------------------------------------------------- #
    chunks.append(lightgbm_joint(panel_imp, panel, clim, masks, thr))
    log("LightGBM joint done")

    # ---- ThermoRoute joint, multiple seeds (resumable per seed) ---------- #
    ckpt = C.PREDICTIONS / "usgs_seed_ckpt"
    ckpt.mkdir(exist_ok=True)
    tr_preds = []
    for sd in range(args.seeds):
        seed_file = ckpt / f"seed{sd}.parquet"
        if seed_file.exists():                       # resume: skip completed seed
            tr_preds.append(pd.read_parquet(seed_file))
            log(f"  ThermoRoute seed{sd}: loaded checkpoint")
            continue
        te = time.time()
        model = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                            n_phys=wd.n_phys, delta_scale=args.delta_scale)
        res = fit_model(model, wd, thr, cfg=CFG, seed=sd, model_name="ThermoRoute",
                        scope="joint_usgs", feature_set="USGS")
        res.pred["seed"] = sd
        res.pred.to_parquet(seed_file)               # checkpoint immediately
        tr_preds.append(res.pred)
        if sd == 0:
            torch.save(model.state_dict(), C.MODELS / "thermoroute_usgs.pt")
        log(f"  ThermoRoute seed{sd}: {res.epochs+1}ep {time.time()-te:.0f}s val={res.best_val:.4f}")
    chunks.append(pd.concat(tr_preds, ignore_index=True))

    # ---- leave-group-out ------------------------------------------------ #
    rng = np.random.default_rng(0)
    perm = rng.permutation(list(stations))
    hold = set(perm[: max(1, len(stations) // 4)])
    trainset = tuple(s for s in stations if s not in hold)
    lgo_file = ckpt / "lgo.parquet"
    if lgo_file.exists():
        chunks.append(pd.read_parquet(lgo_file))
        log("  LGO: loaded checkpoint")
    else:
        te = time.time()
        model = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                            n_phys=wd.n_phys, station_agnostic=True,
                            delta_scale=args.delta_scale)
        res = fit_model(model, wd, thr, cfg=CFG, seed=0, model_name="ThermoRoute-LGO",
                        scope="lgo", feature_set="USGS", train_stations=trainset)
        res.pred["seed"] = 0
        lgo_held = res.pred[res.pred.site_id.isin(hold)]
        lgo_held.to_parquet(lgo_file)
        chunks.append(lgo_held)
        log(f"  LGO ({len(trainset)}→{len(hold)}): {time.time()-te:.0f}s")

    # ---- large-sample module ablations (single seed) -------------------- #
    if args.ablations:
        abl = {"TR-noPrior": dict(use_prior=False),
               "TR-fixedKappa": dict(fixed_kappa=True),
               "TR-noRouter": dict(use_router=False),
               "TR-noMoE": dict(use_moe=False)}
        for name, kw in abl.items():
            af = ckpt / f"{name}.parquet"
            if af.exists():
                chunks.append(pd.read_parquet(af)); log(f"  {name}: loaded"); continue
            te = time.time()
            m = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                            n_phys=wd.n_phys, delta_scale=args.delta_scale, **kw)
            r = fit_model(m, wd, thr, cfg=CFG, seed=0, model_name=name,
                          scope="ablation_usgs", feature_set="USGS")
            r.pred["seed"] = 0
            r.pred.to_parquet(af); chunks.append(r.pred)
            log(f"  {name}: {time.time()-te:.0f}s val={r.best_val:.4f}")

    allp = pd.concat(chunks, ignore_index=True)
    allp.to_parquet(C.PREDICTIONS / "usgs_predictions.parquet")
    log(f"saved predictions ({len(allp)} rows)")

    # ---- headline point report (seed-mean ThermoRoute) ------------------ #
    tr_test = allp[(allp.model == "ThermoRoute") & (allp.split == "test")]
    base = {m: allp[(allp.model == m) & (allp.split == "test")] for m in
            ("Persistence", "DampedPersistence", "Climatology")}

    def rmse_per_station(df, h):
        d = {}
        for s, g in df[df.horizon == h].groupby("site_id"):
            d[s] = float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
        return d

    rows = []
    for h in wd.horizons:
        rp = rmse_per_station(base["Persistence"], h)
        rd = rmse_per_station(base["DampedPersistence"], h)
        # ThermoRoute seed-mean per (station,issue): average y_pred over seeds
        tm = tr_test[tr_test.horizon == h].groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        rt = {s: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
              for s, g in tm.groupby("site_id")}
        for s in stations:
            rows.append({"horizon": h, "site": s, "rmse_persist": rp.get(s, np.nan),
                         "rmse_damped": rd.get(s, np.nan), "rmse_thermo": rt.get(s, np.nan)})
    sc = pd.DataFrame(rows)
    sc.to_csv(C.TABLES / "usgs_scores.csv", index=False)

    L = [f"# USGS large-sample experiment ({len(stations)} stations, {args.seeds} seeds)\n",
         f"_Variables {', '.join(USGS_VARS)}. Observed targets only; identical samples "
         f"across models. ThermoRoute = {args.seeds}-seed mean._\n",
         "| horizon | persist | damped | LightGBM | ThermoRoute | skill vs persist | "
         "skill vs damped | win-rate vs damped |", "|---|---|---|---|---|---|---|---|"]
    lg = allp[(allp.model == "LightGBM") & (allp.split == "test")]
    for h in wd.horizons:
        d = sc[sc.horizon == h]
        rl = rmse_per_station(lg, h)
        ml = np.median([rl[s] for s in stations if s in rl])
        mp, md, mt = d.rmse_persist.median(), d.rmse_damped.median(), d.rmse_thermo.median()
        sk_p = 1 - (d.rmse_thermo / d.rmse_persist).median()
        sk_d = 1 - (d.rmse_thermo / d.rmse_damped).median()
        win = float((d.rmse_thermo < d.rmse_damped).mean())
        L.append(f"| {h} | {mp:.3f} | {md:.3f} | {ml:.3f} | {mt:.3f} | {sk_p:+.3f} | "
                 f"{sk_d:+.3f} | {win:.2f} |")
    # leave-group-out
    lgo = allp[(allp.model == "ThermoRoute-LGO") & (allp.split == "test")]
    L += ["", f"## Leave-group-out transfer ({len(trainset)}→{len(hold)} unseen basins)\n",
          "| horizon | TR transfer RMSE | persistence RMSE | transfer skill |",
          "|---|---|---|---|"]
    for h in wd.horizons:
        g = lgo[lgo.horizon == h]
        rt = float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
        bp = base["Persistence"]
        gp = bp[(bp.horizon == h) & (bp.site_id.isin(hold))]
        rp = float(np.sqrt(((gp.y_pred - gp.y_true) ** 2).mean()))
        L.append(f"| {h} | {rt:.3f} | {rp:.3f} | {1-rt/rp:+.3f} |")

    # ---- ablation summary (median per-station RMSE) --------------------- #
    abl_models = ["ThermoRoute", "TR-noPrior", "TR-fixedKappa", "TR-noRouter", "TR-noMoE"]
    L += ["", f"## Module ablations (median per-station RMSE, delta_scale={args.delta_scale})\n",
          "| variant | h1 | h3 | h7 |", "|---|---|---|---|"]
    for m in abl_models:
        if m == "ThermoRoute":
            sub = tr_test.groupby(["site_id", "horizon", "issue_date"]).agg(
                y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        else:
            sub = allp[(allp.model == m) & (allp.split == "test")]
        if sub.empty:
            continue
        meds = []
        for h in wd.horizons:
            r = rmse_per_station(sub, h)
            meds.append(np.median([r[s] for s in stations if s in r]))
        L.append(f"| {m} | {meds[0]:.3f} | {meds[1]:.3f} | {meds[2]:.3f} |")
    (C.REPORTS / "usgs_experiment.md").write_text("\n".join(L))
    log("DONE")
    print("\n" + "\n".join(L[3:10]))


if __name__ == "__main__":
    main()
