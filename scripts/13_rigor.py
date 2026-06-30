#!/usr/bin/env python3
"""Stage 13 — rigor for claim 2 (transfer) and claim 4 (ablations).

Claim 2: K-fold leave-group-out so every station is held out exactly once; report
mean ± std transfer skill across folds (vs persistence and damped), not a single
random split.
Claim 4: ablations at 3 seeds; per-station paired test of each ablation vs the full
model to confirm which components significantly matter.

Resumable: each fold / ablation-seed is checkpointed.
Run:  PYTHONPATH=src python3 scripts/13_rigor.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

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
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
CFG = C.TrainConfig(batch_size=1536)
DELTA = 1.5
N_FOLDS = 4
CKPT = C.PREDICTIONS / "rigor_ckpt"
CKPT.mkdir(exist_ok=True)
_t0 = time.time()


def log(m): print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


def prep():
    _p100 = ROOT / "data_usgs" / "panel_usgs_100.parquet"
    _pwind = ROOT / "data_usgs" / "panel_usgs_wind.parquet"
    panel_path = _p100 if _p100.exists() else _pwind
    log(f"using panel {panel_path.name}")
    panel = pd.read_parquet(panel_path)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    stations = tuple(sorted(panel.site_id.unique()))
    C.STATIONS = stations; C.UPSTREAM = {s: None for s in stations}
    for v in C.ALL_VARS:
        panel[f"{v}_observed"] = panel[v].notna()
    masks = D.split_masks(panel["DATE"])
    pi = D.Imputer.fit(panel, masks.train).transform(panel)
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    wd = DS.build_windows(pi, masks, clim, variables=USGS_VARS, require_observed_target=True)
    # per-station phi for damped baseline
    phi = {}
    tr = panel.loc[masks.train]
    for st in stations:
        sub = tr[tr.site_id == st].sort_values("DATE")
        a = (sub.WTEMP - clim.predict_dates(st, sub.DATE)).to_numpy(float)
        mm = ~np.isnan(a[1:]) & ~np.isnan(a[:-1])
        phi[st] = float(np.clip(np.corrcoef(a[1:][mm], a[:-1][mm])[0, 1], 0, .999)) if mm.sum() > 30 else .9
    return wd, thr, stations, phi


def base_rmse_held(wd, idx, phi, hold):
    """persistence/damped RMSE on held-out stations' samples, per horizon."""
    site = np.array([C.STATIONS[i] for i in wd.station[idx]])
    sel = np.isin(site, list(hold))
    out = {}
    for hi, h in enumerate(wd.horizons):
        y = wd.y[idx][sel, hi]
        per = wd.wtemp_t[idx][sel]
        cl = wd.clim_tgt[idx][sel, hi]
        ph = np.array([phi[s] for s in site[sel]]) ** h
        dmp = cl + ph * (wd.wtemp_t[idx][sel] - wd.clim_t[idx][sel])
        out[h] = {"persist": float(np.sqrt(np.mean((y - per) ** 2))),
                  "damped": float(np.sqrt(np.mean((y - dmp) ** 2)))}
    return out


def kfold_lgo(wd, thr, stations, phi):
    rng = np.random.default_rng(0)
    perm = list(rng.permutation(list(stations)))
    folds = [set(perm[i::N_FOLDS]) for i in range(N_FOLDS)]
    test_idx = wd.idx("test")
    rows = []
    for fi, hold in enumerate(folds):
        f = CKPT / f"lgo_fold{fi}.parquet"
        if f.exists():
            pred = pd.read_parquet(f); log(f"LGO fold{fi}: loaded")
        else:
            te = time.time()
            train_st = tuple(s for s in stations if s not in hold)
            m = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                            n_phys=wd.n_phys, station_agnostic=True, delta_scale=DELTA)
            r = fit_model(m, wd, thr, cfg=CFG, seed=0, scope="lgo", feature_set="USGS",
                          train_stations=train_st)
            pred = r.pred[(r.pred.split == "test") & (r.pred.site_id.isin(hold))]
            pred.to_parquet(f); log(f"LGO fold{fi} ({len(train_st)}→{len(hold)}): {time.time()-te:.0f}s")
        base = base_rmse_held(wd, test_idx, phi, hold)
        for h in wd.horizons:
            g = pred[pred.horizon == h]
            rt = float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
            rows.append({"fold": fi, "horizon": h, "tr": rt,
                         "skill_persist": 1 - rt / base[h]["persist"],
                         "skill_damped": 1 - rt / base[h]["damped"]})
    return pd.DataFrame(rows)


def ablation_seeds(wd, thr, stations):
    abls = {"TR-noPrior": dict(use_prior=False), "TR-fixedKappa": dict(fixed_kappa=True),
            "TR-noRouter": dict(use_router=False), "TR-noMoE": dict(use_moe=False)}
    frames = []
    for name, kw in abls.items():
        for sd in (1, 2):
            f = CKPT / f"{name}_seed{sd}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f)); continue
            te = time.time()
            m = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                            n_phys=wd.n_phys, delta_scale=DELTA, **kw)
            r = fit_model(m, wd, thr, cfg=CFG, seed=sd, model_name=name,
                          scope="ablation_usgs", feature_set="USGS")
            r.pred["seed"] = sd
            sub = r.pred[r.pred.split == "test"]
            sub.to_parquet(f); frames.append(sub)
            log(f"{name} seed{sd}: {time.time()-te:.0f}s")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    wd, thr, stations, phi = prep()
    log(f"prepared {len(stations)} stations, N={len(wd.X)}")

    lgo = kfold_lgo(wd, thr, stations, phi)
    L = ["# Claim 2 — K-fold leave-group-out transfer (every station held out once)\n",
         f"{N_FOLDS} folds. Mean ± std of transfer skill across folds.\n",
         "| horizon | TR RMSE (mean) | skill vs persistence | skill vs damped |",
         "|---|---|---|---|"]
    for h in C.HORIZONS:
        d = lgo[lgo.horizon == h]
        L.append(f"| {h} | {d.tr.mean():.3f} | {d.skill_persist.mean():+.3f} ± "
                 f"{d.skill_persist.std():.3f} | {d.skill_damped.mean():+.3f} ± "
                 f"{d.skill_damped.std():.3f} |")
    lgo.to_csv(C.TABLES / "claim2_kfold_lgo.csv", index=False)

    # claim 4: ablations 3-seed mean per-station, paired vs full
    abl_new = ablation_seeds(wd, thr, stations)
    _v2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
    _120 = C.PREDICTIONS / "usgs_predictions_120.parquet"
    _40 = C.PREDICTIONS / "usgs_predictions.parquet"
    pred_path = _v2 if _v2.exists() else (_120 if _120.exists() else _40)
    log(f"using predictions {pred_path.name}")
    allp = pd.read_parquet(pred_path)
    full = allp[(allp.model == "ThermoRoute") & (allp.split == "test")]
    abl_seed0 = allp[allp.model.str.startswith("TR-") & (allp.split == "test")]
    abl_all = pd.concat([abl_seed0, abl_new], ignore_index=True)

    def ps_rmse(df, h):
        g = df[df.horizon == h].groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        return {s: float(np.sqrt(((x.y_pred - x.y_true) ** 2).mean()))
                for s, x in g.groupby("site_id")}

    L += ["", "# Claim 4 — ablations (3-seed mean median RMSE) + paired test vs full\n",
          "| variant | h1 | h3 | h7 | Wilcoxon p (h3, vs full) |", "|---|---|---|---|---|"]
    full_ps = {h: ps_rmse(full, h) for h in C.HORIZONS}
    for name in ["ThermoRoute", "TR-noPrior", "TR-fixedKappa", "TR-noRouter", "TR-noMoE"]:
        df = full if name == "ThermoRoute" else abl_all[abl_all.model == name]
        meds, p3 = [], np.nan
        for h in C.HORIZONS:
            ps = ps_rmse(df, h)
            meds.append(np.median([ps[s] for s in stations if s in ps]))
            if h == 3 and name != "ThermoRoute":
                common = [s for s in stations if s in ps and s in full_ps[3]]
                a = np.array([ps[s] for s in common]); b = np.array([full_ps[3][s] for s in common])
                p3 = wilcoxon(a, b).pvalue
        pstr = "—" if name == "ThermoRoute" else f"{p3:.1e}{'*' if p3 < 0.05 else ''}"
        L.append(f"| {name} | {meds[0]:.3f} | {meds[1]:.3f} | {meds[2]:.3f} | {pstr} |")

    (C.REPORTS / "rigor.md").write_text("\n".join(L))
    log("DONE")
    print("\n" + "\n".join(L), flush=True)


if __name__ == "__main__":
    main()
