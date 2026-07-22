#!/usr/bin/env python3
"""Stage 13 — exploratory warm-start transfer and architecture controls.

Diagnostic 2: K-fold leave-group-out so every station is held out exactly once;
this legacy diagnostic permits site-local train-era preprocessing and is
therefore labelled **warm-start**, not zero-shot unseen-basin transfer.  The
held-region, history-dependent gauged transfer is implemented in stage 13c.
Architecture controls use five seeds and descriptive paired effects; they do not
prove that a latent component is physically necessary or causally identified.

Resumable: each fold / ablation-seed is checkpointed.
Run:  PYTHONPATH=src python3 scripts/13_rigor.py
"""
from __future__ import annotations

import os
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

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute.registry import enforce_common_forecast_keys
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.spatial import huc2_cluster_map, load_station_registry
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
CFG = C.TrainConfig(batch_size=1536)
DELTA = C.DELTA_SCALE   # single source (config.py)
N_FOLDS = 4
CKPT = C.PREDICTIONS / "rigor_ckpt_route_a_strict_v1"
CKPT.mkdir(exist_ok=True)
_t0 = time.time()

ABLATIONS = {
    "ThermoRoute": {},
    "TR-noDynamicPrior": {"use_prior": False},
    "TR-fixedKappa": {"fixed_kappa": True},
    "TR-noRouter": {"use_router": False},
    "TR-noMoE": {"use_moe": False},
    "TR-noTCN": {"use_tcn": False},
    "TR-unbounded": {"delta_scale": None},
    "DampedPriorOnly": {"use_prior": False, "residual_model": False},
    # Same full architecture and epoch/update budget; only sampling/selection
    # weights change.  This is a sensitivity analysis, not a module ablation.
    "TR-naturalSampling": {},
}


def log(m): print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


def prep():
    panel_path = Path(os.environ.get(
        "USGS_PANEL", str(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")))
    log(f"using panel {panel_path.name}")
    bundle = D.prepare_dataset_from_panel(str(panel_path))
    panel, pi, masks = bundle["panel_raw"], bundle["panel"], bundle["masks"]
    stations = bundle["stations"]
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    wd = DS.build_windows(pi, masks, clim, variables=USGS_VARS, require_observed_target=True)
    return wd, thr, stations


def base_rmse_held(wd, idx, hold):
    """persistence/damped RMSE on held-out stations' samples, per horizon."""
    site = np.array([C.STATIONS[i] for i in wd.station[idx]])
    sel = np.isin(site, list(hold))
    out = {}
    for hi, h in enumerate(wd.horizons):
        y = wd.y[idx][sel, hi]
        per = wd.wtemp_t[idx][sel]
        dmp = wd.damped_prior[idx][sel, hi]
        out[h] = {"persist": float(np.sqrt(np.mean((y - per) ** 2))),
                  "damped": float(np.sqrt(np.mean((y - dmp) ** 2)))}
    return out


def kfold_lgo(wd, thr, stations):
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
            factory = lambda: ThermoRoute(
                n_vars=len(wd.var_names), n_stations=len(stations), n_phys=wd.n_phys,
                station_agnostic=True, delta_scale=DELTA, safety_anchor="damped")
            r = fit_model(factory, wd, thr, cfg=CFG, seed=0, scope="lgo",
                          feature_set="USGS", train_stations=train_st,
                          station_balanced=True, selection_metric="station_macro")
            pred = r.pred[(r.pred.split == "test") & (r.pred.site_id.isin(hold))]
            pred.to_parquet(f); log(f"LGO fold{fi} ({len(train_st)}→{len(hold)}): {time.time()-te:.0f}s")
        base = base_rmse_held(wd, test_idx, hold)
        for h in wd.horizons:
            g = pred[pred.horizon == h]
            rt = float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
            rows.append({"fold": fi, "horizon": h, "tr": rt,
                         "skill_persist": 1 - rt / base[h]["persist"],
                         "skill_damped": 1 - rt / base[h]["damped"]})
    return pd.DataFrame(rows)


def ablation_seeds(wd, thr, stations):
    frames = []
    for name, kw in ABLATIONS.items():
        for sd in C.USGS_SEEDS:
            f = CKPT / f"{name}_seed{sd}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f)); continue
            te = time.time()
            model_kw = dict(kw)
            model_kw.setdefault("delta_scale", DELTA)
            factory = lambda model_kw=model_kw: ThermoRoute(
                n_vars=len(wd.var_names), n_stations=len(stations),
                n_phys=wd.n_phys, safety_anchor="damped", **model_kw)
            balanced = name != "TR-naturalSampling"
            r = fit_model(factory, wd, thr, cfg=CFG, seed=sd, model_name=name,
                          scope="ablation_usgs", feature_set="USGS",
                          station_balanced=balanced,
                          selection_metric="station_macro" if balanced else "micro")
            r.pred["seed"] = sd
            sub = r.pred[r.pred.split == "test"]
            sub.to_parquet(f); frames.append(sub)
            log(f"{name} seed{sd}: {time.time()-te:.0f}s")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds_only", action="store_true",
                    help="train/checkpoint the LGO folds and exit (lets 13b "
                         "ablation workers run concurrently without racing "
                         "this script's own sequential ablation training)")
    args = ap.parse_args()

    wd, thr, stations = prep()
    log(f"prepared {len(stations)} stations, N={len(wd.X)}")

    lgo = kfold_lgo(wd, thr, stations)
    L = ["# Diagnostic 2 — warm-start K-fold station holdout\n",
         f"{N_FOLDS} folds. Site-local train-era preprocessing is permitted here; "
         "do not cite this table as zero-shot/unseen-basin evidence. Mean ± std "
         "of forecast skill across folds.\n",
         "| horizon | TR RMSE (mean) | skill vs persistence | skill vs damped |",
         "|---|---|---|---|"]
    for h in C.HORIZONS:
        d = lgo[lgo.horizon == h]
        L.append(f"| {h} | {d.tr.mean():.3f} | {d.skill_persist.mean():+.3f} ± "
                 f"{d.skill_persist.std():.3f} | {d.skill_damped.mean():+.3f} ± "
                 f"{d.skill_damped.std():.3f} |")
    lgo.to_csv(C.TABLES / "claim2_kfold_lgo.csv", index=False)
    if args.folds_only:
        log("folds_only: LGO folds checkpointed, claim2 written — exiting "
            "before ablations (run again without the flag to finish rigor.md)")
        return

    # Every exploratory control and the full model are trained independently under
    # the exact same five-seed budget.  No legacy stage-09 checkpoint is mixed in.
    abl_all = ablation_seeds(wd, thr, stations)
    abl_all, audit = enforce_common_forecast_keys(
        abl_all, tuple(ABLATIONS), split="test")
    log(f"ablation registry: {audit.common_unique} shared keys; "
        f"dropped={audit.dropped_rows}; before={audit.before_unique}")
    full = abl_all[(abl_all.model == "ThermoRoute") & (abl_all.split == "test")]

    def ps_rmse(df, h):
        g = df[df.horizon == h].groupby(
            ["site_id", "issue_date", "target_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        return {s: float(np.sqrt(((x.y_pred - x.y_true) ** 2).mean()))
                for s, x in g.groupby("site_id")}

    L += ["", "# Exploratory clean controls (5-member prediction ensemble)\n",
          "The final column is a descriptive paired station-RMSE effect with whole-HUC2 "
          "bootstrap uncertainty. It is not a multiplicity-adjusted component-necessity test.\n",
          "| variant | h1 | h3 | h7 | median h3 variant−full [95% CI] |",
          "|---|---|---|---|---|"]
    full_ps = {h: ps_rmse(full, h) for h in C.HORIZONS}
    cluster_map = huc2_cluster_map(load_station_registry(
        ROOT / "data_usgs" / "station_registry_v1.csv"))
    for variant_index, name in enumerate(ABLATIONS):
        df = abl_all[(abl_all.model == name) & (abl_all.split == "test")]
        meds, h3_effect = [], None
        for h in C.HORIZONS:
            ps = ps_rmse(df, h)
            meds.append(np.median([ps[s] for s in stations if s in ps]))
            if h == 3 and name != "ThermoRoute":
                common = [s for s in stations if s in ps and s in full_ps[3]]
                effect = np.array([ps[s] - full_ps[3][s] for s in common])
                cluster = np.array([
                    cluster_map.get(s, f"UNMAPPED:{s}") for s in common
                ])
                h3_effect = cluster_bootstrap_paired_effect(
                    effect, cluster, n_boot=10000, seed=1400 + variant_index,
                )
        effect_text = (
            "—" if h3_effect is None else
            f"{h3_effect['effect']:+.3f} [{h3_effect['ci_low']:+.3f},"
            f"{h3_effect['ci_high']:+.3f}]"
        )
        L.append(
            f"| {name} | {meds[0]:.3f} | {meds[1]:.3f} | {meds[2]:.3f} | "
            f"{effect_text} |"
        )

    (C.REPORTS / "rigor.md").write_text("\n".join(L))
    log("DONE")
    print("\n" + "\n".join(L), flush=True)


if __name__ == "__main__":
    main()
