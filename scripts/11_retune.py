#!/usr/bin/env python3
"""Stage 11 — retune the residual bound (delta_scale) on the validation split.

On the 3-station data the residual was clamped tight (±0.4 °C) for stability; on
the large sample with forecast headroom a looser clamp may let ThermoRoute add
skill at 3–7 days. This script trains one seed per delta_scale and reports
median per-station RMSE on the **validation split (2016–2017) only**. The blind-
test years (2019–2020) are never read here — the chosen value is then committed
to ``config.TrainConfig`` and propagated to the headline experiments.

Selection rule (fixed before running the sweep): choose the delta_scale that
minimises the unweighted mean over horizons (1, 3, 7 d) of the median
per-station validation RMSE.

Preparation, feature set (incl. WDSP) and batch size mirror
``scripts/09_usgs_experiment.py`` exactly, so the value selected here is
selected for the headline configuration, not a surrogate one.

An earlier version of this script evaluated on the test split; those results
are deprecated and replaced by val-based numbers (see commit history).

Run:  PYTHONPATH=src python3 scripts/11_retune.py --scales 0.4 1.0 1.5 2.0
Parallel workers (one per subset of scales, WORKER_THREADS threads each):
  WORKER_THREADS=4 PYTHONPATH=src python3 scripts/11_retune.py \
      --scales 0.4 1.0 --out usgs_retune_val_a.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["OMP_NUM_THREADS"] = os.environ.get(
    "WORKER_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))

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

torch.set_num_threads(int(os.environ.get("WORKER_THREADS",
                                         os.environ.get("OMP_NUM_THREADS", "8"))))

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

# Parity with scripts/09_usgs_experiment.py: same variables (incl. gridMET wind)
# and the same batch size, so delta_scale is tuned for the headline model.
USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
CFG = C.TrainConfig(batch_size=1536)
_t0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="*", type=float, default=[0.4, 1.0, 1.5, 2.0])
    ap.add_argument("--panel", default=str(ROOT / "data_usgs" / "panel_usgs_100.parquet"))
    ap.add_argument("--out", default="usgs_retune.csv",
                    help="CSV filename under outputs/tables/ (workers use suffixes)")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed (multi-seed tie-break for near-tied scales)")
    args = ap.parse_args()

    b = D.prepare_dataset_from_panel(args.panel)          # same prep as 09
    panel, panel_imp, masks = b["panel_raw"], b["panel"], b["masks"]
    stations = b["stations"]
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    wd = DS.build_windows(panel_imp, masks, clim, variables=USGS_VARS,
                          require_observed_target=True)
    # IMPORTANT: hyperparameter selection MUST happen on the validation split,
    # never on test. Earlier versions of this script read wd.idx("test") which
    # let test labels leak into the chosen delta_scale; the corresponding numbers
    # are deprecated and have been overwritten by this val-based sweep.
    idx = wd.idx("val")
    eval_split = "val"

    def med_rmse(pred):  # median over stations of per-station RMSE, per horizon
        site = np.array([C.STATIONS[i] for i in wd.station[idx]])
        out = {}
        for hi, h in enumerate(wd.horizons):
            y, yp = wd.y[idx][:, hi], pred[hi]
            per = [np.sqrt(np.mean((y[site == s] - yp[site == s]) ** 2))
                   for s in np.unique(site) if (site == s).sum() > 0]
            out[h] = float(np.median(per)) if per else float("nan")
        return out

    print("# delta_scale sweep on VALIDATION split (2016-2017). Test years stay sealed.", flush=True)
    rows = []
    for ds in args.scales:
        te = time.time()
        model = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                            n_phys=wd.n_phys, delta_scale=ds)
        res = fit_model(model, wd, thr, cfg=CFG, seed=args.seed, feature_set="USGS")
        sub = res.pred[res.pred.split == eval_split]
        preds = [sub[sub.horizon == h].set_index(["site_id", "issue_date"]).y_pred
                 .reindex(pd.MultiIndex.from_arrays(
                     [np.array([C.STATIONS[i] for i in wd.station[idx]]),
                      wd.issue_date[idx]])).to_numpy() for h in wd.horizons]
        m = med_rmse(preds)
        rows.append({"delta_scale": ds, "split": eval_split, "seed": args.seed,
                     **{f"h{h}_val": m[h] for h in wd.horizons},
                     "mean_val": float(np.mean([m[h] for h in wd.horizons]))})
        print(f"delta_scale={ds}: val h1={m[1]:.3f} val h3={m[3]:.3f} val h7={m[7]:.3f} "
              f"mean={rows[-1]['mean_val']:.4f} ({time.time()-te:.0f}s, {res.epochs+1}ep)",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / args.out, index=False)
    print("\n" + df.to_string(index=False), flush=True)
    best = df.loc[df.mean_val.idxmin()]
    print(f"\nbest delta_scale (min mean-over-horizons median val RMSE, "
          f"within this worker's scales): {best.delta_scale}", flush=True)
    print(f"total {time.time()-_t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
