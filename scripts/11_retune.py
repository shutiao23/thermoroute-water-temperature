#!/usr/bin/env python3
"""Stage 11 — retune the residual bound (delta_scale) on the large sample.

On the 3-station data the residual was clamped tight (±0.4 °C) for stability; on
the 40-station large sample with real headroom a looser clamp may let ThermoRoute
add skill at 3–7 days, where a strong LightGBM currently leads. This trains one
seed per delta_scale and reports median per-station RMSE against the known damped
(1.261/1.528 at h3/h7) and LightGBM (1.168/1.486) references.

Run:  PYTHONPATH=src python3 scripts/11_retune.py --scales 0.4 1.0 1.5
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

torch.set_num_threads(8)

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH")
CFG = C.TrainConfig(batch_size=2048)
_t0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="*", type=float, default=[0.4, 1.0, 1.5])
    ap.add_argument("--panel", default=str(ROOT / "data_usgs" / "panel_usgs.parquet"))
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    stations = tuple(sorted(panel.site_id.unique()))
    C.STATIONS = stations
    C.UPSTREAM = {s: None for s in stations}
    for v in C.ALL_VARS:
        panel[f"{v}_observed"] = panel[v].notna()
    masks = D.split_masks(panel["DATE"])
    panel_imp = D.Imputer.fit(panel, masks.train).transform(panel)
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    wd = DS.build_windows(panel_imp, masks, clim, variables=USGS_VARS,
                          require_observed_target=True)
    idx = wd.idx("test")

    def med_rmse(pred):  # median over stations of per-station RMSE, per horizon
        site = np.array([C.STATIONS[i] for i in wd.station[idx]])
        out = {}
        for hi, h in enumerate(wd.horizons):
            y, yp = wd.y[idx][:, hi], pred[hi]
            per = [np.sqrt(np.mean((y[site == s] - yp[site == s]) ** 2))
                   for s in np.unique(site)]
            out[h] = float(np.median(per))
        return out

    print(f"refs: damped h3/h7=1.261/1.528, LightGBM=1.168/1.486", flush=True)
    rows = []
    for ds in args.scales:
        te = time.time()
        model = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                            n_phys=wd.n_phys, delta_scale=ds)
        res = fit_model(model, wd, thr, cfg=CFG, seed=0, feature_set="USGS")
        sub = res.pred[res.pred.split == "test"]
        preds = [sub[sub.horizon == h].set_index(["site_id", "issue_date"]).y_pred
                 .reindex(pd.MultiIndex.from_arrays(
                     [np.array([C.STATIONS[i] for i in wd.station[idx]]),
                      wd.issue_date[idx]])).to_numpy() for h in wd.horizons]
        m = med_rmse(preds)
        rows.append({"delta_scale": ds, **{f"h{h}": m[h] for h in wd.horizons}})
        print(f"delta_scale={ds}: h1={m[1]:.3f} h3={m[3]:.3f} h7={m[7]:.3f} "
              f"({time.time()-te:.0f}s, {res.epochs+1}ep)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "usgs_retune.csv", index=False)
    print("\n" + df.to_string(index=False), flush=True)
    print(f"total {time.time()-_t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
