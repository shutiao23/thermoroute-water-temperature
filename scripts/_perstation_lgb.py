#!/usr/bin/env python3
"""M4 — add a per-station LightGBM (the strongest LightGBM config) to v2.

The headline compared ThermoRoute (per-station embeddings) only to a single
GLOBAL LightGBM. This adds a per-station LightGBM so parity is judged against the
stronger of the two learned baselines on the station-adaptation axis. Splices
'LightGBM-perstation' into usgs_predictions_v2.parquet aligned to the shared
test keys.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from thermoroute import config as C, data as D, features as F, baselines as B

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
t0 = time.time()
b = D.prepare_dataset_from_panel(str(ROOT / "data_usgs/panel_usgs_100.parquet"))
panel, pi, masks, stations = b["panel_raw"], b["panel"], b["masks"], b["stations"]
clim = F.HarmonicClimatology.fit(panel, masks.train)
thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9)) for s in stations}
print(f"[{time.time()-t0:.0f}s] building per-station LightGBM tabulars...", flush=True)
tabs = {}
for h in C.HORIZONS:
    tab = F.attach_split(F.build_tabular(pi, h, USGS_VARS, clim,
                         drop_feature_nans=False, require_observed_target=True))
    for c in F.feature_columns(tab):
        tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
    tabs[h] = tab
lgb = B.run_lightgbm(tabs, thr, feature_set="USGS")
lgb["model"] = "LightGBM-perstation"
print(f"[{time.time()-t0:.0f}s] fit per-station LGB. splicing into v2...", flush=True)

pred = C.PREDICTIONS / "usgs_predictions_v2.parquet"
allp = pd.read_parquet(pred)
allp = allp[allp.model != "LightGBM-perstation"]
tr_keys = set(zip(*[allp[(allp.model == "ThermoRoute") & (allp.split == "test")][c]
                    for c in ["site_id", "horizon", "issue_date"]]))
lt = lgb[lgb.split == "test"].copy()
keep = lt[pd.Series(list(zip(lt.site_id, lt.horizon, lt.issue_date)), index=lt.index).isin(tr_keys)]
allp = pd.concat([allp, keep], ignore_index=True)
allp.to_parquet(pred)
for h in C.HORIZONS:
    g = keep[keep.horizon == h]
    print(f"  perstation LGB h{h}: median-per-station RMSE "
          f"{np.median([np.sqrt(((x.y_pred-x.y_true)**2).mean()) for _,x in g.groupby('site_id')]):.3f}", flush=True)
print(f"[{time.time()-t0:.0f}s] DONE, spliced {len(keep)} rows", flush=True)
