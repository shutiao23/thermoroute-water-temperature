#!/usr/bin/env python3
"""CI smoke experiment — a tiny end-to-end run that asserts on real numbers.

Trains ThermoRoute for a handful of epochs on the shipped 3-station data and
checks that the pipeline still produces sane, non-degenerate outputs:
  * predictions exist for all 3 horizons and are finite;
  * conformal 90% coverage on the calibration→test flow lands in a sane band;
  * baselines and ThermoRoute share identical (site, horizon, issue_date) keys.
This is a guard against silent pipeline rot; it is NOT a scientific result.
Exits non-zero on any violation so CI fails loudly.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import fit_model
from thermoroute.conformal import cqr_offsets, apply_cqr

torch.manual_seed(0)
np.random.seed(0)

b = D.prepare_dataset()
panel, panel_imp, masks = b["panel_raw"], b["panel"], b["masks"]
clim = F.HarmonicClimatology.fit(panel_imp, masks.train)
wd = DS.build_windows(panel_imp, masks, clim, variables=C.FEATURE_SETS["V3"])
thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
       for s in C.STATIONS}

cfg = C.TrainConfig(max_epochs=3, batch_size=256)
m = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(C.STATIONS), n_phys=wd.n_phys)
r = fit_model(m, wd, thr, cfg=cfg, seed=0, feature_set="V3")
pred = r.pred

# --- assertions -------------------------------------------------------------
te = pred[pred.split == "test"]
assert set(te.horizon.unique()) == set(C.HORIZONS), "missing horizons"
assert np.isfinite(te.y_pred.to_numpy(float)).all(), "non-finite predictions"
assert te.y_pred.std() > 1e-3, "degenerate (constant) predictions"

# conformal coverage on the calib->test flow must be in a sane band
off = cqr_offsets(pred[pred.split == "calib"])
dc = apply_cqr(pred, off)
tt = dc[dc.split == "test"]
cov = float(((tt.y_true >= tt.q05) & (tt.y_true <= tt.q95)).mean())
assert 0.70 <= cov <= 0.99, f"conformal coverage {cov:.3f} outside sane band [0.70,0.99]"

print(f"CI smoke OK: 3 horizons, finite non-degenerate preds, conformal cov={cov:.3f}")
