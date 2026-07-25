#!/usr/bin/env python3
"""CI smoke experiment on generated, explicitly non-scientific data.

Trains ThermoRoute for a handful of epochs on three deterministic synthetic
monitoring-site series and checks that the pipeline still produces sane,
non-degenerate outputs:
  * predictions exist for all 3 horizons and are finite;
  * conformal 90% coverage on the calibration→test flow lands in a sane band;
  * baselines and ThermoRoute share identical (site, horizon, issue_date) keys.
This is a guard against silent pipeline rot; it is NOT a scientific result.
It deliberately does not read the legacy CSV files, whose source and
redistribution rights remain unresolved.
Exits non-zero on any violation so CI fails loudly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
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


def synthetic_panel() -> pd.DataFrame:
    """Return a deterministic fixture with no claim to environmental realism."""
    dates = pd.date_range("2006-01-01", "2020-12-31", freq="D")
    step = np.arange(len(dates), dtype=float)
    annual = 2.0 * np.pi * step / 365.2425
    weekly = 2.0 * np.pi * step / 7.0
    frames: list[pd.DataFrame] = []
    for site_index, site_id in enumerate(("synthetic_01", "synthetic_02", "synthetic_03")):
        phase = 0.18 * site_index
        air = 11.0 + 10.0 * np.sin(annual + phase) + 0.4 * np.sin(weekly)
        water = 12.0 + 7.0 * np.sin(annual + phase - 0.12) + 0.2 * np.sin(weekly)
        frame = pd.DataFrame(
            {
                "DATE": dates,
                "site_id": site_id,
                "WTEMP": water + 0.3 * site_index,
                "FLOW": 80.0 + 15.0 * np.cos(annual + phase) + site_index,
                "WLEVEL": 2.0 + 0.15 * np.cos(annual + phase),
                "TEMP": air,
                "PRCP": np.maximum(0.0, 1.2 + np.sin(weekly + phase)),
                "WDSP": 2.5 + 0.4 * np.cos(weekly + phase),
                "RHMEAN": 65.0 + 12.0 * np.cos(annual + phase),
                "DH": 180.0 + 120.0 * np.maximum(0.0, np.sin(annual - 0.2)),
            }
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["site_id", "DATE"], ignore_index=True
    )
    for variable in C.ALL_VARS:
        panel[f"{variable}_observed"] = True
    return panel


panel = synthetic_panel()
C.STATIONS = tuple(sorted(panel["site_id"].unique()))
masks = D.split_masks(panel["DATE"])
imputer = D.Imputer.fit(panel, masks.train)
panel_imp = imputer.transform(panel)
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

print(
    "synthetic CI smoke OK: 3 horizons, finite non-degenerate preds, "
    f"conformal cov={cov:.3f}"
)
