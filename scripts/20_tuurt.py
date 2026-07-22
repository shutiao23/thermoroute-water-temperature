#!/usr/bin/env python3
"""Stage 20 — assemble temporal and gauged-site transfer diagnostics.

The model requires water temperature observed through the issue day.  Therefore
neither held-site arm is an ungauged forecast, even when model parameters and
preprocessing statistics are fitted without the held region.  This script keeps
that distinction explicit:

  * Temporal — 2019–2020 already-inspected development evaluation at the same
    stations.
  * Random held-site — a warm-start diagnostic; it is not zero-shot because
    legacy preprocessing used held-site history.
  * Held-region gauged transfer — model/preprocessing fit without each held HUC2,
    while issue-time WTEMP history remains an input.

Writes outputs/reports/tuurt.md.

Run:  PYTHONPATH=src python3 scripts/20_tuurt.py
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

from thermoroute import config as C
from thermoroute import results as R

V2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def temporal_arm():
    """TR 5-seed ensemble per-station median skill vs persistence/damped (in-domain
    future years, same stations)."""
    v2 = R.load_route_a_predictions(
        V2, root=ROOT, panel_path=PANEL, registry_path=REGISTRY
    )
    te = v2[v2.split == "test"]
    out = {}
    for h in C.HORIZONS:
        keys = ["site_id", "horizon", "issue_date", "target_date"]

        def keyed(model):
            g = te[(te.model == model) & (te.horizon == h)]
            if g.empty:
                raise RuntimeError(f"missing {model} temporal predictions at h={h}")
            if (g.groupby(keys).y_true.nunique(dropna=False) > 1).any():
                raise AssertionError(f"{model} y_true differs across seeds")
            return g.groupby(keys).agg(
                y_pred=("y_pred", "mean"), y_true=("y_true", "first")
            ).sort_index()

        frames = {name: keyed(name) for name in
                  ("ThermoRoute", "Persistence", "DampedPersistence")}
        common = frames["ThermoRoute"].index
        for frame in frames.values():
            common = common.intersection(frame.index)
        if len(common) == 0:
            raise RuntimeError(f"no common temporal keys at h={h}")
        truth = frames["ThermoRoute"].loc[common, "y_true"].to_numpy(float)
        for name, frame in frames.items():
            if not np.allclose(frame.loc[common, "y_true"], truth, rtol=0, atol=1e-8):
                raise AssertionError(f"{name} has inconsistent truth on common keys")

        def per_station(model):
            aligned = frames[model].loc[common].reset_index()
            return {s: rmse(x.y_true.to_numpy(float), x.y_pred.to_numpy(float))
                    for s, x in aligned.groupby("site_id")}

        tr = per_station("ThermoRoute")
        pe = per_station("Persistence")
        dm = per_station("DampedPersistence")
        sts = [s for s in tr if s in pe and s in dm]
        out[h] = {"skill_persist": float(np.median([1 - tr[s]/pe[s] for s in sts])),
                  "skill_damped": float(np.median([1 - tr[s]/dm[s] for s in sts])),
                  "n": len(sts)}
    return out


def main():
    temporal = temporal_arm()
    unseen = pd.read_csv(C.TABLES / "claim2_kfold_lgo.csv")
    region = pd.read_csv(C.TABLES / "region_transfer.csv", index_col=0)

    L = ["# Temporal and gauged-site transfer diagnostics\n",
         "Skill = 1 − RMSE(ThermoRoute)/RMSE(reference), median across held-out "
         "stations. Positive values are descriptive evidence for the stated arm, "
         "not proof of universal temporal or spatial generalisation. The model uses "
         "observed WTEMP through issue time, so no arm is labelled ungauged.\n",
         "| arm | protocol | h1 skill vs persist | h3 | h7 |",
         "|---|---|---|---|---|"]
    L.append(f"| **Temporal development** | already-inspected 2019–2020, seen stations (n={temporal[1]['n']}) | "
             f"{temporal[1]['skill_persist']:+.3f} | {temporal[3]['skill_persist']:+.3f} | "
             f"{temporal[7]['skill_persist']:+.3f} |")
    us = {h: unseen[unseen.horizon == h].skill_persist.mean() for h in C.HORIZONS}
    L.append(f"| **Random held-site warm start** | 4-fold; held-site preprocessing history present | "
             f"{us[1]:+.3f} | {us[3]:+.3f} | {us[7]:+.3f} |")
    rg = {h: region.loc[h, "skill_tr_persist"] for h in C.HORIZONS}
    L.append(f"| **Held-region gauged transfer** | leave-HUC2-region-out; issue-time WTEMP present | "
             f"{rg[1]:+.3f} | {rg[3]:+.3f} | {rg[7]:+.3f} |")

    L += ["", "### vs damped persistence (the harder reference)\n",
          "| arm | h1 | h3 | h7 |", "|---|---|---|---|"]
    L.append(f"| Temporal | {temporal[1]['skill_damped']:+.3f} | "
             f"{temporal[3]['skill_damped']:+.3f} | {temporal[7]['skill_damped']:+.3f} |")
    ud = {h: unseen[unseen.horizon == h].skill_damped.mean() for h in C.HORIZONS}
    L.append(f"| Random held-site warm start | {ud[1]:+.3f} | {ud[3]:+.3f} | {ud[7]:+.3f} |")
    rd = {h: region.loc[h, "skill_tr_damped"] for h in C.HORIZONS}
    L.append(f"| Held-region gauged transfer | {rd[1]:+.3f} | {rd[3]:+.3f} | {rd[7]:+.3f} |")

    all_values = [temporal[h][name] for h in C.HORIZONS
                  for name in ("skill_persist", "skill_damped")]
    all_values += list(us.values()) + list(ud.values()) + list(rg.values()) + list(rd.values())
    if all(np.isfinite(value) and value > 0 for value in all_values):
        verdict = ("Every displayed point estimate is positive, but uncertainty, "
                   "multiplicity, and the history-dependent gauged-site contract "
                   "still limit the conclusion.")
    else:
        verdict = ("At least one displayed estimate is non-positive or unavailable; "
                   "the corresponding transfer claim is not supported.")
    L += ["", verdict,
          "A true new-site confirmation is reported separately only after its "
          "metadata-only cohort, model bundle, inputs, and labels have been frozen."]
    (C.REPORTS / "tuurt.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {C.REPORTS/'tuurt.md'}")


if __name__ == "__main__":
    main()
