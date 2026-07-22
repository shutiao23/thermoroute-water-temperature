#!/usr/bin/env python3
"""Stage 15 — stratified / subgroup skill analysis (§S2 of the manuscript).

Answers "is the headline carried by one region or river type?" by reporting the
per-station transfer-free skill (ThermoRoute vs persistence and damped) broken
down by (a) verified HUC2 hydrologic region and (b) drainage-area terciles,
plus a REGION-WEIGHTED summary (mean of per-HUC2
medians) so geographically over-represented regions (PNW) do not dominate.
Reads the v2 predictions; no retraining.

Run:  PYTHONPATH=src python3 scripts/15_stratified.py
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
from thermoroute.spatial import huc2_cluster_map, load_station_registry

PRED = C.PREDICTIONS / "usgs_predictions_v2.parquet"
PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"


def per_station(pred, model, h, ensemble=False):
    sub = pred[(pred.model == model) & (pred.split == "test") & (pred.horizon == h)]
    if ensemble:
        sub = sub.groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
    return {s: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
            for s, g in sub.groupby("site_id")}


def main():
    pred = R.load_route_a_predictions(
        PRED, root=ROOT, panel_path=PANEL, registry_path=REGISTRY
    )
    registry = load_station_registry(REGISTRY)
    huc = huc2_cluster_map(registry)
    meta = registry.set_index("site_no")
    # drainage-area terciles
    da = meta["drain_area_va"].to_dict()
    da_vals = np.array([v for v in da.values() if np.isfinite(v)])
    q1, q2 = np.quantile(da_vals, [1/3, 2/3])

    def da_group(s):
        v = da.get(s, np.nan)
        if not np.isfinite(v):
            return "unknown"
        return "small" if v <= q1 else ("medium" if v <= q2 else "large")

    L = ["# Stratified skill analysis (§S2) — is the headline carried by one subgroup?\n",
         "Per-station 2019–2020 development-evaluation skill of the ThermoRoute "
         "ensemble versus persistence. HUC2 labels are used only where the frozen "
         "registry has an exact site-number match; missing metadata is labelled "
         "unverified rather than inferred from legacy aliases. No name-based "
         "regulated/free-flowing classification is reported.\n"]

    for h in C.HORIZONS:
        tr = per_station(pred, "ThermoRoute", h, ensemble=True)
        per = per_station(pred, "Persistence", h)
        dmp = per_station(pred, "DampedPersistence", h)
        sts = [s for s in tr if s in per and s in dmp]
        df = pd.DataFrame({
            "site_id": sts,
            "skill_persist": [1 - tr[s]/per[s] for s in sts],
            "skill_damped": [1 - tr[s]/dmp[s] for s in sts],
            "huc2": [huc.get(s, f"UNMAPPED:{s}") for s in sts],
            "da_group": [da_group(s) for s in sts],
        })
        L.append(f"\n## Horizon {h} d  (n={len(sts)})\n")
        # region-weighted headline
        per_huc_med = df.groupby("huc2").skill_persist.median()
        L.append(f"- **Pooled median** skill vs persistence: {df.skill_persist.median():+.3f}")
        L.append(f"- **Region-weighted** (mean of per-HUC2 medians, "
                 f"{df.huc2.nunique()} regions): {per_huc_med.mean():+.3f} "
                 f"→ {'robust to region weighting' if abs(per_huc_med.mean()-df.skill_persist.median())<0.03 else 'sensitive to region mix'}")
        # by drainage area
        for lab in ("small", "medium", "large"):
            g = df[df.da_group == lab]
            if len(g):
                L.append(f"- drainage {lab}: n={len(g)}, median skill vs persist "
                         f"{g.skill_persist.median():+.3f}")
        # per-HUC2 table (compact)
        L.append("\n| HUC2 region | n | median skill vs persist | vs damped |")
        L.append("|---|---|---|---|")
        for hc, g in df.groupby("huc2"):
            L.append(f"| {hc} | {len(g)} | {g.skill_persist.median():+.3f} | "
                     f"{g.skill_damped.median():+.3f} |")

    out = C.REPORTS / "stratified.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
