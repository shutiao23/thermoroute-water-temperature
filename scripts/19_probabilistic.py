#!/usr/bin/env python3
"""Stage 19 — probabilistic verification + multi-metric reporting.

The 2025 HESS review (Corona/Feigl, 29:2521) documents that (a) ML stream-temp
studies report almost only point errors, and (b) multi-metric reporting
(NSE/KGE/MAE/PBIAS together) is the expected standard. This stage closes both:

  1. Calibrated probabilistic scores — the SAME split-conformal (CQR) wrapper is
     applied to ThermoRoute, LightGBM AND LSTM (fair foil): PICP, MPIW, CRPS.
  2. High-temperature exceedance verification — Brier, Brier skill vs climatology,
     AUROC, and a reliability diagram (the calibrated-warning contribution).
  3. Multi-metric point table — RMSE/MAE/NSE/KGE/PBIAS at every lead, pooled and
     region-weighted, contextualised against the review's Table-2 thresholds.

Writes: outputs/tables/{multi_metric.csv, probabilistic_scores.csv},
        outputs/figures/fig_reliability.png, outputs/reports/probabilistic.md

Run:  PYTHONPATH=src python3 scripts/19_probabilistic.py
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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import metrics as M
from thermoroute import conformal as CF

PANEL = ROOT / "data_usgs" / "panel_usgs_100.parquet"
HUC = C.TABLES / "usgs_stations_with_huc.csv"
V2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
PROB_MODELS = ["ThermoRoute", "LightGBM", "LSTM"]
POINT_MODELS = ["Persistence", "DampedPersistence", "LightGBM", "LSTM", "ThermoRoute"]
COL = {"ThermoRoute": "#B3132B", "LightGBM": "#185FA5", "LSTM": "#6A4C93"}


def station_thresholds():
    panel = pd.read_parquet(PANEL); panel["DATE"] = pd.to_datetime(panel["DATE"])
    m = D.split_masks(panel["DATE"])
    tr = panel.loc[m.train]
    return {s: float(tr[tr.site_id == s].WTEMP.quantile(C.EXCEEDANCE_QUANTILE))
            for s in panel.site_id.unique()}


def ensemble(v2, model):
    """Seed-mean q/p per (site,horizon,issue,split) for one model."""
    sub = v2[v2.model == model]
    g = sub.groupby(["site_id", "horizon", "issue_date", "split"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        q05=("q05", "mean"), q50=("q50", "mean"), q95=("q95", "mean"),
        p_exceed=("p_exceed", "mean"), target_date=("target_date", "first"))
    return g


def calibrated_test(ens):
    """Apply per-(site,horizon) split-conformal offsets from calib to test."""
    cal = ens[ens.split == "calib"]
    off = CF.cqr_offsets(cal, alpha=0.10) if not cal.empty and cal.q05.notna().any() else {}
    te = ens[ens.split == "test"].copy()
    if off:
        te = CF.apply_cqr(te, off)
    return te


def main():
    thr = station_thresholds()
    v2 = pd.read_parquet(V2)
    meta = pd.read_csv(HUC).set_index("site_id")
    huc = meta["huc2_name"].to_dict() if "huc2_name" in meta.columns else {}

    # ---- 1+2. probabilistic + exceedance on calibrated ensembles ---------- #
    prob_rows, rel = [], {}
    for model in PROB_MODELS:
        ens = ensemble(v2, model)
        te = calibrated_test(ens)
        if te.empty:
            continue
        te["thr"] = te.site_id.map(thr)
        te["ybin"] = (te.y_true > te.thr).astype(int)
        rel[model] = te
        for h in C.HORIZONS:
            g = te[te.horizon == h]
            y = g.y_true.to_numpy(float)
            quants = {0.05: g.q05.to_numpy(float), 0.50: g.q50.to_numpy(float),
                      0.95: g.q95.to_numpy(float)}
            ps = M.probabilistic_scores(y, quants)
            ev = M.event_scores(g.ybin.to_numpy(int), g.p_exceed.to_numpy(float)) \
                if g.p_exceed.notna().any() else {}
            prob_rows.append({"model": model, "horizon": h,
                              "PICP": ps["PICP"], "MPIW": ps["MPIW"], "CRPS": ps["CRPS"],
                              "Brier": ev.get("BRIER", np.nan),
                              "BrierSkill": ev.get("BRIER_SKILL", np.nan),
                              "AUROC": ev.get("AUROC", np.nan),
                              "base_rate": ev.get("BASE_RATE", np.nan)})
    prob = pd.DataFrame(prob_rows)
    prob.to_csv(C.TABLES / "probabilistic_scores.csv", index=False)

    # ---- 3. multi-metric point table (pooled + region-weighted) ----------- #
    mm_rows = []
    for model in POINT_MODELS:
        ens = ensemble(v2, model)
        te = ens[ens.split == "test"]
        if te.empty:
            continue
        for h in C.HORIZONS:
            g = te[te.horizon == h]
            y, yhat = g.y_true.to_numpy(float), g.y_pred.to_numpy(float)
            sc = M.point_scores(y, yhat)
            # region-weighted RMSE (mean of per-HUC2 medians of per-station RMSE)
            perst = {s: M.rmse(x.y_true.to_numpy(float), x.y_pred.to_numpy(float))
                     for s, x in g.groupby("site_id")}
            reg = {}
            for s, r in perst.items():
                reg.setdefault(huc.get(s, "?"), []).append(r)
            rw = float(np.mean([np.median(v) for v in reg.values()]))
            mm_rows.append({"model": model, "horizon": h, "RMSE": sc["RMSE"],
                            "MAE": sc["MAE"], "NSE": sc["NSE"], "KGE": sc["KGE"],
                            "PBIAS": sc["PBIAS"], "RMSE_region_wtd": rw})
    mm = pd.DataFrame(mm_rows)
    mm.to_csv(C.TABLES / "multi_metric.csv", index=False)

    # ---- reliability diagram (exceedance), all horizons pooled ------------ #
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1, label="perfect reliability")
    bins = np.linspace(0, 1, 11)
    for model in PROB_MODELS:
        if model not in rel:
            continue
        g = rel[model]
        p = g.p_exceed.to_numpy(float); yb = g.ybin.to_numpy(int)
        ok = ~np.isnan(p)
        p, yb = p[ok], yb[ok]
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 30:
                xs.append(p[m].mean()); ys.append(yb[m].mean())
        ax.plot(xs, ys, "o-", color=COL[model], lw=1.6, ms=5, label=model)
    ax.set_xlabel("forecast exceedance probability")
    ax.set_ylabel("observed exceedance frequency")
    ax.set_title("Reliability of the high-temperature warning")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_aspect("equal")
    fig.savefig(C.FIGURES / "fig_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----------------------------------------------------------- #
    L = ["# Probabilistic + multi-metric verification\n",
         "The same split-conformal (CQR) wrapper is applied to all three learners "
         "so the comparison is fair; ThermoRoute additionally carries the "
         "Proposition-1 safety floor and an interpretable lag router that a "
         "LightGBM/LSTM cannot.\n",
         "## Calibrated probabilistic scores (90% intervals; CRPS lower = sharper)\n",
         "| model | h | PICP | MPIW | CRPS | Brier skill | AUROC |",
         "|---|---|---|---|---|---|---|"]
    for model in PROB_MODELS:
        for h in C.HORIZONS:
            r = prob[(prob.model == model) & (prob.horizon == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            L.append(f"| {model} | {h} | {r.PICP:.3f} | {r.MPIW:.2f} | {r.CRPS:.3f} | "
                     f"{r.BrierSkill:+.3f} | {r.AUROC:.3f} |")
    L += ["", "## Multi-metric point scores (pooled blind test)\n",
          "_Review Table-2 field norms: median RMSE≈1.35 °C, NSE≈0.93, MAE≈1.09 °C._\n",
          "| model | h | RMSE | MAE | NSE | KGE | PBIAS | RMSE (region-wtd) |",
          "|---|---|---|---|---|---|---|---|"]
    for model in POINT_MODELS:
        for h in C.HORIZONS:
            r = mm[(mm.model == model) & (mm.horizon == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            L.append(f"| {model} | {h} | {r.RMSE:.3f} | {r.MAE:.3f} | {r.NSE:.3f} | "
                     f"{r.KGE:.3f} | {r.PBIAS:+.1f} | {r.RMSE_region_wtd:.3f} |")
    L += ["", "![reliability](../figures/fig_reliability.png)"]
    (C.REPORTS / "probabilistic.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {C.REPORTS/'probabilistic.md'} + fig_reliability.png "
          f"+ {C.TABLES/'multi_metric.csv'} + probabilistic_scores.csv")


if __name__ == "__main__":
    main()
