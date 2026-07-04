#!/usr/bin/env python3
"""Stage 18 — full Relative Economic Value (REV) curves over the cost–loss grid.

'Skill does not equal value' (Modi et al. 2025, HESS 29:5593): two forecasters
can tie on RMSE yet differ sharply in decision value. This decides the
warm-threshold exceedance decision (act iff p>α) across the FULL cost–loss ratio
α∈(0,1), for ThermoRoute vs LightGBM vs LSTM (calibrated probabilities) vs
persistence / climatology (deterministic warnings). The headline is that
ThermoRoute's calibrated exceedance probabilities dominate the value curve even
where point RMSE ties.

Writes:
  * outputs/figures/fig_rev_curve.png  (one panel per lead)
  * outputs/tables/rev_curve.csv       (REV(α) for every model/lead)
  * outputs/reports/rev_curve.md       (REV_max + REV@{0.05,0.1,0.2,0.5})

Run:  PYTHONPATH=src python3 scripts/18_rev_curve.py
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
from thermoroute.decision import rev_curve

PANEL = ROOT / "data_usgs" / "panel_usgs_100.parquet"
V2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
GRID = np.linspace(0.01, 0.99, 99)
# probabilistic models expose a calibrated p_exceed; deterministic ones warn when
# their point forecast crosses the station threshold.
PROB = {"ThermoRoute": "#B3132B", "LightGBM": "#185FA5", "LSTM": "#6A4C93"}
DETERM = {"Persistence": "#777777", "Climatology": "#2E8B57"}
COL = {**PROB, **DETERM}


def station_thresholds():
    panel = pd.read_parquet(PANEL); panel["DATE"] = pd.to_datetime(panel["DATE"])
    m = D.split_masks(panel["DATE"])
    tr = panel.loc[m.train]
    return {s: float(tr[tr.site_id == s].WTEMP.quantile(C.EXCEEDANCE_QUANTILE))
            for s in panel.site_id.unique()}


def main():
    thr = station_thresholds()
    v2 = pd.read_parquet(V2)
    te = v2[v2.split == "test"].copy()
    te["thr"] = te.site_id.map(thr)
    te["event"] = (te.y_true > te.thr).astype(int)

    records = []
    fig, axes = plt.subplots(1, len(C.HORIZONS), figsize=(4.2 * len(C.HORIZONS), 4), sharey=True)
    for ax, h in zip(axes, C.HORIZONS):
        for model, color in COL.items():
            sub = te[(te.model == model) & (te.horizon == h)]
            if sub.empty:
                continue
            probabilistic = model in PROB
            if probabilistic:
                # seed-ensemble p_exceed per (site, issue)
                g = sub.groupby(["site_id", "issue_date"]).agg(
                    p=("p_exceed", "mean"), event=("event", "first")).reset_index()
                if g.p.isna().all():
                    continue
                score = g.p.to_numpy(float); events = g.event.to_numpy(int)
            else:
                g = sub.groupby(["site_id", "issue_date"]).agg(
                    yhat=("y_pred", "mean"), event=("event", "first"),
                    thr=("thr", "first")).reset_index()
                score = (g.yhat.to_numpy(float) > g.thr.to_numpy(float)).astype(float)
                events = g.event.to_numpy(int)
            rev = rev_curve(events, score, GRID, probabilistic=probabilistic)
            ax.plot(GRID, rev, color=color, lw=1.8,
                    label=model, ls="-" if probabilistic else "--")
            for a in (0.05, 0.10, 0.20, 0.50):
                r = rev_curve(events, score, np.array([a]), probabilistic)[0]
                records.append({"horizon": h, "model": model, "alpha": a, "REV": r})
            records.append({"horizon": h, "model": model, "alpha": "max",
                            "REV": float(np.nanmax(rev))})
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xlabel("cost–loss ratio α = C/L")
        ax.set_title(f"h = {h} d")
        ax.grid(alpha=0.25); ax.set_ylim(-0.05, 1.0)
        if h == C.HORIZONS[0]:
            ax.set_ylabel("Relative Economic Value")
            ax.legend(fontsize=8, loc="lower center")
    fig.suptitle("Decision value of the high-temperature exceedance warning "
                 "(calibrated probabilities vs deterministic warnings)", fontsize=11)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_rev_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rec = pd.DataFrame(records)
    rec.to_csv(C.TABLES / "rev_curve.csv", index=False)

    L = ["# Relative Economic Value (REV) of the exceedance warning — full cost–loss curve\n",
         "REV(α) for the decision 'protect iff p>α' against a high-temperature "
         "(train-q90) exceedance, over the full cost–loss grid. Probabilistic models "
         "(ThermoRoute/LightGBM/LSTM) use calibrated p_exceed; persistence/climatology "
         "issue a deterministic warning when their point forecast crosses the "
         "threshold. Framing follows Modi et al. 2025 (HESS 29:5593): value, not RMSE.\n",
         "| horizon | model | REV_max | REV@0.05 | REV@0.1 | REV@0.2 | REV@0.5 |",
         "|---|---|---|---|---|---|---|"]
    piv = rec.pivot_table(index=["horizon", "model"], columns="alpha", values="REV")
    order = ["ThermoRoute", "LightGBM", "LSTM", "DampedPersistence", "Persistence", "Climatology"]
    for h in C.HORIZONS:
        for model in order:
            if (h, model) not in piv.index:
                continue
            r = piv.loc[(h, model)]
            L.append(f"| {h} | {model} | {r.get('max', np.nan):.3f} | "
                     f"{r.get(0.05, np.nan):.3f} | {r.get(0.1, np.nan):.3f} | "
                     f"{r.get(0.2, np.nan):.3f} | {r.get(0.5, np.nan):.3f} |")
    (C.REPORTS / "rev_curve.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {C.REPORTS/'rev_curve.md'} + fig_rev_curve.png + rev_curve.csv")


if __name__ == "__main__":
    main()
