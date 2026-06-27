#!/usr/bin/env python3
"""Stage 8 — forecast decision value (cost–loss / Relative Economic Value).

Turns the calibrated high-temperature exceedance probabilities into a
management-relevant metric: for which decision-makers (cost–loss ratios) does
ThermoRoute's probabilistic warning beat a persistence-based deterministic
warning? Produces REV curves (Fig 11) and a value table.

Run:  PYTHONPATH=src python3 scripts/08_decision_value.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import results as R
from thermoroute import decision as DEC

# model -> (predictions filter, is_probabilistic)
MODELS = [
    ("ThermoRoute", dict(scope="joint", feature_set="V3"), True, "#185FA5"),
    ("LightGBM", dict(feature_set="V3"), True, "#7F77DD"),
    ("DampedPersistence", dict(), False, "#BA7517"),
    ("Persistence", dict(), False, "#888780"),
]


def thresholds():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel.parquet")
    masks = D.split_masks(panel["DATE"])
    return R.exceedance_thresholds(panel, masks)


def subset(pred, model, flt):
    m = (pred.model == model) & (pred.split == "test")
    for k, v in flt.items():
        m &= pred[k] == v
    return pred[m]


def events_and_score(df, thr, probabilistic):
    df = df.copy()
    df["theta"] = df.site_id.map(thr)
    # average over seeds per (site,issue_date)
    g = df.groupby(["site_id", "issue_date"]).agg(
        y_true=("y_true", "mean"), y_pred=("y_pred", "mean"),
        p_exceed=("p_exceed", "mean"), theta=("theta", "first")).reset_index()
    events = (g.y_true.to_numpy() > g.theta.to_numpy()).astype(int)
    if probabilistic:
        score = g.p_exceed.to_numpy()
    else:
        score = (g.y_pred.to_numpy() > g.theta.to_numpy()).astype(float)
    return events, score


def main():
    pred = pd.read_parquet(C.PREDICTIONS / "predictions.parquet")
    thr = thresholds()
    alphas = np.linspace(0.01, 0.99, 99)

    rows, curves = [], {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, h in zip(axes, C.HORIZONS):
        for model, flt, prob, col in MODELS:
            df = subset(pred, model, flt)
            df = df[df.horizon == h]
            if df.empty:
                continue
            ev, sc = events_and_score(df, thr, prob)
            rev = DEC.rev_curve(ev, sc, alphas, probabilistic=prob)
            curves[(model, h)] = rev
            ls = "-" if prob else "--"
            ax.plot(alphas, np.clip(rev, -0.2, 1), ls, color=col, lw=1.6,
                    label=model + ("" if prob else " (det.)"))
            summ = DEC.value_summary(ev, sc, probabilistic=prob)
            summ.update({"model": model, "horizon": h})
            rows.append(summ)
        ax.axhline(0, color="#888780", lw=0.7)
        ax.set_title(f"h = {h} d"); ax.set_xlabel("cost–loss ratio α = C/L")
        ax.set_ylim(-0.2, 1); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Relative Economic Value")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Decision value of high-temperature exceedance warnings (blind test)", y=1.03)
    fig.savefig(C.FIGURES / "fig11_rev_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(C.FIGURES / "fig11_rev_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", C.FIGURES / "fig11_rev_curves.png", flush=True)

    val = pd.DataFrame(rows)
    val.to_csv(C.TABLES / "decision_value.csv", index=False)

    # markdown table
    L = ["# Decision value — Relative Economic Value (blind test)\n",
         "Peak REV and REV at representative cost–loss ratios α. "
         "1.0 = perfect-forecast value; 0 = no better than climatology. "
         "Probabilistic forecasts sweep the optimal threshold; persistence is a "
         "fixed deterministic warning.\n",
         "| model | h | base rate | REV_max | α* | REV@0.05 | REV@0.1 | REV@0.2 | REV@0.5 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in val.sort_values(["horizon", "model"]).iterrows():
        L.append(f"| {r['model']} | {int(r['horizon'])} | {r['base_rate']:.3f} | "
                 f"{r['REV_max']:.3f} | {r['alpha_at_max']:.2f} | {r['REV@0.05']:.3f} | "
                 f"{r['REV@0.1']:.3f} | {r['REV@0.2']:.3f} | {r['REV@0.5']:.3f} |")
    (C.TABLES / "decision_value.md").write_text("\n".join(L))
    print("wrote", C.TABLES / "decision_value.md", flush=True)


if __name__ == "__main__":
    main()
