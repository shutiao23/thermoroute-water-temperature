#!/usr/bin/env python3
"""Stage 6 — publication figures (300 dpi PNG + vector PDF) from saved artifacts.

Reads the processed panel, the predictions table, the scores table and the
mechanism arrays; writes figures to ``outputs/figures/``.  Every figure guards
its own inputs so a partial pipeline still produces what it can.

Run:  PYTHONPATH=src python3 scripts/06_make_figures.py
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from thermoroute import config as C

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "medium", "figure.autolayout": False,
})

FIG = C.FIGURES
STCOLOR = {"b1": "#185FA5", "s2": "#0F6E56", "p3": "#993C1D"}
MODEL_ORDER = ["Persistence", "Climatology", "DampedPersistence", "Air2streamLite",
               "Ridge", "LightGBM", "GRU", "ThermoRoute"]
MODEL_COLOR = {"Persistence": "#888780", "Climatology": "#B4B2A9",
               "DampedPersistence": "#BA7517", "Air2streamLite": "#1D9E75",
               "Ridge": "#D4537E", "LightGBM": "#7F77DD", "GRU": "#5DCAA5",
               "ThermoRoute": "#185FA5"}


def _save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png", flush=True)


def load():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel.parquet")
    pred = pd.read_parquet(C.PREDICTIONS / "predictions.parquet") \
        if (C.PREDICTIONS / "predictions.parquet").exists() else None
    scores = pd.read_csv(C.TABLES / "scores_all.csv") \
        if (C.TABLES / "scores_all.csv").exists() else None
    expl = np.load(C.TABLES / "explain.npz", allow_pickle=True) \
        if (C.TABLES / "explain.npz").exists() else None
    return panel, pred, scores, expl


# --------------------------------------------------------------------------- #
def fig_study_area(panel):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.4),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    axL.set_title("a  Reservoir cascade & travel times")
    axL.axis("off"); axL.set_xlim(0, 10); axL.set_ylim(0, 10)
    pos = {"b1": (1.6, 7.8), "s2": (5.0, 5.0), "p3": (8.4, 2.2)}
    elev = {"b1": 2480, "s2": 1819, "p3": 989}
    for st, (x, y) in pos.items():
        box = FancyBboxPatch((x - 1.0, y - 0.7), 2.0, 1.4, boxstyle="round,pad=0.1",
                             fc=STCOLOR[st], ec="none", alpha=0.9)
        axL.add_patch(box)
        axL.text(x, y + 0.18, st, color="white", ha="center", fontsize=12, weight="bold")
        axL.text(x, y - 0.32, f"{elev[st]} m", color="white", ha="center", fontsize=8)
    for (a, b), lab in [(("b1", "s2"), "flow ~1 d\nthermal ~1 d"),
                        (("s2", "p3"), "flow ~1 d\nthermal ~9 d")]:
        xa, ya = pos[a]; xb, yb = pos[b]
        axL.add_patch(FancyArrowPatch((xa, ya - 0.7), (xb, yb + 0.7),
                      arrowstyle="-|>", mutation_scale=16, lw=2, color="#5F5E5A"))
        axL.text((xa + xb) / 2 + 0.7, (ya + yb) / 2, lab, fontsize=7.5, color="#444441")

    axR.set_title("b  Per-station WTEMP distribution")
    data = [panel[panel.site_id == st].WTEMP.values for st in C.STATIONS]
    bp = axR.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                     tick_labels=list(C.STATIONS), showfliers=False)
    for patch, st in zip(bp["boxes"], C.STATIONS):
        patch.set_facecolor(STCOLOR[st]); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("white")
    axR.set_ylabel("water temperature (°C)")
    axR.grid(axis="y", alpha=0.25)
    _save(fig, "fig1_study_area")


def fig_series_climatology(panel):
    fig, axes = plt.subplots(3, 2, figsize=(9, 6.4),
                             gridspec_kw={"width_ratios": [2.4, 1]})
    for i, st in enumerate(C.STATIONS):
        sub = panel[panel.site_id == st].sort_values("DATE")
        ax = axes[i, 0]
        ax.plot(sub.DATE, sub.WTEMP, lw=0.4, color=STCOLOR[st])
        for lo, hi, c in [(C.SPLIT.val, C.SPLIT.val, "#FAEEDA"),
                          (C.SPLIT.calib, C.SPLIT.calib, "#E1F5EE"),
                          (C.SPLIT.test, C.SPLIT.test, "#FBEAF0")]:
            ax.axvspan(pd.Timestamp(lo[0]), pd.Timestamp(hi[1]), color=c, alpha=0.7, lw=0)
        ax.set_ylabel(f"{st}\nWTEMP (°C)")
        if i < 2:
            ax.set_xticklabels([])
        ax.margins(x=0.01)
        axc = axes[i, 1]
        m_wt = sub.groupby(sub.DATE.dt.month).WTEMP.mean()
        m_air = sub.groupby(sub.DATE.dt.month).TEMP.mean()
        axc.plot(m_wt.index, m_wt.values, "-o", ms=3, color=STCOLOR[st], label="water")
        axc.plot(m_air.index, m_air.values, "--", color="#888780", lw=1, label="air")
        axc.set_xticks([1, 4, 7, 10])
        if i == 0:
            axc.legend(fontsize=7, frameon=False)
        if i < 2:
            axc.set_xticklabels([])
    axes[0, 0].set_title("a  Daily WTEMP 2006–2020 (shaded: val / calib / development eval)")
    axes[0, 1].set_title("b  Monthly climatology")
    axes[2, 1].set_xlabel("month")
    _save(fig, "fig2_series_climatology")


def _test_scores(scores):
    return scores[scores.split == "test"].copy()


def fig_results_heatmap(scores):
    s = _test_scores(scores)
    # headline comparison: one row per model, using the full feature set (V3)
    s = s[s.scope.isin(["per_station", "joint"]) & ~s.feature_set.isin(["V1", "V2"])]
    rmse = s.pivot_table(index="model", columns="horizon", values="RMSE", aggfunc="mean")
    skill = s[s.model != "Persistence"].pivot_table(
        index="model", columns="horizon", values="SKILL_RMSE", aggfunc="mean")
    rmse = rmse.reindex([m for m in MODEL_ORDER if m in rmse.index])
    skill = skill.reindex([m for m in MODEL_ORDER if m in skill.index])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    im1 = a1.imshow(rmse.values, cmap="YlOrRd", aspect="auto")
    a1.set_xticks(range(len(rmse.columns))); a1.set_xticklabels([f"{h}d" for h in rmse.columns])
    a1.set_yticks(range(len(rmse.index))); a1.set_yticklabels(rmse.index)
    a1.set_title("a  Development-evaluation RMSE (°C)")
    for (i, j), v in np.ndenumerate(rmse.values):
        a1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="black" if v < np.nanmax(rmse.values) * 0.6 else "white")
    fig.colorbar(im1, ax=a1, fraction=0.046)

    vmax = np.nanmax(np.abs(skill.values))
    im2 = a2.imshow(skill.values, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
    a2.set_xticks(range(len(skill.columns))); a2.set_xticklabels([f"{h}d" for h in skill.columns])
    a2.set_yticks(range(len(skill.index))); a2.set_yticklabels(skill.index)
    a2.set_title("b  Skill vs persistence (↑ better)")
    for (i, j), v in np.ndenumerate(skill.values):
        a2.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(im2, ax=a2, fraction=0.046)
    _save(fig, "fig3_results_heatmap")


def fig_skill_vs_horizon(scores):
    s = _test_scores(scores)
    s = s[s.scope.isin(["per_station", "joint"]) & (s.model != "Persistence")
          & ~s.feature_set.isin(["V1", "V2"])]
    g = s.groupby(["model", "horizon"]).SKILL_RMSE.mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in [x for x in MODEL_ORDER if x in g.model.unique()]:
        sub = g[g.model == m].sort_values("horizon")
        ax.plot(sub.horizon, sub.SKILL_RMSE, "-o", ms=4, color=MODEL_COLOR.get(m), label=m)
    ax.axhline(0, color="#888780", lw=0.8, ls="--")
    ax.set_xticks(list(C.HORIZONS)); ax.set_xlabel("forecast horizon (days)")
    ax.set_ylabel("RMSE skill vs persistence")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    _save(fig, "fig4_skill_vs_horizon")


def fig_trajectory(pred):
    site, h = "p3", 7
    sub = pred[(pred.split == "test") & (pred.site_id == site) & (pred.horizon == h)]
    if sub.empty:
        return
    tr = sub[sub.model == "ThermoRoute"].groupby("issue_date").mean(numeric_only=True)
    dp = sub[sub.model == "DampedPersistence"].groupby("issue_date").mean(numeric_only=True)
    tr = tr.sort_index(); dp = dp.sort_index()
    win = (tr.index >= pd.Timestamp("2020-04-01")) & (tr.index <= pd.Timestamp("2020-11-01"))
    t = tr.index[win]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.fill_between(t, tr["q05"][win], tr["q95"][win], color="#185FA5", alpha=0.2,
                    label="ThermoRoute 90% interval")
    ax.plot(t, tr["y_true"][win], color="black", lw=1.2, label="observed")
    ax.plot(t, tr["y_pred"][win], color="#185FA5", lw=1.1, label="ThermoRoute")
    ax.plot(dp.index[win], dp["y_pred"][win], color="#BA7517", lw=1.0, ls="--",
            label="damped persistence")
    ax.set_title(f"{site}, {h}-day-ahead development forecast (2020 warm season)")
    ax.set_ylabel("WTEMP (°C)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    _save(fig, "fig5_blindtest_trajectory")


def fig_reliability(scores):
    s = _test_scores(scores)
    tr = s[(s.model == "ThermoRoute") & (s.scope == "joint")]
    lg = s[(s.model == "LightGBM") & (s.feature_set == "V3")]
    if tr.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for df, name, col in [(tr, "ThermoRoute", "#185FA5"), (lg, "LightGBM", "#7F77DD")]:
        g = df.groupby("horizon").PICP.mean()
        a1.plot(g.index, g.values, "-o", ms=4, color=col, label=name)
    a1.axhline(0.90, color="#993C1D", ls="--", lw=1, label="nominal 90%")
    a1.set_xticks(list(C.HORIZONS)); a1.set_ylim(0.5, 1.0)
    a1.set_xlabel("horizon (d)"); a1.set_ylabel("interval coverage (PICP)")
    a1.set_title("a  Calibrated coverage"); a1.legend(fontsize=7, frameon=False)
    a1.grid(alpha=0.25)
    for df, name, col in [(tr, "ThermoRoute", "#185FA5"), (lg, "LightGBM", "#7F77DD")]:
        g = df.groupby("horizon").MPIW.mean()
        a2.plot(g.index, g.values, "-o", ms=4, color=col, label=name)
    a2.set_xticks(list(C.HORIZONS)); a2.set_xlabel("horizon (d)")
    a2.set_ylabel("mean interval width (°C)")
    a2.set_title("b  Sharpness"); a2.grid(alpha=0.25)
    _save(fig, "fig6_reliability")


def fig_lag_maps(expl):
    vn = list(expl["var_names"]); H = expl["horizons"]; ov = expl["overall"]  # [H,V,Lr1]
    Lr1 = ov.shape[2]
    fig, axes = plt.subplots(1, len(H), figsize=(10, 3.4), sharey=True)
    vmax = np.percentile(ov, 99)
    for hi, h in enumerate(H):
        ax = axes[hi]
        im = ax.imshow(ov[hi], cmap="magma_r", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"h = {h} d")
        ax.set_xticks(range(0, Lr1, 2)); ax.set_xticklabels(range(0, Lr1, 2))
        ax.set_xlabel("lag (days)")
        if hi == 0:
            ax.set_yticks(range(len(vn))); ax.set_yticklabels(vn)
    fig.suptitle("Router variable×lag importance by horizon", y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="weight share")
    _save(fig, "fig7_lag_importance")


def fig_dynamic_kappa(expl):
    kappa = expl["kappa"]; lf = expl["logflowz"]; station = expl["station"]
    months = expl["months"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for i, st in enumerate(C.STATIONS):
        sel = station == i
        x = lf[sel]
        bins = np.quantile(x, np.linspace(0, 1, 9))
        idx = np.digitize(x, bins[1:-1])
        mx = [x[idx == b].mean() for b in range(len(bins) - 1)]
        mk = [kappa[sel][idx == b].mean() for b in range(len(bins) - 1)]
        a1.plot(mx, mk, "-o", ms=3, color=STCOLOR[st], label=st)
    a1.set_xlabel("standardised log-flow z(logFLOW)")
    a1.set_ylabel("relaxation rate κ (per day)")
    a1.set_title("a  κ vs flow regime"); a1.legend(fontsize=7, frameon=False)
    a1.grid(alpha=0.25)
    for i, st in enumerate(C.STATIONS):
        sel = station == i
        mk = [kappa[sel][months[sel] == m].mean() for m in range(1, 13)]
        a2.plot(range(1, 13), mk, "-o", ms=3, color=STCOLOR[st], label=st)
    a2.set_xticks([1, 4, 7, 10]); a2.set_xlabel("month")
    a2.set_ylabel("relaxation rate κ"); a2.set_title("b  κ seasonality")
    a2.grid(alpha=0.25)
    _save(fig, "fig8_dynamic_kappa")


def fig_loso(scores):
    s = _test_scores(scores)
    loso = s[s.model == "ThermoRoute-LOSO-WarmStart"]
    joint = s[(s.model == "ThermoRoute") & (s.scope == "joint")]
    if loso.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    width = 0.35
    xs = np.arange(len(C.STATIONS))
    # group bars by station for h=7 (hardest)
    h = 7
    jr = [joint[(joint.site_id == st) & (joint.horizon == h)].RMSE.mean() for st in C.STATIONS]
    lr = [loso[(loso.site_id == st) & (loso.horizon == h)].RMSE.mean() for st in C.STATIONS]
    ax.bar(xs - width / 2, jr, width, color="#185FA5", label="joint (in-sample)")
    ax.bar(xs + width / 2, lr, width, color="#993C1D", label="LOSO (warm start)")
    ax.set_xticks(xs); ax.set_xticklabels(list(C.STATIONS))
    ax.set_ylabel("RMSE at h=7 d (°C)")
    ax.set_title("Leave-one-station-out spatial transfer")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig9_loso")


def fig_flow_lagmaps(expl):
    fm = expl["flow_maps"]; keys = expl["flow_keys"]; vn = list(expl["var_names"])
    H = expl["horizons"]; hi = list(H).index(3)   # show h=3
    Lr1 = fm.shape[3]
    fig, axes = plt.subplots(1, len(keys), figsize=(10, 3.4), sharey=True)
    vmax = np.percentile(fm[:, hi], 99)
    for k, key in enumerate(keys):
        ax = axes[k]
        im = ax.imshow(fm[k, hi], cmap="magma_r", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"{key} flow")
        ax.set_xticks(range(0, Lr1, 2)); ax.set_xticklabels(range(0, Lr1, 2))
        ax.set_xlabel("lag (days)")
        if k == 0:
            ax.set_yticks(range(len(vn))); ax.set_yticklabels(vn)
    fig.suptitle("Router lag importance by flow regime (h = 3 d)", y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="weight share")
    _save(fig, "fig10_flow_lagmaps")


def main():
    panel, pred, scores, expl = load()
    fig_study_area(panel)
    fig_series_climatology(panel)
    if scores is not None:
        fig_results_heatmap(scores)
        fig_skill_vs_horizon(scores)
        fig_reliability(scores)
        fig_loso(scores)
    if pred is not None:
        fig_trajectory(pred)
    if expl is not None:
        fig_lag_maps(expl)
        fig_dynamic_kappa(expl)
        fig_flow_lagmaps(expl)
    print("figures complete", flush=True)


if __name__ == "__main__":
    main()
