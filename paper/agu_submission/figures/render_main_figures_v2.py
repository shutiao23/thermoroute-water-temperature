#!/usr/bin/env python3
r"""Six-figure main-text system (reviewer-layout pass).

figure 1  study sites and evaluation design        (map + timeline + tasks + boundary)
figure 2  prediction framework                     (anchor -> residual learner -> bound)
figure 3  decomposition of reported skill          (RMSE ladder + waterfall + paired)
figure 4  model-class and architecture effects     (forest plots vs damped)
figure 5  spatial transfer and distance            (fold map + arms + distance curve)
figure 6  hydrologic conditions governing gain     (half-life + state strata + seasons)

Style: 139.7 mm wide -- the AGU \textwidth, so `width=\linewidth` neither scales
the figure nor the type in it -- height <= 228 mm, no suptitles (the caption
carries the title), no in-figure prose beyond short labels, constrained layout,
and every figure passes figstyle's text-collision gate before it is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper"))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "paper" / "agu_submission" / "figures"
CONV = ROOT / "outputs" / "conventional"
GEO = OUT / "us_states_geojson.json"

# Draw at final size.  These figures were being drawn 7.1 in wide and then
# scaled to \linewidth (5.5 in) by `width=\linewidth`, so 7.5 pt type printed at
# 5.8 pt -- below AGU's ~8 pt floor -- and because `bbox_inches="tight"` cropped
# each figure to its own content, every figure got a different scale factor and
# therefore a different effective type size.  Drawing at 139.7 mm and saving
# with the standard bbox makes the declared point sizes the printed ones.
W_IN = figstyle.FULL_MM / 25.4
H_CAP = figstyle.MAX_HEIGHT_MM / 25.4
MODEL_COLOUR = {
    "ThermoRoute": figstyle.SERIES["ThermoRoute"],
    "LightGBM": figstyle.SERIES["LightGBM"],
    "LSTM": figstyle.SERIES["LSTM"],
    "Persistence": figstyle.SERIES["Persistence"],
    "DampedPersistence": figstyle.SERIES["DampedPersistence"],
    "Climatology": figstyle.SERIES["Climatology"],
}
ARM_COLOUR = {"random": figstyle.SERIES["LightGBM"], "region": figstyle.SERIES["ThermoRoute"]}

# Display names.  "DampedPersistence" is 17 characters and does not fit beside a
# line end at 7.5 pt in a half-width panel.
LABEL = {
    "ThermoRoute": "ThermoRoute",
    "LightGBM": "LightGBM",
    "LSTM": "LSTM",
    "Persistence": "Persistence",
    "DampedPersistence": "Damped persist.",
    "Climatology": "Climatology",
}

# What each one-factor control actually does.  `m.replace("TR-", "no ")` turned
# `TR-noTCN` into "no noTCN" -- a double negative -- and, worse, turned
# `TR-fixedKappa` into "no fixedKappa" and `TR-unbounded` into "no unbounded",
# which state the opposite of the intervention.  These are spelled out instead.
ABLATION_LABEL = {
    "ThermoRoute": "full model",
    "TR-noTCN": "no TCN",
    "TR-noRouter": "no router",
    "TR-noMoE": "no expert mixture",
    "TR-noDynamicPrior": "no dynamic prior",
    "TR-fixedKappa": "fixed κ (not learned)",
    "TR-unbounded": "correction unbounded",
}


def _load():
    figstyle.use()
    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    registry["site_no8"] = registry["site_no"].astype(str).str.zfill(8)
    sm = pd.read_csv(CONV / "station_metrics_2021_2023.csv")
    sm["site_id"] = sm["site_id"].astype(str).str.zfill(8)
    sm = sm.merge(registry[["site_no8", "huc2", "lat", "lon"]],
                  left_on="site_id", right_on="site_no8", how="left")
    skill = pd.read_csv(CONV / "skill_table_2021_2023.csv")
    paired = pd.read_csv(CONV / "paired_effects_2021_2023.csv")
    mech = json.loads((CONV / "mechanism_2021_2023.json").read_text())
    st = pd.read_csv(CONV / "mechanism_station_level_2021_2023.csv")
    rt = pd.read_csv(CONV / "region_transfer_metrics_2021_2023.csv")
    geo = json.loads(GEO.read_text())
    return registry, sm, skill, paired, mech, st, rt, geo


def _conus(ax):
    geo = json.loads(GEO.read_text())
    for feat in geo["features"]:
        name = feat["properties"].get("name", "")
        if name in ("Alaska", "Hawaii", "Puerto Rico", "District of Columbia"):
            continue
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            ring = poly[0] if g["type"] == "MultiPolygon" else poly
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            ax.plot(xs, ys, color="#BBBBBB", lw=0.4, solid_capstyle="round")


def _region_folds(registry):
    by_reg = {}
    for _, r in registry.iterrows():
        by_reg.setdefault(str(r.huc2), []).append(str(r.site_no8))
    buckets = [[], [], [], []]
    loads = [0, 0, 0, 0]
    for reg, sts in sorted(by_reg.items(), key=lambda kv: -len(kv[1])):
        i = int(np.argmin(loads))
        buckets[i].extend(sts)
        loads[i] += len(sts)
    fold_of = {}
    for i, sts in enumerate(buckets):
        for s in sts:
            fold_of[s] = i
    return fold_of


def save(fig, stem):
    """Write the figure, refusing if any two text artists collide.

    Routed through figstyle so the overlap gate actually runs; the previous
    local implementation wrote unconditionally and with `bbox_inches="tight"`,
    which both skipped the check and changed the final width.
    """
    figstyle.save(fig, stem, OUT, formats=("pdf", "png", "svg"), strict=True)
    w_mm, h_mm = (v * 25.4 for v in fig.get_size_inches())
    smallest = min((t.get_fontsize() for t in fig.findobj(plt.Text)
                    if t.get_visible() and t.get_text().strip()), default=0.0)
    plt.close(fig)
    print(f"wrote {stem:34s} {w_mm:5.1f} x {h_mm:5.1f} mm   min type {smallest:.1f} pt")


def fig1(registry, *rest):
    fold_of = _region_folds(registry)
    fig = plt.figure(figsize=(W_IN, 3.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    _conus(ax_a)
    fold_cols = [figstyle.WONG["blue"], figstyle.WONG["sky"],
                 figstyle.WONG["orange"], figstyle.WONG["vermillion"]]
    for st, r in registry.iterrows():
        ax_a.scatter(r.lon, r.lat, s=5, color=fold_cols[fold_of[str(r.site_no8)]],
                     edgecolor="none", zorder=3)
    ax_a.set_xlim(-125, -66)
    ax_a.set_ylim(24.5, 49.5)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    figstyle.panel_label(ax_a, "(a) 120 sites, 4 whole-region folds")

    # Segment widths are proportional to their span in years, so 2018 is one
    # unit wide while training is ten.  Centring a two-line label on a one-unit
    # segment is what produced "2016-1720182019-20" in the first draft: the
    # labels are far wider than the segments they name.  Labels therefore sit in
    # two staggered rows joined to their segment by a leader, which is the
    # standard fix for a timeline with very unequal spans.
    ax_b = fig.add_subplot(gs[0, 1])
    periods = [("2006\u20132015", "Training", 10, figstyle.WONG["blue"]),
               ("2016\u201317", "Validation", 2, figstyle.WONG["sky"]),
               ("2018", "Calibration", 1, figstyle.WONG["green"]),
               ("2019\u201320", "Development", 2, figstyle.WONG["orange"]),
               ("2021\u201323", "Held-out", 3, figstyle.WONG["vermillion"])]
    rows = (0.62, 1.32)          # near row, far row
    nudge = (0.0, -1.4, -0.6, 1.4, 1.2)
    x0 = 0.0
    for i, (years, name, w, colour) in enumerate(periods):
        ax_b.barh(0.0, w, left=x0, height=0.34, color=colour,
                  edgecolor="white", linewidth=0.5, zorder=2)
        cx = x0 + w / 2
        lx = cx + nudge[i]
        ly = rows[i % 2]
        ax_b.plot([cx, lx], [0.20, ly - 0.08], color=colour, lw=0.5,
                  alpha=0.7, zorder=1)
        ax_b.text(lx, ly, f"{years}\n{name}", ha="center", va="bottom",
                  fontsize=7.5, linespacing=1.25)
        x0 += w
    ax_b.set_xlim(-1.4, x0 + 1.6)
    ax_b.set_ylim(-0.45, 2.05)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    figstyle.panel_label(ax_b, "(b) Temporal partitions")

    ax_c = fig.add_subplot(gs[1, 0])
    tasks = [
        ("Known-site", "site trains; identity used", True),
        ("Random held-site", "random folds; identity removed", True),
        ("Whole-region", "HUC2 folds; identity removed", True),
    ]
    for i, (name, detail, local) in enumerate(tasks):
        colour = figstyle.WONG["sky"] if i == 0 else ARM_COLOUR[
            "random" if i == 1 else "region"]
        ax_c.barh(2 - i, 1.0, height=0.5, color=colour, edgecolor="white",
                  linewidth=0.5, left=0.0)
        ax_c.text(1.05, 2 - i, name, va="center", fontsize=7.5)
        ax_c.text(1.05, 2 - i - 0.28, detail, va="center", fontsize=7.5,
                  color=figstyle.MUTED)
    ax_c.set_xlim(0, 3.6)
    ax_c.set_ylim(-0.5, 2.6)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    figstyle.panel_label(ax_c, "(c) Evaluation tasks")

    # Two columns at fixed x collided: "All models scored on identical keys" is
    # wider than the 0.03-to-0.55 slot it was given, so it ran through the
    # right-hand column.  Stacking the qualifier under its statement removes the
    # constraint entirely and matches panel (c)'s treatment.
    ax_d = fig.add_subplot(gs[1, 1])
    boundary = [("Predictors dated \u2264 issue time t", "local WT history allowed"),
                ("Target observed at t + h", "no future weather"),
                ("All models scored on identical keys",
                 "preprocessing fitted on \u2264 2020")]
    for i, (statement, detail) in enumerate(boundary):
        y = 0.88 - i * 0.33
        ax_d.text(0.0, y, statement, fontsize=7.5, va="center")
        ax_d.text(0.0, y - 0.13, detail, fontsize=7.5, va="center",
                  color=figstyle.MUTED)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for sp in ax_d.spines.values():
        sp.set_visible(False)
    figstyle.panel_label(ax_d, "(d) Issue-time information boundary")

    save(fig, "fig01_study_design")


def fig2(*args):
    """Anchor -> residual learner -> bounded correction.

    Three defects fixed here.  The connectors spanned 0.06 of a 4.0-unit axis
    (about 2.7 mm) while the gap they had to cross was 0.28, so they read as
    specks floating between the blocks; they now span the gap and carry a
    proportionate head.  The footnote was placed at x = 0.5 in *data*
    coordinates on an axis running 0 to 4, which put it an eighth of the way
    across rather than in the middle -- it is now centred in axes coordinates,
    where "centre" cannot drift if the limits change.  And `A_{t+h}` was being
    typeset literally, braces and all.
    """
    fig, ax = plt.subplots(figsize=(W_IN, 1.55))
    blocks = [
        ("Issue-time history\n(WT, flow,\nmeteorology)", figstyle.WONG["sky"], 0.0),
        ("Damped-persistence\nanchor A(t + h)", figstyle.WONG["green"], 1.0),
        ("Temporal residual\nlearner (TCN)", figstyle.WONG["blue"], 2.0),
        ("Bounded correction\n+ final prediction", figstyle.WONG["orange"], 3.0),
    ]
    for label, colour, x in blocks:
        ax.add_patch(plt.Rectangle((x + 0.10, 0.30), 0.80, 0.50,
                                   facecolor=colour, edgecolor="white",
                                   linewidth=0.6, zorder=2))
        ax.text(x + 0.50, 0.55, label, ha="center", va="center",
                fontsize=7.5, zorder=3, linespacing=1.3)
        if x < 3.0:
            ax.annotate("", xy=(x + 1.09, 0.55), xytext=(x + 0.91, 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=figstyle.RULE,
                                        lw=0.9, mutation_scale=9,
                                        shrinkA=0, shrinkB=0))
    ax.text(0.5, 0.045, "same 1/3/7-day horizons \u00b7 no future weather \u00b7 "
                        "correction relative to the anchor",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.5, color=figstyle.MUTED)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "fig02_prediction_framework")


def fig3(registry, sm, skill, paired, mech, *rest):
    fig = plt.figure(figsize=(W_IN, 5.40))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])

    ax_a = fig.add_subplot(gs[0, 0])
    models = ["Persistence", "DampedPersistence", "LightGBM", "LSTM",
              "ThermoRoute"]
    # Four of these six finish inside 0.1 °C of one another at seven days, so
    # end-of-line labels anchored to their own y printed on top of each other
    # ("DampedPersistence" over "Air2stream" over "ThermoRoute").  Collect them
    # and let figstyle separate them in display space, with leaders.
    ends = []
    for model in models:
        g = sm[sm.model == model]
        ys = [g[g.horizon == h].rmse.median() for h in (1, 3, 7)]
        colour = MODEL_COLOUR[model]
        ax_a.plot((1, 3, 7), ys, marker="o", ms=3.5, lw=1.2, color=colour,
                  markeredgecolor="white", markeredgewidth=0.4)
        ends.append((ys[-1], LABEL[model], colour))
    a2s = pd.read_parquet(CONV / "air2stream_2021_2023.parquet")
    a2s["site_id"] = a2s["site_id"].astype(str).str.zfill(8)
    a2s_rmse = a2s.groupby("horizon").apply(
        lambda g: np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)),
        include_groups=False)
    ax_a.plot((1, 3, 7), [a2s_rmse[h] for h in (1, 3, 7)], marker="P", ms=3.5,
              lw=1.1, color=figstyle.WONG["purple"],
              markeredgecolor="white", markeredgewidth=0.4)
    ends.append((float(a2s_rmse[7]), "Air2stream", figstyle.WONG["purple"]))
    ax_a.set_xticks((1, 3, 7))
    ax_a.set_xlim(0.5, 11.2)
    ax_a.set_ylim(0.4, 2.4)
    ax_a.set_xlabel("horizon (d)")
    ax_a.set_ylabel("station-median RMSE (\u00b0C)")
    figstyle.panel_label(ax_a, "(a) RMSE ladder, 2021\u20132023")
    ax_a.grid(axis="y", color="#DDDDDD", lw=0.4)
    figstyle.spread_labels(ax_a, ends, x_data=7, x_text=7.7)

    ax_b = fig.add_subplot(gs[0, 1])
    h7 = mech["memory_learned_median"]["7"]
    starts = [0.0, h7["G_memory"]]
    vals = [h7["G_memory"], h7["G_learned"]]
    labels = ["damping", "learned"]
    # The three level annotations were free-floating text at negative y, which
    # put them straight through the 0.00 and -0.25 tick labels and through each
    # other.  They belong on the x axis: as tick labels they get their space
    # reserved by the layout engine instead of competing for it.
    ax_b.bar([0, 1], vals, bottom=[0, starts[1]],
             color=(figstyle.WONG["sky"], figstyle.WONG["orange"]),
             width=0.42, edgecolor="white", linewidth=0.5)
    for i, (v, s) in enumerate(zip(vals, starts)):
        ax_b.text(i, s + v + 0.012, f"{v:.2f} \u00b0C", ha="center", va="bottom",
                  fontsize=7.5)
    # The step names fit on the axis; the RMSE levels they move between do not
    # -- at 7.5 pt "2.20 \u2192 1.77" is as wide as the whole gap between the two
    # bars -- so the ladder goes once into the free space above the bars.
    ax_b.set_xlim(-0.75, 1.75)
    ax_b.set_ylim(0.0, (starts[1] + vals[1]) * 1.62)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(labels, fontsize=7.5)
    ax_b.tick_params(axis="x", length=0)
    ax_b.text(0.02, 0.97, "station-median RMSE (\u00b0C)\n2.20 \u2192 1.77 \u2192 1.69",
              transform=ax_b.transAxes, fontsize=7.5, va="top",
              color=figstyle.MUTED, linespacing=1.35)
    ax_b.set_ylabel("7 d RMSE reduction (\u00b0C)")
    figstyle.panel_label(ax_b, "(b) Error budget, 7 d")
    ax_b.grid(axis="y", color="#DDDDDD", lw=0.4)

    ax_c = fig.add_subplot(gs[1, :])
    rows = [("damped persist., 1 d", "DampedPersistence", 1),
            ("damped persist., 3 d", "DampedPersistence", 3),
            ("damped persist., 7 d", "DampedPersistence", 7),
            ("LightGBM, 3 d", "LightGBM", 3),
            ("LightGBM, 7 d", "LightGBM", 7)]
    vals, cis = [], []
    for label, ref, h in rows:
        g = paired[(paired.candidate == "ThermoRoute") &
                   (paired.reference == ref) & (paired.horizon == h)]
        if g.empty:
            continue
        g = g.dropna(subset=["effect"])
        vals.append(float(g.effect.median()))
        cis.append((float(g.effect.quantile(0.25)),
                    float(g.effect.quantile(0.75))))
    y = np.arange(len(rows))[::-1]
    for i, (v, ci, yi) in enumerate(zip(vals, cis, y)):
        ax_c.hlines(yi, ci[0], ci[1], color=MODEL_COLOUR["ThermoRoute"], lw=2.2)
        ax_c.scatter(v, yi, s=26, color=MODEL_COLOUR["ThermoRoute"], zorder=3,
                     edgecolor="white", linewidth=0.5)
    ax_c.axvline(0.0, color="black", lw=0.7)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax_c.set_xlim(-0.22, 0.06)
    ax_c.set_xlabel("\u0394RMSE (\u00b0C; \u2212 favors ThermoRoute)")
    figstyle.panel_label(ax_c, "(c) Paired station effects, median and IQR")
    ax_c.grid(axis="x", color="#DDDDDD", lw=0.4)

    save(fig, "fig03_skill_decomposition")


def fig4(registry, sm, skill, paired, mech, *rest):
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 2.95), width_ratios=[0.92, 1.0])
    rows = [("ThermoRoute", "DampedPersistence"), ("LightGBM", "DampedPersistence"),
            ("LSTM", "DampedPersistence")]
    # LightGBM and the LSTM both land near -0.053 at seven days, so their end
    # labels printed one on top of the other.  Same treatment as figure 3(a).
    ends = []
    for j, (cand, ref) in enumerate(rows):
        g = paired[(paired.candidate == cand) & (paired.reference == ref)]
        vals = []
        for h in (1, 3, 7):
            gg = g[(g.horizon == h)].dropna(subset=["effect"])
            vals.append(float(gg.effect.median()))
        colour = MODEL_COLOUR[cand]
        axes[0].plot([1, 3, 7], vals, marker="o", ms=3.5, lw=1.2, color=colour,
                     markeredgecolor="white", markeredgewidth=0.4)
        ends.append((vals[-1], LABEL[cand], colour))
    axes[0].axhline(0.0, color="black", lw=0.6)
    axes[0].set_xticks((1, 3, 7))
    axes[0].set_xlim(0.5, 11.5)
    axes[0].set_xlabel("horizon (d)")
    axes[0].set_ylabel("median \u0394RMSE, model \u2212 damped (\u00b0C)")
    figstyle.panel_label(axes[0], "(a) Learned models")
    axes[0].grid(axis="y", color="#DDDDDD", lw=0.4)
    figstyle.spread_labels(axes[0], ends, x_data=7, x_text=7.6)

    abl = ["ThermoRoute", "TR-noTCN", "TR-noRouter", "TR-noMoE",
           "TR-noDynamicPrior", "TR-fixedKappa", "TR-unbounded"]
    vals, labels = [], []
    for m in abl:
        g = paired[(paired.candidate == m) &
                   (paired.reference == "DampedPersistence") &
                   (paired.horizon == 1)].dropna(subset=["effect"])
        vals.append(float(g.effect.median()))
        labels.append(ABLATION_LABEL[m])
    y2 = np.arange(len(vals))[::-1]
    axes[1].scatter(vals, y2, s=20, color=figstyle.WONG["sky"], zorder=3,
                    edgecolor="white", linewidth=0.4)
    axes[1].axvline(0.0, color="black", lw=0.6)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(labels, fontsize=7.5)
    axes[1].set_xlim(-0.20, 0.06)
    axes[1].set_xlabel("\u0394RMSE, variant \u2212 damped (\u00b0C)")
    figstyle.panel_label(axes[1], "(b) Architecture controls, 1 d")
    axes[1].grid(axis="x", color="#DDDDDD", lw=0.4)
    save(fig, "fig04_model_class_effects")


def fig5(registry, sm, skill, paired, mech, st, rt, geo):
    fig = plt.figure(figsize=(W_IN, 5.30))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])
    fold_of = _region_folds(registry)
    fold_cols = [figstyle.WONG["blue"], figstyle.WONG["sky"],
                 figstyle.WONG["orange"], figstyle.WONG["vermillion"]]

    ax_a = fig.add_subplot(gs[0, 0])
    _conus(ax_a)
    for _, r in registry.iterrows():
        ax_a.scatter(r.lon, r.lat, s=5, color=fold_cols[fold_of[str(r.site_no8)]],
                     edgecolor="none", zorder=3)
    ax_a.set_xlim(-125, -66)
    ax_a.set_ylim(24.5, 49.5)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    figstyle.panel_label(ax_a, "(a) Four whole-region folds")

    ax_b = fig.add_subplot(gs[0, 1])
    for arm, label in (("random", "Random held-site"),
                       ("region", "Whole-region")):
        for h in (1, 3, 7):
            g = rt[(rt.arm == arm) & (rt.horizon == h)]
            v = float(g.rmse.median())
            ax_b.scatter(h, v, s=22, color=ARM_COLOUR[arm],
                         marker="o" if arm == "random" else "s",
                         edgecolor="white", linewidth=0.4, zorder=3)
    med = {a: [float(rt[(rt.arm == a) & (rt.horizon == h)].rmse.median())
               for h in (1, 3, 7)] for a in ("random", "region")}
    for arm, name in (("random", "Random held-site"), ("region", "Whole-region")):
        ax_b.plot([1, 3, 7], med[arm], color=ARM_COLOUR[arm], lw=1.1, label=name)
    ax_b.legend(loc="upper left", fontsize=7.5, frameon=False,
                handlelength=1.4, borderaxespad=0.1)
    # The two arms differ by hundredths of a degree on an axis spanning 1.4 degC,
    # so the panel would otherwise read as "no effect" while the text claims one.
    gap = " / ".join(f"{med['region'][i] - med['random'][i]:+.02f}" for i in range(3))
    ax_b.text(0.98, 0.03, f"region − random: {gap} °C", transform=ax_b.transAxes,
              ha="right", va="bottom", fontsize=7.5, color=figstyle.MUTED)
    ax_b.set_xticks((1, 3, 7))
    ax_b.set_xlim(0.5, 7.5)
    ax_b.set_ylim(0.5, 1.9)
    ax_b.set_xlabel("horizon (d)")
    ax_b.set_ylabel("station-median RMSE (\u00b0C)")
    figstyle.panel_label(ax_b, "(b) Spatial arms")
    ax_b.grid(axis="y", color="#DDDDDD", lw=0.4)

    ax_c = fig.add_subplot(gs[1, :])
    for arm, colour, marker in (("random", ARM_COLOUR["random"], "o"),
                                ("region", ARM_COLOUR["region"], "s")):
        g = rt[(rt.arm == arm) & (rt.horizon == 3)].dropna(subset=["nearest_km"])
        ax_c.scatter(g.nearest_km, g.rmse - g.rmse_damped, s=6, color=colour,
                     marker=marker, alpha=0.5, edgecolor="none")
        b = g.groupby(pd.qcut(g.nearest_km, 5), group_keys=False).apply(
            lambda x: pd.Series({"x": x.nearest_km.median(),
                                 "y": (x.rmse - x.rmse_damped).median()}),
            include_groups=False)
        ax_c.plot(b.x, b.y, color=colour, lw=1.6, marker="D", ms=3,
                  markeredgecolor="white",
                  label="Random held-site" if arm == "random" else "Whole-region")
    ax_c.axhline(0.0, color="black", lw=0.6)
    ax_c.set_xscale("log")
    ax_c.set_xticks([10, 30, 100, 300, 1000])
    ax_c.set_xticklabels(["10", "30", "100", "300", "1000"])
    ax_c.set_xlabel("nearest-training-gauge distance (km)")
    ax_c.set_ylabel("3 d \u0394RMSE, ThermoRoute \u2212 damped (\u00b0C)")
    figstyle.panel_label(ax_c, "(c) Transfer penalty by distance")
    ax_c.legend(loc="lower left", fontsize=7.5, frameon=False, handlelength=1.4)
    ax_c.grid(axis="x", color="#DDDDDD", lw=0.4)

    save(fig, "fig04_spatial_transfer")


def fig6(registry, sm, skill, paired, mech, st, rt, geo):
    pass

def fig6_main(registry, sm, skill, paired, mech, st, rt, geo):
    fig = plt.figure(figsize=(W_IN, 5.50))
    gs = fig.add_gridspec(2, 2)

    ax_a = fig.add_subplot(gs[0, 0])
    hl = st[st.horizon == 1].dropna(subset=["half_life"])
    ax_a.hist(hl.half_life, bins=20, color=figstyle.WONG["sky"],
              edgecolor="white", linewidth=0.4)
    med = float(hl.half_life.median())
    ax_a.axvline(med, color=figstyle.WONG["orange"], lw=1.2)
    ymax = ax_a.get_ylim()[1]
    ax_a.set_ylim(0, ymax * 1.22)
    ax_a.text(med * 1.10, ymax * 1.14, f"median {med:.1f} d", fontsize=7.5,
              color=figstyle.WONG["orange"], va="center", ha="left")
    ax_a.set_xlabel("thermal half-life (d)")
    ax_a.set_ylabel("stations")
    figstyle.panel_label(ax_a, "(a) Thermal memory")
    ax_a.grid(axis="y", color="#DDDDDD", lw=0.4)

    ax_b = fig.add_subplot(gs[0, 1])
    g = st[st.horizon == 1].dropna(subset=["half_life", "G_learned"])
    ax_b.scatter(g.half_life, g.G_learned, s=7, color=figstyle.WONG["blue"],
                 alpha=0.55, edgecolor="none")
    m, c = np.polyfit(g.half_life, g.G_learned, 1)
    xs = np.linspace(g.half_life.min(), g.half_life.max(), 40)
    ax_b.plot(xs, m * xs + c, color=figstyle.WONG["orange"], lw=1.2)
    r = float(np.corrcoef(g.half_life, g.G_learned)[0, 1])
    ax_b.text(0.97, 0.05, f"r = {r:+.2f}, n = {len(g)}", transform=ax_b.transAxes,
              fontsize=7.5, ha="right")
    ax_b.set_xlabel("thermal half-life (d)")
    ax_b.set_ylabel("1 d learned gain over damped (\u00b0C)")
    figstyle.panel_label(ax_b, "(b) Memory vs learned gain")
    ax_b.grid(color="#DDDDDD", lw=0.4)

    ax_c = fig.add_subplot(gs[1, 0])
    strata = mech["stratified_delta_rmse"]["7"]
    order = ["all", "rapid_change10", "warmest10", "high_flow10",
             "low_flow10", "high_disequilibrium10", "coldest10"]
    labels = ["All keys", "Fastest warming 10%", "Warmest 10%", "High flow 10%",
              "Low flow 10%", "High air\u2013water gap 10%", "Coldest 10%"]
    vals = [strata[k]["delta_rmse"] for k in order]
    y = np.arange(len(vals))[::-1]
    ax_c.hlines(y, 0, vals, color=figstyle.WONG["sky"], lw=2.0)
    ax_c.scatter(vals, y, s=20, color=figstyle.WONG["sky"], zorder=3,
                 edgecolor="white", linewidth=0.4)
    ax_c.axvline(0.0, color="black", lw=0.6)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels, fontsize=7.5)
    ax_c.set_xlim(-0.38, 0.05)
    ax_c.set_xlabel("7 d \u0394RMSE, ThermoRoute \u2212 damped (\u00b0C)")
    figstyle.panel_label(ax_c, "(c) Learned gain by hydrologic state")
    ax_c.grid(axis="x", color="#DDDDDD", lw=0.4)

    ax_d = fig.add_subplot(gs[1, 1])
    season = {h: {s: strata_all[s]["delta_rmse"] for s in
                  ("season_DJF", "season_MAM", "season_JJA", "season_SON")}
              for h, strata_all in mech["stratified_delta_rmse"].items()}
    # Labelled at the line ends rather than with a legend: an unframed legend in
    # the lower left sat directly on the 1 d series, which is lowest there --
    # text over a line, which the collision gate cannot see.
    xs = np.arange(4)
    for j, h in enumerate((1, 3, 7)):
        vals = [season[str(h)][s] for s in
                ("season_DJF", "season_MAM", "season_JJA", "season_SON")]
        colour = (figstyle.WONG["sky"], figstyle.WONG["orange"],
                  figstyle.WONG["vermillion"])[j]
        x_end = 3 + (j - 1) * 0.24
        ax_d.plot(xs + (j - 1) * 0.24, vals, marker="o", ms=3,
                  lw=1.1, color=colour,
                  markeredgecolor="white", markeredgewidth=0.4)
        ax_d.text(x_end + 0.13, vals[-1], f"{h} d", fontsize=7.5, color=colour,
                  va="center", ha="left")
    ax_d.axhline(0.0, color="black", lw=0.6)
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(["DJF", "MAM", "JJA", "SON"], fontsize=7.5)
    ax_d.set_xlim(-0.5, 4.05)
    ax_d.set_ylabel("\u0394RMSE, ThermoRoute \u2212 damped (\u00b0C)")
    figstyle.panel_label(ax_d, "(d) Learned gain by season")
    ax_d.grid(axis="y", color="#DDDDDD", lw=0.4)


    save(fig, "fig06_hydrologic_mechanism")


if __name__ == "__main__":
    data = _load()
    fig1(*data)
    fig2(*data)
    fig3(*data)
    fig4(*data)
    fig5(*data)
    fig6_main(*data)
    print("done")
