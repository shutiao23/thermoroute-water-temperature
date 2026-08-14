#!/usr/bin/env python3
r"""Two schematic figures: the shared anchor-residual formulation, and the full
ThermoRoute dataflow.

figure 2   common anchor-residual formulation      (main text, replaces nothing
           numeric -- it is the method schematic the results sections refer to)
figure S3  full ThermoRoute architecture           (Supporting Information)

These replace the `paper/figures_arch/` drafts, which were drawn on an
1800--2300 unit canvas and exported 1350--1725 pt wide.  Embedded at
`width=\linewidth` (397.48 pt) that is a 0.23--0.29 scale factor, so their
15--17 unit body type printed at **2.6--3.8 pt** -- less than half AGU's ~8 pt
floor and illegible on paper.  Same failure the main figure system already
fixed for figures 1--6: draw at 139.7 mm so the declared point sizes are the
printed ones.

Everything is laid out in points on a full-bleed axes, so one data unit is one
printed point and a box height can be reasoned about directly against the type
it must hold.  Both figures go through figstyle's text-collision and
out-of-bounds gate before they are written.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper"))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = ROOT / "paper" / "agu_submission" / "figures"

PT_PER_MM = 72.0 / 25.4
FULL_PT = figstyle.FULL_MM * PT_PER_MM  # 396.0 pt, the AGU \textwidth

# Type scale.  The house floor is 7.5 pt at final size (WRR_FIGURE_STYLE_GUIDE);
# nothing on the page may sit below it, including mathtext superscripts.
TITLE = 9.0
PANEL = 8.0
BOXTITLE = 7.5
BODY = 7.5
# Formulas carrying a superscript are set larger: mathtext draws a superscript
# at 0.7 of the base, so the base is chosen to land the exponent on the floor:
# 10.75 * 0.7 = 7.5.  At the previous 9.0 the exponent printed at 6.3.
MATH = 10.75
MATH_LINE = 11.0

INK = "#1A1A1A"
MUTED = figstyle.MUTED
PANEL_FILL = "#F5F7FA"
PANEL_EDGE = "#C9D1DB"

# Model identity colours are the same ones the result figures use, so a model
# keeps one colour across the whole paper.  Structural boxes therefore may not
# borrow one: drawing the input boxes in LightGBM's blue, as the first pass did,
# made the same hue mean "input" in one panel and "LightGBM" in the next.
C_INPUT = "#3F4A57"  # neutral slate, outside the Wong model palette
C_ANCHOR = figstyle.SERIES["DampedPersistence"]
C_TREE = figstyle.SERIES["LightGBM"]
C_TR = figstyle.SERIES["ThermoRoute"]
C_TCN = figstyle.SERIES["PlainCausalTCN"]
C_FORCING = figstyle.WONG["purple"]
C_OUT = "#00706C"

# A flow arrow needs a gap it can actually be seen in; at the 5 pt stack gap
# only about 2 pt of arrow survives the shrink at each end.
GAP = 5.0
FLOW_GAP = 10.0

# Evaluation strip: heading line plus one row of step boxes.
EVAL_H = 34.0


def canvas(height_pt: float):
    """A figure whose data units are printed points, with no axes furniture."""
    fig = plt.figure(figsize=(FULL_PT / 72.0, height_pt / 72.0))
    fig.set_layout_engine("none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FULL_PT)
    ax.set_ylim(0, height_pt)
    # Both are needed: set_axis_off only hides the tick labels, and figstyle's
    # out-of-bounds gate inspects every Text artist on the figure, visible or
    # not, so the hidden "0", "50", ... would fail the build.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    return fig, ax


def panel(ax, x, y, w, h, title=None, *, fill=PANEL_FILL, edge=PANEL_EDGE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=3",
            linewidth=0.6, facecolor=fill, edgecolor=edge, zorder=1,
        )
    )
    if title:
        ax.text(x + 6, y + h - 9, title, fontsize=PANEL, fontweight="bold",
                color=INK, va="center", ha="left", zorder=4)


# Box metrics.  Heights are derived from these rather than chosen by hand: the
# first draft picked them by eye, and a 22 pt box holding a title plus one 7 pt
# body line pushed its own text 6.5 pt out of the bottom and into the note
# below, which is exactly what figstyle's collision gate then rejected.
BOX_PAD = 4.5
TITLE_LINE = 8.5
TITLE_GAP = 2.5
BODY_LINE = 8.6  # 7 pt at 1.23 leading
PANEL_TITLE_H = 15.0  # panel heading band, above the first box in a stack


def box_height(body_lines: int = 0, *, title: bool = True,
               math: bool = False) -> float:
    """Height a box needs to hold its own type without overflowing."""
    h = 2 * BOX_PAD
    if title:
        h += TITLE_LINE
    if body_lines:
        h += TITLE_GAP + body_lines * (MATH_LINE if math else BODY_LINE)
    return h


class Stack:
    """Places boxes top-down inside a panel, tracking the running cursor."""

    def __init__(self, ax, x, top, w, *, gap=5.0, pad=5.0):
        self.ax, self.x, self.w = ax, x + pad, w - 2 * pad
        self.y = top
        self.gap = gap

    def box(self, title, body=None, colour=INK, **kw):
        lines = 0 if body is None else body.count("\n") + 1
        h = kw.pop("height", None) or box_height(lines, math=kw.get("math", False))
        self.y -= h
        centre = box(self.ax, self.x, self.y, self.w, h, title, body, colour, **kw)
        self.y -= self.gap
        return centre

    def note(self, s, *, lines=None, colour=MUTED):
        n = lines or s.count("\n") + 1
        h = n * BODY_LINE
        self.y -= h
        self.ax.text(self.x, self.y + h / 2, s, fontsize=BODY, color=colour,
                     va="center", ha="left", linespacing=1.23, zorder=4)
        self.y -= self.gap


def box(ax, x, y, w, h, title, body=None, colour=INK, *,
        fill="#FFFFFF", dashed=False, title_size=BOXTITLE, align="left",
        math=False):
    """A labelled box.  Returns its centre, which the arrows attach to."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=2.5",
            linewidth=0.8, facecolor=fill, edgecolor=colour,
            linestyle=(0, (2.5, 1.6)) if dashed else "solid", zorder=2,
        )
    )
    tx = x + w / 2 if align == "center" else x + BOX_PAD
    ha = "center" if align == "center" else "left"
    fig = ax.get_figure()
    watched = fig.__dict__.setdefault("_box_labels", [])
    if body:
        t = ax.text(tx, y + h - BOX_PAD - TITLE_LINE / 2, title, fontsize=title_size,
                    fontweight="bold", color=colour, va="center", ha=ha, zorder=4)
        b = ax.text(tx, y + h - BOX_PAD - TITLE_LINE - TITLE_GAP, body,
                    fontsize=MATH if math else BODY,
                    color=MUTED, va="top", ha=ha, linespacing=1.23, zorder=4)
        watched += [(t, x + BOX_PAD, x + w - BOX_PAD),
                    (b, x + BOX_PAD, x + w - BOX_PAD)]
    else:
        t = ax.text(tx, y + h / 2, title, fontsize=title_size, fontweight="bold",
                    color=colour, va="center", ha=ha, zorder=4)
        watched.append((t, x + BOX_PAD, x + w - BOX_PAD))
    return (x + w / 2, y + h / 2)


def check_boxes_fit(fig, ax, stem: str) -> None:
    """Fail if any label is wider than the box drawn around it.

    figstyle's gate catches text that collides with other text and text that
    leaves the canvas.  It cannot catch text that overruns its own patch,
    because a patch is not a text artist -- so three of the four labels in the
    evaluation strip printed straight through their boxes and every automated
    check still passed.  This closes that gap.
    """
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for artist, x0, x1 in getattr(fig, "_box_labels", []):
        ext = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        (dx0, _), (dx1, _) = inv.transform([(ext.x0, ext.y0), (ext.x1, ext.y1)])
        if dx0 < x0 - 0.25 or dx1 > x1 + 0.25:
            over = max(x0 - dx0, dx1 - x1)
            bad.append(f"{artist.get_text()[:34]!r} overruns its box by {over:.1f} pt")
    if bad:
        raise ValueError(f"{stem}: {len(bad)} label(s) wider than their box -> "
                         + "; ".join(bad[:6]))


def arrow(ax, p0, p1, colour=MUTED, *, dashed=False, rad=0.0, lw=0.9):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>", mutation_scale=6, linewidth=lw,
            color=colour, zorder=3, shrinkA=1.5, shrinkB=1.5,
            linestyle=(0, (2.5, 1.6)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# ---------------------------------------------------------------------------
# Figure 2 -- common anchor-residual formulation
# ---------------------------------------------------------------------------
def draw_concept():
    # Panel (a) is the tallest: three boxes of a title plus two body lines,
    # two gaps, and the panel title.  Everything else is derived from it so a
    # reworded box cannot silently push its text through the panel floor.
    ph = PANEL_TITLE_H + 3 * box_height(2) + 2 * 5.0
    H = EVAL_H + 6.0 + ph + 22.0
    fig, ax = canvas(H)

    ax.text(0, H - 8, "Common anchor–residual formulation under matched information",
            fontsize=TITLE, fontweight="bold", color=INK, va="center", ha="left")

    top = EVAL_H + 6.0
    ax_w, bx_w, cx_w = 106.0, 174.0, 104.0
    ax_x, bx_x, cx_x = 0.0, 112.0, 292.0
    inner = top + ph - PANEL_TITLE_H

    panel(ax, ax_x, top, ax_w, ph, "(a) Information inputs")
    panel(ax, bx_x, top, bx_w, ph, "(b) Prediction decomposition")
    panel(ax, cx_x, top, cx_w, ph, "(c) Forecast")

    # (a) inputs -------------------------------------------------------------
    sa = Stack(ax, ax_x, inner, ax_w)
    p_hist = sa.box("Local thermal state", "WTEMP history, FLOW,\nobservedness masks", C_INPUT)
    p_met = sa.box("Meteorological history", "TEMP, PRCP, RHMEAN,\nradiation, wind", C_INPUT)
    p_fut = sa.box("Future forcing (optional)", "F0 absent\nF3 realized oracle",
                   C_FORCING, dashed=True)

    # (b) decomposition ------------------------------------------------------
    sb = Stack(ax, bx_x, inner, bx_w)
    p_anchor = sb.box("Damped-persistence anchor",
                      "$A = c + \\varphi^{\\,h}\\,(y - c)$", C_ANCHOR, math=True)
    # The headline tree is a raw-target model, not an anchor-residual learner.
    # Drawing every comparator as a residual learner, as the draft did, erases
    # the contrast Section 4.2 turns on.
    row_h = box_height(2)
    sb.y -= row_h
    p_raw = box(ax, sb.x, sb.y, 66, row_h, "Raw target", "LightGBM", C_TREE)
    p_res = box(ax, sb.x + 71, sb.y, sb.w - 71, row_h, "Anchor residual",
                "ResidualLightGBM, plain\ncausal TCN, ThermoRoute", C_TR)
    sb.y -= sb.gap
    sb.note("Identical keys, identical anchor;\nonly the regression target differs.")

    # (c) forecast -----------------------------------------------------------
    sc = Stack(ax, cx_x, inner, cx_w)
    p_fc = sc.box("Point forecast", "$ŷ = A + r$", C_OUT, align="center")
    sc.note("Raw-target models\npredict $ŷ$ directly.")
    sc.note("ThermoRoute bounds\n$|r| < 1$ °C relative to\nthe anchor only.")

    arrow(ax, (ax_x + ax_w - 4, p_hist[1]), (bx_x + 4, p_anchor[1]), C_INPUT, rad=-0.08)
    arrow(ax, (ax_x + ax_w - 4, p_met[1]), (bx_x + 4, p_res[1] + 5), C_INPUT, rad=0.10)
    arrow(ax, (ax_x + ax_w - 4, p_fut[1]), (bx_x + 4, p_res[1] - 5), C_FORCING,
          dashed=True, rad=0.12)
    arrow(ax, (bx_x + bx_w - 4, p_anchor[1]), (cx_x + 4, p_fc[1] + 5), C_ANCHOR, rad=-0.06)
    arrow(ax, (bx_x + bx_w - 4, p_res[1]), (cx_x + 4, p_fc[1] - 5), C_TR, rad=0.08)

    # (d) evaluation contract ------------------------------------------------
    # The heading sits on its own line rather than beside the chain: sharing the
    # row left 74 pt per box, and three of the four labels are wider than that.
    panel(ax, 0, 0, FULL_PT, EVAL_H, None)
    ax.text(5, EVAL_H - 9, "(d) Evaluation", fontsize=PANEL, fontweight="bold",
            color=INK, va="center", ha="left")
    steps = ["Identical keys", "RMSE within station",
             "Paired station contrast", "Median across stations"]
    gap = 7.0
    sw = (FULL_PT - 10.0 - 3 * gap) / 4
    for i, s in enumerate(steps):
        x = 5.0 + i * (sw + gap)
        box(ax, x, 5, sw, 15, s, None, INK, title_size=BODY, align="center")
        if i:
            arrow(ax, (x - gap + 0.5, 12.5), (x - 0.5, 12.5), MUTED, lw=0.8)

    return fig


# ---------------------------------------------------------------------------
# Figure S3 -- full ThermoRoute architecture
# ---------------------------------------------------------------------------
def draw_architecture():
    G = 5.0  # stack gap
    # Interpretive-limits strip: a heading line plus three body lines.  At 7 pt
    # a line of about 105 characters is the most that fits inside 396 pt, so the
    # text is wrapped by hand and the strip is sized from the wrap.
    LIMITS = [
        "±1 °C bounds the deviation from the anchor only — not absolute error, "
        "event-tail error, or interval width.",
        "F3 is a retrospective realized-meteorology oracle, not an operational "
        "forecast.",
        "Fitting, loss, calibration and ablation detail remain in the SI text "
        "and tables.",
    ]
    FOOT = 2 * BOX_PAD + TITLE_LINE + TITLE_GAP + len(LIMITS) * BODY_LINE
    # Tallest stack in each row sets that row's height.  Top row is (a):
    # boxes of 3, 1, 2 and 0 body lines, then a two-line note.  Bottom row is
    # (c): three two-line boxes and a three-line note.
    th = (PANEL_TITLE_H + box_height(3) + box_height(1) + box_height(2)
          + box_height(0) + 2 * BODY_LINE + 4 * G)
    bh = PANEL_TITLE_H + 3 * box_height(2) + 3 * BODY_LINE + 3 * FLOW_GAP
    H = FOOT + 8.0 + bh + 8.0 + th + 26.0
    fig, ax = canvas(H)

    ax.text(0, H - 8, "Full ThermoRoute architecture and calibration dataflow",
            fontsize=TITLE, fontweight="bold", color=INK, va="center", ha="left")
    ax.text(0, H - 19, "Scientific modules; computational operators simplified.",
            fontsize=BODY, color=MUTED, va="center", ha="left")

    pw, gutter = 192.0, 12.0
    lx, rx = 0.0, pw + gutter
    brow = FOOT + 8.0
    trow = brow + bh + 8.0
    tin, bin_ = trow + th - PANEL_TITLE_H, brow + bh - PANEL_TITLE_H

    panel(ax, lx, trow, pw, th, "(a) Inputs and masks")
    panel(ax, rx, trow, pw, th, "(b) Anchor and proposal")
    panel(ax, lx, brow, pw, bh, "(c) Residual representation")
    panel(ax, rx, brow, pw, bh, "(d) Heads and calibration")

    # (a) --------------------------------------------------------------------
    sa = Stack(ax, lx, tin, pw)
    sa.box("32-day construction buffer",
           "WTEMP, FLOW, TEMP, PRCP,\nRHMEAN, DH, WDSP; each value\npaired with an observedness mask",
           C_INPUT)
    sa.box("Calendar and horizon", "season encoding; lead $h$ = 1, 3, 7 days", C_INPUT)
    sa.box("Forcing-regime extension (optional)",
           "future TEMP, PRCP, RHMEAN, DH, wind\nF3 is a realized gridded oracle",
           C_FORCING, dashed=True)
    sa.box("WLEVEL excluded from every head", None, MUTED, dashed=True,
           title_size=BODY)
    sa.note("Predictors are dated ≤ $t$; the target\nis never a predictor.")

    # (b) --------------------------------------------------------------------
    sb = Stack(ax, rx, tin, pw)
    sb.box("Fixed damped-persistence branch", None, C_ANCHOR, title_size=BODY)
    row_h = box_height(1)
    sb.y -= row_h
    b_clim = box(ax, sb.x, sb.y, 86, row_h, "Climatology", "$c$", C_ANCHOR)
    b_dec = box(ax, sb.x + 96, sb.y, sb.w - 96, row_h, "Decay",
                "$\\varphi^{\\,h}$", C_ANCHOR, math=True)
    arrow(ax, (b_clim[0] + 43, b_clim[1]), (b_dec[0] - 43, b_dec[1]), C_ANCHOR)
    sb.y -= sb.gap
    b_anc = sb.box("Anchor  $A$", None, C_ANCHOR, align="center")
    b_prop = sb.box("Learned relaxation proposal",
                    "equilibrium $e$ and rate $\\kappa$\ngive the proposal $P$",
                    C_TR)
    sb.note("$\\kappa$ is a statistical allocation parameter,\nnot a heat-transfer coefficient.")

    # (c) --------------------------------------------------------------------
    sc = Stack(ax, lx, bin_, pw, gap=FLOW_GAP)
    # The paper calls this the router throughout, and the ablation is
    # TR-noRouter; the draft's "variable-lag selector" had no referent in the
    # text.
    c_rout = sc.box("Router (variable × lag)",
                    "7 variables × lags 0–14,\nsparsemax allocation", C_TR)
    c_tcn = sc.box("Left-looking temporal encoder",
                   "TCN: 2 residual blocks, kernel 3,\ndilations 1 and 2", C_TR)
    c_moe = sc.box("Three-expert mixture",
                   "soft gate combines experts and\nemits the residual $r$", C_TR)
    sc.note("The router allocates weight over lags. It is\n"
            "not river-network routing: no verified\ntopology or travel time enters.")
    arrow(ax, (c_rout[0], c_rout[1] - box_height(2) / 2), (c_tcn[0], c_tcn[1] + box_height(2) / 2), C_TR)
    arrow(ax, (c_tcn[0], c_tcn[1] - box_height(2) / 2), (c_moe[0], c_moe[1] + box_height(2) / 2), C_TR)

    # (d) --------------------------------------------------------------------
    sd = Stack(ax, rx, bin_, pw, gap=FLOW_GAP)
    d_comb = sd.box("Combine  $z = P - A + r$", None, INK, align="center")
    d_pt = sd.box("Point head", "$ŷ = A + \\delta\\,\\tanh(z/\\delta)$,  $\\delta = 1.0$ °C",
                  C_OUT, align="center")
    row_h = box_height(1)
    sd.y -= row_h
    d_q = box(ax, sd.x, sd.y, 86, row_h, "Quantile heads", "q05, q50, q95", C_OUT)
    d_e = box(ax, sd.x + 96, sd.y, sd.w - 96, row_h, "Event head",
              "q90 exceedance", C_OUT)
    sd.y -= sd.gap
    sd.box("Calibration fitted on 2018 only",
           "CQR offset for intervals; Platt map\nfor event probability", C_OUT,
           title_size=BODY)
    arrow(ax, (d_comb[0], d_comb[1] - box_height(0) / 2),
          (d_pt[0], d_pt[1] + box_height(1) / 2), C_OUT)
    arrow(ax, (d_pt[0] - 45, d_pt[1] - box_height(1) / 2), (d_q[0], d_q[1] + row_h / 2), C_OUT)
    arrow(ax, (d_pt[0] + 45, d_pt[1] - box_height(1) / 2), (d_e[0], d_e[1] + row_h / 2), C_OUT)

    # interpretive limits ----------------------------------------------------
    panel(ax, 0, 0, FULL_PT, FOOT, None, fill="#FFF6EC", edge="#E8C9A0")
    ax.text(6, FOOT - BOX_PAD - TITLE_LINE / 2, "Interpretive limits",
            fontsize=BODY, fontweight="bold", color="#9A4A0C",
            va="center", ha="left")
    ax.text(6, FOOT - BOX_PAD - TITLE_LINE - TITLE_GAP, "\n".join(LIMITS),
            fontsize=BODY, color="#9A4A0C", va="top", ha="left", linespacing=1.23)

    return fig


def main() -> int:
    figstyle.use()
    family = figstyle.resolved_font()
    # figstyle sets mathtext to DejaVu Sans, which would embed a second typeface
    # in a figure whose prose is Nimbus Sans.  Point mathtext at the resolved
    # family instead and disable the fallback, so a glyph this font lacks fails
    # the build rather than silently arriving as Computer Modern.
    import matplotlib
    matplotlib.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.fallback": None,
    })
    print("resolved font:", family)
    # fig02_model_concept is not rendered here any more.  The manuscript's
    # Figure 2 is owned by render_fig02_one_fork.py; the draw_concept drawing
    # below is the retired boxes-and-arrows version and must not write the
    # shared stem, or the two renderers silently fight over one file.
    for stem, fn in (("figS03_thermoroute_architecture", draw_architecture),):
        fig = fn()
        check_boxes_fit(fig, fig.axes[0], stem)
        written = figstyle.save(fig, stem, OUT)
        plt.close(fig)
        print(f"wrote {stem}: " + ", ".join(p.name for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
