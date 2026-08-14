#!/usr/bin/env python3
r"""Figure 2: ONE PIPE, ONE FORK.

A single wide left-to-right channel.  Shared information enters, passes the
shared damped-persistence anchor, splits at exactly one point -- the regression
target -- and immediately rejoins into one key registry and one station-first
metric chain.  One stream (LightGBM) braids around the anchor instead of
through it, because it is a raw-target model; that exception is drawn as a
shape, not written as a caveat.

Nothing here is a result.  The only curve on the page is the analytic shape of
the anchor, drawn with no axes, no ticks, no numbers and no markers, inside a
panel whose header reads "schematic - not data", under a figure subtitle that
says no results are shown.

Everything is laid out in points on a full-bleed axes, so one data unit is one
printed point: a declared 7.5 pt label is 7.5 pt on the page.

A first version of this figure carried two floating footnotes squeezed between
the anchor inset and the fork panel, at a 69 mm canvas height.  They collided
with the inset (a patch, invisible to the text-text collision gate) and forced
every line stack on the page down to sub-1.0 leading.  The footnotes are gone:
the residual-bound sentence lives in the inset next to the band it describes,
the raw-target sentence is the caption's, and every multi-line block now sets
7.5 pt type on a >= 9 pt pitch.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/lzq/workspace/parttime/thermoroute-water-temperature/paper")
import figstyle  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

import pathlib  # noqa: E402
OUT = pathlib.Path(__file__).resolve().parent

PT_PER_MM = 72.0 / 25.4
W = figstyle.FULL_MM * PT_PER_MM   # 396.0 pt = 139.7 mm, the AGU \textwidth
H = 240.0                          # 84.7 mm

# ---------------------------------------------------------------------------
# Type scale.  Floor is the house 7.5 pt (WRR_FIGURE_STYLE_GUIDE), not the
# checker's 6.0.  The only superscript on the page is the h in phi^h; mathtext
# sets superscripts at 0.7x, so the formula base size is chosen to land the
# exponent exactly on the floor: 10.75 * 0.7 = 7.5.
# ---------------------------------------------------------------------------
TITLE = 10.0
SUB = 7.5
STAMP = 8.0    # zone headers: ONE SHARED RUN / THE ONLY FORK
NAME = 7.5     # model names in the fork
CHIP = 8.5     # the regression target itself
STATION = 7.5  # bold labels under the shared trunk
BOXT = 8.0     # box titles in the right column
BODY = 7.5
SMALL = 7.5
TINY = 7.5
MATH = 10.75

INK = "#1A1A1A"
MUTED = figstyle.MUTED
PIPE_FILL = "#DFE3E8"
PIPE_EDGE = "#AEB6BF"

C_ANCHOR = figstyle.SERIES["DampedPersistence"]   # Wong orange
C_TR = figstyle.SERIES["ThermoRoute"]             # Wong vermillion
MODELS = [
    ("LightGBM", figstyle.SERIES["LightGBM"], r"$y$", True),
    ("ResidualLightGBM", figstyle.WONG["sky"], r"$y - A$", False),
    ("Plain causal TCN", figstyle.SERIES["PlainCausalTCN"], r"$y - A$", False),
    ("ThermoRoute", C_TR, r"$y - A$", False),
]

# ---------------------------------------------------------------------------
# Geometry, in points.  x runs 0..396, y runs 0..240, y up.
# Three columns: shared run + anchor inset (0..150), fork panel (158..286),
# prediction / registry / reduction (292..396).  A banner row of zone headers
# sits at y = 184.5 across all three.
# ---------------------------------------------------------------------------
YC = 134.0       # the axis of the shared channel = centre of fork lane 2
TRUNK_H = 14.0
SEG_A = (0.0, 86.0)      # same information
SEG_B = (86.0, 128.0)    # same anchor -- the island the raw-target stream skirts
FORK_X0, FORK_X1 = 158.0, 286.0
NAME_X = 166.0
CHIP_X0, CHIP_X1 = 240.0, 283.0
ISLAND = (154.0, 63.0, 290.0, 193.0)     # x0, y0, x1, y1
COL_X0, COL_X1 = 292.0, 396.0            # right column
LANE_H = 20.0
FORK_TOP = 170.0                          # top edge of lane 1
LANE_GAPS = (9.0, 6.0, 6.0)              # lane 1 sits apart: it is the exception
BANNER_Y = 184.5

INSET = (0.0, 6.0, 150.0, 86.0)

_fits: list = []


def lane_rows():
    """(y0, y1, ycentre) for the four fork lanes, top down."""
    rows, top = [], FORK_TOP
    for i in range(4):
        rows.append((top - LANE_H, top, top - LANE_H / 2))
        if i < 3:
            top -= LANE_H + LANE_GAPS[i]
    return rows


ROWS = lane_rows()


def tint(hexc: str, f: float = 0.90) -> tuple:
    """Blend a palette colour toward white; f = 1 is white."""
    r, g, b = matplotlib.colors.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def canvas():
    fig = plt.figure(figsize=(W / 72.0, H / 72.0))
    fig.set_layout_engine("none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    # set_axis_off alone leaves the tick Text artists alive, and figstyle's
    # out-of-bounds gate walks every Text on the figure regardless of whether it
    # is drawn -- the hidden "0", "50", ... would fail the build.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    return fig, ax


def txt(ax, x, y, s, size, colour=INK, *, weight="normal", ha="left",
        va="center", box=None, z=6, spacing=1.22):
    t = ax.text(x, y, s, fontsize=size, color=colour, fontweight=weight,
                ha=ha, va=va, zorder=z, linespacing=spacing)
    if box is not None:
        _fits.append((t, box[0], box[1]))
    return t


def rrect(ax, x0, y0, x1, y1, *, fill, edge, lw=0.8, r=3.0, z=1, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, facecolor=fill, edgecolor=edge, zorder=z,
        linestyle=(0, (2.2, 1.4)) if dashed else "solid"))


def sflow(ax, p0, p1, colour, *, lw=2.2, bow=0.5, head=False, z=3,
          c1=None, c2=None):
    """A Sankey-style S-curve from p0 to p1, horizontal at both ends."""
    (x0, y0), (x1, y1) = p0, p1
    if c1 is None:
        c1 = (x0 + (x1 - x0) * bow, y0)
    if c2 is None:
        c2 = (x1 - (x1 - x0) * bow, y1)
    path = Path([p0, c1, c2, p1],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    if head:
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>",
                                     mutation_scale=6, linewidth=lw,
                                     color=colour, zorder=z,
                                     shrinkA=0, shrinkB=0))
    else:
        ax.add_patch(PathPatch(path, fill=False, linewidth=lw,
                               edgecolor=colour, zorder=z, capstyle="round"))


def vdown(ax, x, y0, y1, colour=MUTED, lw=0.9):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=lw, color=colour,
                                 zorder=3, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------------------
def draw_trunk(ax):
    """The shared run: deliberately plain.  Two stages, one bar."""
    y0, y1 = YC - TRUNK_H / 2, YC + TRUNK_H / 2
    # Started off-canvas so the left end is a cut, not a capsule: the channel
    # arrives from outside the frame rather than beginning in mid-air.
    rrect(ax, SEG_A[0] - 6, y0, SEG_B[1], y1, fill=PIPE_FILL, edge=PIPE_EDGE,
          lw=0.7, r=3.0, z=1)
    rrect(ax, SEG_B[0], y0, SEG_B[1], y1, fill=tint(C_ANCHOR, 0.72),
          edge=C_ANCHOR, lw=1.0, r=3.0, z=2)

    txt(ax, 2, BANNER_Y, "ONE SHARED RUN", STAMP, MUTED, weight="bold",
        box=(2, 118))
    # Two label columns under the trunk, separated by a real gutter: the first
    # ends by x = 88 and the second starts at 92.  The fit boxes are the
    # gutter's enforcement, not decoration.
    txt(ax, 2, 115, "Same information set", STATION, INK, weight="bold",
        box=(2, 88))
    # Naming the regime matters: "nothing dated after the issue time" is true
    # of F0, and the paper also runs an F3 arm in which realized future
    # meteorology is handed to every model.  Unqualified, this label would
    # claim a control the paper does not hold everywhere.
    txt(ax, 2, 105.5, "nothing dated after the", SMALL, MUTED, box=(2, 88))
    txt(ax, 2, 96.5, "issue time (regime F0)", SMALL, MUTED, box=(2, 88))

    txt(ax, 90, 115, "Same anchor", STATION, INK, weight="bold",
        box=(90, 144))
    txt(ax, 90, 105.5, "used by three", SMALL, MUTED, box=(90, 144))
    txt(ax, 90, 96.5, "of the four", SMALL, MUTED, box=(90, 144))


def draw_fork(ax):
    x0, y0, x1, y1 = ISLAND
    rrect(ax, x0, y0, x1, y1, fill="#F1F3F6", edge="#5F6A75", lw=1.4, r=4.0, z=1)
    txt(ax, NAME_X, BANNER_Y, "THE ONLY FORK", STAMP, INK, weight="bold",
        box=(NAME_X, 240))
    # Right-aligned to the panel edge rather than centred on the chip column:
    # at house type sizes the centred label is wider than the column it names,
    # and the panel edge is the only fixed landmark it can hang from.
    txt(ax, FORK_X1 - 6, BANNER_Y, "target", SMALL, MUTED,
        ha="right", box=(FORK_X1 - 60, FORK_X1 - 4))

    for (name, colour, target, raw), (ly0, ly1, lyc) in zip(MODELS, ROWS):
        rrect(ax, FORK_X0, ly0, FORK_X1, ly1, fill=tint(colour, 0.90),
              edge=colour, lw=0.9, r=3.0, z=2)
        # Model colour lives in a track marker, not in the type.  Sky blue set
        # as 7 pt bold text drops to a pale grey the moment the journal prints
        # the page in black and white; a 4 pt saturated bar does not.
        rrect(ax, FORK_X0 + 0.6, ly0 + 1.4, FORK_X0 + 4.6, ly1 - 1.4,
              fill=colour, edge=colour, lw=0.0, r=1.2, z=3)
        txt(ax, NAME_X, lyc, name, NAME, INK, weight="bold",
            box=(NAME_X, CHIP_X0 - 5), z=6)
        # The chips are the argument: three identical, one not.  The odd one is
        # separated by shape (dashed, unfilled) as well as by the extra gap
        # above its lane, so the contrast survives greyscale printing.
        rrect(ax, CHIP_X0, lyc - 7.0, CHIP_X1, lyc + 7.0,
              fill="#FFFFFF" if raw else "#EFF1F4",
              edge=INK, lw=1.0 if raw else 0.9, r=2.0, z=3, dashed=raw)
        txt(ax, (CHIP_X0 + CHIP_X1) / 2, lyc, target, CHIP, INK, ha="center",
            box=(CHIP_X0 + 2, CHIP_X1 - 2), z=6)

    # Anchor feeds three of the four lanes.
    for (_, _, lyc), (_, _, _, raw) in zip(ROWS, MODELS):
        if raw:
            continue
        sflow(ax, (SEG_B[1], YC), (FORK_X0, lyc), C_ANCHOR, lw=2.0, head=True,
              c1=(148.0, YC), c2=(FORK_X0 - 12.0, lyc))
    # ... and the fourth braids around the anchor island: same information,
    # no anchor.  It leaves the trunk on the grey segment, before the anchor
    # chip begins, and climbs steeply enough to clear the chip's top edge --
    # a reader who never reads the caption still sees one stream skirt the
    # anchor.
    blue = MODELS[0][1]
    sflow(ax, (58.0, YC + 4.0), (FORK_X0, ROWS[0][2]), blue, lw=2.0,
          head=True, c1=(68.0, 172.0), c2=(112.0, ROWS[0][2]))
    txt(ax, 88.0, 173.0, "skips the anchor", TINY, blue, box=(86, 156))


def draw_merge(ax):
    # The four streams converge behind the registry box: they are one channel
    # from the moment scoring starts, which is the point of the box they enter.
    for (_, colour, _, _), (_, _, lyc) in zip(MODELS, ROWS):
        sflow(ax, (FORK_X1, lyc), (COL_X0 + 10, YC), colour, lw=1.3, bow=0.62)


def draw_column(ax):
    # Prediction box: how a residual becomes a forecast, and the one exception.
    rrect(ax, COL_X0, 160, COL_X1, 196, fill="#FBFBFC", edge="#C9D1DB",
          lw=0.6, r=3.0, z=1)
    txt(ax, COL_X0 + 6, 188.0, "Prediction", BOXT, INK, weight="bold",
        box=(COL_X0 + 5, COL_X1 - 4))
    txt(ax, COL_X0 + 6, 177.0, r"$ŷ = A + r$", CHIP, INK,
        box=(COL_X0 + 5, COL_X1 - 4))
    txt(ax, COL_X0 + 6, 166.5, "LightGBM: ŷ direct", SMALL, MUTED,
        box=(COL_X0 + 5, COL_X1 - 4))

    # The registry box is the fairness claim, so it gets the ink border and
    # the bold close.  7.5 pt on a 9 pt pitch: the first version of this figure
    # set these five lines on 7.3 pt and they read as one smear.
    rrect(ax, COL_X0, 92, COL_X1, 154, fill="#FFFFFF", edge=INK, lw=1.1,
          r=3.0, z=5)
    txt(ax, COL_X0 + 6, 146.0, "One key registry", BOXT, INK, weight="bold",
        box=(COL_X0 + 6, COL_X1 - 5))
    body = (("Every model scored on", MUTED, "normal", 135.5),
            ("the same station × date", MUTED, "normal", 126.5),
            ("× lead keys.", MUTED, "normal", 117.5),
            ("Declining to predict", INK, "bold", 107.5),
            ("is a failure.", INK, "bold", 98.5))
    for line, colour, weight, y in body:
        txt(ax, COL_X0 + 6, y, line, SMALL, colour,
            weight=weight, box=(COL_X0 + 6, COL_X1 - 5))

    steps = (("RMSE within station", 62.0, 77.0),
             ("Paired station contrast", 36.0, 51.0),
             ("Median across stations", 10.0, 25.0))
    prev_bottom = 92.0
    for label, sy0, sy1 in steps:
        vdown(ax, (COL_X0 + COL_X1) / 2, prev_bottom, sy1)
        rrect(ax, COL_X0, sy0, COL_X1, sy1, fill="#FFFFFF", edge=INK, lw=0.7,
              r=2.5, z=2)
        txt(ax, (COL_X0 + COL_X1) / 2, (sy0 + sy1) / 2, label, BODY, INK,
            ha="center", box=(COL_X0 + 3, COL_X1 - 3), z=6)
        prev_bottom = sy0


def draw_inset(ax):
    """The shape of the anchor.  Schematic: no axes, no ticks, no numbers.

    The residual-bound sentence lives here, right-aligned under the |r| band
    it describes, because this is the only place the band is drawn.  In the
    first version it floated in the middle of the figure as a footnote and
    collided with this panel.
    """
    x0, y0, x1, y1 = INSET
    rrect(ax, x0, y0, x1, y1, fill="#FBFBFC", edge="#C9D1DB", lw=0.6, r=3.0, z=1)
    txt(ax, x0 + 5, 78.0, "The anchor A", BODY, INK, weight="bold",
        box=(x0 + 4, 66))
    txt(ax, x1 - 5, 78.0, "schematic — not data", TINY, C_TR, ha="right",
        box=(68, x1 - 3))
    txt(ax, (x0 + x1) / 2, 62.5, r"$A = c + \varphi^{\,h}\,(y - c)$", MATH, INK,
        ha="center", box=(x0 + 4, x1 - 4))

    # One analytic exponential, drawn once, with no markers and no scale.  The
    # value axis carries no ticks and no units, so there is nothing on it a
    # reader could read a number off.
    cx0, cx1 = 16.0, 132.0
    y_c, y_start = 30.0, 48.0
    u = np.linspace(0.0, 1.0, 240)
    xs = cx0 + u * (cx1 - cx0)
    ys = y_c + (y_start - y_c) * np.exp(-3.1 * u)
    band = 2.6
    ax.fill_between(xs, ys - band, ys + band, facecolor=tint(C_TR, 0.72),
                    edgecolor="none", zorder=2)
    ax.plot([cx0 - 4, cx1 + 2], [y_c, y_c], lw=0.7, color=MUTED,
            linestyle=(0, (2.4, 1.6)), zorder=3)
    ax.plot(xs, ys, lw=1.6, color=C_ANCHOR, zorder=4, solid_capstyle="round")
    ax.plot([cx0], [y_start], marker="o", ms=2.6, color=C_ANCHOR, zorder=5)

    txt(ax, cx0 - 5, y_start, "y", SMALL, MUTED, ha="right", box=(x0 + 3, cx0 - 4))
    txt(ax, cx1 + 4, y_c, "c", SMALL, MUTED, box=(cx1 + 3, x1 - 3))
    ax.plot([120, 120], [33.4, 37.4], lw=0.6, color=C_TR, zorder=5)
    txt(ax, x1 - 6, 41.0, r"$|r| < 1$ °C", TINY, C_TR, ha="right",
        box=(96, x1 - 4))
    txt(ax, x1 - 6, 20.5, "bounds r about A, not the error", TINY, MUTED,
        ha="right", box=(24, x1 - 4))
    txt(ax, x0 + 5, 12.0, "lead time h, from the issue time", TINY, MUTED,
        box=(x0 + 4, x1 - 4))


def check_fits(fig, ax, stem="fig02_model_concept"):
    """Fail if a label is wider than the column it was drawn for.

    figstyle's gate catches text on text and text off the canvas.  It cannot
    catch text that overruns the patch drawn around it, because a patch is not
    a Text artist -- so a label can print straight through its box and every
    automated check still passes.
    """
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for artist, lo, hi in _fits:
        e = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        (dx0, _), (dx1, _) = inv.transform([(e.x0, e.y0), (e.x1, e.y1)])
        if dx0 < lo - 0.25 or dx1 > hi + 0.25:
            bad.append(f"{artist.get_text()[:34]!r} overruns by "
                       f"{max(lo - dx0, dx1 - hi):.1f} pt")
    if bad:
        raise ValueError(f"{stem}: {len(bad)} label(s) too wide -> "
                         + "; ".join(bad[:8]))


def build():
    fig, ax = canvas()
    # Not "same anchor" in the headline: the anchor is shared by three of the
    # four, and the figure says so twice below.  Information and keys are the
    # controls that hold for all four; the target is the axis that varies.
    txt(ax, 0, 229.0,
        "Same information, same keys — the only fork is the regression target",
        TITLE, INK, weight="bold", box=(0, W))
    txt(ax, 0, 216.5,
        "Protocol schematic; no results shown. Tuning is not equalised; "
        "the optional F3 forcing axis is not drawn.",
        SUB, MUTED, box=(0, W))
    draw_inset(ax)
    draw_trunk(ax)
    draw_fork(ax)
    draw_merge(ax)
    draw_column(ax)
    return fig, ax


def main() -> int:
    figstyle.use()
    family = figstyle.resolved_font()
    # figstyle points mathtext at DejaVu Sans, which would embed a second
    # typeface next to Nimbus Sans prose.  Point it at the resolved family and
    # kill the fallback, so a missing glyph fails the build instead of arriving
    # silently as Computer Modern.
    matplotlib.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.fallback": None,
    })
    print("resolved font:", family)
    fig, ax = build()
    check_fits(fig, ax)
    written = figstyle.save(fig, "fig02_model_concept", OUT, formats=("png", "pdf"),
                            strict=True)
    print(f"{W / PT_PER_MM:.1f} x {H / PT_PER_MM:.1f} mm")
    print("wrote " + ", ".join(str(p) for p in written))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
