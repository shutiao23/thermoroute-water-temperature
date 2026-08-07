"""Shared figure style for the WRR submission.

One module so every renderer produces the same object: same typeface, same
sizes, same palette, same widths.  Import it and call :func:`use` before
creating any figure.

Why each choice
---------------

**Typeface.**  AGU asks for sans-serif line art with a minimum of ~8 pt at final
size.  Helvetica and Arial are the named preferences, but neither is installed
on every machine, and matplotlib falls back *silently* — declaring only those
two yields DejaVu Sans and nobody notices until a reviewer does.  The chain
below therefore names the metric-compatible free clones explicitly: Nimbus Sans
is URW's Helvetica, Liberation Sans is metrically Arial.  :func:`resolved_font`
reports what actually got used so a build can assert it.

**Palette.**  Wong's eight-colour set from *Nature Methods* (Wong 2011, "Points
of view: Color blindness").  It is the canonical Nature-style qualitative
palette and is safe for deuteranopia, protanopia and tritanopia.  Categorical
colour is never the only channel: pair it with marker or hatch.

**Sizes.**  This project's AGU class gives ``\\textwidth`` = 5.5 in = 139.7 mm,
so full width is 140 mm and single column 85 mm.  AGU's editorial limits are
50–85 mm single, 105–170 mm double, 228 mm maximum height.

**Layout.**  ``constrained_layout`` is on by default.  Every overlap in the
first draft of these figures came from manual placement — a colorbar requested
with ``ax=axes[0]`` lands on ``axes[1]``'s tick labels, an inset lands on a
title.  Constrained layout reserves the space instead of guessing, and
:func:`check_overlaps` fails the build if anything still collides.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1.0 / 25.4

# AGU/WRR widths in mm.  SINGLE and FULL are what this manuscript uses.
#
# FULL is 139.7, not 140.  The AGU class sets \textwidth to 5.5 in = 397.48 TeX
# pt = 139.7 mm; a 140 mm figure is 398.34 pt and overshoots by 0.30 mm.  That
# is invisible in print but it makes pdflatex emit an Overfull \hbox of exactly
# 0.85 pt for *every* figure, which buries real layout warnings in noise.
# Nothing clamps it back: the template's \pandocbounded is provided as an
# identity macro, so `width=\textwidth` is taken literally.
SINGLE_MM = 85.0
FULL_MM = 139.7
MAX_HEIGHT_MM = 228.0

# Wong (2011) Nature Methods colour-blind safe qualitative palette.
WONG = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
# Cycle order avoids yellow early (poor contrast on white) and black (reserved
# for annotation, reference lines and text).
CYCLE = [
    WONG["blue"], WONG["vermillion"], WONG["green"], WONG["orange"],
    WONG["purple"], WONG["sky"], WONG["yellow"],
]

# Semantic assignments used across figures, so a model keeps one colour.
SERIES = {
    "Persistence": WONG["black"],
    "DampedPersistence": WONG["orange"],
    "Climatology": "#7F7F7F",
    "LightGBM": WONG["blue"],
    "LSTM": WONG["sky"],
    "ThermoRoute": WONG["vermillion"],
    "PlainCausalTCN": WONG["green"],
    "PlainMLP": WONG["purple"],
}

GRID = "#D9D9D9"
RULE = "#444444"
MUTED = "#6E6E6E"

# Sequential and diverging maps built from the palette so figures stay in one
# colour world.  Both are monotone in lightness.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "wong_seq", ["#FFFFFF", WONG["sky"], WONG["blue"], "#00344F"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "wong_div", [WONG["blue"], "#9FD3EC", "#F7F7F7", "#F3C08A", WONG["vermillion"]]
)

_FONT_CHAIN = [
    "Helvetica",        # AGU's preference, if the machine has it
    "Nimbus Sans",      # URW Helvetica clone, metrically identical
    "Arial",            # AGU's other preference
    "Liberation Sans",  # metrically Arial
    "DejaVu Sans",      # last resort; matplotlib always ships it
]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": _FONT_CHAIN,
    "mathtext.fontset": "dejavusans",

    # AGU minimum is ~8 pt at final size; ticks may go to 7.5 pt.
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.0,
    "axes.titleweight": "bold",

    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.6,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": RULE,
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": RULE,
    "ytick.color": RULE,

    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.2,

    "axes.prop_cycle": plt.cycler(color=CYCLE),

    # Real text in the vector outputs, not outlines.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "savefig.bbox": "standard",   # constrained layout already fits the figure
    "figure.dpi": 120,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.w_pad": 0.04,
    "figure.constrained_layout.hspace": 0.06,
    "figure.constrained_layout.wspace": 0.06,
}


def use() -> None:
    """Apply the style globally.  Call once, before creating any figure."""
    matplotlib.rcParams.update(RC)


def resolved_font() -> str:
    """Return the family matplotlib actually resolved, for build assertions.

    The chain is passed as a list rather than the string ``"sans-serif"``:
    ``FontProperties`` treats a hyphenated string as a fontconfig pattern and
    fails to parse it.
    """
    from matplotlib.font_manager import FontProperties, findfont, get_font
    chain = matplotlib.rcParams.get("font.sans-serif") or _FONT_CHAIN
    path = findfont(FontProperties(family=list(chain)), fallback_to_default=True)
    return get_font(path).family_name


def figsize(width_mm: float = FULL_MM, height_mm: float | None = None,
            *, ratio: float = 0.62) -> tuple[float, float]:
    """Figure size in inches from a millimetre width.

    ``ratio`` is height/width when ``height_mm`` is not given.  Raises if the
    result would exceed AGU's 228 mm page limit.
    """
    height = height_mm if height_mm is not None else width_mm * ratio
    if height > MAX_HEIGHT_MM:
        raise ValueError(f"height {height:.0f} mm exceeds the {MAX_HEIGHT_MM:.0f} mm limit")
    return width_mm * MM, height * MM


def panel_label(ax, text: str, *, dx: float = -0.02, dy: float = 1.04) -> None:
    """Place a bold panel label in axes coordinates, outside the data area."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", va="bottom", ha="left")


def colorbar(fig, mappable, ax, *, label: str = "", **kwargs):
    """Attach a colorbar without stealing a neighbour's label space.

    ``fig.colorbar(m, ax=axes[0])`` places the bar immediately right of
    ``axes[0]``, which is where ``axes[1]``'s tick labels live.  Constrained
    layout only reserves the room if the colorbar is part of the layout, so
    always route colorbars through here.
    """
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03, **kwargs)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.0, width=0.5, length=2.0)
    if label:
        cb.set_label(label, fontsize=7.5)
    return cb


def check_overlaps(fig, *, tolerance: float = 1.0) -> list[tuple[str, str]]:
    """Return pairs of visible text artists whose boxes intersect.

    Every collision in the first draft of these figures was a text artist
    landing on another one.  A build that renders should assert this is empty
    rather than trust the eye.  ``tolerance`` is in display points; boxes that
    merely touch are not reported.
    """
    fig.canvas.draw()
    items = []
    for ax in fig.axes:
        items.extend(t for t in ax.texts if t.get_visible() and t.get_text().strip())
        items.extend(t for t in ax.get_xticklabels() + ax.get_yticklabels()
                     if t.get_visible() and t.get_text().strip())
        for holder in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if holder.get_visible() and holder.get_text().strip():
                items.append(holder)
    items.extend(t for t in fig.texts if t.get_visible() and t.get_text().strip())

    boxes = []
    for t in items:
        try:
            boxes.append((t, t.get_window_extent(renderer=fig.canvas.get_renderer())))
        except Exception:
            continue

    hits: list[tuple[str, str]] = []
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            dx = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            dy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if dx > tolerance and dy > tolerance:
                hits.append((ti.get_text()[:40], tj.get_text()[:40]))
    return hits


def save(fig, stem: str, out_dir, *, formats=("pdf", "png", "svg"),
         strict: bool = True) -> list:
    """Save a figure in every requested format, refusing on text collisions."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = check_overlaps(fig)
    if hits and strict:
        detail = "; ".join(f"{a!r} x {b!r}" for a, b in hits[:6])
        raise ValueError(f"{stem}: {len(hits)} overlapping text pairs -> {detail}")
    written = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, format=ext)
        written.append(path)
    return written
