#!/usr/bin/env python3
r"""Figure 5: what each axis of the study is worth, on one scale.

The paper's central claim is that what a model is *given* at issue time, not
which model it is, sets the achievable skill on this problem. That claim is
currently spread across four result sections and a dozen tables, and a reader
has to hold nine numbers in their head to see it. It is really one picture:
every quantity the study can vary, plotted on a common log axis, spanning three
orders of magnitude with the estimator at the bottom.

It is deliberately not called a budget.  A budget implies the parts sum to a
whole, and these do not: the forcing and information axes are separate
conditional designs whose component shares already exceed 100%, and the
manuscript's consistency gate rejects the word for exactly that reason.  The
figure compares magnitudes; it does not decompose one.

Design choices worth stating, because each one is a way the figure could
mislead.

*Log axis.* The quantities run from 0.002 to 1.7 degC. On a linear axis
everything except local thermal state collapses onto the zero line and the
figure would say only "one thing is big", which is less than the paper knows.
A log axis makes the ordering and the ratios readable, and ratios are the
claim: local state is two orders of magnitude above the estimator.

*Signed magnitudes on a log axis.* Two entries are negative (the network beats
the tree given oracle forcing). Rather than drop them or take absolute values
silently, they are drawn as open markers and labelled, so the reader sees that
the architecture axis has a sign flip rather than being uniformly small.

*Every row here is post-outcome, so none is distinguished as confirmatory.*
An earlier draft of this figure planned to mark the two statuses differently;
that would have implied some row was confirmatory, and none is. The status is
stated once in the caption instead, which is the honest place for a fact that
applies uniformly.

*One instrument throughout.* All rows are measured with the damped-residual
tree. The information axes could equally use the raw-target tree -- Table 4.15
does -- but the estimator axis is by construction a contrast between two models
sharing an anchor, so using the raw-target tree for some rows and the residual
tree for others would put a change of instrument inside a figure about changes
of information. The ordering is identical either way; only the magnitudes on
the top row shift, by about a sixth.

*The minimum detectable effect is drawn, not omitted.* A band at the cohort's
resolution floor makes the difference between "measured as zero" and "too small
for this design to see" visible, which is the distinction DLOG-035 was written
about.

Every value is read from a scored authority artifact. No number in this file is
typed by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper"))
import figstyle  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = ROOT / "paper" / "agu_submission" / "figures"
FINAL = ROOT / "outputs" / "final"

TREE = "ResidualLightGBM"
LEAD = 7                      # the lead at which every axis has been measured


class FigureDataError(RuntimeError):
    """An authority this figure depends on is missing or has changed shape."""


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FigureDataError(f"missing authority artifact: {path}")
    return pd.read_parquet(path)


def _one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in filters.items():
        mask &= (frame[key] == value).to_numpy()
    rows = frame[mask]
    if len(rows) != 1:
        raise FigureDataError(
            f"expected exactly one row for {filters}, found {len(rows)}"
        )
    return rows.iloc[0]


def collect() -> pd.DataFrame:
    """Assemble the budget from the scored authorities, at the seven-day lead."""
    ladder = _read(FINAL / "information_ladder_v6_authority_v1"
                   / "information_ladder_v6_contrasts.parquet")
    forcing = _read(FINAL / "forcing_regime_v5_observed_inference_authority_v1"
                    / "forcing_v5_inference_contrasts.parquet")
    fxl = _read(FINAL / "forcing_information_interaction_v1"
                / "forcing_information_interaction.parquet")
    geom = _read(FINAL / "ladder_geometry_interaction_v1"
                 / "ladder_geometry_interaction.parquet")
    arch = _read(FINAL / "architecture_geometry_interaction_v1"
                 / "architecture_geometry_interaction.parquet")

    rows: list[dict[str, object]] = []

    def add(group, label, value, low, high, *, post, mde=np.nan,
            interval="cluster bootstrap"):
        rows.append({"group": group, "label": label, "value": float(value),
                     "low": float(low), "high": float(high),
                     "post_outcome": bool(post), "mde": float(mde),
                     "interval_kind": interval})

    # --- information the model is given about the river itself
    for contrast, label in (("L2-L0", "All local thermal state"),
                            ("L2-L1", "Recent temperature sequence"),
                            ("L1-L0", "Own long-term statistics"),
                            ("L2_U2-L2", "Local discharge")):
        row = _one(ladder, contrast=contrast, model=TREE, horizon=LEAD)
        add("Local observation", label, row["median_value_degC"],
            row["ci_low"], row["ci_high"], post=True)


    # --- information about the weather that has not happened yet
    #
    # Taken from the F-by-L authority rather than from Table 4.13, because that
    # table is scored on the primary split while every other row here is scored
    # on the whole-region folds.  Mixing the two would put a 10% geometry
    # difference inside a figure whose readable content is ratios, and a reader
    # differencing two adjacent bars would be differencing two experiments.
    row = _one(fxl, quantity="forcing_value_at_L0", model=TREE, horizon=LEAD)
    add("Future weather", "Realized future meteorology (oracle)",
        row["median_degC"], row["ci_low"], row["ci_high"], post=True)
    row = _one(fxl, quantity="forcing_value_at_L2", model=TREE, horizon=LEAD)
    add("Future weather", "Realized future meteorology, observations withheld",
        row["median_degC"], row["ci_low"], row["ci_high"], post=True)

    # --- how the holdout was drawn
    for quantity, label in (("geometry_penalty_at_L0", "Spatial holdout, gauged"),
                            ("geometry_penalty_at_L2", "Spatial holdout, obs. withheld")):
        row = _one(geom, quantity=quantity, model=TREE, horizon=LEAD)
        add("Study design", label, row["median_degC"],
            row["ci_low"], row["ci_high"], post=True)

    # --- which model class
    for level, label in (("L0", "Estimator, gauged"),
                         ("L2", "Estimator, observations withheld")):
        row = _one(arch, quantity="architecture_penalty_tcn_minus_tree",
                   level=level, geometry="whole_region", forcing="F0", horizon=LEAD)
        add("Estimator", label, row["median_degC"], row["ci_low"], row["ci_high"],
            post=True, mde=row["mde_degC"])
    row = _one(arch, quantity="architecture_penalty_tcn_minus_tree",
               level="L0", geometry="whole_region", forcing="F3_full", horizon=LEAD)
    add("Estimator", "Estimator, gauged + oracle forcing",
        row["median_degC"], row["ci_low"], row["ci_high"],
        post=True, mde=row["mde_degC"])

    # --- how the reference itself was built
    #
    # Reviewer point M06: every other row on this axis is conditional on one
    # anchor, and the anchor is a fitted object whose construction moves the
    # seven-day headline by a factor of 2.4.  Leaving that off the figure let a
    # reader compare an estimator effect against a forcing effect without
    # seeing that the reference specification is larger than either.  The bar
    # is the range across the seven predeclared one-factor variants, NOT a
    # bootstrap interval, and it is flagged as such in the data file.
    anchor = _read(FINAL / "anchor_sensitivity.parquet")
    anchor = anchor[anchor["horizon"] == LEAD]
    if anchor.empty:
        raise FigureDataError("no anchor variants at this lead")
    base = float(anchor.loc[anchor.variant == "baseline",
                            "median_station_delta_rmse"].iloc[0])
    add("Study design", "Reference construction (7 anchor variants)",
        base,
        float(anchor.median_station_delta_rmse.min()),
        float(anchor.median_station_delta_rmse.max()),
        post=False, interval="range over predeclared anchor variants")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FigureDataError("no rows assembled")
    return frame


# ---------------------------------------------------------------- geometry
W_MM = figstyle.FULL_MM          # 139.7, mandatory
H_MM = 56.7                      # the number this candidate is competing on

PITCH_MM = 3.4                   # centre-to-centre, rows inside a group
GROUP_GAP_MM = 1.7               # extra space between groups
PAD_TOP_MM = 1.5                 # air above the top row
PAD_BOT_MM = 2.0                 # air below the bottom row (keeps descenders
                                 # off the x spine)

M_LEFT_MM = 1.2                  # canvas edge -> group name
GUTTER_MM = 2.6                  # group name -> row label column
TICKPAD_MM = 1.3                 # row label -> axis spine
M_RIGHT_MM = 1.6
M_TOP_MM = 0.8
M_BOTTOM_MM = 9.9                # tick labels + axis label + one-line legend

FS_LABEL = 7.0                   # row labels
FS_GROUP = 7.0                   # group names (bold)
FS_TICK = 7.0
FS_AXIS = 7.2
FS_LEGEND = 6.8                  # smallest type in the figure; floor is 6.0

FLOOR = 1e-3
XMAX = 2.6

GROUP_COLOUR = {
    "Local observation": figstyle.SERIES["ThermoRoute"],
    "Future weather": figstyle.SERIES["LightGBM"],
    "Study design": figstyle.SERIES["LSTM"],
    "Estimator": figstyle.SERIES["Persistence"],
}
GROUP_ORDER = ["Local observation", "Future weather", "Study design", "Estimator"]

# The group name absorbs the noun the rows used to repeat; the row keeps the
# part that actually varies.  Nothing here changes what a row means.
GROUP_TITLE = {
    "Local observation": "Local\nobservation",
    "Future weather": "Future weather\n(oracle)",
    "Study design": "Study\ndesign",
    "Estimator": "Estimator\n(TCN - tree)",
}
SHORT = {
    "All local thermal state": "All local thermal state",
    "Recent temperature sequence": "Recent temperature sequence",
    "Own long-term statistics": "Own long-term statistics",
    "Local discharge": "Local discharge",
    "Realized future meteorology (oracle)": "gauged",
    "Realized future meteorology, observations withheld": "observations withheld",
    "Spatial holdout, gauged": "Spatial holdout, gauged",
    "Spatial holdout, obs. withheld": "Spatial holdout, obs. withheld",
    "Reference construction (7 anchor variants)": "Reference construction (7 anchors)",
    "Estimator, gauged": "gauged",
    "Estimator, observations withheld": "observations withheld",
    "Estimator, gauged + oracle forcing": "gauged + oracle forcing",
}


def _darken(hex_colour: str, factor: float = 0.70) -> tuple[float, float, float]:
    """A darker tint of the group colour, for the group name only.

    The row labels inside "Future weather" and "Estimator" say only what
    distinguishes them, so the group name is now load-bearing text rather than
    decoration -- and Wong's sky blue at 7 pt on white is about 2.2:1, which is
    fine for a 1.5 pt rule and marginal for a word a reader has to read.  The
    rule and the markers keep the exact palette colour; only the text is
    darkened, so the hue still ties the name to its block.
    """
    from matplotlib.colors import to_rgb
    return tuple(c * factor for c in to_rgb(hex_colour))


def layout(frame: pd.DataFrame):
    """Row centres in millimetres, measured up from the bottom row."""
    pos: dict[int, float] = {}
    blocks: list[tuple[str, float, float]] = []
    y = 0.0
    for group in reversed(GROUP_ORDER):             # first group ends up on top
        idx = list(frame.index[frame["group"] == group])
        lo = y
        for i in reversed(idx):
            pos[i] = y
            y += PITCH_MM
        blocks.append((group, lo, y - PITCH_MM))
        y += GROUP_GAP_MM
    top = y - GROUP_GAP_MM - PITCH_MM
    return pos, blocks, top


def render(frame: pd.DataFrame):
    figstyle.use()
    fig = plt.figure(figsize=figstyle.figsize(W_MM, H_MM), constrained_layout=False)
    # The rc default is constrained layout; this figure places its one axes by
    # measured millimetres instead, so the engine is switched off explicitly
    # rather than left half-enabled (which warns on save and owns nothing).
    fig.set_layout_engine("none")

    pos, blocks, top = layout(frame)
    axes_h_mm = top + PAD_TOP_MM + PAD_BOT_MM
    if abs((M_TOP_MM + axes_h_mm + M_BOTTOM_MM) - H_MM) > 0.05:
        raise SystemExit(
            f"vertical budget does not close: {M_TOP_MM + axes_h_mm + M_BOTTOM_MM:.2f}"
            f" mm of content in a {H_MM:.2f} mm figure"
        )

    # Provisional rectangle; the left edge is re-derived from measured text below.
    ax = fig.add_axes([0.45, M_BOTTOM_MM / H_MM, 0.5, axes_h_mm / H_MM])

    ticks, labels = [], []
    for i in frame.index:
        row = frame.loc[i]
        y = pos[i]
        colour = GROUP_COLOUR[row["group"]]
        value, low, high = row["value"], row["low"], row["high"]
        negative = value < 0
        mag = max(abs(value), FLOOR)
        lo, hi = sorted((abs(low), abs(high)))
        if low <= 0.0 <= high:          # spans the origin: draw down to the floor
            lo = FLOOR
        lo, hi = max(lo, FLOOR), max(hi, FLOOR * 1.02)
        ax.plot([lo, hi], [y, y], color=colour, lw=1.5,
                solid_capstyle="butt", zorder=2)
        if np.isfinite(row["mde"]):
            ax.plot([row["mde"]], [y], marker="|", ms=6.0, color="0.35",
                    markeredgewidth=1.1, zorder=3)
        ax.plot([mag], [y], marker="o", ms=3.8, zorder=4,
                color="white" if negative else colour,
                markeredgecolor=colour, markeredgewidth=1.15)
        ticks.append(y)
        labels.append(SHORT[row["label"]])

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=FS_LABEL)
    ax.set_ylim(-PAD_BOT_MM, top + PAD_TOP_MM)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR, XMAX)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1"], fontsize=FS_TICK)
    ax.set_xlabel("effect on station-median RMSE at 7 d (°C, log scale)",
                  fontsize=FS_AXIS, labelpad=1.6)
    ax.grid(axis="x", which="major", color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=TICKPAD_MM / 25.4 * 72.0)
    ax.tick_params(axis="x", pad=1.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # Group names: x fixed in figure fractions (independent of the axes rect),
    # y in data millimetres so each name sits on the centre of its own block.
    fig_x_data_y = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    group_texts = []
    for group, lo, hi in blocks:
        colour = GROUP_COLOUR[group]
        t = ax.text(0.0, 0.5 * (lo + hi), GROUP_TITLE[group], transform=fig_x_data_y,
                    ha="right", va="center", fontsize=FS_GROUP, fontweight="bold",
                    color=_darken(colour), linespacing=1.05, clip_on=False)
        group_texts.append(t)
        ax.plot([0.0, 0.0], [lo - 0.45 * PITCH_MM, hi + 0.45 * PITCH_MM],
                transform=fig_x_data_y, color=colour, lw=1.0,
                solid_capstyle="butt", clip_on=False, zorder=1)

    handles = [
        Line2D([], [], marker="o", ls="none", color="0.35", ms=3.8,
               label="filled: positive (withholding it costs skill)"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec="0.35", ms=3.8,
               markeredgewidth=1.15, label="open: negative, magnitude plotted"),
        Line2D([], [], marker="|", ls="none", color="0.35", ms=6.0,
               markeredgewidth=1.1, label="minimum detectable effect"),
    ]
    # borderpad is 0.4 font-units by default, which reserves ~1.7 mm of blank
    # frame around a frameless legend -- pure loss at this height.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=False, columnspacing=1.3, handletextpad=0.35,
               handlelength=1.1, fontsize=FS_LEGEND, borderaxespad=0.0,
               borderpad=0.1)

    # ---- measure, then set the left margin to exactly what the text needs
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_per_mm = fig.dpi / 25.4

    label_w = max(t.get_window_extent(renderer=rend).width
                  for t in ax.get_yticklabels()) / px_per_mm
    group_w = max(t.get_window_extent(renderer=rend).width
                  for t in group_texts) / px_per_mm

    group_right_mm = M_LEFT_MM + group_w
    left_mm = group_right_mm + GUTTER_MM + label_w + TICKPAD_MM
    ax.set_position([left_mm / W_MM, M_BOTTOM_MM / H_MM,
                     (W_MM - M_RIGHT_MM - left_mm) / W_MM, axes_h_mm / H_MM])
    for t in group_texts:
        t.set_x(group_right_mm / W_MM)
    for line in ax.lines:
        if line.get_transform() is fig_x_data_y:
            line.set_xdata([(group_right_mm + 0.5 * GUTTER_MM) / W_MM] * 2)

    report = {
        "label_col_mm": label_w,
        "group_col_mm": group_w,
        "axes_left_mm": left_mm,
        "data_col_mm": W_MM - M_RIGHT_MM - left_mm,
        "row_pitch_mm": PITCH_MM,
    }
    return fig, report


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Group the rows and check every label has a short form.

    `collect()` appends the anchor-variant row last even though its group is
    "Study design", so file order is not group order and the layout must not
    trust it.  The SHORT check is a hard failure rather than a fallback: a new
    row silently keeping its long label would push the label column wider and
    quietly undo the height this layout exists for.
    """
    missing = sorted(set(frame["label"]) - set(SHORT))
    if missing:
        raise SystemExit(f"labels with no short form: {missing}")
    unknown = sorted(set(frame["group"]) - set(GROUP_ORDER))
    if unknown:
        raise SystemExit(f"rows in an unplaced group: {unknown}")
    frame = frame.copy()
    frame["_g"] = frame["group"].map(GROUP_ORDER.index)
    return frame.sort_values("_g", kind="stable").reset_index(drop=True)


def main() -> int:
    frame = collect()
    fig, report = render(_ordered(frame))

    # figstyle's strict gate does not inspect figure-level legends, so the
    # legend is checked against the canvas here rather than assumed to fit.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    for leg in fig.legends:
        b = leg.get_window_extent(renderer=rend)
        if b.x0 < -0.5 or b.x1 > w + 0.5 or b.y0 < -0.5 or b.y1 > h + 0.5:
            raise SystemExit(f"legend runs off the canvas: {b} vs {(w, h)}")

    # This figure places its one axes by measured millimetres, so constrained
    # layout must stay off.  Turning it off on the Figure is not enough: the rc
    # default is on, and matplotlib re-attaches an engine to the figure between
    # the first and second savefig, so the PDF was being laid out by an engine
    # the PNG never saw -- which is exactly the "no gridspecs with layoutgrids"
    # warning.  Disable it for the duration of the write instead.
    with matplotlib.rc_context({"figure.constrained_layout.use": False}):
        figstyle.save(fig, "fig05_information_axes", OUT)
    frame.to_csv(OUT / "fig05_information_axes_data.csv", index=False)
    print(f"figure: {fig.get_size_inches()[0]*25.4:.1f} x "
          f"{fig.get_size_inches()[1]*25.4:.1f} mm")
    for k, v in report.items():
        print(f"  {k:>16s}: {v:.2f}" if isinstance(v, float) else f"  {k:>16s}: {v}")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
