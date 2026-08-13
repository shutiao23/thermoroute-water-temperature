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

import matplotlib.pyplot as plt  # noqa: E402
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


GROUP_COLOUR = {
    "Local observation": figstyle.SERIES["ThermoRoute"],
    "Future weather": figstyle.SERIES["LightGBM"],
    "Study design": figstyle.SERIES["LSTM"],
    "Estimator": figstyle.SERIES["Persistence"],
}
FLOOR = 1e-3        # left edge of the log axis, below every measured quantity


def render(frame: pd.DataFrame):
    figstyle.use()
    fig, ax = plt.subplots(
        figsize=figstyle.figsize(height_mm=104.0), constrained_layout=True
    )

    # Lay the rows out with a spacer above each group and the group name written
    # into that spacer.  A first attempt put rotated group names in the left
    # margin; the collision gate rejected it, because at this width the margin
    # belongs to the row labels and nothing else fits there.
    groups: list[str] = []
    for group in frame["group"]:
        if group not in groups:
            groups.append(group)

    positions: dict[int, float] = {}
    headers: list[tuple[float, str]] = []
    y = 0.0
    for group in reversed(groups):                 # first group ends up on top
        block = frame.index[frame["group"] == group]
        for index in reversed(list(block)):
            positions[index] = y
            y += 1.0
        headers.append((y - 0.5, group))
        y += 1.15

    ticks, labels = [], []
    for index, ypos in positions.items():
        row = frame.loc[index]
        colour = GROUP_COLOUR[row["group"]]
        value, low, high = row["value"], row["low"], row["high"]
        negative = value < 0
        # magnitudes on a log axis; the sign is carried by the marker, not lost
        mag = max(abs(value), FLOOR)
        lo, hi = sorted((abs(low), abs(high)))
        if low <= 0.0 <= high:      # interval spans the origin: draw to the floor
            lo = FLOOR
        lo, hi = max(lo, FLOOR), max(hi, FLOOR * 1.02)
        ax.plot([lo, hi], [ypos, ypos], color=colour, lw=1.7,
                solid_capstyle="butt", zorder=2)
        ax.plot([mag], [ypos], marker="o", ms=4.4, zorder=4,
                color="white" if negative else colour,
                markeredgecolor=colour, markeredgewidth=1.3)
        if np.isfinite(row["mde"]):
            ax.plot([row["mde"]], [ypos], marker="|", ms=8, color="0.35",
                    markeredgewidth=1.2, zorder=3)
        ticks.append(ypos)
        labels.append(row["label"])

    for ypos, group in headers:
        ax.text(FLOOR * 1.06, ypos, group, ha="left", va="center",
                color=GROUP_COLOUR[group], fontsize=plt.rcParams["axes.labelsize"])

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, y - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR, 6.0)
    ax.set_xlabel("effect on station-median RMSE at 7 d (°C, log scale)")
    ax.grid(axis="x", which="major", color="0.88", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="none", color="0.35", ms=4.4,
               label="positive: withholding it costs skill"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec="0.35", ms=4.4,
               markeredgewidth=1.3, label="negative: magnitude plotted"),
        Line2D([], [], marker="|", ls="none", color="0.35", ms=8,
               markeredgewidth=1.2, label="minimum detectable effect"),
    ]
    # Below the axes rather than inside them: at lower right the legend sat on
    # top of the estimator rows, which are the ones a reader most needs to see.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False, columnspacing=1.4,
              fontsize=plt.rcParams["legend.fontsize"], handletextpad=0.4,
              borderaxespad=0.0)
    return fig


def main() -> int:
    frame = collect()
    fig = render(frame)
    figstyle.save(fig, "fig05_information_axes", OUT)
    frame.to_csv(OUT / "fig05_information_axes_data.csv", index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
