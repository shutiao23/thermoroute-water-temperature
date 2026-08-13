#!/usr/bin/env python3
r"""Figure S10: the one-factor architecture controls on the held-out window.

SI07 and SI09 both promise this figure -- "the graphical reading of the control
rows" -- and it has never existed. SI09 explains why: it is "POST-gated and
blocked on the test-window evaluation receipt, so no coordinate in it is
available yet". That was true when it was written and stopped being true when
Section 4.4 ran. The blocker outlived the block, which is the same stale-status
defect the manuscript's Limitations and Conclusions carried, so the figure is
rendered rather than the promise deleted.

What it shows: each one-factor deletion or intervention against damped
persistence, on the exact common held-out keys, as a station-first paired
effect. The full model is drawn as a reference line rather than a bar, because
the question these controls answer is not "is the full model good" but "does
removing this component move it", and a bar-versus-bar reading invites the first
question.

Ordering is by effect rather than alphabetical or by the order the components
appear in the architecture: the interesting content is that one deletion --
the temporal encoder -- separates from the rest, and an ordering that buries it
between `noRouter` and `unbounded` hides the only finding in the panel.

These are deletion and intervention sensitivities on paired keys. They do not
establish that any component is necessary, and the caption says so.
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

OUT = ROOT / "paper" / "agu_submission" / "figures"
EFFECTS = ROOT / "outputs" / "final" / "paired_effects.parquet"
REFERENCE = "DampedPersistence"
FULL = "ThermoRoute"

LABEL = {
    "ThermoRoute": "full model",
    "TR-noTCN": "no temporal encoder",
    "TR-noRouter": "no router",
    "TR-noMoE": "no expert mixture",
    "TR-noDynamicPrior": "no dynamic prior",
    "TR-fixedKappa": "fixed κ (not learned)",
    "TR-unbounded": "correction unbounded",
}


class FigureDataError(RuntimeError):
    """The effects table is missing or does not carry the control arms."""


def collect() -> tuple[pd.DataFrame, dict[int, float]]:
    if not EFFECTS.exists():
        raise FigureDataError(f"missing {EFFECTS}")
    table = pd.read_parquet(EFFECTS)
    rows = table[table["reference"].eq(REFERENCE)
                 & table["candidate"].isin(LABEL)]
    if rows.empty:
        raise FigureDataError("no control arms in the effects table")
    grouped = (rows.groupby(["candidate", "horizon"])["delta_rmse"]
               .agg(["median", "size"]).reset_index())
    if grouped["size"].min() < 100:
        raise FigureDataError("a control cell has fewer than 100 stations")
    full = {int(h): float(v) for h, v in
            grouped[grouped["candidate"].eq(FULL)]
            .set_index("horizon")["median"].items()}
    return grouped[~grouped["candidate"].eq(FULL)], full


def render(grouped: pd.DataFrame, full: dict[int, float]):
    figstyle.use()
    leads = sorted(full)
    fig, axes = plt.subplots(
        1, len(leads), sharey=True,
        figsize=figstyle.figsize(height_mm=78.0), constrained_layout=True,
    )
    # order every panel by the seven-day effect so a component keeps its row
    order = (grouped[grouped["horizon"].eq(leads[-1])]
             .sort_values("median")["candidate"].tolist())
    colour = figstyle.SERIES["ThermoRoute"]

    for ax, lead in zip(np.atleast_1d(axes), leads):
        block = grouped[grouped["horizon"].eq(lead)].set_index("candidate")
        values = [block.loc[c, "median"] for c in order]
        ax.barh(range(len(order)), values, height=0.62, color=colour)
        ax.axvline(full[lead], color="0.25", lw=1.1, ls="--", zorder=3)
        ax.set_title(f"{lead} d")
        ax.set_xlabel("ΔRMSE vs damped (°C)")
        # Three ticks per panel, not the default five: with three narrow shared-y
        # panels the default ticks run one panel's "0.00" into the next panel's
        # "-0.15", which the collision gate rejects and a reader misreads.
        ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune="upper"))
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(order)))
    first.set_yticklabels([LABEL[c] for c in order])
    first.set_ylim(-0.7, len(order) - 0.3)
    first.text(full[leads[0]], len(order) - 0.45, " full model",
               color="0.25", ha="left", va="center",
               fontsize=plt.rcParams["legend.fontsize"])
    return fig


def main() -> int:
    grouped, full = collect()
    fig = render(grouped, full)
    figstyle.save(fig, "figS10_architecture_controls", OUT)
    print(grouped.pivot(index="candidate", columns="horizon",
                        values="median").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
