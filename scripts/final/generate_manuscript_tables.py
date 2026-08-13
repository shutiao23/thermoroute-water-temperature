#!/usr/bin/env python3
"""Generate the manuscript's held-out result tables from outputs/final/.

Every table block in Section 4.6 and Section 4.8 of
``paper/ThermoRoute_paper.md`` must equal the output of this script
(``scripts/final/check_manuscript_consistency.py`` diffs them).  No number is
hand-edited: station-first medians, station counts, skills, paired effects,
decomposition, state effects and the spatial contrasts all come from the
authority tables.

Usage::

    python scripts/final/generate_manuscript_tables.py [--out paper/tables_final.md]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FINAL = ROOT / "outputs" / "final"


def md_table(rows: list[list[str]], align: list[str] | None = None) -> str:
    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join("---" if not align or a == "l" else
                               ("---:" if a == "r" else ":---:") for a in (align or ["l"] * len(rows[0]))) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def f3(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(float(value)) else "—"


def f2(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(float(value)) else "—"


def table_4_6(station: pd.DataFrame, effects: pd.DataFrame) -> str:
    """The five sealed tests, each at the margin it was sealed at.

    This used to test all five rows against zero, which silently converted the
    two H2 rows from the non-inferiority tests they were registered as into
    superiority tests they were not, and reported them as failures.  The margin
    is part of the hypothesis, so it belongs in the table: a row's p-value means
    nothing without the margin it was computed against.
    """
    sealed = pd.read_parquet(
        FINAL / "sealed_confirmatory_family_v1" / "sealed_confirmatory_family.parquet")
    # Nine columns, not thirteen.  A first version carried the test id, the full
    # candidate-vs-reference string, both interval bounds, the station count and
    # the unadjusted p as separate columns; at the AGU text width the columns
    # collided into each other and the table was unreadable in the PDF while
    # looking fine in Markdown.  The candidate is ThermoRoute on every row and
    # the station count is 116 on every row, so both belong in the caption; the
    # test ids and unadjusted p-values are in the artifact.
    rows = [["#", "Reference", "Lead", "Margin (°C)", "ΔRMSE (°C)", "95% CI",
             "Win rate", "Holm p", "Sealed decision"]]
    for i, r in enumerate(sealed.itertuples(), start=1):
        g = effects[(effects.candidate == r.candidate)
                    & (effects.reference == r.reference)
                    & (effects.horizon == r.horizon)]
        rows.append([
            str(i), r.reference, f"{r.horizon} d", f"{r.margin_c:+.2f}",
            f3(r.effect_c), f"[{f3(r.ci_low_c)}, {f3(r.ci_high_c)}]",
            f"{np.mean(g.delta_rmse < 0):.2f}",
            f"{r.holm_p:.1e}", r.decision,
        ])
    return md_table(rows, ["r", "l", "l", "r", "r", "l", "r", "r", "l"])


def accuracy_rows(station: pd.DataFrame, models: list[str]) -> list[list[str]]:
    rows = []
    for model in models:
        cells = [model]
        for h in (1, 3, 7):
            g = station[(station.model == model) & (station.horizon == h)]
            cells.append(f3(g.rmse.median()))
        for h in (1, 3, 7):
            g = station[(station.model == model) & (station.horizon == h)]
            cells.append(f3(g.mae.median()))
        for h in (1, 3, 7):
            g = station[(station.model == model) & (station.horizon == h)]
            cells.append(f3(g.bias.median()))
        rows.append(cells)
    return rows


def table_4_7a(station: pd.DataFrame) -> str:
    """Every model's accuracy AND what it reports against each reference.

    These were two tables. Merging them is not only a page saved: the paper's
    central claim is that a reported number depends on its denominator, and that
    claim is checkable in one glance only when the accuracy and the two skill
    columns sit on the same row. Reading down a column compares models; reading
    across a row shows what the reference alone does to one fitted object.

    MAE and bias are the same rows in SI07 and are not repeated.
    """
    matrix = pd.read_parquet(
        FINAL / "reference_matrix_v1" / "reference_matrix.parquet")
    at7 = matrix[matrix.horizon == 7].set_index(["model", "reference"])

    models = ["Persistence", "DampedPersistence", "Climatology", "LightGBM",
              "LSTM", "PlainMLP-7var", "PlainCausalTCN-7var", "Air2stream",
              "ThermoRoute"]

    def skill(model: str, reference: str) -> str:
        # A reference has no skill against itself, and climatology is not one of
        # the two denominators under test; an em dash says so rather than a zero.
        if (model, reference) not in at7.index:
            return "—"
        row = at7.loc[(model, reference)]
        return (f"{row.median_station_skill:+.3f} "
                f"[{row.skill_ci_low:+.3f}, {row.skill_ci_high:+.3f}]")

    header = ["Model"] + [f"RMSE {h} d" for h in (1, 3, 7)] \
        + ["Skill 7 d vs. persistence", "Skill 7 d vs. damped"]
    rows = [header]
    for cells in accuracy_rows(station, models):
        model = cells[0]
        rows.append(cells[:4] + [skill(model, "Persistence"),
                                 skill(model, "DampedPersistence")])
    return md_table(rows, ["l", "r", "r", "r", "l", "l"])


def skill_rows(effects: pd.DataFrame, models: list[str]) -> list[list[str]]:
    rows = []
    for model in models:
        cells = [model]
        for baseline in ("Persistence", "DampedPersistence"):
            for h in (1, 3, 7):
                g = effects[(effects.candidate == model)
                            & (effects.reference == baseline)
                            & (effects.horizon == h)]
                cells.append(f"+{f3(g.station_skill.median())}"
                             if not g.empty and g.station_skill.median() >= 0
                             else f3(g.station_skill.median()) if not g.empty else "—")
        rows.append(cells)
    return rows




def table_4_12(state: pd.DataFrame, *, horizon: int = 7) -> str:
    issue = state[(state.horizon == horizon) & (state.state_type == "issue_time")]
    outcome = state[(state.horizon == horizon)
                    & (state.state_type == "outcome_conditioned")]
    order_issue = ["issue_low_anomaly", "issue_high_anomaly",
                   "issue_rapid_recent_warming", "issue_rapid_recent_cooling",
                   "issue_strong_warming_pressure", "issue_strong_cooling_pressure",
                   "issue_low_flow", "issue_high_flow",
                   "issue_rapid_flow_rise", "issue_rapid_flow_recession"]
    order_outcome = ["actual_rapid_warming", "actual_rapid_cooling",
                     "actual_warmest_decile", "actual_coldest_decile",
                     "actual_high_flow", "actual_low_flow"]
    all_rows: list[list[str]] = []
    # The "All keys" row is the same estimand as Table 4.6 row 3 (paired
    # ThermoRoute minus DampedPersistence at 7 d, station median), taken from
    # the authority's paired effects; it must NOT be the median over the
    # (site x state) cells of the state table, which mixes states.  Station
    # and key counts are computed from the data, never hard-coded.
    effects = pd.read_parquet(FINAL / "paired_effects.parquet")
    all_eff = effects[(effects.candidate == "ThermoRoute")
                      & (effects.reference == "DampedPersistence")
                      & (effects.horizon == horizon)]
    keys = pd.read_parquet(FINAL / "forecast_keys.parquet")
    per_site_keys = keys[keys.horizon == horizon].groupby("site_id").size()
    all_rows.append(["All keys", f3(all_eff.delta_rmse.median()),
                     str(int(all_eff.delta_rmse.notna().sum())),
                     f"{int(per_site_keys.median()):,}"])
    for name in order_issue:
        g = issue[issue.state_name == name]
        if g.empty:
            continue
        all_rows.append([name.replace("_", " "), f3(g.delta_rmse.median()),
                         str(g.site_id.nunique()),
                         f"{int(g.n_keys.median()):,}"])
    for name in order_outcome:
        g = outcome[outcome.state_name == name]
        if g.empty:
            continue
        all_rows.append([name.replace("_", " "), f3(g.delta_rmse.median()),
                         str(g.site_id.nunique()),
                         f"{int(g.n_keys.median()):,}"])
    rows = [["State (7-day keys)", "ΔRMSE, ThermoRoute − damped (°C)",
             "stations", "median keys"]] + all_rows
    return md_table(rows, ["l", "r", "r", "r"])




def main() -> int:
    station = pd.read_parquet(FINAL / "station_metrics.parquet")
    effects = pd.read_parquet(FINAL / "paired_effects.parquet")
    state = pd.read_parquet(FINAL / "hydrologic_state_effects.parquet")

    out = [
        "<!-- AUTO-GENERATED by scripts/final/generate_manuscript_tables.py -->",
        "",
        "### Table 4.6 — paired comparisons on the held-out window",
        "",
        table_4_6(station, effects),
        "",
        "### Table 4.7a — Accuracy (RMSE, MAE, bias)",
        "",
        table_4_7a(station),
        "",
        "### Table 4.12 — Hydrologic states (7-day keys)",
        "",
        table_4_12(state),
        "",
    ]
    text = "\n".join(out)
    target = ROOT / "paper" / "tables_final.md"
    target.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
