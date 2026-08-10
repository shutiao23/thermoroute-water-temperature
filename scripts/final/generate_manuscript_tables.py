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
    contrasts = [
        ("ThermoRoute", "DampedPersistence", 1, "1 d"),
        ("ThermoRoute", "DampedPersistence", 3, "3 d"),
        ("ThermoRoute", "DampedPersistence", 7, "7 d"),
        ("ThermoRoute", "LightGBM", 3, "3 d"),
        ("ThermoRoute", "LightGBM", 7, "7 d"),
    ]
    import json
    inference = json.loads(
        (FINAL / "cluster_inference_2021_2023.json").read_text(encoding="utf-8"))
    rows = [["#", "Comparison", "Lead", "ΔRMSE (°C)", "CI low", "CI high",
             "Win rate", "Stations", "p (sign flip)", "Holm p"]]
    for i, (cand, ref, h, lead) in enumerate(contrasts, start=1):
        g = effects[(effects.candidate == cand) & (effects.reference == ref)
                    & (effects.horizon == h)]
        record = inference.get(f"{cand}|{ref}|{h}", {})
        p_flip = record.get("p_cluster_sign_flip", np.nan)
        holm = record.get("holm_p", np.nan)
        rows.append([str(i), f"{cand} vs. {ref}", lead,
                     f3(g.delta_rmse.median()),
                     f3(record.get("ci_low", np.nan)),
                     f3(record.get("ci_high", np.nan)),
                     f"{np.mean(g.delta_rmse < 0):.2f}", str(len(g)),
                     f"{p_flip:.1e}" if np.isfinite(float(p_flip)) else "—",
                     f"{holm:.1e}" if np.isfinite(float(holm)) else "—"])
    return md_table(rows, ["r", "l", "l", "r", "r", "r", "r", "r", "r", "r"])


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
    models = ["Persistence", "DampedPersistence", "Climatology", "LightGBM",
              "LSTM", "PlainMLP-7var", "PlainCausalTCN-7var", "Air2stream",
              "ThermoRoute"]
    header = ["Model"] + [f"RMSE {h}" for h in (1, 3, 7)] \
        + [f"MAE {h}" for h in (1, 3, 7)] + [f"bias {h}" for h in (1, 3, 7)]
    rows = [header] + accuracy_rows(station, models)
    return md_table(rows, ["l"] + ["r"] * 9)


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
