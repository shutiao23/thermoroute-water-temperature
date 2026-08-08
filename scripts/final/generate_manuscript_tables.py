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
             "Win rate", "Stations"]]
    for i, (cand, ref, h, lead) in enumerate(contrasts, start=1):
        g = effects[(effects.candidate == cand) & (effects.reference == ref)
                    & (effects.horizon == h)]
        record = inference.get(f"{cand}|{ref}|{h}", {})
        rows.append([str(i), f"{cand} vs. {ref}", lead,
                     f3(g.delta_rmse.median()),
                     f3(record.get("ci_low", np.nan)),
                     f3(record.get("ci_high", np.nan)),
                     f"{np.mean(g.delta_rmse < 0):.2f}", str(len(g))])
    return md_table(rows, ["r", "l", "l", "r", "r", "r", "r", "r"])


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


def table_4_7b(effects: pd.DataFrame) -> str:
    models = ["Persistence", "DampedPersistence", "Climatology", "LightGBM",
              "LSTM", "PlainMLP-7var", "PlainCausalTCN-7var", "Air2stream",
              "ThermoRoute"]
    header = ["Model"] + [f"persist. {h}" for h in (1, 3, 7)] \
        + [f"damped {h}" for h in (1, 3, 7)]
    rows = [header] + skill_rows(effects, models)
    return md_table(rows, ["l"] + ["r"] * 6)


def table_4_11(decomp: pd.DataFrame, station: pd.DataFrame) -> str:
    rows = [["Lead", "RMSE, persistence", "RMSE, damped", "RMSE, ThermoRoute",
             "Median memory gain", "Median learned gain"]]
    for h in (1, 3, 7):
        d = decomp[decomp.horizon == h]
        rows.append([
            f"{h} d",
            f3(station[(station.model == "Persistence") & (station.horizon == h)].rmse.median()),
            f3(station[(station.model == "DampedPersistence") & (station.horizon == h)].rmse.median()),
            f3(station[(station.model == "ThermoRoute") & (station.horizon == h)].rmse.median()),
            f3(d.g_memory.median()), f3(d.g_learned.median()),
        ])
    return md_table(rows, ["l"] + ["r"] * 5)


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
    all_rows.append(["All keys", f3(state[state.horizon == horizon].delta_rmse.median()),
                     "116", "1,023"])
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


def table_4_8a(station: pd.DataFrame) -> str:
    models = ["TR-fixedKappa", "TR-noDynamicPrior", "TR-noMoE", "TR-noRouter",
              "TR-noTCN", "TR-unbounded"]
    header = ["Model"] + [f"RMSE {h}" for h in (1, 3, 7)] \
        + [f"MAE {h}" for h in (1, 3, 7)] + [f"bias {h}" for h in (1, 3, 7)]
    rows = [header] + accuracy_rows(station, models)
    return md_table(rows, ["l"] + ["r"] * 9)


def table_4_8b(effects: pd.DataFrame) -> str:
    models = ["TR-fixedKappa", "TR-noDynamicPrior", "TR-noMoE", "TR-noRouter",
              "TR-noTCN", "TR-unbounded"]
    header = ["Model"] + [f"persist. {h}" for h in (1, 3, 7)] \
        + [f"damped {h}" for h in (1, 3, 7)]
    rows = [header] + skill_rows(effects, models)
    return md_table(rows, ["l"] + ["r"] * 6)


def main() -> int:
    station = pd.read_parquet(FINAL / "station_metrics.parquet")
    effects = pd.read_parquet(FINAL / "paired_effects.parquet")
    decomp = pd.read_parquet(FINAL / "decomposition_effects.parquet")
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
        "### Table 4.7b — Skill against persistence and damped persistence",
        "",
        table_4_7b(effects),
        "",
        "### Table 4.8a — Accuracy (RMSE, MAE, bias) — one-factor ablations",
        "",
        table_4_8a(station),
        "",
        "### Table 4.8b — Skill — one-factor ablations",
        "",
        table_4_8b(effects),
        "",
        "### Table 4.11 — Station-level error-budget decomposition",
        "",
        table_4_11(decomp, station),
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
