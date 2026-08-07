#!/usr/bin/env python3
"""B1 LaTeX result-table builder for the held-out 2021-2023 evaluation.

Reads the long-format metrics CSV produced by
``scripts/conventional_holdout_2021_2023.py`` (columns ``model, horizon,
metric, value, n``; ``thermoroute.conventional_score.compute_metrics_long``)
and renders the paper's held-out result tables as a compilable LaTeX fragment
(booktabs three-line tables, SI units, fixed decimals) written to
``outputs/conventional/``.

Correspondence with the paper (``paper/agu_submission/ThermoRoute_WRR.tex``,
subsection ``Held-out 2021--2023 evaluation``, label
``held-out-20212023-evaluation``):

  * Table 4.7 -- primary models (Persistence, DampedPersistence, Climatology,
    LightGBM, LSTM, ThermoRoute); sidewaystable block at tex lines ~973--1003.
  * Table 4.8 -- one-factor ablations (tr-fixedkappa, tr-nodynamicprior,
    tr-nomoe, tr-norouter, tr-notcn, tr-unbounded); sidewaystable block at
    tex lines ~1005--1035.

Column order and labels replicate the paper tables exactly: Model, $h$ (d),
RMSE ($^\\circ$C), MAE ($^\\circ$C), bias ($^\\circ$C), skill vs.\\ persist,
skill vs.\\ clim, $n$.  NaN cells are rendered as the paper's ``\\ph{\\dots}``
placeholder (reported on stderr); baseline rows show ``--`` for the two skill
columns because the scorer never computes skill for baselines.

Optional content, off by default: pooled-over-horizon rows per model
(--include-pooled); the DampedPriorOnly ablation row appended to Table 4.8
(--include-damped-prior, it is not one of the six ablations the paper lists);
a complementary pooled-external-cohort table (--include-external,
ThermoRoute-ext / LSTM-ext / LightGBM-ext, scored only when their frozen
bundles exist).

Note on n: the CSV ``n`` column is the number of forecast keys
(site_id, issue_date, target_date) for that (model, horizon) cell, not a
station count; the paper text describing Tables 4.7--4.8 says "reportable
station count", which is an editorial mismatch to reconcile when filling the
tables.

Usage:  python3 scripts/build_results_table.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / "outputs" / "conventional" / "holdout_metrics_2021_2023.csv"
DEFAULT_OUT = REPO / "outputs" / "conventional" / "holdout_tables_2021_2023.tex"

# Model blocks of the paper's Tables 4.7 / 4.8 (paper uses lowercase tr-*
# names; the CSV uses the scorer registry names of
# scripts/conventional_holdout_2021_2023.py).
PRIMARY_MODELS = ["Persistence", "DampedPersistence", "Climatology",
                  "LightGBM", "LSTM", "ThermoRoute"]
ABLATION_MODELS = ["TR-fixedKappa", "TR-noDynamicPrior", "TR-noMoE",
                   "TR-noRouter", "TR-noTCN", "TR-unbounded"]
EXTERNAL_MODELS = ["ThermoRoute-ext", "LSTM-ext", "LightGBM-ext"]
BASELINES = {"Persistence", "DampedPersistence", "Climatology"}

HORIZONS = (1, 3, 7)
POOLED = "pooled"

HEADER_LINE = ("Model & $h$ (d) & RMSE ($^\\circ$C) & MAE ($^\\circ$C) & "
               "bias ($^\\circ$C) & skill vs.\\ persist & skill vs.\\ clim & $n$ \\\\")

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def norm_horizon(value: object) -> int | str:
    """CSV stores 1/3/7 as ints and 'pooled' as text; normalise the dtype."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value).strip()


def display_name(model: str) -> str:
    """Paper convention: ablations are lowercase 'tr-*' (Table 4.8)."""
    if model.startswith("TR-"):
        return "tr-" + model[3:].lower()
    return model


def esc(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(c, c) for c in text)


def _val(table: pd.DataFrame, model: str, horizon: object, metric: str) -> float | None:
    try:
        v = table.loc[(model, horizon), metric]
    except KeyError:
        return None
    return None if pd.isna(v) else float(v)


def _fmt(v: float | None, decimals: int, sign: bool = False,
         missing: str | None = None) -> str:
    if v is None:
        return missing if missing is not None else ""
    return f"{v:+.{decimals}f}" if sign else f"{v:.{decimals}f}"


def _skill_cell(table: pd.DataFrame, model: str, horizon: object, metric: str,
                args: argparse.Namespace) -> str:
    v = _val(table, model, horizon, metric)
    if v is not None:
        return _fmt(v, args.skill_decimals)
    # Baselines have no skill rows in the CSV: compute from RMSE vs the
    # reference baseline (Persistence for skill-vs-persist, Climatology for
    # skill-vs-clim). Persistence vs itself -> 0; Climatology vs itself -> 0.
    rmse_model = _val(table, model, horizon, "RMSE")
    ref = "Persistence" if metric == "SKILL_PERSISTENCE" else "Climatology"
    rmse_ref = _val(table, ref, horizon, "RMSE")
    if rmse_model is None or rmse_ref is None or rmse_ref == 0:
        return args.placeholder
    return _fmt(1 - rmse_model / rmse_ref, args.skill_decimals)


def _n_cell(ntab: pd.Series, model: str, horizon: object,
            args: argparse.Namespace) -> str:
    v = ntab.get((model, horizon))
    return f"{int(round(v))}" if v is not None and not pd.isna(v) else args.placeholder


def _metric_cells(table: pd.DataFrame, ntab: pd.Series, model: str, horizon: object,
                  args: argparse.Namespace) -> list[str]:
    return [
        esc(display_name(model)),
        str(horizon),
        _fmt(_val(table, model, horizon, "RMSE"), args.deg_decimals,
             missing=args.placeholder),
        _fmt(_val(table, model, horizon, "MAE"), args.deg_decimals,
             missing=args.placeholder),
        _fmt(_val(table, model, horizon, "BIAS"), args.bias_decimals, sign=True,
             missing=args.placeholder),
        _skill_cell(table, model, horizon, "SKILL_PERSISTENCE", args),
        _skill_cell(table, model, horizon, "SKILL_CLIMATOLOGY", args),
        _n_cell(ntab, model, horizon, args),
    ]


def _n_per_horizon(ntab: pd.Series, models: list[str]) -> dict:
    """n is horizon-only (identical across models); take from the first present model."""
    out: dict = {}
    for h in HORIZONS:
        for m in models:
            v = ntab.get((m, h))
            if v is not None and not pd.isna(v):
                out[h] = int(round(v))
                break
    return out


def _accuracy_cells(table: pd.DataFrame, model: str, horizon: object,
                    args: argparse.Namespace) -> list[str]:
    return [
        esc(display_name(model)),
        str(horizon),
        _fmt(_val(table, model, horizon, "RMSE"), args.deg_decimals,
             missing=args.placeholder),
        _fmt(_val(table, model, horizon, "MAE"), args.deg_decimals,
             missing=args.placeholder),
        _fmt(_val(table, model, horizon, "BIAS"), args.bias_decimals, sign=True,
             missing=args.placeholder),
    ]


def _skill_cells(table: pd.DataFrame, model: str, horizon: object,
                 args: argparse.Namespace) -> list[str]:
    return [
        esc(display_name(model)),
        str(horizon),
        _skill_cell(table, model, horizon, "SKILL_PERSISTENCE", args),
        _skill_cell(table, model, horizon, "SKILL_CLIMATOLOGY", args),
    ]


def render_table(title: str, models: list[str], table: pd.DataFrame,
                 ntab: pd.Series, args: argparse.Namespace) -> list[str]:
    import re
    m = re.match(r"(Table \d+\.\d+)\.\s*(.*)", title)
    prefix = m.group(1) if m else title
    desc = m.group(2).strip() if m else ""
    n_per_h = _n_per_horizon(ntab, models)
    n_note = " $n$ = " + ", ".join(
        f"{n_per_h[h]:,} ($h$={h}\\,d)" for h in HORIZONS if h in n_per_h
    ) + " forecast keys."
    acc_rows: list[str] = []
    sk_rows: list[str] = []
    for model in models:
        if model not in table.index.get_level_values("model"):
            continue
        for h in HORIZONS:
            acc_rows.append(" & ".join(_accuracy_cells(table, model, h, args)) + " \\\\")
            sk_rows.append(" & ".join(_skill_cells(table, model, h, args)) + " \\\\")
        if args.include_pooled:
            acc_rows.append(" & ".join(_accuracy_cells(table, model, POOLED, args)) + " \\\\")
            sk_rows.append(" & ".join(_skill_cells(table, model, POOLED, args)) + " \\\\")
    acc_title = f"{prefix}a. Accuracy (RMSE, MAE, bias)" + (f" --- {desc}" if desc else "")
    sk_title = f"{prefix}b. Skill vs persistence and climatology" + (f" --- {desc}" if desc else "")
    head = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
    ]
    out: list[str] = []
    out += head + [
        f"\\textbf{{{acc_title}}}",
        "\\par\\smallskip",
        "\\textit{" + n_note.strip() + "}",
        "\\par\\medskip",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule\\noalign{}",
        "Model & $h$ (d) & RMSE ($^\\circ$C) & MAE ($^\\circ$C) & bias ($^\\circ$C) \\\\",
        "\\midrule\\noalign{}",
        *acc_rows,
        "\\bottomrule\\noalign{}",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    out += head + [
        f"\\textbf{{{sk_title}}}",
        "\\par\\medskip",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule\\noalign{}",
        "Model & $h$ (d) & skill vs.\\ persist & skill vs.\\ clim \\\\",
        "\\midrule\\noalign{}",
        *sk_rows,
        "\\bottomrule\\noalign{}",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV),
                   help=f"long metrics CSV (default {DEFAULT_CSV})")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"output .tex fragment (default {DEFAULT_OUT})")
    p.add_argument("--deg-decimals", type=int, default=2,
                   help="decimal places for RMSE/MAE in deg C (default 2)")
    p.add_argument("--bias-decimals", type=int, default=2,
                   help="decimal places for bias in deg C (default 2)")
    p.add_argument("--skill-decimals", type=int, default=3,
                   help="decimal places for skill scores (default 3)")
    p.add_argument("--include-pooled", action="store_true",
                   help="append a pooled-over-horizons row per model")
    p.add_argument("--include-damped-prior", action="store_true",
                   help="append the DampedPriorOnly ablation row to Table 4.8")
    p.add_argument("--include-external", action="store_true",
                   help="also render the pooled external cohort table")
    p.add_argument("--placeholder", default=r"\ph{\dots}",
                   help="LaTeX placeholder for missing cells (default \\ph{\\dots})")
    args = p.parse_args(argv)

    path = Path(args.csv)
    if not path.is_file():
        p.error(f"metrics CSV not found: {path} (has the scorer finished?)")
    df = pd.read_csv(path, float_precision="round_trip")
    df["model"] = df["model"].astype(str).str.strip()
    df["horizon"] = df["horizon"].map(norm_horizon)

    table = df.pivot_table(index=["model", "horizon"], columns="metric",
                           values="value", aggfunc="first")
    ntab = (
        df[df["metric"] == "RMSE"]
        .drop_duplicates(["model", "horizon"])
        .set_index(["model", "horizon"])["n"]
        .astype(float)
    )
    present = set(df["model"].unique())

    ablation_models = list(ABLATION_MODELS)
    if args.include_damped_prior:
        ablation_models.append("DampedPriorOnly")

    lines = [
        "% ------------------------------------------------------------------",
        "% Held-out 2021-2023 result tables for paper/agu_submission/ThermoRoute_WRR.tex",
        f"% Generated by: scripts/build_results_table.py (source: {path.name})",
        "% Replaces the placeholder sidewaystable blocks of Tables 4.7 and 4.8 in the",
        "% 'Held-out 2021--2023 evaluation' subsection (label held-out-20212023-evaluation).",
        "% Units: RMSE/MAE/bias in deg C; skill = 1 - RMSE_model/RMSE_baseline on the",
        "% common forecast keys (dimensionless); n = forecast-key count, NOT station",
        "% count (reconcile with the paper text before final submission).",
        "% Skill metrics: SKILL_PERSISTENCE / SKILL_CLIMATOLOGY in the CSV; baselines",
        "% have no skill rows and show '--'.",
        "% ------------------------------------------------------------------",
        "",
    ]
    missing = []
    for title, models in [
        ("Table 4.7. Held-out 2021--2023 pooled metrics, primary models.",
         PRIMARY_MODELS),
        ("Table 4.8. Held-out 2021--2023 pooled metrics, one-factor ablations.",
         ablation_models),
    ]:
        if not any(m in present for m in models):
            missing.append(title.split(".")[0])
            continue
        lines += render_table(title, models, table, ntab, args)
        lines.append("")
    if args.include_external and any(m in present for m in EXTERNAL_MODELS):
        lines += render_table(
            "Held-out 2021--2023 metrics, pooled external cohort (complementary).",
            EXTERNAL_MODELS, table, ntab, args)
        lines.append("")
    if missing:
        lines.append("% not rendered (models absent from the CSV): " + ", ".join(missing) + "\n")

    for model in PRIMARY_MODELS + ablation_models + EXTERNAL_MODELS:
        if model not in present:
            print(f"[warn] model {model!r} not in {path.name}; skipped", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    n_ph = "\n".join(lines).count(r"\ph{")
    print(f"wrote {out_path} ({len(lines)} lines, {n_ph} placeholder cells)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
