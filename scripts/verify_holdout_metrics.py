#!/usr/bin/env python3
"""A1 sanity verification for the conventional 2021-2023 holdout metrics table.

Reads the long-format metrics CSV written by
``scripts/conventional_holdout_2021_2023.py`` (columns ``model, horizon,
metric, value, n``; produced by ``thermoroute.conventional_score.compute_metrics_long``)
and runs the A1 sanity checks before those numbers are filled into the paper.
The script never runs (or re-runs) the scorer; it is a one-command check that
can be invoked as soon as the CSV has appeared.

Checks
------
(a) completeness -- every expected ``model x horizon x metric`` cell exists.
    Expected models mirror the scorer registry
    (``scripts/conventional_holdout_2021_2023.py``: ``TEMPORAL_BUNDLES`` keys
    and ``LSTM_BUNDLE``, plus the three ``baseline_frames`` outputs of
    ``src/thermoroute/conventional_score.py``: Persistence, DampedPersistence,
    Climatology).  Horizons are (1, 3, 7) plus the ``pooled`` row the scorer
    always writes.
(b) plausibility -- RMSE inside a parameterisable band (default 0.5-6 deg C),
    MAE in (0, RMSE], bounded BIAS, positive skill vs persistence, integer n
    that is positive and non-increasing with horizon.
(c) no unexpected NaN -- there are no legitimately-NaN cells in this schema:
    metric rows are only emitted when the underlying forecast keys exist.
(d) baselines -- Persistence / DampedPersistence / Climatology are present and
    inside the RMSE band, and ThermoRoute beats persistence (and damped
    persistence) at every horizon.

Metric names are the code's actual names (``SKILL_PERSISTENCE``,
``SKILL_CLIMATOLOGY``, ``N_SKILL``), not the ``skill_vs_*`` placeholder labels
used in the paper text.

Exit codes: 0 = all checks PASS, 1 = at least one FAIL, 2 = input file
missing or unreadable.

Usage:  python3 scripts/verify_holdout_metrics.py [--csv PATH] [options]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / "outputs" / "conventional" / "holdout_metrics_2021_2023.csv"

# Expected models, mirrored from scripts/conventional_holdout_2021_2023.py:
# TEMPORAL_BUNDLES keys + LSTM_BUNDLE, plus the baselines emitted by
# thermoroute.conventional_score.baseline_frames.
MODELS_REQUIRED = [
    "Persistence", "DampedPersistence", "Climatology",
    "ThermoRoute", "LightGBM", "LSTM",
    "DampedPriorOnly",
    "TR-noDynamicPrior", "TR-fixedKappa", "TR-noRouter", "TR-noMoE",
    "TR-noTCN", "TR-unbounded",
]
# Scored only when the external pooled bundles exist (EXTERNAL_BUNDLES).
MODELS_OPTIONAL = ["ThermoRoute-ext", "LSTM-ext", "LightGBM-ext"]
EXTERNAL_SUFFIX = "-ext"

HORIZONS = (1, 3, 7)
POOLED = "pooled"
ALL_HORIZONS = (*HORIZONS, POOLED)

METRICS_BASE = ("RMSE", "MAE", "BIAS")
METRICS_SKILL = ("SKILL_PERSISTENCE", "SKILL_CLIMATOLOGY", "N_SKILL")
METRICS_ALL = METRICS_BASE + METRICS_SKILL
BASELINE_MODELS = ("Persistence", "DampedPersistence", "Climatology")

CSV_COLUMNS = ("model", "horizon", "metric", "value", "n")

# Hard sanity bounds for skill (dimensionless): it is 1 - RMSE_m/RMSE_b,
# so it can never exceed 1 and should never be hugely negative.
SKILL_MIN_HARD = -10.0


class Reporter:
    """Collect PASS / FAIL / WARN lines and count failures."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def ok(self, check: str, detail: str) -> None:
        self.lines.append(("PASS", check, detail))

    def bad(self, check: str, detail: str) -> None:
        self.lines.append(("FAIL", check, detail))

    def warn(self, check: str, detail: str) -> None:
        self.lines.append(("WARN", check, detail))

    def n_fail(self) -> int:
        return sum(1 for status, _, _ in self.lines if status == "FAIL")

    def print(self) -> None:
        for status, check, detail in self.lines:
            print(f"[{status:4s}] {check}: {detail}")
        n_pass = sum(1 for s, _, _ in self.lines if s == "PASS")
        n_warn = sum(1 for s, _, _ in self.lines if s == "WARN")
        n_fail = self.n_fail()
        verdict = "ALL PASS" if n_fail == 0 else f"{n_fail} FAIL"
        print(f"overall: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL -> {verdict}")


def norm_horizon(value: object) -> int | str:
    """CSV stores 1/3/7 as ints and 'pooled' as text; normalise the dtype."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value).strip()


def _val(table: pd.DataFrame, model: str, horizon: object, metric: str) -> float | None:
    try:
        v = table.loc[(model, horizon), metric]
    except KeyError:
        return None
    return None if pd.isna(v) else float(v)


def _ntab(df: pd.DataFrame) -> pd.Series:
    return (
        df[df["metric"] == "RMSE"]
        .drop_duplicates(["model", "horizon"])
        .set_index(["model", "horizon"])["n"]
        .astype(float)
    )


def check_columns(df: pd.DataFrame, rep: Reporter) -> None:
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        rep.bad("schema.columns", f"missing columns {missing}; got {sorted(df.columns)}")
    else:
        rep.ok("schema.columns", f"columns {list(CSV_COLUMNS)}")
    extra = sorted(set(df.columns) - set(CSV_COLUMNS))
    if extra:
        rep.warn("schema.columns", f"extra columns {extra} (ignored)")


def check_duplicates(df: pd.DataFrame, rep: Reporter) -> None:
    n = int(df.duplicated(subset=["model", "horizon", "metric"], keep=False).sum())
    if n:
        rep.bad("schema.duplicates", f"{n} duplicated (model, horizon, metric) rows")
    else:
        rep.ok("schema.duplicates", "no duplicated (model, horizon, metric) rows")


def check_models(df: pd.DataFrame, required: list[str], optional: list[str],
                 strict_unknown: bool, rep: Reporter) -> None:
    present = set(df["model"].unique())
    for m in required:
        if m in present:
            rep.ok("completeness.models", f"{m!r} present")
        else:
            rep.bad("completeness.models", f"missing expected model {m!r}")
    for m in optional:
        if m not in present:
            rep.warn("completeness.models",
                     f"optional external model {m!r} absent (external bundle may be missing)")
    unknown = sorted(present - set(required) - set(optional))
    for m in unknown:
        if strict_unknown:
            rep.bad("completeness.models", f"model {m!r} not in the expected registry")
        else:
            rep.warn("completeness.models", f"model {m!r} outside the --models scope (ignored)")


def check_horizons(df: pd.DataFrame, rep: Reporter) -> None:
    horizons = set(df["horizon"].unique())
    missing = set(ALL_HORIZONS) - horizons
    extra = horizons - set(ALL_HORIZONS)
    if missing or extra:
        if missing:
            rep.bad("completeness.horizons", f"missing horizons {sorted(missing, key=str)}")
        if extra:
            rep.bad("completeness.horizons", f"unexpected horizons {sorted(extra, key=str)}")
    else:
        rep.ok("completeness.horizons", f"horizons {sorted(horizons, key=str)} complete")


def check_cells(df: pd.DataFrame, rep: Reporter) -> None:
    groups = df.groupby(["model", "horizon"])["metric"].agg(set)
    for (model, horizon), metrics in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        expected = set(METRICS_BASE) if model in BASELINE_MODELS else set(METRICS_BASE + METRICS_SKILL)
        missing = expected - metrics
        unexpected = (metrics & set(METRICS_ALL)) - expected
        unknown = metrics - set(METRICS_ALL)
        label = f"cells.{model}@{horizon}"
        if missing or unexpected or unknown:
            if missing:
                rep.bad(label, f"missing metrics {sorted(missing)}")
            if unexpected:
                rep.bad(label, f"metrics present but not expected here {sorted(unexpected)}")
            if unknown:
                rep.bad(label, f"metric names outside the schema {sorted(unknown)}")
        else:
            rep.ok(label, f"{len(metrics)} metrics present")


def check_nan(df: pd.DataFrame, rep: Reporter) -> None:
    bad = df[df["value"].isna()]
    if len(bad):
        examples = ", ".join(f"{r.model}@{r.horizon}:{r.metric}"
                             for r in bad.head(10).itertuples())
        rep.bad("nan.value", f"{len(bad)} NaN value cells (e.g. {examples})")
    else:
        rep.ok("nan.value", "no NaN in value")
    badn = df[df["n"].isna()]
    if len(badn):
        examples = ", ".join(f"{r.model}@{r.horizon}" for r in badn.head(10).itertuples())
        rep.bad("nan.n", f"{len(badn)} NaN n cells (e.g. {examples})")
    else:
        rep.ok("nan.n", "no NaN in n")


def check_plausibility(df: pd.DataFrame, table: pd.DataFrame, ntab: pd.Series,
                       args: argparse.Namespace, rep: Reporter) -> None:
    for r in df.itertuples(index=False):
        label = f"plausibility.{r.metric.lower()}.{r.model}@{r.horizon}"
        if r.metric == "RMSE":
            if r.value < args.rmse_min or r.value > args.rmse_max:
                rep.bad(label, f"RMSE {r.value:.3f} C outside [{args.rmse_min}, {args.rmse_max}]")
        elif r.metric == "MAE":
            rmse = _val(table, r.model, r.horizon, "RMSE")
            if r.value <= 0:
                rep.bad(label, f"MAE {r.value:.3f} <= 0")
            elif rmse is not None and r.value > rmse + 1e-9:
                rep.bad(label, f"MAE {r.value:.3f} > RMSE {rmse:.3f}")
        elif r.metric == "BIAS":
            if abs(r.value) > args.bias_max:
                rep.bad(label, f"|BIAS| {abs(r.value):.3f} C > --bias-max {args.bias_max}")
        elif r.metric in METRICS_SKILL:
            if r.metric == "N_SKILL":
                n = ntab.get((r.model, r.horizon))
                if r.value < 1:
                    rep.bad(label, f"N_SKILL {r.value:.0f} < 1")
                elif n is not None and r.value > n + 1e-9:
                    rep.bad(label, f"N_SKILL {r.value:.0f} > n {n:.0f}")
            elif r.value < SKILL_MIN_HARD or r.value > 1.0 + 1e-6:
                rep.bad(label, f"{r.metric} {r.value:.3f} outside [{SKILL_MIN_HARD:.0f}, 1]")
            elif r.metric == "SKILL_PERSISTENCE" and r.model not in BASELINE_MODELS:
                if r.value <= args.min_skill:
                    rep.bad(label, f"skill vs persistence {r.value:.3f} <= --min-skill {args.min_skill}")
            elif r.metric == "SKILL_CLIMATOLOGY" and r.model not in BASELINE_MODELS:
                if r.value < 0:
                    rep.warn(label, f"skill vs climatology {r.value:.3f} < 0 (non-fatal)")


def check_n(df: pd.DataFrame, ntab: pd.Series, args: argparse.Namespace, rep: Reporter) -> None:
    models = sorted(set(df["model"].unique()))
    for model in models:
        ns = {h: ntab.get((model, h)) for h in ALL_HORIZONS}
        for h, v in ns.items():
            if v is None:
                continue  # missing cell already reported under completeness
            if math.isnan(v) or not math.isclose(v, round(v), abs_tol=1e-6):
                rep.bad(f"n.{model}@{h}", f"n {v} is not a positive integer")
            elif v < 1:
                rep.bad(f"n.{model}@{h}", f"n {v} < 1")
            if h != POOLED and v > args.max_n:
                rep.bad(f"n.{model}@{h}", f"n {v:.0f} > --max-n {args.max_n} (possible duplicated keys)")
            if h != POOLED and v < args.min_n:
                rep.warn(f"n.{model}@{h}", f"n {v:.0f} < --min-n {args.min_n} (possibly sparse data)")
        vals = [ns.get(h) for h in HORIZONS]
        if all(v is not None for v in vals):
            if vals[0] < vals[1] or vals[1] < vals[2]:
                rep.bad(f"n.{model}.monotonic",
                        f"n {dict(zip(HORIZONS, (v if v is None else int(v) for v in vals)))} "
                        "not non-increasing with horizon")
            else:
                rep.ok(f"n.{model}.monotonic",
                       f"n non-increasing: h1={vals[0]:.0f} >= h3={vals[1]:.0f} >= h7={vals[2]:.0f}")
            pooled = ns.get(POOLED)
            if pooled is not None:
                total = sum(int(round(v)) for v in vals)
                if not math.isclose(pooled, total, abs_tol=1e-6):
                    rep.bad(f"n.{model}.pooled",
                            f"pooled n {pooled:.0f} != sum of horizon n {total}")
                else:
                    rep.ok(f"n.{model}.pooled", f"pooled n {pooled:.0f} == sum of horizon n {total}")
    for h in HORIZONS:
        for cohort, members in _cohorts(models).items():
            vals = {m: ntab.get((m, h)) for m in members if ntab.get((m, h)) is not None}
            if len(vals) < 2:
                continue
            distinct = set(round(v) for v in vals.values())
            if len(distinct) > 1:
                detail = ", ".join(f"{m}={v:.0f}" for m, v in vals.items())
                if args.strict_cross_n:
                    rep.bad(f"n.cross.{cohort}@{h}", f"n differs across models: {detail}")
                else:
                    rep.warn(f"n.cross.{cohort}@{h}", f"n differs across models: {detail}")
            else:
                rep.ok(f"n.cross.{cohort}@{h}",
                       f"n identical ({int(list(vals.values())[0])}) for {len(vals)} models")


def _cohorts(models: list[str]) -> dict[str, list[str]]:
    temporal = [m for m in models if not m.endswith(EXTERNAL_SUFFIX)]
    external = [m for m in models if m.endswith(EXTERNAL_SUFFIX)]
    out = {"temporal": temporal}
    if external:
        out["external"] = external
    return out


def check_baselines(table: pd.DataFrame, rep: Reporter) -> None:
    for h in HORIZONS:
        tr = _val(table, "ThermoRoute", h, "RMSE")
        pers = _val(table, "Persistence", h, "RMSE")
        damp = _val(table, "DampedPersistence", h, "RMSE")
        if tr is not None and pers is not None:
            if tr >= pers:
                rep.bad(f"baselines.tr_vs_persist@{h}",
                        f"ThermoRoute RMSE {tr:.3f} >= persistence {pers:.3f} C")
            else:
                rep.ok(f"baselines.tr_vs_persist@{h}",
                       f"ThermoRoute {tr:.3f} < persistence {pers:.3f} C")
        if tr is not None and damp is not None:
            if tr >= damp:
                rep.bad(f"baselines.tr_vs_damped@{h}",
                        f"ThermoRoute RMSE {tr:.3f} >= damped persistence {damp:.3f} C")
            else:
                rep.ok(f"baselines.tr_vs_damped@{h}",
                       f"ThermoRoute {tr:.3f} < damped persistence {damp:.3f} C")
    for name in BASELINE_MODELS:
        vals = {h: _val(table, name, h, "RMSE") for h in HORIZONS}
        detail = ", ".join(f"h{h}={v:.2f}" for h, v in vals.items() if v is not None)
        if detail:
            rep.ok(f"baselines.{name}", f"RMSE (C): {detail}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV),
                   help=f"long metrics CSV (default {DEFAULT_CSV})")
    p.add_argument("--rmse-min", type=float, default=0.5,
                   help="lower RMSE bound in deg C (default 0.5)")
    p.add_argument("--rmse-max", type=float, default=6.0,
                   help="upper RMSE bound in deg C (default 6.0)")
    p.add_argument("--bias-max", type=float, default=2.0,
                   help="max |BIAS| in deg C (default 2.0)")
    p.add_argument("--min-skill", type=float, default=0.0,
                   help="min skill vs persistence for non-baseline models (default 0.0)")
    p.add_argument("--min-n", type=int, default=30000,
                   help="WARN when n falls below this (default 30000)")
    p.add_argument("--max-n", type=int, default=200000,
                   help="FAIL when n exceeds this; signals duplicated keys (default 200000)")
    p.add_argument("--models", nargs="*", default=None,
                   help="override the required model list (e.g. after --skip-ablations)")
    p.add_argument("--require-external", action="store_true",
                   help="demand the pooled external models (default: optional)")
    p.add_argument("--strict-cross-n", action="store_true",
                   help="upgrade cross-model n mismatch from WARN to FAIL")
    args = p.parse_args(argv)

    path = Path(args.csv)
    if not path.is_file():
        p.error(f"metrics CSV not found: {path} (has the scorer finished?)")
    try:
        df = pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:
        p.error(f"cannot read {path}: {exc}")

    df = df.copy()
    df["model"] = df["model"].astype(str).str.strip()
    df["horizon"] = df["horizon"].map(norm_horizon)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")

    required = list(args.models) if args.models else list(MODELS_REQUIRED)
    if args.require_external:
        required += MODELS_OPTIONAL
    optional = [] if args.require_external else list(MODELS_OPTIONAL)

    rep = Reporter()
    check_columns(df, rep)
    check_duplicates(df, rep)
    df = df.drop_duplicates(subset=["model", "horizon", "metric"], keep="first")
    check_models(df, required, optional, strict_unknown=args.models is None, rep=rep)
    check_horizons(df, rep)
    check_cells(df, rep)
    check_nan(df, rep)
    table = df.pivot_table(index=["model", "horizon"], columns="metric",
                           values="value", aggfunc="first")
    ntab = _ntab(df)
    check_plausibility(df, table, ntab, args, rep)
    check_n(df, ntab, args, rep)
    check_baselines(table, rep)

    rep.print()
    return 1 if rep.n_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
