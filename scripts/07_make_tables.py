#!/usr/bin/env python3
"""Stage 7 — paper tables (Markdown + CSV) with significance testing.

Reads the predictions and scores, writes the main tables to
``outputs/tables/*.md`` and ``*.csv``, including moving-block-bootstrap RMSE CIs
and Diebold-Mariano tests of ThermoRoute vs damped persistence.

Run:  PYTHONPATH=src python3 scripts/07_make_tables.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from thermoroute import config as C
from thermoroute import significance as S

PRED = pd.read_parquet(C.PREDICTIONS / "predictions.parquet")
SCORES = pd.read_csv(C.TABLES / "scores_all.csv")
TEST = SCORES[SCORES.split == "test"].copy()
OUT = []


def w(s=""):
    OUT.append(s)


def mean_over_stations(model_filter, value="RMSE"):
    s = TEST[model_filter].groupby(["model", "horizon"])[value].mean().unstack()
    return s


def seed_pred(model, scope=None, feature_set=None):
    m = PRED.model == model
    if scope:
        m &= PRED.scope == scope
    if feature_set:
        m &= PRED.feature_set == feature_set
    return PRED[m & (PRED.split == "test")]


def main():
    w("# ThermoRoute — result tables\n")
    w("_All metrics on the 2019–2020 blind test. Deep models: mean ± std over "
      "seeds. Stations b1/s2/p3 averaged unless noted._\n")

    # ---- Table 2: headline results -------------------------------------- #
    w("## Table 2 — Overall blind-test accuracy by horizon\n")
    w("RMSE (°C) and skill vs persistence, mean over the three stations.\n")
    order = ["Persistence", "Climatology", "DampedPersistence", "Air2streamLite",
             "Ridge", "LightGBM", "GRU", "ThermoRoute"]
    rmse = mean_over_stations(TEST.scope.isin(["per_station", "joint"]) &
                              (TEST.feature_set != "V1") & (TEST.feature_set != "V2"))
    skill = mean_over_stations(TEST.scope.isin(["per_station", "joint"]) &
                               (TEST.model != "Persistence") &
                               (TEST.feature_set != "V1") & (TEST.feature_set != "V2"),
                               "SKILL_RMSE")
    w("| model | RMSE h1 | RMSE h3 | RMSE h7 | skill h1 | skill h3 | skill h7 |")
    w("|---|---|---|---|---|---|---|")
    for m in [x for x in order if x in rmse.index]:
        r = rmse.loc[m]
        sk = skill.loc[m] if m in skill.index else pd.Series({1: np.nan, 3: np.nan, 7: np.nan})
        w(f"| {m} | {r.get(1):.3f} | {r.get(3):.3f} | {r.get(7):.3f} | "
          f"{sk.get(1):+.3f} | {sk.get(3):+.3f} | {sk.get(7):+.3f} |")
    w("")

    # ---- Table 2b: per-station ThermoRoute vs damped persistence + DM ---- #
    w("## Table 2b — ThermoRoute vs damped persistence, per station (+ significance)\n")
    w("ΔRMSE = RMSE(ThermoRoute) − RMSE(damped). Negative ⇒ ThermoRoute better. "
      "DM p<0.05 marked *.\n")
    w("| station | horizon | RMSE damped | RMSE ThermoRoute | ΔRMSE | DM p |")
    w("|---|---|---|---|---|---|")
    tr = seed_pred("ThermoRoute", scope="joint", feature_set="V3")
    dp = PRED[(PRED.model == "DampedPersistence") & (PRED.split == "test")]
    for st in C.STATIONS:
        for h in C.HORIZONS:
            a = tr[(tr.site_id == st) & (tr.horizon == h)].groupby("issue_date").y_pred.mean()
            b = dp[(dp.site_id == st) & (dp.horizon == h)].set_index("issue_date").y_pred
            y = dp[(dp.site_id == st) & (dp.horizon == h)].set_index("issue_date").y_true
            idx = a.index.intersection(b.index)
            ea = (a.reindex(idx) - y.reindex(idx)).to_numpy()
            eb = (b.reindex(idx) - y.reindex(idx)).to_numpy()
            rmse_a = np.sqrt(np.mean(ea ** 2)); rmse_b = np.sqrt(np.mean(eb ** 2))
            dm, p = S.diebold_mariano(ea, eb, h=h)
            star = "*" if (p is not None and p < 0.05) else ""
            w(f"| {st} | {h} | {rmse_b:.3f} | {rmse_a:.3f} | {rmse_a - rmse_b:+.3f} | "
              f"{p:.3f}{star} |")
    w("")

    # ---- Table 3: feature-set gains ------------------------------------- #
    w("## Table 3 — Variable-set gains (RMSE °C)\n")
    w("Adding hydrology + meteorology to the autoregressive baseline.\n")
    w("| model | set | h1 | h3 | h7 |")
    w("|---|---|---|---|---|")
    for model in ("LightGBM", "ThermoRoute"):
        for fs in ("V1", "V2", "V3"):
            sub = TEST[(TEST.model == model) & (TEST.feature_set == fs)]
            if sub.empty:
                continue
            r = sub.groupby("horizon").RMSE.mean()
            w(f"| {model} | {fs} | {r.get(1, np.nan):.3f} | {r.get(3, np.nan):.3f} "
              f"| {r.get(7, np.nan):.3f} |")
    w("")

    # ---- Table 4: probabilistic + exceedance ---------------------------- #
    w("## Table 4 — Probabilistic & high-temperature warning (blind test)\n")
    w("ThermoRoute (joint, conformal). PICP target = 0.90.\n")
    w("| station | horizon | PICP | MPIW (°C) | CRPS | Brier-skill | AUPRC |")
    w("|---|---|---|---|---|---|---|")
    # pin the SAME configuration as the headline point table (V3 joint)
    trs = TEST[(TEST.model == "ThermoRoute") & (TEST.scope == "joint")
               & (TEST.feature_set == "V3")]
    for st in C.STATIONS:
        for h in C.HORIZONS:
            row = trs[(trs.site_id == st) & (trs.horizon == h)]
            if row.empty:
                continue
            r = row.iloc[0]
            w(f"| {st} | {h} | {r.get('PICP', np.nan):.3f} | {r.get('MPIW', np.nan):.2f} "
              f"| {r.get('CRPS', np.nan):.3f} | {r.get('EVT_BRIER_SKILL', np.nan):+.3f} "
              f"| {r.get('EVT_AUPRC', np.nan):.3f} |")
    w("")

    # ---- Table 5: LOSO -------------------------------------------------- #
    w("## Table 5 — Leave-one-station-out spatial transfer (RMSE °C)\n")
    loso = TEST[TEST.model == "ThermoRoute-LOSO"]
    # pin the V3 joint headline config as the in-sample reference (LOSO is V3-only);
    # without this filter the joint column averages V1/V2/V3 and is not comparable.
    joint = TEST[(TEST.model == "ThermoRoute") & (TEST.scope == "joint")
                 & (TEST.feature_set == "V3")]
    if not loso.empty:
        w("| held-out station | h1 joint | h1 LOSO | h3 joint | h3 LOSO | h7 joint | h7 LOSO |")
        w("|---|---|---|---|---|---|---|")
        for st in C.STATIONS:
            jj = joint[joint.site_id == st].groupby("horizon").RMSE.mean()
            ll = loso[loso.site_id == st].groupby("horizon").RMSE.mean()
            w(f"| {st} | {jj.get(1, np.nan):.3f} | {ll.get(1, np.nan):.3f} "
              f"| {jj.get(3, np.nan):.3f} | {ll.get(3, np.nan):.3f} "
              f"| {jj.get(7, np.nan):.3f} | {ll.get(7, np.nan):.3f} |")
        w("")

    # ---- Table 6: ablations + bootstrap CI ------------------------------ #
    w("## Table 6 — Module ablations (RMSE °C, V3 joint) + RMSE 95% CI\n")
    w("Block-bootstrap 95% CI for the full model (station-pooled).\n")
    w("| variant | h1 | h3 | h7 |")
    w("|---|---|---|---|")
    abl_models = ["ThermoRoute", "TR-noPrior", "TR-fixedKappa", "TR-softmax",
                  "TR-noMoE", "TR-noRouter"]
    for m in abl_models:
        if m == "ThermoRoute":   # reference = full model, V3 joint only
            sub = TEST[(TEST.model == m) & (TEST.scope == "joint") & (TEST.feature_set == "V3")]
        else:
            sub = TEST[(TEST.model == m)]
        if sub.empty:
            continue
        r = sub.groupby("horizon").RMSE.mean()
        w(f"| {m} | {r.get(1, np.nan):.3f} | {r.get(3, np.nan):.3f} | {r.get(7, np.nan):.3f} |")
    w("")
    # bootstrap CI for the station-averaged RMSE (the headline aggregation).
    # Per-station squared errors are block-resampled independently and per-station
    # RMSE is averaged over stations — predictions are NOT averaged across stations
    # (which would cancel independent errors and deflate the RMSE).
    w("**ThermoRoute RMSE 95% CI (moving-block bootstrap, station-averaged):**\n")
    for h in C.HORIZONS:
        sub = tr[tr.horizon == h]
        err2_by_station = {}
        for st, g in sub.groupby("site_id"):
            avg = g.groupby("issue_date").agg(yp=("y_pred", "mean"), yt=("y_true", "mean"))
            err2_by_station[st] = ((avg["yp"] - avg["yt"]) ** 2).to_numpy()
        pt, lo, hi = S.block_bootstrap_station_avg_rmse(err2_by_station, block=30, n_boot=2000)
        w(f"- h={h}: {pt:.3f} [{lo:.3f}, {hi:.3f}] °C")
    w("")

    text = "\n".join(OUT)
    (C.TABLES / "paper_tables.md").write_text(text)
    print("wrote", C.TABLES / "paper_tables.md", f"({len(OUT)} lines)", flush=True)


if __name__ == "__main__":
    main()
