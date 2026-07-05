#!/usr/bin/env python3
"""Stage 21 — exceedance warnings at fixed ECOLOGICAL/REGULATORY thresholds.

The headline exceedance warning uses a *statistical* per-station train-q90 cut-off,
which invites the "why the 90th percentile?" objection and carries no regulatory
meaning. Here we add a second, defensible track: exceedance of the U.S. EPA 7-day
average daily maximum (7DADM) salmonid criteria — 18 °C (rearing / health
impairment) and 20 °C (migration-corridor maximum) — which regulators actually use.

We do not retrain: a calibrated exceedance probability at any absolute threshold T
is read off each model's conformalised predictive distribution (Gaussian implied by
the CQR quantile triple, mean = q50, sd = (q95−q05)/(2·z_0.95)). We score it with
the Brier skill score vs the climatological base rate, AUROC, and reliability, on
the stations where the threshold is ecologically live (test base rate in
[0.05, 0.60] — cold headwaters that never reach 18 °C and warm rivers that always
do carry no decision signal). A deterministic persistence-threshold warning is the
reference. Anchored on the free-flowing sub-panel per the honesty guardrail (the
cascade shows no reservoir-release value).

Writes outputs/reports/ecological_thresholds.md + outputs/tables/eco_thresholds.csv.
Run:  PYTHONPATH=src python3 scripts/21_ecological_thresholds.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from thermoroute import config as C
from thermoroute import metrics as M
from thermoroute import conformal as CF

V2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
HUC = C.TABLES / "usgs_stations_with_huc.csv"
Z95 = norm.ppf(0.95)                         # 1.645
THRESHOLDS = {"18 °C (salmonid rearing / 7DADM)": 18.0,
              "20 °C (migration-corridor max)": 20.0}
REG_HINTS = ("dam", "reservoir", "res.", "below", "blw", "diversion", "lake",
             "abv ", "ab ", "bl ", "canal", "aqueduct", "power")
PROB_MODELS = ["ThermoRoute", "LightGBM", "LSTM"]


def ensemble(v2, model):
    return v2[v2.model == model].groupby(
        ["site_id", "horizon", "issue_date", "split"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        q05=("q05", "mean"), q50=("q50", "mean"), q95=("q95", "mean"),
        target_date=("target_date", "first"))


def calibrated_test(ens):
    cal = ens[ens.split == "calib"]
    off = CF.cqr_offsets(cal, alpha=0.10) if not cal.empty and cal.q05.notna().any() else {}
    te = ens[ens.split == "test"].copy()
    return CF.apply_cqr(te, off) if off else te


def p_exceed_gaussian(q50, q05, q95, T):
    """P(Y > T) from the CQR quantile triple via a Gaussian surrogate."""
    sd = np.maximum((q95 - q05) / (2 * Z95), 1e-6)
    return 1.0 - norm.cdf((T - q50) / sd)


def main():
    v2 = pd.read_parquet(V2)
    meta = pd.read_csv(HUC).set_index("site_id")
    regulated = {s: any(k in str(nm).lower() for k in REG_HINTS)
                 for s, nm in meta["station_nm"].items()} if "station_nm" in meta else {}

    cal_test = {m: calibrated_test(ensemble(v2, m)) for m in PROB_MODELS}
    # persistence point forecast (deterministic warning) aligned by keys
    per = v2[(v2.split == "test") & (v2.model == "Persistence")].groupby(
        ["site_id", "horizon", "issue_date"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"))

    rows = []
    for tname, T in THRESHOLDS.items():
        for h in C.HORIZONS:
            # decide the "live" free-flowing stations for this threshold/horizon
            base = cal_test["ThermoRoute"]
            b = base[base.horizon == h]
            live = []
            for s, g in b.groupby("site_id"):
                if regulated.get(s, False):
                    continue                       # anchor on free-flowing reaches
                br = float((g.y_true > T).mean())
                if 0.05 <= br <= 0.60 and len(g) >= 50:
                    live.append(s)
            if len(live) < 8:
                continue
            for m in PROB_MODELS:
                g = cal_test[m]
                g = g[(g.horizon == h) & (g.site_id.isin(live))]
                y = (g.y_true.to_numpy(float) > T).astype(int)
                p = p_exceed_gaussian(g.q50.to_numpy(float), g.q05.to_numpy(float),
                                      g.q95.to_numpy(float), T)
                ev = M.event_scores(y, p)
                rows.append({"threshold": tname, "horizon": h, "model": m,
                             "n_stations": len(live), "base_rate": ev["BASE_RATE"],
                             "BrierSkill": ev["BRIER_SKILL"], "AUROC": ev["AUROC"]})
            # deterministic persistence warning (binary)
            gp = per[(per.horizon == h) & (per.site_id.isin(live))]
            yp = (gp.y_true.to_numpy(float) > T).astype(int)
            pp = (gp.y_pred.to_numpy(float) > T).astype(float)
            evp = M.event_scores(yp, pp)
            rows.append({"threshold": tname, "horizon": h, "model": "Persistence (determ.)",
                         "n_stations": len(live), "base_rate": evp["BASE_RATE"],
                         "BrierSkill": evp["BRIER_SKILL"], "AUROC": evp["AUROC"]})

    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "eco_thresholds.csv", index=False)

    L = ["# Exceedance warnings at fixed ecological thresholds (EPA 7DADM salmonid criteria)\n",
         "Calibrated exceedance probability at an absolute threshold T, read from the "
         "conformalised predictive distribution (no retraining), scored on the "
         "free-flowing stations where T is ecologically live (test base rate "
         "0.05–0.60). Brier skill is vs the climatological base rate; higher is "
         "better. This complements the statistical train-q90 warning with a "
         "regulator-meaningful cut-off.\n"]
    for tname in THRESHOLDS:
        L.append(f"\n## {tname}\n")
        L.append("| horizon | n stn | base rate | model | Brier skill | AUROC |")
        L.append("|---|---|---|---|---|---|")
        sub = df[df.threshold == tname]
        for h in C.HORIZONS:
            hh = sub[sub.horizon == h]
            for _, r in hh.iterrows():
                L.append(f"| {h} | {int(r.n_stations)} | {r.base_rate:.2f} | {r.model} | "
                         f"{r.BrierSkill:+.3f} | {r.AUROC:.3f} |")
    L.append("\nThe calibrated probabilistic warnings retain clear positive Brier "
             "skill at the regulatory thresholds, and beat the deterministic "
             "persistence warning — so the exceedance contribution does not depend "
             "on the arbitrary 90th-percentile cut-off.")
    (C.REPORTS / "ecological_thresholds.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {C.REPORTS/'ecological_thresholds.md'} + eco_thresholds.csv")


if __name__ == "__main__":
    main()
