#!/usr/bin/env python3
"""Stage 12 — statistical rigor for claims 1 (accuracy) and 3 (calibration).

Treats the 40 stations as the sample unit (the level at which we claim
generality). For claim 1: per-station paired tests of ThermoRoute vs each
baseline (Wilcoxon signed-rank + station-bootstrap CI on median skill + win-rate).
For claim 3: a calibration figure (per-station PICP distribution, PICP and MPIW
vs horizon) plus the achieved-coverage summary.

Run:  PYTHONPATH=src python3 scripts/12_claim_stats.py
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
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermoroute import config as C
from thermoroute import data as D
from thermoroute.conformal import cqr_offsets, apply_cqr
from thermoroute import metrics as M

PRED = pd.read_parquet(C.PREDICTIONS / "usgs_predictions.parquet")


def per_station_rmse(model, h, ensemble=False):
    sub = PRED[(PRED.model == model) & (PRED.split == "test") & (PRED.horizon == h)]
    if ensemble:
        sub = sub.groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
    return {s: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
            for s, g in sub.groupby("site_id")}


def boot_ci_median_skill(tr, ref, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    stations = [s for s in tr if s in ref]
    skill = np.array([1 - tr[s] / ref[s] for s in stations])
    boots = [np.median(skill[rng.integers(0, len(skill), len(skill))]) for _ in range(n)]
    return float(np.median(skill)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def claim1():
    L = ["# Claim 1 — accuracy vs baselines, per-station significance (n=40)\n",
         "ThermoRoute = 5-seed mean. Skill = 1−RMSE/RMSE_ref; win-rate = fraction of "
         "stations where ThermoRoute is better; Wilcoxon = paired signed-rank p.\n",
         "| horizon | reference | median skill [95% CI] | win-rate | Wilcoxon p |",
         "|---|---|---|---|---|"]
    for h in C.HORIZONS:
        tr = per_station_rmse("ThermoRoute", h, ensemble=True)
        for ref_name in ("Persistence", "DampedPersistence", "LightGBM"):
            ref = per_station_rmse(ref_name, h)
            stations = [s for s in tr if s in ref]
            a = np.array([tr[s] for s in stations]); b = np.array([ref[s] for s in stations])
            med, lo, hi = boot_ci_median_skill(tr, ref)
            win = float((a < b).mean())
            p = wilcoxon(a, b).pvalue if len(stations) > 5 else float("nan")
            star = "*" if p < 0.05 else ""
            L.append(f"| {h} | {ref_name} | {med:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{win:.2f} | {p:.1e}{star} |")
    (C.TABLES / "claim1_significance.md").write_text("\n".join(L))
    print("\n".join(L), flush=True)


def claim3():
    # conformalise the 5-seed ThermoRoute ensemble per (station,horizon)
    tr = PRED[PRED.model == "ThermoRoute"].groupby(
        ["site_id", "horizon", "issue_date", "split"], as_index=False).agg(
        y_true=("y_true", "first"), q05=("q05", "mean"), q50=("q50", "mean"),
        q95=("q95", "mean"), target_date=("target_date", "first"))
    off = cqr_offsets(tr[tr.split == "calib"])
    dc = apply_cqr(tr, off)
    te = dc[dc.split == "test"]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.4))
    colors = {1: "#185FA5", 3: "#1D9E75", 7: "#993C1D"}
    picp_rows = []
    for h in C.HORIZONS:
        per = []
        for s, g in te[te.horizon == h].groupby("site_id"):
            per.append(M.coverage(g.y_true.to_numpy(), g.q05.to_numpy(), g.q95.to_numpy()))
        per = np.array(per)
        a1.hist(per, bins=np.linspace(0.6, 1.0, 21), alpha=0.55, color=colors[h], label=f"h={h}d")
        picp_rows.append({"horizon": h, "PICP_mean": per.mean(), "PICP_median": np.median(per),
                          "frac_within_0.05": float((np.abs(per - 0.9) <= 0.05).mean())})
    a1.axvline(0.90, color="black", ls="--", lw=1)
    a1.set_xlabel("per-station PICP (90% target)"); a1.set_ylabel("# stations")
    a1.set_title("a  coverage distribution"); a1.legend(fontsize=8, frameon=False)

    picp = pd.DataFrame(picp_rows)
    a2.plot(picp.horizon, picp.PICP_mean, "-o", color="#185FA5")
    a2.axhline(0.90, color="#993C1D", ls="--"); a2.set_ylim(0.8, 1.0)
    a2.set_xticks(list(C.HORIZONS)); a2.set_xlabel("horizon (d)")
    a2.set_ylabel("mean PICP"); a2.set_title("b  coverage vs lead time"); a2.grid(alpha=0.25)

    mpiw = te.groupby("horizon").apply(lambda g: (g.q95 - g.q05).mean())
    a3.plot(mpiw.index, mpiw.values, "-o", color="#1D9E75")
    a3.set_xticks(list(C.HORIZONS)); a3.set_xlabel("horizon (d)")
    a3.set_ylabel("mean interval width (°C)"); a3.set_title("c  sharpness"); a3.grid(alpha=0.25)
    fig.suptitle("Conformal calibration on the USGS large sample", y=1.03)
    fig.savefig(C.FIGURES / "fig_usgs_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    picp.to_csv(C.TABLES / "claim3_calibration.csv", index=False)
    print("\n=== Claim 3 calibration ===")
    print(picp.round(3).to_string(index=False), flush=True)
    print("wrote fig_usgs_calibration.png", flush=True)


if __name__ == "__main__":
    claim1()
    claim3()
