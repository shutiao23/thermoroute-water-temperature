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

_v2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
_120 = C.PREDICTIONS / "usgs_predictions_120.parquet"
_40 = C.PREDICTIONS / "usgs_predictions.parquet"
PRED_PATH = _v2 if _v2.exists() else (_120 if _120.exists() else _40)
print(f"# claim_stats using {PRED_PATH.name}", flush=True)
PRED = pd.read_parquet(PRED_PATH)


HUC_PATH = C.TABLES / "usgs_stations_with_huc.csv"
HUC2 = (pd.read_csv(HUC_PATH).set_index("site_id").huc2.to_dict()
        if HUC_PATH.exists() else {})


def per_station_rmse(model, h):
    """Per-station RMSE of a single-model baseline (seed 0) on the blind test."""
    sub = PRED[(PRED.model == model) & (PRED.split == "test") & (PRED.horizon == h)]
    return {s: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
            for s, g in sub.groupby("site_id")}


def tr_rmse(h, protocol):
    """ThermoRoute per-station RMSE under one of two disclosed protocols.

    * ``per-seed``  — each of the 5 seed models is scored alone (same model
      budget as each baseline); returns the across-seed mean per station plus
      the per-seed dicts so across-seed spread can be reported.
    * ``ensemble``  — the deployed 5-member seed-averaged forecaster; an
      ensemble-vs-single-model comparison and labelled as such.
    """
    sub = PRED[(PRED.model == "ThermoRoute") & (PRED.split == "test") & (PRED.horizon == h)]
    if protocol == "ensemble":
        e = sub.groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        return ({s: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
                 for s, g in e.groupby("site_id")}, None)
    per_seed = {}
    for sd, gs in sub.groupby("seed"):
        per_seed[sd] = {s: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
                        for s, g in gs.groupby("site_id")}
    sites = set.intersection(*(set(d) for d in per_seed.values()))
    mean_d = {s: float(np.mean([per_seed[sd][s] for sd in per_seed])) for s in sites}
    return mean_d, per_seed


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def boot_ci_median_skill(skill, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [np.median(skill[rng.integers(0, len(skill), len(skill))]) for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cluster_boot_ci_median_skill(skill, clusters, n=5000, seed=0):
    """HUC2 cluster bootstrap: resample whole hydrologic regions with
    replacement, acknowledging spatial correlation between stations."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    by_cl = {c: skill[clusters == c] for c in uniq}
    boots = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots.append(np.median(np.concatenate([by_cl[c] for c in pick])))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(uniq)


def claim1():
    refs = ("Persistence", "DampedPersistence", "LightGBM")
    results = {}          # (protocol, h, ref) -> row dict
    n_seen = set()
    for protocol in ("per-seed", "ensemble"):
        for h in C.HORIZONS:
            tr, per_seed = tr_rmse(h, protocol)
            for ref_name in refs:
                ref = per_station_rmse(ref_name, h)
                stations = sorted(s for s in tr if s in ref
                                  and np.isfinite(tr[s]) and np.isfinite(ref[s]))
                n = len(stations)
                n_seen.add(n)
                a = np.array([tr[s] for s in stations])
                b = np.array([ref[s] for s in stations])
                skill = 1 - a / b
                lo, hi = boot_ci_median_skill(skill)
                wins = int((a < b).sum())
                wlo, whi = wilson_ci(wins, n)
                p = wilcoxon(a, b).pvalue if n > 5 else float("nan")
                row = {"n": n, "med": float(np.median(skill)), "lo": lo, "hi": hi,
                       "wins": wins, "win": wins / n, "wlo": wlo, "whi": whi, "p": p}
                if per_seed is not None:      # across-seed spread of the median skill
                    med_by_seed = [np.median([1 - per_seed[sd][s] / ref[s]
                                              for s in stations]) for sd in per_seed]
                    row["seed_std"] = float(np.std(med_by_seed))
                if protocol == "per-seed" and HUC2:
                    cl = np.array([HUC2.get(s, -1) for s in stations])
                    clo, chi, ncl = cluster_boot_ci_median_skill(skill, cl)
                    row.update({"clo": clo, "chi": chi, "ncl": ncl})
                results[(protocol, h, ref_name)] = row
        # Holm adjustment within each protocol (9 tests: 3 horizons x 3 refs)
        keys = [(protocol, h, r) for h in C.HORIZONS for r in refs]
        adj = holm([results[k]["p"] for k in keys])
        for k, ap in zip(keys, adj):
            results[k]["p_holm"] = float(ap)

    n_stations = max(n_seen)
    L = [f"# Claim 1 — accuracy vs baselines, per-station significance "
         f"(n = {n_stations} blind-test stations)\n",
         "Two disclosed protocols. **per-seed**: each of the 5 ThermoRoute seeds is "
         "scored as a single model (same budget as each baseline) and per-station "
         "RMSE is averaged across seeds; ± is the across-seed std of the median "
         "skill. **ensemble**: the deployed 5-member seed-averaged forecaster vs "
         "single-model baselines — an ensemble-vs-single comparison, labelled as "
         "such. Skill = 1−RMSE/RMSE_ref. Win-rate over the n stations with a "
         "Wilson 95% CI. Wilcoxon = two-sided paired signed-rank; Holm-adjusted "
         "across the 9 (horizon × reference) tests per protocol.\n",
         "| protocol | horizon | reference | n | median skill [boot 95% CI] | "
         "win-rate [Wilson 95% CI] | p (raw) | p (Holm) |",
         "|---|---|---|---|---|---|---|---|"]
    for protocol in ("per-seed", "ensemble"):
        for h in C.HORIZONS:
            for ref_name in refs:
                r = results[(protocol, h, ref_name)]
                med = f"{r['med']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
                if "seed_std" in r:
                    med += f" (±{r['seed_std']:.3f})"
                star = "*" if r["p_holm"] < 0.05 else ""
                L.append(f"| {protocol} | {h} | {ref_name} | {r['n']} | {med} | "
                         f"{r['win']:.2f} ({r['wins']}/{r['n']}) "
                         f"[{r['wlo']:.2f}, {r['whi']:.2f}] | "
                         f"{r['p']:.1e} | {r['p_holm']:.1e}{star} |")
    if HUC2:
        L += ["", "## HUC2 cluster-bootstrap robustness (per-seed protocol)\n",
              "Median-skill 95% CI when whole HUC2 regions are resampled with "
              "replacement (stations within a region are spatially correlated and "
              "are not treated as independent):\n",
              "| horizon | reference | station-boot CI | HUC2-cluster CI (k regions) |",
              "|---|---|---|---|"]
        for h in C.HORIZONS:
            for ref_name in refs:
                r = results[("per-seed", h, ref_name)]
                L.append(f"| {h} | {ref_name} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                         f"[{r['clo']:+.3f}, {r['chi']:+.3f}] (k={r['ncl']}) |")
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


def perstation_figure():
    """Figure 2 — per-station RMSE scatter, ThermoRoute (deployed ensemble) vs
    damped persistence, one panel per horizon. Points below the diagonal are
    stations where ThermoRoute is more accurate. Regenerated from the current
    predictions so the figure and its caption win-rates stay in sync."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, h in zip(axes, C.HORIZONS):
        tr, _ = tr_rmse(h, "ensemble")
        dp = per_station_rmse("DampedPersistence", h)
        sts = sorted(s for s in tr if s in dp
                     and np.isfinite(tr[s]) and np.isfinite(dp[s]))
        x = np.array([dp[s] for s in sts])      # damped RMSE
        y = np.array([tr[s] for s in sts])      # ThermoRoute RMSE
        win = int((y < x).sum())
        below = y < x
        ax.scatter(x[below], y[below], s=16, c="#185FA5", alpha=0.75,
                   label=f"TR better ({win}/{len(sts)})")
        ax.scatter(x[~below], y[~below], s=16, c="#B0B0B0", alpha=0.75,
                   label="damped better")
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("damped persistence RMSE (°C)")
        if h == C.HORIZONS[0]:
            ax.set_ylabel("ThermoRoute RMSE (°C)")
        ax.set_title(f"h = {h} d   ({100*win/len(sts):.0f}% below diagonal)")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.grid(alpha=0.2)
    fig.suptitle("Per-station blind-test RMSE: ThermoRoute vs damped persistence "
                 "(114 USGS stations)", y=1.02)
    fig.savefig(C.FIGURES / "fig_usgs_perstation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_usgs_perstation.png", flush=True)


if __name__ == "__main__":
    claim1()
    claim3()
    perstation_figure()
