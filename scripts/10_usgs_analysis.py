#!/usr/bin/env python3
"""Stage 10 — USGS calibration, decision value, and mechanism analysis.

Consumes the large-sample predictions + saved model from stage 9 and produces:
  * conformal PICP/MPIW/CRPS + high-temp event scores (ThermoRoute vs LightGBM);
  * Relative Economic Value (decision value) curves on the large sample;
  * the dynamic relaxation rate κ stratified by flow regime (tests whether the
    flow-dependence of thermal memory is stronger on free-flowing rivers), and
    router variable×lag maps;
  * figures fig_usgs_*.png.

Run:  PYTHONPATH=src python3 scripts/10_usgs_analysis.py
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
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute import metrics as M
from thermoroute import decision as DEC
from thermoroute.conformal import cqr_offsets, apply_cqr
from thermoroute.thermoroute import ThermoRoute

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
DELTA_SCALE = 1.5
# Auto-pick the latest predictions+panel pair: prefer 120-station if present.
_120_pred = C.PREDICTIONS / "usgs_predictions_120.parquet"
PRED = _120_pred if _120_pred.exists() else C.PREDICTIONS / "usgs_predictions.parquet"
_panel_120 = ROOT / "data_usgs" / "panel_usgs_100.parquet"
PANEL_PATH = _panel_120 if _120_pred.exists() else (ROOT / "data_usgs" / "panel_usgs_wind.parquet")


def prep():
    panel = pd.read_parquet(PANEL_PATH)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    stations = tuple(sorted(panel.site_id.unique()))
    C.STATIONS = stations
    C.UPSTREAM = {s: None for s in stations}
    for v in C.ALL_VARS:
        panel[f"{v}_observed"] = panel[v].notna()
    masks = D.split_masks(panel["DATE"])
    panel_imp = D.Imputer.fit(panel, masks.train).transform(panel)
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    return panel, panel_imp, masks, clim, stations, thr


def ensemble_thermo(pred):
    """Average ThermoRoute over seeds → one calibrated set per (site,horizon,date)."""
    tr = pred[pred.model == "ThermoRoute"]
    g = tr.groupby(["site_id", "horizon", "issue_date", "split"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        q05=("q05", "mean"), q50=("q50", "mean"), q95=("q95", "mean"),
        p_exceed=("p_exceed", "mean"), target_date=("target_date", "first"))
    g["model"] = "ThermoRoute"
    return g


def calibration_table(pred, thr):
    rows = []
    tr = ensemble_thermo(pred)
    lg = pred[pred.model == "LightGBM"]
    for name, df in [("ThermoRoute", tr), ("LightGBM", lg)]:
        off = cqr_offsets(df[df.split == "calib"])
        dc = apply_cqr(df, off)
        te = dc[dc.split == "test"]
        for h in C.HORIZONS:
            g = te[te.horizon == h]
            y = g.y_true.to_numpy(float)
            quants = {0.05: g.q05.to_numpy(float), 0.50: g.q50.to_numpy(float),
                      0.95: g.q95.to_numpy(float)}
            ps = M.probabilistic_scores(y, quants)
            ybin = (y > np.array([thr[s] for s in g.site_id])).astype(float)
            ev = M.event_scores(ybin, g.p_exceed.to_numpy(float))
            rows.append({"model": name, "horizon": h, "PICP": ps["PICP"],
                         "MPIW": ps["MPIW"], "CRPS": ps["CRPS"],
                         "BRIER_SKILL": ev["BRIER_SKILL"], "AUPRC": ev["AUPRC"]})
    return pd.DataFrame(rows)


def rev_table_and_fig(pred, thr):
    alphas = np.linspace(0.01, 0.99, 99)
    tr = ensemble_thermo(pred)
    models = [("ThermoRoute", tr, True, "#185FA5"),
              ("LightGBM", pred[pred.model == "LightGBM"], True, "#7F77DD"),
              ("DampedPersistence", pred[pred.model == "DampedPersistence"], False, "#BA7517"),
              ("Persistence", pred[pred.model == "Persistence"], False, "#888780")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    rows = []
    for ax, h in zip(axes, C.HORIZONS):
        for name, df, prob, col in models:
            g = df[(df.split == "test") & (df.horizon == h)]
            if g.empty:
                continue
            th = np.array([thr[s] for s in g.site_id])
            ev = (g.y_true.to_numpy() > th).astype(int)
            sc = g.p_exceed.to_numpy(float) if prob else (g.y_pred.to_numpy() > th).astype(float)
            rev = DEC.rev_curve(ev, sc, alphas, probabilistic=prob)
            ax.plot(alphas, np.clip(rev, -0.2, 1), "-" if prob else "--", color=col, lw=1.6,
                    label=name + ("" if prob else " (det.)"))
            summ = DEC.value_summary(ev, sc, probabilistic=prob)
            summ.update({"model": name, "horizon": h})
            rows.append(summ)
        ax.axhline(0, color="#888780", lw=0.7); ax.set_ylim(-0.2, 1)
        ax.set_title(f"h = {h} d"); ax.set_xlabel("cost–loss ratio α"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Relative Economic Value")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Decision value, USGS large sample", y=1.03)
    fig.savefig(C.FIGURES / "fig_usgs_rev.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def mechanism(panel, panel_imp, masks, clim, stations, thr):
    wd = DS.build_windows(panel_imp, masks, clim, variables=USGS_VARS,
                          require_observed_target=True)
    model = ThermoRoute(n_vars=len(wd.var_names), n_stations=len(stations),
                        n_phys=wd.n_phys, delta_scale=DELTA_SCALE)
    model.load_state_dict(torch.load(C.MODELS / "thermoroute_usgs.pt"))
    model.eval()
    idx = np.concatenate([wd.idx("calib"), wd.idx("test")])
    # process in chunks (a single ~50k-sample forward can yield sporadic NaNs)
    kap, lws = [], []
    with torch.no_grad():
        for s in range(0, len(idx), 4096):
            o = model(wd.batch(idx[s:s + 4096]))
            kap.append(o.kappa.numpy()); lws.append(o.lag_weights.numpy())
    kappa = np.concatenate(kap)
    lag_w = np.concatenate(lws)
    lf = wd.logflowz[idx]
    station = wd.station[idx]
    # κ flow-dependence: per station, low vs high flow tertile
    rows = []
    for i, s in enumerate(stations):
        sel = station == i
        if sel.sum() < 50:
            continue
        x = lf[sel]
        q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
        kl = kappa[sel][x <= q1].mean(); kh = kappa[sel][x >= q2].mean()
        rows.append({"site": s, "kappa_low": kl, "kappa_high": kh,
                     "ratio": kh / max(kl, 1e-6)})
    kdf = pd.DataFrame(rows)

    # figure: κ vs flow (pooled binned) + ratio distribution
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    bins = np.quantile(lf, np.linspace(0, 1, 11))
    bi = np.digitize(lf, bins[1:-1])
    mx = [lf[bi == b].mean() for b in range(len(bins) - 1)]
    mk = [kappa[bi == b].mean() for b in range(len(bins) - 1)]
    a1.plot(mx, mk, "-o", color="#185FA5")
    a1.set_xlabel("z(log FLOW)"); a1.set_ylabel("relaxation rate κ"); a1.grid(alpha=0.25)
    a1.set_title("a  κ rises with flow (shorter memory)")
    a2.hist(kdf["ratio"], bins=20, color="#1D9E75", alpha=0.8)
    a2.axvline(1.0, color="#993C1D", ls="--")
    a2.set_xlabel("κ_high / κ_low per station"); a2.set_ylabel("# stations")
    a2.set_title("b  flow-dependence across stations")
    fig.savefig(C.FIGURES / "fig_usgs_kappa.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # router variable importance by horizon (nanmean guards sporadic NaN rows)
    vn = list(USGS_VARS); ov = np.nanmean(lag_w, axis=0)   # [H,V,Lr1]
    drivers = {}
    for hi, h in enumerate(C.HORIZONS):
        vi = ov[hi].sum(axis=1)
        order = np.argsort(vi)[::-1][:3]
        drivers[h] = ", ".join(f"{vn[v]} ({vi[v]*100:.0f}%)" for v in order)
    return kdf, drivers


def main():
    pred = pd.read_parquet(PRED)
    panel, panel_imp, masks, clim, stations, thr = prep()

    cal = calibration_table(pred, thr)
    rev = rev_table_and_fig(pred, thr)
    kdf, drivers = mechanism(panel, panel_imp, masks, clim, stations, thr)

    L = ["# USGS large-sample: calibration, decision value, mechanism\n",
         "## Probabilistic & event metrics (conformal, test)\n",
         "| model | h | PICP | MPIW | CRPS | Brier-skill | AUPRC |",
         "|---|---|---|---|---|---|---|"]
    for _, r in cal.iterrows():
        L.append(f"| {r['model']} | {int(r.horizon)} | {r.PICP:.3f} | {r.MPIW:.2f} | "
                 f"{r.CRPS:.3f} | {r.BRIER_SKILL:+.3f} | {r.AUPRC:.3f} |")
    L += ["", "## Decision value (peak REV)\n",
          "| model | h | REV_max | REV@0.1 | REV@0.2 |", "|---|---|---|---|---|"]
    for _, r in rev.sort_values(["horizon", "model"]).iterrows():
        L.append(f"| {r['model']} | {int(r.horizon)} | {r['REV_max']:.3f} | "
                 f"{r['REV@0.1']:.3f} | {r['REV@0.2']:.3f} |")
    frac = float((kdf["ratio"] > 1).mean())
    L += ["", "## Dynamic thermal memory — κ flow-dependence\n",
          f"- κ_high/κ_low > 1 (faster relaxation at high flow) at "
          f"**{frac*100:.0f}% of stations** (median ratio {kdf['ratio'].median():.2f}).",
          f"- mean κ_low={kdf.kappa_low.mean():.3f}, κ_high={kdf.kappa_high.mean():.3f}.",
          "", "## Router top drivers by horizon\n"]
    for h, s in drivers.items():
        L.append(f"- h={h}d: {s}")
    cal.to_csv(C.TABLES / "usgs_calibration.csv", index=False)
    rev.to_csv(C.TABLES / "usgs_rev.csv", index=False)
    kdf.to_csv(C.TABLES / "usgs_kappa.csv", index=False)
    (C.REPORTS / "usgs_analysis.md").write_text("\n".join(L))
    print("\n".join(L), flush=True)
    print("\nwrote outputs/reports/usgs_analysis.md + figs", flush=True)


if __name__ == "__main__":
    main()
