#!/usr/bin/env python3
"""Hydrologic mechanism analysis on the held-out 2021-2023 window.

Derives, from the frozen prediction table and panels only (no re-training):
  1. per-station thermal memory: e-folding half-life from a training-period
     AR(1) fit on damped-seasonal anomalies (phi -> half-life in days);
  2. per-station memory/learned gain decomposition:
        G_memory = RMSE_persistence - RMSE_damped
        G_learned = RMSE_damped - RMSE_model
  3. stratified paired DeltaRMSE (ThermoRoute vs damped persistence) over
     season, warm/cold extremes, rapid warming/cooling, flow quantiles and
     air-water disequilibrium, with the station count per stratum;
  4. flow conditional permutation sensitivity for LightGBM is NOT re-inferred
     here (requires the window pipeline); the development-period synthetic
     perturbation and the stratified flow results are the flow evidence.

Output: outputs/conventional/mechanism_2021_2023.json + summary printout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/conventional")
PRED = OUT / "predictions_2021_2023.parquet"


def fit_half_life(train_panel: pd.DataFrame, stations: list[str]) -> dict:
    """AR(1) on train-period damped-seasonal anomalies -> phi -> half-life."""
    half_life: dict[str, float] = {}
    for st in stations:
        sub = train_panel[train_panel.site_id == st].sort_values("DATE")
        y = sub["WTEMP"].to_numpy(float)
        doy = sub["DATE"].dt.dayofyear.to_numpy()
        m = np.isfinite(y)
        # damped seasonal climatology: least squares harmonic
        X = np.concatenate([np.ones((len(doy), 1)),
                            np.column_stack([
                                np.sin(2*np.pi*k*doy/365.2425) for k in range(1, 3)
                            ] + [
                                np.cos(2*np.pi*k*doy/365.2425) for k in range(1, 3)
                            ])], axis=1)
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        anomaly = y - X @ beta
        a = anomaly[:-1][m[:-1] & m[1:]]
        b = anomaly[1:][m[:-1] & m[1:]]
        if len(a) < 100 or np.var(a) == 0:
            continue
        phi = float(np.dot(a, b) / np.dot(a, a))
        phi = min(max(phi, 0.0), 0.9999)
        if phi <= 0:
            continue
        half_life[st] = float(np.log(0.5) / np.log(phi))
    return half_life


def main() -> None:
    pred = pd.read_parquet(PRED)
    panel = pd.read_parquet(OUT / "panel_2021_2023.parquet")
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    panel["site_id"] = panel["site_id"].astype(str).str.zfill(8)

    train_panel = pd.read_parquet("data_usgs/panel_usgs_120v2.parquet")
    train_panel["DATE"] = pd.to_datetime(train_panel["DATE"])
    reg = pd.read_csv("data_usgs/station_registry_v1.csv",
                      usecols=["site_no", "legacy_site_id"])
    alias = dict(zip(reg["legacy_site_id"].astype(str),
                     reg["site_no"].astype(str).str.zfill(8)))
    train_panel["site_id"] = train_panel["site_id"].astype(str).map(alias)

    stations = sorted(pred.site_id.unique())
    sm = pd.read_csv(OUT / "station_metrics_2021_2023.csv")
    reportable = set(
        sm[(sm.model == "ThermoRoute") & (sm.n >= 100)].site_id.astype(str).str.zfill(8))
    stations = [s for s in stations if s in reportable]
    print(f"reportable stations: {len(stations)}")
    hl = fit_half_life(train_panel, stations)
    print(f"half-life fitted for {len(hl)}/{len(stations)} stations")

    keep = pred.model.isin(["ThermoRoute", "DampedPersistence", "Persistence", "LightGBM", "LSTM"])
    pp = pred[keep].copy()

    # target-day features for stratification
    feat = panel[["DATE", "site_id", "TEMP", "FLOW", "WTEMP"]].rename(
        columns={"DATE": "target_date", "TEMP": "TAIR_t", "FLOW": "FLOW_t",
                 "WTEMP": "WT_t"})
    pp = pp.merge(feat, on=["site_id", "target_date"], how="left", validate="many_to_one")
    feat_i = panel[["DATE", "site_id", "WTEMP"]].rename(
        columns={"DATE": "issue_date", "WTEMP": "WT_i"})
    pp = pp.merge(feat_i, on=["site_id", "issue_date"], how="left", validate="many_to_one")

    pp["dT"] = pp["WT_t"] - pp["WT_i"]
    pp["season"] = pp["target_date"].dt.month.map(
        lambda m: {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA"}.get(m, "SON"))

    out: dict = {}

    # --- 1. memory / learned decomposition per station ---
    wide = pp.pivot_table(index=["site_id", "horizon"], columns="model",
                          values="y_pred", aggfunc="first")
    met = {}
    for (st, h), row in wide.iterrows():
        yt = pp[(pp.site_id == st) & (pp.horizon == h) & pp.model.eq("ThermoRoute")]["y_true"]
        key = (st, h)
        if yt.empty:
            continue
        met[key] = {}
        for mdl in ["Persistence", "DampedPersistence", "ThermoRoute"]:
            y = pp[(pp.site_id == st) & (pp.horizon == h) & pp.model.eq(mdl)]
            met[key][mdl] = float(np.sqrt(np.mean((y.y_pred - y.y_true) ** 2)))
    dec = pd.DataFrame.from_dict(met, orient="index")
    dec.index = pd.MultiIndex.from_tuples(dec.index, names=["site_id", "horizon"])
    dec["G_memory"] = dec["Persistence"] - dec["DampedPersistence"]
    dec["G_learned"] = dec["DampedPersistence"] - dec["ThermoRoute"]
    dec["half_life"] = dec.index.get_level_values("site_id").map(hl)
    med = dec.groupby("horizon")[["G_memory", "G_learned"]].median()
    out["memory_learned_median"] = {
        str(h): {"G_memory": float(r.G_memory), "G_learned": float(r.G_learned)}
        for h, r in med.iterrows()}
    corr = dec.groupby("horizon").apply(
        lambda g: g["half_life"].corr(g["G_learned"]), include_groups=False)
    out["corr_half_life_learned"] = {str(h): float(v) for h, v in corr.items()}
    out["half_life_median"] = float(np.nanmedian(list(hl.values())))
    out["half_life_iqr"] = list(np.nanpercentile(list(hl.values()), [25, 75]))

    # --- 2. stratified pooled DeltaRMSE TR vs damped ---
    d = pp[pp.model.eq("DampedPersistence")][["site_id", "horizon", "issue_date", "y_pred"]].rename(
        columns={"y_pred": "damped"})
    t = pp[pp.model.eq("ThermoRoute")][["site_id", "horizon", "issue_date", "y_pred", "y_true"]].rename(
        columns={"y_pred": "tr"})
    both = t.merge(d, on=["site_id", "horizon", "issue_date"], validate="one_to_one")
    both = both.merge(pp[["site_id", "horizon", "issue_date", "season", "dT", "WT_t", "FLOW_t", "TAIR_t"]].drop_duplicates(),
                      on=["site_id", "horizon", "issue_date"], validate="one_to_one")
    both = both[both.y_true.notna() & both.tr.notna() & both.damped.notna()]
    both = both[both.site_id.isin(reportable)]

    def delta_rmse(sub):
        e_tr = sub.tr - sub.y_true
        e_d = sub.damped - sub.y_true
        return float(np.sqrt(np.mean(e_tr**2)) - np.sqrt(np.mean(e_d**2)))

    strata: dict[str, dict] = {}
    for h in (1, 3, 7):
        sub = both[both.horizon == h]
        strata[str(h)] = {
            "all": {"n_keys": int(len(sub)), "n_stations": int(sub.site_id.nunique()),
                    "delta_rmse": delta_rmse(sub)}}
        for season in ("DJF", "MAM", "JJA", "SON"):
            g = sub[sub.season == season]
            strata[str(h)][f"season_{season}"] = {
                "n_keys": int(len(g)), "n_stations": int(g.site_id.nunique()),
                "delta_rmse": delta_rmse(g)}
        q10, q90 = sub.WT_t.quantile([0.1, 0.9])
        for name, mask in [("warmest10", sub.WT_t >= q90), ("coldest10", sub.WT_t <= q10)]:
            g = sub[mask]
            strata[str(h)][name] = {"n_keys": int(len(g)), "n_stations": int(g.site_id.nunique()),
                                    "delta_rmse": delta_rmse(g)}
        dq = sub.dT.abs()
        hi = sub[dq >= dq.quantile(0.9)]
        strata[str(h)]["rapid_change10"] = {"n_keys": int(len(hi)), "n_stations": int(hi.site_id.nunique()),
                                            "delta_rmse": delta_rmse(hi)}
        if sub.FLOW_t.notna().any():
            fq10, fq90 = sub.FLOW_t.quantile([0.1, 0.9])
            for name, mask in [("low_flow10", sub.FLOW_t <= fq10), ("high_flow10", sub.FLOW_t >= fq90)]:
                g = sub[mask]
                strata[str(h)][name] = {"n_keys": int(len(g)), "n_stations": int(g.site_id.nunique()),
                                        "delta_rmse": delta_rmse(g)}
        airw = (sub.TAIR_t - sub.WT_t).abs()
        hi_aw = sub[airw >= airw.quantile(0.9)]
        strata[str(h)]["high_disequilibrium10"] = {
            "n_keys": int(len(hi_aw)), "n_stations": int(hi_aw.site_id.nunique()),
            "delta_rmse": delta_rmse(hi_aw)}
    out["stratified_delta_rmse"] = strata

    # --- 3. station-level count of TR < damped per lead (win rate by lead) ---
    wins = {}
    for h in (1, 3, 7):
        g = both[both.horizon == h]
        per = g.groupby("site_id").apply(lambda x: delta_rmse(x) < 0, include_groups=False)
        wins[str(h)] = {"win_frac": float(per.mean()), "n": int(len(per))}
    out["tr_win_frac_by_lead"] = wins

    dec_out = dec.reset_index()
    dec_out["half_life"] = dec_out["site_id"].map(hl)
    dec_out.to_csv(OUT / "mechanism_station_level_2021_2023.csv", index=False)

    (OUT / "mechanism_2021_2023.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["memory_learned_median"], indent=1))
    print("corr:", out["corr_half_life_learned"])
    print("half-life median:", out["half_life_median"], out["half_life_iqr"])
    print("wins:", wins)
    print("7d strata:", json.dumps(out["stratified_delta_rmse"]["7"], indent=1))


if __name__ == "__main__":
    main()
