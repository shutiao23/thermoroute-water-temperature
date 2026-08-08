#!/usr/bin/env python3
"""Hydrologic-regime analysis on the independent 2021-2023 window (v2).

Rewrites ``scripts/conventional_mechanism_analysis.py`` per protocol v1 and the
external review (Major Comment 6):

1. **Anomaly half-life from the official anchor.**  ``t_1/2 = ln(0.5)/ln(phi)``
   where ``phi`` comes from the SAME train-fitted
   ``features.DampedPersistenceAnchor`` used by the main models (not a
   re-fitted AR(1)).  It is an anomaly half-life, never an e-folding time.
2. **Issue-time-identifiable states (primary).**  All thresholds are
   station-specific (or station-month-specific) empirical quantiles fitted on
   the 2006-2015 training period only: current thermal anomaly, recent 7-day
   water-temperature trend, signed air-water disequilibrium, station-relative
   flow percentile, recent flow change, season.
3. **Retrospective outcome-conditioned states (secondary).**  Actual rapid
   warming (positive, station-specific) and actual rapid cooling (negative)
   are computed separately; actual warmest/coldest deciles are station-relative
   training quantiles; actual high/low flow uses station-month training
   quantiles.  These are explicitly labelled diagnostics, never
   issue-time-identifiable states.
4. **Station-first state metrics.**  For each station x horizon x state the
   paired DeltaRMSE (ThermoRoute - damped persistence) is computed on the
   common keys within the state (>= 30 keys), then summarized by the median
   across stations -- never a pooled RMSE difference.

Outputs (all under ``outputs/final/``):
  hydrologic_state_effects.parquet
  basin_attributes.parquet
  mechanism_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import features as F  # noqa: E402
from thermoroute import config as C  # noqa: E402

FINAL = ROOT / "outputs" / "final"
OUT_CONV = ROOT / "outputs" / "conventional"

MIN_STATE_KEYS = 30
CANDIDATE = "ThermoRoute"
REFERENCE = "DampedPersistence"
SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
}


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv",
                           dtype={"site_no": str, "legacy_site_id": str})
    alias = dict(zip(registry["legacy_site_id"].astype(str),
                     registry["site_no"].astype(str).str.zfill(8)))
    dev = pd.read_parquet(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")
    dev["DATE"] = pd.to_datetime(dev["DATE"])
    dev["site_id"] = dev["site_id"].astype(str).map(alias)
    hold = pd.read_parquet(OUT_CONV / "panel_2021_2023.parquet")
    hold["DATE"] = pd.to_datetime(hold["DATE"])
    hold["site_id"] = hold["site_id"].astype(str).str.zfill(8)
    registry["site_no"] = registry["site_no"].str.strip()
    return dev, hold, registry


def official_climatology(dev: pd.DataFrame,
                         registry: pd.DataFrame) -> F.HarmonicClimatology:
    dev = dev.sort_values(["site_id", "DATE"])
    train_mask = dev["DATE"] <= pd.Timestamp("2015-12-31")
    stations = tuple(registry["site_no"].astype(str).str.zfill(8))
    return F.HarmonicClimatology.fit(
        dev, train_mask.to_numpy(), fit_stations=stations)


def fit_official_half_life(dev: pd.DataFrame, registry: pd.DataFrame,
                           clim: F.HarmonicClimatology) -> dict[str, dict[str, float]]:
    """phi from the official anchor fit; anomaly half-life in days."""
    dev = dev.sort_values(["site_id", "DATE"])
    train_mask = dev["DATE"] <= pd.Timestamp("2015-12-31")
    stations = tuple(registry["site_no"].astype(str).str.zfill(8))
    anchor = F.DampedPersistenceAnchor.fit(
        dev, train_mask.to_numpy(), clim, fit_stations=stations)
    out: dict[str, dict[str, float]] = {}
    for site in stations:
        phi = float(anchor.phi.get(site, np.nan))
        if not np.isfinite(phi) or phi <= 0.0:
            out[site] = {"phi": float("nan"), "half_life_days": float("nan"),
                         "capped": False}
            continue
        capped = bool(phi >= 0.999)
        t_half = float(np.log(0.5) / np.log(min(phi, 0.999)))
        out[site] = {"phi": phi, "half_life_days": t_half, "capped": capped}
    return out


def station_training_quantiles(dev: pd.DataFrame,
                               clim: F.HarmonicClimatology,
                               stations: tuple[str, ...]) -> dict[str, dict]:
    """Per-station training (2006-2015) empirical q10/q90 for the state
    definitions.  Flow percentiles are station x calendar-month."""
    tr = dev[dev["DATE"] <= pd.Timestamp("2015-12-31")].copy()
    tr = tr.sort_values(["site_id", "DATE"]).reset_index(drop=True)
    for column in ("WTEMP", "TEMP", "FLOW"):
        tr[column] = pd.to_numeric(tr[column], errors="coerce")
    out: dict[str, dict] = {}
    for site, g in tr.groupby("site_id", sort=True):
        g = g.reset_index(drop=True)
        wt = g["WTEMP"].to_numpy(float)
        temp = g["TEMP"].to_numpy(float)
        flow = g["FLOW"].to_numpy(float)
        wt_prev6 = np.full(len(wt), np.nan)
        wt_prev6[6:] = wt[:-6]
        flow_prev3 = np.full(len(flow), np.nan)
        flow_prev3[3:] = flow[:-3]
        trend = (wt - wt_prev6) / 6.0
        dlogq = np.log1p(flow) - np.log1p(flow_prev3)
        diseq = temp - wt
        anomaly = wt - clim.predict_dates(site, g["DATE"])
        stat: dict[str, dict] = {
            "anomaly": {"q10": float(np.nanquantile(anomaly, 0.10)),
                        "q90": float(np.nanquantile(anomaly, 0.90))},
            "trend7": {"q10": float(np.nanquantile(trend, 0.10)),
                       "q90": float(np.nanquantile(trend, 0.90))},
            "disequilibrium": {"q10": float(np.nanquantile(diseq, 0.10)),
                               "q90": float(np.nanquantile(diseq, 0.90))},
            "flow_change": {"q10": float(np.nanquantile(dlogq, 0.10)),
                            "q90": float(np.nanquantile(dlogq, 0.90))},
            "month_flow": {},
            "dT": {},
            "target_wt": {},
        }
        months = pd.to_datetime(g["DATE"]).dt.month.to_numpy()
        for month in range(1, 13):
            values = flow[months == month]
            values = values[np.isfinite(values)]
            if len(values) >= 10:
                stat["month_flow"][month] = {
                    "q10": float(np.quantile(values, 0.10)),
                    "q90": float(np.quantile(values, 0.90)),
                }
        for horizon in (1, 3, 7):
            wt_future = np.full(len(wt), np.nan)
            wt_future[:-horizon] = wt[horizon:]
            dT = wt_future - wt
            stat["dT"][horizon] = {
                "q10": float(np.nanquantile(dT, 0.10)),
                "q90": float(np.nanquantile(dT, 0.90)),
            }
            stat["target_wt"][horizon] = {
                "q10": float(np.nanquantile(wt, 0.10)),
                "q90": float(np.nanquantile(wt, 0.90)),
            }
        out[site] = stat
    return out


def build_issue_features(dev: pd.DataFrame, hold: pd.DataFrame,
                         clim: F.HarmonicClimatology) -> pd.DataFrame:
    """Per-day issue-time covariates on the combined raw panel (no imputation).

    Columns: site_id, DATE (issue date), anomaly, trend7, diseq, dlogq3,
    month, FLOW.  All values are dated no later than the issue date.
    """
    panel = pd.concat([dev, hold], ignore_index=True)
    panel = panel.drop_duplicates(["site_id", "DATE"], keep="last")
    panel = panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)
    for column in ("WTEMP", "TEMP", "FLOW"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["clim"] = np.concatenate([
        clim.predict_dates(site, sub["DATE"]).astype(float)
        for site, sub in panel.groupby("site_id", sort=True)
    ])
    panel = panel.sort_values(["site_id", "DATE"])
    per_site = panel.groupby("site_id", sort=True)
    panel["anomaly"] = panel["WTEMP"] - panel["clim"]
    panel["wt_prev6"] = per_site["WTEMP"].shift(6)
    panel["flow_prev3"] = per_site["FLOW"].shift(3)
    panel["trend7"] = (panel["WTEMP"] - panel["wt_prev6"]) / 6.0
    panel["diseq"] = panel["TEMP"] - panel["WTEMP"]
    panel["dlogq3"] = np.log1p(panel["FLOW"]) - np.log1p(panel["flow_prev3"])
    panel["month"] = panel["DATE"].dt.month
    return panel[["site_id", "DATE", "anomaly", "trend7", "diseq", "dlogq3",
                  "month", "FLOW"]].rename(columns={"DATE": "issue_date"})


def state_of(row: tuple, sq: dict) -> list[str]:
    """Issue-time state labels for one key."""
    states: list[str] = []
    anomaly = row.anomaly
    if np.isfinite(anomaly) and np.isfinite(sq["anomaly"]["q10"]):
        states.append("issue_low_anomaly" if anomaly <= sq["anomaly"]["q10"]
                      else "issue_high_anomaly" if anomaly >= sq["anomaly"]["q90"]
                      else "issue_normal_anomaly")
    trend = row.trend7
    if np.isfinite(trend) and np.isfinite(sq["trend7"]["q10"]):
        states.append("issue_rapid_recent_warming" if trend >= sq["trend7"]["q90"]
                      else "issue_rapid_recent_cooling" if trend <= sq["trend7"]["q10"]
                      else "issue_stable_trend")
    diseq = row.diseq
    if np.isfinite(diseq) and np.isfinite(sq["disequilibrium"]["q10"]):
        states.append("issue_strong_warming_pressure" if diseq >= sq["disequilibrium"]["q90"]
                      else "issue_strong_cooling_pressure" if diseq <= sq["disequilibrium"]["q10"]
                      else "issue_near_equilibrium")
    flow = row.FLOW
    mq = sq["month_flow"].get(row.month) if sq["month_flow"] else None
    if np.isfinite(flow) and mq is not None:
        states.append("issue_high_flow" if flow >= mq["q90"]
                      else "issue_low_flow" if flow <= mq["q10"]
                      else "issue_normal_flow")
    dlogq = row.dlogq3
    if np.isfinite(dlogq) and np.isfinite(sq["flow_change"]["q10"]):
        states.append("issue_rapid_flow_rise" if dlogq >= sq["flow_change"]["q90"]
                      else "issue_rapid_flow_recession" if dlogq <= sq["flow_change"]["q10"]
                      else "issue_stable_flow")
    return states


def outcome_state_of(row: tuple, sq: dict) -> list[str]:
    """Retrospective outcome-conditioned state labels (secondary)."""
    states: list[str] = []
    horizon = int(row.horizon)
    dq = sq["dT"].get(horizon, {})
    wq = sq["target_wt"].get(horizon, {})
    dT = row.dT_actual
    if np.isfinite(dT) and np.isfinite(dq.get("q10")):
        if dT >= dq["q90"]:
            states.append("actual_rapid_warming")
        elif dT <= dq["q10"]:
            states.append("actual_rapid_cooling")
    wt_t = row.WT_t
    if np.isfinite(wt_t) and np.isfinite(wq.get("q10")):
        if wt_t >= wq["q90"]:
            states.append("actual_warmest_decile")
        elif wt_t <= wq["q10"]:
            states.append("actual_coldest_decile")
    mq = sq["month_flow"].get(row.month) if sq["month_flow"] else None
    if np.isfinite(row.FLOW_t) and mq is not None:
        if row.FLOW_t >= mq["q90"]:
            states.append("actual_high_flow")
        elif row.FLOW_t <= mq["q10"]:
            states.append("actual_low_flow")
    return states


def main() -> int:
    dev, hold, registry = load_panels()
    stations = tuple(registry["site_no"].astype(str).str.zfill(8))
    C.STATIONS = stations  # per-station climatology coefficients keyed by site_no

    print("fitting official climatology + damped-persistence anchor...", flush=True)
    clim = official_climatology(dev, registry)
    half_life = fit_official_half_life(dev, registry, clim)
    print("fitting training-period station thresholds...", flush=True)
    q = station_training_quantiles(dev, clim, stations)

    print("building issue-time features...", flush=True)
    feat = build_issue_features(dev, hold, clim)

    pred = pd.read_parquet(FINAL / "predictions.parquet")
    pred = pred[pred.model.isin([CANDIDATE, REFERENCE, "Persistence"])].copy()
    pred["issue_date"] = pd.to_datetime(pred["issue_date"])
    pred["site_id"] = pred["site_id"].astype(str).str.zfill(8)
    keys = pred.merge(feat, on=["site_id", "issue_date"], how="left")

    target_panel = hold.sort_values(["site_id", "DATE"])
    tp = target_panel[["site_id", "DATE", "WTEMP", "FLOW"]].copy()
    tp["WTEMP"] = pd.to_numeric(tp["WTEMP"], errors="coerce")
    tp["FLOW"] = pd.to_numeric(tp["FLOW"], errors="coerce")
    keys = keys.merge(
        tp.rename(columns={"DATE": "target_date", "WTEMP": "WT_t", "FLOW": "FLOW_t"}),
        on=["site_id", "target_date"], how="left", validate="many_to_one")
    keys = keys.merge(
        tp.rename(columns={"DATE": "issue_date", "WTEMP": "WT_i"})[
            ["site_id", "issue_date", "WT_i"]],
        on=["site_id", "issue_date"], how="left", validate="many_to_one")
    keys["dT_actual"] = keys["WT_t"] - keys["WT_i"]
    keys["season"] = keys["issue_date"].dt.month.map(SEASON_BY_MONTH)

    print("classifying keys into states...", flush=True)
    records = keys[["site_id", "horizon", "issue_date", "anomaly", "trend7",
                    "diseq", "dlogq3", "month", "FLOW", "WT_t", "FLOW_t",
                    "dT_actual"]].to_records(index=False)
    issue_labels: list[str] = []
    outcome_labels: list[str] = []
    for record in records:
        sq = q.get(record.site_id)
        if sq is None:
            issue_labels.append("")
            outcome_labels.append("")
            continue
        issue_labels.append("|".join(state_of(record, sq)))
        outcome_labels.append("|".join(outcome_state_of(record, sq)))
    keys["issue_state"] = issue_labels
    keys["outcome_state"] = outcome_labels

    # --- station-first state metrics ---
    print("computing station-first state metrics...", flush=True)
    cand = keys[keys.model == CANDIDATE][
        ["site_id", "horizon", "issue_date", "y_pred", "y_true"]].rename(
        columns={"y_pred": "cand"})
    ref = keys[keys.model == REFERENCE][
        ["site_id", "horizon", "issue_date", "y_pred"]].rename(
        columns={"y_pred": "ref"})
    both = cand.merge(ref, on=["site_id", "horizon", "issue_date"],
                      validate="one_to_one")
    both = both.merge(
        keys[keys.model == CANDIDATE][["site_id", "horizon", "issue_date",
                                       "issue_state", "outcome_state", "season"]],
        on=["site_id", "horizon", "issue_date"], validate="one_to_one")
    both["state_concat"] = (both["issue_state"].fillna("") + "|"
                            + both["outcome_state"].fillna("") + "|"
                            + both["season"].fillna(""))

    state_names = sorted({
        state for states in both["state_concat"]
        for state in states.split("|") if state
    })

    metric_rows: list[dict] = []
    for state in state_names:
        pattern = f"(^|\\|){state}($|\\|)"
        sub = both[both["state_concat"].str.contains(pattern, regex=True)]
        if sub.empty:
            continue
        for (site, horizon), g in sub.groupby(["site_id", "horizon"]):
            if len(g) < MIN_STATE_KEYS:
                continue
            e_c = g["cand"].to_numpy(float) - g["y_true"].to_numpy(float)
            e_r = g["ref"].to_numpy(float) - g["y_true"].to_numpy(float)
            metric_rows.append({
                "state_name": state,
                "state_type": ("issue_time" if state.startswith("issue_")
                               else "outcome_conditioned"
                               if state.startswith("actual_")
                               else "season"),
                "threshold_source": ("training_station_specific"
                                     if state.startswith(("issue_", "actual_"))
                                     else "calendar_known"),
                "site_id": site,
                "horizon": int(horizon),
                "n_keys": int(len(g)),
                "candidate_rmse": float(np.sqrt(np.mean(e_c ** 2))),
                "reference_rmse": float(np.sqrt(np.mean(e_r ** 2))),
                "delta_rmse": float(np.sqrt(np.mean(e_c ** 2))
                                    - np.sqrt(np.mean(e_r ** 2))),
            })
    state_effects = pd.DataFrame(metric_rows)
    state_effects.to_parquet(FINAL / "hydrologic_state_effects.parquet", index=False)

    # --- basin attributes + descriptive summary ---
    print("writing basin attributes and summary...", flush=True)
    hl = pd.DataFrame([
        {"site_id": site, "phi": info["phi"],
         "half_life_days": info["half_life_days"], "capped": info["capped"]}
        for site, info in half_life.items()
    ])
    reg = registry[["site_no", "lat", "lon", "drain_area_va", "huc2"]].rename(
        columns={"site_no": "site_id"})
    basin = reg.merge(hl, on="site_id", how="left")
    temp_mean = dev.groupby("site_id")["TEMP"].mean().rename("mean_annual_air_temp_c")
    basin = basin.merge(temp_mean, on="site_id", how="left")
    basin.to_parquet(FINAL / "basin_attributes.parquet", index=False)

    decomp = pd.read_parquet(FINAL / "decomposition_effects.parquet")
    decomp = decomp[decomp.model == CANDIDATE]
    merged = decomp.merge(
        basin[["site_id", "half_life_days", "drain_area_va", "lat"]],
        on="site_id", how="left")

    summary: dict = {
        "half_life": {
            "n_fit": int(hl["half_life_days"].notna().sum()),
            "median_days": float(np.nanmedian(hl["half_life_days"])),
            "iqr_days": list(np.nanpercentile(hl["half_life_days"], [25, 75])),
            "n_capped": int(hl["capped"].sum()),
            "terminology": ("anomaly half-life t_1/2 = ln(0.5)/ln(phi) from the "
                            "official train-fitted damped-persistence anchor"),
        },
        "learned_gain_correlations": {},
        "state_summaries": {},
    }
    for horizon in (1, 3, 7):
        g = merged[merged.horizon == horizon].dropna(
            subset=["g_learned", "half_life_days"])
        if len(g) > 10:
            corr = {"n": int(len(g)),
                    "pearson_log_half_life": float(np.corrcoef(
                        np.log(g["half_life_days"]), g["g_learned"])[0, 1])}
            da = g.dropna(subset=["drain_area_va"])
            corr["pearson_log_drain_area"] = (
                float(np.corrcoef(np.log(da["drain_area_va"].clip(lower=1)),
                                  da["g_learned"])[0, 1]) if len(da) > 10
                else float("nan"))
            corr["n_drain_area"] = int(len(da))
            corr["pearson_latitude"] = float(np.corrcoef(g["lat"], g["g_learned"])[0, 1])
            summary["learned_gain_correlations"][str(horizon)] = corr
        for state_type in ("issue_time", "outcome_conditioned", "season"):
            s = state_effects[(state_effects.horizon == horizon)
                              & (state_effects.state_type == state_type)]
            block: dict = {}
            for state in sorted(s["state_name"].unique()):
                cell = s[s["state_name"] == state]
                if cell["site_id"].nunique() >= 3:
                    block[state] = {
                        "n_stations": int(cell["site_id"].nunique()),
                        "median_delta_rmse": float(cell["delta_rmse"].median()),
                        "median_n_keys": int(cell["n_keys"].median()),
                    }
            summary["state_summaries"].setdefault(state_type, {})[str(horizon)] = block
    (FINAL / "mechanism_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary["half_life"], indent=1))
    print(json.dumps(summary["learned_gain_correlations"], indent=1))
    print(f"state rows: {len(state_effects)}; states: {state_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
