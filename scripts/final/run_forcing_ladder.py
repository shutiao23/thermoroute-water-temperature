#!/usr/bin/env python3
"""Forcing ladder, protocol v3 (DLOG-015) — the F-axis.

Answers the advisory review's D1 finding: the F0 information set (no
meteorological information beyond the issue date) makes damped persistence an
approximately Bayes-optimal anomaly predictor, so the learned-gain null is a
design consequence.  This runner varies the FORCING information while holding
everything else fixed:

    F0  no future meteorological forcing (status quo ante)
    F1  climatological future forcing (per-station harmonic climatology)
    F3  perfect future forcing (observed panel values; climatology fill where
        the panel ends, counted and reported)

Models: station-agnostic LightGBM and ResidualLightGBM (frozen per-lead
hyperparameters and best iterations of protocol v1/v2), plus the
air2stream-style hybrid under F1 (its published roll-out already uses the
climatology for steps 2..h — the existing authority table) and under F3
(observed air temperature at every roll-out step).

Scoring is on the common forecast-key registry with the SAME two-sided
assertion as the L-axis runner: every arm scores exactly the registry subset,
and no cross-arm comparison is ever made on different key sets.  The deep
architecture gains an F3 arm via its sequence-input channels; that retraining
is scheduled after the tree arms establish the direction (protocol v3
stopping rule; P-5 is evaluated on the best tree model as the interim signal).

Usage::

    python scripts/final/run_forcing_ladder.py [--arms F0,F1,F3]
        [--models LightGBM,ResidualLightGBM,air2stream]
        [--leads 1,3,7] [--max-cells N]

Outputs: ``outputs/final/forcing_effects.parquet`` (per-site metrics per
arm/model/lead) and ``outputs/final/forcing_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "final"))

from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import features as F  # noqa: E402
from thermoroute.baselines import _lgb_fit  # noqa: E402
from run_information_ladder import (  # noqa: E402
    BEST_ITER, FROZEN_PARAMS, USGS_VARS, build_features, fit_preprocessing,
    load_panel, reference_keys,
)

FINAL = ROOT / "outputs" / "final"
OUT_CONV = ROOT / "outputs" / "conventional"

META_VARS = ("TEMP", "PRCP", "DH", "RHMEAN", "WDSP")
MAX_FUTURE_DAYS = 7

ARMS = ("F0", "F1", "F3")


def future_met_columns(panel: pd.DataFrame, clim: F.HarmonicClimatology,
                       arm: str) -> pd.DataFrame:
    """Per (site, DATE) future-met table: ``{var}_fut{k}`` for k=1..7.

    F3 takes observed panel values (climatology fill where the panel ends);
    F1 takes the climatology everywhere; F0 has no columns.  The result is
    keyed for a merge on (site_id, issue_date).
    """
    if arm == "F0":
        return None
    rows = []
    panel = panel.sort_values(["site_id", "DATE"])
    for site, sub in panel.groupby("site_id", sort=True):
        dates = pd.to_datetime(sub["DATE"].to_numpy())
        doy = dates.dayofyear
        out = {"site_id": site, "DATE": dates}
        doy_np = doy.to_numpy()
        for var in META_VARS:
            s = sub[var].to_numpy(float)
            clim_vals = clim.predict(site, doy_np)
            for k in range(1, MAX_FUTURE_DAYS + 1):
                fut = np.full(len(sub), np.nan)
                if k < len(sub):
                    fut[:-k] = s[k:]
                if arm == "F3":
                    # perfect forcing; climatology where the panel ends
                    # (only the last k panel days are affected, and those
                    # rows are excluded by the registry join for scoring)
                    missing = ~np.isfinite(fut)
                    if np.any(missing):
                        idx = np.minimum(np.where(missing)[0] + k,
                                         len(clim_vals) - 1)
                        fut[missing] = clim_vals[idx]
                elif arm == "F1":
                    fut = np.roll(clim_vals, -k)
                    if k < len(sub):
                        fut[-k:] = np.nan
                out[f"{var}_fut{k}"] = fut
        rows.append(pd.DataFrame(out))
    table = pd.concat(rows, ignore_index=True)
    table["site_id"] = table["site_id"].astype(str).str.zfill(8)
    table["DATE"] = pd.to_datetime(table["DATE"])
    return table


def build_arm_features(tab: pd.DataFrame, fut: pd.DataFrame | None,
                       h: int) -> tuple[pd.DataFrame, list[str]]:
    """Attach the lead-consistent future columns to the tabular frame.

    Returns (tab, added_cols); the caller MUST extend the model's feature
    list with ``added_cols``, otherwise the forcing features never reach the
    model and every arm is silently F0.
    """
    if fut is None:
        return tab, []
    added: list[str] = []
    for var in META_VARS:
        parts = [f"{var}_fut{k}" for k in range(1, h + 1)]
        merge_cols = parts + ["site_id", "DATE"]
        tab = tab.merge(fut[merge_cols], left_on=["site_id", "issue_date"],
                        right_on=["site_id", "DATE"], how="left",
                        suffixes=("", "_drop"))
        tab = tab.drop(columns=["DATE"])
        mean_h = f"{var}_futmean{h}"
        tab[mean_h] = tab[parts].mean(axis=1, skipna=False)
        added.append(f"{var}_fut{h}")
        added.append(mean_h)
    return tab, added




def _fit_a2s_station(args):
    """Picklable worker: fit one station on its training rows."""
    st, sub = args
    from thermoroute.air2stream import fit as a2s_fit  # noqa: PLC0415
    Ta = sub["TEMP"].to_numpy(float)
    Q = sub["FLOW"].to_numpy(float)
    T = sub["WTEMP"].to_numpy(float)
    doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
    ta_obs = sub["TEMP_observed"].to_numpy(bool) \
        if "TEMP_observed" in sub else ~np.isnan(Ta)
    flow_obs = sub["FLOW_observed"].to_numpy(bool) \
        if "FLOW_observed" in sub else ~np.isnan(Q)
    wt_obs = sub["WTEMP_observed"].to_numpy(bool) \
        if "WTEMP_observed" in sub else ~np.isnan(T)
    fit_obj = a2s_fit(np.where(ta_obs, Ta, np.nan),
                      np.where(flow_obs, Q, np.nan), T, doy,
                      variant="a8", obs=wt_obs)
    return st, fit_obj


def run_a2s_arm(panel: pd.DataFrame, fit_mask: np.ndarray,
                clim: F.HarmonicClimatology, stations: tuple[str, ...],
                arm: str, ref: pd.DataFrame) -> list[dict]:
    """air2stream-style hybrid under F1/F3 forcing.

    Fits are parallelized across stations (multiprocessing, as in the
    authority pipeline) and identical for both arms; only the roll-out
    forcing differs.  Scoring is on the COMMON REGISTRY keys only, so the
    arm is comparable with the F1 authority table and with the tree arms.
    """
    import multiprocessing as mp  # noqa: PLC0415
    from thermoroute.air2stream import forecast_horizon  # noqa: PLC0415
    from thermoroute.air2stream import fit as a2s_fit  # noqa: PLC0415

    train_rows = np.asarray(fit_mask, dtype=bool)
    sub_frames = []
    for st in stations:
        mask = (panel.site_id.eq(st).to_numpy() & train_rows)
        sub = panel.loc[mask].copy().sort_values("DATE").reset_index(drop=True)
        if sub.empty:
            continue
        sub_frames.append((st, sub))
    with mp.Pool(min(32, len(sub_frames))) as pool:
        fits = dict(pool.imap_unordered(_fit_a2s_station, sub_frames))
    print(f"  fitted {len(fits)} air2stream stations (parallel)", flush=True)

    holdout = panel[~train_rows].copy()
    holdout["doy"] = pd.to_datetime(holdout["DATE"]).dt.dayofyear
    ta_obs_col = (holdout["TEMP_observed"].astype(bool).to_numpy()
                  if "TEMP_observed" in holdout else holdout["TEMP"].notna().to_numpy())
    # per-station O(1) lookup maps: DATE -> (row index, TEMP, WTEMP, FLOW, doy)
    station_maps: dict[str, dict] = {}
    for st, sub in holdout.groupby("site_id"):
        dates = pd.to_datetime(sub["DATE"])
        station_maps[st] = {
            d: (int(sub.index[i]), float(sub["TEMP"].iloc[i]),
                float(sub["WTEMP"].iloc[i]), float(sub["FLOW"].iloc[i]),
                int(dates.iloc[i].dayofyear),
                bool(ta_obs_col[sub.index[i]]))
            for i, d in enumerate(dates)}
    out: list[dict] = []
    for st, g in ref.groupby("site_id"):
        if st not in fits or st not in station_maps:
            continue
        smap = station_maps[st]
        for h in (1, 3, 7):
            kg = g[g.horizon == h]
            if kg.empty:
                continue
            for _, k in kg.iterrows():
                issue = k["issue_date"]
                rec = smap.get(issue)
                if rec is None:
                    continue
                _, T0, Q0, _, doy0, _ = rec
                if not np.isfinite(T0) or not np.isfinite(Q0) or Q0 <= 0:
                    continue
                fut_doy = np.array([int((issue + pd.Timedelta(days=j + 1)).dayofyear)
                                    for j in range(h)])
                if arm == "F3":
                    ta_f = []
                    for j in range(1, h + 1):
                        d = issue + pd.Timedelta(days=j)
                        r = smap.get(d)
                        if r is not None and r[5]:
                            ta_f.append(r[1])
                        else:
                            ta_f.append(float(clim.predict(st, np.array([int(d.dayofyear)]))[0]))
                else:
                    ta_f = [float(clim.predict(st, np.array([d]))[0]) for d in fut_doy]
                yhat = forecast_horizon(fits[st], T0, Q0, doy0,
                                        np.asarray(ta_f, dtype=float),
                                        fut_doy.astype(float))
                out.append({"site_id": st, "horizon": h, "issue_date": issue,
                            "y_pred": yhat, "y_true": k["y_true"],
                            "arm": arm, "model": "air2stream"})
    rows: list[dict] = []
    df = pd.DataFrame(out)
    if df.empty:
        return rows
    for (site, h), g in df.groupby(["site_id", "horizon"]):
        e = g.dropna(subset=["y_pred", "y_true"])
        if e.empty:
            continue
        err = (e.y_pred - e.y_true).to_numpy(float)
        rows.append({"arm": arm, "model": "air2stream", "horizon": int(h),
                     "site_id": site, "n": len(e),
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "rmse_damped": float("nan")})
    return rows




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="F0,F1,F3")
    ap.add_argument("--models", default="LightGBM,ResidualLightGBM")
    ap.add_argument("--leads", default="1,3,7")
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    models = [m for m in args.models.split(",") if m]
    leads = [int(h) for h in args.leads.split(",") if h]

    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    stations = tuple(sorted(registry.site_no.astype(str).str.zfill(8)))
    C.STATIONS = stations
    panel = load_panel(registry)
    dates = panel["DATE"].to_numpy()
    fit_mask = dates <= np.datetime64("2015-12-31")
    val_mask = (dates > np.datetime64("2015-12-31")) & (dates <= np.datetime64("2017-12-31"))
    fit_mask = fit_mask | val_mask
    ref = reference_keys()

    imputer, clim, anchor = fit_preprocessing("L0", panel, fit_mask,
                                              list(stations), stations)
    panel_imp = imputer.transform(panel)
    phi = {site: float(anchor.phi.get(site, 0.9)) for site in stations}
    fut_cache: dict[str, pd.DataFrame | None] = {
        arm: future_met_columns(panel, clim, arm) for arm in arms}

    metric_rows = []
    summary: dict = {"arms": arms, "models": models, "leads": leads,
                     "p5_interim_model": "best tree (LightGBM)"}
    for arm in arms:
        for h in leads:
            tab, cols = build_features(panel_imp, panel_imp, imputer, clim, h,
                                       [], "L0")
            tab, fut_cols = build_arm_features(tab, fut_cache[arm], h)
            if fut_cols:
                # the forcing features MUST enter the model's feature list
                cols = cols + fut_cols
            for model in models:
                if args.max_cells and len(metric_rows) >= args.max_cells:
                    break
                # whole-cohort fit: train on all stations' admissible rows
                tr = tab[tab.split.isin(["train", "val"])
                         & tab["issue_wtemp_observed"].astype(bool)]
                if model == "ResidualLightGBM":
                    ytr = (tr["y"].to_numpy(float)
                           - tr["clim_target"].to_numpy(float)
                           - tr["site_id"].map(phi).to_numpy(float) ** h
                           * (tr["persistence"].to_numpy(float)
                              - tr["clim_t"].to_numpy(float)))
                else:
                    ytr = tr["y"].to_numpy(float)
                params = {"num_leaves": FROZEN_PARAMS[h]["num_leaves"],
                          "min_child_samples": FROZEN_PARAMS[h]["min_child_samples"],
                          "learning_rate": FROZEN_PARAMS[h]["learning_rate"]}
                m = _lgb_fit(tr[cols], ytr,
                             tr[cols].to_numpy(float)[-min(2000, len(tr)):],
                             ytr[-min(2000, len(ytr)):],
                             "regression", n_est=BEST_ITER[h],
                             params_override=params)
                ev = tab[tab.split.eq("confirm")]
                ref_h = ref[(ref.horizon == h)][["site_id", "issue_date"]]
                ev = ev.merge(ref_h, on=["site_id", "issue_date"], how="inner")
                ref_sub = set(ref_h.itertuples(index=False, name=None))
                ev_sub = set(ev[["site_id", "issue_date"]]
                             .itertuples(index=False, name=None))
                if ref_sub - ev_sub or ev_sub - ref_sub:
                    raise AssertionError(
                        f"registry breach {arm}/{model}/h{h}: "
                        f"{len(ref_sub - ev_sub)} missing, "
                        f"{len(ev_sub - ref_sub)} extra keys")
                yhat = m.predict(ev[cols], num_threads=1)
                y_damped = (ev["clim_target"].to_numpy(float)
                            + ev["site_id"].map(phi).to_numpy(float) ** h
                            * (ev["persistence"].to_numpy(float)
                               - ev["clim_t"].to_numpy(float)))
                if model == "ResidualLightGBM":
                    yhat = yhat + y_damped
                frame = pd.DataFrame({
                    "site_id": ev.site_id.to_numpy(),
                    "issue_date": ev.issue_date.to_numpy(),
                    "horizon": h, "y_pred": yhat, "y_true": ev["y"].to_numpy(),
                    "y_damped": y_damped,
                    "arm": arm, "model": model,
                })
                for site, g in frame.groupby("site_id"):
                    e = g.dropna(subset=["y_pred", "y_damped", "y_true"])
                    if e.empty:
                        continue
                    err = (e.y_pred - e.y_true).to_numpy(float)
                    errd = (e.y_damped - e.y_true).to_numpy(float)
                    metric_rows.append({
                        "arm": arm, "model": model, "horizon": h,
                        "site_id": site, "n": len(e),
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "rmse_damped": float(np.sqrt(np.mean(errd ** 2))),
                    })
                per_site = frame.groupby("site_id").apply(
                    lambda g: float(np.sqrt(np.mean(
                        (g["y_pred"] - g["y_true"]) ** 2))))
                print(f"{arm}/{model}/h{h}: {len(frame)} keys, "
                      f"station-median RMSE {per_site.median():.3f}",
                      flush=True)
    if "air2stream" in models:
        for arm in arms:
            if arm == "F0":
                continue
            metric_rows.extend(run_a2s_arm(panel, fit_mask, clim, stations,
                                           arm, ref))
            print(f"{arm}/air2stream: scored {arm}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    effects_path = FINAL / "forcing_effects.parquet"
    if effects_path.exists():
        prev = pd.read_parquet(effects_path)
        key_cols = ["arm", "model", "horizon", "site_id"]
        if prev.duplicated(key_cols).any() or len(prev):
            metrics = (pd.concat([prev, metrics], ignore_index=True)
                       .drop_duplicates(key_cols, keep="last"))
    metrics.to_parquet(effects_path, index=False)

    # station-median summaries + P-5/P-6/P-7 ingredients
    med = (metrics.groupby(["arm", "model", "horizon"])["rmse"]
           .median().reset_index().pivot_table(
               index=["model", "horizon"], columns="arm", values="rmse"))
    if "F0" in med.columns and "F3" in med.columns:
        med["F3_minus_F0"] = med["F3"] - med["F0"]
    if "F0" in med.columns and "F1" in med.columns:
        med["F1_minus_F0"] = med["F1"] - med["F0"]
    def jsonable(table: pd.DataFrame) -> dict:
        return {f"{row[0]}/h{row[1]}": {str(c): (None if pd.isna(v) else round(float(v), 4))
                for c, v in table.loc[row].items()}
                for row in table.index}
    summary["station_median_rmse"] = jsonable(med)
    damped_med = (metrics.groupby(["arm", "model", "horizon"])["rmse_damped"]
                  .median().reset_index().pivot_table(
                      index=["model", "horizon"], columns="arm",
                      values="rmse_damped"))
    summary["station_median_rmse_damped"] = jsonable(damped_med)
    (FINAL / "forcing_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"wrote {FINAL / 'forcing_effects.parquet'} "
          f"({len(metrics)} station-cells) and forcing_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
