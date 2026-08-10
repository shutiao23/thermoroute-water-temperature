#!/usr/bin/env python3
"""Matched spatial-transfer experiment on the independent 2021-2023 window.

Fits a station-agnostic LightGBM (frozen per-horizon hyperparameters and
seed 0, point head) on four deterministic HUC2-region folds and on four
balanced random folds, using 2006-2017 train/validation rows from the
in-fold stations only, and scores each held-out station on its own
2021-2023 common forecast keys.  Preprocessing (imputation, climatology,
per-station) is identical in both arms and identical for held and training
stations (local history is an allowed issue-time input, so this is gauged
regional transfer, not ungauged prediction).

The arm contrast (region vs random) is therefore attributable to the spatial
partition alone.

Outputs:
  outputs/conventional/region_transfer_2021_2023.parquet   (held-site preds)
  outputs/conventional/region_transfer_metrics_2021_2023.csv
  outputs/conventional/region_transfer_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import features as F  # noqa: E402
from thermoroute.baselines import _lgb_fit  # noqa: E402

OUT = ROOT / "outputs" / "conventional"
PRED = OUT / "predictions_2021_2023.parquet"

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")

# Frozen hyperparameters selected on 2016-2017 validation by the main design.
FROZEN_PARAMS = {1: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 3: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 7: {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.03}}
BEST_ITER = {1: 800, 3: 575, 7: 440}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(a))


def region_folds(registry: pd.DataFrame) -> list[tuple[list[str], list[str]]]:
    by_reg: dict[str, list[str]] = {}
    for _, r in registry.iterrows():
        by_reg.setdefault(str(r.huc2), []).append(str(r.site_no).zfill(8))
    folds: list[tuple[list[str], list[str]]] = []
    buckets = [[], [], [], []]
    loads = [0, 0, 0, 0]
    for reg, sts in sorted(by_reg.items(), key=lambda kv: -len(kv[1])):
        i = int(np.argmin(loads))
        buckets[i].extend(sts)
        loads[i] += len(sts)
    for i in range(4):
        hold = sorted(buckets[i])
        train = sorted(set(registry.site_no.astype(str).str.zfill(8)) - set(hold))
        folds.append((train, hold))
    return folds


def random_folds(registry: pd.DataFrame, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    sites = sorted(registry.site_no.astype(str).str.zfill(8))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sites))
    folds = []
    for i in range(4):
        hold = [sites[j] for j in perm[i::4]]
        train = sorted(set(sites) - set(hold))
        folds.append((train, sorted(hold)))
    return folds


def main() -> None:
    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    stations = tuple(sorted(registry.site_no.astype(str).str.zfill(8)))
    C.STATIONS = stations  # build_tabular iterates C.STATIONS

    train_panel = pd.read_parquet(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")
    train_panel["DATE"] = pd.to_datetime(train_panel["DATE"])
    test_panel = pd.read_parquet(OUT / "panel_2021_2023.parquet")
    test_panel["DATE"] = pd.to_datetime(test_panel["DATE"])
    alias = dict(zip(registry.legacy_site_id.astype(str),
                     registry.site_no.astype(str).str.zfill(8)))
    train_panel["site_id"] = train_panel["site_id"].astype(str).map(alias)
    for df in (train_panel, test_panel):
        df["site_id"] = df["site_id"].astype(str).str.zfill(8)
    panel = pd.concat([train_panel, test_panel], ignore_index=True)
    panel = panel.drop_duplicates(subset=["DATE", "site_id"], keep="last")
    panel = panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)

    dates = panel["DATE"].to_numpy()
    train_mask = dates <= np.datetime64("2020-12-31")
    imputer = D.Imputer.fit(panel.loc[train_mask], train_mask[train_mask])
    panel_imp = imputer.transform(panel)
    clim = F.HarmonicClimatology.fit(panel, train_mask, fit_stations=stations)

    ref = pd.read_parquet(PRED, columns=["site_id", "horizon", "issue_date", "y_true"])
    ref["site_id"] = ref["site_id"].astype(str).str.zfill(8)
    ref["issue_date"] = pd.to_datetime(ref["issue_date"])

    dist_map: dict[str, dict[str, float]] = {}
    for _, r in registry.iterrows():
        s = str(r.site_no).zfill(8)
        dist_map[s] = {"lat": float(r.lat), "lon": float(r.lon)}

    rows: list[dict] = []
    arms = [("region", region_folds(registry)), ("random", random_folds(registry))]
    tabs = {}
    for h in (1, 3, 7):
        tab = F.attach_split(F.build_tabular(
            panel_imp, h, USGS_VARS, clim,
            drop_feature_nans=False, require_observed_target=True,
            include_missingness=True))
        tab.loc[tab.issue_date >= np.datetime64("2021-01-01"), "split"] = "confirm"
        cols = F.feature_columns(tab)
        for c in cols:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
        tabs[h] = (tab, cols)
    out_path = OUT / "region_transfer_2021_2023.parquet"
    done: set[tuple[str, int, int]] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        for _, r in existing[["arm", "fold", "horizon"]].drop_duplicates().iterrows():
            done.add((str(r.arm), int(r.fold), int(r.horizon)))
        print(f"resume: {len(done)} cells cached", flush=True)
    for arm, folds in arms:
        for fold_i, (train_st, hold_st) in enumerate(folds):
            print(f"[{arm} fold{fold_i}] train {len(train_st)} hold {len(hold_st)}", flush=True)
            for h in (1, 3, 7):
                if (arm, fold_i, h) in done:
                    print(f"    h{h}: cached", flush=True)
                    continue
                tab, cols = tabs[h]
                tr = tab[tab.split.isin(["train", "val"]) & tab.site_id.isin(train_st)]
                Xtr, ytr = tr[cols], tr["y"].to_numpy(float)
                wtr = D.station_equal_sample_weight(tr["site_id"]) if hasattr(D, "station_equal_sample_weight") else None
                try:
                    from thermoroute.baselines import station_equal_sample_weight
                    wtr = station_equal_sample_weight(tr["site_id"])
                except Exception:
                    wtr = None
                if len(tr) < 500:
                    print(f"    h{h}: skip, only {len(tr)} train rows")
                    continue
                params = {"num_leaves": FROZEN_PARAMS[h]["num_leaves"],
                          "min_child_samples": FROZEN_PARAMS[h]["min_child_samples"],
                          "learning_rate": FROZEN_PARAMS[h]["learning_rate"]}
                m = _lgb_fit(Xtr, ytr, Xtr[-min(2000, len(Xtr)):], ytr[-min(2000, len(ytr)):],
                             "regression", n_est=BEST_ITER[h], params_override=params,
                             sample_weight=wtr)
                ev = tab[tab.split.eq("confirm") & tab.site_id.isin(hold_st)]
                if ev.empty:
                    continue
                yhat = m.predict(ev[cols], num_threads=1)
                out = pd.DataFrame({
                    "site_id": ev.site_id.to_numpy(), "horizon": h,
                    "issue_date": ev.issue_date.to_numpy(),
                    "target_date": ev.target_date.to_numpy(),
                    "y_pred": yhat, "y_true": ev["y"].to_numpy(),
                    "arm": arm, "fold": fold_i})
                rows.append(out)
                pred = pd.concat(rows, ignore_index=True)
                pred.to_parquet(out_path)
                print(f"    h{h}: {len(out)} held-site keys, RMSE "
                      f"{np.sqrt(np.mean((yhat-out.y_true)**2)):.3f}", flush=True)

    if not rows:
        raise SystemExit("no predictions produced")

    damped = pd.read_parquet(PRED, columns=["model", "site_id", "horizon", "issue_date", "y_pred"])
    damped = damped[damped.model == "DampedPersistence"].drop(columns=["model"])
    damped["site_id"] = damped["site_id"].astype(str).str.zfill(8)
    damped["issue_date"] = pd.to_datetime(damped["issue_date"])
    joined = pred.merge(damped, on=["site_id", "horizon", "issue_date"], how="left",
                        suffixes=("", "_damped"))

    metric_rows = []
    for (arm, st, h), g in joined.groupby(["arm", "site_id", "horizon"]):
        m = g.y_pred.notna() & g.y_pred_damped.notna()
        if not m.any():
            continue
        e = g.loc[m]
        metric_rows.append({
            "arm": arm, "site_id": st, "horizon": h, "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean((e.y_pred - e.y_true) ** 2))),
            "rmse_damped": float(np.sqrt(np.mean((e.y_pred_damped - e.y_true) ** 2))),
            "lat": dist_map[st]["lat"], "lon": dist_map[st]["lon"],
        })
    metrics = pd.DataFrame(metric_rows)
    # nearest training station distance per held site
    for arm, g in metrics.groupby("arm"):
        folds = dict([(f[0], f[1]) for f in arms])
        pass
    dists = []
    for arm in ("region", "random"):
        foldlist = region_folds(registry) if arm == "region" else random_folds(registry)
        for fold_i, (train_st, hold_st) in enumerate(foldlist):
            for st in hold_st:
                d = min(haversine(dist_map[st]["lat"], dist_map[st]["lon"],
                                  dist_map[t]["lat"], dist_map[t]["lon"]) for t in train_st)
                dists.append({"arm": arm, "fold": fold_i, "site_id": st, "nearest_km": float(d)})
    dist_df = pd.DataFrame(dists)
    metrics = metrics.merge(dist_df.drop(columns=["fold"]), on=["arm", "site_id"], how="left")
    metrics.to_csv(OUT / "region_transfer_metrics_2021_2023.csv", index=False)

    summary: dict[str, dict] = {}
    for arm in ("region", "random"):
        g = metrics[metrics.arm == arm]
        summary[arm] = {
            "n_stations": int(g.site_id.nunique()),
            "rmse_median": {str(h): float(g[g.horizon == h].rmse.median())
                            for h in (1, 3, 7)},
            "rmse_damped_median": {str(h): float(g[g.horizon == h].rmse_damped.median())
                                   for h in (1, 3, 7)},
            "skill_vs_damped_median": {
                str(h): float(1 - g[g.horizon == h].rmse.median()
                              / g[g.horizon == h].rmse_damped.median())
                for h in (1, 3, 7)},
            "nearest_km_median": float(g.nearest_km.median()),
            "nearest_km_mean": float(g.nearest_km.mean()),
        }
    # station-paired random vs region RMSE at 3d (the headline spatial contrast)
    pv = metrics.pivot_table(index=["site_id", "horizon"], columns="arm", values="rmse")
    pv = pv.dropna()
    for h in (1, 3, 7):
        s = pv[pv.index.get_level_values("horizon") == h]
        summary[f"paired_3d_style_{h}d"] = {
            "n": int(len(s)),
            "region_minus_random_median": float((s["region"] - s["random"]).median()),
            "random_win_frac": float(((s["random"] < s["region"])).mean()),
        }
    (OUT / "region_transfer_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
