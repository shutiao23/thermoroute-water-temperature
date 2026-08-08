#!/usr/bin/env python3
"""Matched spatial-transfer factorial on the independent 2021-2023 window.

Replaces ``scripts/conventional_region_transfer_2021_2023.py`` with the 2x2
factorial of protocol v1 (Major Comment 4 of the external review):

    geometry x adaptation
      {random-site, whole-region} x {target-local, training-pooled}

* **Local adaptation**: per-station climatology, damped anchor and imputation
  medians fitted on each station's own 2006-2015 training rows (the held
  station's long history is an allowed issue-time input -> gauged transfer).
* **Pooled adaptation**: climatology, damped anchor and imputation medians
  fitted over the IN-FOLD training stations only and applied to every station,
  so a held region's long histories never enter preprocessing.

Models: station-agnostic LightGBM on the raw target and on the damped-anchor
residual (residual + anchor at scoring).  Hyperparameters are the frozen
main-design per-lead selections (2016-2017 validation); per-fold re-tuning is
documented as a limitation.  Random splits are repeated over five seeds so the
region-minus-random penalty is read against the random-split distribution.

Outputs:
  outputs/final/spatial_effects.parquet   per arm/fold/seed/site/horizon RMSE
  outputs/final/spatial_summary.json      medians, paired penalties, novelty
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import features as F  # noqa: E402
from thermoroute.baselines import _lgb_fit  # noqa: E402

FINAL = ROOT / "outputs" / "final"
OUT_CONV = ROOT / "outputs" / "conventional"

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
HORIZONS = (1, 3, 7)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
MIN_TRAIN_ROWS = 500

# Frozen main-design per-lead LightGBM selections (2016-2017 validation).
FROZEN_PARAMS = {1: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 3: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 7: {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.03}}
BEST_ITER = {1: 800, 3: 575, 7: 440}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def region_folds(registry: pd.DataFrame) -> list[tuple[list[str], list[str]]]:
    by_reg: dict[str, list[str]] = {}
    for _, r in registry.iterrows():
        by_reg.setdefault(str(r.huc2), []).append(str(r.site_no).zfill(8))
    buckets: list[list[str]] = [[], [], [], []]
    loads = [0, 0, 0, 0]
    for _reg, sts in sorted(by_reg.items(), key=lambda kv: -len(kv[1])):
        i = int(np.argmin(loads))
        buckets[i].extend(sts)
        loads[i] += len(sts)
    return [(sorted(set(registry.site_no.astype(str).str.zfill(8)) - set(b)),
             sorted(b)) for b in buckets]


def random_folds(registry: pd.DataFrame, seed: int) -> list[tuple[list[str], list[str]]]:
    sites = sorted(registry.site_no.astype(str).str.zfill(8))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sites))
    return [(sorted(set(sites) - {sites[j] for j in perm[i::4]}),
             sorted(sites[j] for j in perm[i::4])) for i in range(4)]


def main() -> None:
    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    stations = tuple(sorted(registry.site_no.astype(str).str.zfill(8)))
    C.STATIONS = stations

    train_panel = pd.read_parquet(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")
    train_panel["DATE"] = pd.to_datetime(train_panel["DATE"])
    test_panel = pd.read_parquet(OUT_CONV / "panel_2021_2023.parquet")
    test_panel["DATE"] = pd.to_datetime(test_panel["DATE"])
    alias = dict(zip(registry.legacy_site_id.astype(str),
                     registry.site_no.astype(str).str.zfill(8)))
    train_panel["site_id"] = train_panel["site_id"].astype(str).map(alias)
    for df in (train_panel, test_panel):
        df["site_id"] = df["site_id"].astype(str).str.zfill(8)
    panel = pd.concat([train_panel, test_panel], ignore_index=True)
    panel = panel.drop_duplicates(["DATE", "site_id"], keep="last")
    panel = panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)

    dates = panel["DATE"].to_numpy()
    train_mask = dates <= np.datetime64("2015-12-31")
    val_mask = (dates > np.datetime64("2015-12-31")) & (dates <= np.datetime64("2017-12-31"))
    fit_mask = train_mask | val_mask

    # reference keys on the held-out window (common registry)
    ref = pd.read_parquet(FINAL / "forecast_keys.parquet",
                          columns=["site_id", "horizon", "issue_date", "y_true"])
    ref["site_id"] = ref["site_id"].astype(str).str.zfill(8)
    ref["issue_date"] = pd.to_datetime(ref["issue_date"])

    coord = {
        str(r.site_no).zfill(8): {"lat": float(r.lat), "lon": float(r.lon)}
        for r in registry.itertuples(index=False)
    }

    # hydroclimatic attributes for novelty (training-period only)
    drain_area = {
        str(r.site_no).zfill(8): float(r.drain_area_va)
        for r in registry.itertuples(index=False)
    }
    dev_attr = pd.DataFrame({
        "site_id": stations,
        "mean_annual_air_temp_c": [
            float(train_panel[train_panel.site_id == s]["TEMP"].mean())
            for s in stations
        ],
        "log_drain_area": np.log(np.clip(
            [drain_area.get(s, np.nan) for s in stations], 1.0, None)),
        "abs_lat": np.abs([coord[s]["lat"] for s in stations]),
    })

    def hydro_novelty(hold: list[str], train_st: list[str]) -> dict[str, float]:
        fit = dev_attr[dev_attr.site_id.isin(train_st)]
        z = (dev_attr[["mean_annual_air_temp_c", "log_drain_area", "abs_lat"]]
             - fit[["mean_annual_air_temp_c", "log_drain_area", "abs_lat"]].mean()) \
            / fit[["mean_annual_air_temp_c", "log_drain_area", "abs_lat"]].std()
        z = z.fillna(0.0)
        z_train = z[dev_attr.site_id.isin(train_st)].to_numpy(float)
        out = {}
        for site in hold:
            v = z[dev_attr.site_id == site].to_numpy(float)[0]
            out[site] = float(np.min(np.linalg.norm(z_train - v, axis=1)))
        return out

    folds_by_arm: dict[str, list[tuple[int, list[tuple[list[str], list[str]]]]]] = {
        "region": [(0, region_folds(registry))],
        "random": [(seed, random_folds(registry, seed)) for seed in RANDOM_SEEDS],
    }

    preproc_cache: dict[tuple[str, str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    out_path = FINAL / "spatial_effects.parquet"
    rows: list[dict] = []
    done: set[tuple[str, str, str, int, int, int]] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if "fold" in existing.columns and "rmse" not in existing.columns:
            for _, r in existing[["geometry", "adaptation", "model", "seed", "fold",
                                  "horizon"]].drop_duplicates().iterrows():
                done.add((str(r.geometry), str(r.adaptation), str(r.model),
                          int(r.seed), int(r.fold), int(r.horizon)))
            print(f"resume: {len(done)} cells cached", flush=True)

    for geometry, seed_folds in folds_by_arm.items():
        for seed, folds in seed_folds:
            for fold_i, (train_st, hold_st) in enumerate(folds):
                print(f"[{geometry} seed{seed} fold{fold_i}] train {len(train_st)} "
                      f"hold {len(hold_st)}", flush=True)
                hydro = hydro_novelty(hold_st, train_st)
                dists = {
                    st: min(haversine(coord[st]["lat"], coord[st]["lon"],
                                      coord[t]["lat"], coord[t]["lon"])
                            for t in train_st)
                    for st in hold_st
                }
                for adaptation in ("local", "pooled"):
                    for model in ("LightGBM", "ResidualLightGBM"):
                        for h in HORIZONS:
                            key = (geometry, adaptation, model, seed, fold_i, h)
                            if key in done:
                                continue
                            # preprocess variant: local is fold-independent
                            # (per-station fits); pooled depends on the in-fold
                            # training stations of THIS split (region or random).
                            variant = ("local", 0) if adaptation == "local" \
                                else ("pooled", seed, fold_i)
                            prep_key = (variant[0], variant[1], h)
                            if prep_key not in preproc_cache:
                                if adaptation == "local":
                                    imputer = D.Imputer.fit(panel, fit_mask)
                                    clim = F.HarmonicClimatology.fit(
                                        panel, fit_mask, fit_stations=stations)
                                    anchor = F.DampedPersistenceAnchor.fit(
                                        panel, fit_mask, clim, fit_stations=stations)
                                else:
                                    fold_st = tuple(train_st)
                                    imputer = D.Imputer.fit(
                                        panel, fit_mask, fit_stations=fold_st,
                                        pooled=True)
                                    clim = F.HarmonicClimatology.fit(
                                        panel, fit_mask, fit_stations=fold_st,
                                        pooled=True)
                                    anchor = F.DampedPersistenceAnchor.fit(
                                        panel, fit_mask, clim, fit_stations=fold_st,
                                        pooled=True)
                                panel_imp = imputer.transform(panel)
                                tab = F.attach_split(F.build_tabular(
                                    panel_imp, h, USGS_VARS, clim,
                                    drop_feature_nans=False,
                                    require_observed_target=True,
                                    include_missingness=True))
                                tab.loc[tab.issue_date >= np.datetime64("2021-01-01"),
                                        "split"] = "confirm"
                                cols = F.feature_columns(tab)
                                for c in cols:
                                    tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
                                phi = {site: float(anchor.phi.get(site, 0.9))
                                       for site in stations}
                                preproc_cache[prep_key] = (tab, cols, phi)
                            tab, cols, phi = preproc_cache[prep_key]

                            tr = tab[tab.split.isin(["train", "val"])
                                     & tab.site_id.isin(train_st)]
                            if len(tr) < MIN_TRAIN_ROWS:
                                print(f"    h{h}: skip, only {len(tr)} train rows")
                                continue
                            if model == "ResidualLightGBM":
                                ytr = tr["y"].to_numpy(float) - (
                                    tr["clim_target"].to_numpy(float)
                                    + tr["site_id"].map(phi).to_numpy(float) ** h
                                    * (tr["persistence"].to_numpy(float)
                                       - tr["clim_t"].to_numpy(float)))
                            else:
                                ytr = tr["y"].to_numpy(float)
                            params = {"num_leaves": FROZEN_PARAMS[h]["num_leaves"],
                                      "min_child_samples": FROZEN_PARAMS[h]["min_child_samples"],
                                      "learning_rate": FROZEN_PARAMS[h]["learning_rate"]}
                            m = _lgb_fit(
                                tr[cols], ytr,
                                tr[cols].to_numpy(float)[-min(2000, len(tr)):],
                                ytr[-min(2000, len(ytr)):],
                                "regression", n_est=BEST_ITER[h],
                                params_override=params)
                            ev = tab[tab.split.eq("confirm") & tab.site_id.isin(hold_st)]
                            if ev.empty:
                                continue
                            yhat = m.predict(ev[cols], num_threads=1)
                            ev_damped = (ev["clim_target"].to_numpy(float)
                                         + ev["site_id"].map(phi).to_numpy(float) ** h
                                         * (ev["persistence"].to_numpy(float)
                                            - ev["clim_t"].to_numpy(float)))
                            if model == "ResidualLightGBM":
                                yhat = yhat + ev_damped
                            frame = pd.DataFrame({
                                "site_id": ev.site_id.to_numpy(),
                                "horizon": h,
                                "issue_date": ev.issue_date.to_numpy(),
                                "y_pred": yhat,
                                "y_true": ev["y"].to_numpy(),
                                "y_damped": ev_damped,
                            })
                            frame["geometry"] = geometry
                            frame["adaptation"] = adaptation
                            frame["model"] = model
                            frame["seed"] = seed
                            frame["fold"] = fold_i
                            frame["nearest_km"] = frame["site_id"].map(dists)
                            frame["hydro_novelty"] = frame["site_id"].map(hydro)
                            rows.append(frame)
                            pred = pd.concat(rows, ignore_index=True)
                            pred.to_parquet(out_path)
                            print(f"    h{h} {adaptation} {model}: {len(frame)} keys, "
                                  f"RMSE {np.sqrt(np.mean((yhat-frame.y_true)**2)):.3f}",
                                  flush=True)

    if not rows:
        print("no new predictions produced (all cached)")
    else:
        pred = pd.concat(rows, ignore_index=True)
        pred.to_parquet(out_path)

    pred = pd.read_parquet(out_path)
    metric_rows = []
    for (geometry, adaptation, model, seed, site, h), g in pred.groupby(
            ["geometry", "adaptation", "model", "seed", "site_id", "horizon"]):
        e = g[g.y_pred.notna() & g.y_damped.notna()]
        if e.empty:
            continue
        metric_rows.append({
            "geometry": geometry, "adaptation": adaptation, "model": model,
            "seed": seed, "site_id": site, "horizon": h,
            "n": int(len(e)),
            "rmse": float(np.sqrt(np.mean((e.y_pred - e.y_true) ** 2))),
            "rmse_damped": float(np.sqrt(np.mean((e.y_damped - e.y_true) ** 2))),
            "nearest_km": float(e.nearest_km.iloc[0]),
            "hydro_novelty": float(e.hydro_novelty.iloc[0]),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(FINAL / "spatial_effects.parquet", index=False)

    summary: dict = {}
    for model in ("LightGBM", "ResidualLightGBM"):
        summary[model] = {}
        for h in HORIZONS:
            for adaptation in ("local", "pooled"):
                sub = metrics[(metrics.model == model) & (metrics.horizon == h)
                              & (metrics.adaptation == adaptation)]
                block: dict = {}
                for geometry in ("region", "random"):
                    g = sub[sub.geometry == geometry]
                    block[f"{geometry}_n_stations"] = int(g.site_id.nunique())
                    block[f"{geometry}_rmse_median"] = float(g.rmse.median())
                    block[f"{geometry}_rmse_damped_median"] = float(g.rmse_damped.median())
                    block[f"{geometry}_nearest_km_median"] = float(g.nearest_km.median())
                # per-site penalty: region cell (seed 0) minus the mean of the
                # five random-split cells; the random-split spread is the
                # per-site sd across the five seeds.
                reg = sub[sub.geometry == "region"].set_index("site_id")["rmse"]
                rnd = sub[sub.geometry == "random"].groupby("site_id")["rmse"]
                rnd_mean = rnd.mean()
                rnd_sd = rnd.std()
                sites = reg.index.intersection(rnd_mean.index)
                if len(sites):
                    penalty = reg.loc[sites] - rnd_mean.loc[sites]
                    block["paired_region_minus_random_median"] = float(penalty.median())
                    block["paired_region_minus_random_iqr"] = list(
                        np.percentile(penalty, [25, 75]))
                    block["paired_random_win_frac"] = float(
                        (penalty > 0).mean())
                    block["paired_n_sites"] = int(len(sites))
                    block["random_split_spread_median"] = float(rnd_sd.loc[sites].median())
                    # per-seed penalties: region (seed 0) vs each random seed,
                    # so the seed-0 penalty is read against the seed spread.
                    by_seed: dict[str, float] = {}
                    for seed in RANDOM_SEEDS:
                        r = sub[(sub.geometry == "random")
                                & (sub.seed == seed)].set_index("site_id")["rmse"]
                        common = reg.index.intersection(r.index)
                        if len(common):
                            by_seed[str(seed)] = float(
                                (reg.loc[common] - r.loc[common]).median())
                    block["paired_median_by_seed"] = by_seed
                    corr = sub.dropna(subset=["hydro_novelty"]).copy()
                    corr["penalty"] = corr["rmse"] - corr["rmse_damped"]
                    if len(corr) > 10:
                        block["corr_penalty_log_distance"] = float(np.corrcoef(
                            np.log(corr.nearest_km + 1), corr.penalty)[0, 1])
                        block["corr_penalty_hydro_novelty"] = float(np.corrcoef(
                            corr.hydro_novelty, corr.penalty)[0, 1])
                summary[model].setdefault(str(h), {})[adaptation] = block
    (FINAL / "spatial_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == "__main__":
    main()
