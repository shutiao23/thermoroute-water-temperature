#!/usr/bin/env python3
"""Information-ladder spatial experiment, protocol v2 (DLOG-013).

Replaces the archived 2x2 ``run_spatial_factorial.py`` (DLOG-012 defects:
resume data loss, pooled cache-key fold omission, prediction overwrite) with a
fold-complete, leak-free runner:

    geometry {random-site (5 seeds x 4 folds), whole-region (4 folds)}
    x level {L0 gauged-local, L1 gauged-pooled, L2 ungauged-thermal,
             L2U2 ungauged-no-hydrology, L3 ungauged-regionalized}
    x model {LightGBM, ResidualLightGBM} x horizon {1, 3, 7}

Outputs are key-level shards under ``outputs/final/ladder_shards/``, one file
per cell, written on completion; resume means "shard file exists".  No summary
is written unless the shard set is COMPLETE.  Preprocessing caches carry the
fold in their key and are cleared after every fold.

Key semantics (production mode): every cell scores EXACTLY the common
forecast-key registry (``outputs/final/forecast_keys.parquet``) restricted to
its held stations and horizon — the assertion is two-sided, so a cell may
neither decline a registry key nor score a key outside it.  Training rows obey
the same admissibility rule as the registry (issue-date water temperature
genuinely observed), evaluated on the unmasked panel.  All levels of a cell
therefore score an identical key set, and ladder RMSEs are on the same keys as
the main held-out tables.  With ``--legacy-keys`` the runner reproduces the
archived 2x2 factorial's key semantics (full calendar grid, no registry join,
no training admissibility filter) into ``ladder_shards_legacy/``; this mode
exists solely for the G1 golden test that proves the refactor is
side-effect-free, and its outputs must never be compared with the main
held-out tables.

L2/L3 mask the target station's water-temperature inputs entirely (values
replaced by the pooled day-of-year median, observedness cleared) and remove
the persistence-derived feature columns; labels are repaired from the unmasked
panel so the key registry is identical across all cells.

Usage::

    python scripts/final/run_information_ladder.py
        [--geometry region|random|all]
        [--levels L0,L1,L2,L3] [--models LightGBM,ResidualLightGBM]
        [--seeds 0,1,2,3,4] [--horizons 1,3,7]
        [--max-cells N]          # smoke-test cap (skips completeness gate)
        [--legacy-keys]          # G1: reproduce the archived key semantics
        [--golden-l0]            # G1 (with --legacy-keys): reproduce the
                                 # archived local cells to machine precision;
                                 # G2 (production): verify registry equality
        [--mask-proof]           # L2 perturbation-invariance check (one cell)

Golden tests::

    python scripts/final/run_information_ladder.py --levels L0 --geometry all \
        --seeds 0 --golden-l0 --legacy-keys        # G1 (vs archived run)
    python scripts/final/run_information_ladder.py --levels L0 --geometry all \
        --seeds 0                                   # G2 (registry-scored)
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

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute.baselines import _lgb_fit

FINAL = ROOT / "outputs" / "final"
OUT_CONV = ROOT / "outputs" / "conventional"
SHARDS = FINAL / "ladder_shards"          # production shard directory
SHARDS_LEGACY = FINAL / "ladder_shards_legacy"  # --legacy-keys shard directory
EFFECTS_PATH = FINAL / "ladder_effects.parquet"
EFFECTS_LEGACY_PATH = FINAL / "ladder_effects_legacy.parquet"

MIN_KEYS = 100  # reportability threshold, manuscript Section 3.6

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
HORIZONS = (1, 3, 7)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
MIN_TRAIN_ROWS = 500
CONFIRM_START = np.datetime64("2021-01-01")

# Frozen main-design per-lead LightGBM selections (2016-2017 validation).
FROZEN_PARAMS = {1: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 3: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 7: {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.03}}
BEST_ITER = {1: 800, 3: 575, 7: 440}

# WTEMP-derived columns that carry the target station's thermal memory into
# the model; removed from the feature list (not zeroed) for L2/L3, and logged.
PERSISTENCE_FEATURES = ("persistence", "clim_anom")

MASKED_LEVELS = {"L2", "L2U2", "L3"}


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


def load_panel(registry: pd.DataFrame) -> pd.DataFrame:
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
    return panel


def reference_keys() -> pd.DataFrame:
    ref = pd.read_parquet(FINAL / "forecast_keys.parquet",
                          columns=["site_id", "horizon", "issue_date", "y_true"])
    ref["site_id"] = ref["site_id"].astype(str).str.zfill(8)
    ref["issue_date"] = pd.to_datetime(ref["issue_date"])
    return ref


def fit_preprocessing(level: str, panel: pd.DataFrame, fit_mask: np.ndarray,
                      train_st: list[str], stations: tuple[str, ...]):
    """Return (imputer, clim, anchor) for a (level, fold) combination.

    L0 fits every statistic on each station's own history; L1-L3 fit on the
    in-fold training stations only (pooled), so a held region's long
    histories never enter preprocessing.
    """
    if level == "L0":
        imputer = D.Imputer.fit(panel, fit_mask)
        clim = F.HarmonicClimatology.fit(panel, fit_mask, fit_stations=stations)
        anchor = F.DampedPersistenceAnchor.fit(panel, fit_mask, clim,
                                               fit_stations=stations)
    else:
        fold_st = tuple(train_st)
        imputer = D.Imputer.fit(panel, fit_mask, fit_stations=fold_st, pooled=True)
        clim = F.HarmonicClimatology.fit(panel, fit_mask, fit_stations=fold_st,
                                         pooled=True)
        anchor = F.DampedPersistenceAnchor.fit(panel, fit_mask, clim,
                                               fit_stations=fold_st, pooled=True)
    return imputer, clim, anchor


def mask_station_inputs(panel: pd.DataFrame, imputer: D.Imputer,
                        hold_st: list[str], *, var: str) -> pd.DataFrame:
    """Replace the held stations' ``var`` inputs with the pooled doy median.

    Values (observed and missing) are overwritten and the observedness column
    is cleared, so every derived feature and every missingness indicator
    reports "no record".  The label column is repaired separately by the
    caller.
    """
    out = panel.copy()
    med = imputer.medians.get(("__pooled__", var))
    if med is None:
        raise ValueError(f"mask requires a pooled-imputer {var} doy median")
    doy = pd.to_datetime(out["DATE"]).dt.dayofyear
    gmed = float(imputer.global_median.get(("__pooled__", var), np.nan))
    if np.isnan(gmed):
        gmed = float(med.median())
    for st in hold_st:
        sel = (out.site_id == st).to_numpy()
        fill = pd.Series(doy[sel]).map(med).to_numpy()
        fill = np.where(np.isnan(fill), gmed, fill)
        col = out[var].to_numpy(dtype=float, copy=True)
        col[sel] = fill
        out[var] = col
        obs_name = f"{var}_observed"
        if obs_name in out.columns:
            obs = out[obs_name].to_numpy(dtype=float, copy=True)
            obs[sel] = 0.0
            out[obs_name] = obs
    return out


def build_features(panel_masked: pd.DataFrame, panel_true: pd.DataFrame,
                   imputer: D.Imputer, clim: F.HarmonicClimatology, h: int,
                   hold_st: list[str], level: str, *,
                   legacy_keys: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Feature table for one (level, fold, horizon).

    ``panel_masked`` carries the L2/L3 input masks; for L0/L1 it equals
    ``panel_true``.  Labels (y, y_observed) are always taken from
    ``panel_true`` so the evaluation key registry is identical across levels.

    Production mode enforces the registry's admissibility rule on every row:
    the issue-date water temperature must be genuinely observed (computed
    from the UNMASKED panel), and the label must be finite.  ``legacy_keys``
    reproduces the archived runner's semantics (observed-target rows only,
    no admissibility column, no registry join) and exists solely for the G1
    golden test.
    """
    if legacy_keys:
        tab = F.build_tabular(panel_masked, h, USGS_VARS, clim,
                              drop_feature_nans=False,
                              require_observed_target=True,
                              include_missingness=True)
        tab = F.attach_split(tab, C.SPLIT)
        tab.loc[tab.issue_date >= CONFIRM_START, "split"] = "confirm"
        cols = F.feature_columns(tab)
        for c in cols:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
        return tab, cols

    tab = F.build_tabular(panel_masked, h, USGS_VARS, clim,
                          drop_feature_nans=False,
                          require_observed_target=False,
                          include_missingness=True)
    tab = F.attach_split(tab, C.SPLIT)
    tab.loc[tab.issue_date >= CONFIRM_START, "split"] = "confirm"
    # repair labels from the true panel
    true = F.build_tabular(panel_true, h, USGS_VARS, clim,
                           drop_feature_nans=False,
                           require_observed_target=False,
                           include_missingness=True)
    true = F.attach_split(true, C.SPLIT)
    true.loc[true.issue_date >= CONFIRM_START, "split"] = "confirm"
    key = tab.set_index(["site_id", "issue_date"])
    tkey = true.set_index(["site_id", "issue_date"])
    tab["y"] = tkey["y"].reindex(key.index).to_numpy()
    if "y_observed" in tab.columns:
        # NaN must NOT become True: pd.Series([1,0,nan]).astype(bool) is
        # [True,False,True], which would admit unlabeled keys
        tab["y_observed"] = tkey["y_observed"].reindex(key.index).to_numpy()
        obs = pd.to_numeric(tab["y_observed"], errors="coerce").fillna(0.0)
        tab = tab[obs > 0]
        tab = tab.drop(columns=["y_observed"])
    # labels must be finite (never train or score an imputed target)
    tab = tab[np.isfinite(pd.to_numeric(tab["y"], errors="coerce"))]
    # admissibility: issue-date water temperature genuinely observed, taken
    # from the UNMASKED panel (a masked panel would rule out every held
    # station; the registry-side check below cross-validates this path)
    if "WTEMP_observed" in panel_true.columns:
        adm = (panel_true[["site_id", "DATE", "WTEMP_observed"]]
               .rename(columns={"DATE": "issue_date",
                                "WTEMP_observed": "issue_wtemp_observed"}))
    else:
        adm = panel_true[["site_id", "DATE", "WTEMP"]].copy()
        adm["issue_wtemp_observed"] = adm["WTEMP"].notna().astype(float)
        adm = adm.drop(columns=["WTEMP"]).rename(columns={"DATE": "issue_date"})
    adm["site_id"] = adm["site_id"].astype(str).str.zfill(8)
    adm["issue_date"] = pd.to_datetime(adm["issue_date"])
    tab = tab.merge(adm, on=["site_id", "issue_date"], how="left")
    tab["issue_wtemp_observed"] = tab["issue_wtemp_observed"].fillna(0.0)
    cols = [c for c in F.feature_columns(tab) if c != "issue_wtemp_observed"]
    for c in cols:
        tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
    return tab, cols


def removed_feature_columns(tab: pd.DataFrame, level: str) -> list[str]:
    """WTEMP-derived columns excluded for L2/L3 (logged, not zeroed)."""
    if level not in MASKED_LEVELS:
        return []
    return [c for c in F.feature_columns(tab)
            if c.startswith("WTEMP_") or c in PERSISTENCE_FEATURES]


def anchor_for(level: str, rows: pd.DataFrame, phi: dict[str, float], h: int) -> np.ndarray:
    """Damped-anchor expectation per level.

    L0/L1 include the last-observation term; L2/L2U2/L3 use the pooled
    climatology only, with no last-observation term.
    """
    clim_target = rows["clim_target"].to_numpy(float)
    if level == "L0" or level == "L1":
        return (clim_target
                + rows["site_id"].map(phi).to_numpy(float) ** h
                * (rows["persistence"].to_numpy(float)
                   - rows["clim_t"].to_numpy(float)))
    return clim_target


def shard_path(geometry: str, level: str, model: str, seed: int, fold: int,
               h: int) -> Path:
    return SHARDS / f"{geometry}_{level}_{model}_s{seed}_f{fold}_h{h}.parquet"


def expected_cell_count(geometry: str, levels: list[str], models: list[str],
                        horizons: list[int], seeds: list[int]) -> int:
    if geometry == "region":
        folds = 4
    elif geometry == "random":
        folds = 4 * len(seeds)
    else:
        folds = 4 * (1 + len(seeds))
    return folds * len(levels) * len(models) * len(horizons)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="all", choices=["region", "random", "all"])
    ap.add_argument("--levels", default="L0,L1,L2,L3")
    ap.add_argument("--models", default="LightGBM,ResidualLightGBM")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--horizons", default="1,3,7")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="smoke-test cap; disables the completeness gate")
    ap.add_argument("--legacy-keys", action="store_true",
                    help="G1 mode: reproduce the archived runner's key "
                         "semantics (full calendar grid) into "
                         "ladder_shards_legacy/; never compares with the "
                         "main held-out tables")
    ap.add_argument("--golden-l0", action="store_true",
                    help="G1 (with --legacy-keys): reproduce the archived "
                         "local cells to within 1e-6; G2 (production): "
                         "verify registry equality per cell")
    ap.add_argument("--mask-proof", action="store_true",
                    help="run the L2 perturbation-invariance proof on one cell")
    args = ap.parse_args()

    global SHARDS, EFFECTS_PATH
    if args.legacy_keys:
        SHARDS = SHARDS_LEGACY
        EFFECTS_PATH = EFFECTS_LEGACY_PATH

    levels = [l for l in args.levels.split(",") if l]
    models = [m for m in args.models.split(",") if m]
    horizons = [int(h) for h in args.horizons.split(",") if h]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    if args.mask_proof:
        # standalone perturbation-invariance proof; does not touch shards
        registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
        stations = tuple(sorted(registry.site_no.astype(str).str.zfill(8)))
        C.STATIONS = stations
        panel = load_panel(registry)
        dates = panel["DATE"].to_numpy()
        fit_mask = dates <= np.datetime64("2015-12-31")
        val_mask = (dates > np.datetime64("2015-12-31")) & (dates <= np.datetime64("2017-12-31"))
        return mask_proof(panel, fit_mask | val_mask, registry, stations)

    if args.golden_l0 and set(levels) >= {"L0", "L1", "L2", "L3"}:
        # golden mode only needs the archived local cells
        levels = ["L0"]
        print("golden mode: restricting levels to L0", flush=True)

    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    stations = tuple(sorted(registry.site_no.astype(str).str.zfill(8)))
    C.STATIONS = stations
    panel = load_panel(registry)
    dates = panel["DATE"].to_numpy()
    train_mask = dates <= np.datetime64("2015-12-31")
    val_mask = (dates > np.datetime64("2015-12-31")) & (dates <= np.datetime64("2017-12-31"))
    fit_mask = train_mask | val_mask
    ref = reference_keys()

    geometry_list = ["region", "random"] if args.geometry == "all" else [args.geometry]
    folds_by_arm: dict[str, list[tuple[int, list[tuple[list[str], list[str]]]]]] = {
        "region": [(0, region_folds(registry))],
        "random": [(seed, random_folds(registry, seed)) for seed in seeds],
    }

    SHARDS.mkdir(parents=True, exist_ok=True)
    total = expected_cell_count(args.geometry, levels, models, horizons, seeds)
    done_cells = 0

    # per-fold preprocessing cache: key includes the fold, cleared per fold
    preproc: dict[tuple, object] = {}
    # evaluation key sets per (geometry, seed, fold, model, horizon); every
    # level of the same cell must score the IDENTICAL key set, and each set
    # must cover the common registry's subset for the held stations.
    ev_sets: dict[tuple, frozenset] = {}

    for geometry in geometry_list:
        for seed, folds in folds_by_arm[geometry]:
            for fold_i, (train_st, hold_st) in enumerate(folds):
                for level in levels:
                    regionalizer = None
                    if level == "L3":
                        attr = FINAL / "basin_attributes_v2.parquet"
                        if not attr.exists():
                            print(f"[{geometry} {level}] SKIP: {attr.name} "
                                  f"absent (Phase 2 prerequisite); run degraded "
                                  f"without L3", flush=True)
                            continue
                        from thermoroute import regionalize as R
                        regionalizer = R.AttributeRegionalizer(
                            pd.read_parquet(attr), k=5)
                    prep_key = ("prep", geometry, seed, fold_i, level)
                    if prep_key not in preproc:
                        imputer, clim, anchor = fit_preprocessing(
                            level, panel, fit_mask, train_st, stations)
                        panel_imp = imputer.transform(panel)
                        panel_masked = panel_imp
                        if level in MASKED_LEVELS:
                            panel_masked = mask_station_inputs(
                                panel_imp, imputer, hold_st, var="WTEMP")
                        if level == "L2U2":
                            panel_masked = mask_station_inputs(
                                panel_masked, imputer, hold_st, var="FLOW")
                        phi = {site: float(anchor.phi.get(site, 0.9))
                               for site in stations}
                        if regionalizer is not None:
                            phi = regionalizer.regionalize_phi(
                                phi, train_st, hold_st)
                        preproc[prep_key] = (imputer, clim, anchor, panel_imp,
                                             panel_masked, phi)
                        print(f"[{geometry} seed{seed} fold{fold_i} {level}] "
                              f"preprocessing fitted", flush=True)
                    imputer, clim, anchor, panel_imp, panel_masked, phi = preproc[prep_key]

                    for model in models:
                        for h in horizons:
                            if args.max_cells and done_cells >= args.max_cells:
                                print("smoke cap reached; completeness gate "
                                      "skipped", flush=True)
                                return 0
                            sp = shard_path(geometry, level, model, seed, fold_i, h)
                            if sp.exists() and sp.stat().st_size > 0:
                                done_cells += 1
                                continue
                            tab, cols = build_features(
                                panel_masked, panel_imp, imputer, clim, h,
                                hold_st, level,
                                legacy_keys=args.legacy_keys)
                            drop_cols = removed_feature_columns(tab, level)
                            model_cols = [c for c in cols if c not in drop_cols]
                            if drop_cols:
                                print(f"    {sp.name}: removed WTEMP features "
                                      f"{len(drop_cols)}: "
                                      f"{','.join(drop_cols[:6])}"
                                      + ("..." if len(drop_cols) > 6 else ""),
                                      flush=True)
                            tr = tab[tab.split.isin(["train", "val"])
                                     & tab.site_id.isin(train_st)]
                            if not args.legacy_keys and \
                                    "issue_wtemp_observed" in tab.columns:
                                # training rows obey the registry's
                                # admissibility rule (issue-date WTEMP
                                # observed), evaluated on the unmasked panel
                                tr = tr[tr["issue_wtemp_observed"].astype(bool)]
                            if len(tr) < MIN_TRAIN_ROWS:
                                print(f"    {sp.name}: SKIP, only "
                                      f"{len(tr)} train rows (incomplete cell; "
                                      f"completeness gate will FAIL)",
                                      flush=True)
                                done_cells += 1
                                continue
                            if model == "ResidualLightGBM":
                                ytr = tr["y"].to_numpy(float) - anchor_for(
                                    level, tr, phi, h)
                            else:
                                ytr = tr["y"].to_numpy(float)
                            params = {"num_leaves": FROZEN_PARAMS[h]["num_leaves"],
                                      "min_child_samples": FROZEN_PARAMS[h]["min_child_samples"],
                                      "learning_rate": FROZEN_PARAMS[h]["learning_rate"]}
                            m = _lgb_fit(
                                tr[model_cols], ytr,
                                tr[model_cols].to_numpy(float)[-min(2000, len(tr)):],
                                ytr[-min(2000, len(ytr)):],
                                "regression", n_est=BEST_ITER[h],
                                params_override=params)
                            ev = tab[tab.split.eq("confirm")
                                     & tab.site_id.isin(hold_st)]
                            ref_h = ref[(ref.horizon == h)
                                        & (ref.site_id.isin(hold_st))][
                                ["site_id", "issue_date"]]
                            if not args.legacy_keys:
                                # score EXACTLY the common registry's subset:
                                # inner-join the cell's held-station keys
                                assert pd.api.types.is_datetime64_any_dtype(
                                    ev["issue_date"]), (
                                    "issue_date dtype mismatch; registry "
                                    "merge would silently drop all keys")
                                ev = ev.merge(ref_h, on=["site_id", "issue_date"],
                                              how="inner")
                            if ev.empty:
                                print(f"    {sp.name}: SKIP, no evaluation keys "
                                      f"(incomplete cell; completeness gate "
                                      f"will FAIL)", flush=True)
                                done_cells += 1
                                continue
                            # common-key-registry assertion (protocol v2):
                            # TWO-SIDED — a cell may neither decline a
                            # registry key nor score a key outside the
                            # registry; plus every level of the same cell
                            # scores the IDENTICAL key set, so all ladder
                            # cells are comparable.  Legacy mode is exempt
                            # by design (it reproduces the archived grid).
                            ev_sub = set(ev[["site_id", "issue_date"]]
                                         .itertuples(index=False, name=None))
                            if not args.legacy_keys:
                                ref_sub = set(ref_h.itertuples(
                                    index=False, name=None))
                                missing_keys = ref_sub - ev_sub
                                extra_keys = ev_sub - ref_sub
                                if missing_keys or extra_keys:
                                    raise AssertionError(
                                        f"key registry breach in {sp.name}: "
                                        f"{len(missing_keys)} registry keys "
                                        f"not evaluated, {len(extra_keys)} "
                                        f"evaluated keys outside the registry"
                                        f" (examples: "
                                        f"{sorted(extra_keys)[:3]})")
                            cell_id = (geometry, seed, fold_i, model, h)
                            if cell_id in ev_sets and ev_sets[cell_id] != ev_sub:
                                raise AssertionError(
                                    f"key-set mismatch in {sp.name}: this "
                                    f"level's {len(ev_sub)} keys differ from "
                                    f"the {len(ev_sets[cell_id])} keys scored "
                                    f"by an earlier level of the same cell; "
                                    f"ladder cells are not comparable")
                            ev_sets[cell_id] = ev_sub
                            yhat = m.predict(ev[model_cols], num_threads=1)
                            ev_damped = anchor_for(level, ev, phi, h)
                            if model == "ResidualLightGBM":
                                yhat = yhat + ev_damped
                            frame = pd.DataFrame({
                                "site_id": ev.site_id.to_numpy(),
                                "issue_date": ev.issue_date.to_numpy(),
                                "horizon": h,
                                "y_pred": yhat,
                                "y_true": ev["y"].to_numpy(),
                                "y_damped": ev_damped,
                            })
                            frame["geometry"] = geometry
                            frame["level"] = level
                            frame["model"] = model
                            frame["seed"] = seed
                            frame["fold"] = fold_i
                            frame.to_parquet(sp, index=False)
                            done_cells += 1
                            print(f"    {sp.name}: {len(frame)} keys, RMSE "
                                  f"{np.sqrt(np.mean((yhat - frame.y_true)**2)):.3f}",
                                  flush=True)
                preproc.clear()  # memory: never hold more than one fold

    if args.max_cells:
        print(f"smoke cap ({args.max_cells}); completeness gate skipped")
        return 0

    # completeness gate: every expected shard must exist and be non-empty
    missing = []
    for geometry in geometry_list:
        for seed, folds in folds_by_arm[geometry]:
            for fold_i in range(len(folds)):
                for level in levels:
                    for model in models:
                        for h in horizons:
                            sp = shard_path(geometry, level, model, seed, fold_i, h)
                            if not sp.exists() or sp.stat().st_size == 0:
                                missing.append(f"{geometry}/{level}/{model}/"
                                               f"s{seed}/f{fold_i}/h{h}")
    if missing:
        print(f"INCOMPLETE: {len(missing)} of {total} cells missing; "
              f"no summary written.")
        for m in missing[:20]:
            print("  -", m)
        return 1
    print(f"completeness: {total}/{total} cells")

    # aggregate per-site metrics from the requested geometries' shards
    frames = []
    for geometry in geometry_list:
        for sp in sorted(SHARDS.glob(f"{geometry}_*.parquet")):
            if sp.stat().st_size > 0:
                frames.append(pd.read_parquet(sp))
    if not frames:
        print("no shards; nothing to aggregate")
        return 1
    pred = pd.concat(frames, ignore_index=True)
    metric_rows = []
    for (geometry, level, model, seed, site, h), g in pred.groupby(
            ["geometry", "level", "model", "seed", "site_id", "horizon"]):
        if args.legacy_keys:
            # archive-equivalent metrics (NaN-skipping pandas means), so G1
            # reproduces the archived per-site RMSE to machine precision
            e = g[g.y_pred.notna() & g.y_damped.notna()]
            if e.empty:
                continue
            metric_rows.append({
                "geometry": geometry, "level": level, "model": model,
                "seed": seed, "site_id": site, "horizon": h,
                "n": len(e),
                "rmse": float(np.sqrt(np.mean((e.y_pred - e.y_true) ** 2))),
                "rmse_damped": float(np.sqrt(
                    np.mean((e.y_damped - e.y_true) ** 2))),
            })
            continue
        # production: n counts keys with a REAL label; NaN must surface, not
        # be silently skipped (ndarray mean propagates NaN)
        e = g.dropna(subset=["y_pred", "y_damped", "y_true"])
        if e.empty:
            continue
        err = (e.y_pred - e.y_true).to_numpy(float)
        errd = (e.y_damped - e.y_true).to_numpy(float)
        metric_rows.append({
            "geometry": geometry, "level": level, "model": model,
            "seed": seed, "site_id": site, "horizon": h,
            "n": len(e),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "rmse_damped": float(np.sqrt(np.mean(errd ** 2))),
        })
    metrics = pd.DataFrame(metric_rows)
    if not args.legacy_keys and len(metrics):
        # reportability gate (manuscript Section 3.6): a station-cell needs
        # at least 100 paired keys; non-reportable cells are RETAINED with a
        # flag so downstream analyses filter explicitly and SI can report
        # how many were dropped
        metrics["reportable"] = metrics["n"] >= MIN_KEYS
        n_drop = int((~metrics["reportable"]).sum())
        if n_drop:
            print(f"reportability: {n_drop} station-cells below {MIN_KEYS} "
                  f"keys (retained with reportable=False; downstream must "
                  f"filter)")
    metrics.to_parquet(EFFECTS_PATH, index=False)
    summary = {
        "cells_total": total,
        "cells_written": int(done_cells),
        "models": models,
        "levels": levels,
        "horizons": horizons,
        "legacy_keys": bool(args.legacy_keys),
    }
    (FINAL / "ladder_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"wrote {EFFECTS_PATH} "
          f"({len(metrics)} station-cells)")

    if args.golden_l0:
        return golden_l0(metrics, models, horizons, seeds,
                         legacy_keys=args.legacy_keys)
    return 0


def golden_l0(ladder: pd.DataFrame, models, horizons, seeds,
              legacy_keys: bool) -> int:
    """G1/G2 golden verification of the L0 level.

    G1 (``legacy_keys``): reproduce the archived local-adaptation cells.  The
    observed agreement is at machine precision (worst |Δ| = 4.4e-15 over
    1,380 station-cells); the tolerance is 1e-6.  Stations that the archived
    run lost to the DLOG-012 resume defect are reported, not silently
    dropped.

    G2 (production): the registry-scored cells cannot be compared with the
    archived run (different key sets by design).  The production gate is the
    two-sided registry assertion enforced inline per cell, plus a per-cell
    verification here that every shard's key set equals the registry subset.
    """
    if not legacy_keys:
        problems = _verify_registry_scored_cells(models, horizons, seeds)
        if problems:
            for p in problems:
                print("  -", p)
            print("G2 GOLDEN FAILED")
            return 1
        print("G2 golden passed: every cell scored exactly the common "
              "registry's subset for its held stations")
        return 0
    archived = pd.read_parquet(FINAL / "spatial_effects.parquet")
    arch = archived[archived.adaptation == "local"]
    worst = 0.0
    compared = 0
    mismatched_cells = []
    for (geometry, model, seed, h), g in ladder.groupby(
            ["geometry", "model", "seed", "horizon"]):
        a = arch[(arch.geometry == geometry) & (arch.model == model)
                 & (arch.seed == seed) & (arch.horizon == h)]
        if a.empty:
            print(f"skip {geometry}/{model}/s{seed}/h{h}: no archived cell")
            continue
        joined_all = g.set_index("site_id")[["rmse"]].join(
            a.set_index("site_id")[["rmse"]], rsuffix="_arch")
        joined = joined_all.dropna()
        n_unmatched = len(joined_all) - len(joined)
        if n_unmatched:
            print(f"  NOTE {geometry}/{model}/s{seed}/h{h}: {n_unmatched} "
                  f"stations absent from the archived cell (DLOG-012 resume "
                  f"defect); compared on {len(joined)}")
        if joined.empty:
            continue
        diff = (joined["rmse"] - joined["rmse_arch"]).abs().max()
        worst = max(worst, float(diff))
        compared += len(joined)
        if float(diff) > 1e-6:
            mismatched_cells.append(f"{geometry}/{model}/s{seed}/h{h}")
        print(f"{geometry}/{model}/s{seed}/h{h}: n={len(joined)} "
              f"max|diff|={diff:.2e}")
    print(f"golden L0 (G1 legacy): {compared} station-cells compared, "
          f"worst |diff| = {worst:.2e}")
    if compared == 0:
        print("nothing compared: request L0 cells that the archived run "
              "contains (e.g. --levels L0 --geometry all --seeds 0)")
        return 2
    if worst > 1e-6:
        print("G1 GOLDEN FAILED: reproduction exceeds the 1e-6 tolerance; "
              f"mismatched cells: {mismatched_cells}")
        return 1
    print("G1 golden passed: the refactor is side-effect-free (agreement at "
          "machine precision)")
    return 0


def _verify_registry_scored_cells(models, horizons, seeds) -> list[str]:
    """G2: every production shard's key set equals the registry subset."""
    import glob as _glob
    ref = reference_keys()
    problems = []
    for f in sorted(_glob.glob(str(SHARDS / "*.parquet"))):
        d = pd.read_parquet(f)
        d["site_id"] = d["site_id"].astype(str).str.zfill(8)
        d["issue_date"] = pd.to_datetime(d["issue_date"])
        h = int(d["horizon"].iloc[0])
        held = set(d["site_id"].unique())
        ref_sub = set(ref[(ref.horizon == h) & (ref.site_id.isin(held))]
                      [["site_id", "issue_date"]]
                      .itertuples(index=False, name=None))
        cell = set(d[["site_id", "issue_date"]].itertuples(index=False, name=None))
        missing = ref_sub - cell
        extra = cell - ref_sub
        if missing or extra:
            problems.append(
                f"{Path(f).name}: {len(missing)} registry keys missing, "
                f"{len(extra)} outside (extra examples: "
                f"{sorted(extra)[:3]})")
    return problems


def mask_proof(panel: pd.DataFrame, fit_mask, registry, stations) -> int:
    """Perturbation-invariance proof for L2 (protocol v2 runner requirement).

    Trains the L2 region fold-0 LightGBM 7-day cell twice — once on the
    masked panel and once where the held stations' WTEMP is perturbed by
    +5 C before masking — and asserts bit-identical predictions.  A nonzero
    difference means the mask leaks the target station's water temperature.
    """
    region = region_folds(registry)
    train_st, hold_st = region[0]
    fold_st = tuple(train_st)
    imputer = D.Imputer.fit(panel, fit_mask, fit_stations=fold_st, pooled=True)
    clim = F.HarmonicClimatology.fit(panel, fit_mask, fit_stations=fold_st,
                                     pooled=True)
    F.DampedPersistenceAnchor.fit(panel, fit_mask, clim,
                                       fit_stations=fold_st, pooled=True)
    panel_imp = imputer.transform(panel)
    ref = reference_keys()
    h = 7

    def run_masked(panel_source: pd.DataFrame) -> np.ndarray:
        masked = mask_station_inputs(panel_source, imputer, hold_st, var="WTEMP")
        tab, cols = build_features(masked, panel_imp, imputer, clim, h,
                                   hold_st, "L2")
        drop_cols = removed_feature_columns(tab, "L2")
        model_cols = [c for c in cols if c not in drop_cols]
        tr = tab[tab.split.isin(["train", "val"]) & tab.site_id.isin(train_st)]
        ytr = tr["y"].to_numpy(float)
        params = {"num_leaves": FROZEN_PARAMS[h]["num_leaves"],
                  "min_child_samples": FROZEN_PARAMS[h]["min_child_samples"],
                  "learning_rate": FROZEN_PARAMS[h]["learning_rate"]}
        m = _lgb_fit(tr[model_cols], ytr,
                     tr[model_cols].to_numpy(float)[-min(2000, len(tr)):],
                     ytr[-min(2000, len(ytr)):],
                     "regression", n_est=BEST_ITER[h], params_override=params)
        ev = tab[tab.split.eq("confirm") & tab.site_id.isin(hold_st)]
        ref_h = ref[(ref.horizon == h) & (ref.site_id.isin(hold_st))][
            ["site_id", "issue_date"]]
        ev = ev.merge(ref_h, on=["site_id", "issue_date"], how="inner")
        return m.predict(ev[model_cols], num_threads=1)

    base = run_masked(panel_imp)
    perturbed = panel_imp.copy()
    sel = perturbed.site_id.isin(hold_st).to_numpy()
    wt = perturbed["WTEMP"].to_numpy(dtype=float, copy=True)
    wt[sel] = np.where(np.isnan(wt[sel]), 10.0, wt[sel]) + 5.0
    perturbed["WTEMP"] = wt
    alt = run_masked(perturbed)
    if len(base) != len(alt):
        print("MASK PROOF FAILED: prediction counts differ")
        return 1
    max_diff = float(np.abs(base - alt).max())
    print(f"mask proof: {len(base)} predictions, max |diff| = {max_diff:.3e}")
    if max_diff != 0.0:
        print("MASK PROOF FAILED: predictions respond to masked inputs")
        return 1
    print("mask proof passed: L2 predictions are invariant to the target "
          "station's WTEMP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
