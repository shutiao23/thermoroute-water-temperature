#!/usr/bin/env python
"""Conventional 2021-2023 holdout evaluator for the frozen Route-A models.

STANDALONE: does not import the pre-registration apparatus (opening, model_suite
source-hash enforcement, development_*gate|replay, chronology, inference_gate,
protocols).  It loads the frozen trained bundles from the sibling multicore
worktree, applies the frozen development-period preprocessing baked into each
bundle, runs inference on the 2021-2023 holdout, and scores against the NWIS
observed water temperature.

Pipeline
--------
1. VALIDATION (2019-2020): reproduce the stored development test predictions
   with the frozen bundles.  If the frozen-bundle predictions match the dev
   predictions within tolerance, the loading/preprocessing/inference is correct.
2. HOLDOUT (2021-2023): fetch NWIS + Daymet + gridMET for the 120 stations,
   assemble the panel, build frozen-transform windows, run every model
   ensemble, compute metrics (RMSE/MAE/BIAS/skill vs Persistence & Climatology).

Outputs (under ``outputs/conventional/``)
-----------------------------------------
* ``holdout_metrics_2021_2023.csv``  - long metric table
* ``holdout_summary_2021_2023.json`` - summary + reproduction diagnostics
* ``panel_2021_2023.parquet``        - assembled holdout panel
* ``raw_2021_2023/``                 - cached raw HTTP responses (SnapshotStore)
* ``validation_2019_2020.json``      - reproduction diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thermoroute import config as C
from thermoroute import conventional_score as CS
from thermoroute import frozen_inference as FI
from thermoroute import usgs
from thermoroute.provenance import SnapshotStore

REPO = Path(__file__).resolve().parents[1]
DATA_USGS = REPO / "data_usgs"
OUT = REPO / "outputs" / "conventional"

HOLDOUT_START = "2021-01-01"
HOLDOUT_END = "2023-12-31"
CONTEXT = C.CONTEXT_LENGTH  # 32
# 32-day history before the first holdout target day.
FETCH_START = (pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=CONTEXT)).strftime("%Y-%m-%d")
FETCH_END = HOLDOUT_END
HORIZONS = (1, 3, 7)

# Canonical frozen bundles in the sibling multicore worktree (run_id suffixes).
TEMPORAL_BUNDLES = {
    "ThermoRoute": "thermoroute_usgs_bundle_665afcb4d674161943ae",
    "LightGBM": "lightgbm_usgs_bundle_665afcb4d674161943ae",
    "DampedPriorOnly": "dampedprioronly_bundle_665afcb4d674161943ae",
    "TR-noDynamicPrior": "tr-nodynamicprior_bundle_665afcb4d674161943ae",
    "TR-fixedKappa": "tr-fixedkappa_bundle_665afcb4d674161943ae",
    "TR-noRouter": "tr-norouter_bundle_665afcb4d674161943ae",
    "TR-noMoE": "tr-nomoe_bundle_665afcb4d674161943ae",
    "TR-noTCN": "tr-notcn_bundle_665afcb4d674161943ae",
    "TR-unbounded": "tr-unbounded_bundle_665afcb4d674161943ae",
}
LSTM_BUNDLE = "lstm_usgs_bundle_0b81a6ed744e3393d357"
EXTERNAL_BUNDLES = {
    "ThermoRoute-ext": "external_thermoroute_bundle_310b86788bca69f39517",
    "LSTM-ext": "external_lstm_bundle_310b86788bca69f39517",
    "LightGBM-ext": "external_lightgbm_bundle_310b86788bca69f39517",
}

# Which model ids live in which dev prediction artifact.
DEV_PRED_FILES = {
    "stage9": "outputs/predictions/usgs_predictions_stage9_v2.parquet",
    "final": "outputs/predictions/usgs_predictions_v2.parquet",
}


def log(msg: str) -> None:
    print(f"[conv] {msg}", flush=True)


def load_registry(registry_path: Path) -> pd.DataFrame:
    reg = pd.read_csv(registry_path, dtype={"site_no": str, "legacy_site_id": str})
    reg = reg[["site_no", "legacy_site_id", "lat", "lon"]].copy()
    reg["site_no"] = reg["site_no"].str.strip()
    return reg


def load_dev_panel_mapped(panel_path: Path, registry_path: Path) -> pd.DataFrame:
    """Load the frozen dev panel with stable site_no identifiers (no apparatus)."""
    panel = pd.read_parquet(panel_path)
    reg = load_registry(registry_path)
    mapping = dict(zip(reg["legacy_site_id"], reg["site_no"]))
    panel["site_id"] = panel["site_id"].map(mapping).astype(str)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    return panel


# --------------------------------------------------------------------------- #
# Data acquisition for the holdout panel
# --------------------------------------------------------------------------- #
def assemble_holdout_panel(
    registry: pd.DataFrame,
    store: SnapshotStore,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch NWIS + Daymet + gridMET for every station and assemble the panel."""
    full = pd.date_range(start, end, freq="D", name="DATE")
    frames = []
    failures: list[dict] = []
    n = len(registry)
    for i, row in enumerate(registry.itertuples(index=False)):
        site_no = row.site_no
        lat, lon = float(row.lat), float(row.lon)
        rec: dict = {"site_no": site_no, "nwis": None, "daymet": None, "gridmet": None}
        try:
            nwis = usgs.fetch_nwis_daily(site_no, start, end, snapshot_store=store)
        except Exception as exc:  # network/parse failure -> all-NaN outcomes
            nwis, rec["nwis"] = None, str(exc)[:120]
        try:
            met = usgs.fetch_daymet(lat, lon, start, end, snapshot_store=store)
        except Exception as exc:
            met, rec["daymet"] = None, str(exc)[:120]
        try:
            wind = usgs.fetch_gridmet_wind(lat, lon, start, end, snapshot_store=store)
        except Exception as exc:
            wind, rec["gridmet"] = None, str(exc)[:120]
        cols: dict[str, pd.Series] = {}
        for v in ("WTEMP", "FLOW", "WLEVEL"):
            cols[v] = nwis[v].reindex(full) if nwis is not None and v in nwis else pd.Series(np.nan, index=full)
        for v in ("TEMP", "PRCP", "RHMEAN", "DH"):
            cols[v] = met[v].reindex(full) if met is not None and v in met else pd.Series(np.nan, index=full)
        cols["WDSP"] = wind.reindex(full) if wind is not None else pd.Series(np.nan, index=full)
        df = pd.DataFrame(cols, index=full)
        df = df.reset_index()
        df.insert(1, "site_id", site_no)
        df = df[["DATE", "site_id", "WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP", "WDSP", "RHMEAN", "DH"]]
        frames.append(df)
        if nwis is None or (nwis is not None and nwis["WTEMP"].notna().sum() == 0):
            rec["nwis"] = rec["nwis"] or "no WTEMP observations"
            failures.append(rec)
        if (i + 1) % 20 == 0 or i + 1 == n:
            log(f"  fetched {i + 1}/{n} stations")
    panel = pd.concat(frames, ignore_index=True)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    panel["site_id"] = panel["site_id"].astype(str)
    return panel, failures


# --------------------------------------------------------------------------- #
# Validation against stored development predictions
# --------------------------------------------------------------------------- #
def all_valid_keys_from_wd(wd, station_names, split: str = "confirm") -> set[tuple[str, pd.Timestamp]]:
    """Set of (site_id, issue_date) where all horizons are valid (matches dev test)."""
    idx = wd.idx(split)
    all_valid = wd.target_valid[idx].all(axis=1)
    site = np.asarray([station_names[int(i)] for i in wd.station[idx]], dtype=object)
    issue = pd.to_datetime(wd.issue_date[idx])
    return {(str(s), d) for s, d in zip(site[all_valid], issue[all_valid])}


def filter_to_keys(frame: pd.DataFrame, keyset: set) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = list(zip(
        frame["site_id"].astype(str),
        pd.to_datetime(frame["issue_date"]),
    ))
    mask = np.fromiter((k in keyset for k in keys), dtype=bool, count=len(keys))
    return frame.loc[mask].copy()


def load_dev_predictions(multicore: Path, model: str) -> pd.DataFrame:
    """Load dev test predictions for one model from the multicore worktree."""
    cols = ["model", "seed", "site_id", "horizon", "split", "issue_date", "target_date", "y_pred"]
    candidates = [DEV_PRED_FILES["final"], DEV_PRED_FILES["stage9"]]
    for rel in candidates:
        path = multicore / rel
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=cols)
        sub = df[(df["model"] == model) & (df["split"] == "test")]
        if len(sub):
            return sub
    return pd.DataFrame(columns=cols)


def run_validation(
    multicore: Path,
    panel_path: Path,
    registry_path: Path,
    device: str,
    models: list[str],
) -> dict:
    """Reproduce 2019-2020 dev predictions with frozen bundles and compare."""
    log("=== VALIDATION: reproducing 2019-2020 dev predictions ===")
    panel = load_dev_panel_mapped(panel_path, registry_path)
    station_ids = sorted(panel["site_id"].astype(str).unique())
    # Use the ThermoRoute bundle to build the frozen confirmation windows for 2019-2020.
    tr_dir = multicore / "outputs" / "models" / TEMPORAL_BUNDLES["ThermoRoute"]
    weights, tr_meta = __import__("thermoroute.checkpoint", fromlist=["load_inference_bundle"]).load_inference_bundle(tr_dir)
    wd, transforms, imputed = FI.build_frozen_confirmation_windows(
        panel, tr_meta, station_ids, interval=("2019-01-01", "2020-12-31"), external=False)
    station_names = list(C.STATIONS)
    keyset = all_valid_keys_from_wd(wd, station_names)
    log(f"  built {len(wd.X)} confirmation windows; {len(keyset)} all-horizon-valid issue keys")

    diagnostics: dict = {"interval": "2019-01-01..2020-12-31", "models": {}}
    for model in models:
        if model == "LSTM":
            bundle_dir = multicore / "outputs" / "models" / LSTM_BUNDLE
            dev_model = "LSTM"
        else:
            bundle_dir = multicore / "outputs" / "models" / TEMPORAL_BUNDLES[model]
            dev_model = model
        if not bundle_dir.exists():
            log(f"  skip {model}: bundle missing")
            continue
        log(f"  validating {model} ...")
        if model == "LightGBM":
            ens, member_frames, _ = CS.lightgbm_ensemble(
                bundle_dir, imputed, transforms.climatology, wd, station_names,
                model_name=model, scope="conventional", feature_set="USGS", split="confirm")
        else:
            ens, member_frames, _ = CS.sequence_ensemble(
                bundle_dir, wd, model_name=model, scope="conventional",
                feature_set="USGS", device=device, split="confirm")
        ref = load_dev_predictions(multicore, dev_model)
        if ref.empty:
            log(f"    no dev reference for {dev_model}")
            diagnostics["models"][model] = {"error": "no dev reference"}
            continue
        # filter both to all-horizon-valid keys (dev test is already this set)
        member_filtered = [filter_to_keys(mf, keyset) for mf in member_frames]
        diag = CS.compare_to_reference(member_filtered, ref, atol=1e-2)
        log(f"    {model}: max_abs_diff={diag['max_abs_diff']:.6g} match={diag['match']}")
        diagnostics["models"][model] = diag
    return diagnostics


# --------------------------------------------------------------------------- #
# Holdout scoring
# --------------------------------------------------------------------------- #
def run_holdout(
    multicore: Path,
    registry_path: Path,
    device: str,
    max_stations: int | None,
    skip_external: bool,
    skip_ablations: bool,
) -> dict:
    log("=== HOLDOUT: 2021-2023 conventional evaluation ===")
    registry = load_registry(registry_path)
    if max_stations:
        registry = registry.head(max_stations).copy()
        log(f"  limiting to {len(registry)} stations (--max-stations)")
    store = SnapshotStore(OUT / "raw_2021_2023")

    panel_path = OUT / "panel_2021_2023.parquet"
    failures_path = OUT / "fetch_failures_2021_2023.json"
    if panel_path.exists():
        log(f"  loading cached panel {panel_path}")
        panel = pd.read_parquet(panel_path)
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
        if max_stations:
            keep = set(registry["site_no"].astype(str))
            panel = panel[panel["site_id"].astype(str).isin(keep)].copy()
    else:
        t0 = time.time()
        panel, failures = assemble_holdout_panel(registry, store, FETCH_START, FETCH_END)
        panel.to_parquet(panel_path, index=False)
        failures_path.write_text(json.dumps(failures, indent=2))
        log(f"  fetched panel in {time.time() - t0:.0f}s ({len(failures)} station failures)")
    station_ids = sorted(panel["site_id"].astype(str).unique())
    log(f"  panel: {len(panel)} rows, {len(station_ids)} stations")

    # --- temporal cohort (ThermoRoute + LightGBM + ablations) ---
    tr_dir = multicore / "outputs" / "models" / TEMPORAL_BUNDLES["ThermoRoute"]
    weights, tr_meta = __import__("thermoroute.checkpoint", fromlist=["load_inference_bundle"]).load_inference_bundle(tr_dir)
    wd, transforms, imputed = FI.build_frozen_confirmation_windows(
        panel, tr_meta, station_ids, interval=(HOLDOUT_START, HOLDOUT_END), external=False)
    station_names = list(C.STATIONS)
    log(f"  built {len(wd.X)} temporal holdout windows")

    model_frames: dict[str, pd.DataFrame] = {}
    # baselines (shared with temporal cohort)
    bases = CS.baseline_frames(wd, station_names, split="confirm")
    temporal_models = ["ThermoRoute", "LightGBM"]
    if not skip_ablations:
        temporal_models += ["DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
                            "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded"]
    for model in temporal_models:
        bundle_dir = multicore / "outputs" / "models" / TEMPORAL_BUNDLES[model]
        if not bundle_dir.exists():
            log(f"  skip {model}: bundle missing")
            continue
        log(f"  scoring {model} ...")
        if model == "LightGBM":
            ens, _, _ = CS.lightgbm_ensemble(
                bundle_dir, imputed, transforms.climatology, wd, station_names,
                model_name=model, scope="conventional", feature_set="USGS", split="confirm")
        else:
            ens, _, _ = CS.sequence_ensemble(
                bundle_dir, wd, model_name=model, scope="conventional",
                feature_set="USGS", device=device, split="confirm")
        model_frames[model] = ens

    # --- LSTM (same temporal cohort windows / transforms) ---
    lstm_dir = multicore / "outputs" / "models" / LSTM_BUNDLE
    if lstm_dir.exists():
        log("  scoring LSTM ...")
        ens, _, _ = CS.sequence_ensemble(
            lstm_dir, wd, model_name="LSTM", scope="conventional",
            feature_set="USGS", device=device, split="confirm")
        model_frames["LSTM"] = ens

    # --- external pooled cohort (separate pooled transforms) ---
    if not skip_external:
        ext_dir = multicore / "outputs" / "models" / EXTERNAL_BUNDLES["ThermoRoute-ext"]
        if ext_dir.exists():
            log("  building external pooled windows ...")
            _, ext_meta = __import__("thermoroute.checkpoint", fromlist=["load_inference_bundle"]).load_inference_bundle(ext_dir)
            ext_wd, ext_transforms, ext_imputed = FI.build_frozen_confirmation_windows(
                panel, ext_meta, station_ids, interval=(HOLDOUT_START, HOLDOUT_END), external=True)
            ext_names = list(C.STATIONS)
            for model, suffix in [("ThermoRoute-ext", "ThermoRoute-ext"),
                                  ("LSTM-ext", "LSTM-ext"),
                                  ("LightGBM-ext", "LightGBM-ext")]:
                bdir = multicore / "outputs" / "models" / EXTERNAL_BUNDLES[suffix]
                if not bdir.exists():
                    continue
                log(f"  scoring {model} ...")
                if "LightGBM" in model:
                    ens, _, _ = CS.lightgbm_ensemble(
                        bdir, ext_imputed, ext_transforms.climatology, ext_wd, ext_names,
                        model_name=model, scope="conventional", feature_set="USGS", split="confirm")
                else:
                    ens, _, _ = CS.sequence_ensemble(
                        bdir, ext_wd, model_name=model, scope="conventional",
                        feature_set="USGS", device=device, split="confirm")
                model_frames[model] = ens

    # --- metrics ---
    all_frames = {**model_frames}
    long = CS.compute_metrics_long(all_frames, bases, horizons=HORIZONS)
    long_path = OUT / "holdout_metrics_2021_2023.csv"
    long.to_csv(long_path, index=False)
    log(f"  wrote {long_path}")

    wide = CS.pivot_metrics(long)
    summary = {
        "holdout_interval": f"{HOLDOUT_START}..{HOLDOUT_END}",
        "fetch_start": FETCH_START,
        "n_stations_panel": len(station_ids),
        "station_failures": failures,
        "n_models": len(model_frames),
        "models": sorted(model_frames),
        "metrics_wide": wide.to_dict(orient="records"),
        "n_windows_temporal": int(len(wd.X)),
    }
    (OUT / "holdout_summary_2021_2023.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"  wrote {OUT / 'holdout_summary_2021_2023.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--multicore", default=str(REPO.parent / "thermoroute-water-temperature-multicore"),
                   help="sibling multicore worktree root (read-only frozen bundles)")
    p.add_argument("--panel", default=str(DATA_USGS / "panel_usgs_120v2.parquet"))
    p.add_argument("--registry", default=str(DATA_USGS / "station_registry_v1.csv"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-stations", type=int, default=None, help="limit stations (for testing)")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--skip-holdout", action="store_true")
    p.add_argument("--skip-external", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--validation-models", nargs="*",
                   default=["ThermoRoute", "LightGBM", "LSTM"],
                   help="models to reproduce on 2019-2020")
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    multicore = Path(args.multicore)
    panel_path, registry_path = Path(args.panel), Path(args.registry)

    if not args.skip_validation:
        diag = run_validation(multicore, panel_path, registry_path, args.device, args.validation_models)
        (OUT / "validation_2019_2020.json").write_text(json.dumps(diag, indent=2, default=str))
        log(f"  wrote {OUT / 'validation_2019_2020.json'}")

    if not args.skip_holdout:
        run_holdout(multicore, registry_path, args.device, args.max_stations,
                    args.skip_external, args.skip_ablations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
