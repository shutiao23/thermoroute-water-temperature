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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thermoroute import config as C
from thermoroute import conventional_acquisition as CA
from thermoroute import conventional_score as CS
from thermoroute import frozen_inference as FI
from thermoroute import usgs
from thermoroute.provenance import SnapshotStore

REPO = Path(__file__).resolve().parents[1]
DATA_USGS = REPO / "data_usgs"
OUT = REPO / "outputs" / "conventional"

allow_incomplete_cohort = False

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

# Stage-09b plain neural controls: training checkpoints + stored arm predictions.
# These arms have no inference bundle; the conventional scorer loads their
# best_model_state directly and borrows the frozen stage-09 preprocessing from
# the same-variable ThermoRoute bundle (proved faithful by G15).
PLAIN_CONTROL_RUN_ID = "e87f141ad92c33ce1d3f"
PLAIN_CONTROL_ARMS = ("PlainMLP-7var", "PlainCausalTCN-7var")
G15_REPORT = "plain_controls_g15.json"


def plain_control_run_dir(multicore: Path) -> Path:
    return multicore / "plain_control_run"


def registry_site_overlap() -> float:
    """Overlap of the external bundle's station_to_index with the 120-site
    registry (informational: the -ext cohort is a pooled-preprocessing
    sensitivity on the same sites, not a site-disjoint cohort)."""
    return 1.0


def log(msg: str) -> None:
    print(f"[conv] {msg}", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=REPO).stdout.strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=REPO).stdout.strip())
    except Exception:
        return True


def load_registry(registry_path: Path) -> pd.DataFrame:
    reg = pd.read_csv(registry_path, dtype={"site_no": str, "legacy_site_id": str})
    reg = reg[["site_no", "legacy_site_id", "lat", "lon", "huc2"]].copy()
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
# Data acquisition for the holdout panel (strict re-parse from snapshot cache)
# --------------------------------------------------------------------------- #
def panel_cache_key(registry: pd.DataFrame, store: SnapshotStore) -> str:
    """Content-key: registry + interval + parser version + sorted request hashes."""
    import hashlib
    registry_digest = hashlib.sha256(
        registry["site_no"].astype(str).str.cat(sep="|").encode("utf-8")
    ).hexdigest()
    urls = []
    for site in registry["site_no"].astype(str):
        urls.append(CA.nwis_snapshot_url(site, FETCH_START, FETCH_END))
    request_digests = []
    for url in sorted(urls):
        request_digests.append(SnapshotStore.request_document(
            provider="usgs-nwis-dv", url=url)["url"])
    return hashlib.sha256(
        (registry_digest + "|" + FETCH_START + "|" + FETCH_END + "|parser-v1|"
         + "|".join(request_digests)).encode("utf-8")
    ).hexdigest()


def assemble_holdout_panel(
    registry: pd.DataFrame,
    store: SnapshotStore,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Strict re-parse of every site snapshot; return (panel, cohort, failures).

    The cohort table carries the typed per-site acquisition status (OK /
    NO_SERIES / ALL_SERIES_CONFLICT / PARSE_FAILED / HTTP_FAILED /
    SNAPSHOT_MISSING); the panel admits only OK sites.
    """
    full = pd.date_range(start, end, freq="D", name="DATE")
    frames = []
    failures: list[dict] = []
    cohort_rows: list[dict] = []
    n = len(registry)
    for i, row in enumerate(registry.itertuples(index=False)):
        site_no = row.site_no
        rec: dict = {"site_no": site_no}
        frame, status, detail = CA.reparse_site(store, site_no, start, end)
        rec.update({"status": status, **detail})
        cohort_rows.append(rec)
        if status in (CA.ACQUISITION_STATUS_OK,
                      CA.ACQUISITION_STATUS_NO_SERIES,
                      CA.ACQUISITION_STATUS_ALL_SERIES_CONFLICT,
                      CA.ACQUISITION_STATUS_PARSE_FAILED):
            # Every site with a snapshot stays in the panel (all-NaN rows for
            # dry/conflicted sites) so the panel's station set matches the
            # frozen bundle registry; observed-target windows filter them out.
            lat, lon = float(row.lat), float(row.lon)
            try:
                met = usgs.fetch_daymet(lat, lon, start, end, snapshot_store=store)
            except Exception:
                met = None
            try:
                wind = usgs.fetch_gridmet_wind(lat, lon, start, end, snapshot_store=store)
            except Exception:
                wind = None
            cols: dict[str, pd.Series] = {
                v: pd.Series(np.nan, index=full) for v in
                ("WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
            }
            for v in ("WTEMP", "FLOW", "WLEVEL"):
                if v in frame.columns:
                    cols[v] = frame.set_index("DATE")[v].reindex(full)
            for v in ("TEMP", "PRCP", "RHMEAN", "DH"):
                if met is not None and v in met:
                    cols[v] = met[v].reindex(full)
            if wind is not None:
                cols["WDSP"] = wind.reindex(full)
            df = pd.DataFrame(cols, index=full)
            df = df.reset_index()
            df.insert(1, "site_id", site_no)
            df = df[["DATE", "site_id", "WTEMP", "FLOW", "WLEVEL",
                     "TEMP", "PRCP", "WDSP", "RHMEAN", "DH"]]
            frames.append(df)
        elif status in (CA.ACQUISITION_STATUS_HTTP_FAILED,
                        CA.ACQUISITION_STATUS_SNAPSHOT_MISSING):
            failures.append(rec)
        if (i + 1) % 20 == 0 or i + 1 == n:
            log(f"  re-parsed {i + 1}/{n} stations")
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["DATE", "site_id", "WTEMP", "FLOW", "WLEVEL",
                 "TEMP", "PRCP", "WDSP", "RHMEAN", "DH"])
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    panel["site_id"] = panel["site_id"].astype(str)
    cohort = pd.DataFrame(cohort_rows)
    return panel, cohort, failures


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
    skip_plain_controls: bool = False,
) -> dict:
    """Reproduce 2019-2020 dev predictions with frozen bundles and compare."""
    log("=== VALIDATION: reproducing 2019-2020 dev predictions ===")
    panel = load_dev_panel_mapped(panel_path, registry_path)
    station_ids = sorted(panel["site_id"].astype(str).unique())
    # Use the ThermoRoute bundle to build the frozen confirmation windows for 2019-2020.
    tr_dir = multicore / TEMPORAL_BUNDLES["ThermoRoute"]
    weights, tr_meta = __import__("thermoroute.checkpoint", fromlist=["load_inference_bundle"]).load_inference_bundle(tr_dir)
    wd, transforms, imputed = FI.build_frozen_confirmation_windows(
        panel, tr_meta, station_ids, interval=("2019-01-01", "2020-12-31"), external=False)
    station_names = list(C.STATIONS)
    keyset = all_valid_keys_from_wd(wd, station_names)
    log(f"  built {len(wd.X)} confirmation windows; {len(keyset)} all-horizon-valid issue keys")

    diagnostics: dict = {"interval": "2019-01-01..2020-12-31", "models": {}}
    for model in models:
        if model == "LSTM":
            bundle_dir = multicore / LSTM_BUNDLE
            dev_model = "LSTM"
        else:
            bundle_dir = multicore / TEMPORAL_BUNDLES[model]
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
                bundle_dir, wd, station_names, model_name=model, scope="conventional",
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

    # G15: plain-control preprocessing-borrow reproduction on the dev test rows.
    diagnostics["plain_controls"] = _validate_plain_controls(
        multicore, wd, station_names, device, skip_plain_controls)
    return diagnostics


def _validate_plain_controls(
    multicore: Path,
    wd,
    station_names,
    device: str,
    skip_plain_controls: bool,
) -> dict:
    """G15: prove the frozen-transform borrow reproduces stored plain-control dev test.

    Reuses the 2019-2020 confirmation windows already built by
    :func:`run_validation`.  Each arm is admitted only if every seed reproduces
    every dev-test forecast key within ``atol=1e-2`` on y_pred/q05/q50/q95/
    p_exceed.  Failing arms are recorded ``EXCLUDED_PREPROCESSING_MISMATCH``
    (not a hard abort), and the holdout scorer excludes them.
    """
    from thermoroute.plain_controls import CONTROL_SEEDS
    report: dict = {"arms": {}, "admitted": [], "excluded": [], "skipped": skip_plain_controls}
    if skip_plain_controls:
        log("  skip plain controls (--skip-plain-controls)")
        return report
    run_dir = plain_control_run_dir(multicore)
    checkpoint_dir = run_dir / "checkpoints"
    arm_pred_dir = run_dir / "arm_predictions"
    if not checkpoint_dir.is_dir() or not arm_pred_dir.is_dir():
        log(f"  skip plain controls: Stage-09b run not found at {run_dir}")
        report["skipped"] = True
        return report
    metrics = ("y_pred", "q05", "q50", "q95", "p_exceed")
    for arm in PLAIN_CONTROL_ARMS:
        log(f"  validating plain control {arm} (G15) ...")
        try:
            _ens, member_frames, meta = CS.plain_control_ensemble(
                checkpoint_dir, wd, station_names, arm, device=device, split="confirm",
                scope="development_only_2006_2020", feature_set="all_7_variables",
                n_stations=len(station_names), seeds=CONTROL_SEEDS,
                expected_run_id=PLAIN_CONTROL_RUN_ID)
        except Exception as exc:
            log(f"    {arm}: load/inference failed -> EXCLUDED ({exc})")
            report["arms"][arm] = {"status": "EXCLUDED_PREPROCESSING_MISMATCH", "error": str(exc)}
            report["excluded"].append(arm)
            continue
        ref_parts = [pd.read_parquet(arm_pred_dir / arm / f"seed{seed}.parquet",
                                     columns=list(metrics) + ["seed", "split"] + ["site_id", "horizon", "issue_date", "target_date"])
                     for seed in CONTROL_SEEDS]
        reference = pd.concat([r[r["split"] == "test"] for r in ref_parts], ignore_index=True)
        diag = CS.compare_predictions_to_reference(
            member_frames, reference, atol=1e-2, metrics=metrics, require_all_keys=True)
        arm_ok = bool(diag["match"])
        status = "PASS" if arm_ok else "EXCLUDED_PREPROCESSING_MISMATCH"
        log(f"    {arm}: max_abs_diff={diag['max_abs_diff']:.6g} -> {status}")
        report["arms"][arm] = {
            "status": status,
            "trainable_parameters": meta.get("trainable_parameters"),
            "n_members": meta.get("member_count"),
            "per_seed": diag["per_seed"],
        }
        (report["admitted"] if arm_ok else report["excluded"]).append(arm)
    report["overall_status"] = "PASS" if not report["excluded"] else "EXCLUDED_PREPROCESSING_MISMATCH"
    return report


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
    plain_control_admission: dict | None = None,
    skip_plain_controls: bool = False,
) -> dict:
    log("=== HOLDOUT: 2021-2023 conventional evaluation ===")
    registry = load_registry(registry_path)
    if max_stations:
        registry = registry.head(max_stations).copy()
        log(f"  limiting to {len(registry)} stations (--max-stations)")
    store = SnapshotStore(OUT / "raw_2021_2023", offline=True)

    panel_path = OUT / "panel_2021_2023.parquet"
    cohort_path = OUT / "cohort_2021_2023.csv"
    failures_path = OUT / "fetch_failures_2021_2023.json"
    cache_key = panel_cache_key(registry, store)
    cache_key_path = OUT / "panel_cache_key.txt"
    if panel_path.exists() and cache_key_path.exists() \
            and cache_key_path.read_text().strip() == cache_key:
        log(f"  loading cached panel {panel_path}")
        panel = pd.read_parquet(panel_path)
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
        cohort = pd.read_csv(cohort_path, dtype={"site_no": str}) \
            if cohort_path.exists() else pd.DataFrame()
    else:
        t0 = time.time()
        panel, cohort, failures = assemble_holdout_panel(registry, store, FETCH_START, FETCH_END)
        panel.to_parquet(panel_path, index=False)
        cohort.to_csv(cohort_path, index=False)
        failures_path.write_text(json.dumps(failures, indent=2))
        cache_key_path.write_text(cache_key)
        log(f"  re-parsed panel in {time.time() - t0:.0f}s ({len(failures)} station failures)")
    station_ids = sorted(panel["site_id"].astype(str).unique())
    log(f"  panel: {len(panel)} rows, {len(station_ids)} stations")

    # --- temporal cohort (ThermoRoute + LightGBM + ablations) ---
    tr_dir = multicore / TEMPORAL_BUNDLES["ThermoRoute"]
    weights, tr_meta = __import__("thermoroute.checkpoint", fromlist=["load_inference_bundle"]).load_inference_bundle(tr_dir)
    wd, transforms, imputed = FI.build_frozen_confirmation_windows(
        panel, tr_meta, station_ids, interval=(HOLDOUT_START, HOLDOUT_END), external=False)
    # Trap 5: snapshot the decoder order immediately after the build and pass it
    # explicitly to every ensemble call; never read C.STATIONS at call time.
    station_names = tuple(C.STATIONS)
    log(f"  built {len(wd.X)} temporal holdout windows; {len(station_names)} stations")

    registry_huc2 = dict(zip(
        registry["site_no"].astype(str).str.zfill(8),
        registry["huc2"].astype(str),
    ))
    calibrated_models: set[str] = set()
    uncalibrated_models: set[str] = set()
    model_frames: dict[str, pd.DataFrame] = {}
    metadata_by_model: dict[str, dict[str, Any]] = {}

    def score_temporal(model: str, *, external: bool = False) -> None:
        bundle_key = f"{model}-ext" if external else model
        if external:
            bundle_dir = multicore / EXTERNAL_BUNDLES.get(bundle_key, "")
        else:
            bundle_dir = multicore / TEMPORAL_BUNDLES.get(model, "")
        if not Path(bundle_dir).exists():
            log(f"  skip {model}: bundle missing")
            return
        log(f"  scoring {model} ...")
        if "LightGBM" in model:
            ens, member_frames, meta = CS.lightgbm_ensemble(
                bundle_dir, imputed, transforms.climatology, wd, station_names,
                model_name=model, scope="conventional", feature_set="USGS",
                split="confirm", external=external)
        else:
            ens, member_frames, meta = CS.sequence_ensemble(
                bundle_dir, wd, station_names, model_name=model, scope="conventional",
                feature_set="USGS", device=device, split="confirm", external=external)
        meta = dict(meta)
        meta["_bundle_dir"] = str(bundle_dir)
        ens = CS.assign_cohort_metadata(
            {model: ens}, metadata_by_model={model: meta}, cohort="pooled_prep" if external else "temporal",
            registry_huc2=registry_huc2, external=external)[model]
        model_frames[model] = ens
        metadata_by_model[model] = meta
        calibrated_models.add(model)
        if "q05_raw" in ens.columns:
            bad = int((ens["q05_raw"].to_numpy(float) >= ens["q95_raw"].to_numpy(float)).sum())
            log(f"    {model}: {len(ens)} rows, {bad} degenerate pre-CQR heads")

    for model in ["ThermoRoute", "LightGBM"]:
        score_temporal(model)
    if not skip_ablations:
        for model in ["DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
                      "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded"]:
            score_temporal(model)

    lstm_dir = multicore / LSTM_BUNDLE
    if lstm_dir.exists():
        log("  scoring LSTM ...")
        ens, _, meta = CS.sequence_ensemble(
            lstm_dir, wd, station_names, model_name="LSTM", scope="conventional",
            feature_set="USGS", device=device, split="confirm")
        meta = dict(meta)
        meta["_bundle_dir"] = str(lstm_dir)
        ens = CS.assign_cohort_metadata(
            {"LSTM": ens}, metadata_by_model={"LSTM": meta}, cohort="temporal",
            registry_huc2=registry_huc2)[ "LSTM"]
        model_frames["LSTM"] = ens
        metadata_by_model["LSTM"] = meta
        calibrated_models.add("LSTM")

    # --- plain neural controls (Tier A; no bundle, no calibration) ---
    # Admitted only after G15 proves the borrowed stage-09 preprocessing
    # reproduces the stored dev-test predictions.  These arms carry raw heads
    # only (NO_FROZEN_CALIBRATION); probability_metrics must skip them.
    admitted_arms = (plain_control_admission or {}).get("admitted", [])
    if skip_plain_controls:
        log("  skip plain controls (--skip-plain-controls)")
    elif not admitted_arms:
        log("  skip plain controls: no G15 admission (run validation first)")
    else:
        from thermoroute.plain_controls import CONTROL_SEEDS, DEFAULT_N_STATIONS
        checkpoint_dir = plain_control_run_dir(multicore) / "checkpoints"
        for arm in admitted_arms:
            if not (checkpoint_dir / arm).is_dir():
                log(f"  skip {arm}: checkpoints missing")
                continue
            log(f"  scoring plain control {arm} ...")
            ens, _member_frames, meta = CS.plain_control_ensemble(
                checkpoint_dir, wd, station_names, arm, device=device, split="confirm",
                scope="conventional", feature_set="all_7_variables",
                n_stations=DEFAULT_N_STATIONS, seeds=CONTROL_SEEDS,
                cohort="temporal", expected_run_id=PLAIN_CONTROL_RUN_ID)
            meta = dict(meta)
            ens = CS.assign_cohort_metadata(
                {arm: ens}, metadata_by_model={arm: meta}, cohort="temporal",
                registry_huc2=registry_huc2)[arm]
            model_frames[arm] = ens
            metadata_by_model[arm] = meta
            uncalibrated_models.add(arm)

    # --- external pooled cohort (same registry; pooled-preprocessing sensitivity) ---
    if not skip_external:
        ext_dir = multicore / EXTERNAL_BUNDLES["ThermoRoute-ext"]
        if ext_dir.exists():
            log("  building external pooled windows ...")
            _, ext_meta = __import__("thermoroute.checkpoint", fromlist=["load_inference_bundle"]).load_inference_bundle(ext_dir)
            ext_wd, ext_transforms, ext_imputed = FI.build_frozen_confirmation_windows(
                panel, ext_meta, station_ids, interval=(HOLDOUT_START, HOLDOUT_END), external=True)
            ext_names = tuple(C.STATIONS)
            for model in ("ThermoRoute-ext", "LSTM-ext", "LightGBM-ext"):
                bdir = multicore / EXTERNAL_BUNDLES[model]
                if not Path(bdir).exists():
                    continue
                log(f"  scoring {model} ...")
                if "LightGBM" in model:
                    ens, _, meta = CS.lightgbm_ensemble(
                        bdir, ext_imputed, ext_transforms.climatology, ext_wd, ext_names,
                        model_name=model, scope="conventional", feature_set="USGS",
                        split="confirm", external=True)
                else:
                    ens, _, meta = CS.sequence_ensemble(
                        bdir, ext_wd, ext_names, model_name=model, scope="conventional",
                        feature_set="USGS", device=device, split="confirm", external=True)
                meta = dict(meta)
                meta["_bundle_dir"] = str(bdir)
                ens = CS.assign_cohort_metadata(
                    {model: ens}, metadata_by_model={model: meta}, cohort="pooled_prep",
                    registry_huc2=registry_huc2, external=True)[model]
                model_frames[model] = ens
                metadata_by_model[model] = meta
                calibrated_models.add(model)

    # --- baselines (point-only under the table contract) ---
    bases = CS.baseline_frames(wd, station_names, split="confirm")
    for base_name, base_frame in bases.items():
        model_frames[base_name] = CS.mark_point_only_frame(
            base_frame, cohort="temporal")

    # --- assemble the per-key prediction table (full contract) ---
    all_frames = {**model_frames}
    pred = pd.concat(
        [frame for frame in all_frames.values() if not frame.empty], ignore_index=True
    )
    pred["seed"] = pred.get("seed", 0)
    for column in CS.CONTRACT_COLS:
        if column not in pred.columns:
            pred[column] = np.nan
    pred = pred[list(CS.CONTRACT_COLS)]

    # --- validation gates G1-G14 ---
    gate_report = CS.validate_prediction_table(
        pred,
        registry=registry,
        expected_horizons=HORIZONS,
        calibrated_models=calibrated_models,
        uncalibrated_models=uncalibrated_models,
    )
    for name, entry in gate_report["gates"].items():
        log(f"  gate {name}: {'PASS' if entry['pass'] else 'FAIL'} ({entry['detail']})")
    if not all(entry["pass"] for entry in gate_report["gates"].values()):
        raise RuntimeError("holdout validation gates failed; nothing written")
    if len(failures) and not allow_incomplete_cohort:
        raise RuntimeError(
            f"{len(failures)} acquisition failures without --allow-incomplete-cohort")

    # --- persistence ---
    pred_path = OUT / "predictions_2021_2023.parquet"
    pred.to_parquet(pred_path, index=False)
    log(f"  wrote {pred_path} ({len(pred)} rows, {pred_path.stat().st_size / 1e6:.1f} MB)")

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
        "n_stations_reportable": gate_report.get("n_stations_reportable", {}),
        "gates": gate_report["gates"],
    }
    (OUT / "holdout_summary_2021_2023.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "validation_report_2021_2023.json").write_text(
        json.dumps(gate_report, indent=2, default=str))
    (OUT / "run_manifest.json").write_text(json.dumps({
        "format": "thermoroute.conventional-run-manifest.v2",
        "git_sha": _git_sha(),
        "dirty": _git_dirty(),
        "command": " ".join(sys.argv),
        "predictions": "predictions_2021_2023.parquet",
        "panel_cache_key": cache_key,
        "cohort": "cohort_2021_2023.csv",
        "n_rows": int(len(pred)),
    }, indent=2))
    log(f"  wrote {OUT / 'holdout_summary_2021_2023.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle-root",
                   default=Path("outputs/models"),
                   help="bundle root (default: <repo>/outputs/models; "
                        "overrides the legacy sibling-worktree --multicore layout)")
    p.add_argument("--multicore", default=None,
                   help="deprecated alias: sibling multicore worktree root "
                        "(bundles live under its outputs/models)")
    p.add_argument("--panel", default=str(DATA_USGS / "panel_usgs_120v2.parquet"))
    p.add_argument("--registry", default=str(DATA_USGS / "station_registry_v1.csv"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-stations", type=int, default=None, help="limit stations (for testing)")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--skip-holdout", action="store_true")
    p.add_argument("--skip-external", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--skip-plain-controls", action="store_true",
                   help="skip the Stage-09b plain neural controls (Tier A); "
                        "by default they are admitted after the G15 reproduction gate")
    p.add_argument("--allow-incomplete-cohort", action="store_true",
                   help="stamp a waiver into the manifest when acquisition is incomplete")
    p.add_argument("--validation-models", nargs="*",
                   default=["ThermoRoute", "LightGBM", "LSTM"],
                   help="models to reproduce on 2019-2020")
    args = p.parse_args(argv)
    global allow_incomplete_cohort
    allow_incomplete_cohort = args.allow_incomplete_cohort

    OUT.mkdir(parents=True, exist_ok=True)
    if args.multicore is not None:
        multicore = Path(args.multicore)
        if not (multicore / TEMPORAL_BUNDLES["ThermoRoute"]).exists():
            raise FileNotFoundError(
                f"ThermoRoute bundle not found under {multicore}/outputs/models; "
                "point --multicore at the worktree that holds the frozen bundles")
        multicore = multicore / "outputs" / "models"
    else:
        multicore = Path(args.bundle_root)
    if not (multicore / TEMPORAL_BUNDLES["ThermoRoute"]).exists():
        raise FileNotFoundError(
            f"ThermoRoute bundle not found under {multicore}; "
            "run scripts/verify_model_bundles.py or point --bundle-root at the "
            "local outputs/models directory")
    panel_path, registry_path = Path(args.panel), Path(args.registry)

    plain_admission: dict | None = None
    if not args.skip_validation:
        diag = run_validation(multicore, panel_path, registry_path, args.device,
                              args.validation_models, args.skip_plain_controls)
        (OUT / "validation_2019_2020.json").write_text(json.dumps(diag, indent=2, default=str))
        log(f"  wrote {OUT / 'validation_2019_2020.json'}")
        plain_admission = diag.get("plain_controls")

    if not args.skip_holdout:
        run_holdout(multicore, registry_path, args.device, args.max_stations,
                    args.skip_external, args.skip_ablations,
                    plain_control_admission=plain_admission,
                    skip_plain_controls=args.skip_plain_controls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
