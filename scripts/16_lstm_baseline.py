#!/usr/bin/env python3
"""Stage 16 — the field-standard deep sequence baseline (global LSTM).

Adds a recurrent sequence reference used widely in stream-temperature ML.  Its
training and seed budgets are matched to ThermoRoute so the comparison does not
silently favour either neural architecture.

The same-station arm gives the global LSTM the same 32-day history and a learned
station embedding, matching ThermoRoute's information set.  The held-region arm
is explicitly station-agnostic so it can score sites whose identities were not
seen in fitting.  A predeclared validation-only grid chooses whether its head
uses the frozen climatology, damped anchor and season features already available
to ThermoRoute; calibration/test rows never make that choice.  Heads, composite
loss, splits and budget remain matched; the LSTM is δ-free.
Both models receive the same post-hoc conformal and event-calibration procedures;
the LSTM lacks ThermoRoute's bounded-residual constraint and lag router.  No
result is assumed before the experiment runs.

  --insample   train LSTM × USGS_SEEDS on the full 120-station panel; derive
               final usgs_predictions_v2.parquet from the immutable
               usgs_predictions_with_perstation_v2.parquet parent
  --transfer   train one LSTM per leave-HUC2-region-out fold; checkpoint held-out
               predictions to predictions/region_ckpt/lstm_ctx32_fold{i}.parquet
  --report     3-way (ThermoRoute vs LightGBM vs LSTM) region-transfer table +
               in-sample headline row -> outputs/reports/lstm_baseline.md

Run:  python3 scripts/16_lstm_baseline.py --insample
      python3 scripts/16_lstm_baseline.py --transfer
      python3 scripts/16_lstm_baseline.py --report
"""
from __future__ import annotations

import os
for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Stage 13c is imported below and uses this value when setting Torch threads.
# Keep its default aligned with the formal single-thread contract while still
# failing closed if a caller explicitly requests a different worker count.
os.environ.setdefault("WORKER_THREADS", "1")

import argparse
import importlib.util
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage16-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE16_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE16_NONCE"


def _isolate_project_bytecode() -> None:
    if __name__ != "__main__":
        return
    worker_cache = os.environ.get(_WORKER_CACHE_ENV)
    worker_nonce = os.environ.get(_WORKER_NONCE_ENV)
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    worker_argument = len(sys.argv) > 1 and sys.argv[1] == _WORKER_ARGUMENT
    if worker_cache is not None or worker_nonce is not None or worker_argument:
        if not (worker_cache and worker_nonce and worker_argument):
            raise RuntimeError("Stage 16 formal worker handshake is incomplete")
        expected = Path(worker_cache).resolve()
        flags = (
            int(sys.flags.isolated), int(sys.flags.ignore_environment),
            int(sys.flags.no_user_site), bool(sys.flags.safe_path),
            int(sys.flags.dont_write_bytecode),
        )
        if (
            flags != (1, 1, 1, True, 0)
            or prefix != expected
            or not expected.is_dir()
            or expected == ROOT
            or ROOT in expected.parents
            or (expected / ".controller-nonce").read_text(encoding="utf-8")
            != worker_nonce
        ):
            raise RuntimeError("Stage 16 formal worker isolation contract failed")
        sys.argv.pop(1)
        return
    with tempfile.TemporaryDirectory(prefix="thermoroute-stage16-pycache-") as cache:
        cache_path = Path(cache).resolve()
        if any(cache_path.iterdir()):
            raise RuntimeError("Stage 16 controller pycache was not initially empty")
        nonce = secrets.token_hex(32)
        (cache_path / ".controller-nonce").write_text(nonce, encoding="utf-8")
        environment = os.environ.copy()
        environment[_WORKER_CACHE_ENV] = str(cache_path)
        environment[_WORKER_NONCE_ENV] = nonce
        result = subprocess.run(
            [sys.executable, "-I", "-X", f"pycache_prefix={cache}",
             str(Path(__file__).resolve()), _WORKER_ARGUMENT, *sys.argv[1:]],
            cwd=ROOT,
            env=environment,
            check=False,
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

torch.set_num_threads(1)

from thermoroute import config as C
from thermoroute import conformal as CF
from thermoroute import data as D
from thermoroute import results as R
from thermoroute.checkpoint import load_inference_bundle, save_inference_bundle
from thermoroute.frozen_inference import lstm_factory_from_metadata
from thermoroute.model_suite import (
    ModelSuiteError,
    LSTM_VALIDATION_GRID,
    canonical_development_contract,
    development_prediction_binding,
    file_binding,
    sequence_bundle_metadata,
    torch_entry,
    update_torch_development_prediction,
    verify_sequence_prediction_parity,
    write_component_pointer,
)
from thermoroute.probability import (
    fit_frozen_seasonal_event_reference,
    fit_horizon_calibrators,
)
from thermoroute.registry import (
    ROUTE_A_PRIMARY_MODELS,
    enforce_common_forecast_keys,
)
from thermoroute.repro import (
    assert_formal_numerical_policy,
    cache_is_valid,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from thermoroute.train import (
    LSTMForecaster,
    configure_deterministic_runtime,
    fit_model,
)

configure_deterministic_runtime()

# Reuse 13c's exact fold packing / prep / LightGBM-per-fold so the transfer arm
# is identical to ThermoRoute's (same regions, same in-fold stations).
_spec = importlib.util.spec_from_file_location(
    "region13c", ROOT / "scripts" / "13c_region_transfer.py")
R13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R13)

USGS_VARS = R13.USGS_VARS
# Matched optimisation and seed budget: the LSTM receives the same batch size,
# epoch cap, patience, station-balanced sampler and five seeds as ThermoRoute.
CFG = C.TrainConfig(batch_size=1536)
SEEDS = C.USGS_SEEDS
CKPT = C.PREDICTIONS / "lstm_ckpt_route_a_v2_ctx32_stationembed"
REGION_CKPT = R13.CKPT
PARENT = C.PREDICTIONS / "usgs_predictions_with_perstation_v2.parquet"
V2 = C.PREDICTIONS / "usgs_predictions_v2.parquet"
_t0 = time.time()


def log(m): print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


def _verify_parent(path: Path) -> dict:
    """Verify the exact immutable parent before producing the final derivative."""
    try:
        metadata = validate_artifact_sidecar(
            path, schema=R.PREDICTION_SCHEMA_VERSION
        )
    except ValueError as exc:
        raise FileNotFoundError(
            f"derived Stage-9 predictions and lineage sidecar are required: {path}"
        ) from exc
    return metadata


def _ensemble_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "scope", "feature_set", "site_id", "horizon", "split",
            "issue_date", "target_date"]
    return predictions.groupby(keys, as_index=False, dropna=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        q05=("q05", "mean"), q50=("q50", "mean"), q95=("q95", "mean"),
        p_exceed=("p_exceed", "mean"),
    )


def _calibration_artifacts(predictions: pd.DataFrame, thresholds: dict[str, float]):
    calibration = _ensemble_prediction_frame(predictions)
    calibration = calibration[calibration.split.eq("calib")].copy()
    calibration["threshold"] = calibration.site_id.astype(str).map(thresholds)
    if calibration["threshold"].isna().any():
        raise KeyError("LSTM calibration contains an unknown site")
    calibration["event"] = (
        calibration.y_true.to_numpy(float)
        > calibration.threshold.to_numpy(float)
    ).astype(int)
    offsets = CF.cqr_offsets(calibration, alpha=0.10)
    calibrators = fit_horizon_calibrators(
            calibration, probability_col="p_exceed", outcome_col="event",
            min_samples=100,
        )
    expected = {(str(site), int(horizon)) for site in thresholds
                for horizon in C.HORIZONS}
    if set(offsets) != expected:
        raise ValueError("LSTM calibration lacks the complete site×horizon registry")
    if set(calibrators) != set(C.HORIZONS):
        raise ValueError("LSTM event calibration lacks a declared horizon")
    return offsets, calibrators


def _read_member_bundle(directory: Path, identity, member: str):
    try:
        weights, metadata = load_inference_bundle(directory, expected_member_count=1)
    except (FileNotFoundError, ValueError, RuntimeError):
        return None
    if (
        metadata.get("run_id") != identity.run_id
        or metadata.get("source_sha256") != identity.source_sha256
        or metadata.get("panel_sha256") != identity.panel_sha256
        or metadata.get("registry_sha256") != identity.registry_sha256
        or metadata.get("config_sha256") != identity.config_sha256
        or metadata.get("runtime_sha256") != identity.runtime_sha256
        or set(weights) != {member}
    ):
        return None
    return weights[member]


# --------------------------------------------------------------------------- #
# In-sample: LSTM × seeds on all 120 stations, spliced into v2
# --------------------------------------------------------------------------- #
def insample():
    runtime_policy = assert_formal_numerical_policy()
    parent_lineage = _verify_parent(PARENT)
    parent_sha256 = sha256_file(PARENT)
    run_config = {
        "stage": "16_lstm_baseline_insample",
        "role": "final_route_a_development_predictions",
        "parent_sha256": parent_sha256,
        "models": ROUTE_A_PRIMARY_MODELS,
        "seeds": SEEDS,
        "variables": USGS_VARS,
        "horizons": C.HORIZONS,
        "context_length": C.CONTEXT_LENGTH,
        "station_embedding": True,
        "station_balanced": True,
        "selection_metric": "station_macro",
        "validation_grid": LSTM_VALIDATION_GRID,
        "validation_selection_seed": SEEDS[0],
        "validation_selection_split": "2016-2017 only",
        "event_reference_fit_interval": ("2006-01-01", "2018-12-31"),
        "train_config": CFG,
        "training_device": "cpu",
        "formal_numerical_policy": runtime_policy,
    }
    identity = resolve_run_identity(
        root=ROOT,
        panel=R13.PANEL.resolve(),
        registry=R13.STATION_REGISTRY.resolve(),
        config=run_config,
    )
    run_dir = initialise_run_directory(
        ROOT / "outputs" / "runs" / "16_lstm_baseline", identity, run_config,
        provenance={
            "evidence_role": "prelabel_route_a_model_build_development_only",
            "training_device": "cpu",
        },
    )
    # Lock the exact content-addressed run before dataset materialisation or
    # any checkpoint/cache path can be reached.
    panel, panel_imp, masks, clim, thr, wd, stations = R13.prep()
    event_reference = fit_frozen_seasonal_event_reference(
        panel,
        thr,
        pooled=False,
        fit_interval=("2006-01-01", "2018-12-31"),
    )
    prepared = D.prepare_dataset_from_panel(str(R13.PANEL.resolve()))
    imputer = prepared["imputer"]
    if tuple(prepared["stations"]) != tuple(stations):
        raise AssertionError("LSTM imputer and window station registries differ")
    log(f"{len(stations)} stations | windows N={len(wd.X)}")

    # Architecture selection is a small predeclared grid.  Candidates export
    # validation rows only; calibration and test cannot affect the choice.
    selection_rows = []
    candidates = []
    for candidate_id, candidate in enumerate(LSTM_VALIDATION_GRID):
        factory = lambda candidate=candidate: LSTMForecaster(
            n_vars=len(wd.var_names), n_stations=len(stations),
            context=C.CONTEXT_LENGTH, station_agnostic=False, **candidate,
        )
        result = fit_model(
            factory, wd, thr, cfg=CFG, seed=SEEDS[0],
            model_name=f"LSTM-grid-{candidate_id}", scope="validation_selection",
            feature_set="USGS", station_balanced=True,
            selection_metric="station_macro", export_splits=("val",),
            device="cpu",
            checkpoint_path=run_dir / "selection" / f"candidate{candidate_id}.pt",
            run_id=identity.run_id,
            resolved_config={**run_config, "candidate_id": candidate_id,
                             "candidate": candidate},
        )
        selection_rows.append({
            "candidate_id": candidate_id, **candidate,
            "val_station_macro_rmse": result.best_val,
            "selected": False, "selection_split": "2016-2017 validation",
        })
        candidates.append((result.best_val, candidate_id, candidate))
    _, selected_id, selected = min(candidates, key=lambda value: (value[0], value[1]))
    selection_rows[selected_id]["selected"] = True
    C.TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(
        C.TABLES / "lstm_validation_selection.csv", index=False
    )
    architecture_kwargs = {
        "n_vars": len(wd.var_names), "n_stations": len(stations),
        "context": C.CONTEXT_LENGTH, "station_agnostic": False, **selected,
    }

    preds = []
    ensemble_members = {}
    for sd in SEEDS:
        member = f"seed{sd}"
        f = run_dir / "predictions" / f"{member}.parquet"
        bundle = run_dir / "member_bundles" / member
        cached = None
        if cache_is_valid(f, identity, schema=R.PREDICTION_SCHEMA_VERSION):
            try:
                cached = pd.read_parquet(f)
                R.validate_predictions(cached)
            except Exception:
                cached = None
        cached_weights = _read_member_bundle(bundle, identity, member)
        if cached is not None and cached_weights is not None:
            preds.append(cached)
            ensemble_members[member] = cached_weights
            log(f"LSTM {member}: verified content cache")
            continue
        te = time.time()
        factory = lambda: LSTMForecaster(**architecture_kwargs)
        r = fit_model(factory, wd, thr, cfg=CFG, seed=sd, model_name="LSTM",
                      scope="joint_usgs", feature_set="USGS", verbose=True,
                      device="cpu",
                      station_balanced=True, selection_metric="station_macro",
                      checkpoint_path=run_dir / "checkpoints" / f"{member}.pt",
                      run_id=identity.run_id,
                      resolved_config={**run_config, "selected_candidate": selected,
                                       "arm": "LSTM", "seed": sd})
        r.pred["seed"] = sd
        R.write_predictions(r.pred, f)
        seal_artifact(
            f, identity, kind="lstm_seed_predictions",
            schema=R.PREDICTION_SCHEMA_VERSION,
        )
        member_offsets, member_calibrators = _calibration_artifacts(r.pred, thr)
        save_inference_bundle(
            bundle, members={member: r.model},
            metadata=sequence_bundle_metadata(
                run_id=identity.run_id,
                architecture_class="thermoroute.train.LSTMForecaster",
                architecture_kwargs=architecture_kwargs, train_config=CFG,
                wd=wd, climatology=clim, imputer=imputer, thresholds=thr,
                event_reference_climatology=event_reference,
                conformal_offsets=member_offsets,
                event_calibrators=member_calibrators,
                source_sha256=identity.source_sha256,
                panel_sha256=identity.panel_sha256,
                registry_sha256=identity.registry_sha256,
                config_sha256=identity.config_sha256,
                runtime_sha256=identity.runtime_sha256,
                training_device="cpu",
                development_prediction={},
            ), expected_member_count=1,
        )
        ensemble_members[member] = {
            key: value.detach().cpu().contiguous()
            for key, value in r.model.state_dict().items()
        }
        preds.append(r.pred)
        log(f"LSTM seed{sd}: {r.epochs+1}ep {time.time()-te:.0f}s val_rmse={r.best_val:.4f}")
    lstm = pd.concat(preds, ignore_index=True)

    # Derive, never mutate, the final artifact.  The six-model registry is a
    # protocol constant; optional exploratory rows from the parent cannot alter
    # which examples enter the primary comparison.  Calibration rows remain
    # available for the predeclared post-hoc wrappers.
    allp = pd.read_parquet(PARENT)
    R.validate_predictions(allp)
    allp = allp[allp.model != "LSTM"]
    lt = lstm[lstm.split == "test"].copy()
    lc = lstm[lstm.split == "calib"].copy()          # for the conformal wrapper
    allp = pd.concat([allp, lt, lc], ignore_index=True)
    allp, audit = enforce_common_forecast_keys(
        allp, ROUTE_A_PRIMARY_MODELS, split="test"
    )
    R.write_predictions(allp, V2)
    seal_artifact(
        V2,
        identity,
        kind="final_route_a_development_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents={PARENT.name: parent_sha256},
        extra={
            "parent_run_id": parent_lineage.get("run", {}).get("run_id"),
            "primary_models": ROUTE_A_PRIMARY_MODELS,
            "primary_common_test_keys": audit.common_unique,
            "dropped_primary_rows": audit.dropped_rows,
            "lstm_calibration_rows": len(lc),
        },
    )
    log(f"derived final v2: common={audit.common_unique}, "
        f"dropped={audit.dropped_rows}; {len(lc)} calib rows retained")

    lstm_rows = allp[allp.model.eq("LSTM")]
    offsets, calibrators = _calibration_artifacts(lstm, thr)
    bundle_directory = C.MODELS / f"lstm_usgs_bundle_{identity.run_id}"
    parity_atol = 1e-5
    save_inference_bundle(
        bundle_directory, members=ensemble_members,
        metadata=sequence_bundle_metadata(
            run_id=identity.run_id,
            architecture_class="thermoroute.train.LSTMForecaster",
            architecture_kwargs=architecture_kwargs, train_config=CFG,
            wd=wd, climatology=clim, imputer=imputer, thresholds=thr,
            event_reference_climatology=event_reference,
            conformal_offsets=offsets, event_calibrators=calibrators,
            source_sha256=identity.source_sha256,
            panel_sha256=identity.panel_sha256,
            registry_sha256=identity.registry_sha256,
            config_sha256=identity.config_sha256,
            runtime_sha256=identity.runtime_sha256,
            training_device="cpu",
            development_prediction=development_prediction_binding(
                ROOT, V2, lstm_rows,
                max_abs_difference=parity_atol, atol=parity_atol,
            ),
        ), expected_member_count=len(SEEDS),
    )
    difference = verify_sequence_prediction_parity(
        bundle_directory, wd=wd, expected=lstm_rows,
        model_factory=lambda _member, metadata: lstm_factory_from_metadata(metadata),
        member_seeds={f"seed{seed}": seed for seed in SEEDS},
        atol=parity_atol, splits=("calib", "test"),
    )
    update_torch_development_prediction(
        bundle_directory,
        development_prediction_binding(
            ROOT, V2, lstm_rows,
            max_abs_difference=difference, atol=parity_atol,
        ),
    )
    try:
        development_contract = canonical_development_contract(
            ROOT, ROOT / "data_usgs" / "frozen_panel_v1.json",
            panel_sha256=identity.panel_sha256,
            registry_sha256=identity.registry_sha256,
            source_sha256=identity.source_sha256,
        )
    except ModelSuiteError:
        development_contract = None
    if development_contract is not None and len(ensemble_members) == 5:
        entry = torch_entry(
            ROOT, model_id="LSTM", executor="lstm_bundle",
            directory=bundle_directory, member_count=5,
            raw_feature_order=wd.var_names,
        )
        from thermoroute.repro import atomic_write_json
        atomic_write_json(C.MODELS / "lstm_usgs_bundle.json", {
            "run_id": identity.run_id,
            "bundle_path": bundle_directory.relative_to(ROOT).as_posix(),
            "member_count": 5,
            "metadata_sha256": sha256_file(bundle_directory / "metadata.json"),
            "weights_sha256": sha256_file(bundle_directory / "weights.pt"),
        })
        write_component_pointer(
            C.MODELS / "route_a_lstm_components.json",
            run_id=identity.run_id, cohort="temporal_lstm", entries=[entry],
            raw_feature_order=wd.var_names,
            development_contract=development_contract,
            development_prediction_artifact={
                **file_binding(ROOT, V2),
                "sidecar": file_binding(ROOT, sidecar_path(V2)),
            },
        )
        log("saved formal five-member LSTM bundle and component pointer")
    else:
        log("saved diagnostic LSTM bundle; formal component pointer unchanged")

    # headline: 5-seed ensemble median per-station RMSE vs ThermoRoute
    def ens_rmse(model, h):
        s = allp[(allp.model == model) & (allp.split == "test") & (allp.horizon == h)]
        s = s.groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        return {st: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
                for st, g in s.groupby("site_id")}
    for h in C.HORIZONS:
        lp, tp = ens_rmse("LSTM", h), ens_rmse("ThermoRoute", h)
        comm = [s for s in lp if s in tp]
        a = np.array([tp[s] for s in comm]); b = np.array([lp[s] for s in comm])
        p = wilcoxon(a, b).pvalue
        log(f"  h{h}: LSTM median RMSE {np.median(list(lp.values())):.3f} vs "
            f"TR {np.median(list(tp.values())):.3f} | TR-vs-LSTM paired p={p:.2g} "
            f"| TR wins {100*np.mean(a < b):.0f}%")


# --------------------------------------------------------------------------- #
# Transfer: one LSTM per leave-HUC2-region-out fold
# --------------------------------------------------------------------------- #
def transfer(fold=None):
    REGION_CKPT.mkdir(parents=True, exist_ok=True)
    panel, panel_imp, masks, clim, thr, wd, stations = R13.prep()
    folds, _ = R13.region_folds(stations)
    todo = range(len(folds)) if fold is None else [fold]
    for fi in todo:
        f = REGION_CKPT / f"lstm_ctx32_fold{fi}.parquet"
        if f.exists():
            log(f"LSTM fold{fi}: already done"); continue
        _, _, _, thr, wd, stations, train_st, hold = R13.prep_fold(fi)
        te = time.time()
        log(f"LSTM fold{fi}: train {len(train_st)} -> hold {len(hold)} region stations")
        factory = lambda: LSTMForecaster(
            n_vars=len(wd.var_names), n_stations=len(stations),
            context=C.CONTEXT_LENGTH, station_agnostic=True)
        r = fit_model(factory, wd, thr, cfg=CFG, seed=0, scope="region_lgo",
                      feature_set="USGS", train_stations=train_st,
                      device="cpu",
                      station_balanced=True, selection_metric="station_macro")
        pred = r.pred[(r.pred.split == "test") & (r.pred.site_id.isin(hold))].copy()
        pred["model"] = "LSTM-regionLGO"
        pred.to_parquet(f)
        log(f"LSTM fold{fi}: DONE {r.epochs+1}ep {time.time()-te:.0f}s -> {f.name}")


# --------------------------------------------------------------------------- #
# Report: 3-way transfer table + in-sample headline
# --------------------------------------------------------------------------- #
def report():
    panel, panel_imp, masks, clim, thr, wd, stations = R13.prep()
    folds, _ = R13.region_folds(stations)
    TR = pd.concat([pd.read_parquet(REGION_CKPT / f"tr_fold{fi}.parquet")
                    for fi in range(len(folds))], ignore_index=True)
    LGB = pd.concat([pd.read_parquet(REGION_CKPT / f"lgb_fold{fi}.parquet")
                     for fi in range(len(folds))], ignore_index=True)
    lstm_files = [REGION_CKPT / f"lstm_ctx32_fold{fi}.parquet"
                  for fi in range(len(folds))]
    if not all(f.exists() for f in lstm_files):
        log("LSTM transfer folds missing — run --transfer first"); return
    LSTM = pd.concat([pd.read_parquet(f) for f in lstm_files], ignore_index=True)

    v2 = pd.read_parquet(V2)
    base = v2[(v2.split == "test") & v2.model.isin(["Persistence", "DampedPersistence"])]
    aligned, audit = enforce_common_forecast_keys(
        pd.concat([TR, LGB, LSTM, base], ignore_index=True),
        ("ThermoRoute-regionLGO", "LightGBM-regionLGO",
         "LSTM-regionLGO", "Persistence"), split="test")
    log(f"3-way region registry: common={audit.common_unique}, "
        f"dropped={audit.dropped_rows}")
    TR = aligned[aligned.model == "ThermoRoute-regionLGO"]
    LGB = aligned[aligned.model == "LightGBM-regionLGO"]
    LSTM = aligned[aligned.model == "LSTM-regionLGO"]
    tr_r, lgb_r, lstm_r = R13.ps_rmse(TR), R13.ps_rmse(LGB), R13.ps_rmse(LSTM)

    L = ["# Deep sequence baseline (global LSTM) — development + held-region transfer\n",
         "The same-station top-down LSTM (1 layer, hidden 64, 32-day context, "
         "station embedding, validation-selected persistence-or-damped anchor) uses the same development splits, "
         "loss, station-balanced sampling, epoch/patience limits, history length, "
         "site identity and five-seed budget as ThermoRoute. Its predeclared "
         "validation-only grid may also expose frozen climatology, damped-anchor "
         "and season features; calibration/test data do not select the winner. "
         "Held-region LSTM folds "
         "disable the station embedding. All models are compared on exact common keys. "
         "The LSTM is eligible for the same conformal and probability calibration; "
         "it lacks ThermoRoute's bounded-residual constraint and lag router.\n",
         "## Leave-HUC2-region-out gauged transfer (whole regions held out)\n",
         "Issue-time WTEMP history remains available, so this is not ungauged. "
         "The p-values below are unadjusted exploratory paired-station diagnostics; "
         "primary multiplicity-controlled claims are produced by stage 12.\n",
         "| horizon | n | TR RMSE | LGB RMSE | LSTM RMSE | TR−LSTM p (expl.) | LGB−LSTM p (expl.) | lowest median |",
         "|---|---|---|---|---|---|---|---|"]
    for h in C.HORIZONS:
        sts = sorted(s for s in stations
                     if (s, h) in tr_r and (s, h) in lgb_r and (s, h) in lstm_r)
        a = np.array([tr_r[(s, h)] for s in sts])
        b = np.array([lgb_r[(s, h)] for s in sts])
        c = np.array([lstm_r[(s, h)] for s in sts])
        p_tl = wilcoxon(a, c).pvalue if len(sts) > 5 else float("nan")
        p_gl = wilcoxon(b, c).pvalue if len(sts) > 5 else float("nan")
        meds = {"TR": np.median(a), "LGB": np.median(b), "LSTM": np.median(c)}
        best = min(meds, key=meds.get)
        L.append(f"| {h} | {len(sts)} | {np.median(a):.3f} | {np.median(b):.3f} | "
                 f"{np.median(c):.3f} | {p_tl:.2g} | {p_gl:.2g} | **{best}** |")

    # in-sample headline (5-seed ensembles from v2)
    def ens_rmse(model, h):
        s = v2[(v2.model == model) & (v2.split == "test") & (v2.horizon == h)]
        s = s.groupby(["site_id", "issue_date"]).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first")).reset_index()
        return {st: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
                for st, g in s.groupby("site_id")}
    L += ["", "## Same-station development evaluation (5-seed ensembles)\n",
          "| horizon | persist | LightGBM | LSTM | ThermoRoute | TR−LSTM p (expl.) | TR win rate |",
          "|---|---|---|---|---|---|---|"]
    for h in C.HORIZONS:
        pe = ens_rmse("Persistence", h); lg = ens_rmse("LightGBM", h)
        ls = ens_rmse("LSTM", h); tr = ens_rmse("ThermoRoute", h)
        comm = sorted(s for s in ls if s in tr)
        a = np.array([tr[s] for s in comm]); c = np.array([ls[s] for s in comm])
        p = wilcoxon(a, c).pvalue
        L.append(f"| {h} | {np.median(list(pe.values())):.3f} | "
                 f"{np.median(list(lg.values())):.3f} | {np.median(list(ls.values())):.3f} | "
                 f"{np.median(list(tr.values())):.3f} | {p:.2g} | {100*np.mean(a<c):.0f}% |")

    out = C.REPORTS / "lstm_baseline.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--insample", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.insample:
        insample()
    elif a.transfer:
        transfer(a.fold)
    elif a.report:
        report()
    else:
        print(__doc__)
