#!/usr/bin/env python3
"""Train the station-agnostic Route-A suite on development data only.

Despite its name, this command never reads the frozen 30-site cohort and never
opens post-2020 data.  It fits pooled transforms and station-agnostic models on
the canonical 120-site development panel.  The resulting bundles can later
expand those pooled statistics to any separately frozen external site IDs.

Run: python3 scripts/25_train_external_pooled_suite.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time

for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage25-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE25_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE25_NONCE"


def _isolate_project_bytecode() -> None:
    if __name__ != "__main__":
        return
    worker_cache = os.environ.get(_WORKER_CACHE_ENV)
    worker_nonce = os.environ.get(_WORKER_NONCE_ENV)
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    worker_argument = len(sys.argv) > 1 and sys.argv[1] == _WORKER_ARGUMENT
    if worker_cache is not None or worker_nonce is not None or worker_argument:
        if not (worker_cache and worker_nonce and worker_argument):
            raise RuntimeError("Stage 25 formal worker handshake is incomplete")
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
            raise RuntimeError("Stage 25 formal worker isolation contract failed")
        sys.argv.pop(1)
        return
    with tempfile.TemporaryDirectory(prefix="thermoroute-stage25-pycache-") as cache:
        cache_path = Path(cache).resolve()
        if any(cache_path.iterdir()):
            raise RuntimeError("Stage 25 controller pycache was not initially empty")
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

from thermoroute import config as C
from thermoroute import conformal as CF
from thermoroute import data as D
from thermoroute import datasets as DS
from thermoroute import features as F
from thermoroute import results as R
from thermoroute.checkpoint import save_inference_bundle
from thermoroute.frozen_inference import (
    lstm_factory_from_metadata,
    thermoroute_factory_from_metadata,
)
from thermoroute.model_suite import (
    LSTM_VALIDATION_GRID,
    canonical_development_contract,
    development_predictor_bridge_binding,
    development_prediction_binding,
    file_binding,
    fit_pooled_imputer,
    lightgbm_entry,
    save_lightgbm_bundle,
    sequence_bundle_metadata,
    serialise_offsets,
    serialise_preprocessing,
    torch_entry,
    update_lightgbm_development_prediction,
    update_torch_development_prediction,
    verify_lightgbm_prediction_parity,
    verify_sequence_prediction_parity,
    write_component_pointer,
)
from thermoroute.probability import (
    fit_frozen_seasonal_event_reference,
    fit_horizon_calibrators,
)
from thermoroute.registry import enforce_common_forecast_keys
from thermoroute.repro import (
    assert_formal_numerical_policy,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sidecar_path,
)
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import (
    LSTMForecaster,
    configure_deterministic_runtime,
    fit_model,
)
from thermoroute.weighting import ROW_EQUAL_WEIGHTING

configure_deterministic_runtime()


_stage9_spec = importlib.util.spec_from_file_location(
    "route_a_stage9", ROOT / "scripts" / "09_usgs_experiment.py"
)
STAGE9 = importlib.util.module_from_spec(_stage9_spec)
_stage9_spec.loader.exec_module(STAGE9)

PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
FROZEN_SPEC = ROOT / "data_usgs" / "frozen_panel_v1.json"
USGS_VARS = STAGE9.USGS_VARS
CFG = C.TrainConfig(batch_size=1536)
SEEDS = C.USGS_SEEDS
_t0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - _t0:6.0f}s] {message}", flush=True)


def ensemble_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "scope", "feature_set", "site_id", "horizon", "split",
            "issue_date", "target_date"]
    return predictions.groupby(keys, as_index=False, dropna=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        q05=("q05", "mean"), q50=("q50", "mean"), q95=("q95", "mean"),
        p_exceed=("p_exceed", "mean"),
    )


def pooled_calibration(predictions: pd.DataFrame, threshold: float):
    calibration = ensemble_prediction_frame(predictions)
    calibration = calibration[calibration.split.eq("calib")].copy()
    calibration["event"] = (calibration.y_true.to_numpy(float) > threshold).astype(int)
    offsets = {}
    for horizon, group in calibration.groupby("horizon"):
        score = np.maximum(
            group.q05.to_numpy(float) - group.y_true.to_numpy(float),
            group.y_true.to_numpy(float) - group.q95.to_numpy(float),
        )
        offsets[("__pooled__", int(horizon))] = CF.conformal_quantile(score, 0.10)
    calibrators = fit_horizon_calibrators(
        calibration, probability_col="p_exceed", outcome_col="event",
        min_samples=100,
    )
    if set(offsets) != {("__pooled__", int(h)) for h in C.HORIZONS}:
        raise ValueError("external pooled CQR lacks a declared horizon")
    if set(calibrators) != set(C.HORIZONS):
        raise ValueError("external pooled event calibration lacks a declared horizon")
    return offsets, calibrators


def external_sequence_metadata(
    *, identity, wd, climatology, imputer, pooled_threshold, architecture_class,
    architecture_kwargs, offsets, calibrators, event_reference, prediction_binding,
):
    thresholds = {site: pooled_threshold for site in C.STATIONS}
    metadata = sequence_bundle_metadata(
        run_id=identity.run_id, architecture_class=architecture_class,
        architecture_kwargs=architecture_kwargs, train_config=CFG,
        wd=wd, climatology=climatology, imputer=imputer,
        thresholds=thresholds,
        event_reference_climatology=event_reference,
        conformal_offsets=offsets,
        event_calibrators=calibrators,
        source_sha256=identity.source_sha256,
        panel_sha256=identity.panel_sha256,
        registry_sha256=identity.registry_sha256,
        config_sha256=identity.config_sha256,
        runtime_sha256=identity.runtime_sha256,
        training_device="cpu",
        development_prediction=prediction_binding,
    )
    metadata["event_thresholds"] = {"__pooled__": float(pooled_threshold)}
    metadata["event_threshold_estimator"] = {
        "method": "pooled_training_empirical_quantile_v1",
        "quantile": 0.90,
        "pool_weighting": ROW_EQUAL_WEIGHTING,
        "station_balanced": False,
    }
    metadata["conformal_offsets"] = serialise_offsets(offsets)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-cache",
        help=(
            "immutable LightGBM shard-cache root; defaults to this run's "
            "content-addressed outputs/runs directory"
        ),
    )
    args = parser.parse_args()
    runtime_policy = assert_formal_numerical_policy()
    predictor_bridge = development_predictor_bridge_binding(
        ROOT,
        panel_sha256=sha256_file(PANEL),
        registry_sha256=sha256_file(REGISTRY),
    )
    run_config = {
        "stage": "25_train_external_pooled_suite",
        "role": "prelabel_station_agnostic_development_training",
        "panel": PANEL.name, "registry": REGISTRY.name,
        "variables": USGS_VARS, "horizons": C.HORIZONS,
        "seeds": SEEDS, "train_config": CFG,
        "preprocessing": "pooled_development_train_only",
        "station_agnostic": True,
        "lstm_validation_grid": LSTM_VALIDATION_GRID,
        "lightgbm_validation_grid": STAGE9.LGB_VALIDATION_GRID,
        "event_reference_fit_interval": ("2006-01-01", "2018-12-31"),
        "event_threshold_estimator": {
            "method": "pooled_training_empirical_quantile_v1",
            "quantile": 0.90,
            "pool_weighting": ROW_EQUAL_WEIGHTING,
            "station_balanced": False,
        },
        "post_2020_data_read": False,
        "training_device": "cpu",
        "development_predictor_bridge": predictor_bridge,
        "formal_numerical_policy": runtime_policy,
    }
    identity = resolve_run_identity(
        root=ROOT, panel=PANEL, registry=REGISTRY, config=run_config
    )
    run_dir = initialise_run_directory(
        ROOT / "outputs" / "runs" / "25_external_pooled", identity, run_config,
        provenance={
            "outcome_status": "NO_POST_2020_DATA_READ",
            "training_device": "cpu",
        },
    )
    # Lock before pooled preprocessing materialises arrays and before any
    # checkpoint or external shard-cache path can be reached.
    prepared = D.prepare_dataset_from_panel(str(PANEL))
    panel, masks = prepared["panel_raw"], prepared["masks"]
    stations = tuple(prepared["stations"])
    imputer = fit_pooled_imputer(panel, masks.train, fit_stations=stations)
    panel_imp = imputer.transform(panel)
    climatology = F.HarmonicClimatology.fit(
        panel, masks.train, fit_stations=stations, pooled=True
    )
    wd = DS.build_windows(
        panel_imp, masks, climatology, variables=USGS_VARS,
        require_observed_target=True, scaler_fit_stations=stations,
        pooled_scaler=True, damped_fit_stations=stations, pooled_damped=True,
    )
    training = panel.loc[masks.train & panel.site_id.isin(stations).to_numpy()]
    pooled_threshold = float(training.WTEMP.quantile(0.90))
    thresholds = {site: pooled_threshold for site in stations}
    event_reference = fit_frozen_seasonal_event_reference(
        panel,
        {"__pooled__": pooled_threshold},
        pooled=True,
        fit_interval=("2006-01-01", "2018-12-31"),
    )
    lightgbm_shard_cache = (
        Path(args.shard_cache).expanduser().resolve()
        if args.shard_cache else run_dir / "lightgbm_shards"
    )

    # LSTM architecture selection: validation rows only, station embedding off.
    candidates = []
    for candidate_id, candidate in enumerate(LSTM_VALIDATION_GRID):
        kwargs = {
            "n_vars": len(wd.var_names), "n_stations": len(stations),
            "context": C.CONTEXT_LENGTH, "station_agnostic": True, **candidate,
        }
        result = fit_model(
            lambda kwargs=kwargs: LSTMForecaster(**kwargs), wd, thresholds,
            cfg=CFG, seed=SEEDS[0], model_name=f"LSTM-external-grid-{candidate_id}",
            scope="external_validation_selection", feature_set="USGS",
            station_balanced=True, selection_metric="station_macro",
            device="cpu",
            export_splits=("val",),
            checkpoint_path=run_dir / "selection" / f"lstm_candidate{candidate_id}.pt",
            run_id=identity.run_id,
            resolved_config={**run_config, "candidate_id": candidate_id,
                             "candidate": candidate},
        )
        candidates.append((result.best_val, candidate_id, candidate))
    _, selected_id, selected = min(candidates, key=lambda value: (value[0], value[1]))
    log(f"external LSTM validation winner: candidate {selected_id}")

    tr_kwargs = {
        "n_vars": len(wd.var_names), "n_stations": len(stations),
        "n_phys": wd.n_phys, "station_agnostic": True,
        "use_prior": True, "use_router": True, "use_moe": True,
        "sparse_router": True, "fixed_kappa": False,
        "delta_scale": C.DELTA_SCALE, "use_tcn": True,
        "residual_model": True, "safety_anchor": "damped",
        "use_wlevel": False,
    }
    lstm_kwargs = {
        "n_vars": len(wd.var_names), "n_stations": len(stations),
        "context": C.CONTEXT_LENGTH, "station_agnostic": True, **selected,
    }
    tr_models, lstm_models, tr_predictions, lstm_predictions = {}, {}, [], []
    for seed in SEEDS:
        member = f"seed{seed}"
        tr = fit_model(
            lambda: ThermoRoute(**tr_kwargs), wd, thresholds, cfg=CFG, seed=seed,
            model_name="ThermoRoute", scope="external_pooled_development",
            feature_set="USGS", station_balanced=True,
            device="cpu",
            selection_metric="station_macro",
            checkpoint_path=run_dir / "checkpoints" / f"thermoroute_{member}.pt",
            run_id=identity.run_id,
            resolved_config={**run_config, "arm": "ThermoRoute", "seed": seed},
        )
        tr.pred["seed"] = seed
        tr_models[member] = tr.model
        tr_predictions.append(tr.pred)
        lstm = fit_model(
            lambda: LSTMForecaster(**lstm_kwargs), wd, thresholds,
            cfg=CFG, seed=seed, model_name="LSTM",
            scope="external_pooled_development", feature_set="USGS",
            device="cpu",
            station_balanced=True, selection_metric="station_macro",
            checkpoint_path=run_dir / "checkpoints" / f"lstm_{member}.pt",
            run_id=identity.run_id,
            resolved_config={**run_config, "arm": "LSTM", "seed": seed,
                             "selected_candidate": selected},
        )
        lstm.pred["seed"] = seed
        lstm_models[member] = lstm.model
        lstm_predictions.append(lstm.pred)
    tr_predictions = pd.concat(tr_predictions, ignore_index=True)
    lstm_predictions = pd.concat(lstm_predictions, ignore_index=True)

    (lgb_predictions, lgb_selection, lgb_models, lgb_native_probe,
     lgb_evaluation_design, lgb_design_order) = STAGE9.lightgbm_joint(
        panel_imp, panel, climatology, masks, thresholds, wd,
        station_agnostic=True, scope="external_pooled_development",
        shard_cache=lightgbm_shard_cache,
        shard_identity=identity,
        shard_cohort="external_pooled",
    )
    predictions, audit = enforce_common_forecast_keys(
        pd.concat([tr_predictions, lstm_predictions, lgb_predictions], ignore_index=True),
        ("ThermoRoute", "LSTM", "LightGBM"), split="test",
    )
    prediction_path = C.PREDICTIONS / f"external_pooled_development_{identity.run_id}.parquet"
    R.write_predictions(predictions, prediction_path)
    seal_artifact(
        prediction_path, identity, kind="external_pooled_development_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
        extra={"common_test_keys": audit.common_unique, "post_2020_data_read": False},
    )

    parity_atol = 1e-5
    sequence_artifacts = {}
    for model_id, models, model_predictions, architecture_class, kwargs, factory in (
        ("ThermoRoute", tr_models, tr_predictions,
         "thermoroute.thermoroute.ThermoRoute", tr_kwargs,
         thermoroute_factory_from_metadata),
        ("LSTM", lstm_models, lstm_predictions,
         "thermoroute.train.LSTMForecaster", lstm_kwargs,
         lstm_factory_from_metadata),
    ):
        rows = predictions[predictions.model.eq(model_id)]
        offsets, calibrators = pooled_calibration(model_predictions, pooled_threshold)
        directory = C.MODELS / f"external_{model_id.lower()}_bundle_{identity.run_id}"
        save_inference_bundle(
            directory, members=models,
            metadata=external_sequence_metadata(
                identity=identity, wd=wd, climatology=climatology, imputer=imputer,
                pooled_threshold=pooled_threshold, architecture_class=architecture_class,
                architecture_kwargs=kwargs, offsets=offsets,
                calibrators=calibrators,
                event_reference=event_reference,
                prediction_binding=development_prediction_binding(
                    ROOT, prediction_path, rows,
                    max_abs_difference=parity_atol, atol=parity_atol,
                ),
            ), expected_member_count=5,
        )
        difference = verify_sequence_prediction_parity(
            directory, wd=wd, expected=rows,
            model_factory=lambda _member, metadata, factory=factory: factory(metadata),
            member_seeds={f"seed{seed}": seed for seed in SEEDS},
            atol=parity_atol,
        )
        update_torch_development_prediction(
            directory,
            development_prediction_binding(
                ROOT, prediction_path, rows,
                max_abs_difference=difference, atol=parity_atol,
            ),
        )
        sequence_artifacts[model_id] = directory

    lgb_rows = predictions[predictions.model.eq("LightGBM")]
    lgb_offsets, lgb_calibrators = pooled_calibration(
        lgb_predictions, pooled_threshold
    )
    lgb_manifest = save_lightgbm_bundle(
        C.MODELS / f"external_lightgbm_bundle_{identity.run_id}",
        models=lgb_models, parity_inputs=lgb_native_probe,
        quantile_audit_inputs=lgb_evaluation_design,
        metadata={
            "run_id": identity.run_id,
            "raw_feature_order": list(wd.var_names),
            "design_feature_order": list(lgb_design_order),
            "horizons": list(wd.horizons),
            "members": [f"seed{seed}" for seed in SEEDS], "member_count": 5,
            "station_agnostic": True, "uses_station_categorical": False,
            "station_categories": [],
            "preprocessing": serialise_preprocessing(wd, climatology, imputer),
            "feature_engineering": {
                "builder": "thermoroute.features.build_tabular",
                "include_missingness": True, "numeric_nan_fill": 0.0,
                "station_code": None,
            },
            "training_weighting": "equal_total_weight_per_station",
            "deterministic_training": {
                "deterministic": True, "force_col_wise": True, "n_jobs": 1,
            },
            "validation_selection": lgb_selection.to_dict(orient="records"),
            "event_thresholds": {"__pooled__": pooled_threshold},
            "event_threshold_estimator": {
                "method": "pooled_training_empirical_quantile_v1",
                "quantile": 0.90,
                "pool_weighting": ROW_EQUAL_WEIGHTING,
                "station_balanced": False,
            },
            "event_reference_climatology": dict(event_reference),
            "event_calibrators": {str(h): value.as_dict()
                                  for h, value in sorted(lgb_calibrators.items())},
            "conformal_offsets": serialise_offsets(lgb_offsets),
            "source_sha256": identity.source_sha256,
            "panel_sha256": identity.panel_sha256,
            "registry_sha256": identity.registry_sha256,
            "config_sha256": identity.config_sha256,
            "runtime_sha256": identity.runtime_sha256,
            "training_device": "cpu",
            "development_prediction": development_prediction_binding(
                ROOT, prediction_path, lgb_rows,
                max_abs_difference=1e-12, atol=1e-12,
            ),
        },
    )
    lgb_difference = verify_lightgbm_prediction_parity(
        lgb_manifest, evaluation_design=lgb_evaluation_design,
        expected=lgb_rows,
        member_seeds={f"seed{seed}": seed for seed in SEEDS}, atol=1e-12,
    )
    update_lightgbm_development_prediction(
        lgb_manifest,
        development_prediction_binding(
            ROOT, prediction_path, lgb_rows,
            max_abs_difference=lgb_difference, atol=1e-12,
        ),
    )

    development_contract = canonical_development_contract(
        ROOT, FROZEN_SPEC, panel_sha256=identity.panel_sha256,
        registry_sha256=identity.registry_sha256,
        source_sha256=identity.source_sha256,
    )
    entries = [
        torch_entry(
            ROOT, model_id="ThermoRoute", executor="thermoroute_bundle",
            directory=sequence_artifacts["ThermoRoute"], member_count=5,
            raw_feature_order=wd.var_names,
        ),
        torch_entry(
            ROOT, model_id="LSTM", executor="lstm_bundle",
            directory=sequence_artifacts["LSTM"], member_count=5,
            raw_feature_order=wd.var_names,
        ),
        lightgbm_entry(ROOT, manifest=lgb_manifest, raw_feature_order=wd.var_names),
    ]
    write_component_pointer(
        C.MODELS / "route_a_external_components.json",
        run_id=identity.run_id, cohort="external", entries=entries,
        raw_feature_order=wd.var_names,
        development_contract=development_contract,
        development_prediction_artifact={
            **file_binding(ROOT, prediction_path),
            "sidecar": file_binding(ROOT, sidecar_path(prediction_path)),
        },
    )
    log("saved complete external pooled components: TR5 + LSTM5 + LGB5")


if __name__ == "__main__":
    main()
