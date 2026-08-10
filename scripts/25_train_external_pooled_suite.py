#!/usr/bin/env python3
"""Train the station-agnostic Route-A suite on development data only.

Despite its name, this command never reads the frozen 30-site cohort and never
opens post-2020 data.  It fits pooled transforms and station-agnostic models on
the canonical 120-site development panel.  The resulting bundles can later
expand those pooled statistics to any separately frozen external site IDs.
Success is published only through a self-hashed receipt over the exact file
closure; ``--check`` revalidates it without fitting or opening confirmation
outcomes.

Run: python scripts/25_train_external_pooled_suite.py
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

_IMPORT_SAFE_THREAD_DEFAULT = os.environ.get("OMP_NUM_THREADS") or "1"
STAGE25_THREADS = int(
    os.environ.get("THERMOROUTE_FORMAL_THREADS")
    or ("8" if __name__ == "__main__" else _IMPORT_SAFE_THREAD_DEFAULT)
)

if __name__ == "__main__":
    for _thread_variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_thread_variable, str(STAGE25_THREADS))
    os.environ.setdefault("THERMOROUTE_FORMAL_THREADS", str(STAGE25_THREADS))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage25-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE25_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE25_NONCE"


def _formal_worker_environment(
    cache: Path, nonce: str, threads: int | None = None,
) -> dict[str, str]:
    """Return the complete allowlisted Stage-25 worker environment."""
    if threads is None:
        threads = int(
            os.environ.get("THERMOROUTE_FORMAL_THREADS") or STAGE25_THREADS
        )
    thread_value = str(threads)
    return {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": str(cache.resolve()),
        "OMP_NUM_THREADS": thread_value,
        "MKL_NUM_THREADS": thread_value,
        "OPENBLAS_NUM_THREADS": thread_value,
        "VECLIB_MAXIMUM_THREADS": thread_value,
        "NUMEXPR_NUM_THREADS": thread_value,
        "THERMOROUTE_FORMAL_THREADS": thread_value,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        _WORKER_CACHE_ENV: str(cache.resolve()),
        _WORKER_NONCE_ENV: nonce,
    }


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
            or dict(os.environ)
            != _formal_worker_environment(expected, worker_nonce)
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
        environment = _formal_worker_environment(cache_path, nonce)
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
    STAGE25_COMPLETION_FORMAT,
    STAGE25_COMPLETION_RECEIPT_PATH,
    build_stage25_completion_receipt,
    canonical_development_contract,
    development_predictor_bridge_binding,
    development_prediction_binding,
    file_binding,
    fit_pooled_imputer,
    lightgbm_entry,
    publish_stage25_completion_receipt,
    route_a_calibration_fit_contract,
    save_lightgbm_bundle,
    sequence_bundle_metadata,
    serialise_offsets,
    serialise_preprocessing,
    torch_entry,
    update_lightgbm_development_prediction,
    update_torch_development_prediction,
    validate_stage25_completion_receipt,
    verify_lightgbm_prediction_parity,
    verify_sequence_prediction_parity,
    write_component_pointer,
)
from thermoroute.probability import (
    fit_frozen_seasonal_event_reference,
    fit_horizon_calibrators,
)
from thermoroute.registry import (
    canonicalize_prediction_truth_inplace,
    enforce_common_forecast_keys,
)
from thermoroute.input_closure import (
    compose_input_closure_digest,
    resolve_development_input_closure,
)
from thermoroute.repro import (
    advisory_file_lock,
    assert_formal_numerical_policy,
    assert_role_thread_cap,
    atomic_write_json,
    configure_deterministic_runtime,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sidecar_path,
)
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import (
    LSTMForecaster,
    fit_model,
)
from thermoroute.weighting import ROW_EQUAL_WEIGHTING, STATION_EQUAL_WEIGHTING

configure_deterministic_runtime()
assert_role_thread_cap(ROOT, "stage25")


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
    pooled_cqr = calibration.copy()
    pooled_cqr["site_id"] = "__pooled__"
    offsets, offset_audit = CF.cqr_offsets_with_audit(
        pooled_cqr, alpha=0.10
    )
    calibrators = fit_horizon_calibrators(
        calibration, probability_col="p_exceed", outcome_col="event",
        min_samples=100,
        weighting=STATION_EQUAL_WEIGHTING,
    )
    if set(offsets) != {("__pooled__", int(h)) for h in C.HORIZONS}:
        raise ValueError("external pooled CQR lacks a declared horizon")
    if set(calibrators) != set(C.HORIZONS):
        raise ValueError("external pooled event calibration lacks a declared horizon")
    return offsets, offset_audit, calibrators


def external_sequence_metadata(
    *, identity, wd, climatology, imputer, pooled_threshold, architecture_class,
    architecture_kwargs, offsets, offset_audit, calibrators, event_reference,
    prediction_binding,
):
    thresholds = {site: pooled_threshold for site in C.STATIONS}
    metadata = sequence_bundle_metadata(
        run_id=identity.run_id, architecture_class=architecture_class,
        architecture_kwargs=architecture_kwargs, train_config=CFG,
        wd=wd, climatology=climatology, imputer=imputer,
        thresholds=thresholds,
        event_reference_climatology=event_reference,
        conformal_offsets=offsets,
        conformal_offset_audit=offset_audit,
        event_calibrators=calibrators,
        source_sha256=identity.source_sha256,
        panel_sha256=identity.panel_sha256,
        registry_sha256=identity.registry_sha256,
        config_sha256=identity.config_sha256,
        runtime_sha256=identity.runtime_sha256,
        input_closure_sha256=identity.input_closure_sha256,
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


def _run(args: argparse.Namespace) -> None:
    components_pointer = C.MODELS / "route_a_external_components.json"
    receipt_path = ROOT / STAGE25_COMPLETION_RECEIPT_PATH
    runtime_policy = assert_formal_numerical_policy()
    development_input_closure = resolve_development_input_closure(ROOT)
    input_closure_sha256 = compose_input_closure_digest({
        "development": development_input_closure.binding_digest,
    })
    development_input_closure.assert_unchanged()

    def assert_stage25_publication_inputs() -> None:
        assert_formal_numerical_policy()
        development_input_closure.assert_unchanged()

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
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": len(development_input_closure.inventory),
        "input_closure_component_count": 1,
    }
    identity = resolve_run_identity(
        root=ROOT,
        panel=PANEL,
        registry=REGISTRY,
        config=run_config,
        input_closure_sha256=input_closure_sha256,
    )
    run_dir = initialise_run_directory(
        ROOT / "outputs" / "runs" / "25_external_pooled", identity, run_config,
        provenance={
            "outcome_status": "NO_POST_2020_DATA_READ",
            "training_device": "cpu",
        },
        publication_guard=assert_stage25_publication_inputs,
    )
    # Invalidate any earlier success marker immediately after taking the run
    # lock.  A crash from this point until the final atomic receipt publication
    # therefore leaves a document that --check must reject.
    atomic_write_json(
        receipt_path,
        {
            "format": STAGE25_COMPLETION_FORMAT,
            "status": "INCOMPLETE",
            "stage": "25_train_external_pooled_suite",
            "run_id": identity.run_id,
            "confirmation_outcomes_requested_or_read": False,
        },
        publication_guard=assert_stage25_publication_inputs,
    )
    # Lock before pooled preprocessing materialises arrays and before any
    # checkpoint or external shard-cache path can be reached.
    assert_stage25_publication_inputs()
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
    if lightgbm_shard_cache != run_dir / "lightgbm_shards":
        raise ValueError(
            "formal Stage-25 shard cache must remain inside its run directory"
        )

    # LSTM architecture selection: validation rows only, station embedding off.
    assert_stage25_publication_inputs()
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
            artifact_publication_guard=assert_formal_numerical_policy,
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
            artifact_publication_guard=assert_formal_numerical_policy,
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
            artifact_publication_guard=assert_formal_numerical_policy,
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
    # All long-running fits have completed.  Fail before the first canonical
    # prediction/model artifact if any native runtime left the one-thread mode.
    assert_stage25_publication_inputs()
    predictions, audit = enforce_common_forecast_keys(
        pd.concat([tr_predictions, lstm_predictions, lgb_predictions], ignore_index=True),
        ("ThermoRoute", "LSTM", "LightGBM"), split="test",
    )
    canonicalize_prediction_truth_inplace(predictions)
    prediction_path = C.PREDICTIONS / f"external_pooled_development_{identity.run_id}.parquet"
    R.write_predictions(
        predictions,
        prediction_path,
        publication_guard=assert_stage25_publication_inputs,
    )
    seal_artifact(
        prediction_path, identity, kind="external_pooled_development_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
        extra={"common_test_keys": audit.common_unique, "post_2020_data_read": False},
        publication_guard=assert_stage25_publication_inputs,
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
        offsets, offset_audit, calibrators = pooled_calibration(
            model_predictions, pooled_threshold
        )
        directory = C.MODELS / f"external_{model_id.lower()}_bundle_{identity.run_id}"
        save_inference_bundle(
            directory, members=models,
            metadata=external_sequence_metadata(
                identity=identity, wd=wd, climatology=climatology, imputer=imputer,
                pooled_threshold=pooled_threshold, architecture_class=architecture_class,
                architecture_kwargs=kwargs, offsets=offsets,
                offset_audit=offset_audit,
                calibrators=calibrators,
                event_reference=event_reference,
                prediction_binding=development_prediction_binding(
                    ROOT, prediction_path, rows,
                    max_abs_difference=parity_atol, atol=parity_atol,
                ),
            ), expected_member_count=5,
            publication_guard=assert_stage25_publication_inputs,
        )
        difference = verify_sequence_prediction_parity(
            directory, wd=wd, expected=rows,
            model_factory=lambda _member, metadata, factory=factory: factory(metadata),
            member_seeds={f"seed{seed}": seed for seed in SEEDS},
            atol=parity_atol,
            publication_guard=assert_stage25_publication_inputs,
        )
        assert_stage25_publication_inputs()
        update_torch_development_prediction(
            directory,
            development_prediction_binding(
                ROOT, prediction_path, rows,
                max_abs_difference=difference, atol=parity_atol,
            ),
        )
        sequence_artifacts[model_id] = directory

    lgb_rows = predictions[predictions.model.eq("LightGBM")]
    lgb_offsets, lgb_offset_audit, lgb_calibrators = pooled_calibration(
        lgb_rows, pooled_threshold
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
                "deterministic": True, "force_col_wise": True,
                "n_jobs": STAGE25_THREADS,
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
            "conformal_policy": CF.cqr_policy_contract(),
            "conformal_offset_audit": lgb_offset_audit,
            "calibration_fit_contract": route_a_calibration_fit_contract(
                external=True
            ),
            "source_sha256": identity.source_sha256,
            "panel_sha256": identity.panel_sha256,
            "registry_sha256": identity.registry_sha256,
            "config_sha256": identity.config_sha256,
            "runtime_sha256": identity.runtime_sha256,
            "input_closure_sha256": identity.input_closure_sha256,
            "training_device": "cpu",
            "development_prediction": development_prediction_binding(
                ROOT, prediction_path, lgb_rows,
                max_abs_difference=1e-12, atol=1e-12,
            ),
        },
        publication_guard=assert_stage25_publication_inputs,
    )
    lgb_difference = verify_lightgbm_prediction_parity(
        lgb_manifest, evaluation_design=lgb_evaluation_design,
        expected=lgb_rows,
        member_seeds={f"seed{seed}": seed for seed in SEEDS}, atol=1e-12,
        publication_guard=assert_stage25_publication_inputs,
    )
    assert_stage25_publication_inputs()
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
    assert_stage25_publication_inputs()
    write_component_pointer(
        components_pointer,
        run_id=identity.run_id, cohort="external", entries=entries,
        raw_feature_order=wd.var_names,
        development_contract=development_contract,
        development_prediction_artifact={
            **file_binding(ROOT, prediction_path),
            "sidecar": file_binding(ROOT, sidecar_path(prediction_path)),
        },
        publication_guard=assert_stage25_publication_inputs,
    )
    receipt = build_stage25_completion_receipt(
        root=ROOT,
        run_id=identity.run_id,
        run_manifest=run_dir / "run.json",
        components_pointer=components_pointer,
    )
    # Deliberately the final filesystem write in the transaction.  The publish
    # helper validates the candidate closure before the atomic replace and
    # re-opens the on-disk receipt afterwards.
    assert_stage25_publication_inputs()
    assert_formal_numerical_policy()
    publish_stage25_completion_receipt(
        receipt_path,
        receipt,
        root=ROOT,
        components_pointer=components_pointer,
        publication_guard=assert_stage25_publication_inputs,
    )
    log("saved complete external pooled components: TR5 + LSTM5 + LGB5")
    log(f"saved Stage-25 completion receipt: {receipt_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="read-only validation of the published Stage-25 completion receipt",
    )
    parser.add_argument(
        "--shard-cache",
        help=(
            "immutable LightGBM shard-cache root; defaults to this run's "
            "content-addressed outputs/runs directory"
        ),
    )
    args = parser.parse_args()
    components_pointer = C.MODELS / "route_a_external_components.json"
    receipt_path = ROOT / STAGE25_COMPLETION_RECEIPT_PATH
    if args.check:
        with advisory_file_lock(C.STAGE25_TRANSACTION_LOCK, exclusive=False):
            development_input_closure = resolve_development_input_closure(ROOT)
            expected_input_closure_sha256 = compose_input_closure_digest({
                "development": development_input_closure.binding_digest,
            })
            receipt = validate_stage25_completion_receipt(
                receipt_path,
                root=ROOT,
                components_pointer=components_pointer,
            )
            if (
                receipt.get("run_identity", {}).get("input_closure_sha256")
                != expected_input_closure_sha256
            ):
                raise ValueError("Stage-25 development input closure changed")
            development_input_closure.assert_unchanged()
        assert_formal_numerical_policy()
        print(
            f"Stage-25 COMPLETE: run_id={receipt['run_id']} "
            f"receipt={receipt_path.relative_to(ROOT)}"
        )
        return
    with advisory_file_lock(C.STAGE25_TRANSACTION_LOCK, exclusive=True):
        _run(args)


if __name__ == "__main__":
    main()
