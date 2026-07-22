#!/usr/bin/env python3
"""Stage 9 — comprehensive large-sample experiment on the USGS station set.

Produces the headline result for the (re-)paper: does ThermoRoute beat
persistence AND damped persistence on rivers with real forecast headroom, with
calibrated uncertainty?  The random held-station arm here is only a warm-start
diagnostic; strict held-region transfer is implemented in stage 13c.

Outputs (all in the canonical predictions schema so the analysis stage reuses
conformal/metrics/decision code unchanged):
  * outputs/predictions/usgs_predictions_stage9_v2.parquet
    (immutable Stage-9 parent: baselines + LightGBM + seeds + exploratory arms)
  * outputs/models/thermoroute_usgs_bundle_<run-id>/ (all ensemble members + metadata)
  * outputs/tables/usgs_scores.csv, outputs/reports/usgs_experiment.md
  * outputs/models/route_a_stage09_completion.json
    (last-transaction receipt binding the report, tables, predictions and pointers)

Run:  python3 scripts/09_usgs_experiment.py --seeds 5 --device cpu
"""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
from typing import Callable

for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage09-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE09_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE09_NONCE"


def _isolate_project_bytecode() -> None:
    if __name__ != "__main__":
        return
    worker_cache = os.environ.get(_WORKER_CACHE_ENV)
    worker_nonce = os.environ.get(_WORKER_NONCE_ENV)
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    worker_argument = len(sys.argv) > 1 and sys.argv[1] == _WORKER_ARGUMENT
    if worker_cache is not None or worker_nonce is not None or worker_argument:
        if not (worker_cache and worker_nonce and worker_argument):
            raise RuntimeError("Stage 09 formal worker handshake is incomplete")
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
            raise RuntimeError("Stage 09 formal worker isolation contract failed")
        sys.argv.pop(1)
        return
    with tempfile.TemporaryDirectory(prefix="thermoroute-stage09-pycache-") as cache:
        cache_path = Path(cache).resolve()
        if any(cache_path.iterdir()):
            raise RuntimeError("Stage 09 controller pycache was not initially empty")
        nonce = secrets.token_hex(32)
        (cache_path / ".controller-nonce").write_text(nonce, encoding="utf-8")
        environment = os.environ.copy()
        environment[_WORKER_CACHE_ENV] = str(cache_path)
        environment[_WORKER_NONCE_ENV] = nonce
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                f"pycache_prefix={cache}",
                str(Path(__file__).resolve()),
                _WORKER_ARGUMENT,
                *sys.argv[1:],
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()

import argparse
from dataclasses import asdict
import time
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from thermoroute import config as C
from thermoroute.chronology import STAGE09_ARTIFACT_PATHS
from thermoroute import data as D
from thermoroute.probability import (
    fit_frozen_seasonal_event_reference,
    fit_horizon_calibrators,
)
import lightgbm as lgb
from thermoroute import features as F
from thermoroute import datasets as DS
from thermoroute import results as R
from thermoroute import conformal as CF
from thermoroute.checkpoint import (
    load_inference_bundle,
    neural_output_head_schema,
    save_inference_bundle,
)
from thermoroute.model_suite import (
    ABLATION_INTERVENTIONS,
    MANDATORY_ABLATIONS,
    STAGE9_COMPLETION_RECEIPT_PATH,
    ModelSuiteError,
    build_stage09_completion_receipt,
    canonical_development_contract,
    development_predictor_bridge_binding,
    development_prediction_binding,
    file_binding,
    lightgbm_entry,
    publish_stage09_completion_receipt,
    save_lightgbm_bundle,
    serialise_preprocessing,
    torch_entry,
    update_lightgbm_development_prediction,
    update_torch_development_prediction,
    validate_stage09_prepublication_outputs,
    verify_lightgbm_prediction_parity,
    verify_sequence_prediction_parity,
    write_component_pointer,
)
from thermoroute.frozen_inference import thermoroute_factory_from_metadata
from thermoroute.lgb_shards import (
    LightGBMShardLineage,
    finalize_shard_set,
    lightgbm_design_key_digest,
    save_lightgbm_shard,
    try_load_lightgbm_shard,
)
from thermoroute.repro import (
    assert_formal_numerical_policy,
    atomic_write_bytes,
    atomic_write_json,
    cache_is_valid,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sidecar_path,
)
from thermoroute.registry import (
    FORECAST_KEY,
    STAGE9_PRIMARY_MODELS,
    enforce_common_forecast_keys,
    restrict_tabular_to_window_registry,
)
from thermoroute.quantiles import repair_lightgbm_quantiles
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import (
    configure_deterministic_runtime,
    fit_model,
    resolve_device,
)

configure_deterministic_runtime()

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")  # +gridMET wind
CFG = C.TrainConfig(batch_size=1536)         # larger batch ⇒ fewer steps on 100k+ samples
DELTA_SCALE = C.DELTA_SCALE   # single source (config.py); val-selected (11_retune)
AIR2STREAM_DISPLAY_NAME = "Air2stream-style a4/a8 (unofficial, non-primary)"
LGB_VALIDATION_GRID = (
    {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.03},
    {"num_leaves": 31, "min_child_samples": 40, "learning_rate": 0.03},
    {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
    {"num_leaves": 31, "min_child_samples": 80, "learning_rate": 0.05},
)
_t0 = time.time()


def log(m):
    print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


def formal_publication_candidate(
    *, panel_path: Path, training_device: str, exploratory: bool
) -> bool:
    """Return whether a run may even be considered for formal publication."""
    return (
        panel_path.resolve()
        == (ROOT / "data_usgs" / "panel_usgs_120v2.parquet").resolve()
        and str(training_device) == "cpu"
        and not bool(exploratory)
    )


def seed0_ablation_diagnostic_frames(
    frame: pd.DataFrame,
    *,
    controls: tuple[str, ...] = MANDATORY_ABLATIONS,
    split: str = "test",
) -> dict[str, pd.DataFrame]:
    """Return strictly paired seed-0 frames for the Stage-9 diagnostic table.

    The formal Stage-9 controls are deliberately single-member interventions.
    Their descriptive table must therefore compare each control with the same
    ThermoRoute member, never with the five-member headline ensemble.  Refuse
    publication unless every control contains seed 0 only and its exact
    forecast-key registry and serialized target values equal ThermoRoute seed 0.
    """
    required = {"model", "split", "seed", "y_true", "y_pred", *FORECAST_KEY}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"seed0 ablation diagnostic lacks required columns: {missing}"
        )

    normal = frame.copy()
    normal["issue_date"] = pd.to_datetime(normal["issue_date"])
    normal["target_date"] = pd.to_datetime(normal["target_date"])
    test_rows = normal[normal["split"].eq(split)]

    full_rows = test_rows[test_rows["model"].eq("ThermoRoute")]
    full_seeds = pd.to_numeric(full_rows["seed"], errors="coerce")
    full = full_rows.loc[full_seeds.eq(0)].copy()
    if full.empty:
        raise ValueError(f"ThermoRoute seed=0 is absent from split={split!r}")
    if full.duplicated(list(FORECAST_KEY)).any():
        raise ValueError("ThermoRoute seed=0 has duplicate forecast keys")

    key_columns = list(FORECAST_KEY)
    full_keys = set(full[key_columns].itertuples(index=False, name=None))
    if not full_keys:
        raise ValueError("ThermoRoute seed=0 has no forecast keys")
    result = {"ThermoRoute": full.reset_index(drop=True)}

    for name in controls:
        all_control = normal[normal["model"].eq(name)]
        seeds = pd.to_numeric(all_control["seed"], errors="coerce")
        if (
            all_control.empty
            or seeds.isna().any()
            or not seeds.eq(0).all()
        ):
            raise ValueError(f"{name} must contain exact seed=0 rows only")
        control = all_control[all_control["split"].eq(split)].copy()
        if control.empty:
            raise ValueError(f"{name} is absent from split={split!r}")
        if control.duplicated(key_columns).any():
            raise ValueError(f"{name} seed=0 has duplicate forecast keys")
        control_keys = set(
            control[key_columns].itertuples(index=False, name=None)
        )
        if control_keys != full_keys:
            raise ValueError(
                f"{name} seed=0 forecast keys differ from ThermoRoute seed=0"
            )
        aligned = full[key_columns + ["y_true"]].merge(
            control[key_columns + ["y_true"]],
            on=key_columns,
            how="inner",
            validate="one_to_one",
            suffixes=("_thermoroute", "_control"),
        )
        full_truth = pd.to_numeric(
            aligned["y_true_thermoroute"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        control_truth = pd.to_numeric(
            aligned["y_true_control"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if (
            len(aligned) != len(full_keys)
            or not np.isfinite(full_truth).all()
            or not np.isfinite(control_truth).all()
            or not np.array_equal(full_truth, control_truth)
        ):
            raise ValueError(
                f"{name} seed=0 y_true differs from ThermoRoute seed=0"
            )
        result[name] = control.reset_index(drop=True)
    return result


def rmse_per_station(frame: pd.DataFrame, horizon: int) -> dict[str, float]:
    """Collapse seed rows on exact forecast keys before station-level RMSE."""
    required = {
        "site_id", "horizon", "issue_date", "target_date", "y_pred", "y_true",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RMSE frame lacks exact forecast keys: {sorted(missing)}")
    selected = frame[frame.horizon == horizon].groupby(
        ["site_id", "issue_date", "target_date"], as_index=False
    ).agg(y_pred=("y_pred", "mean"), y_true=("y_true", "first"))
    return {
        str(site): float(np.sqrt(((group.y_pred - group.y_true) ** 2).mean()))
        for site, group in selected.groupby("site_id")
    }


def thermoroute_ensemble_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean headline predictions retaining the complete forecast key."""
    return frame.groupby(
        ["site_id", "horizon", "issue_date", "target_date"], as_index=False
    ).agg(y_pred=("y_pred", "mean"), y_true=("y_true", "first"))


def complete_stage09_transaction(
    *,
    write_report: Callable[[], None],
    validate_outputs: Callable[[], None],
    publish_pointers: Callable[[], None],
    publish_receipt: Callable[[], Path],
) -> Path:
    """Order the formal commit: report, preflight, pointers, then receipt."""
    write_report()
    validate_outputs()
    publish_pointers()
    return publish_receipt()


def prep(panel_path: str):
    """Now a thin shim over the shared D.prepare_dataset_from_panel — the same
    fold-safe preparation used by the 3-station track and any future panel."""
    b = D.prepare_dataset_from_panel(panel_path)
    panel, panel_imp, masks = b["panel_raw"], b["panel"], b["masks"]
    clim = F.HarmonicClimatology.fit(panel, masks.train)
    return panel, panel_imp, masks, clim, b["stations"], b["imputer"]


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _serialise_offsets(offsets):
    return {f"{station}|{int(horizon)}": _finite(value)
            for (station, horizon), value in sorted(offsets.items())}


def bundle_metadata(identity, wd, clim, imputer, thresholds, event_reference,
                    delta_scale,
                    conformal_offsets, event_calibrators, *,
                    training_device, architecture_overrides=None,
                    development_prediction=None):
    kwargs = {
        "n_vars": len(wd.var_names),
        "n_stations": len(C.STATIONS),
        "n_phys": wd.n_phys,
        "station_agnostic": False,
        "use_prior": True,
        "use_router": True,
        "use_moe": True,
        "sparse_router": True,
        "fixed_kappa": False,
        "delta_scale": delta_scale,
        "use_tcn": True,
        "residual_model": True,
        "safety_anchor": "damped",
        "use_wlevel": False,
    }
    kwargs.update(dict(architecture_overrides or {}))
    return {
        "run_id": identity.run_id,
        "architecture": {
            "class": "thermoroute.thermoroute.ThermoRoute",
            "kwargs": kwargs,
            "train_config": asdict(CFG),
        },
        "feature_order": list(wd.var_names),
        "horizons": list(wd.horizons),
        "station_to_index": {station: index for index, station in enumerate(C.STATIONS)},
        "preprocessing": serialise_preprocessing(wd, clim, imputer),
        "event_thresholds": {station: float(value)
                             for station, value in sorted(thresholds.items())},
        "event_reference_climatology": dict(event_reference),
        "event_calibrators": {
            str(horizon): calibrator.as_dict()
            for horizon, calibrator in sorted(event_calibrators.items())
        },
        "conformal_offsets": _serialise_offsets(conformal_offsets),
        "source_sha256": identity.source_sha256,
        "panel_sha256": identity.panel_sha256,
        "registry_sha256": identity.registry_sha256,
        "config_sha256": identity.config_sha256,
        "runtime_sha256": identity.runtime_sha256,
        "training_device": str(training_device),
        "output_head_schema": neural_output_head_schema(),
        "development_prediction": dict(development_prediction or {}),
    }


def read_prediction_cache(path, identity):
    if not cache_is_valid(path, identity, schema=R.PREDICTION_SCHEMA_VERSION):
        return None
    try:
        cached = pd.read_parquet(path)
        R.validate_predictions(cached)
    except Exception as exc:
        log(f"  rejected invalid cache {path.name}: {exc}")
        return None
    return cached


def write_prediction_artifact(frame, path, identity, *, kind, parents=None):
    R.write_predictions(frame, path)
    seal_artifact(
        path,
        identity,
        kind=kind,
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents=parents,
    )


def read_member_bundle(directory, identity, member_name):
    try:
        weights, metadata = load_inference_bundle(directory, expected_member_count=1)
    except (FileNotFoundError, ValueError, RuntimeError):
        return None
    expected_identity = (
        metadata.get("run_id") == identity.run_id
        and metadata.get("source_sha256") == identity.source_sha256
        and metadata.get("panel_sha256") == identity.panel_sha256
        and metadata.get("registry_sha256") == identity.registry_sha256
        and metadata.get("runtime_sha256") == identity.runtime_sha256
    )
    if not expected_identity or set(weights) != {member_name}:
        return None
    return weights[member_name]


def ensemble_prediction_frame(predictions):
    keys = ["model", "scope", "feature_set", "site_id", "horizon", "split",
            "issue_date", "target_date"]
    ensemble = predictions.groupby(keys, as_index=False, dropna=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        q05=("q05", "mean"),
        q50=("q50", "mean"),
        q95=("q95", "mean"),
        p_exceed=("p_exceed", "mean"),
    )
    return ensemble


def calibration_artifacts(predictions, thresholds):
    ensemble = ensemble_prediction_frame(predictions)
    calibration = ensemble[ensemble.split == "calib"].copy()
    calibration["threshold"] = calibration["site_id"].astype(str).map(thresholds)
    if calibration["threshold"].isna().any():
        raise KeyError("calibration rows contain a site without an event threshold")
    calibration["event"] = (
        calibration["y_true"].to_numpy(float)
        > calibration["threshold"].to_numpy(float)
    ).astype(int)
    offsets = CF.cqr_offsets(calibration, alpha=0.10)
    calibrators = fit_horizon_calibrators(
        calibration, probability_col="p_exceed", outcome_col="event",
        min_samples=100,
    )
    expected_offsets = {(str(site), int(horizon)) for site in thresholds
                        for horizon in C.HORIZONS}
    if set(offsets) != expected_offsets:
        missing = sorted(expected_offsets - set(offsets))
        raise ValueError(f"calibration lacks frozen site×horizon offsets: {missing[:5]}")
    if set(calibrators) != set(C.HORIZONS):
        raise ValueError("event calibration lacks a declared horizon")
    return offsets, calibrators


def canon(wd, idx, model_name, preds_by_h, scope="joint_usgs"):
    site = np.array([C.STATIONS[i] for i in wd.station[idx]])
    issue = wd.issue_date[idx]
    frames = []
    for hi, h in enumerate(wd.horizons):
        frames.append(R.make_pred_frame(
            model=model_name, scope=scope, feature_set="USGS", seed=0,
            site_id=site, horizon=np.full(len(idx), h), split=np.full(len(idx), "test"),
            issue_date=issue, target_date=issue + np.timedelta64(h, "D"),
            y_true=wd.y[idx][:, hi], y_pred=preds_by_h[h]))
    return pd.concat(frames, ignore_index=True)


def _station_macro_validation_rmse(frame, prediction):
    values = []
    for station, positions in frame.groupby("site_id", sort=False).indices.items():
        position = np.asarray(positions, dtype=int)
        values.append(np.sqrt(np.mean(
            (prediction[position] - frame.iloc[position]["y"].to_numpy(float)) ** 2
        )))
    return float(np.mean(values))


def lightgbm_joint(panel_imp, panel_raw, clim, masks, thr, wd, *,
                   station_agnostic=False, scope="joint_usgs",
                   shard_cache=None, shard_identity=None, shard_cohort=None):
    """Five matched LightGBM members per horizon and probabilistic head.

    Uses the **imputed** panel and keeps every row whose target WTEMP is observed
    — matching ThermoRoute's windowed sample selection so the two models can be
    compared on identical (site_id, issue_date, horizon) keys (enforced by the
    final join in ``main`` and by ``tests/test_sample_consistency.py``).
    """
    from thermoroute import baselines as B
    shard_arguments = (shard_cache, shard_identity, shard_cohort)
    if any(value is not None for value in shard_arguments) and not all(
        value is not None for value in shard_arguments
    ):
        raise ValueError(
            "LightGBM shard cache, run identity, and cohort must be supplied together"
        )
    shard_cache_path = Path(shard_cache).resolve() if shard_cache is not None else None
    shard_lineages = []
    frames = []
    selection_rows = []
    models = {f"seed{seed}": {} for seed in C.USGS_SEEDS}
    parity_inputs = {}
    evaluation_design = {}
    design_order = None
    for h in C.HORIZONS:
        tab = F.attach_split(F.build_tabular(
            panel_imp, h, USGS_VARS, clim,
            drop_feature_nans=False, require_observed_target=True,
            include_missingness=True))
        tab = restrict_tabular_to_window_registry(tab, wd, C.STATIONS, h)
        cols = F.feature_columns(tab)
        # NaN-safe design matrix: LightGBM natively handles NaN, but we also
        # zero-fill to remove any residual surprise — equivalent to the
        # standardised-mean fallback used by ThermoRoute's encoder.
        for c in cols:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
        # Same-station comparison: expose the stable scientific site identifier
        # just as ThermoRoute receives a station embedding.  Pandas categorical
        # metadata makes LightGBM treat this as nominal, not an ordinal number.
        if not station_agnostic:
            tab["station_code"] = pd.Categorical(
                tab["site_id"].astype(str), categories=list(C.STATIONS)
            )
            cols = [*cols, "station_code"]
        if design_order is None:
            design_order = tuple(cols)
        elif tuple(cols) != design_order:
            raise AssertionError("LightGBM design columns changed across horizons")
        tr, va = tab[tab.split == "train"], tab[tab.split == "val"]
        ev = tab[tab.split.isin(["calib", "test"])]
        Xtr, ytr = tr[cols], tr["y"].to_numpy(float)
        Xva, yva = va[cols], va["y"].to_numpy(float)
        Xev = ev[cols]
        design_key_sha256 = lightgbm_design_key_digest(
            {"train": tr, "validation": va, "evaluation": ev},
            feature_order=cols,
        )
        train_weight = B.station_equal_sample_weight(tr["site_id"])
        validation_weight = B.station_equal_sample_weight(va["site_id"])

        candidates = []
        for candidate_id, params in enumerate(LGB_VALIDATION_GRID):
            candidate = B._lgb_fit(
                Xtr, ytr, Xva, yva, "regression", params_override=params,
                sample_weight=train_weight, val_sample_weight=validation_weight,
            )
            validation_prediction = candidate.predict(Xva)
            score = _station_macro_validation_rmse(va, validation_prediction)
            selection_rows.append({
                "horizon": h,
                "candidate_id": candidate_id,
                **params,
                "val_station_macro_rmse": score,
                "best_iteration": int(candidate.best_iteration_ or candidate.n_estimators),
                "selected": False,
                "selection_split": "2016-2017 validation",
            })
            candidates.append((score, candidate_id, params, candidate))
        _, selected_id, selected_params, _selected_validation_model = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        selection_rows[-len(LGB_VALIDATION_GRID) + selected_id]["selected"] = True
        parity_inputs[h] = Xev.iloc[:min(257, len(Xev))].copy()
        evaluation_design[h] = (
            ev[["site_id", "split", "issue_date", "target_date", "y"]].reset_index(
                drop=True
            ).copy(),
            Xev.reset_index(drop=True).copy(),
        )
        st_arr = ev["site_id"].to_numpy()
        thr_tr = np.array([thr[s] for s in tr["site_id"]])
        thr_va = np.array([thr[s] for s in va["site_id"]])
        for seed in C.USGS_SEEDS:
            member = f"seed{seed}"
            seed_params = {
                **selected_params,
                "seed": int(seed),
                "bagging_seed": int(seed),
                "feature_fraction_seed": int(seed),
                "data_random_seed": int(seed),
            }
            def acquire_shard(head, head_config, trainer):
                if shard_cache_path is None:
                    return trainer()
                lineage = LightGBMShardLineage.from_run_identity(
                    shard_identity,
                    cohort=str(shard_cohort), seed=seed, horizon=h, head=head,
                    design_key_sha256=design_key_sha256,
                    head_config=head_config,
                )
                shard_lineages.append(lineage)
                cached = try_load_lightgbm_shard(
                    shard_cache_path, lineage=lineage, parity_input=parity_inputs[h]
                )
                if cached is not None:
                    return cached
                fitted = trainer()
                return save_lightgbm_shard(
                    shard_cache_path, lineage=lineage, model=fitted,
                    parity_input=parity_inputs[h], parity_atol=1e-12,
                )

            common_regressor_config = {
                "estimator": "lightgbm.LGBMRegressor",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 40,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "n_estimators": 800,
                "verbosity": -1,
                "n_jobs": 1,
                "deterministic": True,
                "force_col_wise": True,
                "early_stopping_rounds": 50,
                "sample_weight": "equal_total_weight_per_station",
                **seed_params,
            }
            point = acquire_shard(
                "point",
                {**common_regressor_config, "objective": "regression", "alpha": None},
                lambda: B._lgb_fit(
                    Xtr, ytr, Xva, yva, "regression",
                    params_override=seed_params,
                    sample_weight=train_weight,
                    val_sample_weight=validation_weight,
                ),
            )
            quantile_models = {}
            quantile_predictions = {}
            for alpha, head in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
                quantile_models[head] = acquire_shard(
                    head,
                    {
                        **common_regressor_config,
                        "objective": "quantile", "alpha": alpha,
                    },
                    lambda alpha=alpha: B._lgb_fit(
                        Xtr, ytr, Xva, yva, "quantile", alpha=alpha,
                        params_override=seed_params,
                        sample_weight=train_weight,
                        val_sample_weight=validation_weight,
                    ),
                )
                quantile_predictions[head] = quantile_models[head].predict(
                    Xev, num_threads=1
                )

            def fit_classifier():
                classifier = lgb.LGBMClassifier(
                    n_estimators=800, **seed_params,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, verbosity=-1, n_jobs=1,
                    deterministic=True, force_col_wise=True,
                )
                classifier.fit(
                    Xtr, (ytr > thr_tr).astype(int),
                    sample_weight=train_weight,
                    eval_set=[(Xva, (yva > thr_va).astype(int))],
                    eval_sample_weight=[validation_weight],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)
                    ],
                )
                return classifier

            classifier = acquire_shard(
                "event",
                {
                    "estimator": "lightgbm.LGBMClassifier",
                    "objective": "binary",
                    "n_estimators": 800,
                    "subsample": 0.8,
                    "subsample_freq": 1,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "verbosity": -1,
                    "n_jobs": 1,
                    "deterministic": True,
                    "force_col_wise": True,
                    "early_stopping_rounds": 50,
                    "sample_weight": "equal_total_weight_per_station",
                    **seed_params,
                },
                fit_classifier,
            )
            models[member][h] = {
                "point": point,
                **quantile_models,
                "event": classifier,
            }
            q05, q50, q95 = repair_lightgbm_quantiles(
                quantile_predictions["q05"],
                quantile_predictions["q50"],
                quantile_predictions["q95"],
            )
            frames.append(R.make_pred_frame(
                model="LightGBM", scope=scope, feature_set="USGS", seed=seed,
                site_id=st_arr, horizon=np.full(len(ev), h),
                split=ev["split"].to_numpy(), issue_date=ev["issue_date"].to_numpy(),
                target_date=ev["target_date"].to_numpy(), y_true=ev["y"].to_numpy(float),
                y_pred=point.predict(Xev, num_threads=1), q05=q05,
                q50=q50, q95=q95,
                p_exceed=(
                    classifier.predict(Xev, num_threads=1)
                    if isinstance(classifier, lgb.Booster)
                    else classifier.predict_proba(Xev)[:, 1]
                )))
    if shard_cache_path is not None:
        finalize_shard_set(
            shard_cache_path, lineages=shard_lineages, parity_inputs=parity_inputs
        )
    return (
        pd.concat(frames, ignore_index=True), pd.DataFrame(selection_rows),
        models, parity_inputs, evaluation_design, tuple(design_order or ()),
    )


def main():
    runtime_policy = assert_formal_numerical_policy()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--panel",
        default=os.environ.get(
            "USGS_PANEL", str(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")),
        help="frozen input panel (default: $USGS_PANEL, then canonical 120-site panel)",
    )
    ap.add_argument("--seeds", type=int, default=len(C.USGS_SEEDS))
    ap.add_argument("--delta_scale", type=float, default=DELTA_SCALE)
    ap.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps", "cuda"),
                    help="CPU for formal Route-A; accelerators require --exploratory")
    ap.add_argument(
        "--exploratory", action="store_true",
        help="allow an explicitly non-formal run; never publish Route-A components",
    )
    ap.add_argument("--eval_batch_size", type=int, default=4096,
                    help="maximum validation/export rows resident on the accelerator")
    ap.add_argument("--station_sampling", choices=("balanced", "natural"),
                    default="balanced",
                    help="equal-station bootstrap (main protocol) or natural row-frequency sensitivity")
    ap.add_argument("--ablations", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--air2stream", action="store_true", default=False,
                    help=f"add {AIR2STREAM_DISPLAY_NAME} physical references (slower)")
    ap.add_argument(
        "--out_predictions",
        default=Path(STAGE09_ARTIFACT_PATHS["predictions"]).name,
    )
    ap.add_argument(
        "--out_report", default=Path(STAGE09_ARTIFACT_PATHS["report"]).name
    )
    ap.add_argument(
        "--out_scores", default=Path(STAGE09_ARTIFACT_PATHS["scores"]).name
    )
    ap.add_argument(
        "--shard-cache",
        help=(
            "immutable LightGBM shard-cache root; defaults to this run's "
            "content-addressed outputs/runs directory"
        ),
    )
    args = ap.parse_args()
    if args.seeds < 1 or args.seeds > len(C.USGS_SEEDS):
        ap.error(f"--seeds must be between 1 and {len(C.USGS_SEEDS)}")

    panel_path = Path(args.panel).resolve()
    registry_path = ROOT / "data_usgs" / "station_registry_v1.csv"
    resolved_device = str(resolve_device(args.device))
    if resolved_device != "cpu" and not args.exploratory:
        ap.error("non-CPU Stage-9 runs require --exploratory and cannot publish formal pointers")
    formal_candidate = formal_publication_candidate(
        panel_path=panel_path,
        training_device=resolved_device,
        exploratory=args.exploratory,
    )
    predictor_bridge = (
        development_predictor_bridge_binding(
            ROOT,
            panel_sha256=sha256_file(panel_path),
            registry_sha256=sha256_file(registry_path),
        )
        if formal_candidate else None
    )
    protocol = f"route_a_strict_v1_{args.station_sampling}_delta{args.delta_scale:g}"
    run_config = {
        "stage": "09_usgs_experiment",
        "protocol": protocol,
        "panel": panel_path.name,
        "station_registry": registry_path.name,
        "variables": USGS_VARS,
        "horizons": C.HORIZONS,
        "time_split": C.SPLIT.as_dict(),
        "train_config": asdict(CFG),
        "thermoroute_seeds": C.USGS_SEEDS[:args.seeds],
        "lightgbm_seeds": C.USGS_SEEDS,
        "delta_scale": args.delta_scale,
        "station_sampling": args.station_sampling,
        "selection_metric": ("station_macro" if args.station_sampling == "balanced"
                             else "micro"),
        "ablations": bool(args.ablations),
        "air2stream": bool(args.air2stream),
        "device": resolved_device,
        "training_device": resolved_device,
        "execution_role": (
            "route_a_formal_candidate" if formal_candidate else "exploratory_only"
        ),
        "development_predictor_bridge": predictor_bridge,
        "eval_batch_size": args.eval_batch_size,
        "lightgbm_validation_grid": LGB_VALIDATION_GRID,
        "event_reference_fit_interval": ("2006-01-01", "2018-12-31"),
        "formal_numerical_policy": runtime_policy,
    }
    identity = resolve_run_identity(
        root=ROOT,
        panel=panel_path,
        registry=registry_path,
        config=run_config,
    )
    run_dir = initialise_run_directory(
        ROOT / "outputs" / "runs" / "09_usgs_experiment",
        identity,
        run_config,
        provenance={
            "evidence_role": (
                "prelabel_route_a_model_build_development_only"
                if formal_candidate else "development_exploratory_2019_2020"
            ),
            "training_device": resolved_device,
        },
    )
    prediction_cache = run_dir / "predictions"
    training_checkpoints = run_dir / "checkpoints"
    member_cache = run_dir / "member_bundles"
    ablation_cache = run_dir / "ablation_bundles"
    lightgbm_shard_cache = (
        Path(args.shard_cache).expanduser().resolve()
        if args.shard_cache else run_dir / "lightgbm_shards"
    )
    log(f"content-addressed run {identity.run_id} | device={resolved_device}")

    panel, panel_imp, masks, clim, stations, imputer = prep(str(panel_path))
    wd = DS.build_windows(panel_imp, masks, clim, variables=USGS_VARS,
                          require_observed_target=True)
    thr = {s: float(panel.loc[masks.train].query("site_id==@s").WTEMP.quantile(0.9))
           for s in stations}
    event_reference = fit_frozen_seasonal_event_reference(
        panel,
        thr,
        pooled=False,
        fit_interval=("2006-01-01", "2018-12-31"),
    )
    test_idx = wd.idx("test")
    log(f"{len(stations)} stations | windows N={len(wd.X)} test={len(test_idx)}")

    chunks = []
    # ---- baselines from identical windowed samples ---------------------- #
    for name, fn in [("Persistence", lambda hi, h: wd.wtemp_t[test_idx]),
                     ("Climatology", lambda hi, h: wd.clim_tgt[test_idx][:, hi]),
                     # Exact same frozen train-fit anchor used inside strict TR.
                     ("DampedPersistence", lambda hi, h: wd.damped_prior[test_idx, hi])]:
        preds = {h: fn(hi, h) for hi, h in enumerate(wd.horizons)}
        chunks.append(canon(wd, test_idx, name, preds))
    log("baselines done")

    # ---- joint LightGBM ------------------------------------------------- #
    if args.air2stream:
        from thermoroute import air2stream as A2S
        clim_air = F.HarmonicClimatology.fit(panel_imp, masks.train, target="TEMP")
        for v in ("a4", "a8"):
            t = time.time()
            chunks.append(A2S.run_air2stream(panel_imp, masks, clim_air,
                                             stations=stations, variant=v))
            log(f"  Air2stream-style {v} (unofficial, non-primary): "
                f"{time.time() - t:.0f}s")
    (lightgbm_predictions, lightgbm_selection, lightgbm_models,
     lightgbm_parity_inputs, lightgbm_evaluation_design,
     lightgbm_design_order) = lightgbm_joint(
        panel_imp, panel, clim, masks, thr, wd,
        shard_cache=lightgbm_shard_cache,
        shard_identity=identity,
        shard_cohort="temporal_stage9",
    )
    chunks.append(lightgbm_predictions)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    lightgbm_selection_path = (
        C.TABLES / "lightgbm_joint_validation_selection.csv"
    )
    atomic_write_bytes(
        lightgbm_selection_path,
        lightgbm_selection.to_csv(index=False).encode("utf-8"),
    )
    log("LightGBM joint done")

    # ---- ThermoRoute joint, multiple seeds (resumable per seed) ---------- #
    tr_preds = []
    ensemble_members = {}
    for sd in C.USGS_SEEDS[:args.seeds]:
        member_name = f"seed{sd}"
        seed_file = prediction_cache / f"thermoroute_{member_name}.parquet"
        seed_bundle = member_cache / member_name
        cached_prediction = read_prediction_cache(seed_file, identity)
        cached_member = read_member_bundle(seed_bundle, identity, member_name)
        if cached_prediction is not None and cached_member is not None:
            tr_preds.append(cached_prediction)
            ensemble_members[member_name] = cached_member
            log(f"  ThermoRoute {member_name}: verified content cache")
            continue
        te = time.time()
        factory = lambda: ThermoRoute(
            n_vars=len(wd.var_names), n_stations=len(stations), n_phys=wd.n_phys,
            delta_scale=args.delta_scale, safety_anchor="damped")
        res = fit_model(factory, wd, thr, cfg=CFG, seed=sd,
                        device=resolved_device, eval_batch_size=args.eval_batch_size,
                        model_name="ThermoRoute", scope="joint_usgs",
                        feature_set="USGS",
                        station_balanced=args.station_sampling == "balanced",
                        selection_metric=("station_macro" if args.station_sampling == "balanced"
                                          else "micro"),
                        checkpoint_path=training_checkpoints / f"{member_name}.pt",
                        run_id=identity.run_id,
                        resolved_config={**run_config, "arm": "ThermoRoute", "seed": sd})
        model = res.model
        res.pred["seed"] = sd
        write_prediction_artifact(
            res.pred, seed_file, identity, kind="thermoroute_seed_predictions"
        )
        tr_preds.append(res.pred)
        seed_offsets, seed_calibrators = calibration_artifacts(res.pred, thr)
        save_inference_bundle(
            seed_bundle,
            members={member_name: model},
            metadata=bundle_metadata(
                identity, wd, clim, imputer, thr, event_reference,
                args.delta_scale,
                seed_offsets, seed_calibrators,
                training_device=resolved_device,
            ),
            expected_member_count=1,
        )
        ensemble_members[member_name] = {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        }
        log(f"  ThermoRoute seed{sd}: {res.epochs+1}ep {time.time()-te:.0f}s val={res.best_val:.4f}")
    chunks.append(pd.concat(tr_preds, ignore_index=True))

    # ---- leave-group-out ------------------------------------------------ #
    rng = np.random.default_rng(0)
    perm = rng.permutation(list(stations))
    hold = set(perm[: max(1, len(stations) // 4)])
    trainset = tuple(s for s in stations if s not in hold)
    lgo_file = prediction_cache / "thermoroute_lgo.parquet"
    cached_lgo = read_prediction_cache(lgo_file, identity)
    if cached_lgo is not None:
        chunks.append(cached_lgo)
        log("  LGO: verified content cache")
    else:
        te = time.time()
        factory = lambda: ThermoRoute(
            n_vars=len(wd.var_names), n_stations=len(stations), n_phys=wd.n_phys,
            station_agnostic=True, delta_scale=args.delta_scale,
            safety_anchor="damped")
        res = fit_model(factory, wd, thr, cfg=CFG, seed=0,
                        device=resolved_device, eval_batch_size=args.eval_batch_size,
                        model_name="ThermoRoute-LGO-WarmStart", scope="warm_start",
                        feature_set="USGS", train_stations=trainset,
                        station_balanced=args.station_sampling == "balanced",
                        selection_metric=("station_macro" if args.station_sampling == "balanced"
                                          else "micro"),
                        checkpoint_path=training_checkpoints / "lgo.pt",
                        run_id=identity.run_id,
                        resolved_config={**run_config, "arm": "ThermoRoute-LGO-WarmStart",
                                         "seed": 0, "train_stations": trainset})
        res.pred["seed"] = 0
        lgo_held = res.pred[res.pred.site_id.isin(hold)]
        write_prediction_artifact(
            lgo_held, lgo_file, identity, kind="thermoroute_lgo_predictions"
        )
        chunks.append(lgo_held)
        log(f"  LGO ({len(trainset)}→{len(hold)}): {time.time()-te:.0f}s")

    # ---- large-sample module ablations (single seed) -------------------- #
    ablation_members = {}
    ablation_predictions = {}
    ablation_architecture = {}
    if args.ablations:
        # Each control changes one declared factor.  In particular noMoE keeps
        # both routed and TCN representations, and noRouter keeps the TCN path.
        abl = {
            "TR-noDynamicPrior": dict(use_prior=False),
            "TR-fixedKappa": dict(fixed_kappa=True),
            "TR-noRouter": dict(use_router=False),
            "TR-noMoE": dict(use_moe=False),
            "TR-noTCN": dict(use_tcn=False),
            "TR-unbounded": dict(delta_scale=None),
            "DampedPriorOnly": dict(use_prior=False, residual_model=False),
        }
        for name, kw in abl.items():
            af = prediction_cache / f"ablation_{name}.parquet"
            ab = ablation_cache / name
            cached_ablation = read_prediction_cache(af, identity)
            cached_weights = read_member_bundle(ab, identity, name)
            model_kw = dict(kw)
            model_kw.setdefault("delta_scale", args.delta_scale)
            if cached_ablation is not None and cached_weights is not None:
                chunks.append(cached_ablation)
                ablation_predictions[name] = cached_ablation
                ablation_members[name] = cached_weights
                ablation_architecture[name] = model_kw
                log(f"  {name}: verified content cache")
                continue
            te = time.time()
            factory = lambda model_kw=model_kw: ThermoRoute(
                n_vars=len(wd.var_names), n_stations=len(stations),
                n_phys=wd.n_phys, safety_anchor="damped", **model_kw)
            r = fit_model(factory, wd, thr, cfg=CFG, seed=0, model_name=name,
                          device=resolved_device, eval_batch_size=args.eval_batch_size,
                          scope="ablation_usgs", feature_set="USGS",
                          station_balanced=args.station_sampling == "balanced",
                          selection_metric=("station_macro" if args.station_sampling == "balanced"
                                            else "micro"),
                          checkpoint_path=training_checkpoints / f"ablation_{name}.pt",
                          run_id=identity.run_id,
                          resolved_config={**run_config, "arm": name, "seed": 0,
                                           "model_kwargs": model_kw})
            r.pred["seed"] = 0
            write_prediction_artifact(
                r.pred, af, identity, kind="thermoroute_ablation_predictions"
            )
            ablation_offsets, ablation_calibrators = calibration_artifacts(r.pred, thr)
            save_inference_bundle(
                ab,
                members={name: r.model},
                metadata=bundle_metadata(
                    identity, wd, clim, imputer, thr, event_reference,
                    args.delta_scale,
                    ablation_offsets, ablation_calibrators,
                    training_device=resolved_device,
                    architecture_overrides=model_kw,
                ),
                expected_member_count=1,
            )
            ablation_predictions[name] = r.pred
            ablation_members[name] = {
                key: value.detach().cpu().contiguous()
                for key, value in r.model.state_dict().items()
            }
            ablation_architecture[name] = model_kw
            chunks.append(r.pred)
            log(f"  {name}: {time.time()-te:.0f}s val={r.best_val:.4f}")

    allp = pd.concat(chunks, ignore_index=True)

    # The primary registry is fixed by protocol.  It must never be inferred
    # from optional models: an Air2stream failure or an exploratory ablation
    # with fewer rows cannot silently remove easier/harder primary examples.
    # The key includes target_date, preserves every seed row, and audits label
    # agreement before a headline metric is computed.
    allp, audit = enforce_common_forecast_keys(
        allp, STAGE9_PRIMARY_MODELS, split="test"
    )
    log(f"sample registry: {audit.common_unique} exact keys across {audit.models}; "
        f"dropped {audit.dropped_rows} non-shared rows (before={audit.before_unique})")
    seed0_diagnostic = (
        seed0_ablation_diagnostic_frames(allp) if args.ablations else {}
    )

    output_predictions = C.PREDICTIONS / args.out_predictions
    write_prediction_artifact(
        allp,
        output_predictions,
        identity,
        kind="canonical_stage9_usgs_predictions",
        parents=(
            {"development_predictor_bridge_v1.json": predictor_bridge["sha256"]}
            if predictor_bridge is not None else None
        ),
    )
    log(f"saved predictions ({len(allp)} rows)")

    # Formal bundles are bound to the immutable Stage-9 prediction artifact.
    # A custom --panel may still produce a diagnostic bundle, but can never move
    # a Route-A component pointer.
    try:
        development_contract = canonical_development_contract(
            ROOT, ROOT / "data_usgs" / "frozen_panel_v1.json",
            panel_sha256=identity.panel_sha256,
            registry_sha256=identity.registry_sha256,
            source_sha256=identity.source_sha256,
        )
        canonical_run = True
    except ModelSuiteError:
        development_contract = None
        canonical_run = False

    parity_atol = 1e-5
    offsets, event_calibrators = calibration_artifacts(
        pd.concat(tr_preds, ignore_index=True), thr
    )
    full_ensemble = args.seeds == len(C.USGS_SEEDS)
    bundle_prefix = "thermoroute_usgs_bundle" if full_ensemble else "thermoroute_usgs_partial_bundle"
    deployment_bundle = C.MODELS / f"{bundle_prefix}_{identity.run_id}"
    save_inference_bundle(
        deployment_bundle,
        members=ensemble_members,
        metadata=bundle_metadata(
            identity, wd, clim, imputer, thr, event_reference,
            args.delta_scale,
            offsets, event_calibrators,
            training_device=resolved_device,
            development_prediction=development_prediction_binding(
                ROOT, output_predictions,
                allp[allp.model.eq("ThermoRoute")],
                max_abs_difference=parity_atol, atol=parity_atol,
            ),
        ),
        expected_member_count=args.seeds,
    )
    loaded_members, loaded_metadata = load_inference_bundle(
        deployment_bundle, expected_member_count=args.seeds
    )
    if set(loaded_members) != set(ensemble_members) or loaded_metadata["run_id"] != identity.run_id:
        raise AssertionError("saved inference ensemble failed round-trip validation")
    tr_difference = verify_sequence_prediction_parity(
        deployment_bundle, wd=wd,
        expected=allp[allp.model.eq("ThermoRoute")],
        model_factory=lambda _member, metadata: thermoroute_factory_from_metadata(metadata),
        member_seeds={f"seed{seed}": seed for seed in C.USGS_SEEDS[:args.seeds]},
        atol=parity_atol, batch_size=args.eval_batch_size,
    )
    update_torch_development_prediction(
        deployment_bundle,
        development_prediction_binding(
            ROOT, output_predictions, allp[allp.model.eq("ThermoRoute")],
            max_abs_difference=tr_difference, atol=parity_atol,
        ),
    )

    # LightGBM is an equally-sized five-member ensemble.  Every point,
    # quantile and event head is a native-text Booster with its own checksum.
    lgb_offsets, lgb_calibrators = calibration_artifacts(lightgbm_predictions, thr)
    lgb_bundle = C.MODELS / f"lightgbm_usgs_bundle_{identity.run_id}"
    lgb_manifest = save_lightgbm_bundle(
        lgb_bundle,
        models=lightgbm_models,
        parity_inputs=lightgbm_parity_inputs,
        quantile_audit_inputs=lightgbm_evaluation_design,
        metadata={
            "run_id": identity.run_id,
            "raw_feature_order": list(wd.var_names),
            "design_feature_order": list(lightgbm_design_order),
            "horizons": list(wd.horizons),
            "members": [f"seed{seed}" for seed in C.USGS_SEEDS],
            "member_count": len(C.USGS_SEEDS),
            "station_agnostic": False,
            "uses_station_categorical": True,
            "station_categories": list(C.STATIONS),
            "preprocessing": serialise_preprocessing(wd, clim, imputer),
            "feature_engineering": {
                "builder": "thermoroute.features.build_tabular",
                "include_missingness": True,
                "numeric_nan_fill": 0.0,
                "station_code": "stable_site_no_pandas_categorical",
            },
            "training_weighting": "equal_total_weight_per_station",
            "deterministic_training": {
                "deterministic": True, "force_col_wise": True, "n_jobs": 1,
            },
            "validation_selection": lightgbm_selection.to_dict(orient="records"),
            "event_thresholds": {str(site): float(value)
                                 for site, value in sorted(thr.items())},
            "event_reference_climatology": dict(event_reference),
            "event_calibrators": {str(h): value.as_dict()
                                  for h, value in sorted(lgb_calibrators.items())},
            "conformal_offsets": _serialise_offsets(lgb_offsets),
            "source_sha256": identity.source_sha256,
            "panel_sha256": identity.panel_sha256,
            "registry_sha256": identity.registry_sha256,
            "config_sha256": identity.config_sha256,
            "runtime_sha256": identity.runtime_sha256,
            "training_device": "cpu",
            "development_prediction": development_prediction_binding(
                ROOT, output_predictions, allp[allp.model.eq("LightGBM")],
                max_abs_difference=1e-12, atol=1e-12,
            ),
        },
    )
    lgb_difference = verify_lightgbm_prediction_parity(
        lgb_manifest, evaluation_design=lightgbm_evaluation_design,
        expected=allp[allp.model.eq("LightGBM")],
        member_seeds={f"seed{seed}": seed for seed in C.USGS_SEEDS},
        atol=1e-12,
    )
    update_lightgbm_development_prediction(
        lgb_manifest,
        development_prediction_binding(
            ROOT, output_predictions, allp[allp.model.eq("LightGBM")],
            max_abs_difference=lgb_difference, atol=1e-12,
        ),
    )

    # Freeze every mandatory one-factor architecture control.  Cached
    # predictions alone are never accepted as a model artifact.
    ablation_deployments = {}
    if args.ablations and set(ablation_members) == set(MANDATORY_ABLATIONS):
        for name in MANDATORY_ABLATIONS:
            pred = allp[allp.model.eq(name)]
            abl_offsets, abl_calibrators = calibration_artifacts(
                ablation_predictions[name], thr
            )
            destination = C.MODELS / f"{name.lower()}_bundle_{identity.run_id}"
            save_inference_bundle(
                destination,
                members={name: ablation_members[name]},
                metadata=bundle_metadata(
                    identity, wd, clim, imputer, thr, event_reference,
                    args.delta_scale,
                    abl_offsets, abl_calibrators,
                    training_device=resolved_device,
                    architecture_overrides=ablation_architecture[name],
                    development_prediction=development_prediction_binding(
                        ROOT, output_predictions, pred,
                        max_abs_difference=parity_atol, atol=parity_atol,
                    ),
                ),
                expected_member_count=1,
            )
            difference = verify_sequence_prediction_parity(
                destination, wd=wd, expected=pred,
                model_factory=lambda _member, metadata:
                    thermoroute_factory_from_metadata(metadata),
                member_seeds={name: 0}, atol=parity_atol,
                batch_size=args.eval_batch_size,
            )
            update_torch_development_prediction(
                destination,
                development_prediction_binding(
                    ROOT, output_predictions, pred,
                    max_abs_difference=difference, atol=parity_atol,
                ),
            )
            ablation_deployments[name] = destination

    formal_complete = (
        canonical_run and full_ensemble and args.ablations
        and formal_candidate and predictor_bridge is not None
        and args.station_sampling == "balanced"
        and np.isclose(args.delta_scale, DELTA_SCALE, rtol=0.0, atol=0.0)
        and set(ablation_deployments) == set(MANDATORY_ABLATIONS)
    )
    component_entries = []
    if formal_complete:
        component_entries = [
            torch_entry(
                ROOT, model_id="ThermoRoute", executor="thermoroute_bundle",
                directory=deployment_bundle, member_count=5,
                raw_feature_order=wd.var_names,
            ),
            lightgbm_entry(
                ROOT, manifest=lgb_manifest, raw_feature_order=wd.var_names,
            ),
        ]
        for name in MANDATORY_ABLATIONS:
            component_entries.append(torch_entry(
                ROOT, model_id=name, executor="thermoroute_bundle",
                directory=ablation_deployments[name], member_count=1,
                raw_feature_order=wd.var_names,
                intervention=ABLATION_INTERVENTIONS[name],
            ))

    # ---- headline point report (seed-mean ThermoRoute) ------------------ #
    tr_test = allp[(allp.model == "ThermoRoute") & (allp.split == "test")]
    base = {m: allp[(allp.model == m) & (allp.split == "test")] for m in
            ("Persistence", "DampedPersistence", "Climatology")}

    rows = []
    for h in wd.horizons:
        rp = rmse_per_station(base["Persistence"], h)
        rd = rmse_per_station(base["DampedPersistence"], h)
        # ThermoRoute seed-mean on the complete forecast key.
        tm = thermoroute_ensemble_summary_frame(
            tr_test[tr_test.horizon == h]
        )
        rt = {s: float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
              for s, g in tm.groupby("site_id")}
        for s in stations:
            rows.append({"horizon": h, "site": s, "rmse_persist": rp.get(s, np.nan),
                         "rmse_damped": rd.get(s, np.nan), "rmse_thermo": rt.get(s, np.nan)})
    sc = pd.DataFrame(rows)
    score_path = C.TABLES / args.out_scores
    atomic_write_bytes(score_path, sc.to_csv(index=False).encode("utf-8"))

    L = [f"# USGS large-sample experiment ({len(stations)} stations, {args.seeds} seeds)\n",
         f"_Variables {', '.join(USGS_VARS)}. Observed targets only; identical samples "
         f"across models. ThermoRoute = {args.seeds}-seed mean. The same-station "
         "LightGBM is also a five-seed mean and receives stable site identity as a "
         "categorical feature; its small "
         "predeclared grid is selected by 2016–2017 station-macro RMSE only._\n",
         f"| horizon | persist | damped | {AIR2STREAM_DISPLAY_NAME} | "
         "LightGBM | ThermoRoute | "
         "skill vs persist | skill vs damped | win-rate vs damped |",
         "|---|---|---|---|---|---|---|---|---|"]
    lg = allp[(allp.model == "LightGBM") & (allp.split == "test")]
    a2s_a4 = allp[(allp.model == "Air2stream-a4") & (allp.split == "test")]
    a2s_a8 = allp[(allp.model == "Air2stream-a8") & (allp.split == "test")]
    for h in wd.horizons:
        d = sc[sc.horizon == h]
        rl = rmse_per_station(lg, h)
        ml = np.median([rl[s] for s in stations if s in rl])
        ra4 = rmse_per_station(a2s_a4, h)
        ra8 = rmse_per_station(a2s_a8, h)
        ma4 = np.median([ra4[s] for s in stations if s in ra4]) if ra4 else float("nan")
        ma8 = np.median([ra8[s] for s in stations if s in ra8]) if ra8 else float("nan")
        mp, md, mt = d.rmse_persist.median(), d.rmse_damped.median(), d.rmse_thermo.median()
        sk_p = 1 - (d.rmse_thermo / d.rmse_persist).median()
        sk_d = 1 - (d.rmse_thermo / d.rmse_damped).median()
        # win-rate over stations that have development-evaluation data for both
        # models; stations with no test samples must not count as losses.
        dv = d.dropna(subset=["rmse_thermo", "rmse_damped"])
        win = float((dv.rmse_thermo < dv.rmse_damped).mean())
        L.append(f"| {h} | {mp:.3f} | {md:.3f} | {ma4:.3f} / {ma8:.3f} | "
                 f"{ml:.3f} | {mt:.3f} | "
                 f"{sk_p:+.3f} | {sk_d:+.3f} | {win:.2f} |")
    # leave-group-out
    lgo = allp[(allp.model == "ThermoRoute-LGO-WarmStart") & (allp.split == "test")]
    L += ["", f"## Random held-station warm-start diagnostic ({len(trainset)}→{len(hold)})\n",
          "Held stations contribute historical observations to the global panel preprocessing; "
          "this arm is not zero-shot spatial transfer and does not establish unseen-basin skill.\n",
          "| horizon | warm-start RMSE | persistence RMSE | warm-start skill |",
          "|---|---|---|---|"]
    for h in wd.horizons:
        g = lgo[lgo.horizon == h]
        rt = float(np.sqrt(((g.y_pred - g.y_true) ** 2).mean()))
        bp = base["Persistence"]
        gp = bp[(bp.horizon == h) & (bp.site_id.isin(hold))]
        rp = float(np.sqrt(((gp.y_pred - gp.y_true) ** 2).mean()))
        L.append(f"| {h} | {rt:.3f} | {rp:.3f} | {1-rt/rp:+.3f} |")

    # ---- ablation summary (median per-station RMSE) --------------------- #
    abl_models = ("ThermoRoute", *MANDATORY_ABLATIONS)
    L += ["", f"## Module ablations (single-seed functionality/intervention "
          f"diagnostic; seed0-vs-seed0; median per-station RMSE, "
          f"delta_scale={args.delta_scale})\n",
          "Audit: every mandatory control is exact seed=0 and is paired with "
          "ThermoRoute seed=0 on identical forecast keys and exact y_true. "
          "Interpretation: this is a single-seed functionality/intervention "
          "diagnostic, seed0-vs-seed0; not evidence of module necessity, causal "
          "mechanism, or cross-seed stability.\n",
          "| variant | h1 | h3 | h7 |", "|---|---|---|---|"]
    for m in abl_models:
        sub = seed0_diagnostic.get(m)
        if sub is None:
            continue
        meds = []
        for h in wd.horizons:
            r = rmse_per_station(sub, h)
            meds.append(np.median([r[s] for s in stations if s in r]))
        L.append(f"| {m} | {meds[0]:.3f} | {meds[1]:.3f} | {meds[2]:.3f} |")
    report_path = C.REPORTS / args.out_report
    report_payload = ("\n".join(L) + "\n").encode("utf-8")

    def write_report() -> None:
        atomic_write_bytes(report_path, report_payload)

    def validate_outputs() -> None:
        validate_stage09_prepublication_outputs(
            root=ROOT,
            run_id=identity.run_id,
            run_manifest=run_dir / "run.json",
            predictions=output_predictions,
            scores=score_path,
            report=report_path,
            lightgbm_selection=lightgbm_selection_path,
        )

    if formal_complete:
        thermoroute_pointer = C.MODELS / "thermoroute_usgs_bundle.json"
        lightgbm_pointer = C.MODELS / "lightgbm_usgs_bundle.json"
        components_pointer = C.MODELS / "route_a_stage9_components.json"
        receipt_path = ROOT / STAGE9_COMPLETION_RECEIPT_PATH

        def publish_pointers() -> None:
            atomic_write_json(thermoroute_pointer, {
                "run_id": identity.run_id,
                "bundle_path": deployment_bundle.relative_to(ROOT).as_posix(),
                "member_count": len(loaded_members),
                "metadata_sha256": sha256_file(
                    deployment_bundle / "metadata.json"
                ),
                "weights_sha256": sha256_file(deployment_bundle / "weights.pt"),
            })
            atomic_write_json(lightgbm_pointer, {
                "run_id": identity.run_id,
                "manifest": file_binding(ROOT, lgb_manifest),
                "member_count": len(C.USGS_SEEDS),
            })
            write_component_pointer(
                components_pointer,
                run_id=identity.run_id,
                cohort="temporal_stage9",
                entries=component_entries,
                raw_feature_order=wd.var_names,
                development_contract=development_contract,
                development_prediction_artifact={
                    **file_binding(ROOT, output_predictions),
                    "sidecar": file_binding(
                        ROOT, sidecar_path(output_predictions)
                    ),
                },
            )

        def publish_receipt() -> Path:
            document = build_stage09_completion_receipt(
                root=ROOT,
                run_id=identity.run_id,
                run_manifest=run_dir / "run.json",
                predictions=output_predictions,
                scores=score_path,
                report=report_path,
                lightgbm_selection=lightgbm_selection_path,
                thermoroute_pointer=thermoroute_pointer,
                lightgbm_pointer=lightgbm_pointer,
                components_pointer=components_pointer,
            )
            publish_stage09_completion_receipt(
                receipt_path,
                document,
                root=ROOT,
                stage9_pointer=components_pointer,
            )
            return receipt_path

        complete_stage09_transaction(
            write_report=write_report,
            validate_outputs=validate_outputs,
            publish_pointers=publish_pointers,
            publish_receipt=publish_receipt,
        )
        log("saved formal Stage-9 model components: TR5 + LGB5 + 7 controls")
        log(f"saved Stage-9 completion receipt: {receipt_path.relative_to(ROOT)}")
    else:
        write_report()
        log("saved diagnostic bundles; formal Stage-9 pointers/receipt unchanged")
    log("DONE")
    print("\n" + "\n".join(L[3:10]))


if __name__ == "__main__":
    main()
