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

import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Mapping

_IMPORT_SAFE_THREAD_DEFAULT = os.environ.get("OMP_NUM_THREADS") or "1"
MAIN_THREADS = int(
    os.environ.get("THERMOROUTE_FORMAL_THREADS")
    or ("8" if __name__ == "__main__" else _IMPORT_SAFE_THREAD_DEFAULT)
)

if __name__ == "__main__":
    for _thread_variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_thread_variable, str(MAIN_THREADS))
    os.environ.setdefault("THERMOROUTE_FORMAL_THREADS", str(MAIN_THREADS))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
_WORKER_ARGUMENT = "--_thermoroute-stage09-worker"
_WORKER_CACHE_ENV = "THERMOROUTE_STAGE09_PYCACHE"
_WORKER_NONCE_ENV = "THERMOROUTE_STAGE09_NONCE"


def _formal_worker_environment(
    cache: Path, nonce: str, threads: int | None = None,
) -> dict[str, str]:
    """Return the complete allowlisted Stage-09 worker environment."""
    if threads is None:
        threads = int(os.environ.get("THERMOROUTE_FORMAL_THREADS") or MAIN_THREADS)
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
            or dict(os.environ)
            != _formal_worker_environment(expected, worker_nonce)
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
        environment = _formal_worker_environment(cache_path, nonce)
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
    CHECKPOINT_METADATA_VERSION,
    CHECKPOINT_VERSION,
    load_inference_bundle,
    neural_output_head_schema,
    save_inference_bundle,
)
from thermoroute.model_suite import (
    ABLATION_INTERVENTIONS,
    MANDATORY_ABLATIONS,
    STAGE9_ABLATION_SEEDS,
    STAGE9_AIR2STREAM_DISPLAY_NAME,
    STAGE9_AIR2STREAM_MODELS,
    STAGE9_COMPLETION_RECEIPT_PATH,
    STAGE9_LIGHTGBM_VALIDATION_GRID,
    build_stage09_completion_receipt,
    canonical_development_contract,
    development_predictor_bridge_binding,
    development_prediction_binding,
    file_binding,
    lightgbm_entry,
    publish_stage09_completion_receipt,
    route_a_calibration_fit_contract,
    save_lightgbm_bundle,
    serialise_offsets,
    serialise_preprocessing,
    stage09_air2stream_report_status,
    stage09_outputs_are_canonical,
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
from thermoroute.input_closure import (
    compose_input_closure_digest,
    resolve_development_input_closure,
)
from thermoroute.stage09_parallel import (
    DEFAULT_CONTROL_WORKERS,
    MAX_CONTROL_WORKERS,
    RECOMMENDED_MAX_CONTROL_WORKERS,
    SEMANTIC_VALIDATION_FORMAT,
    Stage09ParallelError,
    execute_work_orders_bounded,
    extracted_member_payload,
    freeze_control_matrix_receipt,
    freeze_control_plan,
    materialize_member_resume_checkpoint,
    member_work_order_map,
    publish_control_member,
    validate_live_work_order,
    validate_stage09_model_matrix_gate,
)
from thermoroute.repro import (
    RunIdentity,
    assert_formal_numerical_policy,
    assert_role_thread_cap,
    atomic_write_bytes,
    atomic_write_json,
    cache_is_valid,
    configure_deterministic_runtime,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sidecar_path,
    source_tree_hash,
    validate_artifact_sidecar,
)
from thermoroute.registry import (
    FORECAST_KEY,
    STAGE9_PRIMARY_MODELS,
    canonicalize_prediction_truth_inplace,
    enforce_common_forecast_keys,
    restrict_tabular_to_window_registry,
)
from thermoroute.quantiles import repair_lightgbm_quantiles
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import (
    fit_model,
    resolve_device,
)
from thermoroute.weighting import STATION_EQUAL_WEIGHTING

configure_deterministic_runtime()

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")  # +gridMET wind
CFG = C.TrainConfig(batch_size=1536)         # larger batch ⇒ fewer steps on 100k+ samples
DELTA_SCALE = C.DELTA_SCALE   # single source (config.py); val-selected (11_retune)
AIR2STREAM_DISPLAY_NAME = STAGE9_AIR2STREAM_DISPLAY_NAME
LGB_VALIDATION_GRID = STAGE9_LIGHTGBM_VALIDATION_GRID
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


def formal_stage09_configuration(
    *,
    panel_path: Path,
    training_device: str,
    exploratory: bool,
    canonical_output_paths: bool,
    seeds: int,
    station_sampling: str,
    delta_scale: float,
    ablations: bool,
    air2stream: bool,
) -> bool:
    """Whether every pre-training condition permits canonical publication.

    This is deliberately stricter than :func:`formal_publication_candidate`.
    A run with the right panel and device but a partial seed ensemble, omitted
    controls, or a sensitivity configuration is still a diagnostic run.  That
    distinction must be known before the first result artifact is written.
    """
    return (
        formal_publication_candidate(
            panel_path=panel_path,
            training_device=training_device,
            exploratory=exploratory,
        )
        and bool(canonical_output_paths)
        and int(seeds) == len(C.USGS_SEEDS)
        and str(station_sampling) == "balanced"
        and np.isclose(
            float(delta_scale), float(DELTA_SCALE), rtol=0.0, atol=0.0
        )
        and bool(ablations)
        and not bool(air2stream)
    )


def resolve_stage09_publication_paths(
    *,
    root: Path,
    run_dir: Path,
    requested_predictions: Path,
    requested_scores: Path,
    requested_report: Path,
    formal_output_eligible: bool,
) -> dict[str, Path]:
    """Resolve all mutable Stage-9 result files into one authority scope.

    A complete formal configuration retains the frozen canonical paths.
    Everything else is forced beneath its content-addressed run directory;
    caller-supplied or default canonical names can therefore never invalidate
    an already completed formal receipt.
    """
    root = Path(root).resolve()
    run_dir = Path(run_dir).resolve()
    expected_run_root = (
        root / "outputs" / "runs" / "09_usgs_experiment"
    ).resolve()
    if run_dir.parent != expected_run_root:
        raise ValueError("Stage-9 diagnostic outputs require an exact run directory")

    requested = {
        "predictions": Path(requested_predictions).resolve(),
        "scores": Path(requested_scores).resolve(),
        "report": Path(requested_report).resolve(),
    }
    if formal_output_eligible:
        if not stage09_outputs_are_canonical(root=root, **requested):
            raise ValueError(
                "formal Stage-9 publication requires all frozen canonical paths"
            )
        return {
            **requested,
            "lightgbm_selection": (
                root / STAGE09_ARTIFACT_PATHS["lightgbm_selection"]
            ).resolve(),
        }

    diagnostic_root = run_dir / "diagnostic_outputs"
    return {
        label: diagnostic_root / Path(STAGE09_ARTIFACT_PATHS[label]).name
        for label in ("predictions", "scores", "report", "lightgbm_selection")
    }


def multiseed_ablation_diagnostic_frames(
    frame: pd.DataFrame,
    *,
    controls: tuple[str, ...] = MANDATORY_ABLATIONS,
    seeds: tuple[int, ...] = STAGE9_ABLATION_SEEDS,
    split: str = "test",
) -> dict[str, pd.DataFrame]:
    """Return complete same-seed, same-key Stage-9 intervention frames."""
    required = {"model", "split", "seed", "y_true", "y_pred", *FORECAST_KEY}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"multi-seed ablation diagnostic lacks required columns: {missing}"
        )

    normal = frame.copy()
    if (
        not normal["site_id"].map(
            lambda value: isinstance(value, str)
            and bool(value)
            and value == value.strip()
        ).all()
        or pd.api.types.is_bool_dtype(normal["seed"].dtype)
        or not pd.api.types.is_integer_dtype(normal["seed"].dtype)
        or pd.api.types.is_bool_dtype(normal["horizon"].dtype)
        or not pd.api.types.is_integer_dtype(normal["horizon"].dtype)
    ):
        raise ValueError("multi-seed ablation diagnostic has noncanonical key dtypes")
    for column in ("issue_date", "target_date"):
        if (
            str(normal[column].dtype) != "datetime64[ns]"
            or normal[column].isna().any()
            or not normal[column].dt.normalize().equals(normal[column])
        ):
            raise ValueError(
                f"multi-seed ablation diagnostic has noncanonical {column}"
            )
    if (
        not set(normal["split"].tolist())
        <= {*C.SPLIT.as_dict(), "none"}
        or (
            normal["split"].eq("none")
            & ~normal["model"].isin(STAGE9_AIR2STREAM_MODELS)
        ).any()
        or not np.isfinite(
            normal[["y_true", "y_pred"]].to_numpy(dtype=float)
        ).all()
        or normal.duplicated(["model", "seed", *FORECAST_KEY, "split"]).any()
    ):
        raise ValueError(
            "multi-seed ablation diagnostic has invalid split, value, or duplicate key"
        )
    expected_split = np.full(len(normal), "none", dtype=object)
    for split_name, (lower, upper) in C.SPLIT.as_dict().items():
        mask = normal["issue_date"].between(lower, upper) & normal[
            "target_date"
        ].between(lower, upper)
        expected_split[mask.to_numpy()] = split_name
    horizon_days = (
        normal["target_date"] - normal["issue_date"]
    ).dt.days.to_numpy(dtype=int)
    if (
        not np.array_equal(normal["split"].to_numpy(object), expected_split)
        or not np.array_equal(
            normal["horizon"].to_numpy(dtype=int), horizon_days
        )
    ):
        raise ValueError(
            "multi-seed ablation diagnostic split/date/horizon contract changed"
        )
    test_rows = normal[normal["split"].eq(split)]

    full_rows = test_rows[test_rows["model"].eq("ThermoRoute")]
    full_seed_values = pd.to_numeric(full_rows["seed"], errors="coerce")
    if full_rows.empty or full_seed_values.isna().any() or set(
        full_seed_values.astype(int)
    ) != set(seeds):
        raise ValueError(
            f"ThermoRoute lacks the exact ablation seed registry {seeds}"
        )
    full = full_rows.copy()
    paired_key_columns = ["seed", *FORECAST_KEY]
    if full.duplicated(paired_key_columns).any():
        raise ValueError("ThermoRoute has duplicate seed×forecast keys")

    full_keys = set(full[paired_key_columns].itertuples(index=False, name=None))
    if not full_keys:
        raise ValueError("ThermoRoute has no multi-seed forecast keys")
    result = {"ThermoRoute": full.reset_index(drop=True)}

    for name in controls:
        all_control = normal[normal["model"].eq(name)]
        control_seeds = pd.to_numeric(all_control["seed"], errors="coerce")
        if (
            all_control.empty
            or control_seeds.isna().any()
            or set(control_seeds.astype(int)) != set(seeds)
        ):
            raise ValueError(f"{name} must contain the exact seed registry {seeds}")
        control = all_control[all_control["split"].eq(split)].copy()
        if control.empty:
            raise ValueError(f"{name} is absent from split={split!r}")
        if control.duplicated(paired_key_columns).any():
            raise ValueError(f"{name} has duplicate seed×forecast keys")
        control_keys = set(
            control[paired_key_columns].itertuples(index=False, name=None)
        )
        if control_keys != full_keys:
            raise ValueError(
                f"{name} same-seed forecast keys differ from ThermoRoute"
            )
        aligned = full[paired_key_columns + ["y_true"]].merge(
            control[paired_key_columns + ["y_true"]],
            on=paired_key_columns,
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
                f"{name} same-seed y_true differs from ThermoRoute"
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
    return serialise_offsets(offsets)


def bundle_metadata(identity, wd, clim, imputer, thresholds, event_reference,
                    delta_scale,
                    conformal_offsets, conformal_offset_audit,
                    event_calibrators, *,
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
        "conformal_policy": CF.cqr_policy_contract(),
        "conformal_offset_audit": dict(conformal_offset_audit),
        "calibration_fit_contract": route_a_calibration_fit_contract(
            external=False
        ),
        "source_sha256": identity.source_sha256,
        "panel_sha256": identity.panel_sha256,
        "registry_sha256": identity.registry_sha256,
        "config_sha256": identity.config_sha256,
        "runtime_sha256": identity.runtime_sha256,
        "input_closure_sha256": identity.input_closure_sha256,
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


def write_prediction_artifact(
    frame,
    path,
    identity,
    *,
    kind,
    publication_guard,
    parents=None,
):
    R.write_predictions(
        frame,
        path,
        publication_guard=publication_guard,
    )
    seal_artifact(
        path,
        identity,
        kind=kind,
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents=parents,
        publication_guard=publication_guard,
    )


def read_member_bundle(directory, identity, member_name):
    try:
        weights, metadata = load_inference_bundle(
            directory,
            expected_member_count=1,
            publication_guard=assert_formal_numerical_policy,
        )
    except (FileNotFoundError, ValueError, RuntimeError):
        return None
    expected_identity = (
        metadata.get("run_id") == identity.run_id
        and metadata.get("source_sha256") == identity.source_sha256
        and metadata.get("panel_sha256") == identity.panel_sha256
        and metadata.get("registry_sha256") == identity.registry_sha256
        and metadata.get("runtime_sha256") == identity.runtime_sha256
        and metadata.get("input_closure_sha256")
        == identity.input_closure_sha256
    )
    if not expected_identity or set(weights) != {member_name}:
        return None
    return weights[member_name]


def _assert_exact_repository_json(path: Path, value: Mapping[str, object]) -> None:
    expected = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.read_bytes() != expected:
        raise Stage09ParallelError(
            f"Stage-09 semantic JSON is not canonical: {path.name}"
        )


def _normalise_control_semantic_frame(
    frame: pd.DataFrame,
    *,
    model_id: str,
    seed: int,
    scope: str,
) -> pd.DataFrame:
    if tuple(frame.columns) != tuple(R.PRED_COLS) or frame.empty:
        raise Stage09ParallelError(
            f"{model_id} seed{seed} prediction schema/order is not exact"
        )
    try:
        R.validate_predictions(frame)
    except ValueError as exc:
        raise Stage09ParallelError(
            f"{model_id} seed{seed} predictions fail the canonical schema"
        ) from exc
    expected_literals = {
        "model": model_id,
        "scope": scope,
        "feature_set": "USGS",
    }
    if any(
        not frame[column].map(
            lambda value, expected=expected: type(value) is str
            and value == expected
        ).all()
        for column, expected in expected_literals.items()
    ):
        raise Stage09ParallelError(
            f"{model_id} seed{seed} model/scope/feature_set changed"
        )
    seed_values = pd.to_numeric(frame["seed"], errors="coerce")
    if (
        seed_values.isna().any()
        or pd.api.types.is_bool_dtype(frame["seed"].dtype)
        or not pd.api.types.is_integer_dtype(frame["seed"].dtype)
        or not np.equal(seed_values.to_numpy(dtype=float), float(seed)).all()
    ):
        raise Stage09ParallelError(f"{model_id} seed identity changed")
    if (
        pd.api.types.is_bool_dtype(frame["horizon"].dtype)
        or not pd.api.types.is_integer_dtype(frame["horizon"].dtype)
        or not frame["site_id"].map(
            lambda value: type(value) is str
            and bool(value)
            and value == value.strip()
        ).all()
        or not set(frame["split"].tolist()) <= {"val", "calib", "test"}
    ):
        raise Stage09ParallelError(
            f"{model_id} seed{seed} forecast registry dtype changed"
        )
    for date_column in ("issue_date", "target_date"):
        if (
            str(frame[date_column].dtype) != "datetime64[ns]"
            or frame[date_column].isna().any()
            or not frame[date_column].dt.normalize().equals(frame[date_column])
        ):
            raise Stage09ParallelError(
                f"{model_id} seed{seed} {date_column} is not canonical"
            )
    if frame.duplicated(list(FORECAST_KEY)).any():
        raise Stage09ParallelError(
            f"{model_id} seed{seed} has duplicate forecast keys"
        )
    return frame.sort_values(list(FORECAST_KEY), kind="mergesort").reset_index(
        drop=True
    )


def _forecast_key_and_truth_digests(
    frame: pd.DataFrame,
) -> tuple[str, str]:
    key_digest = hashlib.sha256()
    truth_digest = hashlib.sha256()
    columns = [*FORECAST_KEY, "y_true"]
    for site_id, horizon, issue_date, target_date, y_true in frame[
        columns
    ].itertuples(index=False, name=None):
        site_bytes = str(site_id).encode("utf-8", errors="strict")
        key_bytes = b"".join(
            (
                len(site_bytes).to_bytes(4, "big", signed=False),
                site_bytes,
                int(horizon).to_bytes(8, "big", signed=True),
                int(pd.Timestamp(issue_date).value).to_bytes(
                    8, "big", signed=True
                ),
                int(pd.Timestamp(target_date).value).to_bytes(
                    8, "big", signed=True
                ),
            )
        )
        key_digest.update(key_bytes)
        truth_digest.update(key_bytes)
        truth_digest.update(np.asarray([y_true], dtype="<f4").tobytes())
    return key_digest.hexdigest(), truth_digest.hexdigest()


def _validate_control_checkpoint_semantics(
    payload: Path,
    validated,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    checkpoint = payload / "training_checkpoint.pt"
    sidecar = payload / "training_checkpoint.pt.meta.json"
    try:
        sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage09ParallelError(
            "Stage-09 control checkpoint sidecar is invalid"
        ) from exc
    if not isinstance(sidecar_value, dict):
        raise Stage09ParallelError(
            "Stage-09 control checkpoint sidecar is not an object"
        )
    _assert_exact_repository_json(sidecar, sidecar_value)
    expected_sidecar_fields = {
        "format",
        "checkpoint_format",
        "run_id",
        "epoch",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "resolved_config_sha256",
        "extra_sha256",
        "model_class",
        "optimizer_class",
        "scheduler_class",
        "scheduler_present",
    }
    if (
        set(sidecar_value) != expected_sidecar_fields
        or sidecar_value.get("format") != CHECKPOINT_METADATA_VERSION
        or sidecar_value.get("checkpoint_format") != CHECKPOINT_VERSION
        or sidecar_value.get("run_id") != validated.identity.run_id
        or sidecar_value.get("checkpoint_sha256") != sha256_file(checkpoint)
        or sidecar_value.get("checkpoint_bytes") != checkpoint.stat().st_size
    ):
        raise Stage09ParallelError(
            "Stage-09 control checkpoint sidecar lineage changed"
        )
    validated.assert_unchanged()
    try:
        checkpoint_value = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise Stage09ParallelError(
            "Stage-09 control checkpoint cannot be safely loaded"
        ) from exc
    validated.assert_unchanged()
    checkpoint_fields = {
        "format",
        "run_id",
        "resolved_config_json",
        "resolved_config_sha256",
        "extra_json",
        "extra_sha256",
        "epoch",
        "best_epoch",
        "best_metric",
        "model_class",
        "optimizer_class",
        "scheduler_class",
        "model_state",
        "best_model_state",
        "optimizer_state",
        "scheduler_present",
        "scheduler_state",
        "rng_state",
    }
    if not isinstance(checkpoint_value, dict) or set(checkpoint_value) != checkpoint_fields:
        raise Stage09ParallelError(
            "Stage-09 control checkpoint payload schema changed"
        )
    resolved = validated.authorization["resolved_config"]
    scientific = validated.document["scientific_member_config"]
    if not isinstance(resolved, Mapping) or not isinstance(scientific, Mapping):
        raise Stage09ParallelError("Stage-09 checkpoint authority is malformed")
    intervention = scientific.get("model_kwargs_intervention")
    if not isinstance(intervention, Mapping):
        raise Stage09ParallelError("Stage-09 checkpoint intervention is malformed")
    model_kwargs = dict(intervention)
    model_kwargs.setdefault("delta_scale", resolved["delta_scale"])
    expected_resolved = {
        **dict(resolved),
        "arm": validated.member.arm_id,
        "seed": validated.member.seed,
        "model_kwargs": model_kwargs,
    }
    expected_resolved_json = json.dumps(
        expected_resolved,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    best_epoch = checkpoint_value.get("best_epoch")
    best_metric = checkpoint_value.get("best_metric")
    epoch = checkpoint_value.get("epoch")
    model_state = checkpoint_value.get("model_state")
    best_state = checkpoint_value.get("best_model_state")
    extra_json = checkpoint_value.get("extra_json")
    try:
        decoded_extra = json.loads(extra_json) if isinstance(extra_json, str) else None
        canonical_extra = (
            json.dumps(
                decoded_extra,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            if isinstance(decoded_extra, dict)
            else None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        canonical_extra = None

    def safe_tensor_state(value: object) -> bool:
        return bool(
            isinstance(value, dict)
            and value
            and all(type(key) is str for key in value)
            and all(
                isinstance(tensor, torch.Tensor)
                and tensor.device.type == "cpu"
                and (
                    not (tensor.is_floating_point() or tensor.is_complex())
                    or bool(torch.isfinite(tensor).all().item())
                )
                for tensor in value.values()
            )
        )

    if (
        checkpoint_value.get("format") != CHECKPOINT_VERSION
        or checkpoint_value.get("run_id") != validated.identity.run_id
        or checkpoint_value.get("resolved_config_json")
        != expected_resolved_json
        or checkpoint_value.get("resolved_config_sha256")
        != hashlib.sha256(expected_resolved_json.encode("utf-8")).hexdigest()
        or sidecar_value.get("resolved_config_sha256")
        != checkpoint_value.get("resolved_config_sha256")
        or type(extra_json) is not str
        or canonical_extra != extra_json
        or checkpoint_value.get("extra_sha256")
        != hashlib.sha256(extra_json.encode("utf-8")).hexdigest()
        or sidecar_value.get("extra_sha256")
        != checkpoint_value.get("extra_sha256")
        or checkpoint_value.get("model_class")
        != "thermoroute.thermoroute.ThermoRoute"
        or sidecar_value.get("model_class") != checkpoint_value.get("model_class")
        or checkpoint_value.get("optimizer_class")
        != "torch.optim.adamw.AdamW"
        or checkpoint_value.get("scheduler_class")
        != "torch.optim.lr_scheduler.ReduceLROnPlateau"
        or checkpoint_value.get("scheduler_present") is not True
        or sidecar_value.get("optimizer_class")
        != checkpoint_value.get("optimizer_class")
        or sidecar_value.get("scheduler_class")
        != checkpoint_value.get("scheduler_class")
        or sidecar_value.get("scheduler_present")
        != checkpoint_value.get("scheduler_present")
        or type(epoch) is not int
        or epoch < 0
        or sidecar_value.get("epoch") != epoch
        or type(best_epoch) is not int
        or best_epoch < 0
        or best_epoch > epoch
        or isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not np.isfinite(float(best_metric))
        or not safe_tensor_state(model_state)
        or not safe_tensor_state(best_state)
        or set(model_state) != set(best_state)
        or not isinstance(checkpoint_value.get("optimizer_state"), dict)
        or not isinstance(checkpoint_value.get("scheduler_state"), dict)
        or not isinstance(checkpoint_value.get("rng_state"), dict)
    ):
        raise Stage09ParallelError(
            "Stage-09 control checkpoint scientific lineage changed"
        )
    return (
        {
            "checkpoint_payload_sha256": sha256_file(checkpoint),
            "checkpoint_sidecar_sha256": sha256_file(sidecar),
            "checkpoint_best_epoch_index": int(best_epoch),
            "checkpoint_best_selection_metric_value": float(best_metric),
            "checkpoint_weights_safely_loaded": True,
        },
        dict(best_state),
    )


def _validate_stage09_control_payload_semantics(
    payload: Path,
    validated,
    *,
    replay_wd=None,
    prediction_replay: Callable[[Path, pd.DataFrame, object], float] | None = None,
) -> dict[str, object]:
    """Safely load and pair one control with exact same-seed ThermoRoute truth."""
    member = validated.member
    prediction = payload / "predictions.parquet"
    prediction_sidecar = sidecar_path(prediction)
    try:
        lineage = validate_artifact_sidecar(
            prediction,
            identity=validated.identity,
            schema=R.PREDICTION_SCHEMA_VERSION,
            kind="thermoroute_ablation_seed_predictions",
        )
    except (OSError, ValueError) as exc:
        raise Stage09ParallelError(
            f"{member.member_id} prediction lineage changed"
        ) from exc
    _assert_exact_repository_json(prediction_sidecar, lineage)
    try:
        control = pd.read_parquet(prediction)
    except Exception as exc:
        raise Stage09ParallelError(
            f"{member.member_id} prediction parquet cannot be loaded"
        ) from exc
    control = _normalise_control_semantic_frame(
        control,
        model_id=member.arm_id,
        seed=member.seed,
        scope="ablation_usgs",
    )

    member_name = f"seed{member.seed}"
    bundle = payload / "bundle"
    try:
        weights, metadata = load_inference_bundle(
            bundle,
            expected_member_count=1,
            map_location="cpu",
            publication_guard=validated.assert_unchanged,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise Stage09ParallelError(
            f"{member.member_id} inference bundle cannot be safely loaded"
        ) from exc
    metadata_path = bundle / "metadata.json"
    weights_path = bundle / "weights.pt"
    _assert_exact_repository_json(metadata_path, metadata)
    scientific = validated.document["scientific_member_config"]
    if not isinstance(scientific, Mapping):
        raise Stage09ParallelError("Stage-09 scientific member config is malformed")
    intervention = scientific.get("model_kwargs_intervention")
    architecture = metadata.get("architecture")
    if (
        not isinstance(intervention, Mapping)
        or not isinstance(architecture, Mapping)
        or set(architecture) != {"class", "kwargs", "train_config"}
        or architecture.get("class")
        != "thermoroute.thermoroute.ThermoRoute"
        or architecture.get("train_config") != scientific.get("train_config")
        or not isinstance(architecture.get("kwargs"), Mapping)
    ):
        raise Stage09ParallelError(
            f"{member.member_id} bundle architecture metadata changed"
        )
    kwargs = dict(architecture["kwargs"])
    n_phys = kwargs.get("n_phys")
    expected_kwargs = {
        "n_vars": len(USGS_VARS),
        "n_stations": len(C.STATIONS),
        "n_phys": n_phys,
        "station_agnostic": False,
        "use_prior": True,
        "use_router": True,
        "use_moe": True,
        "sparse_router": True,
        "fixed_kappa": False,
        "delta_scale": validated.authorization["resolved_config"]["delta_scale"],
        "use_tcn": True,
        "residual_model": True,
        "safety_anchor": "damped",
        "use_wlevel": False,
    }
    expected_kwargs.update(dict(intervention))
    identity_fields = {
        "run_id": validated.identity.run_id,
        "source_sha256": validated.identity.source_sha256,
        "panel_sha256": validated.identity.panel_sha256,
        "registry_sha256": validated.identity.registry_sha256,
        "config_sha256": validated.identity.config_sha256,
        "runtime_sha256": validated.identity.runtime_sha256,
        "input_closure_sha256": validated.identity.input_closure_sha256,
    }
    state = weights.get(member_name)
    if (
        type(n_phys) is not int
        or n_phys < 1
        or kwargs != expected_kwargs
        or any(metadata.get(key) != value for key, value in identity_fields.items())
        or metadata.get("feature_order") != list(USGS_VARS)
        or metadata.get("horizons") != list(C.HORIZONS)
        or metadata.get("station_to_index")
        != {station: index for index, station in enumerate(C.STATIONS)}
        or metadata.get("training_device") != "cpu"
        or metadata.get("members") != [member_name]
        or metadata.get("member_count") != 1
        or not isinstance(state, dict)
        or not state
        or any(
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or (
                (value.is_floating_point() or value.is_complex())
                and not bool(torch.isfinite(value).all().item())
            )
            for value in state.values()
        )
    ):
        raise Stage09ParallelError(
            f"{member.member_id} bundle identity/intervention changed"
        )

    reference = (
        validated.run_directory
        / "predictions"
        / f"thermoroute_seed{member.seed}.parquet"
    )
    reference_sidecar = sidecar_path(reference)
    try:
        reference_lineage = validate_artifact_sidecar(
            reference,
            identity=validated.identity,
            schema=R.PREDICTION_SCHEMA_VERSION,
            kind="thermoroute_seed_predictions",
        )
    except (OSError, ValueError) as exc:
        raise Stage09ParallelError(
            f"ThermoRoute seed{member.seed} reference lineage changed"
        ) from exc
    _assert_exact_repository_json(reference_sidecar, reference_lineage)
    try:
        reference_frame = pd.read_parquet(reference)
    except Exception as exc:
        raise Stage09ParallelError(
            f"ThermoRoute seed{member.seed} reference parquet cannot be loaded"
        ) from exc
    reference_frame = _normalise_control_semantic_frame(
        reference_frame,
        model_id="ThermoRoute",
        seed=member.seed,
        scope="joint_usgs",
    )
    if (
        len(control) != len(reference_frame)
        or not control.loc[:, list(FORECAST_KEY)].equals(
            reference_frame.loc[:, list(FORECAST_KEY)]
        )
        or not np.array_equal(
            control["split"].to_numpy(dtype=object),
            reference_frame["split"].to_numpy(dtype=object),
        )
        or not np.array_equal(
            control["y_true"].to_numpy(dtype=np.float32),
            reference_frame["y_true"].to_numpy(dtype=np.float32),
        )
    ):
        raise Stage09ParallelError(
            f"{member.member_id} differs from same-seed ThermoRoute keys/truth"
        )
    key_digest, truth_digest = _forecast_key_and_truth_digests(control)
    checkpoint, checkpoint_best_state = _validate_control_checkpoint_semantics(
        payload, validated
    )
    if set(state) != set(checkpoint_best_state) or any(
        not torch.equal(state[name], checkpoint_best_state[name])
        for name in state
    ):
        raise Stage09ParallelError(
            f"{member.member_id} bundle differs from checkpoint best state"
        )
    if replay_wd is not None and prediction_replay is not None:
        raise Stage09ParallelError("Stage-09 prediction replay authority is ambiguous")
    try:
        if prediction_replay is not None:
            replay_difference = prediction_replay(bundle, control, validated)
        elif replay_wd is not None:
            replay_difference = verify_sequence_prediction_parity(
                bundle,
                wd=replay_wd,
                expected=control,
                model_factory=lambda _member, metadata:
                    thermoroute_factory_from_metadata(metadata),
                member_seeds={member_name: member.seed},
                atol=0.0,
                batch_size=int(
                    validated.authorization["resolved_config"]["eval_batch_size"]
                ),
                publication_guard=validated.assert_unchanged,
            )
        else:
            raise Stage09ParallelError(
                "Stage-09 control validation lacks prediction replay data"
            )
    except Stage09ParallelError:
        raise
    except Exception as exc:
        raise Stage09ParallelError(
            f"{member.member_id} bundle prediction replay failed"
        ) from exc
    if (
        isinstance(replay_difference, bool)
        or not isinstance(replay_difference, (int, float))
        or not np.isfinite(float(replay_difference))
        or float(replay_difference) != 0.0
    ):
        raise Stage09ParallelError(
            f"{member.member_id} bundle prediction replay is not exact"
        )
    return {
        "format": SEMANTIC_VALIDATION_FORMAT,
        "model_id": member.arm_id,
        "seed": member.seed,
        "scope": "ablation_usgs",
        "feature_set": "USGS",
        "prediction_schema": R.PREDICTION_SCHEMA_VERSION,
        "prediction_artifact_sha256": sha256_file(prediction),
        "prediction_sidecar_sha256": sha256_file(prediction_sidecar),
        "bundle_metadata_sha256": sha256_file(metadata_path),
        "bundle_weights_sha256": sha256_file(weights_path),
        "bundle_member_id": member_name,
        **checkpoint,
        "same_seed_reference_prediction_sha256": sha256_file(reference),
        "same_seed_reference_sidecar_sha256": sha256_file(reference_sidecar),
        "forecast_record_count": len(control),
        "forecast_key_sha256": key_digest,
        "truth_float32_sha256": truth_digest,
        "prediction_sidecar_run_identity_exact": True,
        "bundle_metadata_identity_exact": True,
        "bundle_member_registry_exact": True,
        "bundle_intervention_exact": True,
        "bundle_weights_safely_loaded": True,
        "bundle_matches_checkpoint_best_state_exact": True,
        "same_seed_forecast_keys_exact": True,
        "same_seed_split_exact": True,
        "same_seed_y_true_float32_exact": True,
        "full_prediction_replay_equivalence_proved": True,
        "full_prediction_replay_max_abs_difference": 0.0,
    }


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
    """Fit CQR/Platt on the independent-horizon 2018 calibration population.

    Callers must supply calib rows whose inclusion already matches confirmation
    (per-horizon observed targets inside ``C.SPLIT.calib``).  Masked / non-finite
    labels are dropped fail-closed before fitting.
    """
    ensemble = ensemble_prediction_frame(predictions)
    calibration = ensemble[ensemble.split == "calib"].copy()
    if calibration.empty:
        raise ValueError("calibration artifacts require a non-empty calib split")
    y_true = calibration["y_true"].to_numpy(float)
    finite = np.isfinite(y_true)
    if not finite.any():
        raise ValueError("calibration artifacts lack finite observed targets")
    if not finite.all():
        calibration = calibration.loc[finite].copy()
    calibration["threshold"] = calibration["site_id"].astype(str).map(thresholds)
    if calibration["threshold"].isna().any():
        raise KeyError("calibration rows contain a site without an event threshold")
    calibration["event"] = (
        calibration["y_true"].to_numpy(float)
        > calibration["threshold"].to_numpy(float)
    ).astype(int)
    offsets, offset_audit = CF.cqr_offsets_with_audit(
        calibration, alpha=0.10
    )
    calibrators = fit_horizon_calibrators(
        calibration, probability_col="p_exceed", outcome_col="event",
        min_samples=100,
        weighting=STATION_EQUAL_WEIGHTING,
    )
    expected_offsets = {(str(site), int(horizon)) for site in thresholds
                        for horizon in C.HORIZONS}
    if set(offsets) != expected_offsets:
        missing = sorted(expected_offsets - set(offsets))
        raise ValueError(f"calibration lacks frozen site×horizon offsets: {missing[:5]}")
    if set(calibrators) != set(C.HORIZONS):
        raise ValueError("event calibration lacks a declared horizon")
    return offsets, offset_audit, calibrators


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
            validation_prediction = candidate.predict(Xva, num_threads=1)
            score = _station_macro_validation_rmse(va, validation_prediction)
            assert_formal_numerical_policy()
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
                    fitted = trainer()
                    assert_formal_numerical_policy()
                    return fitted
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
                    assert_formal_numerical_policy()
                    return cached
                fitted = trainer()
                return save_lightgbm_shard(
                    shard_cache_path, lineage=lineage, model=fitted,
                    parity_input=parity_inputs[h], parity_atol=1e-12,
                    publication_guard=assert_formal_numerical_policy,
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
                "n_jobs": MAIN_THREADS,
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
                    reg_lambda=1.0, verbosity=-1, n_jobs=MAIN_THREADS,
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
                    "n_jobs": MAIN_THREADS,
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
            shard_cache_path,
            lineages=shard_lineages,
            parity_inputs=parity_inputs,
            publication_guard=assert_formal_numerical_policy,
        )
    return (
        pd.concat(frames, ignore_index=True), pd.DataFrame(selection_rows),
        models, parity_inputs, evaluation_design, tuple(design_order or ()),
    )


def _run_stage09_control_member(work_order_path: Path) -> int:
    """Execute exactly one independently authorized control member."""
    validated = validate_live_work_order(
        root=ROOT,
        work_order_path=work_order_path,
    )
    resolved = validated.authorization["resolved_config"]
    scientific = validated.document["scientific_member_config"]
    if not isinstance(resolved, dict) or not isinstance(scientific, dict):
        raise Stage09ParallelError("authorized Stage-09 worker config is malformed")
    if resolved["train_config"] != asdict(CFG):
        raise Stage09ParallelError("authorized Stage-09 TrainConfig changed")
    member = validated.member
    intervention = scientific["model_kwargs_intervention"]
    if (
        not isinstance(intervention, dict)
        or intervention != ABLATION_INTERVENTIONS[member.arm_id]
    ):
        raise Stage09ParallelError(
            "authorized Stage-09 control intervention changed"
        )
    model_kw = dict(intervention)
    model_kw.setdefault("delta_scale", resolved["delta_scale"])
    panel_path = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
    scientific_context: dict[str, object] = {}

    def resolve_scientific_context():
        if scientific_context:
            return (
                scientific_context["panel"],
                scientific_context["panel_imp"],
                scientific_context["masks"],
                scientific_context["clim"],
                scientific_context["stations"],
                scientific_context["imputer"],
                scientific_context["wd"],
                scientific_context["thresholds"],
                scientific_context["event_reference"],
            )
        panel, panel_imp, masks, clim, stations, imputer = prep(str(panel_path))
        wd = DS.build_windows(
            panel_imp,
            masks,
            clim,
            variables=USGS_VARS,
            require_observed_target=True,
        )
        thresholds = {
            station: float(
                panel.loc[masks.train]
                .query("site_id==@station")
                .WTEMP.quantile(0.9)
            )
            for station in stations
        }
        event_reference = fit_frozen_seasonal_event_reference(
            panel,
            thresholds,
            pooled=False,
            fit_interval=("2006-01-01", "2018-12-31"),
        )
        scientific_context.update(
            {
                "panel": panel,
                "panel_imp": panel_imp,
                "masks": masks,
                "clim": clim,
                "stations": stations,
                "imputer": imputer,
                "wd": wd,
                "thresholds": thresholds,
                "event_reference": event_reference,
            }
        )
        return (
            panel,
            panel_imp,
            masks,
            clim,
            stations,
            imputer,
            wd,
            thresholds,
            event_reference,
        )

    def produce(payload: Path) -> dict[str, object]:
        (
            _panel,
            _panel_imp,
            _masks,
            clim,
            stations,
            imputer,
            wd,
            thr,
            event_reference,
        ) = resolve_scientific_context()

        def factory():
            return ThermoRoute(
                n_vars=len(wd.var_names),
                n_stations=len(stations),
                n_phys=wd.n_phys,
                safety_anchor="damped",
                **model_kw,
            )

        # The native-policy callback is kept literal inside fit_model because
        # checkpoint save/recovery executes there.  The complete source/input/
        # matrix guard brackets each larger scientific transaction below.
        validated.assert_unchanged()
        result = fit_model(
            factory,
            wd,
            thr,
            cfg=CFG,
            seed=member.seed,
            model_name=member.arm_id,
            device="cpu",
            eval_batch_size=int(resolved["eval_batch_size"]),
            scope="ablation_usgs",
            feature_set="USGS",
            station_balanced=True,
            selection_metric="station_macro",
            checkpoint_path=validated.resume_checkpoint_path,
            run_id=validated.identity.run_id,
            resolved_config={
                **resolved,
                "arm": member.arm_id,
                "seed": member.seed,
                "model_kwargs": model_kw,
            },
            artifact_publication_guard=assert_formal_numerical_policy,
        )
        validated.assert_unchanged()
        result.pred["seed"] = member.seed
        write_prediction_artifact(
            result.pred,
            payload / "predictions.parquet",
            validated.identity,
            kind="thermoroute_ablation_seed_predictions",
            publication_guard=validated.assert_unchanged,
        )
        validated.assert_unchanged()
        offsets, offset_audit, calibrators = calibration_artifacts(
            result.pred, thr
        )
        member_name = f"seed{member.seed}"
        save_inference_bundle(
            payload / "bundle",
            members={member_name: result.model},
            metadata=bundle_metadata(
                validated.identity,
                wd,
                clim,
                imputer,
                thr,
                event_reference,
                float(resolved["delta_scale"]),
                offsets,
                offset_audit,
                calibrators,
                training_device="cpu",
                architecture_overrides=model_kw,
            ),
            expected_member_count=1,
            publication_guard=validated.assert_unchanged,
        )
        validated.assert_unchanged()
        materialize_member_resume_checkpoint(
            validated=validated,
            payload=payload,
        )
        validated.assert_unchanged()
        return {
            "best_epoch_index": int(result.epochs),
            "best_selection_metric_value": float(result.best_val),
        }

    def validate_payload(payload: Path, view) -> Mapping[str, object]:
        replay_wd = resolve_scientific_context()[6]
        return _validate_stage09_control_payload_semantics(
            payload,
            view,
            replay_wd=replay_wd,
        )

    archive, reused = publish_control_member(
        validated=validated,
        producer=produce,
        semantic_validator=validate_payload,
    )
    state = "verified reuse" if reused else "new create-only publication"
    log(f"  {member.arm_id} seed{member.seed}: {state} at {archive.name}")
    return 0


class _Stage09WorkerLauncher:
    """Own and terminate the isolated member processes for one scheduler run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}
        self._stopping = False

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], value: int) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, value)
        except ProcessLookupError:
            return

    def __call__(self, work_order_path: Path) -> int:
        """Launch one clean interpreter in its own cancellable process group."""
        with tempfile.TemporaryDirectory(
            prefix="thermoroute-stage09-member-pycache-"
        ) as cache:
            cache_path = Path(cache).resolve()
            if any(cache_path.iterdir()):
                raise RuntimeError(
                    "Stage-09 member pycache was not initially empty"
                )
            nonce = secrets.token_hex(32)
            (cache_path / ".controller-nonce").write_text(
                nonce, encoding="utf-8"
            )
            environment = _formal_worker_environment(
                cache_path, nonce, threads=MAIN_THREADS
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-X",
                    f"pycache_prefix={cache}",
                    str(Path(__file__).resolve()),
                    _WORKER_ARGUMENT,
                    "--_control-member-work-order",
                    str(work_order_path.resolve()),
                ],
                cwd=ROOT,
                env=environment,
                start_new_session=True,
            )
            with self._lock:
                stopping = self._stopping
                if not stopping:
                    self._active[process.pid] = process
            if stopping:
                self._signal_group(process, signal.SIGTERM)
            try:
                return int(process.wait())
            finally:
                with self._lock:
                    self._active.pop(process.pid, None)

    def terminate_all(self) -> None:
        """Stop and reap every active process after failure or interruption."""
        with self._lock:
            self._stopping = True
            active = tuple(self._active.values())
        for process in active:
            self._signal_group(process, signal.SIGTERM)
        for process in active:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._signal_group(process, signal.SIGKILL)
        for process in active:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired as exc:  # pragma: no cover - OS fault
                raise RuntimeError(
                    f"Stage-09 worker process would not terminate: {process.pid}"
                ) from exc


def _launch_stage09_control_member(work_order_path: Path) -> int:
    """Compatibility wrapper for one independently owned member launch."""
    return _Stage09WorkerLauncher()(work_order_path)


_SEED_WORKER_STATE: dict[str, object] = {}
SEED_WORKER_PROCESSES = int(
    os.environ.get("THERMOROUTE_SEED_PROCESSES") or "5"
)


class _SeedWorkerLauncher:
    """Own and terminate the isolated ThermoRoute seed worker processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}
        self._stopping = False

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], value: int) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, value)
        except ProcessLookupError:
            return

    def __call__(self, order_path: Path) -> int:
        """Launch one clean interpreter in its own cancellable process group."""
        with tempfile.TemporaryDirectory(
            prefix="thermoroute-stage09-seed-pycache-"
        ) as cache:
            cache_path = Path(cache).resolve()
            if any(cache_path.iterdir()):
                raise RuntimeError(
                    "Stage-09 seed pycache was not initially empty"
                )
            nonce = secrets.token_hex(32)
            (cache_path / ".controller-nonce").write_text(
                nonce, encoding="utf-8"
            )
            environment = _formal_worker_environment(
                cache_path, nonce, threads=MAIN_THREADS
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-X",
                    f"pycache_prefix={cache}",
                    str(Path(__file__).resolve()),
                    _WORKER_ARGUMENT,
                    "--_thermoroute-seed-worker",
                    str(order_path.resolve()),
                ],
                cwd=ROOT,
                env=environment,
                start_new_session=True,
            )
            with self._lock:
                stopping = self._stopping
                if not stopping:
                    self._active[process.pid] = process
            if stopping:
                self._signal_group(process, signal.SIGTERM)
            try:
                return int(process.wait())
            finally:
                with self._lock:
                    self._active.pop(process.pid, None)

    def terminate_all(self) -> None:
        """Stop and reap every active process after failure or interruption."""
        with self._lock:
            self._stopping = True
            active = tuple(self._active.values())
        for process in active:
            self._signal_group(process, signal.SIGTERM)
        for process in active:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._signal_group(process, signal.SIGKILL)
        for process in active:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired as exc:  # pragma: no cover - OS fault
                raise RuntimeError(
                    f"Stage-09 seed worker would not terminate: {process.pid}"
                ) from exc


def _launch_seed_workers(
    orders: Mapping[int, Path],
    *,
    max_workers: int,
    launch: Callable[[Path], int],
    terminate: Callable[[], None],
) -> list:
    """Run the independent seed workers with bounded concurrency, fail fast."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(launch, order): seed for seed, order in orders.items()
        }
        for future in as_completed(futures):
            try:
                return_code = future.result()
            except BaseException:
                terminate()
                raise
            if type(return_code) is not int or return_code != 0:
                terminate()
                raise RuntimeError(
                    f"Stage-09 seed worker failed with exit code {return_code}"
                )
            completed += 1
    if completed != len(orders):
        raise RuntimeError("Stage-09 seed worker set is incomplete")
    return []


def _run_stage09_seed_worker(order_path: Path) -> int:
    """Execute exactly one independently authorized ThermoRoute seed."""
    try:
        order = json.loads(order_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Stage-09 seed order is invalid") from exc
    if (
        not isinstance(order, dict)
        or order.get("format") != "thermoroute.stage09-seed-order.v1"
    ):
        raise RuntimeError("Stage-09 seed order format changed")
    seed = order.get("seed")
    if type(seed) is not int or seed not in C.USGS_SEEDS:
        raise RuntimeError("Stage-09 seed order seed is invalid")
    identity = RunIdentity(**order["identity"])
    run_config = order["run_config"]
    panel_path = Path(order["panel"]).resolve()
    registry_path = ROOT / "data_usgs" / "station_registry_v1.csv"
    resolved_device = str(order["device"])
    delta_scale = float(order["delta_scale"])
    station_sampling = str(order["station_sampling"])
    eval_batch_size = int(order["eval_batch_size"])
    prediction_path = Path(order["prediction"]).resolve()
    bundle_path = Path(order["bundle"]).resolve()
    checkpoint_path = Path(order["checkpoint"]).resolve()

    development_input_closure = resolve_development_input_closure(ROOT)
    input_closure_sha256 = (
        development_input_closure.binding_digest
        if development_input_closure is not None
        else compose_input_closure_digest({
            "panel": sha256_file(panel_path),
            "registry": sha256_file(registry_path),
        })
    )
    rederived = resolve_run_identity(
        root=ROOT,
        panel=panel_path,
        registry=registry_path,
        config=run_config,
        input_closure_sha256=input_closure_sha256,
    )
    if rederived != identity:
        raise RuntimeError("Stage-09 seed order identity changed")

    panel, panel_imp, masks, clim, stations, imputer = prep(str(panel_path))
    wd = DS.build_windows(panel_imp, masks, clim, variables=USGS_VARS,
                          require_observed_target=True)
    thr = {
        station: float(
            panel.loc[masks.train]
            .query("site_id==@station")
            .WTEMP.quantile(0.9)
        )
        for station in stations
    }
    event_reference = fit_frozen_seasonal_event_reference(
        panel,
        thr,
        pooled=False,
        fit_interval=("2006-01-01", "2018-12-31"),
    )
    member_name = f"seed{seed}"
    factory = lambda: ThermoRoute(
        n_vars=len(wd.var_names), n_stations=len(stations), n_phys=wd.n_phys,
        delta_scale=delta_scale, safety_anchor="damped")
    res = fit_model(
        factory, wd, thr, cfg=CFG, seed=seed,
        device=resolved_device, eval_batch_size=eval_batch_size,
        model_name="ThermoRoute", scope="joint_usgs",
        feature_set="USGS",
        station_balanced=station_sampling == "balanced",
        selection_metric=(
            "station_macro" if station_sampling == "balanced" else "micro"
        ),
        checkpoint_path=checkpoint_path,
        run_id=identity.run_id,
        resolved_config={**run_config, "arm": "ThermoRoute", "seed": seed},
        artifact_publication_guard=assert_formal_numerical_policy,
    )
    model = res.model
    res.pred["seed"] = seed
    write_prediction_artifact(
        res.pred, prediction_path, identity,
        kind="thermoroute_seed_predictions",
        publication_guard=assert_formal_numerical_policy,
    )
    seed_offsets, seed_offset_audit, seed_calibrators = calibration_artifacts(
        res.pred, thr
    )
    save_inference_bundle(
        bundle_path,
        members={member_name: model},
        metadata=bundle_metadata(
            identity, wd, clim, imputer, thr, event_reference,
            delta_scale,
            seed_offsets, seed_offset_audit, seed_calibrators,
            training_device=resolved_device,
        ),
        expected_member_count=1,
        publication_guard=assert_formal_numerical_policy,
    )
    log(
        f"  ThermoRoute seed{seed}: {res.epochs+1}ep "
        f"val={res.best_val:.4f}"
    )
    return 0


def main():
    # Importing the module is intentionally side-effect free with respect to a
    # formal role.  The role-specific cap belongs to an actual Stage-09
    # invocation; asserting it during test collection made the repository
    # impossible to collect under the ordinary one-thread test policy.
    assert_role_thread_cap(ROOT, "stage09")
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
                    help=f"add {AIR2STREAM_DISPLAY_NAME} unofficial empirical "
                         "comparators (slower)")
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
    ap.add_argument(
        "--control-workers",
        type=int,
        default=DEFAULT_CONTROL_WORKERS,
        help=(
            "execution-only Stage-09 control-member concurrency (1--"
            f"{MAX_CONTROL_WORKERS}, default {DEFAULT_CONTROL_WORKERS}); "
            f"memory-safe recommendation <= {RECOMMENDED_MAX_CONTROL_WORKERS}; "
            "never enters scientific RunIdentity"
        ),
    )
    ap.add_argument(
        "--_control-member-work-order",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--_thermoroute-seed-worker",
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()
    if args._thermoroute_seed_worker is not None:
        # This mode accepts no ambient scientific override.  Every value is
        # recovered from and checked against the parent-frozen seed order.
        if sys.argv[1:] != [
            "--_thermoroute-seed-worker",
            args._thermoroute_seed_worker,
        ]:
            ap.error(
                "the internal seed worker accepts only its exact "
                "work-order argument"
            )
        return _run_stage09_seed_worker(
            Path(args._thermoroute_seed_worker)
        )
    if args._control_member_work_order is not None:
        # This mode accepts no ambient scientific override.  Every value is
        # recovered from and checked against the parent-frozen work order.
        if sys.argv[1:] != [
            "--_control-member-work-order",
            args._control_member_work_order,
        ]:
            ap.error(
                "the internal control-member worker accepts only its exact "
                "work-order argument"
            )
        return _run_stage09_control_member(
            Path(args._control_member_work_order)
        )
    if not 1 <= args.control_workers <= MAX_CONTROL_WORKERS:
        ap.error(
            f"--control-workers must be between 1 and {MAX_CONTROL_WORKERS}"
        )
    if args.control_workers > RECOMMENDED_MAX_CONTROL_WORKERS:
        log(
            "WARNING: --control-workers above "
            f"{RECOMMENDED_MAX_CONTROL_WORKERS} may exceed memory capacity; "
            "formal safety is preserved but the run may be killed by the OS"
        )
    if args.seeds < 1 or args.seeds > len(C.USGS_SEEDS):
        ap.error(f"--seeds must be between 1 and {len(C.USGS_SEEDS)}")
    if args.ablations and args.seeds != len(STAGE9_ABLATION_SEEDS):
        ap.error(
            "--ablations requires the complete five-seed ThermoRoute ensemble; "
            "use --no-ablations for a reduced-seed exploratory diagnostic"
        )

    panel_path = Path(args.panel).resolve()
    registry_path = ROOT / "data_usgs" / "station_registry_v1.csv"
    requested_predictions = (C.PREDICTIONS / args.out_predictions).resolve()
    requested_scores = (C.TABLES / args.out_scores).resolve()
    requested_report = (C.REPORTS / args.out_report).resolve()
    canonical_output_paths = stage09_outputs_are_canonical(
        root=ROOT,
        predictions=requested_predictions,
        scores=requested_scores,
        report=requested_report,
    )
    if not canonical_output_paths and not args.exploratory:
        ap.error(
            "custom Stage-9 output paths require --exploratory and can never "
            "publish a formal receipt or model suite"
        )
    resolved_device = str(resolve_device(args.device))
    if resolved_device != "cpu" and not args.exploratory:
        ap.error("non-CPU Stage-9 runs require --exploratory and cannot publish formal pointers")
    formal_candidate = formal_publication_candidate(
        panel_path=panel_path,
        training_device=resolved_device,
        exploratory=args.exploratory,
    )
    if formal_candidate and args.air2stream:
        ap.error(
            "--air2stream is an exploratory, model-specific-eligibility "
            "reference; a formal Route-A run must omit it"
        )
    formal_configuration_complete = formal_stage09_configuration(
        panel_path=panel_path,
        training_device=resolved_device,
        exploratory=args.exploratory,
        canonical_output_paths=canonical_output_paths,
        seeds=args.seeds,
        station_sampling=args.station_sampling,
        delta_scale=args.delta_scale,
        ablations=args.ablations,
        air2stream=args.air2stream,
    )
    stage09_matrix_gate = (
        validate_stage09_model_matrix_gate(ROOT)
        if formal_configuration_complete
        else None
    )
    predictor_bridge = (
        development_predictor_bridge_binding(
            ROOT,
            panel_sha256=sha256_file(panel_path),
            registry_sha256=sha256_file(registry_path),
        )
        if formal_configuration_complete else None
    )
    development_input_closure = (
        resolve_development_input_closure(ROOT)
        if formal_configuration_complete else None
    )
    input_closure_sha256 = (
        development_input_closure.binding_digest
        if development_input_closure is not None
        else compose_input_closure_digest({
            "panel": sha256_file(panel_path),
            "registry": sha256_file(registry_path),
        })
    )
    input_closure_file_count = (
        len(development_input_closure.inventory)
        if development_input_closure is not None else 2
    )
    protocol = f"route_a_strict_v1_{args.station_sampling}_delta{args.delta_scale:g}"
    run_config = {
        "stage": "09_usgs_experiment",
        "protocol": protocol,
        "panel": panel_path.name,
        "station_registry": registry_path.name,
        "variables": USGS_VARS,
        "horizons": C.HORIZONS,
        "context_length": C.CONTEXT_LENGTH,
        "seeds": args.seeds,
        "time_split": C.SPLIT.as_dict(),
        "train_config": asdict(CFG),
        "thermoroute_seeds": C.USGS_SEEDS[:args.seeds],
        "lightgbm_seeds": C.USGS_SEEDS,
        "ablation_seeds": STAGE9_ABLATION_SEEDS,
        "delta_scale": args.delta_scale,
        "station_sampling": args.station_sampling,
        "selection_metric": ("station_macro" if args.station_sampling == "balanced"
                             else "micro"),
        "ablations": bool(args.ablations),
        "air2stream": bool(args.air2stream),
        "device": resolved_device,
        "training_device": resolved_device,
        "execution_role": (
            "route_a_formal_candidate"
            if formal_configuration_complete
            else "run_scoped_diagnostic_only"
        ),
        "development_predictor_bridge": predictor_bridge,
        "eval_batch_size": args.eval_batch_size,
        "lightgbm_validation_grid": LGB_VALIDATION_GRID,
        "event_reference_fit_interval": ("2006-01-01", "2018-12-31"),
        "formal_numerical_policy": runtime_policy,
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": input_closure_file_count,
    }
    identity = resolve_run_identity(
        root=ROOT,
        panel=panel_path,
        registry=registry_path,
        config=run_config,
        input_closure_sha256=input_closure_sha256,
    )
    if development_input_closure is not None:
        development_input_closure.assert_unchanged()

    def assert_stage09_publication_inputs() -> None:
        assert_formal_numerical_policy()
        if development_input_closure is not None:
            development_input_closure.assert_unchanged()
        if source_tree_hash(ROOT) != identity.source_sha256:
            raise RuntimeError("Stage-09 source tree changed during execution")
        if stage09_matrix_gate is not None:
            for label in ("amendment", "seal"):
                binding = stage09_matrix_gate.binding[label]
                if (
                    sha256_file(ROOT / binding["path"]) != binding["sha256"]
                    or (ROOT / binding["path"]).stat().st_size
                    != binding["bytes"]
                ):
                    raise RuntimeError(
                        f"Stage-09 model-matrix {label} bytes changed"
                    )
    if formal_configuration_complete:
        # Establish canonical data/source eligibility before any canonical
        # result path can be selected or mutated.  A failed formal preflight is
        # an error, never an implicit downgrade that could overwrite a receipt.
        development_contract = canonical_development_contract(
            ROOT,
            ROOT / "data_usgs" / "frozen_panel_v1.json",
            panel_sha256=identity.panel_sha256,
            registry_sha256=identity.registry_sha256,
            source_sha256=identity.source_sha256,
        )
        canonical_run = True
    else:
        development_contract = None
        canonical_run = False
    run_dir = initialise_run_directory(
        ROOT / "outputs" / "runs" / "09_usgs_experiment",
        identity,
        run_config,
        provenance={
            "evidence_role": (
                "prelabel_route_a_model_build_development_only"
                if formal_configuration_complete
                else "development_exploratory_2019_2020"
            ),
            "training_device": resolved_device,
        },
        publication_guard=assert_formal_numerical_policy,
    )
    if formal_configuration_complete:
        if stage09_matrix_gate is None:  # pragma: no cover - construction above
            raise AssertionError("formal Stage-09 lacks its model-matrix gate")
        control_authorization_path, control_work_orders = freeze_control_plan(
            root=ROOT,
            run_directory=run_dir,
            identity=identity,
            resolved_config=run_config,
            matrix_gate=stage09_matrix_gate,
            interventions=ABLATION_INTERVENTIONS,
            publication_guard=assert_stage09_publication_inputs,
        )
    else:
        control_authorization_path = None
        control_work_orders = ()
    publication_paths = resolve_stage09_publication_paths(
        root=ROOT,
        run_dir=run_dir,
        requested_predictions=requested_predictions,
        requested_scores=requested_scores,
        requested_report=requested_report,
        formal_output_eligible=formal_configuration_complete,
    )
    output_predictions = publication_paths["predictions"]
    score_path = publication_paths["scores"]
    report_path = publication_paths["report"]
    lightgbm_selection_path = publication_paths["lightgbm_selection"]
    prediction_cache = run_dir / "predictions"
    training_checkpoints = run_dir / "checkpoints"
    member_cache = run_dir / "member_bundles"
    ablation_cache = run_dir / "ablation_bundles"
    lightgbm_shard_cache = (
        Path(args.shard_cache).expanduser().resolve()
        if args.shard_cache else run_dir / "lightgbm_shards"
    )
    log(f"content-addressed run {identity.run_id} | device={resolved_device}")
    if not formal_configuration_complete:
        log(f"diagnostic outputs isolated under {publication_paths['predictions'].parent}")

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
    atomic_write_bytes(
        lightgbm_selection_path,
        lightgbm_selection.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ).encode("utf-8"),
        publication_guard=assert_formal_numerical_policy,
    )
    log("LightGBM joint done")

    # ---- ThermoRoute joint, multiple seeds (resumable per seed) ---------- #
    tr_preds = []
    ensemble_members = {}
    seed_work = []
    for sd in C.USGS_SEEDS[:args.seeds]:
        member_name = f"seed{sd}"
        seed_file = prediction_cache / f"thermoroute_{member_name}.parquet"
        seed_bundle = member_cache / member_name
        cached_prediction = read_prediction_cache(seed_file, identity)
        cached_member = read_member_bundle(seed_bundle, identity, member_name)
        if cached_prediction is not None and cached_member is not None:
            assert_formal_numerical_policy()
            tr_preds.append(cached_prediction)
            ensemble_members[member_name] = cached_member
            log(f"  ThermoRoute {member_name}: verified content cache")
            continue
        seed_work.append(sd)
    if seed_work:
        seed_order_dir = run_dir / "stage09_seed_orders_v1"
        seed_order_dir.mkdir(parents=True, exist_ok=True)
        seed_orders: dict[int, Path] = {}
        for sd in seed_work:
            order_path = seed_order_dir / f"seed{sd}.json"
            atomic_write_json(
                order_path,
                {
                    "format": "thermoroute.stage09-seed-order.v1",
                    "seed": sd,
                    "identity": identity.as_dict(),
                    "run_config": run_config,
                    "panel": str(panel_path),
                    "device": resolved_device,
                    "delta_scale": args.delta_scale,
                    "station_sampling": args.station_sampling,
                    "eval_batch_size": args.eval_batch_size,
                    "prediction": str(
                        prediction_cache / f"thermoroute_seed{sd}.parquet"
                    ),
                    "bundle": str(member_cache / f"seed{sd}"),
                    "checkpoint": str(
                        training_checkpoints / f"seed{sd}.pt"
                    ),
                },
            )
            seed_orders[sd] = order_path
        launcher = _SeedWorkerLauncher()
        started = time.time()
        for future in _launch_seed_workers(
            seed_orders,
            max_workers=SEED_WORKER_PROCESSES,
            launch=launcher,
            terminate=launcher.terminate_all,
        ):
            future.result()
        log(f"  ThermoRoute seeds {sorted(seed_orders)}: {time.time()-started:.0f}s")
        for sd in seed_work:
            member_name = f"seed{sd}"
            seed_file = prediction_cache / f"thermoroute_{member_name}.parquet"
            seed_bundle = member_cache / member_name
            cached_prediction = read_prediction_cache(seed_file, identity)
            cached_member = read_member_bundle(seed_bundle, identity, member_name)
            if cached_prediction is None or cached_member is None:
                raise RuntimeError(
                    f"ThermoRoute seed{sd} worker completed without its "
                    "prediction and bundle artifacts"
                )
            tr_preds.append(cached_prediction)
            ensemble_members[member_name] = cached_member
    if {f"seed{sd}" for sd in C.USGS_SEEDS[:args.seeds]} != set(ensemble_members):
        raise RuntimeError(
            "ThermoRoute seed ensemble is incomplete after training"
        )
    chunks.append(pd.concat(tr_preds, ignore_index=True))

    # ---- leave-group-out ------------------------------------------------ #
    rng = np.random.default_rng(0)
    perm = rng.permutation(list(stations))
    hold = set(perm[: max(1, len(stations) // 4)])
    trainset = tuple(s for s in stations if s not in hold)
    lgo_file = prediction_cache / "thermoroute_lgo.parquet"
    cached_lgo = read_prediction_cache(lgo_file, identity)
    if cached_lgo is not None:
        assert_formal_numerical_policy()
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
                                         "seed": 0, "train_stations": trainset},
                        artifact_publication_guard=assert_formal_numerical_policy)
        res.pred["seed"] = 0
        lgo_held = res.pred[res.pred.site_id.isin(hold)]
        write_prediction_artifact(
            lgo_held, lgo_file, identity, kind="thermoroute_lgo_predictions",
            publication_guard=assert_formal_numerical_policy,
        )
        chunks.append(lgo_held)
        log(f"  LGO ({len(trainset)}→{len(hold)}): {time.time()-te:.0f}s")

    # ---- large-sample module ablations (same five seeds) ---------------- #
    ablation_members: dict[str, dict[str, object]] = {}
    ablation_predictions: dict[str, pd.DataFrame] = {}
    ablation_architecture: dict[str, dict[str, object]] = {}
    precomputed_controls: dict[
        tuple[str, int], tuple[pd.DataFrame, dict[str, torch.Tensor]]
    ] = {}
    if args.ablations and formal_configuration_complete:
        if control_authorization_path is None:
            raise AssertionError("formal Stage-09 control authority is absent")
        assert_stage09_publication_inputs()
        control_launcher = _Stage09WorkerLauncher()
        execute_work_orders_bounded(
            control_work_orders,
            max_workers=args.control_workers,
            launch=control_launcher,
            publication_guard=assert_stage09_publication_inputs,
            terminate=control_launcher.terminate_all,
        )
        freeze_control_matrix_receipt(
            root=ROOT,
            authorization_path=control_authorization_path,
            work_orders=control_work_orders,
            thermoroute_references={
                seed: {
                    "prediction": (
                        prediction_cache / f"thermoroute_seed{seed}.parquet"
                    ),
                    "bundle": member_cache / f"seed{seed}",
                }
                for seed in C.USGS_SEEDS
            },
            semantic_validator=lambda payload, validated:
                _validate_stage09_control_payload_semantics(
                    payload,
                    validated,
                    replay_wd=wd,
                ),
            publication_guard=assert_stage09_publication_inputs,
        )
        ordered_work_orders = member_work_order_map(control_work_orders)
        for name in MANDATORY_ABLATIONS:
            for seed in STAGE9_ABLATION_SEEDS:
                work_order = ordered_work_orders[(name, seed)]
                with extracted_member_payload(
                    root=ROOT,
                    work_order_path=work_order,
                ) as payload:
                    prediction = read_prediction_cache(
                        payload / "predictions.parquet", identity
                    )
                    weights = read_member_bundle(
                        payload / "bundle", identity, f"seed{seed}"
                    )
                if prediction is None or weights is None:
                    raise RuntimeError(
                        f"verified Stage-09 archive failed semantic load: "
                        f"{name} seed{seed}"
                    )
                precomputed_controls[(name, seed)] = (prediction, weights)
        if set(precomputed_controls) != {
            (name, seed)
            for name in MANDATORY_ABLATIONS
            for seed in STAGE9_ABLATION_SEEDS
        }:
            raise RuntimeError("Stage-09 precompute load is not exact 7x5")
    if args.ablations:
        # Each control changes one declared factor.  In particular noMoE keeps
        # both routed and TCN representations, and noRouter keeps the TCN path.
        for name in MANDATORY_ABLATIONS:
            model_kw = dict(ABLATION_INTERVENTIONS[name])
            model_kw.setdefault("delta_scale", args.delta_scale)
            ablation_architecture[name] = model_kw
            member_states: dict[str, object] = {}
            member_predictions: list[pd.DataFrame] = []
            for seed in STAGE9_ABLATION_SEEDS:
                member_name = f"seed{seed}"
                if precomputed_controls:
                    prediction, weights = precomputed_controls[(name, seed)]
                    member_predictions.append(prediction)
                    member_states[member_name] = weights
                    log(f"  {name} {member_name}: verified member archive")
                    continue
                prediction_path = (
                    prediction_cache / f"ablation_{name}_{member_name}.parquet"
                )
                member_bundle = ablation_cache / name / member_name
                cached_prediction = read_prediction_cache(prediction_path, identity)
                cached_weights = read_member_bundle(
                    member_bundle, identity, member_name
                )
                if cached_prediction is not None and cached_weights is not None:
                    assert_formal_numerical_policy()
                    member_predictions.append(cached_prediction)
                    member_states[member_name] = cached_weights
                    log(f"  {name} {member_name}: verified content cache")
                    continue
                started = time.time()
                def factory(model_kw=model_kw):
                    return ThermoRoute(
                        n_vars=len(wd.var_names), n_stations=len(stations),
                        n_phys=wd.n_phys, safety_anchor="damped", **model_kw,
                    )
                result = fit_model(
                    factory,
                    wd,
                    thr,
                    cfg=CFG,
                    seed=seed,
                    model_name=name,
                    device=resolved_device,
                    eval_batch_size=args.eval_batch_size,
                    scope="ablation_usgs",
                    feature_set="USGS",
                    station_balanced=args.station_sampling == "balanced",
                    selection_metric=(
                        "station_macro"
                        if args.station_sampling == "balanced" else "micro"
                    ),
                    checkpoint_path=(
                        training_checkpoints / f"ablation_{name}_{member_name}.pt"
                    ),
                    run_id=identity.run_id,
                    resolved_config={
                        **run_config,
                        "arm": name,
                        "seed": seed,
                        "model_kwargs": model_kw,
                    },
                    artifact_publication_guard=assert_formal_numerical_policy,
                )
                result.pred["seed"] = seed
                write_prediction_artifact(
                    result.pred,
                    prediction_path,
                    identity,
                    kind="thermoroute_ablation_seed_predictions",
                    publication_guard=assert_formal_numerical_policy,
                )
                seed_offsets, seed_offset_audit, seed_calibrators = (
                    calibration_artifacts(result.pred, thr)
                )
                save_inference_bundle(
                    member_bundle,
                    members={member_name: result.model},
                    metadata=bundle_metadata(
                        identity,
                        wd,
                        clim,
                        imputer,
                        thr,
                        event_reference,
                        args.delta_scale,
                        seed_offsets,
                        seed_offset_audit,
                        seed_calibrators,
                        training_device=resolved_device,
                        architecture_overrides=model_kw,
                    ),
                    expected_member_count=1,
                    publication_guard=assert_formal_numerical_policy,
                )
                member_predictions.append(result.pred)
                member_states[member_name] = {
                    key: value.detach().cpu().contiguous()
                    for key, value in result.model.state_dict().items()
                }
                log(
                    f"  {name} {member_name}: {time.time()-started:.0f}s "
                    f"val={result.best_val:.4f}"
                )
            combined = pd.concat(member_predictions, ignore_index=True)
            if set(member_states) != {
                f"seed{seed}" for seed in STAGE9_ABLATION_SEEDS
            }:
                raise RuntimeError(f"{name} lacks the complete ablation ensemble")
            ablation_predictions[name] = combined
            ablation_members[name] = member_states
            chunks.append(combined)

    # Re-check the live native pools after every long-running fit and before
    # any canonical artifact can be published.
    assert_stage09_publication_inputs()
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
    ablation_diagnostic = (
        multiseed_ablation_diagnostic_frames(allp) if args.ablations else {}
    )

    # The registry audit above first proves that every primary path refers to
    # the same observation at the precision actually consumed by the models.
    # Persist one canonical representation before any score/report is derived,
    # so the final artifact retains byte-exact y_true equality without a
    # tolerance-based verifier.  This is intentionally in-place: allp contains
    # millions of rows in the formal run.
    canonicalize_prediction_truth_inplace(allp)

    write_prediction_artifact(
        allp,
        output_predictions,
        identity,
        kind=(
            "canonical_stage9_usgs_predictions"
            if formal_configuration_complete
            else "diagnostic_stage9_usgs_predictions"
        ),
        publication_guard=assert_formal_numerical_policy,
        parents=(
            {"development_predictor_bridge_v1.json": predictor_bridge["sha256"]}
            if predictor_bridge is not None else None
        ),
    )
    log(f"saved predictions ({len(allp)} rows)")

    # Formal bundles are bound to the immutable Stage-9 prediction artifact.
    # Non-formal runs use only the diagnostic path selected above and can never
    # move a Route-A component pointer.

    parity_atol = 1e-5
    offsets, offset_audit, event_calibrators = calibration_artifacts(
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
            offsets, offset_audit, event_calibrators,
            training_device=resolved_device,
            development_prediction=development_prediction_binding(
                ROOT, output_predictions,
                allp[allp.model.eq("ThermoRoute")],
                max_abs_difference=parity_atol, atol=parity_atol,
            ),
        ),
        expected_member_count=args.seeds,
        publication_guard=assert_formal_numerical_policy,
    )
    loaded_members, loaded_metadata = load_inference_bundle(
        deployment_bundle,
        expected_member_count=args.seeds,
        publication_guard=assert_formal_numerical_policy,
    )
    if set(loaded_members) != set(ensemble_members) or loaded_metadata["run_id"] != identity.run_id:
        raise AssertionError("saved inference ensemble failed round-trip validation")
    tr_difference = verify_sequence_prediction_parity(
        deployment_bundle, wd=wd,
        expected=allp[allp.model.eq("ThermoRoute")],
        model_factory=lambda _member, metadata: thermoroute_factory_from_metadata(metadata),
        member_seeds={f"seed{seed}": seed for seed in C.USGS_SEEDS[:args.seeds]},
        atol=parity_atol, batch_size=args.eval_batch_size,
        publication_guard=assert_formal_numerical_policy,
    )
    assert_formal_numerical_policy()
    update_torch_development_prediction(
        deployment_bundle,
        development_prediction_binding(
            ROOT, output_predictions, allp[allp.model.eq("ThermoRoute")],
            max_abs_difference=tr_difference, atol=parity_atol,
        ),
    )

    # LightGBM is an equally-sized five-member ensemble.  Every point,
    # quantile and event head is a native-text Booster with its own checksum.
    lgb_offsets, lgb_offset_audit, lgb_calibrators = calibration_artifacts(
        allp[allp.model.eq("LightGBM")], thr
    )
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
                "deterministic": True, "force_col_wise": True, "n_jobs": MAIN_THREADS,
            },
            "validation_selection": lightgbm_selection.to_dict(orient="records"),
            "event_thresholds": {str(site): float(value)
                                 for site, value in sorted(thr.items())},
            "event_reference_climatology": dict(event_reference),
            "event_calibrators": {str(h): value.as_dict()
                                  for h, value in sorted(lgb_calibrators.items())},
            "conformal_offsets": _serialise_offsets(lgb_offsets),
            "conformal_policy": CF.cqr_policy_contract(),
            "conformal_offset_audit": lgb_offset_audit,
            "calibration_fit_contract": route_a_calibration_fit_contract(
                external=False
            ),
            "source_sha256": identity.source_sha256,
            "panel_sha256": identity.panel_sha256,
            "registry_sha256": identity.registry_sha256,
            "config_sha256": identity.config_sha256,
            "runtime_sha256": identity.runtime_sha256,
            "input_closure_sha256": identity.input_closure_sha256,
            "training_device": "cpu",
            "development_prediction": development_prediction_binding(
                ROOT, output_predictions, allp[allp.model.eq("LightGBM")],
                max_abs_difference=1e-12, atol=1e-12,
            ),
        },
        publication_guard=assert_formal_numerical_policy,
    )
    lgb_difference = verify_lightgbm_prediction_parity(
        lgb_manifest, evaluation_design=lightgbm_evaluation_design,
        expected=allp[allp.model.eq("LightGBM")],
        member_seeds={f"seed{seed}": seed for seed in C.USGS_SEEDS},
        atol=1e-12,
        publication_guard=assert_formal_numerical_policy,
    )
    assert_formal_numerical_policy()
    update_lightgbm_development_prediction(
        lgb_manifest,
        development_prediction_binding(
            ROOT, output_predictions, allp[allp.model.eq("LightGBM")],
            max_abs_difference=lgb_difference, atol=1e-12,
        ),
    )

    # Freeze every mandatory one-factor architecture control as a complete
    # five-seed ensemble.  Cached predictions alone are never accepted as a
    # model artifact.
    ablation_deployments = {}
    if args.ablations and set(ablation_members) == set(MANDATORY_ABLATIONS):
        for name in MANDATORY_ABLATIONS:
            pred = allp[allp.model.eq(name)]
            abl_offsets, abl_offset_audit, abl_calibrators = calibration_artifacts(
                ablation_predictions[name], thr
            )
            destination = C.MODELS / f"{name.lower()}_bundle_{identity.run_id}"
            save_inference_bundle(
                destination,
                members=ablation_members[name],
                metadata=bundle_metadata(
                    identity, wd, clim, imputer, thr, event_reference,
                    args.delta_scale,
                    abl_offsets, abl_offset_audit, abl_calibrators,
                    training_device=resolved_device,
                    architecture_overrides=ablation_architecture[name],
                    development_prediction=development_prediction_binding(
                        ROOT, output_predictions, pred,
                        max_abs_difference=parity_atol, atol=parity_atol,
                    ),
                ),
                expected_member_count=len(STAGE9_ABLATION_SEEDS),
                publication_guard=assert_formal_numerical_policy,
            )
            difference = verify_sequence_prediction_parity(
                destination, wd=wd, expected=pred,
                model_factory=lambda _member, metadata:
                    thermoroute_factory_from_metadata(metadata),
                member_seeds={
                    f"seed{seed}": seed for seed in STAGE9_ABLATION_SEEDS
                },
                atol=parity_atol,
                batch_size=args.eval_batch_size,
                publication_guard=assert_formal_numerical_policy,
            )
            assert_formal_numerical_policy()
            update_torch_development_prediction(
                destination,
                development_prediction_binding(
                    ROOT, output_predictions, pred,
                    max_abs_difference=difference, atol=parity_atol,
                ),
            )
            ablation_deployments[name] = destination

    formal_complete = (
        formal_configuration_complete
        and canonical_run and full_ensemble and args.ablations
        and formal_candidate and predictor_bridge is not None
        and canonical_output_paths
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
                directory=ablation_deployments[name],
                member_count=len(STAGE9_ABLATION_SEEDS),
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
    atomic_write_bytes(
        score_path,
        sc.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ).encode("utf-8"),
        publication_guard=assert_formal_numerical_policy,
    )

    L = [f"# USGS large-sample experiment ({len(stations)} stations, {args.seeds} seeds)\n",
         f"_Variables {', '.join(USGS_VARS)}. Observed targets only; identical samples "
         f"across the five primary headline models. ThermoRoute = {args.seeds}-seed "
         "mean. The same-station "
         "LightGBM is also a five-seed mean and receives stable site identity as a "
         "categorical feature; its small "
         "predeclared grid is selected by 2016–2017 station-macro RMSE only._\n",
         stage09_air2stream_report_status(bool(args.air2stream)) + "\n",
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
        if args.air2stream:
            ma4 = np.median([ra4[s] for s in stations if s in ra4])
            ma8 = np.median([ra8[s] for s in stations if s in ra8])
            air_cell = f"{ma4:.3f} / {ma8:.3f}"
        else:
            air_cell = "NOT_RUN / NA"
        mp, md, mt = d.rmse_persist.median(), d.rmse_damped.median(), d.rmse_thermo.median()
        sk_p = 1 - (d.rmse_thermo / d.rmse_persist).median()
        sk_d = 1 - (d.rmse_thermo / d.rmse_damped).median()
        # win-rate over stations that have development-evaluation data for both
        # models; stations with no test samples must not count as losses.
        dv = d.dropna(subset=["rmse_thermo", "rmse_damped"])
        win = float((dv.rmse_thermo < dv.rmse_damped).mean())
        L.append(f"| {h} | {mp:.3f} | {md:.3f} | {air_cell} | "
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
    L += ["", f"## Module ablations (five-seed deletion/intervention "
          f"sensitivity; ensemble-mean median per-station RMSE, "
          f"delta_scale={args.delta_scale})\n",
          "Audit: every mandatory control contains seeds 0--4 and uses the "
          "same five seeds as ThermoRoute, with identical forecast keys and "
          "exact y_true within each paired seed. Interpretation: this is a "
          "five-seed deletion/intervention sensitivity, not evidence of module "
          "necessity, causal mechanism, or capacity-matched attribution.\n",
          "| variant | h1 | h3 | h7 |", "|---|---|---|---|"]
    for m in abl_models:
        sub = ablation_diagnostic.get(m)
        if sub is None:
            continue
        meds = []
        for h in wd.horizons:
            r = rmse_per_station(sub, h)
            meds.append(np.median([r[s] for s in stations if s in r]))
        L.append(f"| {m} | {meds[0]:.3f} | {meds[1]:.3f} | {meds[2]:.3f} |")
    report_payload = ("\n".join(L) + "\n").encode("utf-8")

    def write_report() -> None:
        atomic_write_bytes(
            report_path,
            report_payload,
            publication_guard=assert_stage09_publication_inputs,
        )

    def validate_outputs() -> None:
        assert_stage09_publication_inputs()
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
            assert_stage09_publication_inputs()
            atomic_write_json(
                thermoroute_pointer,
                {
                    "run_id": identity.run_id,
                    "bundle_path": deployment_bundle.relative_to(ROOT).as_posix(),
                    "member_count": len(loaded_members),
                    "metadata_sha256": sha256_file(
                        deployment_bundle / "metadata.json"
                    ),
                    "weights_sha256": sha256_file(
                        deployment_bundle / "weights.pt"
                    ),
                },
                publication_guard=assert_stage09_publication_inputs,
            )
            atomic_write_json(
                lightgbm_pointer,
                {
                    "run_id": identity.run_id,
                    "manifest": file_binding(ROOT, lgb_manifest),
                    "member_count": len(C.USGS_SEEDS),
                },
                publication_guard=assert_stage09_publication_inputs,
            )
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
                publication_guard=assert_stage09_publication_inputs,
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
            assert_formal_numerical_policy()
            assert_stage09_publication_inputs()
            publish_stage09_completion_receipt(
                receipt_path,
                document,
                root=ROOT,
                stage9_pointer=components_pointer,
                publication_guard=assert_stage09_publication_inputs,
            )
            return receipt_path

        assert_stage09_publication_inputs()
        complete_stage09_transaction(
            write_report=write_report,
            validate_outputs=validate_outputs,
            publish_pointers=publish_pointers,
            publish_receipt=publish_receipt,
        )
        log("saved formal Stage-9 model components: TR5 + LGB5 + 7x5 controls")
        log(f"saved Stage-9 completion receipt: {receipt_path.relative_to(ROOT)}")
    else:
        write_report()
        log("saved diagnostic bundles; formal Stage-9 pointers/receipt unchanged")
    log("DONE")
    print("\n" + "\n".join(L[3:10]))


if __name__ == "__main__":
    main()
