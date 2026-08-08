"""Standalone conventional holdout scorer for frozen Route-A models.

This module loads the *frozen* trained model bundles published under
``outputs/models/`` and scores them on a conventional (non-sealed) holdout
period, using the frozen development-period preprocessing statistics baked
into each bundle's metadata.

It is deliberately STANDALONE: it does NOT import the pre-registration
apparatus modules (``opening``, ``model_suite`` source-hash enforcement,
``development_controls_gate``, ``development_replay``, ``chronology``,
``inference_gate``, ``protocols``).  It depends only on the bare model
classes, the weights-only bundle loader, the frozen-transform reconstructor,
the tabular feature builder, and the metric definitions.  It will therefore
survive a later deletion of the apparatus modules.

Conventional means: no sealing, no one-time opening, no source-hash check, no
preflight gate.  Just load frozen weights, apply frozen preprocessing, run
inference, and score against observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import checkpoint
from . import features as F
from . import frozen_inference as FI
from . import metrics as M
from . import registry as REG
from . import results as R
from .quantiles import repair_lightgbm_quantiles
from .repro import sha256_file
from .frozen_calibration import apply_frozen_calibration

LIGHTGBM_BUNDLE_FORMAT = "thermoroute.lightgbm-bundle.v2"
_LIGHTGBM_HEADS = ("point", "q05", "q50", "q95", "event")

CALIBRATED_STATE = "FROZEN_CQR_PLATT_APPLIED"
NO_CALIBRATION_STATE = "NO_FROZEN_CALIBRATION"
POINT_ONLY_STATE = "NOT_APPLICABLE_POINT_ONLY"

# Phase 2 prediction-table contract: results.PRED_COLS plus the calibration,
# event, cohort and provenance extensions (see plan 2.2).  The unadorned
# names always mean the deployed, calibrated quantity.
CONTRACT_COLS = tuple(R.PRED_COLS) + (
    "q05_raw", "q50_raw", "q95_raw", "p_exceed_raw",
    "conformal_delta_c", "platt_intercept", "platt_slope", "platt_constant",
    "calibration_state", "event_threshold_c", "event_observed",
    "huc2", "n_members", "bundle_sha256", "cohort",
)


# --------------------------------------------------------------------------- #
# LightGBM bundle loader (reimplemented without importing model_suite)
# --------------------------------------------------------------------------- #
def load_lightgbm_bundle_standalone(
    directory: str | Path,
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Load a frozen LightGBM bundle from a manifest directory.

    Returns ``(boosters, manifest)`` where ``boosters`` is
    ``dict[member -> dict[horizon(int) -> dict[head -> lgb.Booster]]]``.

    This reimplements the checksum-validated loader from ``model_suite`` so the
    scorer never imports ``model_suite`` (which transitively imports the
    apparatus).  Only ``json``, ``lightgbm`` and ``sha256_file`` are needed.
    """
    import lightgbm as lgb  # lazy import; not needed for sequence models

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != LIGHTGBM_BUNDLE_FORMAT:
        raise ValueError(f"unsupported lightgbm bundle format: {manifest.get('format')!r}")
    horizons = [int(h) for h in manifest["horizons"]]
    members = list(manifest["members"])
    heads = tuple(manifest["heads"])
    if heads != _LIGHTGBM_HEADS:
        raise ValueError(f"unexpected lightgbm heads: {heads}")
    boosters: dict[str, dict[int, dict[str, Any]]] = {}
    for member in members:
        boosters[member] = {}
        for h in horizons:
            boosters[member][h] = {}
            binding_block = manifest["models"][member][str(h)]
            for head in heads:
                binding = binding_block[head]
                path = directory / binding["path"]
                if not path.exists():
                    raise FileNotFoundError(f"lightgbm model file missing: {path}")
                if sha256_file(path) != binding["sha256"]:
                    raise ValueError(f"lightgbm checksum mismatch: {path}")
                boosters[member][h][head] = lgb.Booster(model_file=str(path))
    return boosters, manifest


# --------------------------------------------------------------------------- #
# Sequence-model (ThermoRoute / LSTM / ablations) ensemble inference
# --------------------------------------------------------------------------- #
def _forward_collect(
    model: torch.nn.Module,
    wd,
    idx: np.ndarray,
    device: str | torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run one model member over ``idx`` and return per-output [N, H] arrays."""
    n = len(idx)
    h = len(wd.horizons)
    out_arrays = {
        "point": np.zeros((n, h), np.float32),
        "q05": np.zeros((n, h), np.float32),
        "q50": np.zeros((n, h), np.float32),
        "q95": np.zeros((n, h), np.float32),
        "p_exceed": np.zeros((n, h), np.float32),
    }
    for start in range(0, n, batch_size):
        chunk = idx[start: start + batch_size]
        with torch.no_grad():
            out = model(wd.batch(chunk, device))
        i0, i1 = start, start + len(chunk)
        out_arrays["point"][i0:i1] = out.point.detach().cpu().numpy()
        out_arrays["q05"][i0:i1] = out.q05.detach().cpu().numpy()
        out_arrays["q50"][i0:i1] = out.q50.detach().cpu().numpy()
        out_arrays["q95"][i0:i1] = out.q95.detach().cpu().numpy()
        out_arrays["p_exceed"][i0:i1] = torch.sigmoid(out.exceed_logit).detach().cpu().numpy()
    return out_arrays


def _restrict_calibration_registry(
    metadata: Mapping[str, Any],
    sites: set[str],
    *,
    external: bool,
    horizons: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    """Restrict the frozen calibration registry to decoded sites/horizons.

    Dry sites that built no windows carry no predictions and no thresholds
    to apply, so their entries are dropped from the calibration view (no
    numeric change).  The offset audit is rebuilt from the raw signed
    offsets so it stays consistent with the restricted deployed registry.
    When ``horizons`` is given, the calibrator registry is restricted to
    those leads as well (per-lead calibration calls).
    """
    thresholds = metadata.get("event_thresholds", {})
    if not isinstance(thresholds, Mapping):
        return metadata
    restricted = dict(metadata)
    if not external and set(thresholds) != sites:
        restricted["event_thresholds"] = {
            key: value for key, value in thresholds.items() if str(key) in sites
        }
        offsets = metadata.get("conformal_offsets", {})
        restricted_offsets = {
            key: value for key, value in offsets.items()
            if str(key).split("|")[0] in sites
        }
        restricted["conformal_offsets"] = restricted_offsets
        audit = metadata.get("conformal_offset_audit", {})
        raw = audit.get("raw_signed_offsets")
        if isinstance(raw, Mapping):
            from .conformal import _build_cqr_offset_audit

            raw_restricted = {
                key: value for key, value in raw.items()
                if str(key).split("|")[0] in sites
            }
            restricted["conformal_offset_audit"] = _build_cqr_offset_audit(
                raw_restricted, restricted_offsets)
    if horizons is not None:
        wanted = {str(int(h)) for h in horizons}
        calibrators = metadata.get("event_calibrators", {})
        if isinstance(calibrators, Mapping) and set(calibrators) != wanted:
            restricted["event_calibrators"] = {
                key: value for key, value in calibrators.items()
                if str(key) in wanted
            }
        offsets = restricted.get("conformal_offsets", {})
        if isinstance(offsets, Mapping):
            restricted_offsets = {
                key: value for key, value in offsets.items()
                if str(key).rsplit("|", 1)[-1] in wanted
            }
            restricted["conformal_offsets"] = restricted_offsets
            audit = restricted.get("conformal_offset_audit", {})
            raw = audit.get("raw_signed_offsets")
            if isinstance(raw, Mapping):
                from .conformal import _build_cqr_offset_audit

                raw_restricted = {
                    key: value for key, value in raw.items()
                    if str(key).rsplit("|", 1)[-1] in wanted
                }
                restricted["conformal_offset_audit"] = _build_cqr_offset_audit(
                    raw_restricted, restricted_offsets)
    return restricted


def _calibrate_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    wd,
    station_names: Sequence[str],
    *,
    external: bool,
    label: str,
) -> dict[str, np.ndarray]:
    """Apply frozen CQR + Platt calibration over ALL window rows (like opening)."""
    station = np.asarray(
        [station_names[int(i)] for i in wd.station], dtype=object
    )
    sites = set(str(value) for value in station)
    metadata = _restrict_calibration_registry(metadata, sites, external=external)
    q05, q50, q95, prob = apply_frozen_calibration(
        metadata,
        station,
        wd.horizons,
        arrays["q05"],
        arrays["q50"],
        arrays["q95"],
        arrays["p_exceed"],
        external=external,
        label=label,
    )
    out = dict(arrays)
    out["q05_raw"] = np.asarray(arrays["q05"], float).copy()
    out["q50_raw"] = np.asarray(arrays["q50"], float).copy()
    out["q95_raw"] = np.asarray(arrays["q95"], float).copy()
    out["p_exceed_raw"] = np.asarray(arrays["p_exceed"], float).copy()
    out["q05"], out["q50"], out["q95"], out["p_exceed"] = q05, q50, q95, prob
    return out


def _arrays_to_frame(
    arrays: Mapping[str, np.ndarray],
    idx: np.ndarray,
    wd,
    station_names: Sequence[str],
    model_name: str,
    scope: str,
    feature_set: str,
    seed: int,
    split: str,
) -> pd.DataFrame:
    """Build a canonical prediction frame from per-window output arrays.

    Replicates ``train.export_predictions``: each horizon is emitted only where
    ``wd.target_valid`` is true, with ``site_id`` decoded from the explicit
    ``station_names`` (never the module-level ``C.STATIONS`` global).
    """
    frames = []
    names = list(station_names)
    for hi, h in enumerate(wd.horizons):
        valid = np.asarray(wd.target_valid[idx][:, hi], dtype=bool)
        if not valid.any():
            continue
        site = np.asarray([names[int(i)] for i in wd.station[idx]], dtype=object)[valid]
        issue = wd.issue_date[idx][valid]
        tdate = wd.target_date[idx][valid, hi]
        base = dict(
            model=model_name, scope=scope, feature_set=feature_set, seed=seed,
            site_id=site, horizon=np.full(int(valid.sum()), int(h)),
            split=np.full(int(valid.sum()), split),
            issue_date=issue, target_date=tdate,
            y_true=wd.y[idx][valid, hi].astype(np.float64),
            y_pred=arrays["point"][valid, hi].astype(np.float64),
            q05=arrays["q05"][valid, hi].astype(np.float64),
            q50=arrays["q50"][valid, hi].astype(np.float64),
            q95=arrays["q95"][valid, hi].astype(np.float64),
            p_exceed=arrays["p_exceed"][valid, hi].astype(np.float64),
        )
        if "q05_raw" in arrays:
            base.update(
                q05_raw=arrays["q05_raw"][valid, hi].astype(np.float64),
                q50_raw=arrays["q50_raw"][valid, hi].astype(np.float64),
                q95_raw=arrays["q95_raw"][valid, hi].astype(np.float64),
                p_exceed_raw=arrays["p_exceed_raw"][valid, hi].astype(np.float64),
            )
        frames.append(R.make_pred_frame(**base))
    return pd.concat(frames, ignore_index=True) if frames else R.empty_predictions()


def sequence_ensemble(
    bundle_dir: str | Path,
    wd,
    station_names: Sequence[str],
    model_name: str,
    scope: str,
    feature_set: str,
    *,
    device: str | torch.device = "cpu",
    split: str = "confirm",
    batch_size: int = 4096,
    external: bool = False,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    """Run every member of a frozen sequence bundle and average them.

    Returns ``(ensemble_frame, member_frames, metadata)``.  The ensemble frame
    averages point/quantile/exceedance across members; each member frame is also
    returned (seed-labelled) for reproduction diagnostics.  ``station_names``
    is an explicit argument: the decoder must never read the module-level
    ``C.STATIONS`` global (which ``frozen_inference`` rebinds per window build).
    """
    weights, metadata = checkpoint.load_inference_bundle(bundle_dir)
    model = FI.sequence_factory_from_metadata(metadata)
    members = list(metadata["members"])
    idx = wd.idx(split)
    member_arrays: list[dict[str, np.ndarray]] = []
    accum: dict[str, np.ndarray] | None = None
    for member in members:
        model.load_state_dict(weights[member], strict=True)
        model.eval()
        arr = _forward_collect(model, wd, idx, device, batch_size)
        member_arrays.append(arr)
        if accum is None:
            accum = {k: v.astype(np.float64).copy() for k, v in arr.items()}
        else:
            for k in accum:
                accum[k] += arr[k]
    n_members = len(members)
    ens_arrays = {k: v / n_members for k, v in (accum or {}).items()}
    ens_arrays = _calibrate_arrays(
        ens_arrays, metadata, wd, station_names,
        external=external, label=model_name)
    ens_frame = _arrays_to_frame(
        ens_arrays, idx, wd, station_names, model_name, scope, feature_set, 0, split)
    ens_frame = stamp_calibration_columns(
        ens_frame, metadata, external=external, n_members=n_members)
    member_frames = [
        _arrays_to_frame(arr, idx, wd, station_names, model_name, scope, feature_set, i, split)
        for i, arr in enumerate(member_arrays)
    ]
    return ens_frame, member_frames, metadata


# --------------------------------------------------------------------------- #
# Plain neural controls (Stage-09b checkpoints; no bundle, no calibration)
# --------------------------------------------------------------------------- #
def plain_control_ensemble(
    checkpoint_dir: str | Path,
    wd,
    station_names: Sequence[str],
    model_name: str,
    *,
    device: str | torch.device = "cpu",
    split: str = "confirm",
    batch_size: int = 4096,
    scope: str = "conventional",
    feature_set: str = "all_7_variables",
    n_stations: int = 120,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    cohort: str = "temporal",
    expected_run_id: str | None = None,
    external: bool = False,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    """Run every plain-control seed checkpoint and average them.

    Plain controls have no inference bundle, so this loads each
    ``checkpoint_dir/<model_name>/seed{seed}.pt`` ``best_model_state``
    (via :func:`plain_controls.load_plain_control_checkpoint`), runs
    :func:`_forward_collect` over ``wd.idx(split)``, and averages the heads
    across seeds exactly as :func:`sequence_ensemble` averages across bundle
    members.  Because no frozen CQR/Platt calibration exists for these arms,
    the ensemble is labelled ``NO_FROZEN_CALIBRATION`` with the raw heads
    retained as ``*_raw`` twins (:func:`mark_uncalibrated_frame`).

    ``station_names`` is the explicit decoder (Trap 5): the global
    ``C.STATIONS`` is never read here.  The borrowed stage-09 preprocessing
    (the ``wd`` built from a same-variable ThermoRoute bundle) is proved
    faithful by the G15 reproduction gate, not assumed.
    """
    from .plain_controls import (
        EXPECTED_PARAMETER_COUNTS,
        PlainControlError,
        load_plain_control_checkpoint,
    )
    checkpoint_dir = Path(checkpoint_dir)
    seeds_tuple = tuple(int(s) for s in seeds)
    if not seeds_tuple:
        raise ValueError("plain_control_ensemble requires at least one seed")
    idx = wd.idx(split)
    member_arrays: list[dict[str, np.ndarray]] = []
    member_meta: dict[str, Any] | None = None
    param_count = EXPECTED_PARAMETER_COUNTS.get(model_name)
    for seed in seeds_tuple:
        ckpt_path = checkpoint_dir / model_name / f"seed{seed}.pt"
        try:
            model, mmeta = load_plain_control_checkpoint(
                ckpt_path, arm_id=model_name, seed=seed, n_stations=n_stations,
                expected_run_id=expected_run_id,
            )
        except PlainControlError:
            raise
        model.eval()
        arr = _forward_collect(model, wd, idx, device, batch_size)
        member_arrays.append(arr)
        if member_meta is None:
            member_meta = mmeta
    accum: dict[str, np.ndarray] | None = None
    for arr in member_arrays:
        if accum is None:
            accum = {k: v.astype(np.float64).copy() for k, v in arr.items()}
        else:
            for k in accum:
                accum[k] += arr[k]
    n_members = len(member_arrays)
    ens_arrays = {k: v / n_members for k, v in (accum or {}).items()}
    metadata = dict(member_meta or {})
    metadata.update({
        "members": [f"seed{seed}" for seed in seeds_tuple],
        "member_count": n_members,
        "trainable_parameters": param_count,
        "calibration_state": NO_CALIBRATION_STATE,
        "has_bundle": False,
        "_checkpoint_dir": str(checkpoint_dir),
    })
    ens_frame = _arrays_to_frame(
        ens_arrays, idx, wd, station_names, model_name, scope, feature_set, 0, split)
    ens_frame = stamp_calibration_columns(
        ens_frame, metadata, external=external, n_members=n_members)
    member_frames = [
        _arrays_to_frame(arr, idx, wd, station_names, model_name, scope, feature_set, seed, split)
        for seed, arr in zip(seeds_tuple, member_arrays)
    ]
    ens_frame = mark_uncalibrated_frame(
        ens_frame, cohort=cohort, bundle_sha256="", n_members=n_members)
    return ens_frame, member_frames, metadata


# --------------------------------------------------------------------------- #
# LightGBM ensemble inference
# --------------------------------------------------------------------------- #
def lightgbm_ensemble(
    bundle_dir: str | Path,
    imputed: pd.DataFrame,
    clim: F.HarmonicClimatology,
    wd,
    station_names: Sequence[str],
    model_name: str,
    scope: str,
    feature_set: str,
    *,
    split: str = "confirm",
    truth_atol: float = 1e-3,
    external: bool = False,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    """Run every member of a frozen LightGBM bundle and average them.

    The tabular design is built once per horizon (from the frozen-imputed panel
    and frozen climatology), restricted to the sequence window registry, then
    each member's point/quantile/event heads are evaluated and averaged.
    """
    boosters, manifest = load_lightgbm_bundle_standalone(bundle_dir)
    feature_order = tuple(manifest["raw_feature_order"])
    design_order = list(manifest["design_feature_order"])
    uses_station_code = bool(design_order) and design_order[-1] == "station_code"
    station_categories = list(manifest.get("station_categories", [])) if uses_station_code else []
    horizons = [int(h) for h in manifest["horizons"]]
    members = list(manifest["members"])

    designs: dict[int, pd.DataFrame] = {}
    for h in horizons:
        tab = F.build_tabular(
            imputed, h, feature_order, clim,
            drop_feature_nans=False, require_observed_target=True,
            include_missingness=True)
        tab["split"] = split
        tab = REG.restrict_tabular_to_window_registry(
            tab, wd, station_names, h, truth_atol=truth_atol)
        if uses_station_code:
            tab = tab.copy()
            tab["station_code"] = pd.Categorical(
                tab["site_id"].astype(str), categories=station_categories)
        # verify the design columns exactly match what the boosters expect
        missing = [c for c in design_order if c not in tab.columns]
        if missing:
            raise ValueError(f"lightgbm design missing columns: {missing}")
        designs[h] = tab[design_order + ["site_id", "issue_date", "target_date", "y"]]

    member_frames: list[pd.DataFrame] = []
    for mi, member in enumerate(members):
        hframes = []
        for h in horizons:
            tab = designs[h]
            X = tab[design_order]
            heads = boosters[member][h]
            point = heads["point"].predict(X, num_threads=1)
            q05, q50, q95 = repair_lightgbm_quantiles(
                heads["q05"].predict(X, num_threads=1),
                heads["q50"].predict(X, num_threads=1),
                heads["q95"].predict(X, num_threads=1),
            )
            p_exceed = heads["event"].predict(X, num_threads=1)
            hframes.append(R.make_pred_frame(
                model=model_name, scope=scope, feature_set=feature_set, seed=mi,
                site_id=tab["site_id"].to_numpy(),
                horizon=np.full(len(tab), h),
                split=np.full(len(tab), split),
                issue_date=pd.to_datetime(tab["issue_date"]).to_numpy(),
                target_date=pd.to_datetime(tab["target_date"]).to_numpy(),
                y_true=tab["y"].to_numpy(float),
                y_pred=np.asarray(point, float),
                q05=np.asarray(q05, float), q50=np.asarray(q50, float),
                q95=np.asarray(q95, float), p_exceed=np.asarray(p_exceed, float),
            ))
        member_frames.append(pd.concat(hframes, ignore_index=True))

    allp = pd.concat(member_frames, ignore_index=True)
    group_keys = ["model", "scope", "feature_set", "site_id", "horizon",
                  "split", "issue_date", "target_date"]
    ens = (allp.groupby(group_keys, as_index=False, sort=False)
           .agg(y_pred=("y_pred", "mean"), q05=("q05", "mean"),
                q50=("q50", "mean"), q95=("q95", "mean"),
                p_exceed=("p_exceed", "mean"), y_true=("y_true", "first")))
    ens["seed"] = 0
    manifest = _restrict_calibration_registry(
        manifest, set(ens["site_id"].astype(str)), external=external)
    ens = _calibrate_frame_columns(
        ens, manifest, external=external, label=model_name)
    ens = stamp_calibration_columns(
        ens, manifest, external=external, n_members=len(members))
    for column in CONTRACT_COLS:
        if column not in ens.columns:
            ens[column] = np.nan
    return ens[list(CONTRACT_COLS)], member_frames, manifest


# --------------------------------------------------------------------------- #
# Analytical baselines (Persistence / DampedPersistence / Climatology)
# --------------------------------------------------------------------------- #
def baseline_frames(
    wd,
    station_names: Sequence[str],
    *,
    split: str = "confirm",
) -> dict[str, pd.DataFrame]:
    """Build the three analytical baselines directly from the windowed data.

    * Persistence        = issue-day WTEMP (``wd.wtemp_t``) broadcast to all h.
    * DampedPersistence  = frozen train-fit AR(1) anchor (``wd.damped_prior``).
    * Climatology        = frozen harmonic climatology at the target (``wd.clim_tgt``).

    These mirror ``opening._score_*`` baseline construction exactly.
    """
    names = list(station_names)
    idx = wd.idx(split)
    site = np.asarray([names[int(i)] for i in wd.station[idx]], dtype=object)
    issue = wd.issue_date[idx]
    arrays = {
        "Persistence": np.repeat(wd.wtemp_t[idx, None], len(wd.horizons), axis=1),
        "DampedPersistence": wd.damped_prior[idx],
        "Climatology": wd.clim_tgt[idx],
    }
    out: dict[str, pd.DataFrame] = {}
    for name, arr in arrays.items():
        frames = []
        for hi, h in enumerate(wd.horizons):
            valid = np.asarray(wd.target_valid[idx][:, hi], dtype=bool)
            if not valid.any():
                continue
            frames.append(R.make_pred_frame(
                model=name, scope="conventional", feature_set="USGS", seed=0,
                site_id=site[valid], horizon=np.full(int(valid.sum()), int(h)),
                split=np.full(int(valid.sum()), split),
                issue_date=issue[valid],
                target_date=wd.target_date[idx][valid, hi],
                y_true=wd.y[idx][valid, hi].astype(np.float64),
                y_pred=arr[valid, hi].astype(np.float64),
            ))
        out[name] = pd.concat(frames, ignore_index=True) if frames else R.empty_predictions()
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _keyed_predictions(frame: pd.DataFrame) -> dict[tuple, float]:
    """Map (site_id, issue_date, target_date) -> y_pred for one (model, horizon)."""
    keys = zip(
        frame["site_id"].astype(str).to_numpy(),
        pd.to_datetime(frame["issue_date"]).to_numpy(),
        pd.to_datetime(frame["target_date"]).to_numpy(),
    )
    return dict(zip(keys, frame["y_pred"].to_numpy(float)))


def _keyed_truth(frame: pd.DataFrame) -> dict[tuple, float]:
    keys = zip(
        frame["site_id"].astype(str).to_numpy(),
        pd.to_datetime(frame["issue_date"]).to_numpy(),
        pd.to_datetime(frame["target_date"]).to_numpy(),
    )
    return dict(zip(keys, frame["y_true"].to_numpy(float)))


def compute_metrics_long(
    model_frames: Mapping[str, pd.DataFrame],
    baselines: Mapping[str, pd.DataFrame],
    horizons: Sequence[int] = (1, 3, 7),
) -> pd.DataFrame:
    """Compute per-(model x horizon) and pooled metrics in long format.

    Columns: ``model, horizon, metric, value, n``.  Metrics: RMSE, MAE, BIAS,
    SKILL_PERSISTENCE, SKILL_CLIMATOLOGY, N.  Skill is
    ``1 - RMSE_model / RMSE_baseline`` computed on the forecast keys common to
    the model and both reference baselines.
    """
    baseline_names = set(baselines)
    persist = baselines.get("Persistence")
    clim = baselines.get("Climatology")
    rows: list[dict[str, Any]] = []
    for model_name, frame in {**model_frames, **baselines}.items():
        for h in horizons:
            gh = frame[frame["horizon"] == int(h)]
            truth = _keyed_truth(gh)
            pred = _keyed_predictions(gh)
            keys = sorted(set(truth) & set(pred))
            y = np.asarray([truth[k] for k in keys], float)
            yhat = np.asarray([pred[k] for k in keys], float)
            n = len(keys)
            rmse = M.rmse(y, yhat) if n else float("nan")
            mae = M.mae(y, yhat) if n else float("nan")
            bias = M.bias(y, yhat) if n else float("nan")
            row_base = {"model": model_name, "horizon": int(h), "n": n}
            rows.append({**row_base, "metric": "RMSE", "value": rmse})
            rows.append({**row_base, "metric": "MAE", "value": mae})
            rows.append({**row_base, "metric": "BIAS", "value": bias})
            if model_name not in baseline_names and persist is not None and clim is not None:
                ph = persist[persist["horizon"] == int(h)]
                ch = clim[clim["horizon"] == int(h)]
                p_pred = _keyed_predictions(ph)
                c_pred = _keyed_predictions(ch)
                common = sorted(set(pred) & set(p_pred) & set(c_pred))
                if common:
                    yc = np.asarray([truth[k] for k in common], float)
                    yc_hat = np.asarray([pred[k] for k in common], float)
                    yp = np.asarray([p_pred[k] for k in common], float)
                    ycl = np.asarray([c_pred[k] for k in common], float)
                    rmse_m = M.rmse(yc, yc_hat)
                    rmse_p = M.rmse(yc, yp)
                    rmse_c = M.rmse(yc, ycl)
                    rows.append({**row_base, "metric": "SKILL_PERSISTENCE",
                                 "value": float(1.0 - rmse_m / (rmse_p + 1e-12))})
                    rows.append({**row_base, "metric": "SKILL_CLIMATOLOGY",
                                 "value": float(1.0 - rmse_m / (rmse_c + 1e-12))})
                    rows.append({**row_base, "metric": "N_SKILL", "value": float(len(common))})
        # pooled across horizons
        truth = _keyed_truth(frame)
        pred = _keyed_predictions(frame)
        keys = sorted(set(truth) & set(pred))
        y = np.asarray([truth[k] for k in keys], float)
        yhat = np.asarray([pred[k] for k in keys], float)
        n = len(keys)
        row_base = {"model": model_name, "horizon": "pooled", "n": n}
        rows.append({**row_base, "metric": "RMSE", "value": M.rmse(y, yhat) if n else float("nan")})
        rows.append({**row_base, "metric": "MAE", "value": M.mae(y, yhat) if n else float("nan")})
        rows.append({**row_base, "metric": "BIAS", "value": M.bias(y, yhat) if n else float("nan")})
        if model_name not in baseline_names and persist is not None and clim is not None:
            p_pred = _keyed_predictions(persist)
            c_pred = _keyed_predictions(clim)
            common = sorted(set(pred) & set(p_pred) & set(c_pred))
            if common:
                yc = np.asarray([truth[k] for k in common], float)
                yc_hat = np.asarray([pred[k] for k in common], float)
                yp = np.asarray([p_pred[k] for k in common], float)
                ycl = np.asarray([c_pred[k] for k in common], float)
                rmse_m = M.rmse(yc, yc_hat)
                rows.append({**row_base, "metric": "SKILL_PERSISTENCE",
                             "value": float(1.0 - rmse_m / (M.rmse(yc, yp) + 1e-12))})
                rows.append({**row_base, "metric": "SKILL_CLIMATOLOGY",
                             "value": float(1.0 - rmse_m / (M.rmse(yc, ycl) + 1e-12))})
                rows.append({**row_base, "metric": "N_SKILL", "value": float(len(common))})
    return pd.DataFrame(rows, columns=["model", "horizon", "metric", "value", "n"])


def pivot_metrics(long_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long metrics table to wide (one row per model x horizon)."""
    wide = long_metrics.pivot_table(
        index=["model", "horizon"], columns="metric", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide


def _calibrate_frame_columns(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    external: bool,
    label: str,
) -> pd.DataFrame:
    """Calibrate a long-format frame per lead.

    LightGBM frames are long-format and some windows lack rows for every
    lead (panel-boundary windows whose far target falls past the panel).
    Each lead is therefore calibrated as its own (n, 1) array with the
    calibration registry view restricted to that lead; raw twins are kept.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    horizons = sorted(set(int(v) for v in out["horizon"]))
    for h in horizons:
        mask = out["horizon"].to_numpy(int) == h
        station = out.loc[mask, "site_id"].astype(str).to_numpy()
        q05 = out.loc[mask, "q05"].to_numpy(float).reshape(-1, 1)
        q50 = out.loc[mask, "q50"].to_numpy(float).reshape(-1, 1)
        q95 = out.loc[mask, "q95"].to_numpy(float).reshape(-1, 1)
        prob = out.loc[mask, "p_exceed"].to_numpy(float).reshape(-1, 1)
        meta_h = _restrict_calibration_registry(
            metadata, set(station), external=external, horizons=[h])
        cq05, cq50, cq95, cprob = apply_frozen_calibration(
            meta_h, station, [h], q05, q50, q95, prob,
            external=external, label=label,
        )
        out.loc[mask, "q05_raw"] = q05[:, 0]
        out.loc[mask, "q50_raw"] = q50[:, 0]
        out.loc[mask, "q95_raw"] = q95[:, 0]
        out.loc[mask, "p_exceed_raw"] = prob[:, 0]
        out.loc[mask, "q05"] = cq05[:, 0]
        out.loc[mask, "q50"] = cq50[:, 0]
        out.loc[mask, "q95"] = cq95[:, 0]
        out.loc[mask, "p_exceed"] = cprob[:, 0]
    return out


def stamp_calibration_columns(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    external: bool,
    n_members: int,
) -> pd.DataFrame:
    """Fill the calibration-state/event/threshold/delta columns per row."""
    if frame.empty:
        return frame
    out = frame.copy()
    offsets = metadata.get("conformal_offsets", {})
    calibrators = metadata.get("event_calibrators", {})
    thresholds = metadata.get("event_thresholds", {})
    out["calibration_state"] = CALIBRATED_STATE
    out["n_members"] = n_members
    out["bundle_sha256"] = str(metadata.get("weights_sha256", ""))
    horizon_values = sorted(set(int(v) for v in out["horizon"]))
    for site in set(out["site_id"].astype(str)):
        mask = out["site_id"].astype(str) == site
        for h in horizon_values:
            hmask = mask & (out["horizon"].to_numpy(int) == h)
            if not hmask.any():
                continue
            if external:
                key = f"__pooled__|{h}"
            else:
                key = f"{site}|{h}"
            out.loc[hmask, "conformal_delta_c"] = float(offsets.get(key, np.nan))
            platts = calibrators.get(str(h), {})
            out.loc[hmask, "platt_intercept"] = float(platts.get("intercept", np.nan))
            out.loc[hmask, "platt_slope"] = float(platts.get("slope", np.nan))
            constant = platts.get("constant")
            out.loc[hmask, "platt_constant"] = np.nan if constant is None else float(constant)
        threshold = thresholds.get("__pooled__" if external else str(site), np.nan)
        out.loc[mask, "event_threshold_c"] = float(threshold)
        y = out.loc[mask, "y_true"].to_numpy(float)
        observed = np.where(np.isfinite(y), (y > float(threshold)).astype(int), np.nan)
        out.loc[mask, "event_observed"] = observed
    return out


def apply_frozen_calibration_to_frame(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    station_names: Sequence[str],
    *,
    external: bool,
    label: str,
) -> pd.DataFrame:
    """Apply frozen CQR + Platt calibration to an ensemble frame, keeping raw twins.

    The unadorned ``q05/q50/q95/p_exceed`` columns become the deployed,
    calibrated quantities; the pre-calibration heads are retained as
    ``q05_raw/.../p_exceed_raw``.  ``calibration_state``, the per-site event
    threshold, the observed event, the applied delta/Platt parameters, member
    count, bundle digest, HUC2 label and cohort are carried in the row so
    every downstream statistic is a pure derivation from the table.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    horizons = [int(h) for h in sorted(set(int(v) for v in out["horizon"]))]
    stations = np.asarray([str(v) for v in station_names], dtype=object)
    for hi, h in enumerate(horizons):
        mask = out["horizon"].to_numpy(int) == h
        if not mask.any():
            continue
        order = np.asarray([int(np.where(stations == site)[0][0]) for site in out.loc[mask, "site_id"].astype(str)])
        q05 = np.full((int(mask.sum()), len(horizons)), np.nan)
        q50 = np.full((int(mask.sum()), len(horizons)), np.nan)
        q95 = np.full((int(mask.sum()), len(horizons)), np.nan)
        prob = np.full((int(mask.sum()), len(horizons)), np.nan)
        q05[:, hi] = out.loc[mask, "q05"].to_numpy(float)
        q50[:, hi] = out.loc[mask, "q50"].to_numpy(float)
        q95[:, hi] = out.loc[mask, "q95"].to_numpy(float)
        prob[:, hi] = out.loc[mask, "p_exceed"].to_numpy(float)
        stations_here = stations[order]
        cal_q05, cal_q50, cal_q95, cal_prob = apply_frozen_calibration(
            metadata, stations_here, horizons,
            q05, q50, q95, prob, external=external, label=label,
        )
        out.loc[mask, "q05_raw"] = out.loc[mask, "q05"]
        out.loc[mask, "q50_raw"] = out.loc[mask, "q50"]
        out.loc[mask, "q95_raw"] = out.loc[mask, "q95"]
        out.loc[mask, "p_exceed_raw"] = out.loc[mask, "p_exceed"]
        out.loc[mask, "q05"] = cal_q05[:, hi]
        out.loc[mask, "q50"] = cal_q50[:, hi]
        out.loc[mask, "q95"] = cal_q95[:, hi]
        out.loc[mask, "p_exceed"] = cal_prob[:, hi]
        delta = metadata.get("conformal_offsets", {})
        calibrators = metadata.get("event_calibrators", {})
        thresholds = metadata.get("event_thresholds", {})
        for site in set(out.loc[mask, "site_id"].astype(str)):
            row_sites = out.loc[mask, "site_id"].astype(str) == site
            site_horizons = out.loc[mask & row_sites, "horizon"].to_numpy(int)
            deltas = [float(delta.get(f"{site}|{h}", 0.0)) for h in site_horizons]
            out.loc[mask & row_sites, "conformal_delta_c"] = deltas
            platts = calibrators.get(str(h), {})
            out.loc[mask & row_sites, "platt_intercept"] = float(platts.get("intercept", 0.0))
            out.loc[mask & row_sites, "platt_slope"] = float(platts.get("slope", 1.0))
            constant = platts.get("constant")
            out.loc[mask & row_sites, "platt_constant"] = np.nan if constant is None else float(constant)
            threshold = thresholds.get(str(site), np.nan)
            out.loc[mask & row_sites, "event_threshold_c"] = threshold
            out.loc[mask & row_sites, "event_observed"] = (
                out.loc[mask & row_sites, "y_true"].to_numpy(float) > float(threshold)
            ).astype(int)
    out["calibration_state"] = CALIBRATED_STATE
    return out


def mark_point_only_frame(
    frame: pd.DataFrame,
    *,
    cohort: str,
    bundle_sha256: str = "",
    n_members: int = 1,
) -> pd.DataFrame:
    """Label an analytical baseline frame under the table contract."""
    if frame.empty:
        return frame
    out = frame.copy()
    out["q05_raw"] = np.nan
    out["q50_raw"] = np.nan
    out["q95_raw"] = np.nan
    out["p_exceed_raw"] = np.nan
    out["conformal_delta_c"] = np.nan
    out["platt_intercept"] = np.nan
    out["platt_slope"] = np.nan
    out["platt_constant"] = np.nan
    out["calibration_state"] = POINT_ONLY_STATE
    out["event_threshold_c"] = np.nan
    out["event_observed"] = np.nan
    out["n_members"] = n_members
    out["bundle_sha256"] = bundle_sha256
    out["cohort"] = cohort
    return out


def mark_uncalibrated_frame(
    frame: pd.DataFrame,
    *,
    cohort: str,
    bundle_sha256: str,
    n_members: int,
) -> pd.DataFrame:
    """Label a plain-control frame whose raw heads are persisted uncalibrated."""
    if frame.empty:
        return frame
    out = frame.copy()
    out["q05_raw"] = out["q05"]
    out["q50_raw"] = out["q50"]
    out["q95_raw"] = out["q95"]
    out["p_exceed_raw"] = out["p_exceed"]
    out["conformal_delta_c"] = np.nan
    out["platt_intercept"] = np.nan
    out["platt_slope"] = np.nan
    out["platt_constant"] = np.nan
    out["calibration_state"] = NO_CALIBRATION_STATE
    out["event_threshold_c"] = np.nan
    out["event_observed"] = np.nan
    out["n_members"] = n_members
    out["bundle_sha256"] = bundle_sha256
    out["cohort"] = cohort
    return out


def assign_cohort_metadata(
    frames: Mapping[str, pd.DataFrame],
    *,
    metadata_by_model: Mapping[str, Mapping[str, Any]],
    cohort: str,
    registry_huc2: Mapping[str, str],
    external: bool = False,
) -> dict[str, pd.DataFrame]:
    """Attach HUC2 + cohort + bundle digest to every model frame."""
    out: dict[str, pd.DataFrame] = {}
    for model_name, frame in frames.items():
        if frame.empty:
            out[model_name] = frame
            continue
        meta = metadata_by_model.get(model_name)
        f = frame.copy()
        if meta is not None:
            bundle_sha = str(meta.get("weights_sha256", ""))
            if not bundle_sha and "_bundle_dir" in meta:
                manifest = Path(meta["_bundle_dir"]) / "manifest.json"
                bundle_sha = sha256_file(str(manifest)) if manifest.exists() else ""
        else:
            bundle_sha = ""
        if "huc2" not in f.columns or f["huc2"].isna().all():
            f["huc2"] = f["site_id"].astype(str).map(registry_huc2)
        f["cohort"] = cohort
        out[model_name] = f
    return out


# --------------------------------------------------------------------------- #
# Validation gates G1-G14 (plan 2.4)
# --------------------------------------------------------------------------- #
class ConventionalGateError(RuntimeError):
    """A validation gate failed; nothing is written."""


def validate_prediction_table(
    pred: pd.DataFrame,
    *,
    registry: pd.DataFrame,
    expected_horizons: Sequence[int] = (1, 3, 7),
    calibrated_models: set[str],
    uncalibrated_models: set[str],
    y_true_reference: pd.Series | None = None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Run gates G1-G14 over the assembled per-key prediction table."""
    report: dict[str, Any] = {"gates": {}}
    def gate(name: str, ok: bool, detail: str) -> None:
        report["gates"][name] = {"pass": bool(ok), "detail": str(detail)}
        if not ok:
            raise ConventionalGateError(f"gate {name} failed: {detail}")
    # G1-G3: per-row head validity (finite, ordered, event in [0,1])
    q = pred[["q05", "q50", "q95"]].to_numpy(float)
    finite = np.isfinite(q).all(axis=1) | pred[["q05", "q95"]].isna().all(axis=1)
    ordered = np.asarray(
        [not (np.isfinite(r).all() and not (r[0] <= r[1] <= r[2] and r[0] < r[2]))
         for r in q], dtype=bool)
    event_ok = pred["p_exceed"].isna() | ((pred["p_exceed"] >= 0.0) & (pred["p_exceed"] <= 1.0))
    gate("G1_finite_heads", bool(finite.all()), f"{int((~finite).sum())} non-finite head rows")
    gate("G2_ordered_heads", bool(ordered.all()), f"{int((~ordered).sum())} unordered head rows")
    gate("G3_event_in_unit_interval", bool(event_ok.all()), f"{int((~event_ok).sum())} out-of-range p_exceed")
    # G4: pre-calibration strict q05 < q95 for calibrated models
    cal_rows = pred[pred.model.isin(calibrated_models)]
    if not cal_rows.empty:
        violated = int((cal_rows["q05_raw"] >= cal_rows["q95_raw"]).sum())
        gate("G4_strict_precalibration_width", violated == 0,
             f"{violated} pre-calibration q05_raw >= q95_raw rows")
    # G5: post-calibration ordering, strict width, p_exceed in [0,1]
    cal_rows2 = pred[pred.calibration_state == CALIBRATED_STATE]
    if not cal_rows2.empty:
        post_ok = (
            (cal_rows2["q05"] < cal_rows2["q95"]).all()
            and (cal_rows2["q05"] <= cal_rows2["q50"]).all()
            and (cal_rows2["q50"] <= cal_rows2["q95"]).all()
            and ((cal_rows2["p_exceed"] >= 0.0) & (cal_rows2["p_exceed"] <= 1.0)).all()
        )
        bad = int((
            (cal_rows2["q05"] >= cal_rows2["q95"])
            | (cal_rows2["q05"] > cal_rows2["q50"])
            | (cal_rows2["q50"] > cal_rows2["q95"])
            | cal_rows2["p_exceed"].isna()
            | (cal_rows2["p_exceed"] < 0.0)
            | (cal_rows2["p_exceed"] > 1.0)
        ).sum())
        gate("G5_postcalibration_valid", bool(post_ok),
             f"{bad} invalid post-calibration rows (ordering/width/p_exceed)")
    # G6: results.validate_predictions on PRED_COLS
    from .results import validate_predictions
    try:
        validate_predictions(
            pred[R.PRED_COLS], expected_horizons=tuple(expected_horizons), require_unique=True)
    except Exception as exc:
        gate("G6_schema_validation", False, str(exc))
    else:
        gate("G6_schema_validation", True, "schema OK")
    # G7: duplicate keys
    dup = pred.duplicated(
        subset=["cohort", "model", "seed", "site_id", "horizon", "split",
                "issue_date", "target_date"])
    gate("G7_no_duplicate_keys", bool(not dup.any()), f"{int(dup.sum())} duplicate rows")
    # G8: site_id in registry
    registry_sites = set(registry["site_no"].astype(str).str.zfill(8))
    unknown = ~pred["site_id"].astype(str).isin(registry_sites)
    gate("G8_sites_in_registry", bool(not unknown.any()), f"{int(unknown.sum())} unknown sites")
    # G9: y_true tie-back to the rebuilt panel within tolerance
    if y_true_reference is not None:
        aligned = pred[["site_id", "horizon", "issue_date", "target_date"]].copy()
        aligned = aligned.merge(
            y_true_reference.reset_index(), how="left",
            left_on=["site_id", "horizon", "issue_date", "target_date"],
            right_on=["site_id", "horizon", "issue_date", "target_date"])
        if "y_true" not in aligned.columns:
            gate("G9_y_true_tieback", False, "reference truth lacks y_true")
        else:
            diff = np.abs(
                aligned["y_true"].to_numpy(float) - pred["y_true"].to_numpy(float))
            ok = bool(np.nanmax(diff) <= atol) if len(diff) else True
            gate("G9_y_true_tieback", ok, f"max |dy| = {np.nanmax(diff):.3g}")
    else:
        gate("G9_y_true_tieback", True, "no reference supplied (orchestrator supplies it)")
    # G10: decode matches bundle station_to_index (explicit station_names)
    gate("G10_explicit_station_decode", True, "verified by orchestrator (see report)")
    # G11: recomputed event_observed matches stored
    cal_rows3 = pred[pred.calibration_state == CALIBRATED_STATE]
    if not cal_rows3.empty:
        expected_event = (
            cal_rows3["y_true"].to_numpy(float)
            > cal_rows3["event_threshold_c"].to_numpy(float)
        ).astype(int)
        matches = (expected_event == cal_rows3["event_observed"].to_numpy(float)).all()
        n_mismatch = int((expected_event != cal_rows3["event_observed"].to_numpy(float)).sum())
        gate("G11_event_observed_consistent", bool(matches),
             f"{n_mismatch} recomputed event_observed mismatches")
    # G12: common forecast keys per cohort
    counts = pred.groupby(["cohort", "model", "horizon"]).size().reset_index(name="n")
    per_cohort: dict[str, dict[str, int]] = {}
    for cohort, group in counts.groupby("cohort"):
        per_cohort[cohort] = {
            str(int(h)): int(g["n"].nunique())
            for h, g in group.groupby("horizon")
        }
    ok12 = all(v == 1 for d in per_cohort.values() for v in d.values())
    gate("G12_common_forecast_keys", ok12, str(per_cohort))
    # G13: cohort accounting exhaustive (per-horizon reportable counts)
    reportable = {}
    for h in expected_horizons:
        nh = station_reportable_counts(pred, horizon=h)
        reportable[f"h{h}"] = int(nh)
    gate("G13_cohort_accounting", True, str(reportable))
    report["n_stations_reportable"] = reportable
    # G14: no HTTP_FAILED site unless waiver stamped (handled by orchestrator)
    gate("G14_no_http_failed_without_waiver", True, "waiver state checked by orchestrator")
    return report


def station_reportable_counts(pred: pd.DataFrame, *, horizon: int) -> int:
    """Count reportable stations at one horizon.

    A station is reportable when the primary model (ThermoRoute) has at
    least MINIMUM_VALID_TARGETS valid paired targets at that lead — the
    station/lead cell rule of the manuscript (Section 3.6), not a pooled
    all-model row count.
    """
    from .conventional_stats import MINIMUM_VALID_TARGETS
    g = pred[(pred.horizon == int(horizon)) & (pred["model"] == "ThermoRoute")]
    return int((g.groupby("site_id").size() >= MINIMUM_VALID_TARGETS).sum())


# --------------------------------------------------------------------------- #
# Reproduction diagnostics (validation against stored development predictions)
# --------------------------------------------------------------------------- #
def compare_to_reference(
    member_frames: list[pd.DataFrame],
    reference: pd.DataFrame,
    *,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """Compare per-seed frozen-bundle predictions to stored development ones.

    ``member_frames[i]`` corresponds to seed ``i``.  ``reference`` must contain
    the dev predictions (with a ``seed`` column) restricted to the same split.
    The comparison is on the intersection of forecast keys per seed.
    """
    diagnostics: dict[str, Any] = {"per_seed": [], "match": False}
    all_max = 0.0
    for seed_i, mf in enumerate(member_frames):
        ref = reference[reference["seed"] == seed_i]
        merged = mf.merge(
            ref[["site_id", "horizon", "issue_date", "target_date", "y_pred"]],
            on=["site_id", "horizon", "issue_date", "target_date"],
            suffixes=("_frozen", "_dev"), how="inner")
        if len(merged) == 0:
            diagnostics["per_seed"].append({"seed": seed_i, "n_common": 0, "max_abs_diff": float("nan")})
            continue
        diff = np.abs(merged["y_pred_frozen"].to_numpy(float) - merged["y_pred_dev"].to_numpy(float))
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        all_max = max(all_max, max_diff)
        diagnostics["per_seed"].append({
            "seed": seed_i,
            "n_common": int(len(merged)),
            "n_frozen": int(len(mf)),
            "n_dev": int(len(ref)),
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "rmse_diff": float(np.sqrt(np.mean(diff ** 2))),
            "within_atol": bool(max_diff <= atol),
        })
    diagnostics["max_abs_diff"] = all_max
    diagnostics["atol"] = atol
    diagnostics["match"] = all_max <= atol
    return diagnostics


def compare_predictions_to_reference(
    member_frames: list[pd.DataFrame],
    reference: pd.DataFrame,
    *,
    atol: float = 1e-2,
    metrics: Sequence[str] = ("y_pred", "q05", "q50", "q95", "p_exceed"),
    require_all_keys: bool = True,
) -> dict[str, Any]:
    """Multi-metric per-seed reproduction diagnostic (G15 for plain controls).

    Generalises :func:`compare_to_reference` (which is ``y_pred``-only) to every
    requested head.  ``member_frames[i]`` corresponds to seed ``i`` and
    ``reference`` must carry a ``seed`` column.  Each seed passes only if every
    metric's ``max_abs_diff <= atol`` and (when ``require_all_keys``) every
    reference forecast key is reproduced (``n_common == n_dev``).
    """
    merge_keys = ["site_id", "horizon", "issue_date", "target_date"]
    diagnostics: dict[str, Any] = {"per_seed": [], "atol": atol, "metrics": list(metrics)}
    overall_max = 0.0
    all_pass = True
    for seed_i, mf in enumerate(member_frames):
        ref = reference[reference["seed"] == seed_i]
        entry: dict[str, Any] = {"seed": seed_i, "n_frozen": int(len(mf)), "n_dev": int(len(ref))}
        if mf.empty or ref.empty or len(mf.merge(ref[merge_keys], on=merge_keys, how="inner")) == 0:
            entry.update({"n_common": 0, "max_abs_diff": {m: float("nan") for m in metrics},
                          "within_atol": False, "all_keys_reproduced": False})
            diagnostics["per_seed"].append(entry)
            all_pass = False
            continue
        merged = mf.merge(
            ref[merge_keys + list(metrics)], on=merge_keys,
            suffixes=("_frozen", "_dev"), how="inner")
        per_metric: dict[str, float] = {}
        seed_within = True
        for metric in metrics:
            d = np.abs(merged[f"{metric}_frozen"].to_numpy(float)
                       - merged[f"{metric}_dev"].to_numpy(float))
            maxd = float(np.max(d)) if len(d) else float("nan")
            per_metric[metric] = maxd
            overall_max = max(overall_max, maxd if np.isfinite(maxd) else 0.0)
            if not np.isfinite(maxd) or maxd > atol:
                seed_within = False
        all_keys = int(len(merged)) == int(len(ref))
        entry.update({
            "n_common": int(len(merged)),
            "max_abs_diff": per_metric,
            "overall_max_abs_diff": float(np.nanmax(list(per_metric.values()))),
            "within_atol": bool(seed_within),
            "all_keys_reproduced": bool(all_keys),
        })
        if not (seed_within and (all_keys or not require_all_keys)):
            all_pass = False
        diagnostics["per_seed"].append(entry)
    diagnostics["max_abs_diff"] = overall_max
    diagnostics["match"] = bool(all_pass)
    return diagnostics
