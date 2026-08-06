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

LIGHTGBM_BUNDLE_FORMAT = "thermoroute.lightgbm-bundle.v2"
_LIGHTGBM_HEADS = ("point", "q05", "q50", "q95", "event")


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
    ``wd.target_valid`` is true, with ``site_id`` decoded from ``C.STATIONS``.
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
        frames.append(R.make_pred_frame(
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
        ))
    return pd.concat(frames, ignore_index=True) if frames else R.empty_predictions()


def sequence_ensemble(
    bundle_dir: str | Path,
    wd,
    model_name: str,
    scope: str,
    feature_set: str,
    *,
    device: str | torch.device = "cpu",
    split: str = "confirm",
    batch_size: int = 4096,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    """Run every member of a frozen sequence bundle and average them.

    Returns ``(ensemble_frame, member_frames, metadata)``.  The ensemble frame
    averages point/quantile/exceedance across members; each member frame is also
    returned (seed-labelled) for reproduction diagnostics.
    """
    weights, metadata = checkpoint.load_inference_bundle(bundle_dir)
    model = FI.sequence_factory_from_metadata(metadata)
    members = list(metadata["members"])
    idx = wd.idx(split)
    station_names = list(C.STATIONS)
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
    ens_frame = _arrays_to_frame(
        ens_arrays, idx, wd, station_names, model_name, scope, feature_set, 0, split)
    member_frames = [
        _arrays_to_frame(arr, idx, wd, station_names, model_name, scope, feature_set, i, split)
        for i, arr in enumerate(member_arrays)
    ]
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
    return ens[R.PRED_COLS], member_frames, manifest


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
