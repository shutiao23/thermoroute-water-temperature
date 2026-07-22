"""Canonical predictions schema and the scoring aggregator.

Every model — baseline or deep — writes rows in the same long format so that a
single function turns *all* predictions into a tidy scores table.  Per-day
predictions are the primary artifact; summary metrics are derived, never the
other way round (a reproducibility requirement from the plan).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .repro import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_write_parquet,
    sha256_file,
    sidecar_path,
    source_tree_hash,
)

PREDICTION_SCHEMA_VERSION = "thermoroute.predictions.v1"

PRED_COLS = [
    "model", "scope", "feature_set", "seed",
    "site_id", "horizon", "split",
    "issue_date", "target_date", "y_true", "y_pred",
    "q05", "q50", "q95", "p_exceed",
]


def empty_predictions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PRED_COLS})


def make_pred_frame(**arrays) -> pd.DataFrame:
    """Build a predictions frame; missing probabilistic columns become NaN."""
    n = len(arrays["y_true"])
    data = {}
    for c in PRED_COLS:
        if c in arrays:
            data[c] = arrays[c]
        elif c in ("q05", "q50", "q95", "p_exceed"):
            data[c] = np.full(n, np.nan)
        else:
            data[c] = arrays.get(c)
    return pd.DataFrame(data)[PRED_COLS]


def validate_predictions(pred: pd.DataFrame, *,
                         expected_horizons: tuple[int, ...] = C.HORIZONS,
                         require_unique: bool = True) -> None:
    """Fail closed on malformed or mixed-generation prediction artifacts."""
    missing = set(PRED_COLS) - set(pred)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    required = ["model", "site_id", "horizon", "split", "issue_date", "target_date",
                "y_true", "y_pred"]
    if pred[required].isna().any().any():
        raise ValueError("required prediction fields contain null values")
    issue = pd.to_datetime(pred.issue_date, errors="coerce")
    target = pd.to_datetime(pred.target_date, errors="coerce")
    if issue.isna().any() or target.isna().any():
        raise ValueError("issue_date/target_date contains invalid timestamps")
    horizon = pd.to_numeric(pred.horizon, errors="coerce")
    if horizon.isna().any() or not horizon.astype(int).isin(expected_horizons).all():
        raise ValueError("prediction contains an undeclared horizon")
    actual_days = (target - issue).dt.days.to_numpy()
    if not np.array_equal(actual_days, horizon.astype(int).to_numpy()):
        raise ValueError("target_date must equal issue_date + horizon")
    if not np.isfinite(pred[["y_true", "y_pred"]].to_numpy(float)).all():
        raise ValueError("point outcomes/predictions must be finite")
    optional_quantiles = pred[["q05", "q50", "q95"]]
    any_quantile = optional_quantiles.notna().any(axis=1)
    complete_quantile = optional_quantiles.notna().all(axis=1)
    if (any_quantile & ~complete_quantile).any():
        raise ValueError("quantiles must be all present or all absent per row")
    q = optional_quantiles.loc[complete_quantile].to_numpy(float)
    if len(q) and (not np.isfinite(q).all() or not ((q[:, 0] <= q[:, 1]) &
                                                    (q[:, 1] <= q[:, 2])).all()):
        raise ValueError("quantiles must be finite and ordered q05 <= q50 <= q95")
    probabilities = pred.p_exceed.dropna().to_numpy(float)
    if len(probabilities) and (not np.isfinite(probabilities).all()
                               or not ((0 <= probabilities) & (probabilities <= 1)).all()):
        raise ValueError("p_exceed must lie in [0,1]")
    if require_unique:
        key = ["model", "scope", "feature_set", "seed", "site_id", "horizon", "split",
               "issue_date", "target_date"]
        normalised = pred[key].copy()
        for column in ("scope", "feature_set", "seed"):
            normalised[column] = normalised[column].fillna("__none__").astype(str)
        if normalised.duplicated(key).any():
            duplicate = normalised.loc[normalised.duplicated(key, keep=False), key].head(2)
            raise ValueError(f"duplicate prediction key:\n{duplicate.to_string(index=False)}")


def write_predictions(pred: pd.DataFrame, path, *, require_unique: bool = True) -> None:
    validate_predictions(pred, require_unique=require_unique)
    atomic_write_parquet(pred[PRED_COLS], path, index=False)


def load_route_a_predictions(
    path: str | Path,
    *,
    root: str | Path,
    panel_path: str | Path,
    registry_path: str | Path,
    expected_kind: str = "final_route_a_development_predictions",
    require_current_source: bool = True,
) -> pd.DataFrame:
    """Load a current Route-A prediction artifact or fail closed.

    A bare Parquet file is not scientific evidence: the repository previously
    retained an older ``n00``-identified generation under the same filename.
    This loader therefore binds bytes, schema, panel, station registry, code
    tree, primary-model registry and stable USGS identifiers before returning a
    frame to any table/figure/statistics script.
    """
    from .registry import ROUTE_A_PRIMARY_MODELS, enforce_common_forecast_keys

    root = Path(root).resolve()
    path = Path(path).resolve()
    panel_path = Path(panel_path).resolve()
    registry_path = Path(registry_path).resolve()
    metadata_path = sidecar_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Route-A predictions require both artifact and lineage sidecar: {path}"
        )
    try:
        lineage = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid prediction lineage JSON: {metadata_path}") from exc
    required_lineage = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": expected_kind,
        "content_schema": PREDICTION_SCHEMA_VERSION,
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
    }
    wrong = {
        key: (lineage.get(key), value)
        for key, value in required_lineage.items()
        if lineage.get(key) != value
    }
    if wrong:
        raise ValueError(f"Route-A prediction lineage mismatch: {wrong}")
    run = lineage.get("run")
    if not isinstance(run, dict):
        raise ValueError("Route-A prediction lineage lacks a run identity")
    if run.get("panel_sha256") != sha256_file(panel_path):
        raise ValueError("Route-A predictions were generated from another panel")
    if run.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("Route-A predictions were generated from another station registry")
    if require_current_source and run.get("source_sha256") != source_tree_hash(root):
        raise ValueError(
            "Route-A predictions are stale for the current source tree; rerun the pipeline"
        )
    if not isinstance(lineage.get("parents"), dict) or not lineage["parents"]:
        raise ValueError("final Route-A predictions lack immutable parent lineage")

    frame = pd.read_parquet(path)
    validate_predictions(frame)
    registry = pd.read_csv(registry_path, dtype={"site_no": "string"})
    if "site_no" not in registry or registry.site_no.isna().any():
        raise ValueError("station registry lacks stable site_no values")
    stable_sites = set(registry.site_no.astype(str).str.strip())
    actual_sites = set(frame.site_id.astype(str).str.strip())
    unknown = sorted(actual_sites - stable_sites)
    if unknown:
        raise ValueError(
            "prediction artifact contains legacy or unknown station identifiers: "
            f"{unknown[:5]}"
        )
    if not actual_sites or any(not site.isdigit() or not 8 <= len(site) <= 15
                               for site in actual_sites):
        raise ValueError("prediction artifact does not use stable 8--15 digit USGS site_no")
    test_models = set(frame.loc[frame.split.eq("test"), "model"].astype(str))
    missing_models = sorted(set(ROUTE_A_PRIMARY_MODELS) - test_models)
    if missing_models:
        raise ValueError(f"final Route-A predictions omit primary models: {missing_models}")
    _, audit = enforce_common_forecast_keys(
        frame, ROUTE_A_PRIMARY_MODELS, split="test"
    )
    primary_rows = frame[
        frame.split.eq("test") & frame.model.isin(ROUTE_A_PRIMARY_MODELS)
    ]
    key_counts = (
        primary_rows[
            ["model", "site_id", "horizon", "issue_date", "target_date"]
        ]
        .drop_duplicates()
        .groupby("model")
        .size()
    )
    if not key_counts.eq(audit.common_unique).all():
        raise ValueError("stored primary models are not already on the frozen common registry")
    return frame


def exceedance_thresholds(panel: pd.DataFrame, masks) -> dict[str, float]:
    """Train-period q90 WTEMP per station (statistical high-temp threshold)."""
    tr = panel.loc[masks.train]
    return {st: float(tr[tr.site_id == st][C.TARGET].quantile(C.EXCEEDANCE_QUANTILE))
            for st in C.STATIONS}


def _persistence_ref(pred: pd.DataFrame) -> dict[tuple, pd.Series]:
    """Map (site,horizon,split) -> persistence prediction aligned by issue_date."""
    ref = {}
    sub = pred[pred.model == "Persistence"]
    for (st, h, sp), g in sub.groupby(["site_id", "horizon", "split"]):
        ref[(st, h, sp)] = g.set_index("issue_date")["y_pred"]
    return ref


def evaluate(pred: pd.DataFrame, thresholds: dict[str, float],
             splits: tuple[str, ...] = ("test",)) -> pd.DataFrame:
    """Aggregate predictions into per-(model,scope,feature_set,site,horizon,split)
    scores, averaging across seeds (reporting mean and std)."""
    pred = pred[pred.split.isin(splits)].copy()
    ref = _persistence_ref(pred)
    rows = []
    keys = ["model", "scope", "feature_set", "site_id", "horizon", "split"]
    for key, g in pred.groupby(keys, dropna=False):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("issue_date")
            y = gs["y_true"].to_numpy(float)
            yhat = gs["y_pred"].to_numpy(float)
            st, h, sp = key[3], key[4], key[5]
            r = ref.get((st, h, sp))
            refv = r.reindex(gs["issue_date"]).to_numpy(float) if r is not None else None
            sc = M.point_scores(y, yhat, refv)
            if gs["q05"].notna().all() and len(gs) > 0:
                quants = {0.05: gs["q05"].to_numpy(float),
                          0.50: gs["q50"].to_numpy(float),
                          0.95: gs["q95"].to_numpy(float)}
                sc.update(M.probabilistic_scores(y, quants))
            if gs["p_exceed"].notna().all() and st in thresholds:
                ybin = (y > thresholds[st]).astype(float)
                sc.update({f"EVT_{k}": v
                           for k, v in M.event_scores(ybin, gs["p_exceed"].to_numpy(float)).items()})
            per_seed.append(sc)
        agg = {}
        all_keys = sorted({k for d in per_seed for k in d})
        for mk in all_keys:
            vals = np.array([d[mk] for d in per_seed if mk in d], dtype=float)
            agg[mk] = float(np.nanmean(vals))
            agg[mk + "_std"] = float(np.nanstd(vals))
        row = dict(zip(keys, key))
        row["n_seeds"] = g["seed"].nunique()
        row["n_obs"] = int(len(g) / g["seed"].nunique())
        row.update(agg)
        rows.append(row)
    return pd.DataFrame(rows)
