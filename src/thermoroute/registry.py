"""Prediction-key registry utilities used for fair model comparisons.

Forecast models often produce different subsets of a gappy panel.  Comparing
their aggregate scores without first intersecting the exact forecast keys is a
silent but material experimental error.  This module provides one strict code
path for all main and transfer experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


FORECAST_KEY: tuple[str, ...] = (
    "site_id", "horizon", "issue_date", "target_date",
)


def _float32_truth(values: pd.Series | np.ndarray) -> np.ndarray:
    """Canonicalise labels to the precision consumed by the neural models.

    ``WindowedData`` stores/trains on float32 targets, while the tabular design
    inherits parquet's float64 values.  Comparing those representations in
    float64 creates false label conflicts above roughly 32 degrees C.  Casting
    both sides first preserves the actual model target semantics without hiding
    a one-float32-ULP (or larger) disagreement.
    """
    return np.asarray(values, dtype=np.float32)


def targets_match_at_model_precision(
    left: pd.Series | np.ndarray,
    right: pd.Series | np.ndarray,
) -> bool:
    """Whether two finite target arrays are exactly equal after float32 casting."""
    left_truth = _float32_truth(left)
    right_truth = _float32_truth(right)
    return bool(
        left_truth.shape == right_truth.shape
        and np.isfinite(left_truth).all()
        and np.isfinite(right_truth).all()
        and np.array_equal(left_truth, right_truth)
    )

# These registries are protocol constants, not sets inferred from whatever
# optional models happened to finish.  Stage 9 freezes the five models that it
# can score before the LSTM stage; Stage 16 adds the sixth primary model and
# freezes the final Route-A comparison registry.  Air2stream, per-station LGB,
# warm-start transfer and architecture controls remain available as exploratory
# rows, but can never shrink the primary sample by being missing on a key.
STAGE9_PRIMARY_MODELS: tuple[str, ...] = (
    "Persistence",
    "DampedPersistence",
    "Climatology",
    "LightGBM",
    "ThermoRoute",
)
ROUTE_A_PRIMARY_MODELS: tuple[str, ...] = (
    *STAGE9_PRIMARY_MODELS,
    "LSTM",
)


@dataclass(frozen=True)
class RegistryAudit:
    """Audit information returned by :func:`enforce_common_forecast_keys`."""

    split: str
    models: tuple[str, ...]
    before_unique: dict[str, int]
    common_unique: int
    dropped_rows: int


def _normalise_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    missing = [c for c in FORECAST_KEY if c not in out.columns]
    if missing:
        raise ValueError(f"prediction frame is missing forecast-key columns: {missing}")
    out["issue_date"] = pd.to_datetime(out["issue_date"])
    out["target_date"] = pd.to_datetime(out["target_date"])
    return out


def enforce_common_forecast_keys(
    pred: pd.DataFrame,
    models: Sequence[str],
    *,
    split: str = "test",
    truth_atol: float = 0.0,
    require_all_models: bool = True,
) -> tuple[pd.DataFrame, RegistryAudit]:
    """Restrict comparable models to identical forecast keys and verify labels.

    Multiple seeds may emit duplicate forecast keys; the intersection is taken
    over *unique* keys per model and all seed rows for a retained key are kept.
    Rows belonging to other models or splits are returned untouched.

    Unlike the former ad-hoc set intersection in ``scripts/09``, the canonical
    key includes ``target_date`` and this function also verifies that every model
    stores the same target value for a retained key.
    """

    if not np.isfinite(truth_atol) or truth_atol < 0:
        raise ValueError("truth_atol must be finite and nonnegative")
    if "model" not in pred.columns or "split" not in pred.columns:
        raise ValueError("prediction frame must contain model and split columns")
    out = _normalise_key_columns(pred)
    requested = tuple(dict.fromkeys(models))
    available = set(out.loc[out["split"].eq(split), "model"].astype(str))
    absent = [m for m in requested if m not in available]
    if absent and require_all_models:
        raise ValueError(f"models absent from split={split!r}: {absent}")
    present = tuple(m for m in requested if m in available)
    if len(present) < 2:
        raise ValueError(f"need at least two comparable models, found {present}")

    comp_mask = out["split"].eq(split) & out["model"].isin(present)
    comp = out.loc[comp_mask]
    key_sets: dict[str, set[tuple]] = {}
    for model, group in comp.groupby("model", sort=False):
        key_sets[str(model)] = set(group.loc[:, FORECAST_KEY].itertuples(index=False, name=None))
    common = set.intersection(*(key_sets[m] for m in present))
    if not common:
        raise ValueError(f"no common forecast keys across models {present}")

    common_frame = pd.DataFrame.from_records(list(common), columns=FORECAST_KEY)
    common_frame["issue_date"] = pd.to_datetime(common_frame["issue_date"])
    common_frame["target_date"] = pd.to_datetime(common_frame["target_date"])
    common_frame["__shared_key"] = True
    marked = out.merge(common_frame, on=list(FORECAST_KEY), how="left", sort=False)
    marked_comp = marked["split"].eq(split) & marked["model"].isin(present)
    keep = ~marked_comp | marked["__shared_key"].eq(True)
    filtered = marked.loc[keep].drop(columns="__shared_key").reset_index(drop=True)

    # A shared timestamp is not enough: all models must also be scoring the same
    # observation.  The neural path consumes float32 labels whereas parquet and
    # the tabular path expose float64, so compare the canonical float32 values.
    retained = filtered[
        filtered["split"].eq(split) & filtered["model"].isin(present)
    ]
    if "y_true" not in retained.columns:
        raise ValueError("prediction frame must contain y_true for label-consistency audit")
    truth = _float32_truth(retained["y_true"])
    if np.any(~np.isfinite(truth)):
        raise ValueError("prediction frame contains a non-finite y_true")
    truth_audit = retained.loc[:, FORECAST_KEY].copy()
    truth_audit["__y_true_float32"] = truth
    spread = truth_audit.groupby(list(FORECAST_KEY), observed=True)[
        "__y_true_float32"
    ].agg(lambda x: float(np.max(x.to_numpy(np.float32)) - np.min(x.to_numpy(np.float32))))
    bad = spread[spread > truth_atol]
    if len(bad):
        raise ValueError(
            f"{len(bad)} shared forecast keys disagree on y_true by more than {truth_atol}"
        )

    # Postcondition: every comparable model has exactly the common registry.
    after_sets = {
        str(m): set(g.loc[:, FORECAST_KEY].itertuples(index=False, name=None))
        for m, g in retained.groupby("model", sort=False)
    }
    for model in present:
        if after_sets.get(model, set()) != common:
            raise AssertionError(f"common-key postcondition failed for {model}")

    audit = RegistryAudit(
        split=split,
        models=present,
        before_unique={m: len(key_sets[m]) for m in present},
        common_unique=len(common),
        dropped_rows=int((~keep).sum()),
    )
    return filtered, audit


def align_prediction_frames(
    frames: Iterable[pd.DataFrame],
    *,
    model_names: Sequence[str] | None = None,
    split: str = "test",
) -> tuple[pd.DataFrame, RegistryAudit]:
    """Concatenate frames and apply the strict common-key registry."""

    pred = pd.concat(list(frames), ignore_index=True)
    models = tuple(model_names) if model_names is not None else tuple(pred["model"].unique())
    return enforce_common_forecast_keys(pred, models, split=split)


def restrict_tabular_to_window_registry(
    tab: pd.DataFrame,
    wd,
    station_names: Sequence[str],
    horizon: int,
    *,
    truth_atol: float = 0.0,
) -> pd.DataFrame:
    """Use exactly the sequence model's train/val/calib/test forecast samples.

    Evaluation-only intersection is insufficient when baseline training sets
    differ because of lag warm-up or missing-target rules.  This helper inner
    joins a horizon-specific tabular design to ``WindowedData`` across every
    split and verifies that both pipelines attached the same target value.
    """
    if not np.isfinite(truth_atol) or truth_atol < 0:
        raise ValueError("truth_atol must be finite and nonnegative")
    required = {"site_id", "issue_date", "target_date", "split", "y"}
    missing = sorted(required - set(tab.columns))
    if missing:
        raise ValueError(f"tabular frame is missing registry columns: {missing}")
    try:
        hi = tuple(wd.horizons).index(horizon)
    except ValueError as exc:
        raise ValueError(f"horizon {horizon} absent from WindowedData") from exc
    sites = np.asarray([station_names[int(i)] for i in wd.station], dtype=object)
    registry = pd.DataFrame({
        "site_id": sites,
        "issue_date": pd.to_datetime(wd.issue_date),
        "target_date": pd.to_datetime(wd.target_date[:, hi]),
        "split": wd.split,
        "__window_y": wd.y[:, hi],
    })
    key = ["site_id", "issue_date", "target_date", "split"]
    if registry.duplicated(key).any():
        raise ValueError("WindowedData contains duplicate forecast keys")
    normal = tab.copy()
    normal["issue_date"] = pd.to_datetime(normal["issue_date"])
    normal["target_date"] = pd.to_datetime(normal["target_date"])
    aligned = normal.merge(registry, on=key, how="inner", validate="one_to_one")
    tabular_truth = _float32_truth(aligned["y"])
    window_truth = _float32_truth(aligned["__window_y"])
    disagreement = np.abs(tabular_truth - window_truth)
    if np.any(~np.isfinite(disagreement)) or np.any(disagreement > truth_atol):
        raise ValueError("tabular and window registries disagree on target labels")
    if len(aligned) != len(registry):
        raise ValueError(
            f"tabular design covers {len(aligned)}/{len(registry)} registered "
            f"samples at horizon={horizon}")
    return aligned.drop(columns="__window_y").sort_values(key).reset_index(drop=True)
