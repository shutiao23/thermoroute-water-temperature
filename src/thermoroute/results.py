"""Canonical predictions schema and the scoring aggregator.

Every model — baseline or deep — writes rows in the same long format so that a
single function turns *all* predictions into a tidy scores table.  Per-day
predictions are the primary artifact; summary metrics are derived, never the
other way round (a reproducibility requirement from the plan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M

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


def exceedance_thresholds(panel: pd.DataFrame, masks) -> dict[str, float]:
    """Train-period q90 WTEMP per station (statistical high-temp threshold)."""
    tr = panel.loc[masks.train]
    return {st: float(tr[tr.site_id == st][C.TARGET].quantile(C.EXCEEDANCE_QUANTILE))
            for st in C.STATIONS}


def _persistence_ref(pred: pd.DataFrame) -> dict[tuple, np.ndarray]:
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
