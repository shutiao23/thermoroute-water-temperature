"""Pure-statistics derivation layer for the conventional 2021-2023 holdout.

Consumes the persisted per-key prediction table (the Phase 2 contract in
``results.PRED_COLS`` plus the calibration/event/cohort extensions) and the
station registry, and produces every number the manuscript reports.  No
torch, no bundle loader, no network: importing this module must never pull
in a model framework (enforced by an AST test), so re-deriving every
manuscript statistic costs seconds and touches no model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .significance import (
    cluster_bootstrap_paired_effect,
    cluster_sign_flip_pvalue,
    holm_adjust,
)
from .spatial import huc2_cluster_map

MINIMUM_VALID_TARGETS = 100

POINT_COLS = ("y_true", "y_pred")
PROB_COLS = ("q05", "q50", "q95", "p_exceed")
CALIBRATED_STATE = "FROZEN_CQR_PLATT_APPLIED"

BASELINE_MODELS = ("Persistence", "DampedPersistence", "Climatology")
SKILL_BASELINES = ("Persistence", "DampedPersistence", "Climatology")
HOLM_FAMILY_SIZE = 5

# The frozen five-test family (12_claim_stats.py FROZEN_FAMILY_TEST_IDS):
# ThermoRoute vs damped persistence at every lead, plus vs LightGBM at 3/7 d.
HOLM_FAMILY_CONTRASTS = (
    ("ThermoRoute", "DampedPersistence", 1),
    ("ThermoRoute", "DampedPersistence", 3),
    ("ThermoRoute", "DampedPersistence", 7),
    ("ThermoRoute", "LightGBM", 3),
    ("ThermoRoute", "LightGBM", 7),
)


def station_metrics(
    pred: pd.DataFrame,
    *,
    minimum_targets: int = MINIMUM_VALID_TARGETS,
) -> pd.DataFrame:
    """Per-station RMSE/MAE/BIAS/n per model x horizon (>= minimum_targets)."""
    rows = []
    for (model, horizon), group in pred.groupby(["model", "horizon"]):
        for site, g in group.groupby("site_id"):
            if len(g) < minimum_targets:
                continue
            residual = g.y_pred.to_numpy(float) - g.y_true.to_numpy(float)
            rows.append({
                "model": model,
                "horizon": int(horizon),
                "site_id": str(site),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
                "n": int(len(g)),
            })
    return pd.DataFrame(
        rows,
        columns=["model", "horizon", "site_id", "rmse", "mae", "bias", "n"],
    )


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Pooled RMSE/MAE/BIAS/N per model x horizon.

    Must reproduce the conventional scorer's long table exactly (same
    common-key filtering and aggregation), so P9 can diff the two.
    """
    rows = []
    for (model, horizon), group in pred.groupby(["model", "horizon"]):
        residual = group.y_pred.to_numpy(float) - group.y_true.to_numpy(float)
        rows.append({
            "model": model,
            "horizon": int(horizon),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "n": int(len(group)),
        })
    return pd.DataFrame(
        rows,
        columns=["model", "horizon", "rmse", "mae", "bias", "n"],
    )


def _station_median_rmse(station: pd.DataFrame) -> dict[tuple[str, int], float]:
    out = {}
    for (model, horizon), group in station.groupby(["model", "horizon"]):
        out[(model, int(horizon))] = float(group.rmse.median())
    return out


def _station_median_skill(
    station: pd.DataFrame, model: str, baseline: str, horizon: int,
) -> float:
    """Median over stations of the per-station skill 1 - RMSE_m/RMSE_b.

    The comparison uses the intersection of reportable stations that have
    finite RMSE for both models at this horizon, so model and baseline are
    scored on the same station set.
    """
    a = station[(station.model == model) & (station.horizon == horizon)].set_index(
        "site_id")["rmse"]
    b = station[(station.model == baseline) & (station.horizon == horizon)].set_index(
        "site_id")["rmse"]
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    if a.empty:
        return float("nan")
    return float((1.0 - a / b).median())


def skill_table(
    station: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    baselines: Sequence[str] = SKILL_BASELINES,
) -> pd.DataFrame:
    """Skill = median over reportable stations of per-station
    1 - RMSE_model/RMSE_baseline.

    Per-station skill is computed on the common station set at each
    model x horizon (intersection of reportable stations), and the reported
    value is the unweighted median of those station values -- the same
    station-level estimand as the paired DeltaRMSE of equation (9).  A ratio
    of station-median RMSEs is a different quantity and is deliberately not
    used (Section 3.6).  Baselines are present in ``station``/``pooled`` and
    yield skill rows of zero.
    """
    station_med = _station_median_rmse(station)
    rows = []
    for (model, horizon), value in station_med.items():
        for base in baselines:
            base_key = (base, horizon)
            if base_key not in station_med or base == model:
                continue
            skill = _station_median_skill(station, model, base, horizon)
            rows.append({
                "model": model,
                "horizon": horizon,
                "baseline": base,
                "skill": float(skill),
                "station_rmse": value,
                "baseline_station_rmse": station_med[base_key],
            })
    return pd.DataFrame(
        rows,
        columns=[
            "model", "horizon", "baseline", "skill",
            "station_rmse", "baseline_station_rmse",
        ],
    )


def paired_effects(
    station: pd.DataFrame,
    *,
    contrasts: Sequence[tuple[str, str, int]],
) -> pd.DataFrame:
    """Station-level paired effects, effect = candidate - reference RMSE.

    ``contrasts`` is a sequence of (candidate, reference, horizon) triples.
    Only stations reportable for both arms enter; negative favours the
    candidate.
    """
    station_med = _station_median_rmse(station)
    rows = []
    for candidate, reference, horizon in contrasts:
        candidates = {
            site: float(g.rmse.iloc[0])
            for site, g in station[
                (station.model == candidate) & (station.horizon == horizon)
            ].groupby("site_id")
        }
        references = {
            site: float(g.rmse.iloc[0])
            for site, g in station[
                (station.model == reference) & (station.horizon == horizon)
            ].groupby("site_id")
        }
        sites = sorted(set(candidates) & set(references))
        for site in sites:
            rows.append({
                "candidate": candidate,
                "reference": reference,
                "horizon": horizon,
                "site_id": site,
                "effect": candidates[site] - references[site],
            })
    return pd.DataFrame(
        rows,
        columns=["candidate", "reference", "horizon", "site_id", "effect"],
    )


def cluster_inference(
    effects: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    holm_family: int = HOLM_FAMILY_SIZE,
    holm_family_contrasts: Sequence[tuple[str, str, int]] | None = None,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Cluster bootstrap + sign-flip + Holm per (candidate, reference, horizon).

    Returns one record per contrast: effect (station-median), percentile CI,
    cluster sign-flip p-value, Holm-adjusted p-value, win rate and cluster
    counts.  The Holm family is applied across the first ``holm_family``
    records (the preregistered five-test family).
    """
    cluster_map = huc2_cluster_map(registry)
    out: dict[str, dict[str, Any]] = {}
    records = []
    for (candidate, reference, horizon), group in effects.groupby(
        ["candidate", "reference", "horizon"]
    ):
        sites = group.site_id.astype(str).to_numpy()
        effect = group.effect.to_numpy(float)
        clusters = np.asarray([cluster_map[site] for site in sites], dtype=object)
        bootstrap = cluster_bootstrap_paired_effect(
            effect, clusters, n_boot=n_boot, seed=seed
        )
        p_value = cluster_sign_flip_pvalue(
            effect, clusters, n_randomisations=50000, seed=seed + 5000
        )
        records.append({
            "candidate": candidate,
            "reference": reference,
            "horizon": int(horizon),
            "effect": bootstrap["effect"],
            "ci_low": bootstrap["ci_low"],
            "ci_high": bootstrap["ci_high"],
            "p_cluster_sign_flip": float(p_value),
            "win_rate": float(np.mean(effect < 0.0)),
            "n_stations": int(bootstrap["n_stations"]),
            "n_clusters": int(bootstrap["n_clusters"]),
        })
    for index, record in enumerate(records):
        record["holm_p"] = float(np.nan)
    out = {f"{r['candidate']}|{r['reference']}|{r['horizon']}": r for r in records}
    family_keys = [
        (c, r, int(h)) for c, r, h in holm_family_contrasts
    ] if holm_family_contrasts is not None else [
        (record["candidate"], record["reference"], int(record["horizon"]))
        for record in records[:holm_family]
    ]
    family = np.asarray(
        [out[f"{c}|{r}|{h}"]["p_cluster_sign_flip"] for c, r, h in family_keys],
        dtype=float,
    )
    adjusted = holm_adjust(family)
    for (c, r, h), value in zip(family_keys, adjusted):
        out[f"{c}|{r}|{h}"]["holm_p"] = float(value)
    return out


def probability_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Brier / interval coverage / width / interval score per model x horizon.

    Only models whose rows carry ``calibration_state ==
    FROZEN_CQR_PLATT_APPLIED`` and non-null calibrated heads contribute;
    point-only and uncalibrated models are skipped.
    """
    rows = []
    for (model, horizon), group in pred.groupby(["model", "horizon"]):
        g = group[group.calibration_state == CALIBRATED_STATE]
        if g.empty or g[["q05", "q95"]].isna().all().all():
            continue
        q05 = g.q05.to_numpy(float)
        q50 = g.q50.to_numpy(float)
        q95 = g.q95.to_numpy(float)
        y = g.y_true.to_numpy(float)
        width = q95 - q05
        coverage = float(np.mean((y >= q05) & (y <= q95)))
        brier = float(np.mean((g.p_exceed.to_numpy(float) - g.event_observed.to_numpy(float)) ** 2))
        rows.append({
            "model": model,
            "horizon": int(horizon),
            "interval_coverage": coverage,
            "interval_width": float(np.mean(width)),
            "brier": brier,
            "n": int(len(g)),
        })
    return pd.DataFrame(
        rows,
        columns=["model", "horizon", "interval_coverage", "interval_width", "brier", "n"],
    )


def reliability_bins(pred: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Station-balanced reliability table per model x horizon.

    Rows are probability bins; columns are observed event rate, mean
    forecast probability, count and the absolute calibration gap.
    """
    rows = []
    for (model, horizon), group in pred.groupby(["model", "horizon"]):
        g = group[group.calibration_state == CALIBRATED_STATE]
        if g.empty or g.p_exceed.isna().all():
            continue
        p = g.p_exceed.to_numpy(float)
        y = g.event_observed.to_numpy(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (p >= left) & (p < right)
            if not mask.any():
                continue
            rows.append({
                "model": model,
                "horizon": int(horizon),
                "bin_low": float(left),
                "bin_high": float(right),
                "mean_forecast": float(np.mean(p[mask])),
                "observed_rate": float(np.mean(y[mask])),
                "n": int(mask.sum()),
            })
    return pd.DataFrame(
        rows,
        columns=[
            "model", "horizon", "bin_low", "bin_high",
            "mean_forecast", "observed_rate", "n",
        ],
    )
