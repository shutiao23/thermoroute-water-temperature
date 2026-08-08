"""Single results authority for the WRR strong-accept redesign.

Pure-statistics layer over the persisted per-key prediction tables (no torch,
no bundles, no network).  Every headline number in the manuscript must come
from the tables this module produces under ``outputs/final/``:

* ``forecast_keys.parquet``     one row per common holdout forecast key with a
                                stable sha256 ``key_id`` and context-missingness
                                metrics computed from the raw panels
* ``predictions.parquet``       ensemble per-key predictions for the primary
                                model set only (point + calibrated columns)
* ``station_metrics.parquet``   station-first RMSE/MAE/bias/n per model x
                                horizon (reportable >= 100 paired keys)
* ``pooled_metrics.parquet``    pooled RMSE sensitivity (SI only, never a
                                headline)
* ``paired_effects.parquet``    station-level candidate - reference RMSE and
                                station skill per (candidate, reference, h)
* ``decomposition_effects.parquet`` station-level exact additive
                                G_total = G_memory + G_learned with per-station
                                fractions
* ``paper_values.tex``          \\newcommand macros bound to the claim ledger
* ``claim_ledger_resolved.csv`` one row per claim with source/estimator/value
* ``result_manifest.json``      git state, protocol hash, output digests

Estimand rules enforced here (protocol v1):

1. The sampling unit is the station: RMSE is computed within station on the
   common keys, then aggregated by unweighted median across stations.
2. Skill is the median over stations of the per-station ratio
   ``1 - RMSE_model/RMSE_reference``.  A ratio of station-median RMSEs is a
   different quantity and is never labelled station skill.
3. ``median(G_memory) + median(G_learned) != median(G_total)`` in general, so
   the additive decomposition is only reported per station, with the exact
   identity checked at station level; aggregate waterfall numbers are reported
   as equal-station means only.
4. Station counts are unique: every model at every horizon must be reportable
   on the identical station set at a given horizon (Air2stream is merged onto
   the common keys before reportability is evaluated).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

MINIMUM_VALID_TARGETS = 100
MINIMUM_STATE_KEYS = 30

PRIMARY_MODELS = (
    "Persistence", "DampedPersistence", "Climatology",
    "ThermoRoute", "LightGBM", "LSTM",
    "PlainCausalTCN-7var", "PlainMLP-7var",
)
# Air2stream lives in a separate artifact and is merged onto the common keys.
AIR2STREAM_MODEL = "Air2stream"

SKILL_BASELINES = ("Persistence", "DampedPersistence")

# Contrasts that produce headline numbers.
DECLARED_CONTRASTS = (
    # learned models vs the two references at every lead
    ("ThermoRoute", "Persistence", 1), ("ThermoRoute", "Persistence", 3),
    ("ThermoRoute", "Persistence", 7),
    ("ThermoRoute", "DampedPersistence", 1), ("ThermoRoute", "DampedPersistence", 3),
    ("ThermoRoute", "DampedPersistence", 7),
    ("LightGBM", "Persistence", 1), ("LightGBM", "Persistence", 3),
    ("LightGBM", "Persistence", 7),
    ("LightGBM", "DampedPersistence", 1), ("LightGBM", "DampedPersistence", 3),
    ("LightGBM", "DampedPersistence", 7),
    ("LSTM", "Persistence", 1), ("LSTM", "Persistence", 3),
    ("LSTM", "Persistence", 7),
    ("LSTM", "DampedPersistence", 1), ("LSTM", "DampedPersistence", 3),
    ("LSTM", "DampedPersistence", 7),
    ("PlainCausalTCN-7var", "DampedPersistence", 1),
    ("PlainCausalTCN-7var", "DampedPersistence", 3),
    ("PlainCausalTCN-7var", "DampedPersistence", 7),
    ("PlainMLP-7var", "DampedPersistence", 1),
    ("PlainMLP-7var", "DampedPersistence", 3),
    ("PlainMLP-7var", "DampedPersistence", 7),
    (AIR2STREAM_MODEL, "DampedPersistence", 1),
    (AIR2STREAM_MODEL, "DampedPersistence", 3),
    (AIR2STREAM_MODEL, "DampedPersistence", 7),
    # the frozen five-test family (kept for the SI inference block)
    ("ThermoRoute", "LightGBM", 3), ("ThermoRoute", "LightGBM", 7),
    # matched residual-model-class contrasts (Major Comment 5)
    ("PlainCausalTCN-7var", "ThermoRoute", 1),
    ("PlainCausalTCN-7var", "ThermoRoute", 3),
    ("PlainCausalTCN-7var", "ThermoRoute", 7),
)

DECOMPOSITION_MODEL = "ThermoRoute"


def key_id_for(cohort: str, task: str, site: str, issue: pd.Timestamp,
               target: pd.Timestamp, horizon: int) -> str:
    """Stable sha256 key id (never a DataFrame row number)."""
    payload = (
        f"{cohort}|{task}|{site}|{issue:%Y-%m-%d}|"
        f"{target:%Y-%m-%d}|{int(horizon)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Key registry
# --------------------------------------------------------------------------- #
def build_key_registry(
    predictions: pd.DataFrame,
    *,
    cohort: str = "temporal",
    task: str = "known_site",
    site_columns: Sequence[str] = ("site_id", "horizon", "issue_date",
                                   "target_date", "y_true"),
) -> pd.DataFrame:
    """Common-key intersection across every model in ``predictions``.

    Only keys present in ALL models (ensemble rows) at a horizon enter the
    registry, so no model can gain apparent accuracy by declining to predict
    on hard days (protocol v1, factor: keys).
    """
    required = set(site_columns) | {"model"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions lack required columns: {sorted(missing)}")
    frames = []
    for (model, horizon), group in predictions.groupby(["model", "horizon"]):
        sub = group[list(site_columns)].drop_duplicates()
        sub = sub.copy()
        sub["model"] = model
        sub["horizon"] = int(horizon)
        frames.append(sub)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["site_id", "horizon", "issue_date", "target_date"]
    sizes = combined.groupby(keys + ["model"]).size().unstack("model")
    common = sizes.dropna().index
    registry = (
        combined.drop(columns=["model"]).drop_duplicates(keys)
        .set_index(keys).loc[common].reset_index()
    )
    if len(registry) == 0:
        raise ValueError("empty common-key registry")
    registry["cohort"] = cohort
    registry["task"] = task
    registry["period"] = "2021-2023"
    registry["target_observed"] = np.isfinite(
        registry["y_true"].to_numpy(float)
    ).astype(bool)
    registry["key_id"] = [
        key_id_for(cohort, task, str(site), issue, target, int(h))
        for site, issue, target, h in zip(
            registry["site_id"], registry["issue_date"],
            registry["target_date"], registry["horizon"], strict=True,
        )
    ]
    return registry


def attach_context_missingness(
    registry: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    context_days: Sequence[int] = (7, 14, 32),
) -> pd.DataFrame:
    """Per-key recent water-temperature observed-fraction metrics.

    ``panel`` must be the raw (pre-imputation) combined panel with columns
    ``site_id``, ``DATE``, ``WTEMP``.  Observed means finite ``WTEMP``.
    All counts are left-closed over the ``context_days`` days ending at the
    issue date (``[issue - days + 1, issue]``), matching the construction
    buffer used by the sequence models.
    """
    if {"site_id", "DATE", "WTEMP"} - set(panel.columns):
        raise ValueError("panel lacks site_id/DATE/WTEMP")
    p = panel.copy()
    p["DATE"] = pd.to_datetime(p["DATE"])
    p["site_id"] = p["site_id"].astype(str).str.zfill(8)
    p = p.sort_values(["site_id", "DATE"])
    p = p.drop_duplicates(["site_id", "DATE"], keep="last")
    p["observed"] = np.isfinite(p["WTEMP"].to_numpy(float)).astype(np.int8)
    out = registry.copy()
    out["issue_date"] = pd.to_datetime(out["issue_date"])
    out["site_id"] = out["site_id"].astype(str).str.zfill(8)
    # The panels carry one row per calendar day per site, so a trailing
    # row-rolling sum over the sorted panel equals the calendar-day window
    # count ending at each DATE.  Verified by test_missingness_against_brute_force.
    observed_rolling = {
        days: p.groupby("site_id")["observed"].transform(
            lambda series: series.rolling(days, min_periods=days).sum()
        )
        for days in context_days
    }
    rolling_table = pd.DataFrame(
        {"site_id": p["site_id"], "DATE": p["DATE"],
         **{f"n_observed_wtemp_{days}d": observed_rolling[days]
            for days in context_days}}
    )
    out = out.merge(
        rolling_table, left_on=["site_id", "issue_date"],
        right_on=["site_id", "DATE"], how="left",
    ).drop(columns=["DATE"])
    for days in context_days:
        out[f"context_rows_{days}d"] = days
        out[f"fraction_observed_wtemp_{days}d"] = (
            out[f"n_observed_wtemp_{days}d"] / out[f"context_rows_{days}d"]
        )
    last_obs = (
        p[p.observed == 1]
        .groupby("site_id")["DATE"].max().rename("last_obs_date")
    )
    out = out.merge(last_obs, left_on="site_id", right_index=True, how="left")
    out["days_since_last_observed_wtemp"] = (
        out["issue_date"] - out["last_obs_date"]
    ).dt.days
    out.loc[out["last_obs_date"].isna(), "days_since_last_observed_wtemp"] = -1
    out["issue_wtemp_observed"] = out["days_since_last_observed_wtemp"] == 0
    keep = [c for c in (
        "key_id", "cohort", "task", "site_id", "issue_date", "target_date",
        "horizon", "period", "target_observed", "y_true",
        "issue_wtemp_observed", "days_since_last_observed_wtemp",
    ) if c in out.columns]
    for days in context_days:
        keep += [f"n_observed_wtemp_{days}d", f"fraction_observed_wtemp_{days}d"]
    return out[keep].sort_values(["site_id", "issue_date", "horizon"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Station-first metrics
# --------------------------------------------------------------------------- #
def station_metrics(
    predictions: pd.DataFrame,
    *,
    minimum_targets: int = MINIMUM_VALID_TARGETS,
) -> pd.DataFrame:
    """Station-first RMSE/MAE/bias/n per model x horizon.

    Only stations with at least ``minimum_targets`` paired keys for that
    model-horizon are reportable.  No pooled arithmetic enters here.
    """
    rows: list[dict[str, Any]] = []
    pred = predictions.copy()
    pred["site_id"] = pred["site_id"].astype(str).str.zfill(8)
    for (model, horizon), group in pred.groupby(["model", "horizon"]):
        for site, g in group.groupby("site_id"):
            if len(g) < minimum_targets:
                continue
            residual = g["y_pred"].to_numpy(float) - g["y_true"].to_numpy(float)
            rows.append({
                "model": str(model),
                "horizon": int(horizon),
                "site_id": str(site),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
                "n": int(len(g)),
            })
    if not rows:
        raise ValueError("no reportable station cells")
    return pd.DataFrame(rows, columns=[
        "model", "horizon", "site_id", "rmse", "mae", "bias", "n"])


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pooled RMSE per model x horizon — SI sensitivity only."""
    rows: list[dict[str, Any]] = []
    for (model, horizon), group in predictions.groupby(["model", "horizon"]):
        residual = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        rows.append({
            "model": str(model), "horizon": int(horizon),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "n": int(len(group)),
        })
    return pd.DataFrame(rows, columns=["model", "horizon", "rmse", "mae", "bias", "n"])


def assert_common_station_sets(station: pd.DataFrame) -> None:
    """Every model at a horizon must be reportable on the same station set."""
    counts: dict[int, dict[str, int]] = {}
    for (model, horizon), g in station.groupby(["model", "horizon"]):
        counts.setdefault(int(horizon), {})[str(model)] = int(g["site_id"].nunique())
    for horizon, per_model in counts.items():
        if len(set(per_model.values())) != 1:
            raise ValueError(
                f"horizon {horizon} reportable station counts differ across models: "
                f"{per_model}"
            )


# --------------------------------------------------------------------------- #
# Paired effects and skill
# --------------------------------------------------------------------------- #
def paired_effects(
    station: pd.DataFrame,
    *,
    contrasts: Sequence[tuple[str, str, int]] = DECLARED_CONTRASTS,
) -> pd.DataFrame:
    """Station-level paired effects: delta_rmse = candidate - reference.

    Only stations reportable for both arms enter.  Negative delta favours the
    candidate; station_skill = 1 - RMSE_c/RMSE_r is the per-station ratio.
    """
    rows: list[dict[str, Any]] = []
    for candidate, reference, horizon in contrasts:
        cand = station[(station.model == candidate) & (station.horizon == horizon)] \
            .set_index("site_id")
        ref = station[(station.model == reference) & (station.horizon == horizon)] \
            .set_index("site_id")
        sites = cand.index.intersection(ref.index)
        if sites.empty:
            continue
        for site in sites:
            c, r = cand.loc[site], ref.loc[site]
            rows.append({
                "candidate": candidate,
                "reference": reference,
                "horizon": int(horizon),
                "site_id": site,
                "candidate_rmse": float(c["rmse"]),
                "reference_rmse": float(r["rmse"]),
                "delta_rmse": float(c["rmse"] - r["rmse"]),
                "station_skill": float(1.0 - c["rmse"] / r["rmse"]),
                "n_common_keys": int(min(c["n"], r["n"])),
            })
    if not rows:
        raise ValueError("no paired effects computed")
    return pd.DataFrame(rows, columns=[
        "candidate", "reference", "horizon", "site_id", "candidate_rmse",
        "reference_rmse", "delta_rmse", "station_skill", "n_common_keys"])


def median_station_skill(effects: pd.DataFrame, candidate: str,
                         reference: str, horizon: int) -> float:
    """Median over stations of the per-station skill (never a ratio of
    station-median RMSEs)."""
    g = effects[(effects.candidate == candidate)
                & (effects.reference == reference)
                & (effects.horizon == horizon)]
    if g.empty:
        return float("nan")
    return float(g["station_skill"].median())


# --------------------------------------------------------------------------- #
# Exact station-level decomposition
# --------------------------------------------------------------------------- #
def decomposition_effects(
    station: pd.DataFrame,
    *,
    model: str = DECOMPOSITION_MODEL,
    horizons: Sequence[int] = (1, 3, 7),
) -> pd.DataFrame:
    """Station-level additive error budget.

    G_total = RMSE_persistence - RMSE_model
    G_memory = RMSE_persistence - RMSE_damped
    G_learned = RMSE_damped - RMSE_model
    so G_total == G_memory + G_learned EXACTLY at station level.  Fractions
    are computed only where G_total > 0 (``memory_fraction`` /
    ``learned_fraction``; NaN otherwise).
    """
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        per = {}
        for which in ("Persistence", "DampedPersistence", model):
            g = station[(station.model == which) & (station.horizon == horizon)] \
                .set_index("site_id")["rmse"]
            per[which] = g
        sites = (per["Persistence"].index
                 .intersection(per["DampedPersistence"].index)
                 .intersection(per[model].index))
        if sites.empty:
            continue
        for site in sites:
            p, d, m = per["Persistence"].loc[site], per["DampedPersistence"].loc[site], per[model].loc[site]
            g_total = float(p - m)
            g_memory = float(p - d)
            g_learned = float(d - m)
            if not np.isclose(g_total, g_memory + g_learned, atol=1e-12):
                raise ValueError(f"decomposition not additive at {site}/h{horizon}")
            rows.append({
                "site_id": site,
                "horizon": int(horizon),
                "persistence_rmse": float(p),
                "damped_rmse": float(d),
                "model_rmse": float(m),
                "g_total": g_total,
                "g_memory": g_memory,
                "g_learned": g_learned,
                "g_total_positive": bool(g_total > 0.0),
                "memory_fraction": g_memory / g_total if g_total > 0.0 else float("nan"),
                "learned_fraction": g_learned / g_total if g_total > 0.0 else float("nan"),
                "model": model,
            })
    if not rows:
        raise ValueError("no decomposition rows")
    return pd.DataFrame(rows)


def mean_additive_waterfall(decomp: pd.DataFrame, horizon: int) -> dict[str, float]:
    """Equal-station mean decomposition — the only additive aggregate."""
    g = decomp[decomp.horizon == horizon]
    return {
        "mean_g_total": float(g["g_total"].mean()),
        "mean_g_memory": float(g["g_memory"].mean()),
        "mean_g_learned": float(g["g_learned"].mean()),
        "n_stations": int(len(g)),
    }


# --------------------------------------------------------------------------- #
# Claim ledger resolution
# --------------------------------------------------------------------------- #
def _format(value: float, decimals: int) -> str:
    if np.isnan(float(value)):
        return "NOT_AVAILABLE"
    return f"{float(value):.{decimals}f}"


def resolve_claim_ledger(
    ledger: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Resolve every claim in ``paper/claim_ledger.yaml`` against the tables.

    Returns one row per claim with ``value``, ``status`` and ``latex_macro``.
    Unresolved claims get status ``PENDING`` (missing source table or filter).
    """
    claims = ledger["claims"] if isinstance(ledger, Mapping) and "claims" in ledger \
        else list(ledger)
    rows: list[dict[str, Any]] = []
    for claim in claims:
        source = claim["source_table"]
        macro = claim["latex_macro"]
        table = tables.get(Path(source).name)
        value: float | int | None = None
        status = "PENDING"
        if table is not None:
            try:
                value = _resolve_estimator(table, claim)
                status = "RESOLVED"
            except (KeyError, ValueError) as exc:
                status = f"UNRESOLVED: {exc}"
        rows.append({
            "claim_id": claim["claim_id"],
            "source_table": source,
            "estimator": claim["estimator"],
            "latex_macro": macro,
            "value": "" if value is None else str(value),
            "status": status,
            "used_in": ";".join(claim.get("used_in", [])),
        })
    return pd.DataFrame(rows, columns=[
        "claim_id", "source_table", "estimator", "latex_macro",
        "value", "status", "used_in"])


def _resolve_estimator(table: pd.DataFrame, claim: Mapping[str, Any]) -> Any:
    estimator = claim["estimator"]
    filt = claim.get("filter", {})
    frame = table
    for key, value in filt.items():
        if value is None:
            continue
        if key == "g_total":
            if value == "gt_0":
                frame = frame[frame["g_total"] > 0]
            continue
        if key in ("stat", "period"):
            # `period` is an authority-level invariant (all tables are the
            # 2021-2023 holdout), not a table column.
            continue
        column = "state_name" if key == "state" else key
        frame = frame[frame[column] == value]
    if estimator == "reportable_station_count":
        if frame.empty:
            raise ValueError("empty after filter")
        return int(frame["site_id"].nunique())
    if estimator == "median_station_rmse":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["rmse"].median())
    if estimator == "median_station_paired_delta_rmse":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["delta_rmse"].median())
    if estimator == "median_station_skill":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["station_skill"].median())
    if estimator == "median_station_g_memory":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["g_memory"].median())
    if estimator == "median_station_g_learned":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["g_learned"].median())
    if estimator == "median_station_memory_fraction":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["memory_fraction"].median())
    if estimator == "median_station_state_delta_rmse":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["delta_rmse"].median())
    if estimator == "median":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame[claim["filter"]["stat"]].median())
    if estimator == "median_station_region_minus_random_rmse":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["region_minus_random_rmse"].median())
    if estimator == "median_station_region_minus_random_rmse":
        if frame.empty:
            raise ValueError("empty after filter")
        pivot = frame.pivot_table(
            index="site_id", columns="geometry", values="rmse", aggfunc="mean")
        pivot = pivot.dropna(subset=["region", "random"])
        if pivot.empty:
            raise ValueError("no paired region/random cells")
        return float((pivot["region"] - pivot["random"]).median())
    if estimator == "median_nearest_km":
        if frame.empty:
            raise ValueError("empty after filter")
        return float(frame["nearest_km"].median())
    raise ValueError(f"unknown estimator {estimator}")


def _is_integral(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def write_paper_values(
    resolved: pd.DataFrame,
    path: Path,
    *,
    decimals: Mapping[str, int] | None = None,
) -> None:
    """Emit ``paper_values.tex`` \\newcommand macros for resolved claims."""
    decimals = decimals or {}
    lines = [
        "% Auto-generated by scripts/final/build_results_authority.py -- do not edit.",
        "% Regenerate with: python scripts/final/build_results_authority.py",
        "%",
    ]
    for row in resolved.sort_values("latex_macro").itertuples(index=False):
        if row.status != "RESOLVED":
            lines.append(f"% {{UNRESOLVED}} \\newcommand{{\\{row.latex_macro}}}{{...}} "
                         f"(claim {row.claim_id})")
            continue
        macro = str(row.latex_macro).replace(" ", "")
        value = str(row.value)
        if _is_integral(value):
            pass  # station counts and km distances stay integral
        elif macro.endswith("HalfLifeMedianDays"):
            value = f"{float(value):.1f}"
        else:
            value = f"{float(value):.3f}"
        lines.append(f"\\newcommand{{\\{macro}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    git_sha: str,
    dirty: bool,
    command: str,
    protocol_sha256: str,
    output_hashes: Mapping[str, str],
    started_at: str,
    completed_at: str,
) -> None:
    manifest = {
        "format": "thermoroute.final-results-manifest.v1",
        "git_sha": git_sha,
        "dirty": bool(dirty),
        "command": command,
        "protocol_sha256": protocol_sha256,
        "started_at": started_at,
        "completed_at": completed_at,
        "outputs": {name: digest for name, digest in sorted(output_hashes.items())},
    }
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
