#!/usr/bin/env python3
"""Build the create-only v4 authority for the completed information subset.

This entrypoint is deliberately an *authority builder*, not a training runner.
It accepts only the completed L0/L1/L2 x random/region subset, independently
reconstructs every station metric from the prediction shards, proves two-sided
forecast-key equality, and publishes paired station contrasts.  It fails closed
on any missing or unexpected cell, seed, fold, horizon, model, key, or metric.

The output status is intentionally explicit: the current artifacts are a
completed subset, not the full protocol-v2 information/forcing ladder.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"

DEFAULT_EFFECTS = FINAL / "ladder_effects.parquet"
DEFAULT_SHARDS = FINAL / "ladder_shards"
DEFAULT_REGISTRY = FINAL / "forecast_keys.parquet"
DEFAULT_PROTOCOL = ROOT / "protocols" / "wrr_strong_accept_protocol_v2.yaml"
DEFAULT_OUT_DIR = FINAL / "information_regime_v4"

AUTHORITY_SCHEMA = "information_regime_authority_v4"
AUTHORITY_STATUS = "COMPLETED_SUBSET_NOT_FULL_V2"
OUTPUT_PREFIX = "information_regime_v4"
MIN_REPORTABLE_KEYS = 100
TARGET_ABSOLUTE_TOLERANCE = 2e-6  # registry float32 round-trip versus shard float64

GEOMETRIES = ("random", "region")
LEVELS = ("L0", "L1", "L2")
MODELS = ("LightGBM", "ResidualLightGBM")
HORIZONS = (1, 3, 7)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
REGION_SEEDS = (0,)
FOLDS = (0, 1, 2, 3)

EFFECT_KEY = ("geometry", "level", "model", "seed", "site_id", "horizon")
ABSOLUTE_KEY = ("geometry", "level", "model", "horizon", "site_id")
REGISTRY_KEY = ("site_id", "horizon", "issue_date")

SHARD_RE = re.compile(
    r"^(random|region)_(L0|L1|L2)_"
    r"(LightGBM|ResidualLightGBM)_s([0-9]+)_f([0-9]+)_h([0-9]+)\.parquet$"
)


class AuthorityError(RuntimeError):
    """Raised when an authority precondition cannot be proved."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bound_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    path = path.resolve()
    try:
        info = path.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label} cannot be inspected: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise AuthorityError(f"{label} is not a regular, non-symlink file: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AuthorityError(f"{label} cannot be read: {path}") from exc
    if not payload:
        raise AuthorityError(f"{label} is empty: {path}")
    return payload, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _read_bound_parquet(path: Path, *, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload, binding = _read_bound_bytes(path, label=label)
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise AuthorityError(f"{label} is not readable Parquet: {path}") from exc
    return frame, binding


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AuthorityError(f"{label} lacks required columns: {missing}")


def _integer_series(series: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise AuthorityError(f"{label} is not numeric") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise AuthorityError(f"{label} must contain finite integers")
    return numeric.astype("int64")


def _normalise_sites(series: pd.Series, *, label: str) -> pd.Series:
    if series.isna().any():
        raise AuthorityError(f"{label} contains null site_id values")
    result = series.astype(str).str.strip().str.zfill(8)
    # The study registry contains both eight-digit USGS identifiers and
    # fifteen-digit canonical non-USGS station identifiers.
    valid = result.str.fullmatch(r"[0-9]{8,15}")
    if result.eq("").any() or not valid.all():
        bad = result[~valid].head(3).tolist()
        raise AuthorityError(f"{label} contains invalid site_id values: {bad}")
    return result


def _normalise_dates(series: pd.Series, *, label: str) -> pd.Series:
    try:
        result = pd.to_datetime(series, errors="raise", utc=True).dt.tz_localize(None)
    except Exception as exc:
        raise AuthorityError(f"{label} contains invalid issue_date values") from exc
    if result.isna().any():
        raise AuthorityError(f"{label} contains null issue_date values")
    if not result.eq(result.dt.normalize()).all():
        raise AuthorityError(f"{label} issue_date values must be midnight dates")
    return result


def _finite_numeric(series: pd.Series, *, label: str) -> pd.Series:
    try:
        result = pd.to_numeric(series, errors="raise").astype(float)
    except Exception as exc:
        raise AuthorityError(f"{label} is not numeric") from exc
    if not np.isfinite(result.to_numpy()).all():
        raise AuthorityError(f"{label} contains null or non-finite values")
    return result


def expected_shard_cells() -> set[tuple[str, str, str, int, int, int]]:
    """Return the exact 432-cell completed-subset shard inventory."""
    cells: set[tuple[str, str, str, int, int, int]] = set()
    for geometry in GEOMETRIES:
        seeds = RANDOM_SEEDS if geometry == "random" else REGION_SEEDS
        for level in LEVELS:
            for model in MODELS:
                for seed in seeds:
                    for fold in FOLDS:
                        for horizon in HORIZONS:
                            cells.add((geometry, level, model, seed, fold, horizon))
    if len(cells) != 432:  # defensive guard against accidental constant edits
        raise AssertionError(f"internal expected inventory is {len(cells)}, not 432")
    return cells


def validate_shard_cell_set(
    actual: Iterable[tuple[str, str, str, int, int, int]],
) -> None:
    """Fail closed unless ``actual`` is exactly the completed 432-cell subset."""
    actual_set = set(actual)
    expected = expected_shard_cells()
    missing = sorted(expected - actual_set)
    extra = sorted(actual_set - expected)
    if missing or extra or len(actual_set) != 432:
        raise AuthorityError(
            "shard inventory is not the exact completed subset: "
            f"{len(missing)} missing, {len(extra)} unexpected; "
            f"missing examples={missing[:3]}, unexpected examples={extra[:3]}"
        )


def validate_shard_inventory(
    shard_dir: Path,
) -> dict[tuple[str, str, str, int, int, int], Path]:
    shard_dir = Path(os.path.abspath(os.fspath(shard_dir)))
    try:
        info = shard_dir.lstat()
    except OSError as exc:
        raise AuthorityError(f"shard directory cannot be inspected: {shard_dir}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise AuthorityError(f"shard path is not a non-symlink directory: {shard_dir}")

    inventory: dict[tuple[str, str, str, int, int, int], Path] = {}
    for path in sorted(shard_dir.glob("*.parquet")):
        match = SHARD_RE.fullmatch(path.name)
        if match is None:
            raise AuthorityError(f"unexpected shard filename: {path.name}")
        key = (
            match.group(1),
            match.group(2),
            match.group(3),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
        )
        if key in inventory:
            raise AuthorityError(f"duplicate shard cell: {key}")
        try:
            file_info = path.lstat()
        except OSError as exc:
            raise AuthorityError(f"shard cannot be inspected: {path}") from exc
        if (
            not stat.S_ISREG(file_info.st_mode)
            or stat.S_ISLNK(file_info.st_mode)
            or file_info.st_size <= 0
        ):
            raise AuthorityError(f"shard is empty, non-regular, or a symlink: {path}")
        inventory[key] = path
    validate_shard_cell_set(inventory)
    return inventory


def expected_effect_cells() -> set[tuple[str, str, str, int, int]]:
    cells: set[tuple[str, str, str, int, int]] = set()
    for geometry in GEOMETRIES:
        seeds = RANDOM_SEEDS if geometry == "random" else REGION_SEEDS
        for level in LEVELS:
            for model in MODELS:
                for seed in seeds:
                    for horizon in HORIZONS:
                        cells.add((geometry, level, model, seed, horizon))
    if len(cells) != 108:
        raise AssertionError(f"internal expected effect inventory is {len(cells)}, not 108")
    return cells


def validate_effects_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalise the raw per-seed station effects table."""
    required = set(EFFECT_KEY) | {"n", "rmse", "rmse_damped", "reportable"}
    _require_columns(frame, required, label="ladder_effects")
    effects = frame.loc[:, sorted(required)].copy()
    for column in ("geometry", "level", "model"):
        if effects[column].isna().any():
            raise AuthorityError(f"ladder_effects.{column} contains null values")
        effects[column] = effects[column].astype(str)
    effects["site_id"] = _normalise_sites(effects["site_id"], label="ladder_effects")
    effects["seed"] = _integer_series(effects["seed"], label="ladder_effects.seed")
    effects["horizon"] = _integer_series(
        effects["horizon"], label="ladder_effects.horizon"
    )
    effects["n"] = _integer_series(effects["n"], label="ladder_effects.n")
    if (effects["n"] <= 0).any():
        raise AuthorityError("ladder_effects.n must be positive")
    effects["rmse"] = _finite_numeric(effects["rmse"], label="ladder_effects.rmse")
    effects["rmse_damped"] = _finite_numeric(
        effects["rmse_damped"], label="ladder_effects.rmse_damped"
    )
    if (effects[["rmse", "rmse_damped"]] < 0).any(axis=None):
        raise AuthorityError("ladder_effects contains a negative RMSE")
    if not pd.api.types.is_bool_dtype(effects["reportable"]):
        raise AuthorityError("ladder_effects.reportable must be boolean")
    effects["reportable"] = effects["reportable"].astype(bool)
    expected_reportable = effects["n"] >= MIN_REPORTABLE_KEYS
    if not effects["reportable"].equals(expected_reportable):
        count = int((effects["reportable"] != expected_reportable).sum())
        raise AuthorityError(
            f"ladder_effects has {count} reportable flags inconsistent with "
            f"n >= {MIN_REPORTABLE_KEYS}"
        )
    if effects.duplicated(list(EFFECT_KEY)).any():
        example = effects.loc[effects.duplicated(list(EFFECT_KEY), keep=False), EFFECT_KEY]
        raise AuthorityError(
            "ladder_effects has duplicate station cells: "
            f"{example.head(3).to_dict(orient='records')}"
        )

    actual_cells = set(
        effects[["geometry", "level", "model", "seed", "horizon"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_cells = expected_effect_cells()
    missing = sorted(expected_cells - actual_cells)
    extra = sorted(actual_cells - expected_cells)
    if missing or extra:
        raise AuthorityError(
            "ladder_effects is not the exact L0/L1/L2 completed subset: "
            f"{len(missing)} missing cells, {len(extra)} unexpected cells; "
            f"missing examples={missing[:3]}, unexpected examples={extra[:3]}"
        )

    expected_sites: tuple[str, ...] | None = None
    for cell, group in effects.groupby(
        ["geometry", "level", "model", "seed", "horizon"], sort=True
    ):
        sites = tuple(sorted(group["site_id"].tolist()))
        if expected_sites is None:
            expected_sites = sites
        elif sites != expected_sites:
            raise AuthorityError(
                f"ladder_effects station universe differs in raw cell {cell}"
            )
    if not expected_sites:
        raise AuthorityError("ladder_effects has no station rows")
    return effects.sort_values(list(EFFECT_KEY), kind="mergesort").reset_index(drop=True)


def validate_forecast_registry(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*REGISTRY_KEY, "y_true"}
    _require_columns(frame, required, label="forecast registry")
    registry = frame.loc[:, list(REGISTRY_KEY) + ["y_true"]].copy()
    registry["site_id"] = _normalise_sites(registry["site_id"], label="forecast registry")
    registry["horizon"] = _integer_series(
        registry["horizon"], label="forecast registry.horizon"
    )
    registry["issue_date"] = _normalise_dates(
        registry["issue_date"], label="forecast registry"
    )
    registry["y_true"] = _finite_numeric(
        registry["y_true"], label="forecast registry.y_true"
    )
    horizons = set(registry["horizon"].unique().tolist())
    if horizons != set(HORIZONS):
        raise AuthorityError(
            f"forecast registry horizons are {sorted(horizons)}, expected {list(HORIZONS)}"
        )
    if registry.duplicated(list(REGISTRY_KEY)).any():
        raise AuthorityError("forecast registry has duplicate forecast keys")
    if registry.empty:
        raise AuthorityError("forecast registry is empty")
    return registry.sort_values(list(REGISTRY_KEY), kind="mergesort").reset_index(drop=True)


def assert_two_sided_key_equality(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Prove actual == expected on forecast keys and return their 1:1 join."""
    key = list(REGISTRY_KEY)
    if actual.duplicated(key).any():
        raise AuthorityError(f"{label} has duplicate scored keys")
    if expected.duplicated(key).any():
        raise AuthorityError(f"{label} reference has duplicate registry keys")
    joined = actual.merge(
        expected,
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_registry"),
        sort=True,
    )
    missing = int((joined["_merge"] == "right_only").sum())
    extra = int((joined["_merge"] == "left_only").sum())
    if missing or extra:
        raise AuthorityError(
            f"{label} violates two-sided forecast-key equality: "
            f"{missing} registry keys missing, {extra} keys outside registry"
        )
    return joined.drop(columns="_merge")


def _validate_shard_frame(
    frame: pd.DataFrame,
    *,
    cell: tuple[str, str, str, int, int, int],
    label: str,
) -> pd.DataFrame:
    columns = {
        *REGISTRY_KEY,
        "y_pred",
        "y_true",
        "y_damped",
        "geometry",
        "level",
        "model",
        "seed",
        "fold",
    }
    _require_columns(frame, columns, label=label)
    shard = frame.loc[:, sorted(columns)].copy()
    if shard.empty:
        raise AuthorityError(f"{label} is empty")
    shard["site_id"] = _normalise_sites(shard["site_id"], label=label)
    shard["issue_date"] = _normalise_dates(shard["issue_date"], label=label)
    for column in ("horizon", "seed", "fold"):
        shard[column] = _integer_series(shard[column], label=f"{label}.{column}")
    for column in ("y_pred", "y_true", "y_damped"):
        shard[column] = _finite_numeric(shard[column], label=f"{label}.{column}")
    geometry, level, model, seed, fold, horizon = cell
    expected_values: Mapping[str, Any] = {
        "geometry": geometry,
        "level": level,
        "model": model,
        "seed": seed,
        "fold": fold,
        "horizon": horizon,
    }
    for column, expected in expected_values.items():
        values = set(shard[column].astype(type(expected)).unique().tolist())
        if values != {expected}:
            raise AuthorityError(
                f"{label}.{column} metadata is {sorted(values)}, expected {expected!r}"
            )
    if shard.duplicated(list(REGISTRY_KEY)).any():
        raise AuthorityError(f"{label} has duplicate forecast keys")
    return shard


def audit_shards_and_reconstruct_effects(
    inventory: Mapping[tuple[str, str, str, int, int, int], Path],
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Audit every shard and independently reconstruct raw station metrics."""
    validate_shard_cell_set(inventory)
    registry_groups = {
        (site, int(horizon)): group.loc[:, list(REGISTRY_KEY) + ["y_true"]].copy()
        for (site, horizon), group in registry.groupby(["site_id", "horizon"], sort=False)
    }
    registry_by_horizon = {
        horizon: registry.loc[
            registry["horizon"] == horizon, list(REGISTRY_KEY) + ["y_true"]
        ].copy()
        for horizon in HORIZONS
    }

    raw_rows: list[dict[str, Any]] = []
    shard_bindings: list[dict[str, Any]] = []
    checked_keys = 0
    target_values_checked = 0

    for geometry, level, model, seed, horizon in sorted(expected_effect_cells()):
        arm_frames: list[pd.DataFrame] = []
        for fold in FOLDS:
            cell = (geometry, level, model, seed, fold, horizon)
            path = inventory[cell]
            payload, binding = _read_bound_bytes(path, label=f"shard {path.name}")
            try:
                unvalidated = pd.read_parquet(io.BytesIO(payload))
            except Exception as exc:
                raise AuthorityError(f"shard is not readable Parquet: {path}") from exc
            shard = _validate_shard_frame(unvalidated, cell=cell, label=path.name)
            held_sites = sorted(shard["site_id"].unique().tolist())
            missing_registry_sites = [
                site for site in held_sites if (site, horizon) not in registry_groups
            ]
            if missing_registry_sites:
                raise AuthorityError(
                    f"{path.name} has sites absent from registry: {missing_registry_sites[:3]}"
                )
            expected = pd.concat(
                [registry_groups[(site, horizon)] for site in held_sites],
                ignore_index=True,
            )
            joined = assert_two_sided_key_equality(
                shard,
                expected,
                label=path.name,
            )
            if not np.allclose(
                joined["y_true"].to_numpy(float),
                joined["y_true_registry"].to_numpy(float),
                rtol=0.0,
                atol=TARGET_ABSOLUTE_TOLERANCE,
                equal_nan=False,
            ):
                raise AuthorityError(f"{path.name} y_true values differ from registry")
            checked_keys += len(shard)
            target_values_checked += len(shard)
            binding.update(
                {
                    "name": path.name,
                    "cell": {
                        "geometry": geometry,
                        "level": level,
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "horizon": horizon,
                    },
                    "rows": len(shard),
                }
            )
            shard_bindings.append(binding)
            arm_frames.append(
                shard.loc[:, [*REGISTRY_KEY, "y_pred", "y_true", "y_damped"]]
            )

        arm = pd.concat(arm_frames, ignore_index=True)
        joined_arm = assert_two_sided_key_equality(
            arm,
            registry_by_horizon[horizon],
            label=f"{geometry}/{level}/{model}/seed{seed}/h{horizon} across folds",
        )
        if not np.allclose(
            joined_arm["y_true"].to_numpy(float),
            joined_arm["y_true_registry"].to_numpy(float),
            rtol=0.0,
            atol=TARGET_ABSOLUTE_TOLERANCE,
            equal_nan=False,
        ):
            raise AuthorityError(
                f"{geometry}/{level}/{model}/seed{seed}/h{horizon} targets "
                "differ from the forecast registry"
            )
        for site, station in arm.groupby("site_id", sort=True):
            prediction_error = (
                station["y_pred"].to_numpy(float) - station["y_true"].to_numpy(float)
            )
            damped_error = (
                station["y_damped"].to_numpy(float) - station["y_true"].to_numpy(float)
            )
            n = len(station)
            raw_rows.append(
                {
                    "geometry": geometry,
                    "level": level,
                    "model": model,
                    "seed": seed,
                    "site_id": site,
                    "horizon": horizon,
                    "n": n,
                    "rmse": float(np.sqrt(np.mean(np.square(prediction_error)))),
                    "rmse_damped": float(np.sqrt(np.mean(np.square(damped_error)))),
                    "reportable": bool(n >= MIN_REPORTABLE_KEYS),
                }
            )

    reconstructed = pd.DataFrame(raw_rows)
    reconstructed = validate_effects_subset(reconstructed)
    shard_bindings.sort(key=lambda item: item["name"])
    digest_payload = "".join(
        f"{item['name']}\0{item['sha256']}\0{item['bytes']}\n"
        for item in shard_bindings
    ).encode("utf-8")
    audit = {
        "two_sided_registry_equality": True,
            "target_values_equal_registry": True,
            "target_absolute_tolerance": TARGET_ABSOLUTE_TOLERANCE,
        "checked_shards": len(shard_bindings),
        "checked_scored_keys": checked_keys,
        "checked_target_values": target_values_checked,
        "reconstructed_station_cells": len(reconstructed),
        "shard_set_sha256": _sha256(digest_payload),
    }
    return reconstructed, audit, shard_bindings


def assert_effects_match_reconstruction(
    effects: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> None:
    columns = [*EFFECT_KEY, "n", "rmse", "rmse_damped", "reportable"]
    joined = effects.loc[:, columns].merge(
        reconstructed.loc[:, columns],
        on=list(EFFECT_KEY),
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_effects", "_reconstructed"),
        sort=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int((joined["_merge"] == "right_only").sum())
        extra = int((joined["_merge"] == "left_only").sum())
        raise AuthorityError(
            f"ladder_effects key set differs from shard reconstruction: "
            f"{missing} reconstructed rows missing, {extra} unexpected rows"
        )
    for column in ("n", "reportable"):
        if not joined[f"{column}_effects"].equals(joined[f"{column}_reconstructed"]):
            count = int(
                (joined[f"{column}_effects"] != joined[f"{column}_reconstructed"]).sum()
            )
            raise AuthorityError(
                f"ladder_effects.{column} differs from reconstruction in {count} rows"
            )
    for column in ("rmse", "rmse_damped"):
        if not np.allclose(
            joined[f"{column}_effects"].to_numpy(float),
            joined[f"{column}_reconstructed"].to_numpy(float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ):
            delta = np.abs(
                joined[f"{column}_effects"].to_numpy(float)
                - joined[f"{column}_reconstructed"].to_numpy(float)
            )
            raise AuthorityError(
                f"ladder_effects.{column} differs from reconstruction; "
                f"max absolute difference={float(delta.max()):.3g}"
            )


def common_reportable_sites(effects: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    """Return the horizon-wise intersection reportable in every raw cell."""
    expected_rows_per_site_horizon = len(expected_effect_cells()) // len(HORIZONS)
    result: dict[int, tuple[str, ...]] = {}
    for horizon in HORIZONS:
        subset = effects.loc[effects["horizon"] == horizon]
        groups = subset.groupby("site_id", sort=True).agg(
            rows=("reportable", "size"),
            all_reportable=("reportable", "all"),
        )
        incomplete = groups["rows"] != expected_rows_per_site_horizon
        if incomplete.any():
            examples = groups.index[incomplete].tolist()[:3]
            raise AuthorityError(
                f"h{horizon} stations do not span all {expected_rows_per_site_horizon} "
                f"raw cells: {examples}"
            )
        sites = tuple(groups.index[groups["all_reportable"]].astype(str).tolist())
        if not sites:
            raise AuthorityError(f"h{horizon} has no station reportable in every raw cell")
        result[horizon] = sites
    return result


def build_absolute_station_rows(
    effects: pd.DataFrame,
    *,
    protocol_sha256: str,
) -> tuple[pd.DataFrame, dict[int, tuple[str, ...]]]:
    """Produce seed-first station-level absolute RMSE rows on common sites."""
    effects = validate_effects_subset(effects)
    common = common_reportable_sites(effects)
    keep = pd.Series(False, index=effects.index)
    for horizon, sites in common.items():
        keep |= (effects["horizon"] == horizon) & effects["site_id"].isin(sites)
    filtered = effects.loc[keep].copy()

    rows: list[dict[str, Any]] = []
    for key, group in filtered.groupby(list(ABSOLUTE_KEY), sort=True):
        geometry, level, model, horizon, site = key
        expected_seeds = RANDOM_SEEDS if geometry == "random" else REGION_SEEDS
        seeds = tuple(sorted(group["seed"].astype(int).tolist()))
        if seeds != expected_seeds:
            raise AuthorityError(
                f"absolute station cell {key} has seeds {seeds}, expected {expected_seeds}"
            )
        n_values = tuple(sorted(group["n"].astype(int).unique().tolist()))
        if len(n_values) != 1:
            raise AuthorityError(
                f"absolute station cell {key} has unequal key counts across seeds: {n_values}"
            )
        # Seed-first: every row entering these means is already a station x seed RMSE.
        rows.append(
            {
                "authority_schema": AUTHORITY_SCHEMA,
                "authority_status": AUTHORITY_STATUS,
                "protocol_sha256": protocol_sha256,
                "geometry": geometry,
                "level": level,
                "model": model,
                "horizon": int(horizon),
                "site_id": site,
                "n_keys": n_values[0],
                "seed_count": len(expected_seeds),
                "seed_ids": ",".join(str(seed) for seed in expected_seeds),
                "rmse": float(group["rmse"].mean()),
                "rmse_damped": float(group["rmse_damped"].mean()),
                "reportable_common": True,
                "aggregation_order": "station_then_seed_mean",
            }
        )
    absolute = pd.DataFrame(rows)
    if absolute.empty:
        raise AuthorityError("no absolute station rows were produced")
    if absolute.duplicated(list(ABSOLUTE_KEY)).any():
        raise AuthorityError("absolute station output has duplicate cells")
    expected_count = sum(len(sites) for sites in common.values()) * len(GEOMETRIES) * len(
        LEVELS
    ) * len(MODELS)
    if len(absolute) != expected_count:
        raise AuthorityError(
            f"absolute station row count is {len(absolute)}, expected {expected_count}"
        )
    return absolute.sort_values(list(ABSOLUTE_KEY), kind="mergesort").reset_index(drop=True), common


def _paired_difference(
    absolute: pd.DataFrame,
    *,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    join_columns: list[str],
    contrast_family: str,
    contrast_id: str,
    definition: str,
    conditioning_geometry: str = "",
    conditioning_level: str = "",
) -> pd.DataFrame:
    left = absolute.copy()
    right = absolute.copy()
    for column, value in candidate.items():
        left = left.loc[left[column] == value]
    for column, value in reference.items():
        right = right.loc[right[column] == value]
    left = left.loc[:, join_columns + ["rmse", "n_keys", "protocol_sha256"]]
    right = right.loc[:, join_columns + ["rmse", "n_keys", "protocol_sha256"]]
    paired = left.merge(
        right,
        on=join_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_candidate", "_reference"),
        sort=True,
    )
    if paired.empty or not paired["_merge"].eq("both").all():
        left_only = int((paired["_merge"] == "left_only").sum())
        right_only = int((paired["_merge"] == "right_only").sum())
        raise AuthorityError(
            f"contrast {contrast_id} lacks paired station rows: "
            f"{left_only} candidate-only, {right_only} reference-only"
        )
    if not paired["protocol_sha256_candidate"].equals(
        paired["protocol_sha256_reference"]
    ):
        raise AuthorityError(f"contrast {contrast_id} crosses protocol bindings")
    result = paired.loc[:, join_columns].copy()
    result["authority_schema"] = AUTHORITY_SCHEMA
    result["authority_status"] = AUTHORITY_STATUS
    result["protocol_sha256"] = paired["protocol_sha256_candidate"]
    result["contrast_family"] = contrast_family
    result["contrast_id"] = contrast_id
    result["definition"] = definition
    result["conditioning_geometry"] = conditioning_geometry
    result["conditioning_level"] = conditioning_level
    result["candidate_rmse"] = paired["rmse_candidate"].to_numpy(float)
    result["reference_rmse"] = paired["rmse_reference"].to_numpy(float)
    result["delta_rmse"] = result["candidate_rmse"] - result["reference_rmse"]
    result["n_common_keys"] = np.minimum(
        paired["n_keys_candidate"].to_numpy(int),
        paired["n_keys_reference"].to_numpy(int),
    )
    result["negative_delta_favors"] = "candidate"
    return result


def build_paired_contrasts(absolute: pd.DataFrame) -> pd.DataFrame:
    """Build all predeclared per-station level, geometry, and L x G contrasts."""
    _require_columns(
        absolute,
        {*ABSOLUTE_KEY, "rmse", "n_keys", "protocol_sha256"},
        label="absolute station rows",
    )
    pieces: list[pd.DataFrame] = []
    level_pairs = (("L1", "L0"), ("L2", "L1"), ("L2", "L0"))
    for geometry in GEOMETRIES:
        for candidate_level, reference_level in level_pairs:
            pieces.append(
                _paired_difference(
                    absolute,
                    candidate={"geometry": geometry, "level": candidate_level},
                    reference={"geometry": geometry, "level": reference_level},
                    join_columns=["model", "horizon", "site_id"],
                    contrast_family="level",
                    contrast_id=f"{candidate_level}-{reference_level}",
                    definition=f"RMSE({candidate_level}) - RMSE({reference_level})",
                    conditioning_geometry=geometry,
                )
            )
    for level in LEVELS:
        pieces.append(
            _paired_difference(
                absolute,
                candidate={"geometry": "region", "level": level},
                reference={"geometry": "random", "level": level},
                join_columns=["model", "horizon", "site_id"],
                contrast_family="geometry",
                contrast_id=f"G@{level}",
                definition=f"RMSE(region,{level}) - RMSE(random,{level})",
                conditioning_level=level,
            )
        )

    geometry_rows = pd.concat(
        [piece for piece in pieces if piece["contrast_family"].eq("geometry").all()],
        ignore_index=True,
    )
    l2 = geometry_rows.loc[
        geometry_rows["contrast_id"] == "G@L2",
        [
            "model",
            "horizon",
            "site_id",
            "delta_rmse",
            "n_common_keys",
            "protocol_sha256",
        ],
    ]
    l0 = geometry_rows.loc[
        geometry_rows["contrast_id"] == "G@L0",
        [
            "model",
            "horizon",
            "site_id",
            "delta_rmse",
            "n_common_keys",
            "protocol_sha256",
        ],
    ]
    interaction = l2.merge(
        l0,
        on=["model", "horizon", "site_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_candidate", "_reference"),
        sort=True,
    )
    if interaction.empty or not interaction["_merge"].eq("both").all():
        raise AuthorityError("L x G interaction lacks a common paired station set")
    if not interaction["protocol_sha256_candidate"].equals(
        interaction["protocol_sha256_reference"]
    ):
        raise AuthorityError("L x G interaction crosses protocol bindings")
    interaction_rows = interaction.loc[:, ["model", "horizon", "site_id"]].copy()
    interaction_rows["authority_schema"] = AUTHORITY_SCHEMA
    interaction_rows["authority_status"] = AUTHORITY_STATUS
    interaction_rows["protocol_sha256"] = interaction["protocol_sha256_candidate"]
    interaction_rows["contrast_family"] = "interaction"
    interaction_rows["contrast_id"] = "LxG"
    interaction_rows["definition"] = "[G@L2] - [G@L0]"
    interaction_rows["conditioning_geometry"] = "region-random"
    interaction_rows["conditioning_level"] = "L2-L0"
    interaction_rows["candidate_rmse"] = interaction["delta_rmse_candidate"]
    interaction_rows["reference_rmse"] = interaction["delta_rmse_reference"]
    interaction_rows["delta_rmse"] = (
        interaction_rows["candidate_rmse"] - interaction_rows["reference_rmse"]
    )
    interaction_rows["n_common_keys"] = np.minimum(
        interaction["n_common_keys_candidate"].to_numpy(int),
        interaction["n_common_keys_reference"].to_numpy(int),
    )
    interaction_rows["negative_delta_favors"] = "smaller geometry penalty at L2"
    pieces.append(interaction_rows)

    contrasts = pd.concat(pieces, ignore_index=True, sort=False)
    identity = [
        "contrast_family",
        "contrast_id",
        "conditioning_geometry",
        "conditioning_level",
        "model",
        "horizon",
        "site_id",
    ]
    if contrasts.duplicated(identity).any():
        raise AuthorityError("paired contrast output has duplicate station contrasts")
    station_counts = (
        contrasts.groupby(identity[:-1], dropna=False)["site_id"]
        .nunique()
        .rename("n_sites")
        .reset_index()
    )
    if (station_counts.groupby("horizon")["n_sites"].nunique() != 1).any():
        raise AuthorityError(
            "paired contrasts do not share the horizon-specific common station set"
        )
    return contrasts.sort_values(identity, kind="mergesort").reset_index(drop=True)


def summarize_paired(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Summarise only station-level paired deltas (never marginal medians)."""
    required = {
        "authority_schema",
        "authority_status",
        "protocol_sha256",
        "contrast_family",
        "contrast_id",
        "definition",
        "conditioning_geometry",
        "conditioning_level",
        "model",
        "horizon",
        "site_id",
        "delta_rmse",
    }
    _require_columns(contrasts, required, label="paired contrasts")
    if contrasts.empty:
        raise AuthorityError("paired contrasts are empty")
    if not np.isfinite(contrasts["delta_rmse"].to_numpy(float)).all():
        raise AuthorityError("paired contrasts contain non-finite deltas")
    group_columns = [
        "authority_schema",
        "authority_status",
        "protocol_sha256",
        "contrast_family",
        "contrast_id",
        "definition",
        "conditioning_geometry",
        "conditioning_level",
        "model",
        "horizon",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in contrasts.groupby(group_columns, sort=True, dropna=False):
        values = group["delta_rmse"].to_numpy(float)
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "median_delta_rmse": float(median),
                "q25_delta_rmse": float(q25),
                "q75_delta_rmse": float(q75),
                "iqr_delta_rmse": float(q75 - q25),
                "win_fraction": float(np.mean(values < 0.0)),
                "n": len(values),
                "summary_source": "paired_station_delta_rows_only",
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    if int(summary["n"].sum()) != len(contrasts):
        raise AuthorityError("paired summary row counts do not cover every contrast row")
    return summary.sort_values(
        ["contrast_family", "contrast_id", "model", "horizon"],
        kind="mergesort",
    ).reset_index(drop=True)


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "worktree_dirty": None if status is None else bool(status),
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_staged_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_parquet(path, index=False)
    _fsync_file(path)
    payload = path.read_bytes()
    return {"name": path.name, "rows": len(frame), "bytes": len(payload), "sha256": _sha256(payload)}


def _write_staged_json(value: Any, path: Path) -> dict[str, Any]:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"name": path.name, "bytes": len(payload), "sha256": _sha256(payload)}


def _atomic_exclusive_publish_directory(source: Path, destination: Path) -> None:
    """Atomically rename a private directory to an absent authority name."""
    source = Path(os.path.abspath(os.fspath(source)))
    destination = Path(os.path.abspath(os.fspath(destination)))
    try:
        source_info = source.lstat()
        source_parent = source.parent.lstat()
        destination_parent = destination.parent.lstat()
    except OSError as exc:
        raise AuthorityError("authority publication endpoints cannot be inspected") from exc
    if (
        not stat.S_ISDIR(source_info.st_mode)
        or stat.S_ISLNK(source_info.st_mode)
        or not stat.S_ISDIR(source_parent.st_mode)
        or stat.S_ISLNK(source_parent.st_mode)
        or not stat.S_ISDIR(destination_parent.st_mode)
        or stat.S_ISLNK(destination_parent.st_mode)
        or source_info.st_dev != destination_parent.st_dev
    ):
        raise AuthorityError("authority publication endpoints are unsafe or cross-filesystem")
    if os.path.lexists(destination):
        raise AuthorityError(f"create-only authority collision: {destination}")

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:  # pragma: no cover
            raise AuthorityError("host lacks atomic exclusive renamex_np")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        return_code = renamex_np(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover
            raise AuthorityError("host lacks atomic exclusive renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        return_code = renameat2(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:  # pragma: no cover
        raise AuthorityError(f"host {sys.platform!r} lacks atomic exclusive rename")
    if return_code != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise AuthorityError(f"create-only authority collision: {destination}")
        if error_number == errno.EXDEV:
            raise AuthorityError("authority staging and output are on different filesystems")
        raise AuthorityError("atomic authority publication failed") from OSError(
            error_number, os.strerror(error_number)
        )

    try:
        committed = destination.lstat()
    except OSError as exc:
        raise AuthorityError("atomic publication lacks its destination") from exc
    if (
        os.path.lexists(source)
        or not stat.S_ISDIR(committed.st_mode)
        or stat.S_ISLNK(committed.st_mode)
        or committed.st_dev != source_info.st_dev
        or committed.st_ino != source_info.st_ino
    ):
        raise AuthorityError("atomic exclusive directory rename postcondition failed")
    _fsync_directory(destination.parent)


def publish_authority(
    absolute: pd.DataFrame,
    contrasts: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    manifest_base: dict[str, Any],
) -> Path:
    """Create one complete authority directory atomically and never overwrite."""
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(destination):
        raise AuthorityError(f"authority output already exists; refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_info = destination.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise AuthorityError("authority output parent must be a non-symlink directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        output_bindings = []
        output_bindings.append(
            _write_staged_parquet(
                absolute, stage / f"{OUTPUT_PREFIX}_station_rows.parquet"
            )
        )
        output_bindings.append(
            _write_staged_parquet(
                contrasts, stage / f"{OUTPUT_PREFIX}_paired_contrasts.parquet"
            )
        )
        output_bindings.append(
            _write_staged_parquet(summary, stage / f"{OUTPUT_PREFIX}_summary.parquet")
        )
        summary_records = json.loads(
            summary.to_json(orient="records", double_precision=15)
        )
        summary_binding = _write_staged_json(
            {
                "authority_schema": AUTHORITY_SCHEMA,
                "authority_status": AUTHORITY_STATUS,
                "summary_source": "paired_station_delta_rows_only",
                "rows": summary_records,
            },
            stage / f"{OUTPUT_PREFIX}_summary.json",
        )
        summary_binding["rows"] = len(summary)
        output_bindings.append(summary_binding)

        manifest = dict(manifest_base)
        manifest["outputs"] = output_bindings
        manifest["publication"] = {
            "mode": "atomic_exclusive_directory_rename",
            "create_only": True,
            "manifest_written_last_in_staging": True,
            "destination": str(destination),
        }
        _write_staged_json(manifest, stage / f"{OUTPUT_PREFIX}_manifest.json")
        _fsync_directory(stage)
        _atomic_exclusive_publish_directory(stage, destination)
    except BaseException:
        if os.path.lexists(stage):
            shutil.rmtree(stage)
        raise
    return destination


def build_authority(
    *,
    effects_path: Path,
    shards_path: Path,
    registry_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> Path:
    """Validate, reconstruct, pair, summarise, and atomically publish."""
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(destination):
        raise AuthorityError(f"authority output already exists; refusing overwrite: {destination}")

    protocol_bytes, protocol_binding = _read_bound_bytes(
        Path(protocol_path), label="protocol"
    )
    protocol_sha256 = _sha256(protocol_bytes)
    effects_raw, effects_binding = _read_bound_parquet(
        Path(effects_path), label="ladder_effects"
    )
    registry_raw, registry_binding = _read_bound_parquet(
        Path(registry_path), label="forecast registry"
    )
    effects = validate_effects_subset(effects_raw)
    registry = validate_forecast_registry(registry_raw)
    inventory = validate_shard_inventory(Path(shards_path))
    reconstructed, key_audit, shard_bindings = audit_shards_and_reconstruct_effects(
        inventory, registry
    )
    assert_effects_match_reconstruction(effects, reconstructed)
    absolute, common = build_absolute_station_rows(
        effects, protocol_sha256=protocol_sha256
    )
    contrasts = build_paired_contrasts(absolute)
    paired_summary = summarize_paired(contrasts)

    manifest = {
        "authority_schema": AUTHORITY_SCHEMA,
        "authority_status": AUTHORITY_STATUS,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "levels": list(LEVELS),
            "geometries": list(GEOMETRIES),
            "models": list(MODELS),
            "horizons": list(HORIZONS),
            "random_seeds": list(RANDOM_SEEDS),
            "region_seeds": list(REGION_SEEDS),
            "folds": list(FOLDS),
            "full_protocol_v2_authority": False,
        },
        "protocol_binding": {
            **protocol_binding,
            "exact_bytes_bound": True,
        },
        "inputs": {
            "ladder_effects": effects_binding,
            "forecast_registry": registry_binding,
            "ladder_shards": {
                "path": str(Path(shards_path).resolve()),
                "count": len(shard_bindings),
                "aggregate_sha256": key_audit["shard_set_sha256"],
                "files": shard_bindings,
            },
        },
        "audits": {
            **key_audit,
            "effects_equal_independent_shard_reconstruction": True,
            "common_reportable_station_rule": (
                f"reportable in every raw L0/L1/L2 x random/region x model x seed "
                f"cell at the horizon; each raw cell n >= {MIN_REPORTABLE_KEYS}"
            ),
            "seed_aggregation_order": "station_then_seed_mean",
            "summary_source": "paired_station_delta_rows_only",
            "negative_delta_win_rule": "delta_rmse < 0",
        },
        "common_reportable_stations": {
            str(horizon): {
                "n": len(sites),
                "site_ids_sha256": _sha256(("\n".join(sites) + "\n").encode("utf-8")),
                "site_ids": list(sites),
            }
            for horizon, sites in common.items()
        },
        "row_counts": {
            "raw_effects": len(effects),
            "absolute_station_rows": len(absolute),
            "paired_contrast_rows": len(contrasts),
            "paired_summary_rows": len(paired_summary),
        },
        "software": {
            "builder": str(Path(__file__).resolve()),
            "git": _git_state(),
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    return publish_authority(
        absolute,
        contrasts,
        paired_summary,
        output_dir=destination,
        manifest_base=manifest,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        destination = build_authority(
            effects_path=args.effects,
            shards_path=args.shards,
            registry_path=args.registry,
            protocol_path=args.protocol,
            output_dir=args.out_dir,
        )
    except AuthorityError as exc:
        print(f"AUTHORITY BUILD REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"published {AUTHORITY_STATUS} authority: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
