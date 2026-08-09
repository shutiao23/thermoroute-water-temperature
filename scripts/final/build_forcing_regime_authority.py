#!/usr/bin/env python3
"""Build the create-only authority for the viewed F0/F3_full v4 normalization.

This program does not train or predict.  It accepts exactly the twelve
canonical tree-model cells (F0/F3_full x two models x h1/h3/h7), reconstructs
all station metrics and paired effects from their key-level shards, verifies
the runner's derived artifacts item by item, and atomically publishes a new
authority directory.  The result is permanently labelled post-outcome
normalization of a completed viewed-domain subset: it is neither prospective
nor confirmatory.
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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"

DEFAULT_SHARDS = FINAL / "forcing_shards_v4"
DEFAULT_EFFECTS = FINAL / "forcing_effects_v4.parquet"
DEFAULT_CONTRASTS = FINAL / "forcing_contrasts_v4.parquet"
DEFAULT_SUMMARY = FINAL / "forcing_summary_v4.json"
DEFAULT_FORECAST_REGISTRY = FINAL / "forecast_keys.parquet"
DEFAULT_PROTOCOL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4.yaml"
DEFAULT_STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_TRAIN_PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
DEFAULT_TEST_PANEL = ROOT / "outputs" / "conventional" / "panel_2021_2023.parquet"
DEFAULT_RUNNER = ROOT / "scripts" / "final" / "run_forcing_ladder_v4.py"
DEFAULT_SOURCE_PATHS = (
    ROOT / "scripts" / "final" / "run_information_ladder.py",
    ROOT / "src" / "thermoroute" / "config.py",
    ROOT / "src" / "thermoroute" / "data.py",
    ROOT / "src" / "thermoroute" / "features.py",
    ROOT / "src" / "thermoroute" / "baselines.py",
    ROOT / "src" / "thermoroute" / "air2stream.py",
)
DEFAULT_OUT_DIR = FINAL / "forcing_regime_v4_authority"

AUTHORITY_SCHEMA = "forcing_regime_authority_v4"
AUTHORITY_STATUS = "POST_OUTCOME_NORMALIZATION_ONLY"
SCOPE_STATUS = "COMPLETED_VIEWED_DOMAIN_SUBSET"
OUTPUT_PREFIX = "forcing_regime_v4"
RUNNER_SCHEMA = "thermoroute.forcing-ladder.v4"
EXPECTED_PROTOCOL_ID = "thermoroute_wrr_information_regimes_v4"
EXPECTED_PROTOCOL_VERSION = 4
EXPECTED_PROTOCOL_STATUS = "DRAFT_NOT_SEALED"
MIN_REPORTABLE_KEYS = 100
TARGET_ABSOLUTE_TOLERANCE = 2e-6
TARGET_RELATIVE_TOLERANCE = 0.0

ARMS = ("F0", "F3_full")
MODELS = ("LightGBM", "ResidualLightGBM")
HORIZONS = (1, 3, 7)
META_VARS = ("TEMP", "PRCP", "DH", "RHMEAN", "WDSP")
SUBSTITUTION_COLUMNS = tuple(f"forcing_substitutions_{variable}" for variable in META_VARS)
MODEL_STATUS = "frozen_tree_model"

KEY_LEVEL_COLUMNS = (
    "key_id",
    "arm",
    "protocol_arm",
    "model",
    "site_id",
    "horizon",
    "issue_date",
    "target_date",
    "y_true",
    "y_pred",
    "y_damped",
    *SUBSTITUTION_COLUMNS,
    "forcing_substitution_count",
)
EFFECT_COLUMNS = (
    "arm",
    "protocol_arm",
    "model",
    "model_status",
    "site_id",
    "horizon",
    "n",
    "rmse",
    "rmse_damped",
    *SUBSTITUTION_COLUMNS,
    "forcing_substitution_count",
    "reportable",
)
CONTRAST_COLUMNS = (
    "arm",
    "protocol_arm",
    "reference_arm",
    "reference_protocol_arm",
    "model",
    "contrast_role",
    "model_status",
    "site_id",
    "horizon",
    "n_common_keys",
    "rmse_arm",
    "rmse_reference",
    "delta_rmse",
    "reportable",
)

SHARD_RE = re.compile(r"^(F0|F3_full)_(LightGBM|ResidualLightGBM)_h(1|3|7)\.parquet$")

F3_INPUT_DESCRIPTION = (
    "realized future gridded meteorological estimates at station coordinates; "
    "retrospective oracle inputs, not direct station observations, "
    "catchment-average forcings, or deployable forecasts"
)
F3_FIELD_CAVEATS = {
    "TEMP": "Daymet single-pixel daily-mean air-temperature proxy from tmax/tmin",
    "PRCP": "Daymet single-pixel daily precipitation estimate",
    "RHMEAN": (
        "vapour-pressure/Tetens relative-humidity proxy at the tmax/tmin "
        "midpoint; not a direct daily-mean RH observation"
    ),
    "DH": (
        "legacy name for Daymet daylight-period mean shortwave flux; not a "
        "24-hour mean or daily energy total"
    ),
    "WDSP": "gridMET daily mean wind speed after the frozen CF packing decode",
}


class AuthorityError(RuntimeError):
    """Raised when an authority precondition cannot be proved."""


@dataclass(frozen=True, order=True)
class Cell:
    arm: str
    model: str
    horizon: int

    @property
    def filename(self) -> str:
        return f"{self.arm}_{self.model}_h{self.horizon}.parquet"


def expected_cells() -> tuple[Cell, ...]:
    cells = tuple(
        Cell(arm, model, horizon) for arm in ARMS for model in MODELS for horizon in HORIZONS
    )
    if len(cells) != 12:
        raise AssertionError(f"internal forcing inventory is {len(cells)}, not 12")
    return cells


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bound_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    resolved = path.resolve()
    try:
        info = resolved.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label} cannot be inspected: {resolved}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise AuthorityError(f"{label} is not a regular, non-symlink file: {resolved}")
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise AuthorityError(f"{label} cannot be read: {resolved}") from exc
    if not payload:
        raise AuthorityError(f"{label} is empty: {resolved}")
    return payload, {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _read_bound_parquet(
    path: Path,
    *,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload, binding = _read_bound_bytes(path, label=label)
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise AuthorityError(f"{label} is not readable Parquet: {path}") from exc
    return frame, binding


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_bound_json(path: Path, *, label: str) -> tuple[Any, dict[str, Any]]:
    payload, binding = _read_bound_bytes(path, label=label)
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label} is not strict JSON: {path}") from exc
    return document, binding


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AuthorityError(f"{label} lacks required columns: {missing}")


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    if tuple(frame.columns) != tuple(expected):
        raise AuthorityError(
            f"{label} schema differs: got={list(frame.columns)}, expected={list(expected)}"
        )


def _integer_series(series: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise AuthorityError(f"{label} is not numeric") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise AuthorityError(f"{label} must contain finite integers")
    return numeric.astype("int64")


def _finite_numeric(series: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").astype(float)
    except Exception as exc:
        raise AuthorityError(f"{label} is not numeric") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise AuthorityError(f"{label} contains null or non-finite values")
    return numeric


def _boolean_series(series: pd.Series, *, label: str) -> pd.Series:
    if (
        series.isna().any()
        or not series.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    ):
        raise AuthorityError(f"{label} must contain only booleans")
    return series.astype(bool)


def _normalise_sites(series: pd.Series, *, label: str) -> pd.Series:
    if series.isna().any():
        raise AuthorityError(f"{label} contains null site identifiers")
    sites = series.astype(str).str.strip().str.zfill(8)
    valid = sites.str.fullmatch(r"[0-9]{8,15}")
    if sites.eq("").any() or not valid.all():
        raise AuthorityError(
            f"{label} contains invalid site identifiers: {sites[~valid].head(3).tolist()}"
        )
    return sites


def _normalise_dates(series: pd.Series, *, label: str) -> pd.Series:
    try:
        dates = pd.to_datetime(series, errors="raise", utc=True).dt.tz_localize(None)
    except Exception as exc:
        raise AuthorityError(f"{label} contains invalid dates") from exc
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise AuthorityError(f"{label} must contain non-null midnight dates")
    return dates


def validate_protocol(
    payload: bytes,
    *,
    path: Path,
) -> dict[str, Any]:
    try:
        document = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise AuthorityError(f"protocol is not valid YAML: {path}") from exc
    if not isinstance(document, dict):
        raise AuthorityError("protocol must be a YAML mapping")
    expected_scalars = {
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "version": EXPECTED_PROTOCOL_VERSION,
        "status": EXPECTED_PROTOCOL_STATUS,
        "execution_authorized": False,
    }
    for key, expected in expected_scalars.items():
        if document.get(key) != expected:
            raise AuthorityError(f"protocol {key} is {document.get(key)!r}, expected {expected!r}")
    chronology = document.get("chronology_and_evidence_status")
    if not isinstance(chronology, dict):
        raise AuthorityError("protocol lacks chronology_and_evidence_status")
    viewed_text = " ".join(str(value) for value in chronology.get("already_viewed_domains", []))
    normalization_text = " ".join(
        str(value) for value in chronology.get("normalization_only_domains", [])
    )
    if "F0" not in viewed_text or "F3" not in viewed_text:
        raise AuthorityError("protocol does not identify F0/F3_full as already viewed")
    if "station-paired" not in normalization_text or "F3" not in normalization_text:
        raise AuthorityError("protocol does not declare station-paired F3 normalization")

    key_contract = document.get("common_key_and_reportability_contract")
    if not isinstance(key_contract, dict):
        raise AuthorityError("protocol lacks common-key/reportability contract")
    reportability = key_contract.get("reportability")
    if not isinstance(reportability, dict):
        raise AuthorityError("protocol lacks reportability mapping")
    if reportability.get("minimum_paired_targets_per_station_lead") != MIN_REPORTABLE_KEYS:
        raise AuthorityError("protocol reportability threshold is not exactly 100")
    if (
        reportability.get("station_set_rule")
        != "intersection_across_every_arm_in_the_declared_contrast"
    ):
        raise AuthorityError("protocol common-station rule differs from the authority")
    numeric_contract = str(key_contract.get("target_numeric_reconstruction_contract", ""))
    for token in (
        "2e-6",
        "zero relative tolerance",
        "1.5258789076710855e-6",
        "registry y_true value",
    ):
        if token not in numeric_contract:
            raise AuthorityError(f"protocol target-numeric contract lacks {token!r}")

    estimands = document.get("station_first_estimands")
    if not isinstance(estimands, dict):
        raise AuthorityError("protocol lacks station_first_estimands")
    if estimands.get("forcing_value") != "median_i_of_R_F0_minus_R_Fk":
        raise AuthorityError("protocol forcing-value sign contract differs")
    return document


def validate_shard_inventory(shard_dir: Path) -> dict[Cell, Path]:
    directory = Path(os.path.abspath(os.fspath(shard_dir)))
    try:
        info = directory.lstat()
    except OSError as exc:
        raise AuthorityError(f"shard directory cannot be inspected: {directory}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise AuthorityError(f"shard path is not a non-symlink directory: {directory}")

    inventory: dict[Cell, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        match = SHARD_RE.fullmatch(path.name)
        if match is None:
            raise AuthorityError(f"unexpected entry in exact shard inventory: {path.name}")
        cell = Cell(match.group(1), match.group(2), int(match.group(3)))
        if cell in inventory:
            raise AuthorityError(f"duplicate shard cell {cell}")
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
        inventory[cell] = path

    expected = set(expected_cells())
    actual = set(inventory)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(inventory) != 12:
        raise AuthorityError(
            "shard inventory is not the exact 12 canonical cells: "
            f"{len(missing)} missing, {len(extra)} unexpected; "
            f"missing={missing[:3]}, unexpected={extra[:3]}"
        )
    return inventory


def validate_forecast_registry(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ("key_id", "site_id", "horizon", "issue_date", "target_date", "y_true")
    _require_columns(frame, columns, label="forecast registry")
    registry = frame.loc[:, columns].copy()
    if registry.empty or registry.isna().any(axis=None):
        raise AuthorityError("forecast registry is empty or contains null required values")
    registry["key_id"] = registry["key_id"].astype(str)
    if registry["key_id"].str.strip().eq("").any():
        raise AuthorityError("forecast registry contains empty key_id values")
    registry["site_id"] = _normalise_sites(registry["site_id"], label="forecast registry.site_id")
    registry["horizon"] = _integer_series(registry["horizon"], label="forecast registry.horizon")
    registry["issue_date"] = _normalise_dates(
        registry["issue_date"], label="forecast registry.issue_date"
    )
    registry["target_date"] = _normalise_dates(
        registry["target_date"], label="forecast registry.target_date"
    )
    registry["y_true"] = _finite_numeric(registry["y_true"], label="forecast registry.y_true")
    if set(registry["horizon"].unique()) != set(HORIZONS):
        raise AuthorityError("forecast registry horizons are not exactly 1, 3, and 7")
    if (
        registry.duplicated("key_id").any()
        or registry.duplicated(["site_id", "horizon", "issue_date"]).any()
    ):
        raise AuthorityError("forecast registry contains duplicate identities")
    expected_target = registry["issue_date"] + pd.to_timedelta(registry["horizon"], unit="D")
    if not registry["target_date"].equals(expected_target.rename("target_date")):
        raise AuthorityError("forecast registry target_date != issue_date + horizon")
    return registry.sort_values(["horizon", "site_id", "issue_date"], kind="mergesort").reset_index(
        drop=True
    )


def validate_station_registry(frame: pd.DataFrame) -> tuple[str, ...]:
    _require_columns(frame, ["site_no"], label="station registry")
    sites = _normalise_sites(frame["site_no"], label="station registry.site_no")
    if sites.duplicated().any():
        raise AuthorityError("station registry contains duplicate site_no values")
    return tuple(sorted(sites.tolist()))


def validate_shard_frame(frame: pd.DataFrame, *, cell: Cell, label: str) -> pd.DataFrame:
    _require_exact_columns(frame, KEY_LEVEL_COLUMNS, label=label)
    if frame.empty:
        raise AuthorityError(f"{label} is empty")
    shard = frame.copy()
    shard["key_id"] = shard["key_id"].astype(str)
    if shard["key_id"].str.strip().eq("").any():
        raise AuthorityError(f"{label} contains empty key_id values")
    shard["site_id"] = _normalise_sites(shard["site_id"], label=f"{label}.site_id")
    shard["horizon"] = _integer_series(shard["horizon"], label=f"{label}.horizon")
    shard["issue_date"] = _normalise_dates(shard["issue_date"], label=f"{label}.issue_date")
    shard["target_date"] = _normalise_dates(shard["target_date"], label=f"{label}.target_date")
    for column in ("y_true", "y_pred", "y_damped"):
        shard[column] = _finite_numeric(shard[column], label=f"{label}.{column}")
    if (
        shard.duplicated("key_id").any()
        or shard.duplicated(["site_id", "horizon", "issue_date"]).any()
    ):
        raise AuthorityError(f"{label} contains duplicate key identities")
    identity = set(
        shard[["arm", "protocol_arm", "model", "horizon"]].itertuples(index=False, name=None)
    )
    expected_identity = {(cell.arm, cell.arm, cell.model, cell.horizon)}
    if identity != expected_identity:
        raise AuthorityError(f"{label} arm/protocol_arm/model/horizon identity differs: {identity}")
    expected_target = shard["issue_date"] + pd.to_timedelta(shard["horizon"], unit="D")
    if not shard["target_date"].equals(expected_target.rename("target_date")):
        raise AuthorityError(f"{label} target_date != issue_date + horizon")

    counts = pd.DataFrame(index=shard.index)
    for column in SUBSTITUTION_COLUMNS:
        counts[column] = _integer_series(shard[column], label=f"{label}.{column}")
    total = _integer_series(
        shard["forcing_substitution_count"],
        label=f"{label}.forcing_substitution_count",
    )
    if (counts.to_numpy(int) < 0).any() or (total.to_numpy(int) < 0).any():
        raise AuthorityError(f"{label} contains negative substitution counts")
    if (counts.to_numpy(int) > cell.horizon).any():
        raise AuthorityError(f"{label} per-variable substitutions exceed horizon")
    if (total.to_numpy(int) > len(META_VARS) * cell.horizon).any():
        raise AuthorityError(f"{label} total substitutions exceed variables x horizon")
    if not np.array_equal(counts.sum(axis=1).to_numpy(int), total.to_numpy(int)):
        raise AuthorityError(f"{label} substitution total differs from per-variable sum")
    if cell.arm == "F0" and total.ne(0).any():
        raise AuthorityError(f"{label} F0 rows report forbidden substitutions")
    shard.loc[:, list(SUBSTITUTION_COLUMNS)] = counts
    shard["forcing_substitution_count"] = total
    return shard


def _registry_join(
    shard: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    cell: Cell,
    label: str,
) -> tuple[pd.DataFrame, float, int]:
    expected = registry.loc[registry["horizon"] == cell.horizon].copy()
    joined = shard.merge(
        expected,
        on="key_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_registry"),
        sort=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int((joined["_merge"] == "right_only").sum())
        extra = int((joined["_merge"] == "left_only").sum())
        raise AuthorityError(
            f"{label} two-sided registry breach: {missing} missing, {extra} outside"
        )
    for column in ("site_id", "horizon", "issue_date", "target_date"):
        if not joined[column].equals(joined[f"{column}_registry"]):
            count = int(joined[column].ne(joined[f"{column}_registry"]).sum())
            raise AuthorityError(f"{label} {column} identity differs from registry in {count} rows")
    difference = np.abs(
        joined["y_true"].to_numpy(float) - joined["y_true_registry"].to_numpy(float)
    )
    maximum = float(difference.max(initial=0.0))
    mismatches = int(np.count_nonzero(difference))
    if not np.allclose(
        joined["y_true"].to_numpy(float),
        joined["y_true_registry"].to_numpy(float),
        atol=TARGET_ABSOLUTE_TOLERANCE,
        rtol=TARGET_RELATIVE_TOLERANCE,
        equal_nan=False,
    ):
        raise AuthorityError(
            f"{label} y_true differs beyond absolute tolerance "
            f"{TARGET_ABSOLUTE_TOLERANCE:g}; max={maximum:.17g}"
        )
    return joined, maximum, mismatches


def audit_shards(
    inventory: Mapping[Cell, Path],
    registry: pd.DataFrame,
    station_registry_sites: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    bindings: list[dict[str, Any]] = []
    key_sets: dict[Cell, frozenset[str]] = {}
    station_sets: dict[Cell, tuple[str, ...]] = {}
    maximum_target_difference = 0.0
    nonexact_target_values = 0

    known_sites = set(station_registry_sites)
    for cell in expected_cells():
        path = inventory[cell]
        raw, binding = _read_bound_parquet(path, label=f"shard {path.name}")
        shard = validate_shard_frame(raw, cell=cell, label=path.name)
        joined, maximum, mismatch_count = _registry_join(
            shard,
            registry,
            cell=cell,
            label=path.name,
        )
        maximum_target_difference = max(maximum_target_difference, maximum)
        nonexact_target_values += mismatch_count
        sites = tuple(sorted(shard["site_id"].unique().tolist()))
        outside_station_registry = sorted(set(sites) - known_sites)
        if outside_station_registry:
            raise AuthorityError(
                f"{path.name} contains sites outside station registry: "
                f"{outside_station_registry[:3]}"
            )
        key_sets[cell] = frozenset(shard["key_id"].tolist())
        station_sets[cell] = sites

        registry_targets = joined.set_index("key_id")["y_true_registry"]
        canonical = shard.copy()
        canonical["y_true"] = canonical["key_id"].map(registry_targets).to_numpy(float)
        canonical = canonical.sort_values(["site_id", "issue_date"], kind="mergesort").reset_index(
            drop=True
        )
        predictions.append(canonical)
        binding.update(
            {
                "name": path.name,
                "cell": {
                    "arm": cell.arm,
                    "protocol_arm": cell.arm,
                    "model": cell.model,
                    "horizon": cell.horizon,
                },
                "rows": len(canonical),
                "key_ids_sha256": _sha256(
                    ("\n".join(sorted(canonical["key_id"].tolist())) + "\n").encode()
                ),
            }
        )
        bindings.append(binding)

    first_station_set = station_sets[expected_cells()[0]]
    for cell, sites in station_sets.items():
        if sites != first_station_set:
            raise AuthorityError(f"raw station set differs in cell {cell}")
    for model in MODELS:
        for horizon in HORIZONS:
            f0 = key_sets[Cell("F0", model, horizon)]
            f3 = key_sets[Cell("F3_full", model, horizon)]
            if f0 != f3:
                raise AuthorityError(f"per-arm keys differ for {model}/h{horizon}")
    for horizon in HORIZONS:
        reference_keys = key_sets[Cell("F0", MODELS[0], horizon)]
        for arm in ARMS:
            for model in MODELS:
                if key_sets[Cell(arm, model, horizon)] != reference_keys:
                    raise AuthorityError(
                        f"per arm/model/h key equality failed for {arm}/{model}/h{horizon}"
                    )

    bindings.sort(key=lambda item: item["name"])
    digest_payload = "".join(
        f"{item['name']}\0{item['sha256']}\0{item['bytes']}\n" for item in bindings
    ).encode()
    combined = pd.concat(predictions, ignore_index=True)
    audit = {
        "exact_canonical_cell_count": len(bindings),
        "two_sided_registry_key_equality": True,
        "identity_fields_compared_exactly": [
            "key_id",
            "site_id",
            "horizon",
            "issue_date",
            "target_date",
        ],
        "target_numeric_field": "y_true",
        "target_absolute_tolerance": TARGET_ABSOLUTE_TOLERANCE,
        "target_relative_tolerance": TARGET_RELATIVE_TOLERANCE,
        "maximum_observed_target_absolute_difference": maximum_target_difference,
        "nonexact_target_value_comparisons": nonexact_target_values,
        "metric_target_source": "forecast_key_registry",
        "per_arm_model_h_key_equality": True,
        "raw_common_station_set": True,
        "raw_station_count": len(first_station_set),
        "raw_station_ids": list(first_station_set),
        "shard_set_sha256": _sha256(digest_payload),
    }
    return combined, audit, bindings


def station_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, model, site, horizon), group in predictions.groupby(
        ["arm", "model", "site_id", "horizon"], sort=True
    ):
        error = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        damped_error = group["y_damped"].to_numpy(float) - group["y_true"].to_numpy(float)
        row: dict[str, Any] = {
            "arm": arm,
            "protocol_arm": arm,
            "model": model,
            "model_status": MODEL_STATUS,
            "site_id": site,
            "horizon": int(horizon),
            "n": len(group),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "rmse_damped": float(np.sqrt(np.mean(np.square(damped_error)))),
        }
        for column in SUBSTITUTION_COLUMNS:
            row[column] = int(group[column].sum())
        row["forcing_substitution_count"] = int(group["forcing_substitution_count"].sum())
        row["reportable"] = bool(len(group) >= MIN_REPORTABLE_KEYS)
        rows.append(row)
    result = pd.DataFrame(rows, columns=EFFECT_COLUMNS)
    return result.sort_values(["arm", "model", "site_id", "horizon"], kind="mergesort").reset_index(
        drop=True
    )


def paired_contrasts_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for horizon in HORIZONS:
            model_rows = predictions.loc[
                (predictions["model"] == model) & (predictions["horizon"] == horizon)
            ]
            reference = model_rows.loc[
                model_rows["arm"] == "F0",
                [
                    "key_id",
                    "site_id",
                    "y_true",
                    "y_pred",
                ],
            ].rename(columns={"y_pred": "y_pred_reference"})
            candidate = model_rows.loc[
                model_rows["arm"] == "F3_full",
                ["key_id", "site_id", "y_true", "y_pred"],
            ].rename(columns={"y_pred": "y_pred_arm"})
            paired = candidate.merge(
                reference,
                on=["key_id", "site_id", "y_true"],
                how="outer",
                validate="one_to_one",
                indicator=True,
                sort=True,
            )
            if paired.empty or not paired["_merge"].eq("both").all():
                raise AuthorityError(f"paired keys differ for {model}/h{horizon}")
            for site, group in paired.groupby("site_id", sort=True):
                arm_error = group["y_pred_arm"].to_numpy(float) - group["y_true"].to_numpy(float)
                reference_error = group["y_pred_reference"].to_numpy(float) - group[
                    "y_true"
                ].to_numpy(float)
                rmse_arm = float(np.sqrt(np.mean(np.square(arm_error))))
                rmse_reference = float(np.sqrt(np.mean(np.square(reference_error))))
                rows.append(
                    {
                        "arm": "F3_full",
                        "protocol_arm": "F3_full",
                        "reference_arm": "F0",
                        "reference_protocol_arm": "F0",
                        "model": model,
                        "contrast_role": "primary_tree_F3_full_minus_F0",
                        "model_status": MODEL_STATUS,
                        "site_id": site,
                        "horizon": horizon,
                        "n_common_keys": len(group),
                        "rmse_arm": rmse_arm,
                        "rmse_reference": rmse_reference,
                        "delta_rmse": rmse_arm - rmse_reference,
                        "reportable": bool(len(group) >= MIN_REPORTABLE_KEYS),
                    }
                )
    result = pd.DataFrame(rows, columns=CONTRAST_COLUMNS)
    return result.sort_values(["arm", "model", "site_id", "horizon"], kind="mergesort").reset_index(
        drop=True
    )


def validate_runner_effects(frame: pd.DataFrame) -> pd.DataFrame:
    _require_exact_columns(frame, EFFECT_COLUMNS, label="runner effects")
    effects = frame.copy()
    effects["site_id"] = _normalise_sites(effects["site_id"], label="runner effects.site_id")
    effects["horizon"] = _integer_series(effects["horizon"], label="runner effects.horizon")
    effects["n"] = _integer_series(effects["n"], label="runner effects.n")
    for column in ("rmse", "rmse_damped"):
        effects[column] = _finite_numeric(effects[column], label=f"runner effects.{column}")
    for column in (*SUBSTITUTION_COLUMNS, "forcing_substitution_count"):
        effects[column] = _integer_series(effects[column], label=f"runner effects.{column}")
    effects["reportable"] = _boolean_series(
        effects["reportable"], label="runner effects.reportable"
    )
    if effects.duplicated(["arm", "model", "site_id", "horizon"]).any():
        raise AuthorityError("runner effects contains duplicate station cells")
    if set(effects["arm"]) != set(ARMS):
        raise AuthorityError("runner effects arms are not exactly F0/F3_full")
    if set(effects["model"]) != set(MODELS):
        raise AuthorityError("runner effects models are not exactly the two tree models")
    if set(effects["horizon"]) != set(HORIZONS):
        raise AuthorityError("runner effects horizons are not exactly 1/3/7")
    if not effects["protocol_arm"].equals(effects["arm"].rename("protocol_arm")):
        raise AuthorityError("runner effects protocol_arm differs from canonical arm")
    if not effects["model_status"].eq(MODEL_STATUS).all():
        raise AuthorityError("runner effects model_status differs")
    if not effects["reportable"].equals(effects["n"].ge(MIN_REPORTABLE_KEYS).rename("reportable")):
        raise AuthorityError("runner effects reportable flag differs from n >= 100")
    return effects.sort_values(
        ["arm", "model", "site_id", "horizon"], kind="mergesort"
    ).reset_index(drop=True)


def validate_runner_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    _require_exact_columns(frame, CONTRAST_COLUMNS, label="runner contrasts")
    contrasts = frame.copy()
    contrasts["site_id"] = _normalise_sites(contrasts["site_id"], label="runner contrasts.site_id")
    contrasts["horizon"] = _integer_series(contrasts["horizon"], label="runner contrasts.horizon")
    contrasts["n_common_keys"] = _integer_series(
        contrasts["n_common_keys"], label="runner contrasts.n_common_keys"
    )
    for column in ("rmse_arm", "rmse_reference", "delta_rmse"):
        contrasts[column] = _finite_numeric(contrasts[column], label=f"runner contrasts.{column}")
    contrasts["reportable"] = _boolean_series(
        contrasts["reportable"], label="runner contrasts.reportable"
    )
    if contrasts.duplicated(["model", "site_id", "horizon"]).any():
        raise AuthorityError("runner contrasts contains duplicate station cells")
    expected_constants = {
        "arm": "F3_full",
        "protocol_arm": "F3_full",
        "reference_arm": "F0",
        "reference_protocol_arm": "F0",
        "contrast_role": "primary_tree_F3_full_minus_F0",
        "model_status": MODEL_STATUS,
    }
    for column, expected in expected_constants.items():
        if not contrasts[column].eq(expected).all():
            raise AuthorityError(f"runner contrasts {column} differs from {expected}")
    if set(contrasts["model"]) != set(MODELS) or set(contrasts["horizon"]) != set(HORIZONS):
        raise AuthorityError("runner contrasts model/horizon inventory differs")
    if not contrasts["reportable"].equals(
        contrasts["n_common_keys"].ge(MIN_REPORTABLE_KEYS).rename("reportable")
    ):
        raise AuthorityError("runner contrasts reportability differs from n >= 100")
    return contrasts.sort_values(
        ["arm", "model", "site_id", "horizon"], kind="mergesort"
    ).reset_index(drop=True)


def _assert_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    float_columns: Sequence[str],
    label: str,
) -> None:
    columns = list(expected.columns)
    joined = actual.loc[:, columns].merge(
        expected.loc[:, columns],
        on=list(key_columns),
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_runner", "_reconstructed"),
        sort=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int((joined["_merge"] == "right_only").sum())
        extra = int((joined["_merge"] == "left_only").sum())
        raise AuthorityError(
            f"{label} key set differs from reconstruction: {missing} missing, {extra} unexpected"
        )
    nonkeys = [column for column in columns if column not in key_columns]
    for column in nonkeys:
        left = joined[f"{column}_runner"]
        right = joined[f"{column}_reconstructed"]
        if column in float_columns:
            if not np.allclose(
                left.to_numpy(float),
                right.to_numpy(float),
                atol=1e-12,
                rtol=1e-12,
                equal_nan=False,
            ):
                difference = np.abs(left.to_numpy(float) - right.to_numpy(float))
                raise AuthorityError(
                    f"{label}.{column} differs; max absolute difference="
                    f"{float(difference.max()):.17g}"
                )
        elif not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            count = int(left.ne(right).sum())
            raise AuthorityError(f"{label}.{column} differs in {count} rows")


def assert_effects_equal(
    runner_effects: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> None:
    _assert_frame_equal(
        runner_effects,
        reconstructed,
        key_columns=("arm", "model", "site_id", "horizon"),
        float_columns=("rmse", "rmse_damped"),
        label="runner effects",
    )


def assert_contrasts_equal(
    runner_contrasts: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> None:
    _assert_frame_equal(
        runner_contrasts,
        reconstructed,
        key_columns=("arm", "model", "site_id", "horizon"),
        float_columns=("rmse_arm", "rmse_reference", "delta_rmse"),
        label="runner contrasts",
    )


def common_reportable_sites(effects: pd.DataFrame) -> tuple[str, ...]:
    cell_sets: dict[Cell, tuple[str, ...]] = {}
    for cell in expected_cells():
        group = effects.loc[
            (effects["arm"] == cell.arm)
            & (effects["model"] == cell.model)
            & (effects["horizon"] == cell.horizon)
        ]
        cell_sets[cell] = tuple(
            sorted(group.loc[group["reportable"], "site_id"].astype(str).tolist())
        )
    common = set(cell_sets[expected_cells()[0]])
    for sites in cell_sets.values():
        common &= set(sites)
    if not common:
        raise AuthorityError("no station is reportable in all twelve canonical cells")
    common_tuple = tuple(sorted(common))
    for cell, sites in cell_sets.items():
        if sites != common_tuple:
            raise AuthorityError(
                f"reportable station set for {cell} differs from the common 12-cell set"
            )
    selected = effects.loc[effects["site_id"].isin(common_tuple)]
    if (selected["n"] < MIN_REPORTABLE_KEYS).any():
        raise AuthorityError("common station set contains n < 100 rows")
    expected_rows = len(common_tuple) * len(expected_cells())
    if len(selected) != expected_rows:
        raise AuthorityError(
            f"common station matrix has {len(selected)} rows, expected {expected_rows}"
        )
    return common_tuple


def _runner_absolute_summary(effects: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reportable = effects.loc[effects["reportable"]]
    for (arm, model, horizon), group in reportable.groupby(["arm", "model", "horizon"], sort=True):
        result[f"{model}/h{int(horizon)}/{arm}"] = {
            "protocol_arm": arm,
            "model_status": MODEL_STATUS,
            "n_reportable_stations": len(group),
            "median_station_rmse": float(group["rmse"].median()),
            "median_station_n_keys": int(group["n"].median()),
        }
    return result


def _runner_paired_summary(contrasts: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reportable = contrasts.loc[contrasts["reportable"]]
    for (model, horizon), group in reportable.groupby(["model", "horizon"], sort=True):
        delta = group["delta_rmse"].to_numpy(float)
        key = f"{model}/h{int(horizon)}/F3_full_minus_F0"
        result[key] = {
            "estimand": "median_i[RMSE_i(F3_full)-RMSE_i(F0)]",
            "protocol_arm": "F3_full",
            "reference_protocol_arm": "F0",
            "contrast_role": "primary_tree_F3_full_minus_F0",
            "model_status": MODEL_STATUS,
            "primary_forcing_value": True,
            "is_hybrid": False,
            "validated_hybrid": None,
            "n_reportable_stations": len(group),
            "median_station_delta_rmse": float(np.median(delta)),
            "iqr_station_delta_rmse": [float(value) for value in np.percentile(delta, [25, 75])],
            "station_win_fraction": float(np.mean(delta < 0.0)),
            "median_common_keys": float(group["n_common_keys"].median()),
            "forcing_value_estimand": "median_i[RMSE_i(F0)-RMSE_i(F3_full)]",
            "median_forcing_value_rmse": float(np.median(-delta)),
            "forcing_value_sign_relation": (
                "per-station forcing_value_rmse = -delta_rmse; "
                "positive forcing value means improvement"
            ),
        }
    return result


def _runner_learned_vs_damped_summary(effects: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reportable = effects.loc[effects["reportable"]]
    for (arm, model, horizon), group in reportable.groupby(["arm", "model", "horizon"], sort=True):
        paired = group["rmse"].to_numpy(float) - group["rmse_damped"].to_numpy(float)
        result[f"{model}/h{int(horizon)}/{arm}"] = {
            "estimand": "median_i[RMSE_i(model)-RMSE_i(damped)]",
            "pairing": "within the same reportable station row and exact key set",
            "n_reportable_stations": len(group),
            "median_station_learned_minus_damped_rmse": float(np.median(paired)),
        }
    return result


def _runner_substitution_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (arm, model, horizon), group in predictions.groupby(["arm", "model", "horizon"], sort=True):
        result[f"{model}/h{int(horizon)}/{arm}"] = {
            "protocol_arm": arm,
            "model_status": MODEL_STATUS,
            "n_keys": len(group),
            "keys_with_any_substitution": int(group["forcing_substitution_count"].gt(0).sum()),
            "total_substitutions": int(group["forcing_substitution_count"].sum()),
            "by_variable": {
                variable: int(group[f"forcing_substitutions_{variable}"].sum())
                for variable in META_VARS
            },
        }
    return result


def _runner_p5_summary(
    absolute: Mapping[str, Any],
    learned: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODELS:
        f0_key = f"{model}/h7/F0"
        f3_key = f"{model}/h7/F3_full"
        f3_rmse = float(absolute[f3_key]["median_station_rmse"])
        f3_advantage = float(learned[f3_key]["median_station_learned_minus_damped_rmse"])
        f0_advantage = float(learned[f0_key]["median_station_learned_minus_damped_rmse"])
        expansion = f3_advantage - f0_advantage
        result[model] = {
            "seven_day_F3_full_rmse_range": {
                "registered_range_c": [1.15, 1.25],
                "observed_median_station_rmse": f3_rmse,
                "within_registered_range": bool(1.15 <= f3_rmse <= 1.25),
            },
            "seven_day_learned_advantage_threshold": {
                "registered_threshold_c": -0.30,
                "comparison": ("median_station_learned_minus_damped_rmse <= threshold"),
                "observed_median_station_learned_minus_damped_rmse": f3_advantage,
                "meets_registered_threshold": bool(f3_advantage <= -0.30),
                "F0_median_station_learned_minus_damped_rmse": f0_advantage,
                "F3_full_minus_F0_advantage_change_rmse": expansion,
                "registered_minimum_advantage_expansion_c": 0.10,
                "meets_registered_advantage_expansion": bool(expansion <= -0.10),
            },
            "adjudication_status": "COMPONENT_VALUES_ONLY_NO_AUTOMATIC_VERDICT",
        }
    return result


def reconstruct_runner_summary(
    predictions: pd.DataFrame,
    effects: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    protocol_path: Path,
    protocol_document: Mapping[str, Any],
    protocol_sha256: str,
    formal_threads: int,
) -> dict[str, Any]:
    absolute = _runner_absolute_summary(effects)
    paired = _runner_paired_summary(contrasts)
    learned = _runner_learned_vs_damped_summary(effects)
    p5 = _runner_p5_summary(absolute, learned)
    governance = {
        "analysis_mode": AUTHORITY_STATUS,
        "post_outcome_normalization": True,
        "prospective_or_confirmatory": False,
        "execution_authorized_by_seal": False,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "version": EXPECTED_PROTOCOL_VERSION,
            "status": protocol_document.get("status"),
            "execution_authorized": protocol_document.get("execution_authorized"),
            "sha256": protocol_sha256,
        },
        "seal": None,
        "qualification": (
            "already-viewed 2021-2023 F0/F3_full tree-domain normalization; "
            "not prospective, preregistered, independent, or confirmatory"
        ),
        "execution_resources": {
            "air2stream_worker_budget": 1,
            "THERMOROUTE_FORMAL_THREADS": formal_threads,
        },
    }
    return {
        "schema_version": RUNNER_SCHEMA,
        "analysis_mode": AUTHORITY_STATUS,
        "post_outcome_normalization": True,
        "governance": governance,
        "target_value_binding": {
            "field": "y_true",
            "comparison": "absolute_tolerance_only",
            "absolute_tolerance": TARGET_ABSOLUTE_TOLERANCE,
            "relative_tolerance": TARGET_RELATIVE_TOLERANCE,
            "identity_fields_remain_exact": True,
            "persisted_value_source": "forecast_key_registry",
        },
        "arm_identity": {
            "canonical_cli_arms": ["F0", "F1", "F3_full"],
            "legacy_F3_alias_used": False,
            "legacy_F3_alias_maps_to": "F3_full",
            "F3_temperature_only_implemented": False,
            "interpretation_guard": (
                "Every row's protocol_arm is canonical; F3_full output must "
                "never be interpreted as F3_temperature_only"
            ),
        },
        "execution_resources": {
            "air2stream_worker_budget": 1,
            "constraint": "not above THERMOROUTE_FORMAL_THREADS",
        },
        "F3_full_input_semantics": {
            "description": F3_INPUT_DESCRIPTION,
            "field_caveats": dict(F3_FIELD_CAVEATS),
        },
        "primary_estimand": (
            "tree models: median_i[RMSE_i(F3_full)-RMSE_i(F0)] on identical per-station keys"
        ),
        "diagnostic_estimand": (
            "tree models: median_i[RMSE_i(F1)-RMSE_i(F0)] on identical per-station keys"
        ),
        "secondary_hybrid_estimand": (
            "unofficial unvalidated air2stream: median_i[RMSE_i(F3_full)-RMSE_i(F1)]"
        ),
        "difference_of_marginal_medians_is_not_an_estimand": True,
        "reportability_min_common_keys": MIN_REPORTABLE_KEYS,
        "cells": [
            {
                "arm": cell.arm,
                "protocol_arm": cell.arm,
                "model": cell.model,
                "model_status": MODEL_STATUS,
                "evidence_status": "already_viewed_post_outcome_domain",
                "horizon": cell.horizon,
            }
            for cell in expected_cells()
        ],
        "absolute_station_metrics": absolute,
        "paired_forcing_effects": paired,
        "paired_learned_vs_damped": learned,
        "P5_independent_components": p5,
        "P5_joint_verdict": "NOT_ADJUDICATED_BY_RUNNER",
        "substitution_accounting": _runner_substitution_summary(predictions),
    }


def _formal_threads_from_summary(document: Mapping[str, Any]) -> int:
    governance = document.get("governance")
    if not isinstance(governance, dict):
        raise AuthorityError("runner summary governance is not a mapping")
    resources = governance.get("execution_resources")
    if not isinstance(resources, dict) or set(resources) != {
        "air2stream_worker_budget",
        "THERMOROUTE_FORMAL_THREADS",
    }:
        raise AuthorityError("runner summary governance resources differ")
    if resources.get("air2stream_worker_budget") != 1:
        raise AuthorityError("normalization runner air2stream worker budget is not one")
    threads = resources.get("THERMOROUTE_FORMAL_THREADS")
    if type(threads) is not int or threads < 1:
        raise AuthorityError("THERMOROUTE_FORMAL_THREADS is not a positive integer")
    return threads


def _assert_nested_equal(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AuthorityError(f"{path} is not a mapping")
        if set(actual) != set(expected):
            raise AuthorityError(
                f"{path} keys differ: missing={sorted(set(expected) - set(actual))}, "
                f"unexpected={sorted(set(actual) - set(expected))}"
            )
        for key in expected:
            _assert_nested_equal(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuthorityError(f"{path} list shape differs")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_nested_equal(left, right, path=f"{path}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            raise AuthorityError(f"{path} differs: got={actual!r}, expected={expected!r}")
        return
    if isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise AuthorityError(f"{path} differs: got={actual!r}, expected={expected!r}")
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise AuthorityError(f"{path} is not numeric")
        if not np.isclose(float(actual), expected, atol=1e-12, rtol=1e-12):
            raise AuthorityError(f"{path} differs: got={actual!r}, expected={expected!r}")
        return
    raise TypeError(f"unsupported nested comparison type at {path}: {type(expected)}")


def assert_summary_equal(
    runner_summary: Any,
    predictions: pd.DataFrame,
    effects: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    protocol_path: Path,
    protocol_document: Mapping[str, Any],
    protocol_sha256: str,
) -> None:
    if not isinstance(runner_summary, dict):
        raise AuthorityError("runner summary root is not a mapping")
    formal_threads = _formal_threads_from_summary(runner_summary)
    expected = reconstruct_runner_summary(
        predictions,
        effects,
        contrasts,
        protocol_path=protocol_path,
        protocol_document=protocol_document,
        protocol_sha256=protocol_sha256,
        formal_threads=formal_threads,
    )
    _assert_nested_equal(runner_summary, expected, path="runner_summary")


def build_authority_station_rows(
    effects: pd.DataFrame,
    *,
    common_sites: Sequence[str],
    protocol_sha256: str,
) -> pd.DataFrame:
    rows = effects.loc[effects["site_id"].isin(common_sites)].copy()
    if len(rows) != len(common_sites) * len(expected_cells()):
        raise AuthorityError("authority station matrix is incomplete")
    if not rows["reportable"].all() or (rows["n"] < MIN_REPORTABLE_KEYS).any():
        raise AuthorityError("authority station matrix contains an underpowered row")
    rows.insert(0, "authority_schema", AUTHORITY_SCHEMA)
    rows.insert(1, "authority_status", AUTHORITY_STATUS)
    rows.insert(2, "scope_status", SCOPE_STATUS)
    rows.insert(3, "prospective", False)
    rows.insert(4, "confirmatory", False)
    rows.insert(5, "protocol_sha256", protocol_sha256)
    rows["learned_minus_damped_rmse"] = rows["rmse"].to_numpy(float) - rows["rmse_damped"].to_numpy(
        float
    )
    rows["station_set_rule"] = "common_reportable_across_all_12_cells"
    return rows.sort_values(["arm", "model", "horizon", "site_id"], kind="mergesort").reset_index(
        drop=True
    )


def build_authority_contrast_rows(
    contrasts: pd.DataFrame,
    *,
    common_sites: Sequence[str],
    protocol_sha256: str,
) -> pd.DataFrame:
    rows = contrasts.loc[contrasts["site_id"].isin(common_sites)].copy()
    expected_count = len(common_sites) * len(MODELS) * len(HORIZONS)
    if len(rows) != expected_count:
        raise AuthorityError(
            f"authority contrast matrix has {len(rows)} rows, expected {expected_count}"
        )
    if not rows["reportable"].all() or (rows["n_common_keys"] < MIN_REPORTABLE_KEYS).any():
        raise AuthorityError("authority contrast matrix contains an underpowered row")
    rows.insert(0, "authority_schema", AUTHORITY_SCHEMA)
    rows.insert(1, "authority_status", AUTHORITY_STATUS)
    rows.insert(2, "scope_status", SCOPE_STATUS)
    rows.insert(3, "prospective", False)
    rows.insert(4, "confirmatory", False)
    rows.insert(5, "protocol_sha256", protocol_sha256)
    rows["forcing_value_rmse"] = -rows["delta_rmse"].to_numpy(float)
    rows["delta_estimand"] = "RMSE_i(F3_full)-RMSE_i(F0)"
    rows["forcing_value_estimand"] = "RMSE_i(F0)-RMSE_i(F3_full)"
    rows["forcing_value_sign_rule"] = "forcing_value_rmse = -delta_rmse"
    return rows.sort_values(["model", "horizon", "site_id"], kind="mergesort").reset_index(
        drop=True
    )


def build_authority_summary(
    station_rows: pd.DataFrame,
    contrast_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, horizon), group in contrast_rows.groupby(["model", "horizon"], sort=True):
        delta = group["delta_rmse"].to_numpy(float)
        forcing_value = group["forcing_value_rmse"].to_numpy(float)
        if not np.array_equal(forcing_value, -delta):
            raise AuthorityError(f"forcing-value sign relation failed for {model}/h{horizon}")

        effects = station_rows.loc[
            (station_rows["model"] == model) & (station_rows["horizon"] == horizon),
            [
                "arm",
                "site_id",
                "rmse",
                "rmse_damped",
                "learned_minus_damped_rmse",
            ],
        ]
        f0 = (
            effects.loc[effects["arm"] == "F0"]
            .drop(columns="arm")
            .rename(
                columns={
                    "rmse": "rmse_F0",
                    "rmse_damped": "rmse_damped_F0",
                    "learned_minus_damped_rmse": "learned_minus_damped_F0",
                }
            )
        )
        f3 = (
            effects.loc[effects["arm"] == "F3_full"]
            .drop(columns="arm")
            .rename(
                columns={
                    "rmse": "rmse_F3_full",
                    "rmse_damped": "rmse_damped_F3_full",
                    "learned_minus_damped_rmse": "learned_minus_damped_F3_full",
                }
            )
        )
        paired_anchor = f3.merge(
            f0,
            on="site_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
            sort=True,
        )
        if not paired_anchor["_merge"].eq("both").all() or len(paired_anchor) != len(group):
            raise AuthorityError(f"learned/damped station pairing failed for {model}/h{horizon}")
        f0_advantage = paired_anchor["learned_minus_damped_F0"].to_numpy(float)
        f3_advantage = paired_anchor["learned_minus_damped_F3_full"].to_numpy(float)
        double_difference = f3_advantage - f0_advantage
        q25, median_delta, q75 = np.quantile(delta, [0.25, 0.5, 0.75])
        median_forcing_value = float(np.median(forcing_value))
        if not np.isclose(median_forcing_value, -float(median_delta), atol=1e-12, rtol=0):
            raise AuthorityError(
                f"median forcing-value sign relation failed for {model}/h{horizon}"
            )
        f3_station_median_rmse = float(np.median(paired_anchor["rmse_F3_full"]))
        f3_median_advantage = float(np.median(f3_advantage))
        f0_median_advantage = float(np.median(f0_advantage))
        advantage_expansion = float(np.median(double_difference))
        p5_applicable = int(horizon) == 7
        rows.append(
            {
                "authority_schema": AUTHORITY_SCHEMA,
                "authority_status": AUTHORITY_STATUS,
                "scope_status": SCOPE_STATUS,
                "prospective": False,
                "confirmatory": False,
                "model": model,
                "horizon": int(horizon),
                "n_common_reportable_stations": len(group),
                "median_common_keys": float(group["n_common_keys"].median()),
                "delta_estimand": "median_i[RMSE_i(F3_full)-RMSE_i(F0)]",
                "median_station_delta_rmse": float(median_delta),
                "q25_station_delta_rmse": float(q25),
                "q75_station_delta_rmse": float(q75),
                "station_win_fraction_delta_lt_zero": float(np.mean(delta < 0.0)),
                "forcing_value_estimand": "median_i[RMSE_i(F0)-RMSE_i(F3_full)]",
                "median_forcing_value_rmse": median_forcing_value,
                "forcing_value_sign_rule": (
                    "station forcing_value_rmse = -delta_rmse; positive means improvement"
                ),
                "F0_median_station_learned_minus_damped_rmse": f0_median_advantage,
                "F3_full_median_station_learned_minus_damped_rmse": f3_median_advantage,
                "median_station_learned_advantage_double_difference_rmse": (advantage_expansion),
                "learned_advantage_pairing_rule": (
                    "within station and arm first; F3_full minus F0 second"
                ),
                "F3_full_median_station_rmse": f3_station_median_rmse,
                "P5_applicable": p5_applicable,
                "P5_rmse_range_lower_c": 1.15 if p5_applicable else np.nan,
                "P5_rmse_range_upper_c": 1.25 if p5_applicable else np.nan,
                "P5_rmse_range_component_status": (
                    "MEETS_REGISTERED_RANGE"
                    if p5_applicable and 1.15 <= f3_station_median_rmse <= 1.25
                    else (
                        "DOES_NOT_MEET_REGISTERED_RANGE"
                        if p5_applicable
                        else "NOT_APPLICABLE_NON_7D"
                    )
                ),
                "P5_learned_advantage_threshold_c": -0.30 if p5_applicable else np.nan,
                "P5_learned_advantage_component_status": (
                    "MEETS_REGISTERED_THRESHOLD"
                    if p5_applicable and f3_median_advantage <= -0.30
                    else (
                        "DOES_NOT_MEET_REGISTERED_THRESHOLD"
                        if p5_applicable
                        else "NOT_APPLICABLE_NON_7D"
                    )
                ),
                "P5_registered_minimum_advantage_expansion_c": (0.10 if p5_applicable else np.nan),
                "P5_advantage_expansion_component_status": (
                    "MEETS_REGISTERED_EXPANSION"
                    if p5_applicable and advantage_expansion <= -0.10
                    else (
                        "DOES_NOT_MEET_REGISTERED_EXPANSION"
                        if p5_applicable
                        else "NOT_APPLICABLE_NON_7D"
                    )
                ),
                "P5_joint_verdict": "NOT_ADJUDICATED_BY_AUTHORITY_BUILDER",
            }
        )
    summary = pd.DataFrame(rows)
    if len(summary) != len(MODELS) * len(HORIZONS):
        raise AuthorityError("authority summary does not contain exactly six cells")
    if (
        summary.astype(str)
        .apply(lambda column: column.str.contains("CONFIRMED", regex=False).any())
        .any()
    ):
        raise AuthorityError("authority builder must never auto-emit CONFIRMED")
    return summary.sort_values(["model", "horizon"], kind="mergesort").reset_index(drop=True)


def _git_state(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()

    def run(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AuthorityError(
                f"git state cannot be bound at {root}: git {' '.join(arguments)}"
            ) from exc
        return completed.stdout

    commit = run("rev-parse", "HEAD").decode("utf-8").strip()
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = run("diff", "--binary", "HEAD")
    return {
        "repository_root": str(root),
        "commit": commit,
        "worktree_dirty": bool(status),
        "status_porcelain_v1": status.decode("utf-8").splitlines(),
        "status_porcelain_v1_sha256": _sha256(status),
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff_sha256": _sha256(tracked_diff),
    }


def _bind_sources(
    runner_path: Path,
    source_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    candidates = [Path(__file__).resolve(), runner_path, *source_paths]
    seen: set[Path] = set()
    bindings: list[dict[str, Any]] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _payload, binding = _read_bound_bytes(resolved, label=f"source {resolved.name}")
        binding["role"] = (
            "authority_builder"
            if resolved == Path(__file__).resolve()
            else "forcing_runner"
            if resolved == runner_path.resolve()
            else "runner_dependency"
        )
        bindings.append(binding)
    return bindings


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
    return {
        "name": path.name,
        "rows": len(frame),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _write_staged_json(value: Any, path: Path) -> dict[str, Any]:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"name": path.name, "bytes": len(payload), "sha256": _sha256(payload)}


def _atomic_exclusive_publish_directory(source: Path, destination: Path) -> None:
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
        return_code = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
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
    station_rows: pd.DataFrame,
    contrast_rows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    manifest_base: dict[str, Any],
) -> Path:
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(destination):
        raise AuthorityError(f"authority output already exists; refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_info = destination.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise AuthorityError("authority output parent must be a non-symlink directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        output_bindings = [
            _write_staged_parquet(
                station_rows,
                stage / f"{OUTPUT_PREFIX}_station_metrics.parquet",
            ),
            _write_staged_parquet(
                contrast_rows,
                stage / f"{OUTPUT_PREFIX}_paired_contrasts.parquet",
            ),
            _write_staged_parquet(
                summary,
                stage / f"{OUTPUT_PREFIX}_summary.parquet",
            ),
        ]
        summary_records = json.loads(summary.to_json(orient="records", double_precision=15))
        summary_document = {
            "authority_schema": AUTHORITY_SCHEMA,
            "authority_status": AUTHORITY_STATUS,
            "scope_status": SCOPE_STATUS,
            "prospective": False,
            "confirmatory": False,
            "completed_viewed_domain_subset": True,
            "estimands": {
                "runner_delta": "median_i[RMSE_i(F3_full)-RMSE_i(F0)]",
                "protocol_forcing_value": "median_i[RMSE_i(F0)-RMSE_i(F3_full)]",
                "sign_relation": "forcing_value_rmse = -delta_rmse station by station",
                "learned_vs_damped": "median_i[RMSE_i(model)-RMSE_i(damped)]",
            },
            "P5_adjudication_rule": (
                "seven-day RMSE range and learned-advantage threshold are separate; "
                "this builder emits no automatic joint verdict"
            ),
            "rows": summary_records,
        }
        summary_binding = _write_staged_json(
            summary_document,
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
            "runner_artifacts_never_overwritten": True,
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


def _assert_safe_destination(output_dir: Path, protected_paths: Sequence[Path]) -> Path:
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(destination):
        raise AuthorityError(f"authority output already exists; refusing overwrite: {destination}")
    for protected in protected_paths:
        resolved = protected.resolve()
        if destination == resolved or destination in resolved.parents:
            raise AuthorityError(
                f"authority destination would contain or replace protected input: {resolved}"
            )
    return destination


def build_authority(
    *,
    shards_path: Path,
    effects_path: Path,
    contrasts_path: Path,
    summary_path: Path,
    forecast_registry_path: Path,
    protocol_path: Path,
    station_registry_path: Path,
    train_panel_path: Path,
    test_panel_path: Path,
    runner_path: Path,
    source_paths: Sequence[Path],
    output_dir: Path,
    repository_root: Path = ROOT,
) -> Path:
    protected = [
        shards_path,
        effects_path,
        contrasts_path,
        summary_path,
        forecast_registry_path,
        protocol_path,
        station_registry_path,
        train_panel_path,
        test_panel_path,
        runner_path,
        *source_paths,
    ]
    destination = _assert_safe_destination(output_dir, protected)

    protocol_bytes, protocol_binding = _read_bound_bytes(protocol_path, label="v4 draft protocol")
    protocol_document = validate_protocol(protocol_bytes, path=protocol_path)
    protocol_sha256 = protocol_binding["sha256"]

    station_bytes, station_binding = _read_bound_bytes(
        station_registry_path, label="station registry"
    )
    try:
        station_frame = pd.read_csv(io.BytesIO(station_bytes))
    except Exception as exc:
        raise AuthorityError("station registry is not readable CSV") from exc
    station_sites = validate_station_registry(station_frame)

    _train_panel, train_binding = _read_bound_parquet(train_panel_path, label="training panel")
    _test_panel, test_binding = _read_bound_parquet(test_panel_path, label="test panel")
    registry_raw, forecast_binding = _read_bound_parquet(
        forecast_registry_path, label="forecast registry"
    )
    registry = validate_forecast_registry(registry_raw)
    missing_station_registry = sorted(set(registry["site_id"]) - set(station_sites))
    if missing_station_registry:
        raise AuthorityError(
            "forecast registry contains sites outside station registry: "
            f"{missing_station_registry[:3]}"
        )

    inventory = validate_shard_inventory(shards_path)
    predictions, shard_audit, shard_bindings = audit_shards(
        inventory,
        registry,
        station_sites,
    )
    reconstructed_effects = station_metrics_from_predictions(predictions)
    reconstructed_contrasts = paired_contrasts_from_predictions(predictions)

    runner_effects_raw, effects_binding = _read_bound_parquet(effects_path, label="runner effects")
    runner_contrasts_raw, contrasts_binding = _read_bound_parquet(
        contrasts_path, label="runner contrasts"
    )
    runner_summary, summary_binding = _read_bound_json(summary_path, label="runner summary")
    runner_effects = validate_runner_effects(runner_effects_raw)
    runner_contrasts = validate_runner_contrasts(runner_contrasts_raw)
    assert_effects_equal(runner_effects, reconstructed_effects)
    assert_contrasts_equal(runner_contrasts, reconstructed_contrasts)
    assert_summary_equal(
        runner_summary,
        predictions,
        reconstructed_effects,
        reconstructed_contrasts,
        protocol_path=protocol_path,
        protocol_document=protocol_document,
        protocol_sha256=protocol_sha256,
    )

    common_sites = common_reportable_sites(reconstructed_effects)
    station_rows = build_authority_station_rows(
        reconstructed_effects,
        common_sites=common_sites,
        protocol_sha256=protocol_sha256,
    )
    contrast_rows = build_authority_contrast_rows(
        reconstructed_contrasts,
        common_sites=common_sites,
        protocol_sha256=protocol_sha256,
    )
    authority_summary = build_authority_summary(station_rows, contrast_rows)
    source_bindings = _bind_sources(runner_path, source_paths)
    git_binding = _git_state(repository_root)

    manifest = {
        "authority_schema": AUTHORITY_SCHEMA,
        "authority_status": AUTHORITY_STATUS,
        "scope_status": SCOPE_STATUS,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "governance": {
            "post_outcome_normalization": True,
            "prospective": False,
            "confirmatory": False,
            "completed_viewed_domain_subset": True,
            "full_v4_protocol_authority": False,
            "automatic_P5_joint_verdict": False,
        },
        "scope": {
            "arms": list(ARMS),
            "protocol_arms": list(ARMS),
            "models": list(MODELS),
            "horizons": list(HORIZONS),
            "exact_cell_count": len(expected_cells()),
            "cells": [
                {
                    "arm": cell.arm,
                    "protocol_arm": cell.arm,
                    "model": cell.model,
                    "horizon": cell.horizon,
                }
                for cell in expected_cells()
            ],
        },
        "bindings": {
            "v4_draft_protocol": {**protocol_binding, "exact_bytes_bound": True},
            "data": {
                "station_registry": station_binding,
                "training_panel": train_binding,
                "test_panel": test_binding,
                "forecast_registry": forecast_binding,
            },
            "runner_derived_artifacts": {
                "effects": effects_binding,
                "contrasts": contrasts_binding,
                "summary": summary_binding,
            },
            "shards": {
                "path": str(Path(shards_path).resolve()),
                "count": len(shard_bindings),
                "aggregate_sha256": shard_audit["shard_set_sha256"],
                "files": shard_bindings,
            },
            "source_code": source_bindings,
            "git_state": git_binding,
        },
        "audits": {
            **shard_audit,
            "protocol_sha256_computed_from_current_exact_bytes": protocol_sha256,
            "protocol_forcing_value_sign_contract": ("median_i[RMSE_i(F0)-RMSE_i(Fk)]"),
            "runner_delta_sign_contract": ("median_i[RMSE_i(Fk)-RMSE_i(F0)]"),
            "sign_relation_verified_station_by_station": True,
            "runner_effects_equal_independent_reconstruction_item_by_item": True,
            "runner_contrasts_equal_independent_reconstruction_item_by_item": True,
            "runner_summary_equal_independent_reconstruction_item_by_item": True,
            "substitution_accounting_verified": True,
            "minimum_common_station_keys": MIN_REPORTABLE_KEYS,
            "common_reportable_station_set_across_all_12_cells": True,
            "common_reportable_station_count": len(common_sites),
            "common_reportable_station_ids": list(common_sites),
            "common_reportable_station_ids_sha256": _sha256(
                ("\n".join(common_sites) + "\n").encode()
            ),
            "station_first_estimand_only": True,
            "difference_of_marginal_medians_is_not_an_estimand": True,
            "learned_vs_damped_paired_within_station": True,
            "P5_range_and_learned_threshold_are_independent": True,
            "P5_joint_verdict": "NOT_ADJUDICATED_BY_AUTHORITY_BUILDER",
        },
        "row_counts": {
            "canonical_prediction_rows": len(predictions),
            "runner_effect_rows": len(runner_effects),
            "runner_contrast_rows": len(runner_contrasts),
            "authority_station_rows": len(station_rows),
            "authority_contrast_rows": len(contrast_rows),
            "authority_summary_rows": len(authority_summary),
        },
        "software": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    return publish_authority(
        station_rows,
        contrast_rows,
        authority_summary,
        output_dir=destination,
        manifest_base=manifest,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--contrasts", type=Path, default=DEFAULT_CONTRASTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--forecast-registry", type=Path, default=DEFAULT_FORECAST_REGISTRY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--station-registry", type=Path, default=DEFAULT_STATION_REGISTRY)
    parser.add_argument("--train-panel", type=Path, default=DEFAULT_TRAIN_PANEL)
    parser.add_argument("--test-panel", type=Path, default=DEFAULT_TEST_PANEL)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[],
        help="additional runner dependency to bind (repeatable)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        destination = build_authority(
            shards_path=args.shards,
            effects_path=args.effects,
            contrasts_path=args.contrasts,
            summary_path=args.summary,
            forecast_registry_path=args.forecast_registry,
            protocol_path=args.protocol,
            station_registry_path=args.station_registry,
            train_panel_path=args.train_panel,
            test_panel_path=args.test_panel,
            runner_path=args.runner,
            source_paths=(*DEFAULT_SOURCE_PATHS, *args.source),
            output_dir=args.out_dir,
        )
    except AuthorityError as exc:
        print(f"AUTHORITY BUILD REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"published {AUTHORITY_STATUS}/{SCOPE_STATUS} authority: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
