#!/usr/bin/env python3
"""Build whole-HUC2 descriptive inference for the viewed F0/F3_full subset.

The only scientific inputs are the forcing-regime v4 authority manifest, its
published station-level paired contrasts, and ``station_registry_v1.csv``.
The builder validates their hash chain and fixed inventory before computing
six model-by-horizon descriptive sensitivity cells.  It does not read model
predictions, refit models, adjudicate a registered claim, adjust a p-value
family, or compare forcing effects with architecture effects.

The source outcomes were visible before this normalization was specified.
Consequently every artifact is permanently labelled post-outcome,
non-prospective, non-confirmatory, and approximate fixed-cohort descriptive.
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
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"
DEFAULT_UPSTREAM_DIR = FINAL / "forcing_regime_v4_authority"
DEFAULT_MANIFEST = DEFAULT_UPSTREAM_DIR / "forcing_regime_v4_manifest.json"
DEFAULT_PAIRED = DEFAULT_UPSTREAM_DIR / "forcing_regime_v4_paired_contrasts.parquet"
DEFAULT_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_OUT_DIR = FINAL / "forcing_regime_inference_v4"

AUTHORITY_SCHEMA = "forcing_regime_inference_authority_v4"
UPSTREAM_SCHEMA = "forcing_regime_authority_v4"
AUTHORITY_STATUS = "POST_OUTCOME_NORMALIZATION_ONLY"
SCOPE_STATUS = "COMPLETED_VIEWED_DOMAIN_SUBSET"
INFERENCE_ROLE = "APPROXIMATE_FIXED_COHORT_DESCRIPTIVE"
OUTPUT_PREFIX = "forcing_regime_inference_v4"

MODELS = ("LightGBM", "ResidualLightGBM")
HORIZONS = (1, 3, 7)
ARMS = ("F0", "F3_full")
EXPECTED_UPSTREAM_CELLS = 12
EXPECTED_ANALYSIS_CELLS = 6
EXPECTED_STATIONS = 116
EXPECTED_PAIRED_ROWS = 696
EXPECTED_HUC2_CLUSTERS = 15
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED_NAMESPACE = "thermoroute-forcing-regime-inference-v4-bootstrap"
SIGN_FLIP_TIE_TOLERANCE = 1e-15

FORCING_VALUE_ESTIMAND = "median_i[RMSE_i(F0)-RMSE_i(F3_full)]"
UPSTREAM_PROTOCOL_FORCING_VALUE_SIGN_CONTRACT = "median_i[RMSE_i(F0)-RMSE_i(Fk)]"
STATION_FORCING_VALUE = "RMSE_i(F0)-RMSE_i(F3_full)"
DELTA_ESTIMAND = "RMSE_i(F3_full)-RMSE_i(F0)"
SIGN_RULE = "forcing_value_rmse = -delta_rmse"

PAIRED_COLUMNS = (
    "authority_schema",
    "authority_status",
    "scope_status",
    "prospective",
    "confirmatory",
    "protocol_sha256",
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
    "forcing_value_rmse",
    "delta_estimand",
    "forcing_value_estimand",
    "forcing_value_sign_rule",
)


class InferenceAuthorityError(RuntimeError):
    """Raised when an input or publication precondition cannot be proved."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return _sha256(payload)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_bound_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise InferenceAuthorityError(f"{label} cannot be inspected: {absolute}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise InferenceAuthorityError(f"{label} is not a regular, non-symlink file: {absolute}")
    try:
        payload = absolute.read_bytes()
    except OSError as exc:
        raise InferenceAuthorityError(f"{label} cannot be read: {absolute}") from exc
    if not payload:
        raise InferenceAuthorityError(f"{label} is empty: {absolute}")
    return payload, {
        "path": str(absolute),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise InferenceAuthorityError(f"{label} is not strict JSON") from exc


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceAuthorityError(f"{label} must be a JSON object")
    return value


def _require_columns(frame: pd.DataFrame, expected: Sequence[str], *, label: str) -> None:
    if tuple(frame.columns) != tuple(expected):
        raise InferenceAuthorityError(
            f"{label} schema differs: got={list(frame.columns)}, expected={list(expected)}"
        )


def _integer_series(series: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise InferenceAuthorityError(f"{label} is not numeric") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise InferenceAuthorityError(f"{label} must contain finite integers")
    return numeric.astype("int64")


def _finite_series(series: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").astype(float)
    except Exception as exc:
        raise InferenceAuthorityError(f"{label} is not numeric") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise InferenceAuthorityError(f"{label} contains null or non-finite values")
    return numeric


def _boolean_series(series: pd.Series, *, label: str) -> pd.Series:
    if (
        series.isna().any()
        or not series.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    ):
        raise InferenceAuthorityError(f"{label} must contain only booleans")
    return series.astype(bool)


def _normalise_sites(series: pd.Series, *, label: str) -> pd.Series:
    if series.isna().any():
        raise InferenceAuthorityError(f"{label} contains null site identifiers")
    sites = series.astype(str).str.strip().str.zfill(8)
    valid = sites.str.fullmatch(r"[0-9]{8,15}")
    if not valid.all():
        raise InferenceAuthorityError(
            f"{label} contains invalid site identifiers: {sites[~valid].head(3).tolist()}"
        )
    return sites


def _site_list_sha256(sites: Sequence[str]) -> str:
    return _sha256(("\n".join(sites) + "\n").encode("utf-8"))


def _expected_upstream_cells() -> set[tuple[str, str, int, str]]:
    return {(arm, model, horizon, arm) for arm in ARMS for model in MODELS for horizon in HORIZONS}


def _validate_output_binding(
    manifest: Mapping[str, Any], paired_binding: Mapping[str, Any]
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise InferenceAuthorityError("upstream manifest.outputs must be a non-empty list")
    records = [item for item in outputs if isinstance(item, Mapping)]
    if len(records) != len(outputs):
        raise InferenceAuthorityError("upstream manifest.outputs contains a non-object")
    names = [item.get("name") for item in records]
    if len(set(names)) != len(names):
        raise InferenceAuthorityError("upstream output names are duplicated")
    expected_name = "forcing_regime_v4_paired_contrasts.parquet"
    if Path(str(paired_binding["path"])).name != expected_name:
        raise InferenceAuthorityError(f"paired input must retain canonical name {expected_name}")
    matches = [item for item in records if item.get("name") == expected_name]
    if len(matches) != 1:
        raise InferenceAuthorityError(
            "upstream manifest does not bind exactly one canonical paired-contrast output"
        )
    output = matches[0]
    if (
        output.get("sha256") != paired_binding["sha256"]
        or output.get("bytes") != paired_binding["bytes"]
    ):
        raise InferenceAuthorityError(
            "paired output hash/binding mismatch against upstream manifest"
        )
    if output.get("rows") != EXPECTED_PAIRED_ROWS:
        raise InferenceAuthorityError("upstream paired output does not bind exactly 696 rows")


def _validate_upstream_manifest(
    manifest: Any,
    *,
    paired_binding: Mapping[str, Any],
    registry_binding: Mapping[str, Any],
) -> tuple[tuple[str, ...], str]:
    top = _require_mapping(manifest, label="upstream manifest")
    if top.get("authority_schema") != UPSTREAM_SCHEMA:
        raise InferenceAuthorityError("upstream manifest authority_schema is not forcing v4")
    if top.get("authority_status") != AUTHORITY_STATUS:
        raise InferenceAuthorityError(f"upstream status must be {AUTHORITY_STATUS}")
    if top.get("scope_status") != SCOPE_STATUS:
        raise InferenceAuthorityError(f"upstream scope status must be {SCOPE_STATUS}")

    governance = _require_mapping(top.get("governance"), label="upstream governance")
    required_governance = {
        "post_outcome_normalization": True,
        "completed_viewed_domain_subset": True,
        "prospective": False,
        "confirmatory": False,
        "full_v4_protocol_authority": False,
        "automatic_P5_joint_verdict": False,
    }
    for key, value in required_governance.items():
        if governance.get(key) is not value:
            raise InferenceAuthorityError(f"upstream governance.{key} must be {value}")

    scope = _require_mapping(top.get("scope"), label="upstream scope")
    if tuple(scope.get("arms", ())) != ARMS or tuple(scope.get("protocol_arms", ())) != ARMS:
        raise InferenceAuthorityError("upstream arm scope is not exactly F0/F3_full")
    if tuple(scope.get("models", ())) != MODELS:
        raise InferenceAuthorityError("upstream model scope differs")
    if tuple(scope.get("horizons", ())) != HORIZONS:
        raise InferenceAuthorityError("upstream horizon scope differs")
    if scope.get("exact_cell_count") != EXPECTED_UPSTREAM_CELLS:
        raise InferenceAuthorityError("upstream exact cell count is not 12")
    cells = scope.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_UPSTREAM_CELLS:
        raise InferenceAuthorityError("upstream cell inventory is not exactly 12")
    actual_cells: list[tuple[str, str, int, str]] = []
    for index, raw in enumerate(cells):
        cell = _require_mapping(raw, label=f"upstream scope.cells[{index}]")
        if set(cell) != {"arm", "model", "horizon", "protocol_arm"}:
            raise InferenceAuthorityError("upstream cell fields differ from the canonical layout")
        horizon = cell.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise InferenceAuthorityError("upstream cell horizon is not an integer")
        actual_cells.append(
            (str(cell.get("arm")), str(cell.get("model")), horizon, str(cell.get("protocol_arm")))
        )
    if (
        len(set(actual_cells)) != len(actual_cells)
        or set(actual_cells) != _expected_upstream_cells()
    ):
        raise InferenceAuthorityError("upstream 12-cell inventory is duplicated or incomplete")

    protocol = _require_mapping(
        _require_mapping(top.get("bindings"), label="upstream bindings").get("v4_draft_protocol"),
        label="upstream v4 protocol binding",
    )
    protocol_sha256 = protocol.get("sha256")
    if not _valid_sha256(protocol_sha256):
        raise InferenceAuthorityError("upstream protocol SHA-256 is invalid")
    if protocol.get("exact_bytes_bound") is not True:
        raise InferenceAuthorityError("upstream protocol is not exact-byte bound")
    if not isinstance(protocol.get("bytes"), int) or protocol["bytes"] <= 0:
        raise InferenceAuthorityError("upstream protocol byte count is invalid")

    audits = _require_mapping(top.get("audits"), label="upstream audits")
    if audits.get("exact_canonical_cell_count") != EXPECTED_UPSTREAM_CELLS:
        raise InferenceAuthorityError("upstream audit does not verify exactly 12 cells")
    if audits.get("protocol_sha256_computed_from_current_exact_bytes") != protocol_sha256:
        raise InferenceAuthorityError("upstream protocol hash audit disagrees with its binding")
    if audits.get("common_reportable_station_set_across_all_12_cells") is not True:
        raise InferenceAuthorityError("upstream common station set is not proven across 12 cells")
    if audits.get("station_first_estimand_only") is not True:
        raise InferenceAuthorityError("upstream station-first estimand audit is absent")
    if audits.get("sign_relation_verified_station_by_station") is not True:
        raise InferenceAuthorityError("upstream forcing sign audit is absent")
    if (
        audits.get("protocol_forcing_value_sign_contract")
        != UPSTREAM_PROTOCOL_FORCING_VALUE_SIGN_CONTRACT
    ):
        raise InferenceAuthorityError("upstream forcing-value protocol sign contract differs")
    if audits.get("common_reportable_station_count") != EXPECTED_STATIONS:
        raise InferenceAuthorityError("upstream common station count is not exactly 116")
    raw_sites = audits.get("common_reportable_station_ids")
    if not isinstance(raw_sites, list) or len(raw_sites) != EXPECTED_STATIONS:
        raise InferenceAuthorityError("upstream common station inventory is not exactly 116")
    sites = tuple(str(value).strip().zfill(8) for value in raw_sites)
    if tuple(sorted(set(sites))) != sites:
        raise InferenceAuthorityError("upstream common stations must be sorted and unique")
    if any(re.fullmatch(r"[0-9]{8,15}", site) is None for site in sites):
        raise InferenceAuthorityError("upstream common station inventory is invalid")
    if audits.get("common_reportable_station_ids_sha256") != _site_list_sha256(sites):
        raise InferenceAuthorityError("upstream common station list hash differs")

    _validate_output_binding(top, paired_binding)
    row_counts = _require_mapping(top.get("row_counts"), label="upstream row_counts")
    if row_counts.get("authority_contrast_rows") != EXPECTED_PAIRED_ROWS:
        raise InferenceAuthorityError("upstream row-count inventory is not 696 contrasts")

    data_bindings = _require_mapping(
        _require_mapping(top.get("bindings"), label="upstream bindings").get("data"),
        label="upstream data bindings",
    )
    registry = _require_mapping(
        data_bindings.get("station_registry"), label="upstream station registry binding"
    )
    if Path(str(registry_binding["path"])).name != "station_registry_v1.csv":
        raise InferenceAuthorityError(
            "registry input must retain canonical station_registry_v1.csv name"
        )
    if (
        registry.get("sha256") != registry_binding["sha256"]
        or registry.get("bytes") != registry_binding["bytes"]
    ):
        raise InferenceAuthorityError(
            "station registry hash/binding mismatch against upstream manifest"
        )
    return sites, str(protocol_sha256)


def validate_paired_contrasts(
    frame: pd.DataFrame,
    *,
    common_stations: Sequence[str],
    protocol_sha256: str,
) -> pd.DataFrame:
    """Validate the exact 696-row station-paired forcing-value inventory."""
    _require_columns(frame, PAIRED_COLUMNS, label="paired contrasts")
    if len(frame) != EXPECTED_PAIRED_ROWS:
        raise InferenceAuthorityError(
            f"paired contrasts contain {len(frame)} rows, expected {EXPECTED_PAIRED_ROWS}"
        )
    paired = frame.loc[:, PAIRED_COLUMNS].copy()
    string_columns = (
        "authority_schema",
        "authority_status",
        "scope_status",
        "protocol_sha256",
        "arm",
        "protocol_arm",
        "reference_arm",
        "reference_protocol_arm",
        "model",
        "contrast_role",
        "model_status",
        "delta_estimand",
        "forcing_value_estimand",
        "forcing_value_sign_rule",
    )
    for column in string_columns:
        if paired[column].isna().any():
            raise InferenceAuthorityError(f"paired contrasts.{column} contains nulls")
        paired[column] = paired[column].astype(str)
    paired["site_id"] = _normalise_sites(paired["site_id"], label="paired contrasts.site_id")
    paired["horizon"] = _integer_series(paired["horizon"], label="paired contrasts.horizon")
    paired["n_common_keys"] = _integer_series(
        paired["n_common_keys"], label="paired contrasts.n_common_keys"
    )
    paired["prospective"] = _boolean_series(
        paired["prospective"], label="paired contrasts.prospective"
    )
    paired["confirmatory"] = _boolean_series(
        paired["confirmatory"], label="paired contrasts.confirmatory"
    )
    paired["reportable"] = _boolean_series(
        paired["reportable"], label="paired contrasts.reportable"
    )
    for column in (
        "rmse_arm",
        "rmse_reference",
        "delta_rmse",
        "forcing_value_rmse",
    ):
        paired[column] = _finite_series(paired[column], label=f"paired contrasts.{column}")
    if (paired[["rmse_arm", "rmse_reference"]] < 0.0).any().any():
        raise InferenceAuthorityError("paired contrasts contain a negative RMSE")
    if (paired["n_common_keys"] < 100).any() or not paired["reportable"].all():
        raise InferenceAuthorityError("paired contrasts include a non-reportable station")

    exact_constants: dict[str, Any] = {
        "authority_schema": UPSTREAM_SCHEMA,
        "authority_status": AUTHORITY_STATUS,
        "scope_status": SCOPE_STATUS,
        "prospective": False,
        "confirmatory": False,
        "protocol_sha256": protocol_sha256,
        "arm": "F3_full",
        "protocol_arm": "F3_full",
        "reference_arm": "F0",
        "reference_protocol_arm": "F0",
        "contrast_role": "primary_tree_F3_full_minus_F0",
        "model_status": "frozen_tree_model",
        "delta_estimand": DELTA_ESTIMAND,
        "forcing_value_estimand": STATION_FORCING_VALUE,
        "forcing_value_sign_rule": SIGN_RULE,
    }
    for column, expected in exact_constants.items():
        if set(paired[column].tolist()) != {expected}:
            raise InferenceAuthorityError(f"paired contrasts.{column} violates its fixed contract")
    if set(paired["model"]) != set(MODELS) or set(paired["horizon"]) != set(HORIZONS):
        raise InferenceAuthorityError("paired model/horizon inventory differs")

    delta = paired["rmse_arm"].to_numpy(float) - paired["rmse_reference"].to_numpy(float)
    forcing = -paired["delta_rmse"].to_numpy(float)
    direct_forcing = paired["rmse_reference"].to_numpy(float) - paired["rmse_arm"].to_numpy(float)
    if not np.array_equal(delta, paired["delta_rmse"].to_numpy(float)):
        raise InferenceAuthorityError("delta_rmse is not exactly F3_full minus F0 RMSE")
    if not np.array_equal(forcing, paired["forcing_value_rmse"].to_numpy(float)):
        raise InferenceAuthorityError("forcing-value sign contract is violated")
    if not np.array_equal(direct_forcing, paired["forcing_value_rmse"].to_numpy(float)):
        raise InferenceAuthorityError("forcing value is not exactly F0 minus F3_full RMSE")

    identity = ["model", "horizon", "site_id"]
    if paired.duplicated(identity).any():
        raise InferenceAuthorityError("paired input is not one station row per analysis cell")
    expected_cells = {(model, horizon) for model in MODELS for horizon in HORIZONS}
    actual_cells = set(
        paired[["model", "horizon"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if actual_cells != expected_cells:
        raise InferenceAuthorityError("paired analysis-cell inventory is not exactly six")
    expected_sites = tuple(common_stations)
    for cell, group in paired.groupby(["model", "horizon"], sort=True):
        sites = tuple(sorted(group["site_id"].tolist()))
        if len(group) != EXPECTED_STATIONS or sites != expected_sites:
            raise InferenceAuthorityError(
                f"paired cell {cell} does not contain the exact 116 common stations"
            )
    return paired.sort_values(identity, kind="mergesort").reset_index(drop=True)


def _normalise_huc_code(value: str) -> str:
    stripped = str(value).strip()
    if not stripped:
        return ""
    return stripped.zfill(8 if len(stripped) <= 8 else 12)


def validate_canonical_huc2_registry(
    frame: pd.DataFrame, *, common_stations: Sequence[str]
) -> dict[str, str]:
    required = {"site_no", "huc_cd", "huc2", "huc_metadata_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InferenceAuthorityError(f"station_registry_v1 lacks columns: {missing}")
    registry = frame.loc[:, sorted(required)].copy()
    registry["site_no"] = _normalise_sites(registry["site_no"], label="station_registry_v1.site_no")
    if registry["site_no"].duplicated().any():
        raise InferenceAuthorityError("station registry contains duplicate site_no values")
    for column in ("huc_cd", "huc2", "huc_metadata_status"):
        if registry[column].isna().any():
            raise InferenceAuthorityError(f"station registry {column} contains nulls")
        registry[column] = registry[column].astype(str).str.strip()
    registry["huc_cd"] = registry["huc_cd"].map(_normalise_huc_code)
    registry["huc2"] = registry["huc2"].map(lambda value: value.zfill(2) if value else "")
    indexed = registry.set_index("site_no", verify_integrity=True)
    missing_sites = sorted(set(common_stations) - set(indexed.index))
    if missing_sites:
        raise InferenceAuthorityError(
            f"canonical registry lacks common stations: {missing_sites[:3]}"
        )
    mapping: dict[str, str] = {}
    for site in common_stations:
        row = indexed.loc[site]
        huc_cd = str(row["huc_cd"])
        huc2 = str(row["huc2"])
        if str(row["huc_metadata_status"]) != "USGS_SNAPSHOT_SITE_NO_MATCH":
            raise InferenceAuthorityError(
                f"common station {site} lacks canonical verified HUC metadata"
            )
        if re.fullmatch(r"(?:[0-9]{8}|[0-9]{12})", huc_cd) is None:
            raise InferenceAuthorityError(f"common station {site} has an invalid HUC code")
        if re.fullmatch(r"[0-9]{2}", huc2) is None or huc_cd[:2] != huc2:
            raise InferenceAuthorityError(f"common station {site} has a non-canonical HUC2 mapping")
        mapping[site] = f"HUC2:{huc2}"
    if len(set(mapping.values())) != EXPECTED_HUC2_CLUSTERS:
        raise InferenceAuthorityError(
            f"common stations map to {len(set(mapping.values()))} HUC2 clusters, "
            f"expected {EXPECTED_HUC2_CLUSTERS}"
        )
    return mapping


def load_validated_inputs(
    *, manifest_path: Path, paired_path: Path, registry_path: Path
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Read and bind exactly the three authorized scientific input files."""
    manifest_payload, manifest_binding = _read_bound_bytes(manifest_path, label="upstream manifest")
    paired_payload, paired_binding = _read_bound_bytes(
        paired_path, label="upstream paired contrasts"
    )
    registry_payload, registry_binding = _read_bound_bytes(
        registry_path, label="station_registry_v1"
    )
    manifest = _parse_json(manifest_payload, label="upstream manifest")
    common_stations, protocol_sha256 = _validate_upstream_manifest(
        manifest,
        paired_binding=paired_binding,
        registry_binding=registry_binding,
    )
    try:
        unvalidated_paired = pd.read_parquet(io.BytesIO(paired_payload))
    except Exception as exc:
        raise InferenceAuthorityError("upstream paired contrasts are not Parquet") from exc
    paired = validate_paired_contrasts(
        unvalidated_paired,
        common_stations=common_stations,
        protocol_sha256=protocol_sha256,
    )
    try:
        unvalidated_registry = pd.read_csv(
            io.BytesIO(registry_payload),
            dtype=str,
            keep_default_na=False,
            float_precision="round_trip",
        )
    except Exception as exc:
        raise InferenceAuthorityError("station_registry_v1 is not readable CSV") from exc
    huc2_by_site = validate_canonical_huc2_registry(
        unvalidated_registry, common_stations=common_stations
    )
    canonical_map = [{"site_id": site, "huc2": huc2_by_site[site]} for site in sorted(huc2_by_site)]
    input_records = {
        "upstream_manifest": manifest_binding,
        "upstream_paired_contrasts": {
            **paired_binding,
            "rows": len(paired),
            "hash_validated_against_upstream_manifest": True,
        },
        "station_registry_v1": {
            **registry_binding,
            "hash_validated_against_upstream_manifest": True,
            "canonical_huc2_map_validated": True,
            "canonical_huc2_map_sha256": _canonical_json_sha256(canonical_map),
            "mapped_common_stations": len(canonical_map),
            "canonical_huc2_clusters": sorted(set(huc2_by_site.values())),
            "canonical_huc2_cluster_count": len(set(huc2_by_site.values())),
        },
        "upstream_protocol_binding": dict(manifest["bindings"]["v4_draft_protocol"]),
        "common_reportable_stations": {
            "n": len(common_stations),
            "site_ids_sha256": _site_list_sha256(common_stations),
        },
    }
    input_records["input_set_sha256"] = _canonical_json_sha256(
        {
            key: {field: value[field] for field in ("bytes", "sha256")}
            for key, value in input_records.items()
            if key in {"upstream_manifest", "upstream_paired_contrasts", "station_registry_v1"}
        }
    )
    return paired, huc2_by_site, input_records


def _validate_effect_cluster_arrays(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(effects, dtype=float)
    groups = np.asarray(clusters, dtype=str)
    if values.ndim != 1 or groups.ndim != 1 or len(values) != len(groups):
        raise InferenceAuthorityError(
            "station effects and HUC2 clusters must be aligned one-dimensional arrays"
        )
    if len(values) == 0 or not np.isfinite(values).all():
        raise InferenceAuthorityError("station effects must be non-empty and finite")
    if any(not group for group in groups):
        raise InferenceAuthorityError("HUC2 cluster identifiers must be non-empty")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise InferenceAuthorityError("cluster sensitivity requires at least two HUC2s")
    return values, groups, unique


def bootstrap_seed_for_analysis(analysis_id: str) -> int:
    payload = f"{BOOTSTRAP_SEED_NAMESPACE}\n{analysis_id}\n".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def whole_huc2_cluster_bootstrap_distribution(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
    *,
    draws: int,
    seed: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Resample whole HUC2 groups while retaining the equal-station statistic."""
    values, groups, unique = _validate_effect_cluster_arrays(effects, clusters)
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise InferenceAuthorityError("bootstrap draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise InferenceAuthorityError("bootstrap seed must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise InferenceAuthorityError("bootstrap batch size must be a positive integer")
    cluster_index = np.searchsorted(unique, groups)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_cluster_index = cluster_index[order]
    cluster_count = len(unique)
    probabilities = np.full(cluster_count, 1.0 / cluster_count)
    rng = np.random.default_rng(int(seed))
    result = np.empty(draws, dtype=float)
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        multiplicity = rng.multinomial(cluster_count, probabilities, size=stop - start)
        station_weights = multiplicity[:, sorted_cluster_index]
        cumulative = np.cumsum(station_weights, axis=1)
        total = cumulative[:, -1]
        if (total <= 0).any():  # mathematically impossible; retained fail-closed
            raise InferenceAuthorityError("bootstrap draw contains no station rows")
        lower_rank = (total - 1) // 2
        upper_rank = total // 2
        lower_index = np.argmax(cumulative > lower_rank[:, None], axis=1)
        upper_index = np.argmax(cumulative > upper_rank[:, None], axis=1)
        result[start:stop] = (sorted_values[lower_index] + sorted_values[upper_index]) / 2.0
    return result


def exact_whole_huc2_sign_flip(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
    *,
    null_value: float = 0.0,
) -> dict[str, Any]:
    """Enumerate every whole-HUC2 sign vector and all three raw tails."""
    values, groups, unique = _validate_effect_cluster_arrays(effects, clusters)
    if not np.isfinite(null_value):
        raise InferenceAuthorityError("sign-flip null value must be finite")
    if len(unique) > 20:
        raise InferenceAuthorityError("exact sign-flip is capped at 20 clusters")
    centered = values - float(null_value)
    observed = float(np.median(centered))
    cluster_index = np.searchsorted(unique, groups)
    configurations = 1 << len(unique)
    less_count = 0
    greater_count = 0
    two_sided_count = 0
    for start in range(0, configurations, 4096):
        stop = min(start + 4096, configurations)
        codes = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = (codes >> np.arange(len(unique), dtype=np.uint64)) & 1
        signs = bits.astype(float) * 2.0 - 1.0
        simulated = np.median(signs[:, cluster_index] * centered[None, :], axis=1)
        less_count += int(np.sum(simulated <= observed + SIGN_FLIP_TIE_TOLERANCE))
        greater_count += int(np.sum(simulated >= observed - SIGN_FLIP_TIE_TOLERANCE))
        two_sided_count += int(np.sum(np.abs(simulated) >= abs(observed) - SIGN_FLIP_TIE_TOLERANCE))
    return {
        "observed_forcing_value_minus_null": observed,
        "n_clusters": len(unique),
        "configurations": configurations,
        "p_less_or_equal": float(less_count / configurations),
        "p_greater_or_equal": float(greater_count / configurations),
        "p_two_sided_absolute": float(two_sided_count / configurations),
        "exact": True,
        "seed": None,
    }


def _strict_direction(value: float) -> str:
    if value > 0.0:
        return "POSITIVE_FAVORS_F3_FULL"
    if value < 0.0:
        return "NEGATIVE_FAVORS_F0"
    return "ZERO"


def equal_huc2_sensitivity(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    values, groups, unique = _validate_effect_cluster_arrays(effects, clusters)
    rows: list[dict[str, Any]] = []
    medians: list[float] = []
    for huc2 in unique:
        group_values = values[groups == huc2]
        median = float(np.median(group_values))
        medians.append(median)
        rows.append(
            {
                "huc2": str(huc2),
                "n_stations": len(group_values),
                "median_station_forcing_value_rmse": median,
            }
        )
    return rows, float(np.median(np.asarray(medians, dtype=float)))


def leave_one_huc2_sensitivity(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values, groups, unique = _validate_effect_cluster_arrays(effects, clusters)
    point_direction = _strict_direction(float(np.median(values)))
    rows: list[dict[str, Any]] = []
    loco_values: list[float] = []
    for huc2 in unique:
        keep = groups != huc2
        value = float(np.median(values[keep]))
        loco_values.append(value)
        rows.append(
            {
                "held_out_huc2": str(huc2),
                "held_out_station_count": int(np.sum(~keep)),
                "remaining_station_count": int(np.sum(keep)),
                "remaining_cluster_count": int(len(unique) - 1),
                "equal_station_median_forcing_value_rmse": value,
                "direction_vs_zero": _strict_direction(value),
            }
        )
    loco = np.asarray(loco_values, dtype=float)
    if np.all(loco > 0.0):
        range_direction = "ALL_POSITIVE_FAVOR_F3_FULL"
    elif np.all(loco < 0.0):
        range_direction = "ALL_NEGATIVE_FAVOR_F0"
    else:
        range_direction = "CROSSES_OR_TOUCHES_ZERO"
    stable = bool(
        (
            point_direction == "POSITIVE_FAVORS_F3_FULL"
            and range_direction == "ALL_POSITIVE_FAVOR_F3_FULL"
        )
        or (point_direction == "NEGATIVE_FAVORS_F0" and range_direction == "ALL_NEGATIVE_FAVOR_F0")
    )
    return rows, {
        "loco_min_forcing_value_rmse": float(np.min(loco)),
        "loco_max_forcing_value_rmse": float(np.max(loco)),
        "loco_range_direction_vs_zero": range_direction,
        "loco_direction_stable": stable,
    }


def _analysis_id(model: str, horizon: int) -> str:
    return f"forcing=F0-minus-F3_full|model={model}|h={horizon}"


def build_cluster_sensitivities(
    paired: pd.DataFrame,
    huc2_by_site: Mapping[str, str],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if bootstrap_draws != BOOTSTRAP_DRAWS:
        raise InferenceAuthorityError(
            f"published authority requires exactly {BOOTSTRAP_DRAWS} bootstrap draws"
        )
    missing = sorted(set(paired["site_id"]) - set(huc2_by_site))
    if missing:
        raise InferenceAuthorityError(f"paired rows lack canonical HUC2 mappings: {missing[:3]}")
    summary_rows: list[dict[str, Any]] = []
    per_huc_rows: list[dict[str, Any]] = []
    loco_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for (model, horizon), group in paired.groupby(["model", "horizon"], sort=True):
        horizon = int(horizon)
        analysis_id = _analysis_id(str(model), horizon)
        seed = bootstrap_seed_for_analysis(analysis_id)
        values = group["forcing_value_rmse"].to_numpy(float)
        clusters = np.asarray([huc2_by_site[site] for site in group["site_id"]], dtype=str)
        unique = np.unique(clusters)
        if len(values) != EXPECTED_STATIONS or len(unique) != EXPECTED_HUC2_CLUSTERS:
            raise InferenceAuthorityError(f"{analysis_id} has an invalid station/HUC2 inventory")
        distribution = whole_huc2_cluster_bootstrap_distribution(
            values, clusters, draws=BOOTSTRAP_DRAWS, seed=seed
        )
        ci_low, ci_high = np.percentile(distribution, [2.5, 97.5])
        sign_flip = exact_whole_huc2_sign_flip(values, clusters)
        if sign_flip["configurations"] != 2**EXPECTED_HUC2_CLUSTERS:
            raise InferenceAuthorityError(f"{analysis_id} did not enumerate exactly 2^15 signs")
        per_huc, equal_huc = equal_huc2_sensitivity(values, clusters)
        loco, loco_summary = leave_one_huc2_sensitivity(values, clusters)
        base = {
            "authority_schema": AUTHORITY_SCHEMA,
            "authority_status": AUTHORITY_STATUS,
            "scope_status": SCOPE_STATUS,
            "inference_role": INFERENCE_ROLE,
            "status_permanent": True,
            "prospective": False,
            "confirmatory": False,
            "analysis_id": analysis_id,
            "model": str(model),
            "horizon": horizon,
            "station_forcing_value": STATION_FORCING_VALUE,
            "forcing_value_sign_rule": SIGN_RULE,
        }
        point = float(np.median(values))
        summary_rows.append(
            {
                **base,
                "n_stations": len(values),
                "n_huc2_clusters": len(unique),
                "primary_estimand": FORCING_VALUE_ESTIMAND,
                "primary_weighting": "EQUAL_STATION",
                "equal_station_median_forcing_value_rmse": point,
                "equal_station_direction_vs_zero": _strict_direction(point),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": seed,
                "cluster_bootstrap_unit": "WHOLE_CANONICAL_HUC2",
                "cluster_bootstrap_target_estimand": "EQUAL_STATION_MEDIAN",
                "cluster_bootstrap_ci_low": float(ci_low),
                "cluster_bootstrap_ci_high": float(ci_high),
                "sign_flip_exact_configurations": sign_flip["configurations"],
                "sign_flip_p_less_or_equal_raw_descriptive": sign_flip["p_less_or_equal"],
                "sign_flip_p_greater_or_equal_raw_descriptive": sign_flip["p_greater_or_equal"],
                "sign_flip_p_two_sided_absolute_raw_descriptive": sign_flip["p_two_sided_absolute"],
                "equal_huc2_median_forcing_value_rmse": equal_huc,
                **loco_summary,
                "claim_verdict_emitted": False,
                "multiplicity_adjustment_computed": False,
                "architecture_comparison_in_scope": False,
            }
        )
        for row in per_huc:
            per_huc_rows.append({**base, **row})
        for row in loco:
            loco_rows.append({**base, **row})
        registry.append(
            {
                "analysis_id": analysis_id,
                "model": str(model),
                "horizon": horizon,
                "bootstrap_seed": seed,
                "n_stations": len(values),
                "n_huc2_clusters": len(unique),
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["model", "horizon"], kind="mergesort")
        .reset_index(drop=True)
    )
    per_huc_frame = (
        pd.DataFrame(per_huc_rows)
        .sort_values(["analysis_id", "huc2"], kind="mergesort")
        .reset_index(drop=True)
    )
    loco_frame = (
        pd.DataFrame(loco_rows)
        .sort_values(["analysis_id", "held_out_huc2"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(summary) != EXPECTED_ANALYSIS_CELLS:
        raise InferenceAuthorityError("inference summary inventory is not exactly six cells")
    if len(per_huc_frame) != EXPECTED_ANALYSIS_CELLS * EXPECTED_HUC2_CLUSTERS:
        raise InferenceAuthorityError("per-HUC2 inventory is not exactly 6 x 15")
    if len(loco_frame) != EXPECTED_ANALYSIS_CELLS * EXPECTED_HUC2_CLUSTERS:
        raise InferenceAuthorityError("leave-one-HUC2 inventory is not exactly 6 x 15")
    if summary["analysis_id"].duplicated().any():
        raise InferenceAuthorityError("analysis registry contains duplicate IDs")
    return summary, per_huc_frame, loco_frame, registry


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
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
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
        parent_info = destination.parent.lstat()
    except OSError as exc:
        raise InferenceAuthorityError("publication endpoints cannot be inspected") from exc
    if (
        not stat.S_ISDIR(source_info.st_mode)
        or stat.S_ISLNK(source_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or source_info.st_dev != parent_info.st_dev
    ):
        raise InferenceAuthorityError("publication endpoints are unsafe or cross-filesystem")
    if os.path.lexists(destination):
        raise InferenceAuthorityError(f"create-only authority collision: {destination}")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":  # pragma: no cover
        function = getattr(libc, "renamex_np", None)
        if function is None:
            raise InferenceAuthorityError("host lacks atomic exclusive renamex_np")
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        return_code = function(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:  # pragma: no cover
            raise InferenceAuthorityError("host lacks atomic exclusive renameat2")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        return_code = function(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:  # pragma: no cover
        raise InferenceAuthorityError(f"unsupported host for atomic publish: {sys.platform}")
    if return_code != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise InferenceAuthorityError(f"create-only authority collision: {destination}")
        raise InferenceAuthorityError("atomic authority publication failed") from OSError(
            error_number, os.strerror(error_number)
        )
    _fsync_directory(destination.parent)


def publish_authority(
    summary: pd.DataFrame,
    per_huc: pd.DataFrame,
    loco: pd.DataFrame,
    *,
    output_dir: Path,
    manifest_base: Mapping[str, Any],
) -> Path:
    """Atomically publish a complete authority without overwriting any path."""
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(destination):
        raise InferenceAuthorityError(
            f"authority output already exists; refusing overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_info = destination.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise InferenceAuthorityError("authority output parent must be a real directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        outputs = [
            _write_staged_parquet(summary, stage / f"{OUTPUT_PREFIX}_summary.parquet"),
            _write_staged_parquet(per_huc, stage / f"{OUTPUT_PREFIX}_per_huc2.parquet"),
            _write_staged_parquet(loco, stage / f"{OUTPUT_PREFIX}_leave_one_huc2.parquet"),
        ]
        records = json.loads(summary.to_json(orient="records", double_precision=15))
        summary_json = _write_staged_json(
            {
                "authority_schema": AUTHORITY_SCHEMA,
                "authority_status": AUTHORITY_STATUS,
                "scope_status": SCOPE_STATUS,
                "inference_role": INFERENCE_ROLE,
                "status_permanent": True,
                "prospective": False,
                "confirmatory": False,
                "rows": records,
            },
            stage / f"{OUTPUT_PREFIX}_summary.json",
        )
        summary_json["rows"] = len(summary)
        outputs.append(summary_json)
        manifest = dict(manifest_base)
        manifest["outputs"] = outputs
        manifest["output_set_sha256"] = _canonical_json_sha256(outputs)
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
    *, manifest_path: Path, paired_path: Path, registry_path: Path, output_dir: Path
) -> Path:
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(destination):
        raise InferenceAuthorityError(
            f"authority output already exists; refusing overwrite: {destination}"
        )
    paired, huc2_by_site, input_bindings = load_validated_inputs(
        manifest_path=manifest_path,
        paired_path=paired_path,
        registry_path=registry_path,
    )
    summary, per_huc, loco, analysis_registry = build_cluster_sensitivities(paired, huc2_by_site)
    code_payload, code_binding = _read_bound_bytes(Path(__file__), label="builder code")
    method_definitions = {
        "input_unit": "one station-level forcing_value_rmse per model-horizon cell",
        "station_forcing_value": STATION_FORCING_VALUE,
        "primary_estimand": FORCING_VALUE_ESTIMAND,
        "primary_weighting": "equal station",
        "cluster_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "resampling_unit": "whole canonical HUC2",
            "draw_rule": "sample K of K observed HUC2 clusters uniformly with replacement; retain all stations in every selected cluster with its multiplicity",
            "target_estimand": "equal-station median; cluster resampling does not change the estimand to equal-HUC weighting",
            "interval": "2.5th and 97.5th percentiles",
            "seed_namespace": BOOTSTRAP_SEED_NAMESPACE,
            "seed_algorithm": "big-endian uint32 from first four SHA-256 bytes of namespace newline analysis_id newline",
        },
        "exact_sign_flip": {
            "unit": "whole canonical HUC2",
            "null": 0.0,
            "rule": "apply one common sign to all station forcing values in a HUC2 and enumerate every 2^15 sign vector",
            "statistic": "equal-station median",
            "tails": {
                "less_or_equal": "mean(T_star <= T_observed + tie_tolerance)",
                "greater_or_equal": "mean(T_star >= T_observed - tie_tolerance)",
                "two_sided_absolute": "mean(abs(T_star) >= abs(T_observed) - tie_tolerance)",
            },
            "tie_tolerance": SIGN_FLIP_TIE_TOLERANCE,
            "seed": None,
        },
        "equal_huc2": "median within HUC2, then unweighted median across the 15 HUC2 medians",
        "leave_one_huc2": "remove all stations in one HUC2 and recompute the equal-station median, for each of 15 HUC2s",
        "interpretation": INFERENCE_ROLE,
    }
    manifest = {
        "authority_schema": AUTHORITY_SCHEMA,
        "authority_status": AUTHORITY_STATUS,
        "scope_status": SCOPE_STATUS,
        "inference_role": INFERENCE_ROLE,
        "status_permanent": True,
        "prospective": False,
        "confirmatory": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status_contract": {
            "permanent": True,
            "authority_status": AUTHORITY_STATUS,
            "scope_status": SCOPE_STATUS,
            "inference_role": INFERENCE_ROLE,
            "interpretation": "approximate fixed-cohort descriptive only",
            "prospective": False,
            "confirmatory": False,
        },
        "scope": {
            "models": list(MODELS),
            "horizons": list(HORIZONS),
            "analysis_cells": EXPECTED_ANALYSIS_CELLS,
            "stations_per_cell": EXPECTED_STATIONS,
            "huc2_clusters_per_cell": EXPECTED_HUC2_CLUSTERS,
            "claim_verdict_emitted": False,
            "multiplicity_adjustment_computed": False,
            "architecture_comparison_in_scope": False,
        },
        "inputs": input_bindings,
        "method_definitions": method_definitions,
        "method_definitions_sha256": _canonical_json_sha256(method_definitions),
        "analysis_registry": analysis_registry,
        "analysis_registry_sha256": _canonical_json_sha256(analysis_registry),
        "row_counts": {
            "upstream_paired_rows": len(paired),
            "summary_rows": len(summary),
            "per_huc2_rows": len(per_huc),
            "leave_one_huc2_rows": len(loco),
        },
        "software": {
            "builder": {**code_binding, "sha256_recomputed": _sha256(code_payload)},
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return publish_authority(summary, per_huc, loco, output_dir=destination, manifest_base=manifest)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        destination = build_authority(
            manifest_path=args.manifest,
            paired_path=args.paired,
            registry_path=args.registry,
            output_dir=args.out_dir,
        )
    except InferenceAuthorityError as exc:
        print(f"FORCING INFERENCE AUTHORITY BUILD REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"published {INFERENCE_ROLE} authority: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
