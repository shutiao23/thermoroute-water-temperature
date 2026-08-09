#!/usr/bin/env python3
"""Build cluster-robust descriptive inference for the completed L0-L2 subset.

This authority consumes only the completed-subset manifest, its published
station-level paired-contrast table, and the canonical station registry.  It
does not read predictions, refit a model, or expand the completed experiment.

The 2021-2023 outcomes were already visible before this normalization was
specified.  Every output is therefore marked ``POST_OUTCOME_NORMALIZATION_ONLY``
and ``COMPLETED_SUBSET_NOT_FULL_V2``.  The HUC2 bootstrap, exact sign-flip
enumeration, equal-HUC summary, and leave-one-HUC2 analyses are approximate
fixed-cohort descriptive sensitivities, not confirmatory inference.

Protocol v2's N01-N15 family cannot be completed from these inputs: N07-N09
require L3, and the protocol does not select a model (or, for C1-C3, a
geometry) for presentation.  This builder emits model-stratified raw
sensitivities and a fail-closed family audit; it never computes a Holm value or
confirmatory verdict for that incomplete/ambiguous family.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"
DEFAULT_UPSTREAM_DIR = FINAL / "information_regime_v4"
DEFAULT_MANIFEST = DEFAULT_UPSTREAM_DIR / "information_regime_v4_manifest.json"
DEFAULT_PAIRED = DEFAULT_UPSTREAM_DIR / "information_regime_v4_paired_contrasts.parquet"
DEFAULT_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_OUT_DIR = FINAL / "information_regime_inference_v4"

AUTHORITY_SCHEMA = "information_regime_inference_authority_v4"
UPSTREAM_SCHEMA = "information_regime_authority_v4"
AUTHORITY_STATUS = "COMPLETED_SUBSET_NOT_FULL_V2"
CHRONOLOGY = "POST_OUTCOME_NORMALIZATION_ONLY"
INFERENCE_ROLE = "APPROXIMATE_FIXED_COHORT_DESCRIPTIVE_SENSITIVITY"
OUTPUT_PREFIX = "information_regime_inference_v4"

MODELS = ("LightGBM", "ResidualLightGBM")
HORIZONS = (1, 3, 7)
EXPECTED_HUC2_CLUSTERS = 15
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED_NAMESPACE = "thermoroute-information-regime-inference-v4-bootstrap"
SIGN_FLIP_TIE_TOLERANCE = 1e-15

PAIRED_COLUMNS = (
    "model",
    "horizon",
    "site_id",
    "authority_schema",
    "authority_status",
    "protocol_sha256",
    "contrast_family",
    "contrast_id",
    "definition",
    "conditioning_geometry",
    "conditioning_level",
    "candidate_rmse",
    "reference_rmse",
    "delta_rmse",
    "n_common_keys",
    "negative_delta_favors",
)
CELL_COLUMNS = (
    "contrast_family",
    "contrast_id",
    "definition",
    "conditioning_geometry",
    "conditioning_level",
    "model",
    "horizon",
)

# Exact layouts emitted by build_information_regime_authority.py.  An entry is
# (contrast_family, contrast_id, definition, conditioning_geometry,
#  conditioning_level, negative_delta_favors).
CONTRAST_LAYOUTS = (
    (
        "geometry",
        "G@L0",
        "RMSE(region,L0) - RMSE(random,L0)",
        "",
        "L0",
        "candidate",
    ),
    (
        "geometry",
        "G@L1",
        "RMSE(region,L1) - RMSE(random,L1)",
        "",
        "L1",
        "candidate",
    ),
    (
        "geometry",
        "G@L2",
        "RMSE(region,L2) - RMSE(random,L2)",
        "",
        "L2",
        "candidate",
    ),
    ("level", "L1-L0", "RMSE(L1) - RMSE(L0)", "random", "", "candidate"),
    ("level", "L1-L0", "RMSE(L1) - RMSE(L0)", "region", "", "candidate"),
    ("level", "L2-L0", "RMSE(L2) - RMSE(L0)", "random", "", "candidate"),
    ("level", "L2-L0", "RMSE(L2) - RMSE(L0)", "region", "", "candidate"),
    ("level", "L2-L1", "RMSE(L2) - RMSE(L1)", "random", "", "candidate"),
    ("level", "L2-L1", "RMSE(L2) - RMSE(L1)", "region", "", "candidate"),
    (
        "interaction",
        "LxG",
        "[G@L2] - [G@L0]",
        "region-random",
        "L2-L0",
        "smaller geometry penalty at L2",
    ),
)

V2_TEST_BY_CONTRAST_HORIZON = {
    ("L1-L0", 1): "N01",
    ("L1-L0", 3): "N02",
    ("L1-L0", 7): "N03",
    ("L2-L1", 1): "N04",
    ("L2-L1", 3): "N05",
    ("L2-L1", 7): "N06",
    ("G@L0", 1): "N10",
    ("G@L0", 3): "N11",
    ("G@L0", 7): "N12",
    ("G@L2", 1): "N13",
    ("G@L2", 3): "N14",
    ("G@L2", 7): "N15",
}
MISSING_V2_TESTS = ("N07", "N08", "N09")


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
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        info = path.lstat()
    except OSError as exc:
        raise InferenceAuthorityError(f"{label} cannot be inspected: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise InferenceAuthorityError(f"{label} is not a regular, non-symlink file: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise InferenceAuthorityError(f"{label} cannot be read: {path}") from exc
    if not payload:
        raise InferenceAuthorityError(f"{label} is empty: {path}")
    return payload, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InferenceAuthorityError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except InferenceAuthorityError:
        raise
    except Exception as exc:
        raise InferenceAuthorityError(f"{label} is not valid JSON") from exc


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceAuthorityError(f"{label} must be a JSON object")
    return value


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise InferenceAuthorityError(f"{label} lacks required columns: {missing}")


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


def _validate_upstream_manifest(
    manifest: Any,
    *,
    paired_binding: Mapping[str, Any],
) -> tuple[dict[int, tuple[str, ...]], str]:
    top = _require_mapping(manifest, label="upstream manifest")
    if top.get("authority_schema") != UPSTREAM_SCHEMA:
        raise InferenceAuthorityError("upstream manifest authority_schema is not v4")
    if top.get("authority_status") != AUTHORITY_STATUS:
        raise InferenceAuthorityError(f"upstream status must be {AUTHORITY_STATUS}")

    scope = _require_mapping(top.get("scope"), label="upstream manifest.scope")
    if scope.get("full_protocol_v2_authority") is not False:
        raise InferenceAuthorityError("upstream must explicitly be a non-full-v2 authority")
    if tuple(scope.get("models", ())) != MODELS:
        raise InferenceAuthorityError("upstream model scope differs from the frozen subset")
    if tuple(scope.get("horizons", ())) != HORIZONS:
        raise InferenceAuthorityError("upstream horizon scope differs from the frozen subset")
    if tuple(scope.get("levels", ())) != ("L0", "L1", "L2"):
        raise InferenceAuthorityError("upstream level scope is not exactly L0/L1/L2")
    if tuple(scope.get("geometries", ())) != ("random", "region"):
        raise InferenceAuthorityError("upstream geometry scope is not random/region")

    protocol = _require_mapping(
        top.get("protocol_binding"), label="upstream manifest.protocol_binding"
    )
    protocol_sha256 = protocol.get("sha256")
    if not _valid_sha256(protocol_sha256):
        raise InferenceAuthorityError("upstream protocol SHA-256 is invalid")
    if protocol.get("exact_bytes_bound") is not True:
        raise InferenceAuthorityError("upstream protocol is not exact-byte bound")
    if not isinstance(protocol.get("bytes"), int) or protocol["bytes"] <= 0:
        raise InferenceAuthorityError("upstream protocol byte count is invalid")

    outputs = top.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise InferenceAuthorityError("upstream manifest.outputs must be a non-empty list")
    names = [item.get("name") for item in outputs if isinstance(item, Mapping)]
    if len(names) != len(outputs) or len(set(names)) != len(names):
        raise InferenceAuthorityError("upstream output names are missing or duplicated")
    expected_name = Path(str(paired_binding["path"])).name
    matches = [item for item in outputs if item.get("name") == expected_name]
    if len(matches) != 1:
        raise InferenceAuthorityError(
            f"upstream manifest does not bind exactly one {expected_name} output"
        )
    output = _require_mapping(matches[0], label="paired output binding")
    for field in ("sha256", "bytes"):
        if output.get(field) != paired_binding[field]:
            raise InferenceAuthorityError(
                f"paired output {field} hash/binding mismatch against upstream manifest"
            )
    if not isinstance(output.get("rows"), int) or output["rows"] <= 0:
        raise InferenceAuthorityError("paired output row binding is invalid")
    row_counts = _require_mapping(top.get("row_counts"), label="upstream manifest.row_counts")
    if row_counts.get("paired_contrast_rows") != output["rows"]:
        raise InferenceAuthorityError(
            "upstream paired row-count bindings disagree within the manifest"
        )

    common_raw = _require_mapping(
        top.get("common_reportable_stations"),
        label="upstream manifest.common_reportable_stations",
    )
    if set(common_raw) != {str(horizon) for horizon in HORIZONS}:
        raise InferenceAuthorityError("upstream common-station horizons are incomplete")
    common: dict[int, tuple[str, ...]] = {}
    for horizon in HORIZONS:
        record = _require_mapping(common_raw[str(horizon)], label=f"common stations h{horizon}")
        values = record.get("site_ids")
        if not isinstance(values, list) or not values:
            raise InferenceAuthorityError(f"common stations h{horizon} are absent")
        sites = tuple(str(value).strip().zfill(8) for value in values)
        if tuple(sorted(set(sites))) != sites:
            raise InferenceAuthorityError(f"common stations h{horizon} must be sorted and unique")
        if any(re.fullmatch(r"[0-9]{8,15}", site) is None for site in sites):
            raise InferenceAuthorityError(f"common stations h{horizon} are invalid")
        if record.get("n") != len(sites):
            raise InferenceAuthorityError(f"common-station count mismatch at h{horizon}")
        if record.get("site_ids_sha256") != _site_list_sha256(sites):
            raise InferenceAuthorityError(f"common-station hash mismatch at h{horizon}")
        common[horizon] = sites
    return common, str(protocol_sha256)


def _expected_layout_map() -> dict[tuple[str, str, str], tuple[str, str, str]]:
    result: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for family, contrast, definition, geometry, level, favors in CONTRAST_LAYOUTS:
        key = (contrast, geometry, level)
        if key in result:
            raise AssertionError(f"duplicate internal contrast layout: {key}")
        result[key] = (family, definition, favors)
    return result


def validate_paired_contrasts(
    frame: pd.DataFrame,
    *,
    common_stations: Mapping[int, Sequence[str]],
    protocol_sha256: str,
) -> pd.DataFrame:
    """Validate one station-level row in every exact contrast analysis cell."""
    _require_columns(frame, PAIRED_COLUMNS, label="paired contrasts")
    extra = sorted(set(frame.columns) - set(PAIRED_COLUMNS))
    if extra:
        raise InferenceAuthorityError(f"paired contrasts contain unexpected columns: {extra}")
    paired = frame.loc[:, PAIRED_COLUMNS].copy()
    if paired.empty:
        raise InferenceAuthorityError("paired contrasts are empty")

    string_columns = (
        "model",
        "authority_schema",
        "authority_status",
        "protocol_sha256",
        "contrast_family",
        "contrast_id",
        "definition",
        "conditioning_geometry",
        "conditioning_level",
        "negative_delta_favors",
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
    for column in ("candidate_rmse", "reference_rmse", "delta_rmse"):
        paired[column] = _finite_series(paired[column], label=f"paired contrasts.{column}")
    if (paired["n_common_keys"] < 100).any():
        raise InferenceAuthorityError("paired contrasts include a non-reportable station")
    if not np.array_equal(
        paired["candidate_rmse"].to_numpy(float) - paired["reference_rmse"].to_numpy(float),
        paired["delta_rmse"].to_numpy(float),
    ):
        raise InferenceAuthorityError(
            "paired delta is not exactly candidate_rmse minus reference_rmse"
        )
    if set(paired["authority_schema"]) != {UPSTREAM_SCHEMA}:
        raise InferenceAuthorityError("paired rows cross or misstate authority_schema")
    if set(paired["authority_status"]) != {AUTHORITY_STATUS}:
        raise InferenceAuthorityError("paired rows cross or misstate authority_status")
    if set(paired["protocol_sha256"]) != {protocol_sha256}:
        raise InferenceAuthorityError("paired rows cross the manifest protocol binding")
    if set(paired["model"]) != set(MODELS):
        raise InferenceAuthorityError("paired model set is not the exact completed subset")
    if set(paired["horizon"]) != set(HORIZONS):
        raise InferenceAuthorityError("paired horizon set is not the exact completed subset")

    layout_map = _expected_layout_map()
    actual_layouts = set(
        paired[["contrast_id", "conditioning_geometry", "conditioning_level"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if actual_layouts != set(layout_map):
        missing = sorted(set(layout_map) - actual_layouts)
        extra_layouts = sorted(actual_layouts - set(layout_map))
        raise InferenceAuthorityError(
            "paired contrast layout is incomplete or unexpected: "
            f"missing={missing[:3]}, unexpected={extra_layouts[:3]}"
        )
    for key, group in paired.groupby(
        ["contrast_id", "conditioning_geometry", "conditioning_level"],
        sort=True,
        dropna=False,
    ):
        expected_family, expected_definition, expected_favors = layout_map[key]
        if set(group["contrast_family"]) != {expected_family}:
            raise InferenceAuthorityError(f"contrast {key} has the wrong family")
        if set(group["definition"]) != {expected_definition}:
            raise InferenceAuthorityError(f"contrast {key} has the wrong definition")
        if set(group["negative_delta_favors"]) != {expected_favors}:
            raise InferenceAuthorityError(f"contrast {key} has the wrong sign convention")

    identity = [
        "contrast_id",
        "conditioning_geometry",
        "conditioning_level",
        "model",
        "horizon",
        "site_id",
    ]
    if paired.duplicated(identity).any():
        raise InferenceAuthorityError(
            "paired input is not station-level: duplicate station rows in a contrast cell"
        )
    expected_cells = {
        (contrast, geometry, level, model, horizon)
        for contrast, geometry, level in layout_map
        for model in MODELS
        for horizon in HORIZONS
    }
    actual_cells = set(
        paired[
            [
                "contrast_id",
                "conditioning_geometry",
                "conditioning_level",
                "model",
                "horizon",
            ]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if actual_cells != expected_cells:
        raise InferenceAuthorityError("paired analysis-cell inventory is not exactly 60")

    for cell, group in paired.groupby(identity[:-1], sort=True, dropna=False):
        horizon = int(cell[-1])
        actual_sites = tuple(sorted(group["site_id"].tolist()))
        expected_sites = tuple(common_stations[horizon])
        if actual_sites != expected_sites or len(group) != len(expected_sites):
            raise InferenceAuthorityError(
                f"contrast cell {cell} does not contain exactly one row per common station"
            )
    expected_rows = sum(
        len(common_stations[horizon]) * len(MODELS) * len(CONTRAST_LAYOUTS) for horizon in HORIZONS
    )
    if len(paired) != expected_rows:
        raise InferenceAuthorityError(
            f"paired row count is {len(paired)}, expected {expected_rows}"
        )
    return paired.sort_values(identity, kind="mergesort").reset_index(drop=True)


def _normalise_huc_code(value: str) -> str:
    stripped = str(value).strip()
    if not stripped:
        return ""
    return stripped.zfill(8 if len(stripped) <= 8 else 12)


def validate_canonical_huc2_registry(
    frame: pd.DataFrame,
    *,
    common_stations: Mapping[int, Sequence[str]],
) -> dict[str, str]:
    """Return a strict canonical site->HUC2 map for all common stations."""
    required = {"site_no", "huc_cd", "huc2", "huc_metadata_status"}
    _require_columns(frame, required, label="station_registry_v1")
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

    common_union = sorted(set().union(*(set(sites) for sites in common_stations.values())))
    indexed = registry.set_index("site_no", verify_integrity=True)
    missing = sorted(set(common_union) - set(indexed.index))
    if missing:
        raise InferenceAuthorityError(f"canonical registry lacks common stations: {missing[:3]}")
    mapping: dict[str, str] = {}
    for site in common_union:
        row = indexed.loc[site]
        huc_cd = str(row["huc_cd"])
        huc2 = str(row["huc2"])
        status = str(row["huc_metadata_status"])
        if status != "USGS_SNAPSHOT_SITE_NO_MATCH":
            raise InferenceAuthorityError(
                f"common station {site} lacks canonical verified HUC metadata"
            )
        if re.fullmatch(r"(?:[0-9]{8}|[0-9]{12})", huc_cd) is None:
            raise InferenceAuthorityError(f"common station {site} has invalid HUC code")
        if re.fullmatch(r"[0-9]{2}", huc2) is None or huc_cd[:2] != huc2:
            raise InferenceAuthorityError(f"common station {site} has a non-canonical HUC2 mapping")
        # Match the repository's canonical ``huc2_cluster_map`` namespace so
        # raw codes cannot be confused with another clustering variable.
        mapping[site] = f"HUC2:{huc2}"

    for horizon, sites in common_stations.items():
        clusters = {mapping[site] for site in sites}
        if len(clusters) != EXPECTED_HUC2_CLUSTERS:
            raise InferenceAuthorityError(
                f"h{horizon} maps to {len(clusters)} HUC2 clusters, expected "
                f"{EXPECTED_HUC2_CLUSTERS}"
            )
    return mapping


def load_validated_inputs(
    *,
    manifest_path: Path,
    paired_path: Path,
    registry_path: Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Read and bind exactly the three authorized input artifacts."""
    manifest_payload, manifest_binding = _read_bound_bytes(manifest_path, label="upstream manifest")
    paired_payload, paired_binding = _read_bound_bytes(
        paired_path, label="upstream paired contrasts"
    )
    registry_payload, registry_binding = _read_bound_bytes(
        registry_path, label="station_registry_v1"
    )
    manifest = _parse_json(manifest_payload, label="upstream manifest")
    common, protocol_sha256 = _validate_upstream_manifest(manifest, paired_binding=paired_binding)
    try:
        unvalidated_paired = pd.read_parquet(io.BytesIO(paired_payload))
    except Exception as exc:
        raise InferenceAuthorityError("upstream paired contrasts are not Parquet") from exc
    paired = validate_paired_contrasts(
        unvalidated_paired,
        common_stations=common,
        protocol_sha256=protocol_sha256,
    )
    expected_rows = _require_mapping(manifest, label="upstream manifest")["row_counts"][
        "paired_contrast_rows"
    ]
    if len(paired) != expected_rows:
        raise InferenceAuthorityError("paired Parquet row count differs from the upstream manifest")
    try:
        unvalidated_registry = pd.read_csv(
            io.BytesIO(registry_payload),
            dtype=str,
            keep_default_na=False,
            float_precision="round_trip",
        )
    except Exception as exc:
        raise InferenceAuthorityError("station_registry_v1 is not readable CSV") from exc
    huc2_by_site = validate_canonical_huc2_registry(unvalidated_registry, common_stations=common)
    canonical_huc2_map = [
        {"site_id": site, "huc2": huc2_by_site[site]} for site in sorted(huc2_by_site)
    ]
    bindings = {
        "upstream_manifest": manifest_binding,
        "upstream_paired_contrasts": {
            **paired_binding,
            "rows": len(paired),
            "hash_validated_against_upstream_manifest": True,
        },
        "station_registry_v1": {
            **registry_binding,
            "canonical_huc2_map_validated": True,
            "canonical_huc2_map_sha256": _canonical_json_sha256(canonical_huc2_map),
            "mapped_common_stations": len(canonical_huc2_map),
            "canonical_huc2_clusters": sorted(set(huc2_by_site.values())),
            "canonical_huc2_cluster_count": len(set(huc2_by_site.values())),
        },
        "upstream_protocol_binding": dict(manifest["protocol_binding"]),
        "common_reportable_stations": {
            str(horizon): {
                "n": len(sites),
                "site_ids_sha256": _site_list_sha256(tuple(sites)),
            }
            for horizon, sites in common.items()
        },
    }
    return paired, huc2_by_site, bindings


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
    """Return the fixed uint32 bootstrap seed for a canonical analysis id."""
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
    """Bootstrap station medians by resampling complete HUC2 groups.

    Each draw selects exactly K clusters uniformly with replacement from the K
    observed HUC2s.  A selected cluster contributes all of its station effects;
    if selected multiple times, every one of those station rows receives the
    same multiplicity.  The weighted-median implementation below is exactly the
    median of that conceptually concatenated station array, including the
    average of the two middle order statistics for an even replicated count.
    """
    values, groups, unique = _validate_effect_cluster_arrays(effects, clusters)
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise InferenceAuthorityError("bootstrap draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise InferenceAuthorityError("bootstrap seed must be an integer")
    if batch_size <= 0:
        raise InferenceAuthorityError("bootstrap batch size must be positive")

    cluster_index = np.searchsorted(unique, groups)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_cluster_index = cluster_index[order]
    cluster_count = len(unique)
    probabilities = np.full(cluster_count, 1.0 / cluster_count)
    rng = np.random.default_rng(int(seed))
    result = np.empty(draws, dtype=float)
    start = 0
    while start < draws:
        stop = min(start + batch_size, draws)
        multiplicity = rng.multinomial(cluster_count, probabilities, size=stop - start)
        station_weights = multiplicity[:, sorted_cluster_index]
        cumulative = np.cumsum(station_weights, axis=1)
        total = cumulative[:, -1]
        if (total <= 0).any():  # mathematically impossible, retained fail-closed
            raise InferenceAuthorityError("bootstrap draw contains no station rows")
        lower_rank = (total - 1) // 2
        upper_rank = total // 2
        lower_index = np.argmax(cumulative > lower_rank[:, None], axis=1)
        upper_index = np.argmax(cumulative > upper_rank[:, None], axis=1)
        result[start:stop] = (sorted_values[lower_index] + sorted_values[upper_index]) / 2.0
        start = stop
    return result


def exact_whole_huc2_sign_flip(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
    *,
    null_value: float = 0.0,
) -> dict[str, Any]:
    """Enumerate all whole-HUC2 sign vectors for a station-median statistic."""
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
        "observed_effect_minus_null": observed,
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
        return "POSITIVE"
    if value < 0.0:
        return "NEGATIVE"
    return "ZERO"


def leave_one_huc2_sensitivity(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute whole-HUC2 deletion effects and a strict direction-stability gate."""
    values, groups, unique = _validate_effect_cluster_arrays(effects, clusters)
    point = float(np.median(values))
    point_direction = _strict_direction(point)
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
                "equal_station_median_delta_rmse": value,
                "direction_vs_zero": _strict_direction(value),
            }
        )
    loco = np.asarray(loco_values, dtype=float)
    if np.all(loco > 0.0):
        range_direction = "ALL_POSITIVE"
    elif np.all(loco < 0.0):
        range_direction = "ALL_NEGATIVE"
    else:
        range_direction = "CROSSES_OR_TOUCHES_ZERO"
    gate_pass = bool(
        (point_direction == "POSITIVE" and range_direction == "ALL_POSITIVE")
        or (point_direction == "NEGATIVE" and range_direction == "ALL_NEGATIVE")
    )
    summary = {
        "loco_min_delta_rmse": float(np.min(loco)),
        "loco_max_delta_rmse": float(np.max(loco)),
        "loco_range_direction_vs_zero": range_direction,
        "loco_direction_gate_pass": gate_pass,
        "loco_direction_gate_status": (
            "PASS_STRICT_DIRECTION_STABLE"
            if gate_pass
            else "FAIL_DIRECTION_CROSSES_TOUCHES_OR_DIFFERS_FROM_POINT"
        ),
    }
    return rows, summary


def equal_huc2_sensitivity(
    effects: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    """Return per-HUC station medians and their equally weighted median."""
    values, groups, unique = _validate_effect_cluster_arrays(effects, clusters)
    rows = []
    medians = []
    for huc2 in unique:
        group_values = values[groups == huc2]
        median = float(np.median(group_values))
        medians.append(median)
        rows.append(
            {
                "huc2": str(huc2),
                "n_stations": len(group_values),
                "median_station_delta_rmse": median,
            }
        )
    return rows, float(np.median(np.asarray(medians, dtype=float)))


def _analysis_id(metadata: Mapping[str, Any]) -> str:
    geometry = str(metadata["conditioning_geometry"]) or "none"
    level = str(metadata["conditioning_level"]) or "none"
    return (
        f"contrast={metadata['contrast_id']}|geometry={geometry}|level={level}|"
        f"model={metadata['model']}|h={int(metadata['horizon'])}"
    )


def _v2_test_id(contrast_id: str, horizon: int) -> str:
    return V2_TEST_BY_CONTRAST_HORIZON.get((contrast_id, horizon), "")


def build_cluster_sensitivities(
    paired: pd.DataFrame,
    huc2_by_site: Mapping[str, str],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Build the 60 model-stratified sensitivity cells and detail tables."""
    if bootstrap_draws != BOOTSTRAP_DRAWS:
        # The optional argument exists only to make unit-level method testing
        # explicit.  Publication always calls the fixed 10,000-draw contract.
        raise InferenceAuthorityError(
            f"published authority requires exactly {BOOTSTRAP_DRAWS} bootstrap draws"
        )
    missing_huc = sorted(set(paired["site_id"]) - set(huc2_by_site))
    if missing_huc:
        raise InferenceAuthorityError(
            f"paired rows lack canonical HUC2 mappings: {missing_huc[:3]}"
        )

    summary_rows: list[dict[str, Any]] = []
    per_huc_rows: list[dict[str, Any]] = []
    loco_rows: list[dict[str, Any]] = []
    analysis_registry: list[dict[str, Any]] = []
    for keys, group in paired.groupby(list(CELL_COLUMNS), sort=True, dropna=False):
        metadata = dict(zip(CELL_COLUMNS, keys, strict=True))
        analysis_id = _analysis_id(metadata)
        seed = bootstrap_seed_for_analysis(analysis_id)
        values = group["delta_rmse"].to_numpy(float)
        clusters = np.asarray([huc2_by_site[site] for site in group["site_id"]], dtype=str)
        unique = np.unique(clusters)
        if len(unique) != EXPECTED_HUC2_CLUSTERS:
            raise InferenceAuthorityError(
                f"{analysis_id} has {len(unique)} HUC2 clusters, expected 15"
            )
        distribution = whole_huc2_cluster_bootstrap_distribution(
            values, clusters, draws=BOOTSTRAP_DRAWS, seed=seed
        )
        ci_low, ci_high = np.percentile(distribution, [2.5, 97.5])
        sign_flip = exact_whole_huc2_sign_flip(values, clusters)
        if sign_flip["configurations"] != 2**EXPECTED_HUC2_CLUSTERS:
            raise InferenceAuthorityError(
                f"{analysis_id} did not enumerate exactly 2^15 sign configurations"
            )
        per_huc, equal_huc_median = equal_huc2_sensitivity(values, clusters)
        loco, loco_summary = leave_one_huc2_sensitivity(values, clusters)
        v2_test_id = _v2_test_id(str(metadata["contrast_id"]), int(metadata["horizon"]))
        family_role = (
            "V2_TEST_PRESENTATION_CANDIDATE_MODEL_GEOMETRY_UNRESOLVED"
            if v2_test_id
            else "OUTSIDE_V2_N01_N15_FAMILY"
        )
        base = {
            "authority_schema": AUTHORITY_SCHEMA,
            "authority_status": AUTHORITY_STATUS,
            "chronology": CHRONOLOGY,
            "inference_role": INFERENCE_ROLE,
            "analysis_id": analysis_id,
            **metadata,
            "v2_test_id_candidate": v2_test_id,
            "v2_family_role": family_role,
        }
        point = float(np.median(values))
        summary_rows.append(
            {
                **base,
                "n_stations": len(values),
                "n_huc2_clusters": len(unique),
                "equal_station_median_delta_rmse": point,
                "equal_station_direction_vs_zero": _strict_direction(point),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": seed,
                "cluster_bootstrap_ci_low": float(ci_low),
                "cluster_bootstrap_ci_high": float(ci_high),
                "sign_flip_exact_configurations": sign_flip["configurations"],
                "sign_flip_p_less_or_equal_raw_sensitivity": sign_flip["p_less_or_equal"],
                "sign_flip_p_greater_or_equal_raw_sensitivity": sign_flip["p_greater_or_equal"],
                "sign_flip_p_two_sided_absolute_raw_sensitivity": sign_flip["p_two_sided_absolute"],
                "v2_positive_direction_raw_p_sensitivity": (
                    sign_flip["p_greater_or_equal"] if v2_test_id else np.nan
                ),
                "equal_huc2_median_delta_rmse": equal_huc_median,
                **loco_summary,
                "holm_status": ("NOT_COMPUTED_INCOMPLETE_FAMILY_AND_PRESENTATION_UNRESOLVED"),
            }
        )
        for row in per_huc:
            per_huc_rows.append({**base, **row})
        for row in loco:
            loco_rows.append({**base, **row})
        analysis_registry.append(
            {
                "analysis_id": analysis_id,
                "contrast_id": metadata["contrast_id"],
                "conditioning_geometry": metadata["conditioning_geometry"],
                "conditioning_level": metadata["conditioning_level"],
                "model": metadata["model"],
                "horizon": int(metadata["horizon"]),
                "bootstrap_seed": seed,
                "v2_test_id_candidate": v2_test_id,
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["contrast_id", "conditioning_geometry", "model", "horizon"],
            kind="mergesort",
        )
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
    if len(summary) != 60:
        raise InferenceAuthorityError(f"inference summary has {len(summary)} rows, not 60")
    if len(per_huc_frame) != 60 * EXPECTED_HUC2_CLUSTERS:
        raise InferenceAuthorityError("per-HUC2 table is not 60 x 15")
    if len(loco_frame) != 60 * EXPECTED_HUC2_CLUSTERS:
        raise InferenceAuthorityError("LOCO table is not 60 x 15")
    if summary["analysis_id"].duplicated().any():
        raise InferenceAuthorityError("analysis registry contains duplicate ids")
    return summary, per_huc_frame, loco_frame, analysis_registry


def v2_family_audit(analysis_registry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe why no N01-N15 Holm family may be constructed."""
    expected = [f"N{index:02d}" for index in range(1, 16)]
    presentations: dict[str, list[dict[str, Any]]] = {test_id: [] for test_id in expected}
    for record in analysis_registry:
        test_id = str(record.get("v2_test_id_candidate", ""))
        if test_id:
            presentations[test_id].append(
                {
                    "analysis_id": record["analysis_id"],
                    "model": record["model"],
                    "conditioning_geometry": record["conditioning_geometry"],
                }
            )
    return {
        "protocol_family": "v2_N01_N15",
        "expected_test_ids": expected,
        "family_size": 15,
        "source_contrast_absent_test_ids": list(MISSING_V2_TESTS),
        "source_contrast_absent_reason": "L3-L2 is unavailable in the completed L0-L2 subset",
        "available_but_presentation_ambiguous_test_ids": [
            test_id for test_id in expected if presentations[test_id]
        ],
        "presentation_candidates_by_test_id": presentations,
        "unresolved_selection_rules": [
            "protocol v2 does not select LightGBM versus ResidualLightGBM for N01-N15",
            "protocol v2 does not select random versus region geometry for C1/C2/C3",
        ],
        "holm_computed": False,
        "confirmatory_holm_verdict_emitted": False,
        "blocking_reasons": [
            "N07-N09 are absent because L3 was not completed",
            "model presentation is not specified",
            "C1/C2 geometry presentation is not specified",
            "any post-outcome selection would be outcome-informed",
        ],
        "allowed_output": "model-stratified raw approximate descriptive sensitivities only",
    }


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> bytes | None:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit_raw = run("rev-parse", "HEAD")
    tree_raw = run("rev-parse", "HEAD^{tree}")
    status_raw = run("status", "--porcelain=v1", "-z")
    diff_raw = run("diff", "--binary", "HEAD", "--", ".")
    return {
        "commit": commit_raw.decode().strip() if commit_raw else None,
        "head_tree": tree_raw.decode().strip() if tree_raw else None,
        "worktree_dirty": None if status_raw is None else bool(status_raw),
        "worktree_status_sha256": None if status_raw is None else _sha256(status_raw),
        "tracked_diff_against_head_sha256": None if diff_raw is None else _sha256(diff_raw),
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
        destination_parent_info = destination.parent.lstat()
    except OSError as exc:
        raise InferenceAuthorityError("publication endpoints cannot be inspected") from exc
    if (
        not stat.S_ISDIR(source_info.st_mode)
        or stat.S_ISLNK(source_info.st_mode)
        or not stat.S_ISDIR(destination_parent_info.st_mode)
        or stat.S_ISLNK(destination_parent_info.st_mode)
        or source_info.st_dev != destination_parent_info.st_dev
    ):
        raise InferenceAuthorityError("publication endpoints are unsafe or cross-filesystem")
    if os.path.lexists(destination):
        raise InferenceAuthorityError(f"create-only authority collision: {destination}")

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":  # pragma: no cover - exercised on macOS only
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
    """Publish a complete inference authority atomically and create-only."""
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
        json_binding = _write_staged_json(
            {
                "authority_schema": AUTHORITY_SCHEMA,
                "authority_status": AUTHORITY_STATUS,
                "chronology": CHRONOLOGY,
                "inference_role": INFERENCE_ROLE,
                "rows": records,
            },
            stage / f"{OUTPUT_PREFIX}_summary.json",
        )
        json_binding["rows"] = len(summary)
        outputs.append(json_binding)
        manifest = dict(manifest_base)
        manifest["outputs"] = outputs
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
    manifest_path: Path,
    paired_path: Path,
    registry_path: Path,
    output_dir: Path,
) -> Path:
    """Validate inputs, compute fixed sensitivities, and publish create-only."""
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
    family_audit = v2_family_audit(analysis_registry)
    method_definitions = {
        "input_unit": "one station-level paired delta_rmse per exact analysis cell",
        "primary_point": "unweighted median over stations",
        "cluster_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "unit": "whole canonical HUC2",
            "draw_rule": "sample K of K observed HUC2 clusters uniformly with replacement; retain every station in each selected cluster with the cluster multiplicity",
            "statistic": "unweighted median of the resampled station rows",
            "interval": "2.5th and 97.5th percentiles",
            "seed_namespace": BOOTSTRAP_SEED_NAMESPACE,
            "seed_algorithm": "big-endian uint32 from first four SHA-256 bytes of namespace newline analysis_id newline",
        },
        "exact_sign_flip": {
            "unit": "whole canonical HUC2",
            "null": 0.0,
            "rule": "apply one common sign to all station effects in a HUC2 and enumerate every 2^15 sign vector",
            "statistic": "unweighted station median",
            "tails_reported": ["less_or_equal", "greater_or_equal", "two_sided_absolute"],
            "tie_tolerance": SIGN_FLIP_TIE_TOLERANCE,
            "seed": None,
        },
        "equal_huc2": "median within each HUC2 followed by the unweighted median of 15 HUC2 medians",
        "leave_one_huc2": "remove every station in one HUC2 and recompute the unweighted station median; strict direction gate fails on any zero/crossing/disagreement",
        "interpretation": INFERENCE_ROLE,
    }
    manifest = {
        "authority_schema": AUTHORITY_SCHEMA,
        "authority_status": AUTHORITY_STATUS,
        "chronology": CHRONOLOGY,
        "inference_role": INFERENCE_ROLE,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "models": list(MODELS),
            "horizons": list(HORIZONS),
            "analysis_cells": len(summary),
            "stations_per_cell": sorted(summary["n_stations"].unique().tolist()),
            "huc2_clusters_per_cell": EXPECTED_HUC2_CLUSTERS,
            "full_protocol_v2_authority": False,
            "confirmatory_inference": False,
        },
        "inputs": input_bindings,
        "method_definitions": method_definitions,
        "method_definitions_sha256": _canonical_json_sha256(method_definitions),
        "analysis_registry": analysis_registry,
        "analysis_registry_sha256": _canonical_json_sha256(analysis_registry),
        "v2_family_audit": family_audit,
        "v2_family_audit_sha256": _canonical_json_sha256(family_audit),
        "row_counts": {
            "upstream_station_effect_rows": len(paired),
            "summary_rows": len(summary),
            "per_huc2_rows": len(per_huc),
            "leave_one_huc2_rows": len(loco),
        },
        "software": {
            "builder": {**code_binding, "sha256_recomputed": _sha256(code_payload)},
            "git": _git_state(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return publish_authority(
        summary,
        per_huc,
        loco,
        output_dir=destination,
        manifest_base=manifest,
    )


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
        print(f"INFERENCE AUTHORITY BUILD REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"published {CHRONOLOGY} {AUTHORITY_STATUS} authority: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
