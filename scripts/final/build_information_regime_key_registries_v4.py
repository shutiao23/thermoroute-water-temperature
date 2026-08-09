#!/usr/bin/env python3
"""Freeze score-independent v4 primary and F2a-temperature key registries.

This Phase-1 builder has deliberately narrow authority.  It reconstructs the
2021--2023 admissible water-temperature forecast keys from the station
registry and raw holdout panel, proves that their identities and targets agree
with the immutable legacy key authority, corrects the legacy context metadata,
and freezes reportability before any v4 model score is read.  It separately
builds the F2a/F3-temperature-only intersection after validating a canonical
report from the independent Route-A acquisition verifier and every acquired
Parquet block named by its manifest.

F2a is a retrospectively retrieved, fixed-lead target-day air-temperature
composite.  It is not a coherent initialization, an as-issued or operational
forecast, a trajectory, or a substitute for F2b.  Its matched oracle is
``F3_temperature_only`` (target-day Daymet ``TEMP``), never ``F3_full``.

The command-line interface has no implicit production reads.  Production mode
requires explicit paths plus an explicit authorization flag.  Synthetic mode
requires every evidence/data input to be contained below one synthetic root.
Publishing is optional, atomic, exclusive, and create-only; there is no resume
mode and no artifact is named as a completed model matrix.
"""

from __future__ import annotations

import argparse
import calendar
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_NAME = "information_regime_key_registries_v4"

LEADS = (1, 3, 7)
CONTEXT_WINDOWS = (7, 14, 32)
PRIMARY_ISSUE_START = date(2021, 1, 1)
PRIMARY_TARGET_END = date(2023, 12, 31)
F2A_TARGET_START = date(2021, 3, 30)
MINIMUM_PAIRED_TARGETS = 100
Y_ABS_TOLERANCE_C = 2e-6

EXPECTED_PRODUCTION_REGISTRY_STATIONS = 120
EXPECTED_PRODUCTION_RAW_STATIONS = 118
EXPECTED_PRODUCTION_REPORTABLE_STATIONS = 116
EXPECTED_PRODUCTION_EXCLUDED_SITES = ("01435000", "01608500")

EXPECTED_V4_PROTOCOL_ID = "thermoroute_wrr_information_regimes_v4"
EXPECTED_ROUTE_A_PROTOCOL_ID = "route-a-confirmatory-v1"
EXPECTED_VERIFIER_ID = "route-a-f2a-independent-acquisition-verifier-v1"
EXPECTED_PROVIDER = "Open-Meteo Previous Runs API"
EXPECTED_PROVIDER_DOCUMENTATION = "https://open-meteo.com/en/docs/previous-runs-api"
EXPECTED_UPSTREAM_MODEL = "NOAA NCEP GFS global"
EXPECTED_OPEN_METEO_MODEL = "gfs_global"
EXPECTED_SOURCE_VARIABLE = "temperature_2m"
EXPECTED_ISSUE_SEMANTICS = (
    "rolling_fixed_valid_time_minus_lead_offset; daily composite available by "
    "23:59 UTC on issue_date; not a single model-run initialization"
)
EXPECTED_COVERAGE_THRESHOLD = 0.90

HASH_RE = re.compile(r"[0-9a-f]{64}")
SITE_RE = re.compile(r"(?:[0-9]{8}|[0-9]{15})")
FORBIDDEN_COMPLETION_FRAGMENTS = ("partial", "incomplete", "resume")
FORBIDDEN_SCORE_FRAGMENTS = ("score", "prediction", "metric", "effect", "contrast")

PRODUCTION_CANONICAL_FILES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "protocol_v4_draft": (
            "protocols/wrr_information_regimes_protocol_v4.yaml",
            "66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98",
        ),
        "route_a_protocol": (
            "protocols/route_a_confirmatory_v1.json",
            "93c32e9dbfe976eaa7ef31cc5181ae1f4a415ad2b2e30674a5df64c46b968c05",
        ),
        "station_registry_v1": (
            "data_usgs/station_registry_v1.csv",
            "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9",
        ),
        "raw_holdout_panel": (
            "outputs/conventional/panel_2021_2023.parquet",
            "cecdac459139456202240954e4c98fe18bba1fe8b63e9b06ab268683e0d1c03c",
        ),
        "authority_bound_legacy_forecast_registry": (
            "outputs/final/forecast_keys.parquet",
            "56e00e8befe256c9c90a7245cdc892e8d7539982c16fadeadd34d404079a1e0f",
        ),
        "independent_f2a_verification_report": (
            "outputs/final/f2a_acquisition_verification_v1.json",
            "7d8a019d4966a941e9396a40930b2775ba95660347684701838721fe5032a60e",
        ),
        "f2a_acquisition_manifest": (
            "data_usgs/confirmatory_predictors/gfs-previous-runs-v1/manifest.json",
            "54f85eef3ccd4b3cc69071f3c4ed7af5e3208274d3b9062a49c43c96ea43362c",
        ),
        "f2a_snapshot_index": (
            "data_usgs/raw_snapshots/openmeteo-gfs-previous-runs-v1/snapshot_index.json",
            "433f4b4885f225f5f9fe87ec9e94ba81493c07691bb4cc9a994be1931a406771",
        ),
        "requirements_lock": (
            "requirements-lock.txt",
            "ff2d67915ccaabb750cdf4c630d59500d6c8841b305d5fffb1ffd549195dc047",
        ),
        "requirements_lock_py312_hashed": (
            "requirements-lock-py312-hashed.txt",
            "fa325e30e8e69b9e8ec458e9e76a466a765530a8bffa46f1d1f8e34614f4ab76",
        ),
        "pyproject": (
            "pyproject.toml",
            "ef49ffacf3c73a2e566b89abac22d663bdd997a3f68adc5dcf0f5f187860fcc6",
        ),
    }
)
PRODUCTION_ACQUISITION_ROOT = "data_usgs/confirmatory_predictors/gfs-previous-runs-v1"
PRODUCTION_BUILDER_PATH = "scripts/final/build_information_regime_key_registries_v4.py"
PRODUCTION_RUNTIME = MappingProxyType(
    {
        "python_implementation": "CPython",
        "python_version": "3.11.7",
        "python_compiler": "GCC 11.2.0",
        "numpy_version": "1.26.4",
        "pandas_version": "2.1.4",
        "pyarrow_python_version": "14.0.2",
        "arrow_cpp_version": "14.0.2",
    }
)
RUNTIME_STATUS = (
    "CURRENT_RUNTIME_EXACT_PIN_ENFORCED_REPOSITORY_LOCK_MISMATCH_DISCLOSED_"
    "NOT_CROSS_ENV_REPRODUCIBLE"
)
PRODUCTION_EXPECTED_ROWS = MappingProxyType(
    {
        "primary_raw_key_registry_v4.parquet": 358807,
        "primary_reportable_key_registry_v4.parquet": 358765,
        "f2a_temperature_common_key_registry_v4.parquet": 329648,
        "f2a_temperature_reportable_key_registry_v4.parquet": 329628,
    }
)
PRODUCTION_EXPECTED_PARQUET_SHA256 = MappingProxyType(
    {
        "primary_raw_key_registry_v4.parquet": (
            "7d6c5cfa2ae2905fb725e56f8187df809ade200918fbe5c60291ed7861058268"
        ),
        "primary_reportable_key_registry_v4.parquet": (
            "9a135dcffcfd467cf1e2dda4fc711ba6cf66c8ad54bf3b6f799f4a2d5a3c1d24"
        ),
        "primary_station_inventory_v4.parquet": (
            "d06e9bd51e15d704bce7c96f678bfd2d08b656483345ce9988e50cda438d0783"
        ),
        "f2a_temperature_common_key_registry_v4.parquet": (
            "1861b9cc81551bbcfc408b83f9c7184de730017982a40e03bcfc65ef16ab27a7"
        ),
        "f2a_temperature_reportable_key_registry_v4.parquet": (
            "5cc75fda17abe44cd5201184d156c5a53726960d78182b726edf0620895754b6"
        ),
        "f2a_temperature_station_inventory_v4.parquet": (
            "0ef0cc3bebe70179e691250689d28b1bb077393b95255cc341691a8089692ee4"
        ),
    }
)

STATION_REGISTRY_COLUMNS = (
    "site_no",
    "legacy_site_id",
    "ok",
    "wtemp_cov",
    "flow_cov",
    "wtemp_cov_test",
    "flow_cov_test",
    "wlevel_cov",
    "n_days",
    "station_nm",
    "lat",
    "lon",
    "state",
    "huc_cd",
    "huc2",
    "drain_area_va",
    "huc_metadata_status",
)
PANEL_COLUMNS = (
    "DATE",
    "site_id",
    "WTEMP",
    "FLOW",
    "WLEVEL",
    "TEMP",
    "PRCP",
    "WDSP",
    "RHMEAN",
    "DH",
)
LEGACY_REGISTRY_COLUMNS = (
    "key_id",
    "cohort",
    "task",
    "site_id",
    "issue_date",
    "target_date",
    "horizon",
    "period",
    "target_observed",
    "y_true",
    "issue_wtemp_observed",
    "days_since_last_observed_wtemp",
    "n_observed_wtemp_7d",
    "fraction_observed_wtemp_7d",
    "n_observed_wtemp_14d",
    "fraction_observed_wtemp_14d",
    "n_observed_wtemp_32d",
    "fraction_observed_wtemp_32d",
)
F2A_PARQUET_COLUMNS = (
    "site_no",
    "requested_lat",
    "requested_lon",
    "horizon",
    "issue_date",
    "target_date",
    "air_temp_2m_mean_c",
    "available_hour_count",
    "complete_target_day",
    "lead_field",
    "source_provider",
    "upstream_model",
    "open_meteo_model",
    "issue_semantics",
    "response_grid_lat",
    "response_grid_lon",
    "request_sha256",
    "response_sha256",
)

ACQUISITION_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_role",
    "protocol",
    "protocol_sha256",
    "source_provider",
    "source_documentation",
    "upstream_model",
    "open_meteo_model",
    "source_variable",
    "lead_days",
    "lead_fields",
    "issue_semantics",
    "timezone",
    "daily_aggregation",
    "archive_run_start",
    "common_valid_target_start",
    "target_end",
    "station_count",
    "row_count",
    "complete_row_count",
    "request_partition",
    "immutable_outputs",
    "consumed_by_primary_route_a_models",
    "primary_evaluation_dependency",
    "labels_requested_or_read",
    "outcome_endpoint_called",
    "registry_inputs",
    "raw_snapshot_index",
    "raw_snapshot_index_sha256",
    "chunks",
}
CHUNK_FIELDS = {
    "schema_version",
    "artifact",
    "artifact_sha256",
    "site_no",
    "chunk_start",
    "chunk_end",
    "row_count",
    "complete_row_count",
    "request_sha256",
    "response_sha256",
    "retrieved_at_utc",
    "labels_requested_or_read",
    "protocol_sha256",
}
VERIFICATION_REPORT_FIELDS = {
    "schema_version",
    "verifier_id",
    "status",
    "integrity_verification_passed",
    "coverage_gate_threshold",
    "coverage_gate_passed",
    "arm_promoted",
    "promotion_requires_separate_protocol_authorization",
    "contains_or_reads_outcome_labels",
    "input_access_boundary",
    "information_semantics",
    "inventory",
    "coverage_by_station_and_lead",
    "coverage_by_lead",
    "evidence_hashes",
}

PRIMARY_RAW_FILENAME = "primary_raw_key_registry_v4.parquet"
PRIMARY_REPORTABLE_FILENAME = "primary_reportable_key_registry_v4.parquet"
PRIMARY_STATIONS_FILENAME = "primary_station_inventory_v4.parquet"
F2A_COMMON_FILENAME = "f2a_temperature_common_key_registry_v4.parquet"
F2A_REPORTABLE_FILENAME = "f2a_temperature_reportable_key_registry_v4.parquet"
F2A_STATIONS_FILENAME = "f2a_temperature_station_inventory_v4.parquet"
MANIFEST_FILENAME = "information_regime_key_registries_v4_manifest.json"


class KeyRegistryError(RuntimeError):
    """Raised when any evidence, identity, or publication gate fails closed."""


@dataclass(frozen=True)
class RegistryBuildConfig:
    """Frozen dimensions, with explicit seams only for synthetic contract tests."""

    mode: str
    primary_issue_start: date = PRIMARY_ISSUE_START
    primary_target_end: date = PRIMARY_TARGET_END
    f2a_target_start: date = F2A_TARGET_START
    leads: tuple[int, ...] = LEADS
    minimum_targets: int = MINIMUM_PAIRED_TARGETS
    coverage_threshold: float = EXPECTED_COVERAGE_THRESHOLD
    expected_registry_stations: int | None = EXPECTED_PRODUCTION_REGISTRY_STATIONS
    expected_raw_stations: int | None = EXPECTED_PRODUCTION_RAW_STATIONS
    expected_reportable_stations: int | None = EXPECTED_PRODUCTION_REPORTABLE_STATIONS
    expected_excluded_sites: tuple[str, ...] | None = EXPECTED_PRODUCTION_EXCLUDED_SITES

    def __post_init__(self) -> None:
        _require(self.mode in {"production", "synthetic"}, "invalid build mode")
        _require(self.leads == LEADS, f"lead inventory must be exactly {LEADS}")
        _require(self.minimum_targets >= 1, "minimum_targets must be positive")
        _require(
            self.primary_issue_start <= self.f2a_target_start <= self.primary_target_end,
            "date windows are inconsistent",
        )
        _require(
            math.isclose(self.coverage_threshold, 0.90, abs_tol=0.0, rel_tol=0.0),
            "F2a coverage threshold must remain exactly 0.90",
        )

    @classmethod
    def production(cls) -> RegistryBuildConfig:
        return cls(mode="production")


@dataclass(frozen=True)
class RegistryInputPaths:
    protocol_v4: Path
    route_a_protocol: Path
    station_registry: Path
    raw_holdout_panel: Path
    legacy_forecast_registry: Path
    f2a_verification_report: Path
    f2a_acquisition_manifest: Path
    f2a_acquisition_root: Path


@dataclass(frozen=True)
class BoundBytes:
    path: Path
    sha256: str
    size_bytes: int
    payload: bytes


@dataclass(frozen=True)
class RegistryArtifactBundle:
    files: Mapping[str, bytes]
    manifest_payload: bytes
    primary_raw: pd.DataFrame
    primary_reportable: pd.DataFrame
    primary_station_inventory: pd.DataFrame
    f2a_common: pd.DataFrame
    f2a_reportable: pd.DataFrame
    f2a_station_inventory: pd.DataFrame

    def __post_init__(self) -> None:
        immutable_files: dict[str, bytes] = {}
        for name, payload in self.files.items():
            if type(name) is not str or type(payload) is not bytes:
                raise KeyRegistryError("artifact payload mapping must be str -> immutable bytes")
            immutable_files[name] = payload
        object.__setattr__(self, "files", MappingProxyType(immutable_files))
        if type(self.manifest_payload) is not bytes:
            raise KeyRegistryError("manifest payload must be immutable bytes")

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a fresh parse; no mutable manifest object is retained."""
        return _strict_json(
            self.manifest_payload,
            label="in-memory registry manifest",
            canonical=True,
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KeyRegistryError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_hash(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and HASH_RE.fullmatch(value) is not None, f"{label} invalid")
    return str(value)


def _absolute_without_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_contained_path(path: Path, root: Path, *, label: str) -> Path:
    """Reject lexical escapes and every symlink component before a read."""
    absolute = _absolute_without_resolve(path)
    base = _absolute_without_resolve(root)
    try:
        relative = absolute.relative_to(base)
    except ValueError as exc:
        raise KeyRegistryError(f"{label} escapes containment root: {path}") from exc
    _require(".." not in relative.parts, f"{label} path is not contained")
    current = base
    try:
        root_stat = current.lstat()
    except OSError as exc:
        raise KeyRegistryError(f"containment root is absent: {base}") from exc
    _require(not stat.S_ISLNK(root_stat.st_mode), f"containment root is a symlink: {base}")
    for part in relative.parts:
        current = current / part
        try:
            member_stat = current.lstat()
        except OSError as exc:
            raise KeyRegistryError(f"{label} is absent: {current}") from exc
        _require(not stat.S_ISLNK(member_stat.st_mode), f"{label} crosses a symlink: {current}")
    return absolute


def _read_stable_regular(path: Path, *, root: Path, label: str) -> BoundBytes:
    absolute = _assert_contained_path(path, root, label=label)
    before = absolute.lstat()
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {absolute}")
    _require(before.st_nlink == 1, f"{label} must have exactly one hard link: {absolute}")
    try:
        with absolute.open("rb") as handle:
            payload = handle.read()
        after = absolute.lstat()
    except OSError as exc:
        raise KeyRegistryError(f"{label} changed while read: {absolute}") from exc
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    _require(
        all(getattr(before, field) == getattr(after, field) for field in identity_fields)
        and len(payload) == before.st_size,
        f"{label} changed while read: {absolute}",
    )
    return BoundBytes(absolute, _sha256_bytes(payload), len(payload), payload)


def _strict_json(payload: bytes, *, label: str, canonical: bool) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise KeyRegistryError(f"{label} is not strict JSON") from exc
    _require(isinstance(document, dict), f"{label} top level must be an object")
    if canonical:
        _require(_canonical_json_bytes(document) == payload, f"{label} is not canonical JSON")
    return document


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise KeyRegistryError(f"protocol YAML duplicates key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _strict_yaml(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise KeyRegistryError(f"{label} is not strict YAML") from exc
    _require(isinstance(document, dict), f"{label} top level must be a mapping")
    return document


def _repo_relative(path: Path, repository_root: Path) -> str:
    absolute = _absolute_without_resolve(path)
    root = _absolute_without_resolve(repository_root)
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        return absolute.as_posix()


def _binding(bound: BoundBytes, repository_root: Path) -> dict[str, object]:
    return {
        "path": _repo_relative(bound.path, repository_root),
        "sha256": bound.sha256,
        "size_bytes": bound.size_bytes,
    }


def _validate_canonical_path_hash_contract(
    observed_files: Mapping[str, Mapping[str, object]],
    *,
    expected_files: Mapping[str, tuple[str, str]],
    observed_acquisition_root: str,
    expected_acquisition_root: str,
    observed_builder_path: str,
    expected_builder_path: str,
) -> None:
    """Pure path/hash gate, factored so synthetic tests never read production."""
    _require(
        set(observed_files) == set(expected_files),
        "canonical production binding roles changed",
    )
    for role, (expected_path, expected_sha256) in expected_files.items():
        record = observed_files[role]
        _require(
            set(record) == {"path", "sha256"},
            f"canonical production binding schema changed for {role}",
        )
        _require(
            record.get("path") == expected_path, f"canonical production path changed for {role}"
        )
        _require(
            record.get("sha256") == expected_sha256,
            f"canonical production SHA-256 changed for {role}",
        )
    _require(
        observed_acquisition_root == expected_acquisition_root,
        "canonical production acquisition root changed",
    )
    _require(
        observed_builder_path == expected_builder_path,
        "canonical production builder path changed",
    )


def _validate_production_paths_before_read(
    paths: RegistryInputPaths,
    *,
    repository_root: Path,
) -> None:
    """Reject alternate production inputs before opening any evidence bytes."""
    expected_by_attribute = {
        "protocol_v4": PRODUCTION_CANONICAL_FILES["protocol_v4_draft"][0],
        "route_a_protocol": PRODUCTION_CANONICAL_FILES["route_a_protocol"][0],
        "station_registry": PRODUCTION_CANONICAL_FILES["station_registry_v1"][0],
        "raw_holdout_panel": PRODUCTION_CANONICAL_FILES["raw_holdout_panel"][0],
        "legacy_forecast_registry": PRODUCTION_CANONICAL_FILES[
            "authority_bound_legacy_forecast_registry"
        ][0],
        "f2a_verification_report": PRODUCTION_CANONICAL_FILES[
            "independent_f2a_verification_report"
        ][0],
        "f2a_acquisition_manifest": PRODUCTION_CANONICAL_FILES["f2a_acquisition_manifest"][0],
        "f2a_acquisition_root": PRODUCTION_ACQUISITION_ROOT,
    }
    root = _absolute_without_resolve(repository_root)
    for attribute, relative in expected_by_attribute.items():
        observed = _absolute_without_resolve(getattr(paths, attribute))
        expected = _absolute_without_resolve(root / relative)
        _require(observed == expected, f"canonical production path changed for {attribute}")
    _require(
        _absolute_without_resolve(Path(__file__))
        == _absolute_without_resolve(root / PRODUCTION_BUILDER_PATH),
        "canonical production builder path changed",
    )


def _current_runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "pyarrow_python_version": str(pa.__version__),
        "arrow_cpp_version": str(pa.cpp_version),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "byteorder": os.sys.byteorder,
    }


def _runtime_evidence(
    *,
    requirements_lock: BoundBytes,
    requirements_lock_py312: BoundBytes,
    pyproject: BoundBytes,
    repository_root: Path,
    production: bool,
) -> dict[str, Any]:
    active = _current_runtime_identity()
    if production:
        observed_pin = {key: active[key] for key in PRODUCTION_RUNTIME}
        _require(
            observed_pin == dict(PRODUCTION_RUNTIME),
            "production Parquet runtime differs from the exact current-runtime pin",
        )
        _require(
            active["platform_system"] == "Linux"
            and active["platform_machine"] == "x86_64"
            and active["byteorder"] == "little",
            "production platform differs from the audited Parquet runtime",
        )
    references = {
        "requirements_lock": _binding(requirements_lock, repository_root),
        "requirements_lock_py312_hashed": _binding(requirements_lock_py312, repository_root),
        "pyproject": _binding(pyproject, repository_root),
    }
    return {
        "status": RUNTIME_STATUS,
        "determinism_class": "PINNED_RUNTIME_DETERMINISTIC",
        "active_runtime": active,
        "production_exact_runtime_pin": dict(PRODUCTION_RUNTIME),
        "production_platform_pin": {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "byteorder": "little",
        },
        "byte_identity_scope": (
            "deterministic only under the exact active runtime/platform pin; "
            "cross-environment Parquet byte identity is explicitly not claimed"
        ),
        "repository_dependency_references": references,
        "repository_locks_are_active_runtime_authority": False,
        "repository_locks_match_active_runtime": False,
        "disclosed_lock_mismatches": [
            "both repository locks target Python 3.12 while the active runtime is CPython 3.11.7",
            "both repository locks declare pandas 2.2.2 while the active runtime is pandas 2.1.4",
            "both repository locks declare pyarrow 24.0.0 while the active Python/Arrow runtime is 14.0.2",
            "numpy 1.26.4 matches but does not make either complete lock match the active runtime",
        ],
    }


def _relative_evidence_path(value: object, *, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and value != "", f"{label} path invalid")
    path = PurePosixPath(str(value))
    _require(not path.is_absolute(), f"{label} path must be repository-relative")
    _require(".." not in path.parts and "." not in path.parts, f"{label} path escapes root")
    return path


def _resolve_evidence_path(value: object, *, repository_root: Path, label: str) -> Path:
    relative = _relative_evidence_path(value, label=label)
    return _absolute_without_resolve(repository_root / Path(*relative.parts))


def _validate_exact_columns(frame: pd.DataFrame, expected: Sequence[str], *, label: str) -> None:
    _require(list(frame.columns) == list(expected), f"{label} schema changed")
    forbidden = [column for column in frame.columns if "f3_full" in str(column).lower()]
    _require(not forbidden, f"{label} contains forbidden F3_full fields: {forbidden}")


def _parse_parquet(bound: BoundBytes, *, expected: Sequence[str], label: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(io.BytesIO(bound.payload))
    except Exception as exc:
        raise KeyRegistryError(f"{label} is not readable Parquet") from exc
    _validate_exact_columns(frame, expected, label=label)
    return frame


def _normalise_sites(values: pd.Series, *, label: str) -> pd.Series:
    sites = values.astype("string")
    _require(not sites.isna().any(), f"{label} contains missing site identity")
    _require(
        sites.eq(sites.str.strip()).all(),
        f"{label} contains surrounding whitespace",
    )
    _require(
        sites.map(lambda value: SITE_RE.fullmatch(str(value)) is not None).all(),
        f"{label} must contain exactly 8 or 15 decimal digits",
    )
    return sites.astype(str)


def _normalise_dates(values: pd.Series, *, label: str) -> pd.Series:
    try:
        dates = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise KeyRegistryError(f"{label} contains invalid dates") from exc
    _require(getattr(dates.dt, "tz", None) is None, f"{label} must be timezone-naive")
    _require(dates.dt.normalize().equals(dates), f"{label} must contain midnight daily dates")
    return dates.astype("datetime64[ns]")


def _key_id(site: str, issue: pd.Timestamp, target: pd.Timestamp, lead: int) -> str:
    payload = f"temporal|known_site|{site}|{issue:%Y-%m-%d}|{target:%Y-%m-%d}|{lead}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_v4_protocol(document: Mapping[str, Any], config: RegistryBuildConfig) -> None:
    _require(document.get("protocol_id") == EXPECTED_V4_PROTOCOL_ID, "wrong v4 protocol id")
    _require(document.get("version") == 4, "wrong v4 protocol version")
    _require(document.get("status") == "DRAFT_NOT_SEALED", "v4 protocol is not the bound draft")
    _require(document.get("execution_authorized") is False, "draft cannot authorize execution")
    _require(document.get("seal_path") is None, "draft unexpectedly has a seal")
    axes = document.get("axes")
    _require(isinstance(axes, dict), "v4 protocol lacks axes")
    forcing = axes.get("F_future_forcing")
    _require(isinstance(forcing, dict), "v4 protocol lacks forcing axis")
    f2a = forcing.get("F2a")
    _require(isinstance(f2a, dict), "v4 protocol lacks F2a contract")
    _require(f2a.get("coherent_single_initialization") is False, "F2a coherence changed")
    _require(f2a.get("recovery_denominator") == "F3_temperature_only", "F2a denominator changed")
    scope = f2a.get("frozen_acquisition_scope")
    _require(isinstance(scope, dict), "v4 protocol lacks F2a acquisition scope")
    _require(scope.get("lead_days") == list(config.leads), "F2a protocol leads changed")
    _require(
        scope.get("value_per_forecast_key") == "target-day daily-mean air temperature",
        "F2a target-day value contract changed",
    )
    _require(scope.get("all_intermediate_leads_acquired") is False, "F2a trajectory claim changed")
    common = document.get("common_key_and_reportability_contract")
    _require(isinstance(common, dict), "v4 protocol lacks common-key contract")
    _require(
        common.get("key_fields") == ["site_id", "issue_time", "target_time", "lead_days"],
        "v4 key identity changed",
    )
    reportability = common.get("reportability")
    _require(isinstance(reportability, dict), "v4 reportability contract missing")
    _require(
        reportability.get("minimum_paired_targets_per_station_lead") == config.minimum_targets,
        "v4 reportability threshold changed",
    )
    history = document.get("history_quality_sensitivity")
    _require(isinstance(history, dict), "v4 history-quality contract missing")
    _require(history.get("window_days") == 32, "history window changed")
    _require(history.get("raw_observation_basis") == "before_imputation", "history basis changed")


def _validate_route_a_protocol(document: Mapping[str, Any], config: RegistryBuildConfig) -> None:
    _require(
        document.get("protocol_id") == EXPECTED_ROUTE_A_PROTOCOL_ID, "wrong Route-A protocol id"
    )
    time_holdout = document.get("time_holdout")
    _require(isinstance(time_holdout, dict), "Route-A protocol lacks time holdout")
    _require(
        time_holdout.get("secondary_nwp_common_lead_target_start")
        == config.f2a_target_start.isoformat(),
        "Route-A F2a start changed",
    )
    _require(
        time_holdout.get("end") == config.primary_target_end.isoformat(),
        "Route-A target end changed",
    )
    contract = document.get("secondary_archived_nwp_contract")
    _require(isinstance(contract, dict), "Route-A protocol lacks F2a contract")
    required = {
        "contains_outcome_labels": False,
        "consumed_by_primary_models": False,
        "primary_evaluation_dependency": False,
        "provider": EXPECTED_PROVIDER,
        "api_endpoint": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "upstream_model": EXPECTED_UPSTREAM_MODEL,
        "open_meteo_model_parameter": EXPECTED_OPEN_METEO_MODEL,
        "variable": EXPECTED_SOURCE_VARIABLE,
        "unit": "degrees_C",
        "lead_days": list(config.leads),
        "secondary_common_lead_target_start": config.f2a_target_start.isoformat(),
        "raw_snapshot_required": True,
        "derived_blocks_may_be_overwritten": False,
        "selection_or_tuning_from_predictor_availability": False,
    }
    _require(
        all(contract.get(key) == value for key, value in required.items()),
        "Route-A F2a semantics changed",
    )
    lead_semantics = str(contract.get("lead_semantics", "")).lower()
    _require(
        "rolling fixed-lead composite" in lead_semantics
        and "not the output of one identified model initialization" in lead_semantics,
        "Route-A protocol no longer preserves fixed-lead composite semantics",
    )


def _load_station_registry(bound: BoundBytes) -> pd.DataFrame:
    try:
        registry = pd.read_csv(
            io.BytesIO(bound.payload),
            dtype={
                "site_no": "string",
                "legacy_site_id": "string",
                "huc_cd": "string",
                "huc2": "string",
            },
            float_precision="round_trip",
        )
    except Exception as exc:
        raise KeyRegistryError("station_registry_v1 is not readable CSV") from exc
    _validate_exact_columns(registry, STATION_REGISTRY_COLUMNS, label="station_registry_v1")
    registry = registry.copy()
    registry["site_no"] = _normalise_sites(registry["site_no"], label="station registry site_no")
    _require(not registry["site_no"].duplicated().any(), "station registry duplicates site_no")
    for column in ("lat", "lon"):
        numeric = pd.to_numeric(registry[column], errors="coerce")
        _require(
            np.isfinite(numeric.to_numpy(dtype=float)).all(), f"station registry {column} invalid"
        )
        registry[column] = numeric.astype(float)
    return registry.sort_values("site_no").reset_index(drop=True)


def _load_panel(bound: BoundBytes, *, config: RegistryBuildConfig) -> pd.DataFrame:
    panel = _parse_parquet(bound, expected=PANEL_COLUMNS, label="raw holdout panel")
    panel = panel.copy()
    panel["site_id"] = _normalise_sites(panel["site_id"], label="panel site_id")
    panel["DATE"] = _normalise_dates(panel["DATE"], label="panel DATE")
    _require(not panel.duplicated(["site_id", "DATE"]).any(), "raw panel duplicates site/date")
    for column in PANEL_COLUMNS[2:]:
        _require(pd.api.types.is_numeric_dtype(panel[column]), f"panel {column} must be numeric")
    needed_start = pd.Timestamp(config.primary_issue_start) - pd.Timedelta(
        days=max(CONTEXT_WINDOWS) - 1
    )
    needed_end = pd.Timestamp(config.primary_target_end)
    panel = panel[(panel["DATE"] >= needed_start) & (panel["DATE"] <= needed_end)].copy()
    _require(not panel.empty, "raw holdout panel has no rows in the required window")
    return panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)


def _validate_panel_calendar(
    panel: pd.DataFrame,
    station_sites: Sequence[str],
    *,
    config: RegistryBuildConfig,
) -> None:
    expected_dates = pd.date_range(
        pd.Timestamp(config.primary_issue_start) - pd.Timedelta(days=max(CONTEXT_WINDOWS) - 1),
        pd.Timestamp(config.primary_target_end),
        freq="D",
    )
    observed_sites = set(panel["site_id"])
    expected_sites = set(station_sites)
    _require(observed_sites == expected_sites, "panel/station-registry site sets differ")
    expected_count = len(expected_dates)
    for site, group in panel.groupby("site_id", sort=True):
        dates = pd.DatetimeIndex(group["DATE"])
        _require(
            len(dates) == expected_count and dates.equals(expected_dates),
            f"raw panel lacks the exact daily context/target calendar for {site}",
        )


def _load_legacy_registry(bound: BoundBytes, station_sites: set[str]) -> pd.DataFrame:
    legacy = _parse_parquet(
        bound, expected=LEGACY_REGISTRY_COLUMNS, label="legacy forecast-key registry"
    ).copy()
    legacy["site_id"] = _normalise_sites(legacy["site_id"], label="legacy site_id")
    legacy["issue_date"] = _normalise_dates(legacy["issue_date"], label="legacy issue_date")
    legacy["target_date"] = _normalise_dates(legacy["target_date"], label="legacy target_date")
    _require(pd.api.types.is_integer_dtype(legacy["horizon"]), "legacy horizon dtype changed")
    legacy["horizon"] = legacy["horizon"].astype(int)
    keys = ["site_id", "horizon", "issue_date", "target_date"]
    _require(not legacy.duplicated(keys).any(), "legacy registry duplicates forecast identity")
    _require(not legacy["key_id"].duplicated().any(), "legacy registry duplicates key_id")
    _require(set(legacy["horizon"]) == set(LEADS), "legacy registry lead inventory changed")
    _require(set(legacy["site_id"]).issubset(station_sites), "legacy registry has unknown sites")
    _require(legacy["cohort"].eq("temporal").all(), "legacy cohort identity changed")
    _require(legacy["task"].eq("known_site").all(), "legacy task identity changed")
    _require(legacy["period"].eq("2021-2023").all(), "legacy period identity changed")
    _require(legacy["target_observed"].eq(True).all(), "legacy target-observed flag changed")
    _require(
        np.isfinite(legacy["y_true"].to_numpy(dtype=float)).all(), "legacy y_true is non-finite"
    )
    expected_target = legacy["issue_date"] + pd.to_timedelta(legacy["horizon"], unit="D")
    _require(expected_target.equals(legacy["target_date"]), "legacy target chronology changed")
    expected_ids = [
        _key_id(str(row.site_id), row.issue_date, row.target_date, int(row.horizon))
        for row in legacy.itertuples(index=False)
    ]
    _require(legacy["key_id"].tolist() == expected_ids, "legacy key_id identity changed")
    return legacy.sort_values(keys).reset_index(drop=True)


def _reconstruct_primary_keys(panel: pd.DataFrame, *, config: RegistryBuildConfig) -> pd.DataFrame:
    target_lookup = panel[["site_id", "DATE", "WTEMP"]].rename(
        columns={"DATE": "target_date", "WTEMP": "panel_y_true"}
    )
    issue = panel[
        (panel["DATE"] >= pd.Timestamp(config.primary_issue_start))
        & (panel["DATE"] <= pd.Timestamp(config.primary_target_end))
        & np.isfinite(panel["WTEMP"].to_numpy(dtype=float))
    ][["site_id", "DATE"]].rename(columns={"DATE": "issue_date"})
    pieces: list[pd.DataFrame] = []
    for lead in config.leads:
        candidate = issue.copy()
        candidate["lead_days"] = int(lead)
        candidate["target_date"] = candidate["issue_date"] + pd.Timedelta(days=lead)
        candidate = candidate[candidate["target_date"] <= pd.Timestamp(config.primary_target_end)]
        candidate = candidate.merge(
            target_lookup,
            on=["site_id", "target_date"],
            how="left",
            validate="many_to_one",
        )
        candidate = candidate[np.isfinite(candidate["panel_y_true"].to_numpy(dtype=float))].copy()
        pieces.append(candidate)
    _require(pieces, "no primary lead inventory")
    reconstructed = pd.concat(pieces, ignore_index=True)
    keys = ["site_id", "lead_days", "issue_date", "target_date"]
    reconstructed = reconstructed.sort_values(keys).reset_index(drop=True)
    _require(not reconstructed.empty, "reconstructed primary registry is empty")
    _require(not reconstructed.duplicated(keys).any(), "reconstructed primary keys duplicate")
    chronology = reconstructed["issue_date"] + pd.to_timedelta(reconstructed["lead_days"], unit="D")
    _require(chronology.equals(reconstructed["target_date"]), "reconstructed chronology failed")
    return reconstructed


def _build_context_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute raw calendar-window context and strictly as-of recency.

    This helper intentionally handles issue dates without observed WTEMP too,
    which lets tests prove that a future station-wide observation cannot leak
    backwards.  The primary admissibility filter later retains only rows with
    finite issue-date WTEMP.
    """
    required = {"site_id", "DATE", "WTEMP"}
    _require(required.issubset(panel.columns), "context panel schema incomplete")
    ordered = panel[["site_id", "DATE", "WTEMP"]].copy()
    ordered["site_id"] = _normalise_sites(ordered["site_id"], label="context site_id")
    ordered["DATE"] = _normalise_dates(ordered["DATE"], label="context DATE")
    ordered = ordered.sort_values(["site_id", "DATE"]).reset_index(drop=True)
    _require(not ordered.duplicated(["site_id", "DATE"]).any(), "context panel duplicates dates")
    ordered["issue_wtemp_observed"] = np.isfinite(ordered["WTEMP"].to_numpy(dtype=float))
    observed_dates = ordered["DATE"].where(ordered["issue_wtemp_observed"])
    last_observed_asof = observed_dates.groupby(ordered["site_id"], sort=False).ffill()
    recency = (ordered["DATE"] - last_observed_asof).dt.days
    _require(
        recency.dropna().ge(0).all(),
        "as-of recency became negative (future-observation leakage)",
    )
    ordered["days_since_last_observed_wtemp"] = recency.fillna(-1).astype("int32")
    for window in CONTEXT_WINDOWS:
        counts = ordered.groupby("site_id", sort=False)["issue_wtemp_observed"].transform(
            lambda series, width=window: series.rolling(width, min_periods=width).sum()
        )
        ordered[f"n_observed_wtemp_{window}d"] = counts
        ordered[f"fraction_observed_wtemp_{window}d"] = counts / float(window)
    return ordered.drop(columns=["WTEMP"])


def _attach_corrected_context(keys: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    context = _build_context_table(panel)
    out = keys.merge(
        context,
        left_on=["site_id", "issue_date"],
        right_on=["site_id", "DATE"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["DATE"])
    required_context = [
        "issue_wtemp_observed",
        "days_since_last_observed_wtemp",
        *[
            item
            for window in CONTEXT_WINDOWS
            for item in (
                f"n_observed_wtemp_{window}d",
                f"fraction_observed_wtemp_{window}d",
            )
        ],
    ]
    _require(not out[required_context].isna().any().any(), "context window is incomplete")
    _require(out["issue_wtemp_observed"].eq(True).all(), "issue observability drifted")
    _require(
        out["days_since_last_observed_wtemp"].eq(0).all(), "eligible issue recency is not zero"
    )
    for window in CONTEXT_WINDOWS:
        counts = out[f"n_observed_wtemp_{window}d"]
        _require(counts.between(0, window).all(), f"{window}-day raw count outside bounds")
        expected = counts / float(window)
        _require(
            np.array_equal(
                expected.to_numpy(dtype=float),
                out[f"fraction_observed_wtemp_{window}d"].to_numpy(dtype=float),
            ),
            f"{window}-day raw fraction changed",
        )
        out[f"n_observed_wtemp_{window}d"] = counts.astype("int16")
    out["history_H100"] = out["fraction_observed_wtemp_32d"].eq(1.0)
    out["history_H75"] = out["fraction_observed_wtemp_32d"].ge(0.75)
    out["history_Hall"] = True
    return out


def _match_legacy_targets(
    reconstructed: pd.DataFrame,
    legacy: pd.DataFrame,
    *,
    tolerance: float = Y_ABS_TOLERANCE_C,
) -> tuple[pd.DataFrame, float]:
    keys_left = ["site_id", "lead_days", "issue_date", "target_date"]
    legacy_view = legacy.rename(columns={"horizon": "lead_days"})[["key_id", *keys_left, "y_true"]]
    joined = reconstructed.merge(
        legacy_view,
        on=keys_left,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = joined["_merge"].value_counts()
    _require(
        int(counts.get("left_only", 0)) == 0 and int(counts.get("right_only", 0)) == 0,
        "reconstructed and legacy forecast-key identities differ",
    )
    joined = joined.drop(columns=["_merge"])
    differences = np.abs(
        joined["panel_y_true"].to_numpy(dtype=float) - joined["y_true"].to_numpy(dtype=float)
    )
    max_difference = float(differences.max(initial=0.0))
    _require(
        np.isclose(
            joined["panel_y_true"].to_numpy(dtype=float),
            joined["y_true"].to_numpy(dtype=float),
            atol=tolerance,
            rtol=0.0,
        ).all(),
        f"panel/legacy y_true exceeds absolute {tolerance:g} C tolerance",
    )
    # Persist the legacy authority's exact floating representation, not the
    # independently reconstructed panel representation.
    joined = joined.drop(columns=["panel_y_true"])
    expected_ids = [
        _key_id(str(row.site_id), row.issue_date, row.target_date, int(row.lead_days))
        for row in joined.itertuples(index=False)
    ]
    _require(joined["key_id"].tolist() == expected_ids, "joined key_id identity changed")
    return joined, max_difference


def _freeze_primary_reportability(
    keys: pd.DataFrame,
    *,
    config: RegistryBuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forbidden = [
        column
        for column in keys.columns
        if any(x in column.lower() for x in FORBIDDEN_SCORE_FRAGMENTS)
    ]
    _require(not forbidden, f"reportability input contains score-like columns: {forbidden}")
    counts = keys.groupby(["site_id", "lead_days"], sort=True).size().rename("n_keys").reset_index()
    sites = sorted(keys["site_id"].unique())
    expected_index = pd.MultiIndex.from_product(
        [sites, config.leads], names=["site_id", "lead_days"]
    )
    inventory = (
        counts.set_index(["site_id", "lead_days"])
        .reindex(expected_index, fill_value=0)
        .reset_index()
    )
    inventory["n_keys"] = inventory["n_keys"].astype("int32")
    inventory["meets_minimum_this_lead"] = inventory["n_keys"].ge(config.minimum_targets)
    included = inventory.groupby("site_id", sort=True)["meets_minimum_this_lead"].all()
    inventory["included_all_leads"] = inventory["site_id"].map(included).astype(bool)
    inventory["minimum_targets"] = int(config.minimum_targets)
    reportable_sites = set(included[included].index)
    raw = keys.copy()
    raw["reportable_primary"] = raw["site_id"].isin(reportable_sites)
    reportable = raw[raw["reportable_primary"]].copy()
    raw_count = int(raw["site_id"].nunique())
    reportable_count = len(reportable_sites)
    excluded = tuple(sorted(set(sites) - reportable_sites))
    if config.expected_raw_stations is not None:
        _require(raw_count == config.expected_raw_stations, "raw primary station count changed")
    if config.expected_reportable_stations is not None:
        _require(
            reportable_count == config.expected_reportable_stations,
            "reportable primary station count changed",
        )
    if config.expected_excluded_sites is not None:
        _require(
            excluded == config.expected_excluded_sites, "primary excluded-site identity changed"
        )
    _require(not reportable.empty, "no primary station passes reportability")
    _require(
        set(reportable["site_id"]) == reportable_sites,
        "primary reportable station freeze failed",
    )
    return raw, reportable, inventory


def _month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        month_end = date(cursor.year, cursor.month, last_day)
        chunk_start = max(cursor, start)
        chunk_end = min(month_end, end)
        chunks.append((chunk_start, chunk_end))
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    return chunks


def _validate_binding_record(
    record: object,
    *,
    repository_root: Path,
    containment_root: Path,
    label: str,
) -> BoundBytes:
    _require(
        isinstance(record, dict)
        and set(record) in ({"path", "sha256"}, {"path", "sha256", "size_bytes"}),
        f"{label} binding schema changed",
    )
    path = _resolve_evidence_path(record.get("path"), repository_root=repository_root, label=label)
    bound = _read_stable_regular(path, root=containment_root, label=label)
    _require(
        bound.sha256 == _assert_hash(record.get("sha256"), label=f"{label} sha256"),
        f"{label} hash mismatch",
    )
    if "size_bytes" in record:
        _require(record.get("size_bytes") == bound.size_bytes, f"{label} size mismatch")
    return bound


def _validate_lineage_record(
    record: object,
    *,
    repository_root: Path,
    containment_root: Path,
    expected_bound: BoundBytes,
    expected_rows: int,
    label: str,
) -> None:
    _require(
        isinstance(record, dict) and set(record) == {"path", "sha256", "columns_read", "row_count"},
        f"{label} lineage schema changed",
    )
    expected_path = _repo_relative(expected_bound.path, repository_root)
    _require(record.get("path") == expected_path, f"{label} path binding changed")
    _require(record.get("sha256") == expected_bound.sha256, f"{label} checksum changed")
    _require(record.get("columns_read") == ["site_no", "lat", "lon"], f"{label} columns changed")
    _require(record.get("row_count") == expected_rows, f"{label} row count changed")
    # Re-resolve the path rather than trusting string equality alone.  The
    # bytes were already captured once above; do not open the same input a
    # second time and create a second, weaker TOCTOU window.
    path = _resolve_evidence_path(record.get("path"), repository_root=repository_root, label=label)
    _require(path == expected_bound.path, f"{label} does not resolve to the bound input")
    _require(
        _absolute_without_resolve(path).is_relative_to(_absolute_without_resolve(containment_root)),
        f"{label} escapes containment root",
    )


def _validate_acquisition_manifest(
    document: Mapping[str, Any],
    *,
    manifest_bound: BoundBytes,
    route_protocol_bound: BoundBytes,
    station_registry_bound: BoundBytes,
    station_count: int,
    repository_root: Path,
    input_root: Path,
    acquisition_root: Path,
    config: RegistryBuildConfig,
) -> tuple[list[dict[str, Any]], BoundBytes]:
    _require(
        set(document) == ACQUISITION_MANIFEST_FIELDS, "F2a acquisition manifest schema changed"
    )
    expected_scalars = {
        "schema_version": 1,
        "artifact_role": "OPTIONAL_SECONDARY_NWP_AVAILABILITY_SENSITIVITY",
        "protocol": _repo_relative(route_protocol_bound.path, repository_root),
        "protocol_sha256": route_protocol_bound.sha256,
        "source_provider": EXPECTED_PROVIDER,
        "source_documentation": EXPECTED_PROVIDER_DOCUMENTATION,
        "upstream_model": EXPECTED_UPSTREAM_MODEL,
        "open_meteo_model": EXPECTED_OPEN_METEO_MODEL,
        "source_variable": EXPECTED_SOURCE_VARIABLE,
        "lead_days": list(config.leads),
        "lead_fields": [f"temperature_2m_previous_day{lead}" for lead in config.leads],
        "issue_semantics": EXPECTED_ISSUE_SEMANTICS,
        "timezone": "GMT (UTC+00:00)",
        "daily_aggregation": (
            "arithmetic mean of exactly 24 finite valid-hour values; incomplete "
            "days retained with NaN predictor and availability count"
        ),
        "archive_run_start": "2021-03-23",
        "common_valid_target_start": config.f2a_target_start.isoformat(),
        "target_end": config.primary_target_end.isoformat(),
        "station_count": station_count,
        "request_partition": "one stable site_no by one UTC calendar month",
        "immutable_outputs": True,
        "consumed_by_primary_route_a_models": False,
        "primary_evaluation_dependency": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
    }
    _require(
        all(document.get(key) == value for key, value in expected_scalars.items()),
        "F2a acquisition manifest semantics or identity changed",
    )
    registry_inputs = document.get("registry_inputs")
    _require(
        isinstance(registry_inputs, list) and len(registry_inputs) == 1,
        "F2a registry lineage changed",
    )
    _validate_lineage_record(
        registry_inputs[0],
        repository_root=repository_root,
        containment_root=input_root,
        expected_bound=station_registry_bound,
        expected_rows=station_count,
        label="F2a acquisition station registry",
    )
    snapshot_path = _resolve_evidence_path(
        document.get("raw_snapshot_index"),
        repository_root=repository_root,
        label="F2a snapshot index",
    )
    snapshot_bound = _read_stable_regular(
        snapshot_path, root=input_root, label="F2a snapshot index"
    )
    _require(
        snapshot_bound.sha256
        == _assert_hash(document.get("raw_snapshot_index_sha256"), label="snapshot index sha256"),
        "F2a snapshot-index binding changed",
    )
    chunks = document.get("chunks")
    _require(isinstance(chunks, list) and chunks, "F2a acquisition chunks missing")
    typed_chunks: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    for index, raw_chunk in enumerate(chunks):
        _require(
            isinstance(raw_chunk, dict) and set(raw_chunk) == CHUNK_FIELDS,
            f"F2a chunk {index} schema changed",
        )
        chunk = dict(raw_chunk)
        _require(chunk.get("schema_version") == 1, f"F2a chunk {index} version changed")
        site = str(chunk.get("site_no", ""))
        _require(SITE_RE.fullmatch(site) is not None, f"F2a chunk {index} site identity invalid")
        try:
            chunk_start = date.fromisoformat(str(chunk.get("chunk_start")))
            chunk_end = date.fromisoformat(str(chunk.get("chunk_end")))
        except ValueError as exc:
            raise KeyRegistryError(f"F2a chunk {index} date invalid") from exc
        _require(chunk_start <= chunk_end, f"F2a chunk {index} date order invalid")
        _require(
            config.f2a_target_start <= chunk_start <= chunk_end <= config.primary_target_end,
            f"F2a chunk {index} outside frozen target window",
        )
        _require(
            chunk.get("protocol_sha256") == route_protocol_bound.sha256,
            f"F2a chunk {index} protocol changed",
        )
        _require(
            chunk.get("labels_requested_or_read") is False,
            f"F2a chunk {index} crossed label boundary",
        )
        _assert_hash(chunk.get("artifact_sha256"), label=f"F2a chunk {index} artifact hash")
        _assert_hash(chunk.get("request_sha256"), label=f"F2a chunk {index} request hash")
        _assert_hash(chunk.get("response_sha256"), label=f"F2a chunk {index} response hash")
        _require(
            type(chunk.get("row_count")) is int and int(chunk["row_count"]) >= 0,
            f"F2a chunk {index} row count invalid",
        )
        _require(
            type(chunk.get("complete_row_count")) is int
            and 0 <= int(chunk["complete_row_count"]) <= int(chunk["row_count"]),
            f"F2a chunk {index} complete count invalid",
        )
        identities.append((site, chunk_start.isoformat()))
        typed_chunks.append(chunk)
    _require(
        len(identities) == len(set(identities)), "F2a manifest duplicates a station/month chunk"
    )
    _require(
        identities == sorted(identities), "F2a manifest chunks are not deterministically sorted"
    )
    manifest_path = _absolute_without_resolve(manifest_bound.path)
    _require(
        manifest_path == _absolute_without_resolve(acquisition_root / "manifest.json"),
        "F2a acquisition manifest is not at the root-bound canonical path",
    )
    return typed_chunks, snapshot_bound


def _validate_verification_report(
    report: Mapping[str, Any],
    *,
    report_bound: BoundBytes,
    acquisition_manifest_bound: BoundBytes,
    acquisition_manifest: Mapping[str, Any],
    route_protocol_bound: BoundBytes,
    station_registry_bound: BoundBytes,
    station_count: int,
    snapshot_bound: BoundBytes,
    repository_root: Path,
    input_root: Path,
    config: RegistryBuildConfig,
) -> None:
    _require(set(report) == VERIFICATION_REPORT_FIELDS, "F2a verification report schema changed")
    expected = {
        "schema_version": 1,
        "verifier_id": EXPECTED_VERIFIER_ID,
        "status": "VERIFIED_COMPLETE_COVERAGE_GATE_PASSED",
        "integrity_verification_passed": True,
        "coverage_gate_threshold": config.coverage_threshold,
        "coverage_gate_passed": True,
        "arm_promoted": False,
        "promotion_requires_separate_protocol_authorization": True,
        "contains_or_reads_outcome_labels": False,
    }
    _require(
        all(report.get(key) == value for key, value in expected.items()),
        "F2a verification report did not pass exact gates",
    )
    _require(
        report.get("input_access_boundary")
        == {
            "registry_columns_read": ["site_no", "lat", "lon"],
            "water_temperature_outcomes_read": False,
            "flow_outcomes_read": False,
            "panel_artifacts_read": False,
        },
        "F2a verifier input boundary changed",
    )
    _require(
        report.get("information_semantics")
        == {
            "fixed_lead_composite": True,
            "coherent_single_initialization": False,
            "as_issued_operational_forecast_archive": False,
            "issue_semantics": EXPECTED_ISSUE_SEMANTICS,
        },
        "F2a verification semantics changed",
    )
    inventory = report.get("inventory")
    _require(
        isinstance(inventory, dict)
        and set(inventory)
        == {
            "target_start",
            "target_end",
            "lead_days",
            "station_count",
            "calendar_months_per_station",
            "station_month_chunks",
            "forecast_rows",
            "complete_forecast_rows",
        },
        "F2a verification inventory schema changed",
    )
    expected_months = len(_month_chunks(config.f2a_target_start, config.primary_target_end))
    _require(
        inventory.get("target_start") == config.f2a_target_start.isoformat()
        and inventory.get("target_end") == config.primary_target_end.isoformat()
        and inventory.get("lead_days") == list(config.leads)
        and inventory.get("station_count") == station_count
        and inventory.get("calendar_months_per_station") == expected_months
        and inventory.get("station_month_chunks") == station_count * expected_months,
        "F2a verifier inventory identity changed",
    )
    evidence = report.get("evidence_hashes")
    _require(
        isinstance(evidence, dict)
        and set(evidence)
        == {"protocol", "input_registries", "code", "acquisition_manifest", "snapshot_index"},
        "F2a verification evidence schema changed",
    )
    protocol_record = evidence.get("protocol")
    _require(
        protocol_record
        == {
            "path": _repo_relative(route_protocol_bound.path, repository_root),
            "sha256": route_protocol_bound.sha256,
        },
        "F2a report Route-A protocol binding changed",
    )
    input_registries = evidence.get("input_registries")
    _require(
        isinstance(input_registries, list) and len(input_registries) == 1,
        "F2a report registry lineage changed",
    )
    _validate_lineage_record(
        input_registries[0],
        repository_root=repository_root,
        containment_root=input_root,
        expected_bound=station_registry_bound,
        expected_rows=station_count,
        label="F2a report station registry",
    )
    _require(
        input_registries == acquisition_manifest.get("registry_inputs"),
        "report/manifest registry bindings differ",
    )
    acquisition_record = evidence.get("acquisition_manifest")
    _require(
        acquisition_record
        == {
            "path": _repo_relative(acquisition_manifest_bound.path, repository_root),
            "sha256": acquisition_manifest_bound.sha256,
        },
        "F2a report/acquisition-manifest binding changed",
    )
    snapshot_record = evidence.get("snapshot_index")
    _require(
        snapshot_record
        == {
            "path": _repo_relative(snapshot_bound.path, repository_root),
            "sha256": snapshot_bound.sha256,
        },
        "F2a report/snapshot-index binding changed",
    )
    code = evidence.get("code")
    expected_roles = {
        "acquisition_script",
        "independent_verifier",
        "nwp_contract_module",
        "snapshot_store_module",
    }
    _require(
        isinstance(code, dict) and set(code) == expected_roles, "F2a verifier code binding changed"
    )
    for role in sorted(expected_roles):
        _validate_binding_record(
            code[role],
            repository_root=repository_root,
            containment_root=input_root,
            label=f"F2a verifier code {role}",
        )
    _require(
        report_bound.sha256 == _sha256_bytes(report_bound.payload),
        "verification report byte hash drifted",
    )


def _strict_f2a_arrow_schema(payload: bytes, *, label: str) -> pd.DataFrame:
    try:
        table = pq.read_table(io.BytesIO(payload))
    except Exception as exc:
        raise KeyRegistryError(f"{label} is not readable Parquet") from exc
    _require(table.column_names == list(F2A_PARQUET_COLUMNS), f"{label} schema changed")
    expected_types = {
        "site_no": pa.string(),
        "requested_lat": pa.float64(),
        "requested_lon": pa.float64(),
        "horizon": pa.int64(),
        "issue_date": pa.timestamp("ns"),
        "target_date": pa.timestamp("ns"),
        "air_temp_2m_mean_c": pa.float64(),
        "available_hour_count": pa.int64(),
        "complete_target_day": pa.bool_(),
        "lead_field": pa.string(),
        "source_provider": pa.string(),
        "upstream_model": pa.string(),
        "open_meteo_model": pa.string(),
        "issue_semantics": pa.string(),
        "response_grid_lat": pa.float64(),
        "response_grid_lon": pa.float64(),
        "request_sha256": pa.string(),
        "response_sha256": pa.string(),
    }
    for field in table.schema:
        _require(
            field.type == expected_types[field.name], f"{label} dtype changed for {field.name}"
        )
    return table.to_pandas()


def _walk_regular_files(root: Path, *, containment_root: Path) -> set[Path]:
    root = _assert_contained_path(root, containment_root, label="F2a acquisition root")
    _require(root.is_dir(), "F2a acquisition root is not a directory")
    files: set[Path] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            _require(not child.is_symlink(), f"F2a inventory contains symlink directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            child_stat = child.lstat()
            _require(
                stat.S_ISREG(child_stat.st_mode),
                f"F2a inventory contains non-regular file: {child}",
            )
            _require(
                not stat.S_ISLNK(child_stat.st_mode), f"F2a inventory contains symlink: {child}"
            )
            files.add(_absolute_without_resolve(child))
    return files


def _validate_f2a_block(
    chunk: Mapping[str, Any],
    *,
    artifact_bound: BoundBytes,
    station_coordinates: Mapping[str, tuple[float, float]],
    config: RegistryBuildConfig,
) -> pd.DataFrame:
    site = str(chunk["site_no"])
    label = f"F2a predictor block {site}/{chunk['chunk_start']}"
    _require(artifact_bound.sha256 == chunk["artifact_sha256"], f"{label} checksum mismatch")
    frame = _strict_f2a_arrow_schema(artifact_bound.payload, label=label).copy()
    _require(len(frame) == int(chunk["row_count"]), f"{label} row count changed")
    frame["site_no"] = _normalise_sites(frame["site_no"], label=f"{label} site_no")
    frame["issue_date"] = _normalise_dates(frame["issue_date"], label=f"{label} issue_date")
    frame["target_date"] = _normalise_dates(frame["target_date"], label=f"{label} target_date")
    _require(frame["site_no"].eq(site).all(), f"{label} site identity mismatch")
    _require(pd.api.types.is_integer_dtype(frame["horizon"]), f"{label} horizon dtype changed")
    _require(
        set(frame["horizon"].astype(int)) == set(config.leads), f"{label} lead inventory changed"
    )
    chronology = frame["target_date"] - pd.to_timedelta(frame["horizon"], unit="D")
    _require(chronology.equals(frame["issue_date"]), f"{label} target/issue chronology changed")
    chunk_start = pd.Timestamp(str(chunk["chunk_start"]))
    chunk_end = pd.Timestamp(str(chunk["chunk_end"]))
    _require(
        frame["target_date"].between(chunk_start, chunk_end).all(),
        f"{label} contains an outside target date",
    )
    expected_keys = pd.MultiIndex.from_product(
        [config.leads, pd.date_range(chunk_start, chunk_end, freq="D")],
        names=["horizon", "target_date"],
    )
    observed_keys = pd.MultiIndex.from_frame(
        frame[["horizon", "target_date"]].sort_values(["horizon", "target_date"])
    )
    _require(
        observed_keys.equals(expected_keys),
        f"{label} has an incomplete or outside target-day inventory",
    )
    _require(
        not frame.duplicated(["site_no", "horizon", "issue_date", "target_date"]).any(),
        f"{label} duplicates a fixed-lead key",
    )
    lat, lon = station_coordinates[site]
    _require(frame["requested_lat"].eq(lat).all(), f"{label} requested latitude changed")
    _require(frame["requested_lon"].eq(lon).all(), f"{label} requested longitude changed")
    for column in ("response_grid_lat", "response_grid_lon"):
        _require(
            np.isfinite(frame[column].to_numpy(dtype=float)).all(), f"{label} {column} invalid"
        )
    exact_text = {
        "source_provider": EXPECTED_PROVIDER,
        "upstream_model": EXPECTED_UPSTREAM_MODEL,
        "open_meteo_model": EXPECTED_OPEN_METEO_MODEL,
        "issue_semantics": EXPECTED_ISSUE_SEMANTICS,
        "request_sha256": str(chunk["request_sha256"]),
        "response_sha256": str(chunk["response_sha256"]),
    }
    for column, value in exact_text.items():
        _require(frame[column].eq(value).all(), f"{label} {column} identity changed")
    expected_fields = frame["horizon"].map(lambda lead: f"temperature_2m_previous_day{int(lead)}")
    _require(
        expected_fields.equals(frame["lead_field"]), f"{label} fixed-lead field identity changed"
    )
    _require(
        frame["available_hour_count"].between(0, 24).all(),
        f"{label} hour count outside [0,24]",
    )
    complete = frame["complete_target_day"]
    finite_temperature = np.isfinite(frame["air_temp_2m_mean_c"].to_numpy(dtype=float))
    _require(
        np.array_equal(
            complete.to_numpy(dtype=bool), frame["available_hour_count"].eq(24).to_numpy()
        ),
        f"{label} complete-day/hour-count identity changed",
    )
    _require(
        np.array_equal(complete.to_numpy(dtype=bool), finite_temperature),
        f"{label} complete-day/finite-temperature identity changed",
    )
    _require(
        int(complete.sum()) == int(chunk["complete_row_count"]), f"{label} complete count changed"
    )
    frame["f2a_artifact_sha256"] = artifact_bound.sha256
    return frame


def _load_and_validate_f2a_inventory(
    *,
    chunks: Sequence[Mapping[str, Any]],
    acquisition_manifest: Mapping[str, Any],
    verification_report: Mapping[str, Any],
    acquisition_root: Path,
    repository_root: Path,
    input_root: Path,
    station_registry: pd.DataFrame,
    config: RegistryBuildConfig,
) -> pd.DataFrame:
    coordinates = {
        str(row.site_no): (float(row.lat), float(row.lon))
        for row in station_registry.itertuples(index=False)
    }
    expected_chunk_identities = [
        (site, start.isoformat(), end.isoformat())
        for site in sorted(coordinates)
        for start, end in _month_chunks(config.f2a_target_start, config.primary_target_end)
    ]
    observed_chunk_identities = [
        (str(chunk["site_no"]), str(chunk["chunk_start"]), str(chunk["chunk_end"]))
        for chunk in chunks
    ]
    _require(
        observed_chunk_identities == expected_chunk_identities,
        "F2a acquisition manifest does not cover the exact station/month inventory",
    )
    expected_files = {_absolute_without_resolve(acquisition_root / "manifest.json")}
    frames: list[pd.DataFrame] = []
    for index, chunk in enumerate(chunks):
        relative_artifact = _relative_evidence_path(
            chunk["artifact"], label=f"F2a chunk {index} artifact"
        )
        artifact = _absolute_without_resolve(repository_root / Path(*relative_artifact.parts))
        site = str(chunk["site_no"])
        chunk_start = date.fromisoformat(str(chunk["chunk_start"]))
        canonical_artifact = _absolute_without_resolve(
            acquisition_root / site / f"{chunk_start:%Y-%m}.parquet"
        )
        _require(artifact == canonical_artifact, f"F2a chunk {index} artifact path changed")
        sidecar = artifact.with_suffix(".provenance.json")
        expected_files.update({artifact, sidecar})
        artifact_bound = _read_stable_regular(
            artifact, root=acquisition_root, label=f"F2a predictor block {index}"
        )
        sidecar_bound = _read_stable_regular(
            sidecar, root=acquisition_root, label=f"F2a predictor sidecar {index}"
        )
        sidecar_document = _strict_json(
            sidecar_bound.payload, label=f"F2a predictor sidecar {index}", canonical=True
        )
        _require(
            sidecar_document == chunk, f"F2a sidecar/manifest identity mismatch at chunk {index}"
        )
        frames.append(
            _validate_f2a_block(
                chunk,
                artifact_bound=artifact_bound,
                station_coordinates=coordinates,
                config=config,
            )
        )
    actual_files = _walk_regular_files(acquisition_root, containment_root=input_root)
    _require(
        actual_files == expected_files, "F2a acquisition root has missing or unregistered files"
    )
    acquired = pd.concat(frames, ignore_index=True)
    key = ["site_no", "horizon", "issue_date", "target_date"]
    _require(not acquired.duplicated(key).any(), "F2a predictor key duplicates across chunks")
    acquired = acquired.sort_values(key).reset_index(drop=True)
    total_rows = len(acquired)
    total_complete = int(acquired["complete_target_day"].sum())
    _require(
        acquisition_manifest.get("row_count") == total_rows, "F2a manifest total row count changed"
    )
    _require(
        acquisition_manifest.get("complete_row_count") == total_complete,
        "F2a manifest total complete count changed",
    )
    inventory = verification_report["inventory"]
    _require(
        inventory.get("forecast_rows") == total_rows
        and inventory.get("complete_forecast_rows") == total_complete,
        "F2a verification-report inventory differs from acquired Parquets",
    )
    expected_days = (config.primary_target_end - config.f2a_target_start).days + 1
    coverage_rows: list[dict[str, object]] = []
    for site in sorted(coordinates):
        for lead in config.leads:
            group = acquired[(acquired["site_no"] == site) & (acquired["horizon"] == lead)]
            complete_days = int(group["complete_target_day"].sum())
            fraction = complete_days / expected_days
            coverage_rows.append(
                {
                    "site_no": site,
                    "lead_days": lead,
                    "expected_target_days": expected_days,
                    "complete_target_days": complete_days,
                    "coverage_fraction": fraction,
                    "passes_0_90_gate": fraction >= config.coverage_threshold,
                }
            )
    _require(
        verification_report.get("coverage_by_station_and_lead") == coverage_rows,
        "F2a report station/lead coverage differs from acquired Parquets",
    )
    _require(all(row["passes_0_90_gate"] for row in coverage_rows), "F2a coverage below 0.90")
    coverage_by_lead: list[dict[str, object]] = []
    for lead in config.leads:
        lead_complete = int(acquired.loc[acquired["horizon"].eq(lead), "complete_target_day"].sum())
        denominator = len(coordinates) * expected_days
        passing = sum(row["passes_0_90_gate"] for row in coverage_rows if row["lead_days"] == lead)
        coverage_by_lead.append(
            {
                "lead_days": lead,
                "expected_station_days": denominator,
                "complete_station_days": lead_complete,
                "coverage_fraction": lead_complete / denominator,
                "stations_passing_0_90_gate": passing,
                "station_count": len(coordinates),
            }
        )
    _require(
        verification_report.get("coverage_by_lead") == coverage_by_lead,
        "F2a report lead coverage differs from acquired Parquets",
    )
    return acquired


def _freeze_f2a_temperature_registry(
    primary_raw: pd.DataFrame,
    panel: pd.DataFrame,
    acquired: pd.DataFrame,
    *,
    panel_sha256: str,
    legacy_sha256: str,
    acquisition_manifest_sha256: str,
    verification_report_sha256: str,
    config: RegistryBuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = acquired[
        acquired["complete_target_day"]
        & np.isfinite(acquired["air_temp_2m_mean_c"].to_numpy(dtype=float))
    ][
        [
            "site_no",
            "horizon",
            "issue_date",
            "target_date",
            "air_temp_2m_mean_c",
            "available_hour_count",
            "lead_field",
            "f2a_artifact_sha256",
            "request_sha256",
            "response_sha256",
        ]
    ].rename(columns={"site_no": "site_id", "horizon": "lead_days"})
    primary_keys = ["site_id", "lead_days", "issue_date", "target_date"]
    _require(not complete.duplicated(primary_keys).any(), "complete F2a keys duplicate")
    joined = primary_raw.merge(complete, on=primary_keys, how="inner", validate="one_to_one")
    f3 = panel[["site_id", "DATE", "TEMP"]].rename(
        columns={"DATE": "target_date", "TEMP": "f3_temperature_only_target_day_c"}
    )
    joined = joined.merge(f3, on=["site_id", "target_date"], how="left", validate="many_to_one")
    joined = joined[
        np.isfinite(joined["f3_temperature_only_target_day_c"].to_numpy(dtype=float))
    ].copy()
    _require(not joined.empty, "F2a/F3-temperature-only common registry is empty")
    _require(not joined.duplicated(primary_keys).any(), "F2a common registry duplicates keys")
    primary_identity = set(map(tuple, primary_raw[primary_keys].itertuples(index=False, name=None)))
    common_identity = set(map(tuple, joined[primary_keys].itertuples(index=False, name=None)))
    _require(
        common_identity.issubset(primary_identity), "F2a registry admits an outside primary key"
    )
    chronology = joined["issue_date"] + pd.to_timedelta(joined["lead_days"], unit="D")
    _require(chronology.equals(joined["target_date"]), "F2a common target chronology changed")
    joined["f2a_acquisition_manifest_sha256"] = acquisition_manifest_sha256
    joined["f2a_verification_report_sha256"] = verification_report_sha256
    joined["f3_temperature_panel_sha256"] = panel_sha256
    joined["legacy_target_registry_sha256"] = legacy_sha256
    counts = (
        joined.groupby(["site_id", "lead_days"], sort=True)
        .size()
        .rename("n_common_keys")
        .reset_index()
    )
    primary_counts = (
        primary_raw.groupby(["site_id", "lead_days"], sort=True)
        .size()
        .rename("n_primary_keys")
        .reset_index()
    )
    inventory = primary_counts.merge(
        counts, on=["site_id", "lead_days"], how="left", validate="one_to_one"
    )
    inventory["n_common_keys"] = inventory["n_common_keys"].fillna(0).astype("int32")
    inventory["n_primary_keys"] = inventory["n_primary_keys"].astype("int32")
    inventory["common_fraction_of_primary"] = (
        inventory["n_common_keys"] / inventory["n_primary_keys"]
    )
    inventory["reportable_f2a_temperature"] = inventory["n_common_keys"].ge(config.minimum_targets)
    inventory["minimum_targets"] = int(config.minimum_targets)
    reportable_pairs = set(
        map(
            tuple,
            inventory.loc[
                inventory["reportable_f2a_temperature"], ["site_id", "lead_days"]
            ].itertuples(index=False, name=None),
        )
    )
    joined["reportable_f2a_temperature"] = [
        (site, int(lead)) in reportable_pairs
        for site, lead in zip(joined["site_id"], joined["lead_days"], strict=True)
    ]
    reportable = joined[joined["reportable_f2a_temperature"]].copy()
    _require(not reportable.empty, "no F2a temperature station/lead passes reportability")
    # The output is target-day temperature only.  No prefix/trajectory mean and
    # no F3_full field or denominator is accepted or emitted.
    forbidden = [column for column in joined if "f3_full" in column.lower()]
    _require(not forbidden, "F3_full leaked into the F2a denominator registry")
    return (
        joined.sort_values(primary_keys).reset_index(drop=True),
        reportable.sort_values(primary_keys).reset_index(drop=True),
        inventory.sort_values(["site_id", "lead_days"]).reset_index(drop=True),
    )


PRIMARY_OUTPUT_COLUMNS = (
    "key_id",
    "site_id",
    "issue_date",
    "target_date",
    "lead_days",
    "y_true",
    "issue_wtemp_observed",
    "days_since_last_observed_wtemp",
    "n_observed_wtemp_7d",
    "fraction_observed_wtemp_7d",
    "n_observed_wtemp_14d",
    "fraction_observed_wtemp_14d",
    "n_observed_wtemp_32d",
    "fraction_observed_wtemp_32d",
    "history_H100",
    "history_H75",
    "history_Hall",
    "reportable_primary",
)
PRIMARY_STATION_OUTPUT_COLUMNS = (
    "site_id",
    "lead_days",
    "n_keys",
    "meets_minimum_this_lead",
    "included_all_leads",
    "minimum_targets",
)
F2A_OUTPUT_COLUMNS = (
    *PRIMARY_OUTPUT_COLUMNS,
    "f2a_target_day_air_temperature_c",
    "f3_temperature_only_target_day_c",
    "f2a_available_hour_count",
    "f2a_complete_target_day",
    "f2a_lead_field",
    "f2a_artifact_sha256",
    "f2a_request_sha256",
    "f2a_response_sha256",
    "f2a_acquisition_manifest_sha256",
    "f2a_verification_report_sha256",
    "f3_temperature_panel_sha256",
    "legacy_target_registry_sha256",
    "reportable_f2a_temperature",
)
F2A_STATION_OUTPUT_COLUMNS = (
    "site_id",
    "lead_days",
    "n_primary_keys",
    "n_common_keys",
    "common_fraction_of_primary",
    "reportable_f2a_temperature",
    "minimum_targets",
)

PRIMARY_SCHEMA = pa.schema(
    [
        pa.field("key_id", pa.string(), nullable=False),
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("issue_date", pa.timestamp("ns"), nullable=False),
        pa.field("target_date", pa.timestamp("ns"), nullable=False),
        pa.field("lead_days", pa.int16(), nullable=False),
        pa.field("y_true", pa.float64(), nullable=False),
        pa.field("issue_wtemp_observed", pa.bool_(), nullable=False),
        pa.field("days_since_last_observed_wtemp", pa.int32(), nullable=False),
        pa.field("n_observed_wtemp_7d", pa.int16(), nullable=False),
        pa.field("fraction_observed_wtemp_7d", pa.float64(), nullable=False),
        pa.field("n_observed_wtemp_14d", pa.int16(), nullable=False),
        pa.field("fraction_observed_wtemp_14d", pa.float64(), nullable=False),
        pa.field("n_observed_wtemp_32d", pa.int16(), nullable=False),
        pa.field("fraction_observed_wtemp_32d", pa.float64(), nullable=False),
        pa.field("history_H100", pa.bool_(), nullable=False),
        pa.field("history_H75", pa.bool_(), nullable=False),
        pa.field("history_Hall", pa.bool_(), nullable=False),
        pa.field("reportable_primary", pa.bool_(), nullable=False),
    ]
)
PRIMARY_STATION_SCHEMA = pa.schema(
    [
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("lead_days", pa.int16(), nullable=False),
        pa.field("n_keys", pa.int32(), nullable=False),
        pa.field("meets_minimum_this_lead", pa.bool_(), nullable=False),
        pa.field("included_all_leads", pa.bool_(), nullable=False),
        pa.field("minimum_targets", pa.int32(), nullable=False),
    ]
)
F2A_SCHEMA = pa.schema(
    [
        *PRIMARY_SCHEMA,
        pa.field("f2a_target_day_air_temperature_c", pa.float64(), nullable=False),
        pa.field("f3_temperature_only_target_day_c", pa.float64(), nullable=False),
        pa.field("f2a_available_hour_count", pa.int8(), nullable=False),
        pa.field("f2a_complete_target_day", pa.bool_(), nullable=False),
        pa.field("f2a_lead_field", pa.string(), nullable=False),
        pa.field("f2a_artifact_sha256", pa.string(), nullable=False),
        pa.field("f2a_request_sha256", pa.string(), nullable=False),
        pa.field("f2a_response_sha256", pa.string(), nullable=False),
        pa.field("f2a_acquisition_manifest_sha256", pa.string(), nullable=False),
        pa.field("f2a_verification_report_sha256", pa.string(), nullable=False),
        pa.field("f3_temperature_panel_sha256", pa.string(), nullable=False),
        pa.field("legacy_target_registry_sha256", pa.string(), nullable=False),
        pa.field("reportable_f2a_temperature", pa.bool_(), nullable=False),
    ]
)
F2A_STATION_SCHEMA = pa.schema(
    [
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("lead_days", pa.int16(), nullable=False),
        pa.field("n_primary_keys", pa.int32(), nullable=False),
        pa.field("n_common_keys", pa.int32(), nullable=False),
        pa.field("common_fraction_of_primary", pa.float64(), nullable=False),
        pa.field("reportable_f2a_temperature", pa.bool_(), nullable=False),
        pa.field("minimum_targets", pa.int32(), nullable=False),
    ]
)


def _coerce_output_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    schema: pa.Schema,
    label: str,
) -> pd.DataFrame:
    _validate_exact_columns(frame, columns, label=label)
    _require(not frame.isna().any().any(), f"{label} contains nulls")
    try:
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    except (pa.ArrowException, ValueError, TypeError) as exc:
        raise KeyRegistryError(f"{label} does not satisfy its exact Arrow schema") from exc
    _require(table.schema.remove_metadata() == schema, f"{label} Arrow schema drifted")
    return table.to_pandas()


def _prepare_output_frames(
    primary_raw: pd.DataFrame,
    primary_reportable: pd.DataFrame,
    primary_inventory: pd.DataFrame,
    f2a_common: pd.DataFrame,
    f2a_reportable: pd.DataFrame,
    f2a_inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def primary(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        out = frame[list(PRIMARY_OUTPUT_COLUMNS)].copy()
        out["lead_days"] = out["lead_days"].astype("int16")
        out["days_since_last_observed_wtemp"] = out["days_since_last_observed_wtemp"].astype(
            "int32"
        )
        for window in CONTEXT_WINDOWS:
            out[f"n_observed_wtemp_{window}d"] = out[f"n_observed_wtemp_{window}d"].astype("int16")
        return _coerce_output_frame(
            out, columns=PRIMARY_OUTPUT_COLUMNS, schema=PRIMARY_SCHEMA, label=label
        )

    primary_raw_out = primary(primary_raw, "primary raw output")
    primary_reportable_out = primary(primary_reportable, "primary reportable output")
    primary_inventory_out = primary_inventory[list(PRIMARY_STATION_OUTPUT_COLUMNS)].copy()
    primary_inventory_out["lead_days"] = primary_inventory_out["lead_days"].astype("int16")
    primary_inventory_out["n_keys"] = primary_inventory_out["n_keys"].astype("int32")
    primary_inventory_out["minimum_targets"] = primary_inventory_out["minimum_targets"].astype(
        "int32"
    )
    primary_inventory_out = _coerce_output_frame(
        primary_inventory_out,
        columns=PRIMARY_STATION_OUTPUT_COLUMNS,
        schema=PRIMARY_STATION_SCHEMA,
        label="primary station inventory output",
    )

    def f2a(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        out = frame.rename(
            columns={
                "air_temp_2m_mean_c": "f2a_target_day_air_temperature_c",
                "available_hour_count": "f2a_available_hour_count",
                "lead_field": "f2a_lead_field",
                "request_sha256": "f2a_request_sha256",
                "response_sha256": "f2a_response_sha256",
            }
        ).copy()
        out["f2a_complete_target_day"] = True
        out = out[list(F2A_OUTPUT_COLUMNS)]
        out["lead_days"] = out["lead_days"].astype("int16")
        out["days_since_last_observed_wtemp"] = out["days_since_last_observed_wtemp"].astype(
            "int32"
        )
        for window in CONTEXT_WINDOWS:
            out[f"n_observed_wtemp_{window}d"] = out[f"n_observed_wtemp_{window}d"].astype("int16")
        out["f2a_available_hour_count"] = out["f2a_available_hour_count"].astype("int8")
        return _coerce_output_frame(out, columns=F2A_OUTPUT_COLUMNS, schema=F2A_SCHEMA, label=label)

    f2a_common_out = f2a(f2a_common, "F2a common output")
    f2a_reportable_out = f2a(f2a_reportable, "F2a reportable output")
    f2a_inventory_out = f2a_inventory[list(F2A_STATION_OUTPUT_COLUMNS)].copy()
    f2a_inventory_out["lead_days"] = f2a_inventory_out["lead_days"].astype("int16")
    for column in ("n_primary_keys", "n_common_keys", "minimum_targets"):
        f2a_inventory_out[column] = f2a_inventory_out[column].astype("int32")
    f2a_inventory_out = _coerce_output_frame(
        f2a_inventory_out,
        columns=F2A_STATION_OUTPUT_COLUMNS,
        schema=F2A_STATION_SCHEMA,
        label="F2a station inventory output",
    )
    return (
        primary_raw_out,
        primary_reportable_out,
        primary_inventory_out,
        f2a_common_out,
        f2a_reportable_out,
        f2a_inventory_out,
    )


def _parquet_bytes(frame: pd.DataFrame, schema: pa.Schema) -> bytes:
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    table = table.replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        version="2.6",
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        data_page_version="1.0",
    )
    return sink.getvalue().to_pybytes()


def _identity_digest(frame: pd.DataFrame) -> str:
    rows = [
        f"{row.key_id}|{row.site_id}|{row.issue_date:%Y-%m-%d}|"
        f"{row.target_date:%Y-%m-%d}|{int(row.lead_days)}\n"
        for row in frame.itertuples(index=False)
    ]
    return _sha256_bytes("".join(rows).encode("utf-8"))


def _counts_by_lead(frame: pd.DataFrame) -> list[dict[str, int]]:
    counts = frame.groupby("lead_days", sort=True).size()
    return [{"lead_days": int(lead), "row_count": int(counts.get(lead, 0))} for lead in LEADS]


def _deterministic_reconstruction_receipt(
    manifest_without_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    input_bindings = manifest_without_receipt.get("input_bindings")
    runtime = manifest_without_receipt.get("runtime_determinism")
    outputs = manifest_without_receipt.get("outputs")
    _require(isinstance(input_bindings, dict), "receipt lacks input bindings")
    _require(isinstance(runtime, dict), "receipt lacks runtime binding")
    _require(isinstance(outputs, dict), "receipt lacks output bindings")
    payload = {
        "schema_version": 1,
        "algorithm": (
            "rebuild every registry from canonical captured inputs; serialize every Parquet "
            "with the exact Arrow schema/options; compare all seven bytes before rename"
        ),
        "builder_sha256": input_bindings["builder_code"]["sha256"],
        "input_bindings_sha256": _sha256_bytes(_canonical_json_bytes(input_bindings)),
        "runtime_binding_sha256": _sha256_bytes(_canonical_json_bytes(runtime)),
        "output_bindings_sha256": _sha256_bytes(_canonical_json_bytes(outputs)),
        "publication_contract_sha256": _sha256_bytes(
            _canonical_json_bytes(manifest_without_receipt.get("publication_contract"))
        ),
        "semantic_inventory_sha256": _sha256_bytes(
            _canonical_json_bytes(
                {
                    "primary_registry": manifest_without_receipt.get("primary_registry"),
                    "corrected_context_metadata": manifest_without_receipt.get(
                        "corrected_context_metadata"
                    ),
                    "f2a_temperature_registry": manifest_without_receipt.get(
                        "f2a_temperature_registry"
                    ),
                }
            )
        ),
    }
    return {
        "status": "DETERMINISTIC_RECONSTRUCTION_FINGERPRINT_REQUIRES_SECOND_PASS_BEFORE_PUBLISH",
        "receipt_payload": payload,
        "receipt_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
    }


def build_registry_artifacts(
    paths: RegistryInputPaths,
    *,
    config: RegistryBuildConfig,
    repository_root: Path,
    input_root: Path,
    production_read_authorized: bool = False,
) -> RegistryArtifactBundle:
    """Build deterministic registry bytes without publishing them."""
    repository_root = _absolute_without_resolve(repository_root)
    input_root = _absolute_without_resolve(input_root)
    _require(repository_root == input_root, "repository_root and input_root must be identical")
    if config.mode == "production":
        _require(
            production_read_authorized,
            "production evidence/outcome reads require explicit authorization",
        )
        _require(
            repository_root == _absolute_without_resolve(ROOT),
            "production root must be repository ROOT",
        )
        _validate_production_paths_before_read(paths, repository_root=repository_root)
    else:
        _require(
            repository_root != _absolute_without_resolve(ROOT),
            "synthetic mode cannot use production repository root",
        )

    acquisition_root = _assert_contained_path(
        paths.f2a_acquisition_root, input_root, label="F2a acquisition root"
    )
    _require(acquisition_root.is_dir(), "F2a acquisition root is not a directory")
    report_absolute = _absolute_without_resolve(paths.f2a_verification_report)
    _require(
        not report_absolute.is_relative_to(acquisition_root),
        "independent verification report cannot live inside acquisition root",
    )

    # Each evidence/data file is captured once into immutable bytes.  All
    # parsing and hashing below uses those captured bytes.
    protocol_v4_bound = _read_stable_regular(
        paths.protocol_v4, root=input_root, label="v4 draft protocol"
    )
    route_protocol_bound = _read_stable_regular(
        paths.route_a_protocol, root=input_root, label="Route-A protocol"
    )
    station_registry_bound = _read_stable_regular(
        paths.station_registry, root=input_root, label="station_registry_v1"
    )
    panel_bound = _read_stable_regular(
        paths.raw_holdout_panel, root=input_root, label="raw holdout panel"
    )
    legacy_bound = _read_stable_regular(
        paths.legacy_forecast_registry,
        root=input_root,
        label="authority-bound legacy forecast registry",
    )
    verification_bound = _read_stable_regular(
        paths.f2a_verification_report,
        root=input_root,
        label="independent F2a verification report",
    )
    acquisition_manifest_bound = _read_stable_regular(
        paths.f2a_acquisition_manifest,
        root=acquisition_root,
        label="F2a acquisition manifest",
    )
    builder_bound = _read_stable_regular(
        Path(__file__), root=ROOT, label="v4 key-registry builder code"
    )
    requirements_lock_bound = _read_stable_regular(
        repository_root / "requirements-lock.txt",
        root=input_root,
        label="repository direct dependency lock",
    )
    requirements_lock_py312_bound = _read_stable_regular(
        repository_root / "requirements-lock-py312-hashed.txt",
        root=input_root,
        label="repository hashed Python 3.12 dependency lock",
    )
    pyproject_bound = _read_stable_regular(
        repository_root / "pyproject.toml",
        root=input_root,
        label="repository pyproject",
    )
    runtime_evidence = _runtime_evidence(
        requirements_lock=requirements_lock_bound,
        requirements_lock_py312=requirements_lock_py312_bound,
        pyproject=pyproject_bound,
        repository_root=repository_root,
        production=config.mode == "production",
    )
    if config.mode == "production":
        pre_snapshot_bounds = {
            "protocol_v4_draft": protocol_v4_bound,
            "route_a_protocol": route_protocol_bound,
            "station_registry_v1": station_registry_bound,
            "raw_holdout_panel": panel_bound,
            "authority_bound_legacy_forecast_registry": legacy_bound,
            "independent_f2a_verification_report": verification_bound,
            "f2a_acquisition_manifest": acquisition_manifest_bound,
            "requirements_lock": requirements_lock_bound,
            "requirements_lock_py312_hashed": requirements_lock_py312_bound,
            "pyproject": pyproject_bound,
        }
        pre_snapshot_observed = {
            role: {
                "path": _repo_relative(bound.path, repository_root),
                "sha256": bound.sha256,
            }
            for role, bound in pre_snapshot_bounds.items()
        }
        pre_snapshot_expected = {
            role: PRODUCTION_CANONICAL_FILES[role] for role in pre_snapshot_bounds
        }
        _validate_canonical_path_hash_contract(
            pre_snapshot_observed,
            expected_files=pre_snapshot_expected,
            observed_acquisition_root=_repo_relative(acquisition_root, repository_root),
            expected_acquisition_root=PRODUCTION_ACQUISITION_ROOT,
            observed_builder_path=_repo_relative(builder_bound.path, repository_root),
            expected_builder_path=PRODUCTION_BUILDER_PATH,
        )

    protocol_v4 = _strict_yaml(protocol_v4_bound.payload, label="v4 draft protocol")
    route_protocol = _strict_json(
        route_protocol_bound.payload, label="Route-A protocol", canonical=False
    )
    verification_report = _strict_json(
        verification_bound.payload,
        label="independent F2a verification report",
        canonical=True,
    )
    acquisition_manifest = _strict_json(
        acquisition_manifest_bound.payload,
        label="F2a acquisition manifest",
        canonical=True,
    )
    _validate_v4_protocol(protocol_v4, config)
    _validate_route_a_protocol(route_protocol, config)

    station_registry = _load_station_registry(station_registry_bound)
    if config.expected_registry_stations is not None:
        _require(
            len(station_registry) == config.expected_registry_stations,
            "station_registry_v1 station count changed",
        )
    panel = _load_panel(panel_bound, config=config)
    _validate_panel_calendar(panel, station_registry["site_no"].tolist(), config=config)
    legacy = _load_legacy_registry(legacy_bound, set(station_registry["site_no"]))
    reconstructed = _reconstruct_primary_keys(panel, config=config)
    exact_targets, maximum_y_difference = _match_legacy_targets(reconstructed, legacy)
    exact_targets = _attach_corrected_context(exact_targets, panel)
    primary_raw, primary_reportable, primary_inventory = _freeze_primary_reportability(
        exact_targets, config=config
    )

    chunks, snapshot_bound = _validate_acquisition_manifest(
        acquisition_manifest,
        manifest_bound=acquisition_manifest_bound,
        route_protocol_bound=route_protocol_bound,
        station_registry_bound=station_registry_bound,
        station_count=len(station_registry),
        repository_root=repository_root,
        input_root=input_root,
        acquisition_root=acquisition_root,
        config=config,
    )
    if config.mode == "production":
        expected_snapshot_path, expected_snapshot_sha = PRODUCTION_CANONICAL_FILES[
            "f2a_snapshot_index"
        ]
        _require(
            _repo_relative(snapshot_bound.path, repository_root) == expected_snapshot_path,
            "canonical production path changed for f2a_snapshot_index",
        )
        _require(
            snapshot_bound.sha256 == expected_snapshot_sha,
            "canonical production SHA-256 changed for f2a_snapshot_index",
        )
    _validate_verification_report(
        verification_report,
        report_bound=verification_bound,
        acquisition_manifest_bound=acquisition_manifest_bound,
        acquisition_manifest=acquisition_manifest,
        route_protocol_bound=route_protocol_bound,
        station_registry_bound=station_registry_bound,
        station_count=len(station_registry),
        snapshot_bound=snapshot_bound,
        repository_root=repository_root,
        input_root=input_root,
        config=config,
    )
    if config.mode == "production":
        canonical_observed = {
            "protocol_v4_draft": _binding(protocol_v4_bound, repository_root),
            "route_a_protocol": _binding(route_protocol_bound, repository_root),
            "station_registry_v1": _binding(station_registry_bound, repository_root),
            "raw_holdout_panel": _binding(panel_bound, repository_root),
            "authority_bound_legacy_forecast_registry": _binding(legacy_bound, repository_root),
            "independent_f2a_verification_report": _binding(verification_bound, repository_root),
            "f2a_acquisition_manifest": _binding(acquisition_manifest_bound, repository_root),
            "f2a_snapshot_index": _binding(snapshot_bound, repository_root),
            "requirements_lock": _binding(requirements_lock_bound, repository_root),
            "requirements_lock_py312_hashed": _binding(
                requirements_lock_py312_bound, repository_root
            ),
            "pyproject": _binding(pyproject_bound, repository_root),
        }
        canonical_observed = {
            role: {"path": record["path"], "sha256": record["sha256"]}
            for role, record in canonical_observed.items()
        }
        _validate_canonical_path_hash_contract(
            canonical_observed,
            expected_files=PRODUCTION_CANONICAL_FILES,
            observed_acquisition_root=_repo_relative(acquisition_root, repository_root),
            expected_acquisition_root=PRODUCTION_ACQUISITION_ROOT,
            observed_builder_path=_repo_relative(builder_bound.path, repository_root),
            expected_builder_path=PRODUCTION_BUILDER_PATH,
        )
    acquired = _load_and_validate_f2a_inventory(
        chunks=chunks,
        acquisition_manifest=acquisition_manifest,
        verification_report=verification_report,
        acquisition_root=acquisition_root,
        repository_root=repository_root,
        input_root=input_root,
        station_registry=station_registry,
        config=config,
    )
    f2a_common, f2a_reportable, f2a_inventory = _freeze_f2a_temperature_registry(
        primary_raw,
        panel,
        acquired,
        panel_sha256=panel_bound.sha256,
        legacy_sha256=legacy_bound.sha256,
        acquisition_manifest_sha256=acquisition_manifest_bound.sha256,
        verification_report_sha256=verification_bound.sha256,
        config=config,
    )
    (
        primary_raw,
        primary_reportable,
        primary_inventory,
        f2a_common,
        f2a_reportable,
        f2a_inventory,
    ) = _prepare_output_frames(
        primary_raw,
        primary_reportable,
        primary_inventory,
        f2a_common,
        f2a_reportable,
        f2a_inventory,
    )

    parquet_payloads = {
        PRIMARY_RAW_FILENAME: _parquet_bytes(primary_raw, PRIMARY_SCHEMA),
        PRIMARY_REPORTABLE_FILENAME: _parquet_bytes(primary_reportable, PRIMARY_SCHEMA),
        PRIMARY_STATIONS_FILENAME: _parquet_bytes(primary_inventory, PRIMARY_STATION_SCHEMA),
        F2A_COMMON_FILENAME: _parquet_bytes(f2a_common, F2A_SCHEMA),
        F2A_REPORTABLE_FILENAME: _parquet_bytes(f2a_reportable, F2A_SCHEMA),
        F2A_STATIONS_FILENAME: _parquet_bytes(f2a_inventory, F2A_STATION_SCHEMA),
    }
    frame_by_name = {
        PRIMARY_RAW_FILENAME: primary_raw,
        PRIMARY_REPORTABLE_FILENAME: primary_reportable,
        PRIMARY_STATIONS_FILENAME: primary_inventory,
        F2A_COMMON_FILENAME: f2a_common,
        F2A_REPORTABLE_FILENAME: f2a_reportable,
        F2A_STATIONS_FILENAME: f2a_inventory,
    }
    schema_by_name = {
        PRIMARY_RAW_FILENAME: PRIMARY_SCHEMA,
        PRIMARY_REPORTABLE_FILENAME: PRIMARY_SCHEMA,
        PRIMARY_STATIONS_FILENAME: PRIMARY_STATION_SCHEMA,
        F2A_COMMON_FILENAME: F2A_SCHEMA,
        F2A_REPORTABLE_FILENAME: F2A_SCHEMA,
        F2A_STATIONS_FILENAME: F2A_STATION_SCHEMA,
    }
    outputs = {
        name: {
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
            "row_count": len(frame_by_name[name]),
            "columns": schema_by_name[name].names,
        }
        for name, payload in sorted(parquet_payloads.items())
    }
    if config.mode == "production":
        for name, expected_rows in PRODUCTION_EXPECTED_ROWS.items():
            _require(
                outputs[name]["row_count"] == expected_rows,
                f"production scientific row inventory changed for {name}",
            )
        observed_parquet_hashes = {name: str(record["sha256"]) for name, record in outputs.items()}
        _require(
            observed_parquet_hashes == dict(PRODUCTION_EXPECTED_PARQUET_SHA256),
            "production Parquet bytes differ from the independently audited six-file freeze",
        )
    raw_sites = sorted(primary_raw["site_id"].unique())
    reportable_sites = sorted(primary_reportable["site_id"].unique())
    excluded_sites = sorted(set(raw_sites) - set(reportable_sites))
    f2a_reportable_sites = [
        {
            "lead_days": int(lead),
            "site_ids": sorted(
                f2a_inventory.loc[
                    f2a_inventory["lead_days"].eq(lead)
                    & f2a_inventory["reportable_f2a_temperature"],
                    "site_id",
                ].tolist()
            ),
        }
        for lead in config.leads
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "thermoroute-information-regime-key-registries-v4-phase1",
        "status": "PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY",
        "build_mode": config.mode.upper(),
        "protocol_status": str(protocol_v4["status"]),
        "execution_authorized_by_protocol": False,
        "registry_publication_scope": "PHASE1_CONTRACT_ONLY",
        "model_or_score_output_published": False,
        "model_scores_read_or_accepted": False,
        "reportability_frozen_before_v4_scores": True,
        "input_bindings": {
            "protocol_v4_draft": _binding(protocol_v4_bound, repository_root),
            "route_a_protocol": _binding(route_protocol_bound, repository_root),
            "station_registry_v1": _binding(station_registry_bound, repository_root),
            "raw_holdout_panel": _binding(panel_bound, repository_root),
            "authority_bound_legacy_forecast_registry": _binding(legacy_bound, repository_root),
            "independent_f2a_verification_report": _binding(verification_bound, repository_root),
            "f2a_acquisition_manifest": _binding(acquisition_manifest_bound, repository_root),
            "f2a_snapshot_index": _binding(snapshot_bound, repository_root),
            "builder_code": _binding(builder_bound, repository_root),
            "requirements_lock": _binding(requirements_lock_bound, repository_root),
            "requirements_lock_py312_hashed": _binding(
                requirements_lock_py312_bound, repository_root
            ),
            "pyproject": _binding(pyproject_bound, repository_root),
        },
        "runtime_determinism": runtime_evidence,
        "primary_registry": {
            "scope": "F0_F3_FULL_BASE_REGISTRY_NOT_F2B_INTERSECTION",
            "issue_start": config.primary_issue_start.isoformat(),
            "target_end": config.primary_target_end.isoformat(),
            "lead_days": list(config.leads),
            "admissibility": "finite raw issue-date WTEMP and finite raw target-date WTEMP",
            "raw_station_count": len(raw_sites),
            "raw_site_ids": raw_sites,
            "raw_rows_by_lead": _counts_by_lead(primary_raw),
            "reportability_threshold_per_station_lead": config.minimum_targets,
            "reportable_station_rule": "at_least_threshold_in_every_lead_before_scores",
            "reportable_station_count": len(reportable_sites),
            "reportable_site_ids": reportable_sites,
            "excluded_site_ids": excluded_sites,
            "reportable_rows_by_lead": _counts_by_lead(primary_reportable),
            "raw_key_identity_sha256": _identity_digest(primary_raw),
            "reportable_key_identity_sha256": _identity_digest(primary_reportable),
            "legacy_key_identity_exact_match": True,
            "legacy_y_reconstruction_absolute_tolerance_c": Y_ABS_TOLERANCE_C,
            "maximum_panel_vs_legacy_y_difference_c": maximum_y_difference,
            "persisted_y_true_source": "authority_bound_legacy_forecast_registry_exact_bytes",
        },
        "corrected_context_metadata": {
            "legacy_context_columns_consumed": False,
            "legacy_context_metadata_status": "SUPERSEDED_DEFECTIVE_NOT_CONSUMED",
            "source": "raw_holdout_panel_before_imputation",
            "as_of_rule": "last finite WTEMP on or before each issue date; future observations forbidden",
            "window_days": list(CONTEXT_WINDOWS),
            "window_support": "inclusive calendar windows ending on issue date",
            "H100": "fraction_observed_wtemp_32d == 1.0",
            "H75": "fraction_observed_wtemp_32d >= 0.75",
            "Hall": "all primary-admissible keys",
        },
        "f2a_temperature_registry": {
            "status": "SEPARATE_DIAGNOSTIC_REGISTRY_NOT_ARM_PROMOTION",
            "arm_promoted": False,
            "target_start": config.f2a_target_start.isoformat(),
            "target_end": config.primary_target_end.isoformat(),
            "lead_days": list(config.leads),
            "information_semantics": {
                "fixed_lead_composite": True,
                "one_target_day_value_per_registered_lead": True,
                "intermediate_trajectory_constructed": False,
                "prefix_mean_constructed": False,
                "coherent_single_initialization": False,
                "as_issued_operational_forecast_archive": False,
                "operational_wording_allowed": False,
            },
            "matched_oracle": "F3_temperature_only_target_day_Daymet_TEMP",
            "forbidden_denominator": "F3_full",
            "intersection": (
                "complete finite acquired F2a fixed-target-day temperature AND primary-admissible "
                "key AND finite target-day panel TEMP"
            ),
            "common_rows_by_lead": _counts_by_lead(f2a_common),
            "common_key_identity_sha256": _identity_digest(f2a_common),
            "reportability_threshold_per_station_lead": config.minimum_targets,
            "reportable_sets_by_lead": f2a_reportable_sites,
            "reportable_rows_by_lead": _counts_by_lead(f2a_reportable),
            "reportable_key_identity_sha256": _identity_digest(f2a_reportable),
        },
        "publication_contract": {
            "atomic": True,
            "exclusive": True,
            "create_only": True,
            "resume_supported": False,
            "partial_completion_name_emitted": False,
            "closed_api_accepts_caller_bundle": False,
            "caller_constructed_or_replaced_bundle_publishable": False,
            "independent_build_passes_before_publish": 2,
            "exact_published_file_count": 7,
            "staged_bytes_reverified_immediately_before_rename": True,
        },
        "outputs": outputs,
    }
    manifest["deterministic_reconstruction_receipt"] = _deterministic_reconstruction_receipt(
        manifest
    )
    manifest_payload = _canonical_json_bytes(manifest)
    files: dict[str, bytes] = dict(parquet_payloads)
    files[MANIFEST_FILENAME] = manifest_payload
    return RegistryArtifactBundle(
        files=files,
        manifest_payload=manifest_payload,
        primary_raw=primary_raw,
        primary_reportable=primary_reportable,
        primary_station_inventory=primary_inventory,
        f2a_common=f2a_common,
        f2a_reportable=f2a_reportable,
        f2a_station_inventory=f2a_inventory,
    )


def _publication_schema_by_name() -> dict[str, pa.Schema]:
    return {
        PRIMARY_RAW_FILENAME: PRIMARY_SCHEMA,
        PRIMARY_REPORTABLE_FILENAME: PRIMARY_SCHEMA,
        PRIMARY_STATIONS_FILENAME: PRIMARY_STATION_SCHEMA,
        F2A_COMMON_FILENAME: F2A_SCHEMA,
        F2A_REPORTABLE_FILENAME: F2A_SCHEMA,
        F2A_STATIONS_FILENAME: F2A_STATION_SCHEMA,
    }


def _verify_bundle_integrity(bundle: RegistryArtifactBundle) -> None:
    """Verify payloads, manifest bindings, runtime, schemas, and receipt."""
    _require(
        type(bundle) is RegistryArtifactBundle,
        "publication candidate is not an exact RegistryArtifactBundle",
    )
    mapping_proxy_type = type(MappingProxyType({}))
    _require(
        type(bundle.files) is mapping_proxy_type,
        "artifact payload mapping is not deeply immutable",
    )
    schemas = _publication_schema_by_name()
    expected_files = set(schemas) | {MANIFEST_FILENAME}
    _require(set(bundle.files) == expected_files, "publication candidate file set is not exact")
    _require(
        bundle.files[MANIFEST_FILENAME] == bundle.manifest_payload,
        "manifest payload and file mapping differ",
    )
    manifest = _strict_json(
        bundle.manifest_payload,
        label="publication candidate manifest",
        canonical=True,
    )
    expected_manifest_fields = {
        "schema_version",
        "artifact_id",
        "status",
        "build_mode",
        "protocol_status",
        "execution_authorized_by_protocol",
        "registry_publication_scope",
        "model_or_score_output_published",
        "model_scores_read_or_accepted",
        "reportability_frozen_before_v4_scores",
        "input_bindings",
        "runtime_determinism",
        "primary_registry",
        "corrected_context_metadata",
        "f2a_temperature_registry",
        "publication_contract",
        "outputs",
        "deterministic_reconstruction_receipt",
    }
    _require(set(manifest) == expected_manifest_fields, "publication manifest schema changed")
    _require(manifest.get("schema_version") == 1, "publication manifest version changed")
    _require(
        manifest.get("artifact_id") == "thermoroute-information-regime-key-registries-v4-phase1",
        "publication artifact identity changed",
    )
    _require(
        manifest.get("status") == "PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY",
        "publication authority status changed",
    )
    _require(manifest.get("model_scores_read_or_accepted") is False, "score boundary changed")
    _require(
        manifest.get("reportability_frozen_before_v4_scores") is True,
        "reportability chronology changed",
    )
    _require(
        manifest.get("publication_contract")
        == {
            "atomic": True,
            "exclusive": True,
            "create_only": True,
            "resume_supported": False,
            "partial_completion_name_emitted": False,
            "closed_api_accepts_caller_bundle": False,
            "caller_constructed_or_replaced_bundle_publishable": False,
            "independent_build_passes_before_publish": 2,
            "exact_published_file_count": 7,
            "staged_bytes_reverified_immediately_before_rename": True,
        },
        "publication closure contract changed",
    )
    input_bindings = manifest.get("input_bindings")
    expected_input_roles = {
        "protocol_v4_draft",
        "route_a_protocol",
        "station_registry_v1",
        "raw_holdout_panel",
        "authority_bound_legacy_forecast_registry",
        "independent_f2a_verification_report",
        "f2a_acquisition_manifest",
        "f2a_snapshot_index",
        "builder_code",
        "requirements_lock",
        "requirements_lock_py312_hashed",
        "pyproject",
    }
    _require(
        isinstance(input_bindings, dict) and set(input_bindings) == expected_input_roles,
        "publication input-binding roles changed",
    )
    for role, record in input_bindings.items():
        _require(
            isinstance(record, dict) and set(record) == {"path", "sha256", "size_bytes"},
            f"publication input binding schema changed for {role}",
        )
        _require(isinstance(record["path"], str) and record["path"], f"{role} path invalid")
        _assert_hash(record["sha256"], label=f"{role} SHA-256")
        _require(
            type(record["size_bytes"]) is int and record["size_bytes"] >= 0,
            f"{role} size invalid",
        )
    runtime = manifest.get("runtime_determinism")
    _require(
        isinstance(runtime, dict)
        and set(runtime)
        == {
            "status",
            "determinism_class",
            "active_runtime",
            "production_exact_runtime_pin",
            "production_platform_pin",
            "byte_identity_scope",
            "repository_dependency_references",
            "repository_locks_are_active_runtime_authority",
            "repository_locks_match_active_runtime",
            "disclosed_lock_mismatches",
        },
        "publication runtime schema changed",
    )
    _require(runtime["status"] == RUNTIME_STATUS, "runtime status changed")
    _require(
        runtime["determinism_class"] == "PINNED_RUNTIME_DETERMINISTIC",
        "runtime determinism class changed",
    )
    active_runtime = runtime["active_runtime"]
    expected_active_runtime_fields = set(PRODUCTION_RUNTIME) | {
        "platform_system",
        "platform_machine",
        "byteorder",
    }
    _require(
        isinstance(active_runtime, dict)
        and set(active_runtime) == expected_active_runtime_fields
        and all(type(value) is str and value for value in active_runtime.values()),
        "active runtime identity schema changed",
    )
    _require(
        active_runtime == _current_runtime_identity(),
        "active runtime identity differs from the publishing verifier runtime",
    )
    _require(
        runtime["production_exact_runtime_pin"] == dict(PRODUCTION_RUNTIME),
        "production exact runtime pin changed",
    )
    _require(
        runtime["production_platform_pin"]
        == {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "byteorder": "little",
        },
        "production platform pin changed",
    )
    _require(
        runtime["repository_locks_are_active_runtime_authority"] is False
        and runtime["repository_locks_match_active_runtime"] is False,
        "repository lock mismatch disclosure changed",
    )
    _require(
        "not claimed" in str(runtime["byte_identity_scope"]),
        "cross-environment byte-identity limitation disappeared",
    )
    references = runtime["repository_dependency_references"]
    _require(
        references
        == {
            role: input_bindings[role]
            for role in (
                "requirements_lock",
                "requirements_lock_py312_hashed",
                "pyproject",
            )
        },
        "runtime dependency references differ from input bindings",
    )
    outputs = manifest.get("outputs")
    _require(
        isinstance(outputs, dict) and set(outputs) == set(schemas), "output binding set changed"
    )
    parsed_frames: dict[str, pd.DataFrame] = {}
    for name, schema in schemas.items():
        payload = bundle.files[name]
        record = outputs[name]
        _require(
            isinstance(record, dict)
            and set(record) == {"sha256", "size_bytes", "row_count", "columns"},
            f"output binding schema changed for {name}",
        )
        _require(record["sha256"] == _sha256_bytes(payload), f"output SHA-256 mismatch for {name}")
        _require(record["size_bytes"] == len(payload), f"output size mismatch for {name}")
        try:
            table = pq.read_table(io.BytesIO(payload))
        except Exception as exc:
            raise KeyRegistryError(f"publication output is not Parquet: {name}") from exc
        _require(
            table.schema.remove_metadata() == schema, f"publication schema mismatch for {name}"
        )
        _require(table.column_names == record["columns"], f"manifest columns differ for {name}")
        _require(table.num_rows == record["row_count"], f"manifest row count differs for {name}")
        frame = table.to_pandas()
        _require(
            _parquet_bytes(frame, schema) == payload,
            f"Parquet payload is not a deterministic reconstruction for {name}",
        )
        parsed_frames[name] = frame
    primary = manifest.get("primary_registry")
    f2a = manifest.get("f2a_temperature_registry")
    _require(isinstance(primary, dict) and isinstance(f2a, dict), "semantic inventory missing")
    _require(
        sum(row["row_count"] for row in primary["raw_rows_by_lead"])
        == len(parsed_frames[PRIMARY_RAW_FILENAME]),
        "primary raw semantic inventory differs from payload",
    )
    _require(
        sum(row["row_count"] for row in primary["reportable_rows_by_lead"])
        == len(parsed_frames[PRIMARY_REPORTABLE_FILENAME]),
        "primary reportable semantic inventory differs from payload",
    )
    _require(
        primary["raw_key_identity_sha256"] == _identity_digest(parsed_frames[PRIMARY_RAW_FILENAME]),
        "primary raw identity receipt differs from payload",
    )
    _require(
        primary["reportable_key_identity_sha256"]
        == _identity_digest(parsed_frames[PRIMARY_REPORTABLE_FILENAME]),
        "primary reportable identity receipt differs from payload",
    )
    _require(
        sum(row["row_count"] for row in f2a["common_rows_by_lead"])
        == len(parsed_frames[F2A_COMMON_FILENAME]),
        "F2a common semantic inventory differs from payload",
    )
    _require(
        sum(row["row_count"] for row in f2a["reportable_rows_by_lead"])
        == len(parsed_frames[F2A_REPORTABLE_FILENAME]),
        "F2a reportable semantic inventory differs from payload",
    )
    _require(
        f2a["common_key_identity_sha256"] == _identity_digest(parsed_frames[F2A_COMMON_FILENAME]),
        "F2a common identity receipt differs from payload",
    )
    _require(
        f2a["reportable_key_identity_sha256"]
        == _identity_digest(parsed_frames[F2A_REPORTABLE_FILENAME]),
        "F2a reportable identity receipt differs from payload",
    )
    receipt = manifest["deterministic_reconstruction_receipt"]
    without_receipt = dict(manifest)
    without_receipt.pop("deterministic_reconstruction_receipt")
    _require(
        receipt == _deterministic_reconstruction_receipt(without_receipt),
        "deterministic reconstruction receipt does not verify",
    )


def _verify_bundle_against_independent_reconstruction(
    candidate: RegistryArtifactBundle,
    reconstruction: RegistryArtifactBundle,
) -> None:
    _verify_bundle_integrity(candidate)
    _verify_bundle_integrity(reconstruction)
    _require(
        candidate.manifest_payload == reconstruction.manifest_payload,
        "candidate manifest differs from independent reconstruction",
    )
    _require(
        set(candidate.files) == set(reconstruction.files)
        and all(candidate.files[name] == reconstruction.files[name] for name in candidate.files),
        "candidate payload bytes differ from independent reconstruction",
    )


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory with Linux RENAME_NOREPLACE semantics."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise KeyRegistryError(
                f"refusing to overwrite existing registry authority: {destination}"
            )
        raise KeyRegistryError(
            f"atomic no-replace publication failed ({os.strerror(error)}): {destination}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_registry_artifacts(
    paths: RegistryInputPaths,
    destination: Path,
    *,
    config: RegistryBuildConfig,
    repository_root: Path,
    input_root: Path,
    allowed_output_root: Path,
    production_read_authorized: bool = False,
) -> tuple[Path, RegistryArtifactBundle]:
    """Closed two-pass build-and-publish API; caller bundles are never accepted."""
    _require(
        type(paths) is RegistryInputPaths,
        "closed publisher accepts canonical RegistryInputPaths, never a caller bundle",
    )
    allowed_root = _absolute_without_resolve(allowed_output_root)
    destination = _absolute_without_resolve(destination)
    try:
        relative = destination.relative_to(allowed_root)
    except ValueError as exc:
        raise KeyRegistryError("registry output escapes allowed output root") from exc
    _require(len(relative.parts) == 1, "registry output must be one direct child of output root")
    _require(
        not any(
            fragment in destination.name.lower() for fragment in FORBIDDEN_COMPLETION_FRAGMENTS
        ),
        "registry output name implies partial/resume state",
    )
    if config.mode == "production":
        _require(
            destination.name == DEFAULT_OUTPUT_NAME, "production registry directory name changed"
        )
    _require(destination.name not in {"", ".", ".."}, "invalid registry output name")
    parent = _assert_contained_path(
        destination.parent, allowed_root, label="registry output parent"
    )
    _require(parent == allowed_root and parent.is_dir(), "registry output root is invalid")
    _require(
        not os.path.lexists(destination),
        f"refusing to overwrite existing registry authority: {destination}",
    )
    lock = parent / f".{destination.name}.create.lock"
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise KeyRegistryError(f"another create-only publication is active: {lock}") from exc
    staging: Path | None = None
    try:
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_descriptor)
        os.close(lock_descriptor)
        lock_descriptor = -1
        first = build_registry_artifacts(
            paths,
            config=config,
            repository_root=repository_root,
            input_root=input_root,
            production_read_authorized=production_read_authorized,
        )
        reconstruction = build_registry_artifacts(
            paths,
            config=config,
            repository_root=repository_root,
            input_root=input_root,
            production_read_authorized=production_read_authorized,
        )
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=parent))
        for name, payload in sorted(first.files.items()):
            _require(
                Path(name).name == name and name not in {".", ".."},
                f"invalid output filename: {name}",
            )
            target = staging / name
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        # Revalidate the complete logical authority before the final staged
        # byte comparison.  A token or self-resealed receipt alone cannot pass:
        # the candidate must equal the independently rebuilt authority.
        _verify_bundle_against_independent_reconstruction(first, reconstruction)
        # This block is deliberately adjacent to the atomic rename: verify the
        # exact seven-file set and every staged byte after the more expensive
        # schema/runtime/reconstruction checks have finished.
        staged_names = {path.name for path in staging.iterdir()}
        _require(staged_names == set(first.files), "staged publication file set changed")
        for name, expected_payload in first.files.items():
            staged = _read_stable_regular(
                staging / name,
                root=staging,
                label=f"staged publication payload {name}",
            )
            _require(
                staged.payload == expected_payload,
                f"staged publication bytes changed for {name}",
            )
        _require(
            first.manifest_payload == reconstruction.manifest_payload,
            "builder/input/runtime/reconstruction bindings drifted before rename",
        )
        _require(not os.path.lexists(destination), "registry destination appeared during build")
        _rename_directory_noreplace(staging, destination)
        staging = None
        _fsync_directory(parent)
    finally:
        if "lock_descriptor" in locals() and lock_descriptor >= 0:
            os.close(lock_descriptor)
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if lock.exists():
            lock.unlink()
            _fsync_directory(parent)
    return destination, first


def _parse_iso_date(value: str, *, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be YYYY-MM-DD") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("production", "synthetic"))
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--protocol-v4", required=True, type=Path)
    parser.add_argument("--route-a-protocol", required=True, type=Path)
    parser.add_argument("--station-registry", required=True, type=Path)
    parser.add_argument("--raw-holdout-panel", required=True, type=Path)
    parser.add_argument("--legacy-forecast-registry", required=True, type=Path)
    parser.add_argument("--f2a-verification-report", required=True, type=Path)
    parser.add_argument("--f2a-acquisition-manifest", required=True, type=Path)
    parser.add_argument("--f2a-acquisition-root", required=True, type=Path)
    parser.add_argument(
        "--authorize-production-inputs-after-independent-verification",
        action="store_true",
        help="required in production mode; acknowledges that all listed evidence is final",
    )
    parser.add_argument(
        "--publish", action="store_true", help="atomically create the registry directory"
    )
    parser.add_argument("--output-dir", type=Path, help="required only with --publish")
    parser.add_argument("--synthetic-primary-issue-start", default=PRIMARY_ISSUE_START.isoformat())
    parser.add_argument("--synthetic-target-end", default=PRIMARY_TARGET_END.isoformat())
    parser.add_argument("--synthetic-f2a-target-start", default=F2A_TARGET_START.isoformat())
    parser.add_argument("--synthetic-minimum-targets", type=int, default=MINIMUM_PAIRED_TARGETS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    repository_root = _absolute_without_resolve(args.repository_root)
    if args.mode == "production":
        config = RegistryBuildConfig.production()
        allowed_output_root = repository_root / "outputs" / "final"
    else:
        config = RegistryBuildConfig(
            mode="synthetic",
            primary_issue_start=_parse_iso_date(
                args.synthetic_primary_issue_start, label="synthetic primary issue start"
            ),
            primary_target_end=_parse_iso_date(
                args.synthetic_target_end, label="synthetic target end"
            ),
            f2a_target_start=_parse_iso_date(
                args.synthetic_f2a_target_start, label="synthetic F2a target start"
            ),
            minimum_targets=args.synthetic_minimum_targets,
            expected_registry_stations=None,
            expected_raw_stations=None,
            expected_reportable_stations=None,
            expected_excluded_sites=None,
        )
        allowed_output_root = repository_root
    if args.publish and args.output_dir is None:
        parser.error("--output-dir is required with --publish")
    if not args.publish and args.output_dir is not None:
        parser.error("--output-dir is accepted only with --publish")
    if (
        args.mode == "production"
        and not args.authorize_production_inputs_after_independent_verification
    ):
        parser.error(
            "production mode requires --authorize-production-inputs-after-independent-verification"
        )
    paths = RegistryInputPaths(
        protocol_v4=args.protocol_v4,
        route_a_protocol=args.route_a_protocol,
        station_registry=args.station_registry,
        raw_holdout_panel=args.raw_holdout_panel,
        legacy_forecast_registry=args.legacy_forecast_registry,
        f2a_verification_report=args.f2a_verification_report,
        f2a_acquisition_manifest=args.f2a_acquisition_manifest,
        f2a_acquisition_root=args.f2a_acquisition_root,
    )
    if args.publish:
        published, bundle = publish_registry_artifacts(
            paths,
            args.output_dir,
            config=config,
            repository_root=repository_root,
            input_root=repository_root,
            allowed_output_root=allowed_output_root,
            production_read_authorized=(
                args.authorize_production_inputs_after_independent_verification
            ),
        )
    else:
        bundle = build_registry_artifacts(
            paths,
            config=config,
            repository_root=repository_root,
            input_root=repository_root,
            production_read_authorized=(
                args.authorize_production_inputs_after_independent_verification
            ),
        )
    result: dict[str, object] = {
        "status": "DRY_RUN_VALIDATED_NOT_PUBLISHED",
        "build_mode": config.mode.upper(),
        "primary_raw_rows": len(bundle.primary_raw),
        "primary_reportable_rows": len(bundle.primary_reportable),
        "f2a_common_rows": len(bundle.f2a_common),
        "f2a_reportable_rows": len(bundle.f2a_reportable),
        "output_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(bundle.files.items())
        },
    }
    if args.publish:
        result["status"] = "PHASE1_KEY_REGISTRIES_PUBLISHED_CREATE_ONLY"
        result["output_dir"] = str(published)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyRegistryError as exc:
        print(f"v4 key-registry build failed closed: {exc}", file=os.sys.stderr)
        raise SystemExit(1) from exc
