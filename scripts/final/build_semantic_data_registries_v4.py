#!/usr/bin/env python3
"""Build scratch-only observed-lineage semantic data registries, version 4.

This builder emits exactly two data registries and one manifest:

* ``daily_raw_observed_panel_registry_v4.parquet`` records the canonical
  station-day panel and boolean observedness captured from raw finite values
  before any imputation.
* ``training_example_registry_2006_2017_v4.parquet`` records only examples
  whose issue- and target-day water temperatures are both raw observations,
  with train and validation dates kept disjoint.
* ``semantic_data_registries_v4_candidate_manifest.json`` binds their exact
  schemas, order, semantic content digests, source authorities, and counts.

The result is always ``CANDIDATE_NOT_AUTHORITY`` and always carries
``execution_authorized=false``.  There is no production/default destination,
publish mode, seal mode, overwrite mode, or resume mode.  A caller must name a
scratch/candidate destination explicitly.  The create-only writer builds the
bundle twice, writes through an anchored directory descriptor, verifies the
staged bytes, and uses Linux ``RENAME_NOREPLACE``.

No model, prediction, score, effect, or withdrawn learned-result artifact is
read by this program.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import platform
import secrets
import stat
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

STATUS = "CANDIDATE_NOT_AUTHORITY"
ARTIFACT_ID = "thermoroute-semantic-data-registries-v4-candidate"
DAILY_FILENAME = "daily_raw_observed_panel_registry_v4.parquet"
TRAINING_FILENAME = "training_example_registry_2006_2017_v4.parquet"
MANIFEST_FILENAME = "semantic_data_registries_v4_candidate_manifest.json"

KEY_AUTHORITY_MANIFEST = (
    ROOT
    / "outputs"
    / "final"
    / "information_regime_key_registries_v4"
    / "information_regime_key_registries_v4_manifest.json"
)
DEFECT_AUTHORITY_DIR = ROOT / "outputs" / "final" / "preprocessing_lineage_defect_authority_v1"
DEFECT_AUTHORITY_MANIFEST = (
    DEFECT_AUTHORITY_DIR / "preprocessing_lineage_defect_authority_v1_manifest.json"
)
DEFECT_AUTHORITY_REPORT = DEFECT_AUTHORITY_DIR / "preprocessing_lineage_defect_report_v1.json"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEVELOPMENT_PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
EVALUATION_PANEL = ROOT / "outputs" / "conventional" / "panel_2021_2023.parquet"

KEY_AUTHORITY_MANIFEST_SHA256 = "ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea"
DEFECT_AUTHORITY_MANIFEST_SHA256 = (
    "e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908"
)

SOURCE_SHA256 = MappingProxyType(
    {
        "station_registry": ("090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9"),
        "development_panel": ("0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69"),
        "evaluation_panel": ("cecdac459139456202240954e4c98fe18bba1fe8b63e9b06ab268683e0d1c03c"),
        "defect_authority_report": (
            "5c1b1e05932538dba1b2a0d21ad44fdfe54a9c52e949b96b3a668356910e68d9"
        ),
    }
)

SOURCE_RELATIVE_PATHS = MappingProxyType(
    {
        "station_registry": "data_usgs/station_registry_v1.csv",
        "development_panel": "data_usgs/panel_usgs_120v2.parquet",
        "evaluation_panel": "outputs/conventional/panel_2021_2023.parquet",
        "defect_authority_report": (
            "outputs/final/preprocessing_lineage_defect_authority_v1/"
            "preprocessing_lineage_defect_report_v1.json"
        ),
    }
)

PANEL_VARIABLES = ("WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP", "WDSP", "RHMEAN", "DH")
PANEL_COLUMNS = ("DATE", "site_id", *PANEL_VARIABLES)
OBSERVED_COLUMNS = tuple(f"{variable}_observed" for variable in PANEL_VARIABLES)
DAILY_COLUMNS = (*PANEL_COLUMNS, *OBSERVED_COLUMNS)
TRAINING_COLUMNS = (
    "training_key_id",
    "site_id",
    "split",
    "issue_date",
    "target_date",
    "lead_days",
    "y_true",
    "issue_wtemp_observed",
    "target_wtemp_observed",
)
LEADS = (1, 3, 7)
SPLITS = ("train", "validation")

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

DAILY_SCHEMA = pa.schema(
    [
        pa.field("DATE", pa.timestamp("ns"), nullable=False),
        pa.field("site_id", pa.string(), nullable=False),
        *(pa.field(variable, pa.float64(), nullable=True) for variable in PANEL_VARIABLES),
        *(pa.field(column, pa.bool_(), nullable=False) for column in OBSERVED_COLUMNS),
    ]
)
TRAINING_SCHEMA = pa.schema(
    [
        pa.field("training_key_id", pa.string(), nullable=False),
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("issue_date", pa.timestamp("ns"), nullable=False),
        pa.field("target_date", pa.timestamp("ns"), nullable=False),
        pa.field("lead_days", pa.int16(), nullable=False),
        pa.field("y_true", pa.float64(), nullable=False),
        pa.field("issue_wtemp_observed", pa.bool_(), nullable=False),
        pa.field("target_wtemp_observed", pa.bool_(), nullable=False),
    ]
)

ROUTE_A_CANDIDATE_RUNTIME = MappingProxyType(
    {
        "python_implementation": "CPython",
        "python_version": "3.12.13",
        "python_compiler": "GCC 14.3.0",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "byteorder": "little",
        "numpy_version": "1.26.4",
        "pandas_version": "2.2.2",
        "pyarrow_version": "24.0.0",
        "arrow_cpp_version": "24.0.0",
    }
)

EXPECTED_PRODUCTION_COUNTS = MappingProxyType(
    {
        "development_rows": 657_480,
        "evaluation_rows": 135_240,
        "overlap_rows": 3_840,
        "daily_rows": 788_880,
        "station_count": 120,
        "daily_start": "2006-01-01",
        "daily_end": "2023-12-31",
        "overlap_start": "2020-11-30",
        "overlap_end": "2020-12-31",
        "daily_missing": {
            "WTEMP": 113_970,
            "FLOW": 18_611,
            "WLEVEL": 611_427,
            "TEMP": 480,
            "PRCP": 480,
            "WDSP": 0,
            "RHMEAN": 480,
            "DH": 480,
        },
        "training_examples": {
            "train": {"1": 339_750, "3": 337_725, "7": 335_822},
            "validation": {"1": 83_516, "3": 82_948, "7": 82_233},
        },
    }
)


class SemanticRegistryError(RuntimeError):
    """Raised when evidence, semantic, or create-only output checks fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticRegistryError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative(path: Path, root: Path) -> str:
    try:
        return _absolute(path).relative_to(_absolute(root)).as_posix()
    except ValueError as exc:
        raise SemanticRegistryError(f"input escapes repository/evidence root: {path}") from exc


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SemanticRegistryError(f"{label} cannot be resolved: {absolute}") from exc
    _require(resolved == absolute, f"{label} traverses a symbolic link: {absolute}")
    return absolute


@dataclass(frozen=True)
class BoundFile:
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    stat_signature: tuple[int, int, int, int, int, int, int]


def _read_stable_regular(path: Path, *, root: Path, label: str) -> BoundFile:
    absolute = _absolute(path)
    root_absolute = _require_no_symlink_components(root, label="evidence root")
    _relative(absolute, root_absolute)
    _require_no_symlink_components(absolute, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise SemanticRegistryError(f"cannot safely open {label}: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(before.st_nlink == 1, f"{label} must have exactly one hard link")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    signature = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_mode),
        int(before.st_nlink),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    _require(
        signature
        == (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_nlink),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        and len(payload) == before.st_size,
        f"{label} changed while being captured",
    )
    path_after = absolute.lstat()
    _require(
        stat.S_ISREG(path_after.st_mode)
        and not stat.S_ISLNK(path_after.st_mode)
        and (path_after.st_dev, path_after.st_ino) == (before.st_dev, before.st_ino)
        and path_after.st_nlink == 1,
        f"{label} path identity changed while being captured",
    )
    _require_no_symlink_components(absolute, label=label)
    return BoundFile(absolute, payload, _sha256(payload), len(payload), signature)


def _strict_json(payload: bytes, *, label: str, canonical: bool = True) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise SemanticRegistryError(f"{label} is not strict JSON") from exc
    _require(type(document) is dict, f"{label} top level must be an object")
    if canonical:
        _require(_canonical_json_bytes(document) == payload, f"{label} is not canonical JSON")
    return document


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
        "arrow_cpp_version": pa.cpp_version,
    }


def _validate_candidate_runtime(*, enforce: bool) -> dict[str, Any]:
    active = _runtime_identity()
    if enforce:
        observed = {key: active[key] for key in ROUTE_A_CANDIDATE_RUNTIME}
        _require(
            observed == dict(ROUTE_A_CANDIDATE_RUNTIME),
            "production-input candidate build requires the explicit route-a candidate runtime; "
            f"expected {dict(ROUTE_A_CANDIDATE_RUNTIME)}, observed {observed}",
        )
    return {
        "active_runtime": active,
        "candidate_runtime_expected": dict(ROUTE_A_CANDIDATE_RUNTIME),
        "candidate_runtime_enforced": enforce,
        "determinism_class": "SEMANTIC_DIGEST_CROSS_RUNTIME_PARQUET_BYTES_ACTIVE_RUNTIME_ONLY",
        "authorization_effect": "NONE",
        "runtime_status": "CANDIDATE_SERIALIZATION_RUNTIME_RECORDED_NOT_AUTHORIZED",
    }


def _strict_bool_array(series: pd.Series, *, label: str) -> np.ndarray:
    _require(series.dtype == np.dtype("bool"), f"{label} must have exact non-null bool dtype")
    values = series.to_numpy(copy=True)
    _require(
        values.ndim == 1 and values.dtype == np.dtype("bool"),
        f"{label} bool representation is not exact",
    )
    return values


def _strict_dates(series: pd.Series, *, label: str) -> np.ndarray:
    _require(
        series.dtype == np.dtype("datetime64[ns]"),
        f"{label} must have exact timezone-naive datetime64[ns] dtype",
    )
    values = series.to_numpy(copy=True)
    integers = values.view(np.int64)
    _require(not (integers == np.iinfo(np.int64).min).any(), f"{label} contains NaT")
    _require(
        not np.remainder(integers, 86_400_000_000_000).any(),
        f"{label} must contain midnight dates",
    )
    return values


def _strict_site_ids(series: pd.Series, *, label: str) -> tuple[str, ...]:
    values = tuple(series.to_numpy(copy=True))
    _require(bool(values), f"{label} is empty")
    for value in values:
        _require(
            type(value) is str
            and len(value) in (8, 15)
            and value.isascii()
            and all("0" <= character <= "9" for character in value),
            f"{label} contains a non-canonical ASCII station identifier",
        )
    return values


def _strict_string_series(
    series: pd.Series,
    *,
    label: str,
    allowed: set[str] | None = None,
) -> tuple[str, ...]:
    values = tuple(series.to_numpy(copy=True))
    _require(
        all(type(value) is str and value != "" for value in values),
        f"{label} must contain exact non-empty strings",
    )
    if allowed is not None:
        _require(set(values) <= allowed, f"{label} contains a value outside {sorted(allowed)}")
    return values


def _strict_float_array(
    series: pd.Series,
    *,
    label: str,
    require_finite: bool,
) -> np.ndarray:
    dtype = series.dtype
    _require(
        isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.floating),
        f"{label} must have an exact floating dtype",
    )
    values = series.to_numpy(copy=True).astype(np.float64, copy=False)
    _require(not np.isinf(values).any(), f"{label} contains infinity")
    if require_finite:
        _require(np.isfinite(values).all(), f"{label} contains a missing/non-finite value")
    return values


def _strict_int16_array(series: pd.Series, *, label: str) -> np.ndarray:
    _require(series.dtype == np.dtype("int16"), f"{label} must have exact int16 dtype")
    values = series.to_numpy(copy=True)
    _require(values.ndim == 1, f"{label} must be one-dimensional")
    return values


def _canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Order-sensitive, schema-aware digest independent of Parquet metadata."""

    selected = tuple(columns)
    _require(len(selected) == len(set(selected)), "digest column list contains duplicates")
    _require(set(selected) <= set(frame.columns), "digest column list contains absent columns")
    digest = hashlib.sha256()
    digest.update(b"thermoroute-semantic-data-frame-v4\0")
    digest.update(struct.pack("<QQ", len(frame), len(selected)))
    for column in selected:
        series = frame[column]
        name = column.encode("utf-8")
        dtype_name = str(series.dtype).encode("ascii")
        digest.update(struct.pack("<Q", len(name)))
        digest.update(name)
        digest.update(struct.pack("<Q", len(dtype_name)))
        digest.update(dtype_name)
        if series.dtype == np.dtype("bool"):
            digest.update(b"B")
            digest.update(_strict_bool_array(series, label=column).view(np.uint8).tobytes())
        elif series.dtype == np.dtype("datetime64[ns]"):
            digest.update(b"D")
            dates = _strict_dates(series, label=column)
            digest.update(dates.view(np.int64).astype("<i8", copy=False).tobytes())
        elif series.dtype == np.dtype("int16"):
            digest.update(b"I")
            integers = _strict_int16_array(series, label=column)
            digest.update(integers.astype("<i8", copy=False).tobytes())
        elif isinstance(series.dtype, np.dtype) and np.issubdtype(series.dtype, np.floating):
            digest.update(b"F")
            values = _strict_float_array(series, label=column, require_finite=False)
            present = np.isfinite(values)
            canonical = values.astype("<f8", copy=True)
            canonical[~present] = 0.0
            digest.update(present.view(np.uint8).tobytes())
            digest.update(canonical.tobytes())
        else:
            digest.update(b"S")
            strings = _strict_string_series(series, label=column)
            for value in strings:
                encoded = value.encode("utf-8")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
    return digest.hexdigest()


def _schema_signature(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


@dataclass(frozen=True)
class RegistryInputPaths:
    key_authority_manifest: Path
    defect_authority_manifest: Path
    defect_authority_report: Path
    station_registry: Path
    development_panel: Path
    evaluation_panel: Path

    @classmethod
    def production(cls, root: Path = ROOT) -> RegistryInputPaths:
        root = _absolute(root)
        return cls(
            key_authority_manifest=(
                root
                / "outputs"
                / "final"
                / "information_regime_key_registries_v4"
                / "information_regime_key_registries_v4_manifest.json"
            ),
            defect_authority_manifest=(
                root
                / "outputs"
                / "final"
                / "preprocessing_lineage_defect_authority_v1"
                / "preprocessing_lineage_defect_authority_v1_manifest.json"
            ),
            defect_authority_report=(
                root
                / "outputs"
                / "final"
                / "preprocessing_lineage_defect_authority_v1"
                / "preprocessing_lineage_defect_report_v1.json"
            ),
            station_registry=root / "data_usgs" / "station_registry_v1.csv",
            development_panel=root / "data_usgs" / "panel_usgs_120v2.parquet",
            evaluation_panel=root / "outputs" / "conventional" / "panel_2021_2023.parquet",
        )


@dataclass(frozen=True)
class BuildConfig:
    mode: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    leads: tuple[int, ...] = LEADS
    enforce_route_a_runtime: bool = False
    expected_counts: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _require(self.mode in {"production", "synthetic"}, "invalid build mode")
        _require(self.leads == LEADS, f"lead inventory must be exactly {LEADS}")
        _require(
            self.train_start <= self.train_end < self.validation_start <= self.validation_end,
            "train/validation chronology is invalid",
        )
        if self.mode == "production":
            _require(self.enforce_route_a_runtime, "production mode must enforce route-a runtime")

    @classmethod
    def production(cls) -> BuildConfig:
        return cls(
            mode="production",
            train_start=date(2006, 1, 1),
            train_end=date(2015, 12, 31),
            validation_start=date(2016, 1, 1),
            validation_end=date(2017, 12, 31),
            enforce_route_a_runtime=True,
            expected_counts=EXPECTED_PRODUCTION_COUNTS,
        )

    @classmethod
    def synthetic(
        cls,
        *,
        train_start: date,
        train_end: date,
        validation_start: date,
        validation_end: date,
    ) -> BuildConfig:
        return cls(
            mode="synthetic",
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
        )


@dataclass(frozen=True)
class PanelInventory:
    development_rows: int
    evaluation_rows: int
    overlap_rows: int
    overlap_start: str | None
    overlap_end: str | None
    overlap_differences: Mapping[str, int]
    selected_development_rows: int
    selected_evaluation_rows: int


_BUNDLE_TOKEN = object()


@dataclass(frozen=True)
class CandidateBundle:
    files: Mapping[str, bytes]
    daily: pd.DataFrame
    training_examples: pd.DataFrame
    manifest_payload: bytes
    snapshot: Mapping[str, BoundFile]
    _token: object

    def __post_init__(self) -> None:
        _require(self._token is _BUNDLE_TOKEN, "candidate bundle issuer token changed")
        frozen_files: dict[str, bytes] = {}
        for name, payload in self.files.items():
            _require(type(name) is str and type(payload) is bytes, "invalid bundle payload types")
            frozen_files[name] = bytes(payload)
        frozen_snapshot: dict[str, BoundFile] = {}
        for role, bound in self.snapshot.items():
            _require(type(role) is str and type(bound) is BoundFile, "invalid snapshot binding")
            frozen_snapshot[role] = bound
        object.__setattr__(self, "files", MappingProxyType(frozen_files))
        object.__setattr__(self, "snapshot", MappingProxyType(frozen_snapshot))
        object.__setattr__(self, "manifest_payload", bytes(self.manifest_payload))

    @property
    def manifest(self) -> dict[str, Any]:
        return _strict_json(self.manifest_payload, label="in-memory candidate manifest")


def _validate_production_config(config: BuildConfig) -> None:
    _require(type(config) is BuildConfig, "production config type changed")
    _require(config == BuildConfig.production(), "production config is not exactly canonical")


def _capture_inputs(
    paths: RegistryInputPaths,
    *,
    evidence_root: Path,
    config: BuildConfig,
) -> dict[str, BoundFile]:
    _require(type(paths) is RegistryInputPaths, "input paths type changed")
    root = _absolute(evidence_root)
    if config.mode == "production":
        _validate_production_config(config)
        _require(root == _absolute(ROOT), "production evidence root changed")
        _require(paths == RegistryInputPaths.production(root), "production input paths changed")
    else:
        _require(root != _absolute(ROOT), "synthetic mode cannot read the production repository")
    role_paths = {
        "key_authority_manifest": paths.key_authority_manifest,
        "defect_authority_manifest": paths.defect_authority_manifest,
        "defect_authority_report": paths.defect_authority_report,
        "station_registry": paths.station_registry,
        "development_panel": paths.development_panel,
        "evaluation_panel": paths.evaluation_panel,
    }
    return {
        role: _read_stable_regular(path, root=root, label=role.replace("_", " "))
        for role, path in role_paths.items()
    }


def _binding(bound: BoundFile, *, root: Path) -> dict[str, object]:
    return {
        "path": _relative(bound.path, root),
        "sha256": bound.sha256,
        "size_bytes": bound.size_bytes,
    }


def _validate_binding_record(
    record: object,
    *,
    path: str,
    sha256: str,
    label: str,
) -> None:
    _require(type(record) is dict, f"{label} binding is not an object")
    _require(record.get("path") == path, f"{label} path binding changed")
    _require(record.get("sha256") == sha256, f"{label} SHA-256 binding changed")
    size = record.get("size_bytes")
    _require(type(size) is int and size >= 0, f"{label} size binding is invalid")


def _validate_authorities(
    bounds: Mapping[str, BoundFile],
    *,
    config: BuildConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    key_manifest = _strict_json(
        bounds["key_authority_manifest"].payload,
        label="information-regime key authority manifest",
    )
    defect_manifest = _strict_json(
        bounds["defect_authority_manifest"].payload,
        label="preprocessing-lineage defect authority manifest",
    )
    defect_report = _strict_json(
        bounds["defect_authority_report"].payload,
        label="preprocessing-lineage defect authority report",
    )

    _require(
        key_manifest.get("artifact_id")
        == "thermoroute-information-regime-key-registries-v4-phase1",
        "key authority artifact identity changed",
    )
    _require(
        key_manifest.get("status") == "PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY",
        "key authority status changed",
    )
    _require(
        key_manifest.get("registry_publication_scope") == "PHASE1_CONTRACT_ONLY",
        "key authority scope changed",
    )
    _require(
        key_manifest.get("execution_authorized_by_protocol") is False
        and key_manifest.get("model_or_score_output_published") is False
        and key_manifest.get("model_scores_read_or_accepted") is False,
        "key authority non-execution boundary changed",
    )
    _require(
        defect_manifest.get("artifact_id")
        == "thermoroute-preprocessing-lineage-defect-authority-v1"
        and defect_report.get("artifact_id")
        == "thermoroute-preprocessing-lineage-defect-authority-v1",
        "defect authority identity changed",
    )
    expected_defect_status = "SCIENTIFIC_RESULTS_WITHDRAWN_PENDING_OBSERVED_LINEAGE_RERUN"
    _require(
        defect_manifest.get("status") == expected_defect_status
        and defect_report.get("status") == expected_defect_status,
        "defect authority status changed",
    )
    required_rerun = defect_report.get("required_versioned_rerun")
    _require(type(required_rerun) is dict, "defect authority rerun requirements are absent")
    for field in (
        "raw_observed_flags_must_be_created_before_imputation",
        "observed_masks_must_survive_imputation_bit_exactly",
        "training_issue_and_target_admissibility_must_be_joined_from_raw_lineage",
        "training_key_registries_or_digests_must_be_authority_bound",
        "training_must_use_only_2006_2015_rows",
        "validation_must_use_only_independent_2016_2017_rows",
    ):
        _require(
            required_rerun.get(field) is True, f"defect authority requirement changed: {field}"
        )

    defect_inputs = defect_manifest.get("input_bindings")
    _require(type(defect_inputs) is dict, "defect authority input bindings are absent")
    _validate_binding_record(
        defect_inputs.get("score_independent_key_manifest"),
        path=(
            "outputs/final/information_regime_key_registries_v4/"
            "information_regime_key_registries_v4_manifest.json"
        ),
        sha256=bounds["key_authority_manifest"].sha256,
        label="defect-to-key-authority",
    )
    for authority_role, source_role in (
        ("station_registry", "station_registry"),
        ("training_panel", "development_panel"),
        ("evaluation_panel", "evaluation_panel"),
    ):
        _validate_binding_record(
            defect_inputs.get(authority_role),
            path=SOURCE_RELATIVE_PATHS[source_role],
            sha256=bounds[source_role].sha256,
            label=f"defect authority {source_role}",
        )
    output_binding = defect_manifest.get("output")
    _require(type(output_binding) is dict, "defect authority report binding is not an object")
    _require(
        output_binding.get("name") == DEFECT_AUTHORITY_REPORT.name,
        "defect authority report filename changed",
    )
    _require(
        output_binding.get("sha256") == bounds["defect_authority_report"].sha256
        and output_binding.get("size_bytes") == bounds["defect_authority_report"].size_bytes,
        "defect authority report byte binding changed",
    )

    key_inputs = key_manifest.get("input_bindings")
    _require(type(key_inputs) is dict, "key authority input bindings are absent")
    _validate_binding_record(
        key_inputs.get("station_registry_v1"),
        path=SOURCE_RELATIVE_PATHS["station_registry"],
        sha256=bounds["station_registry"].sha256,
        label="key authority station registry",
    )
    _validate_binding_record(
        key_inputs.get("raw_holdout_panel"),
        path=SOURCE_RELATIVE_PATHS["evaluation_panel"],
        sha256=bounds["evaluation_panel"].sha256,
        label="key authority evaluation panel",
    )

    if config.mode == "production":
        _require(
            bounds["key_authority_manifest"].sha256 == KEY_AUTHORITY_MANIFEST_SHA256,
            "formal key authority manifest SHA-256 changed",
        )
        _require(
            bounds["defect_authority_manifest"].sha256 == DEFECT_AUTHORITY_MANIFEST_SHA256,
            "formal defect authority manifest SHA-256 changed",
        )
        for role, expected in SOURCE_SHA256.items():
            _require(bounds[role].sha256 == expected, f"canonical {role} SHA-256 changed")
    return key_manifest, defect_manifest, defect_report


INPUT_PANEL_SCHEMA = pa.schema(
    [
        pa.field("DATE", pa.timestamp("ns"), nullable=True),
        pa.field("site_id", pa.string(), nullable=True),
        *(pa.field(variable, pa.float64(), nullable=True) for variable in PANEL_VARIABLES),
    ]
)


def _read_parquet_exact(
    bound: BoundFile,
    *,
    schema: pa.Schema,
    label: str,
) -> pd.DataFrame:
    try:
        observed_schema = pq.read_schema(io.BytesIO(bound.payload))
    except Exception as exc:
        raise SemanticRegistryError(f"{label} has unreadable Arrow schema") from exc
    _require(observed_schema.equals(schema), f"{label} Arrow schema/order changed")
    try:
        table = pq.read_table(io.BytesIO(bound.payload), schema=schema)
    except Exception as exc:
        raise SemanticRegistryError(f"{label} is not readable Parquet") from exc
    for field in schema:
        column = table[field.name]
        if field.name in {"DATE", "site_id"}:
            _require(column.null_count == 0, f"{label} {field.name} contains an Arrow null")
        elif field.name in PANEL_VARIABLES:
            valid = column.is_valid().to_numpy(zero_copy_only=False)
            values = column.to_numpy(zero_copy_only=False)
            _require(
                np.isfinite(values[valid]).all(),
                f"{label} {field.name} contains a valid non-finite Arrow value",
            )
    frame = table.to_pandas()
    _require(tuple(frame.columns) == tuple(schema.names), f"{label} column order changed")
    return frame.reset_index(drop=True)


def _load_station_registry(bound: BoundFile) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            io.BytesIO(bound.payload),
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    except Exception as exc:
        raise SemanticRegistryError("station registry is not readable CSV") from exc
    _require(
        tuple(frame.columns) == STATION_REGISTRY_COLUMNS,
        "station registry schema/order changed",
    )
    out = frame[["site_no", "legacy_site_id"]].copy()
    _strict_site_ids(out["site_no"], label="station registry site_no")
    legacy = _strict_string_series(out["legacy_site_id"], label="station registry legacy_site_id")
    _require(
        all(value.isascii() and value.strip() == value for value in legacy),
        "station registry legacy IDs must be exact unpadded ASCII strings",
    )
    _require(not out["site_no"].duplicated().any(), "station registry duplicates site_no")
    _require(
        not out["legacy_site_id"].duplicated().any(),
        "station registry duplicates legacy_site_id",
    )
    ordered = out.sort_values("site_no", kind="mergesort").reset_index(drop=True)
    return ordered


def _load_panel(bound: BoundFile, *, label: str) -> pd.DataFrame:
    frame = _read_parquet_exact(bound, schema=INPUT_PANEL_SCHEMA, label=label)
    _strict_dates(frame["DATE"], label=f"{label} DATE")
    _strict_string_series(frame["site_id"], label=f"{label} site_id")
    _require(
        not any(column.endswith("_observed") for column in frame.columns),
        f"{label} contains pre-existing observedness columns",
    )
    for variable in PANEL_VARIABLES:
        _strict_float_array(
            frame[variable],
            label=f"{label} {variable}",
            require_finite=False,
        )
    _require(
        not frame.duplicated(["site_id", "DATE"]).any(),
        f"{label} duplicates a source station-day identity",
    )
    return frame


def _add_raw_observedness(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Create exact observed flags from raw bytes; no fill operation exists here."""

    _require(tuple(frame.columns) == PANEL_COLUMNS, f"{label} raw schema/order changed")
    out = frame.copy(deep=True).reset_index(drop=True)
    for variable in PANEL_VARIABLES:
        values = _strict_float_array(
            out[variable],
            label=f"{label} raw {variable}",
            require_finite=False,
        )
        observed = np.isfinite(values)
        canonical = values.astype(np.float64, copy=True)
        canonical[~observed] = np.nan
        out[variable] = canonical
        out[f"{variable}_observed"] = observed.astype(np.bool_, copy=False)
    _require(tuple(out.columns) == DAILY_COLUMNS, f"{label} observed schema/order changed")
    return out


def _equal_raw_cells(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.equal(left, right) | (np.isnan(left) & np.isnan(right))


def validate_daily_registry(
    frame: pd.DataFrame,
    *,
    expected_sites: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate exact daily schema, types, masks, identity, order, and calendar."""

    _require(type(frame) is pd.DataFrame, "daily registry must be a DataFrame")
    _require(tuple(frame.columns) == DAILY_COLUMNS, "daily registry schema/order is not exact")
    _require(
        isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1,
        "daily registry index must be the canonical RangeIndex",
    )
    _strict_dates(frame["DATE"], label="daily registry DATE")
    sites = _strict_site_ids(frame["site_id"], label="daily registry site_id")
    _require(
        not frame.duplicated(["site_id", "DATE"]).any(),
        "daily registry duplicates a station-day identity",
    )
    identity = pd.MultiIndex.from_arrays([frame["site_id"], frame["DATE"]])
    _require(identity.is_monotonic_increasing, "daily registry ordering is not site_id, DATE")
    for variable in PANEL_VARIABLES:
        values = _strict_float_array(
            frame[variable],
            label=f"daily registry {variable}",
            require_finite=False,
        )
        observed = _strict_bool_array(
            frame[f"{variable}_observed"],
            label=f"daily registry {variable}_observed",
        )
        _require(
            np.array_equal(observed, np.isfinite(values)),
            f"daily registry {variable}_observed is not raw finite({variable})",
        )
    unique_sites = tuple(sorted(set(sites)))
    if expected_sites is not None:
        expected = tuple(expected_sites)
        _require(tuple(sorted(expected)) == expected, "expected station IDs are not sorted")
        _require(unique_sites == expected, "daily registry station inventory changed")
    reference_dates: np.ndarray | None = None
    for site, group in frame.groupby("site_id", sort=True, observed=True):
        dates = _strict_dates(group["DATE"], label=f"daily registry {site} DATE")
        _require(
            len(dates) == 1 or np.all(np.diff(dates) == np.timedelta64(1, "D")),
            f"daily registry contains a calendar gap for {site}",
        )
        if reference_dates is None:
            reference_dates = dates
        else:
            _require(
                len(dates) == len(reference_dates)
                and dates[0] == reference_dates[0]
                and dates[-1] == reference_dates[-1]
                and np.array_equal(dates, reference_dates),
                "daily registry stations do not share one exact date vector",
            )
    _require(reference_dates is not None, "daily registry contains no station calendars")
    return frame


def _combine_source_panels(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    station_registry: pd.DataFrame,
    *,
    config: BuildConfig,
) -> tuple[pd.DataFrame, PanelInventory]:
    alias = dict(
        zip(
            station_registry["legacy_site_id"].to_numpy(copy=True),
            station_registry["site_no"].to_numpy(copy=True),
            strict=True,
        )
    )
    development_mapped = development.copy(deep=True)
    mapped = development_mapped["site_id"].map(alias)
    _require(mapped.notna().all(), "development panel contains an unmapped legacy site ID")
    development_mapped["site_id"] = mapped.to_numpy(copy=True)
    _strict_site_ids(development_mapped["site_id"], label="mapped development panel")
    _strict_site_ids(evaluation["site_id"], label="evaluation panel")
    expected_sites = tuple(station_registry["site_no"].to_numpy(copy=True))
    _require(
        set(development_mapped["site_id"]) == set(expected_sites),
        "development panel station set differs from the station registry",
    )
    _require(
        set(evaluation["site_id"]) == set(expected_sites),
        "evaluation panel station set differs from the station registry",
    )

    # Observedness is deliberately materialized on each raw source before
    # overlap selection.  No imputer or fill-capable object is imported.
    development_raw = _add_raw_observedness(development_mapped, label="development panel")
    evaluation_raw = _add_raw_observedness(evaluation, label="evaluation panel")

    overlap = development_raw.merge(
        evaluation_raw,
        on=["site_id", "DATE"],
        how="inner",
        validate="one_to_one",
        suffixes=("_development", "_evaluation"),
        sort=False,
    )
    overlap = overlap.sort_values(["site_id", "DATE"], kind="mergesort").reset_index(drop=True)
    overlap_differences: dict[str, int] = {}
    for variable in PANEL_VARIABLES:
        left = _strict_float_array(
            overlap[f"{variable}_development"],
            label=f"overlap development {variable}",
            require_finite=False,
        )
        right = _strict_float_array(
            overlap[f"{variable}_evaluation"],
            label=f"overlap evaluation {variable}",
            require_finite=False,
        )
        overlap_differences[variable] = int((~_equal_raw_cells(left, right)).sum())

    overlap_identity = pd.MultiIndex.from_frame(overlap[["site_id", "DATE"]])
    development_identity = pd.MultiIndex.from_frame(development_raw[["site_id", "DATE"]])
    selected_development = development_raw.loc[~development_identity.isin(overlap_identity)].copy()
    combined = pd.concat([selected_development, evaluation_raw], ignore_index=True)
    combined = combined.sort_values(["site_id", "DATE"], kind="mergesort").reset_index(drop=True)
    validate_daily_registry(combined, expected_sites=expected_sites)

    # Prove evaluation-wins cell content, not merely duplicate elimination.
    selected_overlap = combined.merge(
        overlap[["site_id", "DATE"]],
        on=["site_id", "DATE"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["site_id", "DATE"], kind="mergesort")
    expected_overlap = evaluation_raw.merge(
        overlap[["site_id", "DATE"]],
        on=["site_id", "DATE"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["site_id", "DATE"], kind="mergesort")
    selected_overlap = selected_overlap.reset_index(drop=True)
    expected_overlap = expected_overlap.reset_index(drop=True)
    _require(
        _canonical_frame_sha256(selected_overlap, DAILY_COLUMNS)
        == _canonical_frame_sha256(expected_overlap, DAILY_COLUMNS),
        "combined overlap cells do not exactly equal evaluation-source cells",
    )

    overlap_start = None if overlap.empty else overlap["DATE"].min().date().isoformat()
    overlap_end = None if overlap.empty else overlap["DATE"].max().date().isoformat()
    inventory = PanelInventory(
        development_rows=len(development_raw),
        evaluation_rows=len(evaluation_raw),
        overlap_rows=len(overlap),
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        overlap_differences=MappingProxyType(overlap_differences),
        selected_development_rows=len(selected_development),
        selected_evaluation_rows=len(evaluation_raw),
    )
    _require(
        len(combined) == len(development_raw) + len(evaluation_raw) - len(overlap),
        "daily union row arithmetic changed",
    )
    if config.mode == "production":
        expected_overlap_identity = pd.MultiIndex.from_product(
            [
                expected_sites,
                pd.date_range(
                    EXPECTED_PRODUCTION_COUNTS["overlap_start"],
                    EXPECTED_PRODUCTION_COUNTS["overlap_end"],
                    freq="D",
                ),
            ],
            names=["site_id", "DATE"],
        )
        _require(
            overlap_identity.equals(expected_overlap_identity),
            "production overlap is not the exact 120-site by 32-day Cartesian product",
        )
        expected = EXPECTED_PRODUCTION_COUNTS
        observed = {
            "development_rows": inventory.development_rows,
            "evaluation_rows": inventory.evaluation_rows,
            "overlap_rows": inventory.overlap_rows,
            "daily_rows": len(combined),
            "station_count": len(expected_sites),
            "daily_start": combined["DATE"].min().date().isoformat(),
            "daily_end": combined["DATE"].max().date().isoformat(),
            "overlap_start": inventory.overlap_start,
            "overlap_end": inventory.overlap_end,
        }
        for field, value in observed.items():
            _require(value == expected[field], f"production {field} changed: {value}")
    return combined, inventory


def _training_key_id(
    site_id: str,
    split: str,
    issue_date: pd.Timestamp,
    target_date: pd.Timestamp,
    lead_days: int,
) -> str:
    payload = (
        "thermoroute-training-example-v4|"
        f"{split}|{site_id}|{issue_date.date().isoformat()}|"
        f"{target_date.date().isoformat()}|{lead_days}"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _split_ranges(config: BuildConfig) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        "train": (pd.Timestamp(config.train_start), pd.Timestamp(config.train_end)),
        "validation": (
            pd.Timestamp(config.validation_start),
            pd.Timestamp(config.validation_end),
        ),
    }


def _build_training_examples(daily: pd.DataFrame, *, config: BuildConfig) -> pd.DataFrame:
    validate_daily_registry(daily)
    rows: list[pd.DataFrame] = []
    for split, (lower, upper) in _split_ranges(config).items():
        for lead in config.leads:
            for site, group in daily.groupby("site_id", sort=True, observed=True):
                issue_dates = group["DATE"].to_numpy(copy=True)
                water = _strict_float_array(
                    group["WTEMP"],
                    label=f"training-source {site} WTEMP",
                    require_finite=False,
                )
                observed = _strict_bool_array(
                    group["WTEMP_observed"],
                    label=f"training-source {site} WTEMP_observed",
                )
                target_dates = issue_dates + np.timedelta64(lead, "D")
                target_values = np.full(len(group), np.nan, dtype=np.float64)
                target_observed = np.zeros(len(group), dtype=np.bool_)
                if lead < len(group):
                    target_values[:-lead] = water[lead:]
                    target_observed[:-lead] = observed[lead:]
                selected = (
                    (issue_dates >= lower.to_datetime64())
                    & (issue_dates <= upper.to_datetime64())
                    & (target_dates >= lower.to_datetime64())
                    & (target_dates <= upper.to_datetime64())
                    & observed
                    & target_observed
                    & np.isfinite(target_values)
                )
                if not selected.any():
                    continue
                selected_issue = pd.Series(issue_dates[selected], dtype="datetime64[ns]")
                selected_target = pd.Series(target_dates[selected], dtype="datetime64[ns]")
                count = int(selected.sum())
                rows.append(
                    pd.DataFrame(
                        {
                            "site_id": np.full(count, site, dtype=object),
                            "split": np.full(count, split, dtype=object),
                            "issue_date": selected_issue,
                            "target_date": selected_target,
                            "lead_days": np.full(count, lead, dtype=np.int16),
                            "y_true": target_values[selected].astype(np.float64, copy=False),
                            "issue_wtemp_observed": np.ones(count, dtype=np.bool_),
                            "target_wtemp_observed": np.ones(count, dtype=np.bool_),
                        }
                    )
                )
    _require(bool(rows), "no admissible 2006-2017 training/validation examples were found")
    frame = pd.concat(rows, ignore_index=True)
    split_rank = frame["split"].map({"train": 0, "validation": 1})
    _require(split_rank.notna().all(), "training registry contains an unknown split")
    frame = (
        frame.assign(_split_rank=split_rank.astype(np.int8))
        .sort_values(
            ["_split_rank", "lead_days", "site_id", "issue_date"],
            kind="mergesort",
        )
        .drop(columns="_split_rank")
        .reset_index(drop=True)
    )
    frame.insert(
        0,
        "training_key_id",
        [
            _training_key_id(site, split, issue, target, int(lead))
            for site, split, issue, target, lead in zip(
                frame["site_id"],
                frame["split"],
                frame["issue_date"],
                frame["target_date"],
                frame["lead_days"],
                strict=True,
            )
        ],
    )
    frame = frame[list(TRAINING_COLUMNS)]
    validate_training_registry(
        frame,
        daily=daily,
        config=config,
        verify_derived_key_ids=False,
        verify_daily_lineage=False,
    )
    return frame


def _training_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        selected = frame.loc[frame["split"].eq(split)]
        counts[split] = {str(lead): int(selected["lead_days"].eq(lead).sum()) for lead in LEADS}
    return counts


def _independent_training_counts(
    daily: pd.DataFrame,
    *,
    config: BuildConfig,
) -> dict[str, dict[str, int]]:
    """Count admissible examples without shifts or the registry constructor."""

    result = {split: {str(lead): 0 for lead in config.leads} for split in SPLITS}
    for _site, group in daily.groupby("site_id", sort=False, observed=True):
        date_values = _strict_dates(group["DATE"], label="independent-count dates")
        observed_values = _strict_bool_array(
            group["WTEMP_observed"],
            label="independent-count WTEMP observedness",
        )
        observation_by_date = {
            int(raw_date.astype("datetime64[ns]").astype(np.int64)): bool(raw_observed)
            for raw_date, raw_observed in zip(date_values, observed_values, strict=True)
        }
        for split, (lower, upper) in _split_ranges(config).items():
            lower_value = lower.to_datetime64()
            upper_value = upper.to_datetime64()
            for lead in config.leads:
                count = 0
                delta = np.timedelta64(lead, "D")
                for issue_date, issue_observed in zip(
                    date_values,
                    observed_values,
                    strict=True,
                ):
                    target_date = issue_date + delta
                    if (
                        issue_date < lower_value
                        or issue_date > upper_value
                        or target_date < lower_value
                        or target_date > upper_value
                        or not issue_observed
                    ):
                        continue
                    target_key = int(target_date.astype("datetime64[ns]").astype(np.int64))
                    if observation_by_date.get(target_key) is True:
                        count += 1
                result[split][str(lead)] += count
    return result


def validate_training_registry(
    frame: pd.DataFrame,
    *,
    daily: pd.DataFrame,
    config: BuildConfig,
    verify_derived_key_ids: bool = True,
    verify_daily_lineage: bool = True,
) -> pd.DataFrame:
    """Validate exact schema, raw label lineage, chronology, identity, and order."""

    _require(type(frame) is pd.DataFrame, "training registry must be a DataFrame")
    _require(
        tuple(frame.columns) == TRAINING_COLUMNS,
        "training example registry schema/order is not exact",
    )
    _require(
        isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1,
        "training registry index must be the canonical RangeIndex",
    )
    training_key_ids = _strict_string_series(
        frame["training_key_id"],
        label="training training_key_id",
    )
    _require(
        all(
            len(value) == 64
            and value.isascii()
            and all(character in "0123456789abcdef" for character in value)
            for value in training_key_ids
        ),
        "training registry contains a non-canonical training_key_id",
    )
    _strict_site_ids(frame["site_id"], label="training registry site_id")
    _strict_string_series(frame["split"], label="training registry split", allowed=set(SPLITS))
    issue_dates = _strict_dates(frame["issue_date"], label="training registry issue_date")
    target_dates = _strict_dates(frame["target_date"], label="training registry target_date")
    leads = _strict_int16_array(frame["lead_days"], label="training registry lead_days")
    _require({int(value) for value in leads} == set(config.leads), "training lead set changed")
    y_true = _strict_float_array(
        frame["y_true"],
        label="training registry y_true",
        require_finite=True,
    )
    issue_observed = _strict_bool_array(
        frame["issue_wtemp_observed"],
        label="training registry issue_wtemp_observed",
    )
    target_observed = _strict_bool_array(
        frame["target_wtemp_observed"],
        label="training registry target_wtemp_observed",
    )
    _require(
        issue_observed.all() and target_observed.all(),
        "training registry contains an inadmissible raw issue or target",
    )
    _require(
        np.array_equal(target_dates, issue_dates + leads.astype("timedelta64[D]")),
        "training target_date != issue_date + lead_days",
    )
    for split, (lower, upper) in _split_ranges(config).items():
        selected = frame["split"].eq(split).to_numpy()
        _require(selected.any(), f"training registry contains no {split} examples")
        lower_value = lower.to_datetime64()
        upper_value = upper.to_datetime64()
        _require(
            bool(
                (
                    (issue_dates[selected] >= lower_value)
                    & (issue_dates[selected] <= upper_value)
                    & (target_dates[selected] >= lower_value)
                    & (target_dates[selected] <= upper_value)
                ).all()
            ),
            f"{split} examples cross their exact temporal split",
        )
    _require(
        not frame.duplicated("training_key_id").any(),
        "training registry duplicates training_key_id",
    )
    identity_columns = ["site_id", "split", "issue_date", "target_date", "lead_days"]
    _require(
        not frame.duplicated(identity_columns).any(),
        "training registry duplicates a semantic example identity",
    )

    expected_order = (
        frame.assign(_split_rank=frame["split"].map({"train": 0, "validation": 1}))
        .sort_values(
            ["_split_rank", "lead_days", "site_id", "issue_date"],
            kind="mergesort",
        )
        .index.to_numpy()
    )
    _require(
        np.array_equal(expected_order, np.arange(len(frame))),
        "training registry order is not split, lead_days, site_id, issue_date",
    )

    if verify_derived_key_ids:
        for index, (observed_id, site, split, issue, target, lead) in enumerate(
            zip(
                training_key_ids,
                frame["site_id"],
                frame["split"],
                frame["issue_date"],
                frame["target_date"],
                leads,
                strict=True,
            )
        ):
            expected_id = _training_key_id(site, split, issue, target, int(lead))
            _require(
                observed_id == expected_id,
                f"training training_key_id changed at row {index}",
            )

    if verify_daily_lineage:
        validate_daily_registry(daily)
        lookup = daily.set_index(["site_id", "DATE"], verify_integrity=True)
        issue_keys = pd.MultiIndex.from_arrays([frame["site_id"], frame["issue_date"]])
        target_keys = pd.MultiIndex.from_arrays([frame["site_id"], frame["target_date"]])
        issue_lineage = lookup["WTEMP_observed"].reindex(issue_keys)
        target_lineage = lookup[["WTEMP", "WTEMP_observed"]].reindex(target_keys)
        _require(
            issue_lineage.notna().all() and target_lineage.notna().all(axis=None),
            "training registry has an identity outside the daily raw registry",
        )
        source_issue_observed = _strict_bool_array(
            issue_lineage.reset_index(drop=True),
            label="joined issue observedness",
        )
        source_target_observed = _strict_bool_array(
            target_lineage["WTEMP_observed"].reset_index(drop=True),
            label="joined target observedness",
        )
        source_targets = _strict_float_array(
            target_lineage["WTEMP"].reset_index(drop=True),
            label="joined target WTEMP",
            require_finite=True,
        )
        _require(
            source_issue_observed.all() and source_target_observed.all(),
            "training registry is not admitted from raw observedness",
        )
        _require(
            np.array_equal(source_targets, y_true),
            "training y_true differs from exact raw target WTEMP",
        )
    return frame


def _validate_daily_arrow_table(table: pa.Table, *, label: str) -> None:
    _require(table.schema.equals(DAILY_SCHEMA), f"{label} Arrow schema changed")
    for field in DAILY_SCHEMA:
        column = table[field.name]
        if field.name in PANEL_VARIABLES:
            observed_column = table[f"{field.name}_observed"]
            _require(
                observed_column.null_count == 0,
                f"{label} {field.name}_observed contains an Arrow null",
            )
            valid = column.is_valid().to_numpy(zero_copy_only=False)
            observed = observed_column.to_numpy(zero_copy_only=False)
            _require(
                observed.dtype == np.dtype("bool") and np.array_equal(valid, observed),
                f"{label} {field.name} Arrow validity differs from observedness",
            )
            values = column.to_numpy(zero_copy_only=False)
            _require(
                np.isfinite(values[valid]).all(),
                f"{label} {field.name} has a valid non-finite value",
            )
        else:
            _require(column.null_count == 0, f"{label} {field.name} contains an Arrow null")


def _exact_arrow_table(frame: pd.DataFrame, *, schema: pa.Schema) -> pa.Table:
    """Encode raw missingness as Arrow null iff its explicit observed flag is false."""

    _require(tuple(frame.columns) == tuple(schema.names), "Arrow frame/schema order changed")
    arrays: list[pa.Array] = []
    try:
        for field in schema:
            series = frame[field.name]
            if pa.types.is_floating(field.type):
                float_values = _strict_float_array(
                    series,
                    label=f"Arrow {field.name}",
                    require_finite=(field.name == "y_true"),
                )
                if field.name in PANEL_VARIABLES:
                    observed = _strict_bool_array(
                        frame[f"{field.name}_observed"],
                        label=f"Arrow {field.name}_observed",
                    )
                    _require(
                        np.array_equal(observed, np.isfinite(float_values)),
                        f"Arrow {field.name} raw observedness changed",
                    )
                    array = pa.array(
                        float_values,
                        mask=~observed,
                        type=field.type,
                        from_pandas=False,
                        safe=True,
                    )
                else:
                    array = pa.array(
                        float_values,
                        type=field.type,
                        from_pandas=False,
                        safe=True,
                    )
            elif pa.types.is_boolean(field.type):
                values = _strict_bool_array(series, label=f"Arrow {field.name}")
                array = pa.array(values, type=field.type, from_pandas=False, safe=True)
            elif pa.types.is_timestamp(field.type):
                values = _strict_dates(series, label=f"Arrow {field.name}")
                array = pa.array(values, type=field.type, from_pandas=False, safe=True)
            elif pa.types.is_int16(field.type):
                values = _strict_int16_array(series, label=f"Arrow {field.name}")
                array = pa.array(values, type=field.type, from_pandas=False, safe=True)
            elif pa.types.is_string(field.type):
                values = list(_strict_string_series(series, label=f"Arrow {field.name}"))
                array = pa.array(values, type=field.type, from_pandas=False, safe=True)
            else:  # pragma: no cover - schemas above are a closed allowlist
                raise SemanticRegistryError(f"unsupported Arrow field type: {field}")
            if not field.nullable:
                _require(array.null_count == 0, f"Arrow {field.name} contains a forbidden null")
            arrays.append(array)
        table = pa.Table.from_arrays(arrays, schema=schema)
    except SemanticRegistryError:
        raise
    except Exception as exc:
        raise SemanticRegistryError(
            "cannot coerce candidate frame to exact non-null Arrow arrays"
        ) from exc
    _require(table.schema.equals(schema), "candidate Arrow schema coercion changed")
    if schema.equals(DAILY_SCHEMA):
        _validate_daily_arrow_table(table, label="candidate daily table")
    else:
        _require(
            all(column.null_count == 0 for column in table.columns),
            "candidate Arrow table contains a forbidden null",
        )
    return table


def _parquet_bytes(frame: pd.DataFrame, *, schema: pa.Schema) -> bytes:
    try:
        table = _exact_arrow_table(frame, schema=schema)
    except Exception as exc:
        if isinstance(exc, SemanticRegistryError):
            raise
        raise SemanticRegistryError(
            "cannot coerce candidate frame to its exact Arrow schema"
        ) from exc
    sink = io.BytesIO()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
        row_group_size=max(1, len(frame)),
    )
    return sink.getvalue()


def _parse_candidate_parquet(
    payload: bytes,
    *,
    schema: pa.Schema,
    label: str,
) -> pd.DataFrame:
    try:
        observed_schema = pq.read_schema(io.BytesIO(payload))
        _require(observed_schema.equals(schema), f"{label} Arrow schema changed")
        table = pq.read_table(io.BytesIO(payload), schema=schema)
        if schema.equals(DAILY_SCHEMA):
            _validate_daily_arrow_table(table, label=label)
        else:
            _require(
                all(column.null_count == 0 for column in table.columns),
                f"{label} contains a forbidden Arrow null",
            )
        frame = table.to_pandas()
    except SemanticRegistryError:
        raise
    except Exception as exc:
        raise SemanticRegistryError(f"{label} candidate Parquet is unreadable") from exc
    return frame.reset_index(drop=True)


def _daily_inventory(daily: pd.DataFrame) -> dict[str, Any]:
    validate_daily_registry(daily)
    observed_counts = {
        variable: int(
            _strict_bool_array(
                daily[f"{variable}_observed"],
                label=f"inventory {variable}_observed",
            ).sum()
        )
        for variable in PANEL_VARIABLES
    }
    finite_counts = {
        variable: int(
            np.isfinite(
                _strict_float_array(
                    daily[variable],
                    label=f"inventory {variable}",
                    require_finite=False,
                )
            ).sum()
        )
        for variable in PANEL_VARIABLES
    }
    _require(observed_counts == finite_counts, "independent daily observed counts differ")
    calendar_rows = 0
    for _site, group in daily.groupby("site_id", sort=False, observed=True):
        first = group["DATE"].iloc[0]
        last = group["DATE"].iloc[-1]
        calendar_rows += int((last - first).days) + 1
    _require(calendar_rows == len(daily), "daily count differs from independent calendar count")
    return {
        "row_count": len(daily),
        "independent_calendar_row_count": calendar_rows,
        "station_count": int(daily["site_id"].nunique()),
        "start": daily["DATE"].min().date().isoformat(),
        "end": daily["DATE"].max().date().isoformat(),
        "observed_counts_from_boolean_masks": observed_counts,
        "observed_counts_from_raw_finiteness": finite_counts,
        "missing_counts": {
            variable: len(daily) - observed_counts[variable] for variable in PANEL_VARIABLES
        },
        "all_eight_variable_observed_mask_semantic_sha256": _canonical_frame_sha256(
            daily,
            OBSERVED_COLUMNS,
        ),
    }


def _validate_counts_against_defect_authority(
    daily_inventory: Mapping[str, Any],
    training_counts: Mapping[str, Mapping[str, int]],
    defect_report: Mapping[str, Any],
) -> None:
    raw_inventory = defect_report.get("raw_panel_inventory")
    _require(type(raw_inventory) is dict, "defect authority raw-panel inventory is absent")
    _require(
        raw_inventory.get("combined_rows") == daily_inventory["row_count"]
        and raw_inventory.get("stations") == daily_inventory["station_count"]
        and raw_inventory.get("combined_start") == daily_inventory["start"]
        and raw_inventory.get("combined_end") == daily_inventory["end"],
        "daily registry inventory differs from the defect authority",
    )
    declared_missing = raw_inventory.get("combined_missing")
    _require(type(declared_missing) is dict, "defect authority missing inventory is absent")
    for variable, missing in declared_missing.items():
        _require(
            daily_inventory["missing_counts"].get(variable) == missing,
            f"daily raw missing count differs from defect authority for {variable}",
        )
    temporal = defect_report.get("training_lineage_inventory")
    _require(type(temporal) is dict, "defect authority training inventory is absent")
    declared_by_horizon = temporal.get("by_horizon")
    _require(type(declared_by_horizon) is dict, "defect authority horizon inventory is absent")
    for split, authority_split in (("train", "train"), ("validation", "val")):
        for lead in LEADS:
            lead_record = declared_by_horizon.get(str(lead))
            _require(type(lead_record) is dict, f"defect authority lacks lead {lead}")
            split_record = lead_record.get(authority_split)
            _require(type(split_record) is dict, f"defect authority lacks {split} lead {lead}")
            _require(
                split_record.get("admissible_rows") == training_counts[split][str(lead)],
                f"training count differs from defect authority for {split} lead {lead}",
            )


def _output_record(
    payload: bytes,
    frame: pd.DataFrame,
    *,
    schema: pa.Schema,
    identity_columns: Sequence[str],
) -> dict[str, Any]:
    record = {
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "row_count": len(frame),
        "columns": list(frame.columns),
        "arrow_schema": _schema_signature(schema),
        "ordered_semantic_content_sha256": _canonical_frame_sha256(frame, frame.columns),
        "ordered_identity_sha256": _canonical_frame_sha256(frame, identity_columns),
    }
    if "training_key_id" in frame.columns:
        record["derived_training_key_id_sha256"] = _canonical_frame_sha256(
            frame,
            ("training_key_id",),
        )
    return record


def build_candidate_artifacts(
    paths: RegistryInputPaths,
    *,
    config: BuildConfig,
    evidence_root: Path,
) -> CandidateBundle:
    """Build deterministic candidate bytes in memory; write nothing."""

    _require(type(config) is BuildConfig, "build config type changed")
    evidence_root = _absolute(evidence_root)
    runtime = _validate_candidate_runtime(enforce=config.enforce_route_a_runtime)
    bounds = _capture_inputs(paths, evidence_root=evidence_root, config=config)
    builder_bound = _read_stable_regular(
        Path(__file__),
        root=ROOT,
        label="semantic-registry builder code",
    )
    bounds["builder_code"] = builder_bound
    _key_manifest, _defect_manifest, defect_report = _validate_authorities(
        bounds,
        config=config,
    )

    station_registry = _load_station_registry(bounds["station_registry"])
    development = _load_panel(bounds["development_panel"], label="development panel")
    evaluation = _load_panel(bounds["evaluation_panel"], label="evaluation panel")
    daily, source_inventory = _combine_source_panels(
        development,
        evaluation,
        station_registry,
        config=config,
    )
    training_examples = _build_training_examples(daily, config=config)
    daily_inventory = _daily_inventory(daily)
    registry_counts = _training_counts(training_examples)
    independent_counts = _independent_training_counts(daily, config=config)
    _require(
        registry_counts == independent_counts,
        "training registry counts differ from independent raw-calendar reconstruction",
    )
    _validate_counts_against_defect_authority(
        daily_inventory,
        registry_counts,
        defect_report,
    )
    if config.expected_counts is not None:
        _require(
            registry_counts == dict(config.expected_counts["training_examples"]),
            "training registry counts differ from the production pins",
        )
        _require(
            daily_inventory["missing_counts"] == dict(config.expected_counts["daily_missing"]),
            "daily raw missing counts differ from the production pins",
        )

    daily_payload = _parquet_bytes(daily, schema=DAILY_SCHEMA)
    training_payload = _parquet_bytes(training_examples, schema=TRAINING_SCHEMA)
    outputs = {
        DAILY_FILENAME: _output_record(
            daily_payload,
            daily,
            schema=DAILY_SCHEMA,
            identity_columns=("site_id", "DATE"),
        ),
        TRAINING_FILENAME: _output_record(
            training_payload,
            training_examples,
            schema=TRAINING_SCHEMA,
            identity_columns=(
                "site_id",
                "split",
                "issue_date",
                "target_date",
                "lead_days",
            ),
        ),
    }
    input_bindings = {
        role: _binding(bound, root=ROOT if role == "builder_code" else evidence_root)
        for role, bound in sorted(bounds.items())
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": STATUS,
        "build_mode": (
            "PRODUCTION_INPUTS_CANDIDATE_OUTPUT"
            if config.mode == "production"
            else "SYNTHETIC_CONTRACT_TEST"
        ),
        "formal_authority": False,
        "execution_authorized": False,
        "model_execution_authorized": False,
        "protocol_sealed_or_amended": False,
        "model_or_score_output_published": False,
        "model_predictions_scores_effects_or_contrasts_read": False,
        "candidate_warning": (
            "Scratch-only semantic data-registry candidate. Review and a separate formal "
            "authorization process are required before any runner may consume it."
        ),
        "formal_authority_dependencies": {
            "information_regime_key_registries_v4_manifest_sha256": (
                bounds["key_authority_manifest"].sha256
            ),
            "preprocessing_lineage_defect_authority_v1_manifest_sha256": (
                bounds["defect_authority_manifest"].sha256
            ),
            "dependency_authority_does_not_transfer_to_candidate": True,
        },
        "input_bindings": input_bindings,
        "runtime": runtime,
        "semantic_contract": {
            "ordered_semantic_digest": {
                "algorithm": "SHA-256",
                "domain_prefix_utf8": "thermoroute-semantic-data-frame-v4\u0000",
                "row_and_column_counts": "unsigned_64_bit_little_endian",
                "row_order": "physical_registry_order",
                "column_order": "declared_digest_column_order",
                "per_column_header": (
                    "uint64_le_length_prefixed_UTF-8_column_name_then_"
                    "uint64_le_length_prefixed_ASCII_pandas_dtype_string"
                ),
                "type_tags": {
                    "bool": "B_then_raw_uint8_values",
                    "datetime64_ns": "D_then_signed_int64_little_endian_nanoseconds",
                    "int16": "I_then_values_promoted_to_signed_int64_little_endian",
                    "float": (
                        "F_then_finite_presence_uint8_bitmap_then_float64_little_endian_values_"
                        "with_missing_canonicalized_to_positive_zero"
                    ),
                    "string": "S_then_each_UTF-8_value_uint64_le_length_prefixed",
                },
                "pandas_dtype_string_included": True,
                "parquet_metadata_included": False,
            },
            "daily_raw_observed_panel_registry": {
                "identity": ["site_id", "DATE"],
                "order": ["site_id", "DATE"],
                "raw_observed_definition": "isfinite(value) in captured source before imputation",
                "imputation_performed": False,
                "overlap_policy": "evaluation_panel_wins",
                "overlap_selection_precedence": ["development_panel", "evaluation_panel"],
                "strict_daily_calendar_per_station": True,
                "mask_digest_scope": "all_8_panel_variables_including_WLEVEL",
                "defect_authority_mask_digest_relationship": (
                    "SEPARATE_NOT_EQUAL: the v1 defect authority mask digest covers its seven "
                    "model-used variables and omits WLEVEL"
                ),
                "raw_missing_values_preserved": True,
                "raw_missing_values_filled": False,
                "raw_missing_physical_encoding": (
                    "nullable Arrow float64; validity bitmap equals the corresponding observed "
                    "bool exactly; every valid value is finite"
                ),
            },
            "training_example_registry_2006_2017": {
                "identity": [
                    "site_id",
                    "split",
                    "issue_date",
                    "target_date",
                    "lead_days",
                ],
                "derived_key": {
                    "column": "training_key_id",
                    "algorithm": (
                        "sha256('thermoroute-training-example-v4|' + split + '|' + site_id + "
                        "'|' + issue_date + '|' + target_date + '|' + lead_days)"
                    ),
                    "part_of_natural_identity": False,
                },
                "order": ["split(train,validation)", "lead_days", "site_id", "issue_date"],
                "lead_days": list(config.leads),
                "train_period": [config.train_start.isoformat(), config.train_end.isoformat()],
                "validation_period": [
                    config.validation_start.isoformat(),
                    config.validation_end.isoformat(),
                ],
                "issue_and_target_must_be_in_same_split": True,
                "admissibility": (
                    "issue WTEMP raw-observed AND target WTEMP raw-observed AND y_true exact "
                    "raw target WTEMP"
                ),
                "imputed_labels_allowed": False,
            },
        },
        "source_panel_inventory": {
            "development_rows": source_inventory.development_rows,
            "evaluation_rows": source_inventory.evaluation_rows,
            "overlap_rows": source_inventory.overlap_rows,
            "overlap_start": source_inventory.overlap_start,
            "overlap_end": source_inventory.overlap_end,
            "overlap_raw_cell_differences_by_variable": dict(source_inventory.overlap_differences),
            "selected_development_rows": source_inventory.selected_development_rows,
            "selected_evaluation_rows": source_inventory.selected_evaluation_rows,
            "independent_union_arithmetic_rows": (
                source_inventory.development_rows
                + source_inventory.evaluation_rows
                - source_inventory.overlap_rows
            ),
        },
        "daily_registry_inventory": daily_inventory,
        "training_example_counts": {
            "from_materialized_registry": registry_counts,
            "independent_raw_calendar_reconstruction": independent_counts,
            "counts_match": True,
            "total": len(training_examples),
        },
        "outputs": outputs,
        "output_contract": {
            "caller_must_specify_destination": True,
            "scratch_candidate_only": True,
            "outputs_final_destination_prohibited": True,
            "exact_file_count": 3,
            "create_only": True,
            "overwrite_supported": False,
            "resume_supported": False,
            "independent_build_passes_before_write": 2,
            "atomic_linux_RENAME_NOREPLACE": True,
            "anchored_parent_directory": True,
            "staged_bytes_reverified_before_rename": True,
        },
    }
    manifest_payload = _canonical_json_bytes(manifest)
    files = {
        DAILY_FILENAME: daily_payload,
        TRAINING_FILENAME: training_payload,
        MANIFEST_FILENAME: manifest_payload,
    }
    bundle = CandidateBundle(
        files=files,
        daily=daily,
        training_examples=training_examples,
        manifest_payload=manifest_payload,
        snapshot=bounds,
        _token=_BUNDLE_TOKEN,
    )
    _verify_bundle(bundle, config=config)
    return bundle


def _verify_bundle(bundle: CandidateBundle, *, config: BuildConfig) -> None:
    _require(type(bundle) is CandidateBundle, "candidate bundle type was forged")
    _require(bundle._token is _BUNDLE_TOKEN, "candidate bundle issuer token changed")
    _require(
        set(bundle.files) == {DAILY_FILENAME, TRAINING_FILENAME, MANIFEST_FILENAME},
        "candidate bundle file set changed",
    )
    _require(
        bundle.files[MANIFEST_FILENAME] == bundle.manifest_payload,
        "candidate manifest payload changed",
    )
    manifest = _strict_json(bundle.manifest_payload, label="candidate manifest")
    _require(
        set(manifest)
        == {
            "artifact_id",
            "build_mode",
            "candidate_warning",
            "daily_registry_inventory",
            "execution_authorized",
            "formal_authority",
            "formal_authority_dependencies",
            "input_bindings",
            "model_execution_authorized",
            "model_or_score_output_published",
            "model_predictions_scores_effects_or_contrasts_read",
            "output_contract",
            "outputs",
            "protocol_sealed_or_amended",
            "runtime",
            "schema_version",
            "semantic_contract",
            "source_panel_inventory",
            "status",
            "training_example_counts",
        },
        "candidate manifest top-level schema changed",
    )
    _require(
        manifest["status"] == STATUS
        and manifest["formal_authority"] is False
        and manifest["execution_authorized"] is False
        and manifest["model_execution_authorized"] is False
        and manifest["protocol_sealed_or_amended"] is False,
        "candidate non-authorizing status changed",
    )
    daily = _parse_candidate_parquet(
        bundle.files[DAILY_FILENAME],
        schema=DAILY_SCHEMA,
        label="daily registry",
    )
    training = _parse_candidate_parquet(
        bundle.files[TRAINING_FILENAME],
        schema=TRAINING_SCHEMA,
        label="training registry",
    )
    validate_daily_registry(daily)
    validate_training_registry(
        training,
        daily=daily,
        config=config,
        verify_derived_key_ids=True,
        verify_daily_lineage=True,
    )
    output_frames = {DAILY_FILENAME: daily, TRAINING_FILENAME: training}
    output_schemas = {DAILY_FILENAME: DAILY_SCHEMA, TRAINING_FILENAME: TRAINING_SCHEMA}
    identity_columns = {
        DAILY_FILENAME: ("site_id", "DATE"),
        TRAINING_FILENAME: (
            "site_id",
            "split",
            "issue_date",
            "target_date",
            "lead_days",
        ),
    }
    _require(set(manifest["outputs"]) == set(output_frames), "manifest output set changed")
    for name, frame in output_frames.items():
        observed = manifest["outputs"][name]
        _require(type(observed) is dict, f"manifest output record is invalid: {name}")
        expected = _output_record(
            bundle.files[name],
            frame,
            schema=output_schemas[name],
            identity_columns=identity_columns[name],
        )
        _require(observed == expected, f"manifest output binding changed: {name}")
    _require(
        _canonical_frame_sha256(bundle.daily, DAILY_COLUMNS)
        == _canonical_frame_sha256(daily, DAILY_COLUMNS),
        "in-memory and serialized daily semantic content differ",
    )
    _require(
        _canonical_frame_sha256(bundle.training_examples, TRAINING_COLUMNS)
        == _canonical_frame_sha256(training, TRAINING_COLUMNS),
        "in-memory and serialized training semantic content differ",
    )


def build_twice(
    paths: RegistryInputPaths,
    *,
    config: BuildConfig,
    evidence_root: Path,
) -> CandidateBundle:
    first = build_candidate_artifacts(paths, config=config, evidence_root=evidence_root)
    second = build_candidate_artifacts(paths, config=config, evidence_root=evidence_root)
    _require(set(first.files) == set(second.files), "independent build file sets differ")
    for name in first.files:
        _require(first.files[name] == second.files[name], f"independent build bytes differ: {name}")
    _require(
        set(first.snapshot) == set(second.snapshot),
        "independent build input snapshots differ",
    )
    for role in first.snapshot:
        _require(
            first.snapshot[role].stat_signature == second.snapshot[role].stat_signature
            and first.snapshot[role].payload == second.snapshot[role].payload,
            f"input changed between independent builds: {role}",
        )
    return first


def _require_simple_name(name: str, *, label: str) -> None:
    _require(
        type(name) is str
        and name not in {"", ".", ".."}
        and Path(name).name == name
        and "/" not in name
        and "\\" not in name,
        f"{label} must be one basename",
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    return int(status.st_dev), int(status.st_ino)


def _verify_anchored_directory_path(
    path: Path,
    descriptor: int,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    opened = os.fstat(descriptor)
    _require(stat.S_ISDIR(opened.st_mode), f"opened {label} is not a directory")
    _require(
        _directory_identity(opened) == expected_identity,
        f"opened {label} inode changed",
    )
    try:
        lexical = path.lstat()
    except OSError as exc:
        raise SemanticRegistryError(f"lexical {label} disappeared: {path}") from exc
    _require(
        stat.S_ISDIR(lexical.st_mode)
        and not stat.S_ISLNK(lexical.st_mode)
        and _directory_identity(lexical) == expected_identity,
        f"lexical {label} no longer names the anchored directory",
    )
    _require_no_symlink_components(path, label=label)


def _open_anchored_directory(path: Path, *, label: str) -> tuple[int, tuple[int, int]]:
    _require_no_symlink_components(path, label=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SemanticRegistryError(f"cannot anchor {label}: {path}") from exc
    try:
        status = os.fstat(descriptor)
        _require(stat.S_ISDIR(status.st_mode), f"anchored {label} is not a directory")
        identity = _directory_identity(status)
        _verify_anchored_directory_path(path, descriptor, identity, label=label)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, identity


def _entry_status_at(parent_descriptor: int, name: str) -> os.stat_result | None:
    _require_simple_name(name, label="directory entry")
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _write_bytes_at(parent_descriptor: int, name: str, payload: bytes) -> None:
    _require_simple_name(name, label="staged filename")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            _require(written > 0, f"zero-byte write for staged {name}")
            offset += written
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        _require(
            stat.S_ISREG(status.st_mode)
            and status.st_nlink == 1
            and status.st_size == len(payload),
            f"staged {name} type/size/link count changed",
        )
    finally:
        os.close(descriptor)


def _read_stable_regular_at(parent_descriptor: int, name: str, *, label: str) -> bytes:
    _require_simple_name(name, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise SemanticRegistryError(f"cannot safely open {label}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(before.st_nlink == 1, f"{label} must have exactly one hard link")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    signature_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    _require(
        all(getattr(before, field) == getattr(after, field) for field in signature_fields)
        and len(payload) == before.st_size,
        f"{label} changed while being verified",
    )
    entry = _entry_status_at(parent_descriptor, name)
    _require(
        entry is not None
        and stat.S_ISREG(entry.st_mode)
        and entry.st_nlink == 1
        and _directory_identity(entry) == _directory_identity(before),
        f"{label} directory entry changed while being verified",
    )
    return payload


def _create_staging_directory_at(
    root_descriptor: int,
    *,
    destination_name: str,
) -> tuple[str, int, tuple[int, int]]:
    for _attempt in range(128):
        name = f".{destination_name}.candidate-stage.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=root_descriptor)
        except FileExistsError:
            continue
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=root_descriptor)
            status = os.fstat(descriptor)
            _require(stat.S_ISDIR(status.st_mode), "staging entry is not a directory")
        except Exception:
            try:
                os.rmdir(name, dir_fd=root_descriptor)
            except OSError:
                pass
            raise
        return name, descriptor, _directory_identity(status)
    raise SemanticRegistryError("cannot allocate an exclusive staging directory")


def _remove_tree_at(parent_descriptor: int, name: str) -> None:
    status = _entry_status_at(parent_descriptor, name)
    if status is None:
        return
    if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        child_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        try:
            for child in os.listdir(child_descriptor):
                _remove_tree_at(child_descriptor, child)
            os.fsync(child_descriptor)
        finally:
            os.close(child_descriptor)
        os.rmdir(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _remove_owned_entry_at(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
    strict: bool = False,
) -> bool:
    status = _entry_status_at(parent_descriptor, name)
    if status is None:
        return False
    if _directory_identity(status) != expected_identity:
        _require(not strict, f"refusing to clean replaced {label}")
        return False
    _remove_tree_at(parent_descriptor, name)
    return True


def _rename_directory_noreplace_at(
    old_directory_descriptor: int,
    source_name: str,
    new_directory_descriptor: int,
    destination_name: str,
) -> None:
    _require_simple_name(source_name, label="rename source")
    _require_simple_name(destination_name, label="rename destination")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "Linux atomic RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        old_directory_descriptor,
        os.fsencode(source_name),
        new_directory_descriptor,
        os.fsencode(destination_name),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise SemanticRegistryError(
                f"refusing to overwrite existing candidate destination: {destination_name}"
            )
        raise SemanticRegistryError(
            f"atomic candidate rename failed ({os.strerror(error)}): {destination_name}"
        )


def _revalidate_snapshot(
    snapshot: Mapping[str, BoundFile],
    *,
    evidence_root: Path,
) -> None:
    for role, original in snapshot.items():
        root = ROOT if role == "builder_code" else evidence_root
        current = _read_stable_regular(
            original.path,
            root=root,
            label=f"precommit {role.replace('_', ' ')}",
        )
        _require(
            current.stat_signature == original.stat_signature
            and current.sha256 == original.sha256
            and current.payload == original.payload,
            f"candidate input changed after capture: {role}",
        )


def _validate_candidate_destination(destination: Path) -> tuple[Path, Path]:
    absolute = _absolute(destination)
    _require_simple_name(absolute.name, label="candidate destination")
    parent = absolute.parent
    _require_no_symlink_components(parent, label="candidate output parent")
    final_root = _absolute(ROOT / "outputs" / "final")
    try:
        absolute.relative_to(final_root)
    except ValueError:
        pass
    else:
        raise SemanticRegistryError(
            "candidate output is prohibited under outputs/final; choose a scratch/candidate path"
        )
    _require(not os.path.lexists(absolute), f"candidate destination already exists: {absolute}")
    return absolute, parent


def write_candidate(
    paths: RegistryInputPaths,
    destination: Path,
    *,
    config: BuildConfig,
    evidence_root: Path,
) -> tuple[Path, CandidateBundle]:
    """Closed, create-only scratch writer; caller-created bundles are not accepted."""

    _require(type(paths) is RegistryInputPaths, "writer input-path type changed")
    _require(type(config) is BuildConfig, "writer config type changed")
    destination, parent = _validate_candidate_destination(destination)
    parent_descriptor, parent_identity = _open_anchored_directory(
        parent,
        label="candidate output parent",
    )
    destination_name = destination.name
    lock_name = f".{destination_name}.candidate-create.lock"
    lock_identity: tuple[int, int] | None = None
    staging_name: str | None = None
    staging_descriptor: int | None = None
    staging_identity: tuple[int, int] | None = None
    bundle: CandidateBundle | None = None
    committed = False
    lock_descriptor = -1
    try:
        _require(
            _entry_status_at(parent_descriptor, destination_name) is None,
            f"refusing to overwrite existing candidate destination: {destination}",
        )
        lock_flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_descriptor = os.open(
                lock_name,
                lock_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as exc:
            raise SemanticRegistryError(
                f"another candidate build holds the create lock: {lock_name}"
            ) from exc
        lock_status = os.fstat(lock_descriptor)
        lock_identity = _directory_identity(lock_status)
        lock_payload = f"pid={os.getpid()}\nstatus={STATUS}\n".encode("ascii")
        _require(
            os.write(lock_descriptor, lock_payload) == len(lock_payload),
            "candidate lock write was short",
        )
        os.fsync(lock_descriptor)
        os.close(lock_descriptor)
        lock_descriptor = -1
        os.fsync(parent_descriptor)

        bundle = build_twice(paths, config=config, evidence_root=evidence_root)
        staging_name, staging_descriptor, staging_identity = _create_staging_directory_at(
            parent_descriptor,
            destination_name=destination_name,
        )
        for name, payload in sorted(bundle.files.items()):
            _write_bytes_at(staging_descriptor, name, payload)
        os.fsync(staging_descriptor)
        _verify_bundle(bundle, config=config)
        _require(
            set(os.listdir(staging_descriptor)) == set(bundle.files),
            "staged candidate file set changed",
        )
        for name, payload in bundle.files.items():
            _require(
                _read_stable_regular_at(
                    staging_descriptor,
                    name,
                    label=f"staged {name}",
                )
                == payload,
                f"staged candidate bytes changed: {name}",
            )
        _revalidate_snapshot(bundle.snapshot, evidence_root=_absolute(evidence_root))
        _verify_anchored_directory_path(
            parent,
            parent_descriptor,
            parent_identity,
            label="candidate output parent before rename",
        )
        _require(
            _entry_status_at(parent_descriptor, destination_name) is None,
            "candidate destination appeared during the build",
        )
        _rename_directory_noreplace_at(
            parent_descriptor,
            staging_name,
            parent_descriptor,
            destination_name,
        )
        _verify_anchored_directory_path(
            parent,
            parent_descriptor,
            parent_identity,
            label="candidate output parent after rename",
        )
        destination_status = _entry_status_at(parent_descriptor, destination_name)
        _require(
            destination_status is not None
            and stat.S_ISDIR(destination_status.st_mode)
            and _directory_identity(destination_status) == staging_identity,
            "candidate destination is not the staged directory inode",
        )
        _require(
            set(os.listdir(staging_descriptor)) == set(bundle.files),
            "candidate file set changed after rename",
        )
        for name, payload in bundle.files.items():
            _require(
                _read_stable_regular_at(
                    staging_descriptor,
                    name,
                    label=f"committed candidate {name}",
                )
                == payload,
                f"candidate bytes changed after rename: {name}",
            )
        _remove_owned_entry_at(
            parent_descriptor,
            lock_name,
            lock_identity,
            label="candidate create lock",
            strict=True,
        )
        lock_identity = None
        os.fsync(parent_descriptor)
        _verify_anchored_directory_path(
            parent,
            parent_descriptor,
            parent_identity,
            label="candidate output parent at commit",
        )
        committed = True
    finally:
        try:
            if lock_descriptor >= 0:
                try:
                    os.close(lock_descriptor)
                except OSError:
                    pass
            if staging_descriptor is not None:
                try:
                    os.close(staging_descriptor)
                except OSError:
                    pass
                staging_descriptor = None
            if not committed and staging_identity is not None:
                if staging_name is not None:
                    _remove_owned_entry_at(
                        parent_descriptor,
                        staging_name,
                        staging_identity,
                        label="candidate staging directory",
                    )
                _remove_owned_entry_at(
                    parent_descriptor,
                    destination_name,
                    staging_identity,
                    label="failed candidate destination",
                )
            if lock_identity is not None:
                _remove_owned_entry_at(
                    parent_descriptor,
                    lock_name,
                    lock_identity,
                    label="candidate create lock",
                )
            if not committed or lock_identity is not None:
                os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    _require(bundle is not None, "candidate write completed without a bundle")
    return destination, bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new scratch/candidate directory; outputs/final is always rejected",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BuildConfig.production()
    destination, bundle = write_candidate(
        RegistryInputPaths.production(ROOT),
        args.output_dir,
        config=config,
        evidence_root=ROOT,
    )
    manifest = bundle.manifest
    result = {
        "status": STATUS,
        "execution_authorized": False,
        "output_dir": str(destination),
        "daily_rows": len(bundle.daily),
        "training_example_rows": len(bundle.training_examples),
        "training_example_counts": manifest["training_example_counts"][
            "from_materialized_registry"
        ],
        "file_sha256": {name: _sha256(payload) for name, payload in sorted(bundle.files.items())},
        "semantic_content_sha256": {
            name: record["ordered_semantic_content_sha256"]
            for name, record in sorted(manifest["outputs"].items())
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SemanticRegistryError as exc:
        print(f"semantic data-registry candidate build failed closed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
