#!/usr/bin/env python3
"""Build the scratch-only v4 semantic-contract registry candidate.

This program is intentionally not a protocol, seal, runner, or publication
tool.  It reads one separately produced semantic-data candidate together with
the pinned station, key, and preprocessing-defect authorities.  It writes six
canonical JSON registries and one manifest to a caller-named scratch directory.

Every emitted document is ``CANDIDATE_NOT_AUTHORITY`` and carries
``execution_authorized=false``.  The registered inventory is only the current
12-fit observed-lineage tree correction, the 5,280-fit neural Phase-1 plan,
the 40-fit L2_U2 Phase-1 plan, and 24 protocol-declared LightGBM spatial
logical cells that remain PLANNED without variants or fits.  It must never be
described as the complete v4 matrix or as executed work.  No model, result,
checkpoint, score, effect, or runner is imported or executed here.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import importlib.metadata
import json
import os
import platform
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FINAL_OUTPUT_ROOT = ROOT / "outputs" / "final"

STATUS = "CANDIDATE_NOT_AUTHORITY"
ARTIFACT_ID = "thermoroute-semantic-contract-registries-v4-candidate"

CELL_FILENAME = "cell_registry_v4.json"
FOLD_FILENAME = "fold_registry_v4.json"
MODEL_FILENAME = "model_registry_v4.json"
INPUT_FILENAME = "input_registry_v4.json"
CONTRAST_FILENAME = "contrast_registry_v4.json"
ENVIRONMENT_FILENAME = "environment_registry_v4.json"
MANIFEST_FILENAME = "semantic_contract_registry_manifest_v4_candidate.json"
REGISTRY_FILENAMES = (
    CELL_FILENAME,
    FOLD_FILENAME,
    MODEL_FILENAME,
    INPUT_FILENAME,
    CONTRAST_FILENAME,
    ENVIRONMENT_FILENAME,
)
ALL_FILENAMES = (*REGISTRY_FILENAMES, MANIFEST_FILENAME)

SEMANTIC_DAILY_FILENAME = "daily_raw_observed_panel_registry_v4.parquet"
SEMANTIC_TRAINING_FILENAME = "training_example_registry_2006_2017_v4.parquet"
SEMANTIC_MANIFEST_FILENAME = "semantic_data_registries_v4_candidate_manifest.json"

STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
KEY_AUTHORITY_DIR = ROOT / "outputs" / "final" / "information_regime_key_registries_v4"
KEY_AUTHORITY_MANIFEST = KEY_AUTHORITY_DIR / "information_regime_key_registries_v4_manifest.json"
PRIMARY_KEY_REGISTRY = KEY_AUTHORITY_DIR / "primary_reportable_key_registry_v4.parquet"
DEFECT_AUTHORITY_DIR = ROOT / "outputs" / "final" / "preprocessing_lineage_defect_authority_v1"
DEFECT_AUTHORITY_MANIFEST = (
    DEFECT_AUTHORITY_DIR / "preprocessing_lineage_defect_authority_v1_manifest.json"
)
DEFECT_AUTHORITY_REPORT = DEFECT_AUTHORITY_DIR / "preprocessing_lineage_defect_report_v1.json"
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS_LOCK = ROOT / "requirements-lock.txt"
REQUIREMENTS_LOCK_PY312 = ROOT / "requirements-lock-py312-hashed.txt"

PINNED_SHA256 = MappingProxyType(
    {
        "station_registry": "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9",
        "key_authority_manifest": "ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea",
        "primary_key_registry": "9a135dcffcfd467cf1e2dda4fc711ba6cf66c8ad54bf3b6f799f4a2d5a3c1d24",
        "defect_authority_manifest": "e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908",
        "defect_authority_report": "5c1b1e05932538dba1b2a0d21ad44fdfe54a9c52e949b96b3a668356910e68d9",
        "pyproject": "ef49ffacf3c73a2e566b89abac22d663bdd997a3f68adc5dcf0f5f187860fcc6",
        "requirements_lock": "ff2d67915ccaabb750cdf4c630d59500d6c8841b305d5fffb1ffd549195dc047",
        "requirements_lock_py312": "fa325e30e8e69b9e8ec458e9e76a466a765530a8bffa46f1d1f8e34614f4ab76",
    }
)

PINNED_SIZES = MappingProxyType(
    {
        "station_registry": 19_161,
        "key_authority_manifest": 18_678,
        "primary_key_registry": 13_343_948,
        "defect_authority_manifest": 7_408,
        "defect_authority_report": 14_868,
        "pyproject": 829,
        "requirements_lock": 918,
        "requirements_lock_py312": 69_128,
    }
)

EXPECTED_STATION_HUC2_SHA256 = "6f7891968a841e3eb5476074e08f24462b6a8984fdb1f12eb6b6b5c7d389af5e"
EXPECTED_REGION_FOLD_SHA256 = "c8d7ca871fb364ba1f3efd6a2613ad94a392d49ae3705e13652caa028af7347d"
EXPECTED_REGION_FOLD_RECORD_SHA256 = MappingProxyType(
    {
        0: "397dc053bcdea78d6efa445bd8037f7262516e0a43ddadf83baefb4a17ee6ed1",
        1: "8b8baac11ac42d172bbbf44041aa268d785e80cb13ebcbed872372ba376dd222",
        2: "e9503c90cc0e62cd9ef87d948271b1baa6031469069677ab5a9a6234ee4646b3",
        3: "39e29fa7ddf65f55c4190729d132d2b08c8fd5b4045e28ea9cebdc5c901bbb90",
    }
)
EXPECTED_NEURAL_PHASE1_PLAN_SHA256 = (
    "b8bdff977fcfa441d33fc38a6eb0af43eb737849c3341cfc2a13aea40c220b77"
)
EXPECTED_L2_U2_PHASE1_PLAN_SHA256 = (
    "965e9091d2a97ac1a8abe78c6abb802c8b2ba00b5fd2c7146f281b56f783c786"
)

KNOWN_SITE_TEMPORAL_PARTITION_ID = "partition:known_site_temporal:v5"
PROTOCOL_CELL_STATES = ("PLANNED", "REGISTERED", "EXECUTED", "WITHDRAWN")

ROUTE_A_RUNTIME = MappingProxyType(
    {
        "python_implementation": "CPython",
        "python_version": "3.12.13",
        "python_compiler": "GCC 14.3.0",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "byteorder": "little",
        "numpy_version": "1.26.4",
        "pandas_version": "2.2.2",
        "scipy_version": "1.13.1",
        "scikit_learn_version": "1.5.1",
        "lightgbm_version": "4.6.0",
        "torch_version": "2.12.0+cpu",
        "matplotlib_version": "3.9.2",
        "statsmodels_version": "0.14.2",
        "pyarrow_version": "24.0.0",
        "arrow_cpp_version": "24.0.0",
        "pytest_version": "8.0.0",
        "ruff_version": "0.15.14",
        "mypy_version": "1.11.2",
        "dataretrieval_version": "1.2.0",
        "pypandoc_binary_version": "1.17",
        "setuptools_version": "81.0.0",
    }
)

ROUTE_A_DIRECT_LOCK_PINS = MappingProxyType(
    {
        "numpy": "1.26.4",
        "pandas": "2.2.2",
        "scipy": "1.13.1",
        "scikit-learn": "1.5.1",
        "lightgbm": "4.6.0",
        "torch": "2.12.0",
        "matplotlib": "3.9.2",
        "statsmodels": "0.14.2",
        "pyarrow": "24.0.0",
        "pytest": "8.0.0",
        "ruff": "0.15.14",
        "mypy": "1.11.2",
        "dataretrieval": "1.2.0",
        "pypandoc_binary": "1.17",
        "setuptools": "81.0.0",
    }
)

HISTORICAL_AUTHORITY_RUNTIME = MappingProxyType(
    {
        "python_implementation": "CPython",
        "python_version": "3.11.7",
        "python_compiler": "GCC 11.2.0",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "byteorder": "little",
        "numpy_version": "1.26.4",
        "pandas_version": "2.1.4",
        "pyarrow_version": "14.0.2",
        "arrow_cpp_version": "14.0.2",
    }
)

LEADS = (1, 3, 7)
RANDOM_SPLIT_SEEDS = tuple(range(10))
FOLDS = tuple(range(4))
FIT_SEEDS = tuple(range(5))
HOLD_SIZES = (30, 30, 31, 29)
EXPECTED_REGION_SCOREABLE_COUNTS = (30, 30, 29, 27)
EXPECTED_TRAINING_STATIONS = 120
EXPECTED_REPORTABLE_STATIONS = 116
EXPECTED_REPORTABLE_STATIONS_SHA256 = (
    "ffb99d7c679c67156593d7588148a65b51306a76ee53ec392cd7a7750c720be8"
)
EXPECTED_LOGICAL_CELLS = 74
EXPECTED_PROTOCOL_PRIMARY_CELLS = 72
EXPECTED_VARIANTS = 62
EXPECTED_FITS = 5_332

PANEL_VARIABLES = ("WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP", "WDSP", "RHMEAN", "DH")
DAILY_COLUMNS = (
    "DATE",
    "site_id",
    *PANEL_VARIABLES,
    *(f"{variable}_observed" for variable in PANEL_VARIABLES),
)
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
PRIMARY_KEY_COLUMNS = (
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

DAILY_ARROW_SCHEMA = pa.schema(
    [
        pa.field("DATE", pa.timestamp("ns"), nullable=False),
        pa.field("site_id", pa.string(), nullable=False),
        *(pa.field(variable, pa.float64(), nullable=True) for variable in PANEL_VARIABLES),
        *(
            pa.field(f"{variable}_observed", pa.bool_(), nullable=False)
            for variable in PANEL_VARIABLES
        ),
    ]
)
TRAINING_ARROW_SCHEMA = pa.schema(
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
PRIMARY_KEY_ARROW_SCHEMA = pa.schema(
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


class SemanticContractError(RuntimeError):
    """A source, registry, or create-only boundary failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticContractError(message)


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


def _canonical_sha256(document: object) -> str:
    """Phase-1-compatible digest: canonical JSON without a terminal newline."""

    return hashlib.sha256(_canonical_json_bytes(document)[:-1]).hexdigest()


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
        raise SemanticContractError(f"{label} is not strict JSON") from exc
    _require(type(document) is dict, f"{label} root must be an object")
    if canonical:
        _require(_canonical_json_bytes(document) == payload, f"{label} is not canonical JSON")
    return document


def _keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == expected, f"{label} fields are not the exact closed schema")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be a JSON list")
    assert isinstance(value, list)
    return value


def _schema_signature(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _canonical_site(value: object) -> str:
    _require(
        type(value) is str
        and len(value) in (8, 15)
        and value.isascii()
        and all("0" <= character <= "9" for character in value),
        f"non-canonical station identifier: {value!r}",
    )
    assert isinstance(value, str)
    return value


def _canonical_huc2(value: object) -> str:
    _require(
        type(value) is str and len(value) == 2 and value.isascii() and value.isdigit(),
        f"non-canonical zero-padded HUC2: {value!r}",
    )
    assert isinstance(value, str)
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_to_root(path: Path) -> str:
    try:
        return _absolute(path).relative_to(_absolute(ROOT)).as_posix()
    except ValueError as exc:
        raise SemanticContractError(f"authority path escapes repository: {path}") from exc


def _require_no_symlink_components(path: Path, *, label: str, must_exist: bool = True) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current = current / part
        if not must_exist and index == len(parts) - 1 and not os.path.lexists(current):
            break
        try:
            status = current.lstat()
        except OSError as exc:
            raise SemanticContractError(f"{label} component is unavailable: {current}") from exc
        _require(not stat.S_ISLNK(status.st_mode), f"{label} crosses a symlink: {current}")
    return absolute


@dataclass(frozen=True, slots=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int
    stat_signature: tuple[int, ...]
    payload: bytes | None = None

    def binding(self, *, path: str) -> dict[str, object]:
        return {"path": path, "sha256": self.sha256, "size_bytes": self.size_bytes}


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _capture_regular(path: Path, *, label: str, retain_payload: bool) -> BoundFile:
    absolute = _require_no_symlink_components(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise SemanticContractError(f"cannot open {label}: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(before.st_nlink == 1, f"{label} must have exactly one hard link")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if retain_payload else None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    lexical = absolute.lstat()
    _require(
        _stat_signature(before) == _stat_signature(after)
        and _stat_signature(before) == _stat_signature(lexical),
        f"{label} changed while being captured",
    )
    payload = b"".join(chunks) if chunks is not None else None
    if payload is not None:
        _require(len(payload) == before.st_size, f"{label} size changed during capture")
    return BoundFile(
        path=absolute,
        sha256=digest.hexdigest(),
        size_bytes=int(before.st_size),
        stat_signature=_stat_signature(before),
        payload=payload,
    )


def _require_pinned(bound: BoundFile, role: str) -> None:
    _require(bound.sha256 == PINNED_SHA256[role], f"pinned {role} SHA-256 changed")
    _require(bound.size_bytes == PINNED_SIZES[role], f"pinned {role} size changed")


def _binding_fields(record: object, *, label: str) -> dict[str, Any]:
    result = _keys(record, {"path", "sha256", "size_bytes"}, label=label)
    _require(type(result["path"]) is str and result["path"] != "", f"{label} path is invalid")
    digest = result["sha256"]
    _require(
        type(digest) is str
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} SHA-256 is invalid",
    )
    _require(
        type(result["size_bytes"]) is int and result["size_bytes"] >= 0, f"{label} size is invalid"
    )
    return result


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SemanticContractError(
            f"route-a runtime distribution is unavailable: {distribution}"
        ) from exc


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "pandas_version": __import__("pandas").__version__,
        "scipy_version": _distribution_version("scipy"),
        "scikit_learn_version": _distribution_version("scikit-learn"),
        "lightgbm_version": _distribution_version("lightgbm"),
        "torch_version": _distribution_version("torch"),
        "matplotlib_version": _distribution_version("matplotlib"),
        "statsmodels_version": _distribution_version("statsmodels"),
        "pyarrow_version": pa.__version__,
        "arrow_cpp_version": pa.cpp_version,
        "pytest_version": _distribution_version("pytest"),
        "ruff_version": _distribution_version("ruff"),
        "mypy_version": _distribution_version("mypy"),
        "dataretrieval_version": _distribution_version("dataretrieval"),
        "pypandoc_binary_version": _distribution_version("pypandoc-binary"),
        "setuptools_version": _distribution_version("setuptools"),
    }


@dataclass(frozen=True, slots=True)
class RegistryInputPaths:
    semantic_data_directory: Path
    station_registry: Path = STATION_REGISTRY
    key_authority_manifest: Path = KEY_AUTHORITY_MANIFEST
    primary_key_registry: Path = PRIMARY_KEY_REGISTRY
    defect_authority_manifest: Path = DEFECT_AUTHORITY_MANIFEST
    defect_authority_report: Path = DEFECT_AUTHORITY_REPORT
    pyproject: Path = PYPROJECT
    requirements_lock: Path = REQUIREMENTS_LOCK
    requirements_lock_py312: Path = REQUIREMENTS_LOCK_PY312

    @classmethod
    def production(cls, semantic_data_directory: Path) -> RegistryInputPaths:
        return cls(semantic_data_directory=_absolute(semantic_data_directory))


@dataclass(frozen=True, slots=True)
class BuildConfig:
    enforce_route_a_runtime: bool = True


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    bindings: Mapping[str, Mapping[str, object]]
    stations: tuple[str, ...]
    huc2_by_station: Mapping[str, str]
    reportable_stations: tuple[str, ...]
    semantic_outputs: Mapping[str, Mapping[str, object]]
    signatures: Mapping[str, tuple[int, ...]]


def _validate_parquet_metadata(path: Path, *, schema: pa.Schema, rows: int, label: str) -> None:
    try:
        parquet = pq.ParquetFile(path)
        observed_schema = parquet.schema_arrow
        observed_rows = parquet.metadata.num_rows
    except Exception as exc:
        raise SemanticContractError(f"{label} is not readable Parquet") from exc
    _require(observed_schema.equals(schema), f"{label} Arrow schema/order changed")
    _require(observed_rows == rows, f"{label} row count changed")


def _parse_station_registry(bound: BoundFile) -> tuple[tuple[str, ...], dict[str, str]]:
    _require(bound.payload is not None, "station registry payload was not retained")
    try:
        reader = csv.DictReader(io.StringIO(bound.payload.decode("utf-8")), strict=True)
        _require(
            tuple(reader.fieldnames or ()) == STATION_REGISTRY_COLUMNS,
            "station registry schema/order changed",
        )
        pairs = []
        for row in reader:
            site = _canonical_site(row["site_no"])
            raw_huc = row["huc2"]
            _require(
                type(raw_huc) is str and raw_huc.isascii() and raw_huc.isdigit(),
                "station HUC2 is invalid",
            )
            huc = raw_huc.zfill(2)
            _canonical_huc2(huc)
            pairs.append((site, huc))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SemanticContractError("station registry is not strict UTF-8 CSV") from exc
    pairs.sort()
    _require(
        len(pairs) == 120 and len({site for site, _ in pairs}) == 120,
        "training station universe is not exactly 120 unique stations",
    )
    stations = tuple(site for site, _huc in pairs)
    _require("255534081324000" in stations, "15-digit station 255534081324000 was not preserved")
    huc = dict(pairs)
    _require(
        _canonical_sha256([[site, huc[site]] for site in stations]) == EXPECTED_STATION_HUC2_SHA256,
        "station/HUC2 digest changed",
    )
    return stations, huc


def _validate_key_authority(
    manifest_bound: BoundFile,
    primary_bound: BoundFile,
    station_bound: BoundFile,
    stations: tuple[str, ...],
) -> tuple[str, ...]:
    _require(manifest_bound.payload is not None, "key manifest payload was not retained")
    manifest = _strict_json(manifest_bound.payload, label="key authority manifest")
    _require(
        manifest.get("artifact_id") == "thermoroute-information-regime-key-registries-v4-phase1"
        and manifest.get("status") == "PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY"
        and manifest.get("registry_publication_scope") == "PHASE1_CONTRACT_ONLY",
        "key authority identity/status changed",
    )
    _require(
        manifest.get("execution_authorized_by_protocol") is False
        and manifest.get("model_or_score_output_published") is False
        and manifest.get("model_scores_read_or_accepted") is False,
        "key authority non-execution boundary changed",
    )
    station_binding = manifest.get("input_bindings", {}).get("station_registry_v1")
    _require(
        station_binding == station_bound.binding(path="data_usgs/station_registry_v1.csv"),
        "key authority station binding changed",
    )
    primary_output = manifest.get("outputs", {}).get(PRIMARY_KEY_REGISTRY.name)
    _require(type(primary_output) is dict, "key authority primary output binding is absent")
    _require(
        primary_output.get("sha256") == primary_bound.sha256
        and primary_output.get("size_bytes") == primary_bound.size_bytes
        and primary_output.get("row_count") == 358_765
        and primary_output.get("columns") == list(PRIMARY_KEY_COLUMNS),
        "key authority primary output binding changed",
    )
    primary = manifest.get("primary_registry")
    _require(type(primary) is dict, "key authority primary registry declaration is absent")
    reportable = tuple(primary.get("reportable_site_ids", ()))
    _require(
        len(reportable) == EXPECTED_REPORTABLE_STATIONS
        and reportable == tuple(sorted(reportable))
        and len(set(reportable)) == len(reportable),
        "reportable scoring universe is not exactly 116 sorted unique stations",
    )
    for site in reportable:
        _canonical_site(site)
    _require(
        set(reportable) < set(stations),
        "reportable scoring universe must be a strict subset of the 120-site training universe",
    )
    _require(
        set(stations) - set(reportable) == {"01435000", "01608500", "03058000", "03544970"},
        "120-vs-116 cohort boundary changed",
    )
    _validate_parquet_metadata(
        primary_bound.path,
        schema=PRIMARY_KEY_ARROW_SCHEMA,
        rows=358_765,
        label="primary reportable key registry",
    )
    return reportable


def _validate_defect_authority(
    manifest_bound: BoundFile,
    report_bound: BoundFile,
    key_bound: BoundFile,
    station_bound: BoundFile,
) -> None:
    _require(
        manifest_bound.payload is not None and report_bound.payload is not None,
        "defect authority JSON payload missing",
    )
    manifest = _strict_json(manifest_bound.payload, label="defect authority manifest")
    report = _strict_json(report_bound.payload, label="defect authority report")
    expected_status = "SCIENTIFIC_RESULTS_WITHDRAWN_PENDING_OBSERVED_LINEAGE_RERUN"
    _require(
        manifest.get("artifact_id") == "thermoroute-preprocessing-lineage-defect-authority-v1"
        and report.get("artifact_id") == "thermoroute-preprocessing-lineage-defect-authority-v1"
        and manifest.get("status") == expected_status
        and report.get("status") == expected_status,
        "defect authority identity/status changed",
    )
    _require(
        manifest.get("output")
        == {
            "name": DEFECT_AUTHORITY_REPORT.name,
            "sha256": report_bound.sha256,
            "size_bytes": report_bound.size_bytes,
        },
        "defect authority report binding changed",
    )
    inputs = manifest.get("input_bindings", {})
    _require(
        inputs.get("score_independent_key_manifest", {}).get("sha256") == key_bound.sha256
        and inputs.get("station_registry", {}).get("sha256") == station_bound.sha256,
        "defect authority key/station linkage changed",
    )
    rerun = report.get("required_versioned_rerun")
    _require(type(rerun) is dict, "defect rerun requirements are absent")
    for field in (
        "raw_observed_flags_must_be_created_before_imputation",
        "observed_masks_must_survive_imputation_bit_exactly",
        "imputed_feature_panel_and_raw_label_panel_must_be_distinct_typed_inputs",
        "training_issue_and_target_admissibility_must_be_joined_from_raw_lineage",
        "training_key_registries_or_digests_must_be_authority_bound",
        "training_must_use_only_2006_2015_rows",
        "validation_must_use_only_independent_2016_2017_rows",
    ):
        _require(rerun.get(field) is True, f"defect rerun requirement changed: {field}")


def _validate_semantic_data_candidate(
    directory: Path,
    manifest_bound: BoundFile,
    daily_bound: BoundFile,
    training_bound: BoundFile,
    key_bound: BoundFile,
    defect_bound: BoundFile,
    station_bound: BoundFile,
) -> dict[str, Mapping[str, object]]:
    _require(manifest_bound.payload is not None, "semantic-data manifest payload missing")
    manifest = _strict_json(manifest_bound.payload, label="semantic-data candidate manifest")
    _require(
        manifest.get("artifact_id") == "thermoroute-semantic-data-registries-v4-candidate"
        and manifest.get("status") == STATUS
        and manifest.get("build_mode") == "PRODUCTION_INPUTS_CANDIDATE_OUTPUT",
        "semantic-data candidate identity/status changed",
    )
    for field in (
        "formal_authority",
        "execution_authorized",
        "model_execution_authorized",
        "protocol_sealed_or_amended",
        "model_or_score_output_published",
        "model_predictions_scores_effects_or_contrasts_read",
    ):
        _require(manifest.get(field) is False, f"semantic-data candidate {field} must be false")
    dependencies = manifest.get("formal_authority_dependencies", {})
    _require(
        dependencies.get("information_regime_key_registries_v4_manifest_sha256") == key_bound.sha256
        and dependencies.get("preprocessing_lineage_defect_authority_v1_manifest_sha256")
        == defect_bound.sha256
        and dependencies.get("dependency_authority_does_not_transfer_to_candidate") is True,
        "semantic-data candidate authority dependency boundary changed",
    )
    candidate_inputs = manifest.get("input_bindings", {})
    _require(
        candidate_inputs.get("station_registry", {}).get("sha256") == station_bound.sha256
        and candidate_inputs.get("key_authority_manifest", {}).get("sha256") == key_bound.sha256
        and candidate_inputs.get("defect_authority_manifest", {}).get("sha256")
        == defect_bound.sha256,
        "semantic-data candidate source-authority bindings changed",
    )
    outputs = manifest.get("outputs")
    _require(
        type(outputs) is dict
        and set(outputs) == {SEMANTIC_DAILY_FILENAME, SEMANTIC_TRAINING_FILENAME},
        "semantic-data candidate output set changed",
    )
    expected = {
        SEMANTIC_DAILY_FILENAME: (daily_bound, DAILY_ARROW_SCHEMA, 788_880, DAILY_COLUMNS),
        SEMANTIC_TRAINING_FILENAME: (
            training_bound,
            TRAINING_ARROW_SCHEMA,
            1_261_994,
            TRAINING_COLUMNS,
        ),
    }
    normalized: dict[str, Mapping[str, object]] = {}
    for name, (bound, schema, rows, columns) in expected.items():
        record = outputs[name]
        _require(type(record) is dict, f"semantic-data output record is invalid: {name}")
        _require(
            record.get("sha256") == bound.sha256
            and record.get("size_bytes") == bound.size_bytes
            and record.get("row_count") == rows
            and record.get("columns") == list(columns)
            and record.get("arrow_schema") == _schema_signature(schema),
            f"semantic-data output byte/schema/count binding changed: {name}",
        )
        _validate_parquet_metadata(bound.path, schema=schema, rows=rows, label=name)
        normalized[name] = {
            "path": name,
            "sha256": bound.sha256,
            "size_bytes": bound.size_bytes,
            "row_count": rows,
            "columns": list(columns),
            "arrow_schema": _schema_signature(schema),
            "ordered_semantic_content_sha256": record.get("ordered_semantic_content_sha256"),
            "ordered_identity_sha256": record.get("ordered_identity_sha256"),
        }
    contract = manifest.get("semantic_contract", {})
    training_contract = contract.get("training_example_registry_2006_2017", {})
    _require(
        training_contract.get("train_period") == ["2006-01-01", "2015-12-31"]
        and training_contract.get("validation_period") == ["2016-01-01", "2017-12-31"]
        and training_contract.get("lead_days") == [1, 3, 7]
        and training_contract.get("imputed_labels_allowed") is False,
        "semantic-data training periods/label boundary changed",
    )
    counts = manifest.get("training_example_counts", {}).get("from_materialized_registry")
    _require(
        counts
        == {
            "train": {"1": 339_750, "3": 337_725, "7": 335_822},
            "validation": {"1": 83_516, "3": 82_948, "7": 82_233},
        },
        "semantic-data training-example counts changed",
    )
    daily_inventory = manifest.get("daily_registry_inventory", {})
    _require(
        daily_inventory.get("station_count") == 120
        and daily_inventory.get("row_count") == 788_880
        and daily_inventory.get("start") == "2006-01-01"
        and daily_inventory.get("end") == "2023-12-31",
        "semantic-data daily inventory changed",
    )
    return normalized


def _capture_sources(paths: RegistryInputPaths, config: BuildConfig) -> SourceSnapshot:
    _require(type(paths) is RegistryInputPaths, "input paths object type changed")
    _require(type(config) is BuildConfig, "build config object type changed")
    _require(
        type(config.enforce_route_a_runtime) is bool,
        "route-a runtime enforcement flag must be boolean",
    )
    expected_paths = RegistryInputPaths.production(paths.semantic_data_directory)
    _require(paths == expected_paths, "pinned authority paths changed")
    semantic_directory = _require_no_symlink_components(
        paths.semantic_data_directory, label="semantic-data candidate directory"
    )
    _require(semantic_directory.is_dir(), "semantic-data candidate path is not a directory")
    try:
        semantic_directory.relative_to(_absolute(FINAL_OUTPUT_ROOT))
    except ValueError:
        pass
    else:
        raise SemanticContractError("semantic-data candidate must be outside outputs/final")
    if config.enforce_route_a_runtime:
        _require(
            _runtime_identity() == dict(ROUTE_A_RUNTIME),
            "candidate build must use the exact route-a runtime",
        )

    role_paths = {
        "station_registry": paths.station_registry,
        "key_authority_manifest": paths.key_authority_manifest,
        "primary_key_registry": paths.primary_key_registry,
        "defect_authority_manifest": paths.defect_authority_manifest,
        "defect_authority_report": paths.defect_authority_report,
        "pyproject": paths.pyproject,
        "requirements_lock": paths.requirements_lock,
        "requirements_lock_py312": paths.requirements_lock_py312,
        "semantic_data_manifest": semantic_directory / SEMANTIC_MANIFEST_FILENAME,
        "semantic_daily_registry": semantic_directory / SEMANTIC_DAILY_FILENAME,
        "semantic_training_registry": semantic_directory / SEMANTIC_TRAINING_FILENAME,
    }
    retain = {
        "station_registry",
        "key_authority_manifest",
        "defect_authority_manifest",
        "defect_authority_report",
        "semantic_data_manifest",
    }
    captured = {
        role: _capture_regular(path, label=role.replace("_", " "), retain_payload=role in retain)
        for role, path in role_paths.items()
    }
    for role in PINNED_SHA256:
        _require_pinned(captured[role], role)
    stations, huc = _parse_station_registry(captured["station_registry"])
    reportable = _validate_key_authority(
        captured["key_authority_manifest"],
        captured["primary_key_registry"],
        captured["station_registry"],
        stations,
    )
    _validate_defect_authority(
        captured["defect_authority_manifest"],
        captured["defect_authority_report"],
        captured["key_authority_manifest"],
        captured["station_registry"],
    )
    semantic_outputs = _validate_semantic_data_candidate(
        semantic_directory,
        captured["semantic_data_manifest"],
        captured["semantic_daily_registry"],
        captured["semantic_training_registry"],
        captured["key_authority_manifest"],
        captured["defect_authority_manifest"],
        captured["station_registry"],
    )
    # Re-capture metadata after Arrow inspection so replacement during inspection fails closed.
    for role in ("primary_key_registry", "semantic_daily_registry", "semantic_training_registry"):
        checked = _capture_regular(
            role_paths[role], label=f"post-inspection {role}", retain_payload=False
        )
        _require(
            checked.stat_signature == captured[role].stat_signature
            and checked.sha256 == captured[role].sha256,
            f"{role} changed during schema inspection",
        )

    binding_paths = {
        role: _relative_to_root(bound.path)
        for role, bound in captured.items()
        if role
        not in {"semantic_data_manifest", "semantic_daily_registry", "semantic_training_registry"}
    }
    binding_paths.update(
        {
            "semantic_data_manifest": SEMANTIC_MANIFEST_FILENAME,
            "semantic_daily_registry": SEMANTIC_DAILY_FILENAME,
            "semantic_training_registry": SEMANTIC_TRAINING_FILENAME,
        }
    )
    bindings = {role: captured[role].binding(path=binding_paths[role]) for role in sorted(captured)}
    signatures = {role: bound.stat_signature for role, bound in captured.items()}
    return SourceSnapshot(
        bindings=MappingProxyType(bindings),
        stations=stations,
        huc2_by_station=MappingProxyType(huc),
        reportable_stations=reportable,
        semantic_outputs=MappingProxyType(semantic_outputs),
        signatures=MappingProxyType(signatures),
    )


def _candidate_trust_boundary() -> dict[str, object]:
    return {
        "this_document_trust_class": "SCRATCH_CANDIDATE_NOT_AUTHORITY",
        "semantic_data_candidate_accepted_for": [
            "exact byte binding",
            "closed-schema validation",
            "structural and referential plumbing",
        ],
        "semantic_data_candidate_accepted_as_scientific_evidence": False,
        "authority_dependency_roles_are_narrow_and_do_not_transfer": True,
        "protocol_or_seal_authority_inherited": False,
        "execution_authority_inherited": False,
        "model_score_checkpoint_effect_or_result_artifacts_read_or_accepted": False,
    }


def _registry_header(registry_id: str, schema_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "schema_id": schema_id,
        "registry_id": registry_id,
        "status": STATUS,
        "formal_authority": False,
        "execution_authorized": False,
        "candidate_source_inputs_can_authorize_execution": False,
        "candidate_trust_boundary": _candidate_trust_boundary(),
    }


def _record_guard() -> dict[str, object]:
    return {"status": STATUS, "execution_authorized": False}


def _model_id(family: str, architecture: str, constraint: str, lead: int | None = None) -> str:
    suffix = f":h{lead}" if lead is not None else ""
    return f"model:{family}:{architecture}:{constraint}{suffix}"


def _input_id(family: str, forcing: str, local_level: str, lead: int | None = None) -> str:
    suffix = f":h{lead}" if lead is not None else ""
    return f"input:{family}:{forcing}:{local_level}{suffix}"


def _neural_logical_id(
    forcing: str,
    local_level: str,
    geometry: str,
    architecture: str,
    lead: int,
) -> str:
    return f"cell:neural:{forcing}:{local_level}:{geometry}:{architecture}:h{lead}"


def _neural_variant_id(logical_id: str, constraint: str) -> str:
    return f"variant:{logical_id.removeprefix('cell:')}:{constraint}"


def _protocol_primary_cell_id(
    forcing: str,
    local_level: str,
    geometry: str,
    architecture: str,
    lead: int,
) -> str:
    return f"protocol-cell:primary:{forcing}:{local_level}:{geometry}:{architecture}:h{lead}"


def _protocol_primary_cell_status_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for forcing in ("F0", "F2b", "F3_full"):
        for local_level in ("L0", "L2"):
            for geometry in ("random_site", "whole_region"):
                for architecture in ("LightGBM", "plain_TCN"):
                    for lead in LEADS:
                        registered = forcing in ("F0", "F3_full") and architecture == "plain_TCN"
                        current_logical = forcing in ("F0", "F3_full")
                        logical_id = (
                            _neural_logical_id(
                                forcing,
                                local_level,
                                "random" if geometry == "random_site" else "region",
                                architecture,
                                lead,
                            )
                            if current_logical
                            else None
                        )
                        registered_fit_count = (
                            len(RANDOM_SPLIT_SEEDS) * len(FOLDS) * len(FIT_SEEDS)
                            if registered and geometry == "random_site"
                            else len(FOLDS) * len(FIT_SEEDS)
                            if registered
                            else 0
                        )
                        records.append(
                            {
                                "protocol_cell_id": _protocol_primary_cell_id(
                                    forcing, local_level, geometry, architecture, lead
                                ),
                                "logical_cell_id": logical_id,
                                "forcing": forcing,
                                "local_level": local_level,
                                "geometry": geometry,
                                "architecture": architecture,
                                "lead_days": lead,
                                "protocol_state": "REGISTERED" if registered else "PLANNED",
                                "current_logical_registry_membership": current_logical,
                                "registered_fit_count": registered_fit_count,
                                "execution_receipt_bound": False,
                                "score_or_result_receipt_bound": False,
                            }
                        )
    records.sort(key=lambda record: str(record["protocol_cell_id"]))
    return records


def _protocol_declared_cell_status_map() -> dict[str, object]:
    primary = _protocol_primary_cell_status_records()
    counts = {
        state: sum(record["protocol_state"] == state for record in primary)
        for state in PROTOCOL_CELL_STATES
    }
    return {
        "state_vocabulary": list(PROTOCOL_CELL_STATES),
        "state_semantics": {
            "PLANNED": "declared by the draft protocol but no complete registered fit plan exists",
            "REGISTERED": "logical cell and fit-plan rows are registered; this does not mean run",
            "EXECUTED": "requires an independently authorized execution and completion receipt",
            "WITHDRAWN": "historical scope is retained only for forensic exclusion",
        },
        "candidate_may_assert_executed": False,
        "execution_completion_or_score_receipts_read": False,
        "primary_matrix": {
            "protocol_logical_cell_count": EXPECTED_PROTOCOL_PRIMARY_CELLS,
            "complete_primary_matrix_claimed": False,
            "state_counts": counts,
            "cells": primary,
        },
        "withdrawn_forensic_scope": {
            "scope_id": "historical-corrected-information-ladder-432",
            "protocol_state": "WITHDRAWN",
            "logical_cell_count": 432,
            "included_in_primary_72": False,
            "included_in_current_logical_registry": False,
            "eligible_as_model_or_score_evidence": False,
            "reason": "withdrawn pending an observed-lineage rerun under a future authority",
        },
    }


def _neural_phase1_records() -> list[dict[str, object]]:
    selected_thermoroute = {
        ("F0", "L0", "random"),
        ("F3_full", "L0", "random"),
        ("F0", "L2", "region"),
        ("F3_full", "L2", "region"),
    }
    records: list[dict[str, object]] = []
    for architecture in ("plain_TCN", "ThermoRoute"):
        for forcing in ("F0", "F3_full"):
            for local_level in ("L0", "L2"):
                for geometry in ("random", "region"):
                    if (
                        architecture == "ThermoRoute"
                        and (
                            forcing,
                            local_level,
                            geometry,
                        )
                        not in selected_thermoroute
                    ):
                        continue
                    constraints = (
                        ("unbounded",) if architecture == "plain_TCN" else ("unbounded", "bounded")
                    )
                    for lead in LEADS:
                        for constraint in constraints:
                            split_seeds = RANDOM_SPLIT_SEEDS if geometry == "random" else (0,)
                            for split_seed in split_seeds:
                                for fold in FOLDS:
                                    for fit_seed in FIT_SEEDS:
                                        variant_configuration_id = "__".join(
                                            (
                                                forcing,
                                                local_level,
                                                geometry,
                                                architecture,
                                                f"h{lead}",
                                                constraint,
                                            )
                                        )
                                        fit_id = "__".join(
                                            (
                                                variant_configuration_id,
                                                f"split{split_seed:02d}",
                                                f"fold{fold}",
                                                f"fit{fit_seed}",
                                            )
                                        )
                                        records.append(
                                            {
                                                "forcing": forcing,
                                                "local_level": local_level,
                                                "geometry": geometry,
                                                "protocol_geometry": (
                                                    "random_site"
                                                    if geometry == "random"
                                                    else "whole_region"
                                                ),
                                                "architecture": architecture,
                                                "lead": lead,
                                                "split_seed": split_seed,
                                                "fold": fold,
                                                "fit_seed": fit_seed,
                                                "constraint": constraint,
                                                "variant_configuration_id": (
                                                    variant_configuration_id
                                                ),
                                                "fit_id": fit_id,
                                            }
                                        )
    # This is the dataclass field order used by the independent Phase-1 plan,
    # expressed locally without importing that runner.
    records.sort(
        key=lambda record: (
            record["forcing"],
            record["local_level"],
            record["geometry"],
            record["architecture"],
            record["lead"],
            record["split_seed"],
            record["fold"],
            record["fit_seed"],
            record["constraint"],
        )
    )
    return records


def _l2_u2_phase1_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for architecture in ("LightGBM", "plain_TCN"):
        logical_cell_id = f"F0__L2_U2__whole_region__h7__{architecture}"
        for fold in FOLDS:
            fold_unit_id = f"{logical_cell_id}__fold{fold}"
            for fit_seed in FIT_SEEDS:
                records.append(
                    {
                        "forcing": "F0",
                        "local_level": "L2_U2",
                        "geometry": "whole_region",
                        "lead_days": 7,
                        "architecture": architecture,
                        "fold": fold,
                        "fit_seed": fit_seed,
                        "logical_cell_id": logical_cell_id,
                        "fold_unit_id": fold_unit_id,
                        "fit_id": f"{fold_unit_id}__fit{fit_seed}",
                    }
                )
    records.sort(
        key=lambda record: (
            record["architecture"],
            record["fold"],
            record["fit_seed"],
        )
    )
    return records


def _build_cell_registry() -> dict[str, object]:
    logical_cells: list[dict[str, object]] = []
    variants: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []

    for arm in ("F0", "F3_full"):
        for architecture in ("LightGBM", "ResidualLightGBM"):
            for lead in LEADS:
                logical_id = f"cell:v5:{arm}:L0:known_site_temporal:{architecture}:h{lead}"
                variant_id = f"variant:{logical_id.removeprefix('cell:')}:native"
                logical_cells.append(
                    {
                        "logical_cell_id": logical_id,
                        "family": "v5_observed_lineage_tree_correction",
                        "forcing": arm,
                        "local_level": "L0",
                        "geometry": "known_site_temporal",
                        "architecture": architecture,
                        "lead_days": lead,
                        "post_outcome_status": "POST_OUTCOME_OBSERVED_LINEAGE_CORRECTION",
                        "protocol_state": "REGISTERED",
                        **_record_guard(),
                    }
                )
                variants.append(
                    {
                        "variant_id": variant_id,
                        "logical_cell_id": logical_id,
                        "family": "v5_observed_lineage_tree_correction",
                        "constraint": "native_tree_configuration",
                        **_record_guard(),
                    }
                )
                fits.append(
                    {
                        "fit_id": f"fit:v5:{arm}:L0:known_site_temporal:{architecture}:h{lead}:seed0",
                        "family": "v5_observed_lineage_tree_correction",
                        "logical_cell_id": logical_id,
                        "variant_id": variant_id,
                        "forcing": arm,
                        "local_level": "L0",
                        "geometry": "known_site_temporal",
                        "architecture": architecture,
                        "lead_days": lead,
                        "constraint": "native_tree_configuration",
                        "split_seed": None,
                        "fold_index": None,
                        "fit_seed": 0,
                        "spatial_fold_id": None,
                        "temporal_partition_id": KNOWN_SITE_TEMPORAL_PARTITION_ID,
                        "model_id": _model_id(
                            "v5", architecture, "native_tree_configuration", lead
                        ),
                        "input_id": _input_id("v5", arm, "L0", lead),
                        "environment_id": "environment:route_a_candidate",
                        "phase1_fit_id": None,
                        **_record_guard(),
                    }
                )

    for forcing in ("F0", "F3_full"):
        for local_level in ("L0", "L2"):
            for geometry in ("random", "region"):
                for lead in LEADS:
                    logical_id = _neural_logical_id(
                        forcing, local_level, geometry, "LightGBM", lead
                    )
                    logical_cells.append(
                        {
                            "logical_cell_id": logical_id,
                            "family": "primary_lightgbm_f0_f3_planned",
                            "forcing": forcing,
                            "local_level": local_level,
                            "geometry": (
                                "random_site" if geometry == "random" else "whole_region"
                            ),
                            "architecture": "LightGBM",
                            "lead_days": lead,
                            "post_outcome_status": "PROTOCOL_DECLARED_WITHOUT_REGISTERED_FIT_PLAN",
                            "protocol_state": "PLANNED",
                            **_record_guard(),
                        }
                    )

    neural_phase1 = _neural_phase1_records()
    neural_logical_seen: set[str] = set()
    neural_variant_seen: set[str] = set()
    for phase1 in neural_phase1:
        forcing = str(phase1["forcing"])
        local_level = str(phase1["local_level"])
        geometry = str(phase1["geometry"])
        architecture = str(phase1["architecture"])
        constraint = str(phase1["constraint"])
        lead = int(phase1["lead"])
        logical_id = _neural_logical_id(forcing, local_level, geometry, architecture, lead)
        variant_id = _neural_variant_id(logical_id, constraint)
        if logical_id not in neural_logical_seen:
            logical_cells.append(
                {
                    "logical_cell_id": logical_id,
                    "family": "neural_phase1_f0_f3_registered_subset",
                    "forcing": forcing,
                    "local_level": local_level,
                    "geometry": ("random_site" if geometry == "random" else "whole_region"),
                    "architecture": architecture,
                    "lead_days": lead,
                    "post_outcome_status": "PRE_OUTCOME_UNRUN_PHASE1_PLAN",
                    "protocol_state": "REGISTERED",
                    **_record_guard(),
                }
            )
            neural_logical_seen.add(logical_id)
        if variant_id not in neural_variant_seen:
            variants.append(
                {
                    "variant_id": variant_id,
                    "logical_cell_id": logical_id,
                    "family": "neural_phase1_f0_f3_registered_subset",
                    "constraint": constraint,
                    **_record_guard(),
                }
            )
            neural_variant_seen.add(variant_id)
        phase1_split_seed = int(phase1["split_seed"])
        split_seed = phase1_split_seed if geometry == "random" else None
        fold = int(phase1["fold"])
        registered_fit_id = (
            str(phase1["fit_id"])
            if geometry == "random"
            else "__".join(
                (
                    str(phase1["variant_configuration_id"]),
                    f"fold{fold}",
                    f"fit{int(phase1['fit_seed'])}",
                )
            )
        )
        fits.append(
            {
                "fit_id": f"fit:neural:{registered_fit_id}",
                "family": "neural_phase1_f0_f3_registered_subset",
                "logical_cell_id": logical_id,
                "variant_id": variant_id,
                "forcing": forcing,
                "local_level": local_level,
                "geometry": ("random_site" if geometry == "random" else "whole_region"),
                "architecture": architecture,
                "lead_days": lead,
                "constraint": constraint,
                "split_seed": split_seed,
                "fold_index": fold,
                "fit_seed": int(phase1["fit_seed"]),
                "spatial_fold_id": (
                    f"fold:random:split{split_seed:02d}:fold{fold}"
                    if geometry == "random"
                    else f"fold:whole_region:fold{fold}"
                ),
                "temporal_partition_id": None,
                "model_id": _model_id("neural", architecture, constraint),
                "input_id": _input_id("neural", forcing, local_level),
                "environment_id": "environment:route_a_candidate",
                "phase1_fit_id": str(phase1["fit_id"]),
                **_record_guard(),
            }
        )

    l2_phase1 = _l2_u2_phase1_records()
    for architecture in ("LightGBM", "plain_TCN"):
        logical_id = f"cell:l2_u2:F0:L2_U2:whole_region:{architecture}:h7"
        variant_id = f"variant:{logical_id.removeprefix('cell:')}:unbounded"
        logical_cells.append(
            {
                "logical_cell_id": logical_id,
                "family": "l2_u2_phase1_sensitivity",
                "forcing": "F0",
                "local_level": "L2_U2",
                "geometry": "whole_region",
                "architecture": architecture,
                "lead_days": 7,
                "post_outcome_status": "PRE_OUTCOME_UNRUN_PHASE1_PLAN",
                "protocol_state": "REGISTERED",
                **_record_guard(),
            }
        )
        variants.append(
            {
                "variant_id": variant_id,
                "logical_cell_id": logical_id,
                "family": "l2_u2_phase1_sensitivity",
                "constraint": "unbounded",
                **_record_guard(),
            }
        )
    for phase1 in l2_phase1:
        architecture = str(phase1["architecture"])
        fold = int(phase1["fold"])
        logical_id = f"cell:l2_u2:F0:L2_U2:whole_region:{architecture}:h7"
        variant_id = f"variant:{logical_id.removeprefix('cell:')}:unbounded"
        fits.append(
            {
                "fit_id": f"fit:l2_u2:{phase1['fit_id']}",
                "family": "l2_u2_phase1_sensitivity",
                "logical_cell_id": logical_id,
                "variant_id": variant_id,
                "forcing": "F0",
                "local_level": "L2_U2",
                "geometry": "whole_region",
                "architecture": architecture,
                "lead_days": 7,
                "constraint": "unbounded",
                "split_seed": None,
                "fold_index": fold,
                "fit_seed": int(phase1["fit_seed"]),
                "spatial_fold_id": f"fold:whole_region:fold{fold}",
                "temporal_partition_id": None,
                "model_id": _model_id("l2_u2", architecture, "unbounded"),
                "input_id": _input_id("l2_u2", "F0", "L2_U2"),
                "environment_id": "environment:route_a_candidate",
                "phase1_fit_id": str(phase1["fit_id"]),
                **_record_guard(),
            }
        )

    logical_cells.sort(key=lambda record: str(record["logical_cell_id"]))
    variants.sort(key=lambda record: str(record["variant_id"]))
    fits.sort(key=lambda record: str(record["fit_id"]))
    document = {
        **_registry_header(
            "thermoroute-cell-registry-v4-candidate",
            "thermoroute.semantic-contract.cell-registry.v4-candidate.1",
        ),
        "registered_scope": {
            "scope_name": "CURRENT_REGISTERED_SUBSET_ONLY_NOT_COMPLETE_V4_MATRIX",
            "complete_v4_matrix": False,
            "partial_matrix_claimed_complete": False,
            "families": [
                "v5_observed_lineage_tree_correction",
                "primary_lightgbm_f0_f3_planned",
                "neural_phase1_f0_f3_registered_subset",
                "l2_u2_phase1_sensitivity",
            ],
            "explicitly_excluded_or_degraded": [
                "F2b archived-vintage primary cells: PLANNED_DEGRADED",
                "withdrawn corrected historical 432-cell information ladder: EXCLUDED",
                "future forcing placebos: ABSENT_UNRUN",
                "future forcing component extensions: ABSENT_UNRUN",
                "L3 and hybrid extensions: ABSENT_UNRUN",
                "cohort, novelty, and audit extensions: ABSENT_UNRUN",
                "any scope requiring an absent production loader, trainer, scorer, or writer: BLOCKED",
            ],
        },
        "arithmetic": {
            "logical_cell_count": EXPECTED_LOGICAL_CELLS,
            "variant_count": 62,
            "fit_count": 5_332,
            "v5_logical_cells": 12,
            "v5_variants": 12,
            "v5_fits": 12,
            "primary_lightgbm_f0_f3_planned_logical_cells": 24,
            "primary_lightgbm_f0_f3_variants": 0,
            "primary_lightgbm_f0_f3_fits": 0,
            "neural_plain_tcn_logical_cells": 24,
            "neural_selected_thermoroute_logical_cells": 12,
            "neural_plain_tcn_variants": 24,
            "neural_thermoroute_bounded_unbounded_variants": 24,
            "neural_total_variants": 48,
            "neural_fits": 5_280,
            "l2_u2_logical_cells": 2,
            "l2_u2_variants": 2,
            "l2_u2_fits": 40,
            "fit_sum": "12 + 5280 + 40 = 5332",
        },
        "independent_phase1_plan_receipts": {
            "neural": {
                "algorithm": "canonical JSON SHA-256 over independently reconstructed ordered Phase-1 fit records",
                "expected_sha256": EXPECTED_NEURAL_PHASE1_PLAN_SHA256,
                "observed_sha256": _canonical_sha256(neural_phase1),
                "fit_count": len(neural_phase1),
                "runner_imported_or_executed": False,
            },
            "l2_u2": {
                "algorithm": "canonical JSON SHA-256 over independently reconstructed ordered Phase-1 fit records",
                "expected_sha256": EXPECTED_L2_U2_PHASE1_PLAN_SHA256,
                "observed_sha256": _canonical_sha256(l2_phase1),
                "fit_count": len(l2_phase1),
                "runner_imported_or_executed": False,
            },
        },
        "protocol_declared_cell_status": _protocol_declared_cell_status_map(),
        "logical_cells": logical_cells,
        "variants": variants,
        "fits": fits,
    }
    return document


def _region_memberships(
    stations: Sequence[str], huc2_by_station: Mapping[str, str]
) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    station_tuple = tuple(stations)
    groups: dict[str, list[str]] = {}
    for station in station_tuple:
        groups.setdefault(huc2_by_station[station], []).append(station)
    buckets: list[list[str]] = [[] for _ in FOLDS]
    huc_buckets: list[list[str]] = [[] for _ in FOLDS]
    loads = [0 for _ in FOLDS]
    for huc, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = min(FOLDS, key=lambda index: (loads[index], index))
        buckets[fold].extend(members)
        huc_buckets[fold].append(huc)
        loads[fold] += len(members)
    station_set = set(station_tuple)
    return [
        (
            tuple(sorted(station_set - set(buckets[fold]))),
            tuple(sorted(buckets[fold])),
            tuple(sorted(huc_buckets[fold])),
        )
        for fold in FOLDS
    ]


def _random_memberships(
    stations: Sequence[str], seed: int
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    station_tuple = tuple(stations)
    _require(seed in RANDOM_SPLIT_SEEDS, "random split seed is outside 0..9")
    permutation = np.random.default_rng(seed).permutation(len(station_tuple))
    station_set = set(station_tuple)
    result: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    start = 0
    for size in HOLD_SIZES:
        indices = permutation[start : start + size]
        held = tuple(sorted(station_tuple[int(index)] for index in indices))
        train = tuple(sorted(station_set - set(held)))
        result.append((train, held))
        start += size
    return result


def _fold_record(
    *,
    fold_id: str,
    geometry: str,
    split_seed: int | None,
    fold_index: int | None,
    train: Sequence[str],
    held: Sequence[str],
    scoreable: Sequence[str],
    held_huc2: Sequence[str],
    station_relation: str,
) -> dict[str, object]:
    train_tuple = tuple(train)
    held_tuple = tuple(held)
    scoreable_tuple = tuple(scoreable)
    return {
        "fold_id": fold_id,
        "geometry": geometry,
        "split_seed": split_seed,
        "fold_index": fold_index,
        "station_relation": station_relation,
        "train_stations": list(train_tuple),
        "held_stations": list(held_tuple),
        "scoreable_held_stations": list(scoreable_tuple),
        "held_huc2": list(held_huc2),
        "train_station_count": len(train_tuple),
        "held_station_count": len(held_tuple),
        "scoreable_held_station_count": len(scoreable_tuple),
        "train_stations_sha256": _canonical_sha256(list(train_tuple)),
        "held_stations_sha256": _canonical_sha256(list(held_tuple)),
        "scoreable_held_stations_sha256": _canonical_sha256(list(scoreable_tuple)),
        **_record_guard(),
    }


def _known_site_temporal_partition_record(
    stations: Sequence[str], reportable_stations: Sequence[str]
) -> dict[str, object]:
    station_tuple = tuple(stations)
    reportable_tuple = tuple(reportable_stations)
    return {
        "partition_id": KNOWN_SITE_TEMPORAL_PARTITION_ID,
        "partition_type": "FIXED_DATE_RANGES_ON_THE_SAME_KNOWN_SITE_UNIVERSE",
        "spatial_cv_fold_applicable": False,
        "spatial_fold_id": None,
        "split_seed_applicable": False,
        "split_seed": None,
        "fit_period": "2006-01-01/2015-12-31",
        "validation_period": "2016-01-01/2017-12-31",
        "evaluation_period": "2021-01-01/2023-12-31",
        "fit_stations": list(station_tuple),
        "evaluation_stations": list(station_tuple),
        "scoreable_evaluation_stations": list(reportable_tuple),
        "fit_station_count": len(station_tuple),
        "evaluation_station_count": len(station_tuple),
        "scoreable_evaluation_station_count": len(reportable_tuple),
        "fit_stations_sha256": _canonical_sha256(list(station_tuple)),
        "evaluation_stations_sha256": _canonical_sha256(list(station_tuple)),
        "scoreable_evaluation_stations_sha256": _canonical_sha256(list(reportable_tuple)),
        **_record_guard(),
    }


def _build_fold_registry(
    stations: Sequence[str],
    huc2_by_station: Mapping[str, str],
    reportable_stations: Sequence[str],
) -> dict[str, object]:
    stations = tuple(stations)
    reportable = tuple(reportable_stations)
    reportable_set = set(reportable)
    station_records = [
        {
            "site_id": station,
            "huc2": huc2_by_station[station],
            "in_training_universe": True,
            "in_reportable_scoring_universe": station in reportable_set,
        }
        for station in stations
    ]
    temporal_partitions = [_known_site_temporal_partition_record(stations, reportable)]
    folds: list[dict[str, object]] = []
    for seed in RANDOM_SPLIT_SEEDS:
        for fold, (train, held) in enumerate(_random_memberships(stations, seed)):
            scoreable = tuple(station for station in held if station in reportable_set)
            folds.append(
                _fold_record(
                    fold_id=f"fold:random:split{seed:02d}:fold{fold}",
                    geometry="random_site",
                    split_seed=seed,
                    fold_index=fold,
                    train=train,
                    held=held,
                    scoreable=scoreable,
                    held_huc2=sorted({huc2_by_station[station] for station in held}),
                    station_relation="DISJOINT_PARTITION_OF_120_TRAINING_STATIONS",
                )
            )
    region_phase1_records: list[dict[str, object]] = []
    for fold, (train, held, held_huc2) in enumerate(_region_memberships(stations, huc2_by_station)):
        region_phase1_records.append(
            {
                "fold": fold,
                "geometry": "whole_region",
                "train_stations": list(train),
                "held_stations": list(held),
                "held_huc2": list(held_huc2),
            }
        )
        scoreable = tuple(station for station in held if station in reportable_set)
        folds.append(
                _fold_record(
                    fold_id=f"fold:whole_region:fold{fold}",
                    geometry="whole_region",
                    split_seed=None,
                fold_index=fold,
                train=train,
                held=held,
                scoreable=scoreable,
                held_huc2=held_huc2,
                station_relation="DISJOINT_INTACT_HUC2_PARTITION_OF_120_TRAINING_STATIONS",
            )
        )
    return {
        **_registry_header(
            "thermoroute-fold-registry-v4-candidate",
            "thermoroute.semantic-contract.fold-registry.v4-candidate.1",
        ),
        "universe_contract": {
            "training_station_count": 120,
            "reportable_scoring_station_count": 116,
            "training_stations_sha256": _canonical_sha256(list(stations)),
            "reportable_scoring_stations_sha256": _canonical_sha256(list(reportable)),
            "station_huc2_sha256": _canonical_sha256(
                [[station, huc2_by_station[station]] for station in stations]
            ),
            "known_site_temporal_partition_id": KNOWN_SITE_TEMPORAL_PARTITION_ID,
            "known_site_temporal_is_spatial_cv_fold": False,
            "known_site_temporal_training_universe_is_120": True,
            "known_site_temporal_scoreable_universe_is_116": True,
            "scoreability_rule": "held_stations intersect exact 116-station primary reportable scoring universe",
            "station_255534081324000_preserved": "255534081324000" in stations,
            "huc2_encoding": "exact two-character zero-padded ASCII decimal",
        },
        "algorithm_contract": {
            "random": "numpy Generator(PCG64(seed)).permutation over sorted 120-station universe; contiguous hold sizes 30/30/31/29",
            "random_split_seeds": list(RANDOM_SPLIT_SEEDS),
            "whole_region": "sort intact HUC2 groups by (-station_count,HUC2); greedily assign to lowest-load fold with lowest-index tie break",
            "split_seed_by_geometry": {
                "random_site": list(RANDOM_SPLIT_SEEDS),
                "whole_region": None,
                "known_site_temporal": None,
            },
            "whole_region_split_seed_applicable": False,
            "known_site_temporal_is_fixed_date_partition_not_cv": True,
            "phase1_whole_region_zero_note": "the independently reconstructed Phase-1 plan serializes zero in fit identities, but semantic split_seed is null because HUC2 packing is deterministic",
            "fold_indices": list(FOLDS),
            "held_sizes": list(HOLD_SIZES),
            "whole_region_scoreable_held_sizes": list(EXPECTED_REGION_SCOREABLE_COUNTS),
        },
        "independent_phase1_fold_receipt": {
            "station_huc2_expected_sha256": EXPECTED_STATION_HUC2_SHA256,
            "station_huc2_observed_sha256": _canonical_sha256(
                [[station, huc2_by_station[station]] for station in stations]
            ),
            "whole_region_expected_sha256": EXPECTED_REGION_FOLD_SHA256,
            "whole_region_observed_sha256": _canonical_sha256(region_phase1_records),
            "fold_record_expected_sha256": {
                str(key): value for key, value in EXPECTED_REGION_FOLD_RECORD_SHA256.items()
            },
            "fold_record_observed_sha256": {
                str(index): _canonical_sha256(record)
                for index, record in enumerate(region_phase1_records)
            },
            "runner_imported_or_executed": False,
        },
        "stations": station_records,
        "temporal_partitions": temporal_partitions,
        "folds": folds,
    }


def _v5_lightgbm_parameters(lead: int) -> dict[str, object]:
    _require(lead in LEADS, "v5 lead is not registered")
    return {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 15 if lead == 7 else 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_estimators": {1: 800, 3: 575, 7: 440}[lead],
        "verbosity": -1,
        "seed": 0,
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
    }


def _neural_objective_contract() -> dict[str, object]:
    return {
        "objective_id": "point_mse_only_v1",
        "formula": "mean((point_prediction-target)^2)",
        "trained_output_heads": ["point"],
        "forbidden_auxiliary_losses": [
            "quantile_pinball",
            "event_bce",
            "quantile_crossing",
            "residual_magnitude",
        ],
        "validation_selection_metric": "unweighted_station_macro_rmse",
        "lead_training_unit": "one_model_per_lead",
    }


def _neural_budget_contract() -> dict[str, object]:
    return {
        "model_fit_seeds": list(FIT_SEEDS),
        "hyperparameter_candidates_per_architecture": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.002,
        "weight_decay": 0.0001,
        "batch_size": 1536,
        "max_epochs": 80,
        "patience": 12,
        "gradient_clip_norm": 1.0,
        "station_sampling": "station_balanced",
        "active_parameter_tolerance_fraction": 0.02,
        "active_parameter_definition": (
            "requires_grad parameters on paths contributing to point_prediction; "
            "quantile/event-only heads excluded"
        ),
        "stopping_information": (
            "same 2016-2017 in-fold-station validation keys and station-macro RMSE"
        ),
        "optimizer_step_matching_rule": (
            "paired architectures receive identical eligible examples, batch size, "
            "maximum update cap, patience, and candidate count"
        ),
    }


def _build_model_registry() -> dict[str, object]:
    models: list[dict[str, object]] = []
    for architecture in ("LightGBM", "ResidualLightGBM"):
        for lead in LEADS:
            models.append(
                {
                    "model_id": _model_id("v5", architecture, "native_tree_configuration", lead),
                    "family": "v5_observed_lineage_tree_correction",
                    "architecture": architecture,
                    "constraint": "native_tree_configuration",
                    "lead_days": lead,
                    "configuration_status": "DEFINED_V5_CONFIGURATION_BUT_NOT_EXECUTION_AUTHORITY",
                    "point_target": (
                        "raw_y"
                        if architecture == "LightGBM"
                        else "damped_residual_with_anchor_added_back_to_prediction"
                    ),
                    "objective": {
                        "training_objective": "LightGBM regression",
                        "validation_period": "2016-01-01/2017-12-31",
                        "validation_callback": "early_stopping(50, verbose=False)",
                        "logging_callback": "log_evaluation(0)",
                        "best_iteration_lower_bound": 1,
                        "best_iteration_upper_bound": {1: 800, 3: 575, 7: 440}[lead],
                    },
                    "optimization": _v5_lightgbm_parameters(lead),
                    "fit_seed_policy": {
                        "fit_seeds": [0],
                        "prediction_num_threads": 1,
                    },
                    "active_parameter_contract": {
                        "applicable": False,
                        "count": None,
                        "seal_blocker": False,
                    },
                    "adapter_contract": {
                        "applicable": False,
                        "schema": None,
                        "seal_blocker": False,
                    },
                    "checkpoint_contract": {
                        "schema": "LightGBM native text model expected but not authorized by this candidate",
                        "exact_completion_parser_available_to_this_registry": False,
                        "seal_blocker": True,
                    },
                    "seal_blockers": [
                        "this candidate is not an execution authority",
                        "no execution runtime is authorized by the environment registry",
                        "completion/checkpoint bytes are future results and are not bound here",
                    ],
                    **_record_guard(),
                }
            )

    for architecture, constraint in (
        ("plain_TCN", "unbounded"),
        ("ThermoRoute", "unbounded"),
        ("ThermoRoute", "bounded"),
    ):
        models.append(
            {
                "model_id": _model_id("neural", architecture, constraint),
                "family": "neural_phase1_f0_f3_registered_subset",
                "architecture": architecture,
                "constraint": constraint,
                "lead_days": None,
                "configuration_status": "INCOMPLETE_BLOCKS_SEAL_AND_EXECUTION",
                "point_target": "unbounded_residual_to_level_legal_anchor",
                "objective": _neural_objective_contract(),
                "optimization": _neural_budget_contract(),
                "fit_seed_policy": {
                    "fit_seeds": list(FIT_SEEDS),
                    "effect_aggregation": "arithmetic_mean_of_within_fit_seed_station_paired_effects",
                    "prediction_ensembling_across_fit_seeds": False,
                },
                "active_parameter_contract": {
                    "applicable": True,
                    "count": None,
                    "match_reference": "plain_TCN versus ThermoRoute active point-prediction paths",
                    "maximum_relative_difference": 0.02,
                    "bounded_and_unbounded_thermoroute_counts_must_match": True,
                    "seal_blocker": True,
                },
                "adapter_contract": {
                    "applicable": True,
                    "schema": None,
                    "required_boundary": "exact input-to-architecture tensor/feature adapter with identical information channels",
                    "seal_blocker": True,
                },
                "checkpoint_contract": {
                    "schema": None,
                    "exact_completion_parser_available_to_this_registry": False,
                    "seal_blocker": True,
                },
                "seal_blockers": [
                    "active parameter counts are absent",
                    "architecture adapter schemas are absent",
                    "checkpoint schema and exact completion parser are absent",
                    "production neural loader, trainer, scorer, and writer are absent",
                    "no execution runtime is authorized",
                ],
                **_record_guard(),
            }
        )

    for architecture in ("LightGBM", "plain_TCN"):
        models.append(
            {
                "model_id": _model_id("l2_u2", architecture, "unbounded"),
                "family": "l2_u2_phase1_sensitivity",
                "architecture": architecture,
                "constraint": "unbounded",
                "lead_days": 7,
                "configuration_status": "INCOMPLETE_BLOCKS_SEAL_AND_EXECUTION",
                "point_target": "unbounded_residual_to_identical_training_station_pooled_anchor",
                "objective": {
                    "training_objective": "point regression only",
                    "validation_selection_metric": "unweighted_station_macro_rmse",
                    "exact_architecture_specific_objective_configuration": None,
                },
                "optimization": {
                    "exact_frozen_architecture_configuration": None,
                    "trainer_implemented": False,
                },
                "fit_seed_policy": {
                    "fit_seeds": list(FIT_SEEDS),
                    "effect_aggregation": "arithmetic_mean_of_within_fit_seed_station_paired_effects",
                    "prediction_ensembling_across_fit_seeds": False,
                },
                "active_parameter_contract": {
                    "applicable": architecture == "plain_TCN",
                    "count": None,
                    "seal_blocker": architecture == "plain_TCN",
                },
                "adapter_contract": {
                    "applicable": True,
                    "schema": None,
                    "seal_blocker": True,
                },
                "checkpoint_contract": {
                    "schema": None,
                    "exact_completion_parser_available_to_this_registry": False,
                    "seal_blocker": True,
                },
                "seal_blockers": [
                    f"frozen {architecture} model configuration and trainer are absent",
                    "authority-bound predictor and label loaders are absent",
                    "architecture adapter and checkpoint schemas are absent",
                    "exact prediction parser, scorer, and atomic writer are absent",
                    "no execution runtime is authorized",
                ],
                **_record_guard(),
            }
        )

    models.sort(key=lambda record: str(record["model_id"]))
    return {
        **_registry_header(
            "thermoroute-model-registry-v4-candidate",
            "thermoroute.semantic-contract.model-registry.v4-candidate.1",
        ),
        "global_policy": {
            "point_prediction_only": True,
            "model_fit_seed_effect_aggregation": "mean_of_within_fit_seed_station_paired_effects",
            "fit_seed_prediction_ensembling": False,
            "missing_active_parameter_counts_block_seal": True,
            "missing_adapters_block_seal": True,
            "missing_checkpoint_schemas_block_seal": True,
            "missing_trainers_block_seal": True,
        },
        "models": models,
        "seal_blocker_summary": [
            "neural active parameter counts are not registered",
            "neural and L2_U2 architecture adapters are not registered",
            "neural and L2_U2 checkpoint/completion schemas are not registered",
            "L2_U2 LightGBM and plain_TCN trainers are not implemented",
            "production loaders/scorers/writers required by planned cells are absent",
            "no environment record authorizes execution",
        ],
    }


def _v5_base_feature_columns() -> list[str]:
    columns: list[str] = []
    for variable in ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"):
        for lag in (0, 1, 2, 3, 5, 7, 10, 14):
            columns.extend((f"{variable}_lag{lag}", f"{variable}_observed_lag{lag}"))
        for window in (3, 7, 14, 30):
            columns.extend(
                (
                    f"{variable}_rollmean{window}",
                    f"{variable}_observed_fraction{window}",
                )
            )
            if variable == "WTEMP":
                columns.append(f"{variable}_rollstd{window}")
        columns.extend(
            (
                f"{variable}_delta1",
                f"{variable}_delta3",
                f"{variable}_delta1_observed",
                f"{variable}_delta3_observed",
            )
        )
    columns.extend(f"doy_h{index}" for index in range(6))
    columns.extend(("clim_t", "clim_target", "clim_anom", "persistence"))
    _require(len(columns) == 210 and len(set(columns)) == 210, "v5 base feature namespace changed")
    return columns


def _future_features(lead: int) -> list[str]:
    return [
        column
        for variable in ("TEMP", "PRCP", "DH", "RHMEAN", "WDSP")
        for column in (f"{variable}_fut{lead}", f"{variable}_futmean{lead}")
    ]


def _input_record(
    *,
    input_id: str,
    family: str,
    forcing: str,
    local_level: str,
    lead_days: int | None,
    allowed_history: Sequence[str],
    forbidden_history: Sequence[str],
    future_channels: Sequence[str],
    feature_schema: Sequence[str],
    preprocessing: Mapping[str, object],
) -> dict[str, object]:
    return {
        "input_id": input_id,
        "family": family,
        "forcing": forcing,
        "local_level": local_level,
        "lead_days": lead_days,
        "input_status": "STRUCTURAL_CANDIDATE_NOT_EXECUTION_READY",
        "history_context_days": 32,
        "allowed_history_channels": list(allowed_history),
        "forbidden_history_channels": list(forbidden_history),
        "future_channels": list(future_channels),
        "feature_schema": list(feature_schema),
        "preprocessing": dict(preprocessing),
        "label_boundary": {
            "training_label_source": SEMANTIC_TRAINING_FILENAME,
            "evaluation_label_source": PRIMARY_KEY_REGISTRY.name,
            "training_admissibility": "raw issue WTEMP observed AND raw target WTEMP observed",
            "evaluation_admissibility": "exact primary reportable key registry",
            "y_true_is_raw_not_imputed": True,
            "target_observedness_is_not_a_model_input": True,
            "labels_are_separate_from_imputed_predictor_projection": True,
        },
        "raw_projection_finiteness": {
            "raw_semantic_daily_registry_claimed_fully_finite": False,
            "all_station_training_meteorology_gap_dates": [
                "2008-12-31",
                "2012-12-31",
            ],
            "gap_variables": ["TEMP", "PRCP", "RHMEAN", "DH"],
            "gap_station_count_each_date": 120,
            "WDSP_gap_on_these_dates": False,
            "finite_model_projection_requires_bound_training_only_fill": True,
            "unbound_or_global_full_period_fill_forbidden": True,
        },
        **_record_guard(),
    }


def _build_input_registry(snapshot: SourceSnapshot) -> dict[str, object]:
    inputs: list[dict[str, object]] = []
    v5_history = (
        "WTEMP_values_and_raw_observed_masks",
        "FLOW_values_and_raw_observed_masks",
        "TEMP_values_and_raw_observed_masks",
        "PRCP_values_and_raw_observed_masks",
        "RHMEAN_values_and_raw_observed_masks",
        "DH_values_and_raw_observed_masks",
        "WDSP_values_and_raw_observed_masks",
        "calendar_features",
        "training_only_water_climatology_and_damped_persistence_anchor",
    )
    v5_preprocessing = {
        "fit_period": "2006-01-01/2015-12-31",
        "validation_period": "2016-01-01/2017-12-31",
        "evaluation_period": "2021-01-01/2023-12-31",
        "fit_membership": "all 120 known training stations",
        "fill": "training-period fitted imputer only; raw observed masks preserved bit-exactly",
        "scaler": "not separately defined for LightGBM",
        "anchor": "training-period per-station harmonic water climatology plus training-only damped persistence state",
        "validation_excluded_from_fit": True,
        "evaluation_excluded_from_fit": True,
        "held_station_state_allowed": True,
    }
    base_features = _v5_base_feature_columns()
    for forcing in ("F0", "F3_full"):
        for lead in LEADS:
            future = [] if forcing == "F0" else _future_features(lead)
            inputs.append(
                _input_record(
                    input_id=_input_id("v5", forcing, "L0", lead),
                    family="v5_observed_lineage_tree_correction",
                    forcing=forcing,
                    local_level="L0",
                    lead_days=lead,
                    allowed_history=v5_history,
                    forbidden_history=(
                        "WLEVEL",
                        "future meteorology for F0" if forcing == "F0" else "future discharge",
                        "imputed y_true",
                    ),
                    future_channels=future,
                    feature_schema=(*base_features, *future),
                    preprocessing=v5_preprocessing,
                )
            )

    neural_preprocessing = {
        "fit_period": "2006-01-01/2015-12-31",
        "validation_period": "2016-01-01/2017-12-31",
        "evaluation_period": "2021-01-01/2023-12-31",
        "fit_membership": "exact fold train_stations only",
        "fill": "pooled nanmedian over dated training-station rows only",
        "scaler": "pooled mean and population standard deviation after training-only fill; zero scale becomes 1.0",
        "anchor": "unweighted mean of finite 366-day training-station WTEMP climatology curves",
        "validation_excluded_from_fit": True,
        "evaluation_excluded_from_fit": True,
        "held_station_state_allowed": False,
    }
    for forcing in ("F0", "F3_full"):
        for local_level in ("L0", "L2"):
            allowed = [
                "FLOW_values_and_raw_observed_masks",
                "TEMP_values_and_raw_observed_masks",
                "PRCP_values_and_raw_observed_masks",
                "RHMEAN_values_and_raw_observed_masks",
                "DH_values_and_raw_observed_masks",
                "WDSP_values_and_raw_observed_masks",
                "target_day_season_sin_cos",
                "level_legal_anchor",
            ]
            forbidden = [
                "WLEVEL",
                "labels_or_target_observedness",
                "target_site_identity_in_transfer_models",
            ]
            if local_level == "L0":
                allowed.insert(0, "WTEMP_values_and_raw_observed_masks")
            else:
                forbidden.extend(
                    [
                        "WTEMP_values",
                        "WTEMP_observedness",
                        "target_site_climatology_or_decay_state",
                        "target_site_imputer_or_scaler_state",
                        "target_site_anchor_candidate",
                    ]
                )
            future = (
                []
                if forcing == "F0"
                else [
                    f"{variable}_{statistic}"
                    for variable in ("TEMP", "PRCP", "DH", "RHMEAN", "WDSP")
                    for statistic in ("target_day", "prefix_mean")
                ]
            )
            if forcing == "F0":
                forbidden.append("every future meteorology value/mask/derived statistic")
            inputs.append(
                _input_record(
                    input_id=_input_id("neural", forcing, local_level),
                    family="neural_phase1_f0_f3_registered_subset",
                    forcing=forcing,
                    local_level=local_level,
                    lead_days=None,
                    allowed_history=allowed,
                    forbidden_history=forbidden,
                    future_channels=future,
                    feature_schema=(
                        "32xordered_allowed_history_values",
                        "32xordered_allowed_history_raw_observed_masks",
                        "target_day_season_sin_cos",
                        "level_legal_anchor",
                        *future,
                    ),
                    preprocessing=neural_preprocessing,
                )
            )

    l2_u2_allowed = [
        "TEMP_values_and_raw_observed_masks",
        "PRCP_values_and_raw_observed_masks",
        "RHMEAN_values_and_raw_observed_masks",
        "DH_values_and_raw_observed_masks",
        "WDSP_values_and_raw_observed_masks",
        "target_day_season_sin_cos",
        "training_station_pooled_anchor",
    ]
    inputs.append(
        _input_record(
            input_id=_input_id("l2_u2", "F0", "L2_U2"),
            family="l2_u2_phase1_sensitivity",
            forcing="F0",
            local_level="L2_U2",
            lead_days=7,
            allowed_history=l2_u2_allowed,
            forbidden_history=(
                "WTEMP_values",
                "WTEMP_observedness",
                "FLOW_values",
                "FLOW_observedness",
                "WLEVEL",
                "target_site_identity",
                "target_site_climatology_or_decay_state",
                "target_site_imputer_or_scaler_state",
                "target_site_anchor_candidate",
                "labels_or_target_observedness",
                "every future meteorology value/mask/derived statistic",
            ),
            future_channels=(),
            feature_schema=(
                "32x5_ordered_meteorology_values",
                "32x5_ordered_raw_observed_masks",
                "target_day_season_sin_cos",
                "training_station_pooled_anchor",
            ),
            preprocessing=neural_preprocessing,
        )
    )
    inputs.sort(key=lambda record: str(record["input_id"]))
    source_roles = (
        "semantic_data_manifest",
        "semantic_daily_registry",
        "semantic_training_registry",
        "station_registry",
        "key_authority_manifest",
        "primary_key_registry",
        "defect_authority_manifest",
        "defect_authority_report",
    )
    return {
        **_registry_header(
            "thermoroute-input-registry-v4-candidate",
            "thermoroute.semantic-contract.input-registry.v4-candidate.1",
        ),
        "source_bindings": {role: dict(snapshot.bindings[role]) for role in source_roles},
        "source_authority_boundary": {
            "semantic_data_manifest_status": STATUS,
            "semantic_data_is_formal_authority": False,
            "semantic_data_can_authorize_execution": False,
            "semantic_data_candidate_trusted_only_for_byte_and_structure_validation": True,
            "semantic_data_candidate_is_score_or_result_evidence": False,
            "candidate_model_score_checkpoint_effect_or_result_inputs_read": False,
            "key_authority_role": "KEY_LABEL_AND_REPORTABILITY_AUTHORITY_ONLY_NOT_MODEL_SCORE_AUTHORITY",
            "defect_authority_role": "WITHDRAWAL_AND_RERUN_REQUIREMENTS_NOT_SCORE_AUTHORIZATION",
            "station_authority_role": "PINNED_STATION_AND_HUC2_IDENTITY_ONLY",
            "dependency_authority_transfers_to_this_candidate": False,
        },
        "source_schemas": {
            SEMANTIC_DAILY_FILENAME: dict(snapshot.semantic_outputs[SEMANTIC_DAILY_FILENAME]),
            SEMANTIC_TRAINING_FILENAME: dict(snapshot.semantic_outputs[SEMANTIC_TRAINING_FILENAME]),
            PRIMARY_KEY_REGISTRY.name: {
                "path": PRIMARY_KEY_REGISTRY.name,
                "sha256": snapshot.bindings["primary_key_registry"]["sha256"],
                "size_bytes": snapshot.bindings["primary_key_registry"]["size_bytes"],
                "row_count": 358_765,
                "columns": list(PRIMARY_KEY_COLUMNS),
                "arrow_schema": _schema_signature(PRIMARY_KEY_ARROW_SCHEMA),
            },
        },
        "period_contract": {
            "training": "2006-01-01/2015-12-31",
            "validation": "2016-01-01/2017-12-31",
            "evaluation": "2021-01-01/2023-12-31",
            "training_and_validation_disjoint": True,
            "evaluation_excluded_from_every_preprocessing_fit": True,
            "registered_leads_days": list(LEADS),
        },
        "raw_label_predictor_separation": {
            "raw_observed_masks_captured_before_fill": True,
            "raw_masks_survive_projection_bit_exactly": True,
            "predictor_fill_allowed_only_after_raw_capture": True,
            "raw_y_true_may_be_filled": False,
            "training_and_evaluation_labels_are_separate_typed_inputs": True,
            "semantic_daily_raw_projection_is_fully_finite": False,
        },
        "inputs": inputs,
        "excluded_input_scopes": [
            "F2a and F2b are not registered model inputs here",
            "F1 and F3_temperature_only diagnostics are not registered here",
            "placebo and forcing-component inputs are absent",
            "L1, L3, hybrid, cohort, novelty, and audit inputs are absent",
            "any input requiring an absent production loader is blocked",
        ],
    }


def _geometry_membership_contract(geometry: str) -> dict[str, object]:
    _require(
        geometry in {"known_site_temporal", "random_site", "whole_region"},
        "contrast geometry is not registered",
    )
    if geometry == "known_site_temporal":
        return {
            "split_seed_applicable": False,
            "split_seeds": [],
            "spatial_cv_fold_indices": [],
            "temporal_partition_ids": [KNOWN_SITE_TEMPORAL_PARTITION_ID],
            "station_membership_rule": "same known stations across fixed fit, validation, and evaluation date ranges",
        }
    if geometry == "random_site":
        return {
            "split_seed_applicable": True,
            "split_seeds": list(RANDOM_SPLIT_SEEDS),
            "spatial_cv_fold_indices": list(FOLDS),
            "temporal_partition_ids": [],
            "station_membership_rule": "each station is held in exactly one of four folds within each of ten PCG64 partitions",
        }
    return {
        "split_seed_applicable": False,
        "split_seeds": [],
        "spatial_cv_fold_indices": list(FOLDS),
        "temporal_partition_ids": [],
        "station_membership_rule": "each station is held in exactly one deterministic intact-HUC2 fold; no seed dimension exists",
    }


def _global_geometry_aggregation_contracts() -> dict[str, object]:
    descriptions = {
        "known_site_temporal": "one fixed date partition; no spatial fold, split-seed, or fold aggregation",
        "random_site": "pair effects inside each seed and the station's held fold, then arithmetic-mean the ten seed-specific station effects",
        "whole_region": "use the station's one deterministic intact-HUC2 held fold; no split-seed or across-fold averaging",
    }
    return {
        geometry: {
            **_geometry_membership_contract(geometry),
            "effect_aggregation": descriptions[geometry],
        }
        for geometry in ("known_site_temporal", "random_site", "whole_region")
    }


def _geometry_aggregation_order(geometry: str, fit_seeds: Sequence[int]) -> list[str]:
    if geometry == "known_site_temporal":
        steps = [
            "inside the fixed temporal partition and each fit seed, compute station RMSE over exact two-sided matched evaluation keys",
            "construct any non-geometry selector contrast inside the partition; do not create or average CV folds",
        ]
    elif geometry == "random_site":
        steps = [
            "inside each random split seed, select the one fold that holds the station and compute station RMSE over exact two-sided matched keys",
            "inside that same split seed and fit seed, construct any non-geometry selector contrast; retain side-specific risks when geometry is the contrasted axis",
            "for each station and fit seed, take the arithmetic mean of the ten seed-specific paired risks or effects as required by the registered formula",
        ]
    else:
        _require(geometry == "whole_region", "aggregation geometry is not registered")
        steps = [
            "select the station's one deterministic intact-HUC2 fold and compute station RMSE over exact two-sided matched keys",
            "inside that fold and fit seed, construct any non-geometry selector contrast; retain the risk when geometry is the contrasted axis; do not average folds or split seeds",
        ]
    if len(tuple(fit_seeds)) > 1:
        steps.append(
            "for each station, take the arithmetic mean of paired effects over fit seeds 0..4 after geometry-specific pairing"
        )
    return steps


def _contrast_record(
    *,
    contrast_id: str,
    family: str,
    availability: str,
    estimand: str,
    formula: str,
    absolute_reported: bool,
    selectors: Mapping[str, object],
    geometries: Sequence[str],
    fit_seeds: Sequence[int],
    cross_geometry_aggregation: str | None = None,
    blocker: str | None = None,
) -> dict[str, object]:
    geometry_tuple = tuple(geometries)
    _require(len(geometry_tuple) > 0, "contrast geometry scope is empty")
    geometry_contracts = {
        geometry: _geometry_membership_contract(geometry) for geometry in geometry_tuple
    }
    return {
        "contrast_id": contrast_id,
        "family": family,
        "availability": availability,
        "estimand": estimand,
        "formula": formula,
        "absolute_reported": absolute_reported,
        "selectors": dict(selectors),
        "matched_keys": {
            "identity": ["site_id", "issue_date", "target_date", "lead_days"],
            "two_sided_exact_equality": True,
            "key_source": PRIMARY_KEY_REGISTRY.name,
            "key_declining_allowed": False,
        },
        "matched_stations": {
            "source": "exact 116-station reportable scoring universe intersected with both selector sides",
            "equal_station_weighting": True,
            "different_station_sets_prohibited": True,
        },
        "matched_folds": {
            "geometry_contracts": geometry_contracts,
            "same_spatial_fold_membership_required_within_non_geometry_contrast": True,
            "known_site_temporal_partition_is_not_a_cv_fold": True,
        },
        "matched_fit_seeds": list(fit_seeds),
        "fit_seed_policy": {
            "pair_inside_each_fit_seed": True,
            "aggregate_effects_after_pairing": "arithmetic_mean",
            "prediction_ensembling_across_fit_seeds": False,
        },
        "geometry_aggregation_order": {
            geometry: _geometry_aggregation_order(geometry, fit_seeds)
            for geometry in geometry_tuple
        },
        "cross_geometry_aggregation": (
            cross_geometry_aggregation
            if cross_geometry_aggregation is not None
            else "NOT_APPLICABLE; geometry is fixed or results remain separate by geometry"
        ),
        "final_aggregation_order": [
            "apply the registered selector contrast or signed interaction to geometry-specific station effects",
            "apply an absolute-value transform only when absolute_reported is true",
            "take the unweighted median across exact matched reportable stations",
        ],
        "blocker": blocker,
        **_record_guard(),
    }


def _build_contrast_registry() -> dict[str, object]:
    contrasts = [
        _contrast_record(
            contrast_id="contrast:v5:forcing_value:F3_full_vs_F0",
            family="v5_observed_lineage_tree_correction",
            availability="DEFINED_POST_OUTCOME_CORRECTION_NOT_EXECUTED_BY_THIS_CANDIDATE",
            estimand="station-first realized-future forcing oracle value",
            formula="R_i(F0)-R_i(F3_full)",
            absolute_reported=False,
            selectors={
                "left": {"forcing": "F0"},
                "right": {"forcing": "F3_full"},
                "held_constant": ["architecture", "lead_days", "known_site_temporal"],
            },
            geometries=("known_site_temporal",),
            fit_seeds=(0,),
        ),
        _contrast_record(
            contrast_id="contrast:neural:forcing_value:F3_full_vs_F0",
            family="neural_phase1_f0_f3_registered_subset",
            availability="DEFINED_UNRUN",
            estimand="station-first realized-future forcing oracle value",
            formula="R_i(F0)-R_i(F3_full)",
            absolute_reported=False,
            selectors={
                "left": {"forcing": "F0"},
                "right": {"forcing": "F3_full"},
                "held_constant": [
                    "local_level",
                    "geometry",
                    "architecture",
                    "constraint",
                    "lead_days",
                ],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
        ),
        _contrast_record(
            contrast_id="contrast:neural:local_state_value:L2_vs_L0",
            family="neural_phase1_f0_f3_registered_subset",
            availability="DEFINED_UNRUN",
            estimand="station-first local-state value",
            formula="R_i(L2)-R_i(L0)",
            absolute_reported=False,
            selectors={
                "left": {"local_level": "L2"},
                "right": {"local_level": "L0"},
                "held_constant": [
                    "forcing",
                    "geometry",
                    "architecture",
                    "constraint",
                    "lead_days",
                ],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
        ),
        _contrast_record(
            contrast_id="contrast:neural:geometry_penalty:whole_region_vs_random_site",
            family="neural_phase1_f0_f3_registered_subset",
            availability="DEFINED_UNRUN",
            estimand="station-first geometry penalty",
            formula="R_i(whole_region)-R_i(random_site)",
            absolute_reported=False,
            selectors={
                "left": {"geometry": "whole_region"},
                "right": {"geometry": "random_site"},
                "held_constant": [
                    "forcing",
                    "local_level",
                    "architecture",
                    "constraint",
                    "lead_days",
                ],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
            cross_geometry_aggregation="for every station and fit seed, subtract the random-site risk from the same deterministic whole-region risk inside each random split seed, then average the ten seed-specific differences",
        ),
        _contrast_record(
            contrast_id="contrast:neural:architecture_value:plain_TCN_vs_LightGBM",
            family="registered_cross_family_estimand_only",
            availability="BLOCKED_NO_MATCHED_REGISTERED_LIGHTGBM_NEURAL_FITS",
            estimand="station-first architecture value",
            formula="R_i(LightGBM)-R_i(plain_TCN)",
            absolute_reported=False,
            selectors={
                "left": {"architecture": "LightGBM"},
                "right": {"architecture": "plain_TCN"},
                "held_constant": [
                    "forcing",
                    "local_level",
                    "geometry",
                    "lead_days",
                    "keys",
                    "stations",
                ],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
            blocker="the current neural fit inventory contains plain_TCN and selected ThermoRoute only; it does not contain matched LightGBM neural fits",
        ),
        _contrast_record(
            contrast_id="contrast:neural:interaction:L_by_G_absolute",
            family="neural_phase1_f0_f3_registered_subset",
            availability="DEFINED_UNRUN",
            estimand="absolute local-information by geometry interaction",
            formula="abs((R_region-R_random)_L2-(R_region-R_random)_L0)",
            absolute_reported=True,
            selectors={
                "interaction_axes": ["local_level", "geometry"],
                "held_constant": [
                    "forcing",
                    "architecture",
                    "constraint",
                    "lead_days",
                ],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
            cross_geometry_aggregation="construct the region-minus-random paired effect inside each random split seed before taking the L2-minus-L0 interaction; average the ten signed interactions per station before absolute value",
        ),
        _contrast_record(
            contrast_id="contrast:neural:interaction:F_by_A_absolute",
            family="registered_estimand_blocked_by_missing_lightgbm_neural_fits",
            availability="BLOCKED_NO_MATCHED_REGISTERED_LIGHTGBM_NEURAL_FITS",
            estimand="absolute forcing by architecture interaction",
            formula="abs(V_F3_for_plain_TCN-V_F3_for_LightGBM)",
            absolute_reported=True,
            selectors={
                "interaction_axes": ["forcing", "architecture"],
                "held_constant": ["local_level", "geometry", "lead_days"],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
            blocker="matched LightGBM neural fits are outside the current registered fit inventory",
        ),
        _contrast_record(
            contrast_id="contrast:neural:interaction:L_by_A_absolute",
            family="registered_estimand_blocked_by_missing_lightgbm_neural_fits",
            availability="BLOCKED_NO_MATCHED_REGISTERED_LIGHTGBM_NEURAL_FITS",
            estimand="absolute local-information by architecture interaction",
            formula="abs((R_LightGBM-R_plain_TCN)_L2-(R_LightGBM-R_plain_TCN)_L0)",
            absolute_reported=True,
            selectors={
                "interaction_axes": ["local_level", "architecture"],
                "held_constant": ["forcing", "geometry", "lead_days"],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
            blocker="matched LightGBM neural fits are outside the current registered fit inventory",
        ),
        _contrast_record(
            contrast_id="contrast:neural:interaction:F_by_L_absolute",
            family="neural_phase1_f0_f3_registered_subset",
            availability="DEFINED_UNRUN",
            estimand="absolute forcing by local-information interaction",
            formula="abs(V_F3_at_L2-V_F3_at_L0)",
            absolute_reported=True,
            selectors={
                "interaction_axes": ["forcing", "local_level"],
                "held_constant": [
                    "geometry",
                    "architecture",
                    "constraint",
                    "lead_days",
                ],
            },
            geometries=("random_site", "whole_region"),
            fit_seeds=FIT_SEEDS,
        ),
        _contrast_record(
            contrast_id="contrast:l2_u2:L2_U2_vs_L2",
            family="l2_u2_phase1_sensitivity",
            availability="BLOCKED",
            estimand="fully local-observation-free sensitivity relative to corrected L2",
            formula="R_i(L2_U2)-R_i(L2)",
            absolute_reported=False,
            selectors={
                "left": {"local_level": "L2_U2"},
                "right": {"local_level": "L2"},
                "held_constant": [
                    "F0",
                    "whole_region",
                    "h7",
                    "architecture",
                    "fold",
                    "fit_seed",
                    "keys",
                    "stations",
                    "source_authority",
                ],
            },
            geometries=("whole_region",),
            fit_seeds=FIT_SEEDS,
            blocker="BLOCKED until a corrected same-authority L2 comparator with identical keys, stations, folds, fit seeds, preprocessing lineage, and model configuration exists",
        ),
    ]
    contrasts.sort(key=lambda record: str(record["contrast_id"]))
    return {
        **_registry_header(
            "thermoroute-contrast-registry-v4-candidate",
            "thermoroute.semantic-contract.contrast-registry.v4-candidate.1",
        ),
        "geometry_specific_aggregation_contracts": _global_geometry_aggregation_contracts(),
        "station_first_estimator": {
            "station_risk": "R_i(c,h)=sqrt(mean over exact common keys for station i of (prediction-observation)^2)",
            "primary_summary": "unweighted median of station-level paired effects",
            "random_seed_first": "pair within split seed; mean the ten seed-specific station effects; never concatenate rows across seeds",
            "whole_region_seed_rule": "whole-region packing is deterministic; split_seed is null and no seed averaging is allowed",
            "known_site_temporal_rule": "the fixed date partition is not a CV fold and has no fold or seed aggregation",
            "region_vs_random_rule": "within each station and fit seed, reuse the deterministic region risk against each random seed, average the ten paired differences, then aggregate fit-seed effects",
            "model_fit_seed_first": "pair within fit seed; mean the five within-fit-seed station effects; never ensemble predictions across fit seeds",
            "station_weighting": "each matched reportable station receives equal weight",
            "daily_row_pooling_primary": False,
        },
        "uncertainty_and_sensitivity": {
            "classification": "APPROXIMATE_FIXED_COHORT_DESCRIPTIVE_SENSITIVITIES_ONLY",
            "whole_huc2_cluster_bootstrap_draws": 10_000,
            "bootstrap_seed_namespace": "thermoroute-semantic-contract-registry-v4-bootstrap",
            "bootstrap_seed_formula": "uint32 big-endian first four SHA256 bytes of namespace + newline + contrast_id + newline",
            "bootstrap_draw_rule": "sample K of K observed HUC2 clusters with replacement and retain every station in each selected cluster with multiplicity",
            "exact_whole_huc2_sign_flip": "enumerate all 2^K assignments of one common sign per HUC2; K=15 for the frozen cohort",
            "expected_sign_flip_configurations": 32_768,
            "leave_one_huc2": "remove every station in one HUC2 and recompute the unweighted station median for each of 15 HUC2 groups",
            "confirmatory_or_population_generalizing_claim_allowed": False,
        },
        "recovery_denominator_guard": {
            "ratio_reportable_only_if_paired_oracle_value_positive": True,
            "ratio_reportable_if_oracle_interval_includes_zero": False,
            "minimum_absolute_oracle_value_c": 0.05,
            "otherwise_report_numerator_and_denominator_separately": True,
            "F2a_denominator_if_future_scope_is_ever_registered": "F3_temperature_only_NEVER_F3_full",
            "F2b_denominator_if_upgrade_gates_ever_pass": "F3_full",
        },
        "contrasts": contrasts,
        "prohibited_comparisons": [
            "difference of marginal station-median RMSE values presented as a paired effect",
            "pooled daily-row RMSE as the primary effect",
            "unmatched forecast keys, stations, folds, split seeds, or fit seeds",
            "prediction ensembling across model fit seeds before effect construction",
            "ratios whose numerator and denominator differ in F/L/G/A/key/station conditions",
            "F2a divided by F3_full",
            "F2b comparisons while F2b remains PLANNED_DEGRADED",
            "v5 known-site temporal fits compared directly with neural spatial fits",
            "L2_U2 versus any L2 result not produced under the same corrected source authority",
            "additive information budget without a separately frozen factorial, ANOVA, or Shapley contract",
        ],
    }


def _build_environment_registry(snapshot: SourceSnapshot) -> dict[str, object]:
    environments = [
        {
            "environment_id": "environment:historical_authority_serialization",
            "role": "HISTORICAL_KEY_AND_DEFECT_AUTHORITY_SERIALIZATION_ONLY",
            "runtime": dict(HISTORICAL_AUTHORITY_RUNTIME),
            "packages": {
                "numpy": "1.26.4",
                "pandas": "2.1.4",
                "pyarrow": "14.0.2",
                "arrow_cpp": "14.0.2",
                "lightgbm": "NOT_ASSERTED_BY_THE_BOUND_AUTHORITY_SERIALIZATION_RECEIPTS",
                "scikit_learn": "NOT_ASSERTED_BY_THE_BOUND_AUTHORITY_SERIALIZATION_RECEIPTS",
                "torch": "NOT_ASSERTED_BY_THE_BOUND_AUTHORITY_SERIALIZATION_RECEIPTS",
            },
            "lock_relationship": "historical authority serialization runtime differs from both bound Python-3.12 repository locks",
            "runtime_identity_enforced_at_candidate_build": False,
            "direct_lock_pin_set_complete": False,
            "may_execute_registered_fits": False,
            "thread_policy": {
                "model_training_threads": None,
                "prediction_threads": None,
                "role_is_serialization_not_execution": True,
            },
            "determinism_policy": {
                "applicable_to_model_execution": False,
                "authority_verified": False,
            },
            **_record_guard(),
        },
        {
            "environment_id": "environment:route_a_candidate",
            "role": "ROUTE_A_CANDIDATE_SERIALIZATION_AND_FUTURE_EXECUTION_TARGET_NOT_AUTHORIZED",
            "runtime": dict(ROUTE_A_RUNTIME),
            "packages": dict(ROUTE_A_DIRECT_LOCK_PINS),
            "lock_relationship": "complete direct pins from the bound build-system and requirements lock plus the fully hashed transitive Python-3.12 lock; bindings alone do not authorize execution",
            "runtime_identity_enforced_at_candidate_build": True,
            "direct_lock_pin_set_complete": True,
            "may_execute_registered_fits": False,
            "thread_policy": {
                "THERMOROUTE_FORMAL_THREADS": "1",
                "LightGBM_n_jobs": 1,
                "LightGBM_prediction_num_threads": 1,
                "torch_intraop_threads": 1,
                "torch_interop_threads": 1,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "thread_policy_must_be_verified_by_future_execution_authority": True,
            },
            "determinism_policy": {
                "applicable_to_model_execution": True,
                "LightGBM_deterministic": True,
                "LightGBM_force_col_wise": True,
                "numpy_random_generator": "PCG64",
                "torch_deterministic_algorithms_required": True,
                "authority_verified": False,
            },
            **_record_guard(),
        },
    ]
    return {
        **_registry_header(
            "thermoroute-environment-registry-v4-candidate",
            "thermoroute.semantic-contract.environment-registry.v4-candidate.1",
        ),
        "authority_serialization_bindings": {
            "key_authority_manifest": dict(snapshot.bindings["key_authority_manifest"]),
            "defect_authority_manifest": dict(snapshot.bindings["defect_authority_manifest"]),
        },
        "repository_lock_bindings": {
            "pyproject": dict(snapshot.bindings["pyproject"]),
            "requirements_lock": dict(snapshot.bindings["requirements_lock"]),
            "requirements_lock_py312": dict(snapshot.bindings["requirements_lock_py312"]),
        },
        "runtime_boundary": {
            "historical_serialization_runtime_is_execution_authority": False,
            "route_a_candidate_runtime_is_execution_authority": False,
            "repository_lockfiles_are_execution_authority": False,
            "semantic_data_candidate_runtime_is_execution_authority": False,
            "authorized_execution_environment_id": None,
            "execution_state": "NO_EXECUTION_RUNTIME_AUTHORIZED_YET",
        },
        "environments": environments,
        "seal_blockers": [
            "no independently authorized execution runtime receipt exists",
            "thread and deterministic-kernel policy has not been authority-verified for model execution",
            "missing loaders, trainers, scorers, writers, adapters, active parameter counts, and checkpoint schemas remain unresolved",
        ],
    }


def _raw_registry_documents(snapshot: SourceSnapshot) -> dict[str, dict[str, object]]:
    return {
        CELL_FILENAME: _build_cell_registry(),
        FOLD_FILENAME: _build_fold_registry(
            snapshot.stations,
            snapshot.huc2_by_station,
            snapshot.reportable_stations,
        ),
        MODEL_FILENAME: _build_model_registry(),
        INPUT_FILENAME: _build_input_registry(snapshot),
        CONTRAST_FILENAME: _build_contrast_registry(),
        ENVIRONMENT_FILENAME: _build_environment_registry(snapshot),
    }


def _record_counts(name: str, document: Mapping[str, object]) -> dict[str, int]:
    if name == CELL_FILENAME:
        return {
            "logical_cells": len(document["logical_cells"]),  # type: ignore[arg-type]
            "variants": len(document["variants"]),  # type: ignore[arg-type]
            "fits": len(document["fits"]),  # type: ignore[arg-type]
        }
    if name == FOLD_FILENAME:
        return {
            "stations": len(document["stations"]),  # type: ignore[arg-type]
            "temporal_partitions": len(document["temporal_partitions"]),  # type: ignore[arg-type]
            "folds": len(document["folds"]),  # type: ignore[arg-type]
        }
    if name == MODEL_FILENAME:
        return {"models": len(document["models"])}  # type: ignore[arg-type]
    if name == INPUT_FILENAME:
        return {"inputs": len(document["inputs"])}  # type: ignore[arg-type]
    if name == CONTRAST_FILENAME:
        return {"contrasts": len(document["contrasts"])}  # type: ignore[arg-type]
    if name == ENVIRONMENT_FILENAME:
        return {"environments": len(document["environments"])}  # type: ignore[arg-type]
    raise SemanticContractError(f"unknown registry for record count: {name}")


def _record_field_schema(name: str, document: Mapping[str, object]) -> dict[str, list[str]]:
    collection_names = {
        CELL_FILENAME: ("logical_cells", "variants", "fits"),
        FOLD_FILENAME: ("stations", "temporal_partitions", "folds"),
        MODEL_FILENAME: ("models",),
        INPUT_FILENAME: ("inputs",),
        CONTRAST_FILENAME: ("contrasts",),
        ENVIRONMENT_FILENAME: ("environments",),
    }[name]
    result: dict[str, list[str]] = {}
    for collection_name in collection_names:
        collection = document[collection_name]
        _require(
            isinstance(collection, list) and collection,
            f"{name} {collection_name} is empty",
        )
        fields = sorted(collection[0])
        _require(
            all(type(record) is dict and sorted(record) == fields for record in collection),
            f"{name} {collection_name} record fields are not closed and uniform",
        )
        result[collection_name] = fields
    return result


def _id_digest(values: Sequence[str]) -> str:
    return _canonical_sha256(sorted(values))


def _cross_registry_receipt(documents: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    cell = documents[CELL_FILENAME]
    fold = documents[FOLD_FILENAME]
    model = documents[MODEL_FILENAME]
    inputs = documents[INPUT_FILENAME]
    contrasts = documents[CONTRAST_FILENAME]
    environments = documents[ENVIRONMENT_FILENAME]
    logical_ids = [str(record["logical_cell_id"]) for record in cell["logical_cells"]]  # type: ignore[index]
    variant_ids = [str(record["variant_id"]) for record in cell["variants"]]  # type: ignore[index]
    fit_ids = [str(record["fit_id"]) for record in cell["fits"]]  # type: ignore[index]
    fold_ids = [str(record["fold_id"]) for record in fold["folds"]]  # type: ignore[index]
    temporal_partition_ids = [
        str(record["partition_id"]) for record in fold["temporal_partitions"]  # type: ignore[index]
    ]
    model_ids = [str(record["model_id"]) for record in model["models"]]  # type: ignore[index]
    input_ids = [str(record["input_id"]) for record in inputs["inputs"]]  # type: ignore[index]
    contrast_ids = [str(record["contrast_id"]) for record in contrasts["contrasts"]]  # type: ignore[index]
    environment_ids = [str(record["environment_id"]) for record in environments["environments"]]  # type: ignore[index]
    referenced_folds = sorted(
        {
            str(record["spatial_fold_id"])
            for record in cell["fits"]  # type: ignore[index]
            if record["spatial_fold_id"] is not None
        }
    )
    referenced_temporal_partitions = sorted(
        {
            str(record["temporal_partition_id"])
            for record in cell["fits"]  # type: ignore[index]
            if record["temporal_partition_id"] is not None
        }
    )
    referenced_models = sorted({str(record["model_id"]) for record in cell["fits"]})  # type: ignore[index]
    referenced_inputs = sorted({str(record["input_id"]) for record in cell["fits"]})  # type: ignore[index]
    referenced_environments = sorted(
        {str(record["environment_id"]) for record in cell["fits"]}  # type: ignore[index]
    )
    return {
        "purpose": "REFERENTIAL_AND_BYTE_INTEGRITY_PLUMBING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "logical_cell_count": len(logical_ids),
        "variant_count": len(variant_ids),
        "fit_count": len(fit_ids),
        "fold_count": len(fold_ids),
        "temporal_partition_count": len(temporal_partition_ids),
        "model_count": len(model_ids),
        "input_count": len(input_ids),
        "contrast_count": len(contrast_ids),
        "environment_count": len(environment_ids),
        "logical_cell_ids_sha256": _id_digest(logical_ids),
        "variant_ids_sha256": _id_digest(variant_ids),
        "fit_ids_sha256": _id_digest(fit_ids),
        "fold_ids_sha256": _id_digest(fold_ids),
        "temporal_partition_ids_sha256": _id_digest(temporal_partition_ids),
        "model_ids_sha256": _id_digest(model_ids),
        "input_ids_sha256": _id_digest(input_ids),
        "contrast_ids_sha256": _id_digest(contrast_ids),
        "environment_ids_sha256": _id_digest(environment_ids),
        "referenced_fold_ids": referenced_folds,
        "referenced_temporal_partition_ids": referenced_temporal_partitions,
        "referenced_model_ids": referenced_models,
        "referenced_input_ids": referenced_inputs,
        "referenced_environment_ids": referenced_environments,
        "all_fit_references_closed": True,
        "historical_serialization_environment_is_not_a_fit_environment": True,
    }


def _manifest_document(
    documents: Mapping[str, Mapping[str, object]],
    payloads: Mapping[str, bytes],
    snapshot: SourceSnapshot,
) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for name in REGISTRY_FILENAMES:
        document = documents[name]
        payload = payloads[name]
        outputs[name] = {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "schema_id": document["schema_id"],
            "top_level_fields": sorted(document),
            "record_fields": _record_field_schema(name, document),
            "record_counts": _record_counts(name, document),
        }
    source_roles = (
        "semantic_data_manifest",
        "semantic_daily_registry",
        "semantic_training_registry",
        "station_registry",
        "key_authority_manifest",
        "primary_key_registry",
        "defect_authority_manifest",
        "defect_authority_report",
    )
    return {
        **_registry_header(
            "thermoroute-semantic-contract-registry-manifest-v4-candidate",
            "thermoroute.semantic-contract.registry-manifest.v4-candidate.1",
        ),
        "artifact_id": ARTIFACT_ID,
        "scope": {
            "description": "current registered semantic-contract subset plus explicitly planned logical cells only",
            "complete_v4_matrix": False,
            "partial_matrix_claimed_complete": False,
            "logical_cells": EXPECTED_LOGICAL_CELLS,
            "variants": 62,
            "fits": 5_332,
            "fit_arithmetic": "12 v5 + 5280 neural + 40 L2_U2 = 5332",
            "protocol_primary_cells": EXPECTED_PROTOCOL_PRIMARY_CELLS,
            "protocol_primary_state_counts": {
                "PLANNED": 48,
                "REGISTERED": 24,
                "EXECUTED": 0,
                "WITHDRAWN": 0,
            },
            "withdrawn_historical_cells_forensic_only": 432,
        },
        "source_inputs": {role: dict(snapshot.bindings[role]) for role in source_roles},
        "outputs": outputs,
        "cross_registry_receipt": _cross_registry_receipt(documents),
        "integrity_scope": {
            "sha256_role": "byte identity and transport integrity plumbing only",
            "sha256_is_scientific_evidence": False,
            "sha256_is_quality_evidence": False,
            "protocol_hashes_bound": False,
            "runner_hashes_bound": False,
            "future_checkpoint_prediction_score_or_effect_hashes_bound": False,
            "manifest_self_hash_bound": False,
            "future_result_hashes_bound": False,
        },
        "evidence_boundary": {
            "score_free": True,
            "model_score_checkpoint_prediction_effect_or_result_inputs_read": False,
            "registered_means_executed": False,
            "executed_protocol_cells": 0,
            "candidate_can_promote_planned_or_registered_to_executed": False,
        },
        "publication_contract": {
            "caller_must_specify_destination": True,
            "scratch_candidate_only": True,
            "outputs_final_destination_prohibited": True,
            "exact_file_count": 7,
            "create_only": True,
            "overwrite_supported": False,
            "resume_supported": False,
            "independent_build_passes_before_write": 2,
            "canonical_json_required": True,
            "strict_duplicate_key_rejection": True,
            "anchored_parent_directory": True,
            "hardlinks_and_symlinks_rejected": True,
            "atomic_linux_RENAME_NOREPLACE": True,
            "staged_bytes_reverified_before_rename": True,
        },
        "limitations": [
            "candidate source data and structural registries cannot authorize execution",
            "24 LightGBM F0/F3 spatial logical cells are PLANNED without variants or fits",
            "F2b and the withdrawn historical 432-cell ladder are outside this inventory",
            "future placebos, components, L3, hybrid, cohort, novelty, and audit scopes are absent",
            "missing active parameter counts, adapters, checkpoint schemas, production loaders/trainers/scorers/writers, and an authorized runtime block execution",
            "L2_U2 versus L2 remains blocked until a corrected same-authority L2 comparator exists",
        ],
    }


def _validate_common(document: Mapping[str, object], *, label: str) -> None:
    _require(document.get("status") == STATUS, f"{label} status must be {STATUS}")
    _require(document.get("formal_authority") is False, f"{label} formal_authority must be false")
    _require(
        document.get("execution_authorized") is False,
        f"{label} execution_authorized must be false boolean",
    )
    _require(
        document.get("candidate_source_inputs_can_authorize_execution") is False,
        f"{label} candidate source authorization boundary changed",
    )
    _require(
        document.get("candidate_trust_boundary") == _candidate_trust_boundary(),
        f"{label} candidate trust boundary changed",
    )
    _require(
        type(document.get("schema_version")) is int and document["schema_version"] == 1,
        f"{label} schema version changed",
    )
    _require(type(document.get("schema_id")) is str, f"{label} schema id is invalid")
    _require(type(document.get("registry_id")) is str, f"{label} registry id is invalid")


def _validate_cell_registry(document: Mapping[str, object]) -> None:
    expected = _build_cell_registry()
    _require(
        document == expected, "cell registry differs from the exact closed registered inventory"
    )
    logical = document["logical_cells"]
    variants = document["variants"]
    fits = document["fits"]
    assert isinstance(logical, list) and isinstance(variants, list) and isinstance(fits, list)
    _require(
        (len(logical), len(variants), len(fits)) == (EXPECTED_LOGICAL_CELLS, 62, 5_332),
        "cell arithmetic is not 74/62/5332",
    )
    _require(
        document["registered_scope"]["complete_v4_matrix"] is False,
        "partial matrix was claimed complete",
    )  # type: ignore[index]
    receipts = document["independent_phase1_plan_receipts"]
    _require(
        receipts["neural"]["observed_sha256"] == EXPECTED_NEURAL_PHASE1_PLAN_SHA256,
        "neural Phase-1 plan receipt changed",
    )  # type: ignore[index]
    _require(
        receipts["l2_u2"]["observed_sha256"] == EXPECTED_L2_U2_PHASE1_PLAN_SHA256,
        "L2_U2 Phase-1 plan receipt changed",
    )  # type: ignore[index]
    status_map = document["protocol_declared_cell_status"]
    primary = status_map["primary_matrix"]  # type: ignore[index]
    _require(
        primary["state_counts"]
        == {"PLANNED": 48, "REGISTERED": 24, "EXECUTED": 0, "WITHDRAWN": 0},
        "protocol primary cell-state arithmetic changed",
    )
    _require(
        len(primary["cells"]) == EXPECTED_PROTOCOL_PRIMARY_CELLS,
        "protocol primary cell-state map is not exact 72",
    )
    _require(
        all(
            record["protocol_state"] in {"PLANNED", "REGISTERED", "WITHDRAWN"}
            and record["execution_receipt_bound"] is False
            and record["score_or_result_receipt_bound"] is False
            for record in primary["cells"]
        ),
        "candidate cell-state map falsely claims execution or evidence",
    )
    _require(
        status_map["withdrawn_forensic_scope"]["protocol_state"] == "WITHDRAWN"
        and status_map["withdrawn_forensic_scope"]["included_in_primary_72"] is False,
        "withdrawn 432-cell scope leaked into the primary matrix",
    )
    planned_lightgbm = [
        record
        for record in logical
        if record["family"] == "primary_lightgbm_f0_f3_planned"
    ]
    fit_logical_ids = {record["logical_cell_id"] for record in fits}
    _require(
        len(planned_lightgbm) == 24
        and all(record["protocol_state"] == "PLANNED" for record in planned_lightgbm)
        and not ({record["logical_cell_id"] for record in planned_lightgbm} & fit_logical_ids),
        "planned LightGBM logical cells acquired variants, fits, or execution semantics",
    )


def _snapshot_for_embedded_documents(
    *,
    bindings: Mapping[str, Mapping[str, object]],
    stations: Sequence[str] = (),
    huc2: Mapping[str, str] | None = None,
    reportable: Sequence[str] = (),
    semantic_outputs: Mapping[str, Mapping[str, object]] | None = None,
) -> SourceSnapshot:
    return SourceSnapshot(
        bindings=bindings,
        stations=tuple(stations),
        huc2_by_station={} if huc2 is None else huc2,
        reportable_stations=tuple(reportable),
        semantic_outputs={} if semantic_outputs is None else semantic_outputs,
        signatures={},
    )


def _validate_fold_registry(document: Mapping[str, object]) -> None:
    stations_raw = _list(document.get("stations"), label="fold registry stations")
    stations: list[str] = []
    huc: dict[str, str] = {}
    reportable: list[str] = []
    for index, raw in enumerate(stations_raw):
        record = _keys(
            raw,
            {"site_id", "huc2", "in_training_universe", "in_reportable_scoring_universe"},
            label=f"fold station {index}",
        )
        site = _canonical_site(record["site_id"])
        huc2 = _canonical_huc2(record["huc2"])
        _require(
            record["in_training_universe"] is True, "fold station training-universe flag changed"
        )
        _require(
            type(record["in_reportable_scoring_universe"]) is bool,
            "fold station reportable flag must be boolean",
        )
        stations.append(site)
        huc[site] = huc2
        if record["in_reportable_scoring_universe"] is True:
            reportable.append(site)
    _require(
        tuple(stations) == tuple(sorted(stations)) and len(set(stations)) == 120,
        "fold training universe is not exact sorted 120",
    )
    _require(
        tuple(reportable) == tuple(sorted(reportable)) and len(reportable) == 116,
        "fold reportable universe is not exact sorted 116",
    )
    _require(
        _canonical_sha256([[site, huc[site]] for site in stations]) == EXPECTED_STATION_HUC2_SHA256,
        "fold station/HUC2 mapping digest changed",
    )
    _require(
        _canonical_sha256(reportable) == EXPECTED_REPORTABLE_STATIONS_SHA256,
        "fold reportable station digest changed",
    )
    expected = _build_fold_registry(stations, huc, reportable)
    _require(document == expected, "fold registry differs from exact PCG64/HUC2 memberships")
    temporal_partitions = document["temporal_partitions"]
    spatial_folds = document["folds"]
    _require(
        len(temporal_partitions) == 1 and len(spatial_folds) == 44,
        "temporal partition and spatial fold inventories are not 1/44",
    )
    _require(
        temporal_partitions[0]["spatial_cv_fold_applicable"] is False
        and temporal_partitions[0]["split_seed"] is None,
        "known-site temporal partition was misrepresented as a seeded CV fold",
    )
    _require(
        all(
            (record["geometry"] == "random_site" and type(record["split_seed"]) is int)
            or (record["geometry"] == "whole_region" and record["split_seed"] is None)
            for record in spatial_folds
        ),
        "spatial split-seed semantics do not match geometry",
    )
    region = document["folds"][-4:]  # type: ignore[index]
    _require(
        [record["scoreable_held_station_count"] for record in region] == [30, 30, 29, 27],
        "whole-region scoreable counts changed",
    )


def _validate_binding_map(
    bindings: object, *, roles: set[str], label: str
) -> dict[str, Mapping[str, object]]:
    raw = _keys(bindings, roles, label=label)
    normalized: dict[str, Mapping[str, object]] = {}
    for role in sorted(roles):
        normalized[role] = _binding_fields(raw[role], label=f"{label} {role}")
    return normalized


def _validate_input_registry(document: Mapping[str, object]) -> None:
    source_roles = {
        "semantic_data_manifest",
        "semantic_daily_registry",
        "semantic_training_registry",
        "station_registry",
        "key_authority_manifest",
        "primary_key_registry",
        "defect_authority_manifest",
        "defect_authority_report",
    }
    bindings = _validate_binding_map(
        document.get("source_bindings"), roles=source_roles, label="input source bindings"
    )
    for role in (
        "station_registry",
        "key_authority_manifest",
        "primary_key_registry",
        "defect_authority_manifest",
        "defect_authority_report",
    ):
        _require(
            bindings[role]["sha256"] == PINNED_SHA256[role]
            and bindings[role]["size_bytes"] == PINNED_SIZES[role],
            f"input pinned binding changed: {role}",
        )
    schemas = _keys(
        document.get("source_schemas"),
        {SEMANTIC_DAILY_FILENAME, SEMANTIC_TRAINING_FILENAME, PRIMARY_KEY_REGISTRY.name},
        label="input source schemas",
    )
    semantic_outputs = {
        SEMANTIC_DAILY_FILENAME: schemas[SEMANTIC_DAILY_FILENAME],
        SEMANTIC_TRAINING_FILENAME: schemas[SEMANTIC_TRAINING_FILENAME],
    }
    snapshot = _snapshot_for_embedded_documents(
        bindings=bindings,
        semantic_outputs=semantic_outputs,
    )
    expected = _build_input_registry(snapshot)
    _require(
        document == expected, "input registry differs from exact closed channel/source contract"
    )
    input_records = document["inputs"]
    assert isinstance(input_records, list)
    _require(
        len(input_records) == 11, "input registry must contain exactly 11 input configurations"
    )
    l2_u2 = next(record for record in input_records if record["local_level"] == "L2_U2")
    forbidden = set(l2_u2["forbidden_history_channels"])
    _require(
        {"WTEMP_values", "WTEMP_observedness", "FLOW_values", "FLOW_observedness"} <= forbidden,
        "L2_U2 leaks local thermal or flow channels",
    )


def _validate_environment_registry(document: Mapping[str, object]) -> None:
    authority = _validate_binding_map(
        document.get("authority_serialization_bindings"),
        roles={"key_authority_manifest", "defect_authority_manifest"},
        label="environment authority bindings",
    )
    locks = _validate_binding_map(
        document.get("repository_lock_bindings"),
        roles={"pyproject", "requirements_lock", "requirements_lock_py312"},
        label="environment lock bindings",
    )
    bindings = {**authority, **locks}
    for role in bindings:
        _require(
            bindings[role]["sha256"] == PINNED_SHA256[role]
            and bindings[role]["size_bytes"] == PINNED_SIZES[role],
            f"environment pinned binding changed: {role}",
        )
    expected = _build_environment_registry(_snapshot_for_embedded_documents(bindings=bindings))
    _require(
        document == expected, "environment registry differs from exact historical/route-a contract"
    )
    _require(
        document["runtime_boundary"]["authorized_execution_environment_id"] is None,
        "an execution runtime was incorrectly authorized",
    )  # type: ignore[index]
    route_a = next(
        record
        for record in document["environments"]  # type: ignore[index]
        if record["environment_id"] == "environment:route_a_candidate"
    )
    _require(
        route_a["runtime"] == dict(ROUTE_A_RUNTIME)
        and route_a["packages"] == dict(ROUTE_A_DIRECT_LOCK_PINS)
        and route_a["direct_lock_pin_set_complete"] is True,
        "route-a runtime or complete direct lock pins changed",
    )
    _require(
        route_a["may_execute_registered_fits"] is False
        and route_a["determinism_policy"]["authority_verified"] is False,
        "route-a candidate was incorrectly promoted to execution authority",
    )


def _validate_cross_registry(documents: Mapping[str, Mapping[str, object]]) -> None:
    cell = documents[CELL_FILENAME]
    logical_ids = {record["logical_cell_id"] for record in cell["logical_cells"]}  # type: ignore[index]
    variants = {record["variant_id"]: record for record in cell["variants"]}  # type: ignore[index]
    fold_ids = {record["fold_id"] for record in documents[FOLD_FILENAME]["folds"]}  # type: ignore[index]
    temporal_partition_ids = {
        record["partition_id"]
        for record in documents[FOLD_FILENAME]["temporal_partitions"]  # type: ignore[index]
    }
    model_ids = {record["model_id"] for record in documents[MODEL_FILENAME]["models"]}  # type: ignore[index]
    input_ids = {record["input_id"] for record in documents[INPUT_FILENAME]["inputs"]}  # type: ignore[index]
    environment_ids = {
        record["environment_id"]
        for record in documents[ENVIRONMENT_FILENAME]["environments"]  # type: ignore[index]
    }
    fit_ids: list[object] = []
    for fit in cell["fits"]:  # type: ignore[index]
        fit_ids.append(fit["fit_id"])
        _require(fit["logical_cell_id"] in logical_ids, "fit references missing logical cell")
        _require(fit["variant_id"] in variants, "fit references missing variant")
        _require(
            variants[fit["variant_id"]]["logical_cell_id"] == fit["logical_cell_id"],
            "fit variant/logical-cell relationship mismatches",
        )
        spatial_fold_id = fit["spatial_fold_id"]
        temporal_partition_id = fit["temporal_partition_id"]
        _require(
            (spatial_fold_id is None) != (temporal_partition_id is None),
            "fit must reference exactly one spatial fold or temporal partition",
        )
        if spatial_fold_id is not None:
            _require(spatial_fold_id in fold_ids, "fit references missing spatial fold")
            _require(fit["fold_index"] in FOLDS, "spatial fit fold index is invalid")
            if fit["geometry"] == "random_site":
                _require(
                    fit["split_seed"] in RANDOM_SPLIT_SEEDS,
                    "random-site fit split seed is invalid",
                )
            else:
                _require(
                    fit["geometry"] == "whole_region" and fit["split_seed"] is None,
                    "whole-region fit must have a null split seed",
                )
        else:
            _require(
                temporal_partition_id in temporal_partition_ids
                and fit["geometry"] == "known_site_temporal"
                and fit["split_seed"] is None
                and fit["fold_index"] is None,
                "temporal fit was misrepresented as a seeded spatial CV fold",
            )
        _require(fit["model_id"] in model_ids, "fit references missing model")
        _require(fit["input_id"] in input_ids, "fit references missing input")
        _require(fit["environment_id"] in environment_ids, "fit references missing environment")
    _require(
        len(fit_ids) == len(set(fit_ids)) == 5_332, "fit IDs are not 5,332 globally unique IDs"
    )
    _require(
        set(variants) == {fit["variant_id"] for fit in cell["fits"]},
        "variant inventory is not referentially closed",
    )  # type: ignore[index]
    _require(
        fold_ids
        == {
            fit["spatial_fold_id"]
            for fit in cell["fits"]
            if fit["spatial_fold_id"] is not None
        },
        "spatial fold inventory is not referentially closed",
    )  # type: ignore[index]
    _require(
        temporal_partition_ids
        == {
            fit["temporal_partition_id"]
            for fit in cell["fits"]
            if fit["temporal_partition_id"] is not None
        },
        "temporal partition inventory is not referentially closed",
    )  # type: ignore[index]
    _require(
        model_ids == {fit["model_id"] for fit in cell["fits"]},
        "model inventory is not referentially closed",
    )  # type: ignore[index]
    _require(
        input_ids == {fit["input_id"] for fit in cell["fits"]},
        "input inventory is not referentially closed",
    )  # type: ignore[index]
    _require(
        {fit["environment_id"] for fit in cell["fits"]} == {"environment:route_a_candidate"},
        "fits bind a non-candidate execution environment",
    )  # type: ignore[index]
    _require(
        documents[CONTRAST_FILENAME] == _build_contrast_registry(),
        "contrast selector/matching contract changed",
    )


def validate_registry_documents(
    documents: Mapping[str, Mapping[str, object]],
    *,
    expected_snapshot: SourceSnapshot | None = None,
) -> None:
    _require(set(documents) == set(REGISTRY_FILENAMES), "registry document set is not exact")
    expected_top_fields = {
        CELL_FILENAME: {
            "schema_version",
            "schema_id",
            "registry_id",
            "status",
            "formal_authority",
            "execution_authorized",
            "candidate_source_inputs_can_authorize_execution",
            "candidate_trust_boundary",
            "registered_scope",
            "arithmetic",
            "independent_phase1_plan_receipts",
            "protocol_declared_cell_status",
            "logical_cells",
            "variants",
            "fits",
        },
        FOLD_FILENAME: {
            "schema_version",
            "schema_id",
            "registry_id",
            "status",
            "formal_authority",
            "execution_authorized",
            "candidate_source_inputs_can_authorize_execution",
            "candidate_trust_boundary",
            "universe_contract",
            "algorithm_contract",
            "independent_phase1_fold_receipt",
            "stations",
            "temporal_partitions",
            "folds",
        },
        MODEL_FILENAME: {
            "schema_version",
            "schema_id",
            "registry_id",
            "status",
            "formal_authority",
            "execution_authorized",
            "candidate_source_inputs_can_authorize_execution",
            "candidate_trust_boundary",
            "global_policy",
            "models",
            "seal_blocker_summary",
        },
        INPUT_FILENAME: {
            "schema_version",
            "schema_id",
            "registry_id",
            "status",
            "formal_authority",
            "execution_authorized",
            "candidate_source_inputs_can_authorize_execution",
            "candidate_trust_boundary",
            "source_bindings",
            "source_authority_boundary",
            "source_schemas",
            "period_contract",
            "raw_label_predictor_separation",
            "inputs",
            "excluded_input_scopes",
        },
        CONTRAST_FILENAME: {
            "schema_version",
            "schema_id",
            "registry_id",
            "status",
            "formal_authority",
            "execution_authorized",
            "candidate_source_inputs_can_authorize_execution",
            "candidate_trust_boundary",
            "geometry_specific_aggregation_contracts",
            "station_first_estimator",
            "uncertainty_and_sensitivity",
            "recovery_denominator_guard",
            "contrasts",
            "prohibited_comparisons",
        },
        ENVIRONMENT_FILENAME: {
            "schema_version",
            "schema_id",
            "registry_id",
            "status",
            "formal_authority",
            "execution_authorized",
            "candidate_source_inputs_can_authorize_execution",
            "candidate_trust_boundary",
            "authority_serialization_bindings",
            "repository_lock_bindings",
            "runtime_boundary",
            "environments",
            "seal_blockers",
        },
    }
    for name in REGISTRY_FILENAMES:
        document = documents[name]
        _keys(document, expected_top_fields[name], label=name)
        _validate_common(document, label=name)
    _validate_cell_registry(documents[CELL_FILENAME])
    _validate_fold_registry(documents[FOLD_FILENAME])
    _require(
        documents[MODEL_FILENAME] == _build_model_registry(),
        "model registry differs from exact closed configurations/blockers",
    )
    _validate_input_registry(documents[INPUT_FILENAME])
    _require(
        documents[CONTRAST_FILENAME] == _build_contrast_registry(),
        "contrast registry differs from exact matched estimator contract",
    )
    _validate_environment_registry(documents[ENVIRONMENT_FILENAME])
    _validate_cross_registry(documents)
    if expected_snapshot is not None:
        expected = _raw_registry_documents(expected_snapshot)
        _require(documents == expected, "registry documents differ from captured source snapshot")


_BUNDLE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CandidateBundle:
    files: Mapping[str, bytes]
    documents: Mapping[str, Mapping[str, object]]
    snapshot: SourceSnapshot
    _token: object

    def __post_init__(self) -> None:
        _require(self._token is _BUNDLE_TOKEN, "candidate bundle issuer token changed")

    @property
    def manifest(self) -> dict[str, Any]:
        return _strict_json(self.files[MANIFEST_FILENAME], label="candidate manifest")


def _verify_manifest(
    manifest: Mapping[str, object],
    documents: Mapping[str, Mapping[str, object]],
    payloads: Mapping[str, bytes],
    snapshot: SourceSnapshot,
) -> None:
    expected = _manifest_document(documents, payloads, snapshot)
    _require(
        manifest == expected,
        "candidate manifest differs from exact byte/count/cross-registry receipt",
    )
    _validate_common(manifest, label="candidate manifest")
    _require(
        manifest["scope"]["complete_v4_matrix"] is False, "manifest claims a complete v4 matrix"
    )  # type: ignore[index]
    _require(manifest["scope"]["fits"] == 5_332, "manifest fit count changed")  # type: ignore[index]
    _require(
        manifest["cross_registry_receipt"]["all_fit_references_closed"] is True,
        "manifest reference closure receipt changed",
    )  # type: ignore[index]


def _verify_bundle(bundle: CandidateBundle) -> None:
    _require(
        type(bundle) is CandidateBundle and bundle._token is _BUNDLE_TOKEN,
        "candidate bundle was forged",
    )
    _require(set(bundle.files) == set(ALL_FILENAMES), "candidate bundle file set changed")
    parsed: dict[str, Mapping[str, object]] = {}
    for name in REGISTRY_FILENAMES:
        parsed[name] = _strict_json(bundle.files[name], label=name)
    validate_registry_documents(parsed, expected_snapshot=bundle.snapshot)
    _require(parsed == bundle.documents, "serialized and in-memory registry documents differ")
    manifest = _strict_json(bundle.files[MANIFEST_FILENAME], label=MANIFEST_FILENAME)
    _verify_manifest(
        manifest,
        parsed,
        {name: bundle.files[name] for name in REGISTRY_FILENAMES},
        bundle.snapshot,
    )


def _build_from_snapshot(snapshot: SourceSnapshot) -> CandidateBundle:
    documents = _raw_registry_documents(snapshot)
    validate_registry_documents(documents, expected_snapshot=snapshot)
    payloads = {name: _canonical_json_bytes(documents[name]) for name in REGISTRY_FILENAMES}
    manifest = _manifest_document(documents, payloads, snapshot)
    files = {**payloads, MANIFEST_FILENAME: _canonical_json_bytes(manifest)}
    bundle = CandidateBundle(
        files=MappingProxyType(files),
        documents=MappingProxyType(documents),
        snapshot=snapshot,
        _token=_BUNDLE_TOKEN,
    )
    _verify_bundle(bundle)
    return bundle


def build_twice(paths: RegistryInputPaths, *, config: BuildConfig) -> CandidateBundle:
    first_snapshot = _capture_sources(paths, config)
    first = _build_from_snapshot(first_snapshot)
    second_snapshot = _capture_sources(paths, config)
    second = _build_from_snapshot(second_snapshot)
    _require(first.files == second.files, "independent registry build bytes differ")
    _require(
        first_snapshot.bindings == second_snapshot.bindings,
        "source bindings changed between builds",
    )
    _require(
        first_snapshot.signatures == second_snapshot.signatures,
        "source file identities changed between builds",
    )
    return first


def _simple_name(name: str, *, label: str) -> str:
    _require(
        type(name) is str
        and name not in {"", ".", ".."}
        and Path(name).name == name
        and "/" not in name
        and "\\" not in name,
        f"{label} must be one basename",
    )
    return name


def _entry_at(parent_descriptor: int, name: str) -> os.stat_result | None:
    _simple_name(name, label="directory entry")
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _open_anchored_directory(path: Path, *, label: str) -> tuple[int, tuple[int, int]]:
    absolute = _require_no_symlink_components(path, label=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise SemanticContractError(f"cannot anchor {label}: {absolute}") from exc
    status = os.fstat(descriptor)
    _require(stat.S_ISDIR(status.st_mode), f"{label} is not a directory")
    return descriptor, _directory_identity(status)


def _verify_anchored_directory(
    path: Path, descriptor: int, identity: tuple[int, int], *, label: str
) -> None:
    opened = os.fstat(descriptor)
    lexical = _absolute(path).lstat()
    _require(
        stat.S_ISDIR(opened.st_mode)
        and stat.S_ISDIR(lexical.st_mode)
        and not stat.S_ISLNK(lexical.st_mode)
        and _directory_identity(opened) == identity
        and _directory_identity(lexical) == identity,
        f"{label} inode changed",
    )
    _require_no_symlink_components(path, label=label)


def _write_at(parent_descriptor: int, name: str, payload: bytes) -> tuple[int, int]:
    _simple_name(name, label="candidate filename")
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
            _require(written > 0, f"short write for {name}")
            offset += written
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        _require(
            stat.S_ISREG(status.st_mode)
            and status.st_nlink == 1
            and status.st_size == len(payload),
            f"staged {name} type/link/size changed",
        )
        return _directory_identity(status)
    finally:
        os.close(descriptor)


def _read_at(parent_descriptor: int, name: str, *, label: str) -> bytes:
    _simple_name(name, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise SemanticContractError(f"cannot safely open {label}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(before.st_nlink == 1, f"{label} must have exactly one hard link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    entry = _entry_at(parent_descriptor, name)
    _require(
        _stat_signature(before) == _stat_signature(after)
        and entry is not None
        and _directory_identity(entry) == _directory_identity(before)
        and entry.st_nlink == 1
        and len(payload) == before.st_size,
        f"{label} changed during verification",
    )
    return payload


def _remove_tree_at(parent_descriptor: int, name: str) -> None:
    status = _entry_at(parent_descriptor, name)
    if status is None:
        return
    if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        child = os.open(name, flags, dir_fd=parent_descriptor)
        try:
            for entry in os.listdir(child):
                _remove_tree_at(child, entry)
            os.fsync(child)
        finally:
            os.close(child)
        os.rmdir(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _remove_owned_at(parent_descriptor: int, name: str, identity: tuple[int, int]) -> None:
    status = _entry_at(parent_descriptor, name)
    if status is not None and _directory_identity(status) == identity:
        _remove_tree_at(parent_descriptor, name)


def _rename_noreplace(parent_descriptor: int, source: str, destination: str) -> None:
    _simple_name(source, label="rename source")
    _simple_name(destination, label="rename destination")
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
        parent_descriptor,
        os.fsencode(source),
        parent_descriptor,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise SemanticContractError(
                f"refusing to overwrite existing candidate destination: {destination}"
            )
        raise SemanticContractError(f"atomic RENAME_NOREPLACE failed: {os.strerror(error)}")


def _validate_destination(destination: Path) -> tuple[Path, Path]:
    absolute = _absolute(destination)
    _simple_name(absolute.name, label="candidate destination")
    parent = _require_no_symlink_components(absolute.parent, label="candidate output parent")
    final_root = _absolute(FINAL_OUTPUT_ROOT)
    try:
        absolute.relative_to(final_root)
    except ValueError:
        pass
    else:
        raise SemanticContractError("candidate output is prohibited under outputs/final")
    _require(not os.path.lexists(absolute), f"candidate destination already exists: {absolute}")
    return absolute, parent


def _revalidate_sources(
    paths: RegistryInputPaths, config: BuildConfig, snapshot: SourceSnapshot
) -> None:
    current = _capture_sources(paths, config)
    _require(
        current.bindings == snapshot.bindings and current.signatures == snapshot.signatures,
        "candidate source inputs changed before commit",
    )


def write_candidate(
    paths: RegistryInputPaths,
    destination: Path,
    *,
    config: BuildConfig,
) -> tuple[Path, CandidateBundle]:
    """Build internally and publish one create-only scratch candidate bundle."""

    destination, parent = _validate_destination(destination)
    parent_descriptor, parent_identity = _open_anchored_directory(
        parent, label="candidate output parent"
    )
    destination_name = destination.name
    lock_name = f".{destination_name}.semantic-contract-create.lock"
    lock_identity: tuple[int, int] | None = None
    stage_name: str | None = None
    stage_descriptor: int | None = None
    stage_identity: tuple[int, int] | None = None
    bundle: CandidateBundle | None = None
    committed = False
    try:
        _require(
            _entry_at(parent_descriptor, destination_name) is None,
            "candidate destination appeared before build",
        )
        lock_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_descriptor = os.open(lock_name, lock_flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise SemanticContractError(
                f"candidate create lock already exists: {lock_name}"
            ) from exc
        try:
            lock_status = os.fstat(lock_descriptor)
            lock_identity = _directory_identity(lock_status)
            lock_payload = f"pid={os.getpid()}\nstatus={STATUS}\n".encode("ascii")
            _require(
                os.write(lock_descriptor, lock_payload) == len(lock_payload),
                "candidate lock write was short",
            )
            os.fsync(lock_descriptor)
        finally:
            os.close(lock_descriptor)
        os.fsync(parent_descriptor)

        bundle = build_twice(paths, config=config)
        for _attempt in range(128):
            proposed = f".{destination_name}.candidate-stage.{secrets.token_hex(12)}"
            try:
                os.mkdir(proposed, 0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            stage_name = proposed
            break
        _require(stage_name is not None, "cannot allocate exclusive staging directory")
        stage_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        stage_descriptor = os.open(stage_name, stage_flags, dir_fd=parent_descriptor)
        stage_identity = _directory_identity(os.fstat(stage_descriptor))
        for name in sorted(bundle.files):
            _write_at(stage_descriptor, name, bundle.files[name])
        os.fsync(stage_descriptor)
        _verify_bundle(bundle)
        _require(
            set(os.listdir(stage_descriptor)) == set(ALL_FILENAMES),
            "staged candidate file set changed",
        )
        staged_payloads = {
            name: _read_at(stage_descriptor, name, label=f"staged {name}") for name in ALL_FILENAMES
        }
        _require(staged_payloads == bundle.files, "staged candidate bytes changed")
        _revalidate_sources(paths, config, bundle.snapshot)
        _verify_anchored_directory(
            parent,
            parent_descriptor,
            parent_identity,
            label="candidate output parent before rename",
        )
        _require(
            _entry_at(parent_descriptor, destination_name) is None,
            "candidate destination appeared during build",
        )
        _rename_noreplace(parent_descriptor, stage_name, destination_name)
        destination_status = _entry_at(parent_descriptor, destination_name)
        _require(
            destination_status is not None
            and _directory_identity(destination_status) == stage_identity,
            "committed candidate inode differs from stage",
        )
        _verify_anchored_directory(
            parent, parent_descriptor, parent_identity, label="candidate output parent after rename"
        )
        _require(
            {
                _simple_name(name, label="committed candidate entry")
                for name in os.listdir(stage_descriptor)
            }
            == set(ALL_FILENAMES),
            "committed candidate file set changed",
        )
        for name in ALL_FILENAMES:
            _require(
                _read_at(stage_descriptor, name, label=f"committed {name}") == bundle.files[name],
                f"committed candidate bytes changed: {name}",
            )
        if lock_identity is not None:
            _remove_owned_at(parent_descriptor, lock_name, lock_identity)
            lock_identity = None
        os.fsync(parent_descriptor)
        committed = True
    finally:
        if stage_descriptor is not None:
            try:
                os.close(stage_descriptor)
            except OSError:
                pass
        if not committed and stage_identity is not None:
            if stage_name is not None:
                _remove_owned_at(parent_descriptor, stage_name, stage_identity)
            _remove_owned_at(parent_descriptor, destination_name, stage_identity)
        if lock_identity is not None:
            _remove_owned_at(parent_descriptor, lock_name, lock_identity)
        if not committed:
            try:
                os.fsync(parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)
    _require(bundle is not None, "candidate write completed without a bundle")
    return destination, bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic-data-dir",
        required=True,
        type=Path,
        help="read-only scratch semantic-data candidate directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new scratch destination outside outputs/final",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination, bundle = write_candidate(
        RegistryInputPaths.production(args.semantic_data_dir),
        args.output_dir,
        config=BuildConfig(enforce_route_a_runtime=True),
    )
    result = {
        "status": STATUS,
        "execution_authorized": False,
        "output_dir": str(destination),
        "logical_cells": EXPECTED_LOGICAL_CELLS,
        "variants": 62,
        "fits": 5_332,
        "files": {
            name: {
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(bundle.files.items())
        },
        "limitations": bundle.manifest["limitations"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SemanticContractError as exc:
        print(f"semantic-contract candidate build failed closed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
