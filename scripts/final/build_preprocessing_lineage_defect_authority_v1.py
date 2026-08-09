#!/usr/bin/env python3
"""Build the create-only preprocessing-lineage defect authority, version 1.

This program does not read prediction shards, score tables, or unrun extension
outcomes.  It binds the already-existing point/inference manifests as opaque
bytes and independently reconstructs, from the two raw panel Parquets, the
training-label/admissibility and evaluation-missingness inventories needed to
withdraw their learned-model descendants.

The default invocation is a read-only, two-pass production dry run::

    python scripts/final/build_preprocessing_lineage_defect_authority_v1.py

Publication is an explicit, create-only operation.  It uses two independent
build passes, byte comparison, staged fsyncs, and Linux ``RENAME_NOREPLACE``.
There is deliberately no overwrite or resume mode.
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
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"

STATUS = "SCIENTIFIC_RESULTS_WITHDRAWN_PENDING_OBSERVED_LINEAGE_RERUN"
ARTIFACT_ID = "thermoroute-preprocessing-lineage-defect-authority-v1"
REPORT_FILENAME = "preprocessing_lineage_defect_report_v1.json"
MANIFEST_FILENAME = "preprocessing_lineage_defect_authority_v1_manifest.json"
DEFAULT_OUTPUT_NAME = "preprocessing_lineage_defect_authority_v1"

LEADS = (1, 3, 7)
USED_VARIABLES = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
MET_VARIABLES = ("TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
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
FORMAL_REPORTABLE_KEY_ARROW_SCHEMA = (
    ("key_id", "string", False),
    ("site_id", "string", False),
    ("issue_date", "timestamp[ns]", False),
    ("target_date", "timestamp[ns]", False),
    ("lead_days", "int16", False),
    ("y_true", "double", False),
    ("issue_wtemp_observed", "bool", False),
    ("days_since_last_observed_wtemp", "int32", False),
    ("n_observed_wtemp_7d", "int16", False),
    ("fraction_observed_wtemp_7d", "double", False),
    ("n_observed_wtemp_14d", "int16", False),
    ("fraction_observed_wtemp_14d", "double", False),
    ("n_observed_wtemp_32d", "int16", False),
    ("fraction_observed_wtemp_32d", "double", False),
    ("history_H100", "bool", False),
    ("history_H75", "bool", False),
    ("history_Hall", "bool", False),
    ("reportable_primary", "bool", False),
)
SHORT_LAGS = (0, 1, 2, 3, 5, 7, 10, 14)
ROLLING_WINDOWS = (3, 7, 14, 30)
DELTA_LAGS = (1, 3)
Y_ABSOLUTE_TOLERANCE_C = 2e-6


class LineageAuthorityError(RuntimeError):
    """A fail-closed lineage or publication contract was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LineageAuthorityError(message)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    frozen = _deep_freeze(value)
    _require(type(frozen) is type(MappingProxyType({})), "mapping freeze failed")
    return frozen


PRODUCTION_CANONICAL_FILES: Mapping[str, tuple[str, str]] = _freeze_mapping(
    {
        "station_registry": (
            "data_usgs/station_registry_v1.csv",
            "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9",
        ),
        "training_panel": (
            "data_usgs/panel_usgs_120v2.parquet",
            "0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69",
        ),
        "evaluation_panel": (
            "outputs/conventional/panel_2021_2023.parquet",
            "cecdac459139456202240954e4c98fe18bba1fe8b63e9b06ab268683e0d1c03c",
        ),
        "forecast_registry": (
            "outputs/final/forecast_keys.parquet",
            "56e00e8befe256c9c90a7245cdc892e8d7539982c16fadeadd34d404079a1e0f",
        ),
        "formal_reportable_key_registry": (
            (
                "outputs/final/information_regime_key_registries_v4/"
                "primary_reportable_key_registry_v4.parquet"
            ),
            "9a135dcffcfd467cf1e2dda4fc711ba6cf66c8ad54bf3b6f799f4a2d5a3c1d24",
        ),
        "information_runner": (
            "scripts/final/run_information_ladder.py",
            "8fa6b3c327df6dbb74f5e68f22dc0db861f4fc61cd46cab068bd6b79e206b49c",
        ),
        "forcing_runner": (
            "scripts/final/run_forcing_ladder_v4.py",
            "6b370a4c1271e43f4797408bc1831a44882a7e3d727ccd71a485eb0cccac6eb8",
        ),
        "features_source": (
            "src/thermoroute/features.py",
            "0605254b9ac86bd92c7ffb0f31e7db526dde90777d5ddbcf1b1d7bebfcd4e58b",
        ),
        "data_source": (
            "src/thermoroute/data.py",
            "37fe83c34a72dbe676ce37abd27b24c0d5ac4f91c1d62d9822856bf374751984",
        ),
        "config_source": (
            "src/thermoroute/config.py",
            "7661e82df4a6017dcc351f9f4f07e3b94afd5d2426689f775b83e12b78c41c1f",
        ),
        "information_point_manifest": (
            "outputs/final/information_regime_v4/information_regime_v4_manifest.json",
            "6984d561ceb537814837057707db726c20be5b569709ded27ebd2ddd15aef39b",
        ),
        "information_inference_manifest": (
            "outputs/final/information_regime_inference_v4/information_regime_inference_v4_manifest.json",
            "1bce4a92c0b90193a6c64edf26eaaa43f323e5335374c12dbf78cb82d3fa515c",
        ),
        "forcing_point_manifest": (
            "outputs/final/forcing_regime_v4_authority/forcing_regime_v4_manifest.json",
            "260baa7e9fae08afb64280cb87fe320fc1712f4d2e0bd8b261ef824b133285b8",
        ),
        "forcing_inference_manifest": (
            "outputs/final/forcing_regime_inference_v4/forcing_regime_inference_v4_manifest.json",
            "580fed64c308267c1ed0c49e60dac8f21ff1657651de4ee70e3b179d6ed758d7",
        ),
        "score_independent_key_manifest": (
            "outputs/final/information_regime_key_registries_v4/information_regime_key_registries_v4_manifest.json",
            "ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea",
        ),
        "f2a_acquisition_verification": (
            "outputs/final/f2a_acquisition_verification_v1.json",
            "7d8a019d4966a941e9396a40930b2775ba95660347684701838721fe5032a60e",
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

PRODUCTION_EXPECTED_RUNTIME: Mapping[str, str] = _freeze_mapping(
    {
        "python": "3.11.7",
        "numpy": "1.26.4",
        "pandas": "2.1.4",
        "pyarrow": "14.0.2",
    }
)

PRODUCTION_EXPECTED_PANEL = _freeze_mapping(
    {
        "training_rows": 657_480,
        "evaluation_rows": 135_240,
        "combined_rows": 788_880,
        "stations": 120,
        "combined_start": "2006-01-01",
        "combined_end": "2023-12-31",
        "combined_missing": {
            "WTEMP": 113_970,
            "FLOW": 18_611,
            "TEMP": 480,
            "PRCP": 480,
            "RHMEAN": 480,
            "DH": 480,
            "WDSP": 0,
        },
        "raw_panel_contains_observed_columns": False,
        "all_used_variable_station_global_fallbacks_are_finite": True,
    }
)

PRODUCTION_EXPECTED_TEMPORAL: Mapping[int, Mapping[str, Mapping[str, int]]] = _freeze_mapping(
    {
        1: {
            "train": {
                "current_rows": 438_120,
                "admissible_rows": 339_750,
                "invalid_union_rows": 98_370,
                "issue_missing_rows": 96_588,
                "target_missing_rows": 96_534,
                "both_missing_rows": 94_752,
                "mask_error_rows": 143_554,
                "mask_error_admissible_rows": 46_171,
            },
            "val": {
                "current_rows": 87_600,
                "admissible_rows": 83_516,
                "invalid_union_rows": 4_084,
                "issue_missing_rows": 3_639,
                "target_missing_rows": 3_640,
                "both_missing_rows": 3_195,
                "mask_error_rows": 15_137,
                "mask_error_admissible_rows": 11_270,
            },
        },
        3: {
            "train": {
                "current_rows": 437_880,
                "admissible_rows": 337_725,
                "invalid_union_rows": 100_155,
                "issue_missing_rows": 96_577,
                "target_missing_rows": 96_413,
                "both_missing_rows": 92_835,
                "mask_error_rows": 143_521,
                "mask_error_admissible_rows": 45_503,
            },
            "val": {
                "current_rows": 87_360,
                "admissible_rows": 82_948,
                "invalid_union_rows": 4_412,
                "issue_missing_rows": 3_628,
                "target_missing_rows": 3_628,
                "both_missing_rows": 2_844,
                "mask_error_rows": 15_099,
                "mask_error_admissible_rows": 11_116,
            },
        },
        7: {
            "train": {
                "current_rows": 437_400,
                "admissible_rows": 335_822,
                "invalid_union_rows": 101_578,
                "issue_missing_rows": 96_557,
                "target_missing_rows": 96_169,
                "both_missing_rows": 91_148,
                "mask_error_rows": 143_459,
                "mask_error_admissible_rows": 45_131,
            },
            "val": {
                "current_rows": 86_880,
                "admissible_rows": 82_233,
                "invalid_union_rows": 4_647,
                "issue_missing_rows": 3_609,
                "target_missing_rows": 3_600,
                "both_missing_rows": 2_562,
                "mask_error_rows": 15_034,
                "mask_error_admissible_rows": 11_010,
            },
        },
    }
)

PRODUCTION_EXPECTED_EVALUATION: Mapping[int, Mapping[str, Any]] = _freeze_mapping(
    {
        1: {
            "registry_rows": 120_444,
            "raw_issue_missing_rows": 0,
            "raw_target_missing_rows": 0,
            "mask_error_rows": 17_478,
            "mask_error_WTEMP_rows": 13_962,
            "mask_error_FLOW_rows": 759,
            "mask_error_meteorology_rows": 3_281,
            "mask_error_non_WTEMP_rows": 4_011,
            "F3_keys_with_any_raw_future_substitution": 0,
            "F3_total_raw_future_substitutions": 0,
            "F3_raw_future_substitutions_by_variable": {
                "TEMP": 0,
                "PRCP": 0,
                "RHMEAN": 0,
                "DH": 0,
                "WDSP": 0,
            },
        },
        3: {
            "registry_rows": 119_639,
            "raw_issue_missing_rows": 0,
            "raw_target_missing_rows": 0,
            "mask_error_rows": 17_231,
            "mask_error_WTEMP_rows": 13_725,
            "mask_error_FLOW_rows": 750,
            "mask_error_meteorology_rows": 3_271,
            "mask_error_non_WTEMP_rows": 3_992,
            "F3_keys_with_any_raw_future_substitution": 0,
            "F3_total_raw_future_substitutions": 0,
            "F3_raw_future_substitutions_by_variable": {
                "TEMP": 0,
                "PRCP": 0,
                "RHMEAN": 0,
                "DH": 0,
                "WDSP": 0,
            },
        },
        7: {
            "registry_rows": 118_682,
            "raw_issue_missing_rows": 0,
            "raw_target_missing_rows": 0,
            "mask_error_rows": 17_014,
            "mask_error_WTEMP_rows": 13_528,
            "mask_error_FLOW_rows": 734,
            "mask_error_meteorology_rows": 3_261,
            "mask_error_non_WTEMP_rows": 3_966,
            "F3_keys_with_any_raw_future_substitution": 0,
            "F3_total_raw_future_substitutions": 0,
            "F3_raw_future_substitutions_by_variable": {
                "TEMP": 0,
                "PRCP": 0,
                "RHMEAN": 0,
                "DH": 0,
                "WDSP": 0,
            },
        },
    }
)

PRODUCTION_EXPECTED_LEGACY_EVALUATION: Mapping[int, Mapping[str, Any]] = _freeze_mapping(
    {
        1: {
            "registry_rows": 120_466,
            "raw_issue_missing_rows": 0,
            "raw_target_missing_rows": 0,
            "mask_error_rows": 17_500,
            "mask_error_WTEMP_rows": 13_974,
            "mask_error_FLOW_rows": 759,
            "mask_error_meteorology_rows": 3_291,
            "mask_error_non_WTEMP_rows": 4_021,
            "F3_keys_with_any_raw_future_substitution": 0,
            "F3_total_raw_future_substitutions": 0,
            "F3_raw_future_substitutions_by_variable": {
                "TEMP": 0,
                "PRCP": 0,
                "RHMEAN": 0,
                "DH": 0,
                "WDSP": 0,
            },
        },
        3: {
            "registry_rows": 119_654,
            "raw_issue_missing_rows": 0,
            "raw_target_missing_rows": 0,
            "mask_error_rows": 17_246,
            "mask_error_WTEMP_rows": 13_732,
            "mask_error_FLOW_rows": 750,
            "mask_error_meteorology_rows": 3_279,
            "mask_error_non_WTEMP_rows": 4_000,
            "F3_keys_with_any_raw_future_substitution": 0,
            "F3_total_raw_future_substitutions": 0,
            "F3_raw_future_substitutions_by_variable": {
                "TEMP": 0,
                "PRCP": 0,
                "RHMEAN": 0,
                "DH": 0,
                "WDSP": 0,
            },
        },
        7: {
            "registry_rows": 118_687,
            "raw_issue_missing_rows": 0,
            "raw_target_missing_rows": 0,
            "mask_error_rows": 17_019,
            "mask_error_WTEMP_rows": 13_529,
            "mask_error_FLOW_rows": 734,
            "mask_error_meteorology_rows": 3_265,
            "mask_error_non_WTEMP_rows": 3_970,
            "F3_keys_with_any_raw_future_substitution": 0,
            "F3_total_raw_future_substitutions": 0,
            "F3_raw_future_substitutions_by_variable": {
                "TEMP": 0,
                "PRCP": 0,
                "RHMEAN": 0,
                "DH": 0,
                "WDSP": 0,
            },
        },
    }
)

AFFECTED_MANIFEST_ROLES = (
    "information_point_manifest",
    "information_inference_manifest",
    "forcing_point_manifest",
    "forcing_inference_manifest",
)
PRESERVED_AUTHORITY_ROLES = (
    "formal_reportable_key_registry",
    "score_independent_key_manifest",
    "f2a_acquisition_verification",
)

WITHDRAWN_PATHS = (
    "outputs/final/ladder_shards/",
    "outputs/final/ladder_effects.parquet",
    "outputs/final/ladder_summary.json",
    "outputs/final/information_regime_v4/",
    "outputs/final/information_regime_inference_v4/",
    "outputs/final/forcing_shards_v4/",
    "outputs/final/forcing_effects_v4.parquet",
    "outputs/final/forcing_contrasts_v4.parquet",
    "outputs/final/forcing_summary_v4.json",
    "outputs/final/forcing_regime_v4_authority/",
    "outputs/final/forcing_regime_inference_v4/",
)

WITHDRAWN_CLAIMS = (
    "DLOG-018_REPLACEMENT_L_POINT_VALUES_AND_432_CELL_SCIENTIFIC_USE",
    "DLOG-019_L_BY_GEOMETRY_POINT_VALUES",
    "DLOG-020_F0_F3_FULL_LEARNED_VALUES_AND_P5_P6_COMPONENTS",
    "DLOG-021_INFORMATION_WHOLE_HUC2_INFERENCE",
    "DLOG-022_FORCING_WHOLE_HUC2_INFERENCE",
    "SCIENTIFIC_EVIDENCE_STATUS_ROWS_31_32_33_36_38",
    "EXTERNAL_REVIEW_VERIFICATION_SECTION_L_NUMERIC_RESULTS",
    "SI04_DLOG_018_TO_022_RESULT_SUMMARIES",
)

PRESERVED_SCOPE = (
    "DLOG-018_WITHDRAWAL_OF_OLD_PLUS_0P48_30X_27_TO_1_AND_FULL_BUDGET_CLAIMS",
    "SCORE_INDEPENDENT_PRIMARY_AND_F2A_KEY_REGISTRIES",
    "F2A_LABEL_FREE_ACQUISITION_VERIFICATION",
    "FORECAST_REGISTRY_KEY_IDENTITY_CHRONOLOGY_AND_Y_TRUE_ONLY",
    "RAW_F3_FUTURE_AVAILABILITY_MASK_LOGIC_AND_ZERO_EVALUATION_SUBSTITUTIONS",
    "CONVENTIONAL_ROUTE_A_PRE_IMPUTATION_OBSERVED_FLAG_LIFECYCLE_AND_RESULTS",
    "AIR2STREAM_RAW_PANEL_OBSERVEDNESS_PATH_NOT_AFFECTED_BY_THIS_DEFECT",
    "UNRUN_EXTENSIONS_REMAIN_UNRUN_AND_OUT_OF_SCOPE",
)


@dataclass(frozen=True)
class LineageInputPaths:
    station_registry: Path
    training_panel: Path
    evaluation_panel: Path
    forecast_registry: Path
    formal_reportable_key_registry: Path
    information_runner: Path
    forcing_runner: Path
    features_source: Path
    data_source: Path
    config_source: Path
    information_point_manifest: Path
    information_inference_manifest: Path
    forcing_point_manifest: Path
    forcing_inference_manifest: Path
    score_independent_key_manifest: Path
    f2a_acquisition_verification: Path
    requirements_lock: Path
    requirements_lock_py312_hashed: Path
    pyproject: Path

    @classmethod
    def production(cls, root: Path = ROOT) -> LineageInputPaths:
        values = {
            role: root / relative
            for role, (relative, _sha256_value) in PRODUCTION_CANONICAL_FILES.items()
        }
        return cls(**values)


@dataclass(frozen=True)
class BuildConfig:
    mode: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    expected_temporal: Mapping[int, Mapping[str, Mapping[str, int]]] | None = None
    expected_evaluation: Mapping[int, Mapping[str, Any]] | None = None
    expected_legacy_evaluation: Mapping[int, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        _require(self.mode in {"production", "synthetic"}, "invalid build mode")
        _require(
            self.train_start <= self.train_end < self.val_start <= self.val_end, "invalid split"
        )

    @classmethod
    def production(cls) -> BuildConfig:
        return cls(
            mode="production",
            train_start=date(2006, 1, 1),
            train_end=date(2015, 12, 31),
            val_start=date(2016, 1, 1),
            val_end=date(2017, 12, 31),
            expected_temporal=PRODUCTION_EXPECTED_TEMPORAL,
            expected_evaluation=PRODUCTION_EXPECTED_EVALUATION,
            expected_legacy_evaluation=PRODUCTION_EXPECTED_LEGACY_EVALUATION,
        )


def _validate_production_config(config: BuildConfig) -> None:
    expected = BuildConfig.production()
    _require(type(config) is BuildConfig, "production build config type changed")
    _require(config == expected, "production build config is not the exact canonical config")


@dataclass(frozen=True)
class _BoundFile:
    path: Path
    payload: bytes
    sha256: str


_BUNDLE_TOKEN = object()


@dataclass(frozen=True)
class _ArtifactBundle:
    files: Mapping[str, bytes]
    report_payload: bytes
    manifest_payload: bytes
    _token: object

    def __post_init__(self) -> None:
        _require(self._token is _BUNDLE_TOKEN, "artifact bundle was not issued by the builder")
        copied: dict[str, bytes] = {}
        for name, payload in self.files.items():
            _require(type(name) is str and type(payload) is bytes, "bundle payload types changed")
            copied[name] = bytes(payload)
        object.__setattr__(self, "files", MappingProxyType(copied))
        object.__setattr__(self, "report_payload", bytes(self.report_payload))
        object.__setattr__(self, "manifest_payload", bytes(self.manifest_payload))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    """Resolve an existing path and reject a symlink in any component."""

    absolute = _absolute(path)
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LineageAuthorityError(f"{label} cannot be resolved: {absolute}") from exc
    _require(resolved == absolute, f"{label} traverses a symbolic link: {absolute}")
    return absolute


def _relative(path: Path, root: Path) -> str:
    try:
        return _absolute(path).relative_to(_absolute(root)).as_posix()
    except ValueError as exc:
        raise LineageAuthorityError(f"input escapes repository root: {path}") from exc


def _read_stable_regular(path: Path, *, root: Path, label: str) -> _BoundFile:
    absolute = _absolute(path)
    root_absolute = _require_no_symlink_components(root, label="repository root")
    _relative(absolute, root_absolute)
    _require_no_symlink_components(absolute, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise LineageAuthorityError(f"{label} cannot be opened safely: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{label} changed while being captured",
    )
    _require(len(payload) == before.st_size, f"{label} byte count changed while being captured")
    try:
        path_after = absolute.lstat()
    except OSError as exc:
        raise LineageAuthorityError(f"{label} path disappeared after capture: {absolute}") from exc
    _require(
        not stat.S_ISLNK(path_after.st_mode)
        and stat.S_ISREG(path_after.st_mode)
        and (path_after.st_dev, path_after.st_ino) == (before.st_dev, before.st_ino),
        f"{label} path changed while being captured",
    )
    _require_no_symlink_components(absolute, label=label)
    return _BoundFile(path=absolute, payload=payload, sha256=_sha256(payload))


def _binding(bound: _BoundFile, root: Path, *, opaque: bool = False) -> dict[str, Any]:
    return {
        "path": _relative(bound.path, root),
        "sha256": bound.sha256,
        "size_bytes": len(bound.payload),
        "opaque_bytes_not_semantically_parsed": bool(opaque),
    }


def _validate_canonical_bindings(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    observed = {
        role: (str(record["path"]), str(record["sha256"]))
        for role, record in bindings.items()
        if role != "builder_code"
    }
    expected = dict(PRODUCTION_CANONICAL_FILES)
    _require(set(observed) == set(expected), "production input role set changed")
    for role, pair in expected.items():
        _require(observed[role] == pair, f"canonical production path/hash changed for {role}")


def _runtime_evidence(
    bindings: Mapping[str, Mapping[str, Any]], *, production: bool
) -> dict[str, Any]:
    active = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }
    if production:
        observed = {name: active[name] for name in PRODUCTION_EXPECTED_RUNTIME}
        _require(
            observed == dict(PRODUCTION_EXPECTED_RUNTIME), "production runtime version changed"
        )
    return {
        "determinism_class": "EXACT_ACTIVE_RUNTIME_ONLY",
        "active_runtime": active,
        "repository_lock_bindings": {
            role: bindings[role]
            for role in ("requirements_lock", "requirements_lock_py312_hashed", "pyproject")
        },
        "lock_compatibility_statement": (
            "Both repository locks are bound as evidence, but this authority was serialized under "
            "the active CPython 3.11/pandas/Arrow runtime; no cross-runtime byte identity is claimed."
        ),
    }


def _read_parquet(bound: _BoundFile, *, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(bound.payload))
    except Exception as exc:
        raise LineageAuthorityError(f"{label} is not readable Parquet") from exc


def _normalise_sites(values: pd.Series, *, label: str) -> pd.Series:
    _require(values.notna().all(), f"{label} contains null site identifiers")
    strings = values.astype(str).str.strip().str.zfill(8)
    valid = strings.str.fullmatch(r"(?:[0-9]{8}|[0-9]{15})")
    _require(valid.all(), f"{label} contains invalid site identifiers")
    return strings


def _load_station_registry(bound: _BoundFile) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(bound.payload), dtype=str)
    except Exception as exc:
        raise LineageAuthorityError("station registry is not readable CSV") from exc
    required = {"site_no", "legacy_site_id"}
    _require(required <= set(frame.columns), "station registry lacks required identifiers")
    out = frame[["site_no", "legacy_site_id"]].copy()
    out["site_no"] = _normalise_sites(out["site_no"], label="station registry")
    out["legacy_site_id"] = out["legacy_site_id"].astype(str)
    _require(not out.isna().any(axis=None), "station registry contains null identifiers")
    _require(not out.duplicated("site_no").any(), "station registry duplicates site_no")
    _require(not out.duplicated("legacy_site_id").any(), "station registry duplicates legacy ids")
    return out.sort_values("site_no").reset_index(drop=True)


def _load_raw_panel(bound: _BoundFile, *, label: str) -> pd.DataFrame:
    frame = _read_parquet(bound, label=label)
    _require(tuple(frame.columns) == PANEL_COLUMNS, f"{label} schema/order changed")
    _require(
        not any(column.endswith("_observed") for column in frame.columns),
        f"{label} contains pre-existing observed flags",
    )
    out = frame.copy()
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    _require(out["DATE"].notna().all(), f"{label} contains invalid dates")
    _require(out["DATE"].dt.tz is None, f"{label} dates must be timezone-naive")
    for variable in PANEL_COLUMNS[2:]:
        out[variable] = pd.to_numeric(out[variable], errors="coerce")
        _require(
            not np.isinf(out[variable].to_numpy(dtype=float)).any(),
            f"{label} {variable} contains an infinite value",
        )
    return out


def _combine_panels(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    station_registry: pd.DataFrame,
) -> pd.DataFrame:
    alias = dict(zip(station_registry["legacy_site_id"], station_registry["site_no"], strict=True))
    train = training.copy()
    test = evaluation.copy()
    train["site_id"] = train["site_id"].astype(str).map(alias)
    _require(train["site_id"].notna().all(), "training panel has unmapped legacy stations")
    train["site_id"] = _normalise_sites(train["site_id"], label="training panel")
    test["site_id"] = _normalise_sites(test["site_id"], label="evaluation panel")
    expected_sites = set(station_registry["site_no"])
    _require(set(train["site_id"]) == expected_sites, "training panel station set changed")
    _require(set(test["site_id"]) == expected_sites, "evaluation panel station set changed")
    panel = pd.concat([train, test], ignore_index=True)
    panel = panel.drop_duplicates(["DATE", "site_id"], keep="last")
    panel = panel.sort_values(["site_id", "DATE"], kind="mergesort").reset_index(drop=True)
    _require(
        not panel.duplicated(["site_id", "DATE"]).any(), "combined panel duplicates a station-day"
    )
    for site, group in panel.groupby("site_id", sort=True):
        dates = group["DATE"].to_numpy(dtype="datetime64[D]")
        _require(
            len(dates) == 1 or np.all(np.diff(dates) == np.timedelta64(1, "D")),
            f"combined panel has a calendar gap for {site}",
        )
        for variable in USED_VARIABLES:
            _require(group[variable].notna().any(), f"{variable} has no raw support for {site}")
    _require(
        panel["WTEMP"].isna().any(),
        "raw WTEMP contains no missing values; imputed-as-raw input refused",
    )
    return panel


def _validate_imputer_fit_support(panel: pd.DataFrame, config: BuildConfig) -> None:
    """Prove every per-station fallback used by either audited fit is finite."""

    windows = {
        "training": (pd.Timestamp(config.train_start), pd.Timestamp(config.train_end)),
        "final_train_plus_validation": (
            pd.Timestamp(config.train_start),
            pd.Timestamp(config.val_end),
        ),
    }
    expected_sites = set(panel["site_id"])
    for label, (lower, upper) in windows.items():
        fit_panel = panel.loc[panel["DATE"].between(lower, upper)]
        _require(set(fit_panel["site_id"]) == expected_sites, f"{label} fit lacks a station")
        for variable in USED_VARIABLES:
            finite_by_site = fit_panel.groupby("site_id", sort=True)[variable].agg(
                lambda values: bool(np.isfinite(values.to_numpy(dtype=float)).any())
            )
            _require(
                bool(finite_by_site.all()),
                f"{label} fit has a non-finite {variable} station fallback",
            )


def _raw_observation_masks(panel: pd.DataFrame) -> tuple[dict[str, np.ndarray], str]:
    masks = {
        variable: np.isfinite(panel[variable].to_numpy(dtype=float)) for variable in USED_VARIABLES
    }
    digest = hashlib.sha256()
    digest.update(b"thermoroute-raw-observed-mask-v1\n")
    for variable in USED_VARIABLES:
        digest.update(variable.encode("ascii") + b"\0")
        digest.update(np.packbits(masks[variable], bitorder="little").tobytes())
    return masks, digest.hexdigest()


def _mask_feature_error_rows(
    panel: pd.DataFrame,
    raw_masks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rows where post-imputation fallback masks differ from true raw masks."""

    errors: dict[str, np.ndarray] = {}
    station_key = panel["site_id"]
    for variable in USED_VARIABLES:
        observed = pd.Series(raw_masks[variable].astype(float), index=panel.index)
        inferred_after_imputation = pd.Series(1.0, index=panel.index)
        observed_group = observed.groupby(station_key, sort=False)
        inferred_group = inferred_after_imputation.groupby(station_key, sort=False)
        mismatch = np.zeros(len(panel), dtype=bool)
        for lag in SHORT_LAGS:
            left = observed_group.shift(lag).fillna(0.0).to_numpy(float)
            right = inferred_group.shift(lag).fillna(0.0).to_numpy(float)
            mismatch |= ~np.isclose(left, right, atol=0.0, rtol=0.0)
        for window in ROLLING_WINDOWS:
            left = (
                observed_group.rolling(window)
                .mean()
                .reset_index(level=0, drop=True)
                .sort_index()
                .fillna(0.0)
                .to_numpy(float)
            )
            right = (
                inferred_group.rolling(window)
                .mean()
                .reset_index(level=0, drop=True)
                .sort_index()
                .fillna(0.0)
                .to_numpy(float)
            )
            mismatch |= ~np.isclose(left, right, atol=0.0, rtol=0.0)
        for lag in DELTA_LAGS:
            left = (observed * observed_group.shift(lag)).fillna(0.0).to_numpy(float)
            right = (
                (inferred_after_imputation * inferred_group.shift(lag)).fillna(0.0).to_numpy(float)
            )
            mismatch |= ~np.isclose(left, right, atol=0.0, rtol=0.0)
        errors[variable] = mismatch
    return errors


def _temporal_inventory(
    panel: pd.DataFrame,
    raw_masks: Mapping[str, np.ndarray],
    mask_errors: Mapping[str, np.ndarray],
    config: BuildConfig,
) -> dict[str, dict[str, dict[str, int]]]:
    issue_observed = raw_masks["WTEMP"]
    any_mask_error = np.logical_or.reduce([mask_errors[name] for name in USED_VARIABLES])
    raw_target_observed = pd.Series(raw_masks["WTEMP"], index=panel.index)
    target_observed_by_h = {
        lead: raw_target_observed.groupby(panel["site_id"], sort=False)
        .shift(-lead)
        .fillna(False)
        .to_numpy(dtype=bool)
        for lead in LEADS
    }
    result: dict[str, dict[str, dict[str, int]]] = {}
    split_ranges = {
        "train": (pd.Timestamp(config.train_start), pd.Timestamp(config.train_end)),
        "val": (pd.Timestamp(config.val_start), pd.Timestamp(config.val_end)),
    }
    for lead in LEADS:
        target_date = panel["DATE"] + pd.Timedelta(days=lead)
        target_observed = target_observed_by_h[lead]
        admissible = issue_observed & target_observed
        by_split: dict[str, dict[str, int]] = {}
        for split_name, (lower, upper) in split_ranges.items():
            selected = (
                panel["DATE"].between(lower, upper) & target_date.between(lower, upper)
            ).to_numpy()
            item = {
                "current_rows": int(selected.sum()),
                "admissible_rows": int((selected & admissible).sum()),
                "invalid_union_rows": int((selected & ~admissible).sum()),
                "issue_missing_rows": int((selected & ~issue_observed).sum()),
                "target_missing_rows": int((selected & ~target_observed).sum()),
                "both_missing_rows": int((selected & ~issue_observed & ~target_observed).sum()),
                "mask_error_rows": int((selected & any_mask_error).sum()),
                "mask_error_admissible_rows": int((selected & any_mask_error & admissible).sum()),
            }
            _require(
                item["current_rows"] - item["admissible_rows"] == item["invalid_union_rows"],
                "temporal row arithmetic changed",
            )
            by_split[split_name] = item
        result[str(lead)] = by_split
    return result


def _normalise_registry_frame(
    frame: pd.DataFrame,
    *,
    lead_column: str,
    label: str,
) -> pd.DataFrame:
    required = ("key_id", "site_id", lead_column, "issue_date", "target_date", "y_true")
    _require(set(required) <= set(frame.columns), f"{label} lacks required key/y columns")
    out = frame[list(required)].copy()
    _require(out["key_id"].notna().all(), f"{label} contains null key_id values")
    out["key_id"] = out["key_id"].astype(str)
    _require((out["key_id"].str.len() > 0).all(), f"{label} contains empty key_id values")
    out["site_id"] = _normalise_sites(out["site_id"], label=label)
    numeric_leads = pd.to_numeric(out[lead_column], errors="coerce").to_numpy(float)
    _require(
        np.isfinite(numeric_leads).all() and np.equal(numeric_leads, np.floor(numeric_leads)).all(),
        f"{label} has invalid or non-integral horizons",
    )
    out["horizon"] = numeric_leads.astype(int)
    if lead_column != "horizon":
        out = out.drop(columns=[lead_column])
    out["issue_date"] = pd.to_datetime(out["issue_date"], errors="coerce")
    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    _require(not out.isna().any(axis=None), f"{label} contains null key/y values")
    _require(
        np.isfinite(out["y_true"].to_numpy(float)).all(), f"{label} contains non-finite y_true"
    )
    _require(out["issue_date"].dt.tz is None, f"{label} issue dates must be timezone-naive")
    _require(out["target_date"].dt.tz is None, f"{label} target dates must be timezone-naive")
    _require(
        out["issue_date"].eq(out["issue_date"].dt.normalize()).all()
        and out["target_date"].eq(out["target_date"].dt.normalize()).all(),
        f"{label} contains non-midnight dates",
    )
    _require(set(out["horizon"]) == set(LEADS), f"{label} horizon set changed")
    _require(not out.duplicated("key_id").any(), f"{label} duplicates key_id")
    _require(
        not out.duplicated(["site_id", "horizon", "issue_date"]).any(),
        f"{label} duplicates forecast identities",
    )
    expected_target = out["issue_date"] + pd.to_timedelta(out["horizon"], unit="D")
    _require(
        out["target_date"].equals(expected_target.rename("target_date")),
        f"{label} target chronology changed",
    )
    return out.sort_values(["horizon", "site_id", "issue_date"], kind="mergesort").reset_index(
        drop=True
    )


def _load_legacy_forecast_registry(bound: _BoundFile) -> pd.DataFrame:
    frame = _read_parquet(bound, label="legacy forecast registry")
    return _normalise_registry_frame(
        frame,
        lead_column="horizon",
        label="legacy forecast registry",
    )


def _load_formal_reportable_key_registry(bound: _BoundFile) -> pd.DataFrame:
    try:
        schema = pq.read_schema(io.BytesIO(bound.payload))
    except Exception as exc:
        raise LineageAuthorityError("formal reportable key registry schema is unreadable") from exc
    observed_schema = tuple((field.name, str(field.type), bool(field.nullable)) for field in schema)
    _require(
        observed_schema == FORMAL_REPORTABLE_KEY_ARROW_SCHEMA,
        "formal reportable key registry Arrow schema changed",
    )
    frame = _read_parquet(bound, label="formal reportable key registry")
    for column in (
        "issue_wtemp_observed",
        "history_H100",
        "history_H75",
        "history_Hall",
        "reportable_primary",
    ):
        _require(
            pd.api.types.is_bool_dtype(frame[column].dtype),
            f"formal reportable key registry {column} is not boolean",
        )
    _require(
        bool(frame["issue_wtemp_observed"].all())
        and bool(frame["history_Hall"].all())
        and bool(frame["reportable_primary"].all()),
        "formal reportable key registry contains a non-admissible row",
    )
    return _normalise_registry_frame(
        frame,
        lead_column="lead_days",
        label="formal reportable key registry",
    )


def _validate_registry_relationship(
    formal: pd.DataFrame,
    legacy: pd.DataFrame,
) -> dict[str, Any]:
    formal_ids = set(formal["key_id"])
    legacy_ids = set(legacy["key_id"])
    _require(formal_ids <= legacy_ids, "formal reportable keys are not a legacy-key subset")
    joined = formal.merge(
        legacy,
        on="key_id",
        how="left",
        validate="one_to_one",
        suffixes=("_formal", "_legacy"),
    )
    _require(len(joined) == len(formal), "formal/legacy key relationship join changed")
    for column in ("site_id", "horizon", "issue_date", "target_date"):
        _require(
            np.array_equal(
                joined[f"{column}_formal"].to_numpy(),
                joined[f"{column}_legacy"].to_numpy(),
            ),
            f"formal/legacy {column} binding differs",
        )
    _require(
        np.array_equal(
            joined["y_true_formal"].to_numpy(float),
            joined["y_true_legacy"].to_numpy(float),
        ),
        "formal reportable y_true is not the exact legacy persisted value",
    )
    return {
        "formal_reportable_is_exact_subset_of_legacy_keys": True,
        "formal_reportable_rows": len(formal),
        "legacy_rows": len(legacy),
        "legacy_only_nonreportable_rows": len(legacy_ids - formal_ids),
        "shared_identity_chronology_and_y_true_exact": True,
    }


def _evaluation_inventory(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    mask_errors: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], float]:
    panel_lookup = panel.set_index(["site_id", "DATE"])
    lookup = panel_lookup["WTEMP"]
    issue_index = pd.MultiIndex.from_frame(registry[["site_id", "issue_date"]])
    target_index = pd.MultiIndex.from_frame(registry[["site_id", "target_date"]])
    issue_values = lookup.reindex(issue_index).to_numpy(float)
    target_values = lookup.reindex(target_index).to_numpy(float)
    issue_missing = ~np.isfinite(issue_values)
    target_missing = ~np.isfinite(target_values)
    _require(not issue_missing.any(), "evaluation registry contains raw-missing issue WTEMP")
    _require(not target_missing.any(), "evaluation registry contains raw-missing target WTEMP")
    maximum_y_difference = float(np.max(np.abs(target_values - registry["y_true"].to_numpy(float))))
    _require(
        maximum_y_difference <= Y_ABSOLUTE_TOLERANCE_C,
        "evaluation registry y_true differs from raw target beyond tolerance",
    )

    mask_table = pd.DataFrame(
        {
            "site_id": panel["site_id"],
            "issue_date": panel["DATE"],
            **{name: values for name, values in mask_errors.items()},
        }
    )
    joined = registry.merge(
        mask_table,
        on=["site_id", "issue_date"],
        how="left",
        validate="many_to_one",
    )
    _require(not joined[list(USED_VARIABLES)].isna().any(axis=None), "evaluation mask join failed")
    inventory: dict[str, Any] = {}
    for lead, group in joined.groupby("horizon", sort=True):
        lead = int(lead)
        any_error = group[list(USED_VARIABLES)].any(axis=1)
        met_error = group[list(MET_VARIABLES)].any(axis=1)
        non_wtemp = group[[name for name in USED_VARIABLES if name != "WTEMP"]].any(axis=1)
        future_substitutions = {variable: 0 for variable in MET_VARIABLES}
        key_has_substitution = np.zeros(len(group), dtype=bool)
        for step in range(1, lead + 1):
            future_index = pd.MultiIndex.from_arrays(
                [
                    group["site_id"].to_numpy(),
                    (group["issue_date"] + pd.Timedelta(days=step)).to_numpy(),
                ],
                names=("site_id", "DATE"),
            )
            future_values = panel_lookup[list(MET_VARIABLES)].reindex(future_index)
            raw_missing = ~np.isfinite(future_values.to_numpy(dtype=float))
            key_has_substitution |= raw_missing.any(axis=1)
            for variable_index, variable in enumerate(MET_VARIABLES):
                future_substitutions[variable] += int(raw_missing[:, variable_index].sum())
        total_future_substitutions = sum(future_substitutions.values())
        inventory[str(lead)] = {
            "registry_rows": len(group),
            "raw_issue_missing_rows": 0,
            "raw_target_missing_rows": 0,
            "mask_error_rows": int(any_error.sum()),
            "mask_error_WTEMP_rows": int(group["WTEMP"].sum()),
            "mask_error_FLOW_rows": int(group["FLOW"].sum()),
            "mask_error_meteorology_rows": int(met_error.sum()),
            "mask_error_non_WTEMP_rows": int(non_wtemp.sum()),
            "F3_keys_with_any_raw_future_substitution": int(key_has_substitution.sum()),
            "F3_total_raw_future_substitutions": total_future_substitutions,
            "F3_raw_future_substitutions_by_variable": future_substitutions,
        }
    return inventory, maximum_y_difference


def _normalise_expected_temporal(
    value: Mapping[int, Mapping[str, Mapping[str, int]]],
) -> dict[str, Any]:
    return {
        str(lead): {split: dict(item) for split, item in splits.items()}
        for lead, splits in value.items()
    }


def _normalise_expected_evaluation(value: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {str(lead): dict(item) for lead, item in value.items()}


def _validate_declared_scope(
    withdrawn_paths: Sequence[str],
    withdrawn_claims: Sequence[str],
    preserved_scope: Sequence[str],
) -> None:
    _require(tuple(withdrawn_paths) == WITHDRAWN_PATHS, "withdrawn artifact scope drifted")
    _require(tuple(withdrawn_claims) == WITHDRAWN_CLAIMS, "withdrawn claim scope drifted")
    _require(tuple(preserved_scope) == PRESERVED_SCOPE, "preserved scope drifted")
    _require(not set(withdrawn_claims) & set(preserved_scope), "withdrawn/preserved scope overlaps")


def _validate_production_source_semantics(bounds: Mapping[str, _BoundFile]) -> None:
    information = bounds["information_runner"].payload.decode("utf-8")
    forcing = bounds["forcing_runner"].payload.decode("utf-8")
    data_source = bounds["data_source"].payload.decode("utf-8")
    features = bounds["features_source"].payload.decode("utf-8")
    required_fragments = {
        "information runner": (
            "panel_imp = imputer.transform(panel)",
            "panel_masked, panel_imp, imputer, clim, h",
        ),
        "forcing runner": (
            "panel_imputed = imputer.transform(panel)",
            "panel_imputed,\n                    panel_imputed,",
            "validation_size = min(2000, len(train))",
            "train[list(model_columns)].to_numpy(float)[-validation_size:]",
            "outcome[-validation_size:]",
            "fit_air2stream_models(\n            panel,",
            "station_maps = _build_station_state_maps(panel)",
        ),
        "data source": (
            "def transform(self, panel: pd.DataFrame)",
            "col[miss] = fill",
            'panel[f"{v}_observed"] = panel[v].notna()',
        ),
        "features source": (
            'if f"{var}_observed" in sub.columns else s.notna().astype(float)',
            'feat["y_observed"] = feat["y"].notna().to_numpy()',
        ),
    }
    payloads = {
        "information runner": information,
        "forcing runner": forcing,
        "data source": data_source,
        "features source": features,
    }
    for label, fragments in required_fragments.items():
        for fragment in fragments:
            _require(
                fragment in payloads[label],
                f"bound {label} no longer exhibits audited lineage path",
            )


def _capture_inputs(
    paths: LineageInputPaths, *, root: Path, production: bool
) -> dict[str, _BoundFile]:
    _require(type(paths) is LineageInputPaths, "input path object was forged or replaced")
    bounds = {
        role: _read_stable_regular(path, root=root, label=role.replace("_", " "))
        for role, path in paths.__dict__.items()
    }
    builder = _read_stable_regular(Path(__file__), root=ROOT, label="lineage authority builder")
    bounds["builder_code"] = builder
    bindings = {
        role: _binding(
            bound,
            root if role != "builder_code" or production else ROOT,
            opaque=role in AFFECTED_MANIFEST_ROLES,
        )
        for role, bound in bounds.items()
    }
    if production:
        _validate_canonical_bindings(bindings)
        _validate_production_source_semantics(bounds)
    return bounds


def _build_report(
    *,
    bindings: Mapping[str, Mapping[str, Any]],
    panel_inventory: Mapping[str, Any],
    observed_mask_sha256: str,
    temporal_inventory: Mapping[str, Any],
    evaluation_inventory: Mapping[str, Any],
    maximum_y_difference: float,
    legacy_evaluation_inventory: Mapping[str, Any],
    legacy_maximum_y_difference: float,
    registry_relationship: Mapping[str, Any],
    config: BuildConfig,
) -> dict[str, Any]:
    _validate_declared_scope(WITHDRAWN_PATHS, WITHDRAWN_CLAIMS, PRESERVED_SCOPE)
    f3_substitution_count = sum(
        int(item["F3_total_raw_future_substitutions"]) for item in evaluation_inventory.values()
    )
    legacy_f3_substitution_count = sum(
        int(item["F3_total_raw_future_substitutions"])
        for item in legacy_evaluation_inventory.values()
    )
    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": STATUS,
        "build_mode": config.mode.upper(),
        "scope_guard": {
            "prediction_shards_read": False,
            "model_score_tables_read": False,
            "model_extension_outcomes_read": False,
            "affected_manifests_hashed_as_opaque_bytes_only": True,
            "new_model_result_computed": False,
        },
        "defect_verdict": {
            "confirmed": True,
            "observed_flags_absent_before_imputation": True,
            "imputer_fills_values_but_does_not_create_observed_flags": True,
            "information_runner_passes_imputed_panel_as_panel_true": True,
            "forcing_runner_passes_imputed_panel_as_panel_true": True,
            "imputed_issue_WTEMP_is_admitted": True,
            "imputed_target_WTEMP_is_admitted_as_training_y": True,
            "post_imputation_finite_fallback_corrupts_missingness_features": True,
            "forcing_lightgbm_eval_set_is_tail_of_same_train_plus_val_table": True,
            "forcing_lightgbm_eval_set_maximum_rows": 2_000,
            "independent_2016_2017_validation_passed_to_old_lightgbm_fit": False,
            "scientific_cancellation_across_models_arms_or_folds_assumed": False,
        },
        "input_bindings": dict(bindings),
        "raw_panel_inventory": {
            **dict(panel_inventory),
            "raw_observed_mask_sha256": observed_mask_sha256,
            "observed_definition": "finite value in the captured raw pre-imputation panel",
        },
        "training_lineage_inventory": {
            "split_contract": {
                "train": [config.train_start.isoformat(), config.train_end.isoformat()],
                "val": [config.val_start.isoformat(), config.val_end.isoformat()],
                "issue_and_target_must_remain_within_the_same_split": True,
            },
            "current_rows_definition": (
                "calendar rows admitted after all raw WTEMP gaps are fillable and the imputed "
                "panel is reused as panel_true"
            ),
            "admissible_rows_definition": "raw issue WTEMP finite AND raw target WTEMP finite",
            "by_horizon": dict(temporal_inventory),
            "forcing_repetition": (
                "each horizon inventory is repeated unchanged in F0/F3_full x LightGBM/"
                "ResidualLightGBM"
            ),
            "information_repetition": (
                "each spatial-fold subset is repeated in L0/L1/L2 x LightGBM/"
                "ResidualLightGBM; fold-specific missingness prevents cancellation"
            ),
            "old_forcing_lightgbm_eval_set_defect": {
                "training_pool": "the same table filtered to split in [train,val]",
                "validation_size": "min(2000, len(training_pool))",
                "eval_X": "training_pool[model_columns][-validation_size:]",
                "eval_y": "training_outcome[-validation_size:]",
                "independent_2016_2017_validation_table_used": False,
                "source_semantics_exactly_verified_from_bound_forcing_runner": True,
            },
        },
        "evaluation_inventory": {
            "authority_role": "formal_reportable_key_registry",
            "score_independent_formal_reportable_registry_used": True,
            "by_horizon": dict(evaluation_inventory),
            "maximum_raw_target_vs_registry_y_true_absolute_difference_c": maximum_y_difference,
            "target_tolerance_c": Y_ABSOLUTE_TOLERANCE_C,
            "key_identity_chronology_and_y_true_preserved": True,
            "learned_predictions_preserved": False,
            "F3_raw_future_mask_path_preserved_for_this_specific_defect": True,
            "F3_evaluation_substitution_count_independently_reconstructed": (f3_substitution_count),
        },
        "legacy_forecast_registry_inventory": {
            "authority_role": "forecast_registry",
            "scientific_evaluation_domain": False,
            "role": "legacy raw-key chronology and persisted-y forensic comparison only",
            "legacy_context_columns_consumed": False,
            "by_horizon": dict(legacy_evaluation_inventory),
            "maximum_raw_target_vs_registry_y_true_absolute_difference_c": (
                legacy_maximum_y_difference
            ),
            "target_tolerance_c": Y_ABSOLUTE_TOLERANCE_C,
            "F3_evaluation_substitution_count_independently_reconstructed": (
                legacy_f3_substitution_count
            ),
            "relationship_to_formal_reportable_registry": dict(registry_relationship),
        },
        "withdrawal": {
            "artifact_paths": list(WITHDRAWN_PATHS),
            "claims": list(WITHDRAWN_CLAIMS),
            "old_bytes_must_be_preserved_for_forensic_reproducibility": True,
            "old_authority_hash_and_arithmetic_integrity_remain_descriptive_only": True,
        },
        "preserved": {
            "scope": list(PRESERVED_SCOPE),
            "authority_roles": list(PRESERVED_AUTHORITY_ROLES),
            "forecast_registry_context_columns_preserved": False,
            "forecast_registry_preservation_is_limited_to_key_identity_chronology_and_y_true": True,
            "conventional_route_a": {
                "preserved": True,
                "basis": "bound data source creates raw observed flags before imputation",
            },
            "air2stream": {
                "preserved_for_this_specific_defect": True,
                "basis": "bound forcing runner passes the raw panel to fit/state-map paths",
            },
        },
        "required_versioned_rerun": {
            "old_runners_or_artifacts_may_be_overwritten": False,
            "raw_observed_flags_must_be_created_before_imputation": True,
            "imputed_feature_panel_and_raw_label_panel_must_be_distinct_typed_inputs": True,
            "observed_masks_must_survive_imputation_bit_exactly": True,
            "training_issue_and_target_admissibility_must_be_joined_from_raw_lineage": True,
            "training_key_registries_or_digests_must_be_authority_bound": True,
            "training_must_use_only_2006_2015_rows": True,
            "validation_must_use_only_independent_2016_2017_rows": True,
            "validation_must_not_be_a_tail_slice_of_the_training_table": True,
            "rerun_scope": "ONLY_ALREADY_VIEWED_432_INFORMATION_AND_12_FORCING_TREE_CELLS",
            "unrun_extensions_may_be_opened_by_this_authority": False,
        },
    }


def build_artifacts(
    paths: LineageInputPaths,
    *,
    config: BuildConfig,
    repository_root: Path,
) -> _ArtifactBundle:
    """Independently reconstruct deterministic authority bytes; publish nothing."""

    _require(type(config) is BuildConfig, "build config type changed")
    repository_root = _absolute(repository_root)
    production = config.mode == "production"
    if production:
        _validate_production_config(config)
        _require(repository_root == _absolute(ROOT), "production repository root changed")
        expected_paths = LineageInputPaths.production(repository_root)
        _require(paths == expected_paths, "production input paths are not canonical")
    else:
        _require(repository_root != _absolute(ROOT), "synthetic mode cannot use production root")

    bounds = _capture_inputs(paths, root=repository_root, production=production)
    bindings = {
        role: _binding(
            bound,
            repository_root if role != "builder_code" or production else ROOT,
            opaque=role in AFFECTED_MANIFEST_ROLES,
        )
        for role, bound in sorted(bounds.items())
    }
    runtime = _runtime_evidence(bindings, production=production)
    station_registry = _load_station_registry(bounds["station_registry"])
    training = _load_raw_panel(bounds["training_panel"], label="training panel")
    evaluation = _load_raw_panel(bounds["evaluation_panel"], label="evaluation panel")
    panel = _combine_panels(training, evaluation, station_registry)
    _validate_imputer_fit_support(panel, config)
    raw_masks, observed_mask_sha256 = _raw_observation_masks(panel)
    mask_errors = _mask_feature_error_rows(panel, raw_masks)
    temporal = _temporal_inventory(panel, raw_masks, mask_errors, config)
    legacy_registry = _load_legacy_forecast_registry(bounds["forecast_registry"])
    formal_registry = _load_formal_reportable_key_registry(bounds["formal_reportable_key_registry"])
    registry_relationship = _validate_registry_relationship(formal_registry, legacy_registry)
    evaluation_inventory, maximum_y_difference = _evaluation_inventory(
        panel, formal_registry, mask_errors
    )
    legacy_evaluation_inventory, legacy_maximum_y_difference = _evaluation_inventory(
        panel, legacy_registry, mask_errors
    )
    panel_inventory = {
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "combined_rows": len(panel),
        "stations": panel["site_id"].nunique(),
        "combined_start": panel["DATE"].min().date().isoformat(),
        "combined_end": panel["DATE"].max().date().isoformat(),
        "combined_missing": {
            variable: int(panel[variable].isna().sum()) for variable in USED_VARIABLES
        },
        "raw_panel_contains_observed_columns": False,
        "all_used_variable_station_global_fallbacks_are_finite": True,
    }
    if production:
        _require(
            panel_inventory == dict(PRODUCTION_EXPECTED_PANEL), "production panel inventory changed"
        )
    if config.expected_temporal is not None:
        _require(
            temporal == _normalise_expected_temporal(config.expected_temporal),
            "production temporal lineage counts changed",
        )
    if config.expected_evaluation is not None:
        _require(
            evaluation_inventory == _normalise_expected_evaluation(config.expected_evaluation),
            "production evaluation lineage counts changed",
        )
    if config.expected_legacy_evaluation is not None:
        _require(
            legacy_evaluation_inventory
            == _normalise_expected_evaluation(config.expected_legacy_evaluation),
            "production legacy evaluation lineage counts changed",
        )
    _require(
        sum(item["invalid_union_rows"] for split in temporal.values() for item in split.values())
        > 0,
        "lineage reconstruction found no invalid training rows",
    )
    report = _build_report(
        bindings=bindings,
        panel_inventory=panel_inventory,
        observed_mask_sha256=observed_mask_sha256,
        temporal_inventory=temporal,
        evaluation_inventory=evaluation_inventory,
        maximum_y_difference=maximum_y_difference,
        legacy_evaluation_inventory=legacy_evaluation_inventory,
        legacy_maximum_y_difference=legacy_maximum_y_difference,
        registry_relationship=registry_relationship,
        config=config,
    )
    report_payload = _canonical_json_bytes(report)
    manifest_without_receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": STATUS,
        "build_mode": config.mode.upper(),
        "input_bindings": bindings,
        "runtime_determinism": runtime,
        "output": {
            "name": REPORT_FILENAME,
            "sha256": _sha256(report_payload),
            "size_bytes": len(report_payload),
        },
        "publication_contract": {
            "atomic_linux_RENAME_NOREPLACE": True,
            "exclusive": True,
            "create_only": True,
            "resume_supported": False,
            "overwrite_supported": False,
            "independent_build_passes_before_publish": 2,
            "exact_published_file_count": 2,
            "staged_bytes_reverified_immediately_before_rename": True,
            "closed_publisher_accepts_caller_bundle": False,
            "production_config_exactly_pinned": True,
            "production_output_root_and_destination_exactly_pinned": True,
            "allowed_root_opened_once_with_O_DIRECTORY_O_NOFOLLOW": True,
            "lock_stage_write_verify_rename_fsync_cleanup_use_same_root_dirfd": True,
            "renameat2_uses_anchored_old_and_new_dirfds_plus_basenames": True,
            "lexical_root_inode_verified_before_and_after_rename": True,
            "failed_root_identity_check_cleans_owned_entries_via_anchored_dirfd": True,
        },
    }
    receipt_payload = {
        "algorithm": (
            "capture canonical raw/evidence bytes; independently rebuild raw observed masks, "
            "training and formal-reportable/legacy evaluation inventories, report and manifest "
            "twice; compare exact bytes"
        ),
        "input_bindings_sha256": _sha256(_canonical_json_bytes(bindings)),
        "runtime_binding_sha256": _sha256(_canonical_json_bytes(runtime)),
        "report_sha256": _sha256(report_payload),
        "publication_contract_sha256": _sha256(
            _canonical_json_bytes(manifest_without_receipt["publication_contract"])
        ),
    }
    manifest_without_receipt["deterministic_reconstruction_receipt"] = {
        "payload": receipt_payload,
        "sha256": _sha256(_canonical_json_bytes(receipt_payload)),
    }
    manifest_payload = _canonical_json_bytes(manifest_without_receipt)
    return _ArtifactBundle(
        files={REPORT_FILENAME: report_payload, MANIFEST_FILENAME: manifest_payload},
        report_payload=report_payload,
        manifest_payload=manifest_payload,
        _token=_BUNDLE_TOKEN,
    )


def _parse_canonical_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except Exception as exc:
        raise LineageAuthorityError(f"{label} is not JSON") from exc
    _require(type(document) is dict, f"{label} root is not an object")
    _require(_canonical_json_bytes(document) == payload, f"{label} is not canonical JSON")
    return document


def _verify_bundle(bundle: _ArtifactBundle) -> None:
    _require(type(bundle) is _ArtifactBundle, "publication bundle type was forged")
    _require(bundle._token is _BUNDLE_TOKEN, "publication bundle issuer token changed")
    _require(type(bundle.files) is type(MappingProxyType({})), "bundle mapping is mutable")
    _require(
        set(bundle.files) == {REPORT_FILENAME, MANIFEST_FILENAME},
        "publication file set changed",
    )
    _require(bundle.files[REPORT_FILENAME] == bundle.report_payload, "report payload changed")
    _require(bundle.files[MANIFEST_FILENAME] == bundle.manifest_payload, "manifest payload changed")
    report = _parse_canonical_json(bundle.report_payload, label="lineage report")
    manifest = _parse_canonical_json(bundle.manifest_payload, label="lineage manifest")
    _require(report.get("status") == STATUS and manifest.get("status") == STATUS, "status changed")
    output = manifest.get("output")
    _require(type(output) is dict, "manifest output binding is absent")
    _require(output.get("name") == REPORT_FILENAME, "manifest report filename changed")
    _require(output.get("sha256") == _sha256(bundle.report_payload), "report hash binding changed")
    _require(output.get("size_bytes") == len(bundle.report_payload), "report size binding changed")
    _require(
        report.get("evaluation_inventory", {}).get("authority_role")
        == "formal_reportable_key_registry",
        "formal evaluation authority role changed",
    )
    _require(
        report.get("legacy_forecast_registry_inventory", {}).get("scientific_evaluation_domain")
        is False,
        "legacy registry was promoted to a scientific evaluation domain",
    )
    _validate_declared_scope(
        report["withdrawal"]["artifact_paths"],
        report["withdrawal"]["claims"],
        report["preserved"]["scope"],
    )


def _verify_two_pass(first: _ArtifactBundle, second: _ArtifactBundle) -> None:
    _verify_bundle(first)
    _verify_bundle(second)
    _require(set(first.files) == set(second.files), "two-pass file set differs")
    for name in first.files:
        _require(first.files[name] == second.files[name], f"two-pass bytes differ for {name}")


def build_twice(
    paths: LineageInputPaths,
    *,
    config: BuildConfig,
    repository_root: Path,
) -> _ArtifactBundle:
    first = build_artifacts(paths, config=config, repository_root=repository_root)
    second = build_artifacts(paths, config=config, repository_root=repository_root)
    _verify_two_pass(first, second)
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
    """Require a lexical directory path still names the opened directory."""

    opened = os.fstat(descriptor)
    _require(stat.S_ISDIR(opened.st_mode), f"opened {label} is no longer a directory")
    _require(
        _directory_identity(opened) == expected_identity,
        f"opened {label} inode changed",
    )
    try:
        lexical = path.lstat()
    except OSError as exc:
        raise LineageAuthorityError(f"lexical {label} disappeared: {path}") from exc
    _require(
        not stat.S_ISLNK(lexical.st_mode)
        and stat.S_ISDIR(lexical.st_mode)
        and _directory_identity(lexical) == expected_identity,
        f"lexical {label} no longer names the anchored directory: {path}",
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
        raise LineageAuthorityError(f"cannot anchor {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        _require(stat.S_ISDIR(opened.st_mode), f"anchored {label} is not a directory")
        identity = _directory_identity(opened)
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


def _read_stable_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
) -> bytes:
    _require_simple_name(name, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise LineageAuthorityError(f"cannot safely open {label}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        and len(payload) == before.st_size,
        f"{label} changed while being verified",
    )
    entry = _entry_status_at(parent_descriptor, name)
    _require(
        entry is not None
        and stat.S_ISREG(entry.st_mode)
        and _directory_identity(entry) == _directory_identity(before),
        f"{label} directory entry changed while being verified",
    )
    return payload


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
            _require(written > 0, f"short zero-byte write for staged {name}")
            offset += written
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        _require(
            stat.S_ISREG(status.st_mode) and status.st_size == len(payload),
            f"staged {name} size/type changed",
        )
    finally:
        os.close(descriptor)


def _create_staging_directory_at(
    root_descriptor: int,
    *,
    destination_name: str,
) -> tuple[str, int, tuple[int, int]]:
    for _attempt in range(128):
        name = f".{destination_name}.staging.{secrets.token_hex(12)}"
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
        except Exception:
            os.rmdir(name, dir_fd=root_descriptor)
            raise
        try:
            status = os.fstat(descriptor)
            _require(stat.S_ISDIR(status.st_mode), "staging entry is not a directory")
        except Exception:
            os.close(descriptor)
            os.rmdir(name, dir_fd=root_descriptor)
            raise
        return name, descriptor, _directory_identity(status)
    raise LineageAuthorityError("cannot allocate an exclusive staging directory")


def _remove_tree_at(parent_descriptor: int, name: str) -> None:
    """Remove one anchored entry without following symlinks."""

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
    _require(renameat2 is not None, "atomic no-replace directory publication is unavailable")
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
            raise LineageAuthorityError(
                f"refusing to overwrite existing authority entry: {destination_name}"
            )
        raise LineageAuthorityError(
            f"atomic no-replace publication failed ({os.strerror(error)}): {destination_name}"
        )


def publish_authority(
    paths: LineageInputPaths,
    destination: Path,
    *,
    config: BuildConfig,
    repository_root: Path,
    allowed_output_root: Path,
) -> tuple[Path, _ArtifactBundle]:
    """Closed two-pass create-only publisher; caller bundles are never accepted."""

    _require(type(paths) is LineageInputPaths, "publisher accepts LineageInputPaths only")
    _require(type(config) is BuildConfig, "publisher accepts BuildConfig only")
    root = _absolute(allowed_output_root)
    destination = _absolute(destination)
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise LineageAuthorityError("authority output escapes allowed root") from exc
    _require(len(relative.parts) == 1, "authority output must be one direct child of output root")
    _require(destination.name not in {"", ".", ".."}, "invalid authority output name")
    if config.mode == "production":
        _validate_production_config(config)
        _require(
            _absolute(repository_root) == _absolute(ROOT),
            "production repository root changed",
        )
        _require(root == _absolute(FINAL), "production allowed output root changed")
        _require(
            destination == _absolute(FINAL / DEFAULT_OUTPUT_NAME),
            "production output directory changed",
        )
    root_descriptor, root_identity = _open_anchored_directory(
        root,
        label="allowed output root",
    )
    destination_name = destination.name
    lock_name = f".{destination_name}.create.lock"
    lock_identity: tuple[int, int] | None = None
    staging_name: str | None = None
    staging_descriptor: int | None = None
    staging_identity: tuple[int, int] | None = None
    bundle: _ArtifactBundle | None = None
    committed = False
    try:
        _require(
            _entry_status_at(root_descriptor, destination_name) is None,
            f"refusing to overwrite existing authority: {destination}",
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
                dir_fd=root_descriptor,
            )
        except FileExistsError as exc:
            raise LineageAuthorityError(
                f"another create-only publication is active: {lock_name}"
            ) from exc
        lock_status = os.fstat(lock_descriptor)
        lock_identity = _directory_identity(lock_status)
        lock_payload = f"pid={os.getpid()}\n".encode("ascii")
        _require(
            os.write(lock_descriptor, lock_payload) == len(lock_payload),
            "publication lock write was short",
        )
        os.fsync(lock_descriptor)
        os.close(lock_descriptor)
        lock_descriptor = -1
        os.fsync(root_descriptor)
        bundle = build_twice(paths, config=config, repository_root=repository_root)
        staging_name, staging_descriptor, staging_identity = _create_staging_directory_at(
            root_descriptor,
            destination_name=destination_name,
        )
        for name, payload in sorted(bundle.files.items()):
            _write_bytes_at(staging_descriptor, name, payload)
        os.fsync(staging_descriptor)
        _verify_bundle(bundle)
        _require(
            set(os.listdir(staging_descriptor)) == set(bundle.files),
            "staged file set changed",
        )
        for name, payload in bundle.files.items():
            staged_payload = _read_stable_regular_at(
                staging_descriptor,
                name,
                label=f"staged {name}",
            )
            _require(staged_payload == payload, f"staged bytes changed for {name}")
        _verify_anchored_directory_path(
            root,
            root_descriptor,
            root_identity,
            label="allowed output root before publication",
        )
        _require(
            _entry_status_at(root_descriptor, destination_name) is None,
            "authority destination appeared during build",
        )
        _rename_directory_noreplace_at(
            root_descriptor,
            staging_name,
            root_descriptor,
            destination_name,
        )
        _verify_anchored_directory_path(
            root,
            root_descriptor,
            root_identity,
            label="allowed output root after publication",
        )
        destination_status = _entry_status_at(root_descriptor, destination_name)
        _require(
            destination_status is not None
            and stat.S_ISDIR(destination_status.st_mode)
            and _directory_identity(destination_status) == staging_identity,
            "published destination is not the staged directory inode",
        )
        _require(
            set(os.listdir(staging_descriptor)) == set(bundle.files),
            "published file set changed after rename",
        )
        for name, payload in bundle.files.items():
            _require(
                _read_stable_regular_at(
                    staging_descriptor,
                    name,
                    label=f"published {name}",
                )
                == payload,
                f"published bytes changed for {name}",
            )
        _remove_owned_entry_at(
            root_descriptor,
            lock_name,
            lock_identity,
            label="publication lock",
            strict=True,
        )
        lock_identity = None
        os.fsync(root_descriptor)
        _verify_anchored_directory_path(
            root,
            root_descriptor,
            root_identity,
            label="allowed output root at commit",
        )
        committed = True
    finally:
        try:
            if "lock_descriptor" in locals() and lock_descriptor >= 0:
                try:
                    os.close(lock_descriptor)
                except OSError:
                    pass
                lock_descriptor = -1
            if staging_descriptor is not None:
                try:
                    os.close(staging_descriptor)
                except OSError:
                    pass
                staging_descriptor = None
            if not committed and staging_identity is not None:
                # The stage may still have its random name, or it may already
                # have been renamed when a post-rename root check failed.
                if staging_name is not None:
                    _remove_owned_entry_at(
                        root_descriptor,
                        staging_name,
                        staging_identity,
                        label="staging directory",
                    )
                _remove_owned_entry_at(
                    root_descriptor,
                    destination_name,
                    staging_identity,
                    label="failed published destination",
                )
            if lock_identity is not None:
                _remove_owned_entry_at(
                    root_descriptor,
                    lock_name,
                    lock_identity,
                    label="publication lock",
                )
            if not committed or lock_identity is not None:
                os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
    _require(bundle is not None, "publication completed without an artifact bundle")
    return destination, bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="atomically create the formal authority directory; default is a dry run",
    )
    parser.add_argument("--output-dir", type=Path, help="required only with --publish")
    parser.add_argument(
        "--authorize-withdrawal-publication",
        action="store_true",
        help="required with --publish; acknowledges creation of a formal withdrawal authority",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.publish and args.output_dir is None:
        parser.error("--output-dir is required with --publish")
    if not args.publish and args.output_dir is not None:
        parser.error("--output-dir is accepted only with --publish")
    if args.publish and not args.authorize_withdrawal_publication:
        parser.error("--publish requires --authorize-withdrawal-publication")
    if not args.publish and args.authorize_withdrawal_publication:
        parser.error("--authorize-withdrawal-publication is accepted only with --publish")

    paths = LineageInputPaths.production(ROOT)
    config = BuildConfig.production()
    if args.publish:
        destination, bundle = publish_authority(
            paths,
            args.output_dir,
            config=config,
            repository_root=ROOT,
            allowed_output_root=FINAL,
        )
    else:
        bundle = build_twice(paths, config=config, repository_root=ROOT)
        destination = None
    manifest = _parse_canonical_json(bundle.manifest_payload, label="lineage manifest")
    report = _parse_canonical_json(bundle.report_payload, label="lineage report")
    result: dict[str, Any] = {
        "status": "PUBLISHED_CREATE_ONLY" if args.publish else "DRY_RUN_VALIDATED_NOT_PUBLISHED",
        "scientific_status": STATUS,
        "report_sha256": manifest["output"]["sha256"],
        "manifest_sha256": _sha256(bundle.manifest_payload),
        "combined_panel_rows": report["raw_panel_inventory"]["combined_rows"],
        "evaluation_registry_rows": sum(
            item["registry_rows"] for item in report["evaluation_inventory"]["by_horizon"].values()
        ),
        "legacy_forecast_registry_rows": sum(
            item["registry_rows"]
            for item in report["legacy_forecast_registry_inventory"]["by_horizon"].values()
        ),
        "evaluation_mask_error_rows": sum(
            item["mask_error_rows"]
            for item in report["evaluation_inventory"]["by_horizon"].values()
        ),
        "prediction_or_score_rows_read": 0,
    }
    if destination is not None:
        result["output_dir"] = str(destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LineageAuthorityError as exc:
        print(f"lineage defect authority build failed closed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
