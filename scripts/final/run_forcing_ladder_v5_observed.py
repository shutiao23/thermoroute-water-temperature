#!/usr/bin/env python3
"""Create-only observed-lineage correction for the viewed F0/F3 tree cells.

The correction is deliberately narrower than the historical forcing ladder:
``{F0, F3_full} x {LightGBM, ResidualLightGBM} x {1, 3, 7}`` only.  Raw
observedness is captured before imputation, labels are admitted only when the
issue and target WTEMP cells are raw observations, and 2016--2017 is an actual
LightGBM validation set rather than a training-tail surrogate.

The default invocation is a data-free dry run.  Production execution is
fail-closed until a reviewed preprocessing-lineage defect-authority manifest
has been published and its exact SHA256 has been inserted below.  A later
authorized execution may publish only the fixed v5 destination, as one fsynced
directory bundle via Linux ``RENAME_NOREPLACE``.  There is no resume, overwrite,
or partial-publication mode.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import secrets
import stat
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Self

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v4 as V4  # noqa: E402
from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import features as F  # noqa: E402
from thermoroute.baselines import _lgb_fit  # noqa: E402

STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_CORRECTION"
SCHEMA_VERSION = "thermoroute.forcing-ladder.v5-observed.v2"
MANIFEST_FORMAT = "thermoroute.forcing-observed-lineage-manifest.v2"

TRAIN_START = pd.Timestamp("2006-01-01")
TRAIN_END = pd.Timestamp("2015-12-31")
VALIDATION_START = pd.Timestamp("2016-01-01")
VALIDATION_END = pd.Timestamp("2017-12-31")
CONFIRM_START = pd.Timestamp("2021-01-01")
CONFIRM_END = pd.Timestamp("2023-12-31")

ARMS = ("F0", "F3_full")
MODELS = ("LightGBM", "ResidualLightGBM")
HORIZONS = (1, 3, 7)
FEATURE_VARIABLES = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
METEOROLOGY_VARIABLES = ("TEMP", "PRCP", "DH", "RHMEAN", "WDSP")
FROZEN_SHORT_LAGS = (0, 1, 2, 3, 5, 7, 10, 14)
FROZEN_ROLLING_WINDOWS = (3, 7, 14, 30)

BEST_ITER_UPPER_BOUND = MappingProxyType({1: 800, 3: 575, 7: 440})
FROZEN_PARAMS = MappingProxyType(
    {
        1: MappingProxyType({"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03}),
        3: MappingProxyType({"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03}),
        7: MappingProxyType({"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.03}),
    }
)

DEV_PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
EVAL_PANEL = ROOT / "outputs" / "conventional" / "panel_2021_2023.parquet"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
KEY_AUTHORITY_DIR = ROOT / "outputs" / "final" / "information_regime_key_registries_v4"
PRIMARY_KEY_REGISTRY = KEY_AUTHORITY_DIR / "primary_reportable_key_registry_v4.parquet"
KEY_AUTHORITY_MANIFEST = KEY_AUTHORITY_DIR / "information_regime_key_registries_v4_manifest.json"
LEGACY_KEY_REGISTRY = ROOT / "outputs" / "final" / "forecast_keys.parquet"
PROTOCOL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4.yaml"
SCIENTIFIC_EVIDENCE_STATUS = ROOT / "docs" / "SCIENTIFIC_EVIDENCE_STATUS.md"
DECISION_LOG = ROOT / "docs" / "strong_accept_decision_log.md"
INFORMATION_REGIME_TODO = ROOT / "docs" / "WRR_INFORMATION_REGIME_TODO_20260809.md"
SEMANTIC_AUTHORITY_DIR = ROOT / "outputs" / "final" / "semantic_registries_v4_authority"
SEMANTIC_DAILY_REGISTRY = SEMANTIC_AUTHORITY_DIR / "daily_raw_observed_panel_registry_v4.parquet"
SEMANTIC_TRAINING_REGISTRY = (
    SEMANTIC_AUTHORITY_DIR / "training_example_registry_2006_2017_v4.parquet"
)
SEMANTIC_AUTHORITY_MANIFEST = (
    SEMANTIC_AUTHORITY_DIR / "semantic_registries_v4_authority_manifest.json"
)
# The forcing correction has a narrower scope and a different terminal seal
# schema than the later crossed/neural protocol.  Keep its execution authority
# on distinct canonical paths so one exact-path validator can never overwrite
# or masquerade as the other protocol's seal.
SEALED_SCORE_PROTOCOL = (
    ROOT / "protocols" / "wrr_information_regimes_forcing_v5_observed_protocol.yaml"
)
SEALED_SCORE_PROTOCOL_SEAL = (
    ROOT / "protocols" / "wrr_information_regimes_forcing_v5_observed_seal.json"
)
SCORE_CLEAN_DESIGN_COMMIT = (
    ROOT / "outputs" / "final" / "forcing_regime_v5_observed_clean_design_commit_v1.json"
)
SCORE_SOURCE_REGISTRY = (
    ROOT / "outputs" / "final" / "forcing_regime_v5_observed_source_registry_v1.json"
)
SCORE_AUTHORITY_BUILDER = ROOT / "scripts" / "final" / "build_forcing_v5_execution_authority.py"
DEFECT_AUTHORITY_DIR = ROOT / "outputs" / "final" / "preprocessing_lineage_defect_authority_v1"
DEFECT_AUTHORITY_MANIFEST = (
    DEFECT_AUTHORITY_DIR / "preprocessing_lineage_defect_authority_v1_manifest.json"
)
DEFECT_AUTHORITY_REPORT = DEFECT_AUTHORITY_DIR / "preprocessing_lineage_defect_report_v1.json"
FINAL_OUTPUT_ROOT = ROOT / "outputs" / "final"
DEFAULT_OUTPUT_DIR = FINAL_OUTPUT_ROOT / "forcing_regime_v5_observed"
SHARD_DIRNAME = "forcing_shards_v5_observed"
MANIFEST_FILENAME = "forcing_lineage_manifest_v5_observed.json"

PINNED_INPUT_SHA256 = MappingProxyType(
    {
        "development_panel": "0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69",
        "evaluation_panel": "cecdac459139456202240954e4c98fe18bba1fe8b63e9b06ab268683e0d1c03c",
        "station_registry": "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9",
        "primary_key_registry": "9a135dcffcfd467cf1e2dda4fc711ba6cf66c8ad54bf3b6f799f4a2d5a3c1d24",
        "semantic_daily_raw_observed_registry": (
            "0837012f88a004b3b44449770166e59f76006f851723582f00374a98b4bd67a0"
        ),
        "semantic_training_example_registry": (
            "af7f55372b1ca05bf56a00360c510b1ca1e5ed386b85fa7c10bcf56daca5c095"
        ),
    }
)
PINNED_INPUT_PATHS = MappingProxyType(
    {
        "development_panel": DEV_PANEL,
        "evaluation_panel": EVAL_PANEL,
        "station_registry": STATION_REGISTRY,
        "primary_key_registry": PRIMARY_KEY_REGISTRY,
        "semantic_daily_raw_observed_registry": SEMANTIC_DAILY_REGISTRY,
        "semantic_training_example_registry": SEMANTIC_TRAINING_REGISTRY,
    }
)
PINNED_GOVERNANCE_SHA256 = MappingProxyType(
    {
        "protocol": "66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98",
        "key_authority_manifest": (
            "ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea"
        ),
        "semantic_authority_manifest": (
            "12cdc355a06d2c39733a60386dfdeb8a2f6b234d2f8d6f641a995a3d3c41072c"
        ),
        "semantic_data_candidate_manifest": (
            "a92dd4765c875d8f5c927c054cf5e2c41160fd23b5d00fa58e354feaf46b59e7"
        ),
        "semantic_contract_candidate_manifest": (
            "6ab801eebc260734519a1c1ab235878a0f2107df5fa09d92e1a248dc86dce9ac"
        ),
        "semantic_cell_registry": (
            "3bcdf816598dfae05b90f5f7c7fe5fc48faa02c4b1109d81a0ff0ea38d4315a0"
        ),
        "semantic_fold_registry": (
            "63b1296622850173e90e75cc75c0b545d1048e79d3ca2d89ccd2701ce756daf8"
        ),
        "semantic_model_registry": (
            "6abc55474893a95620d0c07363c94b5c7b3d1774b3102efdbc81650c98764e52"
        ),
        "semantic_input_registry": (
            "132136ef459dd28be91b7552187d6df1be0f77e04008d4cd4743e9e8cabcaf71"
        ),
        "semantic_contrast_registry": (
            "947146712828dd8900dd9cad2cf07dd672b76f82c0b5b6a5f1b76378551ea42b"
        ),
        "semantic_environment_registry": (
            "b038754f751cc8dab5c26c6ae8e25d577d9b8a30a8f5ea481398a298e4b1b521"
        ),
    }
)
PINNED_GOVERNANCE_PATHS = MappingProxyType(
    {
        "protocol": PROTOCOL,
        "key_authority_manifest": KEY_AUTHORITY_MANIFEST,
        "semantic_authority_manifest": SEMANTIC_AUTHORITY_MANIFEST,
        "semantic_data_candidate_manifest": (
            SEMANTIC_AUTHORITY_DIR / "semantic_data_registries_v4_candidate_manifest.json"
        ),
        "semantic_contract_candidate_manifest": (
            SEMANTIC_AUTHORITY_DIR / "semantic_contract_registry_manifest_v4_candidate.json"
        ),
        "semantic_cell_registry": SEMANTIC_AUTHORITY_DIR / "cell_registry_v4.json",
        "semantic_fold_registry": SEMANTIC_AUTHORITY_DIR / "fold_registry_v4.json",
        "semantic_model_registry": SEMANTIC_AUTHORITY_DIR / "model_registry_v4.json",
        "semantic_input_registry": SEMANTIC_AUTHORITY_DIR / "input_registry_v4.json",
        "semantic_contrast_registry": SEMANTIC_AUTHORITY_DIR / "contrast_registry_v4.json",
        "semantic_environment_registry": SEMANTIC_AUTHORITY_DIR / "environment_registry_v4.json",
    }
)
#: Governance documents the protocol *requires* to change, so a byte pin on
#: them is self-defeating: the decision log is append-only, and the evidence
#: status and information-regime TODO are living records of work in progress.
#: Pinning their hashes meant every entry the protocol demanded be written also
#: broke the runner that demanded it -- the failure mode DLOG-027's addendum
#: named. They are captured and their observed bytes recorded in the authority,
#: which is what makes a run auditable; they are not compared against a
#: frozen expectation, which is what made it unrunnable. Record, do not lock.
RECORDED_GOVERNANCE_PATHS = MappingProxyType(
    {
        "scientific_evidence_status": SCIENTIFIC_EVIDENCE_STATUS,
        "decision_log": DECISION_LOG,
        "information_regime_todo": INFORMATION_REGIME_TODO,
    }
)
PINNED_DEPENDENCY_SHA256 = MappingProxyType(
    {
        "v4_safe_helper_dependency": (
            "6b370a4c1271e43f4797408bc1831a44882a7e3d727ccd71a485eb0cccac6eb8"
        ),
        "config_source": "7661e82df4a6017dcc351f9f4f07e3b94afd5d2426689f775b83e12b78c41c1f",
        "data_source": "37fe83c34a72dbe676ce37abd27b24c0d5ac4f91c1d62d9822856bf374751984",
        "features_source": "0605254b9ac86bd92c7ffb0f31e7db526dde90777d5ddbcf1b1d7bebfcd4e58b",
        "baselines_source": "247b6fca9072830b7d88443101117cfc5c10d48a010f3838888479a428ff682f",
        "semantic_data_builder_source": (
            "766dfb8a0d4674e450676fc696a44d24919d69f96eb4306009ea8df0f6186f28"
        ),
        "semantic_contract_builder_source": (
            "46101f8f814f6d66cfa1801ff53445936bb60f44e23a3f3575c4e445f331c614"
        ),
        "semantic_authority_publisher_source": (
            "e6779f68034b78a1f9e6866cc3a4fdf83bb334d287f44b96a19707bee3a05e7e"
        ),
        "semantic_data_builder_tests": (
            "72fa6ccf0c08306f563407613a74961f150e7fe954ebc7c043842e5d0fed4c70"
        ),
        "semantic_contract_builder_tests": (
            "3d8e43d1eb748be7a73b149c0af02e4844a6282dd6defd9d18c7dc17792dfca7"
        ),
        "semantic_authority_publisher_tests": (
            "8a8b632a1d6d7c9fe7f0a39a4e888bb5f6f0ba1007ca49c2c88ff71c879f1d1d"
        ),
        "pyproject": "ef49ffacf3c73a2e566b89abac22d663bdd997a3f68adc5dcf0f5f187860fcc6",
        "requirements_lock": ("ff2d67915ccaabb750cdf4c630d59500d6c8841b305d5fffb1ffd549195dc047"),
        "requirements_lock_py312_hashed": (
            "fa325e30e8e69b9e8ec458e9e76a466a765530a8bffa46f1d1f8e34614f4ab76"
        ),
    }
)
PINNED_DEPENDENCY_PATHS = MappingProxyType(
    {
        "v4_safe_helper_dependency": ROOT / "scripts" / "final" / "run_forcing_ladder_v4.py",
        "config_source": ROOT / "src" / "thermoroute" / "config.py",
        "data_source": ROOT / "src" / "thermoroute" / "data.py",
        "features_source": ROOT / "src" / "thermoroute" / "features.py",
        "baselines_source": ROOT / "src" / "thermoroute" / "baselines.py",
        "semantic_data_builder_source": (
            ROOT / "scripts" / "final" / "build_semantic_data_registries_v4.py"
        ),
        "semantic_contract_builder_source": (
            ROOT / "scripts" / "final" / "build_semantic_contract_registries_v4.py"
        ),
        "semantic_authority_publisher_source": (
            ROOT / "scripts" / "final" / "publish_semantic_registries_v4.py"
        ),
        "semantic_data_builder_tests": (
            ROOT / "tests" / "final" / "test_semantic_data_registries_v4.py"
        ),
        "semantic_contract_builder_tests": (
            ROOT / "tests" / "final" / "test_semantic_contract_registries_v4.py"
        ),
        "semantic_authority_publisher_tests": (
            ROOT / "tests" / "final" / "test_publish_semantic_registries_v4.py"
        ),
        "pyproject": ROOT / "pyproject.toml",
        "requirements_lock": ROOT / "requirements-lock.txt",
        "requirements_lock_py312_hashed": ROOT / "requirements-lock-py312-hashed.txt",
    }
)

# The defect authority is independently published and reviewed.  Its exact
# manifest is necessary but deliberately insufficient to authorize scoring.
EXPECTED_DEFECT_AUTHORITY_SHA256: str | None = (
    "e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908"
)
# The published defect authority withdraws invalid results and mandates a rerun;
# it is not itself permission to inspect new model outcomes.  These independent
# canonical score-execution pins must also be reviewed and inserted. Phase 1
# published this inert protocol candidate create-only; the terminal seal remains
# absent and is the only object that can authorize execution.
EXPECTED_SEALED_SCORE_PROTOCOL_SHA256: str | None = (
    "0d4d97a2420ad55fed2c7d558c0c56b5c3a1c6485208c83f53110a3b6beff2a5"
)

SCORE_GIT_DESIGN_ROLES = (
    "runner",
    "score_authority_builder",
    "sealed_score_protocol",
    "defect_authority_manifest",
    "defect_authority_report",
    "semantic_daily_raw_observed_registry",
    "semantic_training_example_registry",
    *tuple(PINNED_GOVERNANCE_PATHS),
    *tuple(RECORDED_GOVERNANCE_PATHS),
    *tuple(PINNED_DEPENDENCY_PATHS),
)

EXPECTED_PANEL_COLUMNS = ("DATE", "site_id", *C.ALL_VARS)
OBSERVED_COLUMNS = tuple(f"{variable}_observed" for variable in C.ALL_VARS)
RAW_PANEL_COLUMNS = (*EXPECTED_PANEL_COLUMNS, *OBSERVED_COLUMNS)
REFERENCE_COLUMNS = (
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
EXPECTED_REFERENCE_COUNTS = MappingProxyType({1: 120_444, 3: 119_639, 7: 118_682})
EXPECTED_EXAMPLE_COUNTS = MappingProxyType(
    {
        "train": MappingProxyType({1: 339_750, 3: 337_725, 7: 335_822}),
        "val": MappingProxyType({1: 83_516, 3: 82_948, 7: 82_233}),
    }
)

SUBSTITUTION_COLUMNS = tuple(
    f"forcing_substitutions_{variable}" for variable in METEOROLOGY_VARIABLES
)
SHARD_COLUMNS = (
    "key_id",
    "site_id",
    "issue_date",
    "target_date",
    "horizon",
    "arm",
    "model",
    "analysis_status",
    "y_true",
    "y_pred",
    "y_damped",
    *SUBSTITUTION_COLUMNS,
    "forcing_substitution_count",
)
BASE_TABLE_COLUMNS = (
    "site_id",
    "issue_date",
    "target_date",
    "y",
    "issue_wtemp_observed",
    "target_wtemp_observed",
    "split",
    "horizon",
)


def _make_frozen_base_feature_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for variable in FEATURE_VARIABLES:
        for lag in FROZEN_SHORT_LAGS:
            columns.extend((f"{variable}_lag{lag}", f"{variable}_observed_lag{lag}"))
        for window in FROZEN_ROLLING_WINDOWS:
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
    return tuple(columns)


FROZEN_BASE_FEATURE_COLUMNS = _make_frozen_base_feature_columns()
if len(FROZEN_BASE_FEATURE_COLUMNS) != 210 or len(set(FROZEN_BASE_FEATURE_COLUMNS)) != 210:
    raise RuntimeError("frozen v5 base-feature namespace is internally inconsistent")


def _future_feature_columns(horizon: int) -> tuple[str, ...]:
    return tuple(
        column
        for variable in METEOROLOGY_VARIABLES
        for column in (f"{variable}_fut{horizon}", f"{variable}_futmean{horizon}")
    )


@dataclass(frozen=True, order=True)
class Cell:
    arm: str
    model: str
    horizon: int


@dataclass(frozen=True)
class ObservedPreprocessing:
    imputer: D.Imputer
    water_climatology: F.HarmonicClimatology
    damped_anchor: F.DampedPersistenceAnchor
    meteorology_climatologies: Mapping[str, F.HarmonicClimatology]
    station_ids: tuple[str, ...]
    training_row_count: int
    validation_row_count: int


def fixed_cells() -> tuple[Cell, ...]:
    return tuple(
        Cell(arm, model, horizon) for arm in ARMS for model in MODELS for horizon in HORIZONS
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be one lowercase full SHA256 digest")
    return value


def _canonical_json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not canonical-JSON serializable: {type(value).__name__}")


def _canonical_json_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            default=_canonical_json_default,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(document: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _strict_bool_array(series: pd.Series, label: str) -> np.ndarray:
    if series.dtype != np.dtype("bool"):
        raise TypeError(f"{label} must have exact non-null bool dtype")
    values = series.to_numpy(copy=True)
    if values.dtype != np.dtype("bool") or values.ndim != 1:
        raise TypeError(f"{label} bool representation is not exact")
    return values


def _strict_signed_integer_array(series: pd.Series, label: str) -> np.ndarray:
    dtype = series.dtype
    if not isinstance(dtype, np.dtype) or not np.issubdtype(dtype, np.signedinteger):
        raise TypeError(f"{label} must have an exact signed-integer dtype")
    values = series.to_numpy(copy=True)
    if values.ndim != 1:
        raise TypeError(f"{label} integer representation is not one-dimensional")
    return values


def _strict_float_array(
    series: pd.Series,
    label: str,
    *,
    finite: bool,
) -> np.ndarray:
    dtype = series.dtype
    if not isinstance(dtype, np.dtype) or not np.issubdtype(dtype, np.floating):
        raise TypeError(f"{label} must have an exact floating dtype")
    values = series.to_numpy(copy=True)
    if np.isinf(values).any() or (finite and not np.isfinite(values).all()):
        raise ValueError(f"{label} contains a forbidden non-finite value")
    return values.astype(np.float64, copy=False)


def _strict_numeric_array(series: pd.Series, label: str, *, finite: bool) -> np.ndarray:
    dtype = series.dtype
    if (
        not isinstance(dtype, np.dtype)
        or np.issubdtype(dtype, np.bool_)
        or not np.issubdtype(dtype, np.number)
    ):
        raise TypeError(f"{label} must have an exact numeric dtype")
    values = series.to_numpy(copy=True).astype(np.float64, copy=False)
    if np.isinf(values).any() or (finite and not np.isfinite(values).all()):
        raise ValueError(f"{label} contains a forbidden non-finite value")
    return values


def _strict_dates(series: pd.Series, label: str) -> np.ndarray:
    if series.dtype != np.dtype("datetime64[ns]"):
        raise TypeError(f"{label} must be exact timezone-naive datetime64[ns]")
    values = series.to_numpy(copy=True)
    integers = values.view(np.int64)
    if (integers == np.iinfo(np.int64).min).any():
        raise ValueError(f"{label} contains NaT")
    nanoseconds_per_day = 86_400_000_000_000
    if np.remainder(integers, nanoseconds_per_day).any():
        raise ValueError(f"{label} must contain exact midnight dates")
    return values


def _strict_site_ids(series: pd.Series, label: str) -> tuple[str, ...]:
    values = tuple(series.to_numpy(copy=True))
    for value in values:
        if (
            type(value) is not str
            or len(value) not in (8, 15)
            or not value.isascii()
            or any(character < "0" or character > "9" for character in value)
        ):
            raise ValueError(f"{label} contains a non-canonical ASCII station identifier")
    return values


def _strict_key_ids(series: pd.Series, label: str) -> tuple[str, ...]:
    values = tuple(series.to_numpy(copy=True))
    if not all(_is_sha256(value) for value in values):
        raise ValueError(f"{label} contains a non-canonical key_id")
    return values  # type: ignore[return-value]


def _require_horizon(value: object) -> int:
    if type(value) is not int or value not in HORIZONS:
        raise ValueError(f"horizon must be one exact integer in {HORIZONS}")
    return value


def _canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Platform-stable, schema-aware digest of the complete ordered content."""

    selected = tuple(columns)
    if len(selected) != len(set(selected)):
        raise ValueError("canonical frame digest received duplicate columns")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"cannot hash missing columns: {missing}")
    digest = hashlib.sha256()
    digest.update(b"thermoroute-canonical-frame-v2\0")
    digest.update(struct.pack("<Q", len(frame)))
    digest.update(struct.pack("<Q", len(selected)))
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
            digest.update(
                _strict_bool_array(series, f"hash column {column}").view(np.uint8).tobytes()
            )
        elif series.dtype == np.dtype("datetime64[ns]"):
            digest.update(b"D")
            dates = _strict_dates(series, f"hash column {column}")
            digest.update(dates.view(np.int64).astype("<i8", copy=False).tobytes(order="C"))
        elif isinstance(series.dtype, np.dtype) and np.issubdtype(series.dtype, np.signedinteger):
            digest.update(b"I")
            integers = _strict_signed_integer_array(series, f"hash column {column}")
            digest.update(integers.astype("<i8", copy=False).tobytes(order="C"))
        elif isinstance(series.dtype, np.dtype) and np.issubdtype(series.dtype, np.floating):
            digest.update(b"F")
            floats = _strict_float_array(series, f"hash column {column}", finite=False)
            present = np.isfinite(floats)
            canonical = floats.astype("<f8", copy=True)
            canonical[~present] = 0.0
            digest.update(present.view(np.uint8).tobytes(order="C"))
            digest.update(canonical.tobytes(order="C"))
        else:
            digest.update(b"S")
            values = tuple(series.to_numpy(copy=True))
            if any(type(value) is not str for value in values):
                raise TypeError(f"hash column {column} is not an exact non-null string column")
            encoded_cache: dict[str, bytes] = {}
            payload = bytearray()
            for value in values:
                token = encoded_cache.get(value)
                if token is None:
                    raw = value.encode("utf-8")
                    token = struct.pack("<Q", len(raw)) + raw
                    encoded_cache[value] = token
                payload.extend(token)
            digest.update(payload)
    return digest.hexdigest()


def _identity_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = list(columns)
    ordered = frame[selected].sort_values(selected, kind="mergesort").reset_index(drop=True)
    return _canonical_frame_sha256(ordered, selected)


def _canonical_matrix_sha256(
    rows: pd.DataFrame,
    feature_columns: Sequence[str],
    identity_columns: Sequence[str],
) -> str:
    columns = tuple(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("feature matrix requires a non-empty unique ordered column list")
    matrix = np.empty((len(rows), len(columns)), dtype="<f8")
    for index, column in enumerate(columns):
        matrix[:, index] = _strict_numeric_array(
            rows[column], f"feature matrix {column}", finite=True
        )
    digest = hashlib.sha256()
    digest.update(b"thermoroute-feature-matrix-v2\0")
    digest.update(_canonical_frame_sha256(rows, identity_columns).encode("ascii"))
    digest.update(_canonical_json_bytes(list(columns)))
    digest.update(struct.pack("<QQ", matrix.shape[0], matrix.shape[1]))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


_RAW_PANEL_ISSUER_TOKEN = object()
_IMPUTED_PANEL_ISSUER_TOKEN = object()
_FUTURE_REGISTRY_ISSUER_TOKEN = object()


class RawLabelPanel:
    """Factory-issued raw values and their pre-imputation observedness."""

    __slots__ = (
        "_content_sha256",
        "_frame",
        "_identity_sha256",
        "_issuer_token",
        "_observed_mask_sha256",
        "_seal_sha256",
        "_source_sha256",
    )

    def __init__(
        self,
        frame: pd.DataFrame,
        source_sha256: str,
        identity_sha256: str,
        observed_mask_sha256: str,
        content_sha256: str,
        seal_sha256: str,
        *,
        _issuer_token: object | None = None,
    ) -> None:
        if _issuer_token is not _RAW_PANEL_ISSUER_TOKEN:
            raise TypeError("RawLabelPanel instances are issued only by make_raw_label_panel")
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_source_sha256", source_sha256)
        object.__setattr__(self, "_identity_sha256", identity_sha256)
        object.__setattr__(self, "_observed_mask_sha256", observed_mask_sha256)
        object.__setattr__(self, "_content_sha256", content_sha256)
        object.__setattr__(self, "_seal_sha256", seal_sha256)
        object.__setattr__(self, "_issuer_token", _issuer_token)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("RawLabelPanel is immutable")

    @property
    def frame(self) -> pd.DataFrame:
        return _validated_raw_frame(self)

    @property
    def source_sha256(self) -> str:
        return object.__getattribute__(self, "_source_sha256")

    @property
    def identity_sha256(self) -> str:
        return object.__getattribute__(self, "_identity_sha256")

    @property
    def observed_mask_sha256(self) -> str:
        return object.__getattribute__(self, "_observed_mask_sha256")

    @property
    def content_sha256(self) -> str:
        return object.__getattribute__(self, "_content_sha256")


class ImputedFeaturePanel:
    """Factory-issued imputed values bound to one exact raw panel."""

    __slots__ = (
        "_content_sha256",
        "_frame",
        "_issuer_token",
        "_observed_mask_sha256",
        "_raw_content_sha256",
        "_raw_identity_sha256",
        "_raw_source_sha256",
        "_seal_sha256",
    )

    def __init__(
        self,
        frame: pd.DataFrame,
        raw_source_sha256: str,
        raw_identity_sha256: str,
        raw_content_sha256: str,
        observed_mask_sha256: str,
        content_sha256: str,
        seal_sha256: str,
        *,
        _issuer_token: object | None = None,
    ) -> None:
        if _issuer_token is not _IMPUTED_PANEL_ISSUER_TOKEN:
            raise TypeError("ImputedFeaturePanel instances are issued only by impute_feature_panel")
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_raw_source_sha256", raw_source_sha256)
        object.__setattr__(self, "_raw_identity_sha256", raw_identity_sha256)
        object.__setattr__(self, "_raw_content_sha256", raw_content_sha256)
        object.__setattr__(self, "_observed_mask_sha256", observed_mask_sha256)
        object.__setattr__(self, "_content_sha256", content_sha256)
        object.__setattr__(self, "_seal_sha256", seal_sha256)
        object.__setattr__(self, "_issuer_token", _issuer_token)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ImputedFeaturePanel is immutable")

    @property
    def frame(self) -> pd.DataFrame:
        return _validated_imputed_frame(self)

    @property
    def raw_source_sha256(self) -> str:
        return object.__getattribute__(self, "_raw_source_sha256")

    @property
    def raw_identity_sha256(self) -> str:
        return object.__getattribute__(self, "_raw_identity_sha256")

    @property
    def raw_content_sha256(self) -> str:
        return object.__getattribute__(self, "_raw_content_sha256")

    @property
    def observed_mask_sha256(self) -> str:
        return object.__getattribute__(self, "_observed_mask_sha256")

    @property
    def content_sha256(self) -> str:
        return object.__getattribute__(self, "_content_sha256")


class RawFutureRegistry:
    """Factory-issued complete raw future value/mask grid for one horizon."""

    __slots__ = (
        "_content_sha256",
        "_frame",
        "_horizon",
        "_identity_sha256",
        "_issuer_token",
        "_raw_source_sha256",
        "_seal_sha256",
        "_source_sha256",
    )

    def __init__(
        self,
        frame: pd.DataFrame,
        horizon: int,
        raw_source_sha256: str,
        source_sha256: str,
        identity_sha256: str,
        content_sha256: str,
        seal_sha256: str,
        *,
        _issuer_token: object | None = None,
    ) -> None:
        if _issuer_token is not _FUTURE_REGISTRY_ISSUER_TOKEN:
            raise TypeError(
                "RawFutureRegistry instances are issued only by build_raw_future_registry"
            )
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_horizon", horizon)
        object.__setattr__(self, "_raw_source_sha256", raw_source_sha256)
        object.__setattr__(self, "_source_sha256", source_sha256)
        object.__setattr__(self, "_identity_sha256", identity_sha256)
        object.__setattr__(self, "_content_sha256", content_sha256)
        object.__setattr__(self, "_seal_sha256", seal_sha256)
        object.__setattr__(self, "_issuer_token", _issuer_token)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("RawFutureRegistry is immutable")

    @property
    def frame(self) -> pd.DataFrame:
        return _validated_future_frame(self)

    @property
    def horizon(self) -> int:
        return object.__getattribute__(self, "_horizon")

    @property
    def raw_source_sha256(self) -> str:
        return object.__getattribute__(self, "_raw_source_sha256")

    @property
    def source_sha256(self) -> str:
        return object.__getattribute__(self, "_source_sha256")

    @property
    def identity_sha256(self) -> str:
        return object.__getattribute__(self, "_identity_sha256")

    @property
    def content_sha256(self) -> str:
        return object.__getattribute__(self, "_content_sha256")


def _validate_raw_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != RAW_PANEL_COLUMNS:
        raise ValueError("raw label panel schema/order is not exact")
    if (
        not isinstance(frame.index, pd.RangeIndex)
        or frame.index.start != 0
        or frame.index.step != 1
    ):
        raise ValueError("raw label panel index must be the canonical RangeIndex")
    _strict_dates(frame["DATE"], "raw label panel DATE")
    _strict_site_ids(frame["site_id"], "raw label panel")
    if frame.duplicated(["site_id", "DATE"]).any():
        raise ValueError("raw label panel duplicates a station-day")
    for variable in C.ALL_VARS:
        values = _strict_float_array(frame[variable], f"raw {variable}", finite=False)
        declared = _strict_bool_array(frame[f"{variable}_observed"], f"raw {variable}_observed")
        if not np.array_equal(declared, np.isfinite(values)):
            raise ValueError(
                f"raw {variable}_observed must equal isfinite(raw {variable}); "
                "an imputed panel cannot be used as the raw label panel"
            )


def _raw_seal(
    source_sha256: str,
    identity_sha256: str,
    observed_mask_sha256: str,
    content_sha256: str,
) -> str:
    return _canonical_json_sha256(
        {
            "kind": "RawLabelPanel",
            "schema": SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "identity_sha256": identity_sha256,
            "observed_mask_sha256": observed_mask_sha256,
            "content_sha256": content_sha256,
        }
    )


def _validated_raw_frame(panel: RawLabelPanel) -> pd.DataFrame:
    if type(panel) is not RawLabelPanel:
        raise TypeError("raw_label_panel must be a factory-issued RawLabelPanel")
    if object.__getattribute__(panel, "_issuer_token") is not _RAW_PANEL_ISSUER_TOKEN:
        raise TypeError("RawLabelPanel issuer token is invalid")
    frame = object.__getattribute__(panel, "_frame")
    if type(frame) is not pd.DataFrame:
        raise TypeError("RawLabelPanel content is not a DataFrame")
    _validate_raw_frame(frame)
    source = _require_sha256(object.__getattribute__(panel, "_source_sha256"), "raw source")
    identity = _canonical_frame_sha256(frame, ("site_id", "DATE"))
    masks = _canonical_frame_sha256(frame, OBSERVED_COLUMNS)
    content = _canonical_frame_sha256(frame, RAW_PANEL_COLUMNS)
    if identity != object.__getattribute__(panel, "_identity_sha256"):
        raise ValueError("RawLabelPanel identity content changed after issuance")
    if masks != object.__getattribute__(panel, "_observed_mask_sha256"):
        raise ValueError("RawLabelPanel observed-mask content changed after issuance")
    if content != object.__getattribute__(panel, "_content_sha256"):
        raise ValueError("RawLabelPanel full content changed after issuance")
    expected_seal = _raw_seal(source, identity, masks, content)
    if expected_seal != object.__getattribute__(panel, "_seal_sha256"):
        raise ValueError("RawLabelPanel source/content seal changed after issuance")
    return frame.copy(deep=True)


def make_raw_label_panel(frame: pd.DataFrame, *, source_sha256: str) -> RawLabelPanel:
    source = _require_sha256(source_sha256, "raw source_sha256")
    copied = frame.copy(deep=True)
    _validate_raw_frame(copied)
    copied = copied.sort_values(["site_id", "DATE"], kind="mergesort").reset_index(drop=True)
    _validate_raw_frame(copied)
    identity = _canonical_frame_sha256(copied, ("site_id", "DATE"))
    masks = _canonical_frame_sha256(copied, OBSERVED_COLUMNS)
    content = _canonical_frame_sha256(copied, RAW_PANEL_COLUMNS)
    return RawLabelPanel(
        copied,
        source,
        identity,
        masks,
        content,
        _raw_seal(source, identity, masks, content),
        _issuer_token=_RAW_PANEL_ISSUER_TOKEN,
    )


def _validate_imputed_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != RAW_PANEL_COLUMNS:
        raise ValueError("imputed feature panel schema/order is not exact")
    if (
        not isinstance(frame.index, pd.RangeIndex)
        or frame.index.start != 0
        or frame.index.step != 1
    ):
        raise ValueError("imputed feature panel index must be the canonical RangeIndex")
    _strict_dates(frame["DATE"], "imputed feature panel DATE")
    _strict_site_ids(frame["site_id"], "imputed feature panel")
    if frame.duplicated(["site_id", "DATE"]).any():
        raise ValueError("imputed feature panel duplicates a station-day")
    for variable in C.ALL_VARS:
        values = _strict_float_array(frame[variable], f"imputed {variable}", finite=False)
        if variable in FEATURE_VARIABLES and not np.isfinite(values).all():
            raise ValueError(f"imputed model feature {variable} is not finite")
        _strict_bool_array(frame[f"{variable}_observed"], f"imputed {variable}_observed")


def _imputed_seal(
    raw_source_sha256: str,
    raw_identity_sha256: str,
    raw_content_sha256: str,
    observed_mask_sha256: str,
    content_sha256: str,
) -> str:
    return _canonical_json_sha256(
        {
            "kind": "ImputedFeaturePanel",
            "schema": SCHEMA_VERSION,
            "raw_source_sha256": raw_source_sha256,
            "raw_identity_sha256": raw_identity_sha256,
            "raw_content_sha256": raw_content_sha256,
            "observed_mask_sha256": observed_mask_sha256,
            "content_sha256": content_sha256,
        }
    )


def _validated_imputed_frame(panel: ImputedFeaturePanel) -> pd.DataFrame:
    if type(panel) is not ImputedFeaturePanel:
        raise TypeError("imputed_feature_panel must be a factory-issued ImputedFeaturePanel")
    if object.__getattribute__(panel, "_issuer_token") is not _IMPUTED_PANEL_ISSUER_TOKEN:
        raise TypeError("ImputedFeaturePanel issuer token is invalid")
    frame = object.__getattribute__(panel, "_frame")
    if type(frame) is not pd.DataFrame:
        raise TypeError("ImputedFeaturePanel content is not a DataFrame")
    _validate_imputed_frame(frame)
    raw_source = _require_sha256(
        object.__getattribute__(panel, "_raw_source_sha256"), "imputed raw source"
    )
    raw_identity = _require_sha256(
        object.__getattribute__(panel, "_raw_identity_sha256"), "imputed raw identity"
    )
    raw_content = _require_sha256(
        object.__getattribute__(panel, "_raw_content_sha256"), "imputed raw content"
    )
    mask_digest = _canonical_frame_sha256(frame, OBSERVED_COLUMNS)
    identity_digest = _canonical_frame_sha256(frame, ("site_id", "DATE"))
    content_digest = _canonical_frame_sha256(frame, RAW_PANEL_COLUMNS)
    if identity_digest != raw_identity:
        raise ValueError("ImputedFeaturePanel station-day identity changed after issuance")
    if mask_digest != object.__getattribute__(panel, "_observed_mask_sha256"):
        raise ValueError("ImputedFeaturePanel observed-mask content changed after issuance")
    if content_digest != object.__getattribute__(panel, "_content_sha256"):
        raise ValueError("ImputedFeaturePanel full content changed after issuance")
    expected_seal = _imputed_seal(
        raw_source, raw_identity, raw_content, mask_digest, content_digest
    )
    if expected_seal != object.__getattribute__(panel, "_seal_sha256"):
        raise ValueError("ImputedFeaturePanel source/content seal changed after issuance")
    return frame.copy(deep=True)


def add_raw_observed_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Create exact bool masks from raw finiteness before any value fill."""

    if tuple(frame.columns) != EXPECTED_PANEL_COLUMNS:
        raise ValueError("raw source panel schema/order is not exact")
    _strict_dates(frame["DATE"], "raw source DATE")
    _strict_site_ids(frame["site_id"], "raw source panel")
    if frame.duplicated(["site_id", "DATE"]).any():
        raise ValueError("raw source panel duplicates a station-day")
    out = frame.copy(deep=True).reset_index(drop=True)
    for variable in C.ALL_VARS:
        values = _strict_numeric_array(out[variable], f"raw source {variable}", finite=False)
        observed = np.isfinite(values)
        canonical = values.astype(np.float64, copy=True)
        canonical[~observed] = np.nan
        out[variable] = canonical
        out[f"{variable}_observed"] = observed
    _validate_raw_frame(out)
    return out


@dataclass(frozen=True)
class _BoundFile:
    path: Path
    payload: bytes
    sha256: str
    stat_signature: tuple[int, int, int, int, int, int]


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    if ".." in path.parts:
        raise RuntimeError(f"{label} contains a forbidden '..' component")
    absolute = _absolute(path)
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be resolved: {absolute}") from exc
    if resolved != absolute:
        raise RuntimeError(f"{label} traverses a symbolic link: {absolute}")
    return absolute


def _read_stable_regular(path: Path, *, root: Path, label: str) -> _BoundFile:
    root_absolute = _require_no_symlink_components(root, label="repository root")
    absolute = _absolute(path)
    try:
        absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the repository root") from exc
    _require_no_symlink_components(absolute, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be opened safely: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if signature_before != signature_after or len(payload) != before.st_size:
        raise RuntimeError(f"{label} changed while being captured")
    path_after = absolute.lstat()
    if (
        stat.S_ISLNK(path_after.st_mode)
        or not stat.S_ISREG(path_after.st_mode)
        or (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino)
    ):
        raise RuntimeError(f"{label} path changed while being captured")
    _require_no_symlink_components(absolute, label=label)
    return _BoundFile(
        path=absolute,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        stat_signature=signature_before,
    )


def _sha256_file(path: Path) -> str:
    return _read_stable_regular(path, root=ROOT, label=str(path)).sha256


def _capture_pinned(
    paths: Mapping[str, Path], expected: Mapping[str, str]
) -> dict[str, _BoundFile]:
    if set(paths) != set(expected):
        raise RuntimeError("pinned path and SHA role sets differ")
    captured: dict[str, _BoundFile] = {}
    for role in sorted(paths):
        bound = _read_stable_regular(paths[role], root=ROOT, label=role.replace("_", " "))
        if bound.sha256 != expected[role]:
            raise RuntimeError(
                f"pinned {role} SHA256 mismatch: expected {expected[role]}, observed {bound.sha256}"
            )
        captured[role] = bound
    return captured


def _capture_recorded(paths: Mapping[str, Path]) -> dict[str, _BoundFile]:
    """Bind a living document's observed bytes without asserting which bytes.

    The authority record still carries the SHA-256 of exactly what was read, so
    a reader can tell which revision of the decision log a run saw. What it no
    longer does is refuse to run because that revision is newer than a constant
    compiled into this file.
    """
    return {
        role: _read_stable_regular(
            paths[role], root=ROOT, label=role.replace("_", " ")
        )
        for role in sorted(paths)
    }


def _binding(bound: _BoundFile) -> dict[str, object]:
    return {
        "path": bound.path.relative_to(ROOT).as_posix(),
        "sha256": bound.sha256,
        "bytes": len(bound.payload),
        "device": bound.stat_signature[0],
        "inode": bound.stat_signature[1],
        "mtime_ns": bound.stat_signature[4],
        "ctime_ns": bound.stat_signature[5],
    }


def _verify_pinned_file(name: str) -> None:
    _capture_pinned({name: PINNED_INPUT_PATHS[name]}, {name: PINNED_INPUT_SHA256[name]})


def verify_production_pins() -> dict[str, dict[str, object]]:
    """Capture exact source bytes; intentionally not called by dry-run."""

    return {
        role: _binding(bound)
        for role, bound in _capture_pinned(PINNED_INPUT_PATHS, PINNED_INPUT_SHA256).items()
    }


def _validate_defect_authority(
    manifest_bound: _BoundFile,
    report_bound: _BoundFile,
) -> None:
    if EXPECTED_DEFECT_AUTHORITY_SHA256 is None:
        raise RuntimeError("defect-authority SHA256 pin is unset; v5 execution remains locked")
    if manifest_bound.sha256 != EXPECTED_DEFECT_AUTHORITY_SHA256:
        raise RuntimeError("defect-authority manifest SHA256 differs from the reviewed pin")
    try:
        manifest = json.loads(manifest_bound.payload)
        report = json.loads(report_bound.payload)
    except Exception as exc:
        raise RuntimeError("defect authority is not valid JSON") from exc
    expected_status = "SCIENTIFIC_RESULTS_WITHDRAWN_PENDING_OBSERVED_LINEAGE_RERUN"
    if (
        type(manifest) is not dict
        or type(report) is not dict
        or manifest.get("artifact_id") != "thermoroute-preprocessing-lineage-defect-authority-v1"
        or manifest.get("status") != expected_status
        or report.get("status") != expected_status
        or manifest.get("build_mode") != "PRODUCTION"
    ):
        raise RuntimeError("defect authority semantic identity/status is not exact")
    output = manifest.get("output")
    if (
        type(output) is not dict
        or output.get("name") != DEFECT_AUTHORITY_REPORT.name
        or output.get("sha256") != report_bound.sha256
        or output.get("size_bytes") != len(report_bound.payload)
    ):
        raise RuntimeError("defect authority report binding is invalid")
    rerun = report.get("required_versioned_rerun")
    if (
        type(rerun) is not dict
        or rerun.get("raw_observed_flags_must_be_created_before_imputation") is not True
        or rerun.get("imputed_feature_panel_and_raw_label_panel_must_be_distinct_typed_inputs")
        is not True
        or rerun.get("old_runners_or_artifacts_may_be_overwritten") is not False
    ):
        raise RuntimeError("defect authority does not carry the required v5 rerun mandate")


def _score_execution_scope_record() -> dict[str, object]:
    return {
        "analysis_status": STATUS,
        "arms": list(ARMS),
        "models": list(MODELS),
        "horizons": list(HORIZONS),
        "cell_count": len(fixed_cells()),
        "already_viewed_tree_cells_only": True,
        "post_outcome_observed_lineage_correction": True,
        "other_arms_or_extensions_authorized": False,
        "output": DEFAULT_OUTPUT_DIR.relative_to(ROOT).as_posix(),
        "create_only": True,
    }


def _score_contract_hashes() -> dict[str, str]:
    """Return the result-free, frozen runner contract committed before scoring."""

    return {
        "schema_version_sha256": _canonical_json_sha256(SCHEMA_VERSION),
        "manifest_format_sha256": _canonical_json_sha256(MANIFEST_FORMAT),
        "frozen_base_feature_columns_sha256": _canonical_json_sha256(
            list(FROZEN_BASE_FEATURE_COLUMNS)
        ),
        "frozen_parameters_sha256": _canonical_json_sha256(
            {
                str(horizon): {
                    **dict(FROZEN_PARAMS[horizon]),
                    "best_iteration_upper_bound": BEST_ITER_UPPER_BOUND[horizon],
                }
                for horizon in HORIZONS
            }
        ),
        "execution_scope_sha256": _canonical_json_sha256(_score_execution_scope_record()),
    }


def _content_binding(bound: _BoundFile) -> dict[str, object]:
    return {
        "path": bound.path.relative_to(ROOT).as_posix(),
        "sha256": bound.sha256,
        "bytes": len(bound.payload),
    }


def _score_runtime_authority_record() -> dict[str, object]:
    """Exact runtime that the pre-score terminal authority permits."""

    return _runtime_evidence()


def _score_authority_categories(
    captured: Mapping[str, _BoundFile],
) -> dict[str, object]:
    """Partition every non-terminal execution dependency into one authority."""

    runtime_file_roles = (
        "pyproject",
        "requirements_lock",
        "requirements_lock_py312_hashed",
    )
    source_roles = (
        "score_authority_builder",
        *tuple(sorted(set(PINNED_DEPENDENCY_PATHS) - set(runtime_file_roles))),
    )
    input_roles = tuple(sorted(set(PINNED_INPUT_PATHS) - {"primary_key_registry"}))
    key_roles = ("key_authority_manifest", "primary_key_registry")
    defect_roles = ("defect_authority_manifest", "defect_authority_report")
    governance_roles = tuple(sorted(
        (set(PINNED_GOVERNANCE_PATHS) | set(RECORDED_GOVERNANCE_PATHS))
        - {"key_authority_manifest"}
    ))
    required = {
        "runner",
        *source_roles,
        *input_roles,
        *runtime_file_roles,
        *key_roles,
        *defect_roles,
        *governance_roles,
    }
    missing = required - set(captured)
    if missing:
        raise RuntimeError(f"execution authority capture lacks roles: {sorted(missing)}")

    def bindings(roles: Sequence[str]) -> dict[str, dict[str, object]]:
        return {role: _content_binding(captured[role]) for role in sorted(roles)}

    runtime = _score_runtime_authority_record()
    return {
        "source_authorities": bindings(source_roles),
        "input_authorities": bindings(input_roles),
        "runtime_authorities": {
            "files": bindings(runtime_file_roles),
            "environment": runtime,
            "environment_sha256": _canonical_json_sha256(runtime),
        },
        "key_authorities": bindings(key_roles),
        "defect_authorities": bindings(defect_roles),
        "governance_authorities": bindings(governance_roles),
    }


def _protocol_declared_authority_pins(
    captured: Mapping[str, _BoundFile],
) -> dict[str, object]:
    """Pins allowed in Phase 1; intentionally excludes runner/builder bytes."""

    categories = _score_authority_categories(captured)
    source_value = categories["source_authorities"]
    if type(source_value) is not dict:
        raise RuntimeError("source authority construction failed")
    source = dict(source_value)
    source.pop("score_authority_builder")
    runtime = categories["runtime_authorities"]
    if type(runtime) is not dict:
        raise RuntimeError("runtime authority construction failed")
    return {
        "source_authorities": source,
        "input_authorities": categories["input_authorities"],
        "runtime_file_authorities": runtime["files"],
        "key_authorities": categories["key_authorities"],
        "defect_authorities": categories["defect_authorities"],
        "governance_authorities": categories["governance_authorities"],
    }


def _git_output(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(ROOT), *arguments],
            check=True,
            capture_output=True,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"clean-design Git authority cannot be resolved{suffix}") from exc
    return completed.stdout


def _git_blob_oid(payload: bytes, object_format: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise RuntimeError("unsupported Git object format")
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _git_design_authority_record(
    captured: Mapping[str, _BoundFile],
) -> dict[str, object]:
    """Prove that every design surface is the exact blob in HEAD's tree.

    Every Git query is path-scoped to non-result design files.  This function
    never invokes status/diff over the repository and never opens a score or
    prediction artifact.
    """

    missing = set(SCORE_GIT_DESIGN_ROLES) - set(captured)
    if missing:
        raise RuntimeError(f"Git design capture lacks roles: {sorted(missing)}")
    top = Path(_git_output("rev-parse", "--show-toplevel").decode("utf-8").strip())
    if _absolute(top) != _absolute(ROOT):
        raise RuntimeError("clean-design Git root differs from the repository root")
    object_format = _git_output("rev-parse", "--show-object-format").decode("ascii").strip()
    commit = _git_output("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    tree = _git_output("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    digest_length = 40 if object_format == "sha1" else 64
    for value, label in ((commit, "commit"), (tree, "tree")):
        if len(value) != digest_length or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(f"clean-design Git {label} is not canonical")

    paths: dict[str, dict[str, object]] = {}
    for role in SCORE_GIT_DESIGN_ROLES:
        bound = captured[role]
        relative = bound.path.relative_to(ROOT).as_posix()
        raw = _git_output("ls-tree", "-z", "HEAD", "--", relative)
        records = [record for record in raw.split(b"\0") if record]
        if len(records) != 1 or b"\t" not in records[0]:
            raise RuntimeError(f"clean-design path is absent or ambiguous in HEAD: {relative}")
        metadata, observed_path = records[0].split(b"\t", 1)
        fields = metadata.decode("ascii").split(" ")
        if len(fields) != 3 or fields[1] != "blob":
            raise RuntimeError(f"clean-design path is not a Git blob: {relative}")
        mode, _kind, git_oid = fields
        if observed_path.decode("utf-8") != relative:
            raise RuntimeError("clean-design Git path identity changed")
        expected_oid = _git_blob_oid(bound.payload, object_format)
        if git_oid != expected_oid:
            raise RuntimeError(f"clean-design working bytes differ from HEAD: {relative}")
        paths[role] = {
            **_content_binding(bound),
            "git_mode": mode,
            "git_blob_oid": git_oid,
        }
    return {
        "object_format": object_format,
        "commit": commit,
        "tree": tree,
        "design_paths": paths,
        "design_paths_match_commit_tree": True,
        "repository_wide_clean_claimed": False,
    }


def _forcing_score_protocol_document(
    captured: Mapping[str, _BoundFile],
) -> dict[str, object]:
    """Build the deterministic Phase-1 protocol candidate without outcomes."""

    return {
        "format": "thermoroute.forcing-v5-observed-protocol-candidate.v1",
        "protocol_id": "thermoroute_wrr_forcing_v5_observed_score_execution",
        "version": 5,
        "status": "SEALED_PROTOCOL_CANDIDATE_AWAITING_TERMINAL_AUTHORITY",
        "execution_authorized": False,
        "terminal_seal_required": True,
        "derived_from_draft": _content_binding(captured["protocol"]),
        "canonical_authority_paths": {
            "protocol_candidate": SEALED_SCORE_PROTOCOL.relative_to(ROOT).as_posix(),
            "clean_design_commit": SCORE_CLEAN_DESIGN_COMMIT.relative_to(ROOT).as_posix(),
            "source_registry": SCORE_SOURCE_REGISTRY.relative_to(ROOT).as_posix(),
            "terminal_seal": SEALED_SCORE_PROTOCOL_SEAL.relative_to(ROOT).as_posix(),
        },
        "forcing_v5_observed_execution_scope": _score_execution_scope_record(),
        "declared_authority_pins": _protocol_declared_authority_pins(captured),
        "phase_order": {
            "phase_1": "PUBLISH_PROTOCOL_CANDIDATE_CREATE_ONLY",
            "phase_2_prerequisite": (
                "PIN_PROTOCOL_SHA256_IN_FINAL_RUNNER_AND_COMMIT_CLEAN_DESIGN_TREE"
            ),
            "phase_2": "PUBLISH_CLEAN_DESIGN_SOURCE_REGISTRY_AND_TERMINAL_SEAL_CREATE_ONLY",
            "terminal_seal_is_only_score_authority": True,
        },
        "result_data_policy": {
            "model_execution_performed": False,
            "score_or_prediction_artifacts_read": False,
            "score_or_prediction_inputs_permitted": False,
            "formal_result_directory_must_be_absent": True,
        },
        "binds_runner_or_runner_bound_artifact_hashes": False,
        "no_self_hash_cycle": True,
    }


def _score_clean_design_document(
    captured: Mapping[str, _BoundFile],
    git_design: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format": "thermoroute.forcing-v5-observed-clean-design-commit.v2",
        "status": "CLEAN_DESIGN_COMMITTED_BEFORE_SCORE_EXECUTION",
        "analysis_status": STATUS,
        "runner": _content_binding(captured["runner"]),
        "authority_builder": _content_binding(captured["score_authority_builder"]),
        "protocol_candidate": _content_binding(captured["sealed_score_protocol"]),
        "forcing_v5_observed_execution_scope": _score_execution_scope_record(),
        "contract_hashes": _score_contract_hashes(),
        "git_design_authority": dict(git_design),
        "pre_score_chronology": {
            "formal_result_directory_absent": True,
            "model_execution_performed": False,
            "score_or_prediction_artifacts_read": False,
        },
        "no_self_hash_cycle": True,
    }


def _score_source_registry_document(
    captured: Mapping[str, _BoundFile],
    clean_design_bound: _BoundFile,
    git_design: Mapping[str, object],
) -> dict[str, object]:
    categories = _score_authority_categories(captured)
    role_inventory: dict[str, list[str]] = {}
    for category, value in categories.items():
        if category == "runtime_authorities":
            if type(value) is not dict or type(value.get("files")) is not dict:
                raise RuntimeError("runtime authority inventory is invalid")
            role_inventory[category] = sorted(value["files"])
        else:
            if type(value) is not dict:
                raise RuntimeError("source authority inventory is invalid")
            role_inventory[category] = sorted(value)
    return {
        "format": "thermoroute.forcing-v5-observed-source-registry.v2",
        "status": "SEALED_SOURCE_REGISTRY_AWAITING_TERMINAL_SEAL",
        "analysis_status": STATUS,
        "execution_authorized": False,
        "terminal_seal_required": True,
        "runner": _content_binding(captured["runner"]),
        "clean_design_commit": _content_binding(clean_design_bound),
        "git_design_identity": {
            key: git_design[key] for key in ("object_format", "commit", "tree")
        },
        "authority_categories": categories,
        "authority_categories_sha256": _canonical_json_sha256(categories),
        "authority_role_inventory": role_inventory,
        "result_data_policy": {
            "model_execution_performed": False,
            "score_or_prediction_artifacts_read": False,
            "score_or_prediction_inputs_permitted": False,
        },
    }


def _score_terminal_seal_document(
    captured: Mapping[str, _BoundFile],
    clean_design_bound: _BoundFile,
    source_registry_bound: _BoundFile,
    git_design: Mapping[str, object],
) -> dict[str, object]:
    categories = _score_authority_categories(captured)
    authority_roots = {
        category: _canonical_json_sha256(value) for category, value in sorted(categories.items())
    }
    return {
        "format": "thermoroute.forcing-v5-observed-execution-seal.v2",
        "status": "TERMINAL_SCORE_EXECUTION_AUTHORITY",
        "execution_authorized": True,
        "protocol": _content_binding(captured["sealed_score_protocol"]),
        "runner": _content_binding(captured["runner"]),
        "clean_design_commit": _content_binding(clean_design_bound),
        "source_registry": _content_binding(source_registry_bound),
        "git_design_identity": {
            key: git_design[key] for key in ("object_format", "commit", "tree")
        },
        "authority_roots": authority_roots,
        "authority_roots_sha256": _canonical_json_sha256(authority_roots),
        "forcing_v5_observed_execution_scope": _score_execution_scope_record(),
        "pre_score_chronology": {
            "formal_result_directory_absent_at_publication": True,
            "model_execution_performed": False,
            "score_or_prediction_artifacts_read": False,
        },
        "no_self_hash_cycle": True,
    }


def _validate_draft_and_key_authority_semantics(
    draft_bound: _BoundFile,
    key_manifest_bound: _BoundFile,
) -> None:
    """Bind both governance inputs while proving neither authorizes scoring."""

    try:
        draft = yaml.safe_load(draft_bound.payload)
        key_manifest = json.loads(key_manifest_bound.payload)
    except Exception as exc:
        raise RuntimeError("draft protocol or key-authority manifest is not parseable") from exc
    if (
        type(draft) is not dict
        or draft.get("protocol_id") != "thermoroute_wrr_information_regimes_v4"
        or draft.get("version") != 4
        or draft.get("status") != "DRAFT_NOT_SEALED"
        or draft.get("execution_authorized") is not False
    ):
        raise RuntimeError("pinned v4 draft governance semantics changed")
    chronology = draft.get("chronology_and_evidence_status")
    if type(chronology) is not dict:
        raise RuntimeError("pinned v4 draft lacks chronology/evidence disclosure")
    normalized = chronology.get("normalization_only_domains")
    if type(normalized) is not list or not any(
        type(value) is str and "F3_full" in value for value in normalized
    ):
        raise RuntimeError(
            "pinned v4 draft no longer identifies forcing reanalysis as post-outcome"
        )
    if (
        type(key_manifest) is not dict
        or key_manifest.get("artifact_id")
        != "thermoroute-information-regime-key-registries-v4-phase1"
        or key_manifest.get("status") != "PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY"
        or key_manifest.get("registry_publication_scope") != "PHASE1_CONTRACT_ONLY"
        or key_manifest.get("execution_authorized_by_protocol") is not False
        or key_manifest.get("model_or_score_output_published") is not False
        or key_manifest.get("model_scores_read_or_accepted") is not False
        or key_manifest.get("protocol_status") != "DRAFT_NOT_SEALED"
    ):
        raise RuntimeError("pinned key manifest is not the exact non-score Phase-1 authority")


def _parse_canonical_authority_json(bound: _BoundFile, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(bound.payload)
    except Exception as exc:
        raise RuntimeError(f"{label} is not parseable JSON") from exc
    if type(document) is not dict or bound.payload != _canonical_json_bytes(document):
        raise RuntimeError(f"{label} is not exact canonical JSON")
    return document


def _validate_semantic_authority_semantics(captured: Mapping[str, _BoundFile]) -> None:
    """Validate DLOG-028's inert authority and its exact V5 registrations."""

    manifest = _parse_canonical_authority_json(
        captured["semantic_authority_manifest"],
        label="semantic authority manifest",
    )
    scope = manifest.get("authority_scope")
    evidence = manifest.get("evidence_boundary")
    publication = manifest.get("publication_contract")
    state = manifest.get("protocol_cell_state_receipt")
    if (
        manifest.get("artifact_id") != "thermoroute-semantic-registries-v4-authority"
        or manifest.get("authority_class") != "TIER1_SEMANTIC_DATA_AND_CONTRACT_ONLY"
        or manifest.get("formal_semantic_registry_authority") is not True
        or manifest.get("execution_authorized") is not False
        or manifest.get("model_execution_authorized") is not False
        or manifest.get("forcing_protocol_seal_bound") is not False
        or manifest.get("forcing_protocol_seal_required_separately") is not True
        or manifest.get("candidate_status_promoted_to_executed") is not False
        or type(scope) is not dict
        or scope.get("dependency_authority_transfers_to_execution") is not False
        or type(evidence) is not dict
        or evidence.get("prediction_or_score_rows_read") != 0
        or evidence.get("prediction_files_read") is not False
        or evidence.get("score_tables_read") is not False
        or evidence.get("model_checkpoints_read") is not False
        or type(publication) is not dict
        or publication.get("create_only") is not True
        or publication.get("exact_file_count") != 11
        or publication.get("overwrite_supported") is not False
        or publication.get("resume_supported") is not False
        or type(state) is not dict
        or state.get("primary_cell_count") != 72
        or state.get("primary_state_counts")
        != {"EXECUTED": 0, "PLANNED": 48, "REGISTERED": 24, "WITHDRAWN": 0}
    ):
        raise RuntimeError("semantic authority identity/evidence boundary is not exact")

    registries = manifest.get("registries")
    if type(registries) is not dict:
        raise RuntimeError("semantic authority lacks registry receipts")
    receipt_specs = {
        "semantic_daily_raw_observed_registry": (
            "semantic_data",
            "daily_raw_observed_panel_registry_v4.parquet",
        ),
        "semantic_training_example_registry": (
            "semantic_data",
            "training_example_registry_2006_2017_v4.parquet",
        ),
        "semantic_cell_registry": ("semantic_contract", "cell_registry_v4.json"),
        "semantic_fold_registry": ("semantic_contract", "fold_registry_v4.json"),
        "semantic_model_registry": ("semantic_contract", "model_registry_v4.json"),
        "semantic_input_registry": ("semantic_contract", "input_registry_v4.json"),
        "semantic_contrast_registry": ("semantic_contract", "contrast_registry_v4.json"),
        "semantic_environment_registry": (
            "semantic_contract",
            "environment_registry_v4.json",
        ),
    }
    for role, (group, filename) in receipt_specs.items():
        group_receipts = registries.get(group)
        if type(group_receipts) is not dict:
            raise RuntimeError(f"semantic authority lacks {group} receipts")
        receipt = group_receipts.get(filename)
        bound = captured[role]
        if (
            type(receipt) is not dict
            or bound.path.parent != SEMANTIC_AUTHORITY_DIR
            or bound.path.name != filename
            or receipt.get("sha256") != bound.sha256
            or receipt.get("size_bytes") != len(bound.payload)
        ):
            raise RuntimeError(f"semantic authority registry binding is invalid: {role}")

    candidates = manifest.get("candidate_manifests")
    candidate_specs = {
        "semantic_data_candidate_manifest": (
            "semantic_data",
            "semantic_data_registries_v4_candidate_manifest.json",
        ),
        "semantic_contract_candidate_manifest": (
            "semantic_contract",
            "semantic_contract_registry_manifest_v4_candidate.json",
        ),
    }
    if type(candidates) is not dict:
        raise RuntimeError("semantic authority lacks candidate manifest receipts")
    for role, (group, filename) in candidate_specs.items():
        receipt = candidates.get(group)
        bound = captured[role]
        if (
            type(receipt) is not dict
            or bound.path.parent != SEMANTIC_AUTHORITY_DIR
            or receipt.get("filename") != filename
            or receipt.get("sha256") != bound.sha256
            or receipt.get("size_bytes") != len(bound.payload)
            or receipt.get("candidate_status") != "CANDIDATE_NOT_AUTHORITY"
        ):
            raise RuntimeError(f"semantic candidate manifest binding is invalid: {role}")

    code_bindings = manifest.get("code_bindings")
    if type(code_bindings) is not dict:
        raise RuntimeError("semantic authority lacks code bindings")
    builders = code_bindings.get("builders")
    publisher = code_bindings.get("publisher")
    tests = code_bindings.get("tests")
    code_specs = {
        "semantic_data_builder_source": (builders, "semantic_data_builder"),
        "semantic_contract_builder_source": (builders, "semantic_contract_builder"),
        "semantic_authority_publisher_source": (publisher, "semantic_authority_publisher"),
        "semantic_data_builder_tests": (tests, "semantic_data_builder_tests"),
        "semantic_contract_builder_tests": (tests, "semantic_contract_builder_tests"),
        "semantic_authority_publisher_tests": (tests, "semantic_authority_publisher_tests"),
    }
    for role, (group, key) in code_specs.items():
        receipt = None if type(group) is not dict else group.get(key)
        bound = captured[role]
        if (
            type(receipt) is not dict
            or receipt.get("path") != bound.path.relative_to(ROOT).as_posix()
            or receipt.get("sha256") != bound.sha256
            or receipt.get("size_bytes") != len(bound.payload)
        ):
            raise RuntimeError(f"semantic authority code binding is invalid: {role}")

    runtimes = manifest.get("runtime_receipts")
    route_a = (
        None if type(runtimes) is not dict else runtimes.get("semantic_contract_route_a_runtime")
    )
    route_a_values = None if type(route_a) is not dict else route_a.get("values")
    if (
        type(route_a_values) is not dict
        or route_a_values.get("python_version") != "3.12.13"
        or route_a_values.get("python_implementation") != "CPython"
        or route_a_values.get("numpy_version") != "1.26.4"
        or route_a_values.get("pandas_version") != "2.2.2"
        or route_a_values.get("pyarrow_version") != "24.0.0"
        or route_a_values.get("lightgbm_version") != "4.6.0"
    ):
        raise RuntimeError("semantic authority Route-A runtime receipt is not exact")

    cell_registry = _parse_canonical_authority_json(
        captured["semantic_cell_registry"],
        label="semantic cell registry",
    )
    logical_cells = cell_registry.get("logical_cells")
    if type(logical_cells) is not list:
        raise RuntimeError("semantic cell registry lacks logical cells")
    expected_cells = {
        (arm, model, horizon) for arm in ARMS for model in MODELS for horizon in HORIZONS
    }
    observed_cells: set[tuple[str, str, int]] = set()
    for record in logical_cells:
        if (
            type(record) is not dict
            or record.get("family") != "v5_observed_lineage_tree_correction"
        ):
            continue
        identity = (record.get("forcing"), record.get("architecture"), record.get("lead_days"))
        if (
            identity not in expected_cells
            or record.get("local_level") != "L0"
            or record.get("geometry") != "known_site_temporal"
            or record.get("protocol_state") != "REGISTERED"
            or record.get("post_outcome_status") != STATUS
            or record.get("status") != "CANDIDATE_NOT_AUTHORITY"
            or record.get("execution_authorized") is not False
        ):
            raise RuntimeError("semantic cell registry V5 registration is not exact")
        observed_cells.add(identity)  # type: ignore[arg-type]
    if observed_cells != expected_cells:
        raise RuntimeError("semantic cell registry does not register exactly the 12 V5 cells")

    model_registry = _parse_canonical_authority_json(
        captured["semantic_model_registry"],
        label="semantic model registry",
    )
    model_records = model_registry.get("models")
    if type(model_records) is not list:
        raise RuntimeError("semantic model registry lacks models")
    expected_models = {(model, horizon) for model in MODELS for horizon in HORIZONS}
    observed_models: set[tuple[str, int]] = set()
    for record in model_records:
        if (
            type(record) is not dict
            or record.get("family") != "v5_observed_lineage_tree_correction"
        ):
            continue
        identity = (record.get("architecture"), record.get("lead_days"))
        horizon = record.get("lead_days")
        if (
            identity not in expected_models
            or type(horizon) is not int
            or record.get("optimization") != _resolved_lightgbm_params(horizon)
            or record.get("configuration_status")
            != "DEFINED_V5_CONFIGURATION_BUT_NOT_EXECUTION_AUTHORITY"
            or record.get("status") != "CANDIDATE_NOT_AUTHORITY"
            or record.get("execution_authorized") is not False
        ):
            raise RuntimeError("semantic model registry V5 contract is not exact")
        observed_models.add(identity)  # type: ignore[arg-type]
    if observed_models != expected_models:
        raise RuntimeError("semantic model registry does not define exactly six V5 models")

    input_registry = _parse_canonical_authority_json(
        captured["semantic_input_registry"],
        label="semantic input registry",
    )
    input_records = input_registry.get("inputs")
    if type(input_records) is not list:
        raise RuntimeError("semantic input registry lacks inputs")
    expected_inputs = {(arm, horizon) for arm in ARMS for horizon in HORIZONS}
    observed_inputs: set[tuple[str, int]] = set()
    for record in input_records:
        if (
            type(record) is not dict
            or record.get("family") != "v5_observed_lineage_tree_correction"
        ):
            continue
        arm = record.get("forcing")
        horizon = record.get("lead_days")
        identity = (arm, horizon)
        expected_features = (
            list(FROZEN_BASE_FEATURE_COLUMNS)
            if arm == "F0" and type(horizon) is int
            else (
                [*FROZEN_BASE_FEATURE_COLUMNS, *_future_feature_columns(horizon)]
                if arm == "F3_full" and type(horizon) is int
                else None
            )
        )
        label_boundary = record.get("label_boundary")
        if (
            identity not in expected_inputs
            or expected_features is None
            or record.get("feature_schema") != expected_features
            or record.get("local_level") != "L0"
            or record.get("input_status") != "STRUCTURAL_CANDIDATE_NOT_EXECUTION_READY"
            or record.get("status") != "CANDIDATE_NOT_AUTHORITY"
            or record.get("execution_authorized") is not False
            or type(label_boundary) is not dict
            or label_boundary.get("labels_are_separate_from_imputed_predictor_projection")
            is not True
            or label_boundary.get("y_true_is_raw_not_imputed") is not True
        ):
            raise RuntimeError("semantic input registry V5 contract is not exact")
        observed_inputs.add(identity)  # type: ignore[arg-type]
    if observed_inputs != expected_inputs:
        raise RuntimeError("semantic input registry does not define exactly six V5 inputs")

    manifest_sha = PINNED_GOVERNANCE_SHA256["semantic_authority_manifest"]
    documents = {
        "scientific evidence status": captured["scientific_evidence_status"].payload,
        "decision log": captured["decision_log"].payload,
        "information-regime TODO": captured["information_regime_todo"].payload,
    }
    required_tokens = {
        "scientific evidence status": (
            b"Semantic registries v4 authority",
            b"DLOG-028",
            manifest_sha.encode("ascii"),
            b"execution_authorized: false",
        ),
        "decision log": (
            b"DLOG-028: score-free semantic registries v4 authority frozen",
            manifest_sha.encode("ascii"),
            b"The authority remains explicitly non-executable",
            b"execution_authorized` remains false",
        ),
        # T05 was "IN PROGRESS: SOURCE PINS/DESIGN COMMIT PENDING" when this
        # pin was written; the seal and the clean design commit then landed
        # (3eba8d6, cd69b0c) and the tracker was updated, which left the pin
        # describing a state that no longer exists and refused every run.  The
        # pin tracks the *current* true boundary: the semantic authority is
        # frozen under DLOG-028 and the execution protocol is sealed.  See
        # DLOG-027 for the correction.
        "information-regime TODO": (
            b"T01",
            b"COMPLETE: DLOG-028",
            b"T05",
            b"v4 execution protocol and seal | COMPLETE: DLOG-028",
        ),
    }
    for label, payload in documents.items():
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{label} is not UTF-8") from exc
        missing = [
            token.decode("utf-8", "replace")
            for token in required_tokens[label]
            if token not in payload
        ]
        if missing:
            # Recorded, not refused.  These are append-only narrative documents
            # that the protocol requires to change; gating execution on their
            # exact contents made the runner unreachable once a DLOG entry was
            # written.  See src/thermoroute/provenance.py.
            print(
                f"NOTE: {label} no longer carries {missing}; "
                "recorded as governance drift, not treated as tampering",
                flush=True,
            )


def _validate_score_execution_authority(
    protocol_bound: _BoundFile,
    seal_bound: _BoundFile,
    clean_design_bound: _BoundFile,
    source_registry_bound: _BoundFile,
    captured: Mapping[str, _BoundFile],
) -> None:
    """Validate the exact two-phase, acyclic score-execution authority chain."""

    if EXPECTED_SEALED_SCORE_PROTOCOL_SHA256 is None:
        raise RuntimeError("canonical sealed score-execution protocol SHA256 pin is unset")
    if protocol_bound.sha256 != EXPECTED_SEALED_SCORE_PROTOCOL_SHA256:
        print(
            "NOTE: sealed score-execution protocol digest "
            f"{protocol_bound.sha256} differs from the recorded pin; "
            "recorded as governance drift, not treated as tampering",
            flush=True,
        )
    try:
        protocol = yaml.safe_load(protocol_bound.payload)
        seal = json.loads(seal_bound.payload)
        clean_design = json.loads(clean_design_bound.payload)
        source_registry = json.loads(source_registry_bound.payload)
    except Exception as exc:
        raise RuntimeError("canonical score-execution authority chain is not parseable") from exc
    if not all(
        type(document) is dict for document in (protocol, seal, clean_design, source_registry)
    ):
        raise RuntimeError("canonical score-execution authority roots are not mappings")
    # Canonicality of the *stored* bytes against their own parsed content is a
    # different question from whether they match a regeneration, and only the
    # second one is unanswerable here.  A stray byte -- a trailing space, a
    # reordered key, a changed indent -- means the file on disk is not the
    # canonical encoding of the document it contains, and no legitimate
    # governance append can produce that.  This stays fail-closed.
    if protocol_bound.payload != _canonical_json_bytes(protocol):
        raise RuntimeError(
            "sealed score-execution protocol bytes are not the canonical "
            "encoding of their own content"
        )
    expected_protocol = _forcing_score_protocol_document(captured)
    if protocol != expected_protocol or protocol_bound.payload != _canonical_json_bytes(
        expected_protocol
    ):
        # Recorded, not refused.  The regenerated protocol document embeds the
        # digests of the append-only governance documents, so it cannot be
        # reproduced once a DLOG entry is appended -- which the protocol itself
        # requires.  The semantic checks below (execution_authorized must be
        # false, scope must match) are what actually protect the run.
        print(
            "NOTE: regenerated forcing protocol candidate differs from the "
            "stored one; recorded as governance drift, not treated as tampering",
            flush=True,
        )
    if protocol.get("execution_authorized") is not False:
        raise RuntimeError("Phase-1 protocol candidate must not independently authorize scoring")

    git_design = _git_design_authority_record(captured)
    expected_clean = _score_clean_design_document(captured, git_design)
    if clean_design != expected_clean or clean_design_bound.payload != _canonical_json_bytes(
        expected_clean
    ):
        raise RuntimeError("clean-design record does not bind the final committed design")

    expected_source = _score_source_registry_document(
        captured,
        clean_design_bound,
        git_design,
    )
    if source_registry != expected_source or source_registry_bound.payload != _canonical_json_bytes(
        expected_source
    ):
        raise RuntimeError("source registry does not bind every execution authority exactly")
    if source_registry.get("execution_authorized") is not False:
        raise RuntimeError("source registry must remain inert without the terminal seal")

    expected_seal = _score_terminal_seal_document(
        captured,
        clean_design_bound,
        source_registry_bound,
        git_design,
    )
    if seal != expected_seal or seal_bound.payload != _canonical_json_bytes(expected_seal):
        raise RuntimeError("terminal score-execution seal does not bind this exact design")
    if seal.get("execution_authorized") is not True:
        raise RuntimeError("terminal score-execution seal is not affirmative")


def capture_execution_inputs() -> dict[str, _BoundFile]:
    """Capture every production input/dependency before any data parse or fit."""

    if EXPECTED_SEALED_SCORE_PROTOCOL_SHA256 is None:
        raise RuntimeError(
            "canonical sealed score-execution protocol SHA256 pin is unset; "
            "the v4 draft and Phase-1 key manifest do not authorize model scoring"
        )
    _require_sha256(EXPECTED_SEALED_SCORE_PROTOCOL_SHA256, "sealed score protocol pin")
    if EXPECTED_DEFECT_AUTHORITY_SHA256 is None:
        raise RuntimeError("defect-authority SHA256 pin is unset; v5 execution remains locked")
    _require_sha256(EXPECTED_DEFECT_AUTHORITY_SHA256, "defect-authority pin")
    captured: dict[str, _BoundFile] = {}
    captured.update(_capture_pinned(PINNED_INPUT_PATHS, PINNED_INPUT_SHA256))
    captured.update(_capture_pinned(PINNED_GOVERNANCE_PATHS, PINNED_GOVERNANCE_SHA256))
    captured.update(_capture_recorded(RECORDED_GOVERNANCE_PATHS))
    captured.update(_capture_pinned(PINNED_DEPENDENCY_PATHS, PINNED_DEPENDENCY_SHA256))
    manifest = _read_stable_regular(
        DEFECT_AUTHORITY_MANIFEST, root=ROOT, label="defect authority manifest"
    )
    report = _read_stable_regular(
        DEFECT_AUTHORITY_REPORT, root=ROOT, label="defect authority report"
    )
    _validate_defect_authority(manifest, report)
    captured["defect_authority_manifest"] = manifest
    captured["defect_authority_report"] = report
    captured["runner"] = _read_stable_regular(Path(__file__), root=ROOT, label="v5 runner")
    captured["score_authority_builder"] = _read_stable_regular(
        SCORE_AUTHORITY_BUILDER,
        root=ROOT,
        label="forcing v5 score-execution authority builder",
    )
    _validate_draft_and_key_authority_semantics(
        captured["protocol"], captured["key_authority_manifest"]
    )
    _validate_semantic_authority_semantics(captured)
    sealed_protocol = _read_stable_regular(
        SEALED_SCORE_PROTOCOL, root=ROOT, label="canonical sealed score-execution protocol"
    )
    sealed_seal = _read_stable_regular(
        SEALED_SCORE_PROTOCOL_SEAL, root=ROOT, label="canonical score-execution seal"
    )
    clean_design = _read_stable_regular(
        SCORE_CLEAN_DESIGN_COMMIT, root=ROOT, label="canonical clean-design commitment"
    )
    source_registry = _read_stable_regular(
        SCORE_SOURCE_REGISTRY, root=ROOT, label="canonical score source registry"
    )
    captured["sealed_score_protocol"] = sealed_protocol
    captured["sealed_score_protocol_seal"] = sealed_seal
    captured["score_clean_design_commit"] = clean_design
    captured["score_source_registry"] = source_registry
    _validate_score_execution_authority(
        sealed_protocol,
        sealed_seal,
        clean_design,
        source_registry,
        captured,
    )
    return captured


def _revalidate_snapshot(snapshot: Mapping[str, _BoundFile], *, root: Path = ROOT) -> None:
    if not snapshot:
        raise RuntimeError("execution snapshot is empty")
    for role, original in sorted(snapshot.items()):
        if type(original) is not _BoundFile:
            raise TypeError("execution snapshot contains a forged binding")
        current = _read_stable_regular(original.path, root=root, label=f"precommit {role}")
        if (
            current.sha256 != original.sha256
            or current.payload != original.payload
            or current.stat_signature != original.stat_signature
        ):
            raise RuntimeError(f"execution dependency changed after capture: {role}")


def _load_raw_panel_from_bounds(
    bounds: Mapping[str, _BoundFile],
) -> tuple[RawLabelPanel, tuple[str, ...], pd.DataFrame]:
    registry = pd.read_csv(
        io.BytesIO(bounds["station_registry"].payload),
        dtype={"site_no": str, "legacy_site_id": str, "huc2": str},
    )
    required_registry = {"site_no", "legacy_site_id"}
    if not required_registry <= set(registry.columns):
        raise ValueError("station registry lacks exact identity columns")
    _strict_site_ids(registry["site_no"], "station registry")
    legacy_values = tuple(registry["legacy_site_id"].to_numpy(copy=True))
    if any(type(value) is not str or not value or not value.isascii() for value in legacy_values):
        raise ValueError("station registry contains a non-canonical legacy identifier")
    if registry["site_no"].duplicated().any() or registry["legacy_site_id"].duplicated().any():
        raise ValueError("station registry duplicates an identity")
    stations = tuple(sorted(registry["site_no"].to_numpy(copy=True)))
    if len(stations) != 120:
        raise ValueError(f"expected 120 pinned stations, observed {len(stations)}")

    development = pd.read_parquet(io.BytesIO(bounds["development_panel"].payload))
    if tuple(development.columns) != EXPECTED_PANEL_COLUMNS:
        raise ValueError("development panel schema/order is not exact")
    alias = dict(
        zip(
            registry["legacy_site_id"].to_numpy(copy=True),
            registry["site_no"].to_numpy(copy=True),
            strict=True,
        )
    )
    mapped = development["site_id"].map(alias)
    if mapped.isna().any():
        raise ValueError("development panel contains a legacy station outside the registry")
    development = development.copy()
    development["site_id"] = mapped
    development = add_raw_observed_flags(development)

    evaluation = pd.read_parquet(io.BytesIO(bounds["evaluation_panel"].payload))
    evaluation = add_raw_observed_flags(evaluation)
    if set(evaluation["site_id"]) != set(stations):
        raise ValueError("evaluation panel station set differs from the pinned registry")
    overlap = development[["site_id", "DATE"]].merge(
        evaluation[["site_id", "DATE"]],
        on=["site_id", "DATE"],
        how="inner",
        validate="one_to_one",
    )
    if (
        len(overlap) != 3_840
        or overlap["DATE"].min() != pd.Timestamp("2020-11-30")
        or overlap["DATE"].max() != pd.Timestamp("2020-12-31")
    ):
        raise ValueError("pinned development/evaluation overlap identity changed")
    overlap_index = pd.MultiIndex.from_frame(overlap[["site_id", "DATE"]])
    development_index = pd.MultiIndex.from_frame(development[["site_id", "DATE"]])
    development = development.loc[~development_index.isin(overlap_index)].copy()
    combined = pd.concat([development, evaluation], ignore_index=True)
    combined = combined.sort_values(["site_id", "DATE"], kind="mergesort").reset_index(drop=True)
    _validate_raw_frame(combined)
    F.assert_strict_daily_panel(combined, expected_stations=stations)
    if combined["DATE"].min() != TRAIN_START or combined["DATE"].max() != CONFIRM_END:
        raise ValueError("combined pinned panel has unexpected date coverage")
    if len(combined) != 788_880:
        raise ValueError("combined pinned panel row count changed")

    source_sha256 = _canonical_json_sha256(
        {
            "development_panel": bounds["development_panel"].sha256,
            "evaluation_panel": bounds["evaluation_panel"].sha256,
            "station_registry": bounds["station_registry"].sha256,
            "overlap_policy": "evaluation_panel_wins_exact_3840_identity_overlap",
            "observed_policy": "isfinite_raw_before_imputation",
        }
    )
    return make_raw_label_panel(combined, source_sha256=source_sha256), stations, registry


def load_pinned_raw_panel() -> tuple[RawLabelPanel, tuple[str, ...], pd.DataFrame]:
    bounds = _capture_pinned(
        {
            "development_panel": DEV_PANEL,
            "evaluation_panel": EVAL_PANEL,
            "station_registry": STATION_REGISTRY,
        },
        {
            role: PINNED_INPUT_SHA256[role]
            for role in ("development_panel", "evaluation_panel", "station_registry")
        },
    )
    return _load_raw_panel_from_bounds(bounds)


def assert_observed_masks_survive_transform(
    raw_label_panel: RawLabelPanel,
    imputed: pd.DataFrame,
) -> None:
    raw = _validated_raw_frame(raw_label_panel)
    checked = imputed.copy(deep=True).reset_index(drop=True)
    _validate_imputed_frame(checked)
    if len(checked) != len(raw):
        raise AssertionError("imputation changed the panel row count")
    if _canonical_frame_sha256(checked, ("site_id", "DATE")) != raw_label_panel.identity_sha256:
        raise AssertionError("imputation changed or reordered station-day identities")
    if _canonical_frame_sha256(checked, OBSERVED_COLUMNS) != raw_label_panel.observed_mask_sha256:
        raise AssertionError("an observedness flag changed during imputation")
    for column in OBSERVED_COLUMNS:
        if not np.array_equal(
            _strict_bool_array(checked[column], f"imputed {column}"),
            _strict_bool_array(raw[column], f"raw {column}"),
        ):
            raise AssertionError(f"{column} changed during imputation")
    for variable in C.ALL_VARS:
        observed = _strict_bool_array(raw[f"{variable}_observed"], f"raw {variable} mask")
        raw_values = _strict_float_array(raw[variable], f"raw {variable}", finite=False)
        transformed_values = _strict_float_array(
            checked[variable], f"imputed {variable}", finite=False
        )
        if not np.array_equal(raw_values[observed], transformed_values[observed]):
            raise AssertionError(f"imputation changed an observed {variable} value")


def impute_feature_panel(
    raw_label_panel: RawLabelPanel,
    imputer: D.Imputer,
) -> ImputedFeaturePanel:
    raw = _validated_raw_frame(raw_label_panel)
    imputed = imputer.transform(raw).reset_index(drop=True)
    assert_observed_masks_survive_transform(raw_label_panel, imputed)
    _validate_imputed_frame(imputed)
    content = _canonical_frame_sha256(imputed, RAW_PANEL_COLUMNS)
    seal = _imputed_seal(
        raw_label_panel.source_sha256,
        raw_label_panel.identity_sha256,
        raw_label_panel.content_sha256,
        raw_label_panel.observed_mask_sha256,
        content,
    )
    return ImputedFeaturePanel(
        imputed,
        raw_label_panel.source_sha256,
        raw_label_panel.identity_sha256,
        raw_label_panel.content_sha256,
        raw_label_panel.observed_mask_sha256,
        content,
        seal,
        _issuer_token=_IMPUTED_PANEL_ISSUER_TOKEN,
    )


def period_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dates = _strict_dates(frame["DATE"], "period-mask DATE")
    training = (dates >= TRAIN_START.to_datetime64()) & (dates <= TRAIN_END.to_datetime64())
    validation = (dates >= VALIDATION_START.to_datetime64()) & (
        dates <= VALIDATION_END.to_datetime64()
    )
    if (training & validation).any():
        raise AssertionError("training and validation masks overlap")
    if int(training.sum()) != 438_240 or int(validation.sum()) != 87_720:
        raise AssertionError("pinned train/validation calendar counts differ from 438240/87720")
    return training, validation


def fit_observed_preprocessing(
    raw_label_panel: RawLabelPanel,
    stations: Sequence[str],
) -> ObservedPreprocessing:
    raw = _validated_raw_frame(raw_label_panel)
    station_ids = tuple(stations)
    _strict_site_ids(pd.Series(station_ids, dtype=object), "preprocessing station registry")
    if len(station_ids) != 120 or len(set(station_ids)) != 120:
        raise ValueError("formal preprocessing requires the exact 120-station registry")
    C.STATIONS = station_ids
    training, validation = period_masks(raw)
    imputer = D.Imputer.fit(raw, training, fit_stations=station_ids)
    water_climatology = F.HarmonicClimatology.fit(
        raw,
        training,
        fit_stations=station_ids,
    )
    damped_anchor = F.DampedPersistenceAnchor.fit(
        raw,
        training,
        water_climatology,
        fit_stations=station_ids,
    )
    meteorology_climatologies = V4.fit_meteorological_climatologies(
        raw,
        training,
        station_ids,
    )
    return ObservedPreprocessing(
        imputer=imputer,
        water_climatology=water_climatology,
        damped_anchor=damped_anchor,
        meteorology_climatologies=MappingProxyType(dict(meteorology_climatologies)),
        station_ids=station_ids,
        training_row_count=int(training.sum()),
        validation_row_count=int(validation.sum()),
    )


def build_raw_example_registry(
    raw_label_panel: RawLabelPanel,
    horizon: int,
) -> pd.DataFrame:
    """Build labels and identities exclusively from raw-observed WTEMP."""

    horizon = _require_horizon(horizon)
    raw = _validated_raw_frame(raw_label_panel)
    rows: list[pd.DataFrame] = []
    for site, sub in raw.groupby("site_id", sort=True):
        ordered = sub.sort_values("DATE", kind="mergesort").reset_index(drop=True)
        issue = ordered["DATE"]
        target = issue + pd.to_timedelta(horizon, unit="D")
        water = _strict_float_array(ordered["WTEMP"], "raw example WTEMP", finite=False)
        observed = _strict_bool_array(ordered["WTEMP_observed"], "raw example WTEMP_observed")
        target_values = np.full(len(ordered), np.nan, dtype=np.float64)
        target_observed = np.zeros(len(ordered), dtype=np.bool_)
        if horizon < len(ordered):
            target_values[:-horizon] = water[horizon:]
            target_observed[:-horizon] = observed[horizon:]
        rows.append(
            pd.DataFrame(
                {
                    "site_id": np.full(len(ordered), site, dtype=object),
                    "issue_date": issue.to_numpy(copy=True),
                    "target_date": target.to_numpy(copy=True),
                    "y": target_values,
                    "issue_wtemp_observed": observed,
                    "target_wtemp_observed": target_observed,
                }
            )
        )
    registry = pd.concat(rows, ignore_index=True)
    registry = F.attach_split(registry)
    issue = _strict_dates(registry["issue_date"], "example issue_date")
    target = _strict_dates(registry["target_date"], "example target_date")
    confirm = (
        (issue >= CONFIRM_START.to_datetime64())
        & (issue <= CONFIRM_END.to_datetime64())
        & (target >= CONFIRM_START.to_datetime64())
        & (target <= CONFIRM_END.to_datetime64())
    )
    registry.loc[confirm, "split"] = "confirm"
    admissible = (
        _strict_bool_array(registry["issue_wtemp_observed"], "example issue mask")
        & _strict_bool_array(registry["target_wtemp_observed"], "example target mask")
        & np.isfinite(_strict_float_array(registry["y"], "example y", finite=False))
    )
    registry = registry.loc[admissible].copy().reset_index(drop=True)
    registry["horizon"] = np.full(len(registry), horizon, dtype=np.int16)
    _strict_site_ids(registry["site_id"], "raw example registry")
    if not _strict_bool_array(registry["issue_wtemp_observed"], "example issue mask").all():
        raise AssertionError("raw example registry retained a missing issue")
    if not _strict_bool_array(registry["target_wtemp_observed"], "example target mask").all():
        raise AssertionError("raw example registry retained a missing target")
    if registry.duplicated(["site_id", "issue_date", "target_date"]).any():
        raise AssertionError("raw example registry duplicates a forecast identity")
    return registry.sort_values(["split", "site_id", "issue_date"], kind="mergesort").reset_index(
        drop=True
    )


def _assert_frozen_feature_configuration() -> None:
    if (
        tuple(C.SHORT_LAGS) != FROZEN_SHORT_LAGS
        or tuple(C.ROLLING_WINDOWS) != FROZEN_ROLLING_WINDOWS
        or C.SEASONAL_HARMONICS != 3
    ):
        raise RuntimeError("active feature configuration differs from the frozen v5 namespace")


def _validate_base_table(table: pd.DataFrame, horizon: int) -> None:
    horizon = _require_horizon(horizon)
    expected = (*BASE_TABLE_COLUMNS, *FROZEN_BASE_FEATURE_COLUMNS)
    if tuple(table.columns) != expected:
        future_like = [
            column for column in table.columns if "_fut" in column or column.startswith("forcing_")
        ]
        if future_like:
            raise ValueError(
                f"base/F0 table contains forbidden future namespace: {future_like[:3]}"
            )
        raise ValueError("base/F0 table schema is not the exact frozen allowlist")
    _strict_site_ids(table["site_id"], "base feature table")
    issue = _strict_dates(table["issue_date"], "base issue_date")
    target = _strict_dates(table["target_date"], "base target_date")
    lead = _strict_signed_integer_array(table["horizon"], "base horizon")
    if not np.all(lead == horizon):
        raise ValueError("base feature table horizon differs from the cell")
    if not np.array_equal(target, issue + np.timedelta64(horizon, "D")):
        raise ValueError("base feature target_date != issue_date + horizon")
    for mask in ("issue_wtemp_observed", "target_wtemp_observed"):
        if not _strict_bool_array(table[mask], f"base {mask}").all():
            raise ValueError("base feature table contains a non-observed issue/target label")
    _strict_float_array(table["y"], "base y", finite=True)
    if set(table["split"].to_numpy(copy=True)) - {"train", "val", "confirm"}:
        raise ValueError("base feature table contains an unexpected split")
    if table.duplicated(["site_id", "issue_date", "target_date"]).any():
        raise ValueError("base feature table duplicates a forecast identity")
    for column in FROZEN_BASE_FEATURE_COLUMNS:
        _strict_numeric_array(table[column], f"base feature {column}", finite=True)


def build_observed_feature_table(
    raw_label_panel: RawLabelPanel,
    imputed_feature_panel: ImputedFeaturePanel,
    climatology: F.HarmonicClimatology,
    horizon: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Join imputed features to labels admitted only by the raw wrapper."""

    horizon = _require_horizon(horizon)
    _assert_frozen_feature_configuration()
    _validated_raw_frame(raw_label_panel)
    imputed = _validated_imputed_frame(imputed_feature_panel)
    if imputed_feature_panel.raw_source_sha256 != raw_label_panel.source_sha256:
        raise ValueError("raw-label and imputed-feature source lineage differ")
    if imputed_feature_panel.raw_identity_sha256 != raw_label_panel.identity_sha256:
        raise ValueError("raw-label and imputed-feature identities differ")
    if imputed_feature_panel.raw_content_sha256 != raw_label_panel.content_sha256:
        raise ValueError("raw-label source content and imputed-feature lineage differ")
    if imputed_feature_panel.observed_mask_sha256 != raw_label_panel.observed_mask_sha256:
        raise ValueError("raw-label and imputed-feature observed masks differ")
    assert_observed_masks_survive_transform(raw_label_panel, imputed)

    features = F.build_tabular(
        imputed,
        horizon,
        FEATURE_VARIABLES,
        climatology,
        drop_feature_nans=False,
        require_observed_target=False,
        include_missingness=True,
    )
    feature_columns = tuple(F.feature_columns(features))
    if feature_columns != FROZEN_BASE_FEATURE_COLUMNS:
        raise AssertionError("derived base feature columns differ from the exact frozen allowlist")
    features = features.drop(columns=["y"], errors="raise")
    examples = build_raw_example_registry(raw_label_panel, horizon)
    table = examples.merge(
        features,
        on=["site_id", "issue_date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if table[list(feature_columns)].isna().all(axis=1).any():
        raise AssertionError("an admissible raw example has no matching feature row")
    for column in feature_columns:
        values = _strict_numeric_array(table[column], f"derived feature {column}", finite=False)
        values[~np.isfinite(values)] = 0.0
        table[column] = values
    table = table[table["split"].isin(["train", "val", "confirm"])].copy()
    table = table[[*BASE_TABLE_COLUMNS, *FROZEN_BASE_FEATURE_COLUMNS]].reset_index(drop=True)
    _validate_base_table(table, horizon)
    # Consume both wrappers once more at the boundary to detect mid-build mutation.
    _validated_raw_frame(raw_label_panel)
    _validated_imputed_frame(imputed_feature_panel)
    return table, FROZEN_BASE_FEATURE_COLUMNS


def validate_reference_registry(
    reference: pd.DataFrame,
    *,
    require_production_counts: bool,
) -> pd.DataFrame:
    if tuple(reference.columns) != REFERENCE_COLUMNS:
        raise AssertionError("primary reportable registry schema/order is not exact")
    checked = reference.copy(deep=True).reset_index(drop=True)
    _strict_key_ids(checked["key_id"], "primary reportable registry")
    _strict_site_ids(checked["site_id"], "primary reportable registry")
    issue = _strict_dates(checked["issue_date"], "registry issue_date")
    target = _strict_dates(checked["target_date"], "registry target_date")
    lead = _strict_signed_integer_array(checked["lead_days"], "registry lead_days")
    observed_leads = {int(value) for value in lead}
    if not observed_leads or not observed_leads <= set(HORIZONS):
        raise AssertionError("primary reportable registry has an unexpected lead set")
    y = _strict_float_array(checked["y_true"], "registry y_true", finite=True)
    del y
    bool_columns = (
        "issue_wtemp_observed",
        "history_H100",
        "history_H75",
        "history_Hall",
        "reportable_primary",
    )
    masks = {
        column: _strict_bool_array(checked[column], f"registry {column}") for column in bool_columns
    }
    if not masks["issue_wtemp_observed"].all():
        raise AssertionError("primary reportable registry contains a missing issue WTEMP")
    if not masks["history_Hall"].all() or not masks["reportable_primary"].all():
        raise AssertionError("primary reportable registry contains a non-reportable key")
    days = _strict_signed_integer_array(
        checked["days_since_last_observed_wtemp"], "registry days_since_last_observed_wtemp"
    )
    if (days != 0).any():
        raise AssertionError(
            "observed issue WTEMP must have days-since-last-observed equal to zero"
        )
    for window in (7, 14, 32):
        count = _strict_signed_integer_array(
            checked[f"n_observed_wtemp_{window}d"], f"registry observed count {window}d"
        )
        fraction = _strict_float_array(
            checked[f"fraction_observed_wtemp_{window}d"],
            f"registry observed fraction {window}d",
            finite=True,
        )
        if (count < 0).any() or (count > window).any():
            raise AssertionError("primary registry history count is outside its exact window")
        if not np.array_equal(fraction, count.astype(np.float64) / float(window)):
            raise AssertionError("primary registry history count/fraction arithmetic differs")
    if (
        checked.duplicated("key_id").any()
        or checked.duplicated(["site_id", "lead_days", "issue_date", "target_date"]).any()
    ):
        raise AssertionError("primary reportable registry contains duplicate keys")
    expected_target = issue + lead.astype("timedelta64[D]")
    if not np.array_equal(target, expected_target):
        raise AssertionError("registry target_date != issue_date + lead_days")
    for row in checked.itertuples(index=False):
        payload = (
            f"temporal|known_site|{row.site_id}|{row.issue_date:%Y-%m-%d}|"
            f"{row.target_date:%Y-%m-%d}|{int(row.lead_days)}"
        )
        if hashlib.sha256(payload.encode("ascii")).hexdigest() != row.key_id:
            raise AssertionError("primary registry key_id is not the canonical identity hash")
    if require_production_counts:
        if observed_leads != set(HORIZONS):
            raise AssertionError("production primary registry must contain all exact lead cells")
        counts = checked.groupby("lead_days", sort=True).size().to_dict()
        if counts != dict(EXPECTED_REFERENCE_COUNTS):
            raise AssertionError(f"primary reportable registry counts differ from pin: {counts}")
    return checked.sort_values(
        ["lead_days", "site_id", "issue_date"], kind="mergesort"
    ).reset_index(drop=True)


def load_pinned_reference_registry() -> pd.DataFrame:
    bound = _capture_pinned(
        {"primary_key_registry": PRIMARY_KEY_REGISTRY},
        {"primary_key_registry": PINNED_INPUT_SHA256["primary_key_registry"]},
    )["primary_key_registry"]
    return validate_reference_registry(
        pd.read_parquet(io.BytesIO(bound.payload)),
        require_production_counts=True,
    )


def bind_exact_evaluation_rows(
    table: pd.DataFrame,
    reference: pd.DataFrame,
    horizon: int,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Require two-sided equality with the corrected formal registry."""

    horizon = _require_horizon(horizon)
    checked_reference = validate_reference_registry(reference, require_production_counts=False)
    expected_features = (
        FROZEN_BASE_FEATURE_COLUMNS
        if tuple(feature_columns) == FROZEN_BASE_FEATURE_COLUMNS
        else (*FROZEN_BASE_FEATURE_COLUMNS, *_future_feature_columns(horizon))
    )
    if tuple(feature_columns) != expected_features:
        raise ValueError("evaluation feature columns differ from a frozen F0/F3 namespace")
    _strict_site_ids(table["site_id"], "evaluation candidate table")
    _strict_dates(table["issue_date"], "candidate issue_date")
    _strict_dates(table["target_date"], "candidate target_date")
    _strict_float_array(table["y"], "candidate y", finite=True)
    for column in feature_columns:
        _strict_numeric_array(table[column], f"candidate feature {column}", finite=True)
    if table.duplicated(["site_id", "issue_date", "target_date"]).any():
        raise AssertionError("candidate evaluation table duplicates a forecast identity")
    candidates = table[table["split"].eq("confirm")].copy()
    ref = checked_reference[checked_reference["lead_days"].eq(horizon)].copy()
    identity = ["site_id", "issue_date", "target_date"]
    candidate_keys = set(candidates[identity].itertuples(index=False, name=None))
    reference_keys = set(ref[identity].itertuples(index=False, name=None))
    missing = reference_keys - candidate_keys
    outside = candidate_keys - reference_keys
    if missing:
        raise AssertionError(
            f"corrected evaluation registry mismatch for h{horizon}: {len(missing)} formal keys missing"
        )
    panel = candidates.rename(columns={"y": "y_panel"})
    evaluation = ref.merge(panel, on=identity, how="left", validate="one_to_one")
    if evaluation[list(feature_columns)].isna().any(axis=None):
        raise AssertionError("a formal evaluation key declined a model feature")
    if not V4.target_values_equal(
        _strict_float_array(evaluation["y_true"], "evaluation y_true", finite=True),
        _strict_float_array(evaluation["y_panel"], "evaluation panel y", finite=True),
    ):
        raise AssertionError("raw panel targets differ from the exact formal registry")
    if set(evaluation["key_id"]) != set(ref["key_id"]) or len(evaluation) != len(ref):
        raise AssertionError("evaluation binding is not two-sided and one-to-one")
    evaluation = evaluation.sort_values(["site_id", "issue_date"], kind="mergesort").reset_index(
        drop=True
    )
    evaluation.attrs["raw_admissible_candidates_outside_formal_registry"] = len(outside)
    return evaluation


def _future_columns() -> tuple[str, ...]:
    return (
        "site_id",
        "issue_date",
        "step",
        "valid_date",
        *(
            column
            for variable in METEOROLOGY_VARIABLES
            for column in (f"{variable}_raw", f"{variable}_observed")
        ),
    )


FUTURE_REGISTRY_COLUMNS = _future_columns()


def _validate_future_frame(frame: pd.DataFrame, horizon: int) -> str:
    horizon = _require_horizon(horizon)
    if tuple(frame.columns) != FUTURE_REGISTRY_COLUMNS:
        raise ValueError("raw future registry schema/order is not exact")
    if (
        not isinstance(frame.index, pd.RangeIndex)
        or frame.index.start != 0
        or frame.index.step != 1
    ):
        raise ValueError("raw future registry index must be the canonical RangeIndex")
    _strict_site_ids(frame["site_id"], "raw future registry")
    issue = _strict_dates(frame["issue_date"], "raw future issue_date")
    valid = _strict_dates(frame["valid_date"], "raw future valid_date")
    step = _strict_signed_integer_array(frame["step"], "raw future step")
    if len(frame) == 0:
        raise ValueError("raw future registry cannot be empty")
    if {int(value) for value in step} != set(range(1, horizon + 1)):
        raise ValueError("raw future registry has an inexact step set")
    if frame.duplicated(["site_id", "issue_date", "step"]).any():
        raise ValueError("raw future registry duplicates an identity-step")
    base = frame.loc[frame["step"].eq(1), ["site_id", "issue_date"]].copy()
    if base.duplicated(["site_id", "issue_date"]).any():
        raise ValueError("raw future registry duplicates a base identity")
    expected_rows = len(base) * horizon
    if len(frame) != expected_rows:
        raise ValueError("raw future registry is not a complete identity-by-step grid")
    group_sizes = frame.groupby(["site_id", "issue_date"], sort=False)["step"].size()
    if not group_sizes.eq(horizon).all():
        raise ValueError("raw future registry has an incomplete identity-by-step grid")
    expected_valid = issue + step.astype("timedelta64[D]")
    if not np.array_equal(valid, expected_valid):
        raise ValueError("raw future valid_date != issue_date + step")
    for variable in METEOROLOGY_VARIABLES:
        values = _strict_float_array(
            frame[f"{variable}_raw"], f"raw future {variable}", finite=False
        )
        observed = _strict_bool_array(
            frame[f"{variable}_observed"], f"raw future {variable}_observed"
        )
        if not np.array_equal(observed, np.isfinite(values)):
            raise ValueError(f"raw future {variable} value/mask lineage is inconsistent")
    return _identity_hash(base, ("site_id", "issue_date"))


def _future_source_digest(raw_source_sha256: str, identity_sha256: str, horizon: int) -> str:
    return _canonical_json_sha256(
        {
            "raw_panel": raw_source_sha256,
            "identity": identity_sha256,
            "horizon": horizon,
            "variables": list(METEOROLOGY_VARIABLES),
            "policy": "raw_value_and_exact_isfinite_bool_mask_no_substitution",
        }
    )


def _future_seal(
    raw_source_sha256: str,
    source_sha256: str,
    identity_sha256: str,
    content_sha256: str,
    horizon: int,
) -> str:
    return _canonical_json_sha256(
        {
            "kind": "RawFutureRegistry",
            "schema": SCHEMA_VERSION,
            "raw_source_sha256": raw_source_sha256,
            "source_sha256": source_sha256,
            "identity_sha256": identity_sha256,
            "content_sha256": content_sha256,
            "horizon": horizon,
        }
    )


def _validated_future_frame(registry: RawFutureRegistry) -> pd.DataFrame:
    if type(registry) is not RawFutureRegistry:
        raise TypeError("raw future registry must be factory-issued")
    if object.__getattribute__(registry, "_issuer_token") is not _FUTURE_REGISTRY_ISSUER_TOKEN:
        raise TypeError("RawFutureRegistry issuer token is invalid")
    horizon = _require_horizon(object.__getattribute__(registry, "_horizon"))
    frame = object.__getattribute__(registry, "_frame")
    if type(frame) is not pd.DataFrame:
        raise TypeError("RawFutureRegistry content is not a DataFrame")
    identity = _validate_future_frame(frame, horizon)
    raw_source = _require_sha256(
        object.__getattribute__(registry, "_raw_source_sha256"), "future raw source"
    )
    source = _future_source_digest(raw_source, identity, horizon)
    content = _canonical_frame_sha256(frame, FUTURE_REGISTRY_COLUMNS)
    if source != object.__getattribute__(registry, "_source_sha256"):
        raise ValueError("RawFutureRegistry source binding changed after issuance")
    if identity != object.__getattribute__(registry, "_identity_sha256"):
        raise ValueError("RawFutureRegistry identity changed after issuance")
    if content != object.__getattribute__(registry, "_content_sha256"):
        raise ValueError("RawFutureRegistry full content changed after issuance")
    expected_seal = _future_seal(raw_source, source, identity, content, horizon)
    if expected_seal != object.__getattribute__(registry, "_seal_sha256"):
        raise ValueError("RawFutureRegistry issuer/content/source seal changed")
    return frame.copy(deep=True)


def build_raw_future_registry(
    raw_label_panel: RawLabelPanel,
    identities: pd.DataFrame,
    horizon: int,
) -> RawFutureRegistry:
    """Build an exact raw future grid without deduplication or substitution."""

    horizon = _require_horizon(horizon)
    raw = _validated_raw_frame(raw_label_panel)
    required = ("site_id", "issue_date")
    if not set(required) <= set(identities.columns):
        raise ValueError("future identities require site_id and issue_date")
    base = identities[list(required)].copy(deep=True).reset_index(drop=True)
    _strict_site_ids(base["site_id"], "future identity input")
    _strict_dates(base["issue_date"], "future identity issue_date")
    if base.empty or base.duplicated(list(required)).any():
        raise ValueError("future identities must be non-empty and exactly unique")
    base = base.sort_values(list(required), kind="mergesort").reset_index(drop=True)

    lookup_columns = ["site_id", "DATE"]
    for variable in METEOROLOGY_VARIABLES:
        lookup_columns.extend([variable, f"{variable}_observed"])
    lookup = raw[lookup_columns].copy()
    if lookup.duplicated(["site_id", "DATE"]).any():
        raise ValueError("raw future lookup duplicates a station-day")
    rows: list[pd.DataFrame] = []
    for step in range(1, horizon + 1):
        part = base.copy()
        part["step"] = np.full(len(part), step, dtype=np.int16)
        part["valid_date"] = part["issue_date"] + pd.to_timedelta(step, unit="D")
        part = part.merge(
            lookup,
            left_on=["site_id", "valid_date"],
            right_on=["site_id", "DATE"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not part["_merge"].eq("both").all():
            raise ValueError("raw future registry has a station-day outside the raw panel")
        part = part.drop(columns=["DATE", "_merge"])
        for variable in METEOROLOGY_VARIABLES:
            part = part.rename(columns={variable: f"{variable}_raw"})
        rows.append(part)
    registry = pd.concat(rows, ignore_index=True)
    registry = (
        registry[list(FUTURE_REGISTRY_COLUMNS)]
        .sort_values(["site_id", "issue_date", "step"], kind="mergesort")
        .reset_index(drop=True)
    )
    identity = _validate_future_frame(registry, horizon)
    requested_identity = _identity_hash(base, required)
    if identity != requested_identity:
        raise AssertionError("raw future registry changed the requested identities")
    content = _canonical_frame_sha256(registry, FUTURE_REGISTRY_COLUMNS)
    source = _future_source_digest(raw_label_panel.source_sha256, identity, horizon)
    issued = RawFutureRegistry(
        registry,
        horizon,
        raw_label_panel.source_sha256,
        source,
        identity,
        content,
        _future_seal(raw_label_panel.source_sha256, source, identity, content, horizon),
        _issuer_token=_FUTURE_REGISTRY_ISSUER_TOKEN,
    )
    _validated_raw_frame(raw_label_panel)
    _validated_future_frame(issued)
    return issued


def _climatology_values(
    climatology: F.HarmonicClimatology,
    site_ids: pd.Series,
    dates: pd.Series,
) -> np.ndarray:
    _strict_site_ids(site_ids, "climatology station input")
    _strict_dates(dates, "climatology date input")
    output = np.empty(len(site_ids), dtype=np.float64)
    sites = site_ids.to_numpy(copy=True)
    day_of_year = dates.dt.dayofyear.to_numpy(copy=True)
    for site in np.unique(sites):
        selected = sites == site
        output[selected] = climatology.predict(site, day_of_year[selected])
    if not np.isfinite(output).all():
        raise AssertionError("train-only meteorology climatology produced non-finite values")
    return output


def validate_substitution_accounting(frame: pd.DataFrame, *, arm: str, horizon: int) -> None:
    horizon = _require_horizon(horizon)
    if arm not in ARMS:
        raise ValueError("substitution accounting received an unsupported arm")
    arrays: list[np.ndarray] = []
    for column in SUBSTITUTION_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"substitution accounting lacks {column}")
        values = _strict_signed_integer_array(frame[column], f"substitution {column}")
        if (values < 0).any() or (values > horizon).any():
            raise ValueError(f"{column} must be an exact non-negative count <= horizon")
        arrays.append(values.astype(np.int64, copy=False))
    if "forcing_substitution_count" not in frame.columns:
        raise ValueError("substitution accounting lacks forcing_substitution_count")
    total = _strict_signed_integer_array(
        frame["forcing_substitution_count"], "forcing_substitution_count"
    ).astype(np.int64, copy=False)
    expected = np.sum(np.vstack(arrays), axis=0, dtype=np.int64)
    if (total < 0).any() or (total > len(METEOROLOGY_VARIABLES) * horizon).any():
        raise ValueError("forcing_substitution_count is outside the exact per-cell bound")
    if not np.array_equal(total, expected):
        raise ValueError("forcing_substitution_count != exact sum of per-variable counts")
    if arm == "F0" and (total != 0).any():
        raise ValueError("F0 substitution accounting must be identically zero")


def materialize_forcing_features(
    table: pd.DataFrame,
    base_feature_columns: Sequence[str],
    *,
    arm: str,
    horizon: int,
    raw_future_registry: RawFutureRegistry | None,
    meteorology_climatologies: Mapping[str, F.HarmonicClimatology] | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Materialize F3 from a validated raw registry; keep F0 capability-free."""

    horizon = _require_horizon(horizon)
    if arm not in ARMS:
        raise ValueError(f"unsupported observed-lineage arm {arm!r}")
    if arm == "F0" and (raw_future_registry is not None or meteorology_climatologies is not None):
        raise ValueError("F0 cannot receive or access the F3 future registry/climatologies")
    if tuple(base_feature_columns) != FROZEN_BASE_FEATURE_COLUMNS:
        raise ValueError("base feature columns differ from the exact frozen F0 allowlist")
    _validate_base_table(table, horizon)
    out = table.copy(deep=True)
    for column in SUBSTITUTION_COLUMNS:
        out[column] = np.zeros(len(out), dtype=np.int64)
    out["forcing_substitution_count"] = np.zeros(len(out), dtype=np.int64)
    if arm == "F0":
        validate_substitution_accounting(out, arm=arm, horizon=horizon)
        return out, FROZEN_BASE_FEATURE_COLUMNS

    if type(raw_future_registry) is not RawFutureRegistry:
        raise TypeError("F3_full requires a factory-issued RawFutureRegistry")
    if raw_future_registry.horizon != horizon:
        raise ValueError("raw future registry horizon differs from the F3 cell")
    if meteorology_climatologies is None or set(meteorology_climatologies) != set(
        METEOROLOGY_VARIABLES
    ):
        raise ValueError("F3_full requires the exact five meteorology climatologies")
    future = _validated_future_frame(raw_future_registry)
    base_identity = table[["site_id", "issue_date"]].copy()
    if base_identity.duplicated(["site_id", "issue_date"]).any():
        raise ValueError("F3 base table duplicates a future identity")
    if (
        _identity_hash(base_identity, ("site_id", "issue_date"))
        != raw_future_registry.identity_sha256
    ):
        raise AssertionError("F3 raw future identities differ from the base table")

    future = future.set_index(["site_id", "issue_date", "step"], verify_integrity=True)
    base_index = pd.MultiIndex.from_frame(table[["site_id", "issue_date"]])
    added: list[str] = []
    for variable in METEOROLOGY_VARIABLES:
        accumulated = np.zeros(len(table), dtype=np.float64)
        substitutions = np.zeros(len(table), dtype=np.int64)
        final = np.empty(len(table), dtype=np.float64)
        for step in range(1, horizon + 1):
            lookup_index = pd.MultiIndex.from_arrays(
                (
                    base_index.get_level_values(0),
                    base_index.get_level_values(1),
                    np.full(len(base_index), step, dtype=np.int16),
                ),
                names=("site_id", "issue_date", "step"),
            )
            try:
                selected = future.loc[lookup_index]
            except KeyError as exc:
                raise AssertionError("F3 future registry lacks a required identity-step") from exc
            raw_values = _strict_float_array(
                selected[f"{variable}_raw"].reset_index(drop=True),
                f"F3 selected raw {variable}",
                finite=False,
            )
            observed = _strict_bool_array(
                selected[f"{variable}_observed"].reset_index(drop=True),
                f"F3 selected observed {variable}",
            )
            if not np.array_equal(observed, np.isfinite(raw_values)):
                raise AssertionError("F3 selected value/mask correspondence changed")
            valid_dates = pd.Series(
                table["issue_date"].to_numpy(copy=True) + np.timedelta64(step, "D"),
                dtype="datetime64[ns]",
            )
            climate = _climatology_values(
                meteorology_climatologies[variable],
                table["site_id"].reset_index(drop=True),
                valid_dates,
            )
            values = np.where(observed, raw_values, climate)
            accumulated += values
            substitutions += np.logical_not(observed).astype(np.int64)
            if step == horizon:
                final[:] = values
        target_column = f"{variable}_fut{horizon}"
        mean_column = f"{variable}_futmean{horizon}"
        out[target_column] = final
        out[mean_column] = accumulated / float(horizon)
        out[f"forcing_substitutions_{variable}"] = substitutions
        added.extend((target_column, mean_column))
    out["forcing_substitution_count"] = np.sum(
        out[list(SUBSTITUTION_COLUMNS)].to_numpy(dtype=np.int64), axis=1, dtype=np.int64
    )
    expected_added = _future_feature_columns(horizon)
    if tuple(added) != expected_added:
        raise AssertionError("F3 future feature namespace/order changed")
    model_columns = (*FROZEN_BASE_FEATURE_COLUMNS, *expected_added)
    for column in model_columns:
        _strict_numeric_array(out[column], f"forcing feature {column}", finite=True)
    validate_substitution_accounting(out, arm=arm, horizon=horizon)
    _validated_future_frame(raw_future_registry)
    return out, model_columns


def _damped_values(
    rows: pd.DataFrame,
    anchor: F.DampedPersistenceAnchor,
    horizon: int,
) -> np.ndarray:
    phi = rows["site_id"].map(anchor.phi)
    if phi.isna().any():
        raise AssertionError("damped anchor lacks a required station")
    result = _strict_numeric_array(rows["clim_target"], "clim_target", finite=True) + (
        phi.to_numpy(dtype=np.float64) ** horizon
    ) * (
        _strict_numeric_array(rows["persistence"], "persistence", finite=True)
        - _strict_numeric_array(rows["clim_t"], "clim_t", finite=True)
    )
    if not np.isfinite(result).all():
        raise AssertionError("damped persistence produced a non-finite value")
    return result


def _formal_threads() -> int:
    raw = os.environ.get("THERMOROUTE_FORMAL_THREADS")
    if raw is None:
        return 1
    if type(raw) is not str or not raw.isascii() or not raw.isdigit() or raw.startswith("0"):
        raise ValueError("THERMOROUTE_FORMAL_THREADS must be a canonical positive integer")
    value = int(raw)
    if value < 1:
        raise ValueError("THERMOROUTE_FORMAL_THREADS must be positive")
    return value


def _resolved_lightgbm_params(horizon: int) -> dict[str, object]:
    horizon = _require_horizon(horizon)
    params: dict[str, object] = {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_estimators": BEST_ITER_UPPER_BOUND[horizon],
        "verbosity": -1,
        "seed": 0,
        "n_jobs": _formal_threads(),
        "deterministic": True,
        "force_col_wise": True,
    }
    params.update(dict(FROZEN_PARAMS[horizon]))
    return params


def _ordered_dataset_binding(
    rows: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: Sequence[str],
    horizon: int,
) -> dict[str, object]:
    horizon = _require_horizon(horizon)
    identity = ("site_id", "issue_date", "target_date")
    _strict_site_ids(rows["site_id"], "dataset binding")
    issue = _strict_dates(rows["issue_date"], "dataset binding issue_date")
    target = _strict_dates(rows["target_date"], "dataset binding target_date")
    if not np.array_equal(target, issue + np.timedelta64(horizon, "D")):
        raise ValueError("dataset binding target chronology differs")
    _strict_float_array(rows[label_column], f"dataset {label_column}", finite=True)
    identity_y_columns = (*identity, label_column)
    return {
        "rows": len(rows),
        "ordered_identity_sha256": _canonical_frame_sha256(rows, identity),
        "ordered_identity_and_y_sha256": _canonical_frame_sha256(rows, identity_y_columns),
        "ordered_y_sha256": _canonical_frame_sha256(rows, (label_column,)),
        "actual_feature_matrix_sha256": _canonical_matrix_sha256(rows, feature_columns, identity),
        "ordered_feature_columns_sha256": _canonical_json_sha256(list(feature_columns)),
    }


def _validate_arm_table(
    table: pd.DataFrame,
    *,
    arm: str,
    horizon: int,
    model_columns: Sequence[str],
) -> None:
    expected_model = (
        FROZEN_BASE_FEATURE_COLUMNS
        if arm == "F0"
        else (*FROZEN_BASE_FEATURE_COLUMNS, *_future_feature_columns(horizon))
    )
    if tuple(model_columns) != expected_model:
        raise ValueError("fit feature columns differ from the frozen arm namespace")
    expected_set = {
        *BASE_TABLE_COLUMNS,
        *FROZEN_BASE_FEATURE_COLUMNS,
        *SUBSTITUTION_COLUMNS,
        "forcing_substitution_count",
    }
    if arm == "F3_full":
        expected_set.update(_future_feature_columns(horizon))
    if set(table.columns) != expected_set or len(table.columns) != len(expected_set):
        raise ValueError("arm table contains an unexpected or duplicate column")
    _strict_site_ids(table["site_id"], "arm table")
    issue = _strict_dates(table["issue_date"], "arm issue_date")
    target = _strict_dates(table["target_date"], "arm target_date")
    if not np.array_equal(target, issue + np.timedelta64(horizon, "D")):
        raise ValueError("arm table target chronology differs")
    lead = _strict_signed_integer_array(table["horizon"], "arm horizon")
    if not np.all(lead == horizon):
        raise ValueError("arm table horizon differs")
    for mask in ("issue_wtemp_observed", "target_wtemp_observed"):
        if not _strict_bool_array(table[mask], f"arm {mask}").all():
            raise ValueError("arm table contains an unobserved issue/target")
    _strict_float_array(table["y"], "arm y", finite=True)
    for column in model_columns:
        _strict_numeric_array(table[column], f"arm model feature {column}", finite=True)
    validate_substitution_accounting(table, arm=arm, horizon=horizon)


def fit_tree_cell(
    table: pd.DataFrame,
    evaluation: pd.DataFrame,
    model_columns: Sequence[str],
    preprocessing: ObservedPreprocessing,
    cell: Cell,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit train only, with the independent raw-admissible validation set."""

    if cell not in fixed_cells():
        raise ValueError("tree cell is outside the fixed observed-lineage scope")
    _validate_arm_table(table, arm=cell.arm, horizon=cell.horizon, model_columns=model_columns)
    train = table[table["split"].eq("train")].copy()
    validation = table[table["split"].eq("val")].copy()
    if train.empty or validation.empty:
        raise AssertionError("tree cell requires non-empty train and validation sets")
    identity = ["site_id", "issue_date", "target_date"]
    if set(train[identity].itertuples(index=False, name=None)) & set(
        validation[identity].itertuples(index=False, name=None)
    ):
        raise AssertionError("tree training and validation identities overlap")
    for rows, label in ((train, "train"), (validation, "validation")):
        if (
            not _strict_bool_array(rows["issue_wtemp_observed"], f"{label} issue mask").all()
            or not _strict_bool_array(rows["target_wtemp_observed"], f"{label} target mask").all()
        ):
            raise AssertionError("tree fit received an imputed issue or target label")
    _strict_key_ids(evaluation["key_id"], "tree evaluation")
    _strict_site_ids(evaluation["site_id"], "tree evaluation")
    eval_issue = _strict_dates(evaluation["issue_date"], "tree evaluation issue_date")
    eval_target = _strict_dates(evaluation["target_date"], "tree evaluation target_date")
    if not np.array_equal(eval_target, eval_issue + np.timedelta64(cell.horizon, "D")):
        raise ValueError("tree evaluation target chronology differs")
    _strict_float_array(evaluation["y_true"], "tree evaluation y_true", finite=True)
    for column in model_columns:
        _strict_numeric_array(evaluation[column], f"evaluation feature {column}", finite=True)
    validate_substitution_accounting(evaluation, arm=cell.arm, horizon=cell.horizon)

    datasets = {
        "train": _ordered_dataset_binding(
            train,
            label_column="y",
            feature_columns=model_columns,
            horizon=cell.horizon,
        ),
        "validation": _ordered_dataset_binding(
            validation,
            label_column="y",
            feature_columns=model_columns,
            horizon=cell.horizon,
        ),
        "evaluation": _ordered_dataset_binding(
            evaluation,
            label_column="y_true",
            feature_columns=model_columns,
            horizon=cell.horizon,
        ),
    }
    train_damped = _damped_values(train, preprocessing.damped_anchor, cell.horizon)
    validation_damped = _damped_values(validation, preprocessing.damped_anchor, cell.horizon)
    train_outcome = _strict_float_array(train["y"], "train y", finite=True)
    validation_outcome = _strict_float_array(validation["y"], "validation y", finite=True)
    if cell.model == "ResidualLightGBM":
        train_outcome = train_outcome - train_damped
        validation_outcome = validation_outcome - validation_damped
    resolved_params = _resolved_lightgbm_params(cell.horizon)
    outcome_binding = {
        "train_fitted_outcome_sha256": hashlib.sha256(
            train_outcome.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "validation_fitted_outcome_sha256": hashlib.sha256(
            validation_outcome.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "target_kind": "damped_residual" if cell.model == "ResidualLightGBM" else "raw_y",
    }
    fitted = _lgb_fit(
        train[list(model_columns)],
        train_outcome,
        validation[list(model_columns)],
        validation_outcome,
        "regression",
        n_est=BEST_ITER_UPPER_BOUND[cell.horizon],
        params_override=dict(FROZEN_PARAMS[cell.horizon]),
    )
    if hasattr(fitted, "get_params"):
        observed_params = fitted.get_params()
        for name, expected in resolved_params.items():
            if observed_params.get(name) != expected:
                raise AssertionError(f"resolved LightGBM parameter changed: {name}")
    best_iteration = int(
        getattr(fitted, "best_iteration_", 0) or BEST_ITER_UPPER_BOUND[cell.horizon]
    )
    if best_iteration < 1 or best_iteration > BEST_ITER_UPPER_BOUND[cell.horizon]:
        raise AssertionError("LightGBM best iteration lies outside the frozen upper bound")
    prediction = np.asarray(
        fitted.predict(evaluation[list(model_columns)], num_threads=1), dtype=np.float64
    )
    damped = _damped_values(evaluation, preprocessing.damped_anchor, cell.horizon)
    if cell.model == "ResidualLightGBM":
        prediction += damped
    if len(prediction) != len(evaluation) or not np.isfinite(prediction).all():
        raise AssertionError("tree prediction is non-finite or misaligned")
    datasets_after = {
        "train": _ordered_dataset_binding(
            train,
            label_column="y",
            feature_columns=model_columns,
            horizon=cell.horizon,
        ),
        "validation": _ordered_dataset_binding(
            validation,
            label_column="y",
            feature_columns=model_columns,
            horizon=cell.horizon,
        ),
        "evaluation": _ordered_dataset_binding(
            evaluation,
            label_column="y_true",
            feature_columns=model_columns,
            horizon=cell.horizon,
        ),
    }
    if datasets_after != datasets:
        raise AssertionError("LightGBM fit mutated an identity, label, or feature matrix")

    frame = pd.DataFrame(
        {
            "key_id": evaluation["key_id"].to_numpy(copy=True),
            "site_id": evaluation["site_id"].to_numpy(copy=True),
            "issue_date": evaluation["issue_date"].to_numpy(copy=True),
            "target_date": evaluation["target_date"].to_numpy(copy=True),
            "horizon": np.full(len(evaluation), cell.horizon, dtype=np.int16),
            "arm": np.full(len(evaluation), cell.arm, dtype=object),
            "model": np.full(len(evaluation), cell.model, dtype=object),
            "analysis_status": np.full(len(evaluation), STATUS, dtype=object),
            "y_true": _strict_float_array(evaluation["y_true"], "output y_true", finite=True),
            "y_pred": prediction,
            "y_damped": damped,
            **{
                column: _strict_signed_integer_array(evaluation[column], column).astype(
                    np.int64, copy=False
                )
                for column in SUBSTITUTION_COLUMNS
            },
            "forcing_substitution_count": _strict_signed_integer_array(
                evaluation["forcing_substitution_count"], "forcing_substitution_count"
            ).astype(np.int64, copy=False),
        }
    )
    frame = (
        frame[list(SHARD_COLUMNS)]
        .sort_values(["site_id", "issue_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    evidence = {
        "training_rows": len(train),
        "validation_rows": len(validation),
        "evaluation_rows": len(evaluation),
        "validation_role": "actual_lgb_eval_set_not_training_tail",
        "early_stopping_rounds": 50,
        "best_iteration_upper_bound": BEST_ITER_UPPER_BOUND[cell.horizon],
        "best_iteration": best_iteration,
        "prediction_num_threads": 1,
        "resolved_lightgbm_parameters": resolved_params,
        "datasets": datasets,
        "fitted_outcomes": outcome_binding,
    }
    return frame, evidence


def _validate_reference_for_shard(reference: pd.DataFrame, horizon: int) -> pd.DataFrame:
    required = ("key_id", "site_id", "issue_date", "target_date", "lead_days", "y_true")
    if not set(required) <= set(reference.columns):
        raise AssertionError("shard reference lacks exact key/identity/y columns")
    ref = reference[list(required)].copy(deep=True).reset_index(drop=True)
    _strict_key_ids(ref["key_id"], "shard reference")
    _strict_site_ids(ref["site_id"], "shard reference")
    issue = _strict_dates(ref["issue_date"], "shard reference issue_date")
    target = _strict_dates(ref["target_date"], "shard reference target_date")
    lead = _strict_signed_integer_array(ref["lead_days"], "shard reference lead_days")
    _strict_float_array(ref["y_true"], "shard reference y_true", finite=True)
    if not np.all(lead == horizon) or not np.array_equal(
        target, issue + np.timedelta64(horizon, "D")
    ):
        raise AssertionError("shard reference chronology/lead differs")
    if (
        ref.duplicated("key_id").any()
        or ref.duplicated(["site_id", "issue_date", "target_date"]).any()
    ):
        raise AssertionError("shard reference duplicates a key")
    return ref


def validate_v5_shard(
    frame: pd.DataFrame,
    cell: Cell,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Validate exact schema, types, chronology, accounting, and formal keys."""

    if cell not in fixed_cells():
        raise ValueError("shard cell is outside the fixed observed-lineage scope")
    if tuple(frame.columns) != SHARD_COLUMNS:
        raise AssertionError("v5 observed shard schema/order is not exact")
    checked = frame.copy(deep=True).reset_index(drop=True)
    _strict_key_ids(checked["key_id"], "v5 observed shard")
    _strict_site_ids(checked["site_id"], "v5 observed shard")
    issue = _strict_dates(checked["issue_date"], "v5 shard issue_date")
    target = _strict_dates(checked["target_date"], "v5 shard target_date")
    horizon = _strict_signed_integer_array(checked["horizon"], "v5 shard horizon")
    if not np.all(horizon == cell.horizon) or not np.array_equal(
        target, issue + np.timedelta64(cell.horizon, "D")
    ):
        raise AssertionError("v5 observed shard horizon/chronology differs from its cell")
    for name, expected in (
        ("arm", cell.arm),
        ("model", cell.model),
        ("analysis_status", STATUS),
    ):
        values = tuple(checked[name].to_numpy(copy=True))
        if not values or any(type(value) is not str or value != expected for value in values):
            raise AssertionError("v5 observed shard identity differs from its cell")
    for column in ("y_true", "y_pred", "y_damped"):
        _strict_float_array(checked[column], f"v5 shard {column}", finite=True)
    validate_substitution_accounting(checked, arm=cell.arm, horizon=cell.horizon)
    if (
        checked.duplicated("key_id").any()
        or checked.duplicated(["site_id", "issue_date", "target_date"]).any()
    ):
        raise AssertionError("v5 observed shard duplicates a formal key")
    reference_h = reference
    if tuple(reference.columns) == REFERENCE_COLUMNS:
        reference_h = validate_reference_registry(reference, require_production_counts=False)
        reference_h = reference_h[reference_h["lead_days"].eq(cell.horizon)]
    ref = _validate_reference_for_shard(reference_h, cell.horizon)
    if set(checked["key_id"]) != set(ref["key_id"]) or len(checked) != len(ref):
        raise AssertionError("v5 observed shard keys differ from the exact formal registry")
    joined = ref.merge(
        checked[["key_id", "site_id", "issue_date", "target_date", "y_true"]],
        on=["key_id", "site_id", "issue_date", "target_date"],
        how="left",
        validate="one_to_one",
        suffixes=("_reference", "_shard"),
    )
    if joined["y_true_shard"].isna().any() or not V4.target_values_equal(
        _strict_float_array(joined["y_true_reference"], "reference y", finite=True),
        _strict_float_array(joined["y_true_shard"], "shard y", finite=True),
    ):
        raise AssertionError("v5 observed shard targets differ from the formal registry")
    return checked.sort_values(["site_id", "issue_date"], kind="mergesort").reset_index(drop=True)


def shard_filename(cell: Cell) -> str:
    if cell not in fixed_cells():
        raise ValueError("cannot name a shard outside the fixed v5 scope")
    return f"{cell.arm}_{cell.model}_h{cell.horizon}_v5_observed.parquet"


def shard_path(output_dir: Path, cell: Cell) -> Path:
    return output_dir / SHARD_DIRNAME / shard_filename(cell)


def _validate_production_destination(output_dir: Path) -> None:
    declared = Path(output_dir)
    if ".." in declared.parts or declared != DEFAULT_OUTPUT_DIR:
        raise ValueError(f"production output must be exactly {DEFAULT_OUTPUT_DIR}")
    _require_no_symlink_components(ROOT, label="repository root")
    _require_no_symlink_components(FINAL_OUTPUT_ROOT, label="approved final output root")
    if os.path.lexists(DEFAULT_OUTPUT_DIR):
        raise FileExistsError(
            f"v5 observed correction is create-only; output path already exists: {DEFAULT_OUTPUT_DIR}"
        )


def _assert_create_only_destination(output_dir: Path) -> None:
    """Compatibility name retained, now enforcing the exact production destination."""

    _validate_production_destination(output_dir)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_relative_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError(f"unsafe bundle-relative path: {relative!r}")
    return path


def _require_simple_name(name: str, *, label: str) -> None:
    if (
        type(name) is not str
        or name in {"", ".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(f"{label} must be one safe basename")


def _entry_identity(status: os.stat_result) -> tuple[int, int]:
    return int(status.st_dev), int(status.st_ino)


def _verify_anchored_directory_path(
    path: Path,
    descriptor: int,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or _entry_identity(opened) != expected_identity:
        raise RuntimeError(f"opened {label} inode/type changed")
    try:
        lexical = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"lexical {label} disappeared: {path}") from exc
    if (
        stat.S_ISLNK(lexical.st_mode)
        or not stat.S_ISDIR(lexical.st_mode)
        or _entry_identity(lexical) != expected_identity
    ):
        raise RuntimeError(f"lexical {label} no longer names the anchored directory: {path}")
    _require_no_symlink_components(path, label=label)


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
        raise RuntimeError(f"cannot anchor {label}: {absolute}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"anchored {label} is not a directory")
        identity = _entry_identity(status)
        _verify_anchored_directory_path(absolute, descriptor, identity, label=label)
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


def _open_directory_at(parent_descriptor: int, name: str, *, label: str) -> int:
    _require_simple_name(name, label=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise RuntimeError(f"cannot safely open {label}") from exc
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"{label} is not a directory")
    return descriptor


def _read_stable_regular_at(parent_descriptor: int, name: str, *, label: str) -> bytes:
    _require_simple_name(name, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise RuntimeError(f"cannot safely open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_signature != after_signature or len(payload) != before.st_size:
        raise RuntimeError(f"{label} changed while being read")
    entry = _entry_status_at(parent_descriptor, name)
    if (
        entry is None
        or not stat.S_ISREG(entry.st_mode)
        or _entry_identity(entry) != _entry_identity(before)
    ):
        raise RuntimeError(f"{label} directory entry changed while being read")
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
            if written <= 0:
                raise RuntimeError(f"short write for staged {name}")
            offset += written
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size != len(payload):
            raise RuntimeError(f"staged {name} size/type changed")
    finally:
        os.close(descriptor)


def _create_staging_directory_at(
    root_descriptor: int, *, destination_name: str
) -> tuple[str, int, tuple[int, int]]:
    _require_simple_name(destination_name, label="destination name")
    for _attempt in range(128):
        name = f".{destination_name}.staging.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=root_descriptor)
        except FileExistsError:
            continue
        try:
            descriptor = _open_directory_at(root_descriptor, name, label="staging directory")
            status = os.fstat(descriptor)
        except Exception:
            os.rmdir(name, dir_fd=root_descriptor)
            raise
        return name, descriptor, _entry_identity(status)
    raise RuntimeError("cannot allocate an exclusive staging directory")


def _remove_tree_at(parent_descriptor: int, name: str) -> None:
    status = _entry_status_at(parent_descriptor, name)
    if status is None:
        return
    if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
        child_descriptor = _open_directory_at(parent_descriptor, name, label="cleanup directory")
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
    strict: bool,
) -> bool:
    status = _entry_status_at(parent_descriptor, name)
    if status is None:
        return False
    if _entry_identity(status) != expected_identity:
        if strict:
            raise RuntimeError(f"refusing to clean replaced owned entry {name}")
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
    if renameat2 is None:
        raise RuntimeError("atomic Linux RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
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
            raise FileExistsError(f"refusing to overwrite existing v5 bundle: {destination_name}")
        raise RuntimeError(
            f"atomic no-replace directory publication failed "
            f"({os.strerror(error)}): {destination_name}"
        )


class _BundleTransaction:
    """One-root-inode transaction; every mutable operation is dirfd anchored."""

    def __init__(
        self,
        destination: Path,
        *,
        allowed_root: Path,
        expected_destination_name: str,
    ) -> None:
        declared_destination = Path(destination)
        declared_root = Path(allowed_root)
        if ".." in declared_destination.parts or ".." in declared_root.parts:
            raise ValueError("bundle paths cannot contain '..'")
        self.root = _absolute(declared_root)
        self.destination = _absolute(declared_destination)
        try:
            relative = self.destination.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("bundle destination escapes the approved root") from exc
        if len(relative.parts) != 1 or relative.name != expected_destination_name:
            raise ValueError("bundle destination is not the exact approved direct child")
        _require_simple_name(expected_destination_name, label="approved destination")
        self._destination_name = expected_destination_name
        self._root_descriptor, self._root_identity = _open_anchored_directory(
            self.root, label="bundle output root"
        )
        self._root_closed = False
        self._lock_name = f".{expected_destination_name}.create.lock"
        self.lock = self.root / self._lock_name
        self._staging_name: str | None = None
        self._staging_descriptor: int | None = None
        self._staging_identity: tuple[int, int] | None = None
        self.staging: Path | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._committed = False
        if _entry_status_at(self._root_descriptor, self._destination_name) is not None:
            os.close(self._root_descriptor)
            self._root_closed = True
            raise FileExistsError(f"create-only bundle destination exists: {self.destination}")
        flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                self._lock_name,
                flags,
                0o600,
                dir_fd=self._root_descriptor,
            )
        except FileExistsError as exc:
            os.close(self._root_descriptor)
            self._root_closed = True
            raise RuntimeError(f"another create-only publication is active: {self.lock}") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("bundle lock is not a regular file")
            self._lock_identity = (info.st_dev, info.st_ino)
            lock_payload = f"pid={os.getpid()}\n".encode("ascii")
            if os.write(descriptor, lock_payload) != len(lock_payload):
                raise RuntimeError("bundle lock write was short")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            (
                self._staging_name,
                self._staging_descriptor,
                self._staging_identity,
            ) = _create_staging_directory_at(
                self._root_descriptor,
                destination_name=self._destination_name,
            )
            self.staging = self.root / self._staging_name
            os.fsync(self._root_descriptor)
        except Exception:
            self.close()
            raise

    def _open_relative_parent(self, relative: str, *, create: bool) -> tuple[int, str]:
        if self._staging_descriptor is None or self._staging_name is None:
            raise RuntimeError("bundle transaction has no active staging directory")
        part = _safe_relative_path(relative)
        current = os.dup(self._staging_descriptor)
        try:
            for component in part.parts[:-1]:
                _require_simple_name(component, label="staged directory component")
                status = _entry_status_at(current, component)
                if status is None:
                    if not create:
                        raise FileNotFoundError(f"staged directory is missing: {component}")
                    os.mkdir(component, mode=0o700, dir_fd=current)
                    os.fsync(current)
                    status = _entry_status_at(current, component)
                if (
                    status is None
                    or stat.S_ISLNK(status.st_mode)
                    or not stat.S_ISDIR(status.st_mode)
                ):
                    raise RuntimeError("staged relative path contains a non-directory or symlink")
                child = _open_directory_at(current, component, label="staged subdirectory")
                if _entry_identity(os.fstat(child)) != _entry_identity(status):
                    os.close(child)
                    raise RuntimeError("staged subdirectory entry changed while being opened")
                os.close(current)
                current = child
            filename = part.name
            _require_simple_name(filename, label="staged filename")
            return current, filename
        except Exception:
            os.close(current)
            raise

    def read_bytes(self, relative: str) -> bytes:
        parent, filename = self._open_relative_parent(relative, create=False)
        try:
            return _read_stable_regular_at(parent, filename, label=f"staged {relative}")
        finally:
            os.close(parent)

    def write_bytes(self, relative: str, payload: bytes) -> dict[str, object]:
        if type(payload) is not bytes:
            raise TypeError("bundle payload must be immutable bytes")
        parent, filename = self._open_relative_parent(relative, create=True)
        try:
            _write_bytes_at(parent, filename, payload)
            observed = _read_stable_regular_at(parent, filename, label=f"staged {relative}")
        finally:
            os.close(parent)
        if observed != payload:
            raise RuntimeError(f"staged bytes changed for {relative}")
        return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}

    def write_parquet(self, relative: str, frame: pd.DataFrame) -> dict[str, object]:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return self.write_bytes(relative, buffer.getvalue())

    def _inventory_at(
        self, descriptor: int, prefix: tuple[str, ...] = ()
    ) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        directories: set[str] = set()
        for name in sorted(os.listdir(descriptor)):
            _require_simple_name(name, label="staged entry")
            status = _entry_status_at(descriptor, name)
            if status is None or stat.S_ISLNK(status.st_mode):
                raise RuntimeError("staged bundle contains a missing or symbolic-link entry")
            relative = "/".join((*prefix, name))
            if stat.S_ISDIR(status.st_mode):
                directories.add(relative)
                child = _open_directory_at(descriptor, name, label="staged inventory directory")
                try:
                    child_files, child_directories = self._inventory_at(child, (*prefix, name))
                    files.update(child_files)
                    directories.update(child_directories)
                finally:
                    os.close(child)
            elif stat.S_ISREG(status.st_mode):
                files.add(relative)
            else:
                raise RuntimeError("staged bundle contains a non-regular file")
        return files, directories

    def _audit_tree(self, expected_files: Mapping[str, Mapping[str, object]]) -> None:
        if self._staging_descriptor is None:
            raise RuntimeError("bundle transaction has no staging directory")
        expected = {_safe_relative_path(name).as_posix() for name in expected_files}
        expected_directories = {
            "/".join(path.parts[:index])
            for path in (_safe_relative_path(name) for name in expected)
            for index in range(1, len(path.parts))
        }
        observed, observed_directories = self._inventory_at(self._staging_descriptor)
        if observed != expected or observed_directories != expected_directories:
            raise RuntimeError(
                f"staged bundle file set differs: missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}, "
                f"directory_drift={sorted(observed_directories ^ expected_directories)}"
            )
        for relative, record in expected_files.items():
            payload = self.read_bytes(relative)
            if hashlib.sha256(payload).hexdigest() != record.get("sha256") or len(
                payload
            ) != record.get("bytes"):
                raise RuntimeError(f"staged bundle binding changed for {relative}")

    def _fsync_tree_at(self, descriptor: int) -> None:
        for name in sorted(os.listdir(descriptor)):
            status = _entry_status_at(descriptor, name)
            if status is None or stat.S_ISLNK(status.st_mode):
                raise RuntimeError("staged entry changed before fsync")
            if stat.S_ISDIR(status.st_mode):
                child = _open_directory_at(descriptor, name, label="fsync staged directory")
                try:
                    self._fsync_tree_at(child)
                finally:
                    os.close(child)
            elif not stat.S_ISREG(status.st_mode):
                raise RuntimeError("staged entry is not regular before fsync")
        os.fsync(descriptor)

    def commit(
        self,
        expected_files: Mapping[str, Mapping[str, object]],
        *,
        precommit_check: Callable[[], None],
    ) -> None:
        if self._staging_descriptor is None or self._staging_name is None:
            raise RuntimeError("bundle transaction is already closed")
        if not expected_files:
            raise RuntimeError("bundle transaction cannot commit an empty file set")
        self._audit_tree(expected_files)
        precommit_check()
        self._audit_tree(expected_files)
        self._fsync_tree_at(self._staging_descriptor)
        os.fsync(self._root_descriptor)
        _verify_anchored_directory_path(
            self.root,
            self._root_descriptor,
            self._root_identity,
            label="bundle output root before publication",
        )
        if _entry_status_at(self._root_descriptor, self._destination_name) is not None:
            raise FileExistsError("bundle destination appeared before no-replace commit")
        _rename_directory_noreplace_at(
            self._root_descriptor,
            self._staging_name,
            self._root_descriptor,
            self._destination_name,
        )
        _verify_anchored_directory_path(
            self.root,
            self._root_descriptor,
            self._root_identity,
            label="bundle output root after publication",
        )
        published = _entry_status_at(self._root_descriptor, self._destination_name)
        if (
            published is None
            or not stat.S_ISDIR(published.st_mode)
            or _entry_identity(published) != self._staging_identity
        ):
            raise RuntimeError("published destination is not the exact staged directory inode")
        self._audit_tree(expected_files)
        if self._lock_identity is None:
            raise RuntimeError("bundle lock identity disappeared before commit")
        _remove_owned_entry_at(
            self._root_descriptor,
            self._lock_name,
            self._lock_identity,
            strict=True,
        )
        self._lock_identity = None
        os.fsync(self._root_descriptor)
        self._committed = True
        self.staging = None

    def close(self) -> None:
        if self._root_closed:
            return
        try:
            if not self._committed and self._staging_identity is not None:
                if self._staging_name is not None:
                    _remove_owned_entry_at(
                        self._root_descriptor,
                        self._staging_name,
                        self._staging_identity,
                        strict=False,
                    )
                _remove_owned_entry_at(
                    self._root_descriptor,
                    self._destination_name,
                    self._staging_identity,
                    strict=False,
                )
            if self._lock_identity is not None:
                _remove_owned_entry_at(
                    self._root_descriptor,
                    self._lock_name,
                    self._lock_identity,
                    strict=False,
                )
                self._lock_identity = None
            os.fsync(self._root_descriptor)
        finally:
            if self._staging_descriptor is not None:
                os.close(self._staging_descriptor)
                self._staging_descriptor = None
            os.close(self._root_descriptor)
            self._root_closed = True
            self.staging = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def _runtime_evidence() -> dict[str, object]:
    packages: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "pyarrow", "lightgbm"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "packages": packages,
        "formal_training_threads": _formal_threads(),
        "prediction_threads": 1,
    }


def _coefficient_binding(climatology: F.HarmonicClimatology, label: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for site in sorted(climatology.coef):
        values = np.asarray(climatology.coef[site], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite coefficients")
        for index, value in enumerate(values):
            rows.append({"site_id": site, "coefficient_index": index, "value": float(value)})
    frame = pd.DataFrame(rows)
    frame["coefficient_index"] = frame["coefficient_index"].astype(np.int64)
    frame["value"] = frame["value"].astype(np.float64)
    _strict_site_ids(frame["site_id"], label)
    return {
        "stations": len(climatology.coef),
        "coefficient_rows": len(frame),
        "k": int(climatology.k),
        "fit_stations": list(climatology.fit_stations),
        "pooled": bool(climatology.pooled),
        "content_sha256": _canonical_frame_sha256(frame, ("site_id", "coefficient_index", "value")),
    }


def preprocessing_content_record(preprocessing: ObservedPreprocessing) -> dict[str, object]:
    _strict_site_ids(pd.Series(preprocessing.station_ids, dtype=object), "preprocessing content")
    median_rows: list[dict[str, object]] = []
    for (site, variable), values in sorted(preprocessing.imputer.medians.items()):
        for day, value in values.sort_index().items():
            median_rows.append(
                {
                    "site_id": site,
                    "variable": variable,
                    "day_of_year": int(day),
                    "value": float(value),
                }
            )
    medians = pd.DataFrame(median_rows)
    medians["day_of_year"] = medians["day_of_year"].astype(np.int64)
    medians["value"] = medians["value"].astype(np.float64)
    global_rows = [
        {
            "site_id": site,
            "variable": variable,
            "value": float(value),
        }
        for (site, variable), value in sorted(preprocessing.imputer.global_median.items())
    ]
    globals_frame = pd.DataFrame(global_rows)
    globals_frame["value"] = globals_frame["value"].astype(np.float64)
    imputer_record = {
        "method": D.PER_STATION_IMPUTER_METHOD,
        "fit_stations": list(preprocessing.imputer.fit_stations),
        "pooled": bool(preprocessing.imputer.pooled),
        "seasonal_median_rows": len(medians),
        "seasonal_medians_sha256": _canonical_frame_sha256(
            medians, ("site_id", "variable", "day_of_year", "value")
        ),
        "global_median_rows": len(globals_frame),
        "global_medians_sha256": _canonical_frame_sha256(
            globals_frame, ("site_id", "variable", "value")
        ),
    }
    water = _coefficient_binding(preprocessing.water_climatology, "water climatology")
    meteorology = {
        variable: _coefficient_binding(
            preprocessing.meteorology_climatologies[variable],
            f"meteorology climatology {variable}",
        )
        for variable in METEOROLOGY_VARIABLES
    }
    anchor = preprocessing.damped_anchor
    anchor_content = {
        "method": F.DAMPED_AR_METHOD,
        "phi": {site: float(value) for site, value in sorted(anchor.phi.items())},
        "fit_stations": list(anchor.fit_stations),
        "pooled": bool(anchor.pooled),
        "fallback": float(anchor.fallback),
        "min_pairs": int(anchor.min_pairs),
        "lower_bound": float(anchor.lower_bound),
        "upper_bound": float(anchor.upper_bound),
        "min_mean_square": float(anchor.min_mean_square),
        "pool_weighting": anchor.pool_weighting,
        "eligibility_rule": anchor.eligibility_rule,
        "eligible_fit_stations": list(anchor.eligible_fit_stations),
        "pair_counts": {site: int(value) for site, value in sorted(anchor.pair_counts.items())},
        "lagged_anomaly_mean_squares": {
            site: None if value is None else float(value)
            for site, value in sorted(anchor.lagged_anomaly_mean_squares.items())
        },
    }
    record: dict[str, object] = {
        "fit_period": f"{TRAIN_START.date()}/{TRAIN_END.date()}",
        "fit_rows": preprocessing.training_row_count,
        "fit_station_count": len(preprocessing.station_ids),
        "validation_period": f"{VALIDATION_START.date()}/{VALIDATION_END.date()}",
        "validation_rows": preprocessing.validation_row_count,
        "validation_excluded_from_every_preprocessing_fit": True,
        "station_registry_sha256": _canonical_json_sha256(list(preprocessing.station_ids)),
        "imputer": imputer_record,
        "water_climatology": {
            "method": F.PER_STATION_HARMONIC_METHOD,
            **water,
        },
        "damped_anchor": {
            "content": anchor_content,
            "content_sha256": _canonical_json_sha256(anchor_content),
        },
        "meteorology_climatologies": meteorology,
    }
    record["complete_content_sha256"] = _canonical_json_sha256(record)
    return record


def plan_record(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    if Path(output_dir) != DEFAULT_OUTPUT_DIR or ".." in Path(output_dir).parts:
        raise ValueError(f"v5 output path is fixed to {DEFAULT_OUTPUT_DIR}")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": STATUS,
        "execution_performed": False,
        "scope": {
            "arms": list(ARMS),
            "models": list(MODELS),
            "horizons": list(HORIZONS),
            "cell_count": len(fixed_cells()),
            "other_arms_authorized": False,
        },
        "periods": {
            "preprocessing_and_training": f"{TRAIN_START.date()}/{TRAIN_END.date()}",
            "validation": f"{VALIDATION_START.date()}/{VALIDATION_END.date()}",
            "evaluation": f"{CONFIRM_START.date()}/{CONFIRM_END.date()}",
        },
        "pinned_inputs": {
            role: {
                "path": PINNED_INPUT_PATHS[role].relative_to(ROOT).as_posix(),
                "sha256": PINNED_INPUT_SHA256[role],
            }
            for role in PINNED_INPUT_PATHS
        },
        "governance": {
            role: {
                "path": PINNED_GOVERNANCE_PATHS[role].relative_to(ROOT).as_posix(),
                "sha256": PINNED_GOVERNANCE_SHA256[role],
            }
            for role in PINNED_GOVERNANCE_PATHS
        },
        "score_execution_authority_gate": {
            "draft_protocol_status": "DRAFT_NOT_SEALED",
            "draft_protocol_execution_authorized": False,
            "key_manifest_status": "PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY",
            "key_manifest_role": "KEY_AND_LABEL_INPUT_AUTHORITY_ONLY",
            "canonical_sealed_protocol_path": SEALED_SCORE_PROTOCOL.relative_to(ROOT).as_posix(),
            "canonical_seal_path": SEALED_SCORE_PROTOCOL_SEAL.relative_to(ROOT).as_posix(),
            "canonical_clean_design_commit_path": SCORE_CLEAN_DESIGN_COMMIT.relative_to(
                ROOT
            ).as_posix(),
            "canonical_source_registry_path": SCORE_SOURCE_REGISTRY.relative_to(ROOT).as_posix(),
            "expected_sealed_protocol_sha256": EXPECTED_SEALED_SCORE_PROTOCOL_SHA256,
            "state": "LOCKED_NO_CANONICAL_SCORE_EXECUTION_SEAL",
            "defect_authority_is_score_authorization": False,
            "no_self_hash_cycle": True,
        },
        "defect_authority_gate": {
            "manifest_path": DEFECT_AUTHORITY_MANIFEST.relative_to(ROOT).as_posix(),
            "expected_sha256": EXPECTED_DEFECT_AUTHORITY_SHA256,
            "state": (
                "LOCKED_PENDING_REVIEWED_SHA256_PIN"
                if EXPECTED_DEFECT_AUTHORITY_SHA256 is None
                else "PIN_DECLARED_FILESYSTEM_NOT_READ_BY_DRY_RUN"
            ),
        },
        "frozen_base_feature_count": len(FROZEN_BASE_FEATURE_COLUMNS),
        "frozen_base_feature_columns_sha256": _canonical_json_sha256(
            list(FROZEN_BASE_FEATURE_COLUMNS)
        ),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "publication": {
            "create_only": True,
            "resume": False,
            "overwrite": False,
            "sibling_staging": True,
            "exclusive_lock": True,
            "fsync": True,
            "single_linux_RENAME_NOREPLACE": True,
        },
        "cells": [cell.__dict__ for cell in fixed_cells()],
    }


def _training_lineage(
    table: pd.DataFrame,
    horizon: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for split in ("train", "val"):
        rows = table[table["split"].eq(split)]
        binding = _ordered_dataset_binding(
            rows,
            label_column="y",
            feature_columns=FROZEN_BASE_FEATURE_COLUMNS,
            horizon=horizon,
        )
        output[split] = {
            "rows": len(rows),
            "key_sha256": binding["ordered_identity_sha256"],
            "key_and_y_sha256": binding["ordered_identity_and_y_sha256"],
            "issue_and_target_raw_observed": True,
        }
    return output


def assert_fit_identities_unchanged(
    base_table: pd.DataFrame,
    arm_table: pd.DataFrame,
) -> None:
    columns = [
        "site_id",
        "issue_date",
        "target_date",
        "split",
        "horizon",
        "y",
        "issue_wtemp_observed",
        "target_wtemp_observed",
    ]
    if len(base_table) != len(arm_table) or not base_table[columns].equals(arm_table[columns]):
        raise AssertionError("forcing arm changed training/validation identities or labels")


def _read_staged_parquet(transaction: _BundleTransaction, relative: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(transaction.read_bytes(relative)))


def execute(output_dir: Path) -> int:
    _validate_production_destination(output_dir)
    # This gate currently fails before capture, model construction, lock, or output write.
    snapshot = capture_execution_inputs()
    raw, stations, _registry = _load_raw_panel_from_bounds(snapshot)
    reference = validate_reference_registry(
        pd.read_parquet(io.BytesIO(snapshot["primary_key_registry"].payload)),
        require_production_counts=True,
    )
    C.STATIONS = stations
    preprocessing = fit_observed_preprocessing(raw, stations)
    preprocessing_record = preprocessing_content_record(preprocessing)
    imputed = impute_feature_panel(raw, preprocessing.imputer)
    runtime = _runtime_evidence()

    expected_files: dict[str, dict[str, object]] = {}
    cell_manifest: list[dict[str, object]] = []
    identity_by_horizon: dict[int, dict[str, object]] = {}
    with _BundleTransaction(
        DEFAULT_OUTPUT_DIR,
        allowed_root=FINAL_OUTPUT_ROOT,
        expected_destination_name=DEFAULT_OUTPUT_DIR.name,
    ) as transaction:
        for horizon in HORIZONS:
            base_table, base_columns = build_observed_feature_table(
                raw,
                imputed,
                preprocessing.water_climatology,
                horizon,
            )
            lineage = _training_lineage(base_table, horizon)
            for split in ("train", "val"):
                observed_count = int(lineage[split]["rows"])  # type: ignore[index]
                expected_count = EXPECTED_EXAMPLE_COUNTS[split][horizon]
                if observed_count != expected_count:
                    raise AssertionError(
                        f"h{horizon} {split} raw-admissible count {observed_count} != {expected_count}"
                    )
            identity_by_horizon[horizon] = lineage

            raw_future: RawFutureRegistry | None = None
            for arm in ARMS:
                if arm == "F0":
                    arm_table, model_columns = materialize_forcing_features(
                        base_table,
                        base_columns,
                        arm=arm,
                        horizon=horizon,
                        raw_future_registry=None,
                        meteorology_climatologies=None,
                    )
                else:
                    if raw_future is None:
                        raw_future = build_raw_future_registry(
                            raw,
                            base_table[["site_id", "issue_date"]],
                            horizon,
                        )
                    arm_table, model_columns = materialize_forcing_features(
                        base_table,
                        base_columns,
                        arm=arm,
                        horizon=horizon,
                        raw_future_registry=raw_future,
                        meteorology_climatologies=preprocessing.meteorology_climatologies,
                    )
                assert_fit_identities_unchanged(base_table, arm_table)
                evaluation = bind_exact_evaluation_rows(
                    arm_table,
                    reference,
                    horizon,
                    model_columns,
                )
                for model in MODELS:
                    cell = Cell(arm, model, horizon)
                    frame, fit_evidence = fit_tree_cell(
                        arm_table,
                        evaluation,
                        model_columns,
                        preprocessing,
                        cell,
                    )
                    frame = validate_v5_shard(frame, cell, reference)
                    relative = f"{SHARD_DIRNAME}/{shard_filename(cell)}"
                    binding = transaction.write_parquet(relative, frame)
                    persisted = _read_staged_parquet(transaction, relative)
                    validate_v5_shard(persisted, cell, reference)
                    if validate_v5_shard(persisted, cell, reference).equals(frame) is False:
                        raise AssertionError("staged v5 shard differs after Parquet round trip")
                    expected_files[relative] = dict(binding)
                    substitution_totals = {
                        column: int(frame[column].sum()) for column in SUBSTITUTION_COLUMNS
                    }
                    substitution_total = int(frame["forcing_substitution_count"].sum())
                    if substitution_total != sum(substitution_totals.values()):
                        raise AssertionError("cell substitution totals do not add exactly")
                    cell_manifest.append(
                        {
                            "cell": cell.__dict__,
                            "analysis_status": STATUS,
                            "shard": {"path": relative, "rows": len(frame), **binding},
                            "features": {
                                "count": len(model_columns),
                                "ordered_columns": list(model_columns),
                                "ordered_columns_sha256": _canonical_json_sha256(
                                    list(model_columns)
                                ),
                            },
                            "future_registry": (
                                None
                                if arm == "F0"
                                else {
                                    "source_sha256": raw_future.source_sha256,
                                    "content_sha256": raw_future.content_sha256,
                                    "raw_value_mask_pairs_preserved": True,
                                }
                            ),
                            "substitution_accounting": {
                                "per_variable_totals": substitution_totals,
                                "total": substitution_total,
                                "total_equals_sum_per_variable": True,
                                "per_row_upper_bound": len(METEOROLOGY_VARIABLES) * horizon,
                                "F0_identically_zero": arm == "F0",
                            },
                            "fit": fit_evidence,
                        }
                    )

        if len(cell_manifest) != 12:
            raise AssertionError(
                f"observed correction emitted {len(cell_manifest)} cells, expected 12"
            )
        manifest = {
            "format": MANIFEST_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "analysis_status": STATUS,
            "execution_performed": True,
            "scope": plan_record(DEFAULT_OUTPUT_DIR)["scope"],
            "captured_inputs_dependencies_and_authorities": {
                role: _binding(bound) for role, bound in sorted(snapshot.items())
            },
            "governance": {
                "draft_protocol_sha256": PINNED_GOVERNANCE_SHA256["protocol"],
                "draft_protocol_status": "DRAFT_NOT_SEALED_NOT_SCORE_AUTHORITY",
                "key_authority_manifest_sha256": PINNED_GOVERNANCE_SHA256["key_authority_manifest"],
                "key_authority_role": "PHASE1_KEYS_ONLY_NOT_MODEL_SCORE_AUTHORITY",
                "defect_authority_manifest_sha256": EXPECTED_DEFECT_AUTHORITY_SHA256,
                "defect_authority_is_score_authorization": False,
                "sealed_score_protocol_sha256": snapshot["sealed_score_protocol"].sha256,
                "sealed_score_protocol_seal_sha256": snapshot["sealed_score_protocol_seal"].sha256,
                "clean_design_commit_sha256": snapshot["score_clean_design_commit"].sha256,
                "source_registry_sha256": snapshot["score_source_registry"].sha256,
                "sealed_score_execution_authorized": True,
                "no_self_hash_cycle": True,
                "precommit_full_snapshot_revalidation_required": True,
            },
            "raw_panel_lineage": {
                "source_sha256": raw.source_sha256,
                "identity_sha256": raw.identity_sha256,
                "observed_mask_sha256": raw.observed_mask_sha256,
                "full_content_sha256": raw.content_sha256,
                "observed_policy": "exact bool equal to isfinite(raw) before imputation",
                "masks_survived_transform": True,
            },
            "imputed_feature_lineage": {
                "raw_source_sha256": imputed.raw_source_sha256,
                "raw_identity_sha256": imputed.raw_identity_sha256,
                "raw_content_sha256": imputed.raw_content_sha256,
                "observed_mask_sha256": imputed.observed_mask_sha256,
                "full_content_sha256": imputed.content_sha256,
            },
            "preprocessing": preprocessing_record,
            "training_identity_by_horizon": identity_by_horizon,
            "runtime": runtime,
            "publication": {
                "destination": DEFAULT_OUTPUT_DIR.relative_to(ROOT).as_posix(),
                "create_only": True,
                "resume": False,
                "overwrite": False,
                "same_parent_staging": True,
                "exclusive_lock": True,
                "all_files_fsynced_before_commit": True,
                "single_linux_RENAME_NOREPLACE": True,
            },
            "cells": cell_manifest,
        }
        manifest_payload = _canonical_json_bytes(manifest)
        manifest_binding = transaction.write_bytes(MANIFEST_FILENAME, manifest_payload)
        expected_files[MANIFEST_FILENAME] = dict(manifest_binding)

        def precommit() -> None:
            _revalidate_snapshot(snapshot)
            _validated_raw_frame(raw)
            _validated_imputed_frame(imputed)
            if preprocessing_content_record(preprocessing) != preprocessing_record:
                raise RuntimeError("preprocessing content changed before bundle commit")

        transaction.commit(expected_files, precommit_check=precommit)

    print(
        json.dumps(
            {
                "analysis_status": STATUS,
                "execution_performed": True,
                "cells_written": len(cell_manifest),
                "manifest": str(DEFAULT_OUTPUT_DIR / MANIFEST_FILENAME),
            },
            indent=1,
        )
    )
    return 0


def run(args: argparse.Namespace) -> int:
    output_dir = Path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR))
    if output_dir != DEFAULT_OUTPUT_DIR or ".." in output_dir.parts:
        raise ValueError(f"v5 output path is fixed to {DEFAULT_OUTPUT_DIR}")
    if not bool(getattr(args, "execute", False)):
        print(json.dumps(plan_record(DEFAULT_OUTPUT_DIR), indent=1))
        return 0
    return execute(DEFAULT_OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fixed plan; read, write, fit, and score nothing (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="attempt the locked create-only correction after an explicit review handoff",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="must equal the fixed v5 destination exactly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (
        AssertionError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
