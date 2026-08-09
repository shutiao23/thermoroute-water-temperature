#!/usr/bin/env python3
"""Fail-closed Phase-1 scaffold for the v4 L2_U2 sensitivity.

This module is deliberately not an experiment runner yet.  Its dry run reads
only protocol and source-code text.  It does not open station data, the
2021--2023 panel, the outcome registry, an existing prediction, or a
checkpoint; it does not train, score, or write an artifact.

The scaffold freezes the parts that must be settled before outcome access:

* two logical cells (LightGBM and plain_TCN) at F0/L2_U2/whole_region/h7;
* four canonical, intact-HUC2 folds, eight architecture-by-fold units, and
  forty matched model-fit units after the five v4 fit seeds are applied;
* a shared local-observation-free input boundary and a separate label boundary;
* fold-aware, 2006--2015, training-stations-only preprocessing lineage; and
* common-key, provenance, artifact-identity, resume, and completeness gates.

Any invocation without ``--dry-run`` performs the governance check first and
then fails before station metadata, panel, outcome, model, or output access.
Even a future valid seal cannot make this Phase-1 file train: the data loader,
model trainers, prediction-file parser, scorer, and artifact writer are
explicitly unimplemented and must be added and newly sealed in a later phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4.yaml"
SEALED_PROTOCOL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4_sealed.yaml"
SEALED_PROTOCOL_SEAL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4_seal.json"
LEGACY_PROTOCOL = ROOT / "protocols" / "wrr_strong_accept_protocol_v2.yaml"
LEGACY_RUNNER = ROOT / "scripts" / "final" / "run_information_ladder.py"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"

FUTURE_AUTHORITY_ROOT = ROOT / "outputs" / "final" / "l2_u2_information_regime_v4_authority"
SHARED_KEY_REGISTRY = (
    ROOT
    / "outputs"
    / "final"
    / "information_regime_key_registries_v4"
    / "primary_reportable_key_registry_v4.parquet"
)
AUTHORITY_FILE_PATHS = MappingProxyType(
    {
        "station_registry": STATION_REGISTRY,
        "fold_registry": FUTURE_AUTHORITY_ROOT / "fold_registry_v4.json",
        "training_meteorology_registry": FUTURE_AUTHORITY_ROOT
        / "training_meteorology_registry_2006_2015_v4.parquet",
        "training_anchor_registry": FUTURE_AUTHORITY_ROOT
        / "training_anchor_registry_2006_2015_v4.parquet",
        "predictor_registry": FUTURE_AUTHORITY_ROOT / "f0_predictor_registry_v4.parquet",
        "common_key_registry": SHARED_KEY_REGISTRY,
        "label_registry": SHARED_KEY_REGISTRY,
        "contrast_registry": FUTURE_AUTHORITY_ROOT / "contrast_registry_v4.json",
        "model_configuration": FUTURE_AUTHORITY_ROOT / "model_configuration_v4.json",
        "environment": FUTURE_AUTHORITY_ROOT / "environment_v4.json",
    }
)

EXPECTED_PROTOCOL_ID = "thermoroute_wrr_information_regimes_v4"
EXPECTED_PROTOCOL_VERSION = 4
AUTHORIZED_PROTOCOL_STATUS = "SEALED"
SEAL_FORMAT = "thermoroute.wrr-information-regimes-protocol-seal.v4"
EXPECTED_DRAFT_PROTOCOL_SHA256 = "66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98"

# This intentionally remains unset in Phase 1.  A later, reviewed source
# revision must pin the exact bytes of the canonical derived protocol before
# any seal can authorize execution.  A protocol and seal cannot authorize
# themselves merely by agreeing with one another.
EXPECTED_SEALED_PROTOCOL_SHA256: str | None = None

SCHEMA_VERSION = "thermoroute.l2-u2-information-regime.phase1.v1"
MANIFEST_FORMAT = "thermoroute.l2-u2-fit-manifest.v1"
PHASE_STATUS = "PROPOSED_FOR_V4_SEAL_NOT_EXECUTABLE"
EXECUTION_IMPLEMENTED = False
PREDICTOR_BOUND_BYTE_LOADER_AUTHORIZED = False
PREPROCESSING_BOUND_BYTE_LOADERS_AUTHORIZED = False
STRUCTURAL_INPUTS_CAN_AUTHORIZE_EXECUTION = False
UNSCOPED_LABEL_VIEW_STATUS = "UNSCOPED_OR_ALL_H7_NOT_EXECUTION_READY"
FOLD_LABEL_VIEW_STATUS = "AUTHORITY_BOUND_FOLD_HELD_EXACT_KEY_VIEW"
COMPLETION_UNAVAILABLE_STATUS = (
    "UNIMPLEMENTED_MODEL_TRAINING_AND_EXACT_PREDICTION_CONTENT_VALIDATION"
)

FORCING = "F0"
LOCAL_LEVEL = "L2_U2"
GEOMETRY = "whole_region"
LEAD_DAYS = 7
HISTORY_CONTEXT_DAYS = 32
FOLDS = tuple(range(4))
MODEL_FIT_SEEDS = tuple(range(5))
TRAINING_PERIOD = "2006-01-01/2015-12-31"
EXPECTED_STATION_COUNT = 120
EXPECTED_HOLD_SIZES = (30, 30, 31, 29)

# Hashes cover only the canonical site/HUC2 mapping and fold records, not the
# registry's outcome-coverage columns.  A changed cohort or grouping therefore
# cannot silently reuse the plan.
EXPECTED_STATION_HUC2_SHA256 = "6f7891968a841e3eb5476074e08f24462b6a8984fdb1f12eb6b6b5c7d389af5e"
EXPECTED_FOLD_REGISTRY_SHA256 = "c8d7ca871fb364ba1f3efd6a2613ad94a392d49ae3705e13652caa028af7347d"
EXPECTED_PLAN_SHA256 = "965e9091d2a97ac1a8abe78c6abb802c8b2ba00b5fd2c7146f281b56f783c786"
EXPECTED_FOLD_RECORD_SHA256 = MappingProxyType(
    {
        0: "397dc053bcdea78d6efa445bd8037f7262516e0a43ddadf83baefb4a17ee6ed1",
        1: "8b8baac11ac42d172bbbf44041aa268d785e80cb13ebcbed872372ba376dd222",
        2: "e9503c90cc0e62cd9ef87d948271b1baa6031469069677ab5a9a6234ee4646b3",
        3: "39e29fa7ddf65f55c4190729d132d2b08c8fd5b4045e28ea9cebdc5c901bbb90",
    }
)

HISTORY_VARIABLES = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
PROHIBITED_LOCAL_VARIABLES = ("WTEMP", "FLOW")
ALLOWED_METEOROLOGY = ("TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
TRAINING_SOURCE_FORMAT = "thermoroute.l2-u2-training-source-registry.v1"
LABEL_REGISTRY_FORMAT = "thermoroute.l2-u2-label-registry.v1"
FOLD_REGISTRY_FORMAT = "thermoroute.l2-u2-fold-registry.v1"
VALID_SITE_ID_LENGTHS = (8, 15)
SHARED_PRIMARY_REPORTABLE_COLUMNS = (
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
SHARED_PRIMARY_REPORTABLE_ARROW_SCHEMA = pa.schema(
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
SHARED_PRIMARY_REPORTABLE_PARQUET_PHYSICAL_TYPES = (
    "BYTE_ARRAY",
    "BYTE_ARRAY",
    "INT64",
    "INT64",
    "INT32",
    "DOUBLE",
    "BOOLEAN",
    "INT32",
    "INT32",
    "DOUBLE",
    "INT32",
    "DOUBLE",
    "INT32",
    "DOUBLE",
    "BOOLEAN",
    "BOOLEAN",
    "BOOLEAN",
    "BOOLEAN",
)
SHARED_PRIMARY_REPORTABLE_PANDAS_DTYPES = MappingProxyType(
    {
        "key_id": np.dtype("O"),
        "site_id": np.dtype("O"),
        "issue_date": np.dtype("datetime64[ns]"),
        "target_date": np.dtype("datetime64[ns]"),
        "lead_days": np.dtype("int16"),
        "y_true": np.dtype("float64"),
        "issue_wtemp_observed": np.dtype("bool"),
        "days_since_last_observed_wtemp": np.dtype("int32"),
        "n_observed_wtemp_7d": np.dtype("int16"),
        "fraction_observed_wtemp_7d": np.dtype("float64"),
        "n_observed_wtemp_14d": np.dtype("int16"),
        "fraction_observed_wtemp_14d": np.dtype("float64"),
        "n_observed_wtemp_32d": np.dtype("int16"),
        "fraction_observed_wtemp_32d": np.dtype("float64"),
        "history_H100": np.dtype("bool"),
        "history_H75": np.dtype("bool"),
        "history_Hall": np.dtype("bool"),
        "reportable_primary": np.dtype("bool"),
    }
)


class ContractError(ValueError):
    """A plan, fold, input, or artifact violates the frozen Phase-1 contract."""


class GovernanceError(RuntimeError):
    """The protocol/seal does not authorize an L2_U2 execution."""


class Phase1ExecutionUnavailable(RuntimeError):
    """Training/scoring is intentionally absent from this Phase-1 scaffold."""


class Phase1CompletionUnavailable(RuntimeError):
    """Phase 1 cannot attest a completed experiment."""


class Architecture(str, Enum):
    LIGHTGBM = "LightGBM"
    PLAIN_TCN = "plain_TCN"


class ArtifactStatus(str, Enum):
    PLANNED = "PLANNED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


def _canonical_json_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(document: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest") from exc
    return value


def _repository_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"path is outside the repository: {path}") from exc


def _read_stable_regular(path: Path, *, error_type: type[Exception] = ContractError) -> bytes:
    """Read one repository file once, rejecting links and concurrent replacement."""

    candidate = Path(path)
    try:
        lexical_absolute = candidate.absolute()
        lexical_relative = lexical_absolute.relative_to(ROOT.absolute())
        current = ROOT.absolute()
        for part in lexical_relative.parts:
            current = current / part
            if stat.S_ISLNK(current.lstat().st_mode):
                raise error_type(f"trusted repository path crosses a symlink: {candidate}")
        absolute = lexical_absolute.resolve(strict=True)
        absolute.relative_to(ROOT.resolve())
    except (OSError, ValueError) as exc:
        raise error_type(f"trusted repository file is unavailable: {candidate}") from exc
    try:
        before = absolute.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise error_type(f"trusted repository path is not a singly-linked file: {candidate}")
        with absolute.open("rb") as stream:
            payload = stream.read()
            descriptor = stream.fileno()
            opened = Path(f"/proc/self/fd/{descriptor}").stat()
        after = absolute.stat()
    except OSError as exc:
        raise error_type(f"cannot stably read trusted repository file: {candidate}") from exc

    def identity(value: Any) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(opened) or identity(before) != identity(after):
        raise error_type(f"trusted repository file changed while being read: {candidate}")
    if len(payload) != before.st_size:
        raise error_type(f"trusted repository file size changed while being read: {candidate}")
    return payload


def _read_yaml_mapping(path: Path) -> tuple[bytes, Mapping[str, object]]:
    try:
        payload = _read_stable_regular(Path(path))
        document = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ContractError(f"cannot read valid YAML from {path}") from exc
    if not isinstance(document, Mapping):
        raise ContractError(f"YAML root must be a mapping: {path}")
    return payload, document


def _read_frozen_draft() -> tuple[bytes, Mapping[str, object]]:
    payload, document = _read_yaml_mapping(DEFAULT_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_DRAFT_PROTOCOL_SHA256:
        raise ContractError(
            f"immutable v4 draft bytes changed: {digest} != {EXPECTED_DRAFT_PROTOCOL_SHA256}"
        )
    return payload, document


# ---------------------------------------------------------------------------
# Static protocol/legacy audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditFinding:
    finding_id: str
    severity: str
    statement: str

    def to_record(self) -> dict[str, str]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "statement": self.statement,
        }


def audit_v4_and_legacy(
    protocol: Mapping[str, object],
    legacy_protocol_text: str,
    legacy_runner_text: str,
) -> tuple[AuditFinding, ...]:
    """Validate the relevant text and return the non-outcome design audit."""

    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ContractError("unexpected v4 protocol_id")
    if protocol.get("version") != EXPECTED_PROTOCOL_VERSION:
        raise ContractError("unexpected v4 protocol version")

    axes = protocol.get("axes")
    if not isinstance(axes, Mapping):
        raise ContractError("v4 protocol lacks axes")
    local_axis = axes.get("L_local_information")
    geometry_axis = axes.get("G_spatial_geometry")
    architecture_axis = axes.get("A_model_class")
    if not all(
        isinstance(value, Mapping) for value in (local_axis, geometry_axis, architecture_axis)
    ):
        raise ContractError("v4 L/G/A axes must be mappings")
    assert isinstance(local_axis, Mapping)
    assert isinstance(geometry_axis, Mapping)
    assert isinstance(architecture_axis, Mapping)
    l2_u2 = local_axis.get("L2_U2")
    whole_region = geometry_axis.get("whole_region")
    if not isinstance(l2_u2, Mapping) or not isinstance(whole_region, Mapping):
        raise ContractError("v4 lacks L2_U2 or whole_region")
    if l2_u2.get("inherits") != "L2" or l2_u2.get("issue_time_flow") != "masked":
        raise ContractError("v4 L2_U2 inheritance/flow mask differs from the Phase-1 scope")
    if l2_u2.get("required_scope") != "whole_region, 7-day, LightGBM and plain_TCN":
        raise ContractError("v4 L2_U2 required_scope differs from the Phase-1 scope")
    if whole_region.get("folds") != 4:
        raise ContractError("v4 whole_region must contain four folds")
    if architecture_axis.get("primary_levels") != ["LightGBM", "plain_TCN"]:
        raise ContractError("v4 primary architecture levels changed")
    if architecture_axis.get("model_fit_seeds") != [0, 1, 2, 3, 4]:
        raise ContractError("v4 matched model-fit seeds changed")

    legacy_protocol_markers = (
        "U2_fully_ungauged: flow masked (sensitivity; region geometry, 7 d,",
        "LightGBM only, 24 cells)",
        "plus: L2_U2 sensitivity (region, 7 d, LightGBM, 24 cells) = 600 total",
    )
    missing_protocol = [
        marker for marker in legacy_protocol_markers if marker not in legacy_protocol_text
    ]
    if missing_protocol:
        raise ContractError(f"legacy v2 protocol markers changed: {missing_protocol}")
    legacy_runner_markers = (
        'MASKED_LEVELS = {"L2", "L2U2", "L3"}',
        'ap.add_argument("--levels", default="L0,L1,L2,L3")',
        'ap.add_argument("--models", default="LightGBM,ResidualLightGBM")',
        'if level == "L2U2":',
        'var="FLOW"',
        "Trains the L2 region fold-0 LightGBM 7-day cell twice",
        "fit_mask | val_mask",
        'tab.split.isin(["train", "val"])',
        "if sp.exists() and sp.stat().st_size > 0:",
    )
    missing_runner = [
        marker for marker in legacy_runner_markers if marker not in legacy_runner_text
    ]
    if missing_runner:
        raise ContractError(f"legacy runner markers changed: {missing_runner}")

    return (
        AuditFinding(
            "V2_24_CELL_ARITHMETIC_CONTRADICTION",
            "ERROR",
            "The v2 prose fixes one geometry, one lead, one level, and one architecture. "
            "That is one logical cell and four regional fold units, not 24 cells.",
        ),
        AuditFinding(
            "LEGACY_SCOPE_NOT_EXECUTABLE_AS_WRITTEN",
            "ERROR",
            "The legacy default excludes L2U2, while its generic loops allow L2U2 under "
            "arbitrary geometries, models, and leads; the stated special scope is not enforced.",
        ),
        AuditFinding(
            "LEGACY_ARCHITECTURE_MISMATCH",
            "ERROR",
            "The legacy architecture set is LightGBM/ResidualLightGBM; v4 requires "
            "LightGBM/plain_TCN for L2_U2.",
        ),
        AuditFinding(
            "LEGACY_MASK_PROOF_INCOMPLETE",
            "ERROR",
            "The legacy proof covers only WTEMP for one L2 LightGBM cell. It does not test "
            "L2U2 FLOW values/observedness, site state, preprocessing state, or plain_TCN.",
        ),
        AuditFinding(
            "LEGACY_PREPROCESSING_LINEAGE_NOT_TRAIN_ONLY",
            "ERROR",
            "The legacy fit mask and model rows include validation years and have no "
            "fold-bound training-only imputer/scaler/statistic hashes.",
        ),
        AuditFinding(
            "LEGACY_LABEL_AUTHORITY_NOT_SEPARATE",
            "ERROR",
            "The legacy call repairs labels through the imputed panel path instead of a "
            "separate immutable outcome registry with an independent hash.",
        ),
        AuditFinding(
            "LEGACY_RESUME_IS_EXISTENCE_ONLY",
            "ERROR",
            "A non-empty shard path is accepted as resumable without cell identity, "
            "provenance, content hash, exact-key, or target-byte validation.",
        ),
        AuditFinding(
            "V4_L2_U2_FORCING_AXIS_UNSPECIFIED",
            "BLOCKS_SEAL",
            "The v4 L2_U2 required_scope omits F. Phase 1 records F0 as the explicit "
            "no-future-forcing interpretation of the information ladder, but a future "
            "v4 seal must affirm that choice before execution.",
        ),
        AuditFinding(
            "V4_COUNT_RESOLUTION",
            "RESOLVED_IN_PHASE1_PLAN",
            "v4 has 2 logical architecture cells, 8 architecture-by-fold units, and 40 "
            "fit units when all 5 matched model-fit seeds are applied.",
        ),
    )


# ---------------------------------------------------------------------------
# Exact plan arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class FitPlan:
    architecture: Architecture
    fold: int
    fit_seed: int
    forcing: str = FORCING
    local_level: str = LOCAL_LEVEL
    geometry: str = GEOMETRY
    lead_days: int = LEAD_DAYS

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, Architecture):
            raise TypeError("architecture must be an Architecture")
        if type(self.fold) is not int or self.fold not in FOLDS:
            raise ContractError(f"fold must be one of {FOLDS}")
        if type(self.fit_seed) is not int or self.fit_seed not in MODEL_FIT_SEEDS:
            raise ContractError(f"fit_seed must be one of {MODEL_FIT_SEEDS}")
        if (self.forcing, self.local_level, self.geometry, self.lead_days) != (
            FORCING,
            LOCAL_LEVEL,
            GEOMETRY,
            LEAD_DAYS,
        ):
            raise ContractError("L2_U2 Phase-1 axes are immutable")

    @property
    def logical_cell_id(self) -> str:
        return "__".join(
            (
                self.forcing,
                self.local_level,
                self.geometry,
                f"h{self.lead_days}",
                self.architecture.value,
            )
        )

    @property
    def fold_unit_id(self) -> str:
        return f"{self.logical_cell_id}__fold{self.fold}"

    @property
    def fit_id(self) -> str:
        return f"{self.fold_unit_id}__fit{self.fit_seed}"

    def to_record(self) -> dict[str, object]:
        return {
            "forcing": self.forcing,
            "local_level": self.local_level,
            "geometry": self.geometry,
            "lead_days": self.lead_days,
            "architecture": self.architecture.value,
            "fold": self.fold,
            "fit_seed": self.fit_seed,
            "logical_cell_id": self.logical_cell_id,
            "fold_unit_id": self.fold_unit_id,
            "fit_id": self.fit_id,
        }


def build_fit_plan() -> tuple[FitPlan, ...]:
    plan = tuple(
        sorted(
            FitPlan(architecture=architecture, fold=fold, fit_seed=fit_seed)
            for architecture in Architecture
            for fold in FOLDS
            for fit_seed in MODEL_FIT_SEEDS
        )
    )
    validate_fit_plan(plan)
    return plan


def plan_arithmetic(plan: Sequence[FitPlan]) -> dict[str, object]:
    logical = {cell.logical_cell_id for cell in plan}
    fold_units = {cell.fold_unit_id for cell in plan}
    return {
        "logical_cell_count": len(logical),
        "logical_cell_arithmetic": "2 architectures x 1 forcing interpretation x 1 lead "
        "x 1 local level x 1 geometry",
        "architecture_by_fold_unit_count": len(fold_units),
        "fold_unit_arithmetic": "2 logical cells x 4 canonical HUC2 folds",
        "model_fit_seed_count": len(MODEL_FIT_SEEDS),
        "model_fit_seeds": list(MODEL_FIT_SEEDS),
        "fit_unit_count": len(plan),
        "fit_unit_arithmetic": "8 architecture-by-fold units x 5 matched model-fit seeds",
        "legacy_24_inherited": False,
    }


def fit_plan_sha256(plan: Sequence[FitPlan]) -> str:
    return canonical_sha256([cell.to_record() for cell in plan])


def validate_fit_plan(plan: Sequence[FitPlan]) -> None:
    cells = tuple(plan)
    if len(cells) != 40 or any(not isinstance(cell, FitPlan) for cell in cells):
        raise ContractError("the L2_U2 plan must contain exactly 40 typed fit units")
    if len({cell.fit_id for cell in cells}) != len(cells):
        raise ContractError("fit plan contains duplicate identities")
    arithmetic = plan_arithmetic(cells)
    if (
        arithmetic["logical_cell_count"],
        arithmetic["architecture_by_fold_unit_count"],
        arithmetic["fit_unit_count"],
    ) != (2, 8, 40):
        raise ContractError("L2_U2 plan arithmetic changed")
    digest = fit_plan_sha256(cells)
    if digest != EXPECTED_PLAN_SHA256:
        raise ContractError(f"fit plan hash mismatch: {digest} != {EXPECTED_PLAN_SHA256}")


# ---------------------------------------------------------------------------
# Canonical whole-HUC2 folds
# ---------------------------------------------------------------------------


def _canonical_station_id(value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ContractError(f"station id must be an exact digit string: {value!r}")
    if len(value) not in VALID_SITE_ID_LENGTHS or not value.isdigit():
        raise ContractError("station id must contain exactly 8 or 15 decimal digits")
    return value


def _canonical_huc2(value: object) -> str:
    huc = str(value).strip()
    if not huc or not huc.isdigit():
        raise ContractError(f"invalid HUC2 identifier: {value!r}")
    return huc.zfill(2)


def load_station_huc2_metadata(path: Path = STATION_REGISTRY) -> dict[str, str]:
    """Read only site identifiers and HUC2 metadata; never outcome columns."""

    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {"site_no", "huc2"}.issubset(reader.fieldnames):
                raise ContractError("station registry lacks site_no/huc2")
            pairs = [
                (_canonical_station_id(row["site_no"]), _canonical_huc2(row["huc2"]))
                for row in reader
            ]
    except OSError as exc:
        raise ContractError(f"cannot read station metadata: {path}") from exc
    if not pairs or len({site for site, _huc in pairs}) != len(pairs):
        raise ContractError("station/HUC2 metadata is empty or contains duplicate sites")
    return dict(sorted(pairs))


def station_huc2_sha256(huc2_by_station: Mapping[str, str]) -> str:
    items = sorted(
        (_canonical_station_id(site), _canonical_huc2(huc)) for site, huc in huc2_by_station.items()
    )
    if len(items) != len({site for site, _huc in items}):
        raise ContractError("station/HUC2 mapping contains duplicate canonical sites")
    return canonical_sha256(items)


@dataclass(frozen=True, slots=True)
class FoldDefinition:
    fold: int
    train_stations: tuple[str, ...]
    held_stations: tuple[str, ...]
    held_huc2: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.fold) is not int or self.fold not in FOLDS:
            raise ContractError(f"fold must be one of {FOLDS}")
        for label, values in (
            ("train_stations", self.train_stations),
            ("held_stations", self.held_stations),
            ("held_huc2", self.held_huc2),
        ):
            if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
                raise ContractError(f"{label} must be non-empty, unique, and sorted")
        if set(self.train_stations) & set(self.held_stations):
            raise ContractError("fold train and held stations overlap")

    def to_record(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "geometry": GEOMETRY,
            "train_stations": list(self.train_stations),
            "held_stations": list(self.held_stations),
            "held_huc2": list(self.held_huc2),
        }


def build_canonical_huc2_folds(
    huc2_by_station: Mapping[str, str],
) -> tuple[FoldDefinition, ...]:
    """Greedily pack intact HUC2 groups with deterministic tie breaking."""

    canonical = {
        _canonical_station_id(site): _canonical_huc2(huc) for site, huc in huc2_by_station.items()
    }
    if len(canonical) != len(huc2_by_station) or len(canonical) < 4:
        raise ContractError("canonical station/HUC2 registry is invalid")
    groups: dict[str, list[str]] = {}
    for site, huc in canonical.items():
        groups.setdefault(huc, []).append(site)
    buckets: list[list[str]] = [[] for _ in FOLDS]
    huc_buckets: list[list[str]] = [[] for _ in FOLDS]
    loads = [0 for _ in FOLDS]
    for huc, stations in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = min(FOLDS, key=lambda index: (loads[index], index))
        buckets[fold].extend(stations)
        huc_buckets[fold].append(huc)
        loads[fold] += len(stations)
    registry = set(canonical)
    folds = tuple(
        FoldDefinition(
            fold=fold,
            train_stations=tuple(sorted(registry - set(buckets[fold]))),
            held_stations=tuple(sorted(buckets[fold])),
            held_huc2=tuple(sorted(huc_buckets[fold])),
        )
        for fold in FOLDS
    )
    validate_fold_registry(folds, canonical)
    return folds


def validate_fold_registry(
    folds: Sequence[FoldDefinition], huc2_by_station: Mapping[str, str]
) -> None:
    canonical = {
        _canonical_station_id(site): _canonical_huc2(huc) for site, huc in huc2_by_station.items()
    }
    observed = tuple(folds)
    if len(observed) != 4 or tuple(fold.fold for fold in observed) != FOLDS:
        raise ContractError("whole-region registry must have ordered folds 0..3")
    held_once: list[str] = []
    for fold in observed:
        if set(fold.train_stations) | set(fold.held_stations) != set(canonical):
            raise ContractError(f"fold {fold.fold} does not partition the station registry")
        train_huc = {canonical[site] for site in fold.train_stations}
        held_huc = {canonical[site] for site in fold.held_stations}
        if train_huc & held_huc:
            raise ContractError(f"fold {fold.fold} splits an HUC2 group")
        if tuple(sorted(held_huc)) != fold.held_huc2:
            raise ContractError(f"fold {fold.fold} held-HUC2 declaration is wrong")
        held_once.extend(fold.held_stations)
    if Counter(held_once) != Counter(canonical.keys()):
        raise ContractError("four folds must hold every station exactly once")


def fold_registry_sha256(
    folds: Sequence[FoldDefinition], huc2_by_station: Mapping[str, str]
) -> str:
    validate_fold_registry(folds, huc2_by_station)
    return canonical_sha256([fold.to_record() for fold in folds])


def validate_canonical_repository_folds(
    folds: Sequence[FoldDefinition], huc2_by_station: Mapping[str, str]
) -> None:
    if len(huc2_by_station) != EXPECTED_STATION_COUNT:
        raise ContractError(f"v4 L2_U2 cohort must contain {EXPECTED_STATION_COUNT} stations")
    mapping_digest = station_huc2_sha256(huc2_by_station)
    if mapping_digest != EXPECTED_STATION_HUC2_SHA256:
        raise ContractError("station/HUC2 mapping differs from the frozen Phase-1 cohort")
    sizes = tuple(len(fold.held_stations) for fold in folds)
    if sizes != EXPECTED_HOLD_SIZES:
        raise ContractError(f"canonical HUC2 hold sizes {sizes} != {EXPECTED_HOLD_SIZES}")
    digest = fold_registry_sha256(folds, huc2_by_station)
    if digest != EXPECTED_FOLD_REGISTRY_SHA256:
        raise ContractError("canonical HUC2 fold registry hash changed")


def validate_canonical_fold_definition(fold: FoldDefinition) -> None:
    """Bind a training operation to one exact record in the 120-site registry."""

    if not isinstance(fold, FoldDefinition):
        raise TypeError("fold must be a FoldDefinition")
    digest = canonical_sha256(fold.to_record())
    expected = EXPECTED_FOLD_RECORD_SHA256.get(fold.fold)
    if digest != expected:
        raise ContractError(f"fold {fold.fold} is not the frozen canonical 120-site fold record")


# ---------------------------------------------------------------------------
# Fold-aware training-only preprocessing and model-input boundary
# ---------------------------------------------------------------------------


def _immutable_array(value: object, dtype: type) -> np.ndarray:
    """Return an ndarray backed by immutable ``bytes``, not writable owned memory."""

    contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    payload = contiguous.tobytes(order="C")
    immutable = np.frombuffer(payload, dtype=contiguous.dtype).reshape(contiguous.shape)
    if immutable.flags.writeable:
        raise AssertionError("bytes-backed array unexpectedly writable")
    return immutable


def _validate_immutable_array(value: object, *, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.flags.writeable:
        raise ContractError(f"{label} must be a bytes-backed immutable ndarray")
    try:
        value.setflags(write=True)
    except ValueError:
        return value
    value.setflags(write=False)
    raise ContractError(f"{label} can be made writable and is not an immutable boundary")


def _digest_array(hasher: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(label.encode("ascii"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(_canonical_json_bytes(list(array.shape)))
    hasher.update(array.tobytes(order="C"))


def _strict_daily_date(value: object, *, label: str) -> date:
    if not isinstance(value, str):
        raise ContractError(f"{label} must be a canonical YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{label} must be a canonical YYYY-MM-DD string") from exc
    if value != parsed.isoformat():
        raise ContractError(f"{label} must be a canonical YYYY-MM-DD string")
    return parsed


@dataclass(frozen=True, slots=True, order=True)
class ForecastKey:
    """Parsed canonical tuple; its digest matches the v4 primary key registry."""

    site_id: str
    issue_time: str
    target_time: str
    lead_days: int = LEAD_DAYS

    def __post_init__(self) -> None:
        canonical_site = _canonical_station_id(self.site_id)
        if canonical_site != self.site_id:
            raise ContractError("forecast key site_id normalization is forbidden")
        if type(self.lead_days) is not int or self.lead_days != LEAD_DAYS:
            raise ContractError("L2_U2 keys must have lead_days=7")
        issue = _strict_daily_date(self.issue_time, label="forecast issue_time")
        target = _strict_daily_date(self.target_time, label="forecast target_time")
        if target - issue != timedelta(days=LEAD_DAYS):
            raise ContractError("forecast key target_time must be exactly issue_time + 7 days")

    @property
    def key_id(self) -> str:
        payload = (
            f"temporal|known_site|{self.site_id}|{self.issue_time}|"
            f"{self.target_time}|{self.lead_days}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def issue_date(self) -> date:
        return date.fromisoformat(self.issue_time)

    @property
    def target_date(self) -> date:
        return date.fromisoformat(self.target_time)

    @property
    def target_day_of_year(self) -> int:
        return self.target_date.timetuple().tm_yday

    def to_record(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "issue_time": self.issue_time,
            "target_time": self.target_time,
            "lead_days": self.lead_days,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ForecastKey:
        required = {"key_id", "site_id", "issue_time", "target_time", "lead_days"}
        if set(record) != required:
            raise ContractError("forecast-key record must have the exact canonical schema")
        key = cls(
            site_id=record["site_id"],  # type: ignore[arg-type]
            issue_time=record["issue_time"],  # type: ignore[arg-type]
            target_time=record["target_time"],  # type: ignore[arg-type]
            lead_days=record["lead_days"],  # type: ignore[arg-type]
        )
        if record["key_id"] != key.key_id:
            raise ContractError("forecast key_id does not hash its canonical tuple")
        return key


@dataclass(frozen=True, slots=True)
class PreprocessingSources:
    """Typed, dated, training-stations-only source registry snapshot."""

    fold: int
    meteorology_registry_sha256: str
    anchor_registry_sha256: str
    meteorology_dates_by_station: Mapping[str, tuple[str, ...]]
    meteorology_rows_by_station: Mapping[str, np.ndarray]
    anchor_curve_by_station: Mapping[str, np.ndarray]
    registry_format: str = TRAINING_SOURCE_FORMAT
    period_start: str = "2006-01-01"
    period_end: str = "2015-12-31"
    station_huc2_registry_sha256: str = EXPECTED_STATION_HUC2_SHA256
    fold_registry_sha256: str = EXPECTED_FOLD_REGISTRY_SHA256

    def __post_init__(self) -> None:
        if self.registry_format != TRAINING_SOURCE_FORMAT:
            raise ContractError("training source registry format changed")
        if type(self.fold) is not int or self.fold not in FOLDS:
            raise ContractError("training source registry fold is not canonical")
        if (self.period_start, self.period_end) != ("2006-01-01", "2015-12-31"):
            raise ContractError("preprocessing sources must be restricted to 2006--2015")
        if self.station_huc2_registry_sha256 != EXPECTED_STATION_HUC2_SHA256:
            raise ContractError("training sources bind a noncanonical station registry")
        if self.fold_registry_sha256 != EXPECTED_FOLD_REGISTRY_SHA256:
            raise ContractError("training sources bind a noncanonical fold registry")
        _validate_sha256(
            self.meteorology_registry_sha256,
            label="training meteorology_registry_sha256",
        )
        _validate_sha256(self.anchor_registry_sha256, label="training anchor_registry_sha256")

    @property
    def training_period(self) -> str:
        return f"{self.period_start}/{self.period_end}"


@dataclass(frozen=True, slots=True, init=False)
class TrainingOnlyPreprocessor:
    fold: int
    training_stations: tuple[str, ...]
    held_stations: tuple[str, ...]
    held_huc2: tuple[str, ...]
    variables: tuple[str, ...]
    imputer_fill: np.ndarray
    scaler_center: np.ndarray
    scaler_scale: np.ndarray
    pooled_anchor_curve: np.ndarray
    source_sha256: str
    state_sha256: str
    training_period: str
    meteorology_registry_sha256: str
    anchor_registry_sha256: str
    station_huc2_registry_sha256: str
    fold_registry_sha256: str
    attestation: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ContractError(
            "TrainingOnlyPreprocessor has no public constructor; fit exact dated sources"
        )

    def state_record(self, *, include_state_hash: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "fold": self.fold,
            "training_stations": list(self.training_stations),
            "held_stations": list(self.held_stations),
            "held_huc2": list(self.held_huc2),
            "variables": list(self.variables),
            "imputer_fill": np.asarray(self.imputer_fill).tolist(),
            "scaler_center": np.asarray(self.scaler_center).tolist(),
            "scaler_scale": np.asarray(self.scaler_scale).tolist(),
            "pooled_anchor_curve": np.asarray(self.pooled_anchor_curve).tolist(),
            "source_sha256": self.source_sha256,
            "training_period": self.training_period,
            "meteorology_registry_sha256": self.meteorology_registry_sha256,
            "anchor_registry_sha256": self.anchor_registry_sha256,
            "station_huc2_registry_sha256": self.station_huc2_registry_sha256,
            "fold_registry_sha256": self.fold_registry_sha256,
            "attestation": self.attestation,
        }
        if include_state_hash:
            record["state_sha256"] = self.state_sha256
        return record


def validate_training_only_preprocessor(preprocessor: TrainingOnlyPreprocessor) -> None:
    """Revalidate every byte-backed field and hash at each projection boundary."""

    if type(preprocessor) is not TrainingOnlyPreprocessor:
        raise ContractError("preprocessor is not a trusted structural type")
    fold = FoldDefinition(
        fold=preprocessor.fold,
        train_stations=preprocessor.training_stations,
        held_stations=preprocessor.held_stations,
        held_huc2=preprocessor.held_huc2,
    )
    validate_canonical_fold_definition(fold)
    if preprocessor.variables != ALLOWED_METEOROLOGY:
        raise ContractError("preprocessor contains a prohibited local variable")
    if preprocessor.training_period != TRAINING_PERIOD:
        raise ContractError("preprocessor training period is not canonical 2006--2015")
    _validate_sha256(
        preprocessor.meteorology_registry_sha256,
        label="preprocessor meteorology_registry_sha256",
    )
    _validate_sha256(
        preprocessor.anchor_registry_sha256,
        label="preprocessor anchor_registry_sha256",
    )
    if preprocessor.station_huc2_registry_sha256 != EXPECTED_STATION_HUC2_SHA256:
        raise ContractError("preprocessor station registry lineage changed")
    if preprocessor.fold_registry_sha256 != EXPECTED_FOLD_REGISTRY_SHA256:
        raise ContractError("preprocessor fold registry lineage changed")
    width = len(ALLOWED_METEOROLOGY)
    for label, value in (
        ("imputer_fill", preprocessor.imputer_fill),
        ("scaler_center", preprocessor.scaler_center),
        ("scaler_scale", preprocessor.scaler_scale),
    ):
        array = _validate_immutable_array(value, label=label)
        if array.shape != (width,) or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ContractError(f"{label} must be an immutable finite float64 {width}-vector")
    if np.any(preprocessor.scaler_scale <= 0):
        raise ContractError("scaler scales must be positive")
    anchor = _validate_immutable_array(
        preprocessor.pooled_anchor_curve, label="pooled_anchor_curve"
    )
    if anchor.shape != (366,) or anchor.dtype != np.float64 or not np.isfinite(anchor).all():
        raise ContractError("pooled anchor must be immutable finite float64 length 366")
    _validate_sha256(preprocessor.source_sha256, label="preprocessing source_sha256")
    _validate_sha256(preprocessor.state_sha256, label="preprocessing state_sha256")
    expected = canonical_sha256(preprocessor.state_record(include_state_hash=False))
    if preprocessor.state_sha256 != expected:
        raise ContractError("preprocessing state hash does not bind current state bytes")


def _construct_preprocessor(**values: object) -> TrainingOnlyPreprocessor:
    instance = object.__new__(TrainingOnlyPreprocessor)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    validate_training_only_preprocessor(instance)
    return instance


def _hash_training_sources(
    *,
    fold: FoldDefinition,
    dates: Mapping[str, tuple[str, ...]],
    rows: Mapping[str, np.ndarray],
    curves: Mapping[str, np.ndarray],
    meteorology_registry_sha256: str,
    anchor_registry_sha256: str,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        _canonical_json_bytes(
            {
                "format": TRAINING_SOURCE_FORMAT,
                "period": TRAINING_PERIOD,
                "fold": fold.to_record(),
                "station_huc2_registry_sha256": EXPECTED_STATION_HUC2_SHA256,
                "fold_registry_sha256": EXPECTED_FOLD_REGISTRY_SHA256,
                "meteorology_registry_sha256": meteorology_registry_sha256,
                "anchor_registry_sha256": anchor_registry_sha256,
                "dates": {station: list(dates[station]) for station in fold.train_stations},
            }
        )
    )
    for station in fold.train_stations:
        _digest_array(hasher, f"meteorology:{station}", rows[station])
        _digest_array(hasher, f"anchor:{station}", curves[station])
    return hasher.hexdigest()


def fit_training_only_preprocessor(
    fold: FoldDefinition,
    sources: PreprocessingSources,
) -> TrainingOnlyPreprocessor:
    """Fit only exact fold-training rows from a dated 2006--2015 registry."""

    validate_canonical_fold_definition(fold)
    if not isinstance(sources, PreprocessingSources):
        raise TypeError("sources must be PreprocessingSources")
    if sources.fold != fold.fold:
        raise ContractError("training source registry is bound to a different fold")
    expected_members = set(fold.train_stations)
    for label, mapping in (
        ("meteorology dates", sources.meteorology_dates_by_station),
        ("meteorology rows", sources.meteorology_rows_by_station),
        ("anchor curves", sources.anchor_curve_by_station),
    ):
        if set(mapping) != expected_members:
            missing = sorted(expected_members - set(mapping))
            extra = sorted(set(mapping) - expected_members)
            raise ContractError(
                f"{label} must contain exact training stations only: "
                f"missing={missing}, extra={extra}"
            )

    lower = date(2006, 1, 1)
    upper = date(2015, 12, 31)
    rows: dict[str, np.ndarray] = {}
    curves: dict[str, np.ndarray] = {}
    dates: dict[str, tuple[str, ...]] = {}
    width = len(ALLOWED_METEOROLOGY)
    for station in fold.train_stations:
        station_rows = _immutable_array(sources.meteorology_rows_by_station[station], np.float64)
        station_curve = _immutable_array(sources.anchor_curve_by_station[station], np.float64)
        station_dates = tuple(sources.meteorology_dates_by_station[station])
        if station_rows.ndim != 2 or station_rows.shape[1] != width or not len(station_rows):
            raise ContractError(f"training meteorology for {station} must have shape [N,{width}]")
        if (
            len(station_dates) != len(station_rows)
            or tuple(sorted(set(station_dates))) != station_dates
        ):
            raise ContractError(f"training dates for {station} must be aligned, unique, and sorted")
        parsed_dates = [
            _strict_daily_date(value, label=f"training date for {station}")
            for value in station_dates
        ]
        if any(value < lower or value > upper for value in parsed_dates):
            raise ContractError(f"training source for {station} leaves canonical 2006--2015")
        if np.isinf(station_rows).any():
            raise ContractError(f"training meteorology for {station} contains infinity")
        if station_curve.shape != (366,) or not np.isfinite(station_curve).all():
            raise ContractError(f"training anchor curve for {station} must be finite length 366")
        rows[station] = station_rows
        curves[station] = station_curve
        dates[station] = station_dates

    pooled_rows = np.concatenate([rows[station] for station in fold.train_stations], axis=0)
    if np.any(np.all(np.isnan(pooled_rows), axis=0)):
        raise ContractError("a meteorological variable has no training-station observation")
    imputer = np.nanmedian(pooled_rows, axis=0)
    filled = np.where(np.isnan(pooled_rows), imputer, pooled_rows)
    center = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    anchor = np.mean(np.stack([curves[site] for site in fold.train_stations], axis=0), axis=0)
    source_digest = _hash_training_sources(
        fold=fold,
        dates=dates,
        rows=rows,
        curves=curves,
        meteorology_registry_sha256=sources.meteorology_registry_sha256,
        anchor_registry_sha256=sources.anchor_registry_sha256,
    )
    values: dict[str, object] = {
        "fold": fold.fold,
        "training_stations": fold.train_stations,
        "held_stations": fold.held_stations,
        "held_huc2": fold.held_huc2,
        "variables": ALLOWED_METEOROLOGY,
        "imputer_fill": _immutable_array(imputer, np.float64),
        "scaler_center": _immutable_array(center, np.float64),
        "scaler_scale": _immutable_array(scale, np.float64),
        "pooled_anchor_curve": _immutable_array(anchor, np.float64),
        "source_sha256": source_digest,
        "training_period": TRAINING_PERIOD,
        "meteorology_registry_sha256": sources.meteorology_registry_sha256,
        "anchor_registry_sha256": sources.anchor_registry_sha256,
        "station_huc2_registry_sha256": EXPECTED_STATION_HUC2_SHA256,
        "fold_registry_sha256": EXPECTED_FOLD_REGISTRY_SHA256,
        "attestation": "FOLD_AWARE_EXACT_TRAINING_STATIONS_ONLY_NO_HELD_SOURCE_READ",
    }
    provisional = object.__new__(TrainingOnlyPreprocessor)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    values["state_sha256"] = canonical_sha256(provisional.state_record(include_state_hash=False))
    return _construct_preprocessor(**values)


@dataclass(frozen=True, slots=True)
class RawPredictorBatch:
    """F0 predictor rows with exact tuple keys and no local/label channels."""

    keys: tuple[ForecastKey, ...]
    key_ids: tuple[str, ...]
    history_timestamps: tuple[tuple[str, ...], ...]
    history_variables: tuple[str, ...]
    history_values: np.ndarray
    history_observed: np.ndarray
    predictor_registry_sha256: str

    def __post_init__(self) -> None:
        if not self.keys or any(type(key) is not ForecastKey for key in self.keys):
            raise ContractError("predictor keys must be parsed canonical ForecastKey tuples")
        if tuple(sorted(self.keys)) != self.keys or len(set(self.keys)) != len(self.keys):
            raise ContractError("predictor ForecastKey tuples must be unique and sorted")
        expected_ids = tuple(key.key_id for key in self.keys)
        if self.key_ids != expected_ids:
            raise ContractError("predictor key_ids do not hash their canonical ForecastKey tuples")
        if self.history_variables != ALLOWED_METEOROLOGY:
            prohibited = sorted(set(self.history_variables) & set(PROHIBITED_LOCAL_VARIABLES))
            raise ContractError(
                "F0 history schema must contain only shared meteorology; "
                f"prohibited/local channels={prohibited}"
            )
        _validate_sha256(self.predictor_registry_sha256, label="predictor_registry_sha256")
        n = len(self.keys)
        values = _immutable_array(self.history_values, np.float64)
        observed = _immutable_array(self.history_observed, np.bool_)
        if values.shape != (n, HISTORY_CONTEXT_DAYS, len(ALLOWED_METEOROLOGY)):
            raise ContractError(
                "F0 history_values must have exact shape [N,32,5]; zero-length, "
                "short, and variable context windows are forbidden"
            )
        if observed.shape != values.shape:
            raise ContractError("F0 history_observed must match history_values")
        if not np.isfinite(values[observed]).all():
            raise ContractError("observed F0 history values must be finite")
        timestamps = tuple(tuple(row) for row in self.history_timestamps)
        if len(timestamps) != n or any(len(row) != HISTORY_CONTEXT_DAYS for row in timestamps):
            raise ContractError("history timestamps must align exactly with [N,32] predictor rows")
        for key, row in zip(self.keys, timestamps, strict=True):
            expected_dates = tuple(
                (key.issue_date - timedelta(days=offset)).isoformat()
                for offset in range(HISTORY_CONTEXT_DAYS - 1, -1, -1)
            )
            for value in row:
                _strict_daily_date(value, label=f"history timestamp for {key.key_id}")
            if row != expected_dates:
                raise ContractError(
                    "F0 history must be exactly 32 contiguous daily dates ending at issue_time"
                )
        object.__setattr__(self, "history_timestamps", timestamps)
        object.__setattr__(self, "history_values", values)
        object.__setattr__(self, "history_observed", observed)

    @property
    def station_ids(self) -> tuple[str, ...]:
        return tuple(key.site_id for key in self.keys)

    @property
    def target_day_of_year(self) -> np.ndarray:
        return _immutable_array([key.target_day_of_year for key in self.keys], np.int16)

    @property
    def season(self) -> np.ndarray:
        day = self.target_day_of_year.astype(np.float64)
        phase = 2.0 * np.pi * (day - 1.0) / 366.0
        return _immutable_array(np.column_stack((np.sin(phase), np.cos(phase))), np.float64)

    @property
    def state_sha256(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            _canonical_json_bytes(
                {
                    "keys": [key.to_record() for key in self.keys],
                    "key_ids": list(self.key_ids),
                    "history_timestamps": [list(row) for row in self.history_timestamps],
                    "history_variables": list(self.history_variables),
                    "predictor_registry_sha256": self.predictor_registry_sha256,
                }
            )
        )
        _digest_array(hasher, "history_values", self.history_values)
        _digest_array(hasher, "history_observed", self.history_observed)
        return hasher.hexdigest()


def validate_raw_predictor_batch(raw: RawPredictorBatch) -> None:
    if type(raw) is not RawPredictorBatch:
        raise ContractError("raw predictors are not the exact structural batch type")
    _validate_immutable_array(raw.history_values, label="raw history_values")
    _validate_immutable_array(raw.history_observed, label="raw history_observed")
    # Reconstructing invokes all tuple, chronology, channel, and hash checks on
    # current bytes; frozen dataclass syntax alone is not treated as trust.
    RawPredictorBatch(
        keys=raw.keys,
        key_ids=raw.key_ids,
        history_timestamps=raw.history_timestamps,
        history_variables=raw.history_variables,
        history_values=raw.history_values,
        history_observed=raw.history_observed,
        predictor_registry_sha256=raw.predictor_registry_sha256,
    )


@dataclass(frozen=True, slots=True, init=False)
class CanonicalL2U2Inputs:
    """Complete model-agnostic boundary, constructed and rehashed structurally."""

    keys: tuple[ForecastKey, ...]
    key_ids: tuple[str, ...]
    role: str
    meteorology_values: np.ndarray
    meteorology_observed: np.ndarray
    season: np.ndarray
    pooled_anchor: np.ndarray
    preprocessing_sha256: str
    predictor_registry_sha256: str
    raw_predictor_batch_sha256: str
    state_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ContractError("CanonicalL2U2Inputs has no public constructor; use guarded projection")

    @property
    def station_ids(self) -> tuple[str, ...]:
        return tuple(key.site_id for key in self.keys)

    def tensor_mapping(self) -> dict[str, np.ndarray]:
        validate_canonical_inputs(self)
        return {
            "meteorology_values": self.meteorology_values,
            "meteorology_observed": self.meteorology_observed,
            "season": self.season,
            "pooled_anchor": self.pooled_anchor,
        }


def _canonical_inputs_sha256(inputs: CanonicalL2U2Inputs) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        _canonical_json_bytes(
            {
                "keys": [key.to_record() for key in inputs.keys],
                "key_ids": list(inputs.key_ids),
                "role": inputs.role,
                "preprocessing_sha256": inputs.preprocessing_sha256,
                "predictor_registry_sha256": inputs.predictor_registry_sha256,
                "raw_predictor_batch_sha256": inputs.raw_predictor_batch_sha256,
            }
        )
    )
    for label in (
        "meteorology_values",
        "meteorology_observed",
        "season",
        "pooled_anchor",
    ):
        _digest_array(hasher, label, getattr(inputs, label))
    return hasher.hexdigest()


def validate_canonical_inputs(inputs: CanonicalL2U2Inputs) -> None:
    if type(inputs) is not CanonicalL2U2Inputs:
        raise ContractError("canonical input is not the exact guarded structural type")
    if inputs.role not in {"train", "validation", "evaluation"}:
        raise ContractError("input role must be train, validation, or evaluation")
    if inputs.key_ids != tuple(key.key_id for key in inputs.keys):
        raise ContractError("canonical key IDs no longer bind parsed key tuples")
    n = len(inputs.keys)
    values = _validate_immutable_array(inputs.meteorology_values, label="canonical meteorology")
    observed = _validate_immutable_array(
        inputs.meteorology_observed, label="canonical meteorology observedness"
    )
    season = _validate_immutable_array(inputs.season, label="canonical season")
    anchor = _validate_immutable_array(inputs.pooled_anchor, label="canonical pooled anchor")
    if values.shape != (n, HISTORY_CONTEXT_DAYS, len(ALLOWED_METEOROLOGY)):
        raise ContractError("canonical meteorology must have exact shape [N,32,5]")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ContractError("canonical meteorology must be finite immutable float32")
    if observed.shape != values.shape or observed.dtype != np.bool_:
        raise ContractError("canonical observedness must be immutable boolean and aligned")
    if season.shape != (n, 2) or season.dtype != np.float32 or not np.isfinite(season).all():
        raise ContractError("canonical season must be derived finite float32 [N,2]")
    if anchor.shape != (n,) or anchor.dtype != np.float32 or not np.isfinite(anchor).all():
        raise ContractError("canonical pooled anchor must be finite float32 [N]")
    for label in (
        "preprocessing_sha256",
        "predictor_registry_sha256",
        "raw_predictor_batch_sha256",
        "state_sha256",
    ):
        _validate_sha256(getattr(inputs, label), label=f"canonical {label}")
    if inputs.state_sha256 != _canonical_inputs_sha256(inputs):
        raise ContractError("canonical input state hash does not bind current bytes")


def _construct_canonical_inputs(**values: object) -> CanonicalL2U2Inputs:
    instance = object.__new__(CanonicalL2U2Inputs)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    object.__setattr__(instance, "state_sha256", _canonical_inputs_sha256(instance))
    validate_canonical_inputs(instance)
    return instance


def build_canonical_l2_u2_inputs(
    raw: RawPredictorBatch,
    preprocessor: TrainingOnlyPreprocessor,
    sources: PreprocessingSources,
    *,
    role: str,
) -> CanonicalL2U2Inputs:
    """Project exact F0 meteorology; station/season/DOY come only from keys."""

    validate_raw_predictor_batch(raw)
    validate_training_only_preprocessor(preprocessor)
    fold = FoldDefinition(
        fold=preprocessor.fold,
        train_stations=preprocessor.training_stations,
        held_stations=preprocessor.held_stations,
        held_huc2=preprocessor.held_huc2,
    )
    recomputed = fit_training_only_preprocessor(fold, sources)
    if (
        recomputed.source_sha256 != preprocessor.source_sha256
        or recomputed.state_sha256 != preprocessor.state_sha256
    ):
        raise ContractError(
            "preprocessor no longer matches freshly rehashed exact training-source bytes"
        )
    if role not in {"train", "validation", "evaluation"}:
        raise ContractError("input role must be train, validation, or evaluation")
    allowed_stations = (
        set(preprocessor.held_stations)
        if role == "evaluation"
        else set(preprocessor.training_stations)
    )
    outside = sorted(set(raw.station_ids) - allowed_stations)
    if outside:
        raise ContractError(
            f"{role} parsed ForecastKeys contain stations outside their fold partition: {outside}"
        )

    values = np.asarray(raw.history_values)
    observed = np.asarray(raw.history_observed)
    fill = np.asarray(preprocessor.imputer_fill)[None, None, :]
    center = np.asarray(preprocessor.scaler_center)[None, None, :]
    scale = np.asarray(preprocessor.scaler_scale)[None, None, :]
    filled = np.where(observed & np.isfinite(values), values, fill)
    standardized = (filled - center) / scale
    target_doy = raw.target_day_of_year
    anchor = np.asarray(preprocessor.pooled_anchor_curve)[target_doy - 1]
    return _construct_canonical_inputs(
        keys=raw.keys,
        key_ids=raw.key_ids,
        role=role,
        meteorology_values=_immutable_array(standardized, np.float32),
        meteorology_observed=_immutable_array(observed, np.bool_),
        season=_immutable_array(raw.season, np.float32),
        pooled_anchor=_immutable_array(anchor, np.float32),
        preprocessing_sha256=preprocessor.state_sha256,
        predictor_registry_sha256=raw.predictor_registry_sha256,
        raw_predictor_batch_sha256=raw.state_sha256,
    )


@dataclass(frozen=True, slots=True, init=False)
class LightGBMInputs:
    key_ids: tuple[str, ...]
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    preprocessing_sha256: str
    canonical_input_sha256: str
    state_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ContractError("LightGBMInputs has no public constructor; use the guarded adapter")

    def tensor_mapping(self) -> dict[str, np.ndarray]:
        validate_lightgbm_inputs(self)
        return {"matrix": self.matrix}


@dataclass(frozen=True, slots=True, init=False)
class PlainTCNInputs:
    key_ids: tuple[str, ...]
    sequence: np.ndarray
    sequence_observed: np.ndarray
    static_context: np.ndarray
    channel_names: tuple[str, ...]
    static_names: tuple[str, ...]
    preprocessing_sha256: str
    canonical_input_sha256: str
    state_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ContractError("PlainTCNInputs has no public constructor; use the guarded adapter")

    def tensor_mapping(self) -> dict[str, np.ndarray]:
        validate_plain_tcn_inputs(self)
        return {
            "sequence": self.sequence,
            "sequence_observed": self.sequence_observed,
            "static_context": self.static_context,
        }


def _architecture_state_sha256(inputs: LightGBMInputs | PlainTCNInputs) -> str:
    hasher = hashlib.sha256()
    metadata: dict[str, object] = {
        "type": type(inputs).__name__,
        "key_ids": list(inputs.key_ids),
        "preprocessing_sha256": inputs.preprocessing_sha256,
        "canonical_input_sha256": inputs.canonical_input_sha256,
    }
    if type(inputs) is LightGBMInputs:
        metadata["feature_names"] = list(inputs.feature_names)
        arrays = (("matrix", inputs.matrix),)
    elif type(inputs) is PlainTCNInputs:
        metadata["channel_names"] = list(inputs.channel_names)
        metadata["static_names"] = list(inputs.static_names)
        arrays = (
            ("sequence", inputs.sequence),
            ("sequence_observed", inputs.sequence_observed),
            ("static_context", inputs.static_context),
        )
    else:
        raise ContractError("unknown architecture input structural type")
    hasher.update(_canonical_json_bytes(metadata))
    for label, array in arrays:
        _digest_array(hasher, label, array)
    return hasher.hexdigest()


def validate_lightgbm_inputs(inputs: LightGBMInputs) -> None:
    if type(inputs) is not LightGBMInputs:
        raise ContractError("LightGBM input is not the exact guarded adapter type")
    matrix = _validate_immutable_array(inputs.matrix, label="LightGBM matrix")
    if matrix.ndim != 2 or matrix.shape[0] != len(inputs.key_ids) or matrix.dtype != np.float32:
        raise ContractError("LightGBM matrix shape/dtype is invalid")
    if not np.isfinite(matrix).all() or len(inputs.feature_names) != matrix.shape[1]:
        raise ContractError("LightGBM matrix/features are nonfinite or misaligned")
    if any(
        fragment.lower() in name.lower()
        for name in inputs.feature_names
        for fragment in (*PROHIBITED_LOCAL_VARIABLES, "site", "label", "y_true")
    ):
        raise ContractError("LightGBM adapter exposes a prohibited local/identity/label channel")
    for label in ("preprocessing_sha256", "canonical_input_sha256", "state_sha256"):
        _validate_sha256(getattr(inputs, label), label=f"LightGBM {label}")
    if inputs.state_sha256 != _architecture_state_sha256(inputs):
        raise ContractError("LightGBM adapter hash does not bind current bytes")


def validate_plain_tcn_inputs(inputs: PlainTCNInputs) -> None:
    if type(inputs) is not PlainTCNInputs:
        raise ContractError("plain_TCN input is not the exact guarded adapter type")
    sequence = _validate_immutable_array(inputs.sequence, label="plain_TCN sequence")
    observed = _validate_immutable_array(
        inputs.sequence_observed, label="plain_TCN sequence_observed"
    )
    static = _validate_immutable_array(inputs.static_context, label="plain_TCN static_context")
    if sequence.shape != (
        len(inputs.key_ids),
        HISTORY_CONTEXT_DAYS,
        len(ALLOWED_METEOROLOGY),
    ):
        raise ContractError("plain_TCN sequence must have exact shape [N,32,5]")
    if sequence.dtype != np.float32:
        raise ContractError("plain_TCN channel shape/dtype is invalid")
    if observed.shape != sequence.shape or observed.dtype != np.bool_:
        raise ContractError("plain_TCN observedness shape/dtype is invalid")
    if static.shape != (len(inputs.key_ids), 3) or static.dtype != np.float32:
        raise ContractError("plain_TCN static context shape/dtype is invalid")
    if inputs.channel_names != ALLOWED_METEOROLOGY or inputs.static_names != (
        "sin_doy",
        "cos_doy",
        "pooled_anchor",
    ):
        raise ContractError("plain_TCN exposes a noncanonical or prohibited channel")
    if not np.isfinite(sequence).all() or not np.isfinite(static).all():
        raise ContractError("plain_TCN tensors must be finite")
    for label in ("preprocessing_sha256", "canonical_input_sha256", "state_sha256"):
        _validate_sha256(getattr(inputs, label), label=f"plain_TCN {label}")
    if inputs.state_sha256 != _architecture_state_sha256(inputs):
        raise ContractError("plain_TCN adapter hash does not bind current bytes")


def _construct_architecture_inputs(
    cls: type[LightGBMInputs | PlainTCNInputs], **values: object
) -> LightGBMInputs | PlainTCNInputs:
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    object.__setattr__(instance, "state_sha256", _architecture_state_sha256(instance))
    if type(instance) is LightGBMInputs:
        validate_lightgbm_inputs(instance)
    else:
        validate_plain_tcn_inputs(instance)
    return instance


def build_lightgbm_inputs(canonical: CanonicalL2U2Inputs) -> LightGBMInputs:
    validate_canonical_inputs(canonical)
    n, context, _width = canonical.meteorology_values.shape
    value_names = tuple(
        f"{variable}_lag{lag}" for lag in range(context) for variable in ALLOWED_METEOROLOGY
    )
    observed_names = tuple(f"{name}_observed" for name in value_names)
    feature_names = (*value_names, *observed_names, "sin_doy", "cos_doy", "pooled_anchor")
    matrix = np.concatenate(
        (
            canonical.meteorology_values.reshape(n, -1),
            canonical.meteorology_observed.reshape(n, -1).astype(np.float32),
            canonical.season,
            canonical.pooled_anchor[:, None],
        ),
        axis=1,
    )
    result = _construct_architecture_inputs(
        LightGBMInputs,
        key_ids=canonical.key_ids,
        matrix=_immutable_array(matrix, np.float32),
        feature_names=feature_names,
        preprocessing_sha256=canonical.preprocessing_sha256,
        canonical_input_sha256=canonical.state_sha256,
    )
    assert isinstance(result, LightGBMInputs)
    return result


def build_plain_tcn_inputs(canonical: CanonicalL2U2Inputs) -> PlainTCNInputs:
    validate_canonical_inputs(canonical)
    static = np.concatenate((canonical.season, canonical.pooled_anchor[:, None]), axis=1)
    result = _construct_architecture_inputs(
        PlainTCNInputs,
        key_ids=canonical.key_ids,
        sequence=_immutable_array(canonical.meteorology_values, np.float32),
        sequence_observed=_immutable_array(canonical.meteorology_observed, np.bool_),
        static_context=_immutable_array(static, np.float32),
        channel_names=ALLOWED_METEOROLOGY,
        static_names=("sin_doy", "cos_doy", "pooled_anchor"),
        preprocessing_sha256=canonical.preprocessing_sha256,
        canonical_input_sha256=canonical.state_sha256,
    )
    assert isinstance(result, PlainTCNInputs)
    return result


def architecture_input_sha256(inputs: LightGBMInputs | PlainTCNInputs) -> str:
    """Revalidate and hash tensor bytes plus exact key/provenance metadata."""

    if type(inputs) is LightGBMInputs:
        validate_lightgbm_inputs(inputs)
    elif type(inputs) is PlainTCNInputs:
        validate_plain_tcn_inputs(inputs)
    else:
        raise ContractError("unknown architecture input type")
    return inputs.state_sha256


def architecture_tensor_sha256(inputs: LightGBMInputs | PlainTCNInputs) -> str:
    """Revalidate and hash only bytes permitted across the forward boundary."""

    if type(inputs) is LightGBMInputs:
        validate_lightgbm_inputs(inputs)
        tensors = (("matrix", inputs.matrix),)
    elif type(inputs) is PlainTCNInputs:
        validate_plain_tcn_inputs(inputs)
        tensors = (
            ("sequence", inputs.sequence),
            ("sequence_observed", inputs.sequence_observed),
            ("static_context", inputs.static_context),
        )
    else:
        raise ContractError("unknown architecture input type")
    hasher = hashlib.sha256()
    hasher.update(type(inputs).__name__.encode("ascii"))
    for label, value in tensors:
        _digest_array(hasher, label, value)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Label separation and exact common-key contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, init=False)
class LabelRegistry:
    """Bytes-bound supervision authority; never a predictor/model input."""

    keys: tuple[ForecastKey, ...]
    y_true: np.ndarray
    source_path: str
    source_file_sha256: str
    fold: int | None
    view_status: str
    execution_authorized: bool
    y_true_bytes_sha256: str
    state_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ContractError("LabelRegistry has no public constructor; load authority-bound bytes")

    @property
    def key_ids(self) -> tuple[str, ...]:
        return tuple(key.key_id for key in self.keys)

    @property
    def common_key_registry_sha256(self) -> str:
        return canonical_sha256([key.to_record() for key in self.keys])

    @property
    def label_registry_sha256(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.source_path.encode("utf-8"))
        hasher.update(self.source_file_sha256.encode("ascii"))
        hasher.update(
            _canonical_json_bytes(
                {
                    "fold": self.fold,
                    "view_status": self.view_status,
                    "execution_authorized": self.execution_authorized,
                }
            )
        )
        hasher.update(self.common_key_registry_sha256.encode("ascii"))
        _digest_array(hasher, "y_true", self.y_true)
        return hasher.hexdigest()


def validate_label_registry_structure(labels: LabelRegistry) -> None:
    if type(labels) is not LabelRegistry:
        raise ContractError("label registry is not the exact bytes-bound structural type")
    if not labels.keys or tuple(sorted(labels.keys)) != labels.keys:
        raise ContractError("label keys must be non-empty and canonically sorted")
    if len({key.key_id for key in labels.keys}) != len(labels.keys):
        raise ContractError("label registry contains duplicate keys")
    target = _validate_immutable_array(labels.y_true, label="label y_true")
    if target.dtype != np.float64 or target.shape != (len(labels.keys),):
        raise ContractError("label y_true must be immutable float64 aligned with exact keys")
    if not np.isfinite(target).all():
        raise ContractError("label target must be finite")
    if labels.source_path != _repository_relative(AUTHORITY_FILE_PATHS["label_registry"]):
        raise ContractError("label registry does not use the canonical authority path")
    if labels.execution_authorized is not False:
        raise ContractError("Phase-1 label views can never authorize execution")
    if labels.fold is None:
        if labels.view_status != UNSCOPED_LABEL_VIEW_STATUS:
            raise ContractError("unscoped/all-h7 label view has an invalid status")
    elif type(labels.fold) is not int or labels.fold not in FOLDS:
        raise ContractError("fold-scoped label view has a noncanonical fold")
    elif labels.view_status != FOLD_LABEL_VIEW_STATUS:
        raise ContractError("fold-scoped label view lacks authority-bound status")
    for label in ("source_file_sha256", "y_true_bytes_sha256", "state_sha256"):
        _validate_sha256(getattr(labels, label), label=f"label registry {label}")
    expected_y = hashlib.sha256(target.tobytes(order="C")).hexdigest()
    if labels.y_true_bytes_sha256 != expected_y:
        raise ContractError("label registry y_true digest does not bind exact float64 bytes")
    if labels.state_sha256 != labels.label_registry_sha256:
        raise ContractError("label registry state digest does not bind current key/y_true bytes")


def _construct_label_registry(
    *,
    keys: tuple[ForecastKey, ...],
    y_true: object,
    source_file_sha256: str,
    fold: int | None = None,
    view_status: str = UNSCOPED_LABEL_VIEW_STATUS,
) -> LabelRegistry:
    instance = object.__new__(LabelRegistry)
    target = _immutable_array(y_true, np.float64)
    object.__setattr__(instance, "keys", keys)
    object.__setattr__(instance, "y_true", target)
    object.__setattr__(
        instance,
        "source_path",
        _repository_relative(AUTHORITY_FILE_PATHS["label_registry"]),
    )
    object.__setattr__(instance, "source_file_sha256", source_file_sha256)
    object.__setattr__(instance, "fold", fold)
    object.__setattr__(instance, "view_status", view_status)
    object.__setattr__(instance, "execution_authorized", False)
    object.__setattr__(
        instance,
        "y_true_bytes_sha256",
        hashlib.sha256(target.tobytes(order="C")).hexdigest(),
    )
    object.__setattr__(instance, "state_sha256", instance.label_registry_sha256)
    validate_label_registry_structure(instance)
    return instance


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    """Future parser output; content still cannot authorize Phase-1 completion."""

    keys: tuple[ForecastKey, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    label_registry_file_sha256: str

    def __post_init__(self) -> None:
        target = _immutable_array(self.y_true, np.float64)
        prediction = _immutable_array(self.y_pred, np.float64)
        if tuple(sorted(self.keys)) != self.keys or len(set(self.keys)) != len(self.keys):
            raise ContractError("prediction keys must be unique canonical ForecastKey tuples")
        if target.shape != (len(self.keys),) or prediction.shape != target.shape:
            raise ContractError("prediction y_true/y_pred must align with exact keys")
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            raise ContractError("prediction content must be finite")
        _validate_sha256(
            self.label_registry_file_sha256,
            label="prediction label_registry_file_sha256",
        )
        object.__setattr__(self, "y_true", target)
        object.__setattr__(self, "y_pred", prediction)


def validate_model_label_alignment(
    inputs: LightGBMInputs | PlainTCNInputs, labels: LabelRegistry
) -> None:
    if type(inputs) is LightGBMInputs:
        validate_lightgbm_inputs(inputs)
    elif type(inputs) is PlainTCNInputs:
        validate_plain_tcn_inputs(inputs)
    else:
        raise ContractError("unknown architecture input type")
    validate_label_registry_structure(labels)
    if inputs.key_ids != labels.key_ids:
        raise ContractError("model inputs and immutable label registry have different exact keys")


def _validate_prediction_content_structure(
    prediction: PredictionBatch, labels: LabelRegistry
) -> str:
    """Structural proof used beneath the authority-required public boundary."""

    if type(prediction) is not PredictionBatch:
        raise ContractError("prediction must come from the future exact content parser")
    validate_label_registry_structure(labels)
    if tuple(key.key_id for key in prediction.keys) != labels.key_ids:
        raise ContractError("prediction and label authority have different exact keys")
    if prediction.label_registry_file_sha256 != labels.source_file_sha256:
        raise ContractError("prediction binds a different label authority file")
    if prediction.y_true.tobytes(order="C") != labels.y_true.tobytes(order="C"):
        raise ContractError(
            "prediction y_true is not an exact float64 byte copy of authority labels"
        )
    hasher = hashlib.sha256()
    hasher.update(labels.state_sha256.encode("ascii"))
    _digest_array(hasher, "y_true", prediction.y_true)
    _digest_array(hasher, "y_pred", prediction.y_pred)
    return hasher.hexdigest()


def validate_prediction_content(
    prediction: PredictionBatch,
    labels: LabelRegistry,
    authority: ExecutionAuthority,
    *,
    fold: int,
    evaluation_keys: tuple[ForecastKey, ...],
) -> str:
    """Require one authority, exact current label bytes, keys, and y_true copy."""

    validate_label_registry_authority(
        labels,
        authority,
        fold=fold,
        evaluation_keys=evaluation_keys,
    )
    return _validate_prediction_content_structure(prediction, labels)


# ---------------------------------------------------------------------------
# Provenance, artifact identity, resume, and completeness
# ---------------------------------------------------------------------------


REQUIRED_PROVENANCE_HASHES = (
    "authority_sha256",
    "protocol_sha256",
    "protocol_seal_sha256",
    "runner_sha256",
    "station_registry_file_sha256",
    "station_huc2_registry_sha256",
    "fold_registry_file_sha256",
    "fold_registry_sha256",
    "fit_plan_sha256",
    "input_schema_sha256",
    "preprocessing_policy_sha256",
    "common_key_registry_sha256",
    "label_registry_sha256",
    "predictor_registry_sha256",
    "training_meteorology_registry_sha256",
    "training_anchor_registry_sha256",
    "contrast_registry_sha256",
    "model_configuration_sha256",
    "environment_sha256",
)


def input_schema_record() -> dict[str, object]:
    return {
        "regime": {"F": FORCING, "L": LOCAL_LEVEL, "G": GEOMETRY, "lead_days": 7},
        "shared_allowed_history": list(ALLOWED_METEOROLOGY),
        "history_context_days": HISTORY_CONTEXT_DAYS,
        "history_geometry": "32 contiguous daily rows from issue_date-31 through issue_date",
        "raw_history_schema_is_exact_not_column_selected": True,
        "last_history_timestamp_equals_issue_time": True,
        "key_id": "sha256(temporal|known_site|site_id|issue_date|target_date|lead_days)",
        "site_id_decimal_lengths": list(VALID_SITE_ID_LENGTHS),
        "site_id_leading_zeros_preserved": True,
        "site_id_target_doy_and_season_derived_only_from_parsed_key_tuple": True,
        "season_formula": "sin/cos(2*pi*(target_doy-1)/366)",
        "prohibited_from_model": [
            "WTEMP values",
            "WTEMP observedness",
            "FLOW values",
            "FLOW observedness",
            "target-site identity",
            "target-site climatology or decay statistics",
            "target-site imputer/scaler state",
            "target-site anchor candidate",
            "labels or target observedness",
        ],
        "anchor": "unweighted training-station-pooled 2006-2015 WTEMP climatology at target doy",
        "point_target": "unbounded residual to the identical pooled anchor for both architectures",
        "lightgbm": "flattened shared meteorology values/masks + season + pooled anchor",
        "plain_TCN": "shared meteorology sequence/mask + season + pooled anchor",
        "key_ids_are_trainer_metadata_not_model_features": True,
        "station_ids_are_trainer_metadata_not_model_features": True,
        "labels_are_a_separate_immutable_registry": True,
    }


def preprocessing_policy_record() -> dict[str, object]:
    return {
        "period": TRAINING_PERIOD,
        "source_registry_format": TRAINING_SOURCE_FORMAT,
        "source_rows_are_dated_and_canonical_period_checked": True,
        "fit_station_membership": "exact fold.train_stations only",
        "source_membership_excludes_held_stations": True,
        "canonical_120_site_fold_record_hash_required": True,
        "variables_imputed_and_scaled": list(ALLOWED_METEOROLOGY),
        "imputer": "pooled nanmedian over training-station rows",
        "scaler": "pooled mean and population std after training-only imputation",
        "zero_scale_rule": "replace with 1.0",
        "anchor": "unweighted mean of finite 366-day training-station climatology curves",
        "per_site_state_serialized_to_model": False,
        "state_and_source_hash_recomputed_at_projection_boundary": True,
        "authority_file_hash_required_for_execution": True,
    }


def fixed_contract_hashes() -> dict[str, str]:
    return {
        "runner_sha256": hashlib.sha256(_read_stable_regular(Path(__file__).resolve())).hexdigest(),
        "fit_plan_sha256": fit_plan_sha256(build_fit_plan()),
        "input_schema_sha256": canonical_sha256(input_schema_record()),
        "preprocessing_policy_sha256": canonical_sha256(preprocessing_policy_record()),
    }


def validate_provenance_hashes(
    hashes: Mapping[str, str], authority: ExecutionAuthority | None = None
) -> dict[str, str]:
    """Reject self-attestation; only exact current authority bytes can validate."""

    if authority is None:
        raise Phase1ExecutionUnavailable(
            "arbitrary provenance mappings are not authority; a trusted pinned execution "
            "authority and current-byte verification are required"
        )
    return _validate_provenance_against_authority(hashes, authority)


def artifact_stem(cell: FitPlan) -> Path:
    architecture = "lightgbm" if cell.architecture is Architecture.LIGHTGBM else "plain_tcn"
    return Path(
        architecture,
        FORCING,
        LOCAL_LEVEL,
        GEOMETRY,
        f"h{LEAD_DAYS}",
        f"fold{cell.fold}",
        f"fit{cell.fit_seed}",
    )


def artifact_paths(cell: FitPlan) -> dict[str, str]:
    stem = artifact_stem(cell)
    checkpoint_suffix = ".txt" if cell.architecture is Architecture.LIGHTGBM else ".pt"
    return {
        "checkpoint": (Path("checkpoints") / stem).with_suffix(checkpoint_suffix).as_posix(),
        "prediction": (Path("predictions") / stem).with_suffix(".parquet").as_posix(),
        "manifest": (Path("manifests") / stem).with_suffix(".json").as_posix(),
    }


def planned_fit_manifest(
    _cell: FitPlan, _provenance: Mapping[str, str] | ExecutionAuthority
) -> dict[str, object]:
    raise Phase1ExecutionUnavailable(
        "Phase 1 cannot emit or accept fit manifests before the exact prediction parser, "
        "authority-bound label loader, trainer, and atomic writer are implemented"
    )


def validate_resume_manifest(
    _document: Mapping[str, object],
    _cell: FitPlan,
    _provenance: Mapping[str, str] | ExecutionAuthority,
    *,
    artifact_root: Path,
) -> None:
    _ = artifact_root
    raise Phase1CompletionUnavailable(
        f"{COMPLETION_UNAVAILABLE_STATUS}: resume is terminally unavailable in Phase 1; "
        "no manifest, checkpoint, or prediction bytes were trusted or read"
    )


def validate_completion_inventory_metadata(_manifests: Sequence[Mapping[str, object]]) -> None:
    """Never trust self-attested completion inventory in Phase 1."""

    raise Phase1CompletionUnavailable(
        f"{COMPLETION_UNAVAILABLE_STATUS}: completion is terminally unavailable before "
        "authority-bound parsing and byte validation of every one of the 40 fits"
    )


# ---------------------------------------------------------------------------
# Dry-run manifest and governance guard
# ---------------------------------------------------------------------------


PROTOCOL_CONTRACT_BINDING_KEYS = (
    "fit_plan_sha256",
    "station_huc2_registry_sha256",
    "fold_registry_sha256",
    "input_schema_sha256",
    "preprocessing_policy_sha256",
)


def execution_scope_record() -> dict[str, object]:
    return {
        "forcing": FORCING,
        "local_level": LOCAL_LEVEL,
        "geometry": GEOMETRY,
        "lead_days": LEAD_DAYS,
        "architectures": [architecture.value for architecture in Architecture],
        "folds": list(FOLDS),
        "model_fit_seeds": list(MODEL_FIT_SEEDS),
        "logical_cell_count": 2,
        "fit_unit_count": 40,
    }


def dry_run_manifest(*, protocol_path: Path, include_fits: bool = False) -> dict[str, object]:
    if Path(protocol_path).absolute() != DEFAULT_PROTOCOL.absolute():
        raise ContractError("dry-run accepts only the frozen canonical v4 draft path")
    protocol_payload, protocol = _read_frozen_draft()
    try:
        legacy_protocol_payload = _read_stable_regular(LEGACY_PROTOCOL)
        legacy_runner_payload = _read_stable_regular(LEGACY_RUNNER)
        legacy_protocol_text = legacy_protocol_payload.decode("utf-8")
        legacy_runner_text = legacy_runner_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("cannot read immutable legacy protocol/runner for audit") from exc
    findings = audit_v4_and_legacy(protocol, legacy_protocol_text, legacy_runner_text)
    plan = build_fit_plan()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "PHASE_1_CONTRACT_ONLY",
        "status": PHASE_STATUS,
        "execution_implemented": EXECUTION_IMPLEMENTED,
        "reads_2021_2023_panel_or_outcomes": False,
        "trains_or_scores_models": False,
        "writes_artifacts": False,
        "safe_dry_run_reads": [
            str(Path(protocol_path)),
            str(LEGACY_PROTOCOL),
            str(LEGACY_RUNNER),
            str(Path(__file__).resolve()),
        ],
        "forbidden_phase1_reads": [
            "data_usgs/panel_usgs_120v2.parquet",
            "outputs/conventional/panel_2021_2023.parquet",
            "outputs/final/forecast_keys.parquet",
            _repository_relative(SHARED_KEY_REGISTRY),
            "any prediction/checkpoint/outcome artifact",
        ],
        "protocol": {
            "path": str(DEFAULT_PROTOCOL),
            "sha256": hashlib.sha256(protocol_payload).hexdigest(),
            "status": protocol.get("status"),
            "execution_authorized": protocol.get("execution_authorized"),
            "seal_path": protocol.get("seal_path"),
        },
        "static_audit": [finding.to_record() for finding in findings],
        "scope": {
            "F": FORCING,
            "F_status": "PHASE1_CODE_ASSUMPTION_REQUIRES_PINNED_DERIVED_PROTOCOL",
            "L": LOCAL_LEVEL,
            "G": GEOMETRY,
            "A": [architecture.value for architecture in Architecture],
            "lead_days": LEAD_DAYS,
        },
        "inventory": plan_arithmetic(plan),
        "fold_registry": {
            "algorithm": (
                "sort intact HUC2 groups by (-station_count,HUC2); assign each to the "
                "lowest-load fold with lowest-index tie break"
            ),
            "station_count": EXPECTED_STATION_COUNT,
            "station_huc2_sha256": EXPECTED_STATION_HUC2_SHA256,
            "fold_registry_sha256": EXPECTED_FOLD_REGISTRY_SHA256,
            "hold_sizes": list(EXPECTED_HOLD_SIZES),
            "fold_record_sha256": dict(EXPECTED_FOLD_RECORD_SHA256),
            "station_membership_read_in_dry_run": False,
        },
        "model_input_contract": input_schema_record(),
        "preprocessing_contract": preprocessing_policy_record(),
        "common_key_contract": {
            "fields": ["site_id", "issue_time", "target_time", "lead_days"],
            "lead_days": LEAD_DAYS,
            "two_sided_exact_equality": True,
            "same_keys_across_architectures_and_fit_seeds_within_fold": True,
            "shared_key_and_y_source": _repository_relative(SHARED_KEY_REGISTRY),
            "label_view": (
                "authority-fold held-station lead_days=7 view of the identical shared file "
                "bytes; exact two-sided equality with evaluation predictor keys required"
            ),
            "all_h7_label_view_execution_ready": False,
            "independent_key_or_label_authority_allowed": False,
            "prediction_rows_copy_registry_y_true": True,
            "no_key_declining": True,
            "shared_registry_arrow_schema": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in SHARED_PRIMARY_REPORTABLE_ARROW_SCHEMA
            ],
            "physical_schema_validated_before_pandas_conversion": True,
            "permissive_pandas_dtype_coercion_allowed": False,
        },
        "artifact_contract": {
            "format": MANIFEST_FORMAT,
            "required_provenance_hashes": list(REQUIRED_PROVENANCE_HASHES),
            "resume": "TERMINALLY_UNAVAILABLE_IN_PHASE1",
            "all_40_fits_require_one_identical_authority_sha256": True,
            "partial_matrix_publishable": False,
            "completion_status": COMPLETION_UNAVAILABLE_STATUS,
        },
        "future_execution_authority": {
            "canonical_derived_protocol_path": str(SEALED_PROTOCOL),
            "canonical_seal_path": str(SEALED_PROTOCOL_SEAL),
            "derived_protocol_sha256_pinned": EXPECTED_SEALED_PROTOCOL_SHA256,
            "self_signed_protocol_and_seal_sufficient": False,
            "predictor_bound_byte_loader_authorized": PREDICTOR_BOUND_BYTE_LOADER_AUTHORIZED,
            "preprocessing_bound_byte_loaders_authorized": (
                PREPROCESSING_BOUND_BYTE_LOADERS_AUTHORIZED
            ),
            "structural_inputs_can_authorize_execution": (
                STRUCTURAL_INPUTS_CAN_AUTHORIZE_EXECUTION
            ),
            "canonical_file_paths": {
                key: _repository_relative(path) for key, path in AUTHORITY_FILE_PATHS.items()
            },
        },
        "hashes": {
            **fixed_contract_hashes(),
            "legacy_protocol_sha256": hashlib.sha256(legacy_protocol_payload).hexdigest(),
            "legacy_runner_sha256": hashlib.sha256(legacy_runner_payload).hexdigest(),
        },
        "unimplemented_execution_modules": [
            "pinned derived protocol bytes",
            "seal-bound predictor byte loader that rematerializes exact 32-day inputs",
            "seal-bound dated preprocessing byte loaders that rematerialize exact sources",
            "frozen LightGBM model configuration and trainer",
            "frozen plain_TCN model configuration and trainer",
            "exact prediction Parquet parser",
            "model-seed aggregation and station-first estimator",
            "atomic artifact and completion writer",
        ],
    }
    if include_fits:
        manifest["fit_units"] = [cell.to_record() for cell in plan]
    return manifest


def _exact_relative_path(value: object, expected: Path, *, label: str) -> str:
    expected_relative = _repository_relative(expected)
    if not isinstance(value, str) or value != expected_relative:
        raise GovernanceError(f"{label} must equal canonical path {expected_relative!r}")
    if Path(value).is_absolute() or ".." in Path(value).parts or "." in Path(value).parts:
        raise GovernanceError(f"{label} is not a canonical repository-relative path")
    return value


def _strict_json_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GovernanceError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernanceError(f"{label} is not strict JSON") from exc
    if not isinstance(document, Mapping):
        raise GovernanceError(f"{label} root must be an object")
    if _canonical_json_bytes(document) != payload:
        raise GovernanceError(f"{label} must use exact canonical JSON bytes")
    return document


@dataclass(frozen=True, slots=True)
class AuthorityFileBinding:
    name: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.name not in AUTHORITY_FILE_PATHS:
            raise GovernanceError(f"unknown execution authority file key: {self.name}")
        _exact_relative_path(self.path, AUTHORITY_FILE_PATHS[self.name], label=self.name)
        try:
            _validate_sha256(self.sha256, label=f"authority file {self.name}")
        except ContractError as exc:
            raise GovernanceError(str(exc)) from exc


@dataclass(frozen=True, slots=True, init=False)
class ExecutionAuthority:
    """Pinned derived-protocol authority; never constructible from attestations."""

    protocol_path: Path
    seal_path: Path
    protocol_payload: bytes
    seal_payload: bytes
    protocol_sha256: str
    seal_sha256: str
    contract_hashes: Mapping[str, str]
    file_bindings: Mapping[str, AuthorityFileBinding]
    runner_sha256: str
    authority_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise GovernanceError(
            "ExecutionAuthority has no public constructor; validate pinned canonical bytes"
        )


def _expected_contract_hashes() -> dict[str, str]:
    fixed = fixed_contract_hashes()
    return {
        "fit_plan_sha256": fixed["fit_plan_sha256"],
        "station_huc2_registry_sha256": EXPECTED_STATION_HUC2_SHA256,
        "fold_registry_sha256": EXPECTED_FOLD_REGISTRY_SHA256,
        "input_schema_sha256": fixed["input_schema_sha256"],
        "preprocessing_policy_sha256": fixed["preprocessing_policy_sha256"],
    }


def _parse_protocol_bindings(
    bindings: object,
) -> tuple[dict[str, str], dict[str, AuthorityFileBinding]]:
    if not isinstance(bindings, Mapping) or set(bindings) != {"contract_hashes", "files"}:
        raise GovernanceError("derived protocol has wrong L2_U2 execution-binding schema")
    contract = bindings.get("contract_hashes")
    files = bindings.get("files")
    if not isinstance(contract, Mapping) or set(contract) != set(PROTOCOL_CONTRACT_BINDING_KEYS):
        raise GovernanceError("derived protocol contract-hash schema changed")
    validated_contract: dict[str, str] = {}
    for key in PROTOCOL_CONTRACT_BINDING_KEYS:
        try:
            validated_contract[key] = _validate_sha256(contract[key], label=key)
        except ContractError as exc:
            raise GovernanceError(str(exc)) from exc
    if validated_contract != _expected_contract_hashes():
        raise GovernanceError("derived protocol does not bind the exact Phase-1 contracts")
    if not isinstance(files, Mapping) or set(files) != set(AUTHORITY_FILE_PATHS):
        raise GovernanceError("derived protocol file-binding inventory changed")
    validated_files: dict[str, AuthorityFileBinding] = {}
    for name, value in files.items():
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise GovernanceError(f"authority file binding {name} has wrong schema")
        validated_files[name] = AuthorityFileBinding(
            name=name,
            path=value["path"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
        )
    common = validated_files["common_key_registry"]
    labels = validated_files["label_registry"]
    if common.path != labels.path or common.sha256 != labels.sha256:
        raise GovernanceError(
            "common-key and label views must bind the identical shared primary-reportable file"
        )
    return validated_contract, validated_files


def _authority_digest(authority: ExecutionAuthority) -> str:
    return canonical_sha256(
        {
            "protocol_path": _repository_relative(authority.protocol_path),
            "seal_path": _repository_relative(authority.seal_path),
            "protocol_sha256": authority.protocol_sha256,
            "seal_sha256": authority.seal_sha256,
            "runner_sha256": authority.runner_sha256,
            "scope": execution_scope_record(),
            "contract_hashes": dict(authority.contract_hashes),
            "files": {
                name: {"path": binding.path, "sha256": binding.sha256}
                for name, binding in sorted(authority.file_bindings.items())
            },
        }
    )


def _construct_execution_authority(**values: object) -> ExecutionAuthority:
    authority = object.__new__(ExecutionAuthority)
    for name, value in values.items():
        object.__setattr__(authority, name, value)
    object.__setattr__(authority, "authority_sha256", _authority_digest(authority))
    validate_execution_authority_object(authority, reread_governance_bytes=False)
    return authority


def validate_execution_authority_object(
    authority: ExecutionAuthority, *, reread_governance_bytes: bool = True
) -> None:
    if type(authority) is not ExecutionAuthority:
        raise GovernanceError("execution authority is not the exact trusted structural type")
    if authority.protocol_path != SEALED_PROTOCOL or authority.seal_path != SEALED_PROTOCOL_SEAL:
        raise GovernanceError("execution authority paths are not canonical")
    if EXPECTED_SEALED_PROTOCOL_SHA256 is None:
        raise GovernanceError("Phase 1 has no pre-pinned derived protocol SHA-256")
    if hashlib.sha256(authority.protocol_payload).hexdigest() != authority.protocol_sha256:
        raise GovernanceError("authority protocol payload/hash drift")
    if authority.protocol_sha256 != EXPECTED_SEALED_PROTOCOL_SHA256:
        raise GovernanceError("authority protocol is not the pre-pinned derived protocol")
    if hashlib.sha256(authority.seal_payload).hexdigest() != authority.seal_sha256:
        raise GovernanceError("authority seal payload/hash drift")
    if dict(authority.contract_hashes) != _expected_contract_hashes():
        raise GovernanceError("authority contract hashes drifted")
    if set(authority.file_bindings) != set(AUTHORITY_FILE_PATHS):
        raise GovernanceError("authority file inventory drifted")
    for name, binding in authority.file_bindings.items():
        AuthorityFileBinding(name=name, path=binding.path, sha256=binding.sha256)
    common = authority.file_bindings["common_key_registry"]
    labels = authority.file_bindings["label_registry"]
    if common.path != labels.path or common.sha256 != labels.sha256:
        raise GovernanceError("authority split the shared common-key/label source identity")
    current_runner = fixed_contract_hashes()["runner_sha256"]
    if authority.runner_sha256 != current_runner:
        raise GovernanceError("authority does not bind the current runner bytes")
    if authority.authority_sha256 != _authority_digest(authority):
        raise GovernanceError("authority digest does not bind current structural state")
    if reread_governance_bytes:
        if (
            _read_stable_regular(authority.protocol_path, error_type=GovernanceError)
            != authority.protocol_payload
        ):
            raise GovernanceError("canonical protocol bytes changed after authority validation")
        if (
            _read_stable_regular(authority.seal_path, error_type=GovernanceError)
            != authority.seal_payload
        ):
            raise GovernanceError("canonical seal bytes changed after authority validation")


def validate_execution_authority(protocol_path: Path) -> ExecutionAuthority:
    """Validate only pinned canonical governance bytes; never touch data registries."""

    requested = Path(protocol_path).absolute()
    if requested != SEALED_PROTOCOL.absolute():
        raise GovernanceError(
            "L2_U2 execution blocked before station/panel/outcome access: only the fixed "
            f"derived protocol path {SEALED_PROTOCOL} can be considered"
        )
    if EXPECTED_SEALED_PROTOCOL_SHA256 is None:
        raise GovernanceError(
            "L2_U2 execution blocked before station/panel/outcome access: Phase 1 has no "
            "pre-pinned SHA-256 for the future canonical derived protocol"
        )
    protocol_payload = _read_stable_regular(SEALED_PROTOCOL, error_type=GovernanceError)
    protocol_sha256 = hashlib.sha256(protocol_payload).hexdigest()
    if protocol_sha256 != EXPECTED_SEALED_PROTOCOL_SHA256:
        raise GovernanceError(
            "canonical derived protocol bytes do not match the pre-pinned SHA-256"
        )
    try:
        protocol = yaml.safe_load(protocol_payload)
    except yaml.YAMLError as exc:
        raise GovernanceError("canonical derived protocol is not valid YAML") from exc
    if not isinstance(protocol, Mapping):
        raise GovernanceError("canonical derived protocol root must be a mapping")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID or protocol.get("version") != 4:
        raise GovernanceError("canonical derived protocol identity mismatch")
    if (
        protocol.get("status") != AUTHORIZED_PROTOCOL_STATUS
        or protocol.get("execution_authorized") is not True
    ):
        raise GovernanceError("canonical derived protocol does not authorize execution")
    derived = protocol.get("derived_from_draft")
    expected_derived = {
        "path": _repository_relative(DEFAULT_PROTOCOL),
        "sha256": EXPECTED_DRAFT_PROTOCOL_SHA256,
    }
    if derived != expected_derived:
        raise GovernanceError("derived protocol is not byte-bound to the immutable v4 draft")
    _exact_relative_path(protocol.get("seal_path"), SEALED_PROTOCOL_SEAL, label="seal_path")
    if protocol.get("l2_u2_execution_scope") != execution_scope_record():
        raise GovernanceError("derived protocol does not explicitly freeze the F0 L2_U2 scope")
    contract_hashes, file_bindings = _parse_protocol_bindings(
        protocol.get("l2_u2_execution_bindings")
    )

    seal_payload = _read_stable_regular(SEALED_PROTOCOL_SEAL, error_type=GovernanceError)
    seal = _strict_json_mapping(seal_payload, label="canonical derived-protocol seal")
    expected_seal_fields = {
        "format",
        "status",
        "execution_authorized",
        "protocol",
        "runner_sha256",
        "l2_u2_execution_scope",
        "l2_u2_execution_bindings",
    }
    if set(seal) != expected_seal_fields:
        raise GovernanceError("canonical seal has unexpected or missing fields")
    if seal.get("format") != SEAL_FORMAT or seal.get("status") != "SEALED":
        raise GovernanceError("canonical seal format/status mismatch")
    if seal.get("execution_authorized") is not True:
        raise GovernanceError("canonical seal does not authorize execution")
    if seal.get("protocol") != {
        "path": _repository_relative(SEALED_PROTOCOL),
        "sha256": protocol_sha256,
    }:
        raise GovernanceError("canonical seal does not bind exact derived-protocol bytes")
    if seal.get("l2_u2_execution_scope") != execution_scope_record():
        raise GovernanceError("canonical seal scope differs from the pinned protocol")
    if seal.get("l2_u2_execution_bindings") != protocol.get("l2_u2_execution_bindings"):
        raise GovernanceError("canonical seal bindings differ from the pinned protocol")
    runner_sha256 = fixed_contract_hashes()["runner_sha256"]
    if seal.get("runner_sha256") != runner_sha256:
        raise GovernanceError("canonical seal does not bind current runner bytes")
    return _construct_execution_authority(
        protocol_path=SEALED_PROTOCOL,
        seal_path=SEALED_PROTOCOL_SEAL,
        protocol_payload=bytes(protocol_payload),
        seal_payload=bytes(seal_payload),
        protocol_sha256=protocol_sha256,
        seal_sha256=hashlib.sha256(seal_payload).hexdigest(),
        contract_hashes=MappingProxyType(contract_hashes),
        file_bindings=MappingProxyType(file_bindings),
        runner_sha256=runner_sha256,
    )


def verify_authority_file_bindings(authority: ExecutionAuthority) -> dict[str, str]:
    """Rehash every exact bound source from its canonical current bytes."""

    validate_execution_authority_object(authority)
    observed: dict[str, str] = {}
    for name, binding in authority.file_bindings.items():
        payload = _read_stable_regular(AUTHORITY_FILE_PATHS[name])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding.sha256:
            raise ContractError(f"current bytes for authority file {name} do not match its binding")
        observed[name] = digest
    return observed


def provenance_snapshot(authority: ExecutionAuthority) -> dict[str, str]:
    """Build provenance only from one authority plus freshly rehashed exact files."""

    observed = verify_authority_file_bindings(authority)
    fixed = fixed_contract_hashes()
    return {
        "authority_sha256": authority.authority_sha256,
        "protocol_sha256": authority.protocol_sha256,
        "protocol_seal_sha256": authority.seal_sha256,
        "runner_sha256": fixed["runner_sha256"],
        "station_registry_file_sha256": observed["station_registry"],
        "station_huc2_registry_sha256": EXPECTED_STATION_HUC2_SHA256,
        "fold_registry_file_sha256": observed["fold_registry"],
        "fold_registry_sha256": EXPECTED_FOLD_REGISTRY_SHA256,
        "fit_plan_sha256": fixed["fit_plan_sha256"],
        "input_schema_sha256": fixed["input_schema_sha256"],
        "preprocessing_policy_sha256": fixed["preprocessing_policy_sha256"],
        "common_key_registry_sha256": observed["common_key_registry"],
        "label_registry_sha256": observed["label_registry"],
        "predictor_registry_sha256": observed["predictor_registry"],
        "training_meteorology_registry_sha256": observed["training_meteorology_registry"],
        "training_anchor_registry_sha256": observed["training_anchor_registry"],
        "contrast_registry_sha256": observed["contrast_registry"],
        "model_configuration_sha256": observed["model_configuration"],
        "environment_sha256": observed["environment"],
    }


def validate_uniform_fit_authorities(
    assignments: Sequence[tuple[FitPlan, ExecutionAuthority]],
) -> str:
    """Require the exact 40 fits to share one and only one authority object state."""

    expected = {cell.fit_id for cell in build_fit_plan()}
    if len(assignments) != len(expected):
        raise ContractError("authority assignment must cover exactly 40 fit units")
    observed: set[str] = set()
    authority_digests: set[str] = set()
    for cell, authority in assignments:
        if type(cell) is not FitPlan or cell.fit_id in observed:
            raise ContractError("authority assignment has an invalid or duplicate fit unit")
        observed.add(cell.fit_id)
        validate_execution_authority_object(authority)
        authority_digests.add(authority.authority_sha256)
    if observed != expected:
        raise ContractError("authority assignment does not match the exact frozen fit plan")
    if len(authority_digests) != 1:
        raise ContractError("all 40 fits must share one identical execution authority")
    return next(iter(authority_digests))


def _validate_provenance_against_authority(
    hashes: Mapping[str, str], authority: ExecutionAuthority
) -> dict[str, str]:
    if set(hashes) != set(REQUIRED_PROVENANCE_HASHES):
        missing = sorted(set(REQUIRED_PROVENANCE_HASHES) - set(hashes))
        extra = sorted(set(hashes) - set(REQUIRED_PROVENANCE_HASHES))
        raise ContractError(f"provenance schema mismatch: missing={missing}, extra={extra}")
    validated = {
        key: _validate_sha256(hashes[key], label=key) for key in REQUIRED_PROVENANCE_HASHES
    }
    expected = provenance_snapshot(authority)
    if validated != expected:
        drift = sorted(key for key in expected if validated.get(key) != expected[key])
        raise ContractError(f"provenance differs from exact current authority bytes: {drift}")
    return validated


def load_authority_fold_registry(
    authority: ExecutionAuthority,
) -> tuple[FoldDefinition, ...]:
    """Load the exact frozen fold records from their authority-bound bytes."""

    validate_execution_authority_object(authority)
    binding = authority.file_bindings["fold_registry"]
    payload = _read_stable_regular(AUTHORITY_FILE_PATHS["fold_registry"])
    if hashlib.sha256(payload).hexdigest() != binding.sha256:
        raise ContractError("current fold-registry bytes differ from the authority binding")
    document = _strict_json_mapping(payload, label="canonical fold registry")
    if set(document) != {
        "format",
        "station_huc2_registry_sha256",
        "fold_registry_sha256",
        "folds",
    }:
        raise ContractError("canonical fold registry has a noncanonical schema")
    if document.get("format") != FOLD_REGISTRY_FORMAT:
        raise ContractError("canonical fold registry format changed")
    if document.get("station_huc2_registry_sha256") != EXPECTED_STATION_HUC2_SHA256:
        raise ContractError("fold registry binds a different station/HUC2 authority")
    if document.get("fold_registry_sha256") != EXPECTED_FOLD_REGISTRY_SHA256:
        raise ContractError("fold registry declares a different frozen registry hash")
    records = document.get("folds")
    if not isinstance(records, list) or len(records) != len(FOLDS):
        raise ContractError("canonical fold registry must contain exactly four records")
    folds: list[FoldDefinition] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "fold",
            "geometry",
            "train_stations",
            "held_stations",
            "held_huc2",
        }:
            raise ContractError("canonical fold registry record schema changed")
        if record.get("geometry") != GEOMETRY:
            raise ContractError("canonical fold registry geometry changed")
        fold = FoldDefinition(
            fold=record["fold"],  # type: ignore[arg-type]
            train_stations=tuple(record["train_stations"]),  # type: ignore[arg-type]
            held_stations=tuple(record["held_stations"]),  # type: ignore[arg-type]
            held_huc2=tuple(record["held_huc2"]),  # type: ignore[arg-type]
        )
        validate_canonical_fold_definition(fold)
        folds.append(fold)
    result = tuple(folds)
    if tuple(fold.fold for fold in result) != FOLDS:
        raise ContractError("canonical fold registry records are not ordered 0..3")
    if canonical_sha256([fold.to_record() for fold in result]) != EXPECTED_FOLD_REGISTRY_SHA256:
        raise ContractError("canonical fold registry bytes do not reproduce the frozen hash")
    return result


def validate_fold_label_evaluation_boundary(
    labels: LabelRegistry,
    fold: FoldDefinition,
    evaluation_keys: tuple[ForecastKey, ...],
) -> None:
    """Prove exact two-sided fold-held key equality without key declining."""

    validate_label_registry_structure(labels)
    validate_canonical_fold_definition(fold)
    if labels.fold != fold.fold or labels.view_status != FOLD_LABEL_VIEW_STATUS:
        raise ContractError("label view is not scoped to the exact authority fold")
    if not evaluation_keys or any(type(key) is not ForecastKey for key in evaluation_keys):
        raise ContractError("evaluation keys must be parsed ForecastKey tuples")
    if tuple(sorted(evaluation_keys)) != evaluation_keys or len(set(evaluation_keys)) != len(
        evaluation_keys
    ):
        raise ContractError("evaluation keys must be unique and canonically sorted")
    outside = sorted({key.site_id for key in evaluation_keys} - set(fold.held_stations))
    if outside:
        raise ContractError(
            f"evaluation keys contain sites outside fold-held membership: {outside}"
        )
    if labels.keys != evaluation_keys:
        missing_from_predictors = sorted(set(labels.keys) - set(evaluation_keys))
        missing_from_labels = sorted(set(evaluation_keys) - set(labels.keys))
        raise ContractError(
            "fold label/predictor keys are not two-sided exact: "
            f"missing_from_predictors={len(missing_from_predictors)}, "
            f"missing_from_labels={len(missing_from_labels)}"
        )


def _normalize_shared_primary_reportable_frame(
    frame: pd.DataFrame,
) -> tuple[dict[str, object], ...]:
    """Validate already schema-proven pandas values without dtype coercion."""

    if list(frame.columns) != list(SHARED_PRIMARY_REPORTABLE_COLUMNS):
        raise ContractError("shared primary-reportable registry has a noncanonical schema")
    for column, expected_dtype in SHARED_PRIMARY_REPORTABLE_PANDAS_DTYPES.items():
        observed_dtype = frame[column].dtype
        if observed_dtype != expected_dtype:
            raise ContractError(
                "shared primary-reportable registry does not have the exact pandas dtype "
                f"for {column}: {observed_dtype} != {expected_dtype}; coercion is forbidden"
            )
    if frame.empty or frame.isna().any().any():
        raise ContractError("shared primary-reportable registry is empty or contains nulls")
    if any(type(value) is not str for value in frame["key_id"]):
        raise ContractError("shared primary-reportable key_id values must be exact strings")
    if any(type(value) is not str for value in frame["site_id"]):
        raise ContractError("shared primary-reportable site_id values must be exact strings")
    if set(frame["lead_days"].tolist()) != {1, 3, 7}:
        raise ContractError("shared primary-reportable registry must contain exact leads 1/3/7")
    if not frame["reportable_primary"].all():
        raise ContractError("shared primary-reportable registry contains a nonreportable row")
    if frame["key_id"].duplicated().any():
        raise ContractError("shared primary-reportable registry contains duplicate key IDs")
    normalized: list[dict[str, object]] = []
    for row_index in range(len(frame)):
        key_id = frame["key_id"].iat[row_index]
        site = _canonical_station_id(frame["site_id"].iat[row_index])
        issue = frame["issue_date"].iat[row_index]
        target = frame["target_date"].iat[row_index]
        if type(issue) is not pd.Timestamp or type(target) is not pd.Timestamp:
            raise ContractError("shared registry dates must be exact pandas timestamps")
        if issue.tz is not None or target.tz is not None:
            raise ContractError("shared registry dates must be timezone-naive")
        if issue != issue.normalize() or target != target.normalize():
            raise ContractError("shared registry dates must be midnight daily timestamps")
        lead_scalar = frame["lead_days"].iat[row_index]
        if type(lead_scalar) is not np.int16:
            raise ContractError("shared registry lead_days values must be exact int16 scalars")
        lead = lead_scalar.item()
        if target - issue != pd.Timedelta(days=lead):
            raise ContractError("shared registry target_date does not equal issue_date + lead")
        issue_text = issue.strftime("%Y-%m-%d")
        target_text = target.strftime("%Y-%m-%d")
        expected_key_id = hashlib.sha256(
            f"temporal|known_site|{site}|{issue_text}|{target_text}|{lead}".encode()
        ).hexdigest()
        if key_id != expected_key_id:
            raise ContractError("shared registry key_id does not hash its canonical tuple")
        target_scalar = frame["y_true"].iat[row_index]
        if type(target_scalar) is not np.float64:
            raise ContractError("shared registry y_true values must be exact float64 scalars")
        target_value = target_scalar.item()
        if not np.isfinite(target_value):
            raise ContractError("shared registry y_true contains a nonfinite value")
        normalized.append(
            {
                "key_id": expected_key_id,
                "site_id": site,
                "issue_time": issue_text,
                "target_time": target_text,
                "lead_days": lead,
                "y_true": target_value,
            }
        )
    return tuple(normalized)


def _read_shared_primary_reportable_parquet(payload: bytes) -> pd.DataFrame:
    """Prove the exact flat Arrow/Parquet schema before pandas conversion."""

    if type(payload) is not bytes or not payload:
        raise ContractError("authority label registry must be nonempty exact Parquet bytes")
    try:
        parquet_file = pq.ParquetFile(pa.BufferReader(payload))
        arrow_schema = parquet_file.schema_arrow
        parquet_schema = parquet_file.schema
    except Exception as exc:
        raise ContractError("authority label registry is not readable Parquet") from exc

    physical_columns = tuple(
        parquet_schema.column(index) for index in range(len(parquet_schema.names))
    )
    physical_types = tuple(column.physical_type for column in physical_columns)
    is_exact_flat_required_schema = (
        tuple(parquet_schema.names) == SHARED_PRIMARY_REPORTABLE_COLUMNS
        and physical_types == SHARED_PRIMARY_REPORTABLE_PARQUET_PHYSICAL_TYPES
        and all(column.max_definition_level == 0 for column in physical_columns)
        and all(column.max_repetition_level == 0 for column in physical_columns)
    )
    if not is_exact_flat_required_schema or not arrow_schema.equals(
        SHARED_PRIMARY_REPORTABLE_ARROW_SCHEMA,
        check_metadata=False,
    ):
        raise ContractError(
            "shared primary-reportable registry has a noncanonical physical Arrow/Parquet schema"
        )

    try:
        table = parquet_file.read(use_threads=False)
    except Exception as exc:
        raise ContractError("authority label registry Parquet rows are not readable") from exc
    if not table.schema.equals(
        SHARED_PRIMARY_REPORTABLE_ARROW_SCHEMA,
        check_metadata=False,
    ):
        raise ContractError(
            "shared primary-reportable registry read changed its proven physical "
            "Arrow/Parquet schema"
        )
    if table.num_rows == 0 or any(column.null_count for column in table.columns):
        raise ContractError("shared primary-reportable registry is empty or contains nulls")
    try:
        return table.to_pandas()
    except Exception as exc:
        raise ContractError("authority label registry cannot convert exactly to pandas") from exc


def load_label_registry(
    authority: ExecutionAuthority,
    *,
    fold: int,
    evaluation_keys: tuple[ForecastKey, ...],
) -> LabelRegistry:
    """Build an exact authority-fold h7 view, never an all-h7 execution view."""

    validate_execution_authority_object(authority)
    if type(fold) is not int or fold not in FOLDS:
        raise ContractError("label view fold must be one of the four canonical folds")
    fold_definition = load_authority_fold_registry(authority)[fold]
    binding = authority.file_bindings["label_registry"]
    payload = _read_stable_regular(AUTHORITY_FILE_PATHS["label_registry"])
    if hashlib.sha256(payload).hexdigest() != binding.sha256:
        raise ContractError("current label-registry bytes differ from the authority binding")
    frame = _read_shared_primary_reportable_parquet(payload)
    normalized = _normalize_shared_primary_reportable_frame(frame)

    keys: list[ForecastKey] = []
    targets: list[float] = []
    for record in normalized:
        if (
            record["lead_days"] != LEAD_DAYS
            or record["site_id"] not in fold_definition.held_stations
        ):
            continue
        key = ForecastKey.from_record(
            {
                name: record[name]
                for name in ("key_id", "site_id", "issue_time", "target_time", "lead_days")
            }
        )
        target_value = record["y_true"]
        if type(target_value) is not float:
            raise ContractError("normalized shared-registry y_true must remain exact float64")
        keys.append(key)
        targets.append(target_value)
    if not keys or tuple(sorted(keys)) != tuple(keys):
        raise ContractError("authority-fold shared-registry h7 view is empty or unsorted")
    labels = _construct_label_registry(
        keys=tuple(keys),
        y_true=targets,
        source_file_sha256=binding.sha256,
        fold=fold,
        view_status=FOLD_LABEL_VIEW_STATUS,
    )
    validate_fold_label_evaluation_boundary(labels, fold_definition, evaluation_keys)
    return labels


def validate_label_registry_authority(
    labels: LabelRegistry,
    authority: ExecutionAuthority,
    *,
    fold: int,
    evaluation_keys: tuple[ForecastKey, ...],
) -> None:
    validate_label_registry_structure(labels)
    validate_execution_authority_object(authority)
    binding = authority.file_bindings["label_registry"]
    if labels.source_path != binding.path or labels.source_file_sha256 != binding.sha256:
        raise ContractError("label registry path/hash differ from the single execution authority")
    reparsed = load_label_registry(authority, fold=fold, evaluation_keys=evaluation_keys)
    if (
        labels.keys != reparsed.keys
        or labels.y_true.tobytes(order="C") != reparsed.y_true.tobytes(order="C")
        or labels.state_sha256 != reparsed.state_sha256
    ):
        raise ContractError("label registry object differs from freshly reparsed authority bytes")


def authorize_structural_inputs_for_execution(
    _raw: RawPredictorBatch,
    _preprocessor: TrainingOnlyPreprocessor,
    _sources: PreprocessingSources,
    _authority: object,
) -> None:
    """Terminally deny in-memory objects until bound-byte loaders rematerialize them."""

    raise Phase1ExecutionUnavailable(
        "internally self-consistent predictor/preprocessor objects are structural tests only; "
        "predictor and preprocessing bound-byte loader authorization is false"
    )


def authorize_label_view_for_execution(labels: LabelRegistry) -> None:
    """Reject all-h7 and even fold-scoped views while predictor loading is absent."""

    validate_label_registry_structure(labels)
    if labels.fold is None or labels.view_status == UNSCOPED_LABEL_VIEW_STATUS:
        raise Phase1ExecutionUnavailable(
            "an unscoped/all-h7 label view cannot be treated as execution-ready"
        )
    raise Phase1ExecutionUnavailable(
        "fold-scoped labels remain non-executable until the authority-bound predictor "
        "loader rematerializes the exact two-sided evaluation key set"
    )


def execute_training(authority: ExecutionAuthority) -> int:
    """Terminal guard: no loader, model, scorer, or writer exists in Phase 1."""

    validate_execution_authority_object(authority)
    raise Phase1ExecutionUnavailable(
        "L2_U2 Phase 1 contains contracts only; execution code is incomplete. "
        "Exact bound-byte predictor/preprocessing loaders are unauthorized and absent; "
        "no station metadata, panel, outcome, model, prediction, or output path was accessed."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the safe in-memory Phase-1 audit/plan; never train or write",
    )
    parser.add_argument(
        "--print-fits",
        action="store_true",
        help="include all 40 typed fit units in dry-run JSON",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.print_fits and not args.dry_run:
        raise ContractError("--print-fits is allowed only with --dry-run")
    if args.dry_run:
        print(
            json.dumps(
                dry_run_manifest(
                    protocol_path=Path(args.protocol),
                    include_fits=bool(args.print_fits),
                ),
                indent=1,
            )
        )
        return 0

    # Governance is the first external read in non-dry mode.  On the current
    # draft it fails here, before station metadata and every panel/outcome path.
    authority = validate_execution_authority(Path(args.protocol))
    return execute_training(authority)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (
        ContractError,
        GovernanceError,
        Phase1ExecutionUnavailable,
        Phase1CompletionUnavailable,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
