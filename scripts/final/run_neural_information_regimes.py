#!/usr/bin/env python3
"""Phase-1 contract scaffold for the v4 neural information-regime study.

This module deliberately does **not** train or score a model and does not claim
seal readiness.  It defines the typed experiment inventory, spatial-fold rules,
canonical future-feature
schema, L0/L2 input boundary, point-only optimisation policy, and artifact
identity needed before the still-unrun neural hard-regime extension can be
sealed.

Safety boundary
---------------

``--dry-run`` constructs the plan entirely in memory, prints its deterministic
manifest, and reads/writes no files.  Any non-dry execution first validates an
exact v4 protocol/seal byte binding.  Phase 1 then fails closed before loading a
panel, constructing an outcome registry, training, scoring, or writing an
artifact.  A later execution phase must replace :func:`execute_training` only
after its missing modules and tests are complete and the resulting bytes are
bound by a new v4 seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SEALED_PROTOCOL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4_sealed.yaml"
CANONICAL_PROTOCOL_SEAL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4_seal.json"
DEFAULT_PROTOCOL = CANONICAL_SEALED_PROTOCOL
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "final" / "neural_information_regimes_v4"

CANONICAL_NEURAL_REGISTRY_PATHS = MappingProxyType(
    {
        "runner_sha256": Path(__file__).resolve(),
        "cell_plan_sha256": (
            ROOT / "protocols" / "wrr_information_regimes_neural_v4_cell_plan.json"
        ),
        "model_registry_sha256": (
            ROOT / "protocols" / "wrr_information_regimes_neural_v4_model_registry.json"
        ),
        "fold_registry_sha256": (
            ROOT / "protocols" / "wrr_information_regimes_neural_v4_fold_registry.json"
        ),
        "input_registry_sha256": (
            ROOT / "protocols" / "wrr_information_regimes_neural_v4_input_registry.json"
        ),
        "contrast_registry_sha256": (
            ROOT / "protocols" / "wrr_information_regimes_neural_v4_contrast_registry.json"
        ),
    }
)

EXPECTED_PROTOCOL_ID = "thermoroute_wrr_information_regimes_v4"
EXPECTED_PROTOCOL_VERSION = 4
AUTHORIZED_PROTOCOL_STATUS = "SEALED"
PROTOCOL_SEAL_FORMAT = "thermoroute.wrr-information-regimes-protocol-seal.v4"

SCHEMA_VERSION = "thermoroute.neural-information-regimes.phase1.1.v4"
MANIFEST_FORMAT = "thermoroute.neural-information-regime-fit-manifest.v4"
POLICY_STATUS = "PROPOSED_FOR_V4_SEAL_NOT_EXECUTABLE"
SUBSET_DIAGNOSTIC_STATUS = "SUBSET_COMPLETION_VALIDATION_UNAVAILABLE"
FULL_COMPLETION_UNAVAILABLE_STATUS = (
    "UNIMPLEMENTED_PREDICTION_SCHEMA_EXACT_KEY_AND_TARGET_VALIDATION"
)
RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS = (
    "UNIMPLEMENTED_TORCH_CHECKPOINT_AND_PARQUET_SCHEMA_EXACT_KEY_TARGET_VALIDATION"
)
NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS = (
    "UNIMPLEMENTED_CANONICAL_MODEL_FOLD_INPUT_AND_CONTRAST_REGISTRY_SEMANTIC_PARSERS"
)

LEADS = (1, 3, 7)
RANDOM_SPLIT_SEEDS = tuple(range(10))
REGION_SPLIT_SEED = 0
FOLDS = tuple(range(4))
MODEL_FIT_SEEDS = tuple(range(5))
REGION_MATCHED_HOLD_SIZES = (30, 30, 31, 29)
EXPECTED_CELL_PLAN_SHA256 = "b8bdff977fcfa441d33fc38a6eb0af43eb737849c3341cfc2a13aea40c220b77"

HISTORY_VARIABLES = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
HISTORY_LENGTH = 32
WTEMP_HISTORY_INDEX = HISTORY_VARIABLES.index("WTEMP")
GATE_FEATURES = ("sin_doy", "cos_doy", "TEMP", "FLOW", "PRCP", "WTEMP_tendency")
WTEMP_TENDENCY_INDEX = GATE_FEATURES.index("WTEMP_tendency")
METEOROLOGICAL_VARIABLES = ("TEMP", "PRCP", "DH", "RHMEAN", "WDSP")
FUTURE_STATISTICS = ("target_day", "prefix_mean")
SHARED_FOLD_REGISTRY_FORMAT = "thermoroute.neural-shared-fold-registry.v3"
SEMANTIC_FOLD_FINGERPRINT_FORMAT = "thermoroute.neural-semantic-fold-fingerprint.v1"
EVALUATION_START_DATE = date(2021, 1, 1)
EVALUATION_END_DATE = date(2023, 12, 31)
CANONICAL_FORECAST_KEY_FORMULA = (
    "sha256(temporal|known_site|site_id|issue_date|target_date|lead_days)"
)
SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND = "SYNTHETIC_CONTRACT_TEST_ONLY"
FORECAST_KEY_AUTHORITY_BINDING_FORMAT = (
    "thermoroute.neural-synthetic-common-key-authority-binding.v1"
)
CANONICAL_COMMON_KEY_REGISTRY_PATH = (
    "outputs/final/information_regime_key_registries_v4/primary_reportable_key_registry_v4.parquet"
)
PRODUCTION_COMMON_KEY_AUTHORITY_STATUS = (
    "UNAVAILABLE_UNTIL_CANONICAL_V4_COMMON_KEY_LOADER_IS_IMPLEMENTED"
)
MAX_ABS_ENVIRONMENTAL_SOURCE_VALUE = 1_000_000.0
FORMAL_SITE_ID_RE = re.compile(r"(?:[0-9]{8}|[0-9]{15})")


class ContractError(ValueError):
    """A requested plan or data object violates the frozen Phase-1 contract."""


class GovernanceError(RuntimeError):
    """The v4 protocol does not authorize a new-outcome neural execution."""


class Phase1ExecutionUnavailable(RuntimeError):
    """Training/scoring is intentionally unavailable in the Phase-1 runner."""


class PredictionContentValidationUnavailable(RuntimeError):
    """A complete matrix cannot be attested until prediction contents are checked."""


class ResumeContentValidationUnavailable(PredictionContentValidationUnavailable):
    """Resume is disabled until checkpoint and prediction semantics can be checked."""


class NeuralAuthoritySemanticValidationUnavailable(GovernanceError):
    """Canonical registry bytes exist but their required semantics are not validated."""


class ForcingArm(str, Enum):
    F0 = "F0"
    F3_FULL = "F3_full"


class LocalLevel(str, Enum):
    L0 = "L0"
    L2 = "L2"


class Geometry(str, Enum):
    RANDOM = "random"
    REGION = "region"


class Architecture(str, Enum):
    PLAIN_TCN = "plain_TCN"
    THERMOROUTE = "ThermoRoute"


class ConstraintVariant(str, Enum):
    UNBOUNDED = "unbounded"
    BOUNDED = "bounded"


class ArtifactStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


SELECTED_THERMOROUTE_REGIMES = frozenset(
    {
        (ForcingArm.F0, LocalLevel.L0, Geometry.RANDOM),
        (ForcingArm.F3_FULL, LocalLevel.L0, Geometry.RANDOM),
        (ForcingArm.F0, LocalLevel.L2, Geometry.REGION),
        (ForcingArm.F3_FULL, LocalLevel.L2, Geometry.REGION),
    }
)


def _require_enum(value: object, enum_type: type[Enum], field: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field} must be a {enum_type.__name__}, got {value!r}")


@dataclass(frozen=True, slots=True, order=True)
class CellPlan:
    """One indivisible neural fit unit with every experimental axis explicit."""

    forcing: ForcingArm
    local_level: LocalLevel
    geometry: Geometry
    architecture: Architecture
    lead: int
    split_seed: int
    fold: int
    fit_seed: int
    constraint: ConstraintVariant

    def __post_init__(self) -> None:
        _require_enum(self.forcing, ForcingArm, "forcing")
        _require_enum(self.local_level, LocalLevel, "local_level")
        _require_enum(self.geometry, Geometry, "geometry")
        _require_enum(self.architecture, Architecture, "architecture")
        _require_enum(self.constraint, ConstraintVariant, "constraint")
        if type(self.lead) is not int or self.lead not in LEADS:
            raise ContractError(f"lead must be one of {LEADS}, got {self.lead!r}")
        if type(self.fold) is not int or self.fold not in FOLDS:
            raise ContractError(f"fold must be one of {FOLDS}, got {self.fold!r}")
        if type(self.fit_seed) is not int or self.fit_seed not in MODEL_FIT_SEEDS:
            raise ContractError(f"fit_seed must be one of {MODEL_FIT_SEEDS}, got {self.fit_seed!r}")
        if type(self.split_seed) is not int:
            raise TypeError("split_seed must be an int")
        if self.geometry is Geometry.RANDOM:
            if self.split_seed not in RANDOM_SPLIT_SEEDS:
                raise ContractError(f"random split_seed must be one of {RANDOM_SPLIT_SEEDS}")
        elif self.split_seed != REGION_SPLIT_SEED:
            raise ContractError(f"region split_seed is the canonical sentinel {REGION_SPLIT_SEED}")

        regime = (self.forcing, self.local_level, self.geometry)
        if self.architecture is Architecture.PLAIN_TCN:
            if self.constraint is not ConstraintVariant.UNBOUNDED:
                raise ContractError("plain_TCN has only the unbounded residual variant")
        else:
            if regime not in SELECTED_THERMOROUTE_REGIMES:
                raise ContractError("ThermoRoute is limited to the four selected v4 hard regimes")

    @property
    def variant_configuration_id(self) -> str:
        """Identity excluding spatial and model-initialisation replication.

        The constraint is part of this identity.  In particular, the 12
        selected ThermoRoute protocol cells in the degraded F0/F3 design yield
        24 bounded/unbounded *variant configurations*, not 24 protocol logical
        cells.
        """

        return "__".join(
            (
                self.forcing.value,
                self.local_level.value,
                self.geometry.value,
                self.architecture.value,
                f"h{self.lead}",
                self.constraint.value,
            )
        )

    @property
    def fit_id(self) -> str:
        return "__".join(
            (
                self.variant_configuration_id,
                f"split{self.split_seed:02d}",
                f"fold{self.fold}",
                f"fit{self.fit_seed}",
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "forcing": self.forcing.value,
            "local_level": self.local_level.value,
            "geometry": self.geometry.value,
            "protocol_geometry": (
                "random_site" if self.geometry is Geometry.RANDOM else "whole_region"
            ),
            "architecture": self.architecture.value,
            "lead": self.lead,
            "split_seed": self.split_seed,
            "fold": self.fold,
            "fit_seed": self.fit_seed,
            "constraint": self.constraint.value,
            "variant_configuration_id": self.variant_configuration_id,
            "fit_id": self.fit_id,
        }


def _split_seeds(geometry: Geometry) -> tuple[int, ...]:
    return RANDOM_SPLIT_SEEDS if geometry is Geometry.RANDOM else (REGION_SPLIT_SEED,)


@lru_cache(maxsize=1)
def build_cell_plan() -> tuple[CellPlan, ...]:
    """Return the exact 5,280-fit F0/F3 neural Phase-1 inventory."""

    cells: list[CellPlan] = []
    for architecture in Architecture:
        for forcing in ForcingArm:
            for local_level in LocalLevel:
                for geometry in Geometry:
                    regime = (forcing, local_level, geometry)
                    if (
                        architecture is Architecture.THERMOROUTE
                        and regime not in SELECTED_THERMOROUTE_REGIMES
                    ):
                        continue
                    constraints = (
                        (ConstraintVariant.UNBOUNDED,)
                        if architecture is Architecture.PLAIN_TCN
                        else tuple(ConstraintVariant)
                    )
                    for lead in LEADS:
                        for constraint in constraints:
                            for split_seed in _split_seeds(geometry):
                                for fold in FOLDS:
                                    for fit_seed in MODEL_FIT_SEEDS:
                                        cells.append(
                                            CellPlan(
                                                forcing=forcing,
                                                local_level=local_level,
                                                geometry=geometry,
                                                architecture=architecture,
                                                lead=lead,
                                                split_seed=split_seed,
                                                fold=fold,
                                                fit_seed=fit_seed,
                                                constraint=constraint,
                                            )
                                        )
    plan = tuple(sorted(cells))
    validate_cell_plan(plan)
    return plan


@dataclass(frozen=True, slots=True)
class InventoryArithmetic:
    """Machine-checkable distinction between protocol logical cells and fit units."""

    protocol_f0_f3_primary_logical_cells: int
    protocol_lightgbm_primary_logical_cells: int
    protocol_plain_tcn_primary_logical_cells: int
    selected_thermoroute_protocol_cells: int
    selected_thermoroute_variant_configurations: int
    neural_runner_variant_configurations: int
    plain_tcn_random_fits: int
    plain_tcn_region_fits: int
    thermoroute_random_fits: int
    thermoroute_region_fits: int
    total_fit_units: int

    def to_record(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


def inventory_arithmetic(plan: Sequence[CellPlan]) -> InventoryArithmetic:
    configurations = {cell.variant_configuration_id for cell in plan}
    configurations_by_architecture = {
        architecture: {
            cell.variant_configuration_id for cell in plan if cell.architecture is architecture
        }
        for architecture in Architecture
    }
    counts = Counter((cell.architecture, cell.geometry) for cell in plan)
    return InventoryArithmetic(
        protocol_f0_f3_primary_logical_cells=2 * 2 * 2 * 2 * 3,
        protocol_lightgbm_primary_logical_cells=2 * 2 * 2 * 3,
        protocol_plain_tcn_primary_logical_cells=len(
            configurations_by_architecture[Architecture.PLAIN_TCN]
        ),
        selected_thermoroute_protocol_cells=len(SELECTED_THERMOROUTE_REGIMES) * len(LEADS),
        selected_thermoroute_variant_configurations=len(
            configurations_by_architecture[Architecture.THERMOROUTE]
        ),
        neural_runner_variant_configurations=len(configurations),
        plain_tcn_random_fits=counts[(Architecture.PLAIN_TCN, Geometry.RANDOM)],
        plain_tcn_region_fits=counts[(Architecture.PLAIN_TCN, Geometry.REGION)],
        thermoroute_random_fits=counts[(Architecture.THERMOROUTE, Geometry.RANDOM)],
        thermoroute_region_fits=counts[(Architecture.THERMOROUTE, Geometry.REGION)],
        total_fit_units=len(plan),
    )


EXPECTED_ARITHMETIC = InventoryArithmetic(
    protocol_f0_f3_primary_logical_cells=48,
    protocol_lightgbm_primary_logical_cells=24,
    protocol_plain_tcn_primary_logical_cells=24,
    selected_thermoroute_protocol_cells=12,
    selected_thermoroute_variant_configurations=24,
    neural_runner_variant_configurations=48,
    plain_tcn_random_fits=2400,
    plain_tcn_region_fits=240,
    thermoroute_random_fits=2400,
    thermoroute_region_fits=240,
    total_fit_units=5280,
)


def validate_cell_plan(plan: Sequence[CellPlan]) -> None:
    if not plan:
        raise ContractError("cell plan must not be empty")
    if any(not isinstance(cell, CellPlan) for cell in plan):
        raise TypeError("every plan item must be a CellPlan")
    fit_ids = [cell.fit_id for cell in plan]
    if len(fit_ids) != len(set(fit_ids)):
        raise ContractError("cell plan contains duplicate fit identities")
    arithmetic = inventory_arithmetic(plan)
    if arithmetic != EXPECTED_ARITHMETIC:
        raise ContractError(
            "cell-plan arithmetic mismatch: "
            f"{arithmetic.to_record()} != {EXPECTED_ARITHMETIC.to_record()}"
        )
    digest = cell_plan_sha256(plan)
    if digest != EXPECTED_CELL_PLAN_SHA256:
        raise ContractError(f"cell-plan SHA-256 mismatch: {digest} != {EXPECTED_CELL_PLAN_SHA256}")


def _canonical_json_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(document: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def cell_plan_sha256(plan: Sequence[CellPlan]) -> str:
    return canonical_sha256([cell.to_record() for cell in plan])


# ---------------------------------------------------------------------------
# Spatial-fold contract
# ---------------------------------------------------------------------------


def _canonical_stations(stations: Sequence[str], *, label: str) -> tuple[str, ...]:
    values = tuple(str(station) for station in stations)
    if not values:
        raise ContractError(f"{label} must not be empty")
    if len(values) != len(set(values)):
        raise ContractError(f"{label} contains duplicate stations")
    if values != tuple(sorted(values)):
        raise ContractError(f"{label} must be sorted for deterministic hashing")
    return values


@dataclass(frozen=True, slots=True)
class FoldDefinition:
    geometry: Geometry
    split_seed: int
    fold: int
    train_stations: tuple[str, ...]
    hold_stations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum(self.geometry, Geometry, "geometry")
        if type(self.fold) is not int or self.fold not in FOLDS:
            raise ContractError(f"fold must be one of {FOLDS}")
        if type(self.split_seed) is not int:
            raise TypeError("split_seed must be an int")
        if self.geometry is Geometry.RANDOM:
            if self.split_seed not in RANDOM_SPLIT_SEEDS:
                raise ContractError("random fold has an undeclared split seed")
        elif self.split_seed != REGION_SPLIT_SEED:
            raise ContractError("region fold must use split_seed=0")
        _canonical_stations(self.train_stations, label="train_stations")
        _canonical_stations(self.hold_stations, label="hold_stations")

    def to_record(self) -> dict[str, object]:
        return {
            "geometry": self.geometry.value,
            "split_seed": self.split_seed,
            "fold": self.fold,
            "train_stations": list(self.train_stations),
            "hold_stations": list(self.hold_stations),
        }


def validate_station_disjoint(train_stations: Sequence[str], hold_stations: Sequence[str]) -> None:
    train = set(train_stations)
    hold = set(hold_stations)
    overlap = sorted(train & hold)
    if overlap:
        raise ContractError(f"train/hold station overlap: {overlap[:5]}")


def validate_huc_disjoint(
    train_stations: Sequence[str],
    hold_stations: Sequence[str],
    huc2_by_station: Mapping[str, str],
) -> None:
    missing = sorted((set(train_stations) | set(hold_stations)) - set(huc2_by_station))
    if missing:
        raise ContractError(f"HUC2 registry lacks stations: {missing[:5]}")
    train_huc = {str(huc2_by_station[station]) for station in train_stations}
    hold_huc = {str(huc2_by_station[station]) for station in hold_stations}
    overlap = sorted(train_huc & hold_huc)
    if overlap:
        raise ContractError(f"whole-region fold splits HUC2 groups: {overlap}")


def validate_fold_collection(
    folds: Sequence[FoldDefinition],
    stations: Sequence[str],
    *,
    huc2_by_station: Mapping[str, str] | None = None,
    expected_hold_sizes: Sequence[int] = REGION_MATCHED_HOLD_SIZES,
) -> None:
    registry = _canonical_stations(stations, label="station registry")
    expected_sizes = tuple(int(value) for value in expected_hold_sizes)
    if len(folds) != len(FOLDS) or tuple(fold.fold for fold in folds) != FOLDS:
        raise ContractError("a spatial split must contain ordered folds 0,1,2,3")
    if len(expected_sizes) != len(FOLDS) or sum(expected_sizes) != len(registry):
        raise ContractError(
            "expected hold sizes must contain four entries summing to registry size"
        )
    geometry = folds[0].geometry
    split_seed = folds[0].split_seed
    held_once: list[str] = []
    for fold, expected_size in zip(folds, expected_sizes):
        if fold.geometry is not geometry or fold.split_seed != split_seed:
            raise ContractError("all folds in a collection must share geometry and split seed")
        validate_station_disjoint(fold.train_stations, fold.hold_stations)
        if set(fold.train_stations) | set(fold.hold_stations) != set(registry):
            raise ContractError(f"fold {fold.fold} does not partition the station registry")
        if len(fold.hold_stations) != expected_size:
            raise ContractError(
                f"fold {fold.fold} hold size {len(fold.hold_stations)} != {expected_size}"
            )
        if geometry is Geometry.REGION:
            if huc2_by_station is None:
                raise ContractError("whole-region validation requires the HUC2 registry")
            validate_huc_disjoint(fold.train_stations, fold.hold_stations, huc2_by_station)
        held_once.extend(fold.hold_stations)
    if Counter(held_once) != Counter(registry):
        raise ContractError("held folds must cover every station exactly once")


def build_matched_random_folds(
    stations: Sequence[str],
    seed: int,
    *,
    hold_sizes: Sequence[int] = REGION_MATCHED_HOLD_SIZES,
) -> tuple[FoldDefinition, ...]:
    """Build PCG64 folds with the exact 30/30/31/29 regional size vector."""

    registry = _canonical_stations(stations, label="station registry")
    if type(seed) is not int or seed not in RANDOM_SPLIT_SEEDS:
        raise ContractError(f"seed must be one of {RANDOM_SPLIT_SEEDS}")
    sizes = tuple(int(value) for value in hold_sizes)
    if len(sizes) != 4 or any(value <= 0 for value in sizes) or sum(sizes) != len(registry):
        raise ContractError(
            "random hold sizes must be four positive values summing to registry size"
        )
    permutation = np.random.default_rng(seed).permutation(len(registry))
    folds: list[FoldDefinition] = []
    start = 0
    all_stations = set(registry)
    for fold_index, size in enumerate(sizes):
        indices = permutation[start : start + size]
        hold = tuple(sorted(registry[int(index)] for index in indices))
        train = tuple(sorted(all_stations - set(hold)))
        folds.append(
            FoldDefinition(
                geometry=Geometry.RANDOM,
                split_seed=seed,
                fold=fold_index,
                train_stations=train,
                hold_stations=hold,
            )
        )
        start += size
    result = tuple(folds)
    validate_fold_collection(result, registry, expected_hold_sizes=sizes)
    return result


def build_region_folds(
    stations: Sequence[str], huc2_by_station: Mapping[str, str]
) -> tuple[FoldDefinition, ...]:
    """Deterministically pack intact HUC2 groups using the frozen greedy rule."""

    registry = _canonical_stations(stations, label="station registry")
    missing = sorted(set(registry) - set(huc2_by_station))
    if missing:
        raise ContractError(f"HUC2 registry lacks stations: {missing[:5]}")
    groups: dict[str, list[str]] = {}
    for station in registry:
        groups.setdefault(str(huc2_by_station[station]), []).append(station)
    buckets: list[list[str]] = [[] for _ in FOLDS]
    loads = [0 for _ in FOLDS]
    for _huc, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(FOLDS, key=lambda index: (loads[index], index))
        buckets[target].extend(members)
        loads[target] += len(members)
    all_stations = set(registry)
    result = tuple(
        FoldDefinition(
            geometry=Geometry.REGION,
            split_seed=REGION_SPLIT_SEED,
            fold=index,
            train_stations=tuple(sorted(all_stations - set(bucket))),
            hold_stations=tuple(sorted(bucket)),
        )
        for index, bucket in enumerate(buckets)
    )
    expected_sizes = REGION_MATCHED_HOLD_SIZES if len(registry) == 120 else tuple(loads)
    validate_fold_collection(
        result,
        registry,
        huc2_by_station=huc2_by_station,
        expected_hold_sizes=expected_sizes,
    )
    return result


def _canonical_huc2_items(
    stations: Sequence[str], huc2_by_station: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    registry = _canonical_stations(stations, label="station registry")
    if set(huc2_by_station) != set(registry):
        missing = sorted(set(registry) - set(huc2_by_station))
        extra = sorted(set(huc2_by_station) - set(registry))
        raise ContractError(f"HUC2 registry mismatch: missing={missing[:5]}, extra={extra[:5]}")
    items = tuple((station, str(huc2_by_station[station]).strip()) for station in registry)
    if any(not huc for _station, huc in items):
        raise ContractError("HUC2 registry contains an empty grouping identifier")
    return items


def build_canonical_fold_registry(
    stations: Sequence[str], huc2_by_station: Mapping[str, str]
) -> tuple[FoldDefinition, ...]:
    """Build all 44 frozen spatial folds, then validate the exact inventory."""

    registry = _canonical_stations(stations, label="station registry")
    if len(registry) != 120:
        raise ContractError("the v4 neural fold registry requires exactly 120 stations")
    huc_items = _canonical_huc2_items(registry, huc2_by_station)
    canonical_huc = dict(huc_items)
    region = build_region_folds(registry, canonical_huc)
    region_sizes = tuple(len(fold.hold_stations) for fold in region)
    if region_sizes != REGION_MATCHED_HOLD_SIZES:
        raise ContractError(
            f"canonical regional hold sizes {region_sizes} != {REGION_MATCHED_HOLD_SIZES}"
        )
    folds = (
        tuple(
            fold
            for seed in RANDOM_SPLIT_SEEDS
            for fold in build_matched_random_folds(registry, seed, hold_sizes=region_sizes)
        )
        + region
    )
    validate_canonical_fold_registry(folds, registry, canonical_huc)
    return folds


def validate_canonical_fold_registry(
    folds: Sequence[FoldDefinition],
    stations: Sequence[str],
    huc2_by_station: Mapping[str, str],
) -> None:
    """Require the exact 10x4 random plus 1x4 region v4 fold registry."""

    registry = _canonical_stations(stations, label="station registry")
    if len(registry) != 120:
        raise ContractError("the v4 neural fold registry requires exactly 120 stations")
    canonical_huc = dict(_canonical_huc2_items(registry, huc2_by_station))
    observed = tuple(folds)
    if len(observed) != 44 or any(not isinstance(fold, FoldDefinition) for fold in observed):
        raise ContractError("canonical fold registry must contain exactly 44 fold definitions")

    offset = 0
    for seed in RANDOM_SPLIT_SEEDS:
        collection = observed[offset : offset + len(FOLDS)]
        validate_fold_collection(
            collection,
            registry,
            expected_hold_sizes=REGION_MATCHED_HOLD_SIZES,
        )
        if any(
            fold.geometry is not Geometry.RANDOM or fold.split_seed != seed for fold in collection
        ):
            raise ContractError(f"random fold block {seed} has a geometry/seed mismatch")
        expected = build_matched_random_folds(
            registry,
            seed,
            hold_sizes=REGION_MATCHED_HOLD_SIZES,
        )
        if collection != expected:
            raise ContractError(f"random fold block {seed} differs from the canonical PCG64 split")
        offset += len(FOLDS)

    region = observed[offset:]
    validate_fold_collection(
        region,
        registry,
        huc2_by_station=canonical_huc,
        expected_hold_sizes=REGION_MATCHED_HOLD_SIZES,
    )
    if any(
        fold.geometry is not Geometry.REGION or fold.split_seed != REGION_SPLIT_SEED
        for fold in region
    ):
        raise ContractError("regional fold block has a geometry/seed mismatch")
    if region != build_region_folds(registry, canonical_huc):
        raise ContractError("regional fold block differs from canonical greedy HUC2 packing")


def fold_registry_sha256(
    folds: Sequence[FoldDefinition],
    stations: Sequence[str],
    huc2_by_station: Mapping[str, str],
) -> str:
    """Hash only after exact full-registry validation, including HUC2 lineage."""

    validate_canonical_fold_registry(folds, stations, huc2_by_station)
    registry = _canonical_stations(stations, label="station registry")
    return canonical_sha256(
        {
            "stations": list(registry),
            "huc2_by_station": [
                list(item) for item in _canonical_huc2_items(registry, huc2_by_station)
            ],
            "folds": [fold.to_record() for fold in folds],
        }
    )


# ---------------------------------------------------------------------------
# Canonical future-feature contract
# ---------------------------------------------------------------------------


def _parse_iso_date(raw: object, *, label: str) -> date:
    if not isinstance(raw, str):
        raise ContractError(f"{label} must be an ISO date string")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise ContractError(f"{label} is not a valid ISO date") from exc
    if parsed.isoformat() != raw:
        raise ContractError(f"{label} must use canonical YYYY-MM-DD form")
    return parsed


def _require_formal_site_id(value: object, *, label: str) -> str:
    if type(value) is not str or FORMAL_SITE_ID_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must match ASCII-only (?:[0-9]{{8}}|[0-9]{{15}})")
    return value


@lru_cache(maxsize=1)
def _canonical_training_source_dates() -> tuple[str, ...]:
    start = date(2006, 1, 1)
    end = date(2015, 12, 31)
    return tuple(
        (start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)
    )


def _reject_masked_array(value: object, *, label: str) -> None:
    """Reject NumPy masked containers before any coercion can discard their mask."""

    if isinstance(value, np.ma.MaskedArray):
        raise ContractError(f"{label} must not be a NumPy MaskedArray")


def _safe_daily_doy_curve(
    values: np.ndarray,
    date_doys: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Aggregate exact daily values without accepting nonphysical overflow payloads."""

    _reject_masked_array(values, label=f"{label} daily values")
    _reject_masked_array(date_doys, label=f"{label} day-of-year indices")
    if values.shape != date_doys.shape or not np.isfinite(values).all():
        raise ContractError(f"{label} daily source must contain exact finite values")
    if np.any(np.abs(values) > MAX_ABS_ENVIRONMENTAL_SOURCE_VALUE):
        raise ContractError(
            f"{label} daily source exceeds the physically/overflow-safe magnitude limit"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sums = np.bincount(date_doys, weights=values, minlength=366)
        counts = np.bincount(date_doys, minlength=366)
        curve = sums / counts
    if (
        sums.shape != (366,)
        or counts.shape != (366,)
        or curve.shape != (366,)
        or np.any(counts <= 0)
        or not np.isfinite(sums).all()
        or not np.isfinite(counts).all()
        or not np.isfinite(curve).all()
        or np.any(np.abs(curve) > MAX_ABS_ENVIRONMENTAL_SOURCE_VALUE)
    ):
        raise ContractError(f"{label} cannot derive a finite overflow-safe 366-day curve")
    return curve


def _safe_pooled_curve(curves: Sequence[np.ndarray], *, label: str) -> np.ndarray:
    _reject_masked_array(curves, label=f"{label} curve collection")
    candidates = tuple(curves)
    if not candidates:
        raise ContractError(f"{label} requires at least one training-station curve")
    for index, curve in enumerate(candidates):
        _reject_masked_array(curve, label=f"{label} station curve {index}")
    stack = np.stack(candidates, axis=0)
    if stack.ndim != 2 or stack.shape[1] != 366 or not np.isfinite(stack).all():
        raise ContractError(f"{label} station curves must be exact finite 366-day vectors")
    with np.errstate(over="ignore", invalid="ignore"):
        pooled = np.mean(stack, axis=0)
    if (
        pooled.shape != (366,)
        or not np.isfinite(pooled).all()
        or np.any(np.abs(pooled) > MAX_ABS_ENVIRONMENTAL_SOURCE_VALUE)
    ):
        raise ContractError(f"{label} pooled aggregation is nonfinite or overflow-unsafe")
    return pooled


@dataclass(frozen=True, slots=True)
class FutureFeaturePolicy:
    variables: tuple[str, ...] = METEOROLOGICAL_VARIABLES
    statistics: tuple[str, ...] = FUTURE_STATISTICS
    width: int = len(METEOROLOGICAL_VARIABLES) * len(FUTURE_STATISTICS)
    missing_value_rule: str = "training_period_variable_climatology_substitution"
    raw_missingness_rule: str = "raw_value_is_json_null_if_and_only_if_is_missing_is_true"
    substitution_fit_aggregation: str = (
        "unweighted_mean_across_exact_fold_training_station_climatologies"
    )
    substitution_date_rule: str = (
        "one_based_gregorian_valid_date_day_of_year_index_into_366_value_curve"
    )
    substitution_flags_are_model_inputs: bool = False
    raw_trajectory_exposed_to_model: bool = False
    normalization_fit_period: str = "2006-01-01/2015-12-31"
    value_state: str = (
        "canonical physical-unit target/prefix summaries; fold-fit normalization "
        "is intentionally unimplemented in Phase 1"
    )

    def feature_names(self, lead: int) -> tuple[str, ...]:
        if type(lead) is not int or lead not in LEADS:
            raise ContractError(f"future feature lead must be one of {LEADS}")
        return tuple(
            name
            for variable in self.variables
            for name in (f"{variable}_fut{lead}", f"{variable}_futmean{lead}")
        )

    def to_record(self) -> dict[str, object]:
        return {
            "variables": list(self.variables),
            "statistics": list(self.statistics),
            "width": self.width,
            "missing_value_rule": self.missing_value_rule,
            "raw_missingness_rule": self.raw_missingness_rule,
            "substitution_fit_aggregation": self.substitution_fit_aggregation,
            "substitution_date_rule": self.substitution_date_rule,
            "substitution_flags_are_model_inputs": self.substitution_flags_are_model_inputs,
            "substitution_audit_shape": "[N,5,lead]",
            "substitution_audit_retained_and_hash_bound": True,
            "raw_trajectory_exposed_to_model": self.raw_trajectory_exposed_to_model,
            "normalization_fit_period": self.normalization_fit_period,
            "value_state": self.value_state,
            "source_registry_format": FUTURE_SOURCE_REGISTRY_FORMAT,
            "substitution_registry_format": FUTURE_SUBSTITUTION_REGISTRY_FORMAT,
            "fit_provenance_format": FUTURE_FIT_PROVENANCE_FORMAT,
            "fit_fold_registry_format": FUTURE_FIT_FOLD_REGISTRY_FORMAT,
            "substitution_fit_source_format": FUTURE_SUBSTITUTION_FIT_SOURCE_FORMAT,
            "factory_seal_format": FUTURE_FACTORY_SEAL_FORMAT,
            "F3_fit_evidence": (
                "one shared fold/HUC registry, exact daily 2006--2015 training-source "
                "values from which climatology is recomputed, structured ForecastKeys, "
                "and raw-value/missingness registry bytes"
            ),
            "construction_authority": FUTURE_STRUCTURAL_STATUS,
            "feature_names_by_lead": {str(lead): list(self.feature_names(lead)) for lead in LEADS},
        }


FUTURE_FEATURE_POLICY = FutureFeaturePolicy()

FUTURE_SOURCE_REGISTRY_FORMAT = "thermoroute.neural-future-raw-registry.v4"
FUTURE_SUBSTITUTION_REGISTRY_FORMAT = "thermoroute.neural-future-substitution-registry.v4"
FUTURE_FIT_PROVENANCE_FORMAT = "thermoroute.neural-future-fit-provenance.v4"
FUTURE_FIT_FOLD_REGISTRY_FORMAT = SHARED_FOLD_REGISTRY_FORMAT
FUTURE_SUBSTITUTION_FIT_SOURCE_FORMAT = "thermoroute.neural-future-daily-fit-source.v3"
FUTURE_FACTORY_SEAL_FORMAT = "thermoroute.neural-future-factory-seal.v4"
FUTURE_STRUCTURAL_STATUS = "RECOMPUTED_FROM_EXACT_BYTES_NOT_EXECUTION_AUTHORITY"
FUTURE_FIT_TRAINING_PERIOD = "2006-01-01/2015-12-31"


def _readonly_array(value: object, dtype: np.dtype[Any] | type) -> np.ndarray:
    _reject_masked_array(value, label="array input")
    contiguous = np.ascontiguousarray(np.array(value, dtype=dtype, copy=True))
    immutable_payload = contiguous.tobytes(order="C")
    return np.frombuffer(immutable_payload, dtype=contiguous.dtype).reshape(contiguous.shape)


def _strict_float_array(value: object, *, label: str) -> np.ndarray:
    _reject_masked_array(value, label=label)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not a numeric array") from exc
    if raw.dtype.kind not in "iuf":
        raise ContractError(f"{label} must contain only real numeric values")
    return np.asarray(raw, dtype=np.float64)


def _strict_bool_array(value: object, *, label: str) -> np.ndarray:
    _reject_masked_array(value, label=label)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not a boolean array") from exc
    if raw.dtype != np.dtype(np.bool_):
        raise ContractError(f"{label} must contain exact boolean values")
    return np.asarray(raw, dtype=np.bool_)


def _array_sha256(label: str, value: np.ndarray) -> str:
    _reject_masked_array(value, label=f"{label} hash input")
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(_canonical_json_bytes(list(array.shape)))
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _parse_canonical_registry_bytes(payload: bytes, *, label: str) -> Mapping[str, object]:
    if type(payload) is not bytes or not payload:
        raise ContractError(f"{label} must be non-empty immutable bytes")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid canonical JSON") from exc
    if not isinstance(document, Mapping):
        raise ContractError(f"{label} must decode to a mapping")
    if _canonical_json_bytes(document) != payload:
        raise ContractError(f"{label} bytes are not in canonical JSON form")
    return document


def _fold_from_exact_record(record: object, *, label: str) -> FoldDefinition:
    expected_keys = {
        "geometry",
        "split_seed",
        "fold",
        "train_stations",
        "hold_stations",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise ContractError(f"{label} fold-record schema mismatch")
    try:
        geometry = Geometry(record["geometry"])
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} fold geometry is invalid") from exc
    train = record["train_stations"]
    hold = record["hold_stations"]
    if not isinstance(train, list) or not isinstance(hold, list):
        raise ContractError(f"{label} fold station sets must be JSON lists")
    if any(not isinstance(value, str) for value in (*train, *hold)):
        raise ContractError(f"{label} fold station identifiers must be strings")
    for station in (*train, *hold):
        _require_formal_site_id(station, label=f"{label} station identifier")
    return FoldDefinition(
        geometry=geometry,
        split_seed=record["split_seed"],  # type: ignore[arg-type]
        fold=record["fold"],  # type: ignore[arg-type]
        train_stations=tuple(train),
        hold_stations=tuple(hold),
    )


@dataclass(frozen=True, slots=True)
class FutureRawRecord:
    """One ForecastKey's raw future values; ``None`` is the only missing marker."""

    key_id: str
    site_no: str
    huc2: str
    issue_date: str
    target_date: str
    lead: int
    raw_values: Mapping[str, Sequence[float | None]]

    def __post_init__(self) -> None:
        for label, value in (
            ("key_id", self.key_id),
            ("huc2", self.huc2),
        ):
            if type(value) is not str or not value:
                raise ContractError(f"future raw {label} must be a non-empty string")
        _require_formal_site_id(self.site_no, label="future raw site_no")
        issue = _parse_iso_date(self.issue_date, label="future raw issue_date")
        target = _parse_iso_date(self.target_date, label="future raw target_date")
        if type(self.lead) is not int or self.lead not in LEADS:
            raise ContractError(f"future raw lead must be one of {LEADS}")
        if (target - issue).days != self.lead:
            raise ContractError("future raw target_date minus issue_date must equal lead")
        if not isinstance(self.raw_values, Mapping) or set(self.raw_values) != set(
            METEOROLOGICAL_VARIABLES
        ):
            raise ContractError("future raw variables mismatch")
        for variable in METEOROLOGICAL_VARIABLES:
            values = self.raw_values[variable]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ContractError(f"future raw {variable} values must be a sequence")
            if len(values) != self.lead:
                raise ContractError(f"future raw {variable} must contain exactly {self.lead} steps")
            for value in values:
                if value is None:
                    continue
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise ContractError(
                        f"future raw {variable} values must be finite numerics or null"
                    )
                if not np.isfinite(float(value)):
                    raise ContractError(
                        f"future raw {variable} values must be finite numerics or null"
                    )
                if abs(float(value)) > MAX_ABS_ENVIRONMENTAL_SOURCE_VALUE:
                    raise ContractError(
                        f"future raw {variable} exceeds the physically/overflow-safe "
                        "magnitude limit"
                    )


def _parse_future_fit_evidence_bytes(
    fold_registry_bytes: bytes,
    substitution_fit_source_bytes: bytes,
) -> tuple[
    FoldDefinition,
    dict[str, str],
    dict[str, dict[str, np.ndarray]],
    str,
]:
    folds, huc2_by_station = _parse_pooled_l2_fold_registry(fold_registry_bytes)
    source_document = _parse_canonical_registry_bytes(
        substitution_fit_source_bytes, label="future substitution-fit source registry"
    )
    if (
        set(source_document)
        != {
            "format",
            "training_period",
            "geometry",
            "split_seed",
            "fold",
            "records",
        }
        or source_document.get("format") != FUTURE_SUBSTITUTION_FIT_SOURCE_FORMAT
    ):
        raise ContractError("future substitution-fit source registry schema mismatch")
    if source_document.get("training_period") != FUTURE_FIT_TRAINING_PERIOD:
        raise ContractError("future substitution-fit source training period mismatch")
    try:
        geometry = Geometry(source_document.get("geometry"))
    except (TypeError, ValueError) as exc:
        raise ContractError("future substitution-fit geometry is invalid") from exc
    split_seed = source_document.get("split_seed")
    fold_index = source_document.get("fold")
    if type(split_seed) is not int or type(fold_index) is not int:
        raise ContractError("future substitution-fit fold identity must use exact integers")
    fold = _select_exact_fold(
        folds,
        geometry=geometry,
        split_seed=split_seed,
        fold_index=fold_index,
        label="future substitution-fit source",
    )
    records = source_document.get("records")
    if not isinstance(records, list):
        raise ContractError("future substitution-fit source records must be a JSON list")
    expected_dates = _canonical_training_source_dates()
    date_doys = np.asarray(
        [date.fromisoformat(value).timetuple().tm_yday - 1 for value in expected_dates],
        dtype=np.int16,
    )
    climatology: dict[str, dict[str, np.ndarray]] = {}
    observed_stations: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "station_id",
            "daily_records",
        }:
            raise ContractError("future substitution-fit source record schema mismatch")
        station = record.get("station_id")
        daily_records = record.get("daily_records")
        if (
            type(station) is not str
            or station in climatology
            or not isinstance(daily_records, list)
        ):
            raise ContractError("future substitution-fit station/daily fields are invalid")
        _require_formal_site_id(station, label="future substitution-fit station_id")
        if len(daily_records) != len(expected_dates):
            raise ContractError(
                f"future substitution-fit daily source for {station} must cover exact 2006--2015"
            )
        daily_values = {
            variable: np.empty(len(expected_dates), dtype=np.float64)
            for variable in METEOROLOGICAL_VARIABLES
        }
        for row, (daily_record, expected_date) in enumerate(
            zip(daily_records, expected_dates, strict=True)
        ):
            if not isinstance(daily_record, Mapping) or set(daily_record) != {
                "date",
                "values",
            }:
                raise ContractError("future substitution-fit daily record schema mismatch")
            if daily_record.get("date") != expected_date:
                raise ContractError(
                    f"future substitution-fit source dates for {station} are not exact 2006--2015 dates"
                )
            values = daily_record.get("values")
            if not isinstance(values, Mapping) or set(values) != set(METEOROLOGICAL_VARIABLES):
                raise ContractError("future substitution-fit daily variables mismatch")
            for variable in METEOROLOGICAL_VARIABLES:
                raw_value = values[variable]
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise ContractError("future substitution-fit daily values must be numerics")
                numeric = float(raw_value)
                if not np.isfinite(numeric):
                    raise ContractError("future substitution-fit daily values must be finite")
                daily_values[variable][row] = numeric
        station_curves: dict[str, np.ndarray] = {}
        for variable in METEOROLOGICAL_VARIABLES:
            station_curves[variable] = _safe_daily_doy_curve(
                daily_values[variable],
                date_doys,
                label=f"future substitution-fit {station}/{variable}",
            )
        observed_stations.append(station)
        climatology[station] = station_curves
    if tuple(observed_stations) != fold.train_stations:
        missing = sorted(set(fold.train_stations) - set(observed_stations))
        extra = sorted(set(observed_stations) - set(fold.train_stations))
        raise ContractError(
            "future substitution-fit source membership/order mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    semantic_fold_sha256 = _semantic_fold_sha256(fold_registry_bytes, fold, huc2_by_station)
    return fold, huc2_by_station, climatology, semantic_fold_sha256


def _validate_future_fit_evidence_bytes(
    fold_registry_bytes: bytes,
    substitution_fit_source_bytes: bytes,
) -> None:
    _parse_future_fit_evidence_bytes(fold_registry_bytes, substitution_fit_source_bytes)


def build_training_only_future_fit_evidence(
    *,
    fold: FoldDefinition,
    fold_registry_bytes: bytes,
    huc2_by_station: Mapping[str, str],
    training_station_variable_daily_values: Mapping[str, Mapping[str, np.ndarray]],
    training_source_dates_by_station: Mapping[str, Sequence[str]],
) -> tuple[bytes, bytes]:
    """Serialize daily F3 fit evidence; climatology curves are always recomputed."""

    if type(fold) is not FoldDefinition:
        raise TypeError("future fit fold must be a FoldDefinition")
    shared_folds, shared_huc2 = _parse_pooled_l2_fold_registry(fold_registry_bytes)
    selected_fold = _select_exact_fold(
        shared_folds,
        geometry=fold.geometry,
        split_seed=fold.split_seed,
        fold_index=fold.fold,
        label="future fit evidence",
    )
    if selected_fold.to_record() != fold.to_record():
        raise ContractError("future fit fold differs from exact shared fold-registry bytes")
    station_universe = tuple(sorted(set(fold.train_stations) | set(fold.hold_stations)))
    huc_items = _canonical_huc2_items(station_universe, huc2_by_station)
    if dict(huc_items) != shared_huc2:
        raise ContractError("future fit HUC2 mapping differs from shared fold-registry bytes")
    if fold.geometry is Geometry.REGION:
        validate_huc_disjoint(fold.train_stations, fold.hold_stations, dict(huc_items))
    if set(training_station_variable_daily_values) != set(fold.train_stations) or set(
        training_source_dates_by_station
    ) != set(fold.train_stations):
        raise ContractError("future substitution-fit source membership mismatch")
    expected_dates = _canonical_training_source_dates()
    records: list[dict[str, object]] = []
    for station in fold.train_stations:
        dates = tuple(training_source_dates_by_station[station])
        if any(type(value) is not str for value in dates):
            raise ContractError(
                f"future substitution-fit source dates for {station} must be exact strings"
            )
        if dates != expected_dates:
            raise ContractError(
                f"future substitution-fit source dates for {station} are not exact 2006--2015 dates"
            )
        by_variable = training_station_variable_daily_values[station]
        if set(by_variable) != set(METEOROLOGICAL_VARIABLES):
            raise ContractError(f"future substitution-fit daily variables mismatch for {station}")
        canonical_variables: dict[str, np.ndarray] = {}
        for variable in METEOROLOGICAL_VARIABLES:
            values = _strict_float_array(
                by_variable[variable],
                label=f"future substitution-fit {station}/{variable}",
            )
            if values.shape != (len(expected_dates),) or not np.isfinite(values).all():
                raise ContractError(
                    f"future substitution-fit {station}/{variable} must have one finite value "
                    "for every exact 2006--2015 date"
                )
            canonical_variables[variable] = values
        records.append(
            {
                "station_id": station,
                "daily_records": [
                    {
                        "date": source_date,
                        "values": {
                            variable: float(canonical_variables[variable][row])
                            for variable in METEOROLOGICAL_VARIABLES
                        },
                    }
                    for row, source_date in enumerate(dates)
                ],
            }
        )
    fold_bytes = bytes(fold_registry_bytes)
    source_bytes = _canonical_json_bytes(
        {
            "format": FUTURE_SUBSTITUTION_FIT_SOURCE_FORMAT,
            "training_period": FUTURE_FIT_TRAINING_PERIOD,
            "geometry": fold.geometry.value,
            "split_seed": fold.split_seed,
            "fold": fold.fold,
            "records": records,
        }
    )
    _validate_future_fit_evidence_bytes(fold_bytes, source_bytes)
    return fold_bytes, source_bytes


def build_raw_future_registry_bytes(
    *,
    fold: FoldDefinition,
    records: Sequence[FutureRawRecord],
) -> bytes:
    """Serialize raw values and explicit missingness, keyed by site/date/step."""

    if type(fold) is not FoldDefinition:
        raise TypeError("future raw registry fold must be a FoldDefinition")
    for station in (*fold.train_stations, *fold.hold_stations):
        _require_formal_site_id(station, label="future raw fold station identifier")
    candidates = tuple(records)
    if not candidates or any(type(record) is not FutureRawRecord for record in candidates):
        raise ContractError("future raw registry requires typed non-empty records")
    key_ids = tuple(record.key_id for record in candidates)
    canonical_order = tuple(
        (record.site_no, record.issue_date, record.target_date, record.lead, record.key_id)
        for record in candidates
    )
    if canonical_order != tuple(sorted(canonical_order)) or len(key_ids) != len(set(key_ids)):
        raise ContractError("future raw records must have unique keys in canonical tuple order")
    leads = {record.lead for record in candidates}
    if len(leads) != 1:
        raise ContractError("future raw registry must contain exactly one lead")
    lead = next(iter(leads))
    station_universe = set(fold.train_stations) | set(fold.hold_stations)
    canonical_records: list[dict[str, object]] = []
    for record in candidates:
        if record.site_no not in station_universe:
            raise ContractError(f"future raw site is outside selected fold: {record.site_no}")
        issue = _parse_iso_date(record.issue_date, label="future raw issue_date")
        variables: dict[str, list[dict[str, object]]] = {}
        for variable in METEOROLOGICAL_VARIABLES:
            steps: list[dict[str, object]] = []
            for step, raw_value in enumerate(record.raw_values[variable], start=1):
                missing = raw_value is None
                steps.append(
                    {
                        "step": step,
                        "valid_date": (issue + timedelta(days=step)).isoformat(),
                        "raw_value": None if missing else float(raw_value),
                        "is_missing": missing,
                    }
                )
            variables[variable] = steps
        canonical_records.append(
            {
                "key_id": record.key_id,
                "site_no": record.site_no,
                "huc2": record.huc2,
                "issue_date": record.issue_date,
                "target_date": record.target_date,
                "variables": variables,
            }
        )
    return _canonical_json_bytes(
        {
            "format": FUTURE_SOURCE_REGISTRY_FORMAT,
            "forcing": ForcingArm.F3_FULL.value,
            "geometry": fold.geometry.value,
            "split_seed": fold.split_seed,
            "fold": fold.fold,
            "lead": lead,
            "records": canonical_records,
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class CanonicalFutureBatch:
    """Structurally verified future inputs; never an execution attestation.

    Instances retain the exact source, substitution, and fit-evidence bytes.
    Consumers re-materialize every tensor and digest from those bytes instead of
    trusting constructor arguments or caller-supplied digest strings.
    """

    forcing: ForcingArm
    lead: int
    key_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    issue_dates: tuple[str, ...]
    target_dates: tuple[str, ...]
    geometry: Geometry | None
    split_seed: int | None
    fold: int | None
    forecast_keys_sha256: str
    key_authority_binding_sha256: str
    fold_registry_sha256: str
    semantic_fold_sha256: str
    feature_names: tuple[str, ...]
    values: np.ndarray
    mask: np.ndarray
    substitution_mask: np.ndarray
    source_value_sha256: str
    substitution_registry_sha256: str
    fit_provenance_sha256: str
    canonical_value_sha256: str
    canonical_mask_sha256: str
    substitution_mask_sha256: str
    factory_seal_sha256: str
    construction_status: str
    _source_registry_bytes: bytes = field(repr=False, compare=False)
    _substitution_registry_bytes: bytes = field(repr=False, compare=False)
    _fit_provenance_bytes: bytes = field(repr=False, compare=False)
    _forecast_keys_registry_bytes: bytes = field(repr=False, compare=False)
    _fold_registry_bytes: bytes = field(repr=False, compare=False)
    _substitution_fit_source_bytes: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "CanonicalFutureBatch cannot be constructed directly; use build_canonical_future_batch"
        )

    @property
    def n_examples(self) -> int:
        return int(self.values.shape[0])

    @property
    def execution_validated(self) -> bool:
        """Structural synthetic construction is never execution authority."""

        return False

    def substitution_audit_record(self) -> dict[str, object]:
        validate_canonical_future_batch(self)
        by_variable = {
            variable: int(self.substitution_mask[:, index, :].sum())
            for index, variable in enumerate(METEOROLOGICAL_VARIABLES)
        }
        return {
            "forcing": self.forcing.value,
            "lead": self.lead,
            "n_examples": self.n_examples,
            "fold_identity": (
                None
                if self.geometry is None
                else {
                    "geometry": self.geometry.value,
                    "split_seed": self.split_seed,
                    "fold": self.fold,
                }
            ),
            "source_value_sha256": self.source_value_sha256,
            "forecast_keys_sha256": self.forecast_keys_sha256,
            "key_authority_binding_sha256": self.key_authority_binding_sha256,
            "fold_registry_sha256": self.fold_registry_sha256,
            "semantic_fold_sha256": self.semantic_fold_sha256,
            "substitution_registry_sha256": self.substitution_registry_sha256,
            "fit_provenance_sha256": self.fit_provenance_sha256,
            "canonical_value_sha256": self.canonical_value_sha256,
            "canonical_mask_sha256": self.canonical_mask_sha256,
            "substitution_mask_sha256": self.substitution_mask_sha256,
            "factory_seal_sha256": self.factory_seal_sha256,
            "substitution_count_total": int(self.substitution_mask.sum()),
            "substitution_count_by_variable": by_variable,
            "substitution_flags_are_model_inputs": False,
        }


@dataclass(frozen=True, slots=True)
class _ParsedFutureRawRegistry:
    fold: FoldDefinition
    lead: int
    key_ids: tuple[str, ...]
    site_nos: tuple[str, ...]
    huc2s: tuple[str, ...]
    issue_dates: tuple[str, ...]
    target_dates: tuple[str, ...]
    raw_values: np.ndarray
    missing: np.ndarray


def _parse_future_raw_registry(
    source_registry_bytes: bytes,
    *,
    fold: FoldDefinition,
    huc2_by_station: Mapping[str, str],
) -> _ParsedFutureRawRegistry:
    source = _parse_canonical_registry_bytes(source_registry_bytes, label="future raw registry")
    expected_source_keys = {
        "format",
        "forcing",
        "geometry",
        "split_seed",
        "fold",
        "lead",
        "records",
    }
    if set(source) != expected_source_keys or source.get("format") != FUTURE_SOURCE_REGISTRY_FORMAT:
        raise ContractError("future raw registry schema mismatch")
    if source.get("forcing") != ForcingArm.F3_FULL.value:
        raise ContractError("future raw registry must declare F3_full")
    if (
        source.get("geometry") != fold.geometry.value
        or source.get("split_seed") != fold.split_seed
        or source.get("fold") != fold.fold
    ):
        raise ContractError("future raw registry does not select the fit-evidence fold")
    lead = source.get("lead")
    if type(lead) is not int or lead not in LEADS:
        raise ContractError(f"future raw registry lead must be one of {LEADS}")
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("future raw registry records must be a non-empty JSON list")
    n_examples = len(records)
    raw_cube = np.empty((n_examples, len(METEOROLOGICAL_VARIABLES), lead), dtype=np.float64)
    missing_cube = np.empty_like(raw_cube, dtype=np.bool_)
    keys: list[str] = []
    sites: list[str] = []
    huc2s: list[str] = []
    issue_dates: list[str] = []
    target_dates: list[str] = []
    expected_record_keys = {
        "key_id",
        "site_no",
        "huc2",
        "issue_date",
        "target_date",
        "variables",
    }
    station_universe = set(fold.train_stations) | set(fold.hold_stations)
    for row, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise ContractError("future raw record schema mismatch")
        key_id = record.get("key_id")
        site_no = record.get("site_no")
        huc2 = record.get("huc2")
        if type(key_id) is not str or not key_id:
            raise ContractError("future raw key_id must be a non-empty string")
        if type(site_no) is not str:
            raise ContractError("future raw site_no is outside the selected fold")
        _require_formal_site_id(site_no, label="future raw site_no")
        if site_no not in station_universe:
            raise ContractError("future raw site_no is outside the selected fold")
        if type(huc2) is not str or huc2_by_station[site_no] != huc2:
            raise ContractError(f"future raw HUC2 mismatch for {site_no}")
        issue_raw = record.get("issue_date")
        target_raw = record.get("target_date")
        issue = _parse_iso_date(issue_raw, label="future raw issue_date")
        target = _parse_iso_date(target_raw, label="future raw target_date")
        if (target - issue).days != lead:
            raise ContractError("future raw target_date minus issue_date must equal lead")
        variables = record.get("variables")
        if not isinstance(variables, Mapping) or set(variables) != set(METEOROLOGICAL_VARIABLES):
            raise ContractError("future raw variables mismatch")
        for variable_index, variable in enumerate(METEOROLOGICAL_VARIABLES):
            steps = variables[variable]
            if not isinstance(steps, list) or len(steps) != lead:
                raise ContractError(f"future raw {variable} must contain exactly {lead} steps")
            for expected_step, step_record in enumerate(steps, start=1):
                if not isinstance(step_record, Mapping) or set(step_record) != {
                    "step",
                    "valid_date",
                    "raw_value",
                    "is_missing",
                }:
                    raise ContractError("future raw step record schema mismatch")
                if (
                    step_record.get("step") != expected_step
                    or type(step_record.get("step")) is not int
                ):
                    raise ContractError("future raw step indices are not exact and ordered")
                expected_date = (issue + timedelta(days=expected_step)).isoformat()
                if (
                    _parse_iso_date(
                        step_record.get("valid_date"), label="future raw valid_date"
                    ).isoformat()
                    != expected_date
                ):
                    raise ContractError("future raw valid_date does not align with issue/step")
                is_missing = step_record.get("is_missing")
                raw_value = step_record.get("raw_value")
                if type(is_missing) is not bool:
                    raise ContractError("future raw is_missing must be an exact boolean")
                if is_missing:
                    if raw_value is not None:
                        raise ContractError("missing future raw values must be exact JSON null")
                    raw_cube[row, variable_index, expected_step - 1] = np.nan
                else:
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise ContractError("observed future raw values must be JSON numerics")
                    numeric = float(raw_value)
                    if not np.isfinite(numeric):
                        raise ContractError("observed future raw values must be finite")
                    if abs(numeric) > MAX_ABS_ENVIRONMENTAL_SOURCE_VALUE:
                        raise ContractError(
                            "observed future raw value exceeds the physically/overflow-safe "
                            "magnitude limit"
                        )
                    raw_cube[row, variable_index, expected_step - 1] = numeric
                missing_cube[row, variable_index, expected_step - 1] = is_missing
        keys.append(key_id)
        sites.append(site_no)
        huc2s.append(huc2)
        issue_dates.append(issue.isoformat())
        target_dates.append(target.isoformat())
    canonical_order = tuple(
        (sites[index], issue_dates[index], target_dates[index], lead, keys[index])
        for index in range(len(keys))
    )
    if canonical_order != tuple(sorted(canonical_order)) or len(keys) != len(set(keys)):
        raise ContractError("future raw records must have unique keys in canonical tuple order")
    return _ParsedFutureRawRegistry(
        fold=fold,
        lead=lead,
        key_ids=tuple(keys),
        site_nos=tuple(sites),
        huc2s=tuple(huc2s),
        issue_dates=tuple(issue_dates),
        target_dates=tuple(target_dates),
        raw_values=raw_cube,
        missing=missing_cube,
    )


def _materialize_canonical_future_batch(
    *,
    source_registry_bytes: bytes,
    fold_registry_bytes: bytes,
    forecast_keys_registry_bytes: bytes,
    substitution_fit_source_bytes: bytes,
) -> CanonicalFutureBatch:
    key_authority = _materialize_forecast_key_authority(
        fold_registry_bytes=fold_registry_bytes,
        forecast_keys_registry_bytes=forecast_keys_registry_bytes,
    )
    source = _parse_canonical_registry_bytes(source_registry_bytes, label="future source registry")
    try:
        forcing = ForcingArm(source.get("forcing"))
    except (TypeError, ValueError) as exc:
        raise ContractError("future source registry forcing is invalid") from exc
    source_sha256 = hashlib.sha256(source_registry_bytes).hexdigest()

    if forcing is ForcingArm.F0:
        if (
            set(source)
            != {
                "format",
                "forcing",
                "lead",
                "forecast_keys_sha256",
                "key_authority_binding_sha256",
                "semantic_fold_sha256",
                "records",
            }
            or source.get("format") != FUTURE_SOURCE_REGISTRY_FORMAT
        ):
            raise ContractError("F0 future source registry schema mismatch")
        lead = source.get("lead")
        if type(lead) is not int or lead not in LEADS:
            raise ContractError(f"future registry lead must be one of {LEADS}")
        if lead != key_authority.records[0].lead:
            raise ContractError("F0 lead does not match structured ForecastKeys")
        if (
            source.get("forecast_keys_sha256") != key_authority.forecast_keys_sha256
            or source.get("key_authority_binding_sha256")
            != key_authority.key_authority_binding_sha256
            or source.get("semantic_fold_sha256") != key_authority.semantic_fold_sha256
        ):
            raise ContractError("F0 source does not bind structured ForecastKeys/fold bytes")
        keys = tuple(record.key_id for record in key_authority.records)
        if source.get("records") is not None:
            raise ContractError("F0 source registry must not contain future records")
        if substitution_fit_source_bytes:
            raise ContractError("F0 must not retain substitution-fit source bytes")
        substitution_bytes = _canonical_json_bytes(
            {
                "format": FUTURE_SUBSTITUTION_REGISTRY_FORMAT,
                "forcing": forcing.value,
                "lead": lead,
                "key_ids": list(keys),
                "forecast_keys_sha256": key_authority.forecast_keys_sha256,
                "key_authority_binding_sha256": (key_authority.key_authority_binding_sha256),
                "semantic_fold_sha256": key_authority.semantic_fold_sha256,
                "records": None,
            }
        )
        provenance_bytes = _canonical_json_bytes(
            {
                "format": FUTURE_FIT_PROVENANCE_FORMAT,
                "fit_scope": "NOT_APPLICABLE_F0",
                "source_registry_sha256": source_sha256,
                "forecast_keys_sha256": key_authority.forecast_keys_sha256,
                "key_authority_binding_sha256": (key_authority.key_authority_binding_sha256),
                "fold_registry_sha256": key_authority.fold_registry_sha256,
                "semantic_fold_sha256": key_authority.semantic_fold_sha256,
                "substitution_fit_source_sha256": None,
            }
        )
        values = np.zeros((len(keys), FUTURE_FEATURE_POLICY.width), dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.bool_)
        substitution_cube = np.zeros(
            (len(keys), len(METEOROLOGICAL_VARIABLES), lead), dtype=np.bool_
        )
        station_ids = tuple(record.site_no for record in key_authority.records)
        issue_dates = tuple(record.issue_date for record in key_authority.records)
        target_dates = tuple(record.target_date for record in key_authority.records)
        geometry = key_authority.fold.geometry
        split_seed = key_authority.fold.split_seed
        fold_index = key_authority.fold.fold
    else:
        if type(substitution_fit_source_bytes) is not bytes or not substitution_fit_source_bytes:
            raise ContractError("F3_full requires exact non-empty substitution-fit source bytes")
        fold, huc2_by_station, climatology, fit_semantic_fold_sha256 = (
            _parse_future_fit_evidence_bytes(fold_registry_bytes, substitution_fit_source_bytes)
        )
        if (
            fold.to_record() != key_authority.fold.to_record()
            or fit_semantic_fold_sha256 != key_authority.semantic_fold_sha256
        ):
            raise ContractError(
                "F3 ForecastKeys and fit evidence do not share one semantic fold fingerprint"
            )
        parsed = _parse_future_raw_registry(
            source_registry_bytes,
            fold=fold,
            huc2_by_station=huc2_by_station,
        )
        parsed_records = tuple(
            {
                "key_id": parsed.key_ids[index],
                "site_no": parsed.site_nos[index],
                "issue_date": parsed.issue_dates[index],
                "target_date": parsed.target_dates[index],
                "lead": parsed.lead,
                "huc2": parsed.huc2s[index],
            }
            for index in range(len(parsed.key_ids))
        )
        if parsed_records != tuple(record.to_record() for record in key_authority.records):
            raise ContractError("F3 raw registry does not exactly match structured ForecastKeys")
        lead = parsed.lead
        keys = parsed.key_ids
        station_ids = parsed.site_nos
        issue_dates = parsed.issue_dates
        target_dates = parsed.target_dates
        geometry = fold.geometry
        split_seed = fold.split_seed
        fold_index = fold.fold
        pooled_climatology = {
            variable: _safe_pooled_curve(
                [climatology[station][variable] for station in fold.train_stations],
                label=f"future substitution-fit {variable}",
            )
            for variable in METEOROLOGICAL_VARIABLES
        }
        substituted_values = np.array(parsed.raw_values, copy=True)
        derived_records: list[dict[str, object]] = []
        for row, key_id in enumerate(keys):
            issue = _parse_iso_date(parsed.issue_dates[row], label="future raw issue_date")
            derived_variables: dict[str, list[dict[str, object]]] = {}
            for variable_index, variable in enumerate(METEOROLOGICAL_VARIABLES):
                derived_steps: list[dict[str, object]] = []
                for step in range(1, lead + 1):
                    valid_date = issue + timedelta(days=step)
                    substituted = bool(parsed.missing[row, variable_index, step - 1])
                    if substituted:
                        substituted_values[row, variable_index, step - 1] = pooled_climatology[
                            variable
                        ][valid_date.timetuple().tm_yday - 1]
                    value = float(substituted_values[row, variable_index, step - 1])
                    derived_steps.append(
                        {
                            "step": step,
                            "valid_date": valid_date.isoformat(),
                            "value": value,
                            "substituted": substituted,
                        }
                    )
                derived_variables[variable] = derived_steps
            derived_records.append(
                {
                    "key_id": key_id,
                    "site_no": parsed.site_nos[row],
                    "huc2": parsed.huc2s[row],
                    "issue_date": parsed.issue_dates[row],
                    "target_date": parsed.target_dates[row],
                    "variables": derived_variables,
                }
            )
        substitution_bytes = _canonical_json_bytes(
            {
                "format": FUTURE_SUBSTITUTION_REGISTRY_FORMAT,
                "forcing": forcing.value,
                "geometry": fold.geometry.value,
                "split_seed": fold.split_seed,
                "fold": fold.fold,
                "lead": lead,
                "forecast_keys_sha256": key_authority.forecast_keys_sha256,
                "key_authority_binding_sha256": (key_authority.key_authority_binding_sha256),
                "semantic_fold_sha256": key_authority.semantic_fold_sha256,
                "records": derived_records,
            }
        )
        provenance_bytes = _canonical_json_bytes(
            {
                "format": FUTURE_FIT_PROVENANCE_FORMAT,
                "fit_scope": "FOLD_TRAINING_ONLY",
                "source_registry_sha256": source_sha256,
                "forecast_keys_sha256": key_authority.forecast_keys_sha256,
                "key_authority_binding_sha256": (key_authority.key_authority_binding_sha256),
                "fold_registry_sha256": key_authority.fold_registry_sha256,
                "semantic_fold_sha256": key_authority.semantic_fold_sha256,
                "substitution_fit_source_sha256": hashlib.sha256(
                    substitution_fit_source_bytes
                ).hexdigest(),
            }
        )
        columns: list[np.ndarray] = []
        for variable_index, _variable in enumerate(METEOROLOGICAL_VARIABLES):
            trajectory = substituted_values[:, variable_index, :]
            if not np.isfinite(trajectory).all():
                raise ContractError("derived future trajectory contains a non-finite value")
            columns.extend((trajectory[:, -1], trajectory.mean(axis=1)))
        values_float64 = np.stack(columns, axis=1)
        if not np.isfinite(values_float64).all():
            raise ContractError("canonical future summaries are nonfinite or overflowed")
        values = values_float64.astype(np.float32)
        if not np.isfinite(values).all():
            raise ContractError("canonical future summaries overflow float32 model inputs")
        mask = np.ones_like(values, dtype=np.bool_)
        substitution_cube = parsed.missing

    instance = object.__new__(CanonicalFutureBatch)
    object.__setattr__(instance, "forcing", forcing)
    object.__setattr__(instance, "lead", lead)
    object.__setattr__(instance, "key_ids", keys)
    object.__setattr__(instance, "station_ids", station_ids)
    object.__setattr__(instance, "issue_dates", issue_dates)
    object.__setattr__(instance, "target_dates", target_dates)
    object.__setattr__(instance, "geometry", geometry)
    object.__setattr__(instance, "split_seed", split_seed)
    object.__setattr__(instance, "fold", fold_index)
    object.__setattr__(instance, "forecast_keys_sha256", key_authority.forecast_keys_sha256)
    object.__setattr__(
        instance,
        "key_authority_binding_sha256",
        key_authority.key_authority_binding_sha256,
    )
    object.__setattr__(instance, "fold_registry_sha256", key_authority.fold_registry_sha256)
    object.__setattr__(instance, "semantic_fold_sha256", key_authority.semantic_fold_sha256)
    object.__setattr__(instance, "feature_names", FUTURE_FEATURE_POLICY.feature_names(lead))
    object.__setattr__(instance, "values", _readonly_array(values, np.float32))
    object.__setattr__(instance, "mask", _readonly_array(mask, np.bool_))
    object.__setattr__(
        instance,
        "substitution_mask",
        _readonly_array(substitution_cube, np.bool_),
    )
    substitution_sha256 = hashlib.sha256(substitution_bytes).hexdigest()
    fit_provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    value_sha256 = _array_sha256("values", values)
    mask_sha256 = _array_sha256("mask", mask)
    substitution_mask_sha256 = _array_sha256("substitution_mask", substitution_cube)
    factory_seal_sha256 = canonical_sha256(
        {
            "format": FUTURE_FACTORY_SEAL_FORMAT,
            "forcing": forcing.value,
            "lead": lead,
            "key_ids": list(keys),
            "station_ids": list(station_ids),
            "issue_dates": list(issue_dates),
            "target_dates": list(target_dates),
            "geometry": None if geometry is None else geometry.value,
            "split_seed": split_seed,
            "fold": fold_index,
            "forecast_keys_sha256": key_authority.forecast_keys_sha256,
            "key_authority_binding_sha256": key_authority.key_authority_binding_sha256,
            "fold_registry_sha256": key_authority.fold_registry_sha256,
            "semantic_fold_sha256": key_authority.semantic_fold_sha256,
            "source_value_sha256": source_sha256,
            "substitution_registry_sha256": substitution_sha256,
            "fit_provenance_sha256": fit_provenance_sha256,
            "canonical_value_sha256": value_sha256,
            "canonical_mask_sha256": mask_sha256,
            "substitution_mask_sha256": substitution_mask_sha256,
            "construction_status": FUTURE_STRUCTURAL_STATUS,
        }
    )
    object.__setattr__(instance, "source_value_sha256", source_sha256)
    object.__setattr__(instance, "substitution_registry_sha256", substitution_sha256)
    object.__setattr__(instance, "fit_provenance_sha256", fit_provenance_sha256)
    object.__setattr__(instance, "canonical_value_sha256", value_sha256)
    object.__setattr__(instance, "canonical_mask_sha256", mask_sha256)
    object.__setattr__(instance, "substitution_mask_sha256", substitution_mask_sha256)
    object.__setattr__(instance, "factory_seal_sha256", factory_seal_sha256)
    object.__setattr__(instance, "construction_status", FUTURE_STRUCTURAL_STATUS)
    object.__setattr__(instance, "_source_registry_bytes", source_registry_bytes)
    object.__setattr__(instance, "_substitution_registry_bytes", substitution_bytes)
    object.__setattr__(instance, "_fit_provenance_bytes", provenance_bytes)
    object.__setattr__(
        instance,
        "_forecast_keys_registry_bytes",
        forecast_keys_registry_bytes,
    )
    object.__setattr__(instance, "_fold_registry_bytes", fold_registry_bytes)
    object.__setattr__(
        instance,
        "_substitution_fit_source_bytes",
        substitution_fit_source_bytes,
    )
    return instance


def validate_canonical_future_batch(batch: CanonicalFutureBatch) -> None:
    """Recompute all tensors and hashes from retained exact registry bytes."""

    if type(batch) is not CanonicalFutureBatch:
        raise ContractError("future batch must be an exact CanonicalFutureBatch")
    try:
        expected = _materialize_canonical_future_batch(
            source_registry_bytes=batch._source_registry_bytes,
            fold_registry_bytes=batch._fold_registry_bytes,
            forecast_keys_registry_bytes=batch._forecast_keys_registry_bytes,
            substitution_fit_source_bytes=batch._substitution_fit_source_bytes,
        )
    except AttributeError as exc:
        raise ContractError("future batch lacks exact registry-byte evidence") from exc
    scalar_fields = (
        "forcing",
        "lead",
        "key_ids",
        "station_ids",
        "issue_dates",
        "target_dates",
        "geometry",
        "split_seed",
        "fold",
        "forecast_keys_sha256",
        "key_authority_binding_sha256",
        "fold_registry_sha256",
        "semantic_fold_sha256",
        "feature_names",
        "source_value_sha256",
        "substitution_registry_sha256",
        "fit_provenance_sha256",
        "canonical_value_sha256",
        "canonical_mask_sha256",
        "substitution_mask_sha256",
        "factory_seal_sha256",
        "construction_status",
    )
    if any(getattr(batch, name) != getattr(expected, name) for name in scalar_fields):
        raise ContractError("future batch metadata differs from exact registry-byte evidence")
    if (
        batch._substitution_registry_bytes != expected._substitution_registry_bytes
        or batch._fit_provenance_bytes != expected._fit_provenance_bytes
    ):
        raise ContractError("future batch derived audit bytes differ from recomputation")
    for name in ("values", "mask", "substitution_mask"):
        observed = _require_durable_immutable_array(
            getattr(batch, name),
            label=f"future batch {name}",
        )
        wanted = getattr(expected, name)
        if (
            observed.dtype != wanted.dtype
            or observed.shape != wanted.shape
            or not np.array_equal(observed, wanted)
        ):
            raise ContractError(f"future batch {name} differs from exact registry-byte evidence")


def build_canonical_future_batch(
    forcing: ForcingArm,
    lead: int,
    *,
    key_ids: Sequence[str] | None = None,
    forecast_keys_registry_bytes: bytes | None = None,
    fold_registry_bytes: bytes | None = None,
    raw_future_registry_bytes: bytes | None = None,
    trajectories: Mapping[str, np.ndarray] | None = None,
    substitution_masks: Mapping[str, np.ndarray] | None = None,
    substitution_fit_source_bytes: bytes | None = None,
) -> CanonicalFutureBatch:
    """Derive F3 summaries and flags solely from exact raw and fit-evidence bytes."""

    _require_enum(forcing, ForcingArm, "forcing")
    if type(lead) is not int or lead not in LEADS:
        raise ContractError(f"lead must be one of {LEADS}")
    if trajectories is not None or substitution_masks is not None:
        raise ContractError(
            "caller-supplied substituted trajectories or substitution flags are forbidden"
        )
    if key_ids is not None:
        raise ContractError("future key_ids must be derived from structured ForecastKeys")
    if type(forecast_keys_registry_bytes) is not bytes or not forecast_keys_registry_bytes:
        raise ContractError("all forcing arms require exact structured ForecastKeys bytes")
    if type(fold_registry_bytes) is not bytes or not fold_registry_bytes:
        raise ContractError("all forcing arms require exact shared fold-registry bytes")
    key_authority = _materialize_forecast_key_authority(
        fold_registry_bytes=fold_registry_bytes,
        forecast_keys_registry_bytes=forecast_keys_registry_bytes,
    )
    if key_authority.records[0].lead != lead:
        raise ContractError("future lead does not match structured ForecastKeys")
    if forcing is ForcingArm.F0:
        if raw_future_registry_bytes is not None:
            raise ContractError("F0 must not receive a raw future registry")
        if substitution_fit_source_bytes is not None:
            raise ContractError("F0 must not receive substitution-fit source evidence")
        source_bytes = _canonical_json_bytes(
            {
                "format": FUTURE_SOURCE_REGISTRY_FORMAT,
                "forcing": forcing.value,
                "lead": lead,
                "forecast_keys_sha256": key_authority.forecast_keys_sha256,
                "key_authority_binding_sha256": (key_authority.key_authority_binding_sha256),
                "semantic_fold_sha256": key_authority.semantic_fold_sha256,
                "records": None,
            }
        )
        return _materialize_canonical_future_batch(
            source_registry_bytes=source_bytes,
            fold_registry_bytes=fold_registry_bytes,
            forecast_keys_registry_bytes=forecast_keys_registry_bytes,
            substitution_fit_source_bytes=b"",
        )
    if type(raw_future_registry_bytes) is not bytes or not raw_future_registry_bytes:
        raise ContractError("F3_full requires exact non-empty raw future registry bytes")
    if type(substitution_fit_source_bytes) is not bytes or not substitution_fit_source_bytes:
        raise ContractError("F3_full requires exact non-empty substitution-fit source bytes")
    batch = _materialize_canonical_future_batch(
        source_registry_bytes=raw_future_registry_bytes,
        fold_registry_bytes=fold_registry_bytes,
        forecast_keys_registry_bytes=forecast_keys_registry_bytes,
        substitution_fit_source_bytes=substitution_fit_source_bytes,
    )
    if batch.forcing is not forcing or batch.lead != lead:
        raise ContractError("F3_full raw registry forcing/lead mismatch")
    return batch


# ---------------------------------------------------------------------------
# Point-only objective and fit-budget policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointObjectivePolicy:
    status: str = POLICY_STATUS
    objective_id: str = "point_mse_only_v1"
    formula: str = "mean((point_prediction-target)^2)"
    trained_output_heads: tuple[str, ...] = ("point",)
    forbidden_auxiliary_losses: tuple[str, ...] = (
        "quantile_pinball",
        "event_bce",
        "quantile_crossing",
        "residual_magnitude",
    )
    validation_selection_metric: str = "unweighted_station_macro_rmse"
    lead_training_unit: str = "one_model_per_lead"
    model_fit_seed_aggregation: str = "MUST_BE_FROZEN_IN_V4_SEAL"

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "objective_id": self.objective_id,
            "formula": self.formula,
            "trained_output_heads": list(self.trained_output_heads),
            "forbidden_auxiliary_losses": list(self.forbidden_auxiliary_losses),
            "validation_selection_metric": self.validation_selection_metric,
            "lead_training_unit": self.lead_training_unit,
            "model_fit_seed_aggregation": self.model_fit_seed_aggregation,
        }


@dataclass(frozen=True, slots=True)
class FitBudgetPolicy:
    status: str = POLICY_STATUS
    model_fit_seeds: tuple[int, ...] = MODEL_FIT_SEEDS
    hyperparameter_candidates_per_architecture: int = 1
    optimizer: str = "AdamW"
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    batch_size: int = 1536
    max_epochs: int = 80
    patience: int = 12
    gradient_clip_norm: float = 1.0
    station_sampling: str = "station_balanced"
    active_parameter_tolerance_fraction: float = 0.02
    active_parameter_definition: str = (
        "requires_grad parameters on paths contributing to point_prediction; "
        "quantile/event-only heads excluded"
    )
    stopping_information: str = (
        "same 2016-2017 in-fold-station validation keys and station-macro RMSE"
    )
    optimizer_step_matching_rule: str = (
        "paired architectures receive identical eligible examples, batch size, "
        "maximum update cap, patience, and candidate count"
    )

    def __post_init__(self) -> None:
        if self.model_fit_seeds != MODEL_FIT_SEEDS:
            raise ContractError("model-fit seeds must be exactly 0,1,2,3,4")
        if self.hyperparameter_candidates_per_architecture != 1:
            raise ContractError("Phase-1 policy freezes one candidate per architecture")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ContractError("fit budget counts must be positive")
        if not 0.0 <= self.active_parameter_tolerance_fraction <= 0.02:
            raise ContractError("active parameter tolerance cannot exceed two percent")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "model_fit_seeds": list(self.model_fit_seeds),
            "hyperparameter_candidates_per_architecture": (
                self.hyperparameter_candidates_per_architecture
            ),
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "station_sampling": self.station_sampling,
            "active_parameter_tolerance_fraction": (self.active_parameter_tolerance_fraction),
            "active_parameter_definition": self.active_parameter_definition,
            "stopping_information": self.stopping_information,
            "optimizer_step_matching_rule": self.optimizer_step_matching_rule,
        }


POINT_OBJECTIVE_POLICY = PointObjectivePolicy()
FIT_BUDGET_POLICY = FitBudgetPolicy()
ACTIVE_PARAMETER_KEYS = (
    "plain_TCN:unbounded",
    "ThermoRoute:unbounded",
    "ThermoRoute:bounded",
)


def validate_point_objective_policy(policy: PointObjectivePolicy) -> None:
    if not isinstance(policy, PointObjectivePolicy):
        raise TypeError("objective policy must be a PointObjectivePolicy")
    if policy != POINT_OBJECTIVE_POLICY:
        raise ContractError("point objective policy differs from the Phase-1 frozen proposal")
    if policy.trained_output_heads != ("point",) or not policy.forbidden_auxiliary_losses:
        raise ContractError("only the point head may contribute to the objective")


def validate_fit_budget_policy(policy: FitBudgetPolicy) -> None:
    if not isinstance(policy, FitBudgetPolicy):
        raise TypeError("fit budget policy must be a FitBudgetPolicy")
    if policy != FIT_BUDGET_POLICY:
        raise ContractError("fit budget policy differs from the Phase-1 frozen proposal")


def point_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    _reject_masked_array(prediction, label="point prediction")
    _reject_masked_array(target, label="point target")
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape or prediction_array.size == 0:
        raise ContractError("point prediction and target must have the same non-empty shape")
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ContractError("point-MSE inputs must be finite")
    return float(np.mean(np.square(prediction_array - target_array)))


def validate_active_parameter_counts(counts: Mapping[str, int]) -> None:
    if set(counts) != set(ACTIVE_PARAMETER_KEYS):
        raise ContractError(f"active parameter count keys must be exactly {ACTIVE_PARAMETER_KEYS}")
    values = {key: int(value) for key, value in counts.items()}
    if any(type(counts[key]) is not int or values[key] <= 0 for key in counts):
        raise ContractError("active parameter counts must be positive integers")
    if values["ThermoRoute:bounded"] != values["ThermoRoute:unbounded"]:
        raise ContractError("bounded/unbounded ThermoRoute must have identical active parameters")
    reference = values["ThermoRoute:unbounded"]
    relative = abs(values["plain_TCN:unbounded"] / reference - 1.0)
    if relative > FIT_BUDGET_POLICY.active_parameter_tolerance_fraction:
        raise ContractError(f"plain_TCN/ThermoRoute active parameters differ by {relative:.6f}")


# ---------------------------------------------------------------------------
# Outcome-separated L0/L2 input contract
# ---------------------------------------------------------------------------


def _readonly_vector(value: object, *, n: int, label: str) -> np.ndarray:
    array = _readonly_array(value, np.float32)
    if array.shape != (n,):
        raise ContractError(f"{label} must have shape {(n,)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ContractError(f"{label} must be finite")
    return array


def _readonly_matrix(
    value: object, *, n: int, width: int | None, label: str, dtype: type = np.float32
) -> np.ndarray:
    array = _readonly_array(value, dtype)
    if array.ndim != 2 or array.shape[0] != n or (width is not None and array.shape[1] != width):
        expected = f"[N,{width}]" if width is not None else "[N,K]"
        raise ContractError(f"{label} must have shape {expected}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ContractError(f"{label} must be finite")
    return array


@dataclass(frozen=True, slots=True)
class LabelTargets:
    """Supervision and station grouping; never passed to a model forward call."""

    key_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    target: np.ndarray
    forecast_keys_registry_bytes: bytes
    fold_registry_bytes: bytes
    content_sha256: str = field(init=False)
    _forecast_key_authority: ForecastKeyAuthority = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        authority = _materialize_forecast_key_authority(
            fold_registry_bytes=self.fold_registry_bytes,
            forecast_keys_registry_bytes=self.forecast_keys_registry_bytes,
        )
        keys = tuple(self.key_ids)
        stations = tuple(self.station_ids)
        if not keys or len(keys) != len(set(keys)):
            raise ContractError("label key_ids must be non-empty and unique")
        if len(stations) != len(keys):
            raise ContractError("label station_ids must align with key_ids")
        for station in stations:
            _require_formal_site_id(station, label="label station_id")
        if keys != tuple(record.key_id for record in authority.records) or stations != tuple(
            record.site_no for record in authority.records
        ):
            raise ContractError("label key/station vectors differ from structured ForecastKeys")
        target = _readonly_vector(self.target, n=len(keys), label="target")
        object.__setattr__(self, "key_ids", keys)
        object.__setattr__(self, "station_ids", stations)
        object.__setattr__(self, "target", target)
        object.__setattr__(
            self, "forecast_keys_registry_bytes", bytes(self.forecast_keys_registry_bytes)
        )
        object.__setattr__(self, "fold_registry_bytes", bytes(self.fold_registry_bytes))
        object.__setattr__(self, "_forecast_key_authority", authority)
        object.__setattr__(self, "content_sha256", _label_integrity_sha256(self))


@dataclass(frozen=True, slots=True)
class RawInputSources:
    """Input-side arrays built without consulting future WTEMP labels."""

    key_ids: tuple[str, ...]
    station_identity: tuple[str, ...]
    history_timestamps: tuple[tuple[str, ...], ...]
    history_values: np.ndarray
    history_mask: np.ndarray
    wtemp_t: np.ndarray
    clim_t: np.ndarray
    clim_tgt: np.ndarray
    damped_prior: np.ndarray
    phys_std: np.ndarray
    logflowz: np.ndarray
    season: np.ndarray
    gate: np.ndarray
    forecast_keys_registry_bytes: bytes
    fold_registry_bytes: bytes
    content_sha256: str = field(init=False)
    _forecast_key_authority: ForecastKeyAuthority = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        authority = _materialize_forecast_key_authority(
            fold_registry_bytes=self.fold_registry_bytes,
            forecast_keys_registry_bytes=self.forecast_keys_registry_bytes,
        )
        keys = tuple(self.key_ids)
        station_identity = tuple(self.station_identity)
        if not keys or len(keys) != len(set(keys)):
            raise ContractError("input key_ids must be non-empty and unique")
        n = len(keys)
        if len(station_identity) != n:
            raise ContractError("station_identity must align with input keys")
        for station in station_identity:
            _require_formal_site_id(station, label="input station_identity")
        if keys != tuple(
            record.key_id for record in authority.records
        ) or station_identity != tuple(record.site_no for record in authority.records):
            raise ContractError("input key/station vectors differ from structured ForecastKeys")
        try:
            history_timestamps = tuple(tuple(row) for row in self.history_timestamps)
        except TypeError as exc:
            raise ContractError("history timestamps must be a rectangular sequence") from exc
        if len(history_timestamps) != n:
            raise ContractError("history timestamp rows must align with ForecastKeys")
        for record, row in zip(authority.records, history_timestamps, strict=True):
            issue = _parse_iso_date(record.issue_date, label="history issue_date")
            expected = tuple(
                (issue - timedelta(days=HISTORY_LENGTH - 1 - offset)).isoformat()
                for offset in range(HISTORY_LENGTH)
            )
            if row != expected or any(type(value) is not str for value in row):
                raise ContractError(
                    "history timestamps must be exact contiguous issue-31..issue dates"
                )
        history = _readonly_array(self.history_values, np.float32)
        mask = _readonly_array(
            _strict_bool_array(self.history_mask, label="history_mask"),
            np.bool_,
        )
        expected_history_shape = (n, HISTORY_LENGTH, len(HISTORY_VARIABLES))
        if history.shape != expected_history_shape:
            raise ContractError(f"history_values must have shape {expected_history_shape}")
        if mask.shape != history.shape:
            raise ContractError("history_mask must match history_values")
        if not np.isfinite(history[mask]).all():
            raise ContractError("observed history values must be finite")
        phys = _readonly_matrix(self.phys_std, n=n, width=4, label="phys_std")
        season = _readonly_matrix(self.season, n=n, width=2, label="season")
        gate = _readonly_matrix(self.gate, n=n, width=len(GATE_FEATURES), label="gate")
        object.__setattr__(self, "key_ids", keys)
        object.__setattr__(self, "station_identity", station_identity)
        object.__setattr__(self, "history_timestamps", history_timestamps)
        object.__setattr__(self, "history_values", history)
        object.__setattr__(self, "history_mask", mask)
        object.__setattr__(self, "wtemp_t", _readonly_vector(self.wtemp_t, n=n, label="wtemp_t"))
        object.__setattr__(self, "clim_t", _readonly_vector(self.clim_t, n=n, label="clim_t"))
        object.__setattr__(self, "clim_tgt", _readonly_vector(self.clim_tgt, n=n, label="clim_tgt"))
        object.__setattr__(
            self,
            "damped_prior",
            _readonly_vector(self.damped_prior, n=n, label="damped_prior"),
        )
        object.__setattr__(self, "phys_std", phys)
        object.__setattr__(self, "logflowz", _readonly_vector(self.logflowz, n=n, label="logflowz"))
        object.__setattr__(self, "season", season)
        object.__setattr__(self, "gate", gate)
        object.__setattr__(
            self, "forecast_keys_registry_bytes", bytes(self.forecast_keys_registry_bytes)
        )
        object.__setattr__(self, "fold_registry_bytes", bytes(self.fold_registry_bytes))
        object.__setattr__(self, "_forecast_key_authority", authority)
        object.__setattr__(self, "content_sha256", _raw_input_integrity_sha256(self))


def _require_durable_immutable_array(value: object, *, label: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.flags.writeable:
        raise ContractError(f"{label} must be a durable immutable NumPy array")
    try:
        value.setflags(write=True)
    except ValueError:
        return value
    value.setflags(write=False)
    raise ContractError(f"{label} write protection can be re-enabled by a consumer")


def _label_integrity_sha256(labels: LabelTargets) -> str:
    return canonical_sha256(
        {
            "key_ids": list(labels.key_ids),
            "station_ids": list(labels.station_ids),
            "forecast_keys_sha256": labels._forecast_key_authority.forecast_keys_sha256,
            "key_authority_binding_sha256": (
                labels._forecast_key_authority.key_authority_binding_sha256
            ),
            "semantic_fold_sha256": labels._forecast_key_authority.semantic_fold_sha256,
            "target_sha256": _array_sha256("label_target", labels.target),
        }
    )


def validate_label_targets(labels: LabelTargets) -> None:
    if type(labels) is not LabelTargets:
        raise ContractError("labels must have the exact protected LabelTargets type")
    authority = _materialize_forecast_key_authority(
        fold_registry_bytes=labels.fold_registry_bytes,
        forecast_keys_registry_bytes=labels.forecast_keys_registry_bytes,
    )
    validate_forecast_key_authority(labels._forecast_key_authority)
    if (
        labels.key_ids != tuple(record.key_id for record in authority.records)
        or labels.station_ids != tuple(record.site_no for record in authority.records)
        or labels._forecast_key_authority.forecast_keys_sha256 != authority.forecast_keys_sha256
        or labels._forecast_key_authority.key_authority_binding_sha256
        != authority.key_authority_binding_sha256
        or labels._forecast_key_authority.semantic_fold_sha256 != authority.semantic_fold_sha256
    ):
        raise ContractError("label ForecastKeys authority differs from retained exact bytes")
    target = _require_durable_immutable_array(labels.target, label="label target")
    if target.shape != (len(labels.key_ids),) or not np.isfinite(target).all():
        raise ContractError("label target shape/content mismatch")
    if labels.content_sha256 != _label_integrity_sha256(labels):
        raise ContractError("label target or authority digest changed after construction")


def _raw_input_integrity_sha256(sources: RawInputSources) -> str:
    array_digests = {
        name: _array_sha256(name, getattr(sources, name))
        for name in (
            "history_values",
            "history_mask",
            "wtemp_t",
            "clim_t",
            "clim_tgt",
            "damped_prior",
            "phys_std",
            "logflowz",
            "season",
            "gate",
        )
    }
    return canonical_sha256(
        {
            "key_ids": list(sources.key_ids),
            "station_identity": list(sources.station_identity),
            "history_timestamps": [list(row) for row in sources.history_timestamps],
            "forecast_keys_sha256": sources._forecast_key_authority.forecast_keys_sha256,
            "key_authority_binding_sha256": (
                sources._forecast_key_authority.key_authority_binding_sha256
            ),
            "semantic_fold_sha256": sources._forecast_key_authority.semantic_fold_sha256,
            "arrays": array_digests,
        }
    )


def validate_raw_input_sources(sources: RawInputSources) -> None:
    if type(sources) is not RawInputSources:
        raise ContractError("input sources must have the exact protected RawInputSources type")
    authority = _materialize_forecast_key_authority(
        fold_registry_bytes=sources.fold_registry_bytes,
        forecast_keys_registry_bytes=sources.forecast_keys_registry_bytes,
    )
    validate_forecast_key_authority(sources._forecast_key_authority)
    if (
        sources.key_ids != tuple(record.key_id for record in authority.records)
        or sources.station_identity != tuple(record.site_no for record in authority.records)
        or sources._forecast_key_authority.forecast_keys_sha256 != authority.forecast_keys_sha256
        or sources._forecast_key_authority.key_authority_binding_sha256
        != authority.key_authority_binding_sha256
        or sources._forecast_key_authority.semantic_fold_sha256 != authority.semantic_fold_sha256
    ):
        raise ContractError("input ForecastKeys authority differs from retained exact bytes")
    for record, row in zip(authority.records, sources.history_timestamps, strict=True):
        issue = date.fromisoformat(record.issue_date)
        expected = tuple(
            (issue - timedelta(days=HISTORY_LENGTH - 1 - offset)).isoformat()
            for offset in range(HISTORY_LENGTH)
        )
        if row != expected:
            raise ContractError("history timestamps changed or are not issue-31..issue")
    for name in (
        "history_values",
        "history_mask",
        "wtemp_t",
        "clim_t",
        "clim_tgt",
        "damped_prior",
        "phys_std",
        "logflowz",
        "season",
        "gate",
    ):
        _require_durable_immutable_array(getattr(sources, name), label=f"input {name}")
    expected_shape = (len(sources.key_ids), HISTORY_LENGTH, len(HISTORY_VARIABLES))
    if (
        sources.history_values.shape != expected_shape
        or sources.history_mask.shape != expected_shape
    ):
        raise ContractError("history arrays changed from exact [N,32,7] shape")
    if sources.content_sha256 != _raw_input_integrity_sha256(sources):
        raise ContractError(
            "input arrays, timestamps, or authority digest changed after construction"
        )


POOLED_L2_DATE_RULE = (
    "one_based_gregorian_day_of_year_index_into_366_value_training_station_climatology"
)
POOLED_L2_TRAINING_PERIOD = "2006-01-01/2015-12-31"
POOLED_L2_ATTESTATION = "RECOMPUTED_TRAINING_ONLY_NOT_EXECUTION_AUTHORITY"
POOLED_L2_FOLD_REGISTRY_FORMAT = SHARED_FOLD_REGISTRY_FORMAT
POOLED_L2_SOURCE_REGISTRY_FORMAT = "thermoroute.neural-l2-daily-wtemp-source.v3"
POOLED_L2_KEY_REGISTRY_FORMAT = "thermoroute.forecast-keys.synthetic-contract.v5"
POOLED_L2_HUC2_REGISTRY_FORMAT = "thermoroute.neural-l2-huc2-registry.v2"
POOLED_L2_FACTORY_SEAL_FORMAT = "thermoroute.neural-l2-factory-seal.v4"


def canonical_forecast_key_id(
    *,
    site_no: str,
    issue_date: str,
    target_date: str,
    lead: int,
) -> str:
    """Return the formal v4 common-key tuple digest after domain validation."""

    canonical_site = _require_formal_site_id(site_no, label="ForecastKey site_no")
    issue = _parse_iso_date(issue_date, label="ForecastKey issue_date")
    target = _parse_iso_date(target_date, label="ForecastKey target_date")
    if type(lead) is not int or lead not in LEADS:
        raise ContractError(f"ForecastKey lead must be one of {LEADS}")
    if target - issue != timedelta(days=lead):
        raise ContractError("ForecastKey target_date must equal issue_date + lead")
    if issue < EVALUATION_START_DATE or target > EVALUATION_END_DATE:
        raise ContractError(
            "ForecastKey must stay inside the exact 2021-01-01..2023-12-31 evaluation domain"
        )
    payload = f"temporal|known_site|{canonical_site}|{issue_date}|{target_date}|{lead}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _forecast_key_authority_binding_sha256(
    *,
    geometry: Geometry,
    split_seed: int,
    fold_index: int,
    records: Sequence[ForecastKeyRecord],
) -> str:
    return canonical_sha256(
        {
            "format": FORECAST_KEY_AUTHORITY_BINDING_FORMAT,
            "authority_kind": SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND,
            "production_authority_status": PRODUCTION_COMMON_KEY_AUTHORITY_STATUS,
            "canonical_registry_path": CANONICAL_COMMON_KEY_REGISTRY_PATH,
            "evaluation_period": [
                EVALUATION_START_DATE.isoformat(),
                EVALUATION_END_DATE.isoformat(),
            ],
            "key_id_formula": CANONICAL_FORECAST_KEY_FORMULA,
            "geometry": geometry.value,
            "split_seed": split_seed,
            "fold": fold_index,
            "records": [record.to_record() for record in records],
        }
    )


@dataclass(frozen=True, slots=True)
class ForecastKeyRecord:
    key_id: str
    site_no: str
    issue_date: str
    target_date: str
    lead: int
    huc2: str

    def __post_init__(self) -> None:
        if type(self.huc2) is not str or not self.huc2:
            raise ContractError("ForecastKey huc2 must be a non-empty string")
        expected_key_id = canonical_forecast_key_id(
            site_no=self.site_no,
            issue_date=self.issue_date,
            target_date=self.target_date,
            lead=self.lead,
        )
        if self.key_id != expected_key_id:
            raise ContractError("ForecastKey key_id does not hash its canonical tuple")

    def to_record(self) -> dict[str, object]:
        return {
            "key_id": self.key_id,
            "site_no": self.site_no,
            "issue_date": self.issue_date,
            "target_date": self.target_date,
            "lead": self.lead,
            "huc2": self.huc2,
        }


@dataclass(frozen=True, slots=True, init=False)
class PooledL2Lineage:
    geometry: Geometry
    split_seed: int
    fold: int
    training_stations: tuple[str, ...]
    held_stations: tuple[str, ...]
    date_rule: str
    training_period: str
    fold_registry_sha256: str
    semantic_fold_sha256: str
    training_climatology_source_sha256: str
    key_registry_sha256: str
    key_authority_binding_sha256: str
    huc2_registry_sha256: str
    attestation: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("PooledL2Lineage cannot be constructed directly")

    def _validate(self) -> None:
        _require_enum(self.geometry, Geometry, "pooled lineage geometry")
        if type(self.fold) is not int or self.fold not in FOLDS:
            raise ContractError(f"pooled lineage fold must be one of {FOLDS}")
        if type(self.split_seed) is not int:
            raise TypeError("pooled lineage split_seed must be an int")
        if self.geometry is Geometry.RANDOM:
            if self.split_seed not in RANDOM_SPLIT_SEEDS:
                raise ContractError("pooled random lineage has an undeclared split seed")
        elif self.split_seed != REGION_SPLIT_SEED:
            raise ContractError("pooled regional lineage must use split_seed=0")
        training = _canonical_stations(self.training_stations, label="lineage training stations")
        held = _canonical_stations(self.held_stations, label="lineage held stations")
        validate_station_disjoint(training, held)
        if self.date_rule != POOLED_L2_DATE_RULE:
            raise ContractError("pooled L2 date rule differs from the Phase-1.1 contract")
        if self.training_period != POOLED_L2_TRAINING_PERIOD:
            raise ContractError("pooled L2 training period differs from the Phase-1.1 contract")
        if self.attestation != POOLED_L2_ATTESTATION:
            raise ContractError("pooled L2 lineage lacks the non-authoritative structural status")
        for label in (
            "fold_registry_sha256",
            "semantic_fold_sha256",
            "training_climatology_source_sha256",
            "key_registry_sha256",
            "key_authority_binding_sha256",
            "huc2_registry_sha256",
        ):
            _validate_sha256(getattr(self, label), label=label)
        object.__setattr__(self, "training_stations", training)
        object.__setattr__(self, "held_stations", held)

    def to_record(self) -> dict[str, object]:
        return {
            "geometry": self.geometry.value,
            "split_seed": self.split_seed,
            "fold": self.fold,
            "training_stations": list(self.training_stations),
            "held_stations": list(self.held_stations),
            "date_rule": self.date_rule,
            "training_period": self.training_period,
            "fold_registry_sha256": self.fold_registry_sha256,
            "semantic_fold_sha256": self.semantic_fold_sha256,
            "training_climatology_source_sha256": self.training_climatology_source_sha256,
            "key_registry_sha256": self.key_registry_sha256,
            "key_authority_binding_sha256": self.key_authority_binding_sha256,
            "huc2_registry_sha256": self.huc2_registry_sha256,
            "attestation": self.attestation,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_record())


def _materialize_pooled_l2_lineage(
    *,
    geometry: Geometry,
    split_seed: int,
    fold: int,
    training_stations: tuple[str, ...],
    held_stations: tuple[str, ...],
    fold_registry_sha256: str,
    semantic_fold_sha256: str,
    training_climatology_source_sha256: str,
    key_registry_sha256: str,
    key_authority_binding_sha256: str,
    huc2_registry_sha256: str,
) -> PooledL2Lineage:
    instance = object.__new__(PooledL2Lineage)
    for name, value in {
        "geometry": geometry,
        "split_seed": split_seed,
        "fold": fold,
        "training_stations": training_stations,
        "held_stations": held_stations,
        "date_rule": POOLED_L2_DATE_RULE,
        "training_period": POOLED_L2_TRAINING_PERIOD,
        "fold_registry_sha256": fold_registry_sha256,
        "semantic_fold_sha256": semantic_fold_sha256,
        "training_climatology_source_sha256": training_climatology_source_sha256,
        "key_registry_sha256": key_registry_sha256,
        "key_authority_binding_sha256": key_authority_binding_sha256,
        "huc2_registry_sha256": huc2_registry_sha256,
        "attestation": POOLED_L2_ATTESTATION,
    }.items():
        object.__setattr__(instance, name, value)
    instance._validate()
    return instance


@dataclass(frozen=True, slots=True, init=False)
class PooledL2Context:
    """Fold-aware L2 values recomputable from retained exact registry bytes.

    This object deliberately carries no execution-valid attestation.  A future
    sealed loader must bind the same bytes to execution authority before they
    can be used for model training.
    """

    key_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    issue_dates: tuple[str, ...]
    target_dates: tuple[str, ...]
    lead: int
    clim_t: np.ndarray
    clim_tgt: np.ndarray
    lineage: PooledL2Lineage
    construction_status: str
    factory_seal_sha256: str
    _fold_registry_bytes: bytes = field(repr=False, compare=False)
    _training_source_registry_bytes: bytes = field(repr=False, compare=False)
    _key_registry_bytes: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "PooledL2Context cannot be constructed directly; use "
            "build_training_only_pooled_l2_context"
        )

    @property
    def execution_validated(self) -> bool:
        """Synthetic structural construction is never execution authority."""

        return False


def build_pooled_l2_fold_registry_bytes(
    folds: Sequence[FoldDefinition],
    huc2_by_station: Mapping[str, str],
) -> bytes:
    """Build a canonical structured fold/HUC registry for contract tests."""

    candidates = tuple(folds)
    if not candidates or any(type(fold) is not FoldDefinition for fold in candidates):
        raise ContractError("pooled L2 fold registry requires typed folds")
    stations = tuple(sorted(set(candidates[0].train_stations) | set(candidates[0].hold_stations)))
    for station in stations:
        _require_formal_site_id(station, label="shared fold-registry station identifier")
    huc_items = _canonical_huc2_items(stations, huc2_by_station)
    for fold in candidates:
        if set(fold.train_stations) | set(fold.hold_stations) != set(stations):
            raise ContractError("pooled L2 folds do not share one station universe")
        validate_station_disjoint(fold.train_stations, fold.hold_stations)
        if fold.geometry is Geometry.REGION:
            validate_huc_disjoint(fold.train_stations, fold.hold_stations, dict(huc_items))
    identities = {(fold.geometry, fold.split_seed, fold.fold) for fold in candidates}
    if len(identities) != len(candidates):
        raise ContractError("pooled L2 fold registry contains duplicate fold identities")
    return _canonical_json_bytes(
        {
            "format": POOLED_L2_FOLD_REGISTRY_FORMAT,
            "stations": [{"site_no": station, "huc2": huc2} for station, huc2 in huc_items],
            "folds": [fold.to_record() for fold in candidates],
        }
    )


def build_forecast_keys_registry_bytes(
    fold: FoldDefinition,
    records: Sequence[ForecastKeyRecord],
) -> bytes:
    """Serialize explicit synthetic-only keys under the formal common-key semantics.

    This helper is only a structural test authority.  It cannot load or attest
    the production v4 common-key registry and therefore cannot authorize a run.
    """

    if type(fold) is not FoldDefinition:
        raise TypeError("ForecastKeys registry requires a FoldDefinition")
    for station in (*fold.train_stations, *fold.hold_stations):
        _require_formal_site_id(station, label="ForecastKeys fold station identifier")
    candidates = tuple(records)
    if not candidates or any(type(record) is not ForecastKeyRecord for record in candidates):
        raise ContractError("ForecastKeys registry requires typed non-empty records")
    key_ids = tuple(record.key_id for record in candidates)
    canonical_order = tuple(
        (record.site_no, record.issue_date, record.target_date, record.lead, record.key_id)
        for record in candidates
    )
    if canonical_order != tuple(sorted(canonical_order)) or len(key_ids) != len(set(key_ids)):
        raise ContractError("ForecastKeys must have unique records in canonical tuple order")
    authority_binding_sha256 = _forecast_key_authority_binding_sha256(
        geometry=fold.geometry,
        split_seed=fold.split_seed,
        fold_index=fold.fold,
        records=candidates,
    )
    return _canonical_json_bytes(
        {
            "format": POOLED_L2_KEY_REGISTRY_FORMAT,
            "authority_kind": SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND,
            "key_authority_binding_sha256": authority_binding_sha256,
            "production_authority_status": PRODUCTION_COMMON_KEY_AUTHORITY_STATUS,
            "canonical_registry_path": CANONICAL_COMMON_KEY_REGISTRY_PATH,
            "evaluation_period": [
                EVALUATION_START_DATE.isoformat(),
                EVALUATION_END_DATE.isoformat(),
            ],
            "key_id_formula": CANONICAL_FORECAST_KEY_FORMULA,
            "geometry": fold.geometry.value,
            "split_seed": fold.split_seed,
            "fold": fold.fold,
            "records": [record.to_record() for record in candidates],
        }
    )


def build_pooled_l2_training_source_registry_bytes(
    fold: FoldDefinition,
    training_station_daily_wtemp: Mapping[str, np.ndarray],
    training_source_dates_by_station: Mapping[str, Sequence[str]],
) -> bytes:
    """Bind exact daily 2006--2015 WTEMP; station DOY curves are recomputed."""

    if type(fold) is not FoldDefinition:
        raise TypeError("pooled L2 training source requires a FoldDefinition")
    for station in (*fold.train_stations, *fold.hold_stations):
        _require_formal_site_id(station, label="pooled L2 fold station identifier")
    if set(training_station_daily_wtemp) != set(fold.train_stations) or set(
        training_source_dates_by_station
    ) != set(fold.train_stations):
        raise ContractError("pooled L2 training source membership mismatch")
    expected_dates = _canonical_training_source_dates()
    records: list[dict[str, object]] = []
    for station in fold.train_stations:
        dates = tuple(training_source_dates_by_station[station])
        if any(type(value) is not str for value in dates):
            raise ContractError(f"pooled L2 source dates for {station} must be exact strings")
        if dates != expected_dates:
            raise ContractError(
                f"pooled L2 source dates for {station} are not exact 2006--2015 dates"
            )
        daily_values = _strict_float_array(
            training_station_daily_wtemp[station], label=f"pooled L2 daily WTEMP {station}"
        )
        if daily_values.shape != (len(expected_dates),) or not np.isfinite(daily_values).all():
            raise ContractError(
                f"pooled L2 daily WTEMP for {station} must have one finite value "
                "for every exact 2006--2015 date"
            )
        records.append(
            {
                "station_id": station,
                "daily_records": [
                    {"date": source_date, "wtemp": float(daily_values[row])}
                    for row, source_date in enumerate(dates)
                ],
            }
        )
    source_bytes = _canonical_json_bytes(
        {
            "format": POOLED_L2_SOURCE_REGISTRY_FORMAT,
            "training_period": POOLED_L2_TRAINING_PERIOD,
            "records": records,
        }
    )
    _parse_pooled_l2_training_source(source_bytes, fold.train_stations)
    return source_bytes


def _parse_pooled_l2_fold_registry(
    payload: bytes,
) -> tuple[tuple[FoldDefinition, ...], dict[str, str]]:
    document = _parse_canonical_registry_bytes(payload, label="pooled L2 fold registry")
    if (
        set(document) != {"format", "stations", "folds"}
        or document.get("format") != POOLED_L2_FOLD_REGISTRY_FORMAT
    ):
        raise ContractError("pooled L2 fold registry schema mismatch")
    raw_stations = document.get("stations")
    raw_folds = document.get("folds")
    if not isinstance(raw_stations, list) or not isinstance(raw_folds, list) or not raw_folds:
        raise ContractError("pooled L2 fold registry requires stations and folds")
    huc: dict[str, str] = {}
    station_order: list[str] = []
    for record in raw_stations:
        if not isinstance(record, Mapping) or set(record) != {"site_no", "huc2"}:
            raise ContractError("pooled L2 station/HUC record schema mismatch")
        site = record.get("site_no")
        huc2 = record.get("huc2")
        if type(site) is not str or not site or type(huc2) is not str or not huc2 or site in huc:
            raise ContractError("pooled L2 station/HUC identifiers are invalid")
        _require_formal_site_id(site, label="shared fold-registry station identifier")
        station_order.append(site)
        huc[site] = huc2
    if tuple(station_order) != tuple(sorted(station_order)):
        raise ContractError("pooled L2 station/HUC records must be sorted")
    folds = tuple(
        _fold_from_exact_record(record, label="pooled L2 fold registry") for record in raw_folds
    )
    identities = {(fold.geometry, fold.split_seed, fold.fold) for fold in folds}
    if len(identities) != len(folds):
        raise ContractError("pooled L2 fold identities are not unique")
    for fold in folds:
        if set(fold.train_stations) | set(fold.hold_stations) != set(huc):
            raise ContractError("pooled L2 fold/station registry membership mismatch")
        if fold.geometry is Geometry.REGION:
            validate_huc_disjoint(fold.train_stations, fold.hold_stations, huc)
    return folds, huc


def _parse_forecast_keys_registry(
    payload: bytes,
) -> tuple[Geometry, int, int, tuple[ForecastKeyRecord, ...], str]:
    document = _parse_canonical_registry_bytes(payload, label="ForecastKeys registry")
    expected = {
        "format",
        "authority_kind",
        "key_authority_binding_sha256",
        "production_authority_status",
        "canonical_registry_path",
        "evaluation_period",
        "key_id_formula",
        "geometry",
        "split_seed",
        "fold",
        "records",
    }
    if set(document) != expected or document.get("format") != POOLED_L2_KEY_REGISTRY_FORMAT:
        raise ContractError("ForecastKeys registry schema mismatch")
    if (
        document.get("authority_kind") != SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND
        or document.get("production_authority_status") != PRODUCTION_COMMON_KEY_AUTHORITY_STATUS
        or document.get("canonical_registry_path") != CANONICAL_COMMON_KEY_REGISTRY_PATH
        or document.get("evaluation_period")
        != [EVALUATION_START_DATE.isoformat(), EVALUATION_END_DATE.isoformat()]
        or document.get("key_id_formula") != CANONICAL_FORECAST_KEY_FORMULA
    ):
        raise ContractError("ForecastKeys registry lacks the exact synthetic authority contract")
    try:
        geometry = Geometry(document.get("geometry"))
    except (TypeError, ValueError) as exc:
        raise ContractError("ForecastKeys geometry is invalid") from exc
    split_seed = document.get("split_seed")
    fold_index = document.get("fold")
    raw_records = document.get("records")
    if type(split_seed) is not int or type(fold_index) is not int:
        raise ContractError("ForecastKeys split_seed/fold must be exact integers")
    if not isinstance(raw_records, list) or not raw_records:
        raise ContractError("ForecastKeys records must be a non-empty list")
    records: list[ForecastKeyRecord] = []
    expected_fields = {"key_id", "site_no", "issue_date", "target_date", "lead", "huc2"}
    for record in raw_records:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ContractError("ForecastKey record schema mismatch")
        records.append(
            ForecastKeyRecord(
                key_id=record["key_id"],  # type: ignore[arg-type]
                site_no=record["site_no"],  # type: ignore[arg-type]
                issue_date=record["issue_date"],  # type: ignore[arg-type]
                target_date=record["target_date"],  # type: ignore[arg-type]
                lead=record["lead"],  # type: ignore[arg-type]
                huc2=record["huc2"],  # type: ignore[arg-type]
            )
        )
    key_ids = tuple(record.key_id for record in records)
    canonical_order = tuple(
        (record.site_no, record.issue_date, record.target_date, record.lead, record.key_id)
        for record in records
    )
    if canonical_order != tuple(sorted(canonical_order)) or len(key_ids) != len(set(key_ids)):
        raise ContractError("ForecastKeys records must be unique and in canonical tuple order")
    binding = document.get("key_authority_binding_sha256")
    expected_binding = _forecast_key_authority_binding_sha256(
        geometry=geometry,
        split_seed=split_seed,
        fold_index=fold_index,
        records=tuple(records),
    )
    if binding != expected_binding:
        raise ContractError("ForecastKeys key-authority binding does not match exact records")
    return geometry, split_seed, fold_index, tuple(records), expected_binding


def _select_exact_fold(
    folds: Sequence[FoldDefinition],
    *,
    geometry: Geometry,
    split_seed: int,
    fold_index: int,
    label: str,
) -> FoldDefinition:
    matches = tuple(
        fold
        for fold in folds
        if (fold.geometry, fold.split_seed, fold.fold) == (geometry, split_seed, fold_index)
    )
    if len(matches) != 1:
        raise ContractError(f"{label} does not select exactly one shared fold")
    return matches[0]


def _semantic_fold_sha256(
    fold_registry_bytes: bytes,
    fold: FoldDefinition,
    huc2_by_station: Mapping[str, str],
) -> str:
    """Fingerprint selected semantics and the exact shared registry bytes."""

    station_universe = tuple(sorted(set(fold.train_stations) | set(fold.hold_stations)))
    huc_items = _canonical_huc2_items(station_universe, huc2_by_station)
    return canonical_sha256(
        {
            "format": SEMANTIC_FOLD_FINGERPRINT_FORMAT,
            "fold_registry_sha256": hashlib.sha256(fold_registry_bytes).hexdigest(),
            "fold": fold.to_record(),
            "station_huc2": [{"site_no": station, "huc2": huc2} for station, huc2 in huc_items],
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class ForecastKeyAuthority:
    """Structured ForecastKeys bound to one semantic shared-fold fingerprint."""

    records: tuple[ForecastKeyRecord, ...]
    fold: FoldDefinition
    huc2_by_station: Mapping[str, str]
    forecast_keys_sha256: str
    key_authority_binding_sha256: str
    fold_registry_sha256: str
    semantic_fold_sha256: str
    authority_kind: str
    production_authority_status: str
    _forecast_keys_registry_bytes: bytes = field(repr=False, compare=False)
    _fold_registry_bytes: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ForecastKeyAuthority cannot be constructed directly")


def _materialize_forecast_key_authority(
    *,
    fold_registry_bytes: bytes,
    forecast_keys_registry_bytes: bytes,
) -> ForecastKeyAuthority:
    folds, huc2_by_station = _parse_pooled_l2_fold_registry(fold_registry_bytes)
    geometry, split_seed, fold_index, records, authority_binding_sha256 = (
        _parse_forecast_keys_registry(forecast_keys_registry_bytes)
    )
    fold = _select_exact_fold(
        folds,
        geometry=geometry,
        split_seed=split_seed,
        fold_index=fold_index,
        label="ForecastKeys registry",
    )
    for station in (*fold.train_stations, *fold.hold_stations):
        _require_formal_site_id(station, label="ForecastKeys fold station identifier")
    for record in records:
        _require_formal_site_id(record.site_no, label="ForecastKey site_no")
        if record.site_no not in set(fold.hold_stations):
            raise ContractError(
                f"evaluation ForecastKey site is not held by the selected fold: {record.site_no}"
            )
        if huc2_by_station[record.site_no] != record.huc2:
            raise ContractError(f"ForecastKey HUC2 mismatch for {record.site_no}")
    leads = {record.lead for record in records}
    if len(leads) != 1:
        raise ContractError("one ForecastKeys authority must contain exactly one lead")
    instance = object.__new__(ForecastKeyAuthority)
    for name, value in {
        "records": records,
        "fold": fold,
        "huc2_by_station": MappingProxyType(dict(huc2_by_station)),
        "forecast_keys_sha256": hashlib.sha256(forecast_keys_registry_bytes).hexdigest(),
        "key_authority_binding_sha256": authority_binding_sha256,
        "fold_registry_sha256": hashlib.sha256(fold_registry_bytes).hexdigest(),
        "semantic_fold_sha256": _semantic_fold_sha256(fold_registry_bytes, fold, huc2_by_station),
        "authority_kind": SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND,
        "production_authority_status": PRODUCTION_COMMON_KEY_AUTHORITY_STATUS,
        "_forecast_keys_registry_bytes": forecast_keys_registry_bytes,
        "_fold_registry_bytes": fold_registry_bytes,
    }.items():
        object.__setattr__(instance, name, value)
    return instance


def validate_forecast_key_authority(authority: ForecastKeyAuthority) -> None:
    if type(authority) is not ForecastKeyAuthority:
        raise ContractError("ForecastKeys authority must have the exact protected type")
    expected = _materialize_forecast_key_authority(
        fold_registry_bytes=authority._fold_registry_bytes,
        forecast_keys_registry_bytes=authority._forecast_keys_registry_bytes,
    )
    if (
        tuple(record.to_record() for record in authority.records)
        != tuple(record.to_record() for record in expected.records)
        or authority.fold.to_record() != expected.fold.to_record()
        or dict(authority.huc2_by_station) != dict(expected.huc2_by_station)
        or authority.forecast_keys_sha256 != expected.forecast_keys_sha256
        or authority.key_authority_binding_sha256 != expected.key_authority_binding_sha256
        or authority.fold_registry_sha256 != expected.fold_registry_sha256
        or authority.semantic_fold_sha256 != expected.semantic_fold_sha256
        or authority.authority_kind != SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND
        or authority.production_authority_status != PRODUCTION_COMMON_KEY_AUTHORITY_STATUS
    ):
        raise ContractError("ForecastKeys authority differs from retained exact bytes")


def _parse_pooled_l2_training_source(
    payload: bytes,
    training_stations: tuple[str, ...],
) -> dict[str, np.ndarray]:
    document = _parse_canonical_registry_bytes(payload, label="pooled L2 training source")
    if (
        set(document) != {"format", "training_period", "records"}
        or document.get("format") != POOLED_L2_SOURCE_REGISTRY_FORMAT
    ):
        raise ContractError("pooled L2 training source schema mismatch")
    if document.get("training_period") != POOLED_L2_TRAINING_PERIOD:
        raise ContractError("pooled L2 training source period mismatch")
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise ContractError("pooled L2 training source records must be a list")
    curves: dict[str, np.ndarray] = {}
    expected_dates = _canonical_training_source_dates()
    date_doys = np.asarray(
        [date.fromisoformat(value).timetuple().tm_yday - 1 for value in expected_dates],
        dtype=np.int16,
    )
    observed_order: list[str] = []
    for record in raw_records:
        if not isinstance(record, Mapping) or set(record) != {"station_id", "daily_records"}:
            raise ContractError("pooled L2 training source record schema mismatch")
        station = record.get("station_id")
        daily_records = record.get("daily_records")
        if not isinstance(station, str) or station in curves or not isinstance(daily_records, list):
            raise ContractError("pooled L2 training source station/date fields are invalid")
        _require_formal_site_id(station, label="pooled L2 training station_id")
        if len(daily_records) != len(expected_dates):
            raise ContractError(
                f"pooled L2 source dates for {station} are not exact 2006--2015 dates"
            )
        daily_values = np.empty(len(expected_dates), dtype=np.float64)
        for row, (daily_record, expected_date) in enumerate(
            zip(daily_records, expected_dates, strict=True)
        ):
            if not isinstance(daily_record, Mapping) or set(daily_record) != {
                "date",
                "wtemp",
            }:
                raise ContractError("pooled L2 daily WTEMP record schema mismatch")
            if daily_record.get("date") != expected_date:
                raise ContractError(
                    f"pooled L2 source dates for {station} are not exact 2006--2015 dates"
                )
            raw_value = daily_record.get("wtemp")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ContractError("pooled L2 daily WTEMP values must be JSON numerics")
            daily_values[row] = float(raw_value)
        curve = _safe_daily_doy_curve(
            daily_values,
            date_doys,
            label=f"pooled L2 WTEMP {station}",
        )
        observed_order.append(station)
        curves[station] = curve
    if tuple(observed_order) != training_stations:
        raise ContractError("pooled L2 training source membership/order mismatch")
    return curves


def _materialize_pooled_l2_context(
    *,
    fold_registry_bytes: bytes,
    training_source_registry_bytes: bytes,
    key_registry_bytes: bytes,
) -> PooledL2Context:
    authority = _materialize_forecast_key_authority(
        fold_registry_bytes=fold_registry_bytes,
        forecast_keys_registry_bytes=key_registry_bytes,
    )
    fold = authority.fold
    huc = authority.huc2_by_station
    forecast_keys = authority.records
    curves = _parse_pooled_l2_training_source(training_source_registry_bytes, fold.train_stations)
    lead = forecast_keys[0].lead
    keys = tuple(record.key_id for record in forecast_keys)
    stations = tuple(record.site_no for record in forecast_keys)
    issue_dates = tuple(record.issue_date for record in forecast_keys)
    target_dates = tuple(record.target_date for record in forecast_keys)
    issue_doy = np.asarray(
        [
            _parse_iso_date(value, label="ForecastKey issue_date").timetuple().tm_yday
            for value in issue_dates
        ],
        dtype=np.int16,
    )
    target_doy = np.asarray(
        [
            _parse_iso_date(value, label="ForecastKey target_date").timetuple().tm_yday
            for value in target_dates
        ],
        dtype=np.int16,
    )
    pooled_curve = _safe_pooled_curve(
        [curves[station] for station in fold.train_stations],
        label="pooled L2 WTEMP",
    )
    huc_registry_bytes = _canonical_json_bytes(
        {
            "format": POOLED_L2_HUC2_REGISTRY_FORMAT,
            "records": [[station, huc[station]] for station in sorted(huc)],
        }
    )
    lineage = _materialize_pooled_l2_lineage(
        geometry=fold.geometry,
        split_seed=fold.split_seed,
        fold=fold.fold,
        training_stations=fold.train_stations,
        held_stations=fold.hold_stations,
        fold_registry_sha256=authority.fold_registry_sha256,
        semantic_fold_sha256=authority.semantic_fold_sha256,
        training_climatology_source_sha256=hashlib.sha256(
            training_source_registry_bytes
        ).hexdigest(),
        key_registry_sha256=authority.forecast_keys_sha256,
        key_authority_binding_sha256=authority.key_authority_binding_sha256,
        huc2_registry_sha256=hashlib.sha256(huc_registry_bytes).hexdigest(),
    )
    clim_t = _readonly_vector(pooled_curve[issue_doy - 1], n=len(keys), label="pooled clim_t")
    clim_tgt = _readonly_vector(pooled_curve[target_doy - 1], n=len(keys), label="pooled clim_tgt")
    factory_seal_sha256 = canonical_sha256(
        {
            "format": POOLED_L2_FACTORY_SEAL_FORMAT,
            "lineage_sha256": lineage.sha256,
            "lead": lead,
            "issue_dates": list(issue_dates),
            "target_dates": list(target_dates),
            "clim_t_sha256": _array_sha256("pooled_clim_t", clim_t),
            "clim_tgt_sha256": _array_sha256("pooled_clim_tgt", clim_tgt),
            "construction_status": POOLED_L2_ATTESTATION,
        }
    )
    instance = object.__new__(PooledL2Context)
    for name, value in {
        "key_ids": keys,
        "station_ids": stations,
        "issue_dates": issue_dates,
        "target_dates": target_dates,
        "lead": lead,
        "clim_t": clim_t,
        "clim_tgt": clim_tgt,
        "lineage": lineage,
        "construction_status": POOLED_L2_ATTESTATION,
        "factory_seal_sha256": factory_seal_sha256,
        "_fold_registry_bytes": fold_registry_bytes,
        "_training_source_registry_bytes": training_source_registry_bytes,
        "_key_registry_bytes": key_registry_bytes,
    }.items():
        object.__setattr__(instance, name, value)
    return instance


def validate_pooled_l2_context(context: PooledL2Context) -> None:
    """Recompute L2 values and lineage from retained exact registry bytes."""

    if type(context) is not PooledL2Context:
        raise ContractError("pooled L2 context must be an exact PooledL2Context")
    try:
        expected = _materialize_pooled_l2_context(
            fold_registry_bytes=context._fold_registry_bytes,
            training_source_registry_bytes=context._training_source_registry_bytes,
            key_registry_bytes=context._key_registry_bytes,
        )
    except AttributeError as exc:
        raise ContractError("pooled L2 context lacks exact registry-byte evidence") from exc
    if (
        context.key_ids != expected.key_ids
        or context.station_ids != expected.station_ids
        or context.issue_dates != expected.issue_dates
        or context.target_dates != expected.target_dates
        or context.lead != expected.lead
        or context.lineage.to_record() != expected.lineage.to_record()
        or context.construction_status != POOLED_L2_ATTESTATION
        or context.factory_seal_sha256 != expected.factory_seal_sha256
    ):
        raise ContractError("pooled L2 metadata differs from exact registry-byte evidence")
    clim_t = _require_durable_immutable_array(context.clim_t, label="pooled L2 clim_t")
    if (
        clim_t.dtype != expected.clim_t.dtype
        or clim_t.shape != expected.clim_t.shape
        or not np.array_equal(clim_t, expected.clim_t)
    ):
        raise ContractError("pooled L2 issue climatology differs from training-source bytes")
    clim_tgt = _require_durable_immutable_array(context.clim_tgt, label="pooled L2 clim_tgt")
    if (
        clim_tgt.dtype != expected.clim_tgt.dtype
        or clim_tgt.shape != expected.clim_tgt.shape
        or not np.array_equal(clim_tgt, expected.clim_tgt)
    ):
        raise ContractError("pooled L2 target climatology differs from training-source bytes")


def build_training_only_pooled_l2_context(
    *,
    fold_registry_bytes: bytes,
    forecast_keys_registry_bytes: bytes,
    training_source_registry_bytes: bytes,
) -> PooledL2Context:
    """Materialize L2 only from canonical fold, ForecastKeys, and source bytes."""

    return _materialize_pooled_l2_context(
        fold_registry_bytes=fold_registry_bytes,
        training_source_registry_bytes=training_source_registry_bytes,
        key_registry_bytes=forecast_keys_registry_bytes,
    )


@dataclass(frozen=True, slots=True, init=False)
class ModelInputBatch:
    """Exact tensors allowed to cross the model boundary; station identity is absent."""

    key_ids: tuple[str, ...]
    local_level: LocalLevel
    X: np.ndarray
    Mask: np.ndarray
    wtemp_t: np.ndarray
    clim_t: np.ndarray
    clim_tgt: np.ndarray
    damped_prior: np.ndarray
    phys_std: np.ndarray
    logflowz: np.ndarray
    season: np.ndarray
    gate: np.ndarray
    future_context: np.ndarray
    future_mask: np.ndarray
    forecast_keys_sha256: str
    key_authority_binding_sha256: str
    semantic_fold_sha256: str
    pooled_l2_lineage_sha256: str | None
    content_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ModelInputBatch cannot be constructed directly")

    def _validate_and_freeze(self) -> None:
        keys = tuple(str(value) for value in self.key_ids)
        if not keys or len(keys) != len(set(keys)):
            raise ContractError("model-input key_ids must be non-empty and unique")
        _require_enum(self.local_level, LocalLevel, "model-input local_level")
        n = len(keys)
        X = _readonly_array(self.X, np.float32)
        mask = _readonly_array(self.Mask, np.bool_)
        expected_history_shape = (n, HISTORY_LENGTH, len(HISTORY_VARIABLES))
        if X.shape != expected_history_shape:
            raise ContractError(f"model-input X must have shape {expected_history_shape}")
        if mask.shape != X.shape:
            raise ContractError("model-input Mask must match X")
        if not np.isfinite(X).all():
            raise ContractError("model-input X must be finite")
        vectors = {
            name: _readonly_vector(getattr(self, name), n=n, label=f"model-input {name}")
            for name in ("wtemp_t", "clim_t", "clim_tgt", "damped_prior", "logflowz")
        }
        matrices = {
            "phys_std": _readonly_matrix(self.phys_std, n=n, width=4, label="model-input phys_std"),
            "season": _readonly_matrix(self.season, n=n, width=2, label="model-input season"),
            "gate": _readonly_matrix(
                self.gate,
                n=n,
                width=len(GATE_FEATURES),
                label="model-input gate",
            ),
            "future_context": _readonly_matrix(
                self.future_context,
                n=n,
                width=FUTURE_FEATURE_POLICY.width,
                label="model-input future_context",
            ),
        }
        future_mask = _readonly_matrix(
            self.future_mask,
            n=n,
            width=FUTURE_FEATURE_POLICY.width,
            label="model-input future_mask",
            dtype=np.bool_,
        )
        if self.local_level is LocalLevel.L2 and (
            np.any(X[:, :, WTEMP_HISTORY_INDEX])
            or np.any(mask[:, :, WTEMP_HISTORY_INDEX])
            or np.any(matrices["gate"][:, WTEMP_TENDENCY_INDEX])
        ):
            raise ContractError("L2 model inputs expose prohibited WTEMP state")
        if self.local_level is LocalLevel.L2:
            _validate_sha256(
                self.pooled_l2_lineage_sha256,
                label="model-input pooled_l2_lineage_sha256",
            )
        elif self.pooled_l2_lineage_sha256 is not None:
            raise ContractError("L0 model inputs must not carry pooled L2 lineage")
        _validate_sha256(self.forecast_keys_sha256, label="model-input ForecastKeys SHA-256")
        _validate_sha256(
            self.key_authority_binding_sha256,
            label="model-input key-authority binding SHA-256",
        )
        _validate_sha256(self.semantic_fold_sha256, label="model-input semantic fold SHA-256")
        object.__setattr__(self, "key_ids", keys)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Mask", mask)
        for name, array in vectors.items():
            object.__setattr__(self, name, array)
        for name, array in matrices.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "future_mask", future_mask)
        object.__setattr__(self, "content_sha256", _model_input_integrity_sha256(self))

    def _tensor_mapping_unchecked(self) -> dict[str, np.ndarray]:
        return {
            "X": self.X,
            "Mask": self.Mask,
            "wtemp_t": self.wtemp_t,
            "clim_t": self.clim_t,
            "clim_tgt": self.clim_tgt,
            "damped_prior": self.damped_prior,
            "phys_std": self.phys_std,
            "logflowz": self.logflowz,
            "season": self.season,
            "gate": self.gate,
            "future_context": self.future_context,
            "future_mask": self.future_mask,
        }

    def tensor_mapping(self) -> dict[str, np.ndarray]:
        validate_model_input_batch(self)
        return self._tensor_mapping_unchecked()


def _materialize_model_input_batch(
    *,
    key_ids: tuple[str, ...],
    local_level: LocalLevel,
    X: np.ndarray,
    Mask: np.ndarray,
    wtemp_t: np.ndarray,
    clim_t: np.ndarray,
    clim_tgt: np.ndarray,
    damped_prior: np.ndarray,
    phys_std: np.ndarray,
    logflowz: np.ndarray,
    season: np.ndarray,
    gate: np.ndarray,
    future_context: np.ndarray,
    future_mask: np.ndarray,
    forecast_keys_sha256: str,
    key_authority_binding_sha256: str,
    semantic_fold_sha256: str,
    pooled_l2_lineage_sha256: str | None,
) -> ModelInputBatch:
    instance = object.__new__(ModelInputBatch)
    for name, value in {
        "key_ids": key_ids,
        "local_level": local_level,
        "X": X,
        "Mask": Mask,
        "wtemp_t": wtemp_t,
        "clim_t": clim_t,
        "clim_tgt": clim_tgt,
        "damped_prior": damped_prior,
        "phys_std": phys_std,
        "logflowz": logflowz,
        "season": season,
        "gate": gate,
        "future_context": future_context,
        "future_mask": future_mask,
        "forecast_keys_sha256": forecast_keys_sha256,
        "key_authority_binding_sha256": key_authority_binding_sha256,
        "semantic_fold_sha256": semantic_fold_sha256,
        "pooled_l2_lineage_sha256": pooled_l2_lineage_sha256,
    }.items():
        object.__setattr__(instance, name, value)
    instance._validate_and_freeze()
    return instance


def _model_input_integrity_sha256(batch: ModelInputBatch) -> str:
    return canonical_sha256(
        {
            "key_ids": list(batch.key_ids),
            "local_level": batch.local_level.value,
            "forecast_keys_sha256": batch.forecast_keys_sha256,
            "key_authority_binding_sha256": batch.key_authority_binding_sha256,
            "semantic_fold_sha256": batch.semantic_fold_sha256,
            "pooled_l2_lineage_sha256": batch.pooled_l2_lineage_sha256,
            "arrays": {
                name: _array_sha256(name, array)
                for name, array in batch._tensor_mapping_unchecked().items()
            },
        }
    )


def validate_model_input_batch(batch: ModelInputBatch) -> None:
    if type(batch) is not ModelInputBatch:
        raise ContractError("model inputs must have the exact protected ModelInputBatch type")
    for name, array in batch._tensor_mapping_unchecked().items():
        _require_durable_immutable_array(array, label=f"model-input {name}")
    expected_history_shape = (len(batch.key_ids), HISTORY_LENGTH, len(HISTORY_VARIABLES))
    if batch.X.shape != expected_history_shape or batch.Mask.shape != expected_history_shape:
        raise ContractError("model-input history changed from exact [N,32,7] shape")
    if batch.content_sha256 != _model_input_integrity_sha256(batch):
        raise ContractError("model-input arrays or authority digest changed after construction")


@dataclass(frozen=True, slots=True)
class SeparatedBatch:
    model_inputs: ModelInputBatch
    labels: LabelTargets
    future_audit: CanonicalFutureBatch


def build_separated_batch(
    local_level: LocalLevel,
    sources: RawInputSources,
    labels: LabelTargets,
    future: CanonicalFutureBatch,
    *,
    pooled_l2: PooledL2Context | None = None,
) -> SeparatedBatch:
    """Apply L0/L2 only to input tensors while preserving immutable labels."""

    _require_enum(local_level, LocalLevel, "local_level")
    validate_raw_input_sources(sources)
    validate_label_targets(labels)
    validate_canonical_future_batch(future)
    if sources.key_ids != labels.key_ids or sources.key_ids != future.key_ids:
        raise ContractError("input, future, and label key registries differ")
    if sources.station_identity != labels.station_ids or future.station_ids != labels.station_ids:
        raise ContractError("input, future, and label station registries are not key-aligned")
    forecast_keys_hashes = {
        sources._forecast_key_authority.forecast_keys_sha256,
        labels._forecast_key_authority.forecast_keys_sha256,
        future.forecast_keys_sha256,
    }
    semantic_fold_hashes = {
        sources._forecast_key_authority.semantic_fold_sha256,
        labels._forecast_key_authority.semantic_fold_sha256,
        future.semantic_fold_sha256,
    }
    key_authority_bindings = {
        sources._forecast_key_authority.key_authority_binding_sha256,
        labels._forecast_key_authority.key_authority_binding_sha256,
        future.key_authority_binding_sha256,
    }
    exact_forecast_keys_match = (
        sources.forecast_keys_registry_bytes
        == labels.forecast_keys_registry_bytes
        == future._forecast_keys_registry_bytes
    )
    exact_fold_registry_match = (
        sources.fold_registry_bytes == labels.fold_registry_bytes == future._fold_registry_bytes
    )
    if (
        len(forecast_keys_hashes) != 1
        or len(key_authority_bindings) != 1
        or len(semantic_fold_hashes) != 1
        or not exact_forecast_keys_match
        or not exact_fold_registry_match
    ):
        raise ContractError(
            "input, label, and forcing boundaries must share exact ForecastKeys/fold bytes"
        )
    n = len(sources.key_ids)
    if future.n_examples != n:
        raise ContractError("future rows must align with input/label keys")

    X = np.array(sources.history_values, copy=True)
    mask = np.array(sources.history_mask, copy=True)
    wtemp_t = np.array(sources.wtemp_t, copy=True)
    clim_t = np.array(sources.clim_t, copy=True)
    clim_tgt = np.array(sources.clim_tgt, copy=True)
    damped_prior = np.array(sources.damped_prior, copy=True)
    gate = np.array(sources.gate, copy=True)
    if local_level is LocalLevel.L2:
        if pooled_l2 is None or type(pooled_l2) is not PooledL2Context:
            raise ContractError("L2 requires key-aligned training-station-pooled climatology")
        validate_pooled_l2_context(pooled_l2)
        if (
            pooled_l2.key_ids != sources.key_ids
            or pooled_l2.station_ids != labels.station_ids
            or pooled_l2.lead != future.lead
            or pooled_l2.lineage.key_registry_sha256 != future.forecast_keys_sha256
            or pooled_l2.lineage.key_authority_binding_sha256 != future.key_authority_binding_sha256
            or pooled_l2.lineage.semantic_fold_sha256 != future.semantic_fold_sha256
            or pooled_l2._key_registry_bytes != future._forecast_keys_registry_bytes
            or pooled_l2._fold_registry_bytes != future._fold_registry_bytes
        ):
            raise ContractError("L2 requires key-aligned training-station-pooled climatology")
        if (
            pooled_l2.issue_dates != future.issue_dates
            or pooled_l2.target_dates != future.target_dates
            or pooled_l2.lineage.geometry is not future.geometry
            or pooled_l2.lineage.split_seed != future.split_seed
            or pooled_l2.lineage.fold != future.fold
        ):
            raise ContractError("F3 and L2 must share exact ForecastKey dates and fold identity")
        X[:, :, WTEMP_HISTORY_INDEX] = 0.0
        mask[:, :, WTEMP_HISTORY_INDEX] = False
        wtemp_t = np.array(pooled_l2.clim_t, copy=True)
        clim_t = np.array(pooled_l2.clim_t, copy=True)
        clim_tgt = np.array(pooled_l2.clim_tgt, copy=True)
        damped_prior = np.array(pooled_l2.clim_tgt, copy=True)
        gate[:, WTEMP_TENDENCY_INDEX] = 0.0
    elif pooled_l2 is not None:
        raise ContractError("pooled_l2 context is forbidden for L0")

    def freeze(array: np.ndarray, dtype: type = np.float32) -> np.ndarray:
        return _readonly_array(array, dtype)

    model_inputs = _materialize_model_input_batch(
        key_ids=sources.key_ids,
        local_level=local_level,
        X=freeze(X),
        Mask=freeze(mask, np.bool_),
        wtemp_t=freeze(wtemp_t),
        clim_t=freeze(clim_t),
        clim_tgt=freeze(clim_tgt),
        damped_prior=freeze(damped_prior),
        phys_std=freeze(sources.phys_std),
        logflowz=freeze(sources.logflowz),
        season=freeze(sources.season),
        gate=freeze(gate),
        future_context=freeze(future.values),
        future_mask=freeze(future.mask, np.bool_),
        forecast_keys_sha256=future.forecast_keys_sha256,
        key_authority_binding_sha256=future.key_authority_binding_sha256,
        semantic_fold_sha256=future.semantic_fold_sha256,
        pooled_l2_lineage_sha256=(
            pooled_l2.lineage.sha256 if local_level is LocalLevel.L2 else None
        ),
    )
    validate_model_input_batch(model_inputs)
    return SeparatedBatch(model_inputs=model_inputs, labels=labels, future_audit=future)


def _digest_array(hasher: Any, label: str, array: np.ndarray) -> None:
    _reject_masked_array(array, label=f"{label} digest input")
    contiguous = np.ascontiguousarray(array)
    hasher.update(label.encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(_canonical_json_bytes(list(contiguous.shape)))
    hasher.update(contiguous.tobytes(order="C"))


def model_input_sha256(batch: ModelInputBatch) -> str:
    validate_model_input_batch(batch)
    hasher = hashlib.sha256()
    hasher.update(batch.local_level.value.encode("ascii"))
    hasher.update(_canonical_json_bytes(list(batch.key_ids)))
    hasher.update(str(batch.pooled_l2_lineage_sha256).encode("ascii"))
    hasher.update(batch.forecast_keys_sha256.encode("ascii"))
    hasher.update(batch.key_authority_binding_sha256.encode("ascii"))
    hasher.update(batch.semantic_fold_sha256.encode("ascii"))
    for name, array in batch._tensor_mapping_unchecked().items():
        _digest_array(hasher, name, array)
    return hasher.hexdigest()


def label_sha256(labels: LabelTargets) -> str:
    validate_label_targets(labels)
    hasher = hashlib.sha256()
    hasher.update(_canonical_json_bytes(list(labels.key_ids)))
    hasher.update(_canonical_json_bytes(list(labels.station_ids)))
    hasher.update(labels._forecast_key_authority.forecast_keys_sha256.encode("ascii"))
    hasher.update(labels._forecast_key_authority.key_authority_binding_sha256.encode("ascii"))
    hasher.update(labels._forecast_key_authority.semantic_fold_sha256.encode("ascii"))
    _digest_array(hasher, "target", labels.target)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Artifact identity, hashes, resume, and completeness
# ---------------------------------------------------------------------------


DATA_LINEAGE_HASHES = (
    "common_key_registry_sha256",
    "f3_value_registry_sha256",
    "f3_substitution_registry_sha256",
    "fold_preprocessing_sha256",
    "scaler_sha256",
    "anchor_sha256",
    "pooled_l2_lineage_sha256",
    "train_model_input_sha256",
    "validation_model_input_sha256",
    "evaluation_model_input_sha256",
)
EXECUTION_LINEAGE_HASHES = (
    "runner_sha256",
    "model_registry_sha256",
    "contrast_registry_sha256",
)
REQUIRED_PROVENANCE_HASHES = (
    "protocol_sha256",
    "source_tree_sha256",
    "panel_sha256",
    "label_registry_sha256",
    "station_registry_sha256",
    "fold_registry_sha256",
    "input_schema_sha256",
    "future_schema_sha256",
    "objective_policy_sha256",
    "budget_policy_sha256",
    "cell_plan_sha256",
    *DATA_LINEAGE_HASHES,
    *EXECUTION_LINEAGE_HASHES,
)


def _architecture_path(architecture: Architecture) -> str:
    return "plain_tcn" if architecture is Architecture.PLAIN_TCN else "thermoroute"


def artifact_stem(cell: CellPlan) -> Path:
    return Path(
        _architecture_path(cell.architecture),
        cell.forcing.value,
        cell.local_level.value,
        cell.geometry.value,
        f"h{cell.lead}",
        f"split{cell.split_seed:02d}",
        f"fold{cell.fold}",
        f"fit{cell.fit_seed}",
        cell.constraint.value,
    )


def checkpoint_relative_path(cell: CellPlan) -> Path:
    return Path("checkpoints") / artifact_stem(cell).with_suffix(".pt")


def prediction_relative_path(cell: CellPlan) -> Path:
    return Path("predictions") / artifact_stem(cell).with_suffix(".parquet")


def fit_manifest_relative_path(cell: CellPlan) -> Path:
    return Path("manifests") / artifact_stem(cell).with_suffix(".json")


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{label} must be a lowercase SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"{label} must be a lowercase SHA-256 hex digest") from exc
    if value != value.lower():
        raise ContractError(f"{label} must be lowercase")
    return value


def _validate_provenance_hash_shape(hashes: Mapping[str, str]) -> dict[str, str]:
    if set(hashes) != set(REQUIRED_PROVENANCE_HASHES):
        missing = sorted(set(REQUIRED_PROVENANCE_HASHES) - set(hashes))
        extra = sorted(set(hashes) - set(REQUIRED_PROVENANCE_HASHES))
        raise ContractError(f"provenance hash schema mismatch: missing={missing}, extra={extra}")
    return {key: _validate_sha256(hashes[key], label=key) for key in REQUIRED_PROVENANCE_HASHES}


def provenance_hashes_from_authority(
    snapshot: NeuralAuthoritySnapshot,
) -> NoReturn:
    """Terminally block caller-created provenance until input semantics exist."""

    if type(snapshot) is not NeuralAuthoritySnapshot:
        raise ContractError("provenance requires one exact immutable authority snapshot")
    if not snapshot.semantic_validation_complete:
        raise NeuralAuthoritySemanticValidationUnavailable(
            f"{NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS}: provenance "
            "cannot be derived from an opaque input registry"
        )
    raise NeuralAuthoritySemanticValidationUnavailable(
        NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS
    )


def planned_fit_manifest(
    cell: CellPlan,
    authority_snapshot: NeuralAuthoritySnapshot,
) -> NoReturn:
    """Unavailable until provenance can be derived from one semantic snapshot."""

    if not isinstance(cell, CellPlan):
        raise TypeError("planned fit requires a CellPlan")
    del cell
    provenance_hashes_from_authority(authority_snapshot)


def _fit_manifest_identity(cell: CellPlan) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "cell": cell.to_record(),
        "paths": {
            "checkpoint": checkpoint_relative_path(cell).as_posix(),
            "prediction": prediction_relative_path(cell).as_posix(),
            "manifest": fit_manifest_relative_path(cell).as_posix(),
        },
    }


def validate_resume_manifest(
    document: Mapping[str, object],
    cell: CellPlan,
    authority_snapshot: NeuralAuthoritySnapshot | None,
    *,
    artifact_root: Path,
) -> None:
    expected = _fit_manifest_identity(cell)
    expected_top_level = {
        "format",
        "schema_version",
        "status",
        "cell",
        "paths",
        "provenance_hashes",
        "data_lineage_hashes",
        "execution_lineage_hashes",
        "artifact_hashes",
        "training_execution_implemented",
    }
    if set(document) != expected_top_level:
        missing = sorted(expected_top_level - set(document))
        extra = sorted(set(document) - expected_top_level)
        raise ContractError(f"resume manifest schema mismatch: missing={missing}, extra={extra}")
    if document.get("format") != MANIFEST_FORMAT:
        raise ContractError("resume manifest format mismatch")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("resume manifest schema version mismatch")
    if document.get("status") != ArtifactStatus.COMPLETE.value:
        raise ContractError("resume accepts only COMPLETE fit manifests")
    if document.get("cell") != expected["cell"]:
        raise ContractError("resume manifest cell identity mismatch")
    if document.get("paths") != expected["paths"]:
        raise ContractError("resume manifest artifact paths mismatch")
    provenance_hashes = document.get("provenance_hashes")
    data_lineage_hashes = document.get("data_lineage_hashes")
    execution_lineage_hashes = document.get("execution_lineage_hashes")
    if not isinstance(provenance_hashes, Mapping):
        raise ContractError("resume manifest provenance hashes must be a mapping")
    validated_hashes = _validate_provenance_hash_shape(provenance_hashes)  # shape only
    if data_lineage_hashes != {key: validated_hashes[key] for key in DATA_LINEAGE_HASHES}:
        raise ContractError("resume manifest data-lineage hashes mismatch")
    if execution_lineage_hashes != {key: validated_hashes[key] for key in EXECUTION_LINEAGE_HASHES}:
        raise ContractError("resume manifest execution-lineage hashes mismatch")
    artifact_hashes = document.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        raise ContractError("resume manifest lacks artifact hashes")
    if set(artifact_hashes) != {"checkpoint_sha256", "prediction_sha256"}:
        raise ContractError("resume artifact hash schema mismatch")
    declared_hashes = {
        key: _validate_sha256(artifact_hashes[key], label=key)
        for key in ("checkpoint_sha256", "prediction_sha256")
    }
    if document.get("training_execution_implemented") is not True:
        raise ContractError("resume manifest does not attest an implemented execution path")
    del artifact_root, declared_hashes
    if type(authority_snapshot) is not NeuralAuthoritySnapshot:
        raise ResumeContentValidationUnavailable(
            f"{RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS}: resume provenance is not "
            "derived from one immutable authority snapshot"
        )
    if not authority_snapshot.semantic_validation_complete:
        raise ResumeContentValidationUnavailable(
            f"{RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS}: authority snapshot registry "
            "semantics are unavailable"
        )
    raise ResumeContentValidationUnavailable(
        f"{RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS}: Phase 1 has no trusted "
        "torch checkpoint deserializer or Parquet schema/exact-key/registry-y_true "
        "validator; no COMPLETE manifest can be resumed"
    )


def _validate_completion_documents(
    manifests: Sequence[Mapping[str, object]],
    plan: Sequence[CellPlan],
    authority_snapshot: NeuralAuthoritySnapshot | None,
    *,
    artifact_root: Path,
) -> None:
    if not plan or any(not isinstance(cell, CellPlan) for cell in plan):
        raise ContractError("completion plan must contain typed CellPlan records")
    fit_ids = [cell.fit_id for cell in plan]
    if len(fit_ids) != len(set(fit_ids)):
        raise ContractError("completion plan contains duplicate fit identities")
    expected = {cell.fit_id: cell for cell in plan}
    observed: dict[str, Mapping[str, object]] = {}
    for document in manifests:
        cell_record = document.get("cell")
        if not isinstance(cell_record, Mapping) or not isinstance(cell_record.get("fit_id"), str):
            raise ContractError("completion manifest lacks a fit_id")
        fit_id = str(cell_record["fit_id"])
        if fit_id in observed:
            raise ContractError(f"duplicate completion manifest for {fit_id}")
        observed[fit_id] = document
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ContractError(
            f"completion inventory mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for fit_id, cell in expected.items():
        validate_resume_manifest(
            observed[fit_id],
            cell,
            authority_snapshot,
            artifact_root=artifact_root,
        )


def validate_subset_completion_diagnostic(
    manifests: Sequence[Mapping[str, object]],
    subset_plan: Sequence[CellPlan],
    authority_snapshot: NeuralAuthoritySnapshot | None,
    *,
    artifact_root: Path,
) -> None:
    """Fail closed: even a diagnostic subset cannot yet be resumed."""

    canonical = {cell.fit_id: cell for cell in build_cell_plan()}
    subset = tuple(subset_plan)
    if not subset:
        raise ContractError("subset diagnostic plan must not be empty")
    if any(not isinstance(cell, CellPlan) for cell in subset):
        raise ContractError("subset diagnostic plan must contain CellPlan records")
    if len({cell.fit_id for cell in subset}) != len(subset):
        raise ContractError("subset diagnostic plan contains duplicate fit identities")
    for cell in subset:
        if cell.fit_id not in canonical or cell.to_record() != canonical[cell.fit_id].to_record():
            raise ContractError(f"subset contains a non-canonical fit: {cell.fit_id}")
    _validate_completion_documents(
        manifests,
        subset,
        authority_snapshot,
        artifact_root=artifact_root,
    )
    raise ResumeContentValidationUnavailable(RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS)


def validate_full_completion_inventory(
    manifests: Sequence[Mapping[str, object]],
    plan: Sequence[CellPlan],
    authority_snapshot: NeuralAuthoritySnapshot | None,
    *,
    artifact_root: Path,
) -> None:
    """Fail closed until every canonical fit and prediction content can be audited."""

    candidate = tuple(plan)
    canonical = build_cell_plan()
    validate_cell_plan(candidate)
    if len(candidate) != 5_280 or candidate != canonical:
        raise ContractError("full completion requires the exact canonical 5,280-fit plan")
    if cell_plan_sha256(candidate) != EXPECTED_CELL_PLAN_SHA256:
        raise ContractError("full completion cell-plan digest mismatch")
    _validate_completion_documents(
        manifests,
        candidate,
        authority_snapshot,
        artifact_root=artifact_root,
    )
    raise PredictionContentValidationUnavailable(
        f"{FULL_COMPLETION_UNAVAILABLE_STATUS}: Phase 1 cannot semantically validate "
        "torch checkpoints or Parquet schema, exact common keys, and registry y_true; "
        "full completion cannot be returned"
    )


def input_schema_record() -> dict[str, object]:
    return {
        "model_tensor_keys": [
            "X",
            "Mask",
            "wtemp_t",
            "clim_t",
            "clim_tgt",
            "damped_prior",
            "phys_std",
            "logflowz",
            "season",
            "gate",
            "future_context",
            "future_mask",
        ],
        "trainer_only_fields": [
            "key_ids",
            "station_ids",
            "target",
            "future_substitution_mask",
            "future_source_value_sha256",
            "future_substitution_registry_sha256",
            "forecast_keys_sha256",
            "key_authority_binding_sha256",
            "semantic_fold_sha256",
            "pooled_l2_lineage",
        ],
        "station_identity_is_model_input": False,
        "history_variables": list(HISTORY_VARIABLES),
        "history_shape": f"[N,{HISTORY_LENGTH},{len(HISTORY_VARIABLES)}]",
        "history_timestamp_rule": "exact contiguous daily issue-31..issue per ForecastKey",
        "history_mask_rule": "exact bool ndarray; NumPy MaskedArray is forbidden",
        "station_id_rule": "ASCII-only fullmatch (?:[0-9]{8}|[0-9]{15})",
        "array_immutability": (
            "exact ndarray with durable non-writeable backing plus digest recomputation before "
            "every consumer; OWNDATA=True is not required"
        ),
        "boundary_authority": (
            "L0/L2/F0/F3 inputs and labels share exact structured ForecastKeys bytes, "
            "formal tuple hashes, the 2021--2023 domain, an explicit synthetic-only "
            "authority binding, and one semantic shared-fold fingerprint"
        ),
        "forecast_key_authority": {
            "structural_authority_kind": SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND,
            "binding_format": FORECAST_KEY_AUTHORITY_BINDING_FORMAT,
            "key_id_formula": CANONICAL_FORECAST_KEY_FORMULA,
            "evaluation_start": EVALUATION_START_DATE.isoformat(),
            "evaluation_end": EVALUATION_END_DATE.isoformat(),
            "canonical_production_registry_path": CANONICAL_COMMON_KEY_REGISTRY_PATH,
            "production_authority_status": PRODUCTION_COMMON_KEY_AUTHORITY_STATUS,
        },
        "gate_features": list(GATE_FEATURES),
        "L0": {
            "WTEMP_history": "visible",
            "issue_WTEMP": "visible",
            "anchor": "training-period per-site damped climatology",
            "station_identity": "forbidden",
        },
        "L2": {
            "WTEMP_history": "zero value and false mask for train/val/eval",
            "issue_WTEMP": "training-station-pooled climatology",
            "WTEMP_tendency": "zero",
            "anchor": "training-station-pooled target climatology only",
            "station_identity": "forbidden",
            "constructor": (
                "fold-aware exact-byte structural recomputation; not execution authority"
            ),
            "training_source_rule": (
                "station DOY curves are recomputed from exact daily 2006--2015 WTEMP; "
                "caller-supplied 366-value curves are forbidden"
            ),
            "fold_registry_format": POOLED_L2_FOLD_REGISTRY_FORMAT,
            "training_source_registry_format": POOLED_L2_SOURCE_REGISTRY_FORMAT,
            "key_registry_format": POOLED_L2_KEY_REGISTRY_FORMAT,
            "huc2_registry_format": POOLED_L2_HUC2_REGISTRY_FORMAT,
            "factory_seal_format": POOLED_L2_FACTORY_SEAL_FORMAT,
            "date_rule": POOLED_L2_DATE_RULE,
            "lineage_hash_required": True,
        },
        "label_boundary": (
            "durably immutable values bound to exact ForecastKeys/fold bytes; never derived "
            "from the masked input panel"
        ),
    }


def fixed_contract_hashes() -> dict[str, str]:
    return {
        "cell_plan_sha256": cell_plan_sha256(build_cell_plan()),
        "future_schema_sha256": canonical_sha256(FUTURE_FEATURE_POLICY.to_record()),
        "input_schema_sha256": canonical_sha256(input_schema_record()),
        "objective_policy_sha256": canonical_sha256(POINT_OBJECTIVE_POLICY.to_record()),
        "budget_policy_sha256": canonical_sha256(FIT_BUDGET_POLICY.to_record()),
    }


def phase1_plan_manifest(
    plan: Sequence[CellPlan], *, include_cells: bool = False
) -> dict[str, object]:
    validate_cell_plan(plan)
    validate_point_objective_policy(POINT_OBJECTIVE_POLICY)
    validate_fit_budget_policy(FIT_BUDGET_POLICY)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "PHASE_1_CONTRACT_ONLY",
        "status": POLICY_STATUS,
        "execution_implemented": False,
        "reads_panels": False,
        "writes_artifacts": False,
        "training_unit": "one architecture/forcing/local/geometry/lead/split/fold/fit/constraint",
        "inventory": inventory_arithmetic(plan).to_record(),
        "protocol_f0_f3_degraded_primary_matrix": {
            "F": ["F0", "F3_full"],
            "L": ["L0", "L2"],
            "G": ["random_site", "whole_region"],
            "A": ["LightGBM", "plain_TCN"],
            "leads": list(LEADS),
            "logical_cell_count": 48,
            "neural_runner_owned_plain_TCN_cells": 24,
            "tree_runner_owned_LightGBM_cells": 24,
        },
        "selected_thermoroute_regimes": [
            {
                "forcing": forcing.value,
                "local_level": local_level.value,
                "geometry": geometry.value,
            }
            for forcing, local_level, geometry in sorted(
                SELECTED_THERMOROUTE_REGIMES,
                key=lambda item: tuple(value.value for value in item),
            )
        ],
        "selected_thermoroute_inventory_semantics": {
            "protocol_cells": 12,
            "constraint_variants_per_cell": 2,
            "variant_configurations": 24,
            "term_rule": "24 is never called a protocol logical-cell count",
        },
        "replication": {
            "leads": list(LEADS),
            "random_split_seeds": list(RANDOM_SPLIT_SEEDS),
            "region_split_seed_sentinel": REGION_SPLIT_SEED,
            "folds": list(FOLDS),
            "model_fit_seeds": list(MODEL_FIT_SEEDS),
            "region_matched_hold_sizes": list(REGION_MATCHED_HOLD_SIZES),
            "random_fold_algorithm": (
                "numpy.default_rng(seed)/PCG64 permutation of sorted station ids; "
                "contiguous slices sized 30,30,31,29"
            ),
        },
        "future_feature_policy": FUTURE_FEATURE_POLICY.to_record(),
        "input_schema": input_schema_record(),
        "point_objective_policy": POINT_OBJECTIVE_POLICY.to_record(),
        "fit_budget_policy": FIT_BUDGET_POLICY.to_record(),
        "neural_execution_authority_contract": {
            "canonical_protocol_path": str(CANONICAL_SEALED_PROTOCOL),
            "canonical_seal_path": str(CANONICAL_PROTOCOL_SEAL),
            "canonical_registry_paths": {
                key: str(path) for key, path in CANONICAL_NEURAL_REGISTRY_PATHS.items()
            },
            "seal_read_count_per_validation": 1,
            "seal_object_is_recursively_immutable": True,
            "authority_snapshot_is_immutable": True,
            "authority_snapshot_reused_for_all_fits": True,
            "semantic_validation_status": (NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS),
            "common_key_authority_status": PRODUCTION_COMMON_KEY_AUTHORITY_STATUS,
            "canonical_common_key_registry_path": CANONICAL_COMMON_KEY_REGISTRY_PATH,
            "opaque_json_can_authorize_execution": False,
            "binding_keys": list(NEURAL_EXECUTION_BINDING_KEYS),
            "binding_record_schema": ["path", "sha256"],
            "path_rule": (
                "existing non-empty regular file; no symlink component or relative path escape"
            ),
            "digest_rule": "SHA-256 recomputed from exact opened file bytes",
            "runner_rule": "resolved path must equal the executing source path",
            "cell_plan_rule": (
                "exact canonical JSON bytes for the fixed 5,280-fit plan and expected digest"
            ),
            "execution_rule": (
                "non-dry, resume, provenance, and fit creation remain terminally unavailable "
                "until the canonical common-key loader and model/fold/input/contrast "
                "semantic parsers are implemented"
            ),
        },
        "artifact_contract": {
            "manifest_format": MANIFEST_FORMAT,
            "checkpoint_template": (
                "checkpoints/{architecture}/{F}/{L}/{G}/h{lead}/split{seed}/"
                "fold{fold}/fit{fit}/{constraint}.pt"
            ),
            "prediction_template": (
                "predictions/{architecture}/{F}/{L}/{G}/h{lead}/split{seed}/"
                "fold{fold}/fit{fit}/{constraint}.parquet"
            ),
            "required_provenance_hashes": list(REQUIRED_PROVENANCE_HASHES),
            "resume_status": RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS,
            "resume_rule": (
                "disabled until a trusted torch checkpoint validator and Parquet "
                "schema/exact-key/registry-y_true validator are implemented"
            ),
            "subset_diagnostic_status": SUBSET_DIAGNOSTIC_STATUS,
            "full_completion_status": FULL_COMPLETION_UNAVAILABLE_STATUS,
            "completion_rule": (
                "exact canonical 5,280-fit plan and artifacts are necessary but not "
                "sufficient; prediction schema/exact-key/y_true validation is unimplemented"
            ),
            "partial_matrix_rule": "never publish or return a complete-matrix artifact",
        },
        "hashes": fixed_contract_hashes(),
        "unimplemented_execution_modules": [
            (
                "canonical common-key/fold/input/label registry loader for "
                "outputs/final/information_regime_key_registries_v4"
            ),
            "complete fold-aware train-only scaler/anchor/input builder",
            "point-only plain_TCN and ThermoRoute model adapters",
            "trainer/checkpoint writer",
            "exact-key neural scorer and model-seed aggregation",
            "completion-gated station-first estimator",
        ],
    }
    if include_cells:
        manifest["cells"] = [cell.to_record() for cell in plan]
    return manifest


# ---------------------------------------------------------------------------
# Governance and deliberately unavailable execution path
# ---------------------------------------------------------------------------


def _declared_path_candidate(
    raw: object,
    *,
    label: str,
    relative_root: Path = ROOT,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise GovernanceError(f"{label} must be a non-empty path")
    path = Path(raw)
    if ".." in path.parts:
        raise GovernanceError(f"{label} contains a forbidden path-escape component")
    if path.is_absolute():
        return path
    root = Path(relative_root).resolve()
    candidate = root / path
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise GovernanceError(f"{label} escapes its declared root") from exc
    return candidate


def _resolve_declared_path(
    raw: object,
    *,
    label: str,
    relative_root: Path = ROOT,
) -> Path:
    return _declared_path_candidate(raw, label=label, relative_root=relative_root).resolve(
        strict=False
    )


def _read_regular_declared_file(
    raw: object,
    *,
    label: str,
    relative_root: Path = ROOT,
) -> tuple[Path, bytes]:
    """Read one exact regular-file inode without following a final symlink."""

    candidate = _declared_path_candidate(raw, label=label, relative_root=relative_root)
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        try:
            component = cursor.lstat()
        except OSError:
            break
        if stat.S_ISLNK(component.st_mode):
            raise GovernanceError(f"{label} contains a symlink path component: {cursor}")
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise GovernanceError(f"{label} is missing: {candidate}") from exc
    if not stat.S_ISREG(before.st_mode) or candidate.is_symlink():
        raise GovernanceError(f"{label} is not a regular non-symlink file: {candidate}")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise GovernanceError(f"cannot open exact {label}: {candidate}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise GovernanceError(f"{label} changed identity while being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise GovernanceError(f"{label} changed while its bytes were read")
    finally:
        os.close(descriptor)
    try:
        current = candidate.lstat()
    except OSError as exc:
        raise GovernanceError(f"{label} path disappeared after its bytes were read") from exc
    if candidate.is_symlink() or (
        current.st_dev,
        current.st_ino,
    ) != (opened.st_dev, opened.st_ino):
        raise GovernanceError(f"{label} path changed identity while its bytes were read")
    return candidate.resolve(strict=True), b"".join(chunks)


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


NEURAL_EXECUTION_BINDING_KEYS = (
    "runner_sha256",
    "cell_plan_sha256",
    "model_registry_sha256",
    "fold_registry_sha256",
    "input_registry_sha256",
    "contrast_registry_sha256",
)

NEURAL_REGISTRY_FORMATS_REQUIRING_PARSERS = MappingProxyType(
    {
        "model_registry_sha256": "thermoroute.neural-model-registry.v4",
        "fold_registry_sha256": "thermoroute.neural-fold-registry.v4",
        "input_registry_sha256": "thermoroute.neural-input-registry.v4",
        "contrast_registry_sha256": "thermoroute.neural-contrast-registry.v4",
    }
)


def _validate_protocol_and_load_seal_once(
    protocol_path: Path,
) -> tuple[dict[str, object], Mapping[str, object], bytes, bytes]:
    """Read protocol and seal once, then validate one immutable seal object."""

    path, payload = _read_regular_declared_file(str(Path(protocol_path)), label="v4 protocol")
    try:
        document = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise GovernanceError(f"v4 protocol is not valid YAML: {path}") from exc
    if not isinstance(document, Mapping):
        raise GovernanceError("v4 protocol must be a mapping")
    if document.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise GovernanceError("v4 protocol_id mismatch")
    if document.get("version") != EXPECTED_PROTOCOL_VERSION:
        raise GovernanceError("v4 protocol version mismatch")
    status = document.get("status")
    if status != AUTHORIZED_PROTOCOL_STATUS or document.get("execution_authorized") is not True:
        raise GovernanceError(
            "neural hard-cell execution is governance-blocked before panel access: "
            f"status={status!r}, execution_authorized={document.get('execution_authorized')!r}"
        )
    protocol_digest = hashlib.sha256(payload).hexdigest()
    try:
        seal_path, seal_payload = _read_regular_declared_file(
            document.get("seal_path"), label="v4 seal"
        )
        mutable_seal = json.loads(seal_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernanceError(f"cannot read a valid v4 seal: {seal_path}") from exc
    seal = _deep_freeze(mutable_seal)
    if not isinstance(seal, Mapping):
        raise GovernanceError("v4 seal must be a mapping")
    if seal.get("format") != PROTOCOL_SEAL_FORMAT:
        raise GovernanceError("v4 seal format mismatch")
    if seal.get("status") != AUTHORIZED_PROTOCOL_STATUS:
        raise GovernanceError("v4 seal status must be SEALED")
    if seal.get("execution_authorized") is not True:
        raise GovernanceError("v4 seal does not authorize execution")
    binding = seal.get("protocol")
    if not isinstance(binding, Mapping):
        raise GovernanceError("v4 seal lacks a protocol byte binding")
    if set(binding) != {"protocol_id", "version", "path", "sha256"}:
        raise GovernanceError("v4 seal protocol binding schema mismatch")
    if binding.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise GovernanceError("v4 seal protocol_id mismatch")
    if binding.get("version") != EXPECTED_PROTOCOL_VERSION:
        raise GovernanceError("v4 seal protocol version mismatch")
    if _resolve_declared_path(binding.get("path"), label="seal protocol.path") != path:
        raise GovernanceError("v4 seal binds a different protocol path")
    if binding.get("sha256") != protocol_digest:
        raise GovernanceError("v4 seal protocol SHA-256 mismatch")
    byte_binding = {
        "scope": "PROTOCOL_BYTE_BINDING_ONLY",
        "protocol_path": str(path),
        "protocol_sha256": protocol_digest,
        "seal_path": str(seal_path),
        "seal_sha256": hashlib.sha256(seal_payload).hexdigest(),
        "status": AUTHORIZED_PROTOCOL_STATUS,
        "protocol_declares_execution_authorized": True,
        "neural_execution_authorized": False,
    }
    return byte_binding, seal, payload, seal_payload


def validate_protocol_byte_binding_only(protocol_path: Path) -> dict[str, object]:
    """Validate only a SEALED v4 protocol/seal byte loop.

    This deliberately does not return neural execution authority.  Model,
    fold, input, and contrast registries are outside this narrow check.
    """

    byte_binding, _seal, _protocol_bytes, _seal_bytes = _validate_protocol_and_load_seal_once(
        protocol_path
    )
    return byte_binding


@dataclass(frozen=True, slots=True, init=False)
class NeuralAuthoritySnapshot:
    """One immutable set of bytes captured before any fit can be planned."""

    protocol_path: Path
    seal_path: Path
    protocol_bytes: bytes
    seal_bytes: bytes
    seal_document: Mapping[str, object]
    binding_paths: Mapping[str, Path]
    binding_bytes: Mapping[str, bytes]
    binding_sha256: Mapping[str, str]
    snapshot_sha256: str
    semantic_validation_complete: bool

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("NeuralAuthoritySnapshot cannot be constructed directly")


def _capture_neural_authority_snapshot(protocol_path: Path) -> NeuralAuthoritySnapshot:
    """Capture canonical protocol, seal, runner, and registry bytes exactly once."""

    requested = Path(protocol_path).resolve(strict=False)
    canonical_protocol = Path(CANONICAL_SEALED_PROTOCOL).resolve(strict=False)
    if requested != canonical_protocol:
        raise GovernanceError(
            "neural execution authority is pinned to the canonical sealed protocol path: "
            f"{canonical_protocol}"
        )
    byte_binding, seal, protocol_bytes, seal_bytes = _validate_protocol_and_load_seal_once(
        canonical_protocol
    )
    observed_seal_path = Path(str(byte_binding["seal_path"]))
    canonical_seal = Path(CANONICAL_PROTOCOL_SEAL).resolve(strict=False)
    if observed_seal_path != canonical_seal:
        raise GovernanceError(
            "neural execution authority is pinned to the canonical protocol-seal path: "
            f"{canonical_seal}"
        )
    protocol_document = yaml.safe_load(protocol_bytes)
    if not isinstance(protocol_document, Mapping):
        raise GovernanceError("canonical neural protocol must decode to a mapping")
    declared_seal_path = _declared_path_candidate(
        protocol_document.get("seal_path"), label="canonical protocol seal_path"
    ).absolute()
    if declared_seal_path != canonical_seal:
        raise GovernanceError(
            "canonical neural protocol must name the exact canonical protocol-seal path"
        )
    protocol_binding = seal.get("protocol")
    if not isinstance(protocol_binding, Mapping):
        raise GovernanceError("canonical neural seal lacks its protocol binding")
    declared_protocol_path = _declared_path_candidate(
        protocol_binding.get("path"), label="canonical seal protocol.path"
    ).absolute()
    if declared_protocol_path != canonical_protocol:
        raise GovernanceError("canonical neural seal must name the exact canonical protocol path")
    declared = seal.get("neural_execution_bindings")
    if not isinstance(declared, Mapping):
        raise GovernanceError("v4 seal lacks neural_execution_bindings")
    if set(declared) != set(NEURAL_EXECUTION_BINDING_KEYS):
        raise GovernanceError("v4 seal neural execution binding schema mismatch")
    if set(CANONICAL_NEURAL_REGISTRY_PATHS) != set(NEURAL_EXECUTION_BINDING_KEYS):
        raise GovernanceError("canonical neural registry path schema is incomplete")
    observed_paths: dict[str, Path] = {}
    observed_hashes: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for key in NEURAL_EXECUTION_BINDING_KEYS:
        record = declared[key]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise GovernanceError(f"v4 seal neural {key} path/hash schema mismatch")
        try:
            declared_digest = _validate_sha256(
                record.get("sha256"), label=f"seal neural {key}.sha256"
            )
        except ContractError as exc:
            raise GovernanceError(f"v4 seal neural {key} SHA-256 is invalid") from exc
        canonical_path = Path(CANONICAL_NEURAL_REGISTRY_PATHS[key]).resolve(strict=False)
        declared_path = _declared_path_candidate(
            record.get("path"), label=f"neural {key}"
        ).absolute()
        if declared_path != canonical_path:
            raise GovernanceError(
                f"neural {key} path is not canonical: {declared_path} != {canonical_path}"
            )
        resolved, payload = _read_regular_declared_file(str(canonical_path), label=f"neural {key}")
        if resolved != canonical_path:
            raise GovernanceError(f"neural {key} canonical path changed while read")
        if not payload:
            raise GovernanceError(f"neural {key} is empty")
        observed_digest = hashlib.sha256(payload).hexdigest()
        if observed_digest != declared_digest:
            raise GovernanceError(
                f"neural {key} SHA-256 mismatch: {observed_digest} != {declared_digest}"
            )
        observed_paths[key] = resolved
        observed_hashes[key] = observed_digest
        payloads[key] = payload

    executing_runner = Path(__file__).resolve()
    if observed_paths["runner_sha256"] != executing_runner:
        raise GovernanceError("seal runner path does not name the executing Phase-1 source")

    expected_plan_payload = _canonical_json_bytes([cell.to_record() for cell in build_cell_plan()])
    if payloads["cell_plan_sha256"] != expected_plan_payload:
        raise GovernanceError("sealed cell-plan file is not the exact canonical 5,280-fit plan")
    if observed_hashes["cell_plan_sha256"] != EXPECTED_CELL_PLAN_SHA256:
        raise GovernanceError("sealed cell-plan SHA-256 is not the canonical Phase-1.1 plan")
    snapshot_record = {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "seal_sha256": hashlib.sha256(seal_bytes).hexdigest(),
        "bindings": {
            key: {
                "path": str(observed_paths[key]),
                "sha256": observed_hashes[key],
            }
            for key in NEURAL_EXECUTION_BINDING_KEYS
        },
        "semantic_validation_complete": False,
    }
    instance = object.__new__(NeuralAuthoritySnapshot)
    object.__setattr__(instance, "protocol_path", canonical_protocol)
    object.__setattr__(instance, "seal_path", canonical_seal)
    object.__setattr__(instance, "protocol_bytes", bytes(protocol_bytes))
    object.__setattr__(instance, "seal_bytes", bytes(seal_bytes))
    object.__setattr__(instance, "seal_document", seal)
    object.__setattr__(instance, "binding_paths", MappingProxyType(dict(observed_paths)))
    object.__setattr__(instance, "binding_bytes", MappingProxyType(dict(payloads)))
    object.__setattr__(instance, "binding_sha256", MappingProxyType(dict(observed_hashes)))
    object.__setattr__(instance, "snapshot_sha256", canonical_sha256(snapshot_record))
    object.__setattr__(instance, "semantic_validation_complete", False)
    return instance


def _require_neural_registry_semantics(snapshot: NeuralAuthoritySnapshot) -> NoReturn:
    """Terminal guard until all four canonical registry parsers are implemented."""

    if type(snapshot) is not NeuralAuthoritySnapshot:
        raise GovernanceError("neural authority requires an exact immutable snapshot")
    for key, expected_format in NEURAL_REGISTRY_FORMATS_REQUIRING_PARSERS.items():
        payload = snapshot.binding_bytes[key]
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NeuralAuthoritySemanticValidationUnavailable(
                f"{NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS}: {key} is not JSON"
            ) from exc
        if not isinstance(document, Mapping) or document.get("format") != expected_format:
            raise NeuralAuthoritySemanticValidationUnavailable(
                f"{NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS}: "
                f"{key} lacks canonical format {expected_format!r}"
            )
    raise NeuralAuthoritySemanticValidationUnavailable(
        f"{NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS}: canonical bytes "
        "were captured immutably, but model/fold/input/contrast semantics and their "
        "cross-registry invariants are intentionally not implemented"
    )


def validate_neural_execution_authority(protocol_path: Path) -> NoReturn:
    """Never authorize execution until canonical registry semantics are implemented."""

    snapshot = _capture_neural_authority_snapshot(protocol_path)
    _require_neural_registry_semantics(snapshot)


def execute_training(
    _plan: Sequence[CellPlan],
    authority_snapshot: NeuralAuthoritySnapshot,
    _output_dir: Path,
    *,
    resume: bool,
) -> NoReturn:
    """Phase-1 terminal guard: no panel read, training, scoring, or write exists."""

    del resume
    if type(authority_snapshot) is not NeuralAuthoritySnapshot:
        raise Phase1ExecutionUnavailable(
            "Phase-1 execution requires one exact immutable authority snapshot"
        )
    if not authority_snapshot.semantic_validation_complete:
        raise Phase1ExecutionUnavailable(
            f"{NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS}: no fit may start"
        )
    raise Phase1ExecutionUnavailable(
        "Phase-1 contains contracts only: neural training/scoring is intentionally "
        "unimplemented and no panel or outcome registry has been read"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the in-memory Phase-1 plan; read/write/train nothing",
    )
    parser.add_argument(
        "--print-cells",
        action="store_true",
        help="include all 5,280 typed fit records in dry-run JSON",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reserved for a future completion-hash-validated execution phase",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    plan = build_cell_plan()
    if args.print_cells and not args.dry_run:
        raise ContractError("--print-cells is available only with --dry-run")
    if args.resume and args.dry_run:
        raise ContractError("--resume cannot be combined with --dry-run")
    if args.dry_run:
        print(json.dumps(phase1_plan_manifest(plan, include_cells=args.print_cells), indent=1))
        return 0

    # Canonical-path validation is intentionally the first external boundary.
    # The proposed sealed files do not yet exist, and even synthetic canonical
    # bytes remain terminally blocked by unavailable semantic parsers.
    del plan
    validate_neural_execution_authority(Path(args.protocol))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (
        ContractError,
        GovernanceError,
        Phase1ExecutionUnavailable,
        PredictionContentValidationUnavailable,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
