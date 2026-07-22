"""Fail-closed state machine and evidence contracts for Route-A label opening.

Nothing in this module downloads or reads confirmation outcomes during
preflight.  :func:`run_opening_once` launches a fixed isolated orchestrator.  It
creates an exclusive ``OPENING_STARTED`` marker before a separate raw-only NWIS
child is permitted to acquire labels, then launches a fresh trusted scorer that
replays the raw evidence, executes every frozen model, recomputes every formal
test and writes a create-only receipt.  A transport interruption may be resumed
explicitly under the same intent and frozen request ledger.  Trusted products
are validated in a same-filesystem private directory and published as one
directory rename; a scorer interruption may therefore resume without a second
label acquisition.  Any inconsistent raw transaction or invalid canonical
trusted publication remains indeterminate and is never replaced.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence, cast
from urllib.parse import parse_qs, urlsplit
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from . import features as F
from .checkpoint import instantiate_inference_ensemble, load_inference_bundle
from .chronology import (
    CHRONOLOGY_FORMAT,
    CHRONOLOGY_STATUS,
    ChronologyError,
    DEFAULT_RECEIPT as DEFAULT_PRELABEL_CHRONOLOGY_RECEIPT,
    validate_prelabel_chronology,
)
from .confirmatory import CANDIDATE_COLUMNS, replay_candidate_evidence
from .development_replay import validate_development_replay_receipt
from .frozen_inference import (
    FrozenInferenceError,
    build_frozen_confirmation_windows,
    reconstruct_frozen_transforms,
    sequence_factory_from_metadata,
)
from .historical_inputs import (
    DAYMET_PROVIDER,
    GRIDMET_PROVIDER,
    GRIDMET_SCHEMA_PROVIDER,
    REQUEST_MAP_FORMAT,
    USER_AGENT as METEOROLOGY_USER_AGENT,
)
from .inference_gate import (
    AMENDMENT_RELATIVE as INFERENCE_AMENDMENT_RELATIVE,
    AMENDMENT_SEAL_RELATIVE as INFERENCE_AMENDMENT_SEAL_RELATIVE,
    DEFAULT_GATE_RELATIVE as DEFAULT_INFERENCE_GATE,
    InferenceGateError,
    validate_inference_amendment,
    validate_inference_amendment_seal,
    validate_inference_gate_document,
)
from .evidence import EvidenceError, FrozenPanelSpec, select_confirmatory_sites
from .model_suite import (
    ModelSuiteError,
    canonical_frame_digest,
    load_lightgbm_bundle,
    validate_development_prediction_binding,
    validate_model_suite_document,
)
from .opening_contract import (
    AcquisitionContractError,
    MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES,
    RAW_ACQUISITION_FORBIDDEN_STATE_KEYS,
    TRUSTED_STATE_KEYS,
    assert_no_symlink_components,
)
from .outcome_qc import (
    GATE_FORMAT as OUTCOME_QC_GATE_FORMAT,
    POLICY_RELATIVE as OUTCOME_QC_POLICY_RELATIVE,
    OutcomeQCGateError,
    build_outcome_qc_gate_document,
    validate_outcome_qc_gate_document,
    validate_outcome_qc_policy,
)
from .probability import (
    PlattCalibrator,
    logit,
    predict_frozen_seasonal_event_reference,
    validate_frozen_seasonal_event_reference,
)
from .quantiles import QuantileIdentityError, repair_lightgbm_quantiles
from .provenance import canonical_json_bytes, sha256_file
from .registry import targets_match_at_model_precision
from .repro import (
    assert_formal_numerical_policy,
    configure_deterministic_runtime,
    environment_fingerprint,
    numerical_runtime_contract,
    sha256_json,
    source_inventory,
    source_tree_hash,
)
from . import results as R
from .significance import (
    cluster_bootstrap_paired_effect,
    cluster_sign_flip_pvalue,
    holm_adjust,
)
from .spatial import huc2_cluster_map
from .usgs import (
    CONFIRMATORY_OUTCOME_COLUMNS,
    CONFIRMATORY_NWIS_PROVIDER,
    build_daymet_url,
    build_gridmet_wind_url,
    build_gridmet_wind_metadata_url,
    build_nwis_confirmatory_url,
    nwis_confirmatory_series_registry,
    parse_daymet_daily,
    parse_gridmet_wind_daily,
    parse_gridmet_wind_metadata,
    parse_nwis_confirmatory_daily,
)


AUTHORIZATION_FORMAT = "thermoroute.route-a-opening-authorization.v1"
PROTOCOL_SEAL_FORMAT = "thermoroute.route-a-protocol-seal.v1"
DEFAULT_PROTOCOL_SEAL = "protocols/route_a_protocol_seal_v1.json"
MODEL_SUITE_FORMAT = "thermoroute.route-a-model-suite.v1"
INPUT_MANIFEST_FORMAT = "thermoroute.route-a-prelabel-inputs.v1"
ACQUISITION_MANIFEST_FORMAT = "thermoroute.route-a-opened-inputs.v1"
INTENT_FORMAT = "thermoroute.route-a-opening-intent.v1"
RECEIPT_FORMAT = "thermoroute.route-a-opening-receipt.v1"
STATISTICS_FORMAT = "thermoroute.route-a-confirmatory-statistics.v1"
OUTCOME_QUALITY_AUDIT_FORMAT = "thermoroute.route-a-outcome-quality-audit.v1"
APPROVED_TARGET_SENSITIVITY_FORMAT = (
    "thermoroute.route-a-approved-target-sensitivity.v1"
)
SPATIAL_SENSITIVITY_FORMAT = "thermoroute.route-a-spatial-sensitivity.v1"
PROBABILISTIC_EVALUATION_FORMAT = (
    "thermoroute.route-a-probabilistic-evaluation.v1"
)
ACQUISITION_WORK_ORDER_FORMAT = "thermoroute.route-a-acquisition-work-order.v1"
ACQUISITION_REQUEST_MAP_FORMAT = "thermoroute.route-a-opened-request-map.v1"
ACQUISITION_REQUEST_LEDGER_FORMAT = (
    "thermoroute.route-a-acquisition-request-ledger.v1"
)
ACQUISITION_ATTEMPT_START_FORMAT = (
    "thermoroute.route-a-acquisition-attempt-start.v1"
)
ACQUISITION_ATTEMPT_RESULT_FORMAT = (
    "thermoroute.route-a-acquisition-attempt-result.v1"
)
ACQUISITION_ATTEMPT_INDEX_FORMAT = (
    "thermoroute.route-a-acquisition-attempt-index.v1"
)

_FIXED_ENTRYPOINTS = {
    "orchestrator": "scripts/route_a_opening_orchestrator.py",
    "acquisition": "scripts/route_a_outcome_acquisition.py",
    "trusted_scorer": "scripts/route_a_trusted_scorer.py",
}

_TRUSTED_STATE_KEYS = TRUSTED_STATE_KEYS
_TRUSTED_STAGE_PREFIX = ".trusted-stage-v1-"
_TRUSTED_PUBLICATION_LOCK = ".trusted-publication-v1.lock"

OPENING_ACQUIRED_FIELDS = frozenset({"WTEMP", "FLOW", "WLEVEL"})
BUILTIN_MODELS = frozenset({"Persistence", "DampedPersistence", "Climatology"})
SUPPORTED_EXECUTORS = frozenset({
    "builtin",
    "thermoroute_bundle",
    "lstm_bundle",
    "lightgbm_bundle",
})
CONTROL_INTERVENTIONS: Mapping[str, Mapping[str, Any]] = {
    "DampedPriorOnly": {"use_prior": False, "residual_model": False},
    "TR-noDynamicPrior": {"use_prior": False},
    "TR-fixedKappa": {"fixed_kappa": True},
    "TR-noRouter": {"use_router": False},
    "TR-noMoE": {"use_moe": False},
    "TR-noTCN": {"use_tcn": False},
    "TR-unbounded": {"delta_scale": None},
}

THERMOROUTE_INTERVENTION_DEFAULTS: Mapping[str, Any] = {
    "station_agnostic": False,
    "use_prior": True,
    "use_router": True,
    "use_moe": True,
    "sparse_router": True,
    "fixed_kappa": False,
    "use_tcn": True,
    "residual_model": True,
    "safety_anchor": "damped",
    "use_wlevel": False,
}


class OpeningContractError(RuntimeError):
    """A pre-label or post-opening evidence contract is incomplete."""


class OpeningAlreadyStarted(OpeningContractError):
    """The one permitted opening has already begun, completed or crashed."""


@contextmanager
def _secure_directory_chain(path: Path, *, create: bool) -> Iterator[int]:
    """Open a directory from the filesystem root without following symlinks."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in absolute.parts[1:]:
            created = False
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            if created:
                os.fsync(child)
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    except OSError as exc:
        raise OpeningContractError(
            f"opening state directory traversal is unsafe: {absolute}"
        ) from exc
    finally:
        os.close(descriptor)


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise OpeningContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise OpeningContractError(f"{label} must be a JSON object")
    return value


def _resolve_inside(root: Path, relative: object, *, kind: str = "file") -> Path:
    root = root.resolve()
    raw = Path(str(relative))
    if raw.is_absolute():
        raise OpeningContractError(f"frozen artifact path must be relative: {raw}")
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise OpeningContractError(f"frozen artifact escapes repository root: {raw}")
    exists = path.is_dir() if kind == "directory" else path.is_file()
    if not exists:
        raise OpeningContractError(f"frozen {kind} is missing: {raw}")
    return path


def _relative(root: Path, path: str | Path) -> str:
    resolved = Path(path).resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise OpeningContractError(f"artifact is outside repository root: {resolved}")
    return resolved.relative_to(root).as_posix()


def _verify_file_binding(root: Path, binding: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(binding, Mapping):
        raise OpeningContractError(f"{label} binding is not an object")
    if {"path", "sha256"} - set(binding):
        raise OpeningContractError(f"{label} binding lacks path/sha256")
    path = _resolve_inside(root, binding["path"])
    actual = sha256_file(path)
    if actual != str(binding["sha256"]):
        raise OpeningContractError(
            f"{label} checksum mismatch: expected {binding['sha256']}, got {actual}"
        )
    return path


def _verify_canonical_file_binding(
    root: Path,
    binding: Mapping[str, Any],
    *,
    expected_path: str | Path,
    label: str,
) -> Path:
    """Verify bytes and require the binding to name its exact state path."""
    path = _verify_file_binding(root, binding, label=label)
    expected = Path(os.path.abspath(os.fspath(expected_path)))
    if path != expected:
        raise OpeningContractError(f"{label} path is noncanonical")
    return path


def _binding(root: Path, path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": _relative(root, resolved), "sha256": sha256_file(resolved)}


def _logical_binding(
    root: Path,
    physical_path: str | Path,
    logical_path: str | Path,
) -> dict[str, str]:
    """Bind staged bytes to the immutable path they will have after publish."""
    physical = Path(physical_path)
    if not physical.is_file() or physical.is_symlink():
        raise OpeningContractError(
            f"trusted staged artifact is absent or unsafe: {physical}"
        )
    logical = Path(os.path.abspath(os.fspath(logical_path)))
    return {
        "path": _relative(root, logical),
        "sha256": sha256_file(physical),
    }


def exclusive_create_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create and fsync one immutable JSON file without replacement semantics."""
    _atomic_create_bytes(
        Path(os.path.abspath(os.fspath(path))),
        canonical_json_bytes(dict(value)),
    )


def _exclusive_create_bytes(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    with _secure_directory_chain(path.parent, create=True) as parent_descriptor:
        try:
            descriptor = os.open(
                path.name,
                flags,
                0o444,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as exc:
            raise OpeningAlreadyStarted(
                f"refusing to replace one-time artifact: {path}"
            ) from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_descriptor)


def _atomic_create_fault(_point: str, _path: Path) -> None:
    """No-op hook replaced only by atomic-publication crash tests."""


def _cleanup_atomic_create_temps(
    parent_descriptor: int,
    *,
    final_name: str,
    expected_payload: bytes,
    remove: bool = True,
) -> set[str]:
    """Validate and optionally remove safe temps from an interrupted create."""
    parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o022
    ):
        raise OpeningContractError(
            "atomic-create parent is not owner-controlled"
        )
    temporary_pattern = re.compile(
        rf"\.{re.escape(final_name)}\.[a-z0-9_]{{8}}\.tmp"
    )
    try:
        final_metadata = os.stat(
            final_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
    except FileNotFoundError:
        final_metadata = None
    safe: set[str] = set()
    for name in os.listdir(parent_descriptor):
        if temporary_pattern.fullmatch(name) is None:
            continue
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
            published_link = bool(
                final_metadata is not None
                and stat.S_ISREG(final_metadata.st_mode)
                and metadata.st_dev == final_metadata.st_dev
                and metadata.st_ino == final_metadata.st_ino
                and metadata.st_nlink == 2
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != parent.st_dev
                or metadata.st_mode & 0o022
                or (metadata.st_nlink != 1 and not published_link)
                or payload != expected_payload
            ):
                # A SIGKILL during the temporary write can leave an incomplete
                # owner-only nlink=1 file.  It is safe to unlink because the
                # final name was never linked; any other topology fails closed.
                if (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_uid == os.geteuid()
                    and metadata.st_dev == parent.st_dev
                    and not metadata.st_mode & 0o022
                    and metadata.st_nlink == 1
                ):
                    safe.add(name)
                    if remove:
                        os.unlink(name, dir_fd=parent_descriptor)
                    continue
                raise OpeningContractError(
                    "atomic-create temporary artifact has unsafe metadata"
                )
            safe.add(name)
            if remove:
                os.unlink(name, dir_fd=parent_descriptor)
        finally:
            os.close(descriptor)
    if remove:
        os.fsync(parent_descriptor)
    return safe


def _cleanup_atomic_create_path_temps(path: Path, payload: bytes) -> None:
    """Recover safe pre-link or post-link remnants for an existing state path."""
    _validate_atomic_final_file(path, payload, cleanup_temps=True)


def _validate_atomic_final_file(
    path: Path, payload: bytes, *, cleanup_temps: bool
) -> None:
    """Require one immutable final inode or its one known post-link temp."""
    path = Path(os.path.abspath(os.fspath(path)))
    with _secure_directory_chain(path.parent, create=False) as parent_descriptor:
        safe_temps = _cleanup_atomic_create_temps(
            parent_descriptor,
            final_name=path.name,
            expected_payload=payload,
            remove=False,
        )
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            parent = os.fstat(parent_descriptor)
            metadata = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                actual_payload = handle.read()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != parent.st_dev
                or metadata.st_mode & 0o222
                or actual_payload != payload
                or metadata.st_nlink not in {1, 2}
            ):
                raise OpeningContractError(
                    "authoritative atomic final has unsafe metadata or bytes"
                )
            linked_temps = 0
            for name in safe_temps:
                temporary = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (
                    temporary.st_dev == metadata.st_dev
                    and temporary.st_ino == metadata.st_ino
                ):
                    linked_temps += 1
            if (
                metadata.st_nlink == 2 and linked_temps != 1
            ) or (metadata.st_nlink == 1 and linked_temps != 0):
                raise OpeningContractError(
                    "authoritative atomic final has an unknown hard link"
                )
        finally:
            os.close(descriptor)
        if cleanup_temps:
            _cleanup_atomic_create_temps(
                parent_descriptor,
                final_name=path.name,
                expected_payload=payload,
                remove=True,
            )
            final = os.stat(
                path.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if final.st_nlink != 1:
                raise OpeningContractError(
                    "atomic final still has an unknown hard link after cleanup"
                )


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Create immutable bytes atomically; a writer crash cannot expose a prefix."""
    path = Path(os.path.abspath(os.fspath(path)))
    with _secure_directory_chain(path.parent, create=True) as parent_descriptor:
        # The parent directory itself is durable state.  A process can die
        # after mkdir/fsync in ``_secure_directory_chain`` but before mkstemp;
        # expose that real crash window to subprocess tests and recovery.
        _atomic_create_fault("after_parent_directory_create_before_temp", path)
        _cleanup_atomic_create_temps(
            parent_descriptor,
            final_name=path.name,
            expected_payload=payload,
        )
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OpeningAlreadyStarted(
                f"refusing to replace one-time artifact: {path}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                split = max(1, len(payload) // 2) if payload else 0
                handle.write(payload[:split])
                handle.flush()
                _atomic_create_fault("after_temporary_prefix_write", path)
                handle.write(payload[split:])
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                _atomic_create_fault(
                    "after_final_mode_before_inode_fsync", path
                )
                os.fsync(handle.fileno())
            _atomic_create_fault("before_no_replace_link", path)
            try:
                os.link(
                    temporary.name,
                    path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise OpeningAlreadyStarted(
                    f"refusing to replace one-time artifact: {path}"
                ) from exc
            os.fsync(parent_descriptor)
            _atomic_create_fault("after_no_replace_link", path)
        finally:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass


def _exclusive_create_parquet(path: Path, frame: pd.DataFrame) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    with _secure_directory_chain(path.parent, create=True) as parent_descriptor:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OpeningAlreadyStarted(
                f"refusing to replace one-time artifact: {path}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        os.close(descriptor)
        try:
            frame.to_parquet(temporary, index=False)
            temporary_descriptor = os.open(
                temporary.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            try:
                os.fchmod(temporary_descriptor, 0o444)
                os.fsync(temporary_descriptor)
            finally:
                os.close(temporary_descriptor)
            try:
                os.link(
                    temporary.name,
                    path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise OpeningAlreadyStarted(
                    f"refusing to replace one-time artifact: {path}"
                ) from exc
            os.fsync(parent_descriptor)
        finally:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass


_NUMERICAL_LOCK_DISTRIBUTIONS = (
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch",
    "statsmodels", "pyarrow",
)
_GOLDEN_INFERENCE_SHA256 = (
    "1018e45f2145415b096f376548a29d994f0b2dcb2145452bd401e4749d9f719f"
)
_DETERMINISTIC_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}


def _requirements_lock_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise OpeningContractError(
                "Route-A requirements lock contains a non-exact requirement"
            )
        name, version = (value.strip() for value in line.split("==", 1))
        normalised = name.lower().replace("_", "-")
        if not name or not version or normalised in versions:
            raise OpeningContractError("Route-A requirements lock is malformed")
        versions[normalised] = version
    required = {name.lower().replace("_", "-") for name in _NUMERICAL_LOCK_DISTRIBUTIONS}
    if not required <= set(versions):
        raise OpeningContractError("Route-A requirements lock omits a numerical dependency")
    return versions


def _validate_hashed_requirements_lock(path: Path) -> dict[str, Any]:
    """Require one or more SHA-256 hashes on every logical requirement stanza."""
    stanzas: list[str] = []
    current: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continued = stripped.endswith("\\")
        content = stripped[:-1].strip() if continued else stripped
        current.append(content)
        if not continued:
            stanzas.append(" ".join(current))
            current = []
    if current:
        raise OpeningContractError("hashed requirements lock has a dangling continuation")
    if not stanzas:
        raise OpeningContractError("hashed requirements lock contains no requirements")
    versions: dict[str, str] = {}
    hash_count = 0
    hash_pattern = re.compile(r"--hash=sha256:([0-9a-f]{64})(?:\s|$)")
    for stanza in stanzas:
        requirement = stanza.split()[0]
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s\\]+)", requirement)
        if match is None:
            raise OpeningContractError(
                "hashed requirements lock contains a non-exact requirement"
            )
        name = match.group(1).lower().replace("_", "-")
        version = match.group(2)
        if name in versions:
            raise OpeningContractError(
                "hashed requirements lock repeats a distribution"
            )
        hashes = hash_pattern.findall(f"{stanza} ")
        all_hash_options = re.findall(r"--hash=([^\s]+)", stanza)
        if not hashes or len(hashes) != len(all_hash_options):
            raise OpeningContractError(
                f"hashed requirements lock stanza is not fully SHA-256 bound: {name}"
            )
        versions[name] = version
        hash_count += len(hashes)
    return {
        "requirement_stanza_count": len(stanzas),
        "sha256_hash_count": hash_count,
        "every_stanza_hashed": True,
        "versions": versions,
    }


def _golden_inference_probe() -> str:
    """Exercise NumPy, Torch and LightGBM on a fixed CPU numerical probe."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - mandatory production dependency
        raise OpeningContractError("LightGBM is absent from the Route-A runtime") from exc

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work;
        # an already-fixed value is checked in the child environment attestation.
        pass
    torch.use_deterministic_algorithms(True)
    x = np.asarray(
        [[-1.5, 0.0, 2.0], [0.25, -0.75, 1.25], [2.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    weight = np.asarray(
        [[0.25, -0.5, 0.75], [-0.4, 0.1, 0.2]], dtype=np.float64
    )
    bias = np.asarray([0.125, -0.25], dtype=np.float64)
    numpy_output = np.tanh(x @ weight.T + bias)
    torch_output = torch.sigmoid(torch.nn.functional.linear(
        torch.from_numpy(x), torch.from_numpy(weight), torch.from_numpy(bias)
    )).numpy()
    train_x = np.asarray(
        [[float(index), float((index * index) % 7), float(index % 3)]
         for index in range(16)],
        dtype=float,
    )
    train_y = np.asarray(
        [0.25 * index + ((index % 4) - 1.5) * 0.1 for index in range(16)],
        dtype=float,
    )
    dataset = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    booster = lgb.train(
        {
            "objective": "regression", "metric": "None", "learning_rate": 0.2,
            "num_leaves": 3, "max_depth": 2, "min_data_in_leaf": 1,
            "min_data_in_bin": 1, "feature_pre_filter": False,
            "deterministic": True, "force_col_wise": True, "seed": 1729,
            "feature_fraction_seed": 1729, "bagging_seed": 1729,
            "data_random_seed": 1729, "num_threads": 1, "verbosity": -1,
        },
        dataset,
        num_boost_round=4,
    )
    lightgbm_output = booster.predict(train_x[[0, 5, 10, 15]], num_threads=1)
    probe = {
        "version": "route-a-golden-inference-v1",
        "numpy": np.round(numpy_output, 12).tolist(),
        "torch": np.round(torch_output, 12).tolist(),
        "lightgbm": np.round(lightgbm_output, 12).tolist(),
    }
    digest = sha256_json(probe)
    if digest != _GOLDEN_INFERENCE_SHA256:
        raise OpeningContractError(
            "golden CPU inference probe differs from the frozen Route-A runtime"
        )
    return digest


def _route_a_environment_contract(root: Path) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        formal_policy = configure_deterministic_runtime()
    except RuntimeError as exc:
        raise OpeningContractError(
            "cannot apply the formal Route-A numerical policy"
        ) from exc
    lock_path = _resolve_inside(root, "requirements-lock.txt")
    hashed_lock_path = _resolve_inside(
        root, "requirements-lock-py312-hashed.txt"
    )
    locked = _requirements_lock_versions(lock_path)
    hashed_lock = _validate_hashed_requirements_lock(hashed_lock_path)
    if any(
        hashed_lock["versions"].get(name) != version
        for name, version in locked.items()
    ):
        raise OpeningContractError(
            "hashed and direct Route-A requirement locks differ"
        )
    installed: dict[str, str] = {}
    for distribution in _NUMERICAL_LOCK_DISTRIBUTIONS:
        normalised = distribution.lower().replace("_", "-")
        try:
            installed[normalised] = version(distribution)
        except PackageNotFoundError as exc:
            raise OpeningContractError(
                f"Route-A numerical dependency is absent: {distribution}"
            ) from exc
        if installed[normalised] != locked[normalised]:
            raise OpeningContractError(
                f"Route-A runtime {distribution}={installed[normalised]} "
                f"does not match lock {locked[normalised]}"
            )
    runtime = numerical_runtime_contract()
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise OpeningContractError("Route-A Python executable is absent")
    return {
        "format": "thermoroute.route-a-runtime.v1",
        "requirements_lock": _binding(root, lock_path),
        "hashed_requirements_lock": {
            **_binding(root, hashed_lock_path),
            "requirement_stanza_count": hashed_lock[
                "requirement_stanza_count"
            ],
            "sha256_hash_count": hashed_lock["sha256_hash_count"],
            "every_stanza_hashed": True,
        },
        "installed_version_validation": (
            "every installed Route-A numerical distribution is checked against "
            "the direct exact-version lock; the fully hashed Python-3.12 lock "
            "is an additional supply-chain binding"
        ),
        "installed_versions": dict(sorted(installed.items())),
        "numerical_runtime_contract": runtime,
        "runtime_sha256": sha256_json(runtime),
        "python_executable": {
            "invoked_path": sys.executable,
            "realpath": str(executable),
            "sha256": sha256_file(executable),
        },
        "golden_inference_sha256": _golden_inference_probe(),
        "formal_numerical_policy": formal_policy,
        "deterministic_child_policy": {
            "device": "cpu",
            "environment": dict(_DETERMINISTIC_ENVIRONMENT),
            "torch_deterministic_algorithms": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "lightgbm_num_threads": 1,
        },
    }


def _fixed_code_identity(root: Path) -> dict[str, Any]:
    expected_opening = (root / "src" / "thermoroute" / "opening.py").resolve()
    loaded_opening = Path(__file__).resolve()
    if loaded_opening != expected_opening:
        raise OpeningContractError(
            "loaded thermoroute.opening module is not the repository implementation"
        )
    module_names = {
        "thermoroute.opening": "src/thermoroute/opening.py",
        "thermoroute.model_suite": "src/thermoroute/model_suite.py",
        "thermoroute.frozen_inference": "src/thermoroute/frozen_inference.py",
        "thermoroute.datasets": "src/thermoroute/datasets.py",
        "thermoroute.provenance": "src/thermoroute/provenance.py",
        "thermoroute.usgs": "src/thermoroute/usgs.py",
        "thermoroute.inference_gate": "src/thermoroute/inference_gate.py",
        "thermoroute.outcome_qc": "src/thermoroute/outcome_qc.py",
        "thermoroute.quantiles": "src/thermoroute/quantiles.py",
    }
    modules: dict[str, dict[str, str]] = {}
    for name, relative in module_names.items():
        module = sys.modules.get(name)
        loaded_file = getattr(module, "__file__", None)
        expected = (root / relative).resolve()
        if loaded_file is None or Path(loaded_file).resolve() != expected:
            raise OpeningContractError(f"loaded module realpath changed: {name}")
        modules[name] = {
            "path": relative,
            "realpath": str(expected),
            "sha256": sha256_file(expected),
        }
    entrypoints = {
        name: {
            "path": relative,
            "realpath": str(_resolve_inside(root, relative).resolve()),
            "sha256": sha256_file(_resolve_inside(root, relative)),
        }
        for name, relative in _FIXED_ENTRYPOINTS.items()
    }
    files = {
        relative: {
            "path": relative,
            "realpath": str(_resolve_inside(root, relative).resolve()),
            "sha256": sha256_file(_resolve_inside(root, relative)),
        }
        for relative in (
            "src/thermoroute/opening_contract.py",
            "src/thermoroute/outcome_acquisition.py",
        )
    }
    stable = {"modules": modules, "files": files, "entrypoints": entrypoints}
    return {
        "format": "thermoroute.route-a-fixed-code.v1",
        **stable,
        "sha256": sha256_json(stable),
    }


def _validate_portable_runtime_identity(
    frozen: object, current: Mapping[str, Any]
) -> None:
    """Validate a gitless runtime while treating absolute paths as audit-only."""
    if not isinstance(frozen, Mapping):
        raise OpeningContractError("authorized numerical runtime is malformed")
    frozen_view = dict(frozen)
    current_view = dict(current)
    frozen_python = frozen_view.get("python_executable")
    current_python = current_view.get("python_executable")
    if not isinstance(frozen_python, Mapping) or not isinstance(
        current_python, Mapping
    ):
        raise OpeningContractError("authorized Python identity is malformed")
    if frozen_python.get("sha256") != current_python.get("sha256"):
        raise OpeningContractError(
            "gitless release Python binary checksum differs from authorization"
        )
    for python_identity in (frozen_view, current_view):
        value = dict(python_identity["python_executable"])
        if not isinstance(value.get("invoked_path"), str) or not isinstance(
            value.get("realpath"), str
        ):
            raise OpeningContractError("authorized Python audit path is malformed")
        value["invoked_path"] = "<portable-audit-path>"
        value["realpath"] = "<portable-audit-path>"
        python_identity["python_executable"] = value
    if frozen_view != current_view:
        raise OpeningContractError(
            "gitless release numerical runtime differs from authorization"
        )


def _validate_portable_fixed_code_identity(
    frozen: object, current: Mapping[str, Any]
) -> None:
    """Validate relative file identities without requiring the original root."""
    if not isinstance(frozen, Mapping) or frozen.get("format") != current.get(
        "format"
    ):
        raise OpeningContractError("authorized fixed-code identity is malformed")
    frozen_stable = {
        group: frozen.get(group) for group in ("modules", "files", "entrypoints")
    }
    if frozen.get("sha256") != sha256_json(frozen_stable):
        raise OpeningContractError("authorized fixed-code self digest changed")
    for group in ("modules", "files", "entrypoints"):
        old_group = frozen.get(group)
        new_group = current.get(group)
        if not isinstance(old_group, Mapping) or not isinstance(
            new_group, Mapping
        ) or set(old_group) != set(new_group):
            raise OpeningContractError(
                f"gitless release fixed-code {group} registry changed"
            )
        for name in old_group:
            old = old_group[name]
            new = new_group[name]
            if not isinstance(old, Mapping) or not isinstance(new, Mapping):
                raise OpeningContractError("fixed-code file binding is malformed")
            if (
                set(old) != {"path", "realpath", "sha256"}
                or not isinstance(old.get("realpath"), str)
                or old.get("path") != new.get("path")
                or old.get("sha256") != new.get("sha256")
            ):
                raise OpeningContractError(
                    f"gitless release fixed-code bytes changed: {group}/{name}"
                )


def _canonical_state_paths(
    root: Path,
    *,
    protocol_sha256: str,
    source_tree_sha256: str,
    model_suite_sha256: str,
    prelabel_inputs_sha256: str,
    prelabel_chronology_sha256: str,
    inference_gate_sha256: str,
    inference_amendment_seal_sha256: str,
    outcome_qc_policy_sha256: str,
) -> dict[str, str]:
    namespace = sha256_json({
        "protocol_sha256": protocol_sha256,
        "source_tree_sha256": source_tree_sha256,
        "model_suite_sha256": model_suite_sha256,
        "prelabel_inputs_sha256": prelabel_inputs_sha256,
        "prelabel_chronology_sha256": prelabel_chronology_sha256,
        "inference_gate_sha256": inference_gate_sha256,
        "inference_amendment_seal_sha256": inference_amendment_seal_sha256,
        "outcome_qc_policy_sha256": outcome_qc_policy_sha256,
    })[:24]
    base = Path("outputs") / "confirmatory" / f"route_a_{namespace}"
    values = {
        "namespace": namespace,
        "run_directory": base.as_posix(),
        "work_order": (base / "acquisition_work_order_v1.json").as_posix(),
        "intent": (base / "opening_intent_v1.json").as_posix(),
        "transport_root": (base / "transport").as_posix(),
        "raw_nwis_root": (base / "transport" / "raw_nwis_v1").as_posix(),
        "raw_nwis_snapshot_index": (
            base / "transport" / "raw_nwis_v1" / "snapshot_index.json"
        ).as_posix(),
        "acquisition_request_map": (
            base / "acquisition" / "source_request_map_v1.json"
        ).as_posix(),
        "temporal_outcomes": (
            base / "acquisition" / "temporal_outcomes_v1.parquet"
        ).as_posix(),
        "external_outcomes": (
            base / "acquisition" / "external_outcomes_v1.parquet"
        ).as_posix(),
        "acquisition_manifest": (
            base / "acquisition" / "acquisition_manifest_v1.json"
        ).as_posix(),
        "availability_registry": (
            base / "trusted" / "availability_registry_v1.csv"
        ).as_posix(),
        "outcome_quality_audit": (
            base / "trusted" / "outcome_quality_audit_v1.json"
        ).as_posix(),
        "outcome_qc_gate": (
            base / "trusted" / "outcome_qc_gate_v1.json"
        ).as_posix(),
        "approved_target_sensitivity": (
            base / "trusted" / "approved_target_sensitivity_v1.json"
        ).as_posix(),
        "spatial_sensitivity": (
            base / "trusted" / "spatial_sensitivity_v1.json"
        ).as_posix(),
        "probabilistic_evaluation": (
            base / "trusted" / "probabilistic_evaluation_v1.json"
        ).as_posix(),
        "temporal_predictions": (
            base / "trusted" / "temporal_predictions_v1.parquet"
        ).as_posix(),
        "external_predictions": (
            base / "trusted" / "external_predictions_v1.parquet"
        ).as_posix(),
        "statistics": (base / "trusted" / "statistics_v1.json").as_posix(),
        "report": (base / "trusted" / "report_v1.md").as_posix(),
        "receipt": (base / "opening_receipt_v1.json").as_posix(),
        "receipt_sha256": (base / "opening_receipt_v1.sha256").as_posix(),
    }
    for relative in values.values():
        if relative == namespace:
            continue
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise OpeningContractError("canonical opening state path escapes repository")
    return values


def _secure_canonical_state_paths(
    root: Path,
    state_paths: Mapping[str, str],
) -> dict[str, Path]:
    """Preserve lexical state paths and reject a symlink at every component."""
    secured: dict[str, Path] = {}
    for key, value in state_paths.items():
        if key == "namespace":
            continue
        relative = Path(value)
        if relative.is_absolute():
            raise OpeningContractError("opening state path must be relative")
        lexical = root / relative
        try:
            secured[key] = assert_no_symlink_components(root, lexical)
        except AcquisitionContractError as exc:
            raise OpeningContractError(
                f"opening state path is unsafe ({key})"
            ) from exc
    return secured


_UNSAFE_GIT_ENVIRONMENT = frozenset({
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_DIR",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_EXEC_PATH",
    "GIT_EXTERNAL_DIFF",
    "GIT_GRAFT_FILE",
    "GIT_GLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_LITERAL_PATHSPECS",
    "GIT_NAMESPACE",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_QUARANTINE_PATH",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
})


def _safe_git_environment() -> dict[str, str]:
    """Return a Git environment that cannot redirect the repository/history.

    Harmless presentation settings such as ``GIT_PAGER`` remain permitted.  Any
    variable that can replace the object database, index, worktree, pathspec,
    graft/shallow boundary or configuration is rejected rather than silently
    ignored: a formal one-shot opening must make environmental drift visible.
    """
    unsafe = sorted(
        key
        for key, value in os.environ.items()
        if key in _UNSAFE_GIT_ENVIRONMENT
        or key == "GIT_CONFIG_PARAMETERS"
        or key.startswith("GIT_CONFIG_")
        or (key == "GIT_NO_REPLACE_OBJECTS" and value != "1")
    )
    if unsafe:
        raise OpeningContractError(
            "live opening prohibits ambient Git repository/history overrides: "
            f"{unsafe}"
        )
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _run_live_git(
    root: Path,
    *arguments: str,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        cwd=root,
        env=_safe_git_environment(),
        text=text,
        capture_output=True,
        check=False,
    )


def _assert_safe_live_git_repository(root: Path) -> None:
    """Reject Git mechanisms that can rewrite the history seen by preflight."""
    root = root.resolve()
    top = _run_live_git(root, "rev-parse", "--show-toplevel")
    if top.returncode or Path(str(top.stdout).strip()).resolve() != root:
        raise OpeningContractError(
            "live opening requires the exact repository Git top-level"
        )

    shallow = _run_live_git(root, "rev-parse", "--is-shallow-repository")
    if shallow.returncode or str(shallow.stdout).strip() != "false":
        raise OpeningContractError(
            "live opening prohibits a shallow or indeterminate Git repository"
        )

    replacements = _run_live_git(
        root, "for-each-ref", "--format=%(refname)", "refs/replace/"
    )
    replacement_refs = [
        line for line in str(replacements.stdout).splitlines() if line
    ]
    if replacements.returncode or replacement_refs:
        raise OpeningContractError(
            "live opening prohibits Git replacement refs: "
            f"{replacement_refs[:10]}"
        )

    graft_location = _run_live_git(
        root,
        "rev-parse",
        "--path-format=absolute",
        "--git-path",
        "info/grafts",
    )
    if graft_location.returncode or not str(graft_location.stdout).strip():
        raise OpeningContractError("cannot resolve the Git graft file location")
    graft_path = Path(str(graft_location.stdout).strip())
    if graft_path.exists() or graft_path.is_symlink():
        raise OpeningContractError(
            f"live opening prohibits a Git graft file: {graft_path}"
        )


@contextmanager
def _git_replacement_objects_disabled() -> Iterator[None]:
    """Protect imported chronology helpers whose Git subprocesses inherit env."""
    previous = os.environ.get("GIT_NO_REPLACE_OBJECTS")
    os.environ["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GIT_NO_REPLACE_OBJECTS", None)
        else:
            os.environ["GIT_NO_REPLACE_OBJECTS"] = previous


def _git_output(root: Path, *arguments: str) -> str:
    _assert_safe_live_git_repository(root)
    result = _run_live_git(root, *arguments)
    if result.returncode:
        raise OpeningContractError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout


def _live_git_state(root: Path) -> dict[str, Any]:
    """Return exact provenance after the live-history safety gate passes."""
    _assert_safe_live_git_repository(root)
    commit = _run_live_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    status = _run_live_git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    if commit.returncode or status.returncode:
        raise OpeningContractError("cannot resolve the safe live Git state")
    commit_sha = str(commit.stdout).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise OpeningContractError("live Git HEAD is not a full SHA-1 commit")
    return {
        "available": True,
        "commit": commit_sha,
        "dirty": bool(str(status.stdout)),
    }


def _require_only_untracked_authorization(
    root: Path,
    authorization_path: Path,
) -> None:
    """Allow exactly the create-only authorization as post-freeze Git dirt.

    ``--porcelain -z`` disables path quoting, so equality is byte-for-byte even
    when a repository path contains spaces.  Tracked edits, ignored output,
    another untracked file, or committing the authorization all fail closed.
    """
    relative = _relative(root, authorization_path)
    actual = _git_output(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    expected = f"?? {relative}\0"
    if actual != expected:
        raise OpeningContractError(
            "post-freeze worktree must contain only the create-only untracked "
            f"authorization; expected {expected!r}, got {actual!r}"
        )


def _require_authorization_path_trackable(root: Path, path: Path) -> str:
    relative = _relative(root, path)
    _assert_safe_live_git_repository(root)
    result = _run_live_git(root, "check-ignore", "--quiet", "--", relative)
    if result.returncode == 0:
        raise OpeningContractError(
            "opening authorization path is ignored and cannot be the sole audited dirt"
        )
    if result.returncode != 1:
        raise OpeningContractError(
            f"git check-ignore failed for authorization path: {result.stderr.strip()}"
        )
    return relative


def _is_document_only_postopening_descendant(
    root: Path,
    authorization: Mapping[str, Any],
    *,
    compute_commit: object,
    current_commit: object,
) -> bool:
    """Permit the explicitly separated manuscript commit after completion.

    The authorization remains bound to the exact compute commit.  A later HEAD
    is accepted only after the canonical receipt exists and only when the
    cumulative committed diff is additions/modifications under the frozen
    manuscript whitelist.  Chronology validation separately rejects any
    post-model source/control or frozen-artifact touch, including reverted
    commits.
    """
    compute = str(compute_commit)
    current = str(current_commit)
    _assert_safe_live_git_repository(root)
    if not re.fullmatch(r"[0-9a-f]{40}", compute) or not re.fullmatch(
        r"[0-9a-f]{40}", current
    ):
        return False
    state = authorization.get("state_paths")
    if not isinstance(state, Mapping):
        return False
    receipt_relative = state.get("receipt")
    if not isinstance(receipt_relative, str):
        return False
    try:
        _resolve_inside(root, receipt_relative)
    except OpeningContractError:
        return False
    relation = _run_live_git(
        root, "merge-base", "--is-ancestor", compute, current, text=False
    )
    if relation.returncode:
        return False
    history = _run_live_git(
        root, "rev-list", "--reverse", f"{compute}..{current}"
    )
    if history.returncode:
        return False
    commits = [line for line in str(history.stdout).splitlines() if line]
    if not commits:
        return False
    for commit in commits:
        changed = _run_live_git(
            root,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "-r",
            "-m",
            "-z",
            commit,
            text=False,
        )
        if changed.returncode:
            return False
        fields = changed.stdout.split(b"\0")
        if fields and fields[-1] == b"":
            fields.pop()
        if not fields or len(fields) % 2:
            return False
        for offset in range(0, len(fields), 2):
            try:
                status = fields[offset].decode("ascii", errors="strict")
                relative = fields[offset + 1].decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return False
            if status not in {"A", "M"} or not (
                relative == "README.md" or relative.startswith("paper/")
            ):
                return False
    return True


def _required_models(
    protocol: Mapping[str, Any], *, cohort: str
) -> tuple[str, ...]:
    """Resolve model registries only from the machine-readable protocol."""
    inference = protocol.get("primary_inference_contract")
    if not isinstance(inference, Mapping):
        raise OpeningContractError("protocol lacks primary_inference_contract")
    primary = inference.get("primary_models")
    controls = inference.get("mandatory_exploratory_architecture_controls")
    if not isinstance(primary, list) or not primary:
        raise OpeningContractError("protocol primary_models is absent or empty")
    if not isinstance(controls, list):
        raise OpeningContractError(
            "protocol mandatory_exploratory_architecture_controls is absent"
        )
    primary_values = tuple(str(value) for value in primary)
    control_values = tuple(str(value) for value in controls)
    minimum_primary = {
        "Persistence", "DampedPersistence", "Climatology",
        "LightGBM", "LSTM", "ThermoRoute",
    }
    if set(primary_values) != minimum_primary or len(primary_values) != len(minimum_primary):
        raise OpeningContractError(
            "protocol primary_models must be exactly the six declared comparison models"
        )
    if len(control_values) != len(set(control_values)) or set(primary_values) & set(control_values):
        raise OpeningContractError("protocol architecture-control registry is duplicated")
    if set(control_values) != set(CONTROL_INTERVENTIONS):
        raise OpeningContractError(
            "protocol architecture controls differ from the implemented one-factor registry"
        )
    if cohort == "temporal":
        values = (*primary_values, *control_values)
    elif cohort == "external":
        external = protocol.get("external_new_gage_inference_contract")
        if not isinstance(external, Mapping):
            raise OpeningContractError("protocol lacks external new-gage contract")
        required_external = {
            "role": "EXTERNAL_EXPLORATORY_NOT_IN_PRIMARY_FIVE_TEST_FAMILY",
            "observed_wtemp_history_through_issue_date": True,
            "ungauged_claim_allowed": False,
            "station_identity_embedding_allowed": False,
        }
        wrong = {key: external.get(key) for key, value in required_external.items()
                 if external.get(key) != value}
        if wrong:
            raise OpeningContractError(f"external new-gage contract is unsafe: {wrong}")
        # Architecture controls are mandatory for the temporal mechanism audit.
        # The external exploratory suite is exactly the six primary model types,
        # all rebuilt with pooled/station-agnostic preprocessing where applicable.
        values = primary_values
    else:
        raise ValueError("cohort must be temporal or external")
    if len(values) != len(set(values)):
        raise OpeningContractError(f"{cohort} protocol model registry is duplicated")
    return values


def _formal_test_registry(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    inference = protocol.get("primary_inference_contract")
    if not isinstance(inference, Mapping):
        raise OpeningContractError("protocol lacks primary inference contract")
    family = inference.get("confirmatory_family")
    if not isinstance(family, list) or len(family) != 5:
        raise OpeningContractError("protocol must contain exactly five structured tests")
    exact_keys = {
        "test_id", "candidate", "reference", "horizon", "margin_c",
        "alternative", "bootstrap_seed", "sign_flip_seed", "description",
    }
    primary_models = set(str(value) for value in inference.get("primary_models", ()))
    output: list[dict[str, Any]] = []
    for item in family:
        if not isinstance(item, Mapping) or set(item) != exact_keys:
            raise OpeningContractError("confirmatory test object schema changed")
        try:
            horizon = int(item["horizon"])
            margin = float(item["margin_c"])
            bootstrap_seed = int(item["bootstrap_seed"])
            sign_seed = int(item["sign_flip_seed"])
        except (TypeError, ValueError) as exc:
            raise OpeningContractError("confirmatory test value is not numeric") from exc
        candidate = str(item["candidate"])
        reference = str(item["reference"])
        if (
            not str(item["test_id"]).strip()
            or not str(item["description"]).strip()
            or candidate != "ThermoRoute"
            or reference == candidate
            or {candidate, reference} - primary_models
            or horizon not in {1, 3, 7}
            or not np.isfinite(margin)
            or item["alternative"] != "candidate_minus_reference_below_margin"
            or bootstrap_seed < 0
            or sign_seed < 0
        ):
            raise OpeningContractError("confirmatory test object violates its frozen contract")
        output.append({
            "test_id": str(item["test_id"]),
            "candidate": candidate,
            "reference": reference,
            "horizon": horizon,
            "margin_c": margin,
            "bootstrap_seed": bootstrap_seed,
            "sign_flip_seed": sign_seed,
            "description": str(item["description"]),
        })
    ids = [item["test_id"] for item in output]
    family_keys = [
        (item["candidate"], item["reference"], item["horizon"]) for item in output
    ]
    if len(ids) != len(set(ids)) or len(family_keys) != len(set(family_keys)):
        raise OpeningContractError("confirmatory test family is duplicated")
    if inference.get("multiplicity") != (
        "Holm step-down adjustment across exactly the five confirmatory one-sided tests"
    ):
        raise OpeningContractError("confirmatory multiplicity family changed")
    return tuple(output)


def _git_blob_bytes(root: Path, commit: str, relative: str) -> bytes:
    _assert_safe_live_git_repository(root)
    result = _run_live_git(root, "show", f"{commit}:{relative}", text=False)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise OpeningContractError(
            f"cannot replay sealed protocol blob {commit}:{relative}: {detail}"
        )
    return result.stdout


def validate_protocol_seal(
    seal_path: str | Path,
    *,
    protocol_path: str | Path,
    root: str | Path,
    authoritative_commit: str,
    allow_gitless_archive: bool = False,
    frozen_seal_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify final protocol bytes against a prior Git commit and immutable seal."""
    root = Path(root).resolve()
    protocol_path = Path(protocol_path).resolve()
    seal_path = Path(seal_path).resolve()
    if root not in seal_path.parents or not seal_path.is_file():
        raise OpeningContractError("protocol seal escapes or is absent")
    seal_sha256 = sha256_file(seal_path)
    if frozen_seal_sha256 is not None and seal_sha256 != frozen_seal_sha256:
        raise OpeningContractError("protocol seal differs from authorization")
    seal = _load_json(seal_path, label="protocol seal")
    if (
        seal.get("format") != PROTOCOL_SEAL_FORMAT
        or seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
        or seal.get("protocol_id") != "route-a-confirmatory-v1"
    ):
        raise OpeningContractError("unsupported or non-prelabel protocol seal")
    original = seal.get("original_preregistration")
    final = seal.get("final_prelabel_protocol")
    history = seal.get("history_contract")
    attestation = seal.get("prelabel_attestation")
    if not all(isinstance(value, Mapping) for value in (
        original, final, history, attestation
    )):
        raise OpeningContractError("protocol seal sections are malformed")
    assert isinstance(original, Mapping)
    assert isinstance(final, Mapping)
    assert isinstance(history, Mapping)
    assert isinstance(attestation, Mapping)
    if original.get("commit") != authoritative_commit:
        raise OpeningContractError("protocol seal names another original commit")
    final_commit = str(final.get("commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", final_commit):
        raise OpeningContractError("protocol seal final commit is malformed")
    required_history = {
        "original_commit_must_be_ancestor_of_final_commit": True,
        "final_commit_must_be_ancestor_of_authorization_commit": True,
        "git_show_bytes_must_match_every_declared_hash": True,
        "current_protocol_bytes_must_match_final_commit": True,
    }
    if dict(history) != required_history:
        raise OpeningContractError("protocol seal history contract changed")
    required_attestation = {
        "post_2020_wtemp_requested_or_inspected": False,
        "post_2020_flow_or_wlevel_requested_or_inspected": False,
        "confirmation_outcome_artifact_present": False,
        "external_timestamp_or_public_preregistration": False,
        "independent_custodian_or_worm_storage": False,
    }
    if any(attestation.get(key) != value for key, value in required_attestation.items()):
        raise OpeningContractError("protocol seal prelabel attestation changed")
    if not isinstance(attestation.get("scope"), str) or not attestation["scope"].strip():
        raise OpeningContractError("protocol seal security scope is absent")

    original_markdown = original.get("markdown")
    final_json = final.get("json")
    final_markdown = final.get("markdown")
    if not all(isinstance(value, Mapping) for value in (
        original_markdown, final_json, final_markdown
    )):
        raise OpeningContractError("protocol seal file bindings are malformed")
    assert isinstance(original_markdown, Mapping)
    assert isinstance(final_json, Mapping)
    assert isinstance(final_markdown, Mapping)
    expected_json_relative = _relative(root, protocol_path)
    if final_json.get("path") != expected_json_relative:
        raise OpeningContractError("protocol seal binds another JSON protocol")
    markdown_relative = str(final_markdown.get("path", ""))
    markdown_path = _resolve_inside(root, markdown_relative)
    for label, binding, path in (
        ("final JSON", final_json, protocol_path),
        ("final Markdown", final_markdown, markdown_path),
    ):
        digest = binding.get("sha256")
        if not _is_sha256(digest) or sha256_file(path) != digest:
            raise OpeningContractError(f"protocol seal {label} checksum changed")
    if (
        original_markdown.get("path") != markdown_relative
        or not _is_sha256(original_markdown.get("sha256"))
    ):
        raise OpeningContractError("original protocol Markdown binding changed")

    git_directory = root / ".git"
    if git_directory.exists():
        for commit in (authoritative_commit, final_commit):
            _git_output(root, "cat-file", "-e", f"{commit}^{{commit}}")
        for ancestor, descendant, label in (
            (authoritative_commit, final_commit, "original-to-final"),
            (final_commit, "HEAD", "final-to-HEAD"),
        ):
            relation = _run_live_git(
                root,
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
                text=False,
            )
            if relation.returncode:
                raise OpeningContractError(
                    f"protocol seal Git ancestry failed: {label}"
                )
        original_bytes = _git_blob_bytes(
            root, authoritative_commit, str(original_markdown["path"])
        )
        if hashlib.sha256(original_bytes).hexdigest() != original_markdown["sha256"]:
            raise OpeningContractError("original preregistration blob differs from seal")
        for label, binding, current in (
            ("final JSON", final_json, protocol_path),
            ("final Markdown", final_markdown, markdown_path),
        ):
            committed = _git_blob_bytes(root, final_commit, str(binding["path"]))
            if (
                hashlib.sha256(committed).hexdigest() != binding["sha256"]
                or committed != current.read_bytes()
            ):
                raise OpeningContractError(
                    f"{label} bytes differ from the sealed Git commit"
                )
    elif not allow_gitless_archive or not _is_sha256(frozen_seal_sha256):
        raise OpeningContractError(
            "protocol-seal Git history is unavailable outside release mode"
        )
    return {
        "document": seal,
        "path": seal_path,
        "sha256": seal_sha256,
        "final_commit": final_commit,
        "final_markdown_path": markdown_path,
        "final_markdown_sha256": str(final_markdown["sha256"]),
        "external_timestamp_or_public_preregistration": False,
    }


def validate_protocol(
    protocol_path: str | Path,
    *,
    root: str | Path,
    allow_gitless_archive: bool = False,
    frozen_authoritative_markdown_sha256: str | None = None,
    protocol_seal_path: str | Path | None = None,
    frozen_protocol_seal_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the original preregistration and every outcome-independent amendment."""
    root = Path(root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_json(protocol_path, label="confirmatory protocol")
    if protocol.get("schema_version") != 1:
        raise OpeningContractError("unsupported confirmatory protocol schema")
    if protocol.get("status") not in {"FROZEN_NOT_ACQUIRED", "REGISTRY_FROZEN_LABELS_SEALED"}:
        raise OpeningContractError("protocol is not in a pre-opening frozen state")
    if protocol.get("availability_contract", {}).get("maximum_openings") != 1:
        raise OpeningContractError("protocol must permit exactly one opening")
    commit = str(protocol.get("authoritative_protocol_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise OpeningContractError("authoritative protocol commit is not a full SHA-1")
    git_directory = root / ".git"
    if git_directory.exists():
        _git_output(root, "cat-file", "-e", f"{commit}^{{commit}}")
        ancestor = _run_live_git(
            root,
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
            text=False,
        )
        if ancestor.returncode != 0:
            raise OpeningContractError(
                "authoritative protocol commit is not an ancestor of HEAD"
            )
        original_markdown = _git_output(
            root, "show", f"{commit}:protocols/route_a_confirmatory_protocol.md"
        ).encode("utf-8")
        if not original_markdown.strip():
            raise OpeningContractError("authoritative protocol commit has no protocol text")
        authoritative_markdown_sha256 = hashlib.sha256(original_markdown).hexdigest()
        if (
            frozen_authoritative_markdown_sha256 is not None
            and authoritative_markdown_sha256 != frozen_authoritative_markdown_sha256
        ):
            raise OpeningContractError(
                "Git authoritative protocol differs from authorization"
            )
    elif allow_gitless_archive and _is_sha256(frozen_authoritative_markdown_sha256):
        authoritative_markdown_sha256 = str(frozen_authoritative_markdown_sha256)
    else:
        raise OpeningContractError(
            "authoritative protocol Git history is unavailable outside release mode"
        )
    amendments = protocol.get("pre_label_amendments", [])
    if not isinstance(amendments, list):
        raise OpeningContractError("pre_label_amendments must be a list")
    amendment_ids = []
    for amendment in amendments:
        if not isinstance(amendment, Mapping):
            raise OpeningContractError("pre-label amendment is not an object")
        amendment_ids.append(str(amendment.get("amendment_id", "")))
        if amendment.get("outcome_independent") is not True:
            raise OpeningContractError("every pre-label amendment must be outcome-independent")
        if amendment.get("post_2020_wtemp_requested_or_inspected") is not False:
            raise OpeningContractError("an amendment records post-2020 WTEMP access")
    if not all(amendment_ids) or len(amendment_ids) != len(set(amendment_ids)):
        raise OpeningContractError("pre-label amendment identifiers are empty or duplicated")
    seal_info = validate_protocol_seal(
        root / DEFAULT_PROTOCOL_SEAL if protocol_seal_path is None else protocol_seal_path,
        protocol_path=protocol_path,
        root=root,
        authoritative_commit=commit,
        allow_gitless_archive=allow_gitless_archive,
        frozen_seal_sha256=frozen_protocol_seal_sha256,
    )
    holdout = protocol.get("time_holdout", {})
    start = str(holdout.get("primary_target_start", holdout.get("start", "")))
    end = str(holdout.get("end", ""))
    try:
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise OpeningContractError("confirmation interval is reversed")
    except (TypeError, ValueError) as exc:
        raise OpeningContractError("confirmation interval is invalid") from exc
    _required_models(protocol, cohort="temporal")
    _required_models(protocol, cohort="external")
    _formal_test_registry(protocol)
    exact_test = protocol.get("primary_inference_contract", {}).get(
        "one_sided_p_value"
    )
    if not isinstance(exact_test, Mapping) or exact_test.get("method") != (
        "exact whole-HUC2 cluster sign-flip enumeration"
    ):
        raise OpeningContractError("primary p-value is not exact whole-HUC2 sign flipping")
    maximum_configurations = int(
        exact_test.get("maximum_configurations_for_frozen_cohort", -1)
    )
    if (
        maximum_configurations <= 0
        or maximum_configurations & (maximum_configurations - 1)
        or exact_test.get("monte_carlo_correction")
        != "not applicable to exact enumeration"
        or not str(exact_test.get("enumeration_rule", "")).strip()
        or not str(exact_test.get("legacy_seed_field", "")).strip()
    ):
        raise OpeningContractError("exact sign-flip protocol fields changed")
    quality = protocol.get("daily_outcome_quality_contract")
    if not isinstance(quality, Mapping):
        raise OpeningContractError("protocol lacks daily outcome-quality contract")
    audit = quality.get("mandatory_quality_audit")
    sensitivity = quality.get("approved_only_target_sensitivity")
    if (
        quality.get("request_statistic_code") != "00003_daily_mean"
        or not isinstance(audit, Mapping)
        or audit.get("grouping") != [
            "cohort", "site_no", "variable", "raw_qualifier", "series_id",
            "value_status",
        ]
        or audit.get("variables") != ["WTEMP", "FLOW", "WLEVEL"]
        or not isinstance(sensitivity, Mapping)
        or sensitivity.get("role")
        != "EXPLORATORY_DESCRIPTIVE_NOT_IN_CONFIRMATORY_FAMILY"
        or int(sensitivity.get("minimum_valid_targets_per_station_horizon", -1))
        != int(protocol.get("availability_contract", {}).get(
            "minimum_valid_targets_per_station_horizon", -2
        ))
        or sensitivity.get("selection_prohibited") is not True
    ):
        raise OpeningContractError("daily outcome-quality protocol fields changed")
    spatial = protocol.get("primary_inference_contract", {}).get(
        "exploratory_spatial_inference_sensitivity"
    )
    if (
        not isinstance(spatial, Mapping)
        or spatial.get("role") != "DESCRIPTIVE_NOT_IN_CONFIRMATORY_FAMILY"
        or spatial.get("output_artifact") != "trusted/spatial_sensitivity_v1.json"
        or spatial.get("comparison_output_schema") != [
            "test_id", "candidate", "reference", "horizon", "margin_c",
            "station_weighted_median_effect_c", "n_stations", "n_clusters",
            "equal_huc_median_effect_c", "per_huc", "leave_one_huc",
            "influence_min_c", "influence_max_c",
        ]
        or spatial.get("per_huc_schema") != [
            "huc2", "n_stations", "median_station_effect_c"
        ]
        or spatial.get("leave_one_huc_schema") != [
            "held_out_huc2", "n_remaining_stations", "n_remaining_clusters",
            "station_weighted_median_effect_c", "effect_minus_margin_c",
        ]
        or spatial.get("prohibited_outputs") != [
            "p_value", "confidence_interval", "Holm_adjustment",
            "pass_fail_decision",
        ]
        or not str(spatial.get("limitations", "")).strip()
    ):
        raise OpeningContractError("spatial sensitivity protocol fields changed")
    estimand = protocol.get("primary_inference_contract", {}).get(
        "primary_estimand"
    )
    estimand_keys = {
        "population", "admissible_issue", "admissible_target",
        "model_comparison_keys", "station_horizon_reportability",
        "within_station_metric", "between_station_effect", "interpretation_limit",
    }
    if (
        not isinstance(estimand, Mapping)
        or set(estimand) != estimand_keys
        or "32-day history" not in str(estimand["admissible_issue"])
        or "independently" not in str(estimand["admissible_target"])
        or "unweighted RMSE" not in str(estimand["within_station_metric"])
        or "median across reportable stations" not in str(
            estimand["between_station_effect"]
        )
    ):
        raise OpeningContractError("primary estimand protocol fields changed")
    return {
        "document": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "authoritative_commit": commit,
        "authoritative_markdown_sha256": authoritative_markdown_sha256,
        "amendments_sha256": sha256_json(amendments),
        "seal": seal_info,
        "target_start": start,
        "target_end": end,
    }


def _load_registry(
    path: Path, *, expected_count: int, label: str, role: str
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    if role == "development":
        required = {"site_no", "lat", "lon", "huc2", "huc_metadata_status"}
        if not required <= set(header):
            raise OpeningContractError(f"{label} lacks stable site/coordinate/HUC fields")
        # Legacy coverage fields are intentionally left unread and uninterpreted;
        # FrozenPanelSpec closes the complete development file identity later.
        usecols = [
            column for column in (
                "site_no", "lat", "lon", "state", "huc_cd", "huc2",
                "drain_area_va", "huc_metadata_status",
            ) if column in header.columns
        ]
    elif role == "external":
        expected_columns = (*CANDIDATE_COLUMNS, "selection_rank_sha256")
        if tuple(header.columns) != expected_columns:
            raise OpeningContractError("external registry schema differs from seeded selection")
        usecols = list(expected_columns)
    else:
        raise ValueError("registry role must be development or external")
    string_columns = {
        column: "string"
        for column in (
            "site_no", "legacy_site_id", "state", "huc_cd", "huc2",
            "huc_metadata_status",
        )
        if column in usecols
    }
    frame = pd.read_csv(
        path, usecols=usecols, dtype=string_columns, keep_default_na=False
    )
    frame = frame.copy()
    frame["site_no"] = frame["site_no"].astype("string").str.strip()
    if frame.site_no.eq("").any() or frame.site_no.duplicated().any():
        raise OpeningContractError(f"{label} site_no is empty or duplicated")
    if role == "development":
        frame["site_no"] = frame.site_no.str.zfill(8)
        raw_huc = frame.huc_cd.str.strip()
        normalized_huc = raw_huc.map(
            lambda value: (
                value.zfill(8 if len(value) <= 8 else 12)[:8] if value else ""
            )
        )
        frame["huc_cd"] = normalized_huc.astype("string")
        frame["huc2"] = frame.huc2.str.strip().str.zfill(2)
        if (
            frame.huc_cd.str.fullmatch(r"[0-9]{8}").ne(True).any()
            or frame.huc2.str.fullmatch(r"[0-9]{2}").ne(True).any()
            or frame.huc_cd.str[:2].ne(frame.huc2).any()
            or frame.huc_metadata_status.ne("USGS_SNAPSHOT_SITE_NO_MATCH").any()
        ):
            raise OpeningContractError(
                "development registry HUC identifiers/status are not verified"
            )
    if len(frame) != expected_count:
        raise OpeningContractError(
            f"{label} has {len(frame)} sites; protocol requires {expected_count}"
        )
    if not np.isfinite(
        frame[["lat", "lon"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ).all():
        raise OpeningContractError(f"{label} contains invalid coordinates")
    return frame


def validate_registry_lock(
    *,
    root: str | Path,
    protocol_info: Mapping[str, Any],
    development_registry: str | Path,
    external_registry: str | Path,
    external_lock: str | Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    protocol = protocol_info["document"]
    development_path = Path(development_registry).resolve()
    external_path = Path(external_registry).resolve()
    lock_path = Path(external_lock).resolve()
    development = _load_registry(
        development_path, expected_count=120, label="development registry",
        role="development",
    )
    planned_count = int(protocol["new_site_external_validation"]["planned_site_count"])
    external = _load_registry(
        external_path, expected_count=planned_count, label="external registry",
        role="external",
    )
    overlap = set(development.site_no) & set(external.site_no)
    if overlap:
        raise OpeningContractError(
            f"external registry overlaps development sites: {sorted(overlap)[:5]}"
        )
    lock = _load_json(lock_path, label="external registry lock")
    expected = {
        "status": "REGISTRY_FROZEN_LABELS_SEALED",
        "labels_state": "SEALED_NOT_ACQUIRED",
        "opening_count": 0,
        "site_count": planned_count,
        "site_primary_key": "site_no",
        "protocol_id": protocol["protocol_id"],
        "selection_seed": protocol["new_site_external_validation"]["selection_seed"],
        "confirmatory_registry_sha256": sha256_file(external_path),
        "protocol_sha256": protocol_info["protocol_sha256"],
        "authoritative_protocol_commit": protocol_info["authoritative_commit"],
        "pre_label_amendments_sha256": protocol_info["amendments_sha256"],
    }
    wrong = {key: (lock.get(key), value) for key, value in expected.items()
             if lock.get(key) != value}
    if wrong:
        raise OpeningContractError(f"external registry lock mismatch: {wrong}")
    frozen_artifacts = lock.get("frozen_artifacts")
    if not isinstance(frozen_artifacts, Mapping) or set(frozen_artifacts) != {
        "development_panel_spec", "candidate_table", "candidate_provenance",
        "candidate_snapshot_index",
    }:
        raise OpeningContractError("external registry lock lacks candidate evidence bindings")
    development_spec_path = _verify_file_binding(
        root, frozen_artifacts["development_panel_spec"],
        label="external selection development-panel spec",
    )
    candidate_path = _verify_file_binding(
        root, frozen_artifacts["candidate_table"], label="external candidate table"
    )
    provenance_path = _verify_file_binding(
        root, frozen_artifacts["candidate_provenance"],
        label="external candidate provenance",
    )
    candidate_index_path = _verify_file_binding(
        root, frozen_artifacts["candidate_snapshot_index"],
        label="external candidate snapshot index",
    )
    expected_artifact_hashes = {
        "development_panel_spec_sha256": sha256_file(development_spec_path),
        "candidate_table_sha256": sha256_file(candidate_path),
        "candidate_provenance_sha256": sha256_file(provenance_path),
        "candidate_snapshot_index_sha256": sha256_file(candidate_index_path),
    }
    if any(lock.get(key) != value for key, value in expected_artifact_hashes.items()):
        raise OpeningContractError("external registry lock artifact hashes changed")
    try:
        development_spec = FrozenPanelSpec.load(development_spec_path)
        development_evidence = development_spec.verify()
    except Exception as exc:
        raise OpeningContractError("external selection development spec is invalid") from exc
    if (
        development_spec.registry_path.resolve() != development_path
        or development_evidence.get("registry_sha256") != sha256_file(development_path)
    ):
        raise OpeningContractError(
            "external selection development registry differs from FrozenPanelSpec"
        )
    try:
        candidates = replay_candidate_evidence(
            candidate_path,
            provenance_path,
            candidate_index_path,
            protocol_sha256=protocol_info["protocol_sha256"],
            state_universe=protocol["metadata_candidate_contract"]["state_universe"],
        )
        selected = select_confirmatory_sites(
            candidates,
            set(development.site_no.astype(str)),
            n_sites=planned_count,
            selection_seed=protocol["new_site_external_validation"]["selection_seed"],
        )
    except (EvidenceError, KeyError, ValueError) as exc:
        raise OpeningContractError("cannot independently replay external site selection") from exc
    external_comparable = external.copy()
    for column in ("lat", "lon", "drain_area_va"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        external_comparable[column] = pd.to_numeric(
            external_comparable[column], errors="coerce"
        )
    try:
        pd.testing.assert_frame_equal(
            selected.reset_index(drop=True),
            external_comparable.reset_index(drop=True),
            check_dtype=False,
            rtol=0.0,
            atol=0.0,
        )
    except AssertionError as exc:
        raise OpeningContractError(
            "external registry differs from deterministic seeded candidate selection"
        ) from exc
    return {
        "development": development,
        "external": external,
        "development_sha256": sha256_file(development_path),
        "external_sha256": sha256_file(external_path),
        "lock_sha256": sha256_file(lock_path),
        "candidate_table": candidate_path,
        "candidate_provenance": provenance_path,
        "candidate_snapshot_index": candidate_index_path,
        "development_panel_spec": development_spec_path,
    }


def _verify_snapshot_index(
    root: Path,
    path: Path,
    *,
    prelabel: bool,
) -> list[Mapping[str, Any]]:
    document = _load_json(path, label="snapshot index")
    if document.get("schema_version") != 1:
        raise OpeningContractError("unsupported snapshot-index schema")
    records = document.get("records")
    if not isinstance(records, list) or int(document.get("snapshot_count", -1)) != len(records):
        raise OpeningContractError("snapshot index count is inconsistent")
    snapshot_root = path.parent.resolve()
    request_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise OpeningContractError("snapshot index record is not an object")
        required = {
            "provider", "request_sha256", "response_sha256", "retrieved_at_utc",
            "byte_count", "request", "metadata_path", "response_path",
        }
        if required - set(record):
            raise OpeningContractError("snapshot index record is incomplete")
        response = (snapshot_root / str(record.get("response_path", ""))).resolve()
        if snapshot_root not in response.parents or not response.is_file():
            raise OpeningContractError("snapshot response path escapes or is missing")
        payload = response.read_bytes()
        response_sha = hashlib.sha256(payload).hexdigest()
        if response_sha != record.get("response_sha256"):
            raise OpeningContractError("snapshot response checksum mismatch")
        if int(record.get("byte_count", -1)) != len(payload):
            raise OpeningContractError("snapshot response byte count mismatch")
        metadata_path = (
            snapshot_root / str(record.get("metadata_path", ""))
        ).resolve()
        if snapshot_root not in metadata_path.parents or not metadata_path.is_file():
            raise OpeningContractError("snapshot metadata path escapes or is missing")
        request = record.get("request")
        if not isinstance(request, Mapping):
            raise OpeningContractError("snapshot record lacks canonical request")
        request_sha = hashlib.sha256(canonical_json_bytes(dict(request))).hexdigest()
        if request_sha != record.get("request_sha256"):
            raise OpeningContractError("snapshot request fingerprint mismatch")
        if (
            metadata_path.parent != response.parent
            or metadata_path.name != "metadata.json"
            or response.name != "response.bin"
            or metadata_path.parent.name != request_sha
        ):
            raise OpeningContractError("snapshot files do not use canonical request layout")
        if request_sha in request_ids:
            raise OpeningContractError("snapshot index duplicates a canonical request")
        request_ids.add(request_sha)
        metadata = _load_json(metadata_path, label="snapshot metadata")
        expected_metadata = {
            "schema_version": 1,
            "request": dict(request),
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "byte_count": len(payload),
            "http_status": 200,
            "response_file": response.name,
        }
        wrong_metadata = {
            key: metadata.get(key)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if wrong_metadata:
            raise OpeningContractError(
                f"snapshot metadata does not bind request/response: {wrong_metadata}"
            )
        if metadata.get("retrieved_at_utc") != record.get("retrieved_at_utc"):
            raise OpeningContractError("snapshot retrieval timestamp mismatch")
        provider = str(record.get("provider", ""))
        if not provider or request.get("provider") != provider:
            raise OpeningContractError("snapshot provider identity mismatch")
        if not prelabel and provider == CONFIRMATORY_NWIS_PROVIDER:
            expected_record_fields = required | {
                "attempt_number", "metadata_sha256", "series_registry",
            }
            expected_metadata_fields = {
                "schema_version", "opening_id", "authorization_sha256",
                "work_order_self_sha256", "request_ledger_sha256",
                "attempt_number", "request", "request_sha256",
                "retrieved_at_utc", "http_status", "response_headers",
                "final_url", "byte_count", "response_sha256", "response_file",
                "maximum_response_bytes_per_request",
            }
            if (
                set(record) != expected_record_fields
                or set(metadata) != expected_metadata_fields
                or metadata.get("maximum_response_bytes_per_request")
                != MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
                or record.get("metadata_sha256") != sha256_file(metadata_path)
                or record.get("attempt_number") != metadata.get("attempt_number")
                or metadata.get("final_url") != request.get("url")
                or not isinstance(metadata.get("response_headers"), Mapping)
            ):
                raise OpeningContractError(
                    "opened NWIS transaction/metadata schema or attempt binding changed"
                )
            try:
                expected_series = nwis_confirmatory_series_registry(payload)
            except (UnicodeError, ValueError) as exc:
                raise OpeningContractError(
                    "cannot replay opened NWIS series registry"
                ) from exc
            if record.get("series_registry") != expected_series:
                raise OpeningContractError(
                    "opened NWIS series/value/qualifier column registry changed"
                )
        if request.get("schema_version") != 1 or request.get("method") != "GET":
            raise OpeningContractError("snapshot request method/schema is unsupported")
        if not isinstance(request.get("headers"), Mapping):
            raise OpeningContractError("snapshot request headers are not canonical")
        parsed = urlsplit(str(request.get("url", "")))
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise OpeningContractError("snapshot request is not an HTTPS provider URL")
        url = str(request.get("url", "")).lower()
        if prelabel and (
            "/nwis/dv/" in url
            or "parametercd=00010" in url
            or "parametercd=00060" in url
            or "parametercd=00065" in url
        ):
            raise OpeningContractError(
                "pre-label input evidence contains an outcome/history endpoint"
            )
    return records


def _verify_opened_nwis_index(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_sites: set[str],
    history_start: str,
    target_end: str,
) -> None:
    """Require one canonical, full-interval NWIS response for every frozen site."""
    seen: list[str] = []
    for record in records:
        if record.get("provider") != CONFIRMATORY_NWIS_PROVIDER:
            raise OpeningContractError("opened snapshot index contains a non-NWIS provider")
        request = record["request"]
        parsed = urlsplit(str(request["url"]))
        query = parse_qs(parsed.query, keep_blank_values=True)
        if (
            parsed.scheme.lower() != "https"
            or parsed.hostname != "waterservices.usgs.gov"
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/nwis/dv/"
            or parsed.fragment
        ):
            raise OpeningContractError("opened snapshot is not an NWIS daily-values request")
        expected_query = {
            "format": ["rdb"],
            "startDT": [history_start],
            "endDT": [target_end],
            "parameterCd": ["00010,00060,00065"],
            "statCd": ["00003"],
            "siteStatus": ["all"],
        }
        if set(query) != {*expected_query, "sites"}:
            raise OpeningContractError("opened NWIS request has undeclared query fields")
        if request.get("headers") != {}:
            raise OpeningContractError("opened NWIS request has undeclared HTTP headers")
        for key, value in expected_query.items():
            if query.get(key) != value:
                raise OpeningContractError(f"opened NWIS request changed {key}")
        sites = query.get("sites", [])
        if len(sites) != 1 or "," in sites[0]:
            raise OpeningContractError("opened NWIS requests must be one stable site each")
        if str(request["url"]) != build_nwis_confirmatory_url(
            sites[0], history_start, target_end
        ):
            raise OpeningContractError("opened NWIS URL is not the canonical byte string")
        seen.append(str(sites[0]))
    if len(seen) != len(set(seen)) or set(seen) != expected_sites:
        raise OpeningContractError(
            "opened NWIS request registry differs from the frozen 120+30 sites"
        )


def _verify_opened_request_map(
    path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    opening_id: str,
    authorization_sha256: str,
    temporal_sites: set[str],
    external_sites: set[str],
) -> None:
    document = _load_json(path, label="opened NWIS request map")
    if set(document) != {
        "format", "opening_id", "authorization_sha256", "provider",
        "request_count", "requests",
    }:
        raise OpeningContractError("opened NWIS request-map schema changed")
    expected_top = {
        "format": ACQUISITION_REQUEST_MAP_FORMAT,
        "opening_id": opening_id,
        "authorization_sha256": authorization_sha256,
        "provider": CONFIRMATORY_NWIS_PROVIDER,
        "request_count": len(records),
    }
    if any(document.get(key) != value for key, value in expected_top.items()):
        raise OpeningContractError("opened NWIS request-map identity changed")
    requests = document.get("requests")
    if not isinstance(requests, list) or len(requests) != len(records):
        raise OpeningContractError("opened NWIS request-map count changed")
    indexed = {str(record["request_sha256"]): record for record in records}
    seen: set[tuple[str, str]] = set()
    for item in requests:
        if not isinstance(item, Mapping) or set(item) != {
            "cohort", "site_no", "request_sha256", "response_sha256",
            "retrieved_at_utc", "byte_count", "attempt_number", "series_registry",
        }:
            raise OpeningContractError("opened NWIS request-map row schema changed")
        cohort, site = str(item.get("cohort", "")), str(item.get("site_no", ""))
        expected_cohort = (
            "temporal" if site in temporal_sites
            else "external" if site in external_sites else None
        )
        if cohort != expected_cohort or (cohort, site) in seen:
            raise OpeningContractError("opened NWIS request-map site/cohort changed")
        request_sha = str(item.get("request_sha256", ""))
        if request_sha not in indexed:
            raise OpeningContractError("opened request map lacks a raw response")
        record = indexed[request_sha]
        query = parse_qs(urlsplit(str(record["request"]["url"])).query)
        if query.get("sites") != [site]:
            raise OpeningContractError("opened request-map site differs from raw request")
        for field in (
            "request_sha256", "response_sha256", "retrieved_at_utc", "byte_count",
            "attempt_number", "series_registry",
        ):
            if item.get(field) != record.get(field):
                raise OpeningContractError("opened request map does not bind raw evidence")
        seen.add((cohort, site))
    expected_seen = {
        *{("temporal", site) for site in temporal_sites},
        *{("external", site) for site in external_sites},
    }
    if seen != expected_seen:
        raise OpeningContractError("opened request map omits a frozen station")


def _verify_opened_transport_evidence(
    *,
    root: Path,
    acquisition: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    opening_id: str,
    authorization_sha256: str,
    work_order_path: Path,
    raw_root: Path,
) -> dict[str, Any]:
    """Independently close the immutable ledger and every transport attempt."""
    ledger_path = _verify_file_binding(
        root,
        acquisition.get("request_ledger", {}),
        label="opened acquisition request ledger",
    )
    expected_acquisition_root = raw_root.parent.resolve()
    if ledger_path != expected_acquisition_root / "request_ledger_v1.json":
        raise OpeningContractError("opened request ledger path is noncanonical")
    ledger = _load_json(ledger_path, label="opened acquisition request ledger")
    ledger_stable = dict(ledger)
    ledger_self = ledger_stable.pop("request_ledger_self_sha256", None)
    ledger_fields = {
        "format", "status", "opening_id", "authorization_sha256",
        "work_order_self_sha256", "work_order_file_sha256", "provider",
        "maximum_response_bytes_per_request",
        "request_order", "request_count", "requests",
        "station_or_request_replacement_allowed",
        "request_ledger_self_sha256",
    }
    if (
        set(ledger) != ledger_fields
        or ledger.get("format") != ACQUISITION_REQUEST_LEDGER_FORMAT
        or ledger.get("status") != "FROZEN_BEFORE_FIRST_HTTPS_REQUEST"
        or ledger.get("opening_id") != opening_id
        or ledger.get("authorization_sha256") != authorization_sha256
        or ledger.get("work_order_file_sha256") != sha256_file(work_order_path)
        or ledger.get("provider") != CONFIRMATORY_NWIS_PROVIDER
        or ledger.get("maximum_response_bytes_per_request")
        != MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        or ledger.get("request_order")
        != "temporal_then_external_each_site_no_ascending"
        or ledger.get("station_or_request_replacement_allowed") is not False
        or ledger_self
        != hashlib.sha256(canonical_json_bytes(ledger_stable)).hexdigest()
        or ledger_path.read_bytes() != canonical_json_bytes(ledger)
    ):
        raise OpeningContractError("opened acquisition request ledger changed")
    work_order = _load_json(work_order_path, label="acquisition work order")
    if ledger.get("work_order_self_sha256") != work_order.get(
        "work_order_self_sha256"
    ):
        raise OpeningContractError("request ledger differs from its work order")
    rows_by_request = {
        str(row["request_sha256"]): row for row in request_rows
    }
    records_by_request = {
        str(record["request_sha256"]): record for record in records
    }
    expected_requests = []
    ordinal = 0
    for cohort in ("temporal", "external"):
        cohort_rows = sorted(
            (row for row in request_rows if row.get("cohort") == cohort),
            key=lambda row: str(row["site_no"]),
        )
        for row in cohort_rows:
            ordinal += 1
            request_sha = str(row["request_sha256"])
            record = records_by_request.get(request_sha)
            if record is None:
                raise OpeningContractError("request ledger lacks raw evidence")
            expected_requests.append({
                "ordinal": ordinal,
                "cohort": cohort,
                "site_no": str(row["site_no"]),
                "request": dict(record["request"]),
                "request_sha256": request_sha,
            })
    if (
        ledger.get("request_count") != len(records)
        or ledger.get("requests") != expected_requests
        or len(rows_by_request) != len(records)
    ):
        raise OpeningContractError("opened request ledger order/content changed")

    attempt_index_path = _verify_file_binding(
        root,
        acquisition.get("transport_attempt_index", {}),
        label="opened transport-attempt index",
    )
    if attempt_index_path != expected_acquisition_root / (
        "transport_attempt_index_v1.json"
    ):
        raise OpeningContractError("transport-attempt index path is noncanonical")
    attempt_index = _load_json(
        attempt_index_path, label="opened transport-attempt index"
    )
    attempt_stable = dict(attempt_index)
    attempt_self = attempt_stable.pop("attempt_index_self_sha256", None)
    index_fields = {
        "format", "status", "opening_id", "authorization_sha256",
        "work_order_self_sha256", "request_ledger", "request_count",
        "attempt_count", "resume_count", "opening_count",
        "response_replacement_count",
        "completed_before_final_attempt_request_sha256",
        "retrieval_span_utc", "attempts", "attempt_index_self_sha256",
    }
    expected_ledger_binding = _binding(root, ledger_path)
    if (
        set(attempt_index) != index_fields
        or attempt_index.get("format") != ACQUISITION_ATTEMPT_INDEX_FORMAT
        or attempt_index.get("status") != "ALL_LEDGER_TRANSACTIONS_COMPLETE"
        or attempt_index.get("opening_id") != opening_id
        or attempt_index.get("authorization_sha256") != authorization_sha256
        or attempt_index.get("work_order_self_sha256")
        != work_order.get("work_order_self_sha256")
        or attempt_index.get("request_ledger") != expected_ledger_binding
        or attempt_index.get("request_count") != len(records)
        or attempt_index.get("opening_count") != 1
        or attempt_index.get("response_replacement_count") != 0
        or attempt_self
        != hashlib.sha256(canonical_json_bytes(attempt_stable)).hexdigest()
        or attempt_index_path.read_bytes() != canonical_json_bytes(attempt_index)
    ):
        raise OpeningContractError("opened transport-attempt index changed")
    attempts = attempt_index.get("attempts")
    if (
        not isinstance(attempts, list)
        or not attempts
        or attempt_index.get("attempt_count") != len(attempts)
    ):
        raise OpeningContractError("opened transport-attempt count changed")
    request_ids = set(records_by_request)
    starts: dict[int, Mapping[str, Any]] = {}
    results: dict[int, Mapping[str, Any]] = {}
    resume_count = 0
    attempts_root = expected_acquisition_root / "transport_attempts_v1"
    for expected_number, row in enumerate(attempts, start=1):
        if not isinstance(row, Mapping) or set(row) != {
            "attempt_number", "mode", "status", "start", "result",
        } or row.get("attempt_number") != expected_number:
            raise OpeningContractError("opened transport-attempt row changed")
        start_path = _verify_file_binding(
            root, row["start"], label="transport-attempt start"
        )
        if start_path != attempts_root / (
            f"attempt_{expected_number:06d}_start.json"
        ):
            raise OpeningContractError("transport-attempt evidence path changed")
        start = _load_json(start_path, label="transport-attempt start")
        start_stable = dict(start)
        start_self = start_stable.pop("attempt_start_self_sha256", None)
        start_fields = {
            "format", "status", "opening_id", "authorization_sha256",
            "work_order_self_sha256", "request_ledger_sha256",
            "attempt_number", "mode", "opening_count",
            "completed_before_attempt_request_sha256",
            "missing_at_start_request_sha256", "response_replacement_allowed",
            "started_at_utc", "attempt_start_self_sha256",
        }
        common = {
            "opening_id": opening_id,
            "authorization_sha256": authorization_sha256,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "request_ledger_sha256": sha256_file(ledger_path),
            "attempt_number": expected_number,
            "opening_count": 1,
        }
        if (
            set(start) != start_fields
            or start.get("format") != ACQUISITION_ATTEMPT_START_FORMAT
            or any(start.get(key) != value for key, value in common.items())
            or start.get("status") != "TRANSPORT_ATTEMPT_STARTED"
            or start.get("response_replacement_allowed") is not False
            or start_self
            != hashlib.sha256(canonical_json_bytes(start_stable)).hexdigest()
            or start_path.read_bytes() != canonical_json_bytes(start)
            or row.get("mode") != start.get("mode")
        ):
            raise OpeningContractError("transport-attempt evidence changed")
        mode = str(start.get("mode"))
        if mode == "RESUME_SAME_OPENING":
            resume_count += 1
        elif mode != "INITIAL_OPENING_TRANSPORT" or expected_number != 1:
            raise OpeningContractError("transport-attempt mode changed")
        completed_before = start.get(
            "completed_before_attempt_request_sha256"
        )
        missing_before = start.get("missing_at_start_request_sha256")
        for completed, missing in ((completed_before, missing_before),):
            if (
                not isinstance(completed, list)
                or not isinstance(missing, list)
                or len(completed) != len(set(completed))
                or len(missing) != len(set(missing))
                or set(completed) & set(missing)
                or set(completed) | set(missing) != request_ids
            ):
                raise OpeningContractError(
                    "transport attempt does not partition the frozen ledger"
                )
        starts[expected_number] = start
        if row.get("result") is None:
            if (
                expected_number == len(attempts)
                or row.get("status") != "NO_RESULT_PROCESS_TERMINATED"
            ):
                raise OpeningContractError(
                    "only an earlier terminated transport attempt may lack a result"
                )
            continue
        result_path = _verify_file_binding(
            root, row["result"], label="transport-attempt result"
        )
        if result_path != attempts_root / (
            f"attempt_{expected_number:06d}_result.json"
        ):
            raise OpeningContractError("transport-attempt result path changed")
        result = _load_json(result_path, label="transport-attempt result")
        result_stable = dict(result)
        result_self = result_stable.pop("attempt_result_self_sha256", None)
        result_fields = {
            "format", "status", "opening_id", "authorization_sha256",
            "work_order_self_sha256", "request_ledger_sha256",
            "attempt_number", "attempt_start_sha256", "opening_count",
            "completed_request_sha256", "missing_request_sha256",
            "failure_class", "response_replacement_count", "completed_at_utc",
            "attempt_result_self_sha256",
        }
        completed_after = result.get("completed_request_sha256")
        missing_after = result.get("missing_request_sha256")
        if (
            set(result) != result_fields
            or result.get("format") != ACQUISITION_ATTEMPT_RESULT_FORMAT
            or any(result.get(key) != value for key, value in common.items())
            or result.get("response_replacement_count") != 0
            or result.get("attempt_start_sha256") != sha256_file(start_path)
            or result_self
            != hashlib.sha256(canonical_json_bytes(result_stable)).hexdigest()
            or result_path.read_bytes() != canonical_json_bytes(result)
            or row.get("status") != result.get("status")
            or not isinstance(completed_after, list)
            or not isinstance(missing_after, list)
            or len(completed_after) != len(set(completed_after))
            or len(missing_after) != len(set(missing_after))
            or set(completed_after) & set(missing_after)
            or set(completed_after) | set(missing_after) != request_ids
            or result.get("status") not in {
                "ALL_LEDGER_TRANSACTIONS_COMPLETE",
                "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING",
            }
            or (
                result.get("status") == "ALL_LEDGER_TRANSACTIONS_COMPLETE"
                and (missing_after or result.get("failure_class") is not None)
            )
            or (
                result.get("status")
                == "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING"
                and (not missing_after or not isinstance(
                    result.get("failure_class"), str
                ))
            )
        ):
            raise OpeningContractError("transport-attempt result evidence changed")
        results[expected_number] = result
    final_number = len(attempts)
    if (
        results[final_number].get("status")
        != "ALL_LEDGER_TRANSACTIONS_COMPLETE"
        or results[final_number].get("missing_request_sha256") != []
        or attempt_index.get("resume_count") != resume_count
        or attempt_index.get(
            "completed_before_final_attempt_request_sha256"
        )
        != starts[final_number].get(
            "completed_before_attempt_request_sha256"
        )
    ):
        raise OpeningContractError("final transport attempt is not complete")
    for record in records:
        metadata_path = raw_root / str(record["metadata_path"])
        metadata = _load_json(metadata_path, label="opened NWIS metadata")
        number = metadata.get("attempt_number")
        request_sha = str(record["request_sha256"])
        if (
            not isinstance(number, int)
            or number not in starts
            or request_sha not in starts[number]["missing_at_start_request_sha256"]
            or metadata.get("opening_id") != opening_id
            or metadata.get("authorization_sha256") != authorization_sha256
            or metadata.get("work_order_self_sha256")
            != work_order.get("work_order_self_sha256")
            or metadata.get("request_ledger_sha256") != sha256_file(ledger_path)
        ):
            raise OpeningContractError("raw transaction attempt provenance changed")
    timestamps = sorted(str(record["retrieved_at_utc"]) for record in records)
    expected_span = {"first": timestamps[0], "last": timestamps[-1]}
    if attempt_index.get("retrieval_span_utc") != expected_span:
        raise OpeningContractError("transport retrieval span changed")
    expected_summary = {
        "opening_count": 1,
        "attempt_count": len(attempts),
        "resume_count": resume_count,
        "completed_before_final_attempt_request_sha256": starts[final_number][
            "completed_before_attempt_request_sha256"
        ],
        "retrieval_span_utc": expected_span,
    }
    if acquisition.get("transport_summary") != expected_summary:
        raise OpeningContractError("acquisition transport summary changed")
    return expected_summary


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_development_contract(
    suite: Mapping[str, Any],
    *,
    root: Path,
    development_registry_sha256: str,
) -> dict[str, str]:
    contract = suite.get("development_contract")
    if not isinstance(contract, Mapping):
        raise OpeningContractError("model suite lacks canonical development_contract")
    spec_path = _verify_file_binding(
        root, contract.get("frozen_panel_spec", {}), label="suite frozen-panel spec"
    )
    panel_path = _verify_file_binding(
        root, contract.get("panel", {}), label="suite development panel"
    )
    registry_path = _verify_file_binding(
        root, contract.get("registry", {}), label="suite development registry"
    )
    try:
        spec = FrozenPanelSpec.load(spec_path)
        evidence = spec.verify()
    except Exception as exc:
        raise OpeningContractError("suite canonical frozen-panel spec is invalid") from exc
    if panel_path != spec.panel_path.resolve() or registry_path != spec.registry_path.resolve():
        raise OpeningContractError(
            "suite development bindings do not resolve through FrozenPanelSpec"
        )
    if (
        contract["panel"].get("sha256") != evidence["panel_sha256"]
        or contract["registry"].get("sha256") != evidence["registry_sha256"]
    ):
        raise OpeningContractError("suite development binding checksums changed")
    if development_registry_sha256 != evidence["registry_sha256"]:
        raise OpeningContractError(
            "authorized development registry differs from FrozenPanelSpec"
        )
    source_sha = contract.get("source_sha256")
    if not _is_sha256(source_sha):
        raise OpeningContractError("suite development source hash is absent or malformed")
    if source_tree_hash(root) != source_sha:
        raise OpeningContractError(
            "suite development source hash differs from the executing source tree"
        )
    return {
        "panel_sha256": str(evidence["panel_sha256"]),
        "registry_sha256": str(evidence["registry_sha256"]),
        "source_sha256": str(source_sha),
    }


def _validate_development_prediction_parity(
    metadata: Mapping[str, Any],
    *,
    root: Path,
    cohort: str,
    model_id: str,
    lineage: Mapping[str, str],
    member_count: int,
) -> None:
    parity = metadata.get("development_prediction")
    try:
        validate_development_prediction_binding(
            root, parity, label=f"{cohort}/{model_id}"
        )
    except ModelSuiteError as exc:
        raise OpeningContractError(
            f"{cohort}/{model_id} development prediction parity is invalid"
        ) from exc
    if not isinstance(parity, Mapping):  # validator above is authoritative
        raise OpeningContractError(f"{cohort}/{model_id} parity is malformed")
    if tuple(parity.get("forecast_key_columns", ())) != (
        "site_id", "horizon", "issue_date", "target_date"
    ):
        raise OpeningContractError(f"{cohort}/{model_id} parity key schema changed")
    if tuple(parity.get("prediction_columns", ())) != tuple(R.PRED_COLS):
        raise OpeningContractError(f"{cohort}/{model_id} parity value schema changed")
    selection = parity.get("selection", {})
    if selection.get("model") != model_id or len(selection.get("seeds", ())) != member_count:
        raise OpeningContractError(f"{cohort}/{model_id} parity selection changed")
    tolerance = float(parity.get("atol", np.inf))
    difference = float(parity.get("max_abs_difference", np.inf))
    if not 0.0 <= tolerance <= 1e-5 or not 0.0 <= difference <= tolerance:
        raise OpeningContractError(f"{cohort}/{model_id} parity tolerance is unsafe")
    artifact = parity.get("artifact", {})
    artifact_path = _verify_file_binding(
        root, artifact, label=f"{cohort}/{model_id} development predictions"
    )
    sidecar_path = _verify_file_binding(
        root, artifact.get("sidecar", {}),
        label=f"{cohort}/{model_id} development prediction sidecar",
    )
    sidecar = _load_json(sidecar_path, label="development prediction sidecar")
    run = sidecar.get("run")
    expected_run = {
        "source_sha256": lineage["source_sha256"],
        "panel_sha256": lineage["panel_sha256"],
        "registry_sha256": lineage["registry_sha256"],
        "config_sha256": metadata.get("config_sha256"),
    }
    if (
        sidecar.get("schema_version") != "thermoroute.artifact.v1"
        or sidecar.get("artifact_sha256") != sha256_file(artifact_path)
        or sidecar.get("artifact_bytes") != artifact_path.stat().st_size
        or not isinstance(run, Mapping)
        or any(run.get(key) != value for key, value in expected_run.items())
    ):
        raise OpeningContractError(
            f"{cohort}/{model_id} development prediction sidecar lineage changed"
        )


def _validate_bundle_lineage(
    metadata: Mapping[str, Any],
    *,
    root: Path,
    cohort: str,
    model_id: str,
    lineage: Mapping[str, str],
    member_count: int,
) -> None:
    for key in ("panel_sha256", "registry_sha256", "source_sha256"):
        if metadata.get(key) != lineage[key]:
            raise OpeningContractError(
                f"{cohort}/{model_id} {key} differs from canonical development contract"
            )
    if not _is_sha256(metadata.get("config_sha256")):
        raise OpeningContractError(
            f"{cohort}/{model_id} lacks a resolved config SHA-256"
        )
    _validate_development_prediction_parity(
        metadata,
        root=root,
        cohort=cohort,
        model_id=model_id,
        lineage=lineage,
        member_count=member_count,
    )


def _verify_torch_bundle(
    *,
    root: Path,
    entry: Mapping[str, Any],
    expected_features: tuple[str, ...],
    registry_sha256: str,
    external_sites: Sequence[str] | None,
    cohort: str,
    model_id: str,
    lineage: Mapping[str, str],
) -> dict[str, Any]:
    artifact = entry.get("artifact")
    if not isinstance(artifact, Mapping):
        raise OpeningContractError(f"{entry.get('model_id')} lacks bundle artifact")
    directory = _resolve_inside(root, artifact.get("path"), kind="directory")
    for name in ("metadata.json", "weights.pt"):
        key = name.replace(".", "_") + "_sha256"
        # Prefer clear keys but accept the Stage-9 pointer spellings.
        expected = artifact.get(key)
        if expected is None:
            expected = artifact.get("metadata_sha256" if name == "metadata.json"
                                    else "weights_sha256")
        if expected != sha256_file(directory / name):
            raise OpeningContractError(
                f"{entry.get('model_id')} {name} checksum mismatch"
            )
    count = int(entry.get("member_count", 0))
    try:
        _, metadata = load_inference_bundle(directory, expected_member_count=count)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise OpeningContractError(
            f"{entry.get('model_id')} inference bundle is invalid"
        ) from exc
    if tuple(metadata.get("feature_order", ())) != expected_features:
        raise OpeningContractError(
            f"{entry.get('model_id')} bundle uses a different actual feature schema"
        )
    if tuple(int(value) for value in metadata.get("horizons", ())) != (1, 3, 7):
        raise OpeningContractError(f"{entry.get('model_id')} has wrong horizons")
    if metadata.get("registry_sha256") != registry_sha256:
        raise OpeningContractError(
            f"{entry.get('model_id')} is not bound to the requested training registry"
        )
    _validate_bundle_lineage(
        metadata,
        root=root,
        cohort=cohort,
        model_id=model_id,
        lineage=lineage,
        member_count=count,
    )
    try:
        if external_sites is not None:
            reconstruct_frozen_transforms(metadata, external_sites, external=True)
        else:
            ordered = sorted(
                metadata["station_to_index"],
                key=lambda site: int(metadata["station_to_index"][site]),
            )
            reconstruct_frozen_transforms(metadata, ordered, external=False)
    except (FrozenInferenceError, KeyError, TypeError, ValueError) as exc:
        label = "external station-agnostic" if external_sites is not None else "same-station"
        raise OpeningContractError(
            f"{label} {entry.get('model_id')} preprocessing is not reconstructable"
        ) from exc
    return metadata


def _verify_lightgbm_bundle(
    *, root: Path, entry: Mapping[str, Any], expected_features: tuple[str, ...],
    external: bool, cohort: str, lineage: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path = _verify_file_binding(
        root, entry.get("artifact", {}), label=f"{entry.get('model_id')} bundle"
    )
    try:
        boosters, manifest = load_lightgbm_bundle(manifest_path)
    except ModelSuiteError as exc:
        raise OpeningContractError("LightGBM native-text bundle is invalid") from exc
    if tuple(manifest.get("raw_feature_order", ())) != expected_features:
        raise OpeningContractError("LightGBM raw feature schema differs from suite")
    if bool(manifest.get("station_agnostic")) != external:
        raise OpeningContractError("LightGBM station-identity contract is wrong")
    if bool(manifest.get("uses_station_categorical")) == external:
        raise OpeningContractError("external LightGBM must not use a station category")
    members = tuple(str(value) for value in manifest.get("members", ()))
    if (
        int(entry.get("member_count", 0)) != 5
        or int(manifest.get("member_count", 0)) != 5
        or set(boosters) != set(members)
        or any(set(horizons) != {1, 3, 7} for horizons in boosters.values())
        or any(
            set(heads) != {"point", "q05", "q50", "q95", "event"}
            for horizons in boosters.values() for heads in horizons.values()
        )
    ):
        raise OpeningContractError(
            "LightGBM bundle must contain five members and every horizon/head"
        )
    design = tuple(manifest.get("design_feature_order", ()))
    if not design or len(design) != len(set(design)):
        raise OpeningContractError("LightGBM engineered design registry is invalid")
    for key in ("preprocessing", "event_thresholds", "event_calibrators",
                "conformal_offsets"):
        if not isinstance(manifest.get(key), Mapping):
            raise OpeningContractError(f"LightGBM bundle lacks frozen {key}")
    parity = manifest.get("roundtrip_parity")
    if not isinstance(parity, Mapping) or set(parity) != set(members):
        raise OpeningContractError("LightGBM bundle lacks native-text roundtrip parity")
    for member in members:
        horizons = parity[member]
        if not isinstance(horizons, Mapping) or set(horizons) != {"1", "3", "7"}:
            raise OpeningContractError("LightGBM roundtrip parity omits a horizon")
        for horizon in ("1", "3", "7"):
            heads = horizons[horizon]
            if not isinstance(heads, Mapping) or set(heads) != {
                "point", "q05", "q50", "q95", "event"
            }:
                raise OpeningContractError("LightGBM roundtrip parity omits a head")
            for audit in heads.values():
                if not isinstance(audit, Mapping) or int(audit.get("rows", 0)) <= 0:
                    raise OpeningContractError("LightGBM roundtrip parity is empty")
                if float(audit.get("max_abs_difference", np.inf)) > 1e-12:
                    raise OpeningContractError("LightGBM native-text roundtrip parity failed")
                if not _is_sha256(audit.get("prediction_sha256")):
                    raise OpeningContractError(
                        "LightGBM parity prediction digest is malformed"
                    )
    _validate_bundle_lineage(
        manifest,
        root=root,
        cohort=cohort,
        model_id="LightGBM",
        lineage=lineage,
        member_count=int(entry.get("member_count", 0)),
    )
    return manifest


def _normalised_thermoroute_kwargs(metadata: Mapping[str, Any]) -> dict[str, Any]:
    architecture = metadata.get("architecture", {})
    kwargs = architecture.get("kwargs", {}) if isinstance(architecture, Mapping) else {}
    if not isinstance(kwargs, Mapping):
        raise OpeningContractError("ThermoRoute architecture kwargs are malformed")
    output = dict(THERMOROUTE_INTERVENTION_DEFAULTS)
    output.update(dict(kwargs))
    # Horizon values live at bundle top level and are injected by reconstruction.
    output["horizons"] = tuple(int(value) for value in metadata.get("horizons", ()))
    return output


def _validate_temporal_controls(
    entries: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove that each control is the exact registered architecture intervention."""
    if "ThermoRoute" not in metadata:
        raise OpeningContractError("temporal suite lacks primary ThermoRoute metadata")
    primary = _normalised_thermoroute_kwargs(metadata["ThermoRoute"])
    if (
        primary.get("safety_anchor") != "damped"
        or primary.get("station_agnostic") is not False
        or primary.get("use_wlevel") is not False
    ):
        raise OpeningContractError("primary temporal ThermoRoute architecture is unsafe")
    if primary.get("delta_scale") is None:
        raise OpeningContractError("primary temporal ThermoRoute must be bounded")
    for model_id, intervention in CONTROL_INTERVENTIONS.items():
        entry = entries.get(model_id)
        control_metadata = metadata.get(model_id)
        if entry is None or control_metadata is None:
            raise OpeningContractError(f"temporal suite lacks control {model_id}")
        if int(entry.get("member_count", 0)) != 1:
            raise OpeningContractError(f"control {model_id} must contain exactly one seed")
        if entry.get("intervention") != intervention:
            raise OpeningContractError(
                f"control {model_id} intervention registry is not exact"
            )
        expected = dict(primary)
        expected.update(intervention)
        actual = _normalised_thermoroute_kwargs(control_metadata)
        if actual != expected:
            changed = {
                key: (primary.get(key), actual.get(key), expected.get(key))
                for key in sorted(set(primary) | set(actual) | set(expected))
                if actual.get(key) != expected.get(key)
            }
            raise OpeningContractError(
                f"control {model_id} changes fields beyond its intervention: {changed}"
            )


def validate_model_suite(
    suite_path: str | Path,
    *,
    root: str | Path,
    protocol_info: Mapping[str, Any],
    registries: Mapping[str, Any],
) -> dict[str, Any]:
    """Require every declared model for both temporal and external cohorts."""
    try:
        configure_deterministic_runtime()
    except RuntimeError as exc:
        raise OpeningContractError(
            "cannot apply numerical policy before model-suite validation"
        ) from exc
    root = Path(root).resolve()
    suite_path = Path(suite_path).resolve()
    suite = _load_json(suite_path, label="confirmatory model suite")
    if suite.get("format") != MODEL_SUITE_FORMAT:
        raise OpeningContractError("unsupported model-suite format")
    if suite.get("status") != "FROZEN_BEFORE_LABEL_OPENING":
        raise OpeningContractError("model suite is not frozen before opening")
    if suite.get("protocol_sha256") != protocol_info["protocol_sha256"]:
        raise OpeningContractError("model suite is bound to another protocol revision")
    try:
        validate_model_suite_document(suite, root=root)
    except ModelSuiteError as exc:
        raise OpeningContractError("model-suite executable artifact validation failed") from exc
    feature_order = tuple(str(value) for value in suite.get("actual_feature_order", ()))
    if not feature_order or "WTEMP" not in feature_order or len(feature_order) != len(set(feature_order)):
        raise OpeningContractError("model suite has an invalid actual feature schema")
    protocol_schema = protocol_info["document"].get(
        "primary_inference_contract", {}
    ).get("feature_order")
    if tuple(protocol_schema or ()) != feature_order:
        raise OpeningContractError("bundle feature schema differs from amended protocol")
    required_by_cohort = {
        cohort: _required_models(protocol_info["document"], cohort=cohort)
        for cohort in ("temporal", "external")
    }
    lineage = _validate_development_contract(
        suite,
        root=root,
        development_registry_sha256=registries["development_sha256"],
    )
    cohorts = suite.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {"temporal", "external"}:
        raise OpeningContractError("model suite must contain temporal and external cohorts")
    metadata_by_cohort: dict[str, dict[str, Any]] = {}
    entries_by_cohort: dict[str, dict[str, Mapping[str, Any]]] = {}
    for cohort_name, expected_registry, external_sites in (
        ("temporal", registries["development_sha256"], None),
        ("external", registries["development_sha256"],
         tuple(registries["external"].site_no.astype(str))),
    ):
        cohort = cohorts[cohort_name]
        if not isinstance(cohort, Mapping):
            raise OpeningContractError(f"{cohort_name} model cohort is malformed")
        expected_mode = (
            "same_station" if cohort_name == "temporal"
            else "station_agnostic_history_dependent_new_site"
        )
        if cohort.get("site_mode") != expected_mode:
            raise OpeningContractError(f"{cohort_name} site-mode contract is wrong")
        entries = cohort.get("models")
        if not isinstance(entries, list):
            raise OpeningContractError(f"{cohort_name} model registry is not a list")
        required = required_by_cohort[cohort_name]
        ids = tuple(str(entry.get("model_id")) for entry in entries
                    if isinstance(entry, Mapping))
        if set(ids) != set(required) or len(ids) != len(required):
            missing = sorted(set(required) - set(ids))
            extra = sorted(set(ids) - set(required))
            raise OpeningContractError(
                f"{cohort_name} suite is incomplete: missing={missing}, extra={extra}"
            )
        metadata_by_cohort[cohort_name] = {}
        entries_by_cohort[cohort_name] = {
            str(entry["model_id"]): entry for entry in entries
        }
        for entry in entries:
            model_id = str(entry["model_id"])
            executor = str(entry.get("executor", ""))
            if executor not in SUPPORTED_EXECUTORS:
                raise OpeningContractError(
                    f"{cohort_name}/{model_id} has unsupported executor {executor!r}"
                )
            if tuple(entry.get("raw_feature_order", feature_order)) != feature_order:
                raise OpeningContractError(
                    f"{cohort_name}/{model_id} declares another raw feature schema"
                )
            if model_id in BUILTIN_MODELS:
                if executor != "builtin" or "artifact" in entry:
                    raise OpeningContractError(f"{model_id} must be a frozen builtin")
                continue
            if model_id == "LightGBM":
                if executor != "lightgbm_bundle":
                    raise OpeningContractError("LightGBM requires its executable bundle")
                metadata = _verify_lightgbm_bundle(
                    root=root, entry=entry, expected_features=feature_order,
                    external=cohort_name == "external",
                    cohort=cohort_name,
                    lineage=lineage,
                )
                metadata_by_cohort[cohort_name][model_id] = metadata
                continue
            expected_executor = "lstm_bundle" if model_id == "LSTM" else "thermoroute_bundle"
            if executor != expected_executor:
                raise OpeningContractError(
                    f"{model_id} requires executor {expected_executor}"
                )
            metadata = _verify_torch_bundle(
                root=root,
                entry=entry,
                expected_features=feature_order,
                registry_sha256=expected_registry,
                external_sites=external_sites,
                cohort=cohort_name,
                model_id=model_id,
                lineage=lineage,
            )
            expected_class = (
                "thermoroute.train.LSTMForecaster"
                if model_id == "LSTM"
                else "thermoroute.thermoroute.ThermoRoute"
            )
            if metadata.get("architecture", {}).get("class") != expected_class:
                raise OpeningContractError(
                    f"{model_id} bundle architecture is not {expected_class}"
                )
            if model_id == "ThermoRoute" and int(entry.get("member_count", 0)) != 5:
                raise OpeningContractError("primary ThermoRoute must contain all five seeds")
            if model_id == "LSTM" and int(entry.get("member_count", 0)) != 5:
                raise OpeningContractError("LSTM must contain its matched five-seed budget")
            metadata_by_cohort[cohort_name][model_id] = metadata
    _validate_temporal_controls(
        entries_by_cohort["temporal"], metadata_by_cohort["temporal"]
    )
    for cohort_name, station_frame in (
        ("temporal", registries["development"]),
        ("external", registries["external"]),
    ):
        cohort_metadata = metadata_by_cohort[cohort_name]
        primary = cohort_metadata["ThermoRoute"]
        if _normalised_thermoroute_kwargs(primary).get("use_wlevel") is not False:
            raise OpeningContractError(
                f"{cohort_name} ThermoRoute illegally consumes WLEVEL"
            )
        external = cohort_name == "external"
        stations = tuple(station_frame.site_no.astype(str))
        primary_order = _station_order(primary, stations, external=external)
        primary_thresholds = primary.get("event_thresholds")
        primary_event_reference = primary.get("event_reference_climatology")
        if not isinstance(primary_thresholds, Mapping) or not isinstance(
            primary_event_reference, Mapping
        ):
            raise OpeningContractError(
                f"{cohort_name} primary bundle lacks frozen event-reference metadata"
            )
        try:
            validate_frozen_seasonal_event_reference(
                primary_event_reference,
                expected_sites=None if external else set(primary_order),
                pooled=external,
            )
        except ValueError as exc:
            raise OpeningContractError(
                f"{cohort_name} frozen event reference is invalid"
            ) from exc
        for model_id, model_metadata in cohort_metadata.items():
            if model_metadata.get("preprocessing") != primary.get("preprocessing"):
                raise OpeningContractError(
                    f"{cohort_name}/{model_id} preprocessing differs from primary"
                )
            if (
                model_metadata.get("event_thresholds") != primary_thresholds
                or model_metadata.get("event_reference_climatology")
                != primary_event_reference
            ):
                raise OpeningContractError(
                    f"{cohort_name}/{model_id} event threshold/reference "
                    "differs from primary"
                )
            _validate_frozen_calibration_registry(
                model_metadata,
                primary_order,
                (1, 3, 7),
                external=external,
                label=f"{cohort_name}/{model_id}",
            )
            if model_id != "LightGBM" and _station_order(
                model_metadata, stations, external=external
            ) != primary_order:
                raise OpeningContractError(
                    f"{cohort_name}/{model_id} station order differs from primary"
                )
        lightgbm = cohort_metadata["LightGBM"]
        categories = tuple(str(value) for value in lightgbm.get("station_categories", ()))
        design = tuple(str(value) for value in lightgbm.get("design_feature_order", ()))
        if external:
            if categories or "station_code" in design:
                raise OpeningContractError(
                    "external LightGBM cannot consume a station category"
                )
        elif categories != primary_order or "station_code" not in design:
            raise OpeningContractError(
                "same-station LightGBM category registry differs from station order"
            )
    runtime_digests = {
        str(model_metadata.get("runtime_sha256", ""))
        for cohort_metadata in metadata_by_cohort.values()
        for model_metadata in cohort_metadata.values()
    }
    current_runtime_sha256 = sha256_json(numerical_runtime_contract())
    if runtime_digests != {current_runtime_sha256}:
        raise OpeningContractError(
            "frozen model-suite runtime differs from the executing numerical stack"
        )
    return {
        "document": suite,
        "sha256": sha256_file(suite_path),
        "feature_order": feature_order,
        "required_models": required_by_cohort,
        "metadata": metadata_by_cohort,
        "entries": entries_by_cohort,
        "lineage": lineage,
        "runtime_sha256": current_runtime_sha256,
    }


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path, dtype={"site_no": "string"})
    raise OpeningContractError(f"unsupported frozen input table: {path.name}")


def _snapshot_response_path(index_path: Path, record: Mapping[str, Any]) -> Path:
    snapshot_root = index_path.parent.resolve()
    path = (snapshot_root / str(record.get("response_path", ""))).resolve()
    if snapshot_root not in path.parents or not path.is_file():
        raise OpeningContractError("meteorology snapshot response escapes its store")
    return path


def _verify_prelabel_meteorology_replay(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    registries: Mapping[str, Any],
    normalized_tables: Mapping[str, pd.DataFrame],
    required_fields: tuple[str, ...],
    history_start: str,
    target_end: str,
    protocol_sha256: str,
) -> None:
    """Independently replay every Daymet/gridMET byte into consumed tables.

    Producer summaries are deliberately ignored.  Registry coordinates define
    the only admissible canonical request set; the request map and both snapshot
    indexes must be exact bijections onto that set, and pure parsers must rebuild
    the complete Parquet tables byte-for-value (modulo Parquet dtype encoding).
    """
    if set(required_fields) != {"TEMP", "PRCP", "RHMEAN", "DH", "WDSP"}:
        raise OpeningContractError("Route-A meteorology replay has an unknown field set")
    if len(evidence) != 4:
        raise OpeningContractError(
            "pre-label meteorology requires Daymet, gridMET data/schema and request-map evidence"
        )

    snapshot_indexes: dict[str, tuple[Path, list[Mapping[str, Any]]]] = {}
    request_map_path: Path | None = None
    declared_gridmet_contract: Mapping[str, Any] | None = None
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise OpeningContractError("pre-label source evidence is not an object")
        artifact_path = _verify_file_binding(
            root, item.get("artifact", {}), label=f"source evidence {index}"
        )
        evidence_type = item.get("evidence_type")
        if evidence_type == "snapshot_index":
            records = _verify_snapshot_index(root, artifact_path, prelabel=True)
            providers = {str(record.get("provider", "")) for record in records}
            if providers not in (
                {DAYMET_PROVIDER}, {GRIDMET_PROVIDER}, {GRIDMET_SCHEMA_PROVIDER}
            ):
                raise OpeningContractError(
                    "pre-label snapshot index has an unapproved provider identity"
                )
            provider = next(iter(providers))
            if provider in snapshot_indexes:
                raise OpeningContractError("pre-label evidence duplicates a provider index")
            expected_fields = ({"TEMP", "PRCP", "RHMEAN", "DH"}
                               if provider == DAYMET_PROVIDER else {"WDSP"})
            if set(item.get("fields", ())) != expected_fields:
                raise OpeningContractError(
                    f"{provider} evidence field registry is not exact"
                )
            snapshot_indexes[provider] = (artifact_path, records)
            if provider == GRIDMET_SCHEMA_PROVIDER:
                if item.get("contract_attributes") != [
                    "units", "scale_factor", "add_offset"
                ] or not isinstance(item.get("validated_contract"), Mapping):
                    raise OpeningContractError("gridMET schema evidence contract changed")
                declared_gridmet_contract = item["validated_contract"]
        elif evidence_type == "normalized_immutable_snapshot":
            if request_map_path is not None:
                raise OpeningContractError("pre-label evidence duplicates the request map")
            if set(item.get("fields", ())) != set(required_fields):
                raise OpeningContractError("meteorology request-map fields are not exact")
            request_map_path = artifact_path
        else:
            raise OpeningContractError("unsupported source-evidence type")
    if set(snapshot_indexes) != {
        DAYMET_PROVIDER, GRIDMET_PROVIDER, GRIDMET_SCHEMA_PROVIDER
    } or request_map_path is None or declared_gridmet_contract is None:
        raise OpeningContractError("pre-label provider evidence is incomplete")

    registry_inputs = manifest.get("registry_inputs")
    if not isinstance(registry_inputs, Mapping) or set(registry_inputs) != {
        "temporal", "external"
    }:
        raise OpeningContractError("pre-label inputs lack exact registry-coordinate bindings")
    station_registry: dict[tuple[str, str], tuple[float, float]] = {}
    for cohort, checksum_key in (
        ("temporal", "development_sha256"),
        ("external", "external_sha256"),
    ):
        registry = registries["development" if cohort == "temporal" else "external"]
        if not {"site_no", "lat", "lon"} <= set(registry):
            raise OpeningContractError(f"{cohort} registry lacks stable coordinates")
        binding = registry_inputs[cohort]
        if not isinstance(binding, Mapping) or set(binding) != {
            "path", "sha256", "columns_read", "row_count"
        }:
            raise OpeningContractError(f"{cohort} registry-input binding schema changed")
        if (
            binding.get("sha256") != registries[checksum_key]
            or binding.get("columns_read") != ["site_no", "lat", "lon"]
            or int(binding.get("row_count", -1)) != len(registry)
        ):
            raise OpeningContractError(f"{cohort} registry-coordinate binding changed")
        bound_registry = _verify_file_binding(
            root, binding, label=f"{cohort} meteorology coordinate registry"
        )
        header = pd.read_csv(bound_registry, nrows=0)
        if not {"site_no", "lat", "lon"} <= set(header):
            raise OpeningContractError(f"{cohort} bound coordinate registry is incomplete")
        coordinates = pd.read_csv(
            bound_registry,
            usecols=["site_no", "lat", "lon"],
            dtype={"site_no": "string"},
            keep_default_na=False,
        )
        coordinates["site_no"] = coordinates.site_no.astype("string").str.strip()
        coordinates["lat"] = pd.to_numeric(coordinates.lat, errors="coerce")
        coordinates["lon"] = pd.to_numeric(coordinates.lon, errors="coerce")
        expected_coordinates = registry[["site_no", "lat", "lon"]].copy()
        expected_coordinates["site_no"] = (
            expected_coordinates.site_no.astype("string").str.strip()
        )
        expected_coordinates["lat"] = pd.to_numeric(
            expected_coordinates.lat, errors="coerce"
        )
        expected_coordinates["lon"] = pd.to_numeric(
            expected_coordinates.lon, errors="coerce"
        )
        coordinates = coordinates.sort_values("site_no").reset_index(drop=True)
        expected_coordinates = expected_coordinates.sort_values("site_no").reset_index(
            drop=True
        )
        try:
            pd.testing.assert_frame_equal(
                coordinates, expected_coordinates, check_dtype=False, rtol=0.0, atol=0.0
            )
        except AssertionError as exc:
            raise OpeningContractError(
                f"{cohort} coordinate registry differs from opening registry"
            ) from exc
        for row in coordinates.itertuples(index=False):
            key = (cohort, str(row.site_no))
            if key in station_registry:
                raise OpeningContractError("meteorology station registry is duplicated")
            lat, lon = float(row.lat), float(row.lon)
            if not np.isfinite([lat, lon]).all():
                raise OpeningContractError("meteorology registry has non-finite coordinates")
            station_registry[key] = (lat, lon)

    headers = {"User-Agent": METEOROLOGY_USER_AGENT}
    expected_requests: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        DAYMET_PROVIDER: {}, GRIDMET_PROVIDER: {}, GRIDMET_SCHEMA_PROVIDER: {},
    }
    for key, (lat, lon) in station_registry.items():
        urls = {
            DAYMET_PROVIDER: build_daymet_url(lat, lon, history_start, target_end),
            GRIDMET_PROVIDER: build_gridmet_wind_url(lat, lon, history_start, target_end),
        }
        for provider, url in urls.items():
            request = {
                "schema_version": 1,
                "provider": provider,
                "method": "GET",
                "url": url,
                "headers": headers,
            }
            expected_requests[provider][key] = request
    schema_request = {
        "schema_version": 1,
        "provider": GRIDMET_SCHEMA_PROVIDER,
        "method": "GET",
        "url": build_gridmet_wind_metadata_url(),
        "headers": headers,
    }
    expected_requests[GRIDMET_SCHEMA_PROVIDER][("schema", "gridmet")] = schema_request

    indexed_by_provider: dict[str, dict[str, Mapping[str, Any]]] = {}
    for provider, (index_path, records) in snapshot_indexes.items():
        expected_by_sha = {
            hashlib.sha256(canonical_json_bytes(request)).hexdigest(): request
            for request in expected_requests[provider].values()
        }
        actual_by_sha = {str(record["request_sha256"]): record for record in records}
        if set(actual_by_sha) != set(expected_by_sha):
            raise OpeningContractError(
                f"{provider} snapshot requests differ from the exact station-coordinate registry"
            )
        for request_sha, expected_request in expected_by_sha.items():
            record = actual_by_sha[request_sha]
            if dict(record["request"]) != expected_request:
                raise OpeningContractError(f"{provider} canonical request changed")
            _snapshot_response_path(index_path, record)
        indexed_by_provider[provider] = actual_by_sha

    schema_sha = hashlib.sha256(canonical_json_bytes(schema_request)).hexdigest()
    schema_payload = _snapshot_response_path(
        snapshot_indexes[GRIDMET_SCHEMA_PROVIDER][0],
        indexed_by_provider[GRIDMET_SCHEMA_PROVIDER][schema_sha],
    ).read_bytes()
    try:
        gridmet_contract = parse_gridmet_wind_metadata(schema_payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise OpeningContractError("cannot replay gridMET provider packing metadata") from exc
    if dict(declared_gridmet_contract) != gridmet_contract:
        raise OpeningContractError("declared gridMET packing contract differs from raw schema")

    request_map = _load_json(request_map_path, label="meteorology request map")
    required_map_keys = {
        "format", "protocol_sha256", "contains_outcome", "contains_outcome_labels",
        "labels_requested_or_read", "outcome_endpoint_called",
        "post_2020_wtemp_requested_or_inspected", "fields", "history_start",
        "target_end", "request_count", "requests", "gridmet_provider_contract",
    }
    if set(request_map) != required_map_keys:
        raise OpeningContractError("meteorology request-map schema changed")
    expected_map_values = {
        "format": REQUEST_MAP_FORMAT,
        "protocol_sha256": protocol_sha256,
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "post_2020_wtemp_requested_or_inspected": False,
        "fields": list(required_fields),
        "history_start": history_start,
        "target_end": target_end,
        "request_count": len(station_registry),
    }
    if any(request_map.get(key) != value for key, value in expected_map_values.items()):
        raise OpeningContractError("meteorology request-map contract changed")
    schema_record = indexed_by_provider[GRIDMET_SCHEMA_PROVIDER][schema_sha]
    expected_gridmet_map_contract = {
        **gridmet_contract,
        "request_sha256": schema_record["request_sha256"],
        "response_sha256": schema_record["response_sha256"],
        "retrieved_at_utc": schema_record["retrieved_at_utc"],
        "byte_count": schema_record["byte_count"],
    }
    if request_map.get("gridmet_provider_contract") != expected_gridmet_map_contract:
        raise OpeningContractError(
            "meteorology request map does not bind the gridMET schema snapshot"
        )
    map_records = request_map.get("requests")
    if not isinstance(map_records, list) or len(map_records) != len(station_registry):
        raise OpeningContractError("meteorology request-map count is inconsistent")
    map_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in map_records:
        if not isinstance(item, Mapping) or set(item) != {
            "cohort", "site_no", "requested_lat", "requested_lon",
            "contains_outcome", "contains_outcome_labels", "daymet", "gridmet",
        }:
            raise OpeningContractError("meteorology request-map row schema changed")
        key = (str(item.get("cohort", "")), str(item.get("site_no", "")))
        if key not in station_registry or key in map_by_key:
            raise OpeningContractError("meteorology request-map station registry changed")
        lat, lon = station_registry[key]
        if (
            item.get("contains_outcome") is not False
            or item.get("contains_outcome_labels") is not False
            or float(item.get("requested_lat", np.nan)) != lat
            or float(item.get("requested_lon", np.nan)) != lon
        ):
            raise OpeningContractError("meteorology request-map coordinate/label flags changed")
        for nested_name, provider in (
            ("daymet", DAYMET_PROVIDER), ("gridmet", GRIDMET_PROVIDER)
        ):
            nested = item.get(nested_name)
            if not isinstance(nested, Mapping) or set(nested) != {
                "request_sha256", "response_sha256", "retrieved_at_utc", "byte_count"
            }:
                raise OpeningContractError("meteorology request-map source row changed")
            expected_request = expected_requests[provider][key]
            request_sha = hashlib.sha256(
                canonical_json_bytes(expected_request)
            ).hexdigest()
            indexed = indexed_by_provider[provider][request_sha]
            for field in (
                "request_sha256", "response_sha256", "retrieved_at_utc", "byte_count"
            ):
                if nested.get(field) != indexed.get(field):
                    raise OpeningContractError(
                        "meteorology request map does not bind its raw snapshot"
                    )
        map_by_key[key] = item
    if set(map_by_key) != set(station_registry):
        raise OpeningContractError("meteorology request map omits a frozen station")

    rebuilt: dict[str, list[pd.DataFrame]] = {"temporal": [], "external": []}
    for key in sorted(station_registry):
        cohort, site_no = key
        parsed: dict[str, Any] = {}
        for provider in (DAYMET_PROVIDER, GRIDMET_PROVIDER):
            request = expected_requests[provider][key]
            request_sha = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
            index_path = snapshot_indexes[provider][0]
            payload = _snapshot_response_path(
                index_path, indexed_by_provider[provider][request_sha]
            ).read_bytes()
            try:
                if provider == DAYMET_PROVIDER:
                    parsed[provider] = parse_daymet_daily(
                        payload, start=history_start, end=target_end
                    )
                else:
                    parsed[provider] = parse_gridmet_wind_daily(
                        payload,
                        start=history_start,
                        end=target_end,
                        scale_factor=float(gridmet_contract["scale_factor"]),
                        add_offset=float(gridmet_contract["add_offset"]),
                    )
            except (UnicodeDecodeError, ValueError, KeyError) as exc:
                raise OpeningContractError(
                    f"cannot replay {provider} raw bytes for {site_no}"
                ) from exc
        daymet = parsed[DAYMET_PROVIDER]
        wind = parsed[GRIDMET_PROVIDER]
        if not daymet.index.equals(wind.index):
            raise OpeningContractError("meteorology raw-source calendars disagree")
        frame = daymet.copy()
        frame["WDSP"] = wind.to_numpy(float)
        frame = frame.reset_index()
        frame.insert(0, "site_no", site_no)
        rebuilt[cohort].append(frame[["site_no", "DATE", *required_fields]])

    for cohort in ("temporal", "external"):
        expected = pd.concat(rebuilt[cohort], ignore_index=True)
        expected["site_no"] = expected.site_no.astype("string")
        expected["DATE"] = pd.to_datetime(expected.DATE)
        expected = expected.sort_values(["site_no", "DATE"]).reset_index(drop=True)
        actual = normalized_tables[cohort].copy()
        actual["site_no"] = actual.site_no.astype("string").str.strip()
        actual["DATE"] = pd.to_datetime(actual.DATE)
        actual = actual[["site_no", "DATE", *required_fields]].sort_values(
            ["site_no", "DATE"]
        ).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                expected, actual, check_dtype=False, rtol=0.0, atol=1e-12
            )
        except AssertionError as exc:
            raise OpeningContractError(
                f"{cohort} meteorology table cannot be rebuilt from raw snapshots"
            ) from exc


def validate_prelabel_inputs(
    manifest_path: str | Path,
    *,
    root: str | Path,
    protocol_info: Mapping[str, Any],
    registries: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify actual non-outcome tables and their immutable source evidence."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json(manifest_path, label="pre-label actual-input manifest")
    if manifest.get("format") != INPUT_MANIFEST_FORMAT:
        raise OpeningContractError("unsupported pre-label input-manifest format")
    required_flags = {
        "status": "FROZEN_PRELABEL_NO_OUTCOMES",
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "post_2020_wtemp_requested_or_inspected": False,
        "retrospective_provisional_vintage_reconstructable": False,
        "horizon_specific_future_nwp_consumed": False,
    }
    wrong = {key: manifest.get(key) for key, value in required_flags.items()
             if manifest.get(key) != value}
    if wrong:
        raise OpeningContractError(f"pre-label input-manifest flags are unsafe: {wrong}")
    if manifest.get("protocol_sha256") != protocol_info["protocol_sha256"]:
        raise OpeningContractError("pre-label inputs are bound to another protocol")
    if tuple(manifest.get("actual_feature_order", ())) != suite["feature_order"]:
        raise OpeningContractError("pre-label input schema differs from model bundles")
    required_fields = tuple(
        variable for variable in suite["feature_order"]
        if variable not in OPENING_ACQUIRED_FIELDS
    )
    if set(manifest.get("prelabel_fields", ())) != set(required_fields):
        raise OpeningContractError(
            "pre-label field registry is not exactly the non-outcome bundle schema"
        )
    history_start = str(manifest.get("history_start", ""))
    target_end = protocol_info["target_end"]
    try:
        expected_dates = pd.date_range(history_start, target_end, freq="D")
    except (TypeError, ValueError) as exc:
        raise OpeningContractError("pre-label history_start is invalid") from exc
    if len(expected_dates) < 32:
        raise OpeningContractError("pre-label input interval cannot cover 32-day context")
    if (
        manifest.get("target_start") != protocol_info["target_start"]
        or manifest.get("target_end") != target_end
        or int(manifest.get("context_length_days", -1)) != 32
    ):
        raise OpeningContractError("pre-label calendar/context contract changed")
    tables = manifest.get("cohort_tables")
    if not isinstance(tables, Mapping) or set(tables) != {"temporal", "external"}:
        raise OpeningContractError("pre-label inputs need both cohort tables")
    normalized_tables: dict[str, pd.DataFrame] = {}
    for cohort, registry in (
        ("temporal", registries["development"]),
        ("external", registries["external"]),
    ):
        binding = tables[cohort]
        table_path = _verify_file_binding(
            root, binding, label=f"{cohort} retrospective input table"
        )
        table = _read_table(table_path)
        expected_columns = {"site_no", "DATE", *required_fields}
        if set(table) != expected_columns:
            raise OpeningContractError(
                f"{cohort} input columns differ: expected {sorted(expected_columns)}, "
                f"got {sorted(table.columns)}"
            )
        lowered = {str(column).lower() for column in table.columns}
        if any(field.lower() in lowered for field in OPENING_ACQUIRED_FIELDS):
            raise OpeningContractError(f"{cohort} pre-label table contains outcome/history fields")
        table = table.copy()
        table["site_no"] = table.site_no.astype("string").str.strip()
        table["DATE"] = pd.to_datetime(table.DATE, errors="coerce")
        if table.DATE.isna().any() or table.duplicated(["site_no", "DATE"]).any():
            raise OpeningContractError(f"{cohort} input keys are invalid or duplicated")
        sites = set(registry.site_no.astype(str))
        if set(table.site_no.astype(str)) != sites:
            raise OpeningContractError(f"{cohort} input sites differ from frozen registry")
        expected_rows = len(sites) * len(expected_dates)
        if len(table) != expected_rows:
            raise OpeningContractError(
                f"{cohort} table is not a complete site-by-day registry"
            )
        if table.DATE.min() != expected_dates.min() or table.DATE.max() != expected_dates.max():
            raise OpeningContractError(f"{cohort} input date support is incomplete")
        per_site = table.groupby("site_no").DATE.nunique()
        if not per_site.eq(len(expected_dates)).all():
            raise OpeningContractError(f"{cohort} input calendar has station gaps")
        normalized_tables[cohort] = table
    evidence = manifest.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise OpeningContractError("pre-label inputs lack source evidence")
    covered: set[str] = set()
    for item in evidence:
        if not isinstance(item, Mapping) or item.get("contains_outcome_labels") is not False:
            raise OpeningContractError("source evidence does not exclude outcomes")
        if item.get("contains_outcome") is not False:
            raise OpeningContractError("source evidence does not exclude outcome history")
        fields = set(item.get("fields", ()))
        if not fields or not fields <= set(required_fields):
            raise OpeningContractError("source evidence names undeclared fields")
        covered |= fields
    if covered != set(required_fields):
        raise OpeningContractError("source evidence does not cover every consumed field")
    _verify_prelabel_meteorology_replay(
        root=root,
        manifest=manifest,
        evidence=evidence,
        registries=registries,
        normalized_tables=normalized_tables,
        required_fields=required_fields,
        history_start=history_start,
        target_end=target_end,
        protocol_sha256=protocol_info["protocol_sha256"],
    )
    nwp = manifest.get("secondary_nwp_resolution")
    if nwp not in {"ACQUIRED_AND_FROZEN", "EXPLICITLY_NOT_USED"}:
        raise OpeningContractError("optional NWP status is unresolved before opening")
    return {
        "document": manifest,
        "sha256": sha256_file(manifest_path),
        "history_start": history_start,
    }


def _validate_prelabel_chronology_for_opening(
    chronology_receipt: str | Path,
    *,
    root: Path,
    protocol_info: Mapping[str, Any],
    registries: Mapping[str, Any],
    model_suite: str | Path,
    development_replay_receipt: str | Path,
    external_registry: str | Path,
    external_lock: str | Path,
    input_manifest: str | Path,
    allow_gitless_archive: bool = False,
) -> dict[str, Any]:
    """Require the exact Git chronology that precedes label authorization."""
    receipt_path = Path(chronology_receipt).resolve()
    if root not in receipt_path.parents or not receipt_path.is_file():
        raise OpeningContractError(
            "prelabel chronology receipt escapes repository or is absent"
        )
    if _relative(root, receipt_path) != DEFAULT_PRELABEL_CHRONOLOGY_RECEIPT:
        raise OpeningContractError(
            "prelabel chronology receipt does not use its canonical path"
        )
    try:
        if (root / ".git").exists():
            _assert_safe_live_git_repository(root)
            with _git_replacement_objects_disabled():
                chronology = validate_prelabel_chronology(
                    receipt_path,
                    root=root,
                    allow_gitless_archive=allow_gitless_archive,
                )
        else:
            chronology = validate_prelabel_chronology(
                receipt_path,
                root=root,
                allow_gitless_archive=allow_gitless_archive,
            )
    except ChronologyError as exc:
        raise OpeningContractError(
            "repository-internal prelabel chronology is absent or stale"
        ) from exc
    if (
        chronology.get("format") != CHRONOLOGY_FORMAT
        or chronology.get("status") != CHRONOLOGY_STATUS
        or chronology.get("external_timestamp_or_public_preregistration") is not False
        or chronology.get("independent_custodian_or_worm_storage") is not False
    ):
        raise OpeningContractError("prelabel chronology status/scope changed")
    expected_paths = {
        "protocol_seal": _relative(root, protocol_info["seal"]["path"]),
        "model_suite": _relative(root, model_suite),
        "development_replay": _relative(root, development_replay_receipt),
        "candidate_table": _relative(root, registries["candidate_table"]),
        "candidate_provenance": _relative(
            root, registries["candidate_provenance"]
        ),
        "candidate_snapshot_index": _relative(
            root, registries["candidate_snapshot_index"]
        ),
        "external_registry": _relative(root, external_registry),
        "external_lock": _relative(root, external_lock),
        "input_manifest": _relative(root, input_manifest),
    }
    if chronology.get("paths") != expected_paths:
        raise OpeningContractError(
            "prelabel chronology binds another protocol/model/input evidence set"
        )
    history = chronology.get("protocol_history")
    if not isinstance(history, Mapping):
        raise OpeningContractError("prelabel chronology lacks protocol history")
    seal = history.get("seal")
    if (
        not isinstance(seal, Mapping)
        or seal.get("path") != expected_paths["protocol_seal"]
        or seal.get("sha256") != protocol_info["seal"]["sha256"]
        or history.get("original_commit") != protocol_info["authoritative_commit"]
        or history.get("final_prelabel_commit")
        != protocol_info["seal"]["final_commit"]
    ):
        raise OpeningContractError(
            "prelabel chronology binds another protocol Git history"
        )
    order = chronology.get("order")
    if not isinstance(order, Mapping) or order.get("strict_order_verified") is not True:
        raise OpeningContractError("prelabel chronology lacks strict commit order")
    return chronology


def _assert_prepublication_source_snapshot(
    root: Path,
    *,
    initial_git_state: Mapping[str, Any],
    frozen_source_inventory: Mapping[str, str],
) -> None:
    """Close the long-preflight TOCTOU window immediately before publication."""
    current_state = _live_git_state(root)
    if dict(current_state) != dict(initial_git_state):
        raise OpeningContractError(
            "Git HEAD/worktree changed during opening-authorization preflight"
        )
    current_inventory = source_inventory(root)
    if (
        dict(current_inventory) != dict(frozen_source_inventory)
        or sha256_json(current_inventory)
        != sha256_json(dict(frozen_source_inventory))
    ):
        raise OpeningContractError(
            "source inventory changed during opening-authorization preflight"
        )


def freeze_opening_authorization(
    destination: str | Path,
    *,
    root: str | Path,
    protocol_path: str | Path,
    development_registry: str | Path,
    external_registry: str | Path,
    external_lock: str | Path,
    model_suite: str | Path,
    input_manifest: str | Path,
    development_replay_receipt: str | Path = (
        "outputs/model_replay/route_a_development_replay_v1.json"
    ),
    prelabel_chronology_receipt: str | Path = DEFAULT_PRELABEL_CHRONOLOGY_RECEIPT,
    inference_gate: str | Path = DEFAULT_INFERENCE_GATE,
    inference_amendment: str | Path = INFERENCE_AMENDMENT_RELATIVE,
    inference_amendment_seal: str | Path = INFERENCE_AMENDMENT_SEAL_RELATIVE,
    outcome_qc_policy: str | Path = OUTCOME_QC_POLICY_RELATIVE,
) -> dict[str, Any]:
    """Validate all pre-label evidence and create the immutable authorization."""
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    _assert_safe_live_git_repository(root)
    if destination.exists():
        raise OpeningAlreadyStarted(
            f"refusing to replace one-time artifact: {destination}"
        )
    authorization_relative = _require_authorization_path_trackable(root, destination)
    state = _live_git_state(root)
    if not state.get("available") or state.get("dirty") is not False:
        raise OpeningContractError(
            "opening authorization requires a clean, committed Git source tree"
        )
    inference_gate_path = Path(inference_gate)
    if not inference_gate_path.is_absolute():
        inference_gate_path = root / inference_gate_path
    inference_gate_path = inference_gate_path.resolve()
    inference_amendment_path = Path(inference_amendment)
    if not inference_amendment_path.is_absolute():
        inference_amendment_path = root / inference_amendment_path
    inference_amendment_path = inference_amendment_path.resolve()
    inference_amendment_seal_path = Path(inference_amendment_seal)
    if not inference_amendment_seal_path.is_absolute():
        inference_amendment_seal_path = root / inference_amendment_seal_path
    inference_amendment_seal_path = inference_amendment_seal_path.resolve()
    outcome_qc_policy_path = Path(outcome_qc_policy)
    if not outcome_qc_policy_path.is_absolute():
        outcome_qc_policy_path = root / outcome_qc_policy_path
    outcome_qc_policy_path = outcome_qc_policy_path.resolve()
    protocol_info = validate_protocol(protocol_path, root=root)
    try:
        amendment = validate_inference_amendment(
            inference_amendment_path,
            root=root,
            protocol_path=protocol_path,
            protocol_seal_path=protocol_info["seal"]["path"],
        )
        amendment_seal = validate_inference_amendment_seal(
            inference_amendment_seal_path,
            root=root,
            amendment_path=inference_amendment_path,
        )
        inference_gate_document = validate_inference_gate_document(
            inference_gate_path,
            root=root,
            protocol_path=protocol_path,
            protocol_seal_path=protocol_info["seal"]["path"],
            station_registry_path=development_registry,
        )
        outcome_qc_policy_document = validate_outcome_qc_policy(
            outcome_qc_policy_path,
            root=root,
            protocol_path=protocol_path,
        )
    except (InferenceGateError, OutcomeQCGateError) as exc:
        raise OpeningContractError(
            "prelabel inference/QC gate or amendment is absent or stale"
        ) from exc
    registries = validate_registry_lock(
        root=root,
        protocol_info=protocol_info,
        development_registry=development_registry,
        external_registry=external_registry,
        external_lock=external_lock,
    )
    suite = validate_model_suite(
        model_suite,
        root=root,
        protocol_info=protocol_info,
        registries=registries,
    )
    replay_receipt_path = Path(development_replay_receipt).resolve()
    if root not in replay_receipt_path.parents or not replay_receipt_path.is_file():
        raise OpeningContractError(
            "development replay receipt escapes repository or is absent"
        )
    replay_entrypoint = _resolve_inside(
        root, "scripts/27_verify_development_replay.py"
    )
    with tempfile.TemporaryDirectory(
        prefix="thermoroute-route-a-development-replay-check-"
    ) as temporary_name:
        temporary_root = Path(temporary_name).resolve()
        replay_result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(replay_entrypoint),
                "--suite",
                str(Path(model_suite).resolve()),
                "--receipt",
                str(replay_receipt_path),
                "--check",
            ],
            cwd=root,
            env=_sanitized_child_environment(temporary_root=temporary_root),
            text=True,
            capture_output=True,
            check=False,
        )
    if replay_result.returncode:
        detail = (replay_result.stderr or replay_result.stdout).strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise OpeningContractError(
            "fresh isolated full development replay failed: " + detail
        )
    try:
        replay_receipt = validate_development_replay_receipt(
            replay_receipt_path,
            root=root,
            suite_path=model_suite,
        )
    except ModelSuiteError as exc:
        raise OpeningContractError(
            "full isolated development model replay is absent or stale"
        ) from exc
    inputs = validate_prelabel_inputs(
        input_manifest,
        root=root,
        protocol_info=protocol_info,
        registries=registries,
        suite=suite,
    )
    chronology_receipt_path = Path(prelabel_chronology_receipt)
    if not chronology_receipt_path.is_absolute():
        chronology_receipt_path = root / chronology_receipt_path
    chronology_receipt_path = chronology_receipt_path.resolve()
    chronology = _validate_prelabel_chronology_for_opening(
        chronology_receipt_path,
        root=root,
        protocol_info=protocol_info,
        registries=registries,
        model_suite=model_suite,
        development_replay_receipt=replay_receipt_path,
        external_registry=external_registry,
        external_lock=external_lock,
        input_manifest=input_manifest,
    )
    frozen_source_inventory = source_inventory(root)
    source_tree_sha256 = sha256_json(frozen_source_inventory)
    environment = _route_a_environment_contract(root)
    if environment["runtime_sha256"] != suite["runtime_sha256"]:
        raise OpeningContractError(
            "authorization runtime differs from every frozen model bundle"
        )
    fixed_code = _fixed_code_identity(root)
    state_paths = _canonical_state_paths(
        root,
        protocol_sha256=protocol_info["protocol_sha256"],
        source_tree_sha256=source_tree_sha256,
        model_suite_sha256=suite["sha256"],
        prelabel_inputs_sha256=inputs["sha256"],
        prelabel_chronology_sha256=sha256_file(chronology_receipt_path),
        inference_gate_sha256=sha256_file(inference_gate_path),
        inference_amendment_seal_sha256=sha256_file(
            inference_amendment_seal_path
        ),
        outcome_qc_policy_sha256=sha256_file(outcome_qc_policy_path),
    )
    stable = {
        "format": AUTHORIZATION_FORMAT,
        "status": "AUTHORIZED_LABELS_STILL_SEALED",
        "protocol": {
            "path": _relative(root, protocol_path),
            "sha256": protocol_info["protocol_sha256"],
            "seal": _binding(root, protocol_info["seal"]["path"]),
            "final_prelabel_commit": protocol_info["seal"]["final_commit"],
            "authoritative_commit": protocol_info["authoritative_commit"],
            "authoritative_markdown_sha256": protocol_info[
                "authoritative_markdown_sha256"
            ],
            "pre_label_amendments_sha256": protocol_info["amendments_sha256"],
        },
        "registries": {
            "development": _binding(root, development_registry),
            "external": _binding(root, external_registry),
            "external_lock": _binding(root, external_lock),
            "development_panel_spec": _binding(
                root, registries["development_panel_spec"]
            ),
            "candidate_table": _binding(root, registries["candidate_table"]),
            "candidate_provenance": _binding(
                root, registries["candidate_provenance"]
            ),
            "candidate_snapshot_index": _binding(
                root, registries["candidate_snapshot_index"]
            ),
        },
        "model_suite": _binding(root, model_suite),
        "development_replay": {
            **_binding(root, replay_receipt_path),
            "format": replay_receipt["format"],
            "status": replay_receipt["status"],
        },
        "prelabel_chronology": {
            **_binding(root, chronology_receipt_path),
            "format": chronology["format"],
            "status": chronology["status"],
            "order": dict(chronology["order"]),
            "evidence_scope": chronology["evidence_scope"],
        },
        "inference_amendment": {
            **_binding(root, inference_amendment_path),
            "format": amendment["format"],
            "amendment_id": amendment["amendment_id"],
            "seal": _binding(root, inference_amendment_seal_path),
            "final_prelabel_commit": amendment_seal["final_prelabel_commit"],
        },
        "inference_gate": {
            **_binding(root, inference_gate_path),
            "format": inference_gate_document["format"],
            "status": inference_gate_document["status"],
            "claim_eligible": inference_gate_document["claim_eligible"],
            "analysis_mode": inference_gate_document["analysis_mode"],
            "policy_sha256": inference_gate_document["policy_sha256"],
        },
        "outcome_qc_policy": {
            **_binding(root, outcome_qc_policy_path),
            "format": outcome_qc_policy_document["format"],
            "policy_id": outcome_qc_policy_document["policy_id"],
            "required": True,
        },
        "actual_inputs": _binding(root, input_manifest),
        "actual_feature_order": list(suite["feature_order"]),
        "required_models": {
            cohort: list(models)
            for cohort, models in suite["required_models"].items()
        },
        "statistics_contract_sha256": sha256_json(
            protocol_info["document"]["primary_inference_contract"]
        ),
        "runtime": environment,
        "fixed_code": fixed_code,
        "source": {
            "git_commit_before_authorization": state["commit"],
            "source_tree_sha256": source_tree_sha256,
            "source_inventory": frozen_source_inventory,
            "git_clean_before_authorization": True,
            "post_freeze_allowed_git_status": f"?? {authorization_relative}",
            "authorization_path": authorization_relative,
        },
        "acquisition_plan": {
            "history_start": inputs["history_start"],
            "target_start": protocol_info["target_start"],
            "target_end": protocol_info["target_end"],
            "nwis_parameter_codes": ["00010", "00060", "00065"],
            "nwis_statistic_code": "00003",
            "request_partition": "one frozen site_no for the complete interval",
            "no_outcome_based_site_replacement": True,
            "provider": "usgs-nwis-confirmatory-dv",
            "canonical_endpoint": "https://waterservices.usgs.gov/nwis/dv/",
            "transport": "LIVE_HTTPS_ONLY_NO_PRESEEDED_OUTCOMES",
            "maximum_response_bytes_per_request": (
                MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
            ),
        },
        "state_paths": state_paths,
    }
    opening_id = sha256_json(stable)[:24]
    document = {
        **stable,
        "opening_id": opening_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    document["authorization_self_sha256"] = sha256_json(document)
    _assert_prepublication_source_snapshot(
        root,
        initial_git_state=state,
        frozen_source_inventory=frozen_source_inventory,
    )
    final_chronology = _validate_prelabel_chronology_for_opening(
        chronology_receipt_path,
        root=root,
        protocol_info=protocol_info,
        registries=registries,
        model_suite=model_suite,
        development_replay_receipt=replay_receipt_path,
        external_registry=external_registry,
        external_lock=external_lock,
        input_manifest=input_manifest,
    )
    if final_chronology != chronology:
        raise OpeningContractError(
            "prelabel chronology changed during opening-authorization preflight"
        )
    exclusive_create_json(destination, document)
    _require_only_untracked_authorization(root, destination)
    return document


def validate_authorization(
    authorization_path: str | Path,
    *,
    root: str | Path,
    require_clean_source: bool = True,
    allow_gitless_archive: bool = False,
) -> dict[str, Any]:
    """Replay every pre-label dependency without touching outcome artifacts."""
    root = Path(root).resolve()
    authorization_path = Path(authorization_path).resolve()
    if (root / ".git").exists():
        _assert_safe_live_git_repository(root)
    elif not allow_gitless_archive:
        raise OpeningContractError(
            "opening authorization requires safe live Git outside release mode"
        )
    authorization = _load_json(authorization_path, label="opening authorization")
    if authorization.get("format") != AUTHORIZATION_FORMAT:
        raise OpeningContractError("unsupported opening-authorization format")
    if authorization.get("status") != "AUTHORIZED_LABELS_STILL_SEALED":
        raise OpeningContractError("authorization is not in the sealed-label state")
    self_hashed = dict(authorization)
    self_digest = self_hashed.pop("authorization_self_sha256", None)
    if not _is_sha256(self_digest) or sha256_json(self_hashed) != self_digest:
        raise OpeningContractError("opening authorization self-hash is inconsistent")
    protocol_binding = authorization.get("protocol")
    if not isinstance(protocol_binding, Mapping):
        raise OpeningContractError("authorization lacks protocol binding")
    protocol_path = _resolve_inside(root, protocol_binding.get("path"))
    if sha256_file(protocol_path) != protocol_binding.get("sha256"):
        raise OpeningContractError("authorized protocol checksum changed")
    seal_binding = protocol_binding.get("seal")
    if not isinstance(seal_binding, Mapping):
        raise OpeningContractError("authorization lacks final protocol seal binding")
    seal_path = _verify_file_binding(
        root, seal_binding, label="final prelabel protocol seal"
    )
    protocol_info = validate_protocol(
        protocol_path,
        root=root,
        allow_gitless_archive=allow_gitless_archive,
        frozen_authoritative_markdown_sha256=str(
            protocol_binding.get("authoritative_markdown_sha256", "")
        ),
        protocol_seal_path=seal_path,
        frozen_protocol_seal_sha256=str(seal_binding.get("sha256", "")),
    )
    protocol_expected = {
        "sha256": protocol_info["protocol_sha256"],
        "authoritative_commit": protocol_info["authoritative_commit"],
        "authoritative_markdown_sha256": protocol_info["authoritative_markdown_sha256"],
        "pre_label_amendments_sha256": protocol_info["amendments_sha256"],
        "final_prelabel_commit": protocol_info["seal"]["final_commit"],
    }
    for key, value in protocol_expected.items():
        if protocol_binding.get(key) != value:
            raise OpeningContractError(f"authorized protocol {key} changed")
    amendment_binding = authorization.get("inference_amendment")
    if not isinstance(amendment_binding, Mapping) or set(amendment_binding) != {
        "path", "sha256", "format", "amendment_id", "seal",
        "final_prelabel_commit",
    }:
        raise OpeningContractError("authorization lacks inference-amendment binding")
    amendment_path = _verify_file_binding(
        root, amendment_binding, label="inference amendment"
    )
    amendment_seal_binding = amendment_binding.get("seal")
    if not isinstance(amendment_seal_binding, Mapping):
        raise OpeningContractError("authorization lacks inference-amendment seal")
    amendment_seal_path = _verify_file_binding(
        root, amendment_seal_binding, label="inference amendment seal"
    )
    try:
        amendment = validate_inference_amendment(
            amendment_path,
            root=root,
            protocol_path=protocol_path,
            protocol_seal_path=seal_path,
        )
        amendment_seal = validate_inference_amendment_seal(
            amendment_seal_path,
            root=root,
            amendment_path=amendment_path,
            allow_gitless_archive=allow_gitless_archive,
        )
    except InferenceGateError as exc:
        raise OpeningContractError("authorized inference amendment is stale") from exc
    expected_amendment_binding = {
        **_binding(root, amendment_path),
        "format": amendment["format"],
        "amendment_id": amendment["amendment_id"],
        "seal": _binding(root, amendment_seal_path),
        "final_prelabel_commit": amendment_seal["final_prelabel_commit"],
    }
    if dict(amendment_binding) != expected_amendment_binding:
        raise OpeningContractError("authorized inference amendment binding changed")
    outcome_qc_policy_binding = authorization.get("outcome_qc_policy")
    if not isinstance(outcome_qc_policy_binding, Mapping) or set(
        outcome_qc_policy_binding
    ) != {"path", "sha256", "format", "policy_id", "required"}:
        raise OpeningContractError("authorization lacks outcome-QC policy binding")
    outcome_qc_policy_path = _verify_file_binding(
        root, outcome_qc_policy_binding, label="outcome-QC policy"
    )
    try:
        outcome_qc_policy_document = validate_outcome_qc_policy(
            outcome_qc_policy_path,
            root=root,
            protocol_path=protocol_path,
        )
    except OutcomeQCGateError as exc:
        raise OpeningContractError("authorized outcome-QC policy is stale") from exc
    expected_outcome_qc_policy_binding = {
        **_binding(root, outcome_qc_policy_path),
        "format": outcome_qc_policy_document["format"],
        "policy_id": outcome_qc_policy_document["policy_id"],
        "required": True,
    }
    amendment_qc = amendment.get("additional_preopen_gates", {}).get(
        "outcome_qc_policy", {}
    )
    if (
        dict(outcome_qc_policy_binding) != expected_outcome_qc_policy_binding
        or not isinstance(amendment_qc, Mapping)
        or {
            "path": amendment_qc.get("path"),
            "sha256": amendment_qc.get("sha256"),
        } != _binding(root, outcome_qc_policy_path)
    ):
        raise OpeningContractError(
            "authorized outcome-QC policy differs from the sealed amendment"
        )
    bindings = authorization.get("registries")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "development", "external", "external_lock", "development_panel_spec",
        "candidate_table", "candidate_provenance", "candidate_snapshot_index",
    }:
        raise OpeningContractError("authorization lacks registry bindings")
    development_path = _verify_file_binding(
        root, bindings.get("development", {}), label="development registry"
    )
    external_path = _verify_file_binding(
        root, bindings.get("external", {}), label="external registry"
    )
    lock_path = _verify_file_binding(
        root, bindings.get("external_lock", {}), label="external registry lock"
    )
    registries = validate_registry_lock(
        root=root,
        protocol_info=protocol_info,
        development_registry=development_path,
        external_registry=external_path,
        external_lock=lock_path,
    )
    gate_binding = authorization.get("inference_gate")
    if not isinstance(gate_binding, Mapping) or set(gate_binding) != {
        "path", "sha256", "format", "status", "claim_eligible",
        "analysis_mode", "policy_sha256",
    }:
        raise OpeningContractError("authorization lacks inference-gate binding")
    gate_path = _verify_file_binding(root, gate_binding, label="inference gate")
    try:
        gate = validate_inference_gate_document(
            gate_path,
            root=root,
            protocol_path=protocol_path,
            protocol_seal_path=seal_path,
            station_registry_path=development_path,
        )
    except InferenceGateError as exc:
        raise OpeningContractError("authorized inference gate is stale") from exc
    expected_gate_binding = {
        **_binding(root, gate_path),
        "format": gate["format"],
        "status": gate["status"],
        "claim_eligible": gate["claim_eligible"],
        "analysis_mode": gate["analysis_mode"],
        "policy_sha256": gate["policy_sha256"],
    }
    if dict(gate_binding) != expected_gate_binding:
        raise OpeningContractError("authorized inference-gate binding changed")
    if gate["claim_eligible"] is not False:
        raise OpeningContractError("current Route-A gate did not fail closed")
    for key in (
        "development_panel_spec", "candidate_table", "candidate_provenance",
        "candidate_snapshot_index",
    ):
        bound = _verify_file_binding(root, bindings[key], label=f"authorized {key}")
        if bound != Path(registries[key]).resolve():
            raise OpeningContractError(f"authorized {key} differs from external lock")
    suite_path = _verify_file_binding(
        root, authorization.get("model_suite", {}), label="model-suite registry"
    )
    suite = validate_model_suite(
        suite_path, root=root, protocol_info=protocol_info, registries=registries
    )
    replay_binding = authorization.get("development_replay")
    if not isinstance(replay_binding, Mapping) or set(replay_binding) != {
        "path", "sha256", "format", "status"
    }:
        raise OpeningContractError("authorization lacks development-replay binding")
    replay_path = _verify_file_binding(
        root, replay_binding, label="development replay receipt"
    )
    try:
        replay_receipt = validate_development_replay_receipt(
            replay_path,
            root=root,
            suite_path=suite_path,
        )
    except ModelSuiteError as exc:
        raise OpeningContractError(
            "authorized full development model replay is stale"
        ) from exc
    if (
        replay_binding.get("format") != replay_receipt.get("format")
        or replay_binding.get("status") != replay_receipt.get("status")
    ):
        raise OpeningContractError("development replay receipt status changed")
    chronology_binding = authorization.get("prelabel_chronology")
    if not isinstance(chronology_binding, Mapping) or set(chronology_binding) != {
        "path", "sha256", "format", "status", "order", "evidence_scope"
    }:
        raise OpeningContractError("authorization lacks prelabel-chronology binding")
    chronology_path = _verify_file_binding(
        root, chronology_binding, label="prelabel chronology receipt"
    )
    input_path = _verify_file_binding(
        root, authorization.get("actual_inputs", {}), label="actual-input manifest"
    )
    inputs = validate_prelabel_inputs(
        input_path,
        root=root,
        protocol_info=protocol_info,
        registries=registries,
        suite=suite,
    )
    chronology = _validate_prelabel_chronology_for_opening(
        chronology_path,
        root=root,
        protocol_info=protocol_info,
        registries=registries,
        model_suite=suite_path,
        development_replay_receipt=replay_path,
        external_registry=external_path,
        external_lock=lock_path,
        input_manifest=input_path,
        allow_gitless_archive=allow_gitless_archive,
    )
    expected_chronology_binding = {
        **_binding(root, chronology_path),
        "format": chronology["format"],
        "status": chronology["status"],
        "order": dict(chronology["order"]),
        "evidence_scope": chronology["evidence_scope"],
    }
    if dict(chronology_binding) != expected_chronology_binding:
        raise OpeningContractError("authorized prelabel chronology changed")
    if list(suite["feature_order"]) != authorization.get("actual_feature_order"):
        raise OpeningContractError("authorization actual feature schema changed")
    expected_model_registry = {
        cohort: list(models)
        for cohort, models in suite["required_models"].items()
    }
    if expected_model_registry != authorization.get("required_models"):
        raise OpeningContractError("authorization model registry changed")
    if authorization.get("statistics_contract_sha256") != sha256_json(
        protocol_info["document"]["primary_inference_contract"]
    ):
        raise OpeningContractError("confirmatory statistics contract changed")
    environment = _route_a_environment_contract(root)
    if allow_gitless_archive:
        _validate_portable_runtime_identity(
            authorization.get("runtime"), environment
        )
    elif authorization.get("runtime") != environment:
        raise OpeningContractError("authorized numerical runtime/environment changed")
    if suite.get("runtime_sha256") != environment["runtime_sha256"]:
        raise OpeningContractError("model suite runtime differs from authorization runtime")
    fixed_code = _fixed_code_identity(root)
    if allow_gitless_archive:
        _validate_portable_fixed_code_identity(
            authorization.get("fixed_code"), fixed_code
        )
    elif authorization.get("fixed_code") != fixed_code:
        raise OpeningContractError("fixed Route-A executable/module identity changed")
    source = authorization.get("source", {})
    frozen_inventory = source.get("source_inventory")
    current_inventory = source_inventory(root)
    if (
        not isinstance(frozen_inventory, Mapping)
        or dict(frozen_inventory) != current_inventory
        or sha256_json(current_inventory) != source.get("source_tree_sha256")
    ):
        raise OpeningContractError("source tree differs from frozen opening authorization")
    current = _live_git_state(root) if (root / ".git").exists() else {
        "available": False, "commit": None, "dirty": None
    }
    if current.get("available"):
        if current.get("commit") != source.get("git_commit_before_authorization"):
            if require_clean_source or not _is_document_only_postopening_descendant(
                root,
                authorization,
                compute_commit=source.get("git_commit_before_authorization"),
                current_commit=current.get("commit"),
            ):
                raise OpeningContractError(
                    "Git commit differs from opening authorization"
                )
    elif not allow_gitless_archive:
        raise OpeningContractError("Git provenance is unavailable outside release mode")
    if require_clean_source:
        if allow_gitless_archive:
            raise OpeningContractError("gitless release replay cannot claim clean Git state")
        if source.get("git_clean_before_authorization") is not True:
            raise OpeningContractError("authorization does not attest a clean pre-freeze tree")
        if source.get("authorization_path") != _relative(root, authorization_path):
            raise OpeningContractError("authorization source policy names another path")
        if source.get("post_freeze_allowed_git_status") != (
            f"?? {_relative(root, authorization_path)}"
        ):
            raise OpeningContractError("authorization post-freeze Git policy changed")
        _require_only_untracked_authorization(root, authorization_path)
    plan = authorization.get("acquisition_plan", {})
    expected_plan = {
        "history_start": inputs["history_start"],
        "target_start": protocol_info["target_start"],
        "target_end": protocol_info["target_end"],
        "nwis_parameter_codes": ["00010", "00060", "00065"],
        "nwis_statistic_code": "00003",
        "request_partition": "one frozen site_no for the complete interval",
        "no_outcome_based_site_replacement": True,
        "provider": "usgs-nwis-confirmatory-dv",
        "canonical_endpoint": "https://waterservices.usgs.gov/nwis/dv/",
        "transport": "LIVE_HTTPS_ONLY_NO_PRESEEDED_OUTCOMES",
        "maximum_response_bytes_per_request": (
            MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
    }
    if plan != expected_plan:
        raise OpeningContractError("authorized acquisition plan changed")
    state_paths = authorization.get("state_paths")
    expected_state_paths = _canonical_state_paths(
        root,
        protocol_sha256=protocol_info["protocol_sha256"],
        source_tree_sha256=str(source.get("source_tree_sha256", "")),
        model_suite_sha256=suite["sha256"],
        prelabel_inputs_sha256=inputs["sha256"],
        prelabel_chronology_sha256=str(chronology_binding.get("sha256", "")),
        inference_gate_sha256=str(gate_binding.get("sha256", "")),
        inference_amendment_seal_sha256=str(
            amendment_seal_binding.get("sha256", "")
        ),
        outcome_qc_policy_sha256=str(
            outcome_qc_policy_binding.get("sha256", "")
        ),
    )
    if not isinstance(state_paths, Mapping) or dict(state_paths) != expected_state_paths:
        raise OpeningContractError("authorization lacks opening state paths")
    resolved_state_paths = _secure_canonical_state_paths(
        root,
        expected_state_paths,
    )
    intent = resolved_state_paths["intent"]
    receipt = resolved_state_paths["receipt"]
    stable = dict(authorization)
    stable.pop("opening_id", None)
    stable.pop("created_at_utc", None)
    stable.pop("authorization_self_sha256", None)
    if authorization.get("opening_id") != sha256_json(stable)[:24]:
        raise OpeningContractError("opening authorization identity is inconsistent")
    return {
        "authorization": authorization,
        "authorization_sha256": sha256_file(authorization_path),
        "protocol": protocol_info,
        "registries": registries,
        "suite": suite,
        "development_replay": replay_receipt,
        "prelabel_chronology": chronology,
        "inference_amendment": amendment,
        "inference_amendment_seal": amendment_seal,
        "inference_gate": gate,
        "outcome_qc_policy": outcome_qc_policy_document,
        "inputs": inputs,
        "intent_path": intent,
        "receipt_path": receipt,
        "state_paths": resolved_state_paths,
        "runtime": (
            authorization["runtime"] if allow_gitless_archive else environment
        ),
        "executing_runtime": environment,
        "fixed_code": (
            authorization["fixed_code"] if allow_gitless_archive else fixed_code
        ),
        "executing_fixed_code": fixed_code,
    }


def opening_status(*, intent_path: str | Path, receipt_path: str | Path) -> str:
    # A dangling symlink or other non-file node at either irreversible name is
    # still state.  Treating it as absent could authorize a second opening.
    intent = os.path.lexists(intent_path)
    receipt = os.path.lexists(receipt_path)
    if receipt and not intent:
        return "CORRUPT_RECEIPT_WITHOUT_INTENT"
    if receipt:
        return "OPENED_AND_SCORED_ONCE"
    if intent:
        return "OPENING_INCOMPLETE_SAME_OPENING_RESUME_REQUIRES_VALIDATION"
    return "SEALED_READY_OR_NOT_AUTHORIZED"


def inspect_same_opening_transport_resume(
    authorization_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Classify every same-opening recovery checkpoint without changing it."""
    root = Path(root).resolve()
    authorization_path = Path(authorization_path).resolve()
    preflight = validate_authorization(
        authorization_path,
        root=root,
        require_clean_source=False,
    )
    state = preflight["state_paths"]
    base_status = opening_status(
        intent_path=state["intent"], receipt_path=state["receipt"]
    )

    def result(
        status: str,
        *,
        phase: str,
        raw: bool = False,
        acquisition: bool = False,
        trusted: bool = False,
        transport: Mapping[str, Any] | None = None,
        forbidden: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "status": status,
            "resume_phase": phase,
            "raw_transport_resume_allowed": raw,
            "network_free_acquisition_finalization_allowed": acquisition,
            "network_free_trusted_completion_allowed": trusted,
            "transport": None if transport is None else dict(transport),
            "forbidden_existing_outputs": list(forbidden),
        }

    work_order = _expected_acquisition_work_order(preflight, root=root)
    if base_status == "SEALED_READY_OR_NOT_AUTHORIZED":
        run_directory = Path(state["run_directory"])
        preintent_existing = sorted(
            key
            for key, path in state.items()
            if key not in {"namespace", "run_directory"}
            and os.path.lexists(path)
        )
        if preintent_existing:
            return result(
                "OPENING_INDETERMINATE_STATE_WITHOUT_INTENT_NO_RESUME",
                phase="FAIL_CLOSED",
                forbidden=preintent_existing,
            )
        if os.path.lexists(run_directory):
            try:
                recovery, _document = _inspect_or_recover_preintent_temp(
                    state=state,
                    preflight=preflight,
                    root=root,
                    work_order=work_order,
                    publish_or_remove=False,
                )
            except OpeningContractError:
                return result(
                    "OPENING_INDETERMINATE_UNSAFE_STATE_WITHOUT_INTENT_"
                    "NO_RESUME",
                    phase="FAIL_CLOSED",
                )
            return result(
                "SEALED_READY_PRE_INTENT_ATOMIC_RECOVERY_ON_EXECUTE",
                phase={
                    "COMPLETE_VALID": (
                        "PRE_INTENT_COMPLETE_PUBLICATION_ON_EXECUTE"
                    ),
                    "PARTIAL_SAFE": (
                        "PRE_INTENT_PARTIAL_CLEANUP_ON_EXECUTE"
                    ),
                    "EMPTY_SAFE": (
                        "PRE_INTENT_EMPTY_DIRECTORY_RECOVERY_ON_EXECUTE"
                    ),
                }[recovery],
            )
        return result(base_status, phase="TERMINAL_NOT_STARTED")
    if base_status == "CORRUPT_RECEIPT_WITHOUT_INTENT":
        return result(base_status, phase="FAIL_CLOSED")
    try:
        validated_intent = _validated_intent(
            preflight=preflight, root=root, work_order=work_order
        )
    except OpeningContractError:
        return result(
            "OPENING_INDETERMINATE_INVALID_INTENT_NO_RESUME",
            phase="FAIL_CLOSED",
        )
    work_order_path = Path(state["work_order"])
    if not os.path.lexists(work_order_path):
        run_directory = Path(state["run_directory"])
        try:
            with _secure_directory_chain(
                run_directory, create=False
            ) as run_descriptor:
                safe_intent_temps = _cleanup_atomic_create_temps(
                    run_descriptor,
                    final_name=Path(state["intent"]).name,
                    expected_payload=canonical_json_bytes(validated_intent),
                    remove=False,
                )
                safe_work_order_temps = _cleanup_atomic_create_temps(
                    run_descriptor,
                    final_name=work_order_path.name,
                    expected_payload=canonical_json_bytes(work_order),
                    remove=False,
                )
                actual = set(os.listdir(run_descriptor))
        except OpeningContractError:
            return result(
                "OPENING_INDETERMINATE_UNSAFE_STATE_WITHOUT_WORK_ORDER_"
                "NO_RESUME",
                phase="FAIL_CLOSED",
            )
        allowed = {
            Path(state["intent"]).name,
            *safe_intent_temps,
            *safe_work_order_temps,
        }
        unexpected = sorted(actual - allowed)
        if unexpected:
            return result(
                "OPENING_INDETERMINATE_STATE_WITHOUT_WORK_ORDER_NO_RESUME",
                phase="FAIL_CLOSED",
                forbidden=unexpected,
            )
        return result(
            "OPENING_INCOMPLETE_SAME_OPENING_RAW_TRANSPORT_VALIDATED",
            phase="RAW_TRANSPORT",
            raw=True,
            transport={
                "classification": "RESUMABLE_BEFORE_WORK_ORDER_PUBLICATION",
                "completed_request_count": 0,
                "missing_request_count": sum(
                    len(work_order["site_registries"][cohort]["sites"])
                    for cohort in ("temporal", "external")
                ),
                "recoverable_pending_request_count": 0,
                "refetchable_nondurable_response_count": 0,
                "attempt_count": 0,
            },
        )
    if (
        not work_order_path.is_file()
        or work_order_path.is_symlink()
        or _load_json(work_order_path, label="acquisition work order")
        != work_order
    ):
        return result(
            "OPENING_INDETERMINATE_CHANGED_WORK_ORDER_NO_RESUME",
            phase="FAIL_CLOSED",
        )
    try:
        _validate_atomic_final_file(
            work_order_path,
            canonical_json_bytes(work_order),
            cleanup_temps=False,
        )
    except OpeningContractError:
        return result(
            "OPENING_INDETERMINATE_CHANGED_WORK_ORDER_NO_RESUME",
            phase="FAIL_CLOSED",
        )
    try:
        trusted_stage_count = _handle_abandoned_trusted_stages(
            state, remove=False
        )
    except OpeningContractError:
        return result(
            "OPENING_INDETERMINATE_UNSAFE_TRUSTED_STAGE_NO_RESUME",
            phase="FAIL_CLOSED",
        )
    from .outcome_acquisition import (  # local import avoids acquisition coupling
        OutcomeAcquisitionError,
        _acquisition_directory_mode,
        _assert_exact_acquisition_directory,
        _validate_abandoned_acquisition_stages,
        inspect_transport_resume_state,
    )

    try:
        _validate_abandoned_acquisition_stages(
            {key: Path(value) for key, value in state.items()}
        )
    except OutcomeAcquisitionError:
        return result(
            "OPENING_INDETERMINATE_UNSAFE_ACQUISITION_STAGE_NO_RESUME",
            phase="FAIL_CLOSED",
        )

    manifest = Path(state["acquisition_manifest"])
    acquisition_directory = manifest.parent
    trusted_directory = _trusted_directory_from_state(state)
    receipt = Path(state["receipt"])
    sidecar = Path(state["receipt_sha256"])
    manifest_exists = os.path.lexists(manifest)
    acquisition_exists = os.path.lexists(acquisition_directory)
    trusted_exists = os.path.lexists(trusted_directory)
    receipt_exists = os.path.lexists(receipt)
    sidecar_exists = os.path.lexists(sidecar)

    if manifest_exists:
        acquisition_permission_recovery = False
        try:
            _assert_exact_acquisition_directory(
                acquisition_directory,
                {key: Path(value) for key, value in state.items()},
                allow_recoverable_canonical_mode=True,
            )
            acquisition_permission_recovery = (
                _acquisition_directory_mode(
                    {key: Path(value) for key, value in state.items()}
                )
                == 0o700
            )
        except OutcomeAcquisitionError:
            return result(
                "OPENING_INDETERMINATE_INVALID_ACQUISITION_BUNDLE_NO_RESUME",
                phase="FAIL_CLOSED",
            )
        trusted_permission_recovery = False
        if trusted_exists:
            try:
                _assert_exact_trusted_directory(
                    trusted_directory,
                    state,
                    allow_recoverable_canonical_mode=True,
                )
                trusted_permission_recovery = (
                    _trusted_directory_mode(state) == 0o700
                )
            except OpeningContractError:
                return result(
                    "OPENING_INDETERMINATE_INVALID_TRUSTED_DIRECTORY_NO_RESUME",
                    phase="FAIL_CLOSED",
                )
        if acquisition_permission_recovery:
            return result(
                "OPENING_INCOMPLETE_ACQUISITION_PERMISSION_RECOVERY_"
                "REQUIRES_FULL_REPLAY",
                phase="ACQUISITION_PERMISSION_RECOVERY_BY_FULL_REPLAY",
                trusted=True,
            )
        if trusted_permission_recovery:
            return result(
                "OPENING_INCOMPLETE_TRUSTED_PERMISSION_RECOVERY_"
                "REQUIRES_FULL_REPLAY",
                phase="TRUSTED_PERMISSION_RECOVERY_BY_FULL_REPLAY",
                trusted=True,
            )
        if trusted_stage_count and trusted_exists:
            return result(
                "OPENING_INCOMPLETE_TRUSTED_STAGE_CLEANUP_REQUIRES_"
                "FULL_VALIDATION",
                phase="TRUSTED_STAGE_CLEANUP_AFTER_FULL_VALIDATION",
                trusted=True,
            )
        if sidecar_exists and not receipt_exists:
            return result(
                "OPENING_INDETERMINATE_SIDECAR_WITHOUT_RECEIPT_NO_RESUME",
                phase="FAIL_CLOSED",
            )
        if receipt_exists:
            if not trusted_exists:
                return result(
                    "OPENING_INDETERMINATE_RECEIPT_WITHOUT_TRUSTED_NO_RESUME",
                    phase="FAIL_CLOSED",
                )
            try:
                _read_completed_receipt(
                    authorization_path=authorization_path,
                    root=root,
                    require_sidecar=sidecar_exists,
                )
            except OpeningContractError:
                return result(
                    "OPENING_INDETERMINATE_INVALID_RECEIPT_NO_RESUME",
                    phase="FAIL_CLOSED",
                )
            if sidecar_exists:
                return result(
                    "OPENED_AND_SCORED_ONCE",
                    phase="TERMINAL_COMPLETE",
                )
            return result(
                "OPENING_INCOMPLETE_SIDECAR_RECOVERY_VALIDATED",
                phase="SIDECAR_RECOVERY_AFTER_FULL_VALIDATION",
                trusted=True,
            )
        if trusted_exists:
            return result(
                "OPENING_INCOMPLETE_RECEIPT_COMPLETION_REQUIRES_FULL_REPLAY",
                phase="RECEIPT_COMPLETION_AFTER_FULL_REPLAY",
                trusted=True,
            )
        return result(
            "OPENING_INCOMPLETE_TRUSTED_RECOMPUTE_VALIDATED",
            phase="TRUSTED_RECOMPUTE_NETWORK_FREE",
            trusted=True,
        )

    forbidden_existing = sorted(
        key
        for key in RAW_ACQUISITION_FORBIDDEN_STATE_KEYS
        if os.path.lexists(state[key])
    )
    if acquisition_exists:
        forbidden_existing.append("acquisition_directory")
    if trusted_exists:
        forbidden_existing.append("trusted_directory")
    if forbidden_existing:
        return result(
            "OPENING_INDETERMINATE_DERIVED_OR_TRUSTED_OUTPUT_EXISTS_"
            "NO_RESUME",
            phase="FAIL_CLOSED",
            forbidden=sorted(set(forbidden_existing)),
        )
    if trusted_stage_count:
        return result(
            "OPENING_INDETERMINATE_TRUSTED_STAGE_WITHOUT_ACQUISITION_"
            "NO_RESUME",
            phase="FAIL_CLOSED",
        )

    try:
        transport = inspect_transport_resume_state(
            root=root,
            work_order_path=work_order_path,
            work_order=work_order,
            authorization=preflight["authorization"],
            state=state,
        )
    except OutcomeAcquisitionError:
        return result(
            "OPENING_INDETERMINATE_CORRUPT_OR_PARTIAL_TRANSACTION_NO_RESUME",
            phase="FAIL_CLOSED",
        )
    if transport.get("classification") == (
        "RESUMABLE_RAW_COMPLETE_DERIVATION_NOT_PUBLISHED"
    ):
        return result(
            "OPENING_INCOMPLETE_ACQUISITION_FINALIZATION_VALIDATED",
            phase="ACQUISITION_FINALIZATION_NETWORK_FREE",
            acquisition=True,
            transport=transport,
        )
    return result(
        "OPENING_INCOMPLETE_SAME_OPENING_RAW_TRANSPORT_VALIDATED",
        phase="RAW_TRANSPORT",
        raw=True,
        transport=transport,
    )


def _trusted_directory_from_state(state: Mapping[str, Any]) -> Path:
    """Return and strictly validate the single canonical trusted directory."""
    try:
        run_directory = Path(
            os.path.abspath(os.fspath(state["run_directory"]))
        )
        paths = {
            key: Path(os.path.abspath(os.fspath(state[key])))
            for key in _TRUSTED_STATE_KEYS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise OpeningContractError(
            "trusted state-path registry is incomplete"
        ) from exc
    parents = {path.parent for path in paths.values()}
    names = {path.name for path in paths.values()}
    canonical = run_directory / "trusted"
    if (
        parents != {canonical}
        or len(names) != len(_TRUSTED_STATE_KEYS)
        or any(path.parent != canonical for path in paths.values())
    ):
        raise OpeningContractError(
            "trusted artifacts do not share the canonical trusted directory"
        )
    return canonical


def _trusted_state_at_directory(
    state: Mapping[str, Any], directory: Path
) -> dict[str, Any]:
    """Map trusted output names to a private directory without changing names."""
    canonical = _trusted_directory_from_state(state)
    directory = Path(os.path.abspath(os.fspath(directory)))
    if directory.parent != canonical.parent:
        raise OpeningContractError(
            "trusted staging directory is not a same-filesystem sibling"
        )
    staged = dict(state)
    for key in _TRUSTED_STATE_KEYS:
        staged[key] = directory / Path(state[key]).name
    return staged


def _assert_trusted_parent_metadata(descriptor: int) -> os.stat_result:
    """Require a directory that no group/other process identity can mutate."""
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise OpeningContractError(
            "trusted publication parent is not owner-controlled"
        )
    return metadata


@contextmanager
def _exclusive_trusted_publication_lock(
    state: Mapping[str, Any],
) -> Iterator[None]:
    """Hold the process-scoped trusted publisher lock without following links."""
    canonical = _trusted_directory_from_state(state)
    run_directory = canonical.parent
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    with _secure_directory_chain(run_directory, create=False) as parent_descriptor:
        try:
            _assert_trusted_parent_metadata(parent_descriptor)
            descriptor = os.open(
                _TRUSTED_PUBLICATION_LOCK,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
            ):
                raise OpeningContractError(
                    "trusted publication lock has unsafe metadata"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OpeningAlreadyStarted(
                    "another trusted Route-A publisher holds the canonical lock"
                ) from exc
            os.fsync(parent_descriptor)
            yield
        except OSError as exc:
            raise OpeningContractError(
                "trusted publication lock path is unsafe"
            ) from exc
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)


def _new_trusted_stage_directory(state: Mapping[str, Any]) -> Path:
    """Create one private same-filesystem staging directory by directory fd."""
    canonical = _trusted_directory_from_state(state)
    run_directory = canonical.parent
    with _secure_directory_chain(run_directory, create=False) as parent_descriptor:
        parent_stat = _assert_trusted_parent_metadata(parent_descriptor)
        for _attempt in range(128):
            name = f"{_TRUSTED_STAGE_PREFIX}{secrets.token_hex(16)}"
            try:
                os.mkdir(name, 0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_dev != parent_stat.st_dev
                    or metadata.st_uid != os.geteuid()
                ):
                    raise OpeningContractError(
                        "trusted staging directory has unsafe metadata"
                    )
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
            finally:
                os.close(descriptor)
            return run_directory / name
    raise OpeningContractError("cannot allocate a trusted staging directory")


def _remove_safe_abandoned_trusted_stage(
    *,
    parent_descriptor: int,
    name: str,
    parent_metadata: os.stat_result,
    remove: bool,
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_dev != parent_metadata.st_dev
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise OpeningContractError(
                "abandoned trusted stage has unsafe metadata"
            )
        entries = os.listdir(descriptor)
        children: dict[str, os.stat_result] = {}
        inode_counts: dict[tuple[int, int], int] = {}
        for entry in entries:
            child = os.open(
                entry,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                child_metadata = os.fstat(child)
            finally:
                os.close(child)
            if (
                not stat.S_ISREG(child_metadata.st_mode)
                or child_metadata.st_uid != os.geteuid()
                or child_metadata.st_dev != metadata.st_dev
                or child_metadata.st_mode & 0o222
            ):
                raise OpeningContractError(
                    "abandoned trusted stage contains an unsafe entry"
                )
            children[entry] = child_metadata
            inode = (child_metadata.st_dev, child_metadata.st_ino)
            inode_counts[inode] = inode_counts.get(inode, 0) + 1
        if any(
            child.st_nlink
            != inode_counts[(child.st_dev, child.st_ino)]
            for child in children.values()
        ):
            raise OpeningContractError(
                "abandoned trusted stage contains an external hard link"
            )
        if remove:
            for entry in entries:
                os.unlink(entry, dir_fd=descriptor)
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if remove:
        os.rmdir(name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def _handle_abandoned_trusted_stages(
    state: Mapping[str, Any], *, remove: bool
) -> int:
    canonical = _trusted_directory_from_state(state)
    run_directory = canonical.parent
    if not os.path.lexists(run_directory):
        return 0
    count = 0
    with _secure_directory_chain(
        run_directory, create=False
    ) as parent_descriptor:
        parent_metadata = _assert_trusted_parent_metadata(parent_descriptor)
        for name in sorted(os.listdir(parent_descriptor)):
            if not name.startswith(_TRUSTED_STAGE_PREFIX):
                continue
            if re.fullmatch(r"\.trusted-stage-v1-[0-9a-f]{32}", name) is None:
                raise OpeningContractError(
                    "trusted staging namespace contains a noncanonical entry"
                )
            _remove_safe_abandoned_trusted_stage(
                parent_descriptor=parent_descriptor,
                name=name,
                parent_metadata=parent_metadata,
                remove=remove,
            )
            count += 1
    return count


def _assert_exact_trusted_directory(
    directory: Path,
    state: Mapping[str, Any],
    *,
    allow_recoverable_canonical_mode: bool = False,
) -> None:
    """Reject missing, extra, linked, nested, or nonregular trusted artifacts."""
    canonical = _trusted_directory_from_state(state)
    directory = Path(os.path.abspath(os.fspath(directory)))
    if directory.parent != canonical.parent:
        raise OpeningContractError("trusted artifact directory is noncanonical")
    is_canonical = directory == canonical
    expected = {Path(state[key]).name for key in _TRUSTED_STATE_KEYS}
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        raise OpeningContractError(
            "trusted artifact directory is absent or unsafe"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        actual_mode = stat.S_IMODE(metadata.st_mode)
        allowed_modes = (
            {0o555, 0o700}
            if is_canonical and allow_recoverable_canonical_mode
            else {0o555}
            if is_canonical
            else {0o700}
        )
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or actual_mode not in allowed_modes
        ):
            raise OpeningContractError(
                "trusted artifact directory metadata is unsafe"
            )
        actual = set(os.listdir(descriptor))
        if actual != expected:
            raise OpeningContractError(
                "trusted artifact directory is incomplete or has extra entries"
            )
        for name in sorted(expected):
            item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(item.st_mode)
                or item.st_nlink != 1
                or item.st_uid != os.geteuid()
                or item.st_dev != metadata.st_dev
                or item.st_mode & 0o222
            ):
                raise OpeningContractError(
                    f"trusted artifact is linked or nonregular: {name}"
                )
    finally:
        os.close(descriptor)


def _trusted_directory_mode(state: Mapping[str, Any]) -> int:
    canonical = _trusted_directory_from_state(state)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(canonical, flags)
    try:
        return stat.S_IMODE(os.fstat(descriptor).st_mode)
    finally:
        os.close(descriptor)


def _harden_recoverable_trusted_directory(state: Mapping[str, Any]) -> None:
    """Finish only a fully replayed rename-before-chmod trusted publication."""
    canonical = _trusted_directory_from_state(state)
    _assert_exact_trusted_directory(
        canonical, state, allow_recoverable_canonical_mode=True
    )
    if _trusted_directory_mode(state) != 0o700:
        raise OpeningContractError(
            "trusted permission recovery requires exact mode 0700"
        )
    with _secure_directory_chain(
        canonical.parent, create=False
    ) as parent_descriptor:
        descriptor = os.open(
            canonical.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
        finally:
            os.close(descriptor)
    _assert_exact_trusted_directory(canonical, state)


def _atomic_publish_trusted_directory(
    stage_directory: Path, state: Mapping[str, Any]
) -> Path:
    """Publish a validated trusted directory with one same-parent rename."""
    canonical = _trusted_directory_from_state(state)
    stage = Path(os.path.abspath(os.fspath(stage_directory)))
    if stage.parent != canonical.parent or not stage.name.startswith(
        _TRUSTED_STAGE_PREFIX
    ):
        raise OpeningContractError("trusted staging path is noncanonical")
    if re.fullmatch(r"\.trusted-stage-v1-[0-9a-f]{32}", stage.name) is None:
        raise OpeningContractError("trusted staging name is noncanonical")
    _assert_exact_trusted_directory(stage, state)
    with _secure_directory_chain(canonical.parent, create=False) as parent_descriptor:
        _assert_trusted_parent_metadata(parent_descriptor)
        try:
            os.stat(canonical.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OpeningAlreadyStarted(
                "canonical trusted directory already exists and is immutable"
            )
        # This check-to-rename interval is protected by the process-held flock
        # and an owner-only-writable parent.  The documented honest-owner
        # boundary excludes that same owner deliberately bypassing the lock;
        # no other process identity can create the destination in this window.
        stage_descriptor = os.open(
            stage.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            os.fsync(stage_descriptor)
            _trusted_publication_fault("before_trusted_directory_rename")
            try:
                os.rename(
                    stage.name,
                    canonical.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
            except OSError as exc:
                raise OpeningContractError(
                    "atomic trusted-directory publication failed"
                ) from exc
            _trusted_publication_fault(
                "after_trusted_directory_rename_before_hardening"
            )
            # Darwin prohibits renaming a non-writable directory.  Harden the
            # already-complete directory through the still-open descriptor
            # immediately after the one atomic namespace publication.
            os.fchmod(stage_descriptor, 0o555)
            os.fsync(stage_descriptor)
            _trusted_publication_fault(
                "after_trusted_directory_hardening_before_parent_fsync"
            )
        finally:
            os.close(stage_descriptor)
        os.fsync(parent_descriptor)
    _assert_exact_trusted_directory(canonical, state)
    return canonical


def _trusted_publication_fault(_point: str) -> None:
    """No-op production hook monkeypatched only by synthetic crash tests."""


@dataclass(frozen=True)
class OpeningProducts:
    """Canonical artifacts emitted only by the isolated trusted scorer."""

    acquisition_manifest: Path
    availability_registry: Path
    outcome_quality_audit: Path
    outcome_qc_gate: Path
    approved_target_sensitivity: Path
    spatial_sensitivity: Path
    probabilistic_evaluation: Path
    temporal_predictions: Path
    external_predictions: Path
    statistics: Path
    report: Path


def _opening_products_from_state(state: Mapping[str, Any]) -> OpeningProducts:
    return OpeningProducts(
        acquisition_manifest=Path(state["acquisition_manifest"]),
        availability_registry=Path(state["availability_registry"]),
        outcome_quality_audit=Path(state["outcome_quality_audit"]),
        outcome_qc_gate=Path(state["outcome_qc_gate"]),
        approved_target_sensitivity=Path(state["approved_target_sensitivity"]),
        spatial_sensitivity=Path(state["spatial_sensitivity"]),
        probabilistic_evaluation=Path(state["probabilistic_evaluation"]),
        temporal_predictions=Path(state["temporal_predictions"]),
        external_predictions=Path(state["external_predictions"]),
        statistics=Path(state["statistics"]),
        report=Path(state["report"]),
    )

def _expected_acquisition_work_order(
    preflight: Mapping[str, Any], *, root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    authorization = preflight["authorization"]
    stable = {
        "format": ACQUISITION_WORK_ORDER_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_path": authorization["source"]["authorization_path"],
        "authorization_sha256": preflight["authorization_sha256"],
        "source_tree_sha256": authorization["source"]["source_tree_sha256"],
        "runtime_sha256": preflight["runtime"]["runtime_sha256"],
        "fixed_code_sha256": preflight["fixed_code"]["sha256"],
        "acquisition_plan": dict(authorization["acquisition_plan"]),
        "state_paths": dict(authorization["state_paths"]),
        "site_registries": {
            "temporal": {
                "sha256": preflight["registries"]["development_sha256"],
                "sites": sorted(
                    preflight["registries"]["development"].site_no.astype(str)
                ),
            },
            "external": {
                "sha256": preflight["registries"]["external_sha256"],
                "sites": sorted(
                    preflight["registries"]["external"].site_no.astype(str)
                ),
            },
        },
    }
    # Resolve every state path now, before any label-bearing child is launched.
    for key, relative in stable["state_paths"].items():
        if key == "namespace":
            continue
        resolved = (root / str(relative)).resolve()
        if root not in resolved.parents:
            raise OpeningContractError("acquisition work-order path escapes repository")
    return {**stable, "work_order_self_sha256": sha256_json(stable)}

def _preflight_attestation(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "authorization_sha256": preflight["authorization_sha256"],
        "opening_id": preflight["authorization"]["opening_id"],
        "protocol_sha256": preflight["protocol"]["protocol_sha256"],
        "development_registry_sha256": preflight["registries"]["development_sha256"],
        "external_registry_sha256": preflight["registries"]["external_sha256"],
        "external_lock_sha256": preflight["registries"]["lock_sha256"],
        "model_suite_sha256": preflight["suite"]["sha256"],
        "development_replay_sha256": preflight["authorization"]
        ["development_replay"]["sha256"],
        "prelabel_chronology_sha256": preflight["authorization"]
        ["prelabel_chronology"]["sha256"],
        "inference_amendment_sha256": preflight["authorization"]
        ["inference_amendment"]["sha256"],
        "inference_amendment_seal_sha256": preflight["authorization"]
        ["inference_amendment"]["seal"]["sha256"],
        "inference_gate_sha256": preflight["authorization"]
        ["inference_gate"]["sha256"],
        "inference_gate_status": preflight["inference_gate"]["status"],
        "inference_claim_eligible": preflight["inference_gate"]["claim_eligible"],
        "outcome_qc_policy_sha256": preflight["authorization"]
        ["outcome_qc_policy"]["sha256"],
        "prelabel_inputs_sha256": preflight["inputs"]["sha256"],
        "actual_feature_order": list(preflight["suite"]["feature_order"]),
        "required_models": {
            cohort: list(values)
            for cohort, values in preflight["suite"]["required_models"].items()
        },
        "source_tree_sha256": preflight["authorization"]["source"][
            "source_tree_sha256"
        ],
        "runtime_sha256": preflight["runtime"]["runtime_sha256"],
        "requirements_lock_sha256": preflight["runtime"]["requirements_lock"][
            "sha256"
        ],
        "hashed_requirements_lock_sha256": preflight["runtime"]
        ["hashed_requirements_lock"]["sha256"],
        "golden_inference_sha256": preflight["runtime"][
            "golden_inference_sha256"
        ],
        "fixed_code_sha256": preflight["fixed_code"]["sha256"],
        "state_namespace": preflight["authorization"]["state_paths"]["namespace"],
    }


def _trusted_validator_identity(root: Path) -> dict[str, Any]:
    paths = (
        "src/thermoroute/opening.py",
        "src/thermoroute/model_suite.py",
        "src/thermoroute/frozen_inference.py",
        "src/thermoroute/checkpoint.py",
        "src/thermoroute/datasets.py",
        "src/thermoroute/features.py",
        "src/thermoroute/usgs.py",
        "src/thermoroute/results.py",
        "src/thermoroute/significance.py",
    )
    files = {
        relative: sha256_file(_resolve_inside(root, relative)) for relative in paths
    }
    return {
        "implementation": "thermoroute.opening.trusted-validator.v1",
        "files": files,
        "sha256": sha256_json(files),
        "source_tree_sha256": source_tree_hash(root),
    }


def _sanitized_child_environment(*, temporary_root: Path) -> dict[str, str]:
    """Return the complete, non-inheriting environment for Route-A children."""
    environment = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": str(temporary_root.resolve()),
        **_DETERMINISTIC_ENVIRONMENT,
    }
    if any(
        (key.startswith("PYTHON") and key != "PYTHONHASHSEED")
        or key.startswith("DYLD")
        or key == "LD_PRELOAD"
        for key in environment
    ):
        raise OpeningContractError("sanitized child environment is injectable")
    return environment


def _run_fixed_isolated_child(
    *,
    root: Path,
    role: str,
    argument_path: str | Path,
    resume: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one frozen entrypoint; callers cannot supply a command or module."""
    argument_flags = {
        "orchestrator": "--authorization",
        "acquisition": "--work-order",
        "trusted_scorer": "--work-order",
    }
    if role not in argument_flags:
        raise OpeningContractError(f"unsupported fixed Route-A child role: {role}")
    if resume and role not in {"orchestrator", "acquisition"}:
        raise OpeningContractError(
            "same-opening transport resume is limited to fixed transport roles"
        )
    root = root.resolve()
    argument = Path(argument_path).resolve()
    if root not in argument.parents or not argument.is_file():
        raise OpeningContractError("fixed Route-A child argument escapes or is absent")
    identity = _fixed_code_identity(root)
    entry = identity["entrypoints"][role]
    script = Path(entry["realpath"]).resolve()
    if (
        script != (root / _FIXED_ENTRYPOINTS[role]).resolve()
        or sha256_file(script) != entry["sha256"]
    ):
        raise OpeningContractError(f"fixed Route-A {role} entrypoint changed")
    with tempfile.TemporaryDirectory(prefix=f"thermoroute-route-a-{role}-") as name:
        temporary_root = Path(name).resolve()
        pycache = temporary_root / "pycache"
        pycache.mkdir(mode=0o700)
        command = [
            sys.executable,
            "-I",
            "-X",
            f"pycache_prefix={pycache}",
            str(script),
            argument_flags[role],
            str(argument),
        ]
        if resume:
            command.append("--resume")
        result = subprocess.run(
            command,
            cwd=root,
            env=_sanitized_child_environment(temporary_root=temporary_root),
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise OpeningContractError(
            f"fixed isolated Route-A {role} child failed: {detail}"
        )
    return result


def _assert_isolated_role(
    *, preflight: Mapping[str, Any], root: Path, role: str
) -> None:
    """Bind the actually loaded interpreter, entrypoint and child policy."""
    if role not in _FIXED_ENTRYPOINTS or not sys.flags.isolated:
        raise OpeningContractError(f"Route-A {role} must run under python -I")
    entry = preflight["fixed_code"]["entrypoints"][role]
    actual_entry = Path(sys.argv[0]).resolve()
    expected_entry = (root / _FIXED_ENTRYPOINTS[role]).resolve()
    if (
        actual_entry != expected_entry
        or str(actual_entry) != entry.get("realpath")
        or sha256_file(actual_entry) != entry.get("sha256")
    ):
        raise OpeningContractError(f"loaded Route-A {role} entrypoint changed")
    executable = preflight["runtime"]["python_executable"]
    actual_executable = Path(sys.executable).resolve()
    if (
        str(actual_executable) != executable.get("realpath")
        or sha256_file(actual_executable) != executable.get("sha256")
    ):
        raise OpeningContractError("isolated child Python executable changed")
    expected_environment = _sanitized_child_environment(
        temporary_root=Path(os.environ.get("TMPDIR", ""))
    )
    if dict(os.environ) != expected_environment:
        added = sorted(set(os.environ) - set(expected_environment))
        missing = sorted(set(expected_environment) - set(os.environ))
        changed = sorted(
            key for key in set(os.environ) & set(expected_environment)
            if os.environ[key] != expected_environment[key]
        )
        raise OpeningContractError(
            "isolated child environment differs from allowlist "
            f"(added={added}, missing={missing}, changed={changed})"
        )


def _prediction_keys(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
    return set(frame[["site_id", "horizon", "issue_date", "target_date"]].itertuples(
        index=False, name=None
    ))


def validate_prediction_product(
    path: str | Path,
    *,
    required_models: Sequence[str],
    expected_sites: set[str],
    target_start: str,
    target_end: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    R.validate_predictions(frame)
    frame = frame.copy()
    frame["site_id"] = frame.site_id.astype(str)
    frame["issue_date"] = pd.to_datetime(frame.issue_date)
    frame["target_date"] = pd.to_datetime(frame.target_date)
    if set(frame.split.astype(str)) != {"confirm"}:
        raise OpeningContractError("confirmatory predictions contain another split")
    if set(frame.model.astype(str)) != set(required_models):
        raise OpeningContractError("confirmatory predictions omit or add a model")
    if not set(frame.site_id) or not set(frame.site_id) <= expected_sites:
        raise OpeningContractError("predictions contain no frozen sites or an unknown site")
    lower, upper = pd.Timestamp(target_start), pd.Timestamp(target_end)
    if (frame.issue_date < lower).any() or (frame.target_date > upper).any():
        raise OpeningContractError("prediction key leaves the frozen target interval")
    identity = ["model", "site_id", "horizon", "issue_date", "target_date"]
    if frame.duplicated(identity).any():
        raise OpeningContractError(
            "final confirmatory artifact must contain one ensemble row per model/key"
        )
    key_sets = {
        str(model): _prediction_keys(group)
        for model, group in frame.groupby("model", sort=False)
    }
    first = key_sets[str(required_models[0])]
    if not first or any(keys != first for keys in key_sets.values()):
        raise OpeningContractError("models do not share the exact forecast-key registry")
    spread = frame.groupby(["site_id", "horizon", "issue_date", "target_date"])[
        "y_true"
    ].agg(lambda values: float(np.max(values) - np.min(values)))
    if (spread > 1e-6).any():
        raise OpeningContractError("models disagree on confirmation labels")
    return frame


def _validate_availability_registry(
    path: Path,
    *,
    temporal_sites: set[str],
    external_sites: set[str],
    minimum_targets: int,
) -> None:
    frame = pd.read_csv(path, dtype={"site_no": "string"}, keep_default_na=False)
    required = {"cohort", "site_no", "horizon", "n_valid_targets", "reportable"}
    if set(frame) != required:
        raise OpeningContractError("availability registry has a non-frozen schema")
    frame = frame.copy()
    frame["site_no"] = frame.site_no.astype("string").str.strip()
    frame["horizon"] = pd.to_numeric(frame.horizon, errors="coerce")
    frame["n_valid_targets"] = pd.to_numeric(
        frame.n_valid_targets, errors="coerce"
    )
    if frame[["horizon", "n_valid_targets"]].isna().any().any():
        raise OpeningContractError("availability registry has invalid counts/horizons")
    expected = {
        (cohort, site, horizon)
        for cohort, sites in (("temporal", temporal_sites), ("external", external_sites))
        for site in sites
        for horizon in (1, 3, 7)
    }
    actual = set(frame[["cohort", "site_no", "horizon"]].itertuples(
        index=False, name=None
    ))
    if actual != expected or frame.duplicated(["cohort", "site_no", "horizon"]).any():
        raise OpeningContractError("availability registry omits or duplicates a frozen site")
    if (frame.n_valid_targets < 0).any():
        raise OpeningContractError("availability registry has a negative count")
    reportable = frame.reportable.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if reportable.isna().any() or not np.array_equal(
        reportable.to_numpy(bool),
        frame.n_valid_targets.ge(minimum_targets).to_numpy(bool),
    ):
        raise OpeningContractError("availability reportable flag disagrees with protocol")


def _rebuild_opened_nwis_panel(
    snapshot_index_path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    history_start: str,
    target_end: str,
) -> pd.DataFrame:
    """Replay the frozen parser over every raw response, without provider access."""
    snapshot_root = snapshot_index_path.parent.resolve()
    frames = []
    for record in records:
        request = record["request"]
        query = parse_qs(urlsplit(str(request["url"])).query)
        site = str(query["sites"][0])
        response_path = (snapshot_root / str(record["response_path"])).resolve()
        if snapshot_root not in response_path.parents:
            raise OpeningContractError("opened NWIS response escapes snapshot root")
        try:
            frame = parse_nwis_confirmatory_daily(
                response_path.read_bytes(),
                site_no=site,
                start=history_start,
                end=target_end,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise OpeningContractError(
                f"cannot replay frozen NWIS parser for site {site}"
            ) from exc
        frames.append(frame)
    if not frames:
        raise OpeningContractError("opened NWIS snapshot index is empty")
    output = pd.concat(frames, ignore_index=True)
    output["site_no"] = output.site_no.astype(str)
    output["DATE"] = pd.to_datetime(output.DATE)
    if output.duplicated(["site_no", "DATE"]).any():
        raise OpeningContractError("replayed NWIS panel duplicates site/date")
    return output.sort_values(["site_no", "DATE"]).reset_index(drop=True)


def _load_and_verify_normalized_outcomes(
    path: Path,
    *,
    raw_rebuild: pd.DataFrame,
    sites: set[str],
) -> pd.DataFrame:
    stored = _read_table(path)
    required = list(CONFIRMATORY_OUTCOME_COLUMNS)
    if list(stored.columns) != required:
        raise OpeningContractError("normalized outcome panel has a non-frozen schema")
    stored = stored.copy()
    stored["site_no"] = stored.site_no.astype("string").str.strip().astype(str)
    stored["DATE"] = pd.to_datetime(stored.DATE, errors="coerce")
    if stored.DATE.isna().any() or stored.duplicated(["site_no", "DATE"]).any():
        raise OpeningContractError("normalized outcome panel has invalid keys")
    if set(stored.site_no) != sites:
        raise OpeningContractError("normalized outcome panel differs from frozen cohort")
    expected = raw_rebuild[raw_rebuild.site_no.isin(sites)].copy()
    expected = expected.sort_values(["site_no", "DATE"]).reset_index(drop=True)
    stored = stored.sort_values(["site_no", "DATE"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            stored,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise OpeningContractError(
            "normalized outcome panel cannot be rebuilt from raw NWIS snapshots"
        ) from exc
    return stored


def _combined_confirmation_panel(
    outcomes: pd.DataFrame,
    retrospective_inputs: pd.DataFrame,
    *,
    feature_order: Sequence[str],
) -> pd.DataFrame:
    weather = retrospective_inputs.copy()
    weather["site_no"] = weather.site_no.astype("string").str.strip().astype(str)
    weather["DATE"] = pd.to_datetime(weather.DATE)
    merged = outcomes.merge(
        weather,
        on=["site_no", "DATE"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise OpeningContractError("opened outcomes lack a frozen retrospective-input row")
    merged = merged.drop(columns="_merge").rename(columns={"site_no": "site_id"})
    required = ["DATE", "site_id", *feature_order]
    missing = set(required) - set(merged)
    if missing:
        raise OpeningContractError(
            f"combined confirmation panel lacks actual model fields: {sorted(missing)}"
        )
    return merged[required].sort_values(["site_id", "DATE"]).reset_index(drop=True)


def _window_truth_registry(wd, station_order: Sequence[str]) -> pd.DataFrame:
    frames = []
    site = np.asarray([station_order[int(index)] for index in wd.station], dtype=object)
    target_valid = np.asarray(
        getattr(wd, "target_valid", np.ones_like(wd.y, dtype=bool)), dtype=bool
    )
    if target_valid.shape != wd.y.shape:
        raise OpeningContractError("frozen target-validity registry has a wrong shape")
    for column, horizon in enumerate(wd.horizons):
        selected = target_valid[:, column]
        frames.append(pd.DataFrame({
            "site_id": site[selected],
            "horizon": int(horizon),
            "issue_date": pd.to_datetime(wd.issue_date[selected]),
            "target_date": pd.to_datetime(wd.target_date[selected, column]),
            "y_true": wd.y[selected, column].astype(float),
        }))
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["site_id", "horizon", "issue_date", "target_date"]).any():
        raise OpeningContractError("reconstructed frozen windows duplicate a forecast key")
    return output


_TRUSTED_SCOPE = {
    "temporal": "route_a_temporal_confirmation",
    "external": "route_a_external_history_dependent_new_gage",
}
_TRUSTED_FEATURE_SET = "WTEMP+FLOW+TEMP+PRCP+RHMEAN+DH+WDSP"
_TRUSTED_ENSEMBLE_SEED = -1
_FORECAST_KEY = ["site_id", "horizon", "issue_date", "target_date"]


def _station_order(
    metadata: Mapping[str, Any],
    sites: Sequence[str],
    *,
    external: bool,
) -> tuple[str, ...]:
    if external:
        return tuple(sorted(str(site) for site in sites))
    mapping = metadata.get("station_to_index")
    if not isinstance(mapping, Mapping):
        raise OpeningContractError("same-station bundle lacks station_to_index")
    ordered = tuple(
        str(site) for site, _ in sorted(mapping.items(), key=lambda item: int(item[1]))
    )
    if set(ordered) != {str(site) for site in sites}:
        raise OpeningContractError("same-station bundle registry differs from cohort")
    return ordered


def _assert_truth_registry_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> None:
    paired = actual.merge(
        expected,
        on=_FORECAST_KEY,
        how="outer",
        suffixes=("_actual", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired._merge.eq("both").all():
        raise OpeningContractError(f"{label} forecast-key registry changed")
    difference = np.abs(
        paired.y_true_actual.to_numpy(float)
        - paired.y_true_expected.to_numpy(float)
    )
    if not np.isfinite(difference).all() or (difference > 1e-6).any():
        raise OpeningContractError(f"{label} target values changed")


def _trusted_frame(
    wd,
    station_order: Sequence[str],
    *,
    cohort: str,
    model_id: str,
    y_pred: np.ndarray,
    q05: np.ndarray | None = None,
    q50: np.ndarray | None = None,
    q95: np.ndarray | None = None,
    p_exceed: np.ndarray | None = None,
) -> pd.DataFrame:
    count, horizon_count = wd.y.shape
    expected_shape = (count, horizon_count)
    values = {
        "y_pred": np.asarray(y_pred, dtype=float),
        "q05": (np.full(expected_shape, np.nan) if q05 is None
                else np.asarray(q05, dtype=float)),
        "q50": (np.full(expected_shape, np.nan) if q50 is None
                else np.asarray(q50, dtype=float)),
        "q95": (np.full(expected_shape, np.nan) if q95 is None
                else np.asarray(q95, dtype=float)),
        "p_exceed": (np.full(expected_shape, np.nan) if p_exceed is None
                     else np.asarray(p_exceed, dtype=float)),
    }
    if any(array.shape != expected_shape for array in values.values()):
        raise OpeningContractError(f"{cohort}/{model_id} produced a wrong output shape")
    site = np.asarray(
        [station_order[int(index)] for index in wd.station], dtype=object
    )
    target_valid = np.asarray(
        getattr(wd, "target_valid", np.ones(expected_shape, dtype=bool)), dtype=bool
    )
    if target_valid.shape != expected_shape:
        raise OpeningContractError(f"{cohort}/{model_id} target mask has a wrong shape")
    frames = []
    for column, horizon in enumerate(wd.horizons):
        selected = target_valid[:, column]
        frames.append(R.make_pred_frame(
            model=model_id,
            scope=_TRUSTED_SCOPE[cohort],
            feature_set=_TRUSTED_FEATURE_SET,
            seed=_TRUSTED_ENSEMBLE_SEED,
            site_id=site[selected],
            horizon=np.full(int(selected.sum()), int(horizon)),
            split=np.full(int(selected.sum()), "confirm"),
            issue_date=pd.to_datetime(wd.issue_date[selected]),
            target_date=pd.to_datetime(wd.target_date[selected, column]),
            y_true=wd.y[selected, column],
            y_pred=values["y_pred"][selected, column],
            q05=values["q05"][selected, column],
            q50=values["q50"][selected, column],
            q95=values["q95"][selected, column],
            p_exceed=values["p_exceed"][selected, column],
        ))
    output = pd.concat(frames, ignore_index=True)
    try:
        R.validate_predictions(output)
    except ValueError as exc:
        raise OpeningContractError(
            f"trusted {cohort}/{model_id} output is invalid"
        ) from exc
    return output


def _frozen_calibration(
    metadata: Mapping[str, Any],
    station: np.ndarray,
    horizons: Sequence[int],
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    raw_probability: np.ndarray,
    *,
    external: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets, calibrators, _ = _validate_frozen_calibration_registry(
        metadata,
        tuple(str(value) for value in station),
        horizons,
        external=external,
        label=label,
    )
    q05, q50, q95 = (np.asarray(value, dtype=float).copy()
                     for value in (q05, q50, q95))
    probability = np.asarray(raw_probability, dtype=float).copy()
    for column, horizon in enumerate(horizons):
        horizon = int(horizon)
        if external:
            delta = np.full(
                len(station), float(offsets[f"__pooled__|{horizon}"]), dtype=float
            )
        else:
            delta = np.asarray(
                [float(offsets[f"{site}|{horizon}"]) for site in station],
                dtype=float,
            )
        q05[:, column] -= delta
        q95[:, column] += delta
        value = calibrators[str(horizon)]
        constant = value.get("constant")
        calibrator = PlattCalibrator(
            intercept=float(value["intercept"]),
            slope=float(value["slope"]),
            constant=None if constant is None else float(constant),
        )
        probability[:, column] = calibrator.predict(probability[:, column])
    if not (
        np.isfinite(q05).all()
        and np.isfinite(q50).all()
        and np.isfinite(q95).all()
        and np.isfinite(probability).all()
        and (q05 <= q50).all()
        and (q50 <= q95).all()
        and ((0.0 <= probability) & (probability <= 1.0)).all()
    ):
        raise OpeningContractError(f"{label} calibrated heads are invalid")
    return q05, q50, q95, probability


def _validate_frozen_calibration_registry(
    metadata: Mapping[str, Any],
    station_ids: Sequence[str],
    horizons: Sequence[int],
    *,
    external: bool,
    label: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate calibration keys and values without reading confirmation labels."""
    offsets = metadata.get("conformal_offsets")
    calibrators = metadata.get("event_calibrators")
    thresholds = metadata.get("event_thresholds")
    if (
        not isinstance(offsets, Mapping)
        or not isinstance(calibrators, Mapping)
        or not isinstance(thresholds, Mapping)
    ):
        raise OpeningContractError(f"{label} lacks frozen calibration parameters")
    expected_calibrators = {str(int(horizon)) for horizon in horizons}
    if set(calibrators) != expected_calibrators:
        raise OpeningContractError(f"{label} event calibrator registry changed")
    if external:
        if set(thresholds) != {"__pooled__"}:
            raise OpeningContractError(f"{label} external event threshold is not pooled")
        expected_offsets = {f"__pooled__|{int(horizon)}" for horizon in horizons}
    else:
        expected_sites = {str(value) for value in station_ids}
        if set(thresholds) != expected_sites:
            raise OpeningContractError(f"{label} event-threshold site registry changed")
        expected_offsets = {
            f"{site}|{int(horizon)}"
            for site in expected_sites for horizon in horizons
        }
    if set(offsets) != expected_offsets:
        raise OpeningContractError(f"{label} CQR offset registry changed")
    if not np.isfinite(np.asarray(list(thresholds.values()), dtype=float)).all():
        raise OpeningContractError(f"{label} event threshold is non-finite")
    try:
        offset_values = np.asarray(list(offsets.values()), dtype=float)
    except (TypeError, ValueError) as exc:
        raise OpeningContractError(f"{label} CQR offsets are not numeric") from exc
    if not np.isfinite(offset_values).all():
        raise OpeningContractError(f"{label} contains a non-finite CQR offset")
    for horizon in horizons:
        value = calibrators[str(int(horizon))]
        if not isinstance(value, Mapping):
            raise OpeningContractError(f"{label} Platt calibrator is malformed")
        try:
            constant = value.get("constant")
            parameters = np.asarray(
                [float(value["intercept"]), float(value["slope"])], dtype=float
            )
            if constant is not None:
                constant = float(constant)
        except (KeyError, TypeError, ValueError) as exc:
            raise OpeningContractError(f"{label} Platt calibrator is invalid") from exc
        if not np.isfinite(parameters).all() or (
            constant is not None and (not np.isfinite(constant) or not 0.0 < constant < 1.0)
        ):
            raise OpeningContractError(f"{label} Platt calibrator is non-finite")
    return offsets, calibrators, thresholds


def _score_sequence_bundle(
    *,
    root: Path,
    entry: Mapping[str, Any],
    metadata: Mapping[str, Any],
    wd,
    station_order: Sequence[str],
    cohort: str,
    model_id: str,
    external: bool,
    batch_size: int = 4096,
) -> pd.DataFrame:
    artifact = entry.get("artifact")
    if not isinstance(artifact, Mapping):
        raise OpeningContractError(f"{cohort}/{model_id} lacks a sequence artifact")
    directory = _resolve_inside(root, artifact.get("path"), kind="directory")
    count = int(entry.get("member_count", 0))
    try:
        members, loaded_metadata = instantiate_inference_ensemble(
            directory,
            model_factory=lambda _member, bundle: sequence_factory_from_metadata(bundle),
            expected_member_count=count,
            device="cpu",
        )
    except (FrozenInferenceError, RuntimeError, TypeError, ValueError) as exc:
        raise OpeningContractError(
            f"cannot reconstruct trusted {cohort}/{model_id} ensemble"
        ) from exc
    if loaded_metadata != metadata or len(members) != count:
        raise OpeningContractError(f"{cohort}/{model_id} metadata/member registry changed")
    n, horizon_count = wd.y.shape
    sums = {
        name: np.zeros((n, horizon_count), dtype=np.float64)
        for name in ("point", "q05", "q50", "q95", "event")
    }
    index = np.arange(n, dtype=int)
    for member_name in sorted(members):
        model = members[member_name]
        model.eval()
        member_values: dict[str, list[np.ndarray]] = {
            name: [] for name in sums
        }
        for start in range(0, n, batch_size):
            chunk = index[start:start + batch_size]
            with torch.inference_mode():
                output = model(wd.batch(chunk, "cpu"))
            member_values["point"].append(output.point.detach().cpu().numpy())
            member_values["q05"].append(output.q05.detach().cpu().numpy())
            member_values["q50"].append(output.q50.detach().cpu().numpy())
            member_values["q95"].append(output.q95.detach().cpu().numpy())
            member_values["event"].append(
                torch.sigmoid(output.exceed_logit).detach().cpu().numpy()
            )
        arrays = {
            name: np.concatenate(chunks, axis=0).astype(float, copy=False)
            for name, chunks in member_values.items()
        }
        if any(array.shape != (n, horizon_count) for array in arrays.values()):
            raise OpeningContractError(
                f"{cohort}/{model_id}/{member_name} produced a wrong head shape"
            )
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise OpeningContractError(
                f"{cohort}/{model_id}/{member_name} produced non-finite values"
            )
        if not (
            (arrays["q05"] <= arrays["q50"]).all()
            and (arrays["q50"] <= arrays["q95"]).all()
        ):
            raise OpeningContractError(
                f"{cohort}/{model_id}/{member_name} produced crossed quantiles"
            )
        for name in sums:
            sums[name] += arrays[name]
    averaged = {name: value / count for name, value in sums.items()}
    station = np.asarray(
        [station_order[int(index)] for index in wd.station], dtype=object
    )
    q05, q50, q95, probability = _frozen_calibration(
        metadata,
        station,
        wd.horizons,
        averaged["q05"],
        averaged["q50"],
        averaged["q95"],
        averaged["event"],
        external=external,
        label=f"{cohort}/{model_id}",
    )
    return _trusted_frame(
        wd,
        station_order,
        cohort=cohort,
        model_id=model_id,
        y_pred=averaged["point"],
        q05=q05,
        q50=q50,
        q95=q95,
        p_exceed=probability,
    )


def _confirmation_tabular_design(
    imputed: pd.DataFrame,
    climatology,
    expected: pd.DataFrame,
    *,
    feature_order: tuple[str, ...],
    horizon: int,
    station_order: Sequence[str],
    manifest: Mapping[str, Any],
    external: bool,
) -> pd.DataFrame:
    # ``build_tabular`` iterates through this explicit registry; set it to the
    # already-validated cohort order before constructing any rows.
    from . import config as C

    previous_stations, previous_upstream = C.STATIONS, C.UPSTREAM
    try:
        C.STATIONS = tuple(station_order)
        C.UPSTREAM = {site: None for site in station_order}
        tabular = F.build_tabular(
            imputed,
            int(horizon),
            feature_order,
            climatology,
            drop_feature_nans=False,
            require_observed_target=True,
            include_missingness=True,
        )
    finally:
        C.STATIONS, C.UPSTREAM = previous_stations, previous_upstream
    target = expected[expected.horizon.eq(int(horizon))].copy()
    target = target.rename(columns={"y_true": "expected_y"})
    tabular["site_id"] = tabular.site_id.astype(str)
    tabular["issue_date"] = pd.to_datetime(tabular.issue_date)
    tabular["target_date"] = pd.to_datetime(tabular.target_date)
    selected = target.merge(
        tabular,
        on=["site_id", "issue_date", "target_date"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not selected._merge.eq("both").all():
        raise OpeningContractError(
            f"LightGBM h{horizon} design lacks a frozen forecast key"
        )
    if not targets_match_at_model_precision(
        selected.expected_y.to_numpy(float),
        selected.y.to_numpy(float),
    ):
        raise OpeningContractError(f"LightGBM h{horizon} tabular target changed")
    selected = selected.drop(columns=["horizon", "expected_y", "_merge"])
    columns = F.feature_columns(selected)
    for column in columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce").fillna(0.0)
    if external:
        if manifest.get("station_categories") not in (None, [], ()):
            raise OpeningContractError("external LightGBM exposes station categories")
    else:
        categories = tuple(str(value) for value in manifest.get("station_categories", ()))
        if categories != tuple(station_order):
            raise OpeningContractError("LightGBM station-category registry changed")
        selected["station_code"] = pd.Categorical(
            selected.site_id.astype(str), categories=list(categories)
        )
        columns.append("station_code")
    design_order = tuple(str(value) for value in manifest.get("design_feature_order", ()))
    if tuple(columns) != design_order:
        raise OpeningContractError(
            f"LightGBM h{horizon} engineered feature order changed"
        )
    return selected.loc[:, list(design_order)]


def _score_lightgbm_bundle(
    *,
    root: Path,
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    wd,
    imputed: pd.DataFrame,
    climatology,
    expected: pd.DataFrame,
    station_order: Sequence[str],
    cohort: str,
    external: bool,
) -> pd.DataFrame:
    manifest_path = _verify_file_binding(
        root, entry.get("artifact", {}), label=f"{cohort}/LightGBM bundle"
    )
    try:
        members, loaded_manifest = load_lightgbm_bundle(manifest_path)
    except ModelSuiteError as exc:
        raise OpeningContractError("cannot reconstruct trusted LightGBM bundle") from exc
    if loaded_manifest != manifest or len(members) != 5:
        raise OpeningContractError("LightGBM trusted manifest/member registry changed")
    n, horizon_count = wd.y.shape
    point = np.zeros((n, horizon_count), dtype=float)
    quantiles = {
        name: np.zeros((n, horizon_count), dtype=float)
        for name in ("q05", "q50", "q95")
    }
    event = np.zeros((n, horizon_count), dtype=float)
    target_valid = np.asarray(
        getattr(wd, "target_valid", np.ones((n, horizon_count), dtype=bool)),
        dtype=bool,
    )
    for column, horizon in enumerate(wd.horizons):
        selected = np.flatnonzero(target_valid[:, column])
        design = _confirmation_tabular_design(
            imputed,
            climatology,
            expected,
            feature_order=tuple(manifest["raw_feature_order"]),
            horizon=int(horizon),
            station_order=station_order,
            manifest=manifest,
            external=external,
        )
        if len(design) != len(selected):
            raise OpeningContractError(f"LightGBM h{horizon} row registry changed")
        for member_name in sorted(members):
            heads = members[member_name][int(horizon)]
            point[selected, column] += np.asarray(
                heads["point"].predict(design, num_threads=1), dtype=float
            )
            try:
                repaired = repair_lightgbm_quantiles(
                    np.asarray(
                        heads["q05"].predict(design, num_threads=1), dtype=float
                    ),
                    np.asarray(
                        heads["q50"].predict(design, num_threads=1), dtype=float
                    ),
                    np.asarray(
                        heads["q95"].predict(design, num_threads=1), dtype=float
                    ),
                )
            except QuantileIdentityError as exc:
                raise OpeningContractError(
                    f"{cohort}/LightGBM/{member_name}/h{horizon} "
                    "produced invalid nominal quantile heads"
                ) from exc
            for values, name in zip(
                repaired, ("q05", "q50", "q95"), strict=True
            ):
                quantiles[name][selected, column] += values
            event[selected, column] += np.asarray(
                heads["event"].predict(design, num_threads=1), dtype=float
            )
    point[target_valid] /= len(members)
    event[target_valid] /= len(members)
    for value in quantiles.values():
        value[target_valid] /= len(members)
    if not (
        np.isfinite(point[target_valid]).all()
        and np.isfinite(event[target_valid]).all()
        and all(np.isfinite(value[target_valid]).all() for value in quantiles.values())
        and ((0.0 <= event[target_valid]) & (event[target_valid] <= 1.0)).all()
    ):
        raise OpeningContractError("LightGBM trusted heads are non-finite or invalid")
    station = np.asarray(
        [station_order[int(index)] for index in wd.station], dtype=object
    )
    q05, q50, q95, probability = _frozen_calibration(
        manifest,
        station,
        wd.horizons,
        quantiles["q05"],
        quantiles["q50"],
        quantiles["q95"],
        event,
        external=external,
        label=f"{cohort}/LightGBM",
    )
    return _trusted_frame(
        wd,
        station_order,
        cohort=cohort,
        model_id="LightGBM",
        y_pred=point,
        q05=q05,
        q50=q50,
        q95=q95,
        p_exceed=probability,
    )


def score_frozen_confirmation_suite(
    *,
    root: str | Path,
    cohort: str,
    combined_panel: pd.DataFrame,
    sites: Sequence[str],
    interval: tuple[str, str],
    suite: Mapping[str, Any],
) -> pd.DataFrame:
    """Independently execute every frozen model; never trust worker predictions."""
    if cohort not in {"temporal", "external"}:
        raise ValueError("cohort must be temporal or external")
    root = Path(root).resolve()
    external = cohort == "external"
    metadata = suite["metadata"][cohort]
    entries = suite["entries"][cohort]
    primary_metadata = metadata.get("ThermoRoute")
    if not isinstance(primary_metadata, Mapping):
        raise OpeningContractError(f"{cohort} lacks primary ThermoRoute metadata")
    station_order = _station_order(primary_metadata, sites, external=external)
    try:
        wd, transforms, imputed = build_frozen_confirmation_windows(
            combined_panel,
            primary_metadata,
            station_order,
            interval=interval,
            external=external,
        )
    except (FrozenInferenceError, RuntimeError, ValueError, KeyError) as exc:
        raise OpeningContractError(f"cannot build trusted {cohort} windows") from exc
    expected = _window_truth_registry(wd, station_order)
    # Every learned model must consume the same frozen preprocessing.  This also
    # lets one immutable window registry be reused without estimating anything.
    for model_id, value in metadata.items():
        if value.get("preprocessing") != primary_metadata.get("preprocessing"):
            raise OpeningContractError(
                f"{cohort}/{model_id} preprocessing differs from primary ThermoRoute"
            )
        if model_id != "LightGBM":
            order = _station_order(value, sites, external=external)
            if order != station_order:
                raise OpeningContractError(
                    f"{cohort}/{model_id} station index order differs from primary"
                )
    frames = [
        _trusted_frame(
            wd,
            station_order,
            cohort=cohort,
            model_id="Persistence",
            y_pred=np.repeat(wd.wtemp_t[:, None], len(wd.horizons), axis=1),
        ),
        _trusted_frame(
            wd,
            station_order,
            cohort=cohort,
            model_id="DampedPersistence",
            y_pred=wd.damped_prior,
        ),
        _trusted_frame(
            wd,
            station_order,
            cohort=cohort,
            model_id="Climatology",
            y_pred=wd.clim_tgt,
        ),
    ]
    for model_id in suite["required_models"][cohort]:
        if model_id in BUILTIN_MODELS or model_id == "LightGBM":
            continue
        model_metadata = metadata.get(model_id)
        if not isinstance(model_metadata, Mapping):
            raise OpeningContractError(f"{cohort}/{model_id} metadata is absent")
        frames.append(_score_sequence_bundle(
            root=root,
            entry=entries[model_id],
            metadata=model_metadata,
            wd=wd,
            station_order=station_order,
            cohort=cohort,
            model_id=model_id,
            external=external,
        ))
    lightgbm_metadata = metadata.get("LightGBM")
    if not isinstance(lightgbm_metadata, Mapping):
        raise OpeningContractError(f"{cohort} LightGBM metadata is absent")
    if lightgbm_metadata.get("preprocessing") != primary_metadata.get("preprocessing"):
        raise OpeningContractError(
            f"{cohort}/LightGBM preprocessing differs from primary ThermoRoute"
        )
    frames.append(_score_lightgbm_bundle(
        root=root,
        entry=entries["LightGBM"],
        manifest=lightgbm_metadata,
        wd=wd,
        imputed=imputed,
        climatology=transforms.climatology,
        expected=expected,
        station_order=station_order,
        cohort=cohort,
        external=external,
    ))
    output = pd.concat(frames, ignore_index=True)
    required_order = tuple(suite["required_models"][cohort])
    if set(output.model.astype(str)) != set(required_order):
        raise OpeningContractError(f"trusted {cohort} scorer omitted or added a model")
    key_sets = {
        str(model): _prediction_keys(group)
        for model, group in output.groupby("model", sort=False)
    }
    if any(keys != _prediction_keys(expected) for keys in key_sets.values()):
        raise OpeningContractError(f"trusted {cohort} models do not share exact keys")
    return output


def _assert_worker_predictions_equal_trusted(
    worker: pd.DataFrame,
    trusted: pd.DataFrame,
    *,
    cohort: str,
    atol: float = 0.0,
) -> str:
    identity = ["model", "site_id", "horizon", "issue_date", "target_date"]
    left, right = worker.copy(), trusted.copy()
    for frame in (left, right):
        frame["site_id"] = frame.site_id.astype(str)
        frame["issue_date"] = pd.to_datetime(frame.issue_date)
        frame["target_date"] = pd.to_datetime(frame.target_date)
    paired = left.merge(
        right,
        on=identity,
        how="outer",
        suffixes=("_stored", "_trusted"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired._merge.eq("both").all():
        raise OpeningContractError(f"{cohort} stored keys differ from trusted scorer")
    for column in ("scope", "feature_set", "seed", "split"):
        if not paired[f"{column}_stored"].astype(str).eq(
            paired[f"{column}_trusted"].astype(str)
        ).all():
            raise OpeningContractError(
                f"{cohort} stored artifact changed canonical {column} metadata"
            )
    for column in ("y_true", "y_pred", "q05", "q50", "q95", "p_exceed"):
        worker_values = paired[f"{column}_stored"].to_numpy(float)
        trusted_values = paired[f"{column}_trusted"].to_numpy(float)
        if not np.array_equal(np.isnan(worker_values), np.isnan(trusted_values)):
            raise OpeningContractError(
                f"{cohort} stored artifact changed presence of trusted {column}"
            )
        finite = np.isfinite(trusted_values)
        if finite.any() and not np.allclose(
            worker_values[finite], trusted_values[finite], rtol=0.0, atol=atol
        ):
            maximum = float(np.max(np.abs(
                worker_values[finite] - trusted_values[finite]
            )))
            raise OpeningContractError(
                f"{cohort} stored {column} differs from trusted scorer "
                f"(max_abs={maximum:.6g})"
            )
    return canonical_frame_digest(trusted, R.PRED_COLS)


def _assert_predictions_match_frozen_windows(
    predictions: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    model: str,
) -> None:
    key = ["site_id", "horizon", "issue_date", "target_date"]
    actual = predictions[predictions.model.eq(model)][[*key, "y_true"]].copy()
    actual["site_id"] = actual.site_id.astype(str)
    actual["issue_date"] = pd.to_datetime(actual.issue_date)
    actual["target_date"] = pd.to_datetime(actual.target_date)
    paired = actual.merge(
        expected,
        on=key,
        how="outer",
        suffixes=("_stored", "_raw_rebuild"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired._merge.eq("both").all():
        raise OpeningContractError(
            "prediction keys differ from raw outcomes plus frozen preprocessing"
        )
    difference = np.abs(
        paired.y_true_stored.to_numpy(float)
        - paired.y_true_raw_rebuild.to_numpy(float)
    )
    if not np.isfinite(difference).all() or (difference > 1e-6).any():
        raise OpeningContractError("prediction y_true differs from raw NWIS WTEMP")


def _assert_availability_matches_windows(
    availability_path: Path,
    expected_by_cohort: Mapping[str, pd.DataFrame],
    *,
    sites_by_cohort: Mapping[str, set[str]],
) -> None:
    availability = pd.read_csv(
        availability_path, dtype={"site_no": "string"}, keep_default_na=False
    )
    availability["site_no"] = availability.site_no.astype(str)
    availability["horizon"] = pd.to_numeric(availability.horizon).astype(int)
    availability["n_valid_targets"] = pd.to_numeric(
        availability.n_valid_targets
    ).astype(int)
    expected_counts: dict[tuple[str, str, int], int] = {}
    for cohort, sites in sites_by_cohort.items():
        counts = expected_by_cohort[cohort].groupby(
            ["site_id", "horizon"]
        ).size().to_dict()
        for site in sites:
            for horizon in (1, 3, 7):
                expected_counts[(cohort, site, horizon)] = int(
                    counts.get((site, horizon), 0)
                )
    actual_counts = {
        (str(row.cohort), str(row.site_no), int(row.horizon)): int(row.n_valid_targets)
        for row in availability.itertuples(index=False)
    }
    if actual_counts != expected_counts:
        raise OpeningContractError(
            "availability counts differ from raw outcomes and frozen window contract"
        )


def compute_confirmatory_statistics(
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    minimum_targets: int = 100,
) -> dict[str, Any]:
    """Compute exactly the five preregistered station/HUC2 tests."""
    cluster_map = huc2_cluster_map(registry)
    required = _formal_test_registry(protocol)
    inference = protocol["primary_inference_contract"]
    n_boot = int(inference["confidence_interval"]["draws"])
    p_value_contract = inference["one_sided_p_value"]
    maximum_configurations = int(
        p_value_contract["maximum_configurations_for_frozen_cohort"]
    )
    exact_max_clusters = maximum_configurations.bit_length() - 1

    def station_rmse(model: str, horizon: int) -> dict[str, float]:
        selected = predictions[
            predictions.model.eq(model) & predictions.horizon.eq(horizon)
        ]
        output = {}
        for site, group in selected.groupby("site_id"):
            if len(group) >= minimum_targets:
                output[str(site)] = float(np.sqrt(np.mean(
                    (group.y_pred.to_numpy(float) - group.y_true.to_numpy(float)) ** 2
                )))
        return output

    rows = []
    raw_p = []
    for test in required:
        test_id = test["test_id"]
        candidate_name = test["candidate"]
        reference = test["reference"]
        horizon = int(test["horizon"])
        margin = float(test["margin_c"])
        bootstrap_seed = int(test["bootstrap_seed"])
        sign_seed = int(test["sign_flip_seed"])
        candidate = station_rmse(candidate_name, horizon)
        baseline = station_rmse(reference, horizon)
        sites = sorted(set(candidate) & set(baseline))
        effects = np.asarray([candidate[site] - baseline[site] for site in sites])
        clusters = np.asarray([cluster_map[site] for site in sites], dtype=object)
        if len(sites) == 0 or len(set(clusters)) < 2:
            raw_p.append(1.0)
            rows.append({
                "test_id": test_id,
                "candidate": candidate_name,
                "reference": reference,
                "horizon": horizon,
                "margin_c": margin,
                "effect_convention": "station_RMSE_ThermoRoute-minus-reference",
                "status": "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS",
                "median_effect_c": None,
                "ci_low_c": None,
                "ci_high_c": None,
                "n_stations": len(sites),
                "n_clusters": len(set(clusters)),
                "win_rate": None,
                "p_one_sided_raw": 1.0,
                "bootstrap_seed": bootstrap_seed,
                "sign_flip_seed_legacy_ignored": sign_seed,
                "sign_flip_configurations": None,
            })
            continue
        summary = cluster_bootstrap_paired_effect(
            effects,
            clusters,
            statistic="median",
            n_boot=n_boot,
            seed=bootstrap_seed,
            null_margin=margin,
        )
        cluster_count = len(set(clusters))
        configurations = 1 << cluster_count
        if configurations > maximum_configurations:
            raise OpeningContractError(
                "primary Route-A family exceeds the frozen exact sign-flip limit"
            )
        p_value = cluster_sign_flip_pvalue(
            effects,
            clusters,
            statistic="median",
            seed=sign_seed,
            null_margin=margin,
            exact_max_clusters=exact_max_clusters,
        )
        raw_p.append(p_value)
        rows.append({
            "test_id": test_id,
            "candidate": candidate_name,
            "reference": reference,
            "horizon": horizon,
            "margin_c": margin,
            "effect_convention": "station_RMSE_ThermoRoute-minus-reference",
            "status": "ESTIMABLE",
            "median_effect_c": summary["effect"],
            "ci_low_c": summary["ci_low"],
            "ci_high_c": summary["ci_high"],
            "n_stations": summary["n_stations"],
            "n_clusters": summary["n_clusters"],
            "win_rate": float(np.mean(effects < 0.0)),
            "p_one_sided_raw": p_value,
            "bootstrap_seed": bootstrap_seed,
            "sign_flip_seed_legacy_ignored": sign_seed,
            "sign_flip_configurations": configurations,
        })
    adjusted = holm_adjust(np.asarray(raw_p, dtype=float))
    for row, value in zip(rows, adjusted):
        row["p_holm"] = float(value)
        estimable = row["status"] == "ESTIMABLE"
        row["reject_at_0_05"] = bool(estimable and value <= 0.05)
        ci_high = row["ci_high_c"]
        row["confidence_bound_supports_margin"] = bool(
            estimable
            and ci_high is not None
            and float(ci_high) < float(row["margin_c"])
        )
    return {
        "format": STATISTICS_FORMAT,
        "effect_unit": "degrees_C",
        "sampling_unit": "station",
        "cluster_unit": "HUC2_or_stable_unmapped_singleton",
        "confidence_interval": {
            "method": "whole-HUC2 cluster percentile bootstrap",
            "draws": n_boot,
        },
        "p_value": {
            "method": p_value_contract["method"],
            "maximum_configurations_for_frozen_cohort": maximum_configurations,
            "monte_carlo_used": False,
            "assumption": p_value_contract["null_assumption"],
            "enumeration_rule": p_value_contract["enumeration_rule"],
            "legacy_seed_field": p_value_contract["legacy_seed_field"],
        },
        "multiplicity": "Holm step-down across exactly five tests",
        "tests": rows,
    }


def _assert_statistics_equal(
    expected: object,
    actual: object,
    *,
    path: str = "$",
) -> None:
    """Recursively compare a trusted JSON product with its recomputation."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or expected.keys() != actual.keys():
            raise OpeningContractError(
                f"trusted JSON artifact schema differs at {path}"
            )
        for key, left in expected.items():
            _assert_statistics_equal(
                left, actual[key], path=f"{path}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise OpeningContractError(
                f"trusted JSON artifact row count differs at {path}"
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_statistics_equal(left, right, path=f"{path}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        if not isinstance(actual, (int, float, np.integer, np.floating)) or not np.isclose(
            float(expected), float(actual), rtol=1e-10, atol=1e-12
        ):
            raise OpeningContractError(f"trusted JSON value changed at {path}")
        return
    if expected != actual:
        raise OpeningContractError(f"trusted JSON value changed at {path}")


def _verified_acquisition_for_scoring(
    *,
    preflight: Mapping[str, Any],
    root: Path,
    allow_recoverable_publication_mode: bool = False,
) -> tuple[
    dict[str, pd.DataFrame], set[str], set[str], list[Mapping[str, Any]]
]:
    authorization = preflight["authorization"]
    from .outcome_acquisition import _assert_exact_acquisition_directory

    acquisition_state = {
        key: Path(value) for key, value in preflight["state_paths"].items()
    }
    _assert_exact_acquisition_directory(
        Path(preflight["state_paths"]["acquisition_manifest"]).parent,
        acquisition_state,
        allow_recoverable_canonical_mode=(
            allow_recoverable_publication_mode
        ),
    )
    manifest_path = Path(preflight["state_paths"]["acquisition_manifest"])
    acquisition = _load_json(manifest_path, label="opened acquisition manifest")
    expected = {
        "format": ACQUISITION_MANIFEST_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_sha256": preflight["authorization_sha256"],
        "protocol_sha256": preflight["protocol"]["protocol_sha256"],
        "labels_state": "OPENED_ONCE",
        "site_replacement_count": 0,
        "response_replacement_count": 0,
        "history_start": authorization["acquisition_plan"]["history_start"],
        "target_start": authorization["acquisition_plan"]["target_start"],
        "target_end": authorization["acquisition_plan"]["target_end"],
        "maximum_response_bytes_per_request": (
            MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
        "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
    }
    if any(acquisition.get(key) != value for key, value in expected.items()):
        raise OpeningContractError("raw acquisition manifest identity changed")
    raw_index = _verify_canonical_file_binding(
        root, acquisition.get("raw_nwis_snapshot_index", {}),
        expected_path=preflight["state_paths"]["raw_nwis_snapshot_index"],
        label="opened NWIS snapshot index",
    )
    records = _verify_snapshot_index(root, raw_index, prelabel=False)
    temporal_sites = set(preflight["registries"]["development"].site_no.astype(str))
    external_sites = set(preflight["registries"]["external"].site_no.astype(str))
    _verify_opened_nwis_index(
        records,
        expected_sites=temporal_sites | external_sites,
        history_start=authorization["acquisition_plan"]["history_start"],
        target_end=authorization["acquisition_plan"]["target_end"],
    )
    request_map = _verify_canonical_file_binding(
        root,
        acquisition.get("request_map", {}),
        expected_path=preflight["state_paths"]["acquisition_request_map"],
        label="opened NWIS request map",
    )
    _verify_opened_request_map(
        request_map,
        records=records,
        opening_id=authorization["opening_id"],
        authorization_sha256=preflight["authorization_sha256"],
        temporal_sites=temporal_sites,
        external_sites=external_sites,
    )
    request_rows = _load_json(
        request_map, label="opened NWIS request map"
    )["requests"]
    _verify_opened_transport_evidence(
        root=root,
        acquisition=acquisition,
        records=records,
        request_rows=request_rows,
        opening_id=authorization["opening_id"],
        authorization_sha256=preflight["authorization_sha256"],
        work_order_path=Path(preflight["state_paths"]["work_order"]),
        raw_root=raw_index.parent,
    )
    rebuilt = _rebuild_opened_nwis_panel(
        raw_index,
        records,
        history_start=authorization["acquisition_plan"]["history_start"],
        target_end=authorization["acquisition_plan"]["target_end"],
    )
    bindings = acquisition.get("normalized_outcome_tables")
    if not isinstance(bindings, Mapping) or set(bindings) != {"temporal", "external"}:
        raise OpeningContractError("raw acquisition lacks both normalized cohorts")
    normalized: dict[str, pd.DataFrame] = {}
    for cohort, sites in (("temporal", temporal_sites), ("external", external_sites)):
        path = _verify_canonical_file_binding(
            root,
            bindings[cohort],
            expected_path=preflight["state_paths"][f"{cohort}_outcomes"],
            label=f"{cohort} normalized opened outcomes",
        )
        normalized[cohort] = _load_and_verify_normalized_outcomes(
            path, raw_rebuild=rebuilt, sites=sites
        )
    return normalized, temporal_sites, external_sites, request_rows


def _availability_from_truth(
    truth_by_cohort: Mapping[str, pd.DataFrame],
    *,
    sites_by_cohort: Mapping[str, set[str]],
    minimum_targets: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in ("temporal", "external"):
        counts = truth_by_cohort[cohort].groupby(["site_id", "horizon"]).size()
        for site in sorted(sites_by_cohort[cohort]):
            for horizon in (1, 3, 7):
                count = int(counts.get((site, horizon), 0))
                rows.append({
                    "cohort": cohort,
                    "site_no": site,
                    "horizon": horizon,
                    "n_valid_targets": count,
                    "reportable": count >= minimum_targets,
                })
    return pd.DataFrame(rows, columns=[
        "cohort", "site_no", "horizon", "n_valid_targets", "reportable"
    ])


def _build_outcome_quality_audit(
    *,
    normalized: Mapping[str, pd.DataFrame],
    request_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Mechanically count every raw qualifier/series/value-status combination."""
    contract = protocol["daily_outcome_quality_contract"]
    variables = tuple(contract["mandatory_quality_audit"]["variables"])
    counts: dict[tuple[str, str, str, str | None, str | None, str], int] = {}
    conflicts: list[dict[str, Any]] = []
    conflict_constituents: list[dict[str, Any]] = []
    value_status_day_counts: list[dict[str, Any]] = []
    request_by_site: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in request_rows:
        if not isinstance(item, Mapping):
            raise OpeningContractError("opened request-map row is malformed")
        request_key = (str(item.get("cohort", "")), str(item.get("site_no", "")))
        if request_key in request_by_site or not all(
            _is_sha256(item.get(field))
            for field in ("request_sha256", "response_sha256")
        ):
            raise OpeningContractError(
                "opened request-map provenance is incomplete or duplicated"
            )
        request_by_site[request_key] = item
    for cohort in ("temporal", "external"):
        frame = normalized[cohort]
        for variable in variables:
            value = pd.to_numeric(frame[variable], errors="coerce")
            qualifier = frame[f"{variable}_qualifier"].astype("string")
            series = frame[f"{variable}_series_id"].astype("string")
            status = frame[f"{variable}_value_status"].astype(str)
            conflict = frame[f"{variable}_series_conflict"].astype(bool)
            conflict_count = pd.to_numeric(
                frame[f"{variable}_conflicting_series_count"], errors="coerce"
            )
            conflict_ids = frame[f"{variable}_conflicting_series_ids"].astype("string")
            conflict_qualifiers = frame[
                f"{variable}_conflicting_series_qualifiers"
            ].astype("string")
            conflict_provenance = frame[
                f"{variable}_conflicting_series_provenance"
            ].astype("string")
            retained = status.eq("RETAINED_FINITE_VALUE")
            missing = status.eq("MISSING_NO_FINITE_SERIES")
            conflicted = status.eq("MULTIPLE_FINITE_SERIES_CONFLICT")
            if not (retained | missing | conflicted).all():
                raise OpeningContractError("normalized outcome has an unknown value status")
            if (
                not np.array_equal(retained.to_numpy(), np.isfinite(value.to_numpy(float)))
                or not np.array_equal(conflicted.to_numpy(), conflict.to_numpy())
                or (conflicted & conflict_count.lt(2)).any()
                or ((~conflicted) & conflict_count.ne(0)).any()
                or (retained & series.isna()).any()
                or ((~retained) & series.notna()).any()
                or (conflicted & (
                    conflict_ids.isna()
                    | conflict_qualifiers.isna()
                    | conflict_provenance.isna()
                )).any()
                or ((~conflicted) & (
                    conflict_ids.notna()
                    | conflict_qualifiers.notna()
                    | conflict_provenance.notna()
                )).any()
            ):
                raise OpeningContractError(
                    "normalized series/conflict fields disagree with finite values"
                )
            for row_index, row in frame.iterrows():
                row_status = str(status.loc[row_index])
                if row_status == "MULTIPLE_FINITE_SERIES_CONFLICT":
                    constituent_ids = str(conflict_ids.loc[row_index]).split("|")
                    try:
                        constituent_qualifiers = json.loads(
                            str(conflict_qualifiers.loc[row_index])
                        )
                        constituent_provenance = json.loads(
                            str(conflict_provenance.loc[row_index])
                        )
                    except json.JSONDecodeError as exc:
                        raise OpeningContractError(
                            "conflicting NWIS constituent registry is malformed"
                        ) from exc
                    if (
                        not isinstance(constituent_qualifiers, dict)
                        or not isinstance(constituent_provenance, dict)
                        or set(constituent_qualifiers) != set(constituent_ids)
                        or set(constituent_provenance) != set(constituent_ids)
                        or len(constituent_ids) != int(conflict_count.loc[row_index])
                    ):
                        raise OpeningContractError(
                            "conflicting NWIS series/qualifier registry changed"
                        )
                    for constituent_id in constituent_ids:
                        raw_value = constituent_qualifiers[constituent_id]
                        provenance = constituent_provenance[constituent_id]
                        if not isinstance(provenance, Mapping) or set(provenance) != {
                            "value_column", "qualifier_column", "raw_qualifier",
                            "raw_value", "parsed_finite_value",
                        }:
                            raise OpeningContractError(
                                "conflicting NWIS constituent provenance changed"
                            )
                        try:
                            parsed_finite_value = float(
                                provenance["parsed_finite_value"]
                            )
                            reparsed_raw_value = float(provenance["raw_value"])
                        except (TypeError, ValueError) as exc:
                            raise OpeningContractError(
                                "conflicting NWIS constituent value is malformed"
                            ) from exc
                        if (
                            provenance["value_column"] != constituent_id
                            or provenance["raw_qualifier"] != raw_value
                            or not np.isfinite(parsed_finite_value)
                            or not np.isclose(
                                reparsed_raw_value,
                                parsed_finite_value,
                                rtol=0.0,
                                atol=0.0,
                            )
                        ):
                            raise OpeningContractError(
                                "conflicting NWIS constituent does not replay"
                            )
                        request = request_by_site.get(
                            (cohort, str(row["site_no"]))
                        )
                        if request is None:
                            raise OpeningContractError(
                                "conflicting NWIS constituent lacks raw response binding"
                            )
                        conflict_constituents.append({
                            "cohort": cohort,
                            "site_no": str(row["site_no"]),
                            "date": pd.Timestamp(row["DATE"]).strftime("%Y-%m-%d"),
                            "variable": variable,
                            "series_id": constituent_id,
                            "value_column": str(provenance["value_column"]),
                            "qualifier_column": provenance["qualifier_column"],
                            "raw_qualifier": provenance["raw_qualifier"],
                            "raw_value": str(provenance["raw_value"]),
                            "parsed_finite_value": parsed_finite_value,
                            "request_sha256": str(request["request_sha256"]),
                            "response_sha256": str(request["response_sha256"]),
                        })
                        conflict_count_key = (
                            cohort,
                            str(row["site_no"]),
                            variable,
                            None if raw_value is None else str(raw_value),
                            constituent_id,
                            row_status,
                        )
                        counts[conflict_count_key] = (
                            counts.get(conflict_count_key, 0) + 1
                        )
                    continue
                raw_qualifier = qualifier.loc[row_index]
                series_id = series.loc[row_index]
                ordinary_count_key = (
                    cohort,
                    str(row["site_no"]),
                    variable,
                    None if pd.isna(raw_qualifier) else str(raw_qualifier),
                    None if pd.isna(series_id) else str(series_id),
                    row_status,
                )
                counts[ordinary_count_key] = (
                    counts.get(ordinary_count_key, 0) + 1
                )
            for site_no, group in frame.groupby("site_no", sort=True):
                site_conflicts = conflict.loc[group.index]
                site_counts = conflict_count.loc[group.index]
                conflicts.append({
                    "cohort": cohort,
                    "site_no": str(site_no),
                    "variable": variable,
                    "conflict_days": int(site_conflicts.sum()),
                    "maximum_simultaneous_finite_series": int(site_counts.max()),
                })
                for status_name, count in status.loc[group.index].value_counts(
                    sort=False
                ).items():
                    value_status_day_counts.append({
                        "cohort": cohort,
                        "site_no": str(site_no),
                        "variable": variable,
                        "value_status": str(status_name),
                        "day_count": int(count),
                    })
    rows = [
        {
            "cohort": key[0],
            "site_no": key[1],
            "variable": key[2],
            "raw_qualifier": key[3],
            "series_id": key[4],
            "value_status": key[5],
            "count": count,
        }
        for key, count in sorted(
            counts.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    ]
    series_columns: list[dict[str, Any]] = []
    for item in request_rows:
        registry = item.get("series_registry")
        if not isinstance(registry, Mapping) or set(registry) != set(variables):
            raise OpeningContractError("opened request lacks exact NWIS series registry")
        for variable in variables:
            entries = registry[variable]
            if not isinstance(entries, list):
                raise OpeningContractError("opened NWIS series registry is malformed")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise OpeningContractError("opened NWIS series entry is malformed")
                series_columns.append({
                    "cohort": str(item["cohort"]),
                    "site_no": str(item["site_no"]),
                    "variable": variable,
                    "parameter_code": str(entry["parameter_code"]),
                    "value_column": str(entry["value_column"]),
                    "qualifier_column": (
                        None if entry.get("qualifier_column") is None
                        else str(entry["qualifier_column"])
                    ),
                })
    series_columns.sort(key=lambda row: tuple(str(value) for value in row.values()))
    series_lookup = {
        (
            row["cohort"], row["site_no"], row["variable"], row["value_column"]
        ): row["qualifier_column"]
        for row in series_columns
    }
    if len(series_lookup) != len(series_columns):
        raise OpeningContractError("opened NWIS series registry contains duplicates")
    for row in conflict_constituents:
        series_key = (
            row["cohort"], row["site_no"], row["variable"], row["value_column"]
        )
        if (
            series_key not in series_lookup
            or series_lookup[series_key] != row["qualifier_column"]
        ):
            raise OpeningContractError(
                "conflict constituent columns differ from raw series registry"
            )
    conflicts.sort(key=lambda row: (row["cohort"], row["site_no"], row["variable"]))
    conflict_constituents.sort(key=lambda row: (
        row["cohort"], row["site_no"], row["date"], row["variable"],
        row["series_id"],
    ))
    value_status_day_counts.sort(key=lambda row: (
        row["cohort"], row["site_no"], row["variable"], row["value_status"]
    ))
    return {
        "format": OUTCOME_QUALITY_AUDIT_FORMAT,
        "contract_sha256": sha256_json(contract),
        "grouping": list(contract["mandatory_quality_audit"]["grouping"]),
        "parameter_units": dict(contract["parameter_units"]),
        "primary_value_policy": contract["primary_value_policy"],
        "wlevel_policy": contract["wlevel_policy"],
        "daily_time_support": contract["daily_time_support"],
        "sensor_and_datum_limitation": contract["sensor_and_datum_limitation"],
        "counts": rows,
        "value_status_day_counts": value_status_day_counts,
        "series_columns": series_columns,
        "conflicts": conflicts,
        "conflict_constituents": conflict_constituents,
    }


def _approved_target_sensitivity(
    *,
    trusted: Mapping[str, pd.DataFrame],
    normalized: Mapping[str, pd.DataFrame],
    registry: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen exact-A target-only descriptive comparison, without tests."""
    contract = protocol["daily_outcome_quality_contract"][
        "approved_only_target_sensitivity"
    ]
    minimum = int(contract["minimum_valid_targets_per_station_horizon"])
    n_boot = int(
        protocol["primary_inference_contract"]["confidence_interval"]["draws"]
    )
    labels = normalized["temporal"][[
        "site_no", "DATE", "WTEMP_qualifier"
    ]].copy()
    labels["site_no"] = labels.site_no.astype(str)
    labels["DATE"] = pd.to_datetime(labels.DATE)

    def exact_a(value: object) -> bool:
        if pd.isna(value):
            return False
        tokens = {token.strip() for token in str(value).split(",") if token.strip()}
        return tokens == {"A"}

    labels["exact_a"] = labels.WTEMP_qualifier.map(exact_a)
    approved = {
        (str(row.site_no), pd.Timestamp(row.DATE)): bool(row.exact_a)
        for row in labels.itertuples(index=False)
    }
    predictions = trusted["temporal"].copy()
    predictions["site_id"] = predictions.site_id.astype(str)
    predictions["target_date"] = pd.to_datetime(predictions.target_date)
    predictions["exact_a"] = [
        approved.get((site, date), False)
        for site, date in predictions[["site_id", "target_date"]].itertuples(
            index=False, name=None
        )
    ]
    cluster_map = huc2_cluster_map(registry)
    comparisons: list[dict[str, Any]] = []
    for test in _formal_test_registry(protocol):
        horizon = int(test["horizon"])
        candidate = predictions[
            predictions.model.eq(test["candidate"])
            & predictions.horizon.eq(horizon)
            & predictions.exact_a
        ]
        reference = predictions[
            predictions.model.eq(test["reference"])
            & predictions.horizon.eq(horizon)
            & predictions.exact_a
        ]
        key = ["site_id", "issue_date", "target_date"]
        paired = candidate[[*key, "y_true", "y_pred"]].merge(
            reference[[*key, "y_true", "y_pred"]],
            on=key,
            how="outer",
            suffixes=("_candidate", "_reference"),
            indicator=True,
            validate="one_to_one",
        )
        if not paired._merge.eq("both").all() or not np.allclose(
            paired.y_true_candidate.to_numpy(float),
            paired.y_true_reference.to_numpy(float),
            rtol=0.0,
            atol=0.0,
        ):
            raise OpeningContractError("exact-A sensitivity lacks common frozen keys")
        effects: list[float] = []
        sites: list[str] = []
        key_count = 0
        for site_no, group in paired.groupby("site_id", sort=True):
            if len(group) < minimum:
                continue
            candidate_error = (
                group.y_pred_candidate.to_numpy(float)
                - group.y_true_candidate.to_numpy(float)
            )
            reference_error = (
                group.y_pred_reference.to_numpy(float)
                - group.y_true_reference.to_numpy(float)
            )
            effects.append(float(
                np.sqrt(np.mean(candidate_error ** 2))
                - np.sqrt(np.mean(reference_error ** 2))
            ))
            sites.append(str(site_no))
            key_count += len(group)
        clusters = np.asarray([cluster_map[site] for site in sites], dtype=object)
        if not effects or len(set(clusters)) < 2:
            summary: Mapping[str, Any] = {
                "effect": None, "ci_low": None, "ci_high": None,
                "n_stations": len(effects), "n_clusters": len(set(clusters)),
            }
        else:
            summary = cluster_bootstrap_paired_effect(
                np.asarray(effects, dtype=float),
                clusters,
                statistic="median",
                n_boot=n_boot,
                seed=int(test["bootstrap_seed"]),
            )
        comparisons.append({
            "test_id": test["test_id"],
            "candidate": test["candidate"],
            "reference": test["reference"],
            "horizon": horizon,
            "n_stations": int(summary["n_stations"]),
            "n_clusters": int(summary["n_clusters"]),
            "effect_c": summary["effect"],
            "ci_low_c": summary["ci_low"],
            "ci_high_c": summary["ci_high"],
            "n_exact_a_keys": int(key_count),
        })
    return {
        "format": APPROVED_TARGET_SENSITIVITY_FORMAT,
        "role": contract["role"],
        "contract_sha256": sha256_json(contract),
        "key_rule": contract["key_rule"],
        "minimum_valid_targets_per_station_horizon": minimum,
        "retraining_performed": False,
        "p_values_or_holm_computed": False,
        "comparisons": comparisons,
    }


def _spatial_sensitivity(
    *,
    temporal_predictions: pd.DataFrame,
    registry: pd.DataFrame,
    protocol: Mapping[str, Any],
    minimum_targets: int,
) -> dict[str, Any]:
    """Compute the frozen equal-HUC/per-HUC/leave-one-HUC descriptive audit."""
    contract = protocol["primary_inference_contract"][
        "exploratory_spatial_inference_sensitivity"
    ]
    cluster_map = huc2_cluster_map(registry)

    def station_rmse(model: str, horizon: int) -> dict[str, float]:
        selected = temporal_predictions[
            temporal_predictions.model.eq(model)
            & temporal_predictions.horizon.eq(horizon)
        ]
        output: dict[str, float] = {}
        for site, group in selected.groupby("site_id", sort=True):
            if len(group) >= minimum_targets:
                error = group.y_pred.to_numpy(float) - group.y_true.to_numpy(float)
                output[str(site)] = float(np.sqrt(np.mean(error ** 2)))
        return output

    comparisons: list[dict[str, Any]] = []
    for test in sorted(_formal_test_registry(protocol), key=lambda row: row["test_id"]):
        candidate = station_rmse(test["candidate"], int(test["horizon"]))
        reference = station_rmse(test["reference"], int(test["horizon"]))
        sites = sorted(set(candidate) & set(reference))
        effects = {
            site: float(candidate[site] - reference[site]) for site in sites
        }
        full = None if not effects else float(np.median(list(effects.values())))
        by_huc: dict[str, list[float]] = {}
        for site in sites:
            by_huc.setdefault(str(cluster_map[site]), []).append(effects[site])
        per_huc = [
            {
                "huc2": huc,
                "n_stations": len(by_huc[huc]),
                "median_station_effect_c": float(np.median(by_huc[huc])),
            }
            for huc in sorted(by_huc)
        ]
        equal_huc = (
            None if not per_huc else float(np.median([
                row["median_station_effect_c"] for row in per_huc
            ]))
        )
        leave_one: list[dict[str, Any]] = []
        influence: list[float] = []
        for held_out in sorted(by_huc):
            remaining_sites = [
                site for site in sites if str(cluster_map[site]) != held_out
            ]
            remaining_clusters = {
                str(cluster_map[site]) for site in remaining_sites
            }
            remaining_effect = (
                None if not remaining_sites else float(np.median([
                    effects[site] for site in remaining_sites
                ]))
            )
            if remaining_effect is not None and full is not None:
                influence.append(remaining_effect - full)
            leave_one.append({
                "held_out_huc2": held_out,
                "n_remaining_stations": len(remaining_sites),
                "n_remaining_clusters": len(remaining_clusters),
                "station_weighted_median_effect_c": remaining_effect,
                "effect_minus_margin_c": (
                    None if remaining_effect is None
                    else remaining_effect - float(test["margin_c"])
                ),
            })
        comparisons.append({
            "test_id": test["test_id"],
            "candidate": test["candidate"],
            "reference": test["reference"],
            "horizon": int(test["horizon"]),
            "margin_c": float(test["margin_c"]),
            "station_weighted_median_effect_c": full,
            "n_stations": len(sites),
            "n_clusters": len(by_huc),
            "equal_huc_median_effect_c": equal_huc,
            "per_huc": per_huc,
            "leave_one_huc": leave_one,
            "influence_min_c": None if not influence else float(min(influence)),
            "influence_max_c": None if not influence else float(max(influence)),
        })
    return {
        "format": SPATIAL_SENSITIVITY_FORMAT,
        "role": contract["role"],
        "contract_sha256": sha256_json(contract),
        "influence_definition": (
            "leave-one-HUC station-weighted median effect minus the full "
            "station-weighted median effect"
        ),
        "effect_minus_margin_definition": (
            "leave-one-HUC station-weighted median effect minus the test margin"
        ),
        "p_values_confidence_intervals_holm_or_decisions_computed": False,
        "comparisons": comparisons,
    }


def _spatial_cluster_diagnostics(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Derive nondecision small-cluster diagnostics from a spatial row.

    The sealed spatial-sensitivity JSON schema is intentionally left unchanged.
    These quantities are deterministic report annotations derived only from its
    already-frozen per-HUC counts and leave-one-HUC effects.  They do not create
    another p-value, confidence interval, multiplicity adjustment, or decision.
    """
    per_huc = comparison.get("per_huc")
    leave_one = comparison.get("leave_one_huc")
    if not isinstance(per_huc, list) or not isinstance(leave_one, list):
        raise OpeningContractError("spatial comparison lacks cluster diagnostics")
    counts: list[int] = []
    huc_ids: list[str] = []
    for row in per_huc:
        if not isinstance(row, Mapping):
            raise OpeningContractError("spatial per-HUC row is malformed")
        huc = str(row.get("huc2", ""))
        try:
            count = int(row["n_stations"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OpeningContractError("spatial HUC size is malformed") from exc
        if not huc or count <= 0:
            raise OpeningContractError("spatial HUC identity/size is invalid")
        huc_ids.append(huc)
        counts.append(count)
    if len(huc_ids) != len(set(huc_ids)):
        raise OpeningContractError("spatial HUC identities are duplicated")
    try:
        declared_stations = int(comparison["n_stations"])
        declared_clusters = int(comparison["n_clusters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OpeningContractError("spatial comparison totals are malformed") from exc
    if sum(counts) != declared_stations or len(counts) != declared_clusters:
        raise OpeningContractError("spatial comparison totals disagree with per-HUC rows")

    if not counts:
        return {
            "effective_cluster_count_inverse_herfindahl": None,
            "effective_cluster_fraction": None,
            "largest_cluster_share": None,
            "cluster_size_cv": None,
            "loho_direction": "NOT_ESTIMABLE_NO_CLUSTERS",
            "warning_codes": ["NO_REPORTABLE_CLUSTERS"],
            "inference_strength": "NO_STRONG_INFERENCE",
        }
    count_array = np.asarray(counts, dtype=float)
    shares = count_array / float(count_array.sum())
    effective = float(1.0 / np.sum(shares ** 2))
    effective_fraction = float(effective / len(counts))
    largest_share = float(np.max(shares))
    size_cv = float(np.std(count_array, ddof=0) / np.mean(count_array))

    loho_values: list[float] = []
    held_out_ids: list[str] = []
    for row in leave_one:
        if not isinstance(row, Mapping):
            raise OpeningContractError("spatial leave-one-HUC row is malformed")
        held_out = str(row.get("held_out_huc2", ""))
        if not held_out:
            raise OpeningContractError("spatial leave-one-HUC identity is invalid")
        held_out_ids.append(held_out)
        value = row.get("effect_minus_margin_c")
        if value is not None:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise OpeningContractError("spatial leave-one-HUC effect is non-finite")
            loho_values.append(numeric)
    if len(held_out_ids) != len(set(held_out_ids)) or set(held_out_ids) != set(huc_ids):
        raise OpeningContractError("spatial leave-one-HUC rows do not match HUC rows")
    if not loho_values:
        loho_direction = "NOT_ESTIMABLE"
    elif max(loho_values) < 0.0:
        loho_direction = "ALL_BELOW_MARGIN"
    elif min(loho_values) > 0.0:
        loho_direction = "ALL_ABOVE_MARGIN"
    else:
        loho_direction = "CROSSES_OR_TOUCHES_MARGIN"

    warnings_out: list[str] = []
    if len(counts) < 30:
        warnings_out.append("SMALL_CLUSTER_COUNT_LT_30")
    if largest_share >= 0.25:
        warnings_out.append("DOMINANT_CLUSTER_SHARE_GE_0_25")
    if effective_fraction < 0.75:
        warnings_out.append("EFFECTIVE_CLUSTER_FRACTION_LT_0_75")
    if loho_direction in {"CROSSES_OR_TOUCHES_MARGIN", "NOT_ESTIMABLE"}:
        warnings_out.append("LOHO_MARGIN_DIRECTION_UNSTABLE_OR_NOT_ESTIMABLE")
    return {
        "effective_cluster_count_inverse_herfindahl": effective,
        "effective_cluster_fraction": effective_fraction,
        "largest_cluster_share": largest_share,
        "cluster_size_cv": size_cv,
        "loho_direction": loho_direction,
        "warning_codes": warnings_out,
        "inference_strength": (
            "NO_STRONG_INFERENCE" if warnings_out
            else "NO_OBVIOUS_CLUSTER_LEVERAGE"
        ),
    }


def _probabilistic_evaluation(
    *,
    trusted_predictions: Mapping[str, pd.DataFrame],
    suite: Mapping[str, Any],
    availability: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only frozen probabilistic heads and frozen event references.

    Built-in point baselines deliberately have no invented uncertainty or event
    head.  Learned models are required to provide complete q05/q50/q95 and
    calibrated ``p_exceed`` values on every trusted confirmation key.
    """

    contract = protocol["primary_inference_contract"].get(
        "probabilistic_event_contract"
    )
    if not isinstance(contract, Mapping):
        raise OpeningContractError("protocol lacks probabilistic/event contract")
    minimum_targets = int(
        protocol["availability_contract"]
        ["minimum_valid_targets_per_station_horizon"]
    )
    if minimum_targets != 100:
        raise OpeningContractError("probability reportability threshold changed")
    availability_values = availability.copy()
    availability_values["site_no"] = availability_values.site_no.astype(str)
    availability_values["cohort"] = availability_values.cohort.astype(str)
    availability_values["horizon"] = pd.to_numeric(
        availability_values.horizon, errors="raise"
    ).astype(int)
    availability_values["n_valid_targets"] = pd.to_numeric(
        availability_values.n_valid_targets, errors="raise"
    ).astype(int)
    reportable_flags = availability_values.reportable.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if reportable_flags.isna().any() or not np.array_equal(
        reportable_flags.to_numpy(bool),
        availability_values.n_valid_targets.ge(minimum_targets).to_numpy(),
    ):
        raise OpeningContractError(
            "probability evaluation received a changed availability registry"
        )
    availability_values["reportable_bool"] = reportable_flags.to_numpy(bool)

    def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
        value = np.asarray(values, dtype=float)
        weight = np.asarray(weights, dtype=float)
        if value.shape != weight.shape or not (
            np.isfinite(value).all()
            and np.isfinite(weight).all()
            and (weight > 0).all()
            and np.isclose(weight.sum(), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise OpeningContractError("station-balanced metric weights are invalid")
        return float(np.sum(value * weight))

    def reliability_bins(
        outcomes: np.ndarray,
        probabilities: np.ndarray,
        weights: np.ndarray,
        sites: np.ndarray,
    ) -> tuple[list[dict[str, Any]], float]:
        probability = np.clip(
            np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6
        )
        event = np.asarray(outcomes, dtype=float)
        weight = np.asarray(weights, dtype=float)
        edges = np.linspace(0.0, 1.0, 11)
        assignments = np.clip(np.digitize(probability, edges[1:-1]), 0, 9)
        rows: list[dict[str, Any]] = []
        ece = 0.0
        for index in range(10):
            selected = assignments == index
            count = int(selected.sum())
            bin_weight = float(weight[selected].sum())
            mean_probability = (
                None
                if not count
                else float(np.average(probability[selected], weights=weight[selected]))
            )
            event_rate = (
                None
                if not count
                else float(np.average(event[selected], weights=weight[selected]))
            )
            if event_rate is not None and mean_probability is not None:
                ece += bin_weight * abs(
                    event_rate - mean_probability
                )
            rows.append({
                "bin_index": index + 1,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "upper_bound_inclusive": index == 9,
                "n": count,
                "n_sites": int(np.unique(sites[selected]).size),
                "station_balanced_weight": bin_weight,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
            })
        if not np.isclose(
            sum(row["station_balanced_weight"] for row in rows),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise OpeningContractError("reliability-bin weights do not sum to one")
        return rows, float(ece)

    required_by_cohort = suite["required_models"]
    metadata_by_cohort = suite["metadata"]
    cohort_contracts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    metric_names = (
        "coverage_90",
        "mean_interval_width_c",
        "pinball_q05_c",
        "pinball_q50_c",
        "pinball_q95_c",
        "equal_weight_three_quantile_pinball_mean_c",
        "brier_score",
        "frozen_reference_brier_score",
        "brier_skill_frozen_seasonal",
        "log_loss",
        "auroc",
        "auprc",
        "ece_10_equal_width",
        "calibration_intercept",
        "calibration_slope",
        "event_rate",
    )
    for cohort in ("temporal", "external"):
        frame = trusted_predictions[cohort]
        expected_models = tuple(str(value) for value in required_by_cohort[cohort])
        if set(frame.model.astype(str)) != set(expected_models):
            raise OpeningContractError(
                f"probability evaluation {cohort} model registry changed"
            )
        metadata = metadata_by_cohort[cohort]
        primary = metadata.get("ThermoRoute")
        if not isinstance(primary, Mapping):
            raise OpeningContractError(
                f"probability evaluation {cohort} lacks primary metadata"
            )
        thresholds_raw = primary.get("event_thresholds")
        reference = primary.get("event_reference_climatology")
        if not isinstance(thresholds_raw, Mapping) or not isinstance(reference, Mapping):
            raise OpeningContractError(
                f"probability evaluation {cohort} lacks a frozen event reference"
            )
        thresholds = {
            str(site): float(value) for site, value in thresholds_raw.items()
        }
        if not thresholds or not np.isfinite(
            np.asarray(list(thresholds.values()), dtype=float)
        ).all():
            raise OpeningContractError(
                f"probability evaluation {cohort} threshold registry is invalid"
            )
        expected_sites = set(frame.site_id.astype(str))
        external = cohort == "external"
        try:
            validate_frozen_seasonal_event_reference(
                reference,
                expected_sites=None if external else expected_sites,
                pooled=external,
            )
        except ValueError as exc:
            raise OpeningContractError(
                f"probability evaluation {cohort} event reference is invalid"
            ) from exc
        if external:
            if set(thresholds) != {"__pooled__"}:
                raise OpeningContractError(
                    "external probability evaluation threshold is not pooled"
                )
            threshold_scope = "pooled_development_train_q90"
            interpretation = (
                "exploratory pooled statistical tail threshold; non-ecological "
                "and not a waterbody-specific standard"
            )
        else:
            if set(thresholds) != expected_sites:
                raise OpeningContractError(
                    "temporal probability threshold site registry changed"
                )
            threshold_scope = "station_specific_development_train_q90"
            interpretation = (
                "site-local statistical tail diagnostic; not biological, "
                "regulatory, or cross-station comparable"
            )
        for model_id, model_metadata in metadata.items():
            if (
                model_metadata.get("event_thresholds") != thresholds_raw
                or model_metadata.get("event_reference_climatology") != reference
            ):
                raise OpeningContractError(
                    f"{cohort}/{model_id} event threshold/reference differs from primary"
                )
        cohort_contracts[cohort] = {
            "threshold_scope": threshold_scope,
            "threshold_registry_sha256": sha256_json(thresholds),
            "event_reference_format": reference["format"],
            "event_reference_mode": reference["mode"],
            "event_reference_sha256": sha256_json(reference),
            "event_reference_fit_interval": list(reference["fit_interval"]),
            "event_reference_fit_observation_count": int(
                reference["fit_observation_count"]
            ),
            "interpretation": interpretation,
        }

        for model in expected_models:
            for horizon in (1, 3, 7):
                all_selected = frame[
                    frame.model.astype(str).eq(model)
                    & frame.horizon.astype(int).eq(horizon)
                ].copy()
                if all_selected.empty:
                    raise OpeningContractError(
                        f"probability evaluation lacks {cohort}/{model}/h{horizon}"
                    )
                reportable_sites = set(availability_values.loc[
                    availability_values.cohort.eq(cohort)
                    & availability_values.horizon.eq(horizon)
                    & availability_values.reportable_bool,
                    "site_no",
                ])
                selected = all_selected[
                    all_selected.site_id.astype(str).isin(reportable_sites)
                ].copy()
                site_counts = selected.site_id.astype(str).value_counts().sort_index()
                if (
                    set(site_counts.index) != reportable_sites
                    or site_counts.lt(minimum_targets).any()
                ):
                    raise OpeningContractError(
                        "probability reportable-site keys differ from availability"
                    )
                n_sites = int(len(site_counts))
                if n_sites:
                    raw_weights = selected.site_id.astype(str).map(
                        {site: 1.0 / int(count) for site, count in site_counts.items()}
                    ).to_numpy(float)
                    weights = raw_weights / raw_weights.sum()
                    site_total_weights = pd.DataFrame({
                        "site": selected.site_id.astype(str).to_numpy(),
                        "weight": weights,
                    }).groupby("site", sort=True).weight.sum().to_numpy(float)
                    if not np.allclose(
                        site_total_weights,
                        np.full(n_sites, 1.0 / n_sites),
                        rtol=1e-12,
                        atol=1e-12,
                    ):
                        raise OpeningContractError(
                            "probability evaluation does not weight stations equally"
                        )
                else:
                    weights = np.asarray([], dtype=float)
                    site_total_weights = np.asarray([], dtype=float)
                base: dict[str, Any] = {
                    "cohort": cohort,
                    "model": model,
                    "horizon": horizon,
                    "n_forecasts_before_reportability_filter": int(
                        len(all_selected)
                    ),
                    "n_forecasts": int(len(selected)),
                    "n_sites_before_reportability_filter": int(
                        all_selected.site_id.astype(str).nunique()
                    ),
                    "n_sites": n_sites,
                    "minimum_targets_per_retained_site": minimum_targets,
                    "station_balanced_weight_sum": (
                        0.0 if not n_sites else float(weights.sum())
                    ),
                    "minimum_site_total_weight": (
                        None if not n_sites else float(site_total_weights.min())
                    ),
                    "maximum_site_total_weight": (
                        None if not n_sites else float(site_total_weights.max())
                    ),
                    "threshold_scope": threshold_scope,
                }
                if model in BUILTIN_MODELS:
                    optional = all_selected[["q05", "q50", "q95", "p_exceed"]]
                    if optional.notna().any().any():
                        raise OpeningContractError(
                            f"builtin {cohort}/{model} unexpectedly has probability heads"
                        )
                    rows.append({
                        **base,
                        "status": "NOT_AVAILABLE",
                        "reason": "POINT_ONLY_BUILTIN_HAS_NO_FROZEN_PROBABILISTIC_HEAD",
                        "event_count": None,
                        "non_event_count": None,
                        **{name: None for name in metric_names},
                        "undefined_metric_reasons": {
                            name: "POINT_ONLY_BUILTIN_HAS_NO_FROZEN_PROBABILISTIC_HEAD"
                            for name in metric_names
                        },
                        "reliability_bins": [],
                    })
                    continue
                if not n_sites:
                    rows.append({
                        **base,
                        "status": "NOT_ESTIMABLE",
                        "reason": "NO_STATION_HAS_100_COMMON_TARGETS",
                        "event_count": None,
                        "non_event_count": None,
                        **{name: None for name in metric_names},
                        "undefined_metric_reasons": {
                            name: "NO_STATION_HAS_100_COMMON_TARGETS"
                            for name in metric_names
                        },
                        "reliability_bins": [],
                    })
                    continue
                values = selected[
                    ["y_true", "q05", "q50", "q95", "p_exceed"]
                ].to_numpy(float)
                if not np.isfinite(values).all():
                    raise OpeningContractError(
                        f"learned {cohort}/{model}/h{horizon} probability heads are incomplete"
                    )
                y, q05, q50, q95, probability = values.T
                if not ((q05 <= q50).all() and (q50 <= q95).all()):
                    raise OpeningContractError(
                        f"learned {cohort}/{model}/h{horizon} quantiles cross"
                    )
                sites = selected.site_id.astype(str).to_numpy()
                if external:
                    threshold = np.full(len(selected), thresholds["__pooled__"])
                else:
                    threshold = np.asarray(
                        [thresholds[site] for site in sites], dtype=float
                    )
                event = (y > threshold).astype(int)
                reference_probability = predict_frozen_seasonal_event_reference(
                    reference,
                    sites,
                    pd.to_datetime(selected.target_date).to_numpy(),
                )
                if not (
                    np.isfinite(reference_probability).all()
                    and ((0.0 < reference_probability)
                         & (reference_probability < 1.0)).all()
                ):
                    raise OpeningContractError(
                        "frozen event reference produced an invalid probability"
                    )
                p_clip = np.clip(probability, 1e-6, 1.0 - 1e-6)
                pinball_losses = {
                    "pinball_q05_c": np.maximum(
                        0.05 * (y - q05), -0.95 * (y - q05)
                    ),
                    "pinball_q50_c": np.maximum(
                        0.50 * (y - q50), -0.50 * (y - q50)
                    ),
                    "pinball_q95_c": np.maximum(
                        0.95 * (y - q95), -0.05 * (y - q95)
                    ),
                }
                pinballs = {
                    name: weighted_mean(loss, weights)
                    for name, loss in pinball_losses.items()
                }
                three_quantile_mean = float(np.mean(list(pinballs.values())))
                bins, ece = reliability_bins(
                    event, probability, weights, sites
                )
                brier = weighted_mean((probability - event) ** 2, weights)
                reference_brier = weighted_mean(
                    (reference_probability - event) ** 2, weights
                )
                if not reference_brier > 0.0:
                    raise OpeningContractError(
                        "frozen seasonal reference has zero weighted Brier score"
                    )
                brier_skill = float(1.0 - brier / reference_brier)
                undefined: dict[str, str] = {}
                if np.unique(event).size < 2:
                    auroc = auprc = intercept = slope = None
                    for name in (
                        "auroc", "auprc", "calibration_intercept",
                        "calibration_slope",
                    ):
                        undefined[name] = "SINGLE_CLASS_RETAINED_OUTCOMES"
                else:
                    auroc = float(roc_auc_score(
                        event, probability, sample_weight=weights
                    ))
                    auprc = float(average_precision_score(
                        event, probability, sample_weight=weights
                    ))
                    try:
                        calibration = LogisticRegression(
                            C=1e6, solver="lbfgs", max_iter=2000
                        )
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", ConvergenceWarning)
                            calibration.fit(
                                logit(probability).reshape(-1, 1),
                                event,
                                sample_weight=weights,
                            )
                        if any(
                            issubclass(item.category, ConvergenceWarning)
                            for item in caught
                        ):
                            raise ValueError(
                                "weighted calibration regression did not converge"
                            )
                        intercept = float(calibration.intercept_[0])
                        slope = float(calibration.coef_[0, 0])
                        if not np.isfinite([intercept, slope]).all():
                            raise ValueError("non-finite weighted calibration fit")
                    except (RuntimeError, ValueError):
                        intercept = slope = None
                        undefined["calibration_intercept"] = (
                            "WEIGHTED_CALIBRATION_REGRESSION_NOT_ESTIMABLE"
                        )
                        undefined["calibration_slope"] = (
                            "WEIGHTED_CALIBRATION_REGRESSION_NOT_ESTIMABLE"
                        )
                rows.append({
                    **base,
                    "status": "AVAILABLE",
                    "reason": None,
                    "event_count": int(event.sum()),
                    "non_event_count": int(len(event) - event.sum()),
                    "coverage_90": weighted_mean(
                        ((y >= q05) & (y <= q95)).astype(float), weights
                    ),
                    "mean_interval_width_c": weighted_mean(
                        q95 - q05, weights
                    ),
                    **pinballs,
                    "equal_weight_three_quantile_pinball_mean_c": (
                        three_quantile_mean
                    ),
                    "brier_score": float(brier),
                    "frozen_reference_brier_score": float(reference_brier),
                    "brier_skill_frozen_seasonal": float(brier_skill),
                    "log_loss": weighted_mean(
                        -(event * np.log(p_clip)
                          + (1 - event) * np.log(1 - p_clip)),
                        weights,
                    ),
                    "auroc": auroc,
                    "auprc": auprc,
                    "ece_10_equal_width": float(ece),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "event_rate": weighted_mean(event, weights),
                    "undefined_metric_reasons": undefined,
                    "reliability_bins": bins,
                })
    return {
        "format": PROBABILISTIC_EVALUATION_FORMAT,
        "role": contract["role"],
        "contract_sha256": sha256_json(contract),
        "probabilistic_heads": ["q05", "q50", "q95", "p_exceed"],
        "aggregation": contract["aggregation"],
        "minimum_valid_targets_per_station_horizon": minimum_targets,
        "metric_weighting": (
            "station-balanced: each retained station total weight is 1/n_sites"
        ),
        "central_interval_nominal_coverage": 0.90,
        "interval_coverage_claim": (
            "station-balanced empirical marginal coverage only; no "
            "conditional-coverage or exchangeability guarantee"
        ),
        "three_quantile_score_definition": (
            "unscaled equal-weight arithmetic mean of q05/q50/q95 pinball loss"
        ),
        "three_quantile_score_is_crps": False,
        "event_probability_calibration_period": "2018_only_before_confirmation",
        "evaluation_calibration_regression": (
            "weighted logistic regression of event on clipped forecast logit; "
            "sklearn lbfgs, C=1e6, max_iter=2000"
        ),
        "event_probability_clip_for_log_and_calibration_diagnostics": [1e-6, 1 - 1e-6],
        "reliability_bins": "10_equal_width_bins_on_[0,1]",
        "single_class_auroc_auprc_and_calibration_parameters": "NA",
        "brier_skill_reference": (
            "bundle-frozen seasonal development train/calibration climatology"
        ),
        "confirmation_event_rate_used_as_brier_reference": False,
        "rev_status": "REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS",
        "inference_computed": False,
        "cohort_contracts": cohort_contracts,
        "rows": rows,
    }


def _render_confirmatory_report(
    *,
    opening_id: str,
    statistics: Mapping[str, Any],
    availability: pd.DataFrame,
    trusted_predictions: Mapping[str, pd.DataFrame],
    required_models: Mapping[str, Sequence[str]],
    inference_gate: Mapping[str, Any],
    outcome_quality_audit: Mapping[str, Any],
    outcome_qc_gate: Mapping[str, Any],
    approved_target_sensitivity: Mapping[str, Any],
    spatial_sensitivity: Mapping[str, Any],
    probabilistic_evaluation: Mapping[str, Any],
    transport_summary: Mapping[str, Any],
) -> bytes:
    reportable_flags = availability.reportable.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if reportable_flags.isna().any():
        raise OpeningContractError("report received a malformed availability flag")
    if (
        not isinstance(inference_gate.get("status"), str)
        or not isinstance(inference_gate.get("claim_eligible"), bool)
        or not isinstance(
            outcome_qc_gate.get("directional_claims_allowed_by_outcome_qc"),
            bool,
        )
    ):
        raise OpeningContractError("report received a malformed claim gate")
    combined_directional_gate = bool(
        inference_gate["claim_eligible"]
        and outcome_qc_gate["directional_claims_allowed_by_outcome_qc"]
    )
    reportable = int(reportable_flags.sum())
    reportable_registry = availability.copy()
    reportable_registry["site_no"] = reportable_registry.site_no.astype(str)
    reportable_registry["horizon"] = pd.to_numeric(
        reportable_registry.horizon
    ).astype(int)
    reportable_registry["reportable_bool"] = reportable_flags.to_numpy(bool)
    lines = [
        "# Route-A one-time confirmatory result",
        "",
        f"Opening ID: `{opening_id}`",
        "",
        (
            "All predictions and formal tests below were regenerated by the fixed "
            "fresh-process trusted scorer from immutable raw NWIS bytes."
        ),
        "",
        f"Reportable station–horizon cells: {reportable}/{len(availability)}.",
        "",
        "## Directional-result gate status",
        "",
        (
            f"Frozen inference-assumption gate: `{inference_gate['status']}`; "
            f"claim eligible: `{inference_gate['claim_eligible']}`."
        ),
        (
            "Predeclared gross-plausibility and aggregate-sensitivity outcome "
            "gate: "
            f"`{outcome_qc_gate['status']}`; directional wording permitted by "
            "that gate: "
            f"`{outcome_qc_gate['directional_claims_allowed_by_outcome_qc']}`."
        ),
        f"Combined directional-result gate: `{combined_directional_gate}`.",
        (
            "The five unfiltered fixed-cohort effects remain reportable. P-values, "
            "intervals, and margin flags are assumption-conditional diagnostics; "
            "they are not a directional verdict unless the combined gate is true "
            "and the separate deterministic claim ledger renders the wording."
        ),
        "",
        "## Fixed-ledger acquisition transport",
        "",
        (
            f"Opening count: {transport_summary['opening_count']}; transport "
            f"attempts: {transport_summary['attempt_count']}; same-opening resumes: "
            f"{transport_summary['resume_count']}; transactions already complete "
            "before the final attempt: "
            f"{len(transport_summary['completed_before_final_attempt_request_sha256'])}."
        ),
        (
            "Raw retrieval span (UTC): "
            f"`{transport_summary['retrieval_span_utc']['first']}` to "
            f"`{transport_summary['retrieval_span_utc']['last']}`. No completed "
            "response, station, or request was replaced."
        ),
        "",
        "## Predeclared five-test family",
        "",
        (
            "| Test | Candidate | Reference | h | Margin (°C) | Status | "
            "Effect (°C) | 95% CI (°C) | Stations | HUC2 | Win rate | "
            "Raw p | Holm p | Holm≤0.05 flag | CI-high<margin flag |"
        ),
        (
            "|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|"
            "---|---|"
        ),
    ]

    def number(value: object, *, digits: int = 6) -> str:
        if value is None:
            return "NA"
        return f"{float(cast(Any, value)):.{digits}g}"

    for row in statistics["tests"]:
        interval = (
            "NA"
            if row["ci_low_c"] is None or row["ci_high_c"] is None
            else f"[{number(row['ci_low_c'])}, {number(row['ci_high_c'])}]"
        )
        lines.append(
            f"| {row['test_id']} | {row['candidate']} | {row['reference']} | "
            f"{row['horizon']} | {row['margin_c']:.3f} | {row['status']} | "
            f"{number(row['median_effect_c'])} | {interval} | "
            f"{row['n_stations']} | {row['n_clusters']} | "
            f"{number(row['win_rate'])} | {number(row['p_one_sided_raw'])} | "
            f"{number(row['p_holm'])} | {row['reject_at_0_05']} | "
            f"{row['confidence_bound_supports_margin']} |"
        )

    lines.extend(["", "## All frozen models", ""])
    for cohort in ("temporal", "external"):
        frame = trusted_predictions[cohort]
        actual_models = set(frame.model.astype(str))
        expected_models = tuple(str(value) for value in required_models[cohort])
        if actual_models != set(expected_models):
            raise OpeningContractError(
                f"report {cohort} model registry differs from frozen suite"
            )
        role = (
            "primary temporal cohort (architecture controls are exploratory)"
            if cohort == "temporal"
            else "external exploratory history-dependent new-gage cohort"
        )
        lines.extend([
            f"### {cohort}: {role}",
            "",
            (
                "The pooled daily metrics are descriptive micro-averages. The "
                "station median gives each reportable station equal weight."
            ),
            "",
            (
                "| Model | h | Forecasts | Sites | Pooled RMSE (°C) | "
                "Pooled MAE (°C) | Pooled bias (°C) | Reportable stations | "
                "Median station RMSE (°C) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for model in expected_models:
            for horizon in (1, 3, 7):
                selected = frame[
                    frame.model.astype(str).eq(model)
                    & frame.horizon.astype(int).eq(horizon)
                ]
                if selected.empty:
                    raise OpeningContractError(
                        f"report lacks {cohort}/{model}/h{horizon} predictions"
                    )
                error = (
                    selected.y_pred.to_numpy(float)
                    - selected.y_true.to_numpy(float)
                )
                reportable_sites = set(
                    reportable_registry.loc[
                        reportable_registry.cohort.astype(str).eq(cohort)
                        & reportable_registry.horizon.eq(horizon)
                        & reportable_registry.reportable_bool,
                        "site_no",
                    ]
                )
                station_rmse = []
                for site, group in selected.groupby("site_id", sort=True):
                    if str(site) not in reportable_sites:
                        continue
                    station_error = (
                        group.y_pred.to_numpy(float)
                        - group.y_true.to_numpy(float)
                    )
                    station_rmse.append(float(np.sqrt(np.mean(station_error ** 2))))
                lines.append(
                    f"| {model} | {horizon} | {len(selected)} | "
                    f"{selected.site_id.astype(str).nunique()} | "
                    f"{np.sqrt(np.mean(error ** 2)):.6f} | "
                    f"{np.mean(np.abs(error)):.6f} | {np.mean(error):.6f} | "
                    f"{len(station_rmse)} | "
                    f"{number(None if not station_rmse else np.median(station_rmse))} |"
                )
        lines.append("")
    lines.extend([
        "The external cohort and temporal architecture controls are descriptive only; "
        "they are not added to the primary five-test Holm family.",
        "",
        "## Probabilistic and high-temperature event evaluation (exploratory)",
        "",
        (
            "Every metric retains only station–horizon cells with at least 100 "
            "common targets and gives every retained station equal total weight. "
            "Coverage is empirical marginal 90% interval coverage under that "
            "station-balanced weighting. The three-quantile number is the unscaled "
            "equal-weight mean of q05/q50/q95 pinball losses; it is not CRPS. "
            "Point-only built-ins remain NOT_AVAILABLE rather than receiving "
            "invented uncertainty heads."
        ),
        "",
        (
            "| Cohort | Model | h | Status | n | 90% coverage | Width (°C) | "
            "Mean 3Q pinball (°C) |"
        ),
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in probabilistic_evaluation["rows"]:
        lines.append(
            f"| {row['cohort']} | {row['model']} | {row['horizon']} | "
            f"{row['status']} | {row['n_forecasts']} | "
            f"{number(row['coverage_90'])} | "
            f"{number(row['mean_interval_width_c'])} | "
            f"{number(row['equal_weight_three_quantile_pinball_mean_c'])} |"
        )
    lines.extend([
        "",
        (
            "Brier skill below uses only the seasonal climatology frozen in each "
            "model bundle; the confirmation-period event rate is never the reference. "
            "ECE uses 10 equal-width bins. AUROC, AUPRC, and calibration parameters "
            "are NA for a single-class evaluation slice."
        ),
        "",
        (
            "| Cohort | Model | h | Brier | Brier skill | Log loss | AUROC | "
            "AUPRC | ECE | Cal intercept | Cal slope |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in probabilistic_evaluation["rows"]:
        lines.append(
            f"| {row['cohort']} | {row['model']} | {row['horizon']} | "
            f"{number(row['brier_score'])} | "
            f"{number(row['brier_skill_frozen_seasonal'])} | "
            f"{number(row['log_loss'])} | {number(row['auroc'])} | "
            f"{number(row['auprc'])} | {number(row['ece_10_equal_width'])} | "
            f"{number(row['calibration_intercept'])} | "
            f"{number(row['calibration_slope'])} |"
        )
    lines.extend([
        "",
        (
            "Temporal events use station-specific development-train q90 thresholds. "
            "External events use one pooled development-train q90 and are exploratory, "
            "non-ecological, non-regulatory, and not cross-waterbody standards. Full "
            "10-bin reliability counts and rates are retained in the immutable "
            "probabilistic evaluation artifact."
        ),
        "",
        (
            "REV status: `REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS`; no "
            "cost-loss ratio was selected after opening."
        ),
        "",
        "## Frozen-cohort availability",
        "",
        "| Cohort | h | Reportable/total | Min keys | Median keys | Max keys |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    availability_values = availability.copy()
    availability_values["reportable_bool"] = reportable_flags.to_numpy(bool)
    availability_values["n_valid_targets"] = pd.to_numeric(
        availability_values.n_valid_targets
    )
    for (cohort, horizon), group in availability_values.groupby(
        ["cohort", "horizon"], sort=True
    ):
        lines.append(
            f"| {cohort} | {int(horizon)} | "
            f"{int(group.reportable_bool.sum())}/{len(group)} | "
            f"{int(group.n_valid_targets.min())} | "
            f"{float(group.n_valid_targets.median()):.1f} | "
            f"{int(group.n_valid_targets.max())} |"
        )
    lines.extend([
        "",
        "Availability is never used for site replacement or model selection.",
        "",
        "## NWIS outcome-quality audit",
        "",
        "Primary WTEMP/FLOW results use every finite parsed 00003 daily mean; "
        "qualifiers do not filter dates or sites. WLEVEL is retained but not consumed.",
        "",
        "| Cohort | Variable | Value status | Days |",
        "|---|---|---|---:|",
    ])
    audit_counts = pd.DataFrame(
        outcome_quality_audit["value_status_day_counts"]
    )
    for (cohort, variable, status), group in audit_counts.groupby(
        ["cohort", "variable", "value_status"], sort=True
    ):
        lines.append(
            f"| {cohort} | {variable} | {status} | "
            f"{int(group['day_count'].sum())} |"
        )
    lines.extend([
        "",
        "Unknown qualifier strings are preserved and counted without interpretation. "
        "A simultaneous multi-series conflict is set missing; series are never "
        "averaged. Every conflict constituent retains its exact value/qualifier "
        "column, raw qualifier/value, parsed finite value, and raw-response SHA-256.",
        "",
        "## Predeclared gross-plausibility and aggregate-sensitivity outcome gate",
        "",
        (
            f"Gate status: `{outcome_qc_gate['status']}`; pass: "
            f"`{outcome_qc_gate['pass']}`; directional claims allowed by this "
            "gate: "
            f"`{outcome_qc_gate['directional_claims_allowed_by_outcome_qc']}`."
        ),
        (
            "The primary five effects above remain completely unfiltered. This "
            "audit removes no primary row or site, changes no model, and performs "
            "no retraining or recalibration. A failed component withholds "
            "directional wording even if a p-value or interval appears favorable."
        ),
        (
            "This is a deliberately narrow reporting gate, not a complete outcome-"
            "quality certification. It does not establish the absence of in-range "
            "unit mistakes, sensor drift, step changes, flatlines, systematic "
            "qualifier problems, or station-level influences hidden by aggregation. "
            "The separate NWIS audit above reports qualifier and multi-series-conflict "
            "counts, but those counts are not thresholded by this gate."
        ),
        (
            "Finite confirmation-period WTEMP values checked against the frozen "
            f"[{number(outcome_qc_gate['target_plausibility']['lower_inclusive_c'])}, "
            f"{number(outcome_qc_gate['target_plausibility']['upper_inclusive_c'])}] "
            "°C plausibility range: "
            f"{outcome_qc_gate['target_plausibility']['finite_confirmation_values_checked']}; "
            "outside-range values retained and flagged: "
            f"{outcome_qc_gate['target_plausibility']['outside_range_count']}."
        ),
        "",
        (
            "| Test | h | Primary effect (°C) | One extreme/station deleted "
            "effect (°C) | Absolute change (°C) | Margin direction stable | "
            "Max selected combined-SSE share | Pass |"
        ),
        "|---|---:|---:|---:|---:|---|---:|---|",
    ])
    for row in outcome_qc_gate["single_extreme_influence"]:
        lines.append(
            f"| {row['test_id']} | {row['horizon']} | "
            f"{number(row['primary_unfiltered_effect_c'])} | "
            f"{number(row['one_extreme_per_station_deleted_effect_c'])} | "
            f"{number(row['absolute_effect_change_c'])} | "
            f"{row['margin_direction_stable']} | "
            f"{number(row['maximum_selected_combined_sse_share'])} | "
            f"{row['pass']} |"
        )
    lines.extend([
        "",
        "| Test | Full margin direction | All leave-one-HUC directions match | Pass |",
        "|---|---|---|---|",
    ])
    for row in outcome_qc_gate["leave_one_huc_direction"]:
        lines.append(
            f"| {row['test_id']} | {row['full_margin_direction']} | "
            f"{row['all_huc_deletions_match_full_margin_direction']} | "
            f"{row['pass']} |"
        )
    lines.extend([
        "",
        "## Exact-A target-only sensitivity (descriptive)",
        "",
        (
            "This sensitivity changes only the target-label subset. It does not refit "
            "models and creates no p-values, Holm decisions, or new confirmatory tests."
        ),
        "",
        "| Test | Candidate | Reference | h | Stations | HUC2 | Exact-A keys | Effect (°C) | 95% CI (°C) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in approved_target_sensitivity["comparisons"]:
        interval = (
            "NA"
            if row["ci_low_c"] is None or row["ci_high_c"] is None
            else f"[{number(row['ci_low_c'])}, {number(row['ci_high_c'])}]"
        )
        lines.append(
            f"| {row['test_id']} | {row['candidate']} | {row['reference']} | "
            f"{row['horizon']} | {row['n_stations']} | {row['n_clusters']} | "
            f"{row['n_exact_a_keys']} | {number(row['effect_c'])} | {interval} |"
        )
    lines.extend([
        "",
        "## Spatial weighting and leave-one-HUC sensitivity (descriptive)",
        "",
        "No p-values, confidence intervals, Holm adjustments, or decisions are "
        "computed in this sensitivity. Exact sign-flip enumeration in the primary "
        "analysis removes Monte Carlo error only; it does not remove the joint "
        "whole-HUC sign-symmetry assumption or repair a small, unequal cluster "
        "design. The diagnostics below are nondecision warnings derived from the "
        "frozen per-HUC counts and leave-one-HUC effects.",
        "",
        (
            "| Test | h | Margin (°C) | Station median (°C) | Equal-HUC "
            "median (°C) | Stations | HUC2 | Effective HUC2 | Largest share | "
            "Size CV | LOHO margin direction | Evidence warning | "
            "LOHO influence range (°C) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for row in spatial_sensitivity["comparisons"]:
        diagnostics = _spatial_cluster_diagnostics(row)
        influence_range = (
            "NA"
            if row["influence_min_c"] is None or row["influence_max_c"] is None
            else f"[{number(row['influence_min_c'])}, {number(row['influence_max_c'])}]"
        )
        lines.append(
            f"| {row['test_id']} | {row['horizon']} | {row['margin_c']:.3f} | "
            f"{number(row['station_weighted_median_effect_c'])} | "
            f"{number(row['equal_huc_median_effect_c'])} | {row['n_stations']} | "
            f"{row['n_clusters']} | "
            f"{number(diagnostics['effective_cluster_count_inverse_herfindahl'])} | "
            f"{number(diagnostics['largest_cluster_share'])} | "
            f"{number(diagnostics['cluster_size_cv'])} | "
            f"{diagnostics['loho_direction']} | "
            f"{diagnostics['inference_strength']} | {influence_range} |"
        )
        warning_codes = diagnostics["warning_codes"]
        lines.append(
            "  - Nondecision cluster warnings for "
            f"`{row['test_id']}`: "
            + (", ".join(warning_codes) if warning_codes else "none")
            + "."
        )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def produce_trusted_opening_products(
    *,
    preflight: Mapping[str, Any],
    root: str | Path,
    output_state_paths: Mapping[str, Any],
) -> OpeningProducts:
    """Create authoritative products only inside a private trusted stage."""
    root = Path(root).resolve()
    (
        normalized,
        temporal_sites,
        external_sites,
        request_rows,
    ) = _verified_acquisition_for_scoring(preflight=preflight, root=root)
    protocol_info = preflight["protocol"]
    retrospective = preflight["inputs"]["document"]["cohort_tables"]
    trusted: dict[str, pd.DataFrame] = {}
    truth: dict[str, pd.DataFrame] = {}
    for cohort, sites in (("temporal", temporal_sites), ("external", external_sites)):
        weather_path = _verify_file_binding(
            root, retrospective[cohort], label=f"{cohort} frozen retrospective inputs"
        )
        combined = _combined_confirmation_panel(
            normalized[cohort],
            _read_table(weather_path),
            feature_order=preflight["suite"]["feature_order"],
        )
        trusted[cohort] = score_frozen_confirmation_suite(
            root=root,
            cohort=cohort,
            combined_panel=combined,
            sites=tuple(sorted(sites)),
            interval=(protocol_info["target_start"], protocol_info["target_end"]),
            suite=preflight["suite"],
        )
        truth[cohort] = trusted[cohort][trusted[cohort].model.eq("ThermoRoute")][
            [*_FORECAST_KEY, "y_true"]
        ].copy()
    minimum_targets = int(
        protocol_info["document"]["availability_contract"][
            "minimum_valid_targets_per_station_horizon"
        ]
    )
    availability = _availability_from_truth(
        truth,
        sites_by_cohort={"temporal": temporal_sites, "external": external_sites},
        minimum_targets=minimum_targets,
    )
    quality_audit = _build_outcome_quality_audit(
        normalized=normalized,
        request_rows=request_rows,
        protocol=protocol_info["document"],
    )
    approved_sensitivity = _approved_target_sensitivity(
        trusted=trusted,
        normalized=normalized,
        registry=preflight["registries"]["development"],
        protocol=protocol_info["document"],
    )
    spatial_sensitivity = _spatial_sensitivity(
        temporal_predictions=trusted["temporal"],
        registry=preflight["registries"]["development"],
        protocol=protocol_info["document"],
        minimum_targets=minimum_targets,
    )
    try:
        outcome_qc_gate = build_outcome_qc_gate_document(
            root=root,
            policy_path=root
            / preflight["authorization"]["outcome_qc_policy"]["path"],
            protocol=protocol_info["document"],
            temporal_predictions=trusted["temporal"],
            normalized_temporal=normalized["temporal"],
            spatial_sensitivity=spatial_sensitivity,
            minimum_targets=minimum_targets,
        )
    except OutcomeQCGateError as exc:
        raise OpeningContractError(
            "frozen outcome-QC gate could not be executed"
        ) from exc
    probabilistic_evaluation = _probabilistic_evaluation(
        trusted_predictions=trusted,
        suite=preflight["suite"],
        availability=availability,
        protocol=protocol_info["document"],
    )
    canonical_state = preflight["state_paths"]
    state = dict(output_state_paths)
    stage = Path(state["availability_registry"]).parent
    expected_stage_state = _trusted_state_at_directory(
        canonical_state, stage
    )
    if any(
        Path(state[key]) != Path(expected_stage_state[key])
        for key in _TRUSTED_STATE_KEYS
    ):
        raise OpeningContractError("trusted staging state-path map changed")
    if Path(state["acquisition_manifest"]) != Path(
        canonical_state["acquisition_manifest"]
    ):
        raise OpeningContractError(
            "trusted staging cannot relocate raw acquisition evidence"
        )
    exclusive_create_json(state["outcome_quality_audit"], quality_audit)
    exclusive_create_json(state["outcome_qc_gate"], outcome_qc_gate)
    exclusive_create_json(
        state["approved_target_sensitivity"], approved_sensitivity
    )
    exclusive_create_json(state["spatial_sensitivity"], spatial_sensitivity)
    exclusive_create_json(
        state["probabilistic_evaluation"], probabilistic_evaluation
    )
    statistics = compute_confirmatory_statistics(
        trusted["temporal"],
        preflight["registries"]["development"],
        protocol_info["document"],
        minimum_targets=minimum_targets,
    )
    statistics["outcome_quality_artifacts"] = {
        "outcome_quality_audit": _logical_binding(
            root,
            state["outcome_quality_audit"],
            canonical_state["outcome_quality_audit"],
        ),
        "approved_target_sensitivity": _logical_binding(
            root,
            state["approved_target_sensitivity"],
            canonical_state["approved_target_sensitivity"],
        ),
        "spatial_sensitivity": _logical_binding(
            root,
            state["spatial_sensitivity"],
            canonical_state["spatial_sensitivity"],
        ),
    }
    statistics["outcome_qc_gate"] = {
        **_logical_binding(
            root,
            state["outcome_qc_gate"],
            canonical_state["outcome_qc_gate"],
        ),
        "format": OUTCOME_QC_GATE_FORMAT,
        "status": outcome_qc_gate["status"],
        "pass": outcome_qc_gate["pass"],
        "directional_claims_allowed": outcome_qc_gate[
            "directional_claims_allowed_by_outcome_qc"
        ],
    }
    statistics["probabilistic_and_event_artifacts"] = {
        "probabilistic_evaluation": _logical_binding(
            root,
            state["probabilistic_evaluation"],
            canonical_state["probabilistic_evaluation"],
        )
    }
    _exclusive_create_bytes(
        state["availability_registry"],
        availability.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    _exclusive_create_parquet(state["temporal_predictions"], trusted["temporal"])
    _exclusive_create_parquet(state["external_predictions"], trusted["external"])
    exclusive_create_json(state["statistics"], statistics)
    _exclusive_create_bytes(
        state["report"],
        _render_confirmatory_report(
            opening_id=preflight["authorization"]["opening_id"],
            statistics=statistics,
            availability=availability,
            trusted_predictions=trusted,
            required_models=preflight["suite"]["required_models"],
            inference_gate=preflight["inference_gate"],
            outcome_quality_audit=quality_audit,
            outcome_qc_gate=outcome_qc_gate,
            approved_target_sensitivity=approved_sensitivity,
            spatial_sensitivity=spatial_sensitivity,
            probabilistic_evaluation=probabilistic_evaluation,
            transport_summary=_load_json(
                state["acquisition_manifest"],
                label="opened acquisition manifest",
            )["transport_summary"],
        ),
    )
    return _opening_products_from_state(state)


def validate_opening_products(
    products: OpeningProducts,
    *,
    preflight: Mapping[str, Any],
    root: str | Path,
    staged: bool = False,
    allow_recoverable_trusted_mode: bool = False,
) -> dict[str, Any]:
    """Validate outcomes, exact keys, all models and recomputed formal tests."""
    root = Path(root).resolve()
    canonical_state = preflight["state_paths"]
    from .outcome_acquisition import _assert_exact_acquisition_directory

    _assert_exact_acquisition_directory(
        Path(canonical_state["acquisition_manifest"]).parent,
        {key: Path(value) for key, value in canonical_state.items()},
    )
    if staged:
        stage_directory = Path(products.availability_registry).parent
        if (
            re.fullmatch(
                r"\.trusted-stage-v1-[0-9a-f]{32}", stage_directory.name
            )
            is None
        ):
            raise OpeningContractError("trusted validation stage is noncanonical")
        product_state = _trusted_state_at_directory(
            canonical_state, stage_directory
        )
        _assert_exact_trusted_directory(stage_directory, canonical_state)
    else:
        product_state = canonical_state
        _assert_exact_trusted_directory(
            _trusted_directory_from_state(canonical_state),
            canonical_state,
            allow_recoverable_canonical_mode=(
                allow_recoverable_trusted_mode
            ),
        )
    expected_product_paths = {
        "acquisition_manifest": canonical_state["acquisition_manifest"],
        **{key: product_state[key] for key in _TRUSTED_STATE_KEYS},
    }
    if any(
        Path(getattr(products, field)).resolve() != Path(expected).resolve()
        for field, expected in expected_product_paths.items()
    ):
        raise OpeningContractError("trusted product path differs from canonical state namespace")
    authorization = preflight["authorization"]
    opening_id = authorization["opening_id"]
    acquisition = _load_json(products.acquisition_manifest, label="opened acquisition manifest")
    if acquisition.get("format") != ACQUISITION_MANIFEST_FORMAT:
        raise OpeningContractError("unsupported opened acquisition-manifest format")
    expected_acquisition = {
        "opening_id": opening_id,
        "authorization_sha256": preflight["authorization_sha256"],
        "protocol_sha256": preflight["protocol"]["protocol_sha256"],
        "labels_state": "OPENED_ONCE",
        "site_replacement_count": 0,
        "response_replacement_count": 0,
        "history_start": authorization["acquisition_plan"]["history_start"],
        "target_start": authorization["acquisition_plan"]["target_start"],
        "target_end": authorization["acquisition_plan"]["target_end"],
        "maximum_response_bytes_per_request": (
            MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
        "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
    }
    wrong = {key: acquisition.get(key) for key, value in expected_acquisition.items()
             if acquisition.get(key) != value}
    if wrong:
        raise OpeningContractError(f"opened acquisition manifest mismatch: {wrong}")
    raw_index = _verify_canonical_file_binding(
        root, acquisition.get("raw_nwis_snapshot_index", {}),
        expected_path=canonical_state["raw_nwis_snapshot_index"],
        label="opened NWIS snapshot index",
    )
    records = _verify_snapshot_index(root, raw_index, prelabel=False)
    temporal_sites = set(preflight["registries"]["development"].site_no.astype(str))
    external_sites = set(preflight["registries"]["external"].site_no.astype(str))
    _verify_opened_nwis_index(
        records,
        expected_sites=temporal_sites | external_sites,
        history_start=authorization["acquisition_plan"]["history_start"],
        target_end=authorization["acquisition_plan"]["target_end"],
    )
    request_map_path = _verify_canonical_file_binding(
        root,
        acquisition.get("request_map", {}),
        expected_path=canonical_state["acquisition_request_map"],
        label="opened NWIS request map",
    )
    _verify_opened_request_map(
        request_map_path,
        records=records,
        opening_id=opening_id,
        authorization_sha256=preflight["authorization_sha256"],
        temporal_sites=temporal_sites,
        external_sites=external_sites,
    )
    request_rows = _load_json(
        request_map_path, label="opened NWIS request map"
    )["requests"]
    transport_summary = _verify_opened_transport_evidence(
        root=root,
        acquisition=acquisition,
        records=records,
        request_rows=request_rows,
        opening_id=opening_id,
        authorization_sha256=preflight["authorization_sha256"],
        work_order_path=Path(preflight["state_paths"]["work_order"]),
        raw_root=raw_index.parent,
    )
    raw_rebuild = _rebuild_opened_nwis_panel(
        raw_index,
        records,
        history_start=authorization["acquisition_plan"]["history_start"],
        target_end=authorization["acquisition_plan"]["target_end"],
    )
    normalized_bindings = acquisition.get("normalized_outcome_tables")
    if not isinstance(normalized_bindings, Mapping) or set(normalized_bindings) != {
        "temporal", "external"
    }:
        raise OpeningContractError(
            "opened acquisition manifest lacks both normalized outcome panels"
        )
    normalized: dict[str, pd.DataFrame] = {}
    normalized_paths: dict[str, Path] = {}
    for cohort, sites in (("temporal", temporal_sites), ("external", external_sites)):
        normalized_path = _verify_canonical_file_binding(
            root,
            normalized_bindings[cohort],
            expected_path=canonical_state[f"{cohort}_outcomes"],
            label=f"{cohort} normalized opened outcomes",
        )
        normalized[cohort] = _load_and_verify_normalized_outcomes(
            normalized_path, raw_rebuild=raw_rebuild, sites=sites
        )
        normalized_paths[cohort] = normalized_path
    availability_path = Path(products.availability_registry)
    if not availability_path.is_file():
        raise OpeningContractError("trusted site/horizon availability registry is absent")
    minimum_targets = int(
        preflight["protocol"]["document"].get("availability_contract", {}).get(
            "minimum_valid_targets_per_station_horizon", 100
        )
    )
    _validate_availability_registry(
        availability_path,
        temporal_sites=temporal_sites,
        external_sites=external_sites,
        minimum_targets=minimum_targets,
    )
    model_registries = authorization["required_models"]
    temporal_models = tuple(model_registries["temporal"])
    external_models = tuple(model_registries["external"])
    protocol_info = preflight["protocol"]
    temporal = validate_prediction_product(
        products.temporal_predictions,
        required_models=temporal_models,
        expected_sites=temporal_sites,
        target_start=protocol_info["target_start"],
        target_end=protocol_info["target_end"],
    )
    external_predictions = validate_prediction_product(
        products.external_predictions,
        required_models=external_models,
        expected_sites=external_sites,
        target_start=protocol_info["target_start"],
        target_end=protocol_info["target_end"],
    )
    retrospective_bindings = preflight["inputs"]["document"]["cohort_tables"]
    reconstructed_windows: dict[str, pd.DataFrame] = {}
    trusted_frames: dict[str, pd.DataFrame] = {}
    trusted_hashes: dict[str, dict[str, Any]] = {}
    for cohort, sites, predictions in (
        ("temporal", temporal_sites, temporal),
        ("external", external_sites, external_predictions),
    ):
        weather_path = _verify_file_binding(
            root,
            retrospective_bindings[cohort],
            label=f"{cohort} frozen retrospective inputs",
        )
        weather = _read_table(weather_path)
        combined = _combined_confirmation_panel(
            normalized[cohort],
            weather,
            feature_order=preflight["suite"]["feature_order"],
        )
        trusted = score_frozen_confirmation_suite(
            root=root,
            cohort=cohort,
            combined_panel=combined,
            sites=tuple(sorted(sites)),
            interval=(protocol_info["target_start"], protocol_info["target_end"]),
            suite=preflight["suite"],
        )
        expected = trusted[trusted.model.eq("ThermoRoute")][
            [*_FORECAST_KEY, "y_true"]
        ].copy()
        _assert_predictions_match_frozen_windows(
            predictions, expected, model="ThermoRoute"
        )
        trusted_hashes[cohort] = {
            "rows": int(len(trusted)),
            "sha256": _assert_worker_predictions_equal_trusted(
                predictions, trusted, cohort=cohort, atol=0.0
            ),
            "schema": R.PREDICTION_SCHEMA_VERSION,
            "ensemble_rule": (
                "mean frozen members, then frozen CQR and horizon Platt calibration"
            ),
        }
        trusted_frames[cohort] = trusted
        reconstructed_windows[cohort] = expected
    _assert_availability_matches_windows(
        availability_path,
        reconstructed_windows,
        sites_by_cohort={"temporal": temporal_sites, "external": external_sites},
    )
    expected_quality_audit = _build_outcome_quality_audit(
        normalized=normalized,
        request_rows=request_rows,
        protocol=protocol_info["document"],
    )
    actual_quality_audit = _load_json(
        products.outcome_quality_audit, label="outcome-quality audit"
    )
    _assert_statistics_equal(expected_quality_audit, actual_quality_audit)
    expected_approved_sensitivity = _approved_target_sensitivity(
        trusted=trusted_frames,
        normalized=normalized,
        registry=preflight["registries"]["development"],
        protocol=protocol_info["document"],
    )
    actual_approved_sensitivity = _load_json(
        products.approved_target_sensitivity,
        label="approved-target sensitivity",
    )
    _assert_statistics_equal(
        expected_approved_sensitivity, actual_approved_sensitivity
    )
    expected_spatial_sensitivity = _spatial_sensitivity(
        temporal_predictions=trusted_frames["temporal"],
        registry=preflight["registries"]["development"],
        protocol=protocol_info["document"],
        minimum_targets=minimum_targets,
    )
    actual_spatial_sensitivity = _load_json(
        products.spatial_sensitivity, label="spatial sensitivity"
    )
    _assert_statistics_equal(
        expected_spatial_sensitivity, actual_spatial_sensitivity
    )
    try:
        expected_outcome_qc_gate = build_outcome_qc_gate_document(
            root=root,
            policy_path=root
            / preflight["authorization"]["outcome_qc_policy"]["path"],
            protocol=protocol_info["document"],
            temporal_predictions=trusted_frames["temporal"],
            normalized_temporal=normalized["temporal"],
            spatial_sensitivity=expected_spatial_sensitivity,
            minimum_targets=minimum_targets,
        )
        actual_outcome_qc_gate = _load_json(
            products.outcome_qc_gate, label="outcome-QC gate"
        )
        validate_outcome_qc_gate_document(
            actual_outcome_qc_gate,
            root=root,
            policy_path=root
            / preflight["authorization"]["outcome_qc_policy"]["path"],
            protocol=protocol_info["document"],
            temporal_predictions=trusted_frames["temporal"],
            normalized_temporal=normalized["temporal"],
            spatial_sensitivity=expected_spatial_sensitivity,
            minimum_targets=minimum_targets,
        )
    except OutcomeQCGateError as exc:
        raise OpeningContractError(
            "outcome-QC gate differs from trusted recomputation"
        ) from exc
    _assert_statistics_equal(
        expected_outcome_qc_gate, actual_outcome_qc_gate
    )
    expected_probabilistic_evaluation = _probabilistic_evaluation(
        trusted_predictions=trusted_frames,
        suite=preflight["suite"],
        availability=pd.read_csv(
            availability_path,
            dtype={"site_no": "string"},
            keep_default_na=False,
        ),
        protocol=protocol_info["document"],
    )
    actual_probabilistic_evaluation = _load_json(
        products.probabilistic_evaluation,
        label="probabilistic and event evaluation",
    )
    _assert_statistics_equal(
        expected_probabilistic_evaluation, actual_probabilistic_evaluation
    )
    expected_statistics = compute_confirmatory_statistics(
        trusted_frames["temporal"],
        preflight["registries"]["development"],
        protocol_info["document"],
        minimum_targets=minimum_targets,
    )
    expected_statistics["outcome_quality_artifacts"] = {
        "outcome_quality_audit": _logical_binding(
            root,
            products.outcome_quality_audit,
            canonical_state["outcome_quality_audit"],
        ),
        "approved_target_sensitivity": _logical_binding(
            root,
            products.approved_target_sensitivity,
            canonical_state["approved_target_sensitivity"],
        ),
        "spatial_sensitivity": _logical_binding(
            root,
            products.spatial_sensitivity,
            canonical_state["spatial_sensitivity"],
        ),
    }
    expected_statistics["outcome_qc_gate"] = {
        **_logical_binding(
            root,
            products.outcome_qc_gate,
            canonical_state["outcome_qc_gate"],
        ),
        "format": OUTCOME_QC_GATE_FORMAT,
        "status": expected_outcome_qc_gate["status"],
        "pass": expected_outcome_qc_gate["pass"],
        "directional_claims_allowed": expected_outcome_qc_gate[
            "directional_claims_allowed_by_outcome_qc"
        ],
    }
    expected_statistics["probabilistic_and_event_artifacts"] = {
        "probabilistic_evaluation": _logical_binding(
            root,
            products.probabilistic_evaluation,
            canonical_state["probabilistic_evaluation"],
        )
    }
    actual_statistics = _load_json(products.statistics, label="confirmatory statistics")
    _assert_statistics_equal(expected_statistics, actual_statistics)
    report = Path(products.report)
    availability_frame = pd.read_csv(
        availability_path, dtype={"site_no": "string"}, keep_default_na=False
    )
    expected_report = _render_confirmatory_report(
        opening_id=opening_id,
        statistics=expected_statistics,
        availability=availability_frame,
        trusted_predictions=trusted_frames,
        required_models=preflight["suite"]["required_models"],
        inference_gate=preflight["inference_gate"],
        outcome_quality_audit=expected_quality_audit,
        outcome_qc_gate=expected_outcome_qc_gate,
        approved_target_sensitivity=expected_approved_sensitivity,
        spatial_sensitivity=expected_spatial_sensitivity,
        probabilistic_evaluation=expected_probabilistic_evaluation,
        transport_summary=transport_summary,
    )
    if not report.is_file() or report.read_bytes() != expected_report:
        raise OpeningContractError("confirmatory report differs from trusted recomputation")
    artifacts = {
        "acquisition_manifest": _binding(root, products.acquisition_manifest),
        "raw_nwis_snapshot_index": _binding(root, raw_index),
        "acquisition_request_map": _binding(root, request_map_path),
        "temporal_normalized_outcomes": _binding(
            root, normalized_paths["temporal"]
        ),
        "external_normalized_outcomes": _binding(
            root, normalized_paths["external"]
        ),
        "availability_registry": _logical_binding(
            root,
            availability_path,
            canonical_state["availability_registry"],
        ),
        "outcome_quality_audit": _logical_binding(
            root,
            products.outcome_quality_audit,
            canonical_state["outcome_quality_audit"],
        ),
        "outcome_qc_gate": _logical_binding(
            root,
            products.outcome_qc_gate,
            canonical_state["outcome_qc_gate"],
        ),
        "approved_target_sensitivity": _logical_binding(
            root,
            products.approved_target_sensitivity,
            canonical_state["approved_target_sensitivity"],
        ),
        "spatial_sensitivity": _logical_binding(
            root,
            products.spatial_sensitivity,
            canonical_state["spatial_sensitivity"],
        ),
        "probabilistic_evaluation": _logical_binding(
            root,
            products.probabilistic_evaluation,
            canonical_state["probabilistic_evaluation"],
        ),
        "temporal_predictions": _logical_binding(
            root,
            products.temporal_predictions,
            canonical_state["temporal_predictions"],
        ),
        "external_predictions": _logical_binding(
            root,
            products.external_predictions,
            canonical_state["external_predictions"],
        ),
        "statistics": _logical_binding(
            root,
            products.statistics,
            canonical_state["statistics"],
        ),
        "report": _logical_binding(
            root,
            products.report,
            canonical_state["report"],
        ),
    }
    reported_models = {
        cohort: sorted(frame.model.astype(str).unique().tolist())
        for cohort, frame in trusted_frames.items()
    }
    expected_models = {
        cohort: sorted(str(value) for value in values)
        for cohort, values in preflight["suite"]["required_models"].items()
    }
    return {
        "artifacts": artifacts,
        "formal_tests": expected_statistics["tests"],
        "trusted_prediction_hashes": trusted_hashes,
        "reported_models": reported_models,
        "all_required_models_reported": reported_models == expected_models,
    }


def _validated_intent(
    *, preflight: Mapping[str, Any], root: Path, work_order: Mapping[str, Any]
) -> dict[str, Any]:
    intent_path = Path(preflight["state_paths"]["intent"])
    intent = _load_json(intent_path, label="one-time opening intent")
    if intent_path.read_bytes() != canonical_json_bytes(intent):
        raise OpeningContractError("one-time opening intent is noncanonical")
    _validate_atomic_final_file(
        intent_path, canonical_json_bytes(intent), cleanup_temps=False
    )
    _validate_intent_document(
        intent, preflight=preflight, root=root, work_order=work_order
    )
    return intent


def _validate_intent_document(
    intent: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    root: Path,
    work_order: Mapping[str, Any],
) -> None:
    """Validate an intent document independently of its publication name."""
    self_hashed = dict(intent)
    self_digest = self_hashed.pop("intent_self_sha256", None)
    if not _is_sha256(self_digest) or sha256_json(self_hashed) != self_digest:
        raise OpeningContractError("one-time opening intent self-hash changed")
    expected = {
        "format": INTENT_FORMAT,
        "status": "OPENING_STARTED_IRREVERSIBLE",
        "opening_id": preflight["authorization"]["opening_id"],
        "authorization_sha256": preflight["authorization_sha256"],
        "preflight_attestation_sha256": sha256_json(
            _preflight_attestation(preflight)
        ),
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "work_order_file_sha256": hashlib.sha256(
            canonical_json_bytes(dict(work_order))
        ).hexdigest(),
        "fixed_code_sha256": preflight["fixed_code"]["sha256"],
        "runtime_sha256": preflight["runtime"]["runtime_sha256"],
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
    }
    if any(intent.get(key) != value for key, value in expected.items()):
        raise OpeningContractError("one-time opening intent identity changed")
    expected_fields = {
        *expected,
        "trusted_validator",
        "started_at_utc",
        "intent_self_sha256",
    }
    if set(intent) != expected_fields:
        raise OpeningContractError("one-time opening intent schema changed")
    if intent.get("trusted_validator") != _trusted_validator_identity(root):
        raise OpeningContractError("trusted validator differs from opening intent")
    started = intent.get("started_at_utc")
    try:
        timestamp = datetime.fromisoformat(str(started))
    except ValueError as exc:
        raise OpeningContractError(
            "one-time opening intent timestamp is malformed"
        ) from exc
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(None)
    ):
        raise OpeningContractError(
            "one-time opening intent timestamp is not UTC"
        )


def _inspect_or_recover_preintent_temp(
    *,
    state: Mapping[str, Any],
    preflight: Mapping[str, Any],
    root: Path,
    work_order: Mapping[str, Any],
    publish_or_remove: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Classify the sole legal pre-intent crash remnant, optionally recover it."""
    run_directory = Path(state["run_directory"])
    intent_path = Path(state["intent"])
    if not os.path.lexists(run_directory):
        return "ABSENT", None
    with _secure_directory_chain(
        run_directory, create=False
    ) as run_descriptor:
        run_metadata = os.fstat(run_descriptor)
        if (
            not stat.S_ISDIR(run_metadata.st_mode)
            or run_metadata.st_uid != os.geteuid()
            or run_metadata.st_mode & 0o022
        ):
            raise OpeningContractError(
                "pre-intent run directory is not owner-controlled"
            )
        entries = os.listdir(run_descriptor)
        if not entries:
            # A kill can leave only the durable canonical parent directory,
            # before the first atomic temporary file is allocated.  No intent
            # was published, so this is a safe pre-opening recovery state.
            return "EMPTY_SAFE", None
        pattern = re.compile(
            rf"\.{re.escape(intent_path.name)}\.[a-z0-9_]{{8}}\.tmp"
        )
        if len(entries) != 1 or pattern.fullmatch(entries[0]) is None:
            raise OpeningContractError(
                "pre-intent run directory is empty or contains unexpected state"
            )
        temporary_name = entries[0]
        descriptor = os.open(
            temporary_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=run_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != run_metadata.st_dev
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o022
            ):
                raise OpeningContractError(
                    "pre-intent temporary artifact has unsafe metadata"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
            complete = not metadata.st_mode & 0o222
            if complete:
                try:
                    document = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise OpeningContractError(
                        "complete pre-intent temporary JSON is malformed"
                    ) from exc
                if (
                    not isinstance(document, dict)
                    or payload != canonical_json_bytes(document)
                ):
                    raise OpeningContractError(
                        "complete pre-intent temporary JSON is noncanonical"
                    )
                _validate_intent_document(
                    document,
                    preflight=preflight,
                    root=root,
                    work_order=work_order,
                )
                if publish_or_remove:
                    # A writer may have died after fchmod but before its inode
                    # fsync.  Re-fsync the validated complete temp before
                    # publishing its inode under the irreversible intent name.
                    os.fsync(descriptor)
                    _atomic_create_fault(
                        "after_preintent_recovery_inode_fsync_before_link",
                        intent_path,
                    )
                    try:
                        os.link(
                            temporary_name,
                            intent_path.name,
                            src_dir_fd=run_descriptor,
                            dst_dir_fd=run_descriptor,
                            follow_symlinks=False,
                        )
                    except FileExistsError as exc:
                        raise OpeningAlreadyStarted(
                            "one-time intent appeared during pre-intent recovery"
                        ) from exc
                    os.fsync(run_descriptor)
                    os.unlink(temporary_name, dir_fd=run_descriptor)
                    os.fsync(run_descriptor)
                return "COMPLETE_VALID", document
            if not metadata.st_mode & stat.S_IWUSR:
                raise OpeningContractError(
                    "partial pre-intent temporary artifact has unsafe mode"
                )
            if publish_or_remove:
                os.unlink(temporary_name, dir_fd=run_descriptor)
                os.fsync(run_descriptor)
            return "PARTIAL_SAFE", None
        finally:
            os.close(descriptor)


def isolated_orchestrate_opening(
    authorization_path: str | Path,
    *,
    root: str | Path,
    resume: bool = False,
) -> None:
    """Open once, resume raw transport, or finish deterministic trusted scoring."""
    root = Path(root).resolve()
    authorization_path = Path(authorization_path).resolve()
    if root not in authorization_path.parents or not authorization_path.is_file():
        raise OpeningContractError("opening authorization escapes or is absent")
    preflight = validate_authorization(
        authorization_path,
        root=root,
        require_clean_source=not resume,
    )
    _assert_isolated_role(preflight=preflight, root=root, role="orchestrator")
    state = preflight["state_paths"]
    current_state_paths = _secure_canonical_state_paths(
        root,
        preflight["authorization"]["state_paths"],
    )
    if current_state_paths != state:
        raise OpeningContractError(
            "opening state namespace changed after authorization preflight"
        )
    status = opening_status(
        intent_path=state["intent"], receipt_path=state["receipt"]
    )
    attestation = _preflight_attestation(preflight)
    work_order = _expected_acquisition_work_order(preflight, root=root)
    validator = _trusted_validator_identity(root)
    run_acquisition = True
    if resume:
        inspection = inspect_same_opening_transport_resume(
            authorization_path, root=root
        )
        phase = inspection["resume_phase"]
        if phase == "TERMINAL_COMPLETE":
            _read_completed_receipt(
                authorization_path=authorization_path, root=root
            )
            for key in ("intent", "work_order", "receipt", "receipt_sha256"):
                path = Path(state[key])
                _cleanup_atomic_create_path_temps(path, path.read_bytes())
            return
        if phase in {
            "RAW_TRANSPORT",
            "ACQUISITION_FINALIZATION_NETWORK_FREE",
        }:
            run_acquisition = True
        elif phase in {
            "TRUSTED_RECOMPUTE_NETWORK_FREE",
            "RECEIPT_COMPLETION_AFTER_FULL_REPLAY",
            "SIDECAR_RECOVERY_AFTER_FULL_VALIDATION",
            "ACQUISITION_PERMISSION_RECOVERY_BY_FULL_REPLAY",
            "TRUSTED_PERMISSION_RECOVERY_BY_FULL_REPLAY",
            "TRUSTED_STAGE_CLEANUP_AFTER_FULL_VALIDATION",
        }:
            run_acquisition = False
        else:
            raise OpeningAlreadyStarted(
                "Route-A opening is not same-opening resume eligible: "
                f"{inspection['status']}"
            )
        validated_intent = _validated_intent(
            preflight=preflight, root=root, work_order=work_order
        )
        _cleanup_atomic_create_path_temps(
            Path(state["intent"]), canonical_json_bytes(validated_intent)
        )
        work_order_path = Path(state["work_order"])
        if os.path.lexists(work_order_path):
            if (
                not work_order_path.is_file()
                or _load_json(
                    work_order_path, label="acquisition work order"
                ) != work_order
            ):
                raise OpeningContractError(
                    "same-opening resume requires the exact original work order"
                )
            _cleanup_atomic_create_path_temps(
                work_order_path, canonical_json_bytes(work_order)
            )
        else:
            # The intent already binds both hashes of this deterministic work
            # order, so recreating a wholly absent file is not a second opening.
            exclusive_create_json(work_order_path, work_order)
    else:
        if status != "SEALED_READY_OR_NOT_AUTHORIZED":
            raise OpeningAlreadyStarted(f"Route-A opening state is {status}")
        recovery, recovered_intent = _inspect_or_recover_preintent_temp(
            state=state,
            preflight=preflight,
            root=root,
            work_order=work_order,
            publish_or_remove=True,
        )
        recovered_complete = recovery == "COMPLETE_VALID"
        if recovered_complete:
            if recovered_intent is None:
                raise OpeningContractError(
                    "complete pre-intent recovery lost its validated document"
                )
            intent = recovered_intent
        else:
            intent_stable = {
                "format": INTENT_FORMAT,
                "status": "OPENING_STARTED_IRREVERSIBLE",
                "opening_id": preflight["authorization"]["opening_id"],
                "authorization_sha256": preflight["authorization_sha256"],
                "preflight_attestation_sha256": sha256_json(attestation),
                "work_order_self_sha256": work_order["work_order_self_sha256"],
                "work_order_file_sha256": hashlib.sha256(
                    canonical_json_bytes(dict(work_order))
                ).hexdigest(),
                "fixed_code_sha256": preflight["fixed_code"]["sha256"],
                "runtime_sha256": preflight["runtime"]["runtime_sha256"],
                "trusted_validator": validator,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "maximum_openings": 1,
                "retry_after_failure_allowed": False,
                "same_opening_transport_resume_allowed": True,
            }
            intent = {
                **intent_stable,
                "intent_self_sha256": sha256_json(intent_stable),
            }
        allowed_existing = {"namespace", "run_directory"}
        if recovered_complete:
            allowed_existing.add("intent")
        preexisting = sorted(
            key
            for key, path in state.items()
            if key not in allowed_existing
            and os.path.lexists(path)
        )
        if preexisting:
            raise OpeningAlreadyStarted(
                "canonical Route-A state namespace is not empty: "
                f"{preexisting}"
            )
        # This marker is the first state mutation and is never replaced.  A raw
        # continuation remains the same opening ID and the same intent.
        if not recovered_complete:
            exclusive_create_json(state["intent"], intent)
        exclusive_create_json(state["work_order"], work_order)
    if run_acquisition:
        _run_fixed_isolated_child(
            root=root,
            role="acquisition",
            argument_path=state["work_order"],
            resume=resume,
        )
        replayed = validate_authorization(
            authorization_path, root=root, require_clean_source=False
        )
        replayed_attestation = _preflight_attestation(replayed)
        if replayed_attestation != attestation:
            raise OpeningContractError(
                "preflight changed while raw acquisition ran"
            )
        if _trusted_validator_identity(root) != validator:
            raise OpeningContractError(
                "trusted validator changed while acquisition ran"
            )
    if _load_json(state["work_order"], label="acquisition work order") != work_order:
        raise OpeningContractError("acquisition work order changed after publication")
    if not Path(state["acquisition_manifest"]).is_file():
        raise OpeningContractError("raw acquisition child did not publish its manifest")
    _run_fixed_isolated_child(
        root=root, role="trusted_scorer", argument_path=state["work_order"]
    )
    _read_completed_receipt(authorization_path=authorization_path, root=root)


def _release_artifact_formats() -> dict[str, str]:
    return {
        "acquisition_manifest": ACQUISITION_MANIFEST_FORMAT,
        "raw_nwis_snapshot_index": "thermoroute.snapshot-index.v1",
        "acquisition_request_map": ACQUISITION_REQUEST_MAP_FORMAT,
        "temporal_normalized_outcomes": "thermoroute.route-a-normalized-outcomes.v1",
        "external_normalized_outcomes": "thermoroute.route-a-normalized-outcomes.v1",
        "availability_registry": "thermoroute.route-a-availability-registry.v1",
        "outcome_quality_audit": OUTCOME_QUALITY_AUDIT_FORMAT,
        "outcome_qc_gate": OUTCOME_QC_GATE_FORMAT,
        "approved_target_sensitivity": APPROVED_TARGET_SENSITIVITY_FORMAT,
        "spatial_sensitivity": SPATIAL_SENSITIVITY_FORMAT,
        "probabilistic_evaluation": PROBABILISTIC_EVALUATION_FORMAT,
        "temporal_predictions": R.PREDICTION_SCHEMA_VERSION,
        "external_predictions": R.PREDICTION_SCHEMA_VERSION,
        "statistics": STATISTICS_FORMAT,
        "report": "thermoroute.route-a-confirmatory-report.v1",
    }


def _release_bindings(
    *, preflight: Mapping[str, Any], validated: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    formats = _release_artifact_formats()
    artifacts = validated["artifacts"]
    if set(artifacts) != set(formats):
        raise OpeningContractError("release artifact registry is incomplete")
    bound = {
        key: {"format": formats[key], **dict(artifacts[key])}
        for key in sorted(formats)
    }
    authorization = preflight["authorization"]
    return {
        "format": "thermoroute.route-a-release-bindings.v1",
        "opening_id": authorization["opening_id"],
        "state_namespace": authorization["state_paths"]["namespace"],
        "authorization": {
            "format": AUTHORIZATION_FORMAT,
            "path": authorization["source"]["authorization_path"],
            "sha256": preflight["authorization_sha256"],
        },
        "artifacts": bound,
        "receipt": {
            "format": RECEIPT_FORMAT,
            "path": authorization["state_paths"]["receipt"],
            "external_sha256_path": authorization["state_paths"]["receipt_sha256"],
        },
    }


def _assert_validated_artifacts_published(
    *, validated: Mapping[str, Any], root: Path
) -> None:
    artifacts = validated.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise OpeningContractError("validated trusted artifact registry is absent")
    for key, binding in artifacts.items():
        _verify_file_binding(root, binding, label=f"published {key}")


def _receipt_sidecar_bytes(receipt_path: Path) -> bytes:
    return (
        f"{sha256_file(receipt_path)}  {receipt_path.name}\n"
    ).encode("ascii")


def _assert_receipt_matches_validation(
    receipt: Mapping[str, Any], validated: Mapping[str, Any]
) -> None:
    expected = {
        "all_predeclared_models_reported": validated[
            "all_required_models_reported"
        ],
        "reported_models": validated["reported_models"],
        "artifacts": validated["artifacts"],
        "trusted_prediction_hashes": validated["trusted_prediction_hashes"],
        "formal_tests": validated["formal_tests"],
    }
    wrong = [key for key, value in expected.items() if receipt.get(key) != value]
    if wrong:
        raise OpeningContractError(
            "opening receipt differs from trusted recomputation: "
            + ", ".join(sorted(wrong))
        )


def isolated_score_and_receipt(
    work_order_path: str | Path, *, root: str | Path
) -> dict[str, Any]:
    """Replay raw evidence and crash-safely publish trusted products/receipt."""
    root = Path(root).resolve()
    work_order_path = Path(work_order_path).resolve()
    if root not in work_order_path.parents or not work_order_path.is_file():
        raise OpeningContractError("trusted scorer work order escapes or is absent")
    raw_work_order = _load_json(work_order_path, label="acquisition work order")
    authorization_path = _resolve_inside(
        root, raw_work_order.get("authorization_path")
    )
    preflight = validate_authorization(
        authorization_path, root=root, require_clean_source=False
    )
    _assert_isolated_role(preflight=preflight, root=root, role="trusted_scorer")
    expected_work_order = _expected_acquisition_work_order(preflight, root=root)
    if raw_work_order != expected_work_order:
        raise OpeningContractError("trusted scorer received a changed work order")
    _validate_atomic_final_file(
        work_order_path,
        canonical_json_bytes(expected_work_order),
        cleanup_temps=False,
    )
    intent = _validated_intent(
        preflight=preflight, root=root, work_order=expected_work_order
    )
    state = preflight["state_paths"]
    if not Path(state["acquisition_manifest"]).is_file():
        raise OpeningContractError("trusted scorer requires raw acquisition evidence")
    canonical_trusted = _trusted_directory_from_state(state)
    receipt_path = Path(state["receipt"])
    sidecar_path = Path(state["receipt_sha256"])
    with _exclusive_trusted_publication_lock(state):
        abandoned_stage_count = _handle_abandoned_trusted_stages(
            state, remove=False
        )
        from .outcome_acquisition import (
            OutcomeAcquisitionError,
            _acquisition_directory_mode,
            _assert_exact_acquisition_directory,
            _harden_recoverable_acquisition_directory,
        )

        acquisition_state = {
            key: Path(value) for key, value in state.items()
        }
        try:
            _assert_exact_acquisition_directory(
                Path(state["acquisition_manifest"]).parent,
                acquisition_state,
                allow_recoverable_canonical_mode=True,
            )
            acquisition_mode = _acquisition_directory_mode(acquisition_state)
        except OutcomeAcquisitionError as exc:
            raise OpeningContractError(
                "acquisition publication mode/layout is not recoverable"
            ) from exc
        if acquisition_mode == 0o700:
            # No permission is changed until the entire raw bundle has been
            # independently replayed and every canonical binding has closed.
            _verified_acquisition_for_scoring(
                preflight=preflight,
                root=root,
                allow_recoverable_publication_mode=True,
            )
            _harden_recoverable_acquisition_directory(acquisition_state)
        receipt_exists = os.path.lexists(receipt_path)
        sidecar_exists = os.path.lexists(sidecar_path)
        if sidecar_exists and not receipt_exists:
            raise OpeningContractError(
                "opening receipt digest exists without its authoritative receipt"
            )
        try:
            configure_deterministic_runtime()
            assert_formal_numerical_policy()
        except RuntimeError as exc:
            raise OpeningContractError(
                "trusted scorer determinism policy was not applied"
            ) from exc

        canonical_exists = os.path.lexists(canonical_trusted)
        if receipt_exists and not canonical_exists:
            raise OpeningContractError(
                "opening receipt exists without canonical trusted products"
            )
        if canonical_exists:
            products = _opening_products_from_state(state)
            trusted_permission_recovery = (
                _trusted_directory_mode(state) == 0o700
            )
            validated = validate_opening_products(
                products,
                preflight=preflight,
                root=root,
                allow_recoverable_trusted_mode=(
                    trusted_permission_recovery
                ),
            )
            if trusted_permission_recovery:
                _harden_recoverable_trusted_directory(state)
            if abandoned_stage_count:
                # When an authoritative canonical generation exists, do not
                # delete even a metadata-safe abandoned stage until the
                # canonical acquisition and trusted products have both passed
                # their complete deterministic replay.
                _handle_abandoned_trusted_stages(state, remove=True)
        else:
            if abandoned_stage_count:
                _handle_abandoned_trusted_stages(state, remove=True)
            stage_directory = _new_trusted_stage_directory(state)
            staged_state = _trusted_state_at_directory(state, stage_directory)
            products = produce_trusted_opening_products(
                preflight=preflight,
                root=root,
                output_state_paths=staged_state,
            )
            _trusted_publication_fault("after_stage_generation")
            validated = validate_opening_products(
                products,
                preflight=preflight,
                root=root,
                staged=True,
            )
            _trusted_publication_fault("after_stage_validation")
            _atomic_publish_trusted_directory(stage_directory, state)
            _trusted_publication_fault("after_trusted_publish")
            _assert_validated_artifacts_published(
                validated=validated, root=root
            )
        if validated.get("all_required_models_reported") is not True:
            raise OpeningContractError(
                "trusted scorer did not report every frozen model"
            )

        if receipt_exists:
            receipt = _read_completed_receipt(
                authorization_path=authorization_path,
                root=root,
                require_sidecar=False,
            )
            _assert_receipt_matches_validation(receipt, validated)
            _cleanup_atomic_create_path_temps(
                receipt_path, receipt_path.read_bytes()
            )
            if sidecar_exists:
                _cleanup_atomic_create_path_temps(
                    sidecar_path, sidecar_path.read_bytes()
                )
            if not sidecar_exists:
                _atomic_create_bytes(
                    sidecar_path, _receipt_sidecar_bytes(receipt_path)
                )
                _trusted_publication_fault("after_receipt_sidecar_recovery")
            return _read_completed_receipt(
                authorization_path=authorization_path, root=root
            )

        runtime_attestation = environment_fingerprint()
        if runtime_attestation.get(
            "numerical_runtime_sha256"
        ) != preflight["runtime"]["runtime_sha256"]:
            raise OpeningContractError(
                "receipt runtime differs from authorization"
            )
        release_bindings = _release_bindings(
            preflight=preflight, validated=validated, root=root
        )
        receipt_stable = {
            "format": RECEIPT_FORMAT,
            "status": "OPENED_AND_SCORED_ONCE",
            "opening_id": preflight["authorization"]["opening_id"],
            "authorization_sha256": preflight["authorization_sha256"],
            "intent_sha256": sha256_file(state["intent"]),
            "work_order_sha256": sha256_file(state["work_order"]),
            "preflight_attestation": _preflight_attestation(preflight),
            "preflight_attestation_sha256": sha256_json(
                _preflight_attestation(preflight)
            ),
            "trusted_validator": _trusted_validator_identity(root),
            "fixed_code": preflight["fixed_code"],
            "authorized_runtime": preflight["runtime"],
            "completion_environment": runtime_attestation,
            "python_hash_seed_interpreter_effect": (
                "present_but_ignored_under_isolated_mode"
            ),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "opening_count": 1,
            "maximum_openings": 1,
            "retry_after_failure_allowed": False,
            "same_opening_transport_resume_allowed": True,
            "transport_recovery": _load_json(
                state["acquisition_manifest"],
                label="opened acquisition manifest",
            )["transport_summary"],
            "all_predeclared_models_reported": validated[
                "all_required_models_reported"
            ],
            "reported_models": validated["reported_models"],
            "artifacts": validated["artifacts"],
            "trusted_prediction_hashes": validated[
                "trusted_prediction_hashes"
            ],
            "formal_tests": validated["formal_tests"],
            "state_paths": dict(preflight["authorization"]["state_paths"]),
            "release_bindings": release_bindings,
            "intent_self_sha256": intent["intent_self_sha256"],
            "security_boundary": (
                "misoperation/replay guard for an honest filesystem owner; not a "
                "defense against an owner who can replace the interpreter or files"
            ),
        }
        receipt = {
            **receipt_stable,
            "receipt_self_sha256": sha256_json(receipt_stable),
        }
        _atomic_create_bytes(
            receipt_path, canonical_json_bytes(receipt)
        )
        _trusted_publication_fault("after_receipt_publish")
        _atomic_create_bytes(
            sidecar_path, _receipt_sidecar_bytes(receipt_path)
        )
        _trusted_publication_fault("after_receipt_sidecar_publish")
        return _read_completed_receipt(
            authorization_path=authorization_path, root=root
        )


def _read_completed_receipt(
    *,
    authorization_path: str | Path,
    root: Path,
    allow_gitless_archive: bool = False,
    require_sidecar: bool = True,
) -> dict[str, Any]:
    authorization_path = Path(authorization_path).resolve()
    authorization = _load_json(authorization_path, label="opening authorization")
    self_hashed = dict(authorization)
    self_digest = self_hashed.pop("authorization_self_sha256", None)
    if not _is_sha256(self_digest) or sha256_json(self_hashed) != self_digest:
        raise OpeningContractError("opening authorization self-hash changed")
    if authorization.get("format") != AUTHORIZATION_FORMAT:
        raise OpeningContractError("unsupported opening authorization")
    if authorization.get("source", {}).get("authorization_path") != _relative(
        root, authorization_path
    ):
        raise OpeningContractError("authorization path differs from frozen source policy")
    state = authorization.get("state_paths")
    if not isinstance(state, Mapping):
        raise OpeningContractError("authorization lacks canonical state paths")
    secured_state = _secure_canonical_state_paths(root, state)
    from .outcome_acquisition import (
        OutcomeAcquisitionError,
        _assert_exact_acquisition_directory,
    )

    # Check the trusted publication first: a receipt is authoritative only for
    # this exact immutable directory, and diagnostics must not be masked by an
    # unrelated missing acquisition fixture.
    _assert_exact_trusted_directory(
        _trusted_directory_from_state(secured_state), secured_state
    )
    try:
        _assert_exact_acquisition_directory(
            Path(secured_state["acquisition_manifest"]).parent,
            {key: Path(value) for key, value in secured_state.items()},
        )
    except OutcomeAcquisitionError as exc:
        raise OpeningContractError(
            "opening receipt acquisition bundle is not exact"
        ) from exc
    receipt_path = _resolve_inside(root, state.get("receipt"))
    raw_sidecar = Path(str(state.get("receipt_sha256")))
    if raw_sidecar.is_absolute():
        raise OpeningContractError("opening receipt sidecar path must be relative")
    try:
        sidecar_path = assert_no_symlink_components(root, root / raw_sidecar)
    except AcquisitionContractError as exc:
        raise OpeningContractError(
            "opening receipt sidecar path is unsafe"
        ) from exc
    if require_sidecar and not sidecar_path.is_file():
        raise OpeningContractError("external opening-receipt SHA-256 is absent")
    receipt = _load_json(receipt_path, label="opening receipt")
    if receipt_path.read_bytes() != canonical_json_bytes(receipt):
        raise OpeningContractError("opening receipt is noncanonical")
    _validate_atomic_final_file(
        receipt_path, canonical_json_bytes(receipt), cleanup_temps=False
    )
    receipt_stable = dict(receipt)
    receipt_self = receipt_stable.pop("receipt_self_sha256", None)
    if not _is_sha256(receipt_self) or sha256_json(receipt_stable) != receipt_self:
        raise OpeningContractError("opening receipt self-hash changed")
    expected = {
        "format": RECEIPT_FORMAT,
        "status": "OPENED_AND_SCORED_ONCE",
        "opening_id": authorization.get("opening_id"),
        "authorization_sha256": sha256_file(authorization_path),
        "opening_count": 1,
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
        "all_predeclared_models_reported": True,
        "state_paths": dict(state),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise OpeningContractError("opening receipt identity/completion state changed")
    acquisition_manifest = _load_json(
        _resolve_inside(root, state.get("acquisition_manifest")),
        label="opened acquisition manifest",
    )
    if receipt.get("transport_recovery") != acquisition_manifest.get(
        "transport_summary"
    ):
        raise OpeningContractError("opening receipt transport evidence changed")
    preflight = validate_authorization(
        authorization_path,
        root=root,
        require_clean_source=False,
        allow_gitless_archive=allow_gitless_archive,
    )
    expected_attestation = _preflight_attestation(preflight)
    if (
        receipt.get("preflight_attestation") != expected_attestation
        or receipt.get("preflight_attestation_sha256")
        != sha256_json(expected_attestation)
        or receipt.get("fixed_code") != preflight["fixed_code"]
        or receipt.get("authorized_runtime") != preflight["runtime"]
        or receipt.get("trusted_validator") != _trusted_validator_identity(root)
    ):
        raise OpeningContractError("opening receipt preflight/code/runtime binding changed")
    intent_path = _resolve_inside(root, state.get("intent"))
    if receipt.get("intent_sha256") != sha256_file(intent_path):
        raise OpeningContractError("opening receipt does not bind its intent")
    work_order_path = _resolve_inside(root, state.get("work_order"))
    expected_work_order = _expected_acquisition_work_order(preflight, root=root)
    if (
        receipt.get("work_order_sha256") != sha256_file(work_order_path)
        or _load_json(work_order_path, label="acquisition work order")
        != expected_work_order
    ):
        raise OpeningContractError("opening receipt does not bind its work order")
    _validate_atomic_final_file(
        work_order_path,
        canonical_json_bytes(expected_work_order),
        cleanup_temps=False,
    )
    _validated_intent(
        preflight=preflight, root=root, work_order=expected_work_order
    )
    expected_sidecar = (
        f"{sha256_file(receipt_path)}  {receipt_path.name}\n"
    ).encode("ascii")
    if os.path.lexists(sidecar_path):
        if not sidecar_path.is_file() or sidecar_path.read_bytes() != expected_sidecar:
            raise OpeningContractError("external opening-receipt SHA-256 is invalid")
        _validate_atomic_final_file(
            sidecar_path, expected_sidecar, cleanup_temps=False
        )
    if require_sidecar and not os.path.lexists(sidecar_path):
        raise OpeningContractError("external opening-receipt SHA-256 is absent")
    bindings = receipt.get("release_bindings")
    if (
        not isinstance(bindings, Mapping)
        or bindings.get("format") != "thermoroute.route-a-release-bindings.v1"
        or bindings.get("opening_id") != authorization.get("opening_id")
        or bindings.get("state_namespace") != state.get("namespace")
    ):
        raise OpeningContractError("opening receipt release bindings changed")
    formats = _release_artifact_formats()
    artifacts = receipt.get("artifacts")
    released_artifacts = bindings.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(formats)
        or not isinstance(released_artifacts, Mapping)
        or set(released_artifacts) != set(formats)
    ):
        raise OpeningContractError("opening receipt artifact registry is incomplete")
    for key, format_name in formats.items():
        binding = artifacts[key]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise OpeningContractError(f"receipt artifact binding is malformed: {key}")
        artifact_path = _verify_file_binding(root, binding, label=f"receipt {key}")
        expected_release = {"format": format_name, **dict(binding)}
        if released_artifacts[key] != expected_release:
            raise OpeningContractError(f"release binding differs for {key}")
        canonical_key = {
            "acquisition_manifest": "acquisition_manifest",
            "raw_nwis_snapshot_index": "raw_nwis_snapshot_index",
            "acquisition_request_map": "acquisition_request_map",
            "temporal_normalized_outcomes": "temporal_outcomes",
            "external_normalized_outcomes": "external_outcomes",
            "availability_registry": "availability_registry",
            "outcome_quality_audit": "outcome_quality_audit",
            "outcome_qc_gate": "outcome_qc_gate",
            "approved_target_sensitivity": "approved_target_sensitivity",
            "spatial_sensitivity": "spatial_sensitivity",
            "probabilistic_evaluation": "probabilistic_evaluation",
            "temporal_predictions": "temporal_predictions",
            "external_predictions": "external_predictions",
            "statistics": "statistics",
            "report": "report",
        }.get(key)
        if canonical_key is not None and artifact_path != (
            root / str(state[canonical_key])
        ).resolve():
            raise OpeningContractError(f"receipt artifact path is noncanonical: {key}")
    expected_release_identity = {
        "format": "thermoroute.route-a-release-bindings.v1",
        "opening_id": authorization["opening_id"],
        "state_namespace": state["namespace"],
        "authorization": {
            "format": AUTHORIZATION_FORMAT,
            "path": authorization["source"]["authorization_path"],
            "sha256": sha256_file(authorization_path),
        },
        "artifacts": dict(released_artifacts),
        "receipt": {
            "format": RECEIPT_FORMAT,
            "path": state["receipt"],
            "external_sha256_path": state["receipt_sha256"],
        },
    }
    if dict(bindings) != expected_release_identity:
        raise OpeningContractError("release binding identity/path registry changed")
    expected_models = {
        cohort: sorted(str(value) for value in values)
        for cohort, values in preflight["suite"]["required_models"].items()
    }
    if receipt.get("reported_models") != expected_models:
        raise OpeningContractError("receipt reported-model registry changed")
    return receipt


def validate_completed_receipt(
    authorization_path: str | Path,
    *,
    root: str | Path,
    allow_gitless_archive: bool = False,
) -> dict[str, Any]:
    """Public read-only verifier for status, release and claim-gating tools."""
    return _read_completed_receipt(
        authorization_path=authorization_path,
        root=Path(root).resolve(),
        allow_gitless_archive=allow_gitless_archive,
    )


def isolated_verify_release(
    authorization_path: str | Path, *, root: str | Path
) -> dict[str, Any]:
    """Read-only, network-free replay for an extracted post-opening archive."""
    root = Path(root).resolve()
    authorization_path = Path(authorization_path).resolve()
    if not sys.flags.isolated:
        raise OpeningContractError("release replay must run under python -I")
    if root not in authorization_path.parents or not authorization_path.is_file():
        raise OpeningContractError("release authorization escapes or is absent")
    preflight = validate_authorization(
        authorization_path,
        root=root,
        require_clean_source=False,
        allow_gitless_archive=True,
    )
    entry = preflight["fixed_code"]["entrypoints"]["trusted_scorer"]
    actual_entry = Path(sys.argv[0]).resolve()
    if (
        actual_entry != (root / _FIXED_ENTRYPOINTS["trusted_scorer"]).resolve()
        or entry.get("path") != _FIXED_ENTRYPOINTS["trusted_scorer"]
        or sha256_file(actual_entry) != entry.get("sha256")
    ):
        raise OpeningContractError("release replay entrypoint identity changed")
    state = preflight["state_paths"]
    products = _opening_products_from_state(state)
    validated = validate_opening_products(products, preflight=preflight, root=root)
    receipt = _read_completed_receipt(
        authorization_path=authorization_path,
        root=root,
        allow_gitless_archive=True,
    )
    if validated["artifacts"] != receipt.get("artifacts"):
        raise OpeningContractError("release replay artifacts differ from receipt")
    return {
        "status": "ROUTE_A_RELEASE_REPLAY_VALID",
        "opening_id": receipt["opening_id"],
        "state_namespace": receipt["state_paths"]["namespace"],
        "source_tree_sha256": preflight["authorization"]["source"][
            "source_tree_sha256"
        ],
        "runtime_sha256": preflight["runtime"]["runtime_sha256"],
        "all_predeclared_models_reported": receipt[
            "all_predeclared_models_reported"
        ],
        "artifact_count": len(validated["artifacts"]),
        "network_used": False,
        "files_written": False,
    }


def run_opening_once(
    authorization_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Launch the sole fixed opening chain; no callback or command is accepted."""
    root = Path(root).resolve()
    authorization_path = Path(authorization_path).resolve()
    if root not in authorization_path.parents or not authorization_path.is_file():
        raise OpeningContractError("opening authorization escapes or is absent")
    _run_fixed_isolated_child(
        root=root, role="orchestrator", argument_path=authorization_path
    )
    return _read_completed_receipt(
        authorization_path=authorization_path, root=root
    )


def resume_opening_once(
    authorization_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Resume one ledger or deterministic completion; HTTP is not exactly once."""
    root = Path(root).resolve()
    authorization_path = Path(authorization_path).resolve()
    if root not in authorization_path.parents or not authorization_path.is_file():
        raise OpeningContractError("opening authorization escapes or is absent")
    _run_fixed_isolated_child(
        root=root,
        role="orchestrator",
        argument_path=authorization_path,
        resume=True,
    )
    return _read_completed_receipt(
        authorization_path=authorization_path, root=root
    )
