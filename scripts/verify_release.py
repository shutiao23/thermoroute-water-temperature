#!/usr/bin/env python3
"""Verify a ThermoRoute ZIP in an isolated temporary directory.

The default check is fast and network-free: validate the archive boundary, verify
its checksum sidecar when present, extract it, and run the embedded provenance
checker.  ``--run-data-smoke`` additionally executes stage 01 from the extracted
copy, proving that the raw three-station inputs were actually shipped.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from fnmatch import fnmatch
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import types
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse
import zipfile


ARCHIVE_ROOT = "thermoroute"
PROFILE_FORMAT = "thermoroute.route-a-release-profile.v1"
PROFILE_MARKER = "data_usgs/release_profile_v1.json"
CLAIM_AUDIT_PATH = "evidence/release_claim_audit_v1.json"
GIT_BUNDLE_PATH = "evidence/route_a_compute_history.bundle"
REPRODUCIBILITY_LOCK = "requirements-lock-py312-hashed.txt"
PREOPEN_PROFILE = "PREOPEN_NOT_COMPLETE"
POSTOPEN_PROFILE = "ROUTE_A_OPENED_COMPLETE"
RELEASE_PROFILES = (PREOPEN_PROFILE, POSTOPEN_PROFILE)
PREOPEN_WARNING = (
    "This archive predates the one-time Route-A label opening. It cannot support "
    "a Route-A confirmatory result or conclusion."
)
HASHED_LOCK_ROLE = (
    "FULLY_HASHED_PACKAGE_PORTABILITY_AID; NOT_THE_OPENING_RUNTIME_IDENTITY "
    "UNLESS_AN_OPENING_AUTHORIZATION_EXPLICITLY_BINDS_THIS_FILE"
)
AUTHORIZATION_FORMAT = "thermoroute.route-a-opening-authorization.v1"
INTENT_FORMAT = "thermoroute.route-a-opening-intent.v1"
RECEIPT_FORMAT = "thermoroute.route-a-opening-receipt.v1"
STATISTICS_FORMAT = "thermoroute.route-a-confirmatory-statistics.v1"
PROTOCOL_SEAL_FORMAT = "thermoroute.route-a-protocol-seal.v1"
PROTOCOL_SEAL_PATH = "protocols/route_a_protocol_seal_v1.json"
CHRONOLOGY_FORMAT = "thermoroute.route-a-prelabel-chronology.v1"
CHRONOLOGY_STATUS = "PASS_REPOSITORY_INTERNAL_PRELABEL_ORDER"
CHRONOLOGY_PATH = "outputs/prelabel/route_a_prelabel_chronology_v1.json"
CHRONOLOGY_EVIDENCE_SCOPE = (
    "repository-internal Git ancestry and SHA-256 evidence for an honest owner; "
    "not proof against owner-controlled Git-history rewriting"
)
OUTCOME_QC_AMENDMENT_ROLE = (
    "predeclared_nonfiltering_gross_plausibility_and_aggregate_sensitivity_"
    "directional_reporting_gate_not_complete_outcome_quality_certification"
)
TEMPORAL_COVERAGE_POLICY_PATH = (
    "protocols/route_a_temporal_coverage_policy_v1.json"
)
TEMPORAL_COVERAGE_POLICY_SHA256 = (
    "6b08850ced16de6f97ceda8b16ce89b301d5c5cccb794a4427da0a3e39e211ad"
)
TEMPORAL_COVERAGE_POLICY_FORMAT = (
    "thermoroute.route-a-temporal-coverage-policy.v1"
)
TEMPORAL_COVERAGE_POLICY_ID = "route-a-temporal-coverage-audit-001"
TEMPORAL_COVERAGE_AUDIT_FORMAT = (
    "thermoroute.route-a-temporal-coverage-audit.v1"
)
TEMPORAL_COVERAGE_CORE_STATUS = "DERIVED_CORE_REQUIRES_RECEIPT_BINDING"
TEMPORAL_COVERAGE_AMENDMENT_ROLE = (
    "predeclared_nonfiltering_temporal_coverage_and_equal_cell_descriptive_"
    "sensitivity_never_changes_formal_result_or_decision"
)
TRUSTED_SCORING_RECOVERY_CONTRACT = {
    "maximum_logical_openings": 1,
    "maximum_frozen_request_ledgers_per_opening": 1,
    "second_logical_opening_allowed": False,
    "http_retries_within_or_across_transport_processes_allowed": True,
    "http_delivery_semantics": (
        "at_least_once_until_the_response_transaction_directory_is_complete_"
        "and_durably_published"
    ),
    "response_received_but_transaction_not_durable_may_be_requested_again": True,
    "exactly_once_http_delivery_claimed": False,
    "durable_canonical_response_replacement_allowed": False,
    "raw_transport_resume_before_acquisition_manifest": (
        "same_opening_identifier_and_exact_frozen_request_ledger_only; "
        "refetch_only_when_no_complete_durable_verifiable_canonical_"
        "transaction_exists; partial_invalid_or_noncanonical_canonical_"
        "transactions_fail_closed_without_overwrite; only_unpublished_"
        "owner_private_temporary_or_pending_state_may_be_cleaned; "
        "normalized_derived_and_trusted_outputs_must_all_be_absent"
    ),
    "raw_transport_resume_after_acquisition_manifest_allowed": False,
    "raw_acquisition_child_after_acquisition_manifest_allowed": False,
    "acquisition_bundle_publication": (
        "request_map_two_normalized_tables_and_manifest_are_generated_and_"
        "validated_in_one_private_same_filesystem_stage_then_published_by_one_"
        "directory_rename"
    ),
    "trusted_completion_after_acquisition_manifest": (
        "network_free_deterministic_replay_under_the_same_opening_identifier"
    ),
    "trusted_publication": (
        "generate_and_validate_one_complete_private_same_filesystem_directory_"
        "then_publish_by_one_directory_rename"
    ),
    "absent_canonical_trusted_directory": (
        "delete_only_strictly_validated_canonical_named_owner_private_same_"
        "device_read_only_regular_file_stages_without_external_hardlinks_"
        "then_recompute_a_complete_generation; any_unsafe_stage_fails_closed"
    ),
    "complete_canonical_trusted_directory_without_receipt": (
        "fully_replay_and_validate_all_trusted_artifacts_then_create_the_"
        "receipt_only"
    ),
    "partial_invalid_or_noncanonical_trusted_directory": (
        "FAIL_CLOSED_NO_REPLACEMENT"
    ),
    "valid_receipt_without_external_sha256_sidecar": (
        "fully_validate_receipt_and_bound_artifacts_then_derive_the_missing_"
        "sidecar_only"
    ),
    "external_sha256_sidecar_without_receipt": "FAIL_CLOSED",
    "security_scope": (
        "honest_owner_misoperation_and_replay_guard_not_a_malicious_same_uid_"
        "or_owner_security_boundary"
    ),
}

# The outer verifier treats a release ZIP as hostile input.  These limits are
# deliberately far above the canonical release footprint while bounding disk,
# memory-metadata and decompression work before any archive member is written.
MAX_ARCHIVE_MEMBERS = 50_000
MAX_ARCHIVE_FILE_BYTES = 4 * 1024**3
MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES = 256 * 1024**2
MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 8 * 1024**3
MAX_ARCHIVE_COMPRESSION_RATIO = 200
ARCHIVE_COPY_CHUNK_BYTES = 1024 * 1024

_FORBIDDEN_AMBIENT_GIT_VARIABLES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_EXTERNAL_DIFF",
        "GIT_GLOB_PATHSPECS",
        "GIT_GRAFT_FILE",
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
    }
)

SOURCE_INVENTORY_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "tests/**/*.py",
    "protocols/**/*.json",
    "protocols/**/*.md",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock*.txt",
)
PROTECTED_DIRECTORIES = ("src", "scripts", "tests", "protocols", ".github")
PROTECTED_EXACT_FILES = (".gitignore", "pyproject.toml")
PROTECTED_ROOT_PATTERNS = ("requirements*.txt", "*lock*", "*.lock")

REQUIRED_MEMBERS = {
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    REPRODUCIBILITY_LOCK,
    ".github/workflows/ci.yml",
    "src/thermoroute/config.py",
    "scripts/run_all.sh",
    "scripts/14_manifest.py",
    "scripts/deterministic_zip.py",
    "scripts/verify_release.py",
    "scripts/26_validate_claims.py",
    "tests/test_leakage.py",
    "protocols/route_a_confirmatory_v1.json",
    "protocols/route_a_confirmatory_protocol.md",
    PROTOCOL_SEAL_PATH,
    "protocols/route_a_claim_registry_v1.json",
    "data/b1.csv",
    "data/s2.csv",
    "data/p3.csv",
    "data_usgs/panel_usgs_120v2.parquet",
    "data_usgs/station_registry_v1.csv",
    "data_usgs/stations_meta_120v2.csv",
    "data_usgs/frozen_panel_v1.json",
    "data_usgs/huc_metadata_usgs_v1.csv",
    "data_usgs/huc_metadata_usgs_v1.provenance.json",
    "data_usgs/raw_snapshots/huc-v1/snapshot_index.json",
    PROFILE_MARKER,
    CLAIM_AUDIT_PATH,
    GIT_BUNDLE_PATH,
    "outputs/manifest.json",
}

FORBIDDEN_MEMBERS = {
    # Different 120-site cohort: 18 keys differ from the frozen registry.
    "outputs/tables/usgs_stations_with_huc.csv",
}

CANONICAL_DEVELOPMENT_PATHS = (
    "data_usgs/panel_usgs_120v2.parquet",
    "data_usgs/station_registry_v1.csv",
    "data_usgs/stations_meta_120v2.csv",
    "data_usgs/frozen_panel_v1.json",
    "data_usgs/huc_metadata_usgs_v1.csv",
    "data_usgs/huc_metadata_usgs_v1.provenance.json",
)

REQUIRED_STATE_PATHS = {
    "namespace",
    "run_directory",
    "work_order",
    "intent",
    "raw_nwis_root",
    "acquisition_request_map",
    "temporal_outcomes",
    "external_outcomes",
    "acquisition_manifest",
    "availability_registry",
    "outcome_quality_audit",
    "outcome_qc_gate",
    "approved_target_sensitivity",
    "spatial_sensitivity",
    "probabilistic_evaluation",
    "temporal_predictions",
    "external_predictions",
    "statistics",
    "temporal_coverage_audit",
    "report",
    "receipt",
    "receipt_sha256",
}

REQUIRED_RECEIPT_ARTIFACTS = {
    "acquisition_manifest",
    "raw_nwis_snapshot_index",
    "acquisition_request_map",
    "temporal_normalized_outcomes",
    "external_normalized_outcomes",
    "availability_registry",
    "outcome_quality_audit",
    "outcome_qc_gate",
    "approved_target_sensitivity",
    "spatial_sensitivity",
    "probabilistic_evaluation",
    "temporal_predictions",
    "external_predictions",
    "statistics",
    "temporal_coverage_audit",
    "report",
}

REQUIRED_POSTOPEN_CATEGORIES = {
    "canonical_development",
    "authorization",
    "inference_gates",
    "registries",
    "candidate_evidence",
    "model_suite",
    "model_bundles",
    "prelabel_chronology",
    "prelabel_inputs",
    "raw_meteorology",
    "opening_intent",
    "raw_nwis",
    "normalized_outcomes",
    "trusted_predictions",
    "availability",
    "sensitivity_audits",
    "outcome_qc",
    "probabilistic_evaluation",
    "statistics",
    "temporal_coverage",
    "report",
    "receipt",
    "environment_attestations",
    "reproducibility_lock",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    # This is the lineage-hash canonical form used by thermoroute.repro.
    # JSON files may be serialized with ensure_ascii=False for readability,
    # but self/lineage digests deliberately use the stdlib default escaping so
    # roots and interpreter paths containing non-ASCII text hash identically
    # in the live opening and in the independent release verifier.
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _matches_source_inventory(relative: str) -> bool:
    """Match the frozen source globs, including ``**`` matching zero levels."""
    for pattern in SOURCE_INVENTORY_PATTERNS:
        if fnmatch(relative, pattern):
            return True
        if "/**/" in pattern and fnmatch(relative, pattern.replace("/**/", "/")):
            return True
    return False


def _working_source_inventory_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for pattern in SOURCE_INVENTORY_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                paths.add(path.relative_to(root).as_posix())
    return paths


def _is_model_control_path(relative: str) -> bool:
    if relative in PROTECTED_EXACT_FILES:
        return True
    if any(
        relative == directory or relative.startswith(f"{directory}/")
        for directory in PROTECTED_DIRECTORIES
    ):
        return True
    return "/" not in relative and any(
        fnmatch(relative, pattern) for pattern in PROTECTED_ROOT_PATTERNS
    )


def _working_model_control_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for directory in PROTECTED_DIRECTORIES:
        base = root / directory
        if not base.exists():
            continue
        if base.is_symlink() or not base.is_dir():
            paths.add(directory)
            continue
        for path in base.rglob("*"):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                raise ValueError(
                    "compiled Python cache is prohibited in protected release paths: "
                    f"{path.relative_to(root).as_posix()}"
                )
            if path.is_file() or path.is_symlink():
                paths.add(path.relative_to(root).as_posix())
    for relative in PROTECTED_EXACT_FILES:
        path = root / relative
        if path.exists() or path.is_symlink():
            paths.add(relative)
    for path in root.iterdir():
        if (path.is_file() or path.is_symlink()) and any(
            fnmatch(path.name, pattern) for pattern in PROTECTED_ROOT_PATTERNS
        ):
            paths.add(path.name)
    return paths


def _safe_git_environment() -> dict[str, str]:
    """Return a deterministic Git environment with redirectors rejected."""
    forbidden = sorted(
        name
        for name, value in os.environ.items()
        if name in _FORBIDDEN_AMBIENT_GIT_VARIABLES
        or name.startswith("GIT_CONFIG_KEY_")
        or name.startswith("GIT_CONFIG_VALUE_")
        or (name == "GIT_NO_REPLACE_OBJECTS" and value != "1")
    )
    if forbidden:
        raise ValueError(
            "ambient Git repository/configuration override is prohibited: "
            f"{forbidden}"
        )
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _run_git(
    repository: Path,
    *arguments: str,
    text: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run Git without replacement refs or ambient repository redirection."""
    return subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-c",
            "core.useReplaceRefs=false",
            "-C",
            str(repository),
            *arguments,
        ],
        env=_safe_git_environment(),
        text=text,
        capture_output=True,
        check=check,
    )


def _assert_safe_git_repository(root: Path, *, bare: bool = False) -> None:
    """Reject history overlays and incomplete object stores before auditing Git."""
    root = root.resolve()
    if bare:
        state = _run_git(root, "rev-parse", "--is-bare-repository", text=True)
        if state.returncode or state.stdout.strip() != "true":
            raise ValueError("Git bundle audit requires the exact bare repository")
    else:
        top = _run_git(root, "rev-parse", "--show-toplevel", text=True)
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            raise ValueError("release audit requires the exact repository Git root")
    shallow = _run_git(root, "rev-parse", "--is-shallow-repository", text=True)
    if shallow.returncode or shallow.stdout.strip() != "false":
        raise ValueError("shallow or indeterminate Git history is prohibited")
    replacements = _run_git(
        root, "for-each-ref", "--format=%(refname)", "refs/replace/", text=True
    )
    replacement_refs = [line for line in replacements.stdout.splitlines() if line]
    if replacements.returncode or replacement_refs:
        raise ValueError(f"Git replacement refs are prohibited: {replacement_refs[:10]}")
    for label, relative in (
        ("legacy grafts", "info/grafts"),
        ("object alternates", "objects/info/alternates"),
    ):
        location = _run_git(
            root,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            relative,
            text=True,
        )
        if location.returncode or not location.stdout.strip():
            raise ValueError(f"cannot resolve Git {label} location")
        path = Path(location.stdout.strip())
        if path.exists() or path.is_symlink():
            raise ValueError(f"Git {label} are prohibited: {path}")


def assert_no_hidden_git_index_flags(root: str | Path) -> None:
    """Reject index flags that can conceal tracked worktree changes from Git."""
    root = Path(root).resolve()
    _assert_safe_git_repository(root)
    listed = _run_git(root, "ls-files", "-v", "-z")
    if listed.returncode:
        raise ValueError("cannot audit Git assume-unchanged/skip-worktree flags")
    hidden: list[str] = []
    for record in listed.stdout.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise ValueError("Git index flag audit returned a malformed record")
        tag = chr(record[0])
        if tag == "S" or tag.islower():
            hidden.append(record[2:].decode("utf-8", errors="strict"))
    if hidden:
        raise ValueError(
            "Git index contains forbidden assume-unchanged/skip-worktree flags: "
            f"{sorted(hidden)[:10]}"
        )


def _git_commits_between(repository: Path, start: str, end: str) -> list[str]:
    result = _run_git(repository, "rev-list", "--reverse", f"{start}..{end}", text=True)
    if result.returncode:
        raise ValueError("cannot enumerate Git commits for release audit")
    return [line for line in result.stdout.splitlines() if line]


def _git_path_exists(repository: Path, commit: str, relative: str) -> bool:
    result = _run_git(repository, "cat-file", "-e", f"{commit}:{relative}")
    if result.returncode not in {0, 1, 128}:
        raise ValueError("cannot inspect a Git path lifetime")
    return result.returncode == 0


def _git_path_creation_commits(
    repository: Path, tip: str, relative: str
) -> list[str]:
    """Find births over the complete reachable DAG, without path simplification."""
    history = _run_git(repository, "rev-list", "--reverse", "--parents", tip, text=True)
    if history.returncode:
        raise ValueError("cannot enumerate complete Git history for path lifetime")
    created: list[str] = []
    for line in history.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        commit, parents = fields[0], fields[1:]
        if not _git_path_exists(repository, commit, relative):
            continue
        if not parents or all(
            not _git_path_exists(repository, parent, relative) for parent in parents
        ):
            created.append(commit)
    return created


def _git_commit_name_status(repository: Path, commit: str) -> list[tuple[str, str]]:
    result = _run_git(
        repository,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-status",
        "--no-renames",
        "-r",
        "-m",
        "-z",
        commit,
    )
    if result.returncode:
        raise ValueError("cannot audit paths changed by a Git commit")
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise ValueError("Git commit path audit returned malformed name-status data")
    changes: list[tuple[str, str]] = []
    for offset in range(0, len(fields), 2):
        try:
            status = fields[offset].decode("ascii", errors="strict")
            relative = fields[offset + 1].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("Git commit path audit contains an invalid path") from exc
        changes.append((status, relative))
    return changes


def _relative(root: Path, path: Path, *, label: str) -> str:
    root, path = root.resolve(), path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} path escapes the release root")
    return path.relative_to(root).as_posix()


def _resolve_release_path(
    root: Path,
    value: object,
    *,
    label: str,
    base: Path | None = None,
    expected_sha256: str | None = None,
) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} path must be a non-empty release-relative path")
    posix = PurePosixPath(value)
    if (
        "\\" in value
        or posix.as_posix() != value
        or any(part in {".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"{label} path is not canonical: {value}")
    root = root.resolve()
    candidates = [root / Path(*posix.parts)]
    if base is not None:
        base = base.resolve()
        if base != root and root not in base.parents:
            raise ValueError(f"{label} base escapes the release root")
        candidate = base / Path(*posix.parts)
        if candidate not in candidates:
            candidates.append(candidate)
    inside = [
        candidate for candidate in candidates
        if candidate == root or root in candidate.parents
    ]
    if not inside:
        raise ValueError(f"{label} path escapes the release root")
    existing: list[Path] = []
    for candidate in inside:
        state = _release_entry_state(root, candidate, label=label)
        if state is None:
            continue
        if stat.S_ISREG(state.st_mode):
            if state.st_nlink != 1:
                raise ValueError(f"{label} artifact is hard-linked")
        elif not stat.S_ISDIR(state.st_mode):
            raise ValueError(f"{label} artifact is not a regular file or directory")
        existing.append(candidate)
    if expected_sha256 is not None:
        matched = [
            candidate for candidate in existing
            if candidate.is_file() and sha256_file(candidate) == expected_sha256
        ]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1 and len({str(path) for path in matched}) == 1:
            return matched[0]
        if existing:
            raise ValueError(f"{label} checksum mismatch")
    if len(existing) != 1:
        reason = "is absent" if not existing else "is ambiguous"
        raise ValueError(f"{label} path {reason}: {value}")
    return existing[0]


def _release_entry_state(
    root: Path,
    path: Path,
    *,
    label: str,
) -> os.stat_result | None:
    """Lstat one release path without accepting aliases through symlinks."""
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} artifact escapes the release root") from exc
    current = root
    try:
        root_state = current.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(root_state.st_mode):
        raise ValueError(f"{label} release root is a symlink")
    state = root_state
    parts = relative.parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            state = current.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"{label} path contains a symlink: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(state.st_mode):
            raise ValueError(f"{label} path crosses a non-directory entry")
    return state


def _binding_for(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": _relative(root, path, label="closure artifact"),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _category_bindings(
    root: Path, categories: Mapping[str, set[Path]]
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for category in sorted(categories):
        paths = sorted(
            categories[category], key=lambda path: _relative(root, path, label=category)
        )
        if not paths:
            raise ValueError(f"release closure category is empty: {category}")
        output[category] = [_binding_for(root, path) for path in paths]
    return output


def _add_path(
    root: Path,
    categories: dict[str, set[Path]],
    category: str,
    path: Path,
) -> list[Path]:
    root = root.resolve()
    if not path.is_absolute():
        path = root / path
    if path != root and root not in path.parents:
        raise ValueError(f"{category} artifact escapes the release root")
    state = _release_entry_state(root, path, label=category)
    if state is None:
        raise ValueError(f"{category} artifact is absent: {path}")
    if stat.S_ISREG(state.st_mode):
        if state.st_nlink != 1:
            raise ValueError(f"{category} artifact is hard-linked")
        categories.setdefault(category, set()).add(path)
        return [path]
    if not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"{category} artifact is absent: {path}")
    files = []
    for member in sorted(path.rglob("*")):
        member_state = _release_entry_state(root, member, label=category)
        if member_state is None:
            raise ValueError(f"{category} directory member disappeared: {member}")
        if stat.S_ISREG(member_state.st_mode):
            if member_state.st_nlink != 1:
                raise ValueError(
                    f"{category} directory contains a hard-linked file: {member}"
                )
            categories.setdefault(category, set()).add(member)
            files.append(member)
        elif not stat.S_ISDIR(member_state.st_mode):
            raise ValueError(f"{category} directory contains a non-regular entry")
    if not files:
        raise ValueError(f"{category} artifact directory is empty: {path}")
    return files


def _add_binding(
    root: Path,
    categories: dict[str, set[Path]],
    category: str,
    binding: object,
    *,
    label: str,
    base: Path | None = None,
) -> Path:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} binding is malformed")
    expected = binding.get("sha256")
    if expected is not None and (
        not isinstance(expected, str) or len(expected) != 64
    ):
        raise ValueError(f"{label} binding has an invalid SHA-256")
    path = _resolve_release_path(
        root,
        binding.get("path"),
        label=label,
        base=base,
        expected_sha256=expected if isinstance(expected, str) else None,
    )
    _add_path(root, categories, category, path)
    if path.is_dir():
        for name, key in (
            ("metadata.json", "metadata_sha256"),
            ("weights.pt", "weights_sha256"),
        ):
            digest = binding.get(key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{label} directory lacks {key}")
            member = path / name
            if not member.is_file() or sha256_file(member) != digest:
                raise ValueError(f"{label} directory {name} checksum mismatch")
    return path


def _walk_json_dependencies(
    root: Path,
    categories: dict[str, set[Path]],
    category: str,
    json_path: Path,
    *,
    visited: set[tuple[str, str]] | None = None,
) -> None:
    """Collect every file/directory binding reachable from one JSON document."""
    if visited is None:
        visited = set()
    identity = (category, str(json_path.resolve()))
    if identity in visited:
        return
    visited.add(identity)
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse closure JSON: {json_path}") from exc

    def walk(value: object, *, base: Path, trail: str) -> None:
        if isinstance(value, Mapping):
            if "path" in value and any(
                key in value
                for key in ("sha256", "metadata_sha256", "weights_sha256")
            ):
                path = _add_binding(
                    root,
                    categories,
                    category,
                    value,
                    label=f"{category} dependency {trail}",
                    base=base,
                )
                if path.is_file() and path.suffix == ".json":
                    _walk_json_dependencies(
                        root, categories, category, path, visited=visited
                    )
                elif path.is_dir():
                    for child in sorted(path.rglob("*.json")):
                        _walk_json_dependencies(
                            root, categories, category, child, visited=visited
                        )
            for key, item in value.items():
                walk(item, base=base, trail=f"{trail}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, base=base, trail=f"{trail}[{index}]")

    walk(document, base=json_path.parent, trail=json_path.name)
    # Snapshot indexes use index-relative metadata/response paths.  Including
    # the exact directory makes those raw bytes part of the file-level closure.
    if json_path.name == "snapshot_index.json":
        _add_path(root, categories, category, json_path.parent)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_canonical_outcome_qc_module(root: Path) -> Any:
    """Load the release root's one shared outcome-QC implementation.

    A private package name prevents an already-imported ``thermoroute`` from a
    different checkout from satisfying this security-sensitive import.  The
    source bytes are separately covered by the fixed-code and Git closures.
    """
    source_dir = (root / "src" / "thermoroute").resolve()
    source = source_dir / "outcome_qc.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("canonical outcome-QC verifier source is absent")
    fingerprint = hashlib.sha256(
        str(root).encode("utf-8") + b"\0" + source.read_bytes()
    ).hexdigest()[:20]
    package_name = f"_thermoroute_release_{fingerprint}"
    module_name = f"{package_name}.outcome_qc"
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != source:
            raise ValueError("cached outcome-QC verifier is noncanonical")
        return existing

    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(source_dir)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot construct canonical outcome-QC verifier import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise ValueError("cannot load canonical outcome-QC verifier") from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    if Path(getattr(module, "__file__", "")).resolve() != source:
        raise ValueError("imported outcome-QC verifier is noncanonical")
    for name in (
        "validate_outcome_qc_policy",
        "validate_outcome_qc_gate_structure",
    ):
        if not callable(getattr(module, name, None)):
            raise ValueError("canonical outcome-QC verifier API is incomplete")
    return module


def _load_canonical_coverage_bridge_module(root: Path) -> Any:
    """Load the Git-verified archive bridge under an isolated package name.

    ``verify_release_profile`` must establish the archive's compute-commit and
    fixed-code byte identity before this function is reachable.  The private
    package prevents ambient ``thermoroute`` modules from satisfying relative
    imports of the coverage core or reproducibility helpers.
    """
    source_dir = (root / "src" / "thermoroute").resolve()
    source = source_dir / "coverage_bridge.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("canonical temporal-coverage bridge source is absent")
    fingerprint = hashlib.sha256(
        str(root).encode("utf-8") + b"\0coverage\0" + source.read_bytes()
    ).hexdigest()[:20]
    package_name = f"_thermoroute_coverage_release_{fingerprint}"
    module_name = f"{package_name}.coverage_bridge"
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != source:
            raise ValueError("cached temporal-coverage bridge is noncanonical")
        return existing

    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(source_dir)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot construct canonical temporal-coverage import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise ValueError("cannot load canonical temporal-coverage bridge") from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    if Path(getattr(module, "__file__", "")).resolve() != source:
        raise ValueError("imported temporal-coverage bridge is noncanonical")
    if not callable(
        getattr(module, "replay_temporal_coverage_from_physical_files", None)
    ):
        raise ValueError("canonical temporal-coverage bridge API is incomplete")
    return module


def _load_protocol_seal(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Validate the current final protocol bytes against their canonical seal."""
    seal_path = _resolve_release_path(root, PROTOCOL_SEAL_PATH, label="protocol seal")
    seal = _load_json(seal_path, label="protocol seal")
    if (
        seal.get("format") != PROTOCOL_SEAL_FORMAT
        or seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
        or seal.get("protocol_id") != protocol.get("protocol_id")
    ):
        raise ValueError("release protocol seal is malformed or not prelabel")
    original = seal.get("original_preregistration")
    final = seal.get("final_prelabel_protocol")
    if not isinstance(original, Mapping) or not isinstance(final, Mapping):
        raise ValueError("release protocol seal lacks original/final sections")
    authoritative = str(protocol.get("authoritative_protocol_commit", ""))
    final_commit = str(final.get("commit", ""))
    if (
        original.get("commit") != authoritative
        or not re.fullmatch(r"[0-9a-f]{40}", authoritative)
        or not re.fullmatch(r"[0-9a-f]{40}", final_commit)
    ):
        raise ValueError("release protocol seal commit identities changed")
    expected = (
        (
            "final JSON",
            final.get("json"),
            "protocols/route_a_confirmatory_v1.json",
        ),
        (
            "final Markdown",
            final.get("markdown"),
            "protocols/route_a_confirmatory_protocol.md",
        ),
    )
    for label, binding, relative in expected:
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError(f"protocol seal {label} binding changed")
        path = _resolve_release_path(
            root,
            binding.get("path"),
            label=f"protocol seal {label}",
            expected_sha256=(
                str(binding.get("sha256"))
                if isinstance(binding.get("sha256"), str)
                else None
            ),
        )
        if not path.is_file() or not re.fullmatch(
            r"[0-9a-f]{64}", str(binding.get("sha256", ""))
        ):
            raise ValueError(f"protocol seal {label} checksum is malformed")
    attestation = seal.get("prelabel_attestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("external_timestamp_or_public_preregistration") is not False
        or attestation.get("independent_custodian_or_worm_storage") is not False
    ):
        raise ValueError("protocol seal overstates external/WORM evidence")
    return seal_path, seal


def _chronology_self_sha256(value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _chronology_artifact(
    root: Path,
    categories: dict[str, set[Path]],
    value: object,
    *,
    label: str,
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "sha256", "byte_count", "git_blob_oid"
    }:
        raise ValueError(f"{label} chronology binding schema changed")
    path = _add_binding(
        root, categories, "prelabel_chronology", value, label=label
    )
    if path.stat().st_size != value.get("byte_count"):
        raise ValueError(f"{label} chronology byte count changed")
    if not re.fullmatch(r"[0-9a-f]{40,64}", str(value.get("git_blob_oid", ""))):
        raise ValueError(f"{label} chronology Git blob OID is malformed")
    return path


def _authorization_path(value: object, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"authorization {label} binding is malformed")
    path = value.get("path")
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError(f"authorization {label} path is malformed")
    return path


def _validate_prelabel_chronology_structure(
    root: Path,
    categories: dict[str, set[Path]],
    authorization: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Close the opened release over the repository-internal order receipt."""
    binding = authorization.get("prelabel_chronology")
    if not isinstance(binding, Mapping) or set(binding) != {
        "path", "sha256", "format", "status", "order", "evidence_scope"
    }:
        raise ValueError("authorization lacks the exact prelabel chronology binding")
    chronology_path = _add_binding(
        root,
        categories,
        "prelabel_chronology",
        binding,
        label="authorized prelabel chronology",
    )
    if _relative(root, chronology_path, label="prelabel chronology") != CHRONOLOGY_PATH:
        raise ValueError("authorization uses a noncanonical prelabel chronology path")
    chronology = _load_json(chronology_path, label="prelabel chronology")
    stable = dict(chronology)
    self_digest = stable.pop("receipt_self_sha256", None)
    if self_digest != _chronology_self_sha256(stable):
        raise ValueError("prelabel chronology self hash is inconsistent")
    if (
        chronology.get("format") != CHRONOLOGY_FORMAT
        or chronology.get("status") != CHRONOLOGY_STATUS
        or chronology.get("external_timestamp_or_public_preregistration") is not False
        or chronology.get("independent_custodian_or_worm_storage") is not False
        or chronology.get("evidence_scope") != CHRONOLOGY_EVIDENCE_SCOPE
        or chronology.get("post_freeze_artifact_mutation_count") != 0
        or chronology.get("fallback_if_validation_fails")
        != "TRANSDUCTIVE_RETROSPECTIVE_EXPLORATION_CONFIRMATION_CLAIMS_PROHIBITED"
    ):
        raise ValueError("prelabel chronology status or evidence-scope disclosure changed")
    order = chronology.get("order")
    if not isinstance(order, Mapping) or set(order) != {
        "model_freeze_commit",
        "input_evidence_commit",
        "receipt_creation_base_commit",
        "strict_order_verified",
    } or order.get("strict_order_verified") is not True:
        raise ValueError("prelabel chronology order schema changed")
    commits = [
        str(order[key]) for key in (
            "model_freeze_commit",
            "input_evidence_commit",
            "receipt_creation_base_commit",
        )
    ]
    if any(not re.fullmatch(r"[0-9a-f]{40}", commit) for commit in commits) or len(
        set(commits)
    ) != 3:
        raise ValueError("prelabel chronology commit identities are malformed")
    expected_authorized = {
        "format": chronology["format"],
        "status": chronology["status"],
        "order": dict(order),
        "evidence_scope": chronology["evidence_scope"],
    }
    if any(binding.get(key) != value for key, value in expected_authorized.items()):
        raise ValueError("authorization chronology metadata differs from its receipt")

    protocol = authorization.get("protocol")
    registries = authorization.get("registries")
    if not isinstance(protocol, Mapping) or not isinstance(registries, Mapping):
        raise ValueError("authorization cannot resolve chronology dependencies")
    expected_paths = {
        "protocol_seal": _authorization_path(protocol.get("seal"), label="protocol seal"),
        "model_suite": _authorization_path(
            authorization.get("model_suite"), label="model suite"
        ),
        "development_replay": _authorization_path(
            authorization.get("development_replay"), label="development replay"
        ),
        "candidate_table": _authorization_path(
            registries.get("candidate_table"), label="candidate table"
        ),
        "candidate_provenance": _authorization_path(
            registries.get("candidate_provenance"), label="candidate provenance"
        ),
        "candidate_snapshot_index": _authorization_path(
            registries.get("candidate_snapshot_index"),
            label="candidate snapshot index",
        ),
        "external_registry": _authorization_path(
            registries.get("external"), label="external registry"
        ),
        "external_lock": _authorization_path(
            registries.get("external_lock"), label="external lock"
        ),
        "input_manifest": _authorization_path(
            authorization.get("actual_inputs"), label="actual inputs"
        ),
    }
    if chronology.get("paths") != expected_paths:
        raise ValueError("prelabel chronology binds another authorized evidence set")

    protocol_history = chronology.get("protocol_history")
    if not isinstance(protocol_history, Mapping):
        raise ValueError("prelabel chronology lacks protocol history")
    seal = protocol_history.get("seal")
    seal_path = _chronology_artifact(
        root, categories, seal, label="chronology protocol seal"
    )
    if (
        _relative(root, seal_path, label="chronology protocol seal")
        != expected_paths["protocol_seal"]
        or not isinstance(seal, Mapping)
        or seal.get("sha256") != protocol.get("seal", {}).get("sha256")
        or protocol_history.get("original_commit")
        != protocol.get("authoritative_commit")
        or protocol_history.get("final_prelabel_commit")
        != protocol.get("final_prelabel_commit")
    ):
        raise ValueError("prelabel chronology protocol history changed")
    declared = protocol_history.get("declared_git_show_bindings")
    if not isinstance(declared, list) or len(declared) != 3:
        raise ValueError("prelabel chronology Git-show registry is incomplete")
    roles = {
        str(item.get("role")) for item in declared if isinstance(item, Mapping)
    }
    if roles != {"original_markdown", "final_json", "final_markdown"}:
        raise ValueError("prelabel chronology Git-show roles changed")

    required_gate_paths = {
        "src/thermoroute/chronology.py",
        "src/thermoroute/outcome_qc.py",
        "scripts/28_freeze_prelabel_chronology.py",
        "tests/test_chronology.py",
        "protocols/route_a_outcome_qc_policy_v1.json",
    }
    observed_by_field: dict[str, set[str]] = {}
    bindings_by_path: dict[str, Mapping[str, Any]] = {}
    for field, minimum in (
        ("required_gate_files_at_model_freeze", len(required_gate_paths)),
        ("model_source_control_artifacts", 1),
        ("model_freeze_artifacts", 1),
        ("input_evidence_artifacts", 1),
    ):
        values = chronology.get(field)
        if not isinstance(values, list) or len(values) < minimum:
            raise ValueError(f"prelabel chronology {field} is incomplete")
        observed: set[str] = set()
        for index, item in enumerate(values):
            path = _chronology_artifact(
                root, categories, item, label=f"chronology {field}[{index}]"
            )
            relative = _relative(root, path, label=f"chronology {field}")
            if relative in observed:
                raise ValueError(f"prelabel chronology {field} duplicates {relative}")
            observed.add(relative)
            assert isinstance(item, Mapping)
            bindings_by_path[relative] = item
        observed_by_field[field] = observed
        if field == "required_gate_files_at_model_freeze" and observed != required_gate_paths:
            raise ValueError("prelabel chronology required-gate registry changed")
    declared_control = observed_by_field["model_source_control_artifacts"]
    if declared_control != _working_model_control_paths(root):
        raise ValueError(
            "archive source/control path set differs from prelabel chronology"
        )
    source_paths = _working_source_inventory_paths(root)
    source_inventory = {
        relative: str(bindings_by_path[relative]["sha256"])
        for relative in sorted(source_paths)
        if relative in bindings_by_path
    }
    if (
        not source_paths
        or set(source_inventory) != source_paths
        or chronology.get("source_tree_sha256") != _sha256_json(source_inventory)
    ):
        raise ValueError("prelabel chronology source-tree lineage changed")
    if not {
        expected_paths["model_suite"], expected_paths["development_replay"]
    } <= observed_by_field["model_freeze_artifacts"]:
        raise ValueError("prelabel chronology omits its model suite or development replay")
    required_input_evidence = {
        expected_paths[key] for key in (
            "candidate_table",
            "candidate_provenance",
            "candidate_snapshot_index",
            "external_registry",
            "external_lock",
            "input_manifest",
        )
    }
    if not required_input_evidence <= observed_by_field["input_evidence_artifacts"]:
        raise ValueError("prelabel chronology omits required candidate/input evidence")

    absence = chronology.get("absence_at_model_freeze")
    if (
        not isinstance(absence, Mapping)
        or absence.get("present_paths") != []
        or not isinstance(absence.get("checked_paths"), list)
        or not absence["checked_paths"]
    ):
        raise ValueError("prelabel chronology absence audit is incomplete")
    control = chronology.get("post_model_control_audit")
    if (
        not isinstance(control, Mapping)
        or set(control) != {
            "protected_directories",
            "protected_exact_files",
            "protected_root_patterns",
            "committed_touches",
            "worktree_changes",
        }
        or control.get("protected_directories") != list(PROTECTED_DIRECTORIES)
        or control.get("protected_exact_files") != list(PROTECTED_EXACT_FILES)
        or control.get("protected_root_patterns") != list(PROTECTED_ROOT_PATTERNS)
        or control.get("committed_touches") != []
        or control.get("worktree_changes") != []
    ):
        raise ValueError("prelabel chronology post-model control audit is not clean")
    return chronology_path, chronology


def _canonical_categories(root: Path) -> dict[str, set[Path]]:
    categories: dict[str, set[Path]] = {
        "canonical_development": set(),
        "reproducibility_lock": set(),
    }
    for relative in CANONICAL_DEVELOPMENT_PATHS:
        path = _resolve_release_path(root, relative, label="canonical development")
        _add_path(root, categories, "canonical_development", path)
    huc_root = _resolve_release_path(
        root,
        "data_usgs/raw_snapshots/huc-v1",
        label="canonical raw HUC evidence",
    )
    _add_path(root, categories, "canonical_development", huc_root)
    hashed_lock = _resolve_release_path(
        root, REPRODUCIBILITY_LOCK, label="fully hashed Python 3.12 lock"
    )
    _verify_fully_hashed_lock(hashed_lock)
    _add_path(root, categories, "reproducibility_lock", hashed_lock)
    return categories


def _verify_fully_hashed_lock(path: Path) -> None:
    """Fail closed on unpinned or unhashed requirement blocks."""
    requirement = re.compile(
        r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?)"
        r"==(?P<version>[^\s\\]+)(?:\s+\\)?$"
    )
    hash_token = re.compile(r"--hash=sha256:[0-9a-f]{64}(?=\s|\\|$)")
    current: str | None = None
    current_has_hash = False
    names: set[str] = set()
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1].isspace():
            if "--hash=" in stripped:
                valid = hash_token.findall(stripped)
                if not valid or re.sub(hash_token, "", stripped).strip(" \\"):
                    raise ValueError(f"hashed lock has a malformed hash at line {number}")
                current_has_hash = True
                count += len(valid)
            elif not stripped.startswith("#"):
                raise ValueError(f"hashed lock has an unsupported continuation at line {number}")
            continue
        if current is not None and not current_has_hash:
            raise ValueError(f"hashed lock requirement has no SHA-256: {current}")
        match = requirement.fullmatch(stripped)
        if match is None:
            raise ValueError(f"hashed lock requirement is not exactly pinned at line {number}")
        current = match.group("name").lower().replace("_", "-")
        if current in names:
            raise ValueError(f"hashed lock duplicates requirement: {current}")
        names.add(current)
        current_has_hash = False
    if current is None or not current_has_hash or count == 0:
        raise ValueError("fully hashed lock is empty or its final requirement is unhashed")


def _merge_categories(
    target: dict[str, set[Path]], source: Mapping[str, Iterable[Path]]
) -> None:
    for category, paths in source.items():
        target.setdefault(category, set()).update(path.resolve() for path in paths)


def _validate_authorization_structure(
    root: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    authorization = _load_json(authorization_path, label="Route-A authorization")
    if (
        authorization.get("format") != AUTHORIZATION_FORMAT
        or authorization.get("status") != "AUTHORIZED_LABELS_STILL_SEALED"
    ):
        raise ValueError("post-opening release lacks a production sealed authorization")
    self_hashed = dict(authorization)
    claimed = self_hashed.pop("authorization_self_sha256", None)
    if not isinstance(claimed, str) or claimed != _sha256_json(self_hashed):
        raise ValueError("opening authorization self hash is inconsistent")
    opening_id = authorization.get("opening_id")
    if not isinstance(opening_id, str) or len(opening_id) != 24:
        raise ValueError("opening authorization lacks a stable opening_id")
    source = authorization.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("authorization_path")
        != _relative(root, authorization_path, label="authorization")
    ):
        raise ValueError("authorization source policy names another authorization path")
    state = authorization.get("state_paths")
    if not isinstance(state, Mapping):
        raise ValueError("authorization canonical state-path registry is malformed")
    if set(state) != REQUIRED_STATE_PATHS:
        missing = sorted(REQUIRED_STATE_PATHS - set(state))
        extra = sorted(set(state) - REQUIRED_STATE_PATHS)
        raise ValueError(
            "authorization canonical state-path registry changed: "
            f"missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, str) or not value for value in state.values()):
        raise ValueError("authorization contains a malformed canonical state path")
    namespace = str(state["namespace"])
    if len(namespace) != 24 or any(character not in "0123456789abcdef" for character in namespace):
        raise ValueError("authorization state namespace is not a 24-hex digest")
    base = f"outputs/confirmatory/route_a_{namespace}"
    expected = {
        "run_directory": base,
        "work_order": f"{base}/acquisition_work_order_v1.json",
        "intent": f"{base}/opening_intent_v1.json",
        "raw_nwis_root": f"{base}/acquisition/raw_nwis_v1",
        "acquisition_request_map": f"{base}/acquisition/source_request_map_v1.json",
        "temporal_outcomes": f"{base}/acquisition/temporal_outcomes_v1.parquet",
        "external_outcomes": f"{base}/acquisition/external_outcomes_v1.parquet",
        "acquisition_manifest": f"{base}/acquisition/acquisition_manifest_v1.json",
        "availability_registry": f"{base}/trusted/availability_registry_v1.csv",
        "outcome_quality_audit": f"{base}/trusted/outcome_quality_audit_v1.json",
        "outcome_qc_gate": f"{base}/trusted/outcome_qc_gate_v1.json",
        "approved_target_sensitivity": f"{base}/trusted/approved_target_sensitivity_v1.json",
        "spatial_sensitivity": f"{base}/trusted/spatial_sensitivity_v1.json",
        "probabilistic_evaluation": f"{base}/trusted/probabilistic_evaluation_v1.json",
        "temporal_predictions": f"{base}/trusted/temporal_predictions_v1.parquet",
        "external_predictions": f"{base}/trusted/external_predictions_v1.parquet",
        "statistics": f"{base}/trusted/statistics_v1.json",
        "temporal_coverage_audit": (
            f"{base}/trusted/temporal_coverage_audit_v1.json"
        ),
        "report": f"{base}/trusted/report_v1.md",
        "receipt": f"{base}/opening_receipt_v1.json",
        "receipt_sha256": f"{base}/opening_receipt_v1.sha256",
    }
    wrong = {key: state.get(key) for key, value in expected.items() if state.get(key) != value}
    if wrong:
        raise ValueError(f"authorization state paths leave the canonical namespace: {wrong}")
    qc_policy = authorization.get("outcome_qc_policy")
    if (
        not isinstance(qc_policy, Mapping)
        or set(qc_policy) != {"path", "sha256", "format", "policy_id", "required"}
        or qc_policy.get("path") != "protocols/route_a_outcome_qc_policy_v1.json"
        or qc_policy.get("format") != "thermoroute.route-a-outcome-qc-policy.v1"
        or qc_policy.get("policy_id") != "route-a-outcome-qc-and-influence-001"
        or qc_policy.get("required") is not True
    ):
        raise ValueError("authorization lacks the canonical required outcome-QC policy")
    coverage_policy = authorization.get("temporal_coverage_policy")
    if (
        not isinstance(coverage_policy, Mapping)
        or set(coverage_policy)
        != {"path", "sha256", "format", "policy_id", "status", "required"}
        or coverage_policy.get("path") != TEMPORAL_COVERAGE_POLICY_PATH
        or coverage_policy.get("sha256") != TEMPORAL_COVERAGE_POLICY_SHA256
        or coverage_policy.get("format") != TEMPORAL_COVERAGE_POLICY_FORMAT
        or coverage_policy.get("policy_id") != TEMPORAL_COVERAGE_POLICY_ID
        or coverage_policy.get("status") != "FROZEN_PRELABEL_OUTCOME_FREE"
        or coverage_policy.get("required") is not True
    ):
        raise ValueError(
            "authorization lacks the canonical temporal-coverage policy"
        )
    amendment = authorization.get("inference_amendment")
    if (
        not isinstance(amendment, Mapping)
        or set(amendment) != {
            "path", "sha256", "format", "amendment_id", "seal",
            "final_prelabel_commit",
        }
        or amendment.get("path")
        != "protocols/route_a_inference_amendment_v1.json"
        or amendment.get("format")
        != "thermoroute.route-a-inference-amendment.v1"
        or amendment.get("amendment_id")
        != "route-a-prelabel-inference-scope-014"
        or not isinstance(amendment.get("seal"), Mapping)
        or amendment["seal"].get("path")
        != "protocols/route_a_inference_amendment_seal_v1.json"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(amendment.get("final_prelabel_commit", ""))
        )
    ):
        raise ValueError("authorization lacks the canonical inference amendment seal")
    inference_gate = authorization.get("inference_gate")
    if (
        not isinstance(inference_gate, Mapping)
        or set(inference_gate) != {
            "path", "sha256", "format", "status", "claim_eligible",
            "analysis_mode", "policy_sha256",
        }
        or inference_gate.get("path")
        != "outputs/prelabel/route_a_inference_gate_v1.json"
        or inference_gate.get("format")
        != "thermoroute.route-a-inference-gate.v1"
        or inference_gate.get("status") != "FAIL_CLOSED_DESCRIPTIVE_ONLY"
        or inference_gate.get("claim_eligible") is not False
        or inference_gate.get("analysis_mode")
        != "FIXED_COHORT_DESCRIPTIVE_ONLY"
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(inference_gate.get("policy_sha256", ""))
        )
    ):
        raise ValueError("authorization inference gate is not fail-closed")
    runtime = authorization.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("authorization lacks an environment attestation")
    required_runtime = {
        "format", "requirements_lock", "hashed_requirements_lock",
        "installed_version_validation", "numerical_runtime_contract",
        "runtime_sha256", "python_executable", "golden_inference_sha256",
        "formal_numerical_policy", "deterministic_child_policy",
    }
    if not required_runtime <= set(runtime):
        raise ValueError("authorization environment attestation is incomplete")
    for key in ("runtime_sha256", "golden_inference_sha256"):
        value = runtime.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"authorization runtime has an invalid {key}")
    return authorization, {key: str(value) for key, value in state.items()}


def validate_postopen_git_dirt(
    root: str | Path, authorization_path: str | Path
) -> dict[str, object]:
    """Allow a clean documentation-only descendant plus canonical opening dirt."""
    root, authorization_path = Path(root).resolve(), Path(authorization_path).resolve()
    authorization, state = _validate_authorization_structure(root, authorization_path)
    return _postopen_revision_contract(
        root, authorization_path, authorization, state, require_git=True
    )


def _postopen_revision_contract(
    root: Path,
    authorization_path: Path,
    authorization: Mapping[str, Any],
    state: Mapping[str, str],
    *,
    require_git: bool,
) -> dict[str, object]:
    """Bind immutable compute code separately from a doc-only manuscript commit."""
    root, authorization_path = root.resolve(), authorization_path.resolve()
    compute_commit = str(
        authorization.get("source", {}).get("git_commit_before_authorization", "")
    )
    if len(compute_commit) != 40 or any(
        character not in "0123456789abcdef" for character in compute_commit
    ):
        if require_git:
            raise ValueError("authorization lacks the computational Git commit")
        compute_commit = "0" * 40
    whitelist = ["README.md", "paper/**"]

    def allowed_document(relative: str) -> bool:
        return relative == "README.md" or relative.startswith("paper/")

    def git(*arguments: str) -> subprocess.CompletedProcess[Any]:
        return _run_git(root, *arguments, text=True)

    top = git("rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != root:
        if require_git:
            raise ValueError("post-opening dirt policy requires the repository Git root")
        return {
            "compute_commit": compute_commit,
            "manuscript_commit": compute_commit,
            "committed_document_whitelist": whitelist,
            "committed_document_diff": [],
            "tracked_changes_allowed": False,
            "staged_changes_allowed": False,
            "untracked_exact": [
                _relative(root, authorization_path, label="authorization")
            ],
            "untracked_prefixes": [state["run_directory"].rstrip("/") + "/"],
        }
    _assert_safe_git_repository(root)
    assert_no_hidden_git_index_flags(root)
    for arguments, label in (
        (("diff", "--name-only"), "tracked worktree"),
        (("diff", "--cached", "--name-only"), "staged"),
    ):
        result = git(*arguments)
        paths = [line for line in result.stdout.splitlines() if line]
        if result.returncode or paths:
            raise ValueError(f"post-opening release has forbidden {label} changes: {paths}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode:
        raise ValueError("cannot audit post-opening Git dirt")
    authorization_relative = _relative(root, authorization_path, label="authorization")
    namespace_prefix = state["run_directory"].rstrip("/") + "/"
    observed: list[str] = []
    forbidden: list[str] = []
    for line in status.stdout.splitlines():
        if not line:
            continue
        if not line.startswith("?? "):
            forbidden.append(line)
            continue
        relative = line[3:]
        observed.append(relative)
        if relative != authorization_relative and not relative.startswith(namespace_prefix):
            forbidden.append(line)
    if authorization_relative not in observed:
        raise ValueError("post-opening authorization is not the expected untracked audit dirt")
    if forbidden:
        raise ValueError(f"post-opening release has extra Git dirt: {forbidden}")
    head = git("rev-parse", "HEAD")
    manuscript_commit = head.stdout.strip()
    ancestor = git("merge-base", "--is-ancestor", compute_commit, manuscript_commit)
    if head.returncode or ancestor.returncode:
        raise ValueError("manuscript commit is not a descendant of the compute commit")
    for commit in _git_commits_between(root, compute_commit, manuscript_commit):
        forbidden_intermediate = [
            f"{status} {relative}"
            for status, relative in _git_commit_name_status(root, commit)
            if status not in {"A", "M"} or not allowed_document(relative)
        ]
        if forbidden_intermediate:
            raise ValueError(
                "post-opening commits modify files outside the documentation "
                f"whitelist or delete/rename documents: {commit}/"
                f"{forbidden_intermediate[:10]}"
            )
    diff = git("diff", "--name-only", f"{compute_commit}..{manuscript_commit}")
    changed_documents = sorted(set(line for line in diff.stdout.splitlines() if line))
    forbidden_committed = [path for path in changed_documents if not allowed_document(path)]
    if diff.returncode or forbidden_committed:
        raise ValueError(
            "post-opening commits modify files outside the documentation whitelist: "
            f"{forbidden_committed}"
        )
    document_bindings = []
    for relative in changed_documents:
        path = _resolve_release_path(root, relative, label="post-opening document")
        if not path.is_file():
            raise ValueError(f"post-opening document was deleted: {relative}")
        document_bindings.append(_binding_for(root, path))
    policy = {
        "compute_commit": compute_commit,
        "manuscript_commit": manuscript_commit,
        "committed_document_whitelist": whitelist,
        "committed_document_diff": document_bindings,
        "tracked_changes_allowed": False,
        "staged_changes_allowed": False,
        "untracked_exact": [authorization_relative],
        "untracked_prefixes": [namespace_prefix],
    }
    return policy


def _required_model_ids(cohort: str) -> set[str]:
    primary = {
        "Persistence", "DampedPersistence", "Climatology",
        "LightGBM", "LSTM", "ThermoRoute",
    }
    if cohort == "external":
        return primary
    return primary | {
        "DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
        "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded",
    }


def _stage09b_release_members() -> tuple[tuple[str, int], ...]:
    ladder = (
        "01_WTEMP", "02_plus_FLOW", "03_plus_TEMP", "04_plus_PRCP",
        "05_plus_RHMEAN", "06_plus_DH", "07_plus_WDSP",
    )
    return (
        *(("PlainMLP-7var", seed) for seed in range(5)),
        *(("PlainCausalTCN-7var", seed) for seed in range(5)),
        *(
            (f"ThermoRoute-ladder-{rung}", seed)
            for rung in ladder for seed in range(3)
        ),
    )


def _stage09b_formal_configuration(
    value: object, *, expected_bridge: object,
) -> dict[str, Any]:
    """Validate every scientific configuration field without archive imports."""
    if not isinstance(value, Mapping):
        raise ValueError("Stage-09b formal configuration is malformed")
    ladder = (
        ("01_WTEMP", ("WTEMP",)),
        ("02_plus_FLOW", ("WTEMP", "FLOW")),
        ("03_plus_TEMP", ("WTEMP", "FLOW", "TEMP")),
        ("04_plus_PRCP", ("WTEMP", "FLOW", "TEMP", "PRCP")),
        ("05_plus_RHMEAN", ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN")),
        ("06_plus_DH", ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH")),
        (
            "07_plus_WDSP",
            ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"),
        ),
    )
    full = list(ladder[-1][1])
    arms = [
        {
            "arm_id": "PlainMLP-7var", "family": "PlainMLP",
            "feature_set": "all_7_variables", "variables": full,
            "seeds": [0, 1, 2, 3, 4],
        },
        {
            "arm_id": "PlainCausalTCN-7var", "family": "PlainCausalTCN",
            "feature_set": "all_7_variables", "variables": full,
            "seeds": [0, 1, 2, 3, 4],
        },
        *[
            {
                "arm_id": f"ThermoRoute-ladder-{rung}", "family": "ThermoRoute",
                "feature_set": f"feature_ladder_{rung}", "variables": list(variables),
                "seeds": [0, 1, 2],
            }
            for rung, variables in ladder
        ],
    ]
    train = {
        "d_model": 40, "encoder_blocks": 2, "kernel_size": 3,
        "dropout": 0.15, "n_experts": 3, "station_embed_dim": 8,
        "lr": 0.002, "weight_decay": 0.0001, "batch_size": 1536,
        "max_epochs": 80, "patience": 12, "grad_clip": 1.0,
        "lambda_event": 0.3, "lambda_residual": 0.01,
        "lambda_crossing": 1.0,
    }
    parameter_counts = {
        "PlainMLP-7var": 38_545,
        "PlainCausalTCN-7var": 38_031,
        "ThermoRoute-ladder-01_WTEMP": 37_775,
        "ThermoRoute-ladder-02_plus_FLOW": 37_896,
        "ThermoRoute-ladder-03_plus_TEMP": 38_018,
        "ThermoRoute-ladder-04_plus_PRCP": 38_139,
        "ThermoRoute-ladder-05_plus_RHMEAN": 38_261,
        "ThermoRoute-ladder-06_plus_DH": 38_383,
        "ThermoRoute-ladder-07_plus_WDSP": 38_505,
    }
    neural_common = {
        "format_version": 2,
        "module": "thermoroute.neural_baselines",
        "future_keys_never_read": ["y", "clim_tgt", "damped_prior", "target_date"],
        "input_keys_read": ["X", "Mask", "station"],
        "output_keys": ["point", "q_lo", "q_med", "q_hi", "event_logit"],
        "point_objective": "mse_conditional_mean",
        "q50_is_independent_from_point": True,
        "quantile_levels": [0.05, 0.5, 0.95],
        "budget_matching_note": (
            "Match trainable parameters, optimiser steps, input schema, "
            "early-stopping rule, and tuning budget externally; the constructor "
            "defaults do not establish fairness."
        ),
        "initialization_seed_policy": "exact declared member seed",
    }
    templates: dict[str, Any] = {}
    for arm in arms:
        arm_id = str(arm["arm_id"])
        variables = list(arm["variables"])
        if arm["family"] == "PlainMLP":
            templates[arm_id] = {
                **neural_common,
                "architecture_id": "plain_history_mlp_v2",
                "class_name": "PlainMLPForecaster",
                "constructor_kwargs": {
                    "n_vars": 7, "context_length": 32, "horizons": [1, 3, 7],
                    "n_stations": 120, "station_agnostic": False,
                    "station_embed_dim": 8, "hidden_dim": 70, "depth": 2,
                    "dropout": 0.15, "min_spread": 0.0001,
                    "init_seed": "member_seed",
                },
                "trainable_parameters": parameter_counts[arm_id],
            }
        elif arm["family"] == "PlainCausalTCN":
            templates[arm_id] = {
                **neural_common,
                "architecture_id": "plain_causal_tcn_v2",
                "class_name": "PlainCausalTCNForecaster",
                "constructor_kwargs": {
                    "n_vars": 7, "context_length": 32, "horizons": [1, 3, 7],
                    "n_stations": 120, "station_agnostic": False,
                    "station_embed_dim": 8, "channels": 54, "blocks": 4,
                    "kernel_size": 3, "dropout": 0.15, "min_spread": 0.0001,
                    "init_seed": "member_seed",
                },
                "trainable_parameters": parameter_counts[arm_id],
            }
        else:
            n_phys = sum(item in {"TEMP", "RHMEAN", "DH", "WDSP"} for item in variables)
            templates[arm_id] = {
                "format_version": 2,
                "architecture_id": "thermoroute_full_v2",
                "module": "thermoroute.thermoroute",
                "class_name": "ThermoRoute",
                "constructor_kwargs": {
                    "n_vars": len(variables), "n_stations": 120,
                    "horizons": [1, 3, 7], "train_config": train,
                    "n_phys": n_phys, "station_agnostic": False,
                    "delta_scale": 1.0, "safety_anchor": "damped",
                },
                "initialization_seed": "member_seed",
                "trainable_parameters": parameter_counts[arm_id],
                "input_variables": variables,
                "initialization_seed_policy": "exact declared member seed",
            }
    hash_policy = "canonical-sort-identity-collections-independent-of-hash-secret"
    formal_policy = {
        "thread_environment": {
            name: "1" for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
        "cublas_workspace_config": ":4096:8",
        "python_hash_environment_declaration": "0",
        "python_hash_randomization_enabled": True,
        "python_hash_policy": hash_policy,
        "required": {
            "threads": 1, "cublas_workspace_config": ":4096:8",
            "python_hash_policy": hash_policy,
            "torch_deterministic_algorithms": True, "tf32": False,
            "float32_matmul_precision": "highest",
        },
        "torch": {
            "num_threads": 1, "num_interop_threads": 1,
            "deterministic_algorithms": True, "cudnn_deterministic": True,
            "cudnn_benchmark": False, "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False, "float32_matmul_precision": "highest",
        },
    }
    eval_batch_size = value.get("eval_batch_size")
    if type(eval_batch_size) is not int or eval_batch_size < 1:
        raise ValueError("Stage-09b eval batch size is absent from run identity")
    expected = {
        "stage": "09b_development_controls",
        "format": "thermoroute.development-controls.v2",
        "execution_role": "prelabel_relative_to_unopened_post_2020_confirmation",
        "evidence_role": "development_only_exploratory",
        "development_disclosure": (
            "2019-2020 outcomes were already inspected during development; this is "
            "exploratory development evidence, not a blind or confirmatory test."
        ),
        "panel_date_range": ["2006-01-01", "2020-12-31"],
        "development_evaluation_interval": ["2019-01-01", "2020-12-31"],
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "training_device": "cpu",
        "variables": full,
        "context_length": 32,
        "horizons": [1, 3, 7],
        "time_split": {
            "train": ["2006-01-01", "2015-12-31"],
            "val": ["2016-01-01", "2017-12-31"],
            "calib": ["2018-01-01", "2018-12-31"],
            "test": ["2019-01-01", "2020-12-31"],
        },
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "train_config": train,
        "arms": arms,
        "expected_member_registry": [list(member) for member in _stage09b_release_members()],
        "parameter_counts": parameter_counts,
        "architecture_templates": templates,
        "parameter_match_tolerance_fraction": 0.02,
        "architecture_candidates_per_arm": 1,
        "historical_tuning_budget_equalized": False,
        "development_predictor_bridge": expected_bridge,
        "formal_numerical_policy": formal_policy,
        "eval_batch_size": eval_batch_size,
    }
    if dict(value) != expected:
        raise ValueError("Stage-09b formal architecture/training configuration changed")
    return expected


def _validate_receipt_self_hash(receipt: Mapping[str, Any], *, label: str) -> None:
    stable = {
        key: value for key, value in receipt.items()
        if key != "receipt_self_sha256"
    }
    if receipt.get("receipt_self_sha256") != _sha256_json(stable):
        raise ValueError(f"{label} self hash changed")


_STAGE09B_PREDICTION_COLUMNS = [
    "model", "scope", "feature_set", "seed", "site_id", "horizon", "split",
    "issue_date", "target_date", "y_true", "y_pred", "q05", "q50", "q95",
    "p_exceed",
]
_STAGE09B_KEY_COLUMNS = [
    "split", "site_id", "horizon", "issue_date", "target_date",
]


def _stage09b_assert_prediction_arrow_schema(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(path).schema_arrow
    if schema.names != _STAGE09B_PREDICTION_COLUMNS:
        raise ValueError("Stage-09b prediction Arrow columns/order changed")
    text_columns = {"model", "scope", "feature_set", "site_id", "split"}
    integer_columns = {"seed", "horizon"}
    date_columns = {"issue_date", "target_date"}
    float_columns = {"y_true", "y_pred", "q05", "q50", "q95", "p_exceed"}
    for field in schema:
        if field.name in text_columns and not pa.types.is_string(field.type):
            raise ValueError(
                f"Stage-09b prediction {field.name} Arrow type changed"
            )
        if field.name in integer_columns and not pa.types.is_integer(field.type):
            raise ValueError(
                f"Stage-09b prediction {field.name} Arrow type changed"
            )
        if field.name in date_columns and not (
            pa.types.is_timestamp(field.type)
            and field.type.unit == "ns"
            and field.type.tz is None
        ):
            raise ValueError(
                f"Stage-09b prediction {field.name} Arrow type changed"
            )
        if field.name in float_columns and not pa.types.is_floating(field.type):
            raise ValueError(
                f"Stage-09b prediction {field.name} Arrow type changed"
            )


def _stage09b_window_registry_digest(frame: Any) -> str:
    """Independent implementation of the canonical window-registry digest."""
    import numpy as np
    import pandas as pd

    columns = ["split", "site_id", "horizon", "issue_date", "target_date", "y_true"]
    ordered = frame.loc[:, columns].copy()
    ordered["split"] = ordered["split"].astype(str)
    ordered["site_id"] = ordered["site_id"].astype(str)
    ordered["horizon"] = pd.to_numeric(ordered["horizon"], errors="raise").astype("int64")
    ordered["issue_date"] = pd.to_datetime(ordered["issue_date"], errors="raise")
    ordered["target_date"] = pd.to_datetime(ordered["target_date"], errors="raise")
    ordered["y_true"] = pd.to_numeric(ordered["y_true"], errors="raise").astype("float64")
    key = ["split", "site_id", "horizon", "issue_date", "target_date"]
    ordered = ordered.sort_values(key, kind="mergesort").reset_index(drop=True)
    if ordered.duplicated(key).any() or not np.isfinite(ordered["y_true"]).all():
        raise ValueError("Stage-09b canonical window registry is invalid")
    digest = hashlib.sha256()
    digest.update(b"thermoroute.window-registry-digest.v1")
    digest.update(struct.pack("<Q", len(ordered)))
    for column in columns:
        encoded = column.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        if column == "horizon":
            digest.update(np.asarray(ordered[column], dtype="<i8").tobytes(order="C"))
        elif column in {"issue_date", "target_date"}:
            values = pd.to_datetime(ordered[column]).to_numpy(
                dtype="datetime64[ns]"
            ).astype("<i8", copy=False)
            digest.update(values.tobytes(order="C"))
        elif column == "y_true":
            values = np.asarray(ordered[column], dtype="<f8").copy()
            values[values == 0.0] = 0.0
            digest.update(values.tobytes(order="C"))
        else:
            for value in ordered[column].astype(str):
                payload = value.encode("utf-8")
                digest.update(struct.pack("<Q", len(payload)))
                digest.update(payload)
    return digest.hexdigest()


def _stage09b_rebuild_canonical_windows(
    panel_path: Path, registry_path: Path, spec_path: Path,
) -> tuple[Any, int, str, str, tuple[str, ...]]:
    """Reconstruct train/evaluation keys directly from the frozen panel.

    This outer-release implementation deliberately imports no archive module.
    It uses only the frozen table, stable-ID map and the declared 32-day,
    1/3/7-day, fully-observed-target split rules.
    """
    import numpy as np
    import pandas as pd

    spec = _load_json(spec_path, label="Stage-09b frozen panel specification")
    panel_spec = spec.get("panel")
    registry_spec = spec.get("station_registry")
    if (
        spec.get("schema_version") != 1
        or spec.get("evidence_role") != "development_exploratory"
        or not isinstance(panel_spec, Mapping)
        or not isinstance(registry_spec, Mapping)
        or panel_spec.get("date_start") != "2006-01-01"
        or panel_spec.get("date_end") != "2020-12-31"
        or panel_spec.get("row_count") != 657_480
        or panel_spec.get("station_count") != 120
        or registry_spec.get("station_count") != 120
        or panel_spec.get("sha256") != sha256_file(panel_path)
        or registry_spec.get("sha256") != sha256_file(registry_path)
        or (spec_path.parent / str(panel_spec.get("path"))).resolve() != panel_path.resolve()
        or (spec_path.parent / str(registry_spec.get("path"))).resolve()
        != registry_path.resolve()
    ):
        raise ValueError("Stage-09b frozen panel specification changed")
    registry = pd.read_csv(
        registry_path,
        usecols=["site_no", "legacy_site_id"],
        dtype={"site_no": "string", "legacy_site_id": "string"},
        keep_default_na=False,
    )
    registry["site_no"] = registry["site_no"].astype("string").str.strip()
    registry["legacy_site_id"] = registry["legacy_site_id"].astype("string").str.strip()
    if (
        len(registry) != 120
        or registry["site_no"].eq("").any()
        or registry["legacy_site_id"].eq("").any()
        or registry["site_no"].duplicated().any()
        or registry["legacy_site_id"].duplicated().any()
        or any(
            not site.isdigit() or not 8 <= len(site) <= 15
            for site in registry["site_no"].astype(str)
        )
    ):
        raise ValueError("Stage-09b stable station registry changed")
    mapping = dict(zip(
        registry["legacy_site_id"].astype(str), registry["site_no"].astype(str),
        strict=True,
    ))
    panel = pd.read_parquet(panel_path, columns=["DATE", "site_id", "WTEMP"])
    panel["DATE"] = pd.to_datetime(panel["DATE"], errors="raise").dt.normalize()
    panel["site_id"] = panel["site_id"].astype(str)
    panel["WTEMP"] = pd.to_numeric(panel["WTEMP"], errors="coerce").astype("float64")
    if (
        len(panel) != 657_480
        or panel.duplicated(["site_id", "DATE"]).any()
        or set(panel["site_id"]) != set(mapping)
        or panel["DATE"].min() != pd.Timestamp("2006-01-01")
        or panel["DATE"].max() != pd.Timestamp("2020-12-31")
    ):
        raise ValueError("Stage-09b frozen panel dimensions/keys changed")
    expected_dates = pd.date_range("2006-01-01", "2020-12-31", freq="D")
    horizons = (1, 3, 7)
    splits = {
        "train": (pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31")),
        "val": (pd.Timestamp("2016-01-01"), pd.Timestamp("2017-12-31")),
        "calib": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-12-31")),
        "test": (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31")),
    }
    eval_frames: list[Any] = []
    train_frames: list[Any] = []
    train_examples = 0
    for legacy, site_no in sorted(mapping.items(), key=lambda item: item[1]):
        station = panel.loc[panel["site_id"].eq(legacy)].sort_values("DATE")
        dates = pd.DatetimeIndex(station["DATE"])
        truth = station["WTEMP"].to_numpy(dtype="float64")
        if len(station) != len(expected_dates) or not dates.equals(expected_dates):
            raise ValueError("Stage-09b panel is not an exact daily station rectangle")
        candidate = np.arange(31, len(station) - 7, dtype=np.int64)
        observed = np.isfinite(truth[candidate])
        for horizon in horizons:
            observed &= np.isfinite(truth[candidate + horizon])
        for split, (lower, upper) in splits.items():
            issue_dates = dates[candidate]
            inside = (
                (issue_dates >= lower)
                & (issue_dates <= upper)
                & (issue_dates + pd.Timedelta(days=7) <= upper)
            )
            selected = candidate[observed & np.asarray(inside)]
            if split == "train":
                train_examples += len(selected)
            target_frames = train_frames if split == "train" else eval_frames
            for horizon in horizons:
                target_frames.append(pd.DataFrame({
                    "split": split,
                    "site_id": site_no,
                    "horizon": horizon,
                    "issue_date": dates[selected].to_numpy(),
                    "target_date": dates[selected + horizon].to_numpy(),
                    "y_true": truth[selected + horizon],
                }))
    evaluation = pd.concat(eval_frames, ignore_index=True).sort_values(
        _STAGE09B_KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    training = pd.concat(train_frames, ignore_index=True).sort_values(
        _STAGE09B_KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    stations = tuple(sorted(registry["site_no"].astype(str)))
    if (
        len(stations) != 120
        or len(training) != train_examples * len(horizons)
        or set(evaluation["split"]) != {"val", "calib", "test"}
        or set(evaluation["site_id"]) != set(stations)
        or set(training["site_id"]) != set(stations)
    ):
        raise ValueError("Stage-09b reconstructed window registry is incomplete")
    return (
        evaluation,
        train_examples,
        _stage09b_window_registry_digest(evaluation),
        _stage09b_window_registry_digest(training),
        stations,
    )


def _stage09b_prediction_content_digest(frame: Any) -> str:
    import numpy as np
    import pandas as pd

    digest = hashlib.sha256()
    digest.update(b"thermoroute.prediction-content-digest.v1")
    digest.update(struct.pack("<Q", len(frame)))
    integer_columns = {"seed", "horizon"}
    date_columns = {"issue_date", "target_date"}
    float_columns = {"y_true", "y_pred", "q05", "q50", "q95", "p_exceed"}
    for column in _STAGE09B_PREDICTION_COLUMNS:
        encoded = column.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        if column in integer_columns:
            digest.update(np.asarray(frame[column], dtype="<i8").tobytes(order="C"))
        elif column in date_columns:
            values = pd.to_datetime(frame[column], errors="raise").to_numpy(
                dtype="datetime64[ns]"
            ).astype("<i8", copy=False)
            digest.update(values.tobytes(order="C"))
        elif column in float_columns:
            values = np.asarray(frame[column], dtype="<f8").copy()
            values[values == 0.0] = 0.0
            digest.update(values.tobytes(order="C"))
        else:
            for value in frame[column].astype(str):
                payload = value.encode("utf-8")
                digest.update(struct.pack("<Q", len(payload)))
                digest.update(payload)
    return digest.hexdigest()


def _normalise_stage09b_release_prediction(
    frame: Any,
    *,
    arm_id: str,
    seed: int,
    feature_set: str,
    reference: Any | None,
) -> Any:
    import numpy as np
    import pandas as pd

    if list(frame.columns) != _STAGE09B_PREDICTION_COLUMNS:
        raise ValueError("Stage-09b prediction columns/order changed")
    output = frame.loc[:, _STAGE09B_PREDICTION_COLUMNS].copy()
    for column in ("model", "scope", "feature_set", "site_id", "split"):
        values = output[column]
        if not (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
        ) or not all(
            isinstance(value, str) and bool(value.strip()) for value in values
        ):
            raise ValueError(
                f"Stage-09b prediction {column} values are not non-empty strings"
            )
    for column in ("seed", "horizon"):
        values = output[column]
        if not all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ) or (
            not pd.api.types.is_integer_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
        ):
            raise ValueError(
                f"Stage-09b prediction {column} values are not true integers"
            )
    for column in ("issue_date", "target_date"):
        values = output[column]
        if str(values.dtype) != "datetime64[ns]":
            raise ValueError(
                f"Stage-09b prediction {column} is not naive datetime64[ns]"
            )
        if values.isna().any() or not values.equals(values.dt.normalize()):
            raise ValueError(
                "Stage-09b prediction dates are not timezone-naive normalized days"
            )
    for column in ("y_true", "y_pred", "q05", "q50", "q95", "p_exceed"):
        values = output[column]
        if not pd.api.types.is_float_dtype(values.dtype) or not all(
            isinstance(value, (float, np.floating)) for value in values
        ):
            raise ValueError(
                f"Stage-09b prediction {column} values are not floating point"
            )
    for column in ("model", "scope", "feature_set", "site_id", "split"):
        if output[column].isna().any():
            raise ValueError(f"Stage-09b prediction {column} contains nulls")
        output[column] = output[column].astype(str)
    if (
        set(output["model"]) != {arm_id}
        or set(output["scope"]) != {"development_only_2006_2020"}
        or set(output["feature_set"]) != {feature_set}
    ):
        raise ValueError("Stage-09b prediction static identity changed")
    numeric_seed = output["seed"].astype("int64")
    horizon = output["horizon"].astype("int64")
    if set(numeric_seed) != {seed}:
        raise ValueError("Stage-09b prediction seed/horizon changed")
    output["seed"] = numeric_seed
    output["horizon"] = horizon
    numerical = ("y_true", "y_pred", "q05", "q50", "q95", "p_exceed")
    for column in numerical:
        output[column] = pd.to_numeric(output[column], errors="raise").astype("float64")
    if not np.isfinite(output.loc[:, numerical].to_numpy(float)).all():
        raise ValueError("Stage-09b prediction numerical values are non-finite")
    if not (
        (output["q05"] <= output["q50"])
        & (output["q50"] <= output["q95"])
    ).all() or not output["p_exceed"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Stage-09b prediction probabilistic semantics changed")
    if set(output["split"]) != {"val", "calib", "test"} or set(
        output["horizon"]
    ) != {1, 3, 7}:
        raise ValueError("Stage-09b prediction split/horizon registry changed")
    sites = set(output["site_id"])
    if any(not site.isdigit() or not 8 <= len(site) <= 15 for site in sites):
        raise ValueError("Stage-09b prediction does not use stable site_no")
    if not (
        output["issue_date"] + pd.to_timedelta(output["horizon"], unit="D")
    ).equals(output["target_date"]):
        raise ValueError("Stage-09b prediction target-date arithmetic changed")
    if output.duplicated(_STAGE09B_KEY_COLUMNS).any():
        raise ValueError("Stage-09b prediction has duplicate forecast keys")
    output = output.sort_values(
        _STAGE09B_KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    if reference is not None:
        if (
            not output[_STAGE09B_KEY_COLUMNS].equals(
                reference[_STAGE09B_KEY_COLUMNS]
            )
            or not np.array_equal(
                output["y_true"].to_numpy(dtype="<f4"),
                reference["y_true"].to_numpy(dtype="<f4"),
            )
        ):
            raise ValueError("Stage-09b member forecast registry/truth changed")
        output["y_true"] = reference["y_true"].to_numpy(dtype="float64")
    return output


def _stage09b_validate_combined_stream(
    combined_path: Path,
    member_paths: Mapping[tuple[str, int], Path],
) -> int:
    """Compare combined/member Parquet values in bounded Arrow batches."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    combined_file = pq.ParquetFile(combined_path)
    if combined_file.schema_arrow.names != _STAGE09B_PREDICTION_COLUMNS:
        raise ValueError("Stage-09b combined prediction schema changed")
    combined_iterator = iter(combined_file.iter_batches(
        columns=_STAGE09B_PREDICTION_COLUMNS, batch_size=65_536,
    ))
    current: Any | None = None
    current_offset = 0
    total = 0

    def take(rows: int) -> Any:
        nonlocal current, current_offset
        pieces: list[Any] = []
        remaining = rows
        while remaining:
            if current is None or current_offset == current.num_rows:
                try:
                    current = next(combined_iterator)
                except StopIteration as exc:
                    raise ValueError("Stage-09b combined predictions end early") from exc
                current_offset = 0
            count = min(remaining, current.num_rows - current_offset)
            pieces.append(current.slice(current_offset, count))
            current_offset += count
            remaining -= count
        return pa.Table.from_batches(pieces).combine_chunks()

    for member in _stage09b_release_members():
        path = member_paths[member]
        member_file = pq.ParquetFile(path)
        if member_file.schema_arrow.names != _STAGE09B_PREDICTION_COLUMNS:
            raise ValueError("Stage-09b member prediction schema changed")
        for batch in member_file.iter_batches(
            columns=_STAGE09B_PREDICTION_COLUMNS, batch_size=65_536,
        ):
            expected = pa.Table.from_batches([batch]).combine_chunks()
            observed = take(batch.num_rows)
            if not observed.equals(expected, check_metadata=False):
                raise ValueError("Stage-09b combined/member prediction columns differ")
            total += batch.num_rows
    if current is not None and current_offset < current.num_rows:
        raise ValueError("Stage-09b combined predictions contain extra rows")
    try:
        next(combined_iterator)
    except StopIteration:
        return total
    raise ValueError("Stage-09b combined predictions contain extra rows")


def _stage09b_recompute_summary(frames: Mapping[tuple[str, int], Any]) -> Any:
    import numpy as np
    import pandas as pd

    station_metrics = _stage09b_recompute_station_rmse(frames)
    rows: list[dict[str, object]] = []
    for (arm_id, seed), frame in frames.items():
        for (split, horizon), group in frame.groupby(["split", "horizon"], sort=True):
            with np.errstate(over="ignore", invalid="ignore"):
                error = (
                    group["y_pred"].to_numpy(dtype="float64")
                    - group["y_true"].to_numpy(dtype="float64")
                )
            micro_rmse, micro_mae = _stage09b_stable_error_metrics(error)
            stations = station_metrics.loc[
                station_metrics["arm_id"].eq(str(arm_id))
                & station_metrics["seed"].eq(int(seed))
                & station_metrics["split"].eq(str(split))
                & station_metrics["horizon"].eq(int(horizon))
            ]
            if stations.empty:
                raise ValueError("Stage-09b member metric lacks station units")
            station_median = float(
                np.median(stations["station_rmse_c"].to_numpy(dtype="float64"))
            )
            if not math.isfinite(station_median):
                raise ValueError("Stage-09b station-median RMSE is non-finite")
            rows.append({
                "arm_id": arm_id,
                "seed": seed,
                "split": str(split),
                "horizon": int(horizon),
                "forecast_keys": len(group),
                "stations": len(stations),
                "median_station_rmse_c": station_median,
                "micro_rmse_c": micro_rmse,
                "micro_mae_c": micro_mae,
            })
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "arm_id", "seed", "split", "horizon", "forecast_keys", "stations",
            "median_station_rmse_c", "micro_rmse_c", "micro_mae_c",
        ),
    ).sort_values(
        ["arm_id", "seed", "split", "horizon"], kind="mergesort"
    ).reset_index(drop=True)


def _stage09b_stable_error_metrics(error: Any) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(error, dtype="float64")
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("Stage-09b metric arithmetic overflowed")
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 0.0, 0.0
    scaled = values / scale
    rmse = scale * float(np.sqrt(np.mean(scaled ** 2)))
    mae = scale * float(np.mean(np.abs(scaled)))
    if not math.isfinite(rmse) or not math.isfinite(mae):
        raise ValueError("Stage-09b prediction-derived metric is non-finite")
    return rmse, mae


def _stage09b_recompute_station_rmse(
    frames: Mapping[tuple[str, int], Any],
) -> Any:
    import numpy as np
    import pandas as pd

    rows: list[dict[str, object]] = []
    for (arm_id, seed), frame in frames.items():
        for (split, horizon, site_id), group in frame.groupby(
            ["split", "horizon", "site_id"], sort=True,
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                error = (
                    group["y_pred"].to_numpy(dtype="float64")
                    - group["y_true"].to_numpy(dtype="float64")
                )
            station_rmse, _station_mae = _stage09b_stable_error_metrics(error)
            rows.append({
                "arm_id": str(arm_id), "seed": int(seed), "split": str(split),
                "horizon": int(horizon), "site_id": str(site_id),
                "forecast_keys": int(len(group)), "station_rmse_c": station_rmse,
            })
    columns = (
        "arm_id", "seed", "split", "horizon", "site_id", "forecast_keys",
        "station_rmse_c",
    )
    output = pd.DataFrame.from_records(rows, columns=columns)
    if output.empty:
        raise ValueError("Stage-09b station RMSE registry is empty")
    return output.sort_values(
        ["arm_id", "seed", "split", "horizon", "site_id"], kind="mergesort"
    ).reset_index(drop=True)


def _stage09b_paired_comparison_registry() -> list[dict[str, object]]:
    full = "ThermoRoute-ladder-07_plus_WDSP"
    ladder = (
        "01_WTEMP", "02_plus_FLOW", "03_plus_TEMP", "04_plus_PRCP",
        "05_plus_RHMEAN", "06_plus_DH", "07_plus_WDSP",
    )
    controls = [
        {
            "comparison_family": "full_vs_control",
            "comparison_id": f"{full}-minus-{reference}",
            "candidate_arm_id": full,
            "reference_arm_id": reference,
            "seeds": [0, 1, 2],
        }
        for reference in ("PlainMLP-7var", "PlainCausalTCN-7var")
    ]
    arms = [f"ThermoRoute-ladder-{rung}" for rung in ladder]
    adjacent = [
        {
            "comparison_family": "adjacent_feature_ladder",
            "comparison_id": f"{candidate}-minus-{reference}",
            "candidate_arm_id": candidate,
            "reference_arm_id": reference,
            "seeds": [0, 1, 2],
        }
        for reference, candidate in zip(arms[:-1], arms[1:], strict=True)
    ]
    return controls + adjacent


def _stage09b_recompute_paired_effects(station_metrics: Any) -> Any:
    import numpy as np
    import pandas as pd

    observed_members = set(zip(
        station_metrics["arm_id"].astype(str),
        station_metrics["seed"].astype(int),
        strict=True,
    ))
    if observed_members != set(_stage09b_release_members()):
        raise ValueError("Stage-09b paired effects lack the exact 31 members")
    if set(station_metrics["split"].astype(str)) != {"val", "calib", "test"}:
        raise ValueError("Stage-09b paired-effect split registry changed")
    if set(station_metrics["horizon"].astype(int)) != {1, 3, 7}:
        raise ValueError("Stage-09b paired-effect horizon registry changed")
    rows: list[dict[str, object]] = []
    pair_keys = ["split", "horizon", "site_id"]
    for comparison in _stage09b_paired_comparison_registry():
        for seed in comparison["seeds"]:
            candidate = station_metrics.loc[
                station_metrics["arm_id"].eq(comparison["candidate_arm_id"])
                & station_metrics["seed"].eq(seed)
            ].sort_values(pair_keys, kind="mergesort").reset_index(drop=True)
            reference = station_metrics.loc[
                station_metrics["arm_id"].eq(comparison["reference_arm_id"])
                & station_metrics["seed"].eq(seed)
            ].sort_values(pair_keys, kind="mergesort").reset_index(drop=True)
            if (
                candidate.empty
                or reference.empty
                or not candidate[pair_keys].equals(reference[pair_keys])
                or not np.array_equal(
                    candidate["forecast_keys"].to_numpy(dtype="int64"),
                    reference["forecast_keys"].to_numpy(dtype="int64"),
                )
            ):
                raise ValueError("Stage-09b paired station registry changed")
            paired = candidate[pair_keys + ["forecast_keys"]].copy()
            paired["effect_c"] = (
                candidate["station_rmse_c"].to_numpy(dtype="float64")
                - reference["station_rmse_c"].to_numpy(dtype="float64")
            )
            if not np.isfinite(paired["effect_c"]).all():
                raise ValueError("Stage-09b paired effect is non-finite")
            for (split, horizon), group in paired.groupby(
                ["split", "horizon"], sort=True,
            ):
                rows.append({
                    "comparison_family": comparison["comparison_family"],
                    "comparison_id": comparison["comparison_id"],
                    "candidate_arm_id": comparison["candidate_arm_id"],
                    "reference_arm_id": comparison["reference_arm_id"],
                    "seed": int(seed), "split": str(split),
                    "horizon": int(horizon),
                    "common_forecast_keys": int(group["forecast_keys"].sum()),
                    "stations": int(len(group)),
                    "median_paired_station_rmse_difference_c": float(
                        np.median(group["effect_c"].to_numpy(dtype="float64"))
                    ),
                })
    columns = (
        "comparison_family", "comparison_id", "candidate_arm_id",
        "reference_arm_id", "seed", "split", "horizon", "common_forecast_keys",
        "stations", "median_paired_station_rmse_difference_c",
    )
    output = pd.DataFrame.from_records(rows, columns=columns)
    expected_identities = [
        (
            comparison["comparison_family"], comparison["comparison_id"],
            comparison["candidate_arm_id"], comparison["reference_arm_id"],
            seed, split, horizon,
        )
        for comparison in _stage09b_paired_comparison_registry()
        for seed in (0, 1, 2)
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
    ]
    observed_identities = list(output[[
        "comparison_family", "comparison_id", "candidate_arm_id",
        "reference_arm_id", "seed", "split", "horizon",
    ]].itertuples(index=False, name=None))
    if observed_identities != expected_identities:
        raise ValueError("Stage-09b paired-effect registry is incomplete")
    return output.reset_index(drop=True)


def _stage09b_scientific_summary(paired_effects: Any) -> dict[str, object]:
    records = [
        {
            "comparison_family": str(row.comparison_family),
            "comparison_id": str(row.comparison_id),
            "candidate_arm_id": str(row.candidate_arm_id),
            "reference_arm_id": str(row.reference_arm_id),
            "seed": int(row.seed), "split": str(row.split),
            "horizon": int(row.horizon),
            "common_forecast_keys": int(row.common_forecast_keys),
            "stations": int(row.stations),
            "median_paired_station_rmse_difference_c": float(
                row.median_paired_station_rmse_difference_c
            ),
        }
        for row in paired_effects.itertuples(index=False)
    ]
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    ladder_variables = (
        ("01_WTEMP", ["WTEMP"]),
        ("02_plus_FLOW", ["WTEMP", "FLOW"]),
        ("03_plus_TEMP", ["WTEMP", "FLOW", "TEMP"]),
        ("04_plus_PRCP", ["WTEMP", "FLOW", "TEMP", "PRCP"]),
        ("05_plus_RHMEAN", ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN"]),
        ("06_plus_DH", ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH"]),
        (
            "07_plus_WDSP",
            ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"],
        ),
    )
    return {
        "format": "thermoroute.development-controls-scientific-summary.v1",
        "metric_summary_format": (
            "thermoroute.development-controls-metric-summary.v2"
        ),
        "primary_member_estimand": {
            "name": "median_across_stations_of_within_station_rmse_c",
            "column": "median_station_rmse_c", "unit": "degree_Celsius",
            "aggregation": "median_of_within_station_RMSE",
            "station_weighting": "one_station_one_value",
        },
        "secondary_member_estimands": {
            "micro_rmse_c": {
                "role": "secondary_not_primary_estimand",
                "aggregation": "RMSE_over_all_forecast_keys",
            },
            "micro_mae_c": {
                "role": "secondary_not_primary_estimand",
                "aggregation": "MAE_over_all_forecast_keys",
            },
        },
        "paired_descriptive_effects": {
            "estimand": (
                "median_across_stations_of_candidate_rmse_minus_reference_rmse_c"
            ),
            "effect_convention": "candidate_minus_reference",
            "negative_favours": "candidate", "same_seed": True,
            "exact_common_forecast_keys_verified": True,
            "comparison_registry": _stage09b_paired_comparison_registry(),
            "feature_ladder_order": [
                {"rung": rung, "variables": variables}
                for rung, variables in ladder_variables
            ],
            "feature_ladder_fixed_order_path_dependent": True,
            "independent_feature_contribution_claimed": False,
            "causal_effect_claimed": False,
            "records_sha256": hashlib.sha256(payload).hexdigest(),
            "records": records,
        },
    }


def _stage09b_validate_scientific_summary_document(value: object) -> None:
    import pandas as pd

    if not isinstance(value, Mapping):
        raise ValueError("Stage-09b scientific summary is malformed")
    paired = value.get("paired_descriptive_effects")
    records = paired.get("records") if isinstance(paired, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("Stage-09b paired effects are malformed")
    columns = (
        "comparison_family", "comparison_id", "candidate_arm_id",
        "reference_arm_id", "seed", "split", "horizon", "common_forecast_keys",
        "stations", "median_paired_station_rmse_difference_c",
    )
    expected_identities = [
        (
            comparison["comparison_family"], comparison["comparison_id"],
            comparison["candidate_arm_id"], comparison["reference_arm_id"],
            seed, split, horizon,
        )
        for comparison in _stage09b_paired_comparison_registry()
        for seed in (0, 1, 2)
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
    ]
    observed: list[tuple[object, ...]] = []
    for record in records:
        effect = (
            record.get("median_paired_station_rmse_difference_c")
            if isinstance(record, Mapping) else None
        )
        if (
            not isinstance(record, Mapping)
            or set(record) != set(columns)
            or type(record.get("seed")) is not int
            or type(record.get("horizon")) is not int
            or type(record.get("common_forecast_keys")) is not int
            or int(record["common_forecast_keys"]) < 1
            or type(record.get("stations")) is not int
            or int(record["stations"]) < 1
            or type(effect) not in {int, float}
            or not math.isfinite(float(effect))
        ):
            raise ValueError("Stage-09b paired-effect record is malformed")
        observed.append((
            record["comparison_family"], record["comparison_id"],
            record["candidate_arm_id"], record["reference_arm_id"],
            record["seed"], record["split"], record["horizon"],
        ))
    if observed != expected_identities:
        raise ValueError("Stage-09b paired-effect registry changed")
    frame = pd.DataFrame.from_records(records, columns=columns)
    if dict(value) != _stage09b_scientific_summary(frame):
        raise ValueError("Stage-09b scientific summary contract changed")


def _stage09b_markdown_table(frame: Any) -> str:
    import numpy as np

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not math.isfinite(float(value)) else f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _stage09b_expected_report(
    *, run_id: str, audit: Mapping[str, Any], budget: Any, summary: Any,
    paired_effects: Any,
) -> bytes:
    development = summary.loc[summary["split"].eq("test")]
    aggregate = (
        development.groupby(["arm_id", "horizon"], as_index=False)
        .agg(
            primary_median_station_rmse_seed_mean_c=(
                "median_station_rmse_c", "mean",
            ),
            primary_median_station_rmse_seed_sd_c=(
                "median_station_rmse_c", "std",
            ),
            secondary_micro_rmse_seed_mean_c=("micro_rmse_c", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(
            ["horizon", "primary_median_station_rmse_seed_mean_c", "arm_id"],
            kind="mergesort",
        )
    )
    result_table = _stage09b_markdown_table(aggregate)
    development_pairs = paired_effects.loc[paired_effects["split"].eq("test")]
    pair_columns = [
        "candidate_arm_id", "reference_arm_id", "seed", "horizon", "stations",
        "median_paired_station_rmse_difference_c",
    ]
    control_effects = _stage09b_markdown_table(development_pairs.loc[
        development_pairs["comparison_family"].eq("full_vs_control"), pair_columns,
    ])
    ladder_effects = _stage09b_markdown_table(development_pairs.loc[
        development_pairs["comparison_family"].eq("adjacent_feature_ladder"),
        pair_columns,
    ])
    budget_table = _stage09b_markdown_table(budget[[
        "arm_id", "variables", "seed_count", "trainable_parameters",
        "parameter_ratio_to_full_thermoroute", "maximum_optimizer_steps_per_seed",
    ]])
    splits = audit["splits"]
    return f"""# Development-only neural controls and feature ladder

Run ID: `{run_id}`

Status: **COMPLETE BEST-MODEL-STATE PREDICTION REPLAY**. Every stored prediction
member is reproduced from the safely loaded checkpoint `best_model_state` and
derived artifacts are regenerated. This is not optimiser-step/trajectory replay
and is not part of the sealed confirmatory model suite.

> 2019-2020 outcomes were already inspected during development; this is exploratory development evidence, not a blind or confirmatory test.

## Design

All models use the frozen 120-site 2006--2020 panel, 32 days of history,
horizons 1/3/7 days, CPU-only deterministic execution, equal-station fixed-size
bootstrap sampling, AdamW, the same declared maximum optimisation budget, and
early-stopping rule. PlainMLP and PlainCausalTCN receive the seven declared
history variables and masks. ThermoRoute additionally receives its declared
train-fit/calendar-derived physical-anchor inputs. The feature ladder adds one
declared variable at a time in the fixed order WTEMP, FLOW, TEMP, PRCP, RHMEAN,
DH, WDSP.

The two pure-neural controls are parameter-matched within 2% of the full
ThermoRoute architecture. Each architecture has one fixed candidate here.
This does not equalise ThermoRoute's historical tuning advantage, so
`historical_tuning_budget_equalized` remains false.

Exact member count: {audit['expected_members']}. Common forecast keys per member:
{audit['common_forecast_keys']}. Total prediction rows: {audit['prediction_rows']}.
Validated splits: {', '.join(splits)}.

## Architecture and declared maximum optimisation budget

{budget_table}

## 2019--2020 development-evaluation results

These values are deterministically derived from the machine-readable summary,
which is itself recomputed from every stored prediction row. `test` means the
already-inspected 2019--2020 development partition, never a blind test.

The primary member-level estimand is median station RMSE: RMSE is first computed
within each station on the exact common forecast keys and then the station RMSEs
are aggregated by their median. Micro RMSE is retained only as a secondary,
non-primary estimand; it weights stations according to their available row count.

{result_table}

## Same-seed station-paired descriptive effects

Every effect below is the median across stations of candidate RMSE minus
reference RMSE on the same seed and exact common forecast keys. Negative values
favour the candidate. These are descriptive development effects, not hypothesis
tests.

### Full ThermoRoute versus matched neural controls

{control_effects}

### Adjacent cumulative feature-ladder rungs

{ladder_effects}

The ladder order is fixed as WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP. Each
adjacent contrast is path-dependent on every preceding rung. It is not an
independent feature contribution, feature importance score, or causal effect.

## Interpretation boundary

These artifacts describe architecture comparisons and fixed-path adjacent
ladder contrasts on historical development data. They verify best-state
prediction replay, not the
full training trajectory, and cannot establish prospective, operational, causal,
safety, or confirmatory performance. They do not modify the frozen Route-A suite
pointer.
""".encode("utf-8")


def _stage09b_expected_final_extra(
    audit: Mapping[str, Any], *, role: str,
) -> dict[str, Any]:
    """Reconstruct the exact stable metadata written for a Stage-09b output."""
    return {
        "format": "thermoroute.development-controls.v2",
        "artifact_role": role,
        "expected_members": audit["expected_members"],
        "prediction_rows": audit["prediction_rows"],
        "common_forecast_keys_per_member": audit["common_forecast_keys"],
        "splits": audit["splits"],
        "reference_member": audit["reference_member"],
        "development_only": True,
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
    }


def _validate_preopening_completion_gates(
    root: Path,
    categories: dict[str, set[Path]],
    suite: Mapping[str, Any],
    development: Mapping[str, Any],
    suite_runtime: str,
) -> None:
    """Independently verify both pre-opening admission receipts.

    This verifier deliberately does not import or execute archive Python.  It
    checks the receipt schemas, self hashes, byte bindings, 31-member registry,
    sidecar/run alignment and architecture-budget registry with the standard
    library before any trusted replay is considered.
    """
    gates = suite.get("preopening_gates")
    required = {"stage09_completion", "stage09b_development_controls"}
    if not isinstance(gates, Mapping) or set(gates) != required:
        raise ValueError("authorized model suite lacks Stage-9/09b completion gates")

    def add(binding: object, *, label: str) -> Path:
        return _add_binding(
            root, categories, "model_suite", binding, label=label
        )

    stage9_path = add(gates["stage09_completion"], label="Stage-9 completion gate")
    stage9 = _load_json(stage9_path, label="Stage-9 completion receipt")
    stage9_keys = {
        "format", "status", "stage", "run_id", "run_identity",
        "formal_configuration", "confirmation_outcomes_requested_or_read",
        "artifacts", "receipt_self_sha256",
    }
    stage9_artifacts = {
        "run_manifest", "predictions", "prediction_sidecar", "scores", "report",
        "lightgbm_selection", "thermoroute_pointer", "lightgbm_pointer",
        "components_pointer",
    }
    stage9_identity = stage9.get("run_identity")
    if (
        set(stage9) != stage9_keys
        or stage9.get("format") != "thermoroute.stage09-completion-receipt.v1"
        or stage9.get("status") != "PASS_FORMAL_STAGE09_COMPLETE"
        or stage9.get("stage") != "09_usgs_experiment"
        or stage9.get("confirmation_outcomes_requested_or_read") is not False
        or not isinstance(stage9_identity, Mapping)
        or stage9_identity.get("panel_sha256") != development.get("panel", {}).get("sha256")
        or stage9_identity.get("registry_sha256")
        != development.get("registry", {}).get("sha256")
        or stage9_identity.get("source_sha256") != development.get("source_sha256")
        or stage9_identity.get("runtime_sha256") != suite_runtime
    ):
        raise ValueError("authorized Stage-9 completion receipt is stale or malformed")
    _validate_receipt_self_hash(stage9, label="Stage-9 receipt")
    artifacts = stage9.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != stage9_artifacts:
        raise ValueError("authorized Stage-9 completion artifact registry is incomplete")
    for label, binding in artifacts.items():
        add(binding, label=f"Stage-9 receipt {label}")

    controls_path = add(
        gates["stage09b_development_controls"],
        label="Stage-09b development-controls completion gate",
    )
    controls = _load_json(
        controls_path, label="Stage-09b development-controls completion receipt"
    )
    control_keys = {
        "format", "status", "stage", "run_id", "run_identity",
        "formal_configuration", "evidence_scope", "training_replay_verified",
        "best_model_state_prediction_replay_verified",
        "matrix_audit", "member_registry", "artifacts",
        "post_2020_outcomes_requested_or_read", "receipt_self_sha256",
    }
    control_artifacts = {
        "run_manifest", "frozen_panel_spec", "panel", "registry",
        "predictor_bridge", "predictions", "prediction_sidecar",
        "architecture_budget", "architecture_budget_sidecar", "metric_summary",
        "metric_summary_sidecar", "report", "report_sidecar", "semantic_audit",
        "semantic_audit_sidecar",
    }
    identity = controls.get("run_identity")
    config = controls.get("formal_configuration")
    if (
        set(controls) != control_keys
        or controls.get("format") != "thermoroute.stage09b-completion-receipt.v3"
        or controls.get("status")
        != "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY"
        or controls.get("stage") != "09b_development_controls"
        or controls.get("post_2020_outcomes_requested_or_read") is not False
        or controls.get("evidence_scope") != "best_model_state_prediction_replay"
        or controls.get("training_replay_verified") is not False
        or controls.get("best_model_state_prediction_replay_verified") is not True
        or not isinstance(identity, Mapping)
        or not isinstance(config, Mapping)
        or controls.get("run_id") != identity.get("run_id")
        or identity.get("panel_sha256") != development.get("panel", {}).get("sha256")
        or identity.get("registry_sha256")
        != development.get("registry", {}).get("sha256")
        or identity.get("source_sha256") != development.get("source_sha256")
        or identity.get("runtime_sha256") != suite_runtime
    ):
        raise ValueError("authorized Stage-09b completion receipt is stale or malformed")
    config = _stage09b_formal_configuration(
        config, expected_bridge=development.get("predictor_bridge")
    )
    identity_fields = {
        "run_id", "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "schema_version",
    }
    if (
        set(identity) != identity_fields
        or identity.get("schema_version") != "thermoroute.run.v1"
        or identity.get("config_sha256") != _sha256_json(config)
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(identity.get(field, "")))
            for field in identity_fields - {"run_id", "schema_version"}
        )
        or identity.get("run_id") != _sha256_json({
            "schema_version": identity["schema_version"],
            "panel_sha256": identity["panel_sha256"],
            "registry_sha256": identity["registry_sha256"],
            "config_sha256": identity["config_sha256"],
            "source_sha256": identity["source_sha256"],
            "runtime_sha256": identity["runtime_sha256"],
        })[:20]
    ):
        raise ValueError("authorized Stage-09b run identity is not content addressed")
    _validate_receipt_self_hash(controls, label="Stage-09b receipt")
    artifacts = controls.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != control_artifacts:
        raise ValueError("authorized Stage-09b completion artifact registry is incomplete")
    if (
        artifacts.get("frozen_panel_spec") != development.get("frozen_panel_spec")
        or artifacts.get("panel") != development.get("panel")
        or artifacts.get("registry") != development.get("registry")
        or artifacts.get("predictor_bridge") != development.get("predictor_bridge")
    ):
        raise ValueError("authorized Stage-09b receipt binds another development contract")
    resolved_artifacts = {
        label: add(binding, label=f"Stage-09b receipt {label}")
        for label, binding in artifacts.items()
    }
    run_dir = (
        root / "outputs" / "runs" / "09b_development_controls"
        / str(identity["run_id"])
    ).resolve()
    for candidate in run_dir.rglob("*"):
        if (
            (candidate.name.startswith(".") and candidate.name.endswith(".tmp"))
            or candidate.name.endswith(".recovery-probe")
        ):
            raise ValueError("Stage-09b run retains an unbound transaction temp")
    canonical_final_paths = {
        "run_manifest": run_dir / "run.json",
        "frozen_panel_spec": (root / "data_usgs/frozen_panel_v1.json").resolve(),
        "panel": (root / "data_usgs/panel_usgs_120v2.parquet").resolve(),
        "registry": (root / "data_usgs/station_registry_v1.csv").resolve(),
        "predictor_bridge": (
            root / "data_usgs/development_predictor_bridge_v1.json"
        ).resolve(),
        "predictions": run_dir / "development_controls_predictions.parquet",
        "architecture_budget": run_dir / "development_controls_architecture_budget.csv",
        "metric_summary": run_dir / "development_controls_metric_summary.csv",
        "report": run_dir / "development_controls_report.md",
        "semantic_audit": run_dir / "development_controls_semantic_audit.json",
    }
    if any(
        resolved_artifacts[label].resolve() != expected
        for label, expected in canonical_final_paths.items()
    ):
        raise ValueError("Stage-09b final artifact path is noncanonical")
    for artifact, sidecar in (
        ("predictions", "prediction_sidecar"),
        ("architecture_budget", "architecture_budget_sidecar"),
        ("metric_summary", "metric_summary_sidecar"),
        ("report", "report_sidecar"),
        ("semantic_audit", "semantic_audit_sidecar"),
    ):
        if resolved_artifacts[sidecar] != resolved_artifacts[artifact].with_name(
            resolved_artifacts[artifact].name + ".meta.json"
        ):
            raise ValueError("Stage-09b final artifact/sidecar alignment changed")
    final_sidecar_specs = (
        (
            "predictions", "prediction_sidecar",
            "development_controls_combined_predictions",
            "thermoroute.predictions.v1", "combined_predictions",
        ),
        (
            "architecture_budget", "architecture_budget_sidecar",
            "development_controls_budget",
            "thermoroute.development-controls-architecture-budget.v1",
            "architecture_budget",
        ),
        (
            "metric_summary", "metric_summary_sidecar",
            "development_controls_metric_summary",
            "thermoroute.development-controls-metric-summary.v2",
            "metric_summary",
        ),
        (
            "report", "report_sidecar", "development_controls_report",
            "thermoroute.development-controls-report.v2", "report",
        ),
        (
            "semantic_audit", "semantic_audit_sidecar",
            "development_controls_semantic_audit",
            "thermoroute.development-controls-semantic-audit.v3",
            "semantic_audit",
        ),
    )
    for artifact, sidecar, kind, content_schema, _role in final_sidecar_specs:
        metadata = _load_json(
            resolved_artifacts[sidecar], label=f"Stage-09b {artifact} sidecar"
        )
        extra = metadata.get("extra")
        try:
            created = datetime.fromisoformat(str(metadata.get("created_utc")))
        except ValueError:
            created = None
        if (
            set(metadata) != {
                "schema_version", "kind", "artifact", "artifact_sha256",
                "artifact_bytes", "content_schema", "run", "parents", "extra",
                "created_utc",
            }
            or metadata.get("schema_version") != "thermoroute.artifact.v1"
            or metadata.get("kind") != kind
            or metadata.get("artifact") != resolved_artifacts[artifact].name
            or metadata.get("artifact_sha256") != artifacts[artifact].get("sha256")
            or metadata.get("artifact_bytes") != resolved_artifacts[artifact].stat().st_size
            or metadata.get("content_schema") != content_schema
            or metadata.get("run") != identity
            or created is None
            or created.tzinfo is None
            or created.utcoffset() is None
            or not isinstance(metadata.get("parents"), Mapping)
            or not isinstance(extra, Mapping)
            or extra.get("expected_members") != 31
            or extra.get("development_only") is not True
            or extra.get("blind_or_confirmatory") is not False
            or extra.get("evidence_scope") != "best_model_state_prediction_replay"
            or extra.get("training_replay_verified") is not False
            or extra.get("best_model_state_prediction_replay_verified") is not True
        ):
            raise ValueError("authorized Stage-09b final sidecar changed")
    run_manifest = _load_json(
        resolved_artifacts["run_manifest"], label="Stage-09b run manifest"
    )
    if (
        run_manifest.get("identity") != identity
        or run_manifest.get("resolved_config") != config
        or resolved_artifacts["run_manifest"].resolve()
        != (
            root / "outputs" / "runs" / "09b_development_controls"
            / str(identity["run_id"]) / "run.json"
        ).resolve()
        or set(run_manifest) != {
            "schema_version", "identity", "resolved_config", "created_utc",
            "environment", "git", "provenance",
        }
        or run_manifest.get("schema_version") != "thermoroute.run.v1"
    ):
        raise ValueError("Stage-09b run manifest differs from its receipt")

    (
        canonical_evaluation,
        canonical_train_examples,
        canonical_evaluation_sha256,
        canonical_train_sha256,
        canonical_sites,
    ) = _stage09b_rebuild_canonical_windows(
        resolved_artifacts["panel"],
        resolved_artifacts["registry"],
        resolved_artifacts["frozen_panel_spec"],
    )

    audit = controls.get("matrix_audit")
    members = controls.get("member_registry")
    expected_members = _stage09b_release_members()
    if (
        not isinstance(audit, Mapping)
        or set(audit) != {
            "expected_members", "prediction_rows", "common_forecast_keys",
            "splits", "reference_member",
        }
        or audit.get("expected_members") != 31
        or audit.get("common_forecast_keys") != len(canonical_evaluation)
        or audit.get("prediction_rows") != 31 * audit["common_forecast_keys"]
        or audit.get("splits") != ["calib", "test", "val"]
        or audit.get("reference_member") != "PlainMLP-7var/seed0"
        or not isinstance(members, list)
        or len(members) != 31
    ):
        raise ValueError("authorized Stage-09b matrix audit is incomplete")
    observed: list[tuple[str, int]] = []
    member_prediction_paths: dict[tuple[str, int], Path] = {}
    member_checkpoint_paths: dict[tuple[str, int], Path] = {}
    member_checkpoint_sidecars: dict[tuple[str, int], Path] = {}
    arm_documents = {
        str(arm["arm_id"]): dict(arm) for arm in config["arms"]
    }
    for entry in members:
        if not isinstance(entry, Mapping) or set(entry) != {
            "arm_id", "seed", "checkpoint", "checkpoint_sidecar",
            "prediction", "prediction_sidecar",
        }:
            raise ValueError("authorized Stage-09b member binding is malformed")
        arm_id, seed = entry.get("arm_id"), entry.get("seed")
        if not isinstance(arm_id, str) or type(seed) is not int:
            raise ValueError("authorized Stage-09b member identity is malformed")
        observed.append((arm_id, seed))
        prediction = add(
            entry["prediction"], label=f"Stage-09b {arm_id}/seed{seed} prediction"
        )
        member_prediction_paths[(arm_id, seed)] = prediction
        member_sidecar = add(
            entry["prediction_sidecar"],
            label=f"Stage-09b {arm_id}/seed{seed} sidecar",
        )
        if member_sidecar != prediction.with_name(prediction.name + ".meta.json"):
            raise ValueError("authorized Stage-09b member sidecar path changed")
        checkpoint = add(
            entry["checkpoint"], label=f"Stage-09b {arm_id}/seed{seed} checkpoint"
        )
        checkpoint_sidecar = add(
            entry["checkpoint_sidecar"],
            label=f"Stage-09b {arm_id}/seed{seed} checkpoint sidecar",
        )
        expected_member_root = run_dir / "arm_predictions" / arm_id
        expected_checkpoint_root = run_dir / "checkpoints" / arm_id
        if (
            prediction.resolve() != (expected_member_root / f"seed{seed}.parquet").resolve()
            or checkpoint.resolve() != (expected_checkpoint_root / f"seed{seed}.pt").resolve()
            or checkpoint_sidecar.resolve()
            != checkpoint.with_name(checkpoint.name + ".meta.json").resolve()
        ):
            raise ValueError("authorized Stage-09b member/checkpoint path changed")
        member_checkpoint_paths[(arm_id, seed)] = checkpoint
        member_checkpoint_sidecars[(arm_id, seed)] = checkpoint_sidecar
        checkpoint_metadata = _load_json(
            checkpoint_sidecar, label="Stage-09b checkpoint sidecar"
        )
        arm_config = {
            **config,
            "arm": arm_documents[arm_id],
            "seed": seed,
            "trainable_parameters": config["parameter_counts"][arm_id],
        }
        expected_model_class = {
            "PlainMLP": "thermoroute.neural_baselines.PlainMLPForecaster",
            "PlainCausalTCN": "thermoroute.neural_baselines.PlainCausalTCNForecaster",
            "ThermoRoute": "thermoroute.thermoroute.ThermoRoute",
        }[str(arm_documents[arm_id]["family"])]
        if (
            set(checkpoint_metadata) != {
                "format", "checkpoint_format", "run_id", "epoch",
                "checkpoint_bytes", "checkpoint_sha256",
                "resolved_config_sha256", "extra_sha256", "model_class",
                "optimizer_class", "scheduler_class", "scheduler_present",
            }
            or checkpoint_metadata.get("format")
            != "thermoroute.training-checkpoint-metadata.v2"
            or checkpoint_metadata.get("checkpoint_format")
            != "thermoroute.training-checkpoint.v3"
            or checkpoint_metadata.get("run_id") != identity["run_id"]
            or type(checkpoint_metadata.get("epoch")) is not int
            or checkpoint_metadata["epoch"] < 0
            or checkpoint_metadata.get("checkpoint_bytes") != checkpoint.stat().st_size
            or checkpoint_metadata.get("checkpoint_sha256") != sha256_file(checkpoint)
            or checkpoint_metadata.get("checkpoint_sha256")
            != entry["checkpoint"].get("sha256")
            or checkpoint_metadata.get("resolved_config_sha256")
            != _sha256_json(arm_config)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(checkpoint_metadata.get("extra_sha256", ""))
            )
            or checkpoint_metadata.get("model_class") != expected_model_class
            or checkpoint_metadata.get("optimizer_class") != "torch.optim.adamw.AdamW"
            or checkpoint_metadata.get("scheduler_class")
            != "torch.optim.lr_scheduler.ReduceLROnPlateau"
            or checkpoint_metadata.get("scheduler_present") is not True
        ):
            raise ValueError("authorized Stage-09b checkpoint metadata changed")
        metadata = _load_json(member_sidecar, label="Stage-09b member sidecar")
        extra = metadata.get("extra")
        expected_parents = {
            "frozen_panel": identity["panel_sha256"],
            "frozen_station_registry": identity["registry_sha256"],
            "development_predictor_bridge": artifacts["predictor_bridge"]["sha256"],
            "training_checkpoint": entry["checkpoint"]["sha256"],
            "training_checkpoint_sidecar": entry["checkpoint_sidecar"]["sha256"],
        }
        training_summary = extra.get("training_summary") if isinstance(extra, Mapping) else None
        if (
            metadata.get("kind") != "development_control_arm_predictions"
            or metadata.get("artifact_sha256") != entry["prediction"].get("sha256")
            or metadata.get("run") != identity
            or not isinstance(extra, Mapping)
            or extra.get("arm_id") != arm_id
            or extra.get("seed") != seed
            or extra.get("training_device") != "cpu"
            or extra.get("development_only") is not True
            or extra.get("blind_or_confirmatory") is not False
            or extra.get("eval_batch_size") != config["eval_batch_size"]
            or metadata.get("parents") != dict(sorted(expected_parents.items()))
            or not isinstance(training_summary, Mapping)
            or set(training_summary) != {
                "best_validation_metric", "selected_epoch", "checkpoint_final_epoch",
            }
            or type(training_summary.get("best_validation_metric")) not in {int, float}
            or not math.isfinite(float(training_summary["best_validation_metric"]))
            or float(training_summary["best_validation_metric"]) < 0.0
            or type(training_summary.get("selected_epoch")) is not int
            or training_summary["selected_epoch"] < 0
            or training_summary.get("checkpoint_final_epoch")
            != checkpoint_metadata["epoch"]
            or training_summary["selected_epoch"] > checkpoint_metadata["epoch"]
        ):
            raise ValueError("authorized Stage-09b member sidecar changed")
    if tuple(observed) != expected_members:
        raise ValueError("authorized Stage-09b receipt does not bind exactly 31 members")

    expected_final_parents = {
        "frozen_panel": identity["panel_sha256"],
        "frozen_station_registry": identity["registry_sha256"],
        "development_predictor_bridge": artifacts["predictor_bridge"]["sha256"],
        **{
            f"arm::{entry['arm_id']}::seed{entry['seed']}::prediction": (
                entry["prediction"]["sha256"]
            )
            for entry in members
        },
        **{
            f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint": (
                entry["checkpoint"]["sha256"]
            )
            for entry in members
        },
        **{
            f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint_sidecar": (
                entry["checkpoint_sidecar"]["sha256"]
            )
            for entry in members
        },
    }
    for artifact, sidecar, _kind, _content_schema, role in final_sidecar_specs:
        metadata = _load_json(
            resolved_artifacts[sidecar], label=f"Stage-09b {artifact} sidecar"
        )
        if (
            metadata.get("parents") != dict(sorted(expected_final_parents.items()))
            or metadata.get("extra")
            != _stage09b_expected_final_extra(audit, role=role)
        ):
            raise ValueError("authorized Stage-09b final parent closure changed")

    try:
        import pandas as pd

        features = {
            "PlainMLP-7var": "all_7_variables",
            "PlainCausalTCN-7var": "all_7_variables",
            **{
                f"ThermoRoute-ladder-{rung}": f"feature_ladder_{rung}"
                for rung in (
                    "01_WTEMP", "02_plus_FLOW", "03_plus_TEMP", "04_plus_PRCP",
                    "05_plus_RHMEAN", "06_plus_DH", "07_plus_WDSP",
                )
            },
        }
        member_digests: dict[tuple[str, int], str] = {}
        summary_frames: list[Any] = []
        station_frames: list[Any] = []
        for member in expected_members:
            _stage09b_assert_prediction_arrow_schema(
                member_prediction_paths[member]
            )
            frame = pd.read_parquet(member_prediction_paths[member])
            normalised = _normalise_stage09b_release_prediction(
                frame,
                arm_id=member[0],
                seed=member[1],
                feature_set=features[member[0]],
                reference=canonical_evaluation,
            )
            member_digests[member] = _stage09b_prediction_content_digest(normalised)
            summary_frames.append(_stage09b_recompute_summary({member: normalised}))
            station_frames.append(
                _stage09b_recompute_station_rmse({member: normalised})
            )
            del frame, normalised
        recomputed_summary = pd.concat(summary_frames, ignore_index=True).sort_values(
            ["arm_id", "seed", "split", "horizon"], kind="mergesort"
        ).reset_index(drop=True)
        paired_effects = _stage09b_recompute_paired_effects(
            pd.concat(station_frames, ignore_index=True)
        )
        expected_summary_bytes = recomputed_summary.to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ).encode("utf-8")
        if resolved_artifacts["metric_summary"].read_bytes() != expected_summary_bytes:
            raise ValueError("Stage-09b metric summary is not prediction-derived")

        combined_rows = _stage09b_validate_combined_stream(
            resolved_artifacts["predictions"], member_prediction_paths,
        )
        if combined_rows != audit["prediction_rows"]:
            raise ValueError("Stage-09b combined prediction row count changed")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("authorized Stage-09b prediction semantics are unreadable") from exc

    expected_parameters = {
        "PlainMLP-7var": 38_545,
        "PlainCausalTCN-7var": 38_031,
        "ThermoRoute-ladder-01_WTEMP": 37_775,
        "ThermoRoute-ladder-02_plus_FLOW": 37_896,
        "ThermoRoute-ladder-03_plus_TEMP": 38_018,
        "ThermoRoute-ladder-04_plus_PRCP": 38_139,
        "ThermoRoute-ladder-05_plus_RHMEAN": 38_261,
        "ThermoRoute-ladder-06_plus_DH": 38_383,
        "ThermoRoute-ladder-07_plus_WDSP": 38_505,
    }
    budget_columns = (
        "arm_id", "family", "feature_set", "variables", "variable_count",
        "seed_count", "seeds", "trainable_parameters",
        "thermoroute_full_reference_parameters",
        "parameter_difference_from_full_thermoroute",
        "parameter_ratio_to_full_thermoroute", "matched_within_2pct_of_full_thermoroute",
        "context_length", "horizons", "optimizer", "learning_rate", "weight_decay",
        "batch_size", "max_epochs", "early_stopping_patience", "selection_metric",
        "station_sampling", "train_examples_per_epoch",
        "maximum_optimizer_steps_per_seed", "architecture_candidates_in_this_entrypoint",
        "architecture_configuration", "mlp_hidden_dim", "mlp_depth", "tcn_channels",
        "tcn_blocks", "tcn_kernel_size", "thermoroute_d_model",
        "historical_tuning_budget_equalized", "training_device", "evidence_role",
    )
    try:
        with resolved_artifacts["architecture_budget"].open(
            newline="", encoding="utf-8"
        ) as budget_handle:
            rows = list(csv.DictReader(budget_handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("authorized Stage-09b architecture budget is unreadable") from exc
    train_example_values = {
        int(row.get("train_examples_per_epoch", "-1")) for row in rows
    }
    expected_arm_rows = {
        str(arm["arm_id"]): arm for arm in config["arms"]
    }
    if (
        not rows
        or tuple(rows[0]) != budget_columns
        or [row.get("arm_id") for row in rows] != list(expected_parameters)
        or train_example_values != {canonical_train_examples}
        or any(
            row.get("family") != expected_arm_rows[row["arm_id"]]["family"]
            or row.get("feature_set") != expected_arm_rows[row["arm_id"]]["feature_set"]
            or row.get("variables")
            != "+".join(expected_arm_rows[row["arm_id"]]["variables"])
            or int(row.get("variable_count", "-1"))
            != len(expected_arm_rows[row["arm_id"]]["variables"])
            or int(row.get("seed_count", "-1"))
            != len(expected_arm_rows[row["arm_id"]]["seeds"])
            or row.get("seeds")
            != ",".join(str(seed) for seed in expected_arm_rows[row["arm_id"]]["seeds"])
            or int(row.get("trainable_parameters", "-1"))
            != expected_parameters[row["arm_id"]]
            or int(row.get("thermoroute_full_reference_parameters", "-1")) != 38_505
            or int(row.get("parameter_difference_from_full_thermoroute", "-999999"))
            != expected_parameters[row["arm_id"]] - 38_505
            or not math.isclose(
                float(row.get("parameter_ratio_to_full_thermoroute", "nan")),
                expected_parameters[row["arm_id"]] / 38_505,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or row.get("matched_within_2pct_of_full_thermoroute") != "True"
            or int(row.get("context_length", "-1")) != 32
            or row.get("horizons") != "1,3,7"
            or row.get("optimizer") != "torch.optim.AdamW"
            or float(row.get("learning_rate", "nan")) != 0.002
            or float(row.get("weight_decay", "nan")) != 0.0001
            or int(row.get("batch_size", "-1")) != 1536
            or int(row.get("max_epochs", "-1")) != 80
            or int(row.get("early_stopping_patience", "-1")) != 12
            or row.get("selection_metric") != "station_macro_rmse"
            or row.get("station_sampling") != "equal_station_fixed_size_bootstrap"
            or int(row.get("maximum_optimizer_steps_per_seed", "-1"))
            != math.ceil(int(row["train_examples_per_epoch"]) / 1536) * 80
            or int(row.get("architecture_candidates_in_this_entrypoint", "-1")) != 1
            or json.loads(row.get("architecture_configuration", "null"))
            != config["architecture_templates"][row["arm_id"]]
            or row.get("mlp_hidden_dim")
            != ("70" if row["arm_id"] == "PlainMLP-7var" else "")
            or row.get("mlp_depth")
            != ("2" if row["arm_id"] == "PlainMLP-7var" else "")
            or row.get("tcn_channels")
            != ("54" if row["arm_id"] == "PlainCausalTCN-7var" else "")
            or row.get("tcn_blocks")
            != ("4" if row["arm_id"] == "PlainCausalTCN-7var" else "")
            or row.get("tcn_kernel_size")
            != ("3" if row["arm_id"] == "PlainCausalTCN-7var" else "")
            or row.get("thermoroute_d_model")
            != ("40" if row["arm_id"].startswith("ThermoRoute-") else "")
            or row.get("training_device") != "cpu"
            or row.get("historical_tuning_budget_equalized") != "False"
            or row.get("evidence_role") != "development_only_exploratory"
            for row in rows
        )
    ):
        raise ValueError("authorized Stage-09b architecture budget changed")
    try:
        with resolved_artifacts["metric_summary"].open(
            newline="", encoding="utf-8"
        ) as summary_handle:
            summary_rows = list(csv.DictReader(summary_handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("authorized Stage-09b metric summary is unreadable") from exc
    summary_registry = {
        (row.get("arm_id"), int(row.get("seed", "-1")), row.get("split"),
         int(row.get("horizon", "-1")))
        for row in summary_rows
    }
    expected_summary_registry = {
        (arm, seed, split, horizon)
        for arm, seed in expected_members
        for split in ("val", "calib", "test")
        for horizon in (1, 3, 7)
    }
    if (
        summary_registry != expected_summary_registry
        or len(summary_rows) != len(expected_summary_registry)
        or any(
            int(row.get("forecast_keys", "0")) < 1
            or int(row.get("stations", "0")) < 1
            or not math.isfinite(
                float(row.get("median_station_rmse_c", "nan"))
            )
            or not math.isfinite(float(row.get("micro_rmse_c", "nan")))
            or not math.isfinite(float(row.get("micro_mae_c", "nan")))
            for row in summary_rows
        )
    ):
        raise ValueError("authorized Stage-09b metric summary registry changed")
    try:
        budget_frame = pd.read_csv(resolved_artifacts["architecture_budget"])
        expected_report_bytes = _stage09b_expected_report(
            run_id=str(controls["run_id"]),
            audit=audit,
            budget=budget_frame,
            summary=recomputed_summary,
            paired_effects=paired_effects,
        )
    except Exception as exc:
        raise ValueError("authorized Stage-09b report cannot be regenerated") from exc
    if resolved_artifacts["report"].read_bytes() != expected_report_bytes:
        raise ValueError("authorized Stage-09b report is not summary-derived")
    semantic = _load_json(
        resolved_artifacts["semantic_audit"], label="Stage-09b semantic audit"
    )
    semantic_stable = dict(semantic)
    semantic_self = semantic_stable.pop("semantic_audit_self_sha256", None)
    semantic_members = semantic.get("members")
    derived = semantic.get("derived_artifacts")
    canonical_window = semantic.get("canonical_window_registry")
    if (
        set(semantic) != {
            "format", "status", "run_id", "evidence_scope",
            "training_replay_verified",
            "best_model_state_prediction_replay_verified",
            "post_2020_outcomes_requested_or_read", "matrix_audit",
            "canonical_window_registry", "scientific_summary", "members",
            "derived_artifacts",
            "semantic_audit_self_sha256",
        }
        or semantic.get("format")
        != "thermoroute.development-controls-semantic-audit.v3"
        or semantic.get("status")
        != "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY"
        or semantic.get("run_id") != controls.get("run_id")
        or semantic.get("evidence_scope") != "best_model_state_prediction_replay"
        or semantic.get("training_replay_verified") is not False
        or semantic.get("best_model_state_prediction_replay_verified") is not True
        or semantic.get("post_2020_outcomes_requested_or_read") is not False
        or semantic.get("matrix_audit") != audit
        or semantic_self != _sha256_json(semantic_stable)
        or not isinstance(semantic_members, list)
        or len(semantic_members) != 31
        or not isinstance(canonical_window, Mapping)
        or set(canonical_window)
        != {
            "sha256", "common_forecast_keys", "train_examples_per_epoch",
            "train_registry_sha256",
        }
        or canonical_window.get("sha256") != canonical_evaluation_sha256
        or canonical_window.get("train_registry_sha256") != canonical_train_sha256
        or canonical_window.get("common_forecast_keys")
        != audit["common_forecast_keys"]
        or canonical_window.get("train_examples_per_epoch")
        != canonical_train_examples
        or semantic.get("scientific_summary")
        != _stage09b_scientific_summary(paired_effects)
        or not isinstance(derived, Mapping)
        or set(derived) != {
            "architecture_budget", "combined_predictions", "metric_summary", "report"
        }
    ):
        raise ValueError("authorized Stage-09b semantic audit changed")
    for semantic_member, receipt_member, member in zip(
        semantic_members, members, expected_members, strict=True
    ):
        prediction_path = member_prediction_paths[member]
        prediction_sidecar_path = Path(
            root / receipt_member["prediction_sidecar"]["path"]
        )
        if (
            not isinstance(semantic_member, Mapping)
            or set(semantic_member) != {
                "arm_id", "seed", "checkpoint", "checkpoint_sidecar",
                "prediction", "prediction_sidecar",
                "normalised_prediction_sha256",
                "best_model_state_prediction_replay_verified",
            }
            or semantic_member.get("arm_id") != receipt_member.get("arm_id")
            or semantic_member.get("seed") != receipt_member.get("seed")
            or semantic_member.get("prediction", {}).get("sha256")
            != receipt_member.get("prediction", {}).get("sha256")
            or semantic_member.get("prediction_sidecar", {}).get("sha256")
            != receipt_member.get("prediction_sidecar", {}).get("sha256")
            or semantic_member.get("prediction", {}).get("bytes")
            != prediction_path.stat().st_size
            or semantic_member.get("prediction_sidecar", {}).get("bytes")
            != prediction_sidecar_path.stat().st_size
            or semantic_member.get("checkpoint", {}).get("sha256")
            != receipt_member.get("checkpoint", {}).get("sha256")
            or semantic_member.get("checkpoint_sidecar", {}).get("sha256")
            != receipt_member.get("checkpoint_sidecar", {}).get("sha256")
            or semantic_member.get("checkpoint", {}).get("bytes")
            != member_checkpoint_paths[member].stat().st_size
            or semantic_member.get("checkpoint_sidecar", {}).get("bytes")
            != member_checkpoint_sidecars[member].stat().st_size
            or semantic_member.get("best_model_state_prediction_replay_verified") is not True
            or semantic_member.get("normalised_prediction_sha256")
            != member_digests[member]
        ):
            raise ValueError("authorized Stage-09b semantic member audit changed")
    derived_labels = {
        "architecture_budget": ("architecture_budget", "architecture_budget_sidecar"),
        "combined_predictions": ("predictions", "prediction_sidecar"),
        "metric_summary": ("metric_summary", "metric_summary_sidecar"),
        "report": ("report", "report_sidecar"),
    }
    for semantic_label, (artifact_label, sidecar_label) in derived_labels.items():
        value = derived.get(semantic_label)
        if (
            not isinstance(value, Mapping)
            or value.get("artifact", {}).get("sha256")
            != artifacts[artifact_label].get("sha256")
            or value.get("sidecar", {}).get("sha256")
            != artifacts[sidecar_label].get("sha256")
            or value.get("artifact", {}).get("bytes")
            != resolved_artifacts[artifact_label].stat().st_size
            or value.get("sidecar", {}).get("bytes")
            != resolved_artifacts[sidecar_label].stat().st_size
        ):
            raise ValueError("authorized Stage-09b semantic artifact audit changed")
    _walk_json_dependencies(root, categories, "model_suite", stage9_path)
    _walk_json_dependencies(root, categories, "model_suite", controls_path)


def _independent_station_geometry(path: Path) -> dict[str, object]:
    """Recompute the outcome-free HUC2 concentration gate from frozen CSV bytes."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or {"site_no", "huc2"} - set(
                reader.fieldnames
            ):
                raise ValueError("station registry lacks site_no/huc2")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("cannot parse inference-gate station registry") from exc
    sites: set[str] = set()
    counts: dict[str, int] = {}
    for row in rows:
        site = str(row.get("site_no", "")).strip()
        huc2 = str(row.get("huc2", "")).strip()
        if not site or not huc2 or site in sites:
            raise ValueError("inference-gate station geometry is malformed")
        sites.add(site)
        counts[huc2] = counts.get(huc2, 0) + 1
    if not sites or not counts:
        raise ValueError("inference-gate station geometry is empty")
    n_stations = len(sites)
    cluster_sizes = sorted(counts.values())
    shares = [count / n_stations for count in cluster_sizes]
    effective = 1.0 / sum(share * share for share in shares)
    n_clusters = len(cluster_sizes)
    mean_size = n_stations / n_clusters
    size_cv = math.sqrt(
        sum((count - mean_size) ** 2 for count in cluster_sizes) / n_clusters
    ) / mean_size
    return {
        "n_stations": n_stations,
        "n_clusters": n_clusters,
        "cluster_sizes_sorted": cluster_sizes,
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "cluster_size_cv": size_cv,
        "largest_cluster_share": max(shares),
        "effective_cluster_count_inverse_herfindahl": effective,
        "effective_cluster_fraction": effective / n_clusters,
    }


def _validate_inference_closure(
    root: Path,
    categories: dict[str, set[Path]],
    authorization: Mapping[str, Any],
    *,
    protocol_binding: Mapping[str, Any],
    protocol_document: Mapping[str, Any],
    protocol_seal_path: Path,
    outcome_qc_policy: Mapping[str, Any],
    temporal_coverage_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate the prelabel amendment, seal, and fail-closed gate."""
    def exact_binding(path: Path) -> dict[str, str]:
        return {
            "path": _relative(root, path, label="inference closure artifact"),
            "sha256": sha256_file(path),
        }

    family = protocol_document.get("primary_inference_contract", {}).get(
        "confirmatory_family"
    )
    if not isinstance(family, list) or len(family) != 5:
        raise ValueError("inference closure lacks the exact five-test family")

    amendment_binding = authorization.get("inference_amendment")
    assert isinstance(amendment_binding, Mapping)
    amendment_path = _add_binding(
        root,
        categories,
        "inference_gates",
        amendment_binding,
        label="authorized inference amendment",
    )
    amendment = _load_json(amendment_path, label="inference amendment")
    if (
        set(amendment)
        != {
            "format", "status", "amendment_id", "recorded_date",
            "post_2020_wtemp_requested_or_inspected", "outcome_independent",
            "base_protocol", "base_protocol_seal", "scientific_comparisons",
            "estimand_scope", "inference_scope", "decision_overlay",
            "additional_preopen_gates", "trusted_scoring_recovery_contract",
            "lineage_contract",
        }
        or amendment.get("format")
        != "thermoroute.route-a-inference-amendment.v1"
        or amendment.get("status") != "FROZEN_PRELABEL_OUTCOME_FREE"
        or amendment.get("amendment_id")
        != "route-a-prelabel-inference-scope-014"
        or amendment.get("post_2020_wtemp_requested_or_inspected") is not False
        or amendment.get("outcome_independent") is not True
        or amendment.get("base_protocol")
        != {
            "path": protocol_binding.get("path"),
            "sha256": protocol_binding.get("sha256"),
        }
        or amendment.get("base_protocol_seal")
        != exact_binding(protocol_seal_path)
    ):
        raise ValueError("inference amendment identity or outcome-free attestation changed")
    comparisons = amendment.get("scientific_comparisons")
    if (
        not isinstance(comparisons, Mapping)
        or set(comparisons)
        != {
            "count", "confirmatory_family_sha256", "objects", "change_allowed"
        }
        or comparisons.get("count") != 5
        or comparisons.get("objects") != family
        or comparisons.get("confirmatory_family_sha256") != _sha256_json(family)
        or comparisons.get("change_allowed") is not False
    ):
        raise ValueError("inference amendment changed a scientific comparison")
    decision = amendment.get("decision_overlay")
    if (
        not isinstance(decision, Mapping)
        or decision.get("gate_artifact")
        != "outputs/prelabel/route_a_inference_gate_v1.json"
        or decision.get("all_gate_components_must_pass") is not True
        or decision.get("missing_unknown_or_not_run_is_failure") is not True
        or decision.get("gate_failure_verdict")
        != "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED"
        or decision.get("supported_claim_allowed_when_gate_fails") is not False
        or decision.get("strong_p_value_or_favorable_interval_cannot_override_gate")
        is not True
        or decision.get("all_five_comparisons_must_still_be_rendered_exactly_once")
        is not True
    ):
        raise ValueError("inference amendment decision overlay is not fail-closed")
    policy_overlays = amendment.get("additional_preopen_gates")
    if not isinstance(policy_overlays, Mapping) or set(policy_overlays) != {
        "outcome_qc_policy",
        "temporal_coverage_policy",
    }:
        raise ValueError("inference amendment preopen-gate registry changed")
    policy_overlay = policy_overlays.get("outcome_qc_policy")
    expected_policy_overlay = {
        "path": authorization["outcome_qc_policy"]["path"],
        "sha256": authorization["outcome_qc_policy"]["sha256"],
        "required": True,
        "role": OUTCOME_QC_AMENDMENT_ROLE,
    }
    if policy_overlay != expected_policy_overlay:
        raise ValueError("inference amendment binds another outcome-QC policy")
    expected_coverage_overlay = {
        "path": authorization["temporal_coverage_policy"]["path"],
        "sha256": authorization["temporal_coverage_policy"]["sha256"],
        "required": True,
        "role": TEMPORAL_COVERAGE_AMENDMENT_ROLE,
    }
    if (
        policy_overlays.get("temporal_coverage_policy")
        != expected_coverage_overlay
        or temporal_coverage_policy.get("format")
        != TEMPORAL_COVERAGE_POLICY_FORMAT
        or temporal_coverage_policy.get("policy_id")
        != TEMPORAL_COVERAGE_POLICY_ID
        or temporal_coverage_policy.get("status")
        != "FROZEN_PRELABEL_OUTCOME_FREE"
    ):
        raise ValueError(
            "inference amendment binds another temporal-coverage policy"
        )
    recovery = amendment.get("trusted_scoring_recovery_contract")
    if (
        not isinstance(recovery, Mapping)
        or dict(recovery) != TRUSTED_SCORING_RECOVERY_CONTRACT
    ):
        raise ValueError(
            "inference amendment trusted-scoring recovery contract changed"
        )
    if amendment.get("lineage_contract") != {
        "base_v1_files_remain_immutable": True,
        "separate_amendment_seal_required": True,
        "seal_path": "protocols/route_a_inference_amendment_seal_v1.json",
        "amendment_commit_must_precede_seal_commit": True,
    }:
        raise ValueError("inference amendment lineage contract changed")

    seal_path = _add_binding(
        root,
        categories,
        "inference_gates",
        amendment_binding.get("seal"),
        label="authorized inference amendment seal",
    )
    seal = _load_json(seal_path, label="inference amendment seal")
    if (
        set(seal)
        != {
            "format", "status", "amendment_id", "amendment",
            "base_protocol_seal", "final_prelabel_commit", "history_contract",
            "prelabel_attestation",
        }
        or seal.get("format")
        != "thermoroute.route-a-inference-amendment-seal.v1"
        or seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
        or seal.get("amendment_id") != amendment.get("amendment_id")
        or seal.get("amendment") != exact_binding(amendment_path)
        or seal.get("base_protocol_seal") != exact_binding(protocol_seal_path)
        or seal.get("final_prelabel_commit")
        != amendment_binding.get("final_prelabel_commit")
        or seal.get("history_contract")
        != {
            "base_protocol_commit_must_be_ancestor": True,
            "amendment_blob_must_match_commit": True,
            "amendment_commit_must_be_ancestor_of_authorization": True,
            "seal_is_created_only_after_amendment_commit": True,
        }
        or seal.get("prelabel_attestation")
        != {
            "post_2020_wtemp_requested_or_inspected": False,
            "outcome_independent": True,
        }
    ):
        raise ValueError("inference amendment seal semantics changed")

    gate_binding = authorization.get("inference_gate")
    assert isinstance(gate_binding, Mapping)
    gate_path = _add_binding(
        root,
        categories,
        "inference_gates",
        gate_binding,
        label="authorized inference gate",
    )
    gate = _load_json(gate_path, label="inference gate")
    gate_stable = dict(gate)
    gate_self = gate_stable.pop("gate_self_sha256", None)
    inputs = gate.get("inputs")
    gate_family = gate.get("confirmatory_family")
    policy = gate.get("policy")
    cluster_geometry = gate.get("cluster_geometry")
    station_registry_path = _resolve_release_path(
        root,
        authorization["registries"]["development"]["path"],
        label="inference-gate station registry",
    )
    expected_geometry = _independent_station_geometry(station_registry_path)
    threshold_failures: list[str] = []
    if int(expected_geometry["n_clusters"]) < 30:
        threshold_failures.append("SMALL_CLUSTER_COUNT_LT_30")
    if float(expected_geometry["effective_cluster_fraction"]) < 0.75:
        threshold_failures.append("EFFECTIVE_CLUSTER_FRACTION_LT_0_75")
    if float(expected_geometry["largest_cluster_share"]) >= 0.25:
        threshold_failures.append("DOMINANT_CLUSTER_SHARE_GE_0_25")
    structural_failures = [
        "INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING",
        "JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY",
    ]
    blocking = [
        *(f"STRUCTURAL_ASSUMPTION_NOT_ESTABLISHED:{value}"
          for value in structural_failures),
        *threshold_failures,
        "NULL_SIMULATION_NOT_PASSING",
    ]
    if (
        set(gate)
        != {
            "format", "status", "contains_confirmation_outcomes",
            "post_2020_outcomes_requested_or_inspected", "network_used", "inputs",
            "confirmatory_family", "policy", "policy_sha256", "cluster_geometry",
            "cluster_gate", "structural_assumption_gate", "null_simulation_gate",
            "claim_eligible", "analysis_mode", "blocking_reasons",
            "gate_self_sha256",
        }
        or gate_self != _sha256_json(gate_stable)
        or gate.get("format") != "thermoroute.route-a-inference-gate.v1"
        or gate.get("status") != "FAIL_CLOSED_DESCRIPTIVE_ONLY"
        or gate.get("contains_confirmation_outcomes") is not False
        or gate.get("post_2020_outcomes_requested_or_inspected") is not False
        or gate.get("network_used") is not False
        or gate.get("claim_eligible") is not False
        or gate.get("analysis_mode") != "FIXED_COHORT_DESCRIPTIVE_ONLY"
        or not isinstance(inputs, Mapping)
        or inputs.get("base_protocol")
        != {
            "path": protocol_binding.get("path"),
            "sha256": protocol_binding.get("sha256"),
        }
        or inputs.get("base_protocol_seal") != exact_binding(protocol_seal_path)
        or inputs.get("station_registry")
        != {
            "path": authorization["registries"]["development"]["path"],
            "sha256": authorization["registries"]["development"]["sha256"],
        }
        or inputs.get("source")
        != {
            "source_tree_sha256": authorization["source"]["source_tree_sha256"],
            "source_inventory": authorization["source"]["source_inventory"],
        }
        or gate_family
        != {
            "count": 5,
            "sha256": _sha256_json(family),
            "objects": family,
            "candidate_reference_horizon_margin_unchanged": True,
        }
        or not isinstance(policy, Mapping)
        or gate.get("policy_sha256") != _sha256_json(policy)
        or gate.get("policy_sha256") != gate_binding.get("policy_sha256")
        or policy.get("cluster_thresholds")
        != {
            "minimum_clusters": 30,
            "minimum_effective_cluster_fraction": 0.75,
            "maximum_largest_cluster_share_exclusive": 0.25,
        }
        or policy.get("decision", {}).get("all_components_must_pass") is not True
        or policy.get("decision", {}).get("missing_unknown_or_not_run_is_failure")
        is not True
        or policy.get("decision", {}).get("failed_mode")
        != "FIXED_COHORT_DESCRIPTIVE_ONLY"
        or cluster_geometry != expected_geometry
        or gate.get("cluster_gate")
        != {"pass": not threshold_failures, "failure_codes": threshold_failures}
        or gate.get("structural_assumption_gate")
        != {"pass": False, "failure_codes": structural_failures}
        or gate.get("null_simulation_gate", {}).get("pass") is not False
        or gate.get("null_simulation_gate", {}).get("outcomes_read") is not False
        or gate.get("null_simulation_gate", {}).get("network_used") is not False
        or gate.get("blocking_reasons") != blocking
        or {key: gate_binding.get(key) for key in (
            "format", "status", "claim_eligible", "analysis_mode", "policy_sha256"
        )}
        != {key: gate.get(key) for key in (
            "format", "status", "claim_eligible", "analysis_mode", "policy_sha256"
        )}
    ):
        raise ValueError("inference gate semantics or frozen inputs changed")
    if outcome_qc_policy.get("confirmatory_family_sha256") != gate_family["sha256"]:
        raise ValueError("inference and outcome-QC gates bind different test families")
    return dict(gate)


def _gather_postopen_categories(
    root: Path, authorization_path: Path
) -> tuple[dict[str, set[Path]], dict[str, Any], dict[str, str]]:
    root, authorization_path = root.resolve(), authorization_path.resolve()
    categories = _canonical_categories(root)
    authorization, state = _validate_authorization_structure(root, authorization_path)
    _add_path(root, categories, "authorization", authorization_path)
    work_order_path = _resolve_release_path(
        root, state["work_order"], label="acquisition work order"
    )
    _add_path(root, categories, "authorization", work_order_path)

    protocol_binding = authorization.get("protocol")
    protocol = _add_binding(
        root, categories, "authorization", protocol_binding,
        label="authorized protocol",
    )
    if protocol.suffix != ".json":
        raise ValueError("authorized protocol is not machine-readable JSON")
    if not isinstance(protocol_binding, Mapping):
        raise ValueError("authorized protocol binding is malformed")
    protocol_document = _load_json(protocol, label="authorized protocol")
    outcome_qc_policy_path = _add_binding(
        root, categories, "authorization", authorization.get("outcome_qc_policy"),
        label="authorized outcome-QC policy",
    )
    outcome_qc_policy = _load_json(
        outcome_qc_policy_path, label="authorized outcome-QC policy",
    )
    temporal_coverage_policy_path = _add_binding(
        root,
        categories,
        "authorization",
        authorization.get("temporal_coverage_policy"),
        label="authorized temporal-coverage policy",
    )
    temporal_coverage_policy = _load_json(
        temporal_coverage_policy_path,
        label="authorized temporal-coverage policy",
    )
    if (
        _relative(
            root,
            temporal_coverage_policy_path,
            label="temporal-coverage policy",
        )
        != TEMPORAL_COVERAGE_POLICY_PATH
        or sha256_file(temporal_coverage_policy_path)
        != TEMPORAL_COVERAGE_POLICY_SHA256
    ):
        raise ValueError("authorized temporal-coverage policy bytes changed")
    outcome_qc_module = _load_canonical_outcome_qc_module(root)
    try:
        validated_outcome_qc_policy = outcome_qc_module.validate_outcome_qc_policy(
            outcome_qc_policy_path,
            root=root,
            protocol_path=protocol,
        )
    except Exception as exc:
        raise ValueError("authorized outcome-QC policy semantics changed") from exc
    family = protocol_document.get("primary_inference_contract", {}).get(
        "confirmatory_family"
    )
    if (
        validated_outcome_qc_policy != outcome_qc_policy
        or not isinstance(family, list)
        or len(family) != 5
    ):
        raise ValueError("authorized outcome-QC policy semantics changed")
    seal_path, seal = _load_protocol_seal(root, protocol_document)
    final_protocol = seal.get("final_prelabel_protocol")
    if not isinstance(final_protocol, Mapping):
        raise ValueError("final protocol seal section is malformed")
    for key in ("json", "markdown"):
        _add_binding(
            root,
            categories,
            "authorization",
            final_protocol.get(key),
            label=f"sealed final protocol {key}",
        )
    authorized_seal = _add_binding(
        root,
        categories,
        "authorization",
        protocol_binding.get("seal"),
        label="authorized final protocol seal",
    )
    if authorized_seal != seal_path:
        raise ValueError("authorization binds a noncanonical protocol seal")
    final = final_protocol
    original = seal.get("original_preregistration")
    if (
        not isinstance(final, Mapping)
        or not isinstance(original, Mapping)
        or protocol_binding.get("final_prelabel_commit") != final.get("commit")
        or protocol_binding.get("authoritative_commit") != original.get("commit")
    ):
        raise ValueError("authorization protocol chronology differs from its seal")
    _validate_inference_closure(
        root,
        categories,
        authorization,
        protocol_binding=protocol_binding,
        protocol_document=protocol_document,
        protocol_seal_path=seal_path,
        outcome_qc_policy=outcome_qc_policy,
        temporal_coverage_policy=temporal_coverage_policy,
    )

    registries = authorization.get("registries")
    required_registries = {
        "development", "external", "external_lock", "development_panel_spec",
        "candidate_table", "candidate_provenance", "candidate_snapshot_index",
    }
    if not isinstance(registries, Mapping) or set(registries) != required_registries:
        raise ValueError("authorization registry/evidence bindings are incomplete")
    for key in ("development", "external", "external_lock", "development_panel_spec"):
        path = _add_binding(
            root, categories, "registries", registries[key],
            label=f"authorized registry {key}",
        )
        if path.suffix == ".json":
            _walk_json_dependencies(root, categories, "registries", path)
    for key in ("candidate_table", "candidate_provenance", "candidate_snapshot_index"):
        path = _add_binding(
            root, categories, "candidate_evidence", registries[key],
            label=f"authorized candidate evidence {key}",
        )
        if path.suffix == ".json":
            _walk_json_dependencies(root, categories, "candidate_evidence", path)

    suite_path = _add_binding(
        root, categories, "model_suite", authorization.get("model_suite"),
        label="authorized model suite",
    )
    suite = _load_json(suite_path, label="model suite")
    if (
        suite.get("format") != "thermoroute.route-a-model-suite.v1"
        or suite.get("status") != "FROZEN_BEFORE_LABEL_OPENING"
    ):
        raise ValueError("authorized model suite is not frozen before opening")
    suite_runtime = suite.get("numerical_runtime_sha256")
    if (
        suite.get("training_device") != "cpu"
        or not isinstance(suite_runtime, str)
        or not re.fullmatch(r"[0-9a-f]{64}", suite_runtime)
        or suite_runtime != authorization.get("runtime", {}).get("runtime_sha256")
    ):
        raise ValueError(
            "authorized model suite is not bound to the exact CPU numerical runtime"
        )
    development = suite.get("development_contract")
    if not isinstance(development, Mapping):
        raise ValueError("authorized model suite lacks its development contract")
    bridge_path = _add_binding(
        root,
        categories,
        "model_suite",
        development.get("predictor_bridge"),
        label="development predictor bridge",
    )
    bridge = _load_json(bridge_path, label="development predictor bridge")
    if (
        bridge.get("format") != "thermoroute.development-predictor-bridge.v1"
        or bridge.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or bridge.get("outcome_values_requested_or_read") is not False
        or bridge.get("panel") != development.get("panel")
        or bridge.get("registry") != development.get("registry")
        # The bridge records the source that produced the earlier outcome-free
        # compatibility audit.  Its manifest bytes are frozen below; it is not
        # expected to equal the later, gate-hardened training source identity.
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(bridge.get("source_tree_sha256", ""))
        )
    ):
        raise ValueError("development predictor bridge is stale or not an exact PASS")
    _walk_json_dependencies(root, categories, "model_suite", bridge_path)
    _validate_preopening_completion_gates(
        root, categories, suite, development, suite_runtime
    )
    cohorts = suite.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {"temporal", "external"}:
        raise ValueError("model suite lacks temporal/external cohorts")
    for cohort in ("temporal", "external"):
        item = cohorts[cohort]
        entries = item.get("models") if isinstance(item, Mapping) else None
        if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
            raise ValueError(f"model suite {cohort} registry is malformed")
        ids = [str(entry.get("model_id")) for entry in entries]
        if len(ids) != len(set(ids)) or set(ids) != _required_model_ids(cohort):
            raise ValueError(f"model suite {cohort} is incomplete")
        for entry in entries:
            if entry.get("executor") == "builtin":
                if "artifact" in entry:
                    raise ValueError(f"builtin {cohort}/{entry.get('model_id')} has an artifact")
                continue
            artifact = _add_binding(
                root, categories, "model_bundles", entry.get("artifact"),
                label=f"model bundle {cohort}/{entry.get('model_id')}",
            )
            if artifact.is_file() and artifact.suffix == ".json":
                _walk_json_dependencies(root, categories, "model_bundles", artifact)
            elif artifact.is_dir():
                for child in sorted(artifact.rglob("*.json")):
                    _walk_json_dependencies(root, categories, "model_bundles", child)
    _walk_json_dependencies(root, categories, "model_suite", suite_path)
    replay_path = _add_binding(
        root,
        categories,
        "model_suite",
        authorization.get("development_replay"),
        label="authorized full development replay receipt",
    )
    replay = _load_json(replay_path, label="development replay receipt")
    if (
        replay.get("format") != "thermoroute.route-a-development-replay.v1"
        or replay.get("status")
        != "PASS_FULL_DEVELOPMENT_REPLAY_NO_CONFIRMATION_DATA"
        or replay.get("suite") != authorization.get("model_suite")
        or replay.get("source_tree_sha256")
        != authorization.get("source", {}).get("source_tree_sha256")
        or replay.get("runtime_sha256")
        != authorization.get("runtime", {}).get("runtime_sha256")
        or replay.get("source_tree_sha256") != development.get("source_sha256")
        or replay.get("runtime_sha256") != suite_runtime
        or replay.get("confirmation_period_read") is not False
    ):
        raise ValueError("authorized development replay receipt is stale or malformed")

    _validate_prelabel_chronology_structure(root, categories, authorization)

    inputs_path = _add_binding(
        root, categories, "prelabel_inputs", authorization.get("actual_inputs"),
        label="authorized pre-label input manifest",
    )
    inputs = _load_json(inputs_path, label="pre-label input manifest")
    expected_prelabel = {
        "format": "thermoroute.route-a-prelabel-inputs.v1",
        "status": "FROZEN_PRELABEL_NO_OUTCOMES",
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "post_2020_wtemp_requested_or_inspected": False,
    }
    wrong_inputs = {
        key: inputs.get(key) for key, value in expected_prelabel.items()
        if inputs.get(key) != value
    }
    if wrong_inputs:
        raise ValueError(f"pre-label input safety contract changed: {wrong_inputs}")
    tables = inputs.get("cohort_tables")
    if not isinstance(tables, Mapping) or set(tables) != {"temporal", "external"}:
        raise ValueError("pre-label input manifest lacks both cohort tables")
    for cohort, binding in tables.items():
        _add_binding(
            root, categories, "prelabel_inputs", binding,
            label=f"pre-label {cohort} table",
        )
    evidence = inputs.get("source_evidence")
    if not isinstance(evidence, list) or len(evidence) < 4:
        raise ValueError("pre-label input manifest lacks raw meteorology evidence")
    for index, item in enumerate(evidence):
        if (
            not isinstance(item, Mapping)
            or item.get("contains_outcome") is not False
            or item.get("contains_outcome_labels") is not False
        ):
            raise ValueError("raw meteorology evidence does not exclude outcomes")
        path = _add_binding(
            root, categories, "raw_meteorology", item.get("artifact"),
            label=f"meteorology evidence {index}",
        )
        if path.suffix == ".json":
            _walk_json_dependencies(root, categories, "raw_meteorology", path)

    intent_path = _resolve_release_path(root, state["intent"], label="opening intent")
    intent = _load_json(intent_path, label="opening intent")
    _add_path(root, categories, "opening_intent", intent_path)
    intent_stable = dict(intent)
    intent_self = intent_stable.pop("intent_self_sha256", None)
    if not isinstance(intent_self, str) or intent_self != _sha256_json(intent_stable):
        raise ValueError("opening intent self hash is inconsistent")
    if (
        intent.get("format") != INTENT_FORMAT
        or intent.get("status") != "OPENING_STARTED_IRREVERSIBLE"
        or intent.get("opening_id") != authorization["opening_id"]
        or intent.get("maximum_openings") != 1
        or intent.get("retry_after_failure_allowed") is not False
        or intent.get("same_opening_transport_resume_allowed") is not True
        or intent.get("unsafe_test_only") is not None
    ):
        raise ValueError("opening intent is not a production one-shot marker")

    receipt_path = _resolve_release_path(root, state["receipt"], label="opening receipt")
    receipt = _load_json(receipt_path, label="opening receipt")
    _add_path(root, categories, "receipt", receipt_path)
    receipt_stable = dict(receipt)
    receipt_self = receipt_stable.pop("receipt_self_sha256", None)
    if not isinstance(receipt_self, str) or receipt_self != _sha256_json(receipt_stable):
        raise ValueError("opening receipt self hash is inconsistent")
    authorization_sha = sha256_file(authorization_path)
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("status") != "OPENED_AND_SCORED_ONCE"
        or receipt.get("opening_id") != authorization["opening_id"]
        or receipt.get("authorization_sha256") != authorization_sha
        or receipt.get("intent_sha256") != sha256_file(intent_path)
        or receipt.get("opening_count") != 1
        or receipt.get("maximum_openings") != 1
        or receipt.get("retry_after_failure_allowed") is not False
        or receipt.get("same_opening_transport_resume_allowed") is not True
        or receipt.get("all_predeclared_models_reported") is not True
        or receipt.get("state_paths") != dict(state)
        or receipt.get("unsafe_test_only") is not None
    ):
        raise ValueError("opening receipt is not a complete production one-shot receipt")
    if intent.get("authorization_sha256") != authorization_sha:
        raise ValueError("opening intent is bound to another authorization")
    if intent.get("trusted_validator") != receipt.get("trusted_validator"):
        raise ValueError("intent/receipt trusted-validator attestations differ")
    validator = receipt.get("trusted_validator")
    if not isinstance(validator, Mapping) or not isinstance(validator.get("sha256"), str):
        raise ValueError("receipt lacks a trusted-validator environment attestation")
    work_order_document = _load_json(
        work_order_path, label="acquisition work order"
    )
    work_order_stable = dict(work_order_document)
    work_order_self = work_order_stable.pop("work_order_self_sha256", None)
    if not isinstance(work_order_self, str) or work_order_self != _sha256_json(work_order_stable):
        raise ValueError("acquisition work-order self hash is inconsistent")
    preflight = receipt.get("preflight_attestation")
    if (
        not isinstance(preflight, Mapping)
        or receipt.get("preflight_attestation_sha256") != _sha256_json(preflight)
        or intent.get("preflight_attestation_sha256") != _sha256_json(preflight)
        or preflight.get("prelabel_chronology_sha256")
        != authorization.get("prelabel_chronology", {}).get("sha256")
        or receipt.get("work_order_sha256") != sha256_file(
            _resolve_release_path(root, state["work_order"], label="acquisition work order")
        )
        or intent.get("work_order_self_sha256") != work_order_self
        or intent.get("work_order_file_sha256") != sha256_file(
            _resolve_release_path(root, state["work_order"], label="acquisition work order")
        )
        or intent.get("fixed_code_sha256") != authorization.get("fixed_code", {}).get("sha256")
        or intent.get("runtime_sha256") != authorization["runtime"].get("runtime_sha256")
        or receipt.get("fixed_code") != authorization.get("fixed_code")
        or receipt.get("authorized_runtime") != authorization.get("runtime")
        or receipt.get("intent_self_sha256") != intent_self
    ):
        raise ValueError("intent/receipt work-order, preflight, code or runtime binding changed")
    completion = receipt.get("completion_environment")
    if (
        not isinstance(completion, Mapping)
        or completion.get("numerical_runtime_sha256")
        != authorization["runtime"].get("runtime_sha256")
    ):
        raise ValueError("receipt completion environment differs from authorization")
    required_models = authorization.get("required_models")
    if not isinstance(required_models, Mapping) or set(required_models) != {"temporal", "external"}:
        raise ValueError("authorization lacks the required-model registry")
    expected_reported_models = {
        cohort: sorted(str(value) for value in values)
        for cohort, values in required_models.items()
    }
    if receipt.get("reported_models") != expected_reported_models:
        raise ValueError("receipt reported-model registry differs from authorization")
    _add_path(root, categories, "environment_attestations", authorization_path)
    _add_path(root, categories, "environment_attestations", intent_path)
    _add_path(root, categories, "environment_attestations", receipt_path)
    lock_path = _add_binding(
        root, categories, "environment_attestations",
        authorization["runtime"].get("requirements_lock"),
        label="authorized requirements lock",
    )
    if lock_path.name != "requirements-lock.txt":
        raise ValueError("runtime attestation names another dependency lock")
    hashed_lock_path = _add_binding(
        root,
        categories,
        "reproducibility_lock",
        authorization["runtime"].get("hashed_requirements_lock"),
        label="authorized fully hashed requirements lock",
    )
    if hashed_lock_path.name != REPRODUCIBILITY_LOCK:
        raise ValueError("runtime attestation names another hashed dependency lock")
    _verify_fully_hashed_lock(hashed_lock_path)
    fixed_code = authorization.get("fixed_code")
    if not isinstance(fixed_code, Mapping):
        raise ValueError("authorization lacks fixed-code attestation")
    for group in ("modules", "files", "entrypoints"):
        values = fixed_code.get(group)
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"fixed-code attestation lacks {group}")
        for name, binding in values.items():
            _add_binding(
                root, categories, "environment_attestations", binding,
                label=f"fixed code {group}/{name}",
            )

    acquisition_path = _resolve_release_path(
        root, state["acquisition_manifest"], label="acquisition manifest"
    )
    acquisition = _load_json(acquisition_path, label="acquisition manifest")
    _add_path(root, categories, "raw_nwis", acquisition_path)
    if (
        acquisition.get("opening_id") != authorization["opening_id"]
        or acquisition.get("authorization_sha256") != authorization_sha
        or acquisition.get("labels_state") != "OPENED_ONCE"
        or acquisition.get("site_replacement_count") != 0
        or acquisition.get("response_replacement_count") != 0
        or acquisition.get("producer_role") != "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS"
    ):
        raise ValueError("acquisition manifest identity or raw-only role changed")
    transport = acquisition.get("transport_summary")
    completed_before_final = (
        transport.get("completed_before_final_attempt_request_sha256")
        if isinstance(transport, Mapping) else None
    )
    retrieval_span = transport.get("retrieval_span_utc") if isinstance(
        transport, Mapping
    ) else None
    if (
        not isinstance(transport, Mapping)
        or set(transport) != {
            "opening_count", "attempt_count", "resume_count",
            "completed_before_final_attempt_request_sha256",
            "retrieval_span_utc",
        }
        or transport.get("opening_count") != 1
        or type(transport.get("attempt_count")) is not int
        or transport["attempt_count"] < 1
        or type(transport.get("resume_count")) is not int
        or not 0 <= transport["resume_count"] < transport["attempt_count"]
        or not isinstance(completed_before_final, list)
        or len(completed_before_final) != len(set(completed_before_final))
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in completed_before_final
        )
        or not isinstance(retrieval_span, Mapping)
        or set(retrieval_span) != {"first", "last"}
        or any(
            not isinstance(retrieval_span.get(field), str)
            or not retrieval_span[field]
            for field in ("first", "last")
        )
        or receipt.get("transport_recovery") != transport
    ):
        raise ValueError("receipt/acquisition transport evidence differs")
    raw_root = _resolve_release_path(root, state["raw_nwis_root"], label="raw NWIS root")
    _add_path(root, categories, "raw_nwis", raw_root)
    for key in (
        "request_ledger",
        "transport_attempt_index",
        "raw_nwis_snapshot_index",
        "request_map",
    ):
        path = _add_binding(
            root, categories, "raw_nwis", acquisition.get(key),
            label=f"acquisition {key}",
        )
        if path.suffix == ".json":
            _walk_json_dependencies(root, categories, "raw_nwis", path)
    normalized = acquisition.get("normalized_outcome_tables")
    if not isinstance(normalized, Mapping) or set(normalized) != {"temporal", "external"}:
        raise ValueError("acquisition manifest lacks both normalized outcome tables")
    for cohort, binding in normalized.items():
        _add_binding(
            root, categories, "normalized_outcomes", binding,
            label=f"normalized {cohort} outcomes",
        )

    receipt_artifacts = receipt.get("artifacts")
    if not isinstance(receipt_artifacts, Mapping) or set(receipt_artifacts) != REQUIRED_RECEIPT_ARTIFACTS:
        raise ValueError("receipt artifact registry is incomplete or contains extras")
    receipt_categories = {
        "acquisition_manifest": "raw_nwis",
        "raw_nwis_snapshot_index": "raw_nwis",
        "acquisition_request_map": "raw_nwis",
        "temporal_normalized_outcomes": "normalized_outcomes",
        "external_normalized_outcomes": "normalized_outcomes",
        "availability_registry": "availability",
        "outcome_quality_audit": "sensitivity_audits",
        "outcome_qc_gate": "outcome_qc",
        "approved_target_sensitivity": "sensitivity_audits",
        "spatial_sensitivity": "sensitivity_audits",
        "probabilistic_evaluation": "probabilistic_evaluation",
        "temporal_predictions": "trusted_predictions",
        "external_predictions": "trusted_predictions",
        "statistics": "statistics",
        "temporal_coverage_audit": "temporal_coverage",
        "report": "report",
    }
    resolved_receipt_artifacts: dict[str, Path] = {}
    for key, category in receipt_categories.items():
        resolved_receipt_artifacts[key] = _add_binding(
            root, categories, category, receipt_artifacts[key],
            label=f"receipt artifact {key}",
        )
    release_bindings = receipt.get("release_bindings")
    released = release_bindings.get("artifacts") if isinstance(release_bindings, Mapping) else None
    if (
        not isinstance(release_bindings, Mapping)
        or release_bindings.get("format") != "thermoroute.route-a-release-bindings.v1"
        or release_bindings.get("opening_id") != authorization["opening_id"]
        or release_bindings.get("state_namespace") != state["namespace"]
        or release_bindings.get("authorization") != {
            "format": AUTHORIZATION_FORMAT,
            "path": _relative(root, authorization_path, label="authorization"),
            "sha256": authorization_sha,
        }
        or not isinstance(released, Mapping)
        or set(released) != REQUIRED_RECEIPT_ARTIFACTS
        or release_bindings.get("receipt") != {
            "format": RECEIPT_FORMAT,
            "path": state["receipt"],
            "external_sha256_path": state["receipt_sha256"],
        }
    ):
        raise ValueError("receipt release-binding registry is incomplete")
    for key, binding in receipt_artifacts.items():
        released_binding = released[key]
        if (
            not isinstance(released_binding, Mapping)
            or not isinstance(released_binding.get("format"), str)
            or not released_binding.get("format")
            or {field: released_binding.get(field) for field in ("path", "sha256")}
            != dict(binding)
        ):
            raise ValueError(f"release binding differs from receipt artifact: {key}")
    expected_paths = {
        "acquisition_manifest": state["acquisition_manifest"],
        "acquisition_request_map": state["acquisition_request_map"],
        "temporal_normalized_outcomes": state["temporal_outcomes"],
        "external_normalized_outcomes": state["external_outcomes"],
        "availability_registry": state["availability_registry"],
        "outcome_quality_audit": state["outcome_quality_audit"],
        "outcome_qc_gate": state["outcome_qc_gate"],
        "approved_target_sensitivity": state["approved_target_sensitivity"],
        "spatial_sensitivity": state["spatial_sensitivity"],
        "probabilistic_evaluation": state["probabilistic_evaluation"],
        "temporal_predictions": state["temporal_predictions"],
        "external_predictions": state["external_predictions"],
        "statistics": state["statistics"],
        "temporal_coverage_audit": state["temporal_coverage_audit"],
        "report": state["report"],
    }
    for key, relative in expected_paths.items():
        if _relative(root, resolved_receipt_artifacts[key], label=key) != relative:
            raise ValueError(f"receipt {key} leaves its canonical state path")
    statistics = _load_json(resolved_receipt_artifacts["statistics"], label="statistics")
    tests = statistics.get("tests")
    if (
        statistics.get("format") != STATISTICS_FORMAT
        or not isinstance(tests, list)
        or len(tests) != 5
        or receipt.get("formal_tests") != tests
    ):
        raise ValueError("receipt/statistics do not contain the exact five-test family")
    coverage_binding = receipt_artifacts["temporal_coverage_audit"]
    if (
        released["temporal_coverage_audit"].get("format")
        != TEMPORAL_COVERAGE_AUDIT_FORMAT
    ):
        raise ValueError("release binding temporal-coverage format changed")
    expected_coverage_receipt = {
        **dict(coverage_binding),
        "format": TEMPORAL_COVERAGE_AUDIT_FORMAT,
        "core_status": TEMPORAL_COVERAGE_CORE_STATUS,
        "physical_replay_verified": True,
        "source_binding_count": 11,
    }
    if receipt.get("temporal_coverage_audit") != expected_coverage_receipt:
        raise ValueError(
            "opening receipt temporal-coverage replay attestation changed"
        )
    coverage_audit = _load_json(
        resolved_receipt_artifacts["temporal_coverage_audit"],
        label="temporal-coverage audit",
    )
    coverage_module = _load_canonical_coverage_bridge_module(root)
    try:
        replayed_coverage = (
            coverage_module.replay_temporal_coverage_from_physical_files(
                root=root,
                authorization=authorization,
                receipt_artifacts=receipt_artifacts,
                expected_audit=coverage_audit,
            )
        )
    except Exception as exc:
        raise ValueError(
            "temporal-coverage audit cannot be replayed from physical evidence"
        ) from exc
    if replayed_coverage != coverage_audit:
        raise ValueError(
            "temporal-coverage audit differs from independent physical replay"
        )
    outcome_qc_gate = _load_json(
        resolved_receipt_artifacts["outcome_qc_gate"], label="outcome-QC gate",
    )
    availability_contract = protocol_document.get("availability_contract")
    minimum_targets = (
        availability_contract.get("minimum_valid_targets_per_station_horizon")
        if isinstance(availability_contract, Mapping)
        else None
    )
    if type(minimum_targets) is not int or minimum_targets < 2:
        raise ValueError("authorized protocol has an invalid outcome-QC minimum")
    try:
        validated_outcome_qc_gate = (
            outcome_qc_module.validate_outcome_qc_gate_structure(
                outcome_qc_gate,
                root=root,
                policy_path=outcome_qc_policy_path,
                protocol=protocol_document,
                minimum_targets=minimum_targets,
            )
        )
    except Exception as exc:
        raise ValueError("outcome-QC gate semantics changed") from exc
    if validated_outcome_qc_gate != outcome_qc_gate:
        raise ValueError("outcome-QC gate semantics changed")
    gate_pass = outcome_qc_gate["pass"]
    expected_gate_binding = {
        "path": receipt_artifacts["outcome_qc_gate"]["path"],
        "sha256": receipt_artifacts["outcome_qc_gate"]["sha256"],
        "format": "thermoroute.route-a-outcome-qc-gate.v1",
        "status": outcome_qc_gate.get("status"),
        "pass": gate_pass,
        "directional_claims_allowed": outcome_qc_gate.get(
            "directional_claims_allowed_by_outcome_qc"
        ),
    }
    if statistics.get("outcome_qc_gate") != expected_gate_binding:
        raise ValueError("outcome-QC gate semantics or statistics binding changed")
    if resolved_receipt_artifacts["report"].stat().st_size == 0:
        raise ValueError("trusted report is empty")

    sidecar = _resolve_release_path(root, state["receipt_sha256"], label="receipt checksum")
    _add_path(root, categories, "receipt", sidecar)
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or fields[0] != sha256_file(receipt_path):
        raise ValueError("receipt checksum sidecar does not bind the receipt")
    if not REQUIRED_POSTOPEN_CATEGORIES <= set(categories):
        missing = sorted(REQUIRED_POSTOPEN_CATEGORIES - set(categories))
        raise ValueError(f"post-opening closure categories are absent: {missing}")
    return categories, authorization, state


def _derive_release_claim_status(
    root: Path,
    authorization: Mapping[str, Any],
    state: Mapping[str, str],
) -> dict[str, object]:
    """Separate completed scoring from claim support and recompute every verdict."""
    gate = _load_json(
        _resolve_release_path(
            root,
            str(authorization["inference_gate"]["path"]),
            label="release inference gate",
        ),
        label="release inference gate",
    )
    outcome_gate = _load_json(
        _resolve_release_path(
            root, state["outcome_qc_gate"], label="release outcome-QC gate"
        ),
        label="release outcome-QC gate",
    )
    statistics = _load_json(
        _resolve_release_path(root, state["statistics"], label="release statistics"),
        label="release statistics",
    )
    inference_allowed = gate.get("claim_eligible") is True
    gross_plausibility_and_sensitivity_allowed = (
        outcome_gate.get("directional_claims_allowed_by_outcome_qc") is True
    )
    directional_allowed = (
        inference_allowed and gross_plausibility_and_sensitivity_allowed
    )
    tests = statistics.get("tests")
    if not isinstance(tests, list) or len(tests) != 5:
        raise ValueError("cannot derive release claim status from malformed tests")
    supported: list[str] = []
    for row in tests:
        if not isinstance(row, Mapping) or not isinstance(row.get("test_id"), str):
            raise ValueError("cannot derive release claim status from malformed test")
        p_support = row.get("reject_at_0_05") is True
        interval_support = row.get("confidence_bound_supports_margin") is True
        if p_support != interval_support and row.get("status") == "ESTIMABLE":
            # The claim validator renders this as an evidence conflict.  Keeping
            # it out of the support list is mandatory even when both gates pass.
            continue
        if (
            directional_allowed
            and row.get("status") == "ESTIMABLE"
            and p_support
            and interval_support
        ):
            supported.append(str(row["test_id"]))
    return {
        "confirmatory_scoring_completed": True,
        "directional_claims_allowed": directional_allowed,
        "inference_gate_claim_eligible": inference_allowed,
        "gross_plausibility_and_aggregate_sensitivity_gate_passed": (
            gross_plausibility_and_sensitivity_allowed
        ),
        "supported_test_ids": supported,
        "supports_route_a_confirmatory_conclusions": bool(supported),
    }


def build_release_profile(
    root: str | Path,
    profile: str,
    *,
    authorization_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, set[Path]]]:
    root = Path(root).resolve()
    if profile not in RELEASE_PROFILES:
        raise ValueError(f"unknown release profile: {profile}")
    if profile == PREOPEN_PROFILE:
        if authorization_path is not None:
            raise ValueError("pre-opening profile must not accept an authorization")
        categories = _canonical_categories(root)
        document: dict[str, Any] = {
            "format": PROFILE_FORMAT,
            "profile": PREOPEN_PROFILE,
            "status": PREOPEN_PROFILE,
            "confirmatory_scoring_completed": False,
            "directional_claims_allowed": False,
            "supported_test_ids": [],
            "supports_route_a_confirmatory_conclusions": False,
            "labels_included": False,
            "warning": PREOPEN_WARNING,
            "fully_hashed_lock_role": HASHED_LOCK_ROLE,
            "forbidden_prefixes": ["outputs/confirmatory/"],
            "forbidden_path_components": ["labels"],
            "artifact_closure": _category_bindings(root, categories),
        }
        return document, categories
    if authorization_path is None:
        raise ValueError("ROUTE_A_OPENED_COMPLETE requires --authorization")
    authorization_path = Path(authorization_path).resolve()
    categories, authorization, state = _gather_postopen_categories(
        root, authorization_path
    )
    claim_status = _derive_release_claim_status(root, authorization, state)
    document = {
        "format": PROFILE_FORMAT,
        "profile": POSTOPEN_PROFILE,
        "status": POSTOPEN_PROFILE,
        **claim_status,
        "labels_included": True,
        "opening_id": authorization["opening_id"],
        "state_namespace": state["namespace"],
        "fully_hashed_lock_role": HASHED_LOCK_ROLE,
        "authorization": _binding_for(root, authorization_path),
        "authorized_worktree_dirt_policy": _postopen_revision_contract(
            root,
            authorization_path,
            authorization,
            state,
            require_git=(root / ".git").exists(),
        ),
        "trusted_replay_interface": {
            "entrypoint": "scripts/route_a_trusted_scorer.py",
            "arguments": ["--verify-release", "--authorization", _relative(
                root, authorization_path, label="authorization"
            )],
            "policy": "fixed-entrypoint-fresh-python-I",
        },
        "artifact_closure": _category_bindings(root, categories),
    }
    return document, categories


def _copy_file(source_root: Path, stage_root: Path, source: Path) -> None:
    relative = _relative(source_root, source, label="release artifact")
    destination = stage_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise ValueError(f"staged artifact conflicts with profile closure: {relative}")
        return
    shutil.copyfile(source, destination)
    destination.chmod(0o644)


def materialize_release_profile(
    source_root: str | Path,
    stage_root: str | Path,
    profile: str,
    *,
    authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    source_root, stage_root = Path(source_root).resolve(), Path(stage_root).resolve()
    if not stage_root.is_dir():
        raise ValueError("release stage root must already exist")
    if (source_root / ".git").exists():
        assert_no_hidden_git_index_flags(source_root)
    document, categories = build_release_profile(
        source_root, profile, authorization_path=authorization_path
    )
    for path in sorted(
        set().union(*categories.values()),
        key=lambda value: _relative(source_root, value, label="release artifact"),
    ):
        _copy_file(source_root, stage_root, path)
    marker = stage_root / PROFILE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    marker.chmod(0o644)
    return document


def materialize_claim_audit(stage_root: str | Path, profile: str) -> dict[str, Any]:
    """Run the fixed claim gate and bind every scanned document into the marker."""
    stage_root = Path(stage_root).resolve()
    if profile not in RELEASE_PROFILES:
        raise ValueError(f"unknown release profile: {profile}")
    validator = stage_root / "scripts" / "26_validate_claims.py"
    registry = stage_root / "protocols" / "route_a_claim_registry_v1.json"
    marker_path = stage_root / PROFILE_MARKER
    if not validator.is_file() or not registry.is_file() or not marker_path.is_file():
        raise ValueError("claim audit requires staged validator, registry and profile marker")
    marker = _load_json(marker_path, label="release profile marker")
    if marker.get("profile") != profile:
        raise ValueError("claim audit profile differs from release marker")
    # Never execute a staged validator until the outer verifier has proved its
    # exact blob and every other protected source/control byte from the bundle.
    _verify_git_history_evidence(stage_root, marker, profile)
    command = [
        sys.executable, str(validator), "--root", str(stage_root),
        "--registry", str(registry),
    ]
    require_complete = profile == POSTOPEN_PROFILE
    if require_complete:
        command.append("--require-complete")
    result = subprocess.run(
        command,
        cwd=stage_root,
        env={
            "PATH": os.defpath,
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"Route-A claim gate failed: {detail[-8000:]}")
    registry_document = _load_json(registry, label="claim registry")
    raw_patterns = registry_document.get("documents")
    if not isinstance(raw_patterns, list) or not all(
        isinstance(pattern, str) for pattern in raw_patterns
    ):
        raise ValueError("claim registry document patterns are malformed")
    patterns: list[str] = list(raw_patterns)
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ValueError("claim registry document patterns are malformed")
    scanned = sorted(
        path for path in stage_root.rglob("*")
        if path.is_file()
        and any(
            fnmatch(path.relative_to(stage_root).as_posix(), pattern)
            for pattern in patterns
        )
    )
    if not scanned:
        raise ValueError("claim audit selected no release documents")
    audit = {
        "format": "thermoroute.route-a-release-claim-audit.v1",
        "profile": profile,
        "require_complete": require_complete,
        "validator": _binding_for(stage_root, validator),
        "registry": _binding_for(stage_root, registry),
        "scanned_documents": [_binding_for(stage_root, path) for path in scanned],
        "violation_count": 0,
        "validator_stdout": result.stdout.strip(),
    }
    audit_path = stage_root / CLAIM_AUDIT_PATH
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    marker["claim_validation"] = _binding_for(stage_root, audit_path)
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return audit


def _verify_claim_audit(
    root: Path,
    marker: Mapping[str, Any],
    profile: str,
    *,
    execute_validator: bool,
) -> None:
    audit_path = _add_binding(
        root, {}, "claim_validation", marker.get("claim_validation"),
        label="release claim audit",
    )
    if _relative(root, audit_path, label="claim audit") != CLAIM_AUDIT_PATH:
        raise ValueError("claim audit leaves its canonical evidence path")
    audit = _load_json(audit_path, label="release claim audit")
    expected_complete = profile == POSTOPEN_PROFILE
    if (
        audit.get("format") != "thermoroute.route-a-release-claim-audit.v1"
        or audit.get("profile") != profile
        or audit.get("require_complete") is not expected_complete
        or audit.get("violation_count") != 0
    ):
        raise ValueError("claim audit status/profile is inconsistent")
    validator = _add_binding(
        root, {}, "claim_validation", audit.get("validator"),
        label="claim validator",
    )
    registry = _add_binding(
        root, {}, "claim_validation", audit.get("registry"),
        label="claim registry",
    )
    if (
        _relative(root, validator, label="claim validator")
        != "scripts/26_validate_claims.py"
        or _relative(root, registry, label="claim registry")
        != "protocols/route_a_claim_registry_v1.json"
    ):
        raise ValueError("claim audit uses a noncanonical validator/registry")
    registry_document = _load_json(registry, label="claim registry")
    raw_patterns = registry_document.get("documents")
    if not isinstance(raw_patterns, list) or not all(
        isinstance(pattern, str) for pattern in raw_patterns
    ):
        raise ValueError("claim registry document patterns are malformed")
    patterns: list[str] = list(raw_patterns)
    selected = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and any(
            fnmatch(path.relative_to(root).as_posix(), pattern)
            for pattern in patterns
        )
    )
    expected_documents = [_binding_for(root, path) for path in selected]
    if audit.get("scanned_documents") != expected_documents:
        raise ValueError("claim audit document registry/checksums changed")
    if not execute_validator:
        return
    command = [
        sys.executable, str(validator), "--root", str(root),
        "--registry", str(registry),
    ]
    if expected_complete:
        command.append("--require-complete")
    result = subprocess.run(
        command, cwd=root,
        env={
            "PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"archived Route-A claim gate failed: {(result.stderr or result.stdout)[-8000:]}")


def materialize_git_history_evidence(
    source_root: str | Path, stage_root: str | Path, profile: str
) -> dict[str, Any]:
    """Create a self-contained bundle and bind original sealed protocol bytes."""
    source_root, stage_root = Path(source_root).resolve(), Path(stage_root).resolve()
    if profile not in RELEASE_PROFILES:
        raise ValueError(f"unknown release profile: {profile}")
    marker_path = stage_root / PROFILE_MARKER
    marker = _load_json(marker_path, label="release profile marker")
    if marker.get("profile") != profile:
        raise ValueError("Git evidence profile differs from release marker")
    _assert_safe_git_repository(source_root)
    assert_no_hidden_git_index_flags(source_root)
    if profile == PREOPEN_PROFILE:
        dirt = _run_git(
            source_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            text=True,
        )
        dirty_paths = [line for line in dirt.stdout.splitlines() if line]
        if dirt.returncode or dirty_paths:
            raise ValueError(
                "formal pre-opening release requires a clean Git worktree: "
                f"{dirty_paths[:10]}"
            )
    head = _run_git(source_root, "rev-parse", "HEAD", text=True)
    if head.returncode:
        raise ValueError("cannot resolve manuscript Git HEAD for release bundle")
    bundle = stage_root / GIT_BUNDLE_PATH
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        raise ValueError("refusing to replace staged Git history evidence")
    result = _run_git(source_root, "bundle", "create", str(bundle), "HEAD", text=True)
    if result.returncode or not bundle.is_file():
        raise ValueError(f"cannot create release Git bundle: {(result.stderr or result.stdout).strip()}")
    protocol = _load_json(
        stage_root / "protocols" / "route_a_confirmatory_v1.json",
        label="Route-A protocol",
    )
    seal_path, seal = _load_protocol_seal(stage_root, protocol)
    authoritative = str(protocol.get("authoritative_protocol_commit", ""))
    protocol_markdown = "protocols/route_a_confirmatory_protocol.md"
    original_binding = seal.get("original_preregistration", {}).get("markdown", {})
    final_section = seal.get("final_prelabel_protocol", {})
    final_commit = str(final_section.get("commit", ""))
    original = _run_git(source_root, "show", f"{authoritative}:{protocol_markdown}")
    if original.returncode:
        raise ValueError("cannot recover original sealed protocol from Git history")
    original_sha = hashlib.sha256(original.stdout).hexdigest()
    if (
        not isinstance(original_binding, Mapping)
        or original_binding.get("path") != protocol_markdown
        or original_binding.get("sha256") != original_sha
    ):
        raise ValueError("original protocol Git blob differs from final protocol seal")
    if profile == POSTOPEN_PROFILE:
        authorization = _load_json(
            stage_root / str(marker.get("authorization", {}).get("path", "")),
            label="opening authorization",
        )
        expected_sha = authorization.get("protocol", {}).get(
            "authoritative_markdown_sha256"
        )
        if expected_sha != original_sha:
            raise ValueError("Git bundle protocol blob differs from opening authorization")
        authorized_protocol = authorization.get("protocol", {})
        authorized_seal = (
            authorized_protocol.get("seal")
            if isinstance(authorized_protocol, Mapping)
            else None
        )
        if (
            not isinstance(authorized_protocol, Mapping)
            or not isinstance(authorized_seal, Mapping)
            or authorized_protocol.get("final_prelabel_commit") != final_commit
            or authorized_seal.get("path") != PROTOCOL_SEAL_PATH
            or authorized_seal.get("sha256") != sha256_file(seal_path)
        ):
            raise ValueError("opening authorization differs from final protocol seal")
        compute = str(marker["authorized_worktree_dirt_policy"]["compute_commit"])
        manuscript = str(marker["authorized_worktree_dirt_policy"]["manuscript_commit"])
        chronology_path, chronology = _validate_prelabel_chronology_structure(
            stage_root, {}, authorization
        )
        chronology_relative = _relative(
            stage_root, chronology_path, label="prelabel chronology"
        )
        added_commits = _git_path_creation_commits(
            source_root, compute, chronology_relative
        )
        if len(added_commits) != 1:
            raise ValueError(
                "prelabel chronology receipt was not added exactly once before authorization"
            )
        chronology_receipt_commit = added_commits[0]
        receipt_blob = _run_git(
            source_root, "show", f"{chronology_receipt_commit}:{chronology_relative}"
        )
        receipt_oid = _run_git(
            source_root,
            "rev-parse",
            f"{chronology_receipt_commit}:{chronology_relative}",
            text=True,
        )
        if (
            receipt_blob.returncode
            or receipt_oid.returncode
            or receipt_blob.stdout != chronology_path.read_bytes()
        ):
            raise ValueError("prelabel chronology receipt cannot be replayed from Git")
        chronology_evidence: dict[str, Any] | None = {
            "receipt": _binding_for(stage_root, chronology_path),
            "receipt_commit": chronology_receipt_commit,
            "receipt_git_blob_oid": receipt_oid.stdout.strip(),
            "order": dict(chronology["order"]),
            "model_source_control_artifact_count": len(
                chronology["model_source_control_artifacts"]
            ),
            "model_freeze_artifact_count": len(
                chronology["model_freeze_artifacts"]
            ),
            "input_evidence_artifact_count": len(
                chronology["input_evidence_artifacts"]
            ),
        }
    else:
        compute = head.stdout.strip()
        manuscript = compute
        chronology_evidence = None
    for ancestor, descendant, label in (
        (authoritative, final_commit, "original-to-final protocol"),
        (final_commit, compute, "final protocol-to-compute"),
        (compute, manuscript, "compute-to-manuscript"),
    ):
        relation = _run_git(
            source_root, "merge-base", "--is-ancestor", ancestor, descendant
        )
        if relation.returncode:
            raise ValueError(f"release Git chronology failed: {label}")
    if chronology_evidence is not None:
        order = chronology_evidence["order"]
        strict_relations = (
            (final_commit, order["model_freeze_commit"], "final-to-model-freeze"),
            (
                order["model_freeze_commit"],
                order["input_evidence_commit"],
                "model-freeze-to-input-evidence",
            ),
            (
                order["input_evidence_commit"],
                order["receipt_creation_base_commit"],
                "input-evidence-to-receipt-base",
            ),
            (
                order["receipt_creation_base_commit"],
                chronology_evidence["receipt_commit"],
                "receipt-base-to-receipt-commit",
            ),
        )
        for ancestor, descendant, label in strict_relations:
            relation = _run_git(
                source_root, "merge-base", "--is-ancestor", ancestor, descendant
            )
            if ancestor == descendant or relation.returncode:
                raise ValueError(f"release prelabel chronology failed: {label}")
        relation = _run_git(
            source_root,
            "merge-base",
            "--is-ancestor",
            str(chronology_evidence["receipt_commit"]),
            compute,
        )
        if relation.returncode:
            raise ValueError("chronology receipt commit is later than authorization compute")
    final_blobs = []
    for key in ("json", "markdown"):
        binding = final_section.get(key)
        if not isinstance(binding, Mapping):
            raise ValueError("final protocol seal lacks a blob binding")
        relative = str(binding.get("path", ""))
        blob = _run_git(source_root, "show", f"{final_commit}:{relative}")
        staged_path = _resolve_release_path(
            stage_root, relative, label=f"final protocol {key}"
        )
        digest = hashlib.sha256(blob.stdout).hexdigest()
        if (
            blob.returncode
            or digest != binding.get("sha256")
            or blob.stdout != staged_path.read_bytes()
        ):
            raise ValueError(f"final protocol {key} cannot be replayed from Git")
        final_blobs.append({
            "commit": final_commit,
            "path": relative,
            "sha256": digest,
            "bytes": len(blob.stdout),
        })
    evidence = {
        "format": "thermoroute.route-a-git-history-evidence.v1",
        "profile": profile,
        "bundle": _binding_for(stage_root, bundle),
        "compute_commit": compute,
        "manuscript_commit": manuscript,
        "authoritative_protocol_commit": authoritative,
        "final_prelabel_protocol_commit": final_commit,
        "protocol_seal": _binding_for(stage_root, seal_path),
        "sealed_protocol_blob": {
            "commit": authoritative,
            "path": protocol_markdown,
            "sha256": original_sha,
            "bytes": len(original.stdout),
        },
        "final_protocol_blobs": final_blobs,
        "external_timestamp_or_public_preregistration": False,
    }
    if chronology_evidence is not None:
        evidence["prelabel_chronology"] = chronology_evidence
    marker["git_history_evidence"] = evidence
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    # This is a build-time trust boundary: the staged tree may execute Python
    # only after an independent replay of the just-created bundle proves that
    # every protected source/control byte is the committed byte.
    _verify_git_history_evidence(stage_root, marker, profile)
    return evidence


def _verify_protected_tree_from_bundle(
    *, root: Path, bare: Path, commit: str
) -> None:
    """Require archive protected source/control bytes to equal one Git tree."""
    tree = _run_git(bare, "ls-tree", "-r", "-z", commit)
    if tree.returncode:
        raise ValueError("cannot enumerate protected source/control Git tree")
    try:
        entries: dict[str, tuple[str, str]] = {}
        for record in tree.stdout.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, _oid = metadata.decode("ascii", errors="strict").split()
            relative = raw_path.decode("utf-8", errors="strict")
            if _is_model_control_path(relative):
                entries[relative] = (mode, object_type)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("protected Git tree contains a malformed entry/path") from exc
    unsafe_entries = {
        relative: value
        for relative, value in entries.items()
        if value[1] != "blob" or value[0] not in {"100644", "100755"}
    }
    if unsafe_entries:
        raise ValueError(
            "protected Git tree contains symlink/submodule/non-file entries: "
            f"{list(unsafe_entries.items())[:10]}"
        )
    expected = set(entries)
    archived = _working_model_control_paths(root)
    if archived != expected:
        missing = sorted(expected - archived)
        extra = sorted(archived - expected)
        raise ValueError(
            "archive protected source/control path set differs from compute Git tree: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for relative in sorted(expected):
        current = root / relative
        if current.is_symlink() or not current.is_file():
            raise ValueError(f"archive protected path is not a regular file: {relative}")
        blob = _run_git(bare, "show", f"{commit}:{relative}")
        if blob.returncode or blob.stdout != current.read_bytes():
            raise ValueError(
                "archive protected source/control differs from compute Git blob: "
                f"{relative}"
            )


def _verify_authorized_compute_tree_from_bundle(
    *,
    root: Path,
    bare: Path,
    authorization: Mapping[str, Any],
    compute_commit: str,
) -> None:
    """Bind authorization, compute Git blobs, and archived bytes exactly."""
    source = authorization.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("authorization lacks its frozen source identity")
    if source.get("git_commit_before_authorization") != compute_commit:
        raise ValueError("authorization compute commit differs from Git evidence")
    inventory = source.get("source_inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("authorization lacks its exact source inventory")
    frozen_inventory: dict[str, str] = {}
    for relative, digest in inventory.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or not _matches_source_inventory(relative)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("authorization source inventory is malformed")
        frozen_inventory[relative] = digest
    if source.get("source_tree_sha256") != _sha256_json(
        dict(sorted(frozen_inventory.items()))
    ):
        raise ValueError("authorization source-tree digest is inconsistent")

    tree = _run_git(bare, "ls-tree", "-r", "--name-only", "-z", compute_commit)
    if tree.returncode:
        raise ValueError("cannot enumerate the authorized compute Git tree")
    try:
        git_paths = {
            item.decode("utf-8", errors="strict")
            for item in tree.stdout.split(b"\0") if item
        }
    except UnicodeDecodeError as exc:
        raise ValueError("compute Git tree contains a non-UTF-8 path") from exc
    expected_paths = {path for path in git_paths if _matches_source_inventory(path)}
    archived_paths = _working_source_inventory_paths(root)
    if set(frozen_inventory) != expected_paths:
        raise ValueError(
            "authorization source inventory path set differs from compute Git tree"
        )
    if archived_paths != expected_paths:
        raise ValueError(
            "archive source inventory path set differs from compute Git tree"
        )

    def verify_blob(relative: str, digest: str, *, label: str) -> None:
        current = _resolve_release_path(root, relative, label=label)
        if not current.is_file() or sha256_file(current) != digest:
            raise ValueError(f"{label} differs from authorization: {relative}")
        blob = _run_git(bare, "show", f"{compute_commit}:{relative}")
        if (
            blob.returncode
            or hashlib.sha256(blob.stdout).hexdigest() != digest
            or blob.stdout != current.read_bytes()
        ):
            raise ValueError(
                f"{label} differs across authorization, compute Git blob, and archive: "
                f"{relative}"
            )

    for relative, digest in sorted(frozen_inventory.items()):
        verify_blob(relative, digest, label="authorized source")

    fixed_code = authorization.get("fixed_code")
    if (
        not isinstance(fixed_code, Mapping)
        or set(fixed_code) != {
            "format", "modules", "files", "entrypoints", "sha256"
        }
        or fixed_code.get("format") != "thermoroute.route-a-fixed-code.v1"
    ):
        raise ValueError("authorization fixed-code identity is malformed")
    stable_fixed = {
        group: fixed_code.get(group)
        for group in ("modules", "files", "entrypoints")
    }
    if fixed_code.get("sha256") != _sha256_json(stable_fixed):
        raise ValueError("authorization fixed-code digest is inconsistent")
    expected_fixed = {
        "modules": {
            "thermoroute.opening": "src/thermoroute/opening.py",
            "thermoroute.model_suite": "src/thermoroute/model_suite.py",
            "thermoroute.frozen_inference": "src/thermoroute/frozen_inference.py",
            "thermoroute.datasets": "src/thermoroute/datasets.py",
            "thermoroute.provenance": "src/thermoroute/provenance.py",
            "thermoroute.usgs": "src/thermoroute/usgs.py",
            "thermoroute.inference_gate": "src/thermoroute/inference_gate.py",
            "thermoroute.outcome_qc": "src/thermoroute/outcome_qc.py",
            "thermoroute.quantiles": "src/thermoroute/quantiles.py",
            "thermoroute.coverage_audit": (
                "src/thermoroute/coverage_audit.py"
            ),
            "thermoroute.coverage_bridge": (
                "src/thermoroute/coverage_bridge.py"
            ),
            "thermoroute.repro": "src/thermoroute/repro.py",
        },
        "files": {
            "src/thermoroute/opening_contract.py": (
                "src/thermoroute/opening_contract.py"
            ),
            "src/thermoroute/outcome_acquisition.py": (
                "src/thermoroute/outcome_acquisition.py"
            ),
        },
        "entrypoints": {
            "orchestrator": "scripts/route_a_opening_orchestrator.py",
            "acquisition": "scripts/route_a_outcome_acquisition.py",
            "trusted_scorer": "scripts/route_a_trusted_scorer.py",
        },
    }
    fixed_paths: set[str] = set()
    for group in ("modules", "files", "entrypoints"):
        values = fixed_code.get(group)
        if not isinstance(values, Mapping) or set(values) != set(expected_fixed[group]):
            raise ValueError(f"authorization fixed-code {group} registry is malformed")
        for name, binding in values.items():
            if (
                not isinstance(name, str)
                or not isinstance(binding, Mapping)
                or set(binding) != {"path", "realpath", "sha256"}
                or not isinstance(binding.get("realpath"), str)
            ):
                raise ValueError(f"authorization fixed-code {group} binding is malformed")
            relative = binding.get("path")
            digest = binding.get("sha256")
            if (
                not isinstance(relative, str)
                or relative != expected_fixed[group][name]
                or relative not in frozen_inventory
                or not isinstance(digest, str)
                or digest != frozen_inventory[relative]
                or relative in fixed_paths
            ):
                raise ValueError(
                    f"authorization fixed-code {group} binding leaves frozen source"
                )
            fixed_paths.add(relative)
            verify_blob(relative, digest, label=f"authorized fixed-code {group}")

    runtime = authorization.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("authorization runtime identity is malformed")
    for key, canonical in (
        ("requirements_lock", "requirements-lock.txt"),
        ("hashed_requirements_lock", REPRODUCIBILITY_LOCK),
    ):
        binding = runtime.get(key)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != canonical
            or not isinstance(binding.get("sha256"), str)
            or binding.get("sha256") != frozen_inventory.get(canonical)
        ):
            raise ValueError(f"authorization {key} binding is inconsistent")
        verify_blob(
            canonical,
            str(binding["sha256"]),
            label=f"authorized {key}",
        )

    amendment = authorization.get("inference_amendment")
    if not isinstance(amendment, Mapping):
        raise ValueError("authorization lacks inference amendment Git lineage")
    amendment_relative = _git_declared_binding_path(
        bare,
        compute_commit,
        amendment,
        label="authorized inference amendment",
    )
    seal_relative = _git_declared_binding_path(
        bare,
        compute_commit,
        amendment.get("seal"),
        label="authorized inference amendment seal",
    )
    gate_relative = _git_declared_binding_path(
        bare,
        compute_commit,
        authorization.get("inference_gate"),
        label="authorized inference gate",
    )
    if (
        amendment_relative != "protocols/route_a_inference_amendment_v1.json"
        or seal_relative
        != "protocols/route_a_inference_amendment_seal_v1.json"
        or gate_relative != "outputs/prelabel/route_a_inference_gate_v1.json"
    ):
        raise ValueError("authorized inference Git artifacts are noncanonical")
    amendment_commit = str(amendment.get("final_prelabel_commit", ""))
    if (
        not re.fullmatch(r"[0-9a-f]{40}", amendment_commit)
        or _run_git(
            bare, "merge-base", "--is-ancestor", amendment_commit, compute_commit
        ).returncode
    ):
        raise ValueError("inference amendment commit is not a compute ancestor")
    protocol = authorization.get("protocol")
    final_protocol_commit = (
        str(protocol.get("final_prelabel_commit", ""))
        if isinstance(protocol, Mapping)
        else ""
    )
    if (
        not re.fullmatch(r"[0-9a-f]{40}", final_protocol_commit)
        or _run_git(
            bare,
            "merge-base",
            "--is-ancestor",
            final_protocol_commit,
            amendment_commit,
        ).returncode
    ):
        raise ValueError("base protocol commit is not an inference-amendment ancestor")
    sealed_blob = _run_git(
        bare, "show", f"{amendment_commit}:{amendment_relative}"
    )
    if (
        sealed_blob.returncode
        or hashlib.sha256(sealed_blob.stdout).hexdigest() != amendment.get("sha256")
    ):
        raise ValueError("sealed inference-amendment Git blob changed")


def _normalise_git_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is empty or malformed")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} path is not canonical: {value!r}")
    return value


def _git_json_document(
    bare: Path, commit: str, relative: str, *, label: str
) -> dict[str, Any]:
    blob = _run_git(bare, "show", f"{commit}:{relative}")
    try:
        value = json.loads(blob.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label} from Git") from exc
    if blob.returncode or not isinstance(value, dict):
        raise ValueError(f"cannot replay {label} from Git")
    return value


def _git_declared_binding_path(
    bare: Path,
    commit: str,
    value: object,
    *,
    label: str,
    base: str | None = None,
) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is malformed")
    raw = _normalise_git_relative(value.get("path"), label=label)
    if base is None:
        relative = raw
    else:
        relative = (PurePosixPath(base).parent / raw).as_posix()
        relative = _normalise_git_relative(relative, label=label)
    digest = value.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} lacks an exact SHA-256")
    blob = _run_git(bare, "show", f"{commit}:{relative}")
    if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != digest:
        raise ValueError(f"{label} Git blob differs from its nested binding")
    return relative


def _prediction_dependency_paths(
    bare: Path, commit: str, value: object, *, label: str
) -> set[str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} prediction lineage is malformed")
    artifact = value.get("artifact")
    path = _git_declared_binding_path(
        bare, commit, artifact, label=f"{label} prediction"
    )
    assert isinstance(artifact, Mapping)
    sidecar = _git_declared_binding_path(
        bare,
        commit,
        artifact.get("sidecar"),
        label=f"{label} prediction sidecar",
    )
    return {path, sidecar}


def _validate_git_lightgbm_quantile_contract(manifest: Mapping[str, Any]) -> None:
    repair = manifest.get("quantile_repair")
    if repair != {
        "method": "median_preserving_endpoint_clip_v1",
        "version": 1,
        "nominal_head_levels": {"q05": 0.05, "q50": 0.50, "q95": 0.95},
        "q05_operation": "minimum(raw_q05,raw_q50)",
        "q50_operation": "raw_q50_unchanged",
        "q95_operation": "maximum(raw_q95,raw_q50)",
        "nominal_median_preserved_exactly": True,
    }:
        raise ValueError("Git LightGBM quantile-head identity contract changed")
    members = manifest.get("members")
    horizons = manifest.get("horizons")
    audit = manifest.get("raw_quantile_crossing_audit")
    if (
        not isinstance(members, list)
        or not members
        or len(members) != len(set(members))
        or manifest.get("member_count") != len(members)
        or not isinstance(horizons, list)
        or not horizons
        or len(horizons) != len(set(horizons))
        or not isinstance(audit, Mapping)
        or set(audit)
        != {
            "format", "scope", "key_columns", "repair_method", "members",
            "audit_sha256",
        }
        or audit.get("format")
        != "thermoroute.raw-quantile-crossing-audit.v1"
        or audit.get("scope") != "development_export_rows_before_repair"
        or audit.get("key_columns")
        != ["site_id", "horizon", "split", "issue_date", "target_date"]
        or audit.get("repair_method") != "median_preserving_endpoint_clip_v1"
    ):
        raise ValueError("Git LightGBM raw-crossing audit registry changed")
    stable = {key: value for key, value in audit.items() if key != "audit_sha256"}
    member_audits = audit.get("members")
    expected_horizons = {str(int(value)) for value in horizons}
    if (
        audit.get("audit_sha256") != _sha256_json(stable)
        or not isinstance(member_audits, Mapping)
        or set(member_audits) != set(members)
    ):
        raise ValueError("Git LightGBM raw-crossing audit self hash changed")
    horizon_identity: dict[str, tuple[int, str]] = {}
    fields = {
        "rows", "forecast_key_sha256", "raw_prediction_sha256",
        "q05_above_q50_count", "q50_above_q95_count", "any_crossing_count",
        "any_crossing_rate", "maximum_crossing_gap_c",
    }
    for member in members:
        values = member_audits[member]
        if not isinstance(values, Mapping) or set(values) != expected_horizons:
            raise ValueError("Git LightGBM raw-crossing horizons changed")
        for horizon, summary in values.items():
            if not isinstance(summary, Mapping) or set(summary) != fields:
                raise ValueError("Git LightGBM raw-crossing summary schema changed")
            integers = [
                summary.get("rows"), summary.get("q05_above_q50_count"),
                summary.get("q50_above_q95_count"),
                summary.get("any_crossing_count"),
            ]
            if any(type(value) is not int for value in integers):
                raise ValueError("Git LightGBM raw-crossing counts are not integers")
            rows, lower, upper, crossing = (int(value) for value in integers)
            rate = summary.get("any_crossing_rate")
            gap = summary.get("maximum_crossing_gap_c")
            if (
                rows < 1
                or min(lower, upper, crossing) < 0
                or max(lower, upper, crossing) > rows
                or crossing < max(lower, upper)
                or crossing > lower + upper
                or isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or float(rate) != crossing / rows
                or isinstance(gap, bool)
                or not isinstance(gap, (int, float))
                or not math.isfinite(float(gap))
                or float(gap) < 0.0
                or (crossing == 0) != (float(gap) == 0.0)
                or any(
                    not re.fullmatch(r"[0-9a-f]{64}", str(summary.get(field, "")))
                    for field in ("forecast_key_sha256", "raw_prediction_sha256")
                )
            ):
                raise ValueError("Git LightGBM raw-crossing summary is inconsistent")
            identity = (rows, str(summary["forecast_key_sha256"]))
            if horizon in horizon_identity and horizon_identity[horizon] != identity:
                raise ValueError("Git LightGBM raw-crossing keys differ by member")
            horizon_identity[horizon] = identity


def _git_preopening_gate_dependency_paths(
    bare: Path, commit: str, suite: Mapping[str, Any]
) -> set[str]:
    gates = suite.get("preopening_gates")
    expected_gates = {
        "stage09_completion": (
            "outputs/models/route_a_stage09_completion.json",
            "thermoroute.stage09-completion-receipt.v1",
            "PASS_FORMAL_STAGE09_COMPLETE",
        ),
        "stage09b_development_controls": (
            "outputs/models/route_a_stage09b_completion.json",
            "thermoroute.stage09b-completion-receipt.v3",
            "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
        ),
    }
    if not isinstance(gates, Mapping) or set(gates) != set(expected_gates):
        raise ValueError("Git model suite lacks exact Stage-09/09b completion gates")
    output: set[str] = set()
    for gate_name, (
        expected_receipt_path, expected_format, expected_status,
    ) in expected_gates.items():
        gate_binding = gates[gate_name]
        if not isinstance(gate_binding, Mapping) or set(gate_binding) != {
            "path", "sha256"
        }:
            raise ValueError(f"Git {gate_name} binding is not exact")
        receipt_path = _git_declared_binding_path(
            bare, commit, gate_binding, label=f"Git {gate_name}",
        )
        if receipt_path != expected_receipt_path:
            raise ValueError(f"Git {gate_name} receipt path is noncanonical")
        output.add(receipt_path)
        receipt = _git_json_document(
            bare, commit, receipt_path, label=f"Git {gate_name}",
        )
        stable = dict(receipt)
        self_hash = stable.pop("receipt_self_sha256", None)
        if (
            receipt.get("format") != expected_format
            or receipt.get("status") != expected_status
            or self_hash != _sha256_json(stable)
        ):
            raise ValueError(f"Git {gate_name} receipt changed")
        run_id = receipt.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Git {gate_name} run identity changed")
        if gate_name == "stage09_completion":
            expected_receipt_keys = {
                "format", "status", "stage", "run_id", "run_identity",
                "formal_configuration", "confirmation_outcomes_requested_or_read",
                "artifacts", "receipt_self_sha256",
            }
            expected_artifact_paths = {
                "run_manifest": f"outputs/runs/09_usgs_experiment/{run_id}/run.json",
                "predictions": "outputs/predictions/usgs_predictions_stage9_v2.parquet",
                "prediction_sidecar": (
                    "outputs/predictions/usgs_predictions_stage9_v2.parquet.meta.json"
                ),
                "scores": "outputs/tables/usgs_scores.csv",
                "report": "outputs/reports/usgs_experiment.md",
                "lightgbm_selection": (
                    "outputs/tables/lightgbm_joint_validation_selection.csv"
                ),
                "thermoroute_pointer": "outputs/models/thermoroute_usgs_bundle.json",
                "lightgbm_pointer": "outputs/models/lightgbm_usgs_bundle.json",
                "components_pointer": "outputs/models/route_a_stage9_components.json",
            }
            if (
                set(receipt) != expected_receipt_keys
                or receipt.get("stage") != "09_usgs_experiment"
                or receipt.get("confirmation_outcomes_requested_or_read") is not False
                or not isinstance(receipt.get("run_identity"), Mapping)
                or not isinstance(receipt.get("formal_configuration"), Mapping)
            ):
                raise ValueError("Git Stage-09 receipt contract changed")
        else:
            expected_receipt_keys = {
                "format", "status", "stage", "run_id", "run_identity",
                "formal_configuration", "evidence_scope",
                "training_replay_verified", "matrix_audit", "member_registry",
                "best_model_state_prediction_replay_verified",
                "artifacts", "post_2020_outcomes_requested_or_read",
                "receipt_self_sha256",
            }
        run_dir = f"outputs/runs/09b_development_controls/{run_id}"
        run_tree = _run_git(
            bare, "ls-tree", "-r", "--name-only", commit, "--", run_dir,
        )
        if run_tree.returncode:
            raise ValueError("cannot enumerate Git Stage-09b run directory")
        for raw_path in run_tree.stdout.decode("utf-8").splitlines():
            name = PurePosixPath(raw_path).name
            if (
                (name.startswith(".") and name.endswith(".tmp"))
                or name.endswith(".recovery-probe")
            ):
                raise ValueError("Git Stage-09b run retains an unbound transaction temp")
            expected_artifact_paths = {
                "run_manifest": f"{run_dir}/run.json",
                "frozen_panel_spec": "data_usgs/frozen_panel_v1.json",
                "panel": "data_usgs/panel_usgs_120v2.parquet",
                "registry": "data_usgs/station_registry_v1.csv",
                "predictor_bridge": "data_usgs/development_predictor_bridge_v1.json",
                "predictions": (
                    f"{run_dir}/development_controls_predictions.parquet"
                ),
                "prediction_sidecar": (
                    f"{run_dir}/development_controls_predictions.parquet.meta.json"
                ),
                "architecture_budget": (
                    f"{run_dir}/development_controls_architecture_budget.csv"
                ),
                "architecture_budget_sidecar": (
                    f"{run_dir}/development_controls_architecture_budget.csv.meta.json"
                ),
                "metric_summary": (
                    f"{run_dir}/development_controls_metric_summary.csv"
                ),
                "metric_summary_sidecar": (
                    f"{run_dir}/development_controls_metric_summary.csv.meta.json"
                ),
                "report": f"{run_dir}/development_controls_report.md",
                "report_sidecar": (
                    f"{run_dir}/development_controls_report.md.meta.json"
                ),
                "semantic_audit": (
                    f"{run_dir}/development_controls_semantic_audit.json"
                ),
                "semantic_audit_sidecar": (
                    f"{run_dir}/development_controls_semantic_audit.json.meta.json"
                ),
            }
            audit = receipt.get("matrix_audit")
            common_keys = audit.get("common_forecast_keys") if isinstance(
                audit, Mapping
            ) else None
            if (
                set(receipt) != expected_receipt_keys
                or receipt.get("stage") != "09b_development_controls"
                or receipt.get("evidence_scope")
                != "best_model_state_prediction_replay"
                or receipt.get("training_replay_verified") is not False
                or receipt.get("best_model_state_prediction_replay_verified") is not True
                or receipt.get("post_2020_outcomes_requested_or_read") is not False
                or not isinstance(receipt.get("run_identity"), Mapping)
                or not isinstance(receipt.get("formal_configuration"), Mapping)
                or not isinstance(audit, Mapping)
                or set(audit) != {
                    "expected_members", "prediction_rows", "common_forecast_keys",
                    "splits", "reference_member",
                }
                or audit.get("expected_members") != 31
                or type(common_keys) is not int
                or common_keys < 1
                or audit.get("prediction_rows") != 31 * common_keys
                or audit.get("splits") != ["calib", "test", "val"]
                or audit.get("reference_member") != "PlainMLP-7var/seed0"
            ):
                raise ValueError("Git Stage-09b receipt contract changed")
        artifacts = receipt.get("artifacts")
        if (
            not isinstance(artifacts, Mapping)
            or set(artifacts) != set(expected_artifact_paths)
        ):
            raise ValueError(f"Git {gate_name} artifact registry changed")
        resolved: dict[str, str] = {}
        descriptors: dict[str, dict[str, object]] = {}
        for label, binding in artifacts.items():
            if not isinstance(binding, Mapping) or set(binding) != {
                "path", "sha256"
            }:
                raise ValueError(f"Git {gate_name} {label} binding is not exact")
            resolved[str(label)] = _git_declared_binding_path(
                bare, commit, binding, label=f"Git {gate_name} {label}",
            )
            output.add(resolved[str(label)])
            blob = _run_git(bare, "show", f"{commit}:{resolved[str(label)]}")
            if blob.returncode:
                raise ValueError(f"cannot replay Git {gate_name} {label}")
            descriptors[str(label)] = {
                "sha256": hashlib.sha256(blob.stdout).hexdigest(),
                "bytes": len(blob.stdout),
            }
        if resolved != expected_artifact_paths:
            raise ValueError(f"Git {gate_name} artifact paths are noncanonical")
        if gate_name != "stage09b_development_controls":
            continue
        members = receipt.get("member_registry")
        expected_members = _stage09b_release_members()
        if not isinstance(members, list) or len(members) != len(expected_members):
            raise ValueError("Git Stage-09b member registry changed")
        member_descriptors: dict[
            tuple[str, int], tuple[
                dict[str, object], dict[str, object],
                dict[str, object], dict[str, object],
            ]
        ] = {}
        for member, (arm_id, seed) in zip(members, expected_members):
            if (
                not isinstance(member, Mapping)
                or set(member) != {
                    "arm_id", "seed", "checkpoint", "checkpoint_sidecar",
                    "prediction", "prediction_sidecar",
                }
                or (member.get("arm_id"), member.get("seed")) != (arm_id, seed)
            ):
                raise ValueError("Git Stage-09b member binding is malformed")
            expected_prediction = (
                f"{run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
            )
            expected_checkpoint = f"{run_dir}/checkpoints/{arm_id}/seed{seed}.pt"
            member_output: list[tuple[str, dict[str, object]]] = []
            for label, expected_path in (
                ("prediction", expected_prediction),
                ("prediction_sidecar", f"{expected_prediction}.meta.json"),
                ("checkpoint", expected_checkpoint),
                ("checkpoint_sidecar", f"{expected_checkpoint}.meta.json"),
            ):
                binding = member[label]
                if not isinstance(binding, Mapping) or set(binding) != {
                    "path", "sha256"
                }:
                    raise ValueError("Git Stage-09b member binding is not exact")
                path = _git_declared_binding_path(
                    bare, commit, binding,
                    label=f"Git Stage-09b {arm_id}/seed{seed} {label}",
                )
                if path != expected_path:
                    raise ValueError("Git Stage-09b member path is noncanonical")
                output.add(path)
                blob = _run_git(bare, "show", f"{commit}:{path}")
                if blob.returncode:
                    raise ValueError("cannot replay Git Stage-09b member")
                member_output.append((path, {
                    "sha256": hashlib.sha256(blob.stdout).hexdigest(),
                    "bytes": len(blob.stdout),
                }))
            member_descriptors[(arm_id, seed)] = (
                member_output[0][1], member_output[1][1],
                member_output[2][1], member_output[3][1],
            )
        semantic_path = resolved.get("semantic_audit")
        if semantic_path is None:
            raise ValueError("Git Stage-09b receipt lacks semantic audit")
        semantic = _git_json_document(
            bare, commit, semantic_path, label="Git Stage-09b semantic audit",
        )
        stable_semantic = dict(semantic)
        semantic_hash = stable_semantic.pop("semantic_audit_self_sha256", None)
        expected_semantic_keys = {
            "format", "status", "run_id", "evidence_scope",
            "training_replay_verified", "post_2020_outcomes_requested_or_read",
            "best_model_state_prediction_replay_verified",
            "matrix_audit", "canonical_window_registry", "scientific_summary",
            "members",
            "derived_artifacts", "semantic_audit_self_sha256",
        }
        if (
            set(semantic) != expected_semantic_keys
            or semantic.get("format")
            != "thermoroute.development-controls-semantic-audit.v3"
            or semantic.get("status")
            != "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY"
            or semantic.get("run_id") != run_id
            or semantic.get("evidence_scope")
            != "best_model_state_prediction_replay"
            or semantic.get("training_replay_verified") is not False
            or semantic.get("best_model_state_prediction_replay_verified") is not True
            or semantic.get("post_2020_outcomes_requested_or_read") is not False
            or semantic.get("matrix_audit") != receipt.get("matrix_audit")
            or semantic_hash != _sha256_json(stable_semantic)
        ):
            raise ValueError("Git Stage-09b semantic audit changed")
        _stage09b_validate_scientific_summary_document(
            semantic.get("scientific_summary")
        )
        registry = semantic.get("canonical_window_registry")
        if (
            not isinstance(registry, Mapping)
            or set(registry) != {
                "sha256", "common_forecast_keys", "train_examples_per_epoch",
                "train_registry_sha256",
            }
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(registry.get(field, "")))
                for field in ("sha256", "train_registry_sha256")
            )
            or type(registry.get("common_forecast_keys")) is not int
            or registry["common_forecast_keys"] < 1
            or type(registry.get("train_examples_per_epoch")) is not int
            or registry["train_examples_per_epoch"] < 1
        ):
            raise ValueError("Git Stage-09b canonical-window registry changed")
        semantic_members = semantic.get("members")
        if (
            not isinstance(semantic_members, list)
            or len(semantic_members) != len(expected_members)
        ):
            raise ValueError("Git Stage-09b semantic member registry changed")
        for row, expected_member in zip(semantic_members, expected_members):
            digest = row.get("normalised_prediction_sha256") if isinstance(
                row, Mapping
            ) else None
            (
                prediction_descriptor, sidecar_descriptor,
                checkpoint_descriptor, checkpoint_sidecar_descriptor,
            ) = member_descriptors[expected_member]
            if (
                not isinstance(row, Mapping)
                or set(row) != {
                    "arm_id", "seed", "prediction", "prediction_sidecar",
                    "normalised_prediction_sha256",
                    "checkpoint", "checkpoint_sidecar",
                    "best_model_state_prediction_replay_verified",
                }
                or (row.get("arm_id"), row.get("seed")) != expected_member
                or row.get("prediction") != prediction_descriptor
                or row.get("prediction_sidecar") != sidecar_descriptor
                or row.get("checkpoint") != checkpoint_descriptor
                or row.get("checkpoint_sidecar") != checkpoint_sidecar_descriptor
                or row.get("best_model_state_prediction_replay_verified") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", str(digest or ""))
            ):
                raise ValueError("Git Stage-09b semantic member evidence changed")
        derived_labels = {
            "architecture_budget": (
                "architecture_budget", "architecture_budget_sidecar"
            ),
            "combined_predictions": ("predictions", "prediction_sidecar"),
            "metric_summary": ("metric_summary", "metric_summary_sidecar"),
            "report": ("report", "report_sidecar"),
        }
        derived = semantic.get("derived_artifacts")
        if not isinstance(derived, Mapping) or set(derived) != set(derived_labels):
            raise ValueError("Git Stage-09b derived-artifact registry changed")
        for label, (artifact_label, sidecar_label) in derived_labels.items():
            if derived[label] != {
                "artifact": descriptors[artifact_label],
                "sidecar": descriptors[sidecar_label],
            }:
                raise ValueError("Git Stage-09b derived-artifact evidence changed")
    return output


def _reconstruct_model_dependency_paths(
    bare: Path,
    commit: str,
    *,
    suite_path: str,
    replay_path: str,
) -> set[str]:
    output = {suite_path, replay_path}
    suite = _git_json_document(bare, commit, suite_path, label="model suite")
    if (
        suite.get("format") != "thermoroute.route-a-model-suite.v1"
        or suite.get("status") != "FROZEN_BEFORE_LABEL_OPENING"
    ):
        raise ValueError("Git model suite is not a canonical frozen Route-A suite")
    suite_runtime = suite.get("numerical_runtime_sha256")
    if (
        suite.get("training_device") != "cpu"
        or not isinstance(suite_runtime, str)
        or not re.fullmatch(r"[0-9a-f]{64}", suite_runtime)
    ):
        raise ValueError("Git model suite lacks its exact CPU numerical runtime")
    development = suite.get("development_contract")
    if not isinstance(development, Mapping):
        raise ValueError("Git model suite lacks its development contract")
    development_paths: dict[str, str] = {}
    for name in ("frozen_panel_spec", "panel", "registry"):
        development_paths[name] = _git_declared_binding_path(
            bare,
            commit,
            development.get(name),
            label=f"model-suite development {name}",
        )
        output.add(development_paths[name])
    suite_source_sha = development.get("source_sha256")
    if not isinstance(suite_source_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", suite_source_sha
    ):
        raise ValueError("Git model suite lacks a valid source-tree SHA-256")
    bridge_path = _git_declared_binding_path(
        bare,
        commit,
        development.get("predictor_bridge"),
        label="model-suite development predictor bridge",
    )
    output.add(bridge_path)
    bridge = _git_json_document(
        bare, commit, bridge_path, label="development predictor bridge"
    )
    if (
        bridge.get("format") != "thermoroute.development-predictor-bridge.v1"
        or bridge.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or bridge.get("outcome_values_requested_or_read") is not False
        or not isinstance(bridge.get("source_tree_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", bridge["source_tree_sha256"])
        or bridge.get("panel") != development.get("panel")
        or bridge.get("registry") != development.get("registry")
    ):
        raise ValueError("Git development predictor bridge is stale or not an exact PASS")
    for field in ("normalized", "raw_snapshot_indexes"):
        bindings = bridge.get(field)
        if not isinstance(bindings, Mapping):
            raise ValueError(f"Git development predictor bridge lacks {field}")
        expected = {"frozen", "refreshed"} if field == "normalized" else {
            "daymet", "gridmet", "gridmet_schema"
        }
        if set(bindings) != expected:
            raise ValueError(f"Git development predictor bridge {field} changed")
        for name, binding in bindings.items():
            relative = _git_declared_binding_path(
                bare,
                commit,
                binding,
                label=f"development predictor bridge {field}/{name}",
            )
            output.add(relative)
            if field == "raw_snapshot_indexes":
                if PurePosixPath(relative).name != "snapshot_index_v2.json":
                    raise ValueError(
                        "Git development bridge lacks metadata-byte-bound raw index v2"
                    )
                output |= _snapshot_dependency_paths(
                    bare, commit, relative, require_metadata_binding=True,
                )
    for field in ("report", "request_map"):
        output.add(
            _git_declared_binding_path(
                bare,
                commit,
                bridge.get(field),
                label=f"development predictor bridge {field}",
            )
        )
    output |= _git_preopening_gate_dependency_paths(bare, commit, suite)
    versioned = suite.get("versioned_suite")
    if versioned is not None:
        versioned_path = _git_declared_binding_path(
            bare, commit, versioned, label="versioned model suite"
        )
        output.add(versioned_path)
        versioned_document = _git_json_document(
            bare, commit, versioned_path, label="versioned model suite"
        )
        alias = dict(suite)
        alias.pop("versioned_suite", None)
        if versioned_document != alias:
            raise ValueError("Git versioned model suite differs from its alias")
    cohorts = suite.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {"temporal", "external"}:
        raise ValueError("Git model suite lacks exact temporal/external cohorts")
    learned = 0
    for cohort_name, cohort in cohorts.items():
        entries = cohort.get("models") if isinstance(cohort, Mapping) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Git {cohort_name} model registry is empty")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Git model-suite entry is malformed")
            executor = entry.get("executor")
            if executor == "builtin":
                if "artifact" in entry:
                    raise ValueError("Git builtin model unexpectedly binds an artifact")
                continue
            artifact = entry.get("artifact")
            if not isinstance(artifact, Mapping):
                raise ValueError("Git learned model lacks an artifact binding")
            learned += 1
            if executor == "lightgbm_bundle":
                manifest_path = _git_declared_binding_path(
                    bare, commit, artifact, label="LightGBM manifest"
                )
                output.add(manifest_path)
                manifest = _git_json_document(
                    bare, commit, manifest_path, label="LightGBM manifest"
                )
                if manifest.get("format") != "thermoroute.lightgbm-bundle.v2":
                    raise ValueError("Git LightGBM manifest format changed")
                if (
                    manifest.get("training_device") != "cpu"
                    or manifest.get("runtime_sha256") != suite_runtime
                    or manifest.get("heads")
                    != ["point", "q05", "q50", "q95", "event"]
                ):
                    raise ValueError(
                        "Git LightGBM manifest differs from the suite numerical runtime"
                    )
                _validate_git_lightgbm_quantile_contract(manifest)
                models = manifest.get("models")
                if not isinstance(models, Mapping) or not models:
                    raise ValueError("Git LightGBM manifest has no models")
                count = 0
                for horizons in models.values():
                    if not isinstance(horizons, Mapping):
                        raise ValueError("Git LightGBM horizons are malformed")
                    for heads in horizons.values():
                        if not isinstance(heads, Mapping):
                            raise ValueError("Git LightGBM heads are malformed")
                        for binding in heads.values():
                            output.add(
                                _git_declared_binding_path(
                                    bare,
                                    commit,
                                    binding,
                                    label="LightGBM model",
                                    base=manifest_path,
                                )
                            )
                            count += 1
                if count < 1:
                    raise ValueError("Git LightGBM manifest resolved no model files")
                output |= _prediction_dependency_paths(
                    bare,
                    commit,
                    manifest.get("development_prediction"),
                    label="LightGBM",
                )
            elif executor in {"thermoroute_bundle", "lstm_bundle"}:
                directory = _normalise_git_relative(
                    artifact.get("path"), label="Torch model directory"
                )
                for name, hash_key in (
                    ("metadata.json", "metadata_sha256"),
                    ("weights.pt", "weights_sha256"),
                ):
                    relative = f"{directory}/{name}"
                    output.add(
                        _git_declared_binding_path(
                            bare,
                            commit,
                            {"path": relative, "sha256": artifact.get(hash_key)},
                            label=f"Torch {name}",
                        )
                    )
                metadata_path = f"{directory}/metadata.json"
                metadata = _git_json_document(
                    bare, commit, metadata_path, label="Torch metadata"
                )
                if metadata.get("weights_sha256") != artifact.get("weights_sha256"):
                    raise ValueError("Git Torch metadata binds another weights file")
                if (
                    metadata.get("training_device") != "cpu"
                    or metadata.get("runtime_sha256") != suite_runtime
                ):
                    raise ValueError(
                        "Git Torch metadata differs from the suite numerical runtime"
                    )
                output |= _prediction_dependency_paths(
                    bare,
                    commit,
                    metadata.get("development_prediction"),
                    label=str(entry.get("model_id", executor)),
                )
            else:
                raise ValueError(f"unsafe executor in Git model suite: {executor}")
    if learned < 1:
        raise ValueError("Git model suite contains no learned artifact")
    replay = _git_json_document(bare, commit, replay_path, label="development replay")
    if replay.get("format") != "thermoroute.route-a-development-replay.v1":
        raise ValueError("Git development replay format changed")
    replay_suite = replay.get("suite")
    if (
        not isinstance(replay_suite, Mapping)
        or replay_suite.get("path") != suite_path
        or _git_declared_binding_path(
            bare, commit, replay_suite, label="development replay suite"
        ) != suite_path
    ):
        raise ValueError("Git development replay binds another model suite")
    if replay.get("source_tree_sha256") != suite_source_sha:
        raise ValueError("Git development replay and model suite source trees differ")
    if replay.get("runtime_sha256") != suite_runtime:
        raise ValueError("Git development replay and model suite runtimes differ")
    return output


def _snapshot_dependency_paths(
    bare: Path, commit: str, index_path: str, *,
    require_metadata_binding: bool = False,
) -> set[str]:
    index = _git_json_document(bare, commit, index_path, label="snapshot index")
    records = index.get("records")
    if (
        not isinstance(records, list) or not records
        or (
            require_metadata_binding
            and (
                set(index) != {"schema_version", "snapshot_count", "records"}
                or index.get("schema_version") != 2
                or type(index.get("snapshot_count")) is not int
                or index["snapshot_count"] != len(records)
            )
        )
    ):
        raise ValueError(f"Git snapshot index is empty or malformed: {index_path}")
    output: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Git snapshot-index record is malformed")
        if require_metadata_binding and (
            set(record) != {
                "provider", "request_sha256", "response_sha256",
                "metadata_sha256", "metadata_byte_count", "retrieved_at_utc",
                "byte_count", "request", "metadata_path", "response_path",
            }
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("metadata_sha256", ""))
            )
            or type(record.get("metadata_byte_count")) is not int
            or record["metadata_byte_count"] < 1
        ):
            raise ValueError("Git snapshot-index metadata binding is malformed")
        for field in ("metadata_path", "response_path"):
            raw = _normalise_git_relative(
                record.get(field), label=f"snapshot {field}"
            )
            relative = (PurePosixPath(index_path).parent / raw).as_posix()
            relative = _normalise_git_relative(relative, label=f"snapshot {field}")
            blob = _run_git(bare, "cat-file", "-e", f"{commit}:{relative}")
            if blob.returncode:
                raise ValueError(f"Git snapshot dependency is absent: {relative}")
            expected_sha = (
                record.get("response_sha256")
                if field == "response_path" else record.get("metadata_sha256")
            )
            if isinstance(expected_sha, str):
                payload = _run_git(bare, "show", f"{commit}:{relative}")
                if hashlib.sha256(payload.stdout).hexdigest() != expected_sha:
                    raise ValueError(f"Git snapshot {field} SHA-256 changed")
                expected_bytes = record.get(
                    "byte_count" if field == "response_path" else "metadata_byte_count"
                )
                if require_metadata_binding and len(payload.stdout) != expected_bytes:
                    raise ValueError(f"Git snapshot {field} byte count changed")
            output.add(relative)
    return output


def _reconstruct_input_dependency_paths(
    bare: Path,
    commit: str,
    paths: Mapping[str, str],
) -> set[str]:
    required = {
        "candidate_table",
        "candidate_provenance",
        "candidate_snapshot_index",
        "external_registry",
        "external_lock",
        "input_manifest",
    }
    if set(paths) != required:
        raise ValueError("cannot independently resolve exact input evidence paths")
    output = set(paths.values())
    for relative in output:
        if _run_git(bare, "cat-file", "-e", f"{commit}:{relative}").returncode:
            raise ValueError(f"Git input dependency is absent: {relative}")
    lock = _git_json_document(
        bare, commit, paths["external_lock"], label="external registry lock"
    )
    if lock.get("status") != "REGISTRY_FROZEN_LABELS_SEALED":
        raise ValueError("Git external registry lock is not sealed")
    registry_blob = _run_git(
        bare, "show", f"{commit}:{paths['external_registry']}"
    )
    if (
        registry_blob.returncode
        or lock.get("confirmatory_registry_sha256")
        != hashlib.sha256(registry_blob.stdout).hexdigest()
    ):
        raise ValueError("Git external registry lock binds another registry")
    frozen = lock.get("frozen_artifacts")
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "development_panel_spec",
        "candidate_table",
        "candidate_provenance",
        "candidate_snapshot_index",
    }:
        raise ValueError("Git external registry lock dependency set changed")
    for name, binding in frozen.items():
        relative = _git_declared_binding_path(
            bare, commit, binding, label=f"external lock {name}"
        )
        if name != "development_panel_spec" and relative != paths[name]:
            raise ValueError(f"Git external lock names another {name}")
        output.add(relative)
    output |= _snapshot_dependency_paths(
        bare, commit, paths["candidate_snapshot_index"]
    )
    manifest = _git_json_document(
        bare, commit, paths["input_manifest"], label="actual-input manifest"
    )
    if (
        manifest.get("format") != "thermoroute.route-a-prelabel-inputs.v1"
        or manifest.get("status") != "FROZEN_PRELABEL_NO_OUTCOMES"
        or manifest.get("contains_outcome") is not False
        or manifest.get("contains_outcome_labels") is not False
        or manifest.get("post_2020_wtemp_requested_or_inspected") is not False
    ):
        raise ValueError("Git actual-input manifest is not canonical outcome-free evidence")
    for field in ("cohort_tables", "registry_inputs"):
        values = manifest.get(field)
        if not isinstance(values, Mapping) or set(values) != {"temporal", "external"}:
            raise ValueError(f"Git actual-input {field} set changed")
        for cohort, binding in values.items():
            relative = _git_declared_binding_path(
                bare, commit, binding, label=f"actual-input {field}/{cohort}"
            )
            if field == "registry_inputs" and cohort == "external" and relative != paths[
                "external_registry"
            ]:
                raise ValueError("Git actual-input manifest uses another external registry")
            output.add(relative)
    evidence = manifest.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Git actual-input manifest lacks source evidence")
    for index, item in enumerate(evidence):
        if (
            not isinstance(item, Mapping)
            or item.get("contains_outcome") is not False
            or item.get("contains_outcome_labels") is not False
        ):
            raise ValueError("Git actual-input source evidence is not outcome-free")
        relative = _git_declared_binding_path(
            bare,
            commit,
            item.get("artifact"),
            label=f"actual-input source evidence {index}",
        )
        output.add(relative)
        if item.get("evidence_type") == "snapshot_index":
            output |= _snapshot_dependency_paths(bare, commit, relative)
        elif item.get("evidence_type") != "normalized_immutable_snapshot":
            raise ValueError("Git actual-input evidence type changed")
    return output


def _verify_prelabel_chronology_from_bundle(
    *,
    root: Path,
    bare: Path,
    evidence: Mapping[str, Any],
    authorization: Mapping[str, Any],
    compute_commit: str,
    manuscript_commit: str,
    final_protocol_commit: str,
) -> None:
    """Independently replay every chronology assertion from the Git bundle."""
    chronology_path, chronology = _validate_prelabel_chronology_structure(
        root, {}, authorization
    )
    declared = evidence.get("prelabel_chronology")
    if not isinstance(declared, Mapping) or set(declared) != {
        "receipt",
        "receipt_commit",
        "receipt_git_blob_oid",
        "order",
        "model_source_control_artifact_count",
        "model_freeze_artifact_count",
        "input_evidence_artifact_count",
    }:
        raise ValueError("Git history evidence lacks exact prelabel chronology evidence")
    if (
        declared.get("receipt") != _binding_for(root, chronology_path)
        or declared.get("order") != chronology.get("order")
        or declared.get("model_source_control_artifact_count")
        != len(chronology["model_source_control_artifacts"])
        or declared.get("model_freeze_artifact_count")
        != len(chronology["model_freeze_artifacts"])
        or declared.get("input_evidence_artifact_count")
        != len(chronology["input_evidence_artifacts"])
    ):
        raise ValueError("Git chronology evidence differs from its receipt")
    order = chronology["order"]
    model_commit = str(order["model_freeze_commit"])
    input_commit = str(order["input_evidence_commit"])
    creation_commit = str(order["receipt_creation_base_commit"])
    receipt_commit = str(declared.get("receipt_commit", ""))
    chronology_commits = (
        model_commit, input_commit, creation_commit, receipt_commit
    )
    if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in chronology_commits):
        raise ValueError("Git chronology evidence has a malformed commit")
    for commit in chronology_commits:
        result = _run_git(bare, "cat-file", "-e", f"{commit}^{{commit}}")
        if result.returncode:
            raise ValueError("Git chronology evidence references an absent commit")
    for ancestor, descendant, label, strict in (
        (
            final_protocol_commit,
            model_commit,
            "final-protocol-to-model-freeze",
            True,
        ),
        (model_commit, input_commit, "model-freeze-to-input-evidence", True),
        (input_commit, creation_commit, "input-evidence-to-receipt-base", True),
        (creation_commit, receipt_commit, "receipt-base-to-receipt-commit", True),
        (receipt_commit, compute_commit, "receipt-commit-to-compute", False),
        (compute_commit, manuscript_commit, "compute-to-manuscript", False),
    ):
        relation = _run_git(
            bare, "merge-base", "--is-ancestor", ancestor, descendant
        )
        if relation.returncode or (strict and ancestor == descendant):
            raise ValueError(f"Git bundle prelabel chronology failed: {label}")

    chronology_relative = _relative(
        root, chronology_path, label="prelabel chronology"
    )
    receipt_blob = _run_git(
        bare, "show", f"{receipt_commit}:{chronology_relative}"
    )
    receipt_oid = _run_git(
        bare, "rev-parse", f"{receipt_commit}:{chronology_relative}", text=True
    )
    if (
        receipt_blob.returncode
        or receipt_oid.returncode
        or receipt_blob.stdout != chronology_path.read_bytes()
        or receipt_oid.stdout.strip() != declared.get("receipt_git_blob_oid")
    ):
        raise ValueError("prelabel chronology receipt differs from its Git commit")
    receipt_creations = _git_path_creation_commits(
        bare, compute_commit, chronology_relative
    )
    if receipt_creations != [receipt_commit]:
        raise ValueError("prelabel chronology receipt was not added exactly once")

    def replay_binding(commit: str, value: object, label: str) -> str:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} chronology binding is malformed")
        relative = str(value.get("path", ""))
        blob = _run_git(bare, "show", f"{commit}:{relative}")
        oid = _run_git(bare, "rev-parse", f"{commit}:{relative}", text=True)
        if (
            blob.returncode
            or oid.returncode
            or hashlib.sha256(blob.stdout).hexdigest() != value.get("sha256")
            or len(blob.stdout) != value.get("byte_count")
            or oid.stdout.strip() != value.get("git_blob_oid")
        ):
            raise ValueError(f"{label} cannot be replayed at its chronology commit")
        return relative

    model_paths: set[str] = set()
    control_bindings: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(chronology["model_source_control_artifacts"]):
        relative = replay_binding(
            model_commit, value, f"model_source_control_artifacts[{index}]"
        )
        if relative in control_bindings or not isinstance(value, Mapping):
            raise ValueError("chronology model source/control registry is duplicated")
        control_bindings[relative] = value
        model_paths.add(relative)
    model_tree = _run_git(
        bare, "ls-tree", "-r", "--name-only", "-z", model_commit
    )
    if model_tree.returncode:
        raise ValueError("cannot enumerate model-freeze source/control Git tree")
    try:
        expected_control = {
            item.decode("utf-8", errors="strict")
            for item in model_tree.stdout.split(b"\0")
            if item and _is_model_control_path(item.decode("utf-8", errors="strict"))
        }
    except UnicodeDecodeError as exc:
        raise ValueError("model-freeze Git tree contains a non-UTF-8 path") from exc
    if set(control_bindings) != expected_control:
        raise ValueError(
            "chronology source/control path set differs from model-freeze Git tree"
        )
    model_source_inventory = {
        relative: str(control_bindings[relative]["sha256"])
        for relative in sorted(control_bindings)
        if _matches_source_inventory(relative)
    }
    if (
        not model_source_inventory
        or chronology.get("source_tree_sha256")
        != _sha256_json(model_source_inventory)
    ):
        raise ValueError(
            "chronology source-tree digest differs from model-freeze Git blobs"
        )
    declared_model_artifacts: set[str] = set()
    for field in ("required_gate_files_at_model_freeze", "model_freeze_artifacts"):
        for index, value in enumerate(chronology[field]):
            relative = replay_binding(model_commit, value, f"{field}[{index}]")
            model_paths.add(relative)
            if field == "model_freeze_artifacts":
                declared_model_artifacts.add(relative)
    seal = chronology["protocol_history"]["seal"]
    model_paths.add(replay_binding(model_commit, seal, "protocol seal"))
    input_paths = {
        replay_binding(input_commit, value, f"input_evidence_artifacts[{index}]")
        for index, value in enumerate(chronology["input_evidence_artifacts"])
    }
    chronology_paths = chronology.get("paths")
    if not isinstance(chronology_paths, Mapping):
        raise ValueError("chronology paths registry is malformed")
    reconstructed_models = _reconstruct_model_dependency_paths(
        bare,
        model_commit,
        suite_path=str(chronology_paths.get("model_suite", "")),
        replay_path=str(chronology_paths.get("development_replay", "")),
    )
    if declared_model_artifacts != reconstructed_models:
        raise ValueError(
            "chronology model dependency registry is not the independently "
            "reconstructed exact set"
        )
    input_names = (
        "candidate_table",
        "candidate_provenance",
        "candidate_snapshot_index",
        "external_registry",
        "external_lock",
        "input_manifest",
    )
    reconstructed_inputs = _reconstruct_input_dependency_paths(
        bare,
        input_commit,
        {name: str(chronology_paths.get(name, "")) for name in input_names},
    )
    if input_paths != reconstructed_inputs:
        raise ValueError(
            "chronology input dependency registry is not the independently "
            "reconstructed exact set"
        )
    for item in chronology["protocol_history"]["declared_git_show_bindings"]:
        if not isinstance(item, Mapping):
            raise ValueError("chronology Git-show binding is malformed")
        role = str(item.get("role", ""))
        expected_commit = (
            authorization["protocol"]["authoritative_commit"]
            if role == "original_markdown"
            else final_protocol_commit
        )
        blob = _run_git(
            bare, "show", f"{item.get('commit')}:{item.get('path')}"
        )
        if (
            blob.returncode
            or item.get("commit") != expected_commit
            or hashlib.sha256(blob.stdout).hexdigest() != item.get("sha256")
        ):
            raise ValueError(f"chronology protocol Git-show binding failed: {role}")

    for relative in chronology["absence_at_model_freeze"]["checked_paths"]:
        absent = _run_git(
            bare,
            "ls-tree",
            "-r",
            "--name-only",
            model_commit,
            "--",
            str(relative),
            text=True,
        )
        if absent.returncode or absent.stdout.strip():
            raise ValueError(
                f"confirmation-period artifact existed at model freeze: {relative}"
            )

    def commits_between(start: str, end: str) -> list[str]:
        return _git_commits_between(bare, start, end)

    def touched(commit: str) -> set[str]:
        return {relative for _status, relative in _git_commit_name_status(bare, commit)}

    for commit in commits_between(model_commit, compute_commit):
        changed = touched(commit)
        forbidden = sorted(path for path in changed if _is_model_control_path(path))
        if forbidden:
            raise ValueError(
                f"Git bundle has post-model source/control touch: {commit}/{forbidden[:3]}"
            )
        changed_models = sorted(changed & model_paths)
        if changed_models:
            raise ValueError(
                f"Git bundle has post-freeze model artifact touch: {changed_models[:3]}"
            )
    for commit in commits_between(input_commit, compute_commit):
        changed_inputs = sorted(touched(commit) & input_paths)
        if changed_inputs:
            raise ValueError(
                f"Git bundle has post-freeze input artifact touch: {changed_inputs[:3]}"
            )


def _verify_git_history_evidence(
    root: Path, marker: Mapping[str, Any], profile: str
) -> None:
    evidence = marker.get("git_history_evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("format") != "thermoroute.route-a-git-history-evidence.v1"
        or evidence.get("profile") != profile
        or evidence.get("external_timestamp_or_public_preregistration") is not False
    ):
        raise ValueError("release lacks honest Git/preregistration evidence status")
    manifest_path = root / "outputs" / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, label="release manifest")
        expected_release_evidence = {
            "profile": profile,
            "claim_validation": marker.get("claim_validation"),
            "git_history_evidence": dict(evidence),
            "reproducibility_lock": marker.get("artifact_closure", {}).get(
                "reproducibility_lock"
            ),
        }
        if manifest.get("release_evidence") != expected_release_evidence:
            raise ValueError("manifest does not bind claim/Git/lock release evidence")
    bundle = _add_binding(
        root, {}, "git_history", evidence.get("bundle"), label="compute Git bundle"
    )
    if _relative(root, bundle, label="Git bundle") != GIT_BUNDLE_PATH:
        raise ValueError("Git history bundle leaves its canonical evidence path")
    commits = [
        str(evidence.get("compute_commit", "")),
        str(evidence.get("manuscript_commit", "")),
        str(evidence.get("authoritative_protocol_commit", "")),
        str(evidence.get("final_prelabel_protocol_commit", "")),
    ]
    if any(len(commit) != 40 for commit in commits):
        raise ValueError("Git history evidence has a malformed commit")
    with tempfile.TemporaryDirectory(prefix="thermoroute-release-git-") as name:
        bare = Path(name) / "audit.git"
        bare.mkdir()
        for arguments, label in (
            (("init", "--bare", "-q"), "initialize isolated Git audit"),
            (("bundle", "verify", str(bundle)), "verify release Git bundle"),
            (
                (
                    "fetch",
                    "-q",
                    str(bundle),
                    "HEAD:refs/heads/release-evidence",
                ),
                "fetch release Git bundle",
            ),
        ):
            result = _run_git(bare, *arguments)
            if result.returncode:
                raise ValueError(f"cannot {label}")
        _assert_safe_git_repository(bare, bare=True)
        for commit in commits:
            result = _run_git(bare, "cat-file", "-e", f"{commit}^{{commit}}")
            if result.returncode:
                raise ValueError("Git history evidence references an absent commit")
        bundled_head = _run_git(
            bare, "rev-parse", "refs/heads/release-evidence", text=True
        )
        if bundled_head.returncode or bundled_head.stdout.strip() != commits[1]:
            raise ValueError("Git bundle HEAD differs from manuscript commit")
        if profile == PREOPEN_PROFILE and commits[0] != commits[1]:
            raise ValueError("pre-opening release compute and manuscript commits differ")
        for ancestor, descendant, label in (
            (commits[0], commits[1], "compute-to-manuscript"),
            (commits[2], commits[3], "original-to-final protocol"),
            (commits[3], commits[0], "final protocol-to-compute"),
        ):
            relation = _run_git(
                bare, "merge-base", "--is-ancestor", ancestor, descendant
            )
            if relation.returncode:
                raise ValueError(
                    f"Git bundle {label} commit is not an ancestor of manuscript"
                )
        if profile == POSTOPEN_PROFILE:
            policy = marker.get("authorized_worktree_dirt_policy")
            documents = (
                policy.get("committed_document_diff")
                if isinstance(policy, Mapping) else None
            )
            if not isinstance(documents, list):
                raise ValueError("Git evidence lacks the committed document bindings")
            expected_documents: dict[str, Mapping[str, Any]] = {}
            for binding in documents:
                if not isinstance(binding, Mapping):
                    raise ValueError("Git evidence document binding is malformed")
                relative = str(binding.get("path", ""))
                if relative in expected_documents:
                    raise ValueError("Git evidence document binding is duplicated")
                expected_documents[relative] = binding
            for commit in _git_commits_between(bare, commits[0], commits[1]):
                forbidden_intermediate = [
                    f"{status} {relative}"
                    for status, relative in _git_commit_name_status(bare, commit)
                    if status not in {"A", "M"}
                    or not (
                        relative == "README.md"
                        or relative.startswith("paper/")
                    )
                ]
                if forbidden_intermediate:
                    raise ValueError(
                        "Git bundle contains a forbidden compute-to-manuscript "
                        "intermediate change: "
                        f"{commit}/{forbidden_intermediate[:10]}"
                    )
            changed = _run_git(
                bare,
                "diff",
                "--name-status",
                "--no-renames",
                "-z",
                f"{commits[0]}..{commits[1]}",
            )
            if changed.returncode:
                raise ValueError("cannot replay compute-to-manuscript Git diff")
            fields = changed.stdout.split(b"\0")
            if fields and fields[-1] == b"":
                fields.pop()
            if len(fields) % 2:
                raise ValueError("compute-to-manuscript Git diff is malformed")
            observed: dict[str, str] = {}
            for offset in range(0, len(fields), 2):
                status = fields[offset].decode("ascii", errors="strict")
                relative = fields[offset + 1].decode("utf-8", errors="strict")
                if (
                    status not in {"A", "M"}
                    or relative in observed
                    or not (
                        relative == "README.md"
                        or relative.startswith("paper/")
                    )
                ):
                    raise ValueError(
                        "Git bundle contains a forbidden compute-to-manuscript change"
                    )
                observed[relative] = status
            if sorted(observed) != sorted(expected_documents):
                raise ValueError(
                    "Git bundle document diff differs from release revision bindings"
                )
            for relative, binding in expected_documents.items():
                blob = _run_git(bare, "show", f"{commits[1]}:{relative}")
                if (
                    blob.returncode
                    or hashlib.sha256(blob.stdout).hexdigest()
                    != binding.get("sha256")
                    or len(blob.stdout) != binding.get("bytes")
                ):
                    raise ValueError(
                        f"Git manuscript blob differs from release document: {relative}"
                    )
        sealed = evidence.get("sealed_protocol_blob")
        if not isinstance(sealed, Mapping):
            raise ValueError("Git history evidence lacks sealed protocol blob")
        _verify_protected_tree_from_bundle(
            root=root, bare=bare, commit=commits[0]
        )
        blob = _run_git(
            bare, "show", f"{sealed.get('commit')}:{sealed.get('path')}"
        )
        if (
            blob.returncode
            or hashlib.sha256(blob.stdout).hexdigest() != sealed.get("sha256")
            or len(blob.stdout) != sealed.get("bytes")
            or sealed.get("commit") != commits[2]
            or sealed.get("path") != "protocols/route_a_confirmatory_protocol.md"
        ):
            raise ValueError("sealed protocol blob cannot be replayed from Git bundle")
        seal_path = _add_binding(
            root,
            {},
            "git_history",
            evidence.get("protocol_seal"),
            label="final protocol seal",
        )
        if _relative(root, seal_path, label="protocol seal") != PROTOCOL_SEAL_PATH:
            raise ValueError("Git evidence uses a noncanonical protocol seal")
        protocol_document = _load_json(
            root / "protocols" / "route_a_confirmatory_v1.json",
            label="Route-A protocol",
        )
        canonical_seal_path, seal_document = _load_protocol_seal(
            root, protocol_document
        )
        if canonical_seal_path != seal_path:
            raise ValueError("Git evidence protocol seal path changed")
        final_blobs = evidence.get("final_protocol_blobs")
        if not isinstance(final_blobs, list) or len(final_blobs) != 2:
            raise ValueError("Git evidence lacks the final JSON/Markdown protocol blobs")
        expected_final = seal_document.get("final_prelabel_protocol", {})
        observed_keys: set[str] = set()
        for binding in final_blobs:
            if not isinstance(binding, Mapping):
                raise ValueError("final protocol blob evidence is malformed")
            relative = str(binding.get("path", ""))
            key = {
                "protocols/route_a_confirmatory_v1.json": "json",
                "protocols/route_a_confirmatory_protocol.md": "markdown",
            }.get(relative)
            if key is None or key in observed_keys:
                raise ValueError("final protocol blob evidence is duplicated/noncanonical")
            observed_keys.add(key)
            expected_binding = expected_final.get(key)
            final_blob = _run_git(
                bare, "show", f"{binding.get('commit')}:{relative}"
            )
            current = _resolve_release_path(
                root, relative, label=f"final protocol {key}"
            )
            if (
                not isinstance(expected_binding, Mapping)
                or binding.get("commit") != commits[3]
                or expected_binding.get("sha256") != binding.get("sha256")
                or final_blob.returncode
                or hashlib.sha256(final_blob.stdout).hexdigest()
                != binding.get("sha256")
                or len(final_blob.stdout) != binding.get("bytes")
                or final_blob.stdout != current.read_bytes()
            ):
                raise ValueError(
                    f"final protocol {key} cannot be replayed from Git bundle"
                )
        if profile == POSTOPEN_PROFILE:
            authorization = _load_json(
                root / str(marker["authorization"]["path"]),
                label="opening authorization",
            )
            authorized_protocol = authorization.get("protocol")
            authorized_seal = (
                authorized_protocol.get("seal")
                if isinstance(authorized_protocol, Mapping)
                else None
            )
            if (
                not isinstance(authorized_protocol, Mapping)
                or not isinstance(authorized_seal, Mapping)
                or authorized_protocol.get("authoritative_markdown_sha256")
                != sealed.get("sha256")
                or authorized_protocol.get("final_prelabel_commit") != commits[3]
                or authorized_seal.get("path") != PROTOCOL_SEAL_PATH
                or authorized_seal.get("sha256") != sha256_file(seal_path)
            ):
                raise ValueError("sealed protocol blob differs from authorization")
            _verify_authorized_compute_tree_from_bundle(
                root=root,
                bare=bare,
                authorization=authorization,
                compute_commit=commits[0],
            )
            _verify_prelabel_chronology_from_bundle(
                root=root,
                bare=bare,
                evidence=evidence,
                authorization=authorization,
                compute_commit=commits[0],
                manuscript_commit=commits[1],
                final_protocol_commit=commits[3],
            )


def _preflight_zip_container(path: Path) -> None:
    """Bound the central directory before ``zipfile`` allocates per-member state."""
    size = path.stat().st_size
    if size < 22 or size > MAX_ARCHIVE_FILE_BYTES:
        raise ValueError("archive compressed size is outside the safety limit")
    tail_size = min(size, 65_557)
    with path.open("rb") as handle:
        handle.seek(size - tail_size)
        tail = handle.read(tail_size)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("archive lacks a canonical end-of-central-directory record")
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", tail, offset)
    if signature != b"PK\x05\x06" or offset + 22 + comment_size != len(tail):
        raise ValueError("archive end-of-central-directory record is malformed")
    if comment_size != 0:
        raise ValueError("archive comments are prohibited by the deterministic format")
    if disk_number or central_disk or disk_entries != total_entries:
        raise ValueError("multi-disk ZIP archives are prohibited")
    if total_entries in {0, 0xFFFF} or central_size == 0xFFFFFFFF:
        raise ValueError("empty or ZIP64 archive containers are prohibited")
    if total_entries > MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"archive member count exceeds safety limit: {total_entries}")
    if central_size > MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES:
        raise ValueError("archive central directory exceeds safety limit")
    eocd_absolute = size - tail_size + offset
    if central_offset + central_size != eocd_absolute:
        raise ValueError("archive central-directory offsets are inconsistent")


def _validate_archive_resource_limits(infos: list[zipfile.ZipInfo]) -> None:
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(
            f"archive member count exceeds safety limit: {len(infos)}"
        )
    total = 0
    for info in infos:
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted archive member is prohibited: {info.filename!r}")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError(
                f"unsupported archive compression method: {info.filename!r}"
            )
        if info.file_size < 0 or info.compress_size < 0:
            raise ValueError(f"archive member has a negative size: {info.filename!r}")
        if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError(
                f"archive member exceeds uncompressed safety limit: {info.filename!r}"
            )
        total += info.file_size
        if total > MAX_ARCHIVE_TOTAL_BYTES:
            raise ValueError("archive total uncompressed size exceeds safety limit")
        if info.file_size:
            if info.compress_size == 0:
                raise ValueError(
                    f"archive member has an impossible compression size: {info.filename!r}"
                )
            ratio = info.file_size / info.compress_size
            if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
                raise ValueError(
                    "archive member compression ratio exceeds safety limit: "
                    f"{info.filename!r}"
                )


def normalised_members(archive: zipfile.ZipFile) -> set[str]:
    """Return paths below the single archive root after security validation."""
    members: set[str] = set()
    infos = archive.infolist()
    _validate_archive_resource_limits(infos)
    names = [info.filename for info in infos]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("archive entries are not uniquely ordered")
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe archive path: {info.filename!r}")
        if path.parts[0] != ARCHIVE_ROOT:
            raise ValueError(f"archive member outside {ARCHIVE_ROOT}/: {info.filename!r}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f"symbolic links are not allowed in release: {info.filename!r}")
        if info.date_time != (1980, 1, 1, 0, 0, 0):
            raise ValueError(f"archive member has a non-deterministic timestamp: {info.filename!r}")
        permission = (info.external_attr >> 16) & 0o777
        expected_kind: int
        if info.is_dir():
            expected_permission = 0o755
            expected_kind = stat.S_IFDIR
        else:
            relative = PurePosixPath(*path.parts[1:])
            expected_permission = (
                0o755
                if relative.parts[:1] == ("scripts",)
                and relative.suffix in {".py", ".sh"}
                else 0o644
            )
            expected_kind = stat.S_IFREG
        if permission != expected_permission or mode != expected_kind:
            raise ValueError(f"archive member has a non-canonical mode: {info.filename!r}")
        if len(path.parts) > 1 and not info.is_dir():
            members.add(PurePosixPath(*path.parts[1:]).as_posix())
    return members


def _extract_archive_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    """Stream regular members below destination while re-enforcing size limits."""
    destination = destination.resolve()
    total_written = 0
    for info in archive.infolist():
        posix = PurePosixPath(info.filename)
        target = destination.joinpath(*posix.parts)
        resolved = target.resolve()
        if resolved != destination and destination not in resolved.parents:
            raise ValueError(f"archive extraction path escapes destination: {info.filename!r}")
        permission = (info.external_attr >> 16) & 0o777
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(permission)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError(f"archive extraction target already exists: {info.filename!r}")
        written = 0
        with archive.open(info, "r") as source, target.open("xb") as output:
            while True:
                chunk = source.read(ARCHIVE_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                total_written += len(chunk)
                if written > info.file_size or written > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ValueError(
                        f"archive member expanded beyond declared limit: {info.filename!r}"
                    )
                if total_written > MAX_ARCHIVE_TOTAL_BYTES:
                    raise ValueError("archive expanded beyond total safety limit")
                output.write(chunk)
        if written != info.file_size:
            raise ValueError(
                f"archive member expanded size differs from metadata: {info.filename!r}"
            )
        target.chmod(permission)


def validate_members(members: set[str]) -> None:
    missing = sorted(REQUIRED_MEMBERS - members)
    if missing:
        raise ValueError("release is missing required members: " + ", ".join(missing))
    forbidden = sorted(FORBIDDEN_MEMBERS & members)
    if forbidden:
        raise ValueError("release contains stale mixed-generation evidence: "
                         + ", ".join(forbidden))
    if not any(path.startswith("paper/") for path in members):
        raise ValueError("release contains no manuscript files")


def _read_profile_marker(root: Path) -> dict[str, Any]:
    marker = _load_json(root / PROFILE_MARKER, label="release profile marker")
    if marker.get("format") != PROFILE_FORMAT:
        raise ValueError("release carries an unsupported profile marker")
    if marker.get("profile") not in RELEASE_PROFILES:
        raise ValueError("release profile marker has an unknown profile")
    if marker.get("status") != marker.get("profile"):
        raise ValueError("release profile status differs from its profile")
    return marker


def _verify_declared_closure(
    root: Path,
    declared: object,
    actual: Mapping[str, set[Path]],
) -> None:
    if not isinstance(declared, Mapping):
        raise ValueError("release marker lacks an artifact closure")
    expected = _category_bindings(root, actual)
    if dict(declared) != expected:
        raise ValueError("release artifact closure differs from canonical dependencies")
    for category, bindings in declared.items():
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(f"release closure category is empty: {category}")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError(f"release closure binding is malformed: {category}")
            path = _resolve_release_path(
                root,
                binding.get("path"),
                label=f"release closure {category}",
                expected_sha256=str(binding.get("sha256", "")),
            )
            if not path.is_file() or binding.get("bytes") != path.stat().st_size:
                raise ValueError(f"release closure size changed: {category}/{path.name}")


def _closure_paths(declared: object) -> set[str]:
    if not isinstance(declared, Mapping):
        return set()
    output: set[str] = set()
    for bindings in declared.values():
        if isinstance(bindings, list):
            for binding in bindings:
                if isinstance(binding, Mapping) and isinstance(binding.get("path"), str):
                    output.add(str(binding["path"]))
    return output


def _verify_archived_revision_contract(
    root: Path,
    marker: Mapping[str, Any],
    authorization: Mapping[str, Any],
    state: Mapping[str, str],
) -> None:
    policy = marker.get("authorized_worktree_dirt_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("opened release lacks compute/manuscript revision separation")
    compute = str(authorization.get("source", {}).get("git_commit_before_authorization", ""))
    expected_static = {
        "compute_commit": compute,
        "committed_document_whitelist": ["README.md", "paper/**"],
        "tracked_changes_allowed": False,
        "staged_changes_allowed": False,
        "untracked_exact": [str(marker["authorization"]["path"])],
        "untracked_prefixes": [state["run_directory"].rstrip("/") + "/"],
    }
    if any(policy.get(key) != value for key, value in expected_static.items()):
        raise ValueError("archived post-opening revision policy is inconsistent")
    manuscript = policy.get("manuscript_commit")
    if not isinstance(manuscript, str) or len(manuscript) != 40:
        raise ValueError("release lacks a manuscript commit")
    documents = policy.get("committed_document_diff")
    if not isinstance(documents, list):
        raise ValueError("release lacks the committed document diff")
    paths: list[str] = []
    for binding in documents:
        if not isinstance(binding, Mapping):
            raise ValueError("committed document binding is malformed")
        path = str(binding.get("path", ""))
        if not (path == "README.md" or path.startswith("paper/")):
            raise ValueError("committed document diff leaves its whitelist")
        if path in paths:
            raise ValueError("committed document diff duplicates a path")
        paths.append(path)
        resolved = _resolve_release_path(
            root, path, label="committed document",
            expected_sha256=str(binding.get("sha256", "")),
        )
        if binding.get("bytes") != resolved.stat().st_size:
            raise ValueError("committed document size differs from revision contract")
    if paths != sorted(paths):
        raise ValueError("committed document diff is not deterministic")
    manifest_path = root / "outputs" / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, label="release manifest")
        if (
            manifest.get("git", {}).get("commit") != manuscript
            or manifest.get("release_revision") != dict(policy)
        ):
            raise ValueError("manifest does not bind compute/manuscript revision identities")


def _run_trusted_replay(root: Path, marker: Mapping[str, Any]) -> None:
    interface = marker.get("trusted_replay_interface")
    expected_authorization = marker.get("authorization", {}).get("path")
    expected = {
        "entrypoint": "scripts/route_a_trusted_scorer.py",
        "arguments": ["--verify-release", "--authorization", expected_authorization],
        "policy": "fixed-entrypoint-fresh-python-I",
    }
    if interface != expected:
        raise ValueError("trusted replay interface is mutable or malformed")
    entrypoint = _resolve_release_path(
        root, expected["entrypoint"], label="trusted replay entrypoint"
    )
    authorization = _resolve_release_path(
        root, expected_authorization, label="trusted replay authorization"
    )
    environment = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(entrypoint),
            "--verify-release",
            "--authorization",
            str(authorization),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"trusted Route-A replay failed: {detail[-4000:]}")


def verify_release_profile(
    root: str | Path,
    members: set[str] | None = None,
    *,
    run_trusted_replay: bool = True,
) -> str:
    root = Path(root).resolve()
    marker = _read_profile_marker(root)
    profile = str(marker["profile"])
    if members is None:
        members = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*") if path.is_file()
        }
    if profile == PREOPEN_PROFILE:
        if (
            marker.get("confirmatory_scoring_completed") is not False
            or marker.get("directional_claims_allowed") is not False
            or marker.get("supported_test_ids") != []
            or marker.get("supports_route_a_confirmatory_conclusions") is not False
            or marker.get("labels_included") is not False
            or marker.get("warning") != PREOPEN_WARNING
            or marker.get("fully_hashed_lock_role") != HASHED_LOCK_ROLE
        ):
            raise ValueError("pre-opening marker overstates its evidentiary status")
        forbidden = sorted(
            path for path in members
            if path.startswith("outputs/confirmatory/")
            or "labels" in PurePosixPath(path).parts
            or (path.startswith("outputs/") and path != "outputs/manifest.json")
        )
        if forbidden:
            raise ValueError(
                "PREOPEN_NOT_COMPLETE contains confirmation/label/result artifacts: "
                + ", ".join(forbidden[:10])
            )
        expected_categories = _canonical_categories(root)
        if set(marker.get("artifact_closure", {})) != {
            "canonical_development", "reproducibility_lock"
        }:
            raise ValueError("pre-opening archive declares non-canonical result evidence")
        _verify_declared_closure(root, marker.get("artifact_closure"), expected_categories)
        if run_trusted_replay:
            _verify_git_history_evidence(root, marker, profile)
        _verify_claim_audit(
            root,
            marker,
            profile,
            execute_validator=run_trusted_replay,
        )
        return profile

    if (
        marker.get("confirmatory_scoring_completed") is not True
        or marker.get("labels_included") is not True
        or marker.get("fully_hashed_lock_role") != HASHED_LOCK_ROLE
    ):
        raise ValueError("opened-complete marker does not acknowledge opened labels")
    authorization_binding = marker.get("authorization")
    authorization = _add_binding(
        root, {}, "authorization", authorization_binding,
        label="release authorization",
    )
    # The release root is hostile input.  Establish the compute Git tree,
    # source inventory and fixed-code bytes before importing any Python module
    # from that root.  This is mandatory even when the expensive model replay
    # is disabled; coverage replay remains an evidence-validation requirement.
    _verify_git_history_evidence(root, marker, profile)
    categories, document, state = _gather_postopen_categories(root, authorization)
    expected_claim_status = _derive_release_claim_status(root, document, state)
    if (
        marker.get("opening_id") != document.get("opening_id")
        or marker.get("state_namespace") != state.get("namespace")
        or set(marker.get("artifact_closure", {})) != REQUIRED_POSTOPEN_CATEGORIES
        or any(
            marker.get(key) != value
            for key, value in expected_claim_status.items()
        )
    ):
        raise ValueError(
            "opened-complete marker identity/categories/claim status are inconsistent"
        )
    _verify_archived_revision_contract(root, marker, document, state)
    _verify_declared_closure(root, marker.get("artifact_closure"), categories)
    closure = _closure_paths(marker.get("artifact_closure"))
    unexplained_scientific = sorted(
        path for path in members
        if (
            (path.startswith("data_usgs/") and path != PROFILE_MARKER)
            or (path.startswith("outputs/") and path != "outputs/manifest.json")
        )
        and path not in closure
    )
    if unexplained_scientific:
        raise ValueError(
            "opened release contains scientific artifacts outside the authorization closure: "
            + ", ".join(unexplained_scientific[:10])
        )
    _verify_claim_audit(
        root,
        marker,
        profile,
        execute_validator=run_trusted_replay,
    )
    if run_trusted_replay:
        _run_trusted_replay(root, marker)
    return profile


def verify_checksum_sidecar(archive_path: Path) -> None:
    sidecar = Path(str(archive_path) + ".sha256")
    if not sidecar.is_file():
        return
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields:
        raise ValueError(f"empty checksum sidecar: {sidecar}")
    actual = sha256_file(archive_path)
    if fields[0] != actual:
        raise ValueError(f"archive checksum mismatch: expected {fields[0]}, got {actual}")


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _canonical_json_bytes(value: object) -> bytes:
    """Match the newline-terminated canonical JSON used by SnapshotStore."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _inside(root: Path, relative: str, *, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"{label} path must be relative")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the release") from exc
    if not path.is_file():
        raise ValueError(f"{label} file is absent")
    return path


def _parse_usgs_rdb(payload: bytes) -> list[dict[str, str]]:
    """Pure-stdlib parser for the exact NWIS tabular response in the archive."""
    try:
        lines = [
            line for line in payload.decode("utf-8").splitlines()
            if line and not line.startswith("#")
        ]
    except UnicodeDecodeError as exc:
        raise ValueError("HUC raw response is not UTF-8 NWIS RDB") from exc
    if len(lines) < 3:
        raise ValueError("HUC raw response has no NWIS RDB rows")
    reader = csv.DictReader(io.StringIO("\n".join([lines[0], *lines[2:]])), delimiter="\t")
    rows = [{str(key): str(value) for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("HUC raw response contains no sites")
    return rows


def _same_decimal(raw: str, derived: str) -> bool:
    raw, derived = raw.strip(), derived.strip()
    if not raw or not derived:
        return raw == derived
    try:
        return Decimal(raw) == Decimal(derived)
    except InvalidOperation:
        return False


def _verify_raw_huc_derivation(
    root: Path,
    *,
    spec: dict[str, object],
    registry_rows: list[dict[str, str]],
    huc_rows: list[dict[str, str]],
    huc_path: Path,
    provenance_path: Path,
) -> None:
    """Replay the released HUC CSV from its immutable NWIS response bytes."""
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported HUC provenance schema")
    if provenance.get("outcome_data_requested") is not False:
        raise ValueError("HUC evidence is not metadata-only")
    if provenance.get("join_key") != "site_no":
        raise ValueError("HUC evidence was not joined by stable site_no")
    if provenance.get("derived_csv_sha256") != sha256_file(huc_path):
        raise ValueError("HUC provenance does not bind the derived CSV")
    panel = spec.get("panel", {})
    station = spec.get("station_registry", {})
    if not isinstance(panel, dict) or not isinstance(station, dict):
        raise ValueError("frozen panel contract is malformed")
    if provenance.get("development_panel_sha256") != panel.get("sha256"):
        raise ValueError("HUC provenance is bound to another development panel")
    if provenance.get("development_metadata_sha256") != station.get(
        "source_metadata_sha256"
    ):
        raise ValueError("HUC provenance is bound to other station metadata")
    if provenance.get("site_count") != len(registry_rows):
        raise ValueError("HUC provenance site count differs from the registry")

    index_relative = provenance.get("raw_snapshot_index")
    if not isinstance(index_relative, str):
        raise ValueError("HUC provenance lacks a raw snapshot index")
    index_path = _inside(root, index_relative, label="HUC snapshot index")
    if provenance.get("raw_snapshot_index_sha256") != sha256_file(index_path):
        raise ValueError("HUC raw snapshot-index checksum mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index.get("records")
    if (
        index.get("schema_version") != 1
        or not isinstance(records, list)
        or index.get("snapshot_count") != len(records)
        or len(records) != 1
    ):
        raise ValueError("HUC raw snapshot index is malformed")
    record = records[0]
    if not isinstance(record, dict) or record.get("provider") != "usgs-nwis-site-metadata":
        raise ValueError("HUC raw snapshot has an unexpected provider")
    request = record.get("request")
    if not isinstance(request, dict):
        raise ValueError("HUC raw snapshot lacks its request")
    request_sha = hashlib.sha256(_canonical_json_bytes(request)).hexdigest()
    if request_sha != record.get("request_sha256") or request_sha != provenance.get(
        "request_sha256"
    ):
        raise ValueError("HUC raw request hash mismatch")
    if request != {
        "schema_version": 1,
        "provider": "usgs-nwis-site-metadata",
        "method": "GET",
        "url": request.get("url"),
        "headers": {},
    }:
        raise ValueError("HUC raw request contains an undeclared method/header")
    parsed_url = urlparse(str(request.get("url", "")))
    query = parse_qs(parsed_url.query, keep_blank_values=True)
    expected_sites = sorted(row["site_no"].strip() for row in registry_rows)
    requested_sites = sorted(query.get("sites", [""])[0].split(","))
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "waterservices.usgs.gov"
        or parsed_url.path != "/nwis/site/"
        or query.get("format") != ["rdb"]
        or query.get("siteOutput") != ["expanded"]
        or query.get("siteStatus") != ["all"]
        or set(query) != {"format", "sites", "siteOutput", "siteStatus"}
        or requested_sites != expected_sites
    ):
        raise ValueError("HUC raw request is not the frozen metadata-only site query")

    snapshot_root = index_path.parent.resolve()
    metadata_relative = record.get("metadata_path")
    response_relative = record.get("response_path")
    if not isinstance(metadata_relative, str) or not isinstance(response_relative, str):
        raise ValueError("HUC raw snapshot paths are malformed")
    metadata_path = _inside(snapshot_root, metadata_relative, label="HUC metadata")
    response_path = _inside(snapshot_root, response_relative, label="HUC response")
    payload = response_path.read_bytes()
    if (
        record.get("response_sha256") != sha256_file(response_path)
        or record.get("response_sha256") != provenance.get("response_sha256")
        or record.get("byte_count") != len(payload)
        or record.get("retrieved_at_utc") != provenance.get("retrieved_at_utc")
    ):
        raise ValueError("HUC raw response binding mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, expected in {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha,
        "response_file": response_path.name,
        "response_sha256": record.get("response_sha256"),
        "retrieved_at_utc": record.get("retrieved_at_utc"),
        "byte_count": len(payload),
        "http_status": 200,
    }.items():
        if metadata.get(key) != expected:
            raise ValueError(f"HUC snapshot metadata mismatch: {key}")
    actual_snapshot_files = {
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file()
    }
    if actual_snapshot_files != {
        index_path.relative_to(snapshot_root).as_posix(),
        metadata_path.relative_to(snapshot_root).as_posix(),
        response_path.relative_to(snapshot_root).as_posix(),
    }:
        raise ValueError("HUC raw snapshot contains unindexed files")

    raw_rows = _parse_usgs_rdb(payload)
    required = {
        "site_no", "station_nm", "dec_lat_va", "dec_long_va", "huc_cd"
    }
    if any(not required <= set(row) for row in raw_rows):
        raise ValueError("HUC raw response lacks required site columns")
    raw_by_site = {row["site_no"].strip(): row for row in raw_rows}
    if len(raw_by_site) != len(raw_rows) or set(raw_by_site) != set(expected_sites):
        raise ValueError("HUC raw response station keys differ from the registry")
    huc_by_site = {row["site_no"].strip(): row for row in huc_rows}
    for site_no, derived in huc_by_site.items():
        raw = raw_by_site[site_no]
        raw_huc = raw["huc_cd"].strip()
        if (
            derived.get("station_nm", "") != raw["station_nm"].strip()
            or derived.get("huc_cd", "").strip() != raw_huc
            or derived.get("huc2", "").strip() != raw_huc[:2]
            or not _same_decimal(raw["dec_lat_va"], derived.get("dec_lat_va", ""))
            or not _same_decimal(raw["dec_long_va"], derived.get("dec_long_va", ""))
            or not _same_decimal(raw.get("drain_area_va", ""), derived.get("drain_area_va", ""))
        ):
            raise ValueError(f"released HUC row cannot be replayed from raw NWIS: {site_no}")


def verify_canonical_huc_closure(root: Path) -> None:
    """Prove that the only released HUC table is the panel's frozen generation."""
    spec_path = root / "data_usgs" / "frozen_panel_v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    station = spec.get("station_registry", {})
    huc = station.get("huc_metadata", {}) if isinstance(station, dict) else {}
    expected = {
        "registry": (station.get("path"), station.get("sha256")),
        "huc": (huc.get("source_path"), huc.get("source_sha256")),
        "provenance": (huc.get("provenance_path"), huc.get("provenance_sha256")),
    }
    paths: dict[str, Path] = {}
    for label, (relative, digest) in expected.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"frozen panel lacks {label} HUC binding")
        path = (spec_path.parent / relative).resolve()
        if spec_path.parent.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"frozen panel {label} HUC path escapes or is missing")
        if sha256_file(path) != digest:
            raise ValueError(f"frozen panel {label} HUC checksum mismatch")
        paths[label] = path

    registry_rows = _csv_rows(paths["registry"])
    huc_rows = _csv_rows(paths["huc"])
    registry_by_site = {row.get("site_no", "").strip(): row for row in registry_rows}
    huc_by_site = {row.get("site_no", "").strip(): row for row in huc_rows}
    if "" in registry_by_site or "" in huc_by_site:
        raise ValueError("canonical HUC evidence has an empty site_no")
    if len(registry_by_site) != len(registry_rows) or len(huc_by_site) != len(huc_rows):
        raise ValueError("canonical HUC evidence has duplicate site_no keys")
    if set(registry_by_site) != set(huc_by_site):
        raise ValueError("canonical registry and HUC snapshot contain different stations")
    for site in registry_by_site:
        registry_huc = registry_by_site[site].get("huc2", "").strip().zfill(2)
        source_huc = huc_by_site[site].get("huc2", "").strip().zfill(2)
        if registry_huc != source_huc:
            raise ValueError(f"canonical HUC2 mismatch for site {site}")
    _verify_raw_huc_derivation(
        root,
        spec=spec,
        registry_rows=registry_rows,
        huc_rows=huc_rows,
        huc_path=paths["huc"],
        provenance_path=paths["provenance"],
    )


def verify_archive(
    archive_path: Path,
    *,
    run_data_smoke: bool = False,
    run_trusted_replay: bool = True,
) -> str:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    verify_checksum_sidecar(archive_path)
    _preflight_zip_container(archive_path)

    with tempfile.TemporaryDirectory(prefix="thermoroute-clean-room-") as tmp:
        destination = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            members = normalised_members(archive)
            validate_members(members)
            _extract_archive_safely(archive, destination)

        root = destination / ARCHIVE_ROOT
        profile = verify_release_profile(
            root, members, run_trusted_replay=run_trusted_replay
        )
        verify_canonical_huc_closure(root)
        manifest = root / "outputs" / "manifest.json"
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if document.get("schema_version") != "thermoroute.provenance-manifest.v2":
            raise ValueError("release carries a legacy or unsupported manifest")
        git = document.get("git", {})
        if not git.get("available") or not git.get("commit") or not git.get("tree"):
            raise ValueError("release manifest is not bound to an origin Git revision")

        run_checked([
            sys.executable, "scripts/14_manifest.py", "--root", str(root),
            "--manifest", str(manifest), "--check", "--no-git",
        ], cwd=root)

        if run_data_smoke:
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            run_checked([sys.executable, "scripts/01_prepare_data.py"], cwd=root, env=env)
            processed = root / "data" / "processed" / "panel.parquet"
            if not processed.is_file() or processed.stat().st_size == 0:
                raise RuntimeError("release data smoke did not create processed panel")
            # Verify the frozen USGS panel, stable site_no registry, and their
            # HUC raw-snapshot dependencies without performing a network call.
            run_checked([
                sys.executable,
                "-c",
                (
                    "from thermoroute.evidence import FrozenPanelSpec; "
                    "e=FrozenPanelSpec.load().verify(); "
                    "assert e['station_count']==120 and "
                    "e['site_primary_key']=='site_no'; print('USGS evidence OK')"
                ),
            ], cwd=root, env=env)
        return profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--run-data-smoke", action="store_true",
                        help="also execute stage 01 inside the extracted archive")
    parser.add_argument(
        "--materialize-profile",
        type=Path,
        metavar="STAGE_ROOT",
        help="build-time helper: copy the selected profile closure into a stage",
    )
    parser.add_argument(
        "--materialize-claim-audit",
        type=Path,
        metavar="STAGE_ROOT",
        help="build-time helper: run and bind the fixed claim validator",
    )
    parser.add_argument(
        "--materialize-git-history",
        type=Path,
        metavar="STAGE_ROOT",
        help="build-time helper: bind a self-contained compute/history Git bundle",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--profile", choices=RELEASE_PROFILES)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument(
        "--check-postopen-dirt",
        action="store_true",
        help="audit the exact authorization-derived Git dirt allowed after opening",
    )
    args = parser.parse_args()
    try:
        if args.materialize_git_history is not None:
            if args.archive is not None or args.materialize_profile is not None:
                raise ValueError("Git-history materialization is a standalone operation")
            if args.profile is None or args.source_root is None:
                raise ValueError("Git-history materialization requires --profile/--source-root")
            evidence = materialize_git_history_evidence(
                args.source_root, args.materialize_git_history, args.profile
            )
            print(json.dumps({
                "profile": evidence["profile"],
                "bundle": evidence["bundle"]["path"],
                "compute_commit": evidence["compute_commit"],
                "manuscript_commit": evidence["manuscript_commit"],
            }, indent=2))
            return 0
        if args.materialize_claim_audit is not None:
            if args.archive is not None or args.materialize_profile is not None:
                raise ValueError("claim-audit materialization is a standalone operation")
            if args.profile is None:
                raise ValueError("claim-audit materialization requires --profile")
            audit = materialize_claim_audit(
                args.materialize_claim_audit, args.profile
            )
            print(json.dumps({
                "profile": audit["profile"],
                "claim_violations": audit["violation_count"],
                "audit": str(args.materialize_claim_audit / CLAIM_AUDIT_PATH),
            }, indent=2))
            return 0
        if args.check_postopen_dirt:
            if args.archive is not None or args.materialize_profile is not None:
                raise ValueError("--check-postopen-dirt is a standalone operation")
            if args.source_root is None or args.authorization is None:
                raise ValueError("post-opening dirt audit requires --source-root/--authorization")
            policy = validate_postopen_git_dirt(
                args.source_root, args.authorization
            )
            print(json.dumps(policy, indent=2, sort_keys=True))
            return 0
        if args.materialize_profile is not None:
            if args.archive is not None:
                raise ValueError("archive and --materialize-profile are mutually exclusive")
            if args.source_root is None or args.profile is None:
                raise ValueError("profile materialization requires --source-root and --profile")
            document = materialize_release_profile(
                args.source_root,
                args.materialize_profile,
                args.profile,
                authorization_path=args.authorization,
            )
            print(json.dumps({
                "profile": document["profile"],
                "status": document["status"],
                "marker": str(args.materialize_profile / PROFILE_MARKER),
            }, indent=2))
            return 0
        if args.archive is None:
            raise ValueError("an archive is required")
        profile = verify_archive(
            args.archive,
            run_data_smoke=args.run_data_smoke,
            run_trusted_replay=True,
        )
    except Exception as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"release OK [{profile}]: {args.archive}"
          f"{' + data smoke' if args.run_data_smoke else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
