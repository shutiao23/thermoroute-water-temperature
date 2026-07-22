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
from decimal import Decimal, InvalidOperation
from fnmatch import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
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

REQUIRED_MEMBERS = {
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    REPRODUCIBILITY_LOCK,
    ".zenodo.json",
    ".github/workflows/ci.yml",
    "src/thermoroute/config.py",
    "scripts/run_all.sh",
    "scripts/14_manifest.py",
    "scripts/deterministic_zip.py",
    "scripts/verify_release.py",
    "tests/test_leakage.py",
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
    "approved_target_sensitivity",
    "spatial_sensitivity",
    "probabilistic_evaluation",
    "temporal_predictions",
    "external_predictions",
    "statistics",
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
    "approved_target_sensitivity",
    "spatial_sensitivity",
    "probabilistic_evaluation",
    "temporal_predictions",
    "external_predictions",
    "statistics",
    "report",
}

REQUIRED_POSTOPEN_CATEGORIES = {
    "canonical_development",
    "authorization",
    "registries",
    "candidate_evidence",
    "model_suite",
    "model_bundles",
    "prelabel_inputs",
    "raw_meteorology",
    "opening_intent",
    "raw_nwis",
    "normalized_outcomes",
    "trusted_predictions",
    "availability",
    "sensitivity_audits",
    "probabilistic_evaluation",
    "statistics",
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
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    candidates = [(root / value).resolve()]
    if base is not None:
        candidate = (base / value).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    root = root.resolve()
    inside = [
        candidate for candidate in candidates
        if candidate == root or root in candidate.parents
    ]
    if not inside:
        raise ValueError(f"{label} path escapes the release root")
    existing = [candidate for candidate in inside if candidate.is_file() or candidate.is_dir()]
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
    root, path = root.resolve(), path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{category} artifact escapes the release root")
    if path.is_symlink():
        raise ValueError(f"{category} artifact is a symlink")
    if path.is_file():
        categories.setdefault(category, set()).add(path)
        return [path]
    if not path.is_dir():
        raise ValueError(f"{category} artifact is absent: {path}")
    files = []
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"{category} directory contains a symlink: {member}")
        if member.is_file():
            categories.setdefault(category, set()).add(member.resolve())
            files.append(member.resolve())
        elif not member.is_dir():
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
    if not isinstance(state, Mapping) or not REQUIRED_STATE_PATHS <= set(state):
        missing = sorted(REQUIRED_STATE_PATHS - set(state or {}))
        raise ValueError(f"authorization lacks canonical state paths: {missing}")
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
        "approved_target_sensitivity": f"{base}/trusted/approved_target_sensitivity_v1.json",
        "spatial_sensitivity": f"{base}/trusted/spatial_sensitivity_v1.json",
        "probabilistic_evaluation": f"{base}/trusted/probabilistic_evaluation_v1.json",
        "temporal_predictions": f"{base}/trusted/temporal_predictions_v1.parquet",
        "external_predictions": f"{base}/trusted/external_predictions_v1.parquet",
        "statistics": f"{base}/trusted/statistics_v1.json",
        "report": f"{base}/trusted/report_v1.md",
        "receipt": f"{base}/opening_receipt_v1.json",
        "receipt_sha256": f"{base}/opening_receipt_v1.sha256",
    }
    wrong = {key: state.get(key) for key, value in expected.items() if state.get(key) != value}
    if wrong:
        raise ValueError(f"authorization state paths leave the canonical namespace: {wrong}")
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
    whitelist = [".zenodo.json", "README.md", "paper/**"]

    def allowed_document(relative: str) -> bool:
        return relative in {"README.md", ".zenodo.json"} or relative.startswith("paper/")

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
        )

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


def _gather_postopen_categories(
    root: Path, authorization_path: Path
) -> tuple[dict[str, set[Path]], dict[str, Any], dict[str, str]]:
    root, authorization_path = root.resolve(), authorization_path.resolve()
    categories = _canonical_categories(root)
    authorization, state = _validate_authorization_structure(root, authorization_path)
    _add_path(root, categories, "authorization", authorization_path)
    work_order = _resolve_release_path(root, state["work_order"], label="acquisition work order")
    _add_path(root, categories, "authorization", work_order)

    protocol = _add_binding(
        root, categories, "authorization", authorization.get("protocol"),
        label="authorized protocol",
    )
    if protocol.suffix != ".json":
        raise ValueError("authorized protocol is not machine-readable JSON")

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
        or replay.get("confirmation_period_read") is not False
    ):
        raise ValueError("authorized development replay receipt is stale or malformed")

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
    work_order = _load_json(work_order, label="acquisition work order")
    work_order_stable = dict(work_order)
    work_order_self = work_order_stable.pop("work_order_self_sha256", None)
    if not isinstance(work_order_self, str) or work_order_self != _sha256_json(work_order_stable):
        raise ValueError("acquisition work-order self hash is inconsistent")
    preflight = receipt.get("preflight_attestation")
    if (
        not isinstance(preflight, Mapping)
        or receipt.get("preflight_attestation_sha256") != _sha256_json(preflight)
        or intent.get("preflight_attestation_sha256") != _sha256_json(preflight)
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
        or acquisition.get("producer_role") != "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS"
    ):
        raise ValueError("acquisition manifest identity or raw-only role changed")
    raw_root = _resolve_release_path(root, state["raw_nwis_root"], label="raw NWIS root")
    _add_path(root, categories, "raw_nwis", raw_root)
    for key in ("raw_nwis_snapshot_index", "request_map"):
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
        "approved_target_sensitivity": "sensitivity_audits",
        "spatial_sensitivity": "sensitivity_audits",
        "probabilistic_evaluation": "probabilistic_evaluation",
        "temporal_predictions": "trusted_predictions",
        "external_predictions": "trusted_predictions",
        "statistics": "statistics",
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
        "approved_target_sensitivity": state["approved_target_sensitivity"],
        "spatial_sensitivity": state["spatial_sensitivity"],
        "probabilistic_evaluation": state["probabilistic_evaluation"],
        "temporal_predictions": state["temporal_predictions"],
        "external_predictions": state["external_predictions"],
        "statistics": state["statistics"],
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
    document = {
        "format": PROFILE_FORMAT,
        "profile": POSTOPEN_PROFILE,
        "status": POSTOPEN_PROFILE,
        "supports_route_a_confirmatory_conclusions": True,
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
    patterns = registry_document.get("documents")
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
    marker = _load_json(marker_path, label="release profile marker")
    marker["claim_validation"] = _binding_for(stage_root, audit_path)
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return audit


def _verify_claim_audit(root: Path, marker: Mapping[str, Any], profile: str) -> None:
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
    patterns = registry_document.get("documents")
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
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root,
        text=True, capture_output=True, check=False,
    )
    if head.returncode:
        raise ValueError("cannot resolve manuscript Git HEAD for release bundle")
    bundle = stage_root / GIT_BUNDLE_PATH
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        raise ValueError("refusing to replace staged Git history evidence")
    result = subprocess.run(
        ["git", "bundle", "create", str(bundle), "HEAD"],
        cwd=source_root, text=True, capture_output=True, check=False,
    )
    if result.returncode or not bundle.is_file():
        raise ValueError(f"cannot create release Git bundle: {(result.stderr or result.stdout).strip()}")
    protocol = _load_json(
        stage_root / "protocols" / "route_a_confirmatory_v1.json",
        label="Route-A protocol",
    )
    authoritative = str(protocol.get("authoritative_protocol_commit", ""))
    protocol_markdown = "protocols/route_a_confirmatory_protocol.md"
    original = subprocess.run(
        ["git", "show", f"{authoritative}:{protocol_markdown}"],
        cwd=source_root, capture_output=True, check=False,
    )
    if original.returncode:
        raise ValueError("cannot recover original sealed protocol from Git history")
    original_sha = hashlib.sha256(original.stdout).hexdigest()
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
        compute = str(marker["authorized_worktree_dirt_policy"]["compute_commit"])
        manuscript = str(marker["authorized_worktree_dirt_policy"]["manuscript_commit"])
    else:
        compute = head.stdout.strip()
        manuscript = compute
    evidence = {
        "format": "thermoroute.route-a-git-history-evidence.v1",
        "profile": profile,
        "bundle": _binding_for(stage_root, bundle),
        "compute_commit": compute,
        "manuscript_commit": manuscript,
        "authoritative_protocol_commit": authoritative,
        "sealed_protocol_blob": {
            "commit": authoritative,
            "path": protocol_markdown,
            "sha256": original_sha,
            "bytes": len(original.stdout),
        },
        "external_timestamp_or_public_preregistration": False,
    }
    marker["git_history_evidence"] = evidence
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return evidence


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
    ]
    if any(len(commit) != 40 for commit in commits):
        raise ValueError("Git history evidence has a malformed commit")
    with tempfile.TemporaryDirectory(prefix="thermoroute-release-git-") as name:
        bare = Path(name) / "audit.git"
        run_checked(["git", "init", "--bare", "-q", str(bare)], cwd=root)
        run_checked(["git", "-C", str(bare), "bundle", "verify", str(bundle)], cwd=root)
        run_checked([
            "git", "-C", str(bare), "fetch", "-q", str(bundle),
            "HEAD:refs/heads/release-evidence",
        ], cwd=root)
        for commit in commits:
            run_checked(
                ["git", "-C", str(bare), "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=root,
            )
        bundled_head = subprocess.run(
            ["git", "-C", str(bare), "rev-parse", "refs/heads/release-evidence"],
            text=True, capture_output=True, check=False,
        )
        if bundled_head.returncode or bundled_head.stdout.strip() != commits[1]:
            raise ValueError("Git bundle HEAD differs from manuscript commit")
        for ancestor, label in (
            (commits[0], "compute"),
            (commits[2], "authoritative protocol"),
        ):
            relation = subprocess.run(
                [
                    "git", "-C", str(bare), "merge-base", "--is-ancestor",
                    ancestor, commits[1],
                ],
                capture_output=True, check=False,
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
            changed = subprocess.run(
                [
                    "git", "-C", str(bare), "diff", "--name-status", "--no-renames",
                    "-z", f"{commits[0]}..{commits[1]}",
                ],
                capture_output=True, check=False,
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
                        relative in {"README.md", ".zenodo.json"}
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
                blob = subprocess.run(
                    ["git", "-C", str(bare), "show", f"{commits[1]}:{relative}"],
                    capture_output=True, check=False,
                )
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
        blob = subprocess.run(
            [
                "git", "-C", str(bare), "show",
                f"{sealed.get('commit')}:{sealed.get('path')}",
            ],
            capture_output=True, check=False,
        )
        if (
            blob.returncode
            or hashlib.sha256(blob.stdout).hexdigest() != sealed.get("sha256")
            or len(blob.stdout) != sealed.get("bytes")
            or sealed.get("commit") != commits[2]
            or sealed.get("path") != "protocols/route_a_confirmatory_protocol.md"
        ):
            raise ValueError("sealed protocol blob cannot be replayed from Git bundle")
        if profile == POSTOPEN_PROFILE:
            authorization = _load_json(
                root / str(marker["authorization"]["path"]),
                label="opening authorization",
            )
            if authorization.get("protocol", {}).get(
                "authoritative_markdown_sha256"
            ) != sealed.get("sha256"):
                raise ValueError("sealed protocol blob differs from authorization")


def normalised_members(archive: zipfile.ZipFile) -> set[str]:
    """Return paths below the single archive root after security validation."""
    members: set[str] = set()
    infos = archive.infolist()
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
        "committed_document_whitelist": [".zenodo.json", "README.md", "paper/**"],
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
        if not (path in {"README.md", ".zenodo.json"} or path.startswith("paper/")):
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
            marker.get("supports_route_a_confirmatory_conclusions") is not False
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
        _verify_claim_audit(root, marker, profile)
        if run_trusted_replay:
            _verify_git_history_evidence(root, marker, profile)
        return profile

    if (
        marker.get("supports_route_a_confirmatory_conclusions") is not True
        or marker.get("labels_included") is not True
        or marker.get("fully_hashed_lock_role") != HASHED_LOCK_ROLE
    ):
        raise ValueError("opened-complete marker does not acknowledge opened labels")
    authorization_binding = marker.get("authorization")
    authorization = _add_binding(
        root, {}, "authorization", authorization_binding,
        label="release authorization",
    )
    categories, document, state = _gather_postopen_categories(root, authorization)
    if (
        marker.get("opening_id") != document.get("opening_id")
        or marker.get("state_namespace") != state.get("namespace")
        or set(marker.get("artifact_closure", {})) != REQUIRED_POSTOPEN_CATEGORIES
    ):
        raise ValueError("opened-complete marker identity/categories are inconsistent")
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
    _verify_claim_audit(root, marker, profile)
    if run_trusted_replay:
        _verify_git_history_evidence(root, marker, profile)
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

    with tempfile.TemporaryDirectory(prefix="thermoroute-clean-room-") as tmp:
        destination = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            members = normalised_members(archive)
            validate_members(members)
            archive.extractall(destination)

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
