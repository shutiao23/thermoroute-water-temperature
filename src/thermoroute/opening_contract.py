"""Lightweight stdlib-only contract for the isolated Route-A acquisition child."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
from typing import Any, Mapping


AUTHORIZATION_FORMAT = "thermoroute.route-a-opening-authorization.v1"
ACQUISITION_WORK_ORDER_FORMAT = "thermoroute.route-a-acquisition-work-order.v1"
ACQUISITION_MANIFEST_FORMAT = "thermoroute.route-a-opened-inputs.v1"
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
INTENT_FORMAT = "thermoroute.route-a-opening-intent.v1"
MODEL_MATRIX_AMENDMENT_FORMAT = (
    "thermoroute.route-a-model-matrix-amendment.v1"
)
MODEL_MATRIX_AMENDMENT_STATUS = "FROZEN_PRELABEL_OUTCOME_FREE"
MODEL_MATRIX_AMENDMENT_ID = "route-a-prelabel-model-matrix-replication-017"
MODEL_MATRIX_AMENDMENT_RELATIVE = (
    "protocols/route_a_model_matrix_amendment_v1.json"
)
MODEL_MATRIX_AMENDMENT_SEAL_RELATIVE = (
    "protocols/route_a_model_matrix_amendment_seal_v1.json"
)
AUTHORIZATION_TOP_LEVEL_FIELDS = frozenset({
    "format",
    "status",
    "protocol",
    "registries",
    "model_suite",
    "development_replay",
    "prelabel_chronology",
    "inference_amendment",
    "probability_metric_erratum",
    "model_matrix_amendment",
    "inference_gate",
    "outcome_qc_policy",
    "temporal_coverage_policy",
    "actual_inputs",
    "actual_feature_order",
    "required_models",
    "statistics_contract_sha256",
    "runtime",
    "fixed_code",
    "source",
    "acquisition_plan",
    "state_paths",
    "opening_id",
    "created_at_utc",
    "authorization_self_sha256",
})
MODEL_MATRIX_AMENDMENT_BINDING_FIELDS = frozenset({
    "path",
    "sha256",
    "format",
    "status",
    "amendment_id",
    "seal",
    "amendment_document_commit",
})
ACQUISITION_WORK_ORDER_FIELDS = frozenset({
    "format",
    "opening_id",
    "authorization_path",
    "authorization_sha256",
    "source_tree_sha256",
    "runtime_sha256",
    "fixed_code_sha256",
    "model_matrix_amendment_seal_sha256",
    "acquisition_plan",
    "state_paths",
    "site_registries",
    "work_order_self_sha256",
})
INTENT_FIELDS = frozenset({
    "format",
    "status",
    "opening_id",
    "authorization_sha256",
    "preflight_attestation_sha256",
    "work_order_self_sha256",
    "work_order_file_sha256",
    "fixed_code_sha256",
    "runtime_sha256",
    "model_matrix_amendment_seal_sha256",
    "trusted_validator",
    "started_at_utc",
    "maximum_openings",
    "retry_after_failure_allowed",
    "same_opening_transport_resume_allowed",
    "intent_self_sha256",
})
RECEIPT_FIELDS = frozenset({
    "format",
    "status",
    "opening_id",
    "authorization_sha256",
    "model_matrix_amendment_seal_sha256",
    "intent_sha256",
    "work_order_sha256",
    "preflight_attestation",
    "preflight_attestation_sha256",
    "trusted_validator",
    "fixed_code",
    "authorized_runtime",
    "completion_environment",
    "python_hash_seed_interpreter_effect",
    "completed_at_utc",
    "opening_count",
    "maximum_openings",
    "retry_after_failure_allowed",
    "same_opening_transport_resume_allowed",
    "transport_recovery",
    "all_predeclared_models_reported",
    "reported_models",
    "artifacts",
    "trusted_prediction_hashes",
    "formal_tests",
    "temporal_coverage_audit",
    "state_paths",
    "release_bindings",
    "intent_self_sha256",
    "security_boundary",
    "receipt_self_sha256",
})
RAW_PREFLIGHT_TRANSCRIPT_FORMAT = (
    "thermoroute.route-a-raw-preflight-transcript.v1"
)
RAW_PREFLIGHT_TRANSCRIPT_FIELDS = frozenset({
    "format",
    "status",
    "challenge",
    "opening_id",
    "authorization_path",
    "authorization_sha256",
    "authorization_self_sha256",
    "work_order_path",
    "work_order_sha256",
    "work_order_self_sha256",
    "intent_path",
    "intent_sha256",
    "intent_self_sha256",
    "preflight_attestation",
    "preflight_attestation_sha256",
    "trusted_validator",
    "state_namespace",
    "resume_phase",
    "raw_transport_resume_allowed",
    "network_free_acquisition_finalization_allowed",
    "source_tree_sha256",
    "runtime_sha256",
    "fixed_code_sha256",
    "model_matrix_amendment_sha256",
    "model_matrix_amendment_seal_sha256",
    "development_registry_sha256",
    "external_registry_sha256",
    "validator_entrypoint_path",
    "validator_entrypoint_sha256",
    "outcome_values_parsed",
    "network_used",
})
RAW_PREFLIGHT_ATTESTATION_FIELDS = frozenset({
    "authorization_sha256",
    "opening_id",
    "protocol_sha256",
    "development_registry_sha256",
    "external_registry_sha256",
    "external_lock_sha256",
    "model_suite_sha256",
    "development_replay_sha256",
    "prelabel_chronology_sha256",
    "inference_amendment_sha256",
    "inference_amendment_seal_sha256",
    "model_matrix_amendment_sha256",
    "model_matrix_amendment_seal_sha256",
    "model_matrix_amendment_id",
    "model_matrix_amendment_status",
    "inference_gate_sha256",
    "inference_gate_status",
    "inference_claim_eligible",
    "outcome_qc_policy_sha256",
    "temporal_coverage_policy_sha256",
    "prelabel_inputs_sha256",
    "actual_feature_order",
    "required_models",
    "source_tree_sha256",
    "runtime_sha256",
    "requirements_lock_sha256",
    "hashed_requirements_lock_sha256",
    "golden_inference_sha256",
    "fixed_code_sha256",
    "state_namespace",
})
MAX_RAW_PREFLIGHT_TRANSCRIPT_BYTES = 1024 * 1024
MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES = 32 * 1024 * 1024
TRUSTED_STATE_KEYS = (
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
)
RAW_DERIVED_STATE_KEYS = (
    "acquisition_manifest",
    "acquisition_request_map",
    "temporal_outcomes",
    "external_outcomes",
)
RAW_ACQUISITION_FORBIDDEN_STATE_KEYS = (
    *RAW_DERIVED_STATE_KEYS,
    *TRUSTED_STATE_KEYS,
    "receipt",
    "receipt_sha256",
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
_REQUIRED_ACQUISITION_MODULES = {
    "thermoroute.opening_contract": "src/thermoroute/opening_contract.py",
    "thermoroute.outcome_acquisition": (
        "src/thermoroute/outcome_acquisition.py"
    ),
    "thermoroute.provenance": "src/thermoroute/provenance.py",
    "thermoroute.usgs": "src/thermoroute/usgs.py",
}


class AcquisitionContractError(RuntimeError):
    """The isolated acquisition input does not match its frozen authorization."""


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
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


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AcquisitionContractError(
                f"acquisition contract contains duplicate JSON key: {key}"
            )
        value[key] = item
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _model_matrix_amendment_seal_sha256(
    authorization: Mapping[str, Any],
) -> str:
    """Validate the raw child's exact outer governance binding."""
    binding = authorization.get("model_matrix_amendment")
    if (
        not isinstance(binding, Mapping)
        or set(binding) != set(MODEL_MATRIX_AMENDMENT_BINDING_FIELDS)
        or binding.get("path") != MODEL_MATRIX_AMENDMENT_RELATIVE
        or not _is_sha256(binding.get("sha256"))
        or binding.get("format") != MODEL_MATRIX_AMENDMENT_FORMAT
        or binding.get("status") != MODEL_MATRIX_AMENDMENT_STATUS
        or binding.get("amendment_id") != MODEL_MATRIX_AMENDMENT_ID
        or not isinstance(binding.get("amendment_document_commit"), str)
        or re.fullmatch(
            r"[0-9a-f]{40}", str(binding.get("amendment_document_commit"))
        )
        is None
    ):
        raise AcquisitionContractError(
            "opening authorization model-matrix amendment binding changed"
        )
    seal = binding.get("seal")
    if (
        not isinstance(seal, Mapping)
        or set(seal) != {"path", "sha256"}
        or seal.get("path") != MODEL_MATRIX_AMENDMENT_SEAL_RELATIVE
        or not _is_sha256(seal.get("sha256"))
    ):
        raise AcquisitionContractError(
            "opening authorization model-matrix amendment seal changed"
        )
    return str(seal["sha256"])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_immutable_atomic_final(path: Path, *, label: str) -> None:
    metadata = os.lstat(path)
    parent = os.lstat(path.parent)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != parent.st_dev
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
    ):
        raise AcquisitionContractError(
            f"{label} is not one immutable atomic final file"
        )


def assert_no_symlink_components(
    root: str | Path,
    path: str | Path,
    *,
    require_file: bool = False,
) -> Path:
    """Validate one lexical repository path with an lstat-only component walk."""
    canonical_root = Path(root).resolve()
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical != canonical_root and canonical_root not in lexical.parents:
        raise AcquisitionContractError(
            "acquisition path escapes the lexical repository root"
        )
    relative = lexical.relative_to(canonical_root)
    current = canonical_root
    missing_component = False
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            missing_component = True
            continue
        if missing_component:
            raise AcquisitionContractError(
                "acquisition path has an impossible missing-parent topology"
            )
        if stat.S_ISLNK(status.st_mode):
            raise AcquisitionContractError(
                f"acquisition path contains a symlink: {current}"
            )
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise AcquisitionContractError(
                f"acquisition path parent is not a directory: {current}"
            )
    if lexical.resolve(strict=False) != lexical:
        raise AcquisitionContractError(
            "acquisition path lexical and resolved identities differ"
        )
    if require_file:
        try:
            status = os.lstat(lexical)
        except FileNotFoundError as exc:
            raise AcquisitionContractError(
                "acquisition contract file is absent"
            ) from exc
        if not stat.S_ISREG(status.st_mode):
            raise AcquisitionContractError(
                "acquisition contract file is not a regular file"
            )
    return lexical


def _inside(root: Path, relative: object, *, file: bool = False) -> Path:
    raw = Path(str(relative))
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise AcquisitionContractError("acquisition contract path must be relative")
    return assert_no_symlink_components(root, root / raw, require_file=file)


def _source_inventory(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for pattern in SOURCE_INVENTORY_PATTERNS:
        for candidate in root.glob(pattern):
            if "__pycache__" in candidate.parts:
                continue
            try:
                status = os.lstat(candidate)
            except FileNotFoundError as exc:
                raise AcquisitionContractError(
                    "source inventory changed during acquisition validation"
                ) from exc
            if stat.S_ISLNK(status.st_mode):
                raise AcquisitionContractError(
                    f"source inventory contains a symlink: {candidate}"
                )
            if not stat.S_ISREG(status.st_mode):
                continue
            lexical = assert_no_symlink_components(
                root, candidate, require_file=True
            )
            files[lexical.relative_to(root).as_posix()] = sha256_file(lexical)
    return dict(sorted(files.items()))


def validate_frozen_source_identity(
    *,
    root: str | Path,
    authorization: Mapping[str, Any],
) -> dict[str, str]:
    """Replay the complete frozen source tree and every loaded project module."""
    canonical_root = Path(root).resolve()
    source = authorization.get("source")
    if not isinstance(source, Mapping):
        raise AcquisitionContractError("authorization lacks frozen source identity")
    frozen = source.get("source_inventory")
    if not isinstance(frozen, Mapping) or not all(
        isinstance(path, str) and isinstance(digest, str)
        for path, digest in frozen.items()
    ):
        raise AcquisitionContractError("authorization source inventory is malformed")
    frozen_inventory = dict(frozen)
    current_inventory = _source_inventory(canonical_root)
    if current_inventory != frozen_inventory:
        changed = sorted(
            path
            for path in set(current_inventory) | set(frozen_inventory)
            if current_inventory.get(path) != frozen_inventory.get(path)
        )
        raise AcquisitionContractError(
            "source tree differs from opening authorization: "
            f"{changed[:20]}"
        )
    digest = _sha256_json(current_inventory)
    if digest != source.get("source_tree_sha256"):
        raise AcquisitionContractError(
            "authorization source-tree digest is inconsistent"
        )

    loaded_project_modules: dict[str, str] = {}
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "thermoroute" and not module_name.startswith(
            "thermoroute."
        ):
            continue
        loaded_file = getattr(module, "__file__", None)
        if not loaded_file:
            raise AcquisitionContractError(
                f"loaded project module lacks a realpath: {module_name}"
            )
        lexical = Path(os.path.abspath(os.fspath(loaded_file)))
        try:
            lexical = assert_no_symlink_components(
                canonical_root, lexical, require_file=True
            )
        except AcquisitionContractError as exc:
            raise AcquisitionContractError(
                f"loaded project module escapes frozen source: {module_name}"
            ) from exc
        relative = lexical.relative_to(canonical_root).as_posix()
        if (
            relative not in frozen_inventory
            or sha256_file(lexical) != frozen_inventory[relative]
        ):
            raise AcquisitionContractError(
                f"loaded project module bytes changed: {module_name}"
            )
        loaded_project_modules[module_name] = relative
    for module_name, relative in _REQUIRED_ACQUISITION_MODULES.items():
        if loaded_project_modules.get(module_name) != relative:
            raise AcquisitionContractError(
                f"required acquisition module identity changed: {module_name}"
            )
    return current_inventory


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(f"cannot read acquisition contract: {path}") from exc
    if not isinstance(value, dict):
        raise AcquisitionContractError("acquisition contract must be a JSON object")
    if raw != _canonical_json_bytes(value):
        raise AcquisitionContractError(
            f"acquisition contract is not exact canonical JSON: {path}"
        )
    return value


def _registry_sites(root: Path, binding: Mapping[str, Any]) -> list[str]:
    path = _inside(root, binding.get("path"), file=True)
    if sha256_file(path) != binding.get("sha256"):
        raise AcquisitionContractError("acquisition registry checksum changed")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "site_no" not in reader.fieldnames:
            raise AcquisitionContractError("acquisition registry lacks site_no")
        sites = [str(row["site_no"]).strip() for row in reader]
    if not sites or any(not site for site in sites) or len(sites) != len(set(sites)):
        raise AcquisitionContractError("acquisition registry site IDs are invalid")
    return sorted(sites)


def _validate_acquisition_environment() -> None:
    """Require the complete non-inheriting child environment allowlist."""
    temporary_root_raw = os.environ.get("TMPDIR", "")
    temporary_root = Path(temporary_root_raw)
    if (
        not temporary_root_raw
        or not temporary_root.is_absolute()
        or not temporary_root.is_dir()
    ):
        raise AcquisitionContractError(
            "raw acquisition TMPDIR is not an existing absolute directory"
        )
    expected_environment = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": str(temporary_root.resolve()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }
    if dict(os.environ) != expected_environment:
        added = sorted(set(os.environ) - set(expected_environment))
        missing = sorted(set(expected_environment) - set(os.environ))
        changed = sorted(
            key for key in set(os.environ) & set(expected_environment)
            if os.environ[key] != expected_environment[key]
        )
        raise AcquisitionContractError(
            "raw acquisition environment differs from the complete allowlist "
            f"(added={added}, missing={missing}, changed={changed})"
        )


def validate_acquisition_work_order(
    work_order_path: str | Path,
    *,
    root: str | Path,
    entrypoint_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Verify auth/work-order/entrypoint identity without importing scorer code."""
    root = Path(root).resolve()
    work_order_path = assert_no_symlink_components(
        root,
        Path(os.path.abspath(os.fspath(work_order_path))),
        require_file=True,
    )
    _require_immutable_atomic_final(
        work_order_path, label="acquisition work order"
    )
    work_order = _read_json(work_order_path)
    if work_order.get("format") != ACQUISITION_WORK_ORDER_FORMAT:
        raise AcquisitionContractError("unsupported acquisition work-order format")
    if set(work_order) != set(ACQUISITION_WORK_ORDER_FIELDS):
        raise AcquisitionContractError("acquisition work-order schema changed")
    self_hashed = dict(work_order)
    self_digest = self_hashed.pop("work_order_self_sha256", None)
    if self_digest != _sha256_json(self_hashed):
        raise AcquisitionContractError("acquisition work-order self hash changed")

    authorization_path = _inside(root, work_order.get("authorization_path"), file=True)
    _require_immutable_atomic_final(
        authorization_path, label="opening authorization"
    )
    authorization = _read_json(authorization_path)
    if authorization.get("format") != AUTHORIZATION_FORMAT:
        raise AcquisitionContractError("unsupported opening authorization")
    if set(authorization) != set(AUTHORIZATION_TOP_LEVEL_FIELDS):
        raise AcquisitionContractError("opening authorization schema changed")
    self_hashed_authorization = dict(authorization)
    auth_self_digest = self_hashed_authorization.pop("authorization_self_sha256", None)
    if auth_self_digest != _sha256_json(self_hashed_authorization):
        raise AcquisitionContractError("opening authorization self hash changed")
    model_matrix_amendment_seal_sha256 = (
        _model_matrix_amendment_seal_sha256(authorization)
    )
    source = authorization.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("authorization_path")
        != work_order.get("authorization_path")
    ):
        raise AcquisitionContractError(
            "authorization source policy names another authorization file"
        )
    matrix_binding = authorization["model_matrix_amendment"]
    matrix_path = _inside(root, matrix_binding["path"], file=True)
    seal_binding = matrix_binding["seal"]
    seal_path = _inside(root, seal_binding["path"], file=True)
    if (
        sha256_file(matrix_path) != matrix_binding["sha256"]
        or sha256_file(seal_path) != seal_binding["sha256"]
    ):
        raise AcquisitionContractError(
            "model-matrix amendment or seal bytes changed"
        )
    expected_equal = {
        "opening_id": authorization.get("opening_id"),
        "authorization_sha256": sha256_file(authorization_path),
        "source_tree_sha256": source.get("source_tree_sha256"),
        "runtime_sha256": authorization.get("runtime", {}).get("runtime_sha256"),
        "fixed_code_sha256": authorization.get("fixed_code", {}).get("sha256"),
        "model_matrix_amendment_seal_sha256": (
            model_matrix_amendment_seal_sha256
        ),
        "acquisition_plan": authorization.get("acquisition_plan"),
        "state_paths": authorization.get("state_paths"),
    }
    if any(work_order.get(key) != value for key, value in expected_equal.items()):
        raise AcquisitionContractError("work order differs from opening authorization")
    if authorization.get("status") != "AUTHORIZED_LABELS_STILL_SEALED":
        raise AcquisitionContractError("opening authorization is not sealed/authorized")
    plan = authorization.get("acquisition_plan")
    if (
        not isinstance(plan, Mapping)
        or plan.get("maximum_response_bytes_per_request")
        != MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
    ):
        raise AcquisitionContractError(
            "acquisition response-byte limit differs from authorization"
        )
    validate_frozen_source_identity(root=root, authorization=authorization)

    state = work_order.get("state_paths")
    if not isinstance(state, Mapping):
        raise AcquisitionContractError("work order state paths are malformed")
    resolved = {
        key: _inside(root, value)
        for key, value in state.items() if key != "namespace"
    }
    run_directory = resolved.get("run_directory")
    if run_directory is None or run_directory == root:
        raise AcquisitionContractError(
            "work-order state namespace lacks a confined run directory"
        )
    if any(
        path != run_directory and run_directory not in path.parents
        for path in resolved.values()
    ):
        raise AcquisitionContractError(
            "work-order state path escapes its canonical run namespace"
        )
    if work_order_path != resolved["work_order"]:
        raise AcquisitionContractError("work order is not at its canonical state path")
    intent = _read_json(resolved["intent"])
    _require_immutable_atomic_final(
        resolved["intent"], label="opening intent"
    )
    self_hashed_intent = dict(intent)
    if set(intent) != set(INTENT_FIELDS):
        raise AcquisitionContractError("opening intent schema changed")
    intent_self_digest = self_hashed_intent.pop("intent_self_sha256", None)
    if intent_self_digest != _sha256_json(self_hashed_intent):
        raise AcquisitionContractError("opening intent self hash changed")
    expected_intent = {
        "format": INTENT_FORMAT,
        "status": "OPENING_STARTED_IRREVERSIBLE",
        "opening_id": authorization.get("opening_id"),
        "authorization_sha256": sha256_file(authorization_path),
        "work_order_self_sha256": work_order.get("work_order_self_sha256"),
        "work_order_file_sha256": sha256_file(work_order_path),
        "fixed_code_sha256": authorization.get("fixed_code", {}).get("sha256"),
        "runtime_sha256": authorization.get("runtime", {}).get("runtime_sha256"),
        "model_matrix_amendment_seal_sha256": (
            model_matrix_amendment_seal_sha256
        ),
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
    }
    if any(intent.get(key) != value for key, value in expected_intent.items()):
        raise AcquisitionContractError("opening intent/work-order binding changed")

    registries = authorization.get("registries")
    if not isinstance(registries, Mapping):
        raise AcquisitionContractError("authorization lacks registry bindings")
    expected_sites = {
        "temporal": _registry_sites(root, registries["development"]),
        "external": _registry_sites(root, registries["external"]),
    }
    declared_sites = work_order.get("site_registries")
    if not isinstance(declared_sites, Mapping):
        raise AcquisitionContractError("work order lacks site registries")
    for cohort, binding_key in (("temporal", "development"), ("external", "external")):
        declared = declared_sites.get(cohort)
        if not isinstance(declared, Mapping) or declared.get("sites") != expected_sites[cohort]:
            raise AcquisitionContractError("work-order site registry changed")
        if declared.get("sha256") != registries[binding_key].get("sha256"):
            raise AcquisitionContractError("work-order registry checksum changed")
    if set(expected_sites["temporal"]) & set(expected_sites["external"]):
        raise AcquisitionContractError("work-order cohorts overlap")

    fixed_code = authorization.get("fixed_code", {})
    acquisition_entry = fixed_code.get("entrypoints", {}).get("acquisition", {})
    entrypoint = assert_no_symlink_components(
        root,
        Path(os.path.abspath(os.fspath(entrypoint_path))),
        require_file=True,
    )
    if (
        entrypoint != _inside(root, acquisition_entry.get("path"), file=True)
        or str(entrypoint) != acquisition_entry.get("realpath")
        or sha256_file(entrypoint) != acquisition_entry.get("sha256")
    ):
        raise AcquisitionContractError("loaded acquisition entrypoint identity changed")
    for module_name, relative in (
        ("thermoroute.opening_contract", "src/thermoroute/opening_contract.py"),
        ("thermoroute.outcome_acquisition", "src/thermoroute/outcome_acquisition.py"),
    ):
        binding = fixed_code.get("files", {}).get(relative, {})
        module = sys.modules.get(module_name)
        loaded = Path(str(getattr(module, "__file__", ""))).resolve()
        expected = _inside(root, relative, file=True)
        if (
            loaded != expected
            or binding.get("path") != relative
            or binding.get("realpath") != str(expected)
            or binding.get("sha256") != sha256_file(expected)
        ):
            raise AcquisitionContractError(f"loaded acquisition module changed: {module_name}")
    if not sys.flags.isolated:
        raise AcquisitionContractError("acquisition child must run under python -I")
    prohibited_modules = sorted(
        name for name in sys.modules
        if name == "thermoroute.opening"
        or name == "torch" or name.startswith("torch.")
        or name == "lightgbm" or name.startswith("lightgbm.")
    )
    if prohibited_modules:
        raise AcquisitionContractError(
            "raw-only acquisition imported scorer/model modules: "
            f"{prohibited_modules}"
        )
    _validate_acquisition_environment()

    return work_order, authorization, resolved


def _validate_raw_preflight_transcript(
    transcript: Mapping[str, Any],
    *,
    challenge: str,
    root: Path,
    work_order_path: Path,
    work_order: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the full validator's challenge-bound canonical assertion."""
    if set(transcript) != set(RAW_PREFLIGHT_TRANSCRIPT_FIELDS):
        raise AcquisitionContractError(
            "raw preflight transcript top-level schema changed"
        )
    if (
        transcript.get("format") != RAW_PREFLIGHT_TRANSCRIPT_FORMAT
        or transcript.get("status")
        != "FULL_PREFLIGHT_VALIDATED_NETWORK_FREE"
        or transcript.get("challenge") != challenge
        or not re.fullmatch(r"[0-9a-f]{64}", challenge)
    ):
        raise AcquisitionContractError(
            "raw preflight transcript challenge/status changed"
        )
    authorization_path = _inside(
        root, work_order.get("authorization_path"), file=True
    )
    state = work_order.get("state_paths")
    if not isinstance(state, Mapping):
        raise AcquisitionContractError(
            "raw preflight work-order state paths are malformed"
        )
    intent_path = _inside(root, state.get("intent"), file=True)
    fixed_code = authorization.get("fixed_code")
    source = authorization.get("source")
    runtime = authorization.get("runtime")
    registries = authorization.get("registries")
    matrix = authorization.get("model_matrix_amendment")
    if (
        not isinstance(fixed_code, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(registries, Mapping)
        or not isinstance(matrix, Mapping)
    ):
        raise AcquisitionContractError(
            "raw preflight authorization bindings are malformed"
        )
    entrypoints = fixed_code.get("entrypoints")
    matrix_seal = matrix.get("seal")
    development_registry = registries.get("development")
    external_registry = registries.get("external")
    if (
        not isinstance(entrypoints, Mapping)
        or not isinstance(matrix_seal, Mapping)
        or not isinstance(development_registry, Mapping)
        or not isinstance(external_registry, Mapping)
    ):
        raise AcquisitionContractError(
            "raw preflight nested authorization bindings are malformed"
        )
    orchestrator = entrypoints.get("orchestrator")
    if not isinstance(orchestrator, Mapping):
        raise AcquisitionContractError(
            "raw preflight validator entrypoint binding is malformed"
        )
    validator_path = _inside(root, orchestrator.get("path"), file=True)
    expected = {
        "opening_id": authorization.get("opening_id"),
        "authorization_path": work_order.get("authorization_path"),
        "authorization_sha256": sha256_file(authorization_path),
        "authorization_self_sha256": authorization.get(
            "authorization_self_sha256"
        ),
        "work_order_path": work_order_path.relative_to(root).as_posix(),
        "work_order_sha256": sha256_file(work_order_path),
        "work_order_self_sha256": work_order.get(
            "work_order_self_sha256"
        ),
        "intent_path": state.get("intent"),
        "intent_sha256": sha256_file(intent_path),
        "intent_self_sha256": _read_json(intent_path).get(
            "intent_self_sha256"
        ),
        "state_namespace": state.get("namespace"),
        "source_tree_sha256": source.get("source_tree_sha256"),
        "runtime_sha256": runtime.get("runtime_sha256"),
        "fixed_code_sha256": fixed_code.get("sha256"),
        "model_matrix_amendment_sha256": matrix.get("sha256"),
        "model_matrix_amendment_seal_sha256": matrix_seal.get("sha256"),
        "development_registry_sha256": development_registry.get("sha256"),
        "external_registry_sha256": external_registry.get("sha256"),
        "validator_entrypoint_path": orchestrator.get("path"),
        "validator_entrypoint_sha256": sha256_file(validator_path),
        "outcome_values_parsed": False,
        "network_used": False,
    }
    wrong = [
        key
        for key, value in expected.items()
        if transcript.get(key) != value
    ]
    if wrong:
        raise AcquisitionContractError(
            "raw preflight transcript binding changed: "
            + ", ".join(sorted(wrong))
        )
    phase = transcript.get("resume_phase")
    raw_allowed = transcript.get("raw_transport_resume_allowed")
    finalization_allowed = transcript.get(
        "network_free_acquisition_finalization_allowed"
    )
    if (
        phase == "RAW_TRANSPORT"
        and (raw_allowed is not True or finalization_allowed is not False)
    ) or (
        phase == "ACQUISITION_FINALIZATION_NETWORK_FREE"
        and (raw_allowed is not False or finalization_allowed is not True)
    ) or phase not in {
        "RAW_TRANSPORT",
        "ACQUISITION_FINALIZATION_NETWORK_FREE",
    }:
        raise AcquisitionContractError(
            "raw preflight transcript state is not acquisition eligible"
        )
    attestation = transcript.get("preflight_attestation")
    if (
        not isinstance(attestation, Mapping)
        or set(attestation) != set(RAW_PREFLIGHT_ATTESTATION_FIELDS)
        or transcript.get("preflight_attestation_sha256")
        != _sha256_json(attestation)
    ):
        raise AcquisitionContractError(
            "raw preflight attestation schema/hash changed"
        )
    attestation_expected = {
        "authorization_sha256": expected["authorization_sha256"],
        "opening_id": expected["opening_id"],
        "development_registry_sha256": expected[
            "development_registry_sha256"
        ],
        "external_registry_sha256": expected[
            "external_registry_sha256"
        ],
        "model_matrix_amendment_sha256": expected[
            "model_matrix_amendment_sha256"
        ],
        "model_matrix_amendment_seal_sha256": expected[
            "model_matrix_amendment_seal_sha256"
        ],
        "source_tree_sha256": expected["source_tree_sha256"],
        "runtime_sha256": expected["runtime_sha256"],
        "fixed_code_sha256": expected["fixed_code_sha256"],
        "state_namespace": expected["state_namespace"],
    }
    if any(
        attestation.get(key) != value
        for key, value in attestation_expected.items()
    ):
        raise AcquisitionContractError(
            "raw preflight attestation differs from raw contract"
        )
    trusted_validator = transcript.get("trusted_validator")
    if (
        not isinstance(trusted_validator, Mapping)
        or set(trusted_validator)
        != {"implementation", "files", "sha256", "source_tree_sha256"}
        or trusted_validator.get("implementation")
        != "thermoroute.opening.trusted-validator.v1"
        or not isinstance(trusted_validator.get("files"), Mapping)
        or trusted_validator.get("sha256")
        != _sha256_json(trusted_validator["files"])
        or trusted_validator.get("source_tree_sha256")
        != expected["source_tree_sha256"]
    ):
        raise AcquisitionContractError(
            "raw preflight trusted-validator identity changed"
        )
    return dict(transcript)


def run_full_raw_preflight_validator(
    work_order_path: str | Path,
    *,
    root: str | Path,
    work_order: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the fixed full validator before the raw child mutates transport."""
    root = Path(root).resolve()
    canonical_work_order = assert_no_symlink_components(
        root,
        Path(os.path.abspath(os.fspath(work_order_path))),
        require_file=True,
    )
    challenge = secrets.token_hex(32)
    fixed_code = authorization.get("fixed_code", {})
    entry = fixed_code.get("entrypoints", {}).get("orchestrator", {})
    if (
        not isinstance(entry, Mapping)
        or entry.get("path") != "scripts/route_a_opening_orchestrator.py"
    ):
        raise AcquisitionContractError(
            "raw preflight fixed validator entrypoint changed"
        )
    validator = _inside(root, entry.get("path"), file=True)
    if (
        str(validator) != entry.get("realpath")
        or sha256_file(validator) != entry.get("sha256")
    ):
        raise AcquisitionContractError(
            "raw preflight fixed validator bytes changed"
        )
    command = [
        sys.executable,
        "-I",
        "-B",
        str(validator),
        "--raw-preflight-work-order",
        str(canonical_work_order),
        "--challenge",
        challenge,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=dict(os.environ),
            capture_output=True,
            check=False,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AcquisitionContractError(
            "fixed raw preflight validator could not complete"
        ) from exc
    if result.returncode != 0 or result.stderr != b"":
        raise AcquisitionContractError(
            "fixed raw preflight validator failed or wrote stderr"
        )
    stdout = result.stdout
    if not stdout or len(stdout) > MAX_RAW_PREFLIGHT_TRANSCRIPT_BYTES:
        raise AcquisitionContractError(
            "raw preflight transcript has an invalid byte length"
        )
    if not stdout.endswith(b"\n") or b"\n" in stdout[:-1]:
        raise AcquisitionContractError(
            "raw preflight transcript is not exactly one line"
        )
    try:
        transcript = json.loads(
            stdout[:-1].decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(
            "raw preflight transcript is not valid JSON"
        ) from exc
    if (
        not isinstance(transcript, dict)
        or stdout != _canonical_json_bytes(transcript)
    ):
        raise AcquisitionContractError(
            "raw preflight transcript is not exact canonical JSON"
        )
    return _validate_raw_preflight_transcript(
        transcript,
        challenge=challenge,
        root=root,
        work_order_path=canonical_work_order,
        work_order=work_order,
        authorization=authorization,
    )


def revalidate_raw_preflight_volatile_bindings(
    work_order_path: str | Path,
    *,
    root: str | Path,
    entrypoint_path: str | Path,
    transcript: Mapping[str, Any],
    expected_work_order: Mapping[str, Any],
    expected_authorization: Mapping[str, Any],
) -> None:
    """Replay all raw-visible bytes immediately before every socket open."""
    current_work_order, current_authorization, _state = (
        validate_acquisition_work_order(
            work_order_path,
            root=root,
            entrypoint_path=entrypoint_path,
        )
    )
    if (
        current_work_order != dict(expected_work_order)
        or current_authorization != dict(expected_authorization)
    ):
        raise AcquisitionContractError(
            "raw preflight documents changed after validation"
        )
    challenge = transcript.get("challenge")
    if not isinstance(challenge, str):
        raise AcquisitionContractError(
            "raw preflight transcript lost its challenge"
        )
    _validate_raw_preflight_transcript(
        transcript,
        challenge=challenge,
        root=Path(root).resolve(),
        work_order_path=Path(
            os.path.abspath(os.fspath(work_order_path))
        ),
        work_order=current_work_order,
        authorization=current_authorization,
    )
