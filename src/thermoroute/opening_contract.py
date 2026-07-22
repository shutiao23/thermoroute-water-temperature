"""Lightweight stdlib-only contract for the isolated Route-A acquisition child."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import stat
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(f"cannot read acquisition contract: {path}") from exc
    if not isinstance(value, dict):
        raise AcquisitionContractError("acquisition contract must be a JSON object")
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
    self_hashed = dict(work_order)
    self_digest = self_hashed.pop("work_order_self_sha256", None)
    if self_digest != _sha256_json(self_hashed):
        raise AcquisitionContractError("acquisition work-order self hash changed")

    authorization_path = _inside(root, work_order.get("authorization_path"), file=True)
    authorization = _read_json(authorization_path)
    if authorization.get("format") != AUTHORIZATION_FORMAT:
        raise AcquisitionContractError("unsupported opening authorization")
    self_hashed_authorization = dict(authorization)
    auth_self_digest = self_hashed_authorization.pop("authorization_self_sha256", None)
    if auth_self_digest != _sha256_json(self_hashed_authorization):
        raise AcquisitionContractError("opening authorization self hash changed")
    expected_equal = {
        "opening_id": authorization.get("opening_id"),
        "authorization_sha256": sha256_file(authorization_path),
        "source_tree_sha256": authorization.get("source", {}).get("source_tree_sha256"),
        "runtime_sha256": authorization.get("runtime", {}).get("runtime_sha256"),
        "fixed_code_sha256": authorization.get("fixed_code", {}).get("sha256"),
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
