#!/usr/bin/env python3
"""Rebuild and create-only publish the Tier-1 semantic registries v4 authority.

The authority contains the exact semantic-data candidate (two Parquet
registries and its manifest), the exact semantic-contract candidate (six JSON
registries and its manifest), and one authority manifest.  It rebuilds each
candidate through that candidate builder's independent two-pass API.  It then
cross-checks serialized bytes, semantic-data digests, contract references,
source bindings, runtime receipts, and the protocol cell-state arithmetic.

This is a semantic data/contract authority only.  It never reads a model,
checkpoint, prediction, score, effect, contrast result, or runner output.  It
does not authorize execution and cannot replace a forcing-protocol seal.  The
writer has no overwrite or resume mode: it stages in the destination's parent,
fsyncs files and directories, and commits with Linux ``RENAME_NOREPLACE``.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FINAL_OUTPUT_ROOT = ROOT / "outputs" / "final"
CANONICAL_AUTHORITY_DIRECTORY = FINAL_OUTPUT_ROOT / "semantic_registries_v4_authority"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = importlib.import_module("scripts.final.build_semantic_data_registries_v4")
CONTRACT = importlib.import_module("scripts.final.build_semantic_contract_registries_v4")

STATUS = "TIER1_SEMANTIC_DATA_AND_CONTRACT_AUTHORITY_NOT_EXECUTION_AUTHORITY"
ARTIFACT_ID = "thermoroute-semantic-registries-v4-authority"
AUTHORITY_MANIFEST_FILENAME = "semantic_registries_v4_authority_manifest.json"

DATA_REGISTRY_FILENAMES = (DATA.DAILY_FILENAME, DATA.TRAINING_FILENAME)
CONTRACT_REGISTRY_FILENAMES = tuple(CONTRACT.REGISTRY_FILENAMES)
CANDIDATE_MANIFEST_FILENAMES = (DATA.MANIFEST_FILENAME, CONTRACT.MANIFEST_FILENAME)
BOUND_CANDIDATE_FILENAMES = (
    *DATA_REGISTRY_FILENAMES,
    DATA.MANIFEST_FILENAME,
    *CONTRACT_REGISTRY_FILENAMES,
    CONTRACT.MANIFEST_FILENAME,
)
ALL_FILENAMES = (*BOUND_CANDIDATE_FILENAMES, AUTHORITY_MANIFEST_FILENAME)

EXPECTED_PRIMARY_STATE_COUNTS = MappingProxyType(
    {"PLANNED": 48, "REGISTERED": 24, "EXECUTED": 0, "WITHDRAWN": 0}
)
EXPECTED_PRIMARY_CELL_COUNT = 72
EXPECTED_FORENSIC_WITHDRAWN_CELL_COUNT = 432


class SemanticAuthorityError(RuntimeError):
    """A candidate, attestation, or create-only publication check failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticAuthorityError(message)


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


def _canonical_document_sha256(document: object) -> str:
    return _sha256(_canonical_json_bytes(document))


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
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
        raise SemanticAuthorityError(f"{label} is not strict JSON") from exc
    _require(type(document) is dict, f"{label} root must be an object")
    _require(_canonical_json_bytes(document) == payload, f"{label} is not canonical JSON")
    return document


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_to_root(path: Path) -> str:
    try:
        return _absolute(path).relative_to(_absolute(ROOT)).as_posix()
    except ValueError as exc:
        raise SemanticAuthorityError(f"attestation source escapes repository: {path}") from exc


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current = current / part
        try:
            status = current.lstat()
        except OSError as exc:
            raise SemanticAuthorityError(f"{label} component is unavailable: {current}") from exc
        _require(not stat.S_ISLNK(status.st_mode), f"{label} crosses a symlink: {current}")
    return absolute


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


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


@dataclass(frozen=True, slots=True)
class BoundSource:
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    stat_signature: tuple[int, ...]

    def binding(self) -> dict[str, object]:
        return {
            "path": _relative_to_root(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _capture_repository_source(path: Path, *, label: str) -> BoundSource:
    absolute = _require_no_symlink_components(path, label=label)
    _relative_to_root(absolute)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise SemanticAuthorityError(f"cannot safely open {label}: {absolute}") from exc
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
    lexical = absolute.lstat()
    _require(
        _stat_signature(before) == _stat_signature(after)
        and _stat_signature(before) == _stat_signature(lexical)
        and len(payload) == before.st_size,
        f"{label} changed while being captured",
    )
    return BoundSource(
        path=absolute,
        payload=payload,
        sha256=_sha256(payload),
        size_bytes=len(payload),
        stat_signature=_stat_signature(before),
    )


def _attestation_paths() -> dict[str, Path]:
    return {
        "semantic_data_builder": ROOT / "scripts/final/build_semantic_data_registries_v4.py",
        "semantic_contract_builder": (
            ROOT / "scripts/final/build_semantic_contract_registries_v4.py"
        ),
        "semantic_authority_publisher": Path(__file__),
        "semantic_data_builder_tests": (ROOT / "tests/final/test_semantic_data_registries_v4.py"),
        "semantic_contract_builder_tests": (
            ROOT / "tests/final/test_semantic_contract_registries_v4.py"
        ),
        "semantic_authority_publisher_tests": (
            ROOT / "tests/final/test_publish_semantic_registries_v4.py"
        ),
    }


def _capture_attestation_sources() -> dict[str, BoundSource]:
    return {
        role: _capture_repository_source(path, label=role.replace("_", " "))
        for role, path in sorted(_attestation_paths().items())
    }


def _revalidate_attestation_sources(original: Mapping[str, BoundSource]) -> None:
    _require(set(original) == set(_attestation_paths()), "attestation source role set changed")
    current = _capture_attestation_sources()
    for role in original:
        _require(
            current[role].stat_signature == original[role].stat_signature
            and current[role].sha256 == original[role].sha256
            and current[role].payload == original[role].payload,
            f"attestation source changed before commit: {role}",
        )


def _binding_fields(record: object, *, label: str) -> dict[str, object]:
    _require(type(record) is dict, f"{label} binding must be an object")
    assert isinstance(record, dict)
    _require(set(record) == {"path", "sha256", "size_bytes"}, f"{label} fields changed")
    path = record["path"]
    digest = record["sha256"]
    size = record["size_bytes"]
    _require(type(path) is str and path != "", f"{label} path is invalid")
    _require(
        type(digest) is str
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} SHA-256 is invalid",
    )
    _require(type(size) is int and size >= 0, f"{label} size is invalid")
    return dict(record)


def _verify_data_candidate(
    bundle: Any,
    *,
    config: Any,
    require_production: bool,
) -> dict[str, Any]:
    try:
        DATA._verify_bundle(bundle, config=config)
    except DATA.SemanticRegistryError as exc:
        raise SemanticAuthorityError(f"semantic-data candidate verification failed: {exc}") from exc
    manifest = bundle.manifest
    _require(
        manifest.get("artifact_id") == DATA.ARTIFACT_ID and manifest.get("status") == DATA.STATUS,
        "semantic-data candidate identity/status changed",
    )
    if require_production:
        _require(config == DATA.BuildConfig.production(), "semantic-data config is not production")
        _require(
            manifest.get("build_mode") == "PRODUCTION_INPUTS_CANDIDATE_OUTPUT",
            "semantic-data candidate is not a production-input reconstruction",
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
    _require(
        set(bundle.files) == {DATA.DAILY_FILENAME, DATA.TRAINING_FILENAME, DATA.MANIFEST_FILENAME},
        "semantic-data candidate file set changed",
    )
    _require(
        bundle.files[DATA.MANIFEST_FILENAME] == DATA._canonical_json_bytes(manifest),
        "semantic-data candidate manifest bytes changed",
    )
    outputs = manifest.get("outputs")
    _require(
        type(outputs) is dict and set(outputs) == set(DATA_REGISTRY_FILENAMES),
        "semantic-data output binding set changed",
    )
    for name in DATA_REGISTRY_FILENAMES:
        record = outputs[name]
        _require(type(record) is dict, f"semantic-data output record is invalid: {name}")
        _require(
            record.get("sha256") == _sha256(bundle.files[name])
            and record.get("size_bytes") == len(bundle.files[name]),
            f"semantic-data output byte binding changed: {name}",
        )
        for digest_field in ("ordered_semantic_content_sha256", "ordered_identity_sha256"):
            value = record.get(digest_field)
            _require(
                type(value) is str and len(value) == 64,
                f"semantic-data {name} lacks {digest_field}",
            )
    return manifest


def _verify_contract_candidate(bundle: Any) -> dict[str, Any]:
    try:
        CONTRACT._verify_bundle(bundle)
    except CONTRACT.SemanticContractError as exc:
        raise SemanticAuthorityError(
            f"semantic-contract candidate verification failed: {exc}"
        ) from exc
    manifest = bundle.manifest
    _require(
        manifest.get("artifact_id") == CONTRACT.ARTIFACT_ID
        and manifest.get("status") == CONTRACT.STATUS,
        "semantic-contract candidate identity/status changed",
    )
    _require(
        manifest.get("formal_authority") is False and manifest.get("execution_authorized") is False,
        "semantic-contract candidate incorrectly claims authority or execution",
    )
    _require(set(bundle.files) == set(CONTRACT.ALL_FILENAMES), "contract file set changed")
    _require(
        manifest.get("evidence_boundary", {}).get("score_free") is True
        and manifest.get("evidence_boundary", {}).get(
            "model_score_checkpoint_prediction_effect_or_result_inputs_read"
        )
        is False,
        "semantic-contract score-free boundary changed",
    )
    for name in CONTRACT.REGISTRY_FILENAMES:
        record = manifest.get("outputs", {}).get(name)
        _require(type(record) is dict, f"contract output record is invalid: {name}")
        _require(
            record.get("sha256") == _sha256(bundle.files[name])
            and record.get("size_bytes") == len(bundle.files[name]),
            f"semantic-contract output byte binding changed: {name}",
        )
    return manifest


def _verify_candidate_cross_bindings(
    data_bundle: Any,
    contract_bundle: Any,
    *,
    data_config: Any,
    require_production: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_manifest = _verify_data_candidate(
        data_bundle,
        config=data_config,
        require_production=require_production,
    )
    contract_manifest = _verify_contract_candidate(contract_bundle)
    expected_data_bindings = {
        "semantic_data_manifest": {
            "path": DATA.MANIFEST_FILENAME,
            "sha256": _sha256(data_bundle.files[DATA.MANIFEST_FILENAME]),
            "size_bytes": len(data_bundle.files[DATA.MANIFEST_FILENAME]),
        },
        "semantic_daily_registry": {
            "path": DATA.DAILY_FILENAME,
            "sha256": _sha256(data_bundle.files[DATA.DAILY_FILENAME]),
            "size_bytes": len(data_bundle.files[DATA.DAILY_FILENAME]),
        },
        "semantic_training_registry": {
            "path": DATA.TRAINING_FILENAME,
            "sha256": _sha256(data_bundle.files[DATA.TRAINING_FILENAME]),
            "size_bytes": len(data_bundle.files[DATA.TRAINING_FILENAME]),
        },
    }
    contract_sources = contract_manifest.get("source_inputs")
    _require(type(contract_sources) is dict, "contract source-input bindings are absent")
    assert isinstance(contract_sources, dict)
    input_sources = contract_bundle.documents[CONTRACT.INPUT_FILENAME].get("source_bindings")
    _require(type(input_sources) is dict, "input registry source bindings are absent")
    assert isinstance(input_sources, dict)
    for role, expected in expected_data_bindings.items():
        _require(
            _binding_fields(contract_sources.get(role), label=f"contract {role}") == expected,
            f"contract manifest does not bind exact semantic-data bytes: {role}",
        )
        _require(
            _binding_fields(input_sources.get(role), label=f"input registry {role}") == expected,
            f"input registry does not bind exact semantic-data bytes: {role}",
        )

    if require_production:
        data_inputs = data_manifest.get("input_bindings")
        _require(type(data_inputs) is dict, "semantic-data input bindings are absent")
        assert isinstance(data_inputs, dict)
        for role in (
            "station_registry",
            "key_authority_manifest",
            "defect_authority_manifest",
            "defect_authority_report",
        ):
            _require(
                _binding_fields(data_inputs.get(role), label=f"semantic-data common {role}")
                == _binding_fields(
                    contract_sources.get(role),
                    label=f"semantic-contract common {role}",
                ),
                f"candidate builders do not bind the same production source: {role}",
            )

    state_map = contract_bundle.documents[CONTRACT.CELL_FILENAME].get(
        "protocol_declared_cell_status"
    )
    _require(type(state_map) is dict, "protocol cell-state map is absent")
    assert isinstance(state_map, dict)
    primary = state_map.get("primary_matrix")
    forensic = state_map.get("withdrawn_forensic_scope")
    _require(type(primary) is dict and type(forensic) is dict, "cell-state scopes are invalid")
    assert isinstance(primary, dict) and isinstance(forensic, dict)
    _require(
        primary.get("protocol_logical_cell_count") == EXPECTED_PRIMARY_CELL_COUNT
        and primary.get("state_counts") == dict(EXPECTED_PRIMARY_STATE_COUNTS),
        "primary 72-cell state arithmetic changed",
    )
    cells = primary.get("cells")
    _require(
        type(cells) is list and len(cells) == EXPECTED_PRIMARY_CELL_COUNT, "primary cells changed"
    )
    assert isinstance(cells, list)
    _require(
        all(
            type(record) is dict
            and record.get("protocol_state") in {"PLANNED", "REGISTERED"}
            and record.get("execution_receipt_bound") is False
            and record.get("score_or_result_receipt_bound") is False
            for record in cells
        ),
        "a primary candidate cell claims execution or score/result evidence",
    )
    _require(
        forensic.get("protocol_state") == "WITHDRAWN"
        and forensic.get("logical_cell_count") == EXPECTED_FORENSIC_WITHDRAWN_CELL_COUNT
        and forensic.get("included_in_primary_72") is False
        and forensic.get("eligible_as_model_or_score_evidence") is False,
        "432-cell withdrawn forensic boundary changed",
    )
    scope = contract_manifest.get("scope")
    evidence = contract_manifest.get("evidence_boundary")
    _require(type(scope) is dict and type(evidence) is dict, "contract scope boundary is absent")
    assert isinstance(scope, dict) and isinstance(evidence, dict)
    _require(
        scope.get("protocol_primary_cells") == EXPECTED_PRIMARY_CELL_COUNT
        and scope.get("protocol_primary_state_counts") == dict(EXPECTED_PRIMARY_STATE_COUNTS)
        and scope.get("withdrawn_historical_cells_forensic_only")
        == EXPECTED_FORENSIC_WITHDRAWN_CELL_COUNT
        and evidence.get("executed_protocol_cells") == 0
        and evidence.get("candidate_can_promote_planned_or_registered_to_executed") is False,
        "contract manifest cell-state receipt changed",
    )
    return data_manifest, contract_manifest


def _code_binding_document(sources: Mapping[str, BoundSource]) -> dict[str, Any]:
    builders = {
        role: sources[role].binding()
        for role in ("semantic_data_builder", "semantic_contract_builder")
    }
    publisher = {"semantic_authority_publisher": sources["semantic_authority_publisher"].binding()}
    tests = {
        role: sources[role].binding()
        for role in (
            "semantic_data_builder_tests",
            "semantic_contract_builder_tests",
            "semantic_authority_publisher_tests",
        )
    }
    binding_set = {"builders": builders, "publisher": publisher, "tests": tests}
    return {
        **binding_set,
        "canonical_binding_set_sha256": _canonical_document_sha256(binding_set),
    }


def _source_input_binding_document(
    data_bundle: Any,
    contract_bundle: Any,
    data_manifest: Mapping[str, Any],
    contract_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    data_inputs = data_manifest.get("input_bindings")
    contract_inputs = contract_manifest.get("source_inputs")
    _require(type(data_inputs) is dict, "semantic-data input bindings are absent")
    _require(type(contract_inputs) is dict, "semantic-contract source inputs are absent")
    assert isinstance(data_inputs, dict) and isinstance(contract_inputs, dict)
    for role, record in data_inputs.items():
        _binding_fields(record, label=f"semantic-data input {role}")
    for role, record in contract_inputs.items():
        _binding_fields(record, label=f"semantic-contract input {role}")
    contract_snapshot = {
        role: _binding_fields(record, label=f"contract snapshot {role}")
        for role, record in sorted(contract_bundle.snapshot.bindings.items())
    }
    data_snapshot = {
        role: {
            "path": data_inputs[role]["path"],
            "sha256": bound.sha256,
            "size_bytes": bound.size_bytes,
        }
        for role, bound in sorted(data_bundle.snapshot.items())
    }
    _require(data_snapshot == data_inputs, "data snapshot and manifest bindings differ")
    maps = {
        "semantic_data_candidate_inputs": dict(data_inputs),
        "semantic_contract_candidate_inputs": dict(contract_inputs),
        "semantic_data_builder_snapshot": data_snapshot,
        "semantic_contract_builder_snapshot": contract_snapshot,
    }
    return {**maps, "canonical_binding_maps_sha256": _canonical_document_sha256(maps)}


def _runtime_receipt_document(
    data_manifest: Mapping[str, Any],
    contract_bundle: Any,
    *,
    require_production: bool,
) -> dict[str, object]:
    data_runtime = data_manifest.get("runtime")
    _require(type(data_runtime) is dict, "semantic-data runtime receipt is absent")
    assert isinstance(data_runtime, dict)
    data_active = data_runtime.get("active_runtime")
    data_expected = data_runtime.get("candidate_runtime_expected")
    _require(
        type(data_active) is dict and type(data_expected) is dict,
        "semantic-data runtime identities are invalid",
    )
    assert isinstance(data_active, dict) and isinstance(data_expected, dict)
    environment = contract_bundle.documents[CONTRACT.ENVIRONMENT_FILENAME]
    environments = environment.get("environments")
    _require(type(environments) is list, "contract environment inventory is invalid")
    assert isinstance(environments, list)
    route_a = next(
        (
            record
            for record in environments
            if type(record) is dict
            and record.get("environment_id") == "environment:route_a_candidate"
        ),
        None,
    )
    _require(type(route_a) is dict, "route-a candidate runtime record is absent")
    assert isinstance(route_a, dict)
    route_runtime = route_a.get("runtime")
    route_packages = route_a.get("packages")
    _require(
        type(route_runtime) is dict and type(route_packages) is dict,
        "route-a runtime/package receipts are invalid",
    )
    assert isinstance(route_runtime, dict) and isinstance(route_packages, dict)
    authority_active = CONTRACT._runtime_identity()
    if require_production:
        _require(
            authority_active == dict(CONTRACT.ROUTE_A_RUNTIME)
            and route_runtime == dict(CONTRACT.ROUTE_A_RUNTIME),
            "authority and contract candidate must use the exact route-a runtime",
        )
        _require(data_active == dict(DATA.ROUTE_A_CANDIDATE_RUNTIME), "data runtime changed")
        _require(data_expected == dict(DATA.ROUTE_A_CANDIDATE_RUNTIME), "data runtime pin changed")
    records = {
        "authority_active_runtime": {
            "values": dict(authority_active),
            "canonical_sha256": _canonical_document_sha256(authority_active),
        },
        "semantic_data_active_runtime": {
            "values": dict(data_active),
            "canonical_sha256": _canonical_document_sha256(data_active),
        },
        "semantic_data_expected_runtime": {
            "values": dict(data_expected),
            "canonical_sha256": _canonical_document_sha256(data_expected),
        },
        "semantic_contract_route_a_runtime": {
            "values": dict(route_runtime),
            "canonical_sha256": _canonical_document_sha256(route_runtime),
        },
        "semantic_contract_route_a_packages": {
            "values": dict(route_packages),
            "canonical_sha256": _canonical_document_sha256(route_packages),
        },
    }
    return {
        **records,
        "runtime_receipt_set_sha256": _canonical_document_sha256(records),
        "runtime_identity_is_execution_authority": False,
    }


def _bound_registry_records(
    data_bundle: Any,
    contract_bundle: Any,
    data_manifest: Mapping[str, Any],
    contract_manifest: Mapping[str, Any],
) -> dict[str, object]:
    data_records: dict[str, dict[str, Any]] = {}
    for name in DATA_REGISTRY_FILENAMES:
        candidate_record = data_manifest["outputs"][name]
        data_records[name] = {
            "candidate_bundle": "semantic_data",
            "candidate_manifest": DATA.MANIFEST_FILENAME,
            **dict(candidate_record),
        }
    contract_records: dict[str, dict[str, Any]] = {}
    for name in CONTRACT_REGISTRY_FILENAMES:
        candidate_record = contract_manifest["outputs"][name]
        contract_records[name] = {
            "candidate_bundle": "semantic_contract",
            "candidate_manifest": CONTRACT.MANIFEST_FILENAME,
            **dict(candidate_record),
        }
    _require(
        all(
            record["sha256"] == _sha256(data_bundle.files[name])
            and record["size_bytes"] == len(data_bundle.files[name])
            for name, record in data_records.items()
        ),
        "data authority registry bindings differ from candidate bytes",
    )
    _require(
        all(
            record["sha256"] == _sha256(contract_bundle.files[name])
            and record["size_bytes"] == len(contract_bundle.files[name])
            for name, record in contract_records.items()
        ),
        "contract authority registry bindings differ from candidate bytes",
    )
    records = {"semantic_data": data_records, "semantic_contract": contract_records}
    return {**records, "canonical_registry_receipts_sha256": _canonical_document_sha256(records)}


def _candidate_manifest_records(data_bundle: Any, contract_bundle: Any) -> dict[str, object]:
    records = {
        "semantic_data": {
            "filename": DATA.MANIFEST_FILENAME,
            "artifact_id": DATA.ARTIFACT_ID,
            "candidate_status": DATA.STATUS,
            "sha256": _sha256(data_bundle.files[DATA.MANIFEST_FILENAME]),
            "size_bytes": len(data_bundle.files[DATA.MANIFEST_FILENAME]),
            "canonical_json": True,
        },
        "semantic_contract": {
            "filename": CONTRACT.MANIFEST_FILENAME,
            "artifact_id": CONTRACT.ARTIFACT_ID,
            "candidate_status": CONTRACT.STATUS,
            "sha256": _sha256(contract_bundle.files[CONTRACT.MANIFEST_FILENAME]),
            "size_bytes": len(contract_bundle.files[CONTRACT.MANIFEST_FILENAME]),
            "canonical_json": True,
        },
    }
    return {
        **records,
        "canonical_candidate_manifest_receipts_sha256": _canonical_document_sha256(records),
    }


def _authority_manifest_document(
    data_bundle: Any,
    contract_bundle: Any,
    sources: Mapping[str, BoundSource],
    *,
    data_config: Any,
    require_production: bool,
) -> dict[str, object]:
    data_manifest, contract_manifest = _verify_candidate_cross_bindings(
        data_bundle,
        contract_bundle,
        data_config=data_config,
        require_production=require_production,
    )
    code_bindings = _code_binding_document(sources)
    source_input_bindings = _source_input_binding_document(
        data_bundle,
        contract_bundle,
        data_manifest,
        contract_manifest,
    )
    _require(
        source_input_bindings["semantic_data_candidate_inputs"]["builder_code"]
        == code_bindings["builders"]["semantic_data_builder"],
        "semantic-data candidate and authority bind different builder bytes",
    )
    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": STATUS,
        "authority_class": "TIER1_SEMANTIC_DATA_AND_CONTRACT_ONLY",
        "formal_semantic_registry_authority": True,
        "execution_authorized": False,
        "model_execution_authorized": False,
        "forcing_protocol_seal_bound": False,
        "forcing_protocol_seal_required_separately": True,
        "may_replace_or_amend_forcing_protocol_seal": False,
        "candidate_status_promoted_to_executed": False,
        "model_prediction_score_effect_or_result_artifacts_read": False,
        "authority_scope": {
            "authoritative_for": [
                "exact bytes of the two bound candidate manifests",
                "exact bytes and declared semantics of the two semantic-data registries",
                "exact bytes and closed contracts of the six semantic-contract registries",
                "the bound builder, test, source-input, and runtime receipts",
            ],
            "not_authoritative_for": [
                "forcing protocol authorization or seal amendment",
                "model execution",
                "checkpoint completion",
                "predictions, scores, effects, contrasts, or scientific results",
            ],
            "candidate_internal_non_authorizing_fields_remain_false": True,
            "dependency_authority_transfers_to_execution": False,
        },
        "candidate_manifests": _candidate_manifest_records(data_bundle, contract_bundle),
        "registries": _bound_registry_records(
            data_bundle,
            contract_bundle,
            data_manifest,
            contract_manifest,
        ),
        "code_bindings": code_bindings,
        "source_input_bindings": source_input_bindings,
        "runtime_receipts": _runtime_receipt_document(
            data_manifest,
            contract_bundle,
            require_production=require_production,
        ),
        "candidate_rebuild_and_cross_verification": {
            "semantic_data_builder_api": "build_twice via closed write_candidate",
            "semantic_data_independent_build_passes": 2,
            "semantic_contract_builder_api": "build_twice",
            "semantic_contract_independent_build_passes": 2,
            "candidate_manifest_bytes_verified": True,
            "all_registry_bytes_verified_against_candidate_manifests": True,
            "semantic_data_parquet_schema_content_identity_and_lineage_verified": True,
            "semantic_contract_closed_schema_and_cross_registry_references_verified": True,
            "contract_to_semantic_data_byte_bindings_cross_verified": True,
            "common_production_source_bindings_cross_verified": require_production,
            "caller_supplied_candidate_bundle_accepted": False,
        },
        "protocol_cell_state_receipt": {
            "primary_cell_count": EXPECTED_PRIMARY_CELL_COUNT,
            "primary_state_counts": dict(EXPECTED_PRIMARY_STATE_COUNTS),
            "primary_executed_cell_count": 0,
            "candidate_may_assert_executed": False,
            "withdrawn_forensic_cell_count": EXPECTED_FORENSIC_WITHDRAWN_CELL_COUNT,
            "withdrawn_forensic_state": "WITHDRAWN",
            "withdrawn_forensic_included_in_primary_72": False,
            "withdrawn_forensic_eligible_as_score_or_result_evidence": False,
        },
        "evidence_boundary": {
            "exact_read_scope": (
                "pinned station/panel/key/defect/lock inputs plus the six bound code/test files"
            ),
            "prediction_files_read": False,
            "score_tables_read": False,
            "model_checkpoints_read": False,
            "effect_or_contrast_results_read": False,
            "runner_imported_or_executed": False,
            "prediction_or_score_rows_read": 0,
        },
        "publication_contract": {
            "caller_must_specify_destination": True,
            "recommended_canonical_destination": ("outputs/final/semantic_registries_v4_authority"),
            "exact_file_count": len(ALL_FILENAMES),
            "bound_candidate_file_count": len(BOUND_CANDIDATE_FILENAMES),
            "authority_manifest_self_hash_bound": False,
            "create_only": True,
            "same_parent_directory_staging": True,
            "file_fsync_before_commit": True,
            "staging_directory_fsync_before_commit": True,
            "parent_directory_fsync_before_and_after_commit": True,
            "atomic_linux_RENAME_NOREPLACE": True,
            "overwrite_supported": False,
            "resume_supported": False,
            "staged_and_committed_bytes_reverified": True,
        },
    }


_BUNDLE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AuthorityBundle:
    files: Mapping[str, bytes]
    data_candidate: Any
    contract_candidate: Any
    attestation_sources: Mapping[str, BoundSource]
    data_config: Any
    require_production: bool
    _token: object

    def __post_init__(self) -> None:
        _require(self._token is _BUNDLE_TOKEN, "authority bundle issuer token changed")
        frozen_files: dict[str, bytes] = {}
        for name, payload in self.files.items():
            _require(type(name) is str and type(payload) is bytes, "invalid authority file payload")
            frozen_files[name] = bytes(payload)
        object.__setattr__(self, "files", MappingProxyType(frozen_files))
        object.__setattr__(
            self,
            "attestation_sources",
            MappingProxyType(dict(self.attestation_sources)),
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return _strict_json(self.files[AUTHORITY_MANIFEST_FILENAME], label="authority manifest")


def _build_authority_bundle(
    data_candidate: Any,
    contract_candidate: Any,
    attestation_sources: Mapping[str, BoundSource],
    *,
    data_config: Any,
    require_production: bool,
) -> AuthorityBundle:
    _require(
        set(attestation_sources) == set(_attestation_paths()),
        "authority attestation source set changed",
    )
    candidate_files = {**dict(data_candidate.files), **dict(contract_candidate.files)}
    _require(
        len(candidate_files) == len(BOUND_CANDIDATE_FILENAMES)
        and set(candidate_files) == set(BOUND_CANDIDATE_FILENAMES),
        "candidate file namespaces overlap or changed",
    )
    manifest = _authority_manifest_document(
        data_candidate,
        contract_candidate,
        attestation_sources,
        data_config=data_config,
        require_production=require_production,
    )
    files = {**candidate_files, AUTHORITY_MANIFEST_FILENAME: _canonical_json_bytes(manifest)}
    bundle = AuthorityBundle(
        files=files,
        data_candidate=data_candidate,
        contract_candidate=contract_candidate,
        attestation_sources=attestation_sources,
        data_config=data_config,
        require_production=require_production,
        _token=_BUNDLE_TOKEN,
    )
    _verify_authority_bundle(bundle)
    return bundle


def _verify_authority_bundle(bundle: AuthorityBundle) -> None:
    _require(
        type(bundle) is AuthorityBundle and bundle._token is _BUNDLE_TOKEN,
        "authority bundle was forged",
    )
    _require(set(bundle.files) == set(ALL_FILENAMES), "authority bundle file set changed")
    for name in DATA_REGISTRY_FILENAMES:
        _require(
            bundle.files[name] == bundle.data_candidate.files[name],
            f"authority data registry bytes differ from rebuilt candidate: {name}",
        )
    for name in (DATA.MANIFEST_FILENAME, *CONTRACT.ALL_FILENAMES):
        source = (
            bundle.data_candidate.files[name]
            if name == DATA.MANIFEST_FILENAME
            else bundle.contract_candidate.files[name]
        )
        _require(bundle.files[name] == source, f"authority candidate bytes changed: {name}")
    observed = bundle.manifest
    expected = _authority_manifest_document(
        bundle.data_candidate,
        bundle.contract_candidate,
        bundle.attestation_sources,
        data_config=bundle.data_config,
        require_production=bundle.require_production,
    )
    _require(observed == expected, "authority manifest differs from exact rebuilt receipts")
    _require(
        observed.get("status") == STATUS
        and observed.get("formal_semantic_registry_authority") is True
        and observed.get("execution_authorized") is False
        and observed.get("model_execution_authorized") is False
        and observed.get("forcing_protocol_seal_bound") is False
        and observed.get("may_replace_or_amend_forcing_protocol_seal") is False,
        "authority semantic-only/non-execution boundary changed",
    )
    _require(
        observed["protocol_cell_state_receipt"]["primary_state_counts"]
        == dict(EXPECTED_PRIMARY_STATE_COUNTS)
        and observed["protocol_cell_state_receipt"]["primary_executed_cell_count"] == 0
        and observed["protocol_cell_state_receipt"]["withdrawn_forensic_cell_count"]
        == EXPECTED_FORENSIC_WITHDRAWN_CELL_COUNT,
        "authority protocol cell-state receipt changed",
    )
    for name in (DATA.MANIFEST_FILENAME, CONTRACT.MANIFEST_FILENAME, *CONTRACT_REGISTRY_FILENAMES):
        _strict_json(bundle.files[name], label=f"bound {name}")


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
        raise SemanticAuthorityError(f"cannot anchor {label}: {absolute}") from exc
    try:
        status = os.fstat(descriptor)
        _require(stat.S_ISDIR(status.st_mode), f"{label} is not a directory")
        identity = _directory_identity(status)
        lexical = absolute.lstat()
        _require(
            stat.S_ISDIR(lexical.st_mode)
            and not stat.S_ISLNK(lexical.st_mode)
            and _directory_identity(lexical) == identity,
            f"{label} path identity changed while being anchored",
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, identity


def _verify_anchored_directory(
    path: Path,
    descriptor: int,
    identity: tuple[int, int],
    *,
    label: str,
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


def _create_directory_at(
    parent_descriptor: int,
    *,
    prefix: str,
) -> tuple[str, int, tuple[int, int]]:
    _simple_name(prefix, label="temporary directory prefix")
    for _attempt in range(128):
        name = f".{prefix}.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            status = os.fstat(descriptor)
            _require(stat.S_ISDIR(status.st_mode), "temporary entry is not a directory")
        except Exception:
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
        return name, descriptor, _directory_identity(status)
    raise SemanticAuthorityError("cannot allocate an exclusive temporary directory")


def _write_at(parent_descriptor: int, name: str, payload: bytes) -> tuple[int, int]:
    _simple_name(name, label="authority filename")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as exc:
        raise SemanticAuthorityError(f"cannot create staged authority file: {name}") from exc
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            _require(written > 0, f"zero-byte write for staged {name}")
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
        raise SemanticAuthorityError(f"cannot safely open {label}") from exc
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
        and stat.S_ISREG(entry.st_mode)
        and entry.st_nlink == 1
        and _directory_identity(entry) == _directory_identity(before)
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
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        try:
            for child in os.listdir(descriptor):
                _remove_tree_at(descriptor, child)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rmdir(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _remove_owned_at(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, int],
    *,
    label: str,
    strict: bool = False,
) -> bool:
    status = _entry_at(parent_descriptor, name)
    if status is None:
        return False
    if _directory_identity(status) != identity:
        _require(not strict, f"refusing to clean replaced {label}")
        return False
    _remove_tree_at(parent_descriptor, name)
    return True


def _rename_noreplace(parent_descriptor: int, source: str, destination: str) -> None:
    _simple_name(source, label="rename source")
    _simple_name(destination, label="rename destination")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "Linux atomic RENAME_NOREPLACE is unavailable")
    assert renameat2 is not None
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
            raise SemanticAuthorityError(
                f"refusing to overwrite existing authority destination: {destination}"
            )
        raise SemanticAuthorityError(f"atomic RENAME_NOREPLACE failed: {os.strerror(error)}")


def _validate_destination(destination: Path) -> tuple[Path, Path]:
    absolute = _absolute(destination)
    _simple_name(absolute.name, label="authority destination")
    parent = _require_no_symlink_components(absolute.parent, label="authority output parent")
    _require(not os.path.lexists(absolute), f"authority destination already exists: {absolute}")
    return absolute, parent


def _rebuild_parent(output_parent: Path) -> Path:
    absolute_parent = _absolute(output_parent)
    final_root = _absolute(FINAL_OUTPUT_ROOT)
    try:
        absolute_parent.relative_to(final_root)
    except ValueError:
        return absolute_parent
    return _require_no_symlink_components(final_root.parent, label="candidate rebuild parent")


def _rebuild_candidates(work_directory: Path) -> tuple[Any, Any, Any, Any]:
    data_config = DATA.BuildConfig.production()
    data_paths = DATA.RegistryInputPaths.production(ROOT)
    semantic_data_directory = work_directory / "semantic-data-candidate"
    try:
        written, data_candidate = DATA.write_candidate(
            data_paths,
            semantic_data_directory,
            config=data_config,
            evidence_root=ROOT,
        )
    except DATA.SemanticRegistryError as exc:
        raise SemanticAuthorityError(f"semantic-data candidate rebuild failed: {exc}") from exc
    _require(
        _absolute(written) == _absolute(semantic_data_directory),
        "semantic-data candidate writer returned another destination",
    )
    contract_paths = CONTRACT.RegistryInputPaths.production(semantic_data_directory)
    contract_config = CONTRACT.BuildConfig(enforce_route_a_runtime=True)
    try:
        contract_candidate = CONTRACT.build_twice(contract_paths, config=contract_config)
    except CONTRACT.SemanticContractError as exc:
        raise SemanticAuthorityError(f"semantic-contract candidate rebuild failed: {exc}") from exc
    _verify_candidate_cross_bindings(
        data_candidate,
        contract_candidate,
        data_config=data_config,
        require_production=True,
    )
    return data_candidate, contract_candidate, data_config, (contract_paths, contract_config)


def _revalidate_candidate_sources(
    data_candidate: Any,
    contract_candidate: Any,
    contract_context: tuple[Any, Any],
) -> None:
    try:
        DATA._revalidate_snapshot(data_candidate.snapshot, evidence_root=ROOT)
    except DATA.SemanticRegistryError as exc:
        raise SemanticAuthorityError(f"semantic-data source revalidation failed: {exc}") from exc
    contract_paths, contract_config = contract_context
    try:
        CONTRACT._revalidate_sources(
            contract_paths,
            contract_config,
            contract_candidate.snapshot,
        )
    except CONTRACT.SemanticContractError as exc:
        raise SemanticAuthorityError(
            f"semantic-contract source revalidation failed: {exc}"
        ) from exc


def publish_authority(destination: Path) -> tuple[Path, AuthorityBundle]:
    """Rebuild fixed production candidates and create exactly one authority directory."""

    destination, output_parent = _validate_destination(destination)
    output_descriptor, output_identity = _open_anchored_directory(
        output_parent, label="authority output parent"
    )
    destination_name = destination.name
    lock_name = f".{destination_name}.semantic-authority-create.lock"
    lock_identity: tuple[int, int] | None = None
    stage_name: str | None = None
    stage_descriptor: int | None = None
    stage_identity: tuple[int, int] | None = None
    rebuild_parent = _rebuild_parent(output_parent)
    rebuild_descriptor: int | None = None
    rebuild_parent_identity: tuple[int, int] | None = None
    work_name: str | None = None
    work_descriptor: int | None = None
    work_identity: tuple[int, int] | None = None
    bundle: AuthorityBundle | None = None
    committed = False
    lock_descriptor = -1
    try:
        _require(
            _entry_at(output_descriptor, destination_name) is None,
            "authority destination appeared before build",
        )
        lock_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_descriptor = os.open(lock_name, lock_flags, 0o600, dir_fd=output_descriptor)
        except FileExistsError as exc:
            raise SemanticAuthorityError(
                f"authority create lock already exists: {lock_name}; resume is unsupported"
            ) from exc
        lock_status = os.fstat(lock_descriptor)
        _require(
            stat.S_ISREG(lock_status.st_mode) and lock_status.st_nlink == 1,
            "authority create lock type/link count changed",
        )
        lock_identity = _directory_identity(lock_status)
        lock_payload = f"pid={os.getpid()}\nstatus={STATUS}\n".encode("ascii")
        _require(
            os.write(lock_descriptor, lock_payload) == len(lock_payload), "lock write was short"
        )
        os.fsync(lock_descriptor)
        os.close(lock_descriptor)
        lock_descriptor = -1
        os.fsync(output_descriptor)

        attestations = _capture_attestation_sources()
        rebuild_descriptor, rebuild_parent_identity = _open_anchored_directory(
            rebuild_parent, label="candidate rebuild parent"
        )
        work_name, work_descriptor, work_identity = _create_directory_at(
            rebuild_descriptor,
            prefix=f"{destination_name}.semantic-authority-rebuild",
        )
        os.fsync(rebuild_descriptor)
        work_directory = rebuild_parent / work_name
        _verify_anchored_directory(
            work_directory,
            work_descriptor,
            work_identity,
            label="candidate rebuild workspace before build",
        )
        data_candidate, contract_candidate, data_config, contract_context = _rebuild_candidates(
            work_directory
        )
        bundle = _build_authority_bundle(
            data_candidate,
            contract_candidate,
            attestations,
            data_config=data_config,
            require_production=True,
        )

        stage_name, stage_descriptor, stage_identity = _create_directory_at(
            output_descriptor,
            prefix=f"{destination_name}.semantic-authority-stage",
        )
        for name in sorted(bundle.files):
            _write_at(stage_descriptor, name, bundle.files[name])
        os.fsync(stage_descriptor)
        os.fsync(output_descriptor)
        _verify_authority_bundle(bundle)
        _require(
            set(os.listdir(stage_descriptor)) == set(ALL_FILENAMES),
            "staged authority file set changed",
        )
        for name, payload in bundle.files.items():
            _require(
                _read_at(stage_descriptor, name, label=f"staged {name}") == payload,
                f"staged authority bytes changed: {name}",
            )

        _revalidate_attestation_sources(attestations)
        _revalidate_candidate_sources(data_candidate, contract_candidate, contract_context)
        _verify_authority_bundle(bundle)
        _verify_anchored_directory(
            work_directory,
            work_descriptor,
            work_identity,
            label="candidate rebuild workspace before cleanup",
        )
        os.close(work_descriptor)
        work_descriptor = None
        _remove_owned_at(
            rebuild_descriptor,
            work_name,
            work_identity,
            label="candidate rebuild workspace",
            strict=True,
        )
        work_name = None
        work_identity = None
        os.fsync(rebuild_descriptor)
        _verify_anchored_directory(
            output_parent,
            output_descriptor,
            output_identity,
            label="authority output parent before rename",
        )
        _verify_anchored_directory(
            rebuild_parent,
            rebuild_descriptor,
            rebuild_parent_identity,
            label="candidate rebuild parent before commit",
        )
        current_lock_status = _entry_at(output_descriptor, lock_name)
        _require(
            current_lock_status is not None
            and _directory_identity(current_lock_status) == lock_identity
            and stat.S_ISREG(current_lock_status.st_mode)
            and current_lock_status.st_nlink == 1,
            "authority create lock was replaced before commit",
        )
        _require(
            _entry_at(output_descriptor, destination_name) is None,
            "authority destination appeared during build",
        )
        _rename_noreplace(output_descriptor, stage_name, destination_name)
        destination_status = _entry_at(output_descriptor, destination_name)
        _require(
            destination_status is not None
            and stat.S_ISDIR(destination_status.st_mode)
            and _directory_identity(destination_status) == stage_identity,
            "committed authority inode differs from stage",
        )
        _verify_anchored_directory(
            output_parent,
            output_descriptor,
            output_identity,
            label="authority output parent after rename",
        )
        _require(
            set(os.listdir(stage_descriptor)) == set(ALL_FILENAMES),
            "committed authority file set changed",
        )
        for name, payload in bundle.files.items():
            _require(
                _read_at(stage_descriptor, name, label=f"committed {name}") == payload,
                f"committed authority bytes changed: {name}",
            )
        _remove_owned_at(
            output_descriptor,
            lock_name,
            lock_identity,
            label="authority create lock",
            strict=True,
        )
        lock_identity = None
        os.fsync(output_descriptor)
        committed = True
    finally:
        if lock_descriptor >= 0:
            try:
                os.close(lock_descriptor)
            except OSError:
                pass
        if stage_descriptor is not None:
            try:
                os.close(stage_descriptor)
            except OSError:
                pass
            stage_descriptor = None
        if not committed and stage_identity is not None:
            if stage_name is not None:
                _remove_owned_at(
                    output_descriptor,
                    stage_name,
                    stage_identity,
                    label="authority staging directory",
                )
            _remove_owned_at(
                output_descriptor,
                destination_name,
                stage_identity,
                label="failed authority destination",
            )
        if lock_identity is not None:
            _remove_owned_at(
                output_descriptor,
                lock_name,
                lock_identity,
                label="authority create lock",
            )
        if work_descriptor is not None:
            try:
                os.close(work_descriptor)
            except OSError:
                pass
            work_descriptor = None
        if rebuild_descriptor is not None:
            if work_name is not None and work_identity is not None:
                _remove_owned_at(
                    rebuild_descriptor,
                    work_name,
                    work_identity,
                    label="candidate rebuild workspace",
                    strict=True,
                )
                os.fsync(rebuild_descriptor)
            os.close(rebuild_descriptor)
        if not committed or lock_identity is not None:
            try:
                os.fsync(output_descriptor)
            except OSError:
                pass
        os.close(output_descriptor)
    _require(bundle is not None, "authority publication completed without a bundle")
    return destination, bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "new create-only authority directory; canonical production path is "
            "outputs/final/semantic_registries_v4_authority"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination, bundle = publish_authority(args.output_dir)
    manifest = bundle.manifest
    result = {
        "status": STATUS,
        "authority_class": manifest["authority_class"],
        "execution_authorized": False,
        "forcing_protocol_seal_bound": False,
        "output_dir": str(destination),
        "primary_state_counts": dict(EXPECTED_PRIMARY_STATE_COUNTS),
        "withdrawn_forensic_cells": EXPECTED_FORENSIC_WITHDRAWN_CELL_COUNT,
        "files": {
            name: {"sha256": _sha256(payload), "size_bytes": len(payload)}
            for name, payload in sorted(bundle.files.items())
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SemanticAuthorityError as exc:
        print(f"semantic-registry authority publication failed closed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
