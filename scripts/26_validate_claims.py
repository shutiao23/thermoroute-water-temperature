#!/usr/bin/env python3
"""Validate the evidence-derived Route-A claim ledger and manuscript blocks.

The validator has no command-line phase switch.  For the v2 ledger the phase is
derived only from the canonical authorization, its canonical intent marker and
``thermoroute.opening.validate_completed_receipt``.  Consequently an interrupted
opening is not silently treated as either PRE or POST.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
from fnmatch import fnmatch
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence, cast


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "protocols" / "route_a_claim_registry_v1.json"

V1_FORMAT = "thermoroute.route-a-claim-registry.v1"
V2_FORMAT = "thermoroute.route-a-claim-ledger.v2"
PRE_PHASE = "PRE_CONFIRMATION_LABELS_SEALED"
POST_PHASE = "POST_CONFIRMATION_VERIFIED"
INDETERMINATE_PHASE = "INDETERMINATE_FAIL_CLOSED"
AUTHORIZATION_FORMAT = "thermoroute.route-a-opening-authorization.v1"
STATISTICS_FORMAT = "thermoroute.route-a-confirmatory-statistics.v1"

BLOCK_START = re.compile(
    r"^<!-- ROUTE_A_CLAIM (?P<claim_id>[A-Za-z0-9_.-]+) "
    r"sha256=(?P<sha256>[0-9a-f]{64}) -->\n",
    flags=re.MULTILINE,
)
BLOCK_END = "\n<!-- END ROUTE_A_CLAIM -->"
BLOCK_TOKEN = "ROUTE_A_CLAIM"


class ClaimRegistryError(RuntimeError):
    """The claim evidence or registry is malformed or indeterminate."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ClaimRegistryError("claim ledger contains non-canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: object, *, require_file: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ClaimRegistryError("claim-ledger path is not a non-empty relative path")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ClaimRegistryError(f"claim-ledger path escapes repository: {relative}")
    if require_file and not path.is_file():
        raise ClaimRegistryError(f"required claim evidence is absent: {relative}")
    return path


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClaimRegistryError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ClaimRegistryError(f"{label} is not a JSON object")
    return value


def _load_registry(path: Path) -> Mapping[str, Any]:
    document = _load_json(path, label="claim registry")
    format_name = document.get("format")
    if format_name not in {V1_FORMAT, V2_FORMAT}:
        raise ClaimRegistryError("unsupported claim-registry format")
    claims = document.get("claims")
    if format_name == V1_FORMAT:
        if not isinstance(claims, list) or not claims:
            raise ClaimRegistryError("claim registry is empty")
        identifiers = [str(claim.get("claim_id", "")) for claim in claims]
        if not all(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ClaimRegistryError("claim identifiers are empty or duplicated")
        return document

    required = {
        "format",
        "scope",
        "protocol_binding",
        "documents",
        "phase_resolver",
        "permanent_constraints",
        "claim_templates",
        "result_claim_specs",
        "required_documents",
        "required_postopen_coverage",
        "required_permanent_coverage",
        "decision_rule",
        "free_text_lints",
        "free_text_policy",
        "preopen_document_sha256",
        "postopen_document_transform",
    }
    if set(document) != required:
        raise ClaimRegistryError("v2 claim-ledger top-level schema changed")
    if document.get("scope") != "Route A only":
        raise ClaimRegistryError("v2 claim ledger has an unsupported scope")
    patterns = document.get("documents")
    if (
        not isinstance(patterns, list)
        or not patterns
        or not all(isinstance(item, str) and item for item in patterns)
    ):
        raise ClaimRegistryError("document registry is malformed")
    required_documents = document.get("required_documents")
    if (
        not isinstance(required_documents, list)
        or not required_documents
        or not all(
            isinstance(item, str) and item and not any(character in item for character in "*?[")
            for item in required_documents
        )
    ):
        raise ClaimRegistryError("exact required-document registry is malformed")
    if len(required_documents) != len(set(required_documents)) or any(
        not any(fnmatch(item, pattern) for pattern in patterns) for item in required_documents
    ):
        raise ClaimRegistryError("required documents are duplicated or outside scan scope")
    preopen_document_sha256 = document.get("preopen_document_sha256")
    if (
        not isinstance(preopen_document_sha256, Mapping)
        or set(preopen_document_sha256) != set(required_documents)
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in preopen_document_sha256.values()
        )
    ):
        raise ClaimRegistryError(
            "pre-opening document SHA-256 registry is not an exact required-document map"
        )
    if document.get("postopen_document_transform") != {
        "mode": "EXACT_PREOPEN_BYTES_PLUS_DETERMINISTIC_RESULT_SUFFIX",
        "heading": "# Route-A receipt-derived results",
        "preamble": (
            "This section was generated only after a verified one-time receipt. "
            "It is the canonical post-opening result layer and supersedes the "
            "pre-opening readiness/status statements above; it does not alter "
            "the frozen methods or limitations."
        ),
        "separator": "TWO_NEWLINES",
        "terminal_newline": True,
    }:
        raise ClaimRegistryError("post-opening document transform changed")
    resolver = document.get("phase_resolver")
    if not isinstance(resolver, Mapping) or resolver.get("mode") != ("DERIVE_NEVER_CLI_OVERRIDE"):
        raise ClaimRegistryError("claim phase is not evidence-derived")
    authorization = resolver.get("canonical_authorization")
    namespace_glob = resolver.get("namespace_glob")
    if (
        authorization != "data_usgs/confirmatory_opening_authorization_v1.json"
        or namespace_glob != "outputs/confirmatory/route_a_*"
        or set(resolver) != {"mode", "canonical_authorization", "namespace_glob"}
    ):
        raise ClaimRegistryError("phase resolver does not use canonical Route-A paths")

    constraints = document.get("permanent_constraints")
    templates = document.get("claim_templates")
    specs = document.get("result_claim_specs")
    coverage = document.get("required_postopen_coverage")
    permanent_coverage = document.get("required_permanent_coverage")
    lints = document.get("free_text_lints")
    decision_rule = document.get("decision_rule")
    if not isinstance(constraints, list) or not constraints:
        raise ClaimRegistryError("permanent constraint registry is empty")
    if not isinstance(templates, Mapping) or not templates:
        raise ClaimRegistryError("claim-template registry is empty")
    if not isinstance(specs, list) or len(specs) != 5:
        raise ClaimRegistryError("result claim registry must contain exactly five specs")
    if not isinstance(coverage, Mapping) or set(coverage) != {"claim_ids", "multiplicity", "phase"}:
        raise ClaimRegistryError("post-opening coverage contract is malformed")
    if coverage.get("multiplicity") != "EACH_EXACTLY_ONCE" or coverage.get("phase") != POST_PHASE:
        raise ClaimRegistryError("post-opening coverage is not exact-once POST")
    claim_ids = coverage.get("claim_ids")
    if not isinstance(claim_ids, list) or len(claim_ids) != 5:
        raise ClaimRegistryError("post-opening coverage must name exactly five claims")
    if len(set(str(value) for value in claim_ids)) != 5:
        raise ClaimRegistryError("post-opening coverage claim IDs are duplicated")
    spec_ids = [str(spec.get("claim_id", "")) for spec in specs]
    if spec_ids != [str(value) for value in claim_ids]:
        raise ClaimRegistryError("result specs and exact coverage order differ")
    if not isinstance(lints, list) or not lints:
        raise ClaimRegistryError("free-text lint registry is empty")
    if document.get("free_text_policy") != (
        "REQUIRED_PREOPEN_DOCUMENT_BYTES_ARE_SHA256_FROZEN; POST MAY ONLY "
        "APPEND THE DETERMINISTIC RESULT SUFFIX TO DECLARED TARGETS; "
        "STRUCTURED_CLAIM_BLOCKS_ARE_AUTHORITATIVE; regexes are "
        "defense-in-depth lint only"
    ):
        raise ClaimRegistryError("formal-result/free-text trust boundary changed")
    expected_decision_rule = {
        "protocol_predicates": [
            {
                "pointer": (
                    "/primary_inference_contract/"
                    "confirmatory_claim_decision_contract/"
                    "all_five_tests_must_be_rendered_exactly_once"
                ),
                "operator": "equals",
                "expected": True,
            },
            {
                "pointer": (
                    "/primary_inference_contract/"
                    "confirmatory_claim_decision_contract/supported_if_and_only_if"
                ),
                "operator": "contains",
                "expected": "ci_high_c < margin_c",
            },
            {
                "pointer": (
                    "/primary_inference_contract/"
                    "confirmatory_claim_decision_contract/"
                    "noninferiority_wording_limit"
                ),
                "operator": "contains",
                "expected": "may not say equivalent, parity",
            },
        ],
        "support_requires": [
            "authorization.inference_gate.claim_eligible == true",
            "statistics.outcome_qc_gate.directional_claims_allowed == true",
            "receipt.temporal_coverage_audit.physical_replay_verified == true",
            "status == ESTIMABLE",
            "p_holm <= 0.05",
            "ci_high_c < margin_c",
        ],
        "inference_gate_failure": "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
        "outcome_qc_gate_failure": ("DESCRIPTIVE_ONLY_OUTCOME_QC_GATE_FAILED"),
        "both_gates_failure": "DESCRIPTIVE_ONLY_BOTH_GATES_FAILED",
        "p_ci_disagreement": "EVIDENCE_CONFLICT_NOT_SUPPORTED",
        "coherent_failure": "NOT_ESTABLISHED",
        "nonestimable": "NOT_ESTIMABLE",
        "h1_interpretation": ("margin-0 superiority for the named comparison only"),
        "h2_interpretation": (
            "satisfies only the preregistered +0.05 degrees C numerical "
            "non-inferiority ceiling; never equivalence or parity"
        ),
    }
    if decision_rule != expected_decision_rule:
        raise ClaimRegistryError("confirmatory claim decision rule changed")
    if any(
        template_id not in templates
        for template_id in (
            "SUPERIORITY_SUPPORTED",
            "NONINFERIORITY_SUPPORTED",
            "EVIDENCE_CONFLICT_NOT_SUPPORTED",
            "NOT_ESTABLISHED",
            "NOT_ESTIMABLE",
            "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
            "DESCRIPTIVE_ONLY_OUTCOME_QC_GATE_FAILED",
            "DESCRIPTIVE_ONLY_BOTH_GATES_FAILED",
        )
    ):
        raise ClaimRegistryError("decision rule lacks a fixed result template")

    all_entries: list[Mapping[str, Any]] = []
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            raise ClaimRegistryError("permanent constraint is not an object")
        entry = constraint.get("claim")
        predicates = constraint.get("protocol_predicates")
        if (
            set(constraint) != {"constraint_id", "protocol_predicates", "claim", "lint_regex"}
            or not str(constraint.get("constraint_id", ""))
            or not isinstance(predicates, list)
            or not predicates
            or not isinstance(entry, Mapping)
        ):
            raise ClaimRegistryError("permanent constraint schema is malformed")
        all_entries.append(entry)
        regexes = constraint.get("lint_regex")
        if not isinstance(regexes, list) or not all(isinstance(item, str) for item in regexes):
            raise ClaimRegistryError("permanent-constraint lint registry is malformed")
    all_entries.extend(spec for spec in specs if isinstance(spec, Mapping))
    if len(all_entries) != len(constraints) + len(specs):
        raise ClaimRegistryError("claim entry registry is malformed")
    entry_ids = []
    for entry in all_entries:
        _validate_claim_entry(entry, templates=templates)
        entry_ids.append(str(entry["claim_id"]))
    if len(entry_ids) != len(set(entry_ids)):
        raise ClaimRegistryError("v2 claim IDs are duplicated")
    if spec_ids != entry_ids[-5:]:
        raise ClaimRegistryError("result claim registry is not exact")
    permanent_ids = [str(constraint["claim"]["claim_id"]) for constraint in constraints]
    if (
        not isinstance(permanent_coverage, Mapping)
        or set(permanent_coverage) != {"claim_ids", "multiplicity", "phases"}
        or permanent_coverage.get("claim_ids") != permanent_ids
        or permanent_coverage.get("multiplicity") != "EACH_EXACTLY_ONCE"
        or permanent_coverage.get("phases") != [PRE_PHASE, POST_PHASE]
    ):
        raise ClaimRegistryError("permanent limitation coverage is not exact-once")
    for entry in all_entries:
        if any(
            not any(fnmatch(target, pattern) for pattern in patterns)
            for target in entry["render_targets"]
        ):
            raise ClaimRegistryError("claim render target is outside canonical documents")
        if any(target not in required_documents for target in entry["render_targets"]):
            raise ClaimRegistryError("claim render target is not a SHA-frozen required document")
        if entry["kind"] == "CONFIRMATORY_RESULT" and len(entry["render_targets"]) != 1:
            raise ClaimRegistryError("exact-once result claim must have one render target")
    return document


def _validate_claim_entry(entry: Mapping[str, Any], *, templates: Mapping[str, Any]) -> None:
    required = {
        "claim_id",
        "scope",
        "kind",
        "polarity",
        "phase_allowed",
        "template_id",
        "evidence",
        "render_targets",
    }
    if set(entry) != required:
        raise ClaimRegistryError("claim entry fields differ from ledger v2")
    if not all(
        str(entry.get(key, "")).strip()
        for key in ("claim_id", "scope", "kind", "polarity", "template_id")
    ):
        raise ClaimRegistryError("claim entry contains an empty identity field")
    phases = entry.get("phase_allowed")
    if not isinstance(phases, list) or not phases or set(phases) - {PRE_PHASE, POST_PHASE}:
        raise ClaimRegistryError("claim entry has an invalid phase allowance")
    evidence = entry.get("evidence")
    targets = entry.get("render_targets")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ClaimRegistryError("claim entry lacks structured evidence")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(value, str) and value for value in targets)
    ):
        raise ClaimRegistryError("claim entry lacks render targets")
    if entry["kind"] == "NEGATED_LIMITATION":
        if entry["polarity"] != "NEGATED" or entry["template_id"] not in templates:
            raise ClaimRegistryError("limitation claim is not a fixed negated template")
        if evidence.get("artifact") != "protocol":
            raise ClaimRegistryError("limitation claim is not protocol-backed")
    elif entry["kind"] == "CONFIRMATORY_RESULT":
        if (
            entry["polarity"] != "AUTO_FROM_VERIFIED_STATISTICS"
            or entry["template_id"] != "AUTO_CONFIRMATORY_DECISION"
            or phases != [POST_PHASE]
            or set(evidence) != {"artifact", "test_id"}
            or evidence.get("artifact") != "statistics"
        ):
            raise ClaimRegistryError("result claim is not statistics-derived POST-only")
    else:
        raise ClaimRegistryError("unsupported structured claim kind")


def _documents(root: Path, patterns: Sequence[str]) -> list[Path]:
    candidates = [path for path in root.rglob("*") if path.is_file()]
    selected = {
        path
        for path in candidates
        if any(fnmatch(path.relative_to(root).as_posix(), pattern) for pattern in patterns)
    }
    return sorted(selected)


def _json_pointer(document: object, pointer: object) -> object:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ClaimRegistryError("protocol predicate has an invalid JSON pointer")
    current = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ClaimRegistryError(f"protocol predicate pointer is absent: {pointer}")
    return current


def _validate_protocol_binding(root: Path, registry: Mapping[str, Any]) -> Mapping[str, Any]:
    binding = registry.get("protocol_binding")
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ClaimRegistryError("protocol binding is malformed")
    expected_sha = binding.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ClaimRegistryError("protocol binding lacks an exact SHA-256")
    path = _inside(root, binding.get("path"), require_file=True)
    if _sha256_file(path) != expected_sha:
        raise ClaimRegistryError("claim ledger protocol SHA-256 changed")
    protocol = _load_json(path, label="claim-ledger protocol")

    def validate_predicate(predicate: object, *, label: str) -> None:
        if not isinstance(predicate, Mapping) or set(predicate) != {
            "pointer",
            "operator",
            "expected",
        }:
            raise ClaimRegistryError("protocol predicate schema is malformed")
        actual = _json_pointer(protocol, predicate["pointer"])
        operator = predicate["operator"]
        expected = predicate["expected"]
        passed = False
        if operator == "equals":
            passed = actual == expected
        elif operator == "contains":
            passed = isinstance(actual, str) and isinstance(expected, str) and expected in actual
        elif operator == "contains_all":
            passed = (
                isinstance(actual, list)
                and isinstance(expected, list)
                and all(value in actual for value in expected)
            )
        else:
            raise ClaimRegistryError(f"unsupported protocol predicate operator: {operator}")
        if not passed:
            raise ClaimRegistryError(f"protocol no longer supports {label}: {predicate['pointer']}")

    for constraint in registry["permanent_constraints"]:
        for predicate in constraint["protocol_predicates"]:
            validate_predicate(
                predicate, label=f"permanent constraint {constraint['constraint_id']}"
            )
    for predicate in registry["decision_rule"]["protocol_predicates"]:
        validate_predicate(predicate, label="confirmatory claim decision rule")
    return protocol


def _opening_api(root: Path) -> tuple[Any, Any]:
    """Load the repository's frozen verifier, never a caller-supplied phase."""
    source = (root / "src").resolve()
    if not source.is_dir():
        raise ClaimRegistryError("Route-A opening verifier source is absent")
    source_text = str(source)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        module = importlib.import_module("thermoroute.opening")
    except Exception as exc:  # dependency/import failure makes phase unknowable
        raise ClaimRegistryError("cannot load the canonical opening verifier") from exc
    finally:
        if inserted and sys.path and sys.path[0] == source_text:
            sys.path.pop(0)
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != (source / "thermoroute" / "opening.py").resolve():
        raise ClaimRegistryError("imported opening verifier is noncanonical")
    return module.validate_authorization, module.validate_completed_receipt


def _outcome_qc_structure_api(root: Path) -> Any:
    """Load the single canonical deep gate validator used by opening."""
    source = (root / "src").resolve()
    source_text = str(source)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        module = importlib.import_module("thermoroute.outcome_qc")
    except Exception as exc:
        raise ClaimRegistryError("cannot load the canonical outcome-QC verifier") from exc
    finally:
        if inserted and sys.path and sys.path[0] == source_text:
            sys.path.pop(0)
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != (source / "thermoroute" / "outcome_qc.py").resolve():
        raise ClaimRegistryError("imported outcome-QC verifier is noncanonical")
    return module.validate_outcome_qc_gate_structure


def _coverage_replay_api(root: Path) -> Any:
    """Load the canonical physical coverage replay only after POST verification."""
    source = (root / "src").resolve()
    source_text = str(source)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        module = importlib.import_module("thermoroute.coverage_bridge")
    except Exception as exc:
        raise ClaimRegistryError(
            "cannot load the canonical temporal-coverage verifier"
        ) from exc
    finally:
        if inserted and sys.path and sys.path[0] == source_text:
            sys.path.pop(0)
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != (source / "thermoroute" / "coverage_bridge.py").resolve():
        raise ClaimRegistryError(
            "imported temporal-coverage verifier is noncanonical"
        )
    return module.replay_temporal_coverage_from_physical_files


def _validate_authorization(
    authorization_path: Path, *, root: Path, allow_gitless_archive: bool
) -> Mapping[str, Any]:
    validate, _ = _opening_api(root)
    try:
        return validate(
            authorization_path,
            root=root,
            require_clean_source=False,
            allow_gitless_archive=allow_gitless_archive,
        )
    except Exception as exc:
        raise ClaimRegistryError("canonical opening authorization did not verify") from exc


def _validate_completed_receipt(
    authorization_path: Path, *, root: Path, allow_gitless_archive: bool
) -> Mapping[str, Any]:
    _, validate = _opening_api(root)
    try:
        return validate(
            authorization_path,
            root=root,
            allow_gitless_archive=allow_gitless_archive,
        )
    except Exception as exc:
        raise ClaimRegistryError("canonical completed receipt did not verify") from exc


def _canonical_state(authorization: Mapping[str, Any], *, root: Path) -> Mapping[str, str]:
    if (
        authorization.get("format") != AUTHORIZATION_FORMAT
        or authorization.get("status") != "AUTHORIZED_LABELS_STILL_SEALED"
    ):
        raise ClaimRegistryError("canonical authorization identity/status is invalid")
    state = authorization.get("state_paths")
    if not isinstance(state, Mapping):
        raise ClaimRegistryError("canonical authorization lacks state paths")
    namespace = state.get("namespace")
    if not isinstance(namespace, str) or not re.fullmatch(r"[0-9a-f]{24}", namespace):
        raise ClaimRegistryError("Route-A state namespace is malformed")
    base = f"outputs/confirmatory/route_a_{namespace}"
    expected = {
        "run_directory": base,
        "intent": f"{base}/opening_intent_v1.json",
        "statistics": f"{base}/trusted/statistics_v1.json",
        "temporal_coverage_audit": (
            f"{base}/trusted/temporal_coverage_audit_v1.json"
        ),
        "outcome_qc_gate": f"{base}/trusted/outcome_qc_gate_v1.json",
        "report": f"{base}/trusted/report_v1.md",
        "receipt": f"{base}/opening_receipt_v1.json",
        "receipt_sha256": f"{base}/opening_receipt_v1.sha256",
    }
    for key, relative in expected.items():
        if state.get(key) != relative:
            raise ClaimRegistryError(f"authorization has noncanonical state path: {key}")
        _inside(root, relative)
    return {str(key): str(value) for key, value in state.items()}


def _namespace_entries(root: Path, namespace_glob: str) -> list[Path]:
    entries: list[Path] = []
    for directory in root.glob(namespace_glob):
        if directory.is_file():
            entries.append(directory)
        elif directory.is_dir():
            entries.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(entries)


def resolve_phase(*, root: Path, registry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Derive PRE/POST; every partial or unverifiable state raises."""
    root = root.resolve()
    resolver = registry["phase_resolver"]
    authorization_relative = resolver["canonical_authorization"]
    authorization_path = _inside(root, authorization_relative)
    namespace_entries = _namespace_entries(root, resolver["namespace_glob"])
    if not authorization_path.exists():
        if namespace_entries:
            raise ClaimRegistryError(
                "Route-A phase is INDETERMINATE: namespace artifacts exist without "
                "the canonical authorization"
            )
        return {"phase": PRE_PHASE, "authorization": None, "receipt": None}
    if not authorization_path.is_file():
        raise ClaimRegistryError("Route-A phase is INDETERMINATE: authorization is not a file")
    authorization = _load_json(authorization_path, label="canonical authorization")
    source = authorization.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("authorization_path") != authorization_relative
    ):
        raise ClaimRegistryError("canonical authorization points to another authorization")
    binding = authorization.get("protocol")
    expected_binding = registry["protocol_binding"]
    if (
        not isinstance(binding, Mapping)
        or binding.get("path") != expected_binding["path"]
        or binding.get("sha256") != expected_binding["sha256"]
    ):
        raise ClaimRegistryError("authorization and claim ledger bind different protocols")
    state = _canonical_state(authorization, root=root)
    intent_path = _inside(root, state["intent"])
    receipt_path = _inside(root, state["receipt"])
    intent_exists = intent_path.is_file()
    receipt_exists = receipt_path.is_file()
    if intent_exists != receipt_exists:
        raise ClaimRegistryError(
            "Route-A phase is INDETERMINATE: intent/receipt completion is partial"
        )
    if not intent_exists:
        if namespace_entries:
            raise ClaimRegistryError(
                "Route-A phase is INDETERMINATE: namespace artifacts exist before intent"
            )
        allow_gitless = not (root / ".git").exists()
        preflight = _validate_authorization(
            authorization_path, root=root, allow_gitless_archive=allow_gitless
        )
        return {
            "phase": PRE_PHASE,
            "authorization": authorization,
            "preflight": preflight,
            "receipt": None,
        }
    allow_gitless = not (root / ".git").exists()
    receipt = _validate_completed_receipt(
        authorization_path, root=root, allow_gitless_archive=allow_gitless
    )
    return {
        "phase": POST_PHASE,
        "authorization": authorization,
        "authorization_path": authorization_path,
        "receipt": receipt,
        "state": state,
    }


def _holm(raw: Sequence[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda index: (raw[index], index))
    adjusted = [0.0] * len(raw)
    running = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ClaimRegistryError(f"statistics {label} is boolean, not numeric")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ClaimRegistryError(f"statistics {label} is not numeric") from exc
    if not math.isfinite(number):
        raise ClaimRegistryError(f"statistics {label} is not finite")
    return number


def _load_verified_statistics(
    *,
    root: Path,
    registry: Mapping[str, Any],
    phase: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    if phase.get("phase") != POST_PHASE:
        raise ClaimRegistryError("result statistics requested outside verified POST")
    receipt = phase.get("receipt")
    state = phase.get("state")
    if not isinstance(receipt, Mapping) or not isinstance(state, Mapping):
        raise ClaimRegistryError("verified POST lacks receipt/state evidence")
    artifacts = receipt.get("artifacts")
    binding = artifacts.get("statistics") if isinstance(artifacts, Mapping) else None
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ClaimRegistryError("receipt lacks an exact statistics binding")
    if binding.get("path") != state.get("statistics"):
        raise ClaimRegistryError("receipt statistics path is noncanonical")
    statistics_path = _inside(root, binding["path"], require_file=True)
    if _sha256_file(statistics_path) != binding.get("sha256"):
        raise ClaimRegistryError("receipt-bound statistics SHA-256 changed")
    statistics = _load_json(statistics_path, label="receipt-bound statistics")
    if statistics.get("format") != STATISTICS_FORMAT:
        raise ClaimRegistryError("receipt-bound statistics format changed")
    inference = protocol.get("primary_inference_contract")
    if not isinstance(inference, Mapping):
        raise ClaimRegistryError("bound protocol lacks primary inference contract")
    protocol_ci = inference.get("confidence_interval")
    protocol_p = inference.get("one_sided_p_value")
    statistics_ci = statistics.get("confidence_interval")
    statistics_p = statistics.get("p_value")
    if (
        not isinstance(protocol_ci, Mapping)
        or not isinstance(protocol_p, Mapping)
        or not isinstance(statistics_ci, Mapping)
        or statistics_ci.get("method") != protocol_ci.get("method")
        or statistics_ci.get("draws") != protocol_ci.get("draws")
        or not isinstance(statistics_p, Mapping)
        or statistics_p.get("method") != protocol_p.get("method")
        or statistics_p.get("maximum_configurations_for_frozen_cohort")
        != protocol_p.get("maximum_configurations_for_frozen_cohort")
        or statistics_p.get("monte_carlo_used") is not False
        or statistics_p.get("assumption") != protocol_p.get("null_assumption")
        or statistics_p.get("enumeration_rule") != protocol_p.get("enumeration_rule")
        or statistics_p.get("legacy_seed_field") != protocol_p.get("legacy_seed_field")
        or statistics.get("multiplicity") != "Holm step-down across exactly five tests"
    ):
        raise ClaimRegistryError("statistics inference method differs from the protocol")
    rows = statistics.get("tests")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ClaimRegistryError("receipt-bound statistics must contain exactly five tests")
    formal_tests = receipt.get("formal_tests")
    if formal_tests != rows:
        raise ClaimRegistryError("statistics tests differ from receipt formal_tests")
    family = inference.get("confirmatory_family")
    if not isinstance(family, list) or len(family) != 5:
        raise ClaimRegistryError("bound protocol lacks the exact five-test family")
    protocol_ids = [str(item.get("test_id", "")) for item in family]
    required_claims = registry["required_postopen_coverage"]["claim_ids"]
    result_specs = registry["result_claim_specs"]
    spec_test_ids = [str(item["evidence"]["test_id"]) for item in result_specs]
    if spec_test_ids != protocol_ids or len(set(protocol_ids)) != 5:
        raise ClaimRegistryError("claim coverage differs from the protocol five-test family")
    if [str(item["claim_id"]) for item in result_specs] != required_claims:
        raise ClaimRegistryError("claim coverage registry changed")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ClaimRegistryError("statistics test row is not an object")
        test_id = str(row.get("test_id", ""))
        if not test_id or test_id in by_id:
            raise ClaimRegistryError("statistics test IDs are empty or duplicated")
        by_id[test_id] = row
    if list(by_id) != protocol_ids:
        raise ClaimRegistryError("statistics does not cover the exact five tests in order")
    raw_values: list[float] = []
    exact_row_keys = {
        "test_id",
        "candidate",
        "reference",
        "horizon",
        "margin_c",
        "effect_convention",
        "status",
        "median_effect_c",
        "ci_low_c",
        "ci_high_c",
        "n_stations",
        "n_clusters",
        "win_rate",
        "p_one_sided_raw",
        "bootstrap_seed",
        "sign_flip_seed_legacy_ignored",
        "sign_flip_configurations",
        "p_holm",
        "reject_at_0_05",
        "confidence_bound_supports_margin",
    }
    for test, row in zip(family, rows):
        if set(row) != exact_row_keys:
            raise ClaimRegistryError(
                f"statistics row schema changed for {test.get('test_id', '<unknown>')}"
            )
        expected_identity = {
            "test_id": test["test_id"],
            "candidate": test["candidate"],
            "reference": test["reference"],
            "horizon": int(test["horizon"]),
            "margin_c": float(test["margin_c"]),
        }
        for key, value in expected_identity.items():
            actual = row.get(key)
            if key == "margin_c":
                actual = _finite_number(actual, label=f"{test['test_id']}.{key}")
            if actual != value:
                raise ClaimRegistryError(
                    f"statistics identity changed for {test['test_id']}: {key}"
                )
        status = row.get("status")
        if row.get("effect_convention") != ("station_RMSE_ThermoRoute-minus-reference"):
            raise ClaimRegistryError("statistics effect convention changed")
        if row.get("bootstrap_seed") != test.get("bootstrap_seed") or row.get(
            "sign_flip_seed_legacy_ignored"
        ) != test.get("sign_flip_seed"):
            raise ClaimRegistryError("statistics frozen seed identity changed")
        raw = _finite_number(row.get("p_one_sided_raw"), label=f"{test['test_id']}.p_raw")
        holm = _finite_number(row.get("p_holm"), label=f"{test['test_id']}.p_holm")
        if not 0.0 <= raw <= 1.0 or not 0.0 <= holm <= 1.0:
            raise ClaimRegistryError("statistics p-value lies outside [0,1]")
        raw_values.append(raw)
        if status == "ESTIMABLE":
            effect = _finite_number(row.get("median_effect_c"), label="median_effect_c")
            ci_low = _finite_number(row.get("ci_low_c"), label="ci_low_c")
            ci_high = _finite_number(row.get("ci_high_c"), label="ci_high_c")
            win_rate = _finite_number(row.get("win_rate"), label="win_rate")
            n_stations_value = _finite_number(row.get("n_stations"), label="n_stations")
            n_clusters_value = _finite_number(row.get("n_clusters"), label="n_clusters")
            if not n_stations_value.is_integer() or not n_clusters_value.is_integer():
                raise ClaimRegistryError("statistics sample counts are not integers")
            n_stations, n_clusters = int(n_stations_value), int(n_clusters_value)
            if (
                n_stations <= 0
                or n_clusters < 2
                or not math.isfinite(effect)
                or ci_low > ci_high
                or not 0.0 <= win_rate <= 1.0
            ):
                raise ClaimRegistryError("estimable statistics has invalid sample counts")
            configurations_value = _finite_number(
                row.get("sign_flip_configurations"), label="sign_flip_configurations"
            )
            if not configurations_value.is_integer():
                raise ClaimRegistryError("sign-flip configuration count is not an integer")
            configurations = int(configurations_value)
            if configurations != 1 << n_clusters:
                raise ClaimRegistryError("exact sign-flip configuration count changed")
            if row.get("reject_at_0_05") is not (holm <= 0.05):
                raise ClaimRegistryError("statistics reject flag differs from Holm p-value")
            if row.get("confidence_bound_supports_margin") is not (
                ci_high < float(test["margin_c"])
            ):
                raise ClaimRegistryError("statistics CI/margin flag is inconsistent")
        elif status == "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS":
            for key in ("median_effect_c", "ci_low_c", "ci_high_c", "win_rate"):
                if row.get(key) is not None:
                    raise ClaimRegistryError("non-estimable statistics reports an effect")
            n_stations_value = _finite_number(row.get("n_stations"), label="n_stations")
            n_clusters_value = _finite_number(row.get("n_clusters"), label="n_clusters")
            if (
                not n_stations_value.is_integer()
                or not n_clusters_value.is_integer()
                or n_stations_value < 0
                or n_clusters_value < 0
            ):
                raise ClaimRegistryError("non-estimable attrition counts are invalid")
            if (
                raw != 1.0
                or row.get("reject_at_0_05") is not False
                or row.get("confidence_bound_supports_margin") is not False
                or row.get("sign_flip_configurations") is not None
            ):
                raise ClaimRegistryError("non-estimable statistics reports support")
        else:
            raise ClaimRegistryError(f"unsupported confirmatory status: {status}")
    expected_holm = _holm(raw_values)
    for row, expected in zip(rows, expected_holm):
        actual = _finite_number(row.get("p_holm"), label="p_holm")
        if actual != expected:
            raise ClaimRegistryError("Holm values are not the exact five-test adjustment")
    return statistics, str(binding["sha256"])


def _outcome_qc_claim_eligibility(
    *,
    root: Path,
    phase: Mapping[str, Any],
    statistics: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> bool:
    """Verify the receipt-bound QC gate before permitting directional prose."""
    receipt = phase.get("receipt")
    state = phase.get("state")
    authorization = phase.get("authorization")
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(state, Mapping)
        or not isinstance(authorization, Mapping)
    ):
        raise ClaimRegistryError("verified POST lacks outcome-QC evidence")
    artifacts = receipt.get("artifacts")
    receipt_binding = artifacts.get("outcome_qc_gate") if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(receipt_binding, Mapping)
        or set(receipt_binding) != {"path", "sha256"}
        or receipt_binding.get("path") != state.get("outcome_qc_gate")
    ):
        raise ClaimRegistryError("receipt lacks the canonical outcome-QC gate")
    gate_path = _inside(root, receipt_binding["path"], require_file=True)
    if _sha256_file(gate_path) != receipt_binding.get("sha256"):
        raise ClaimRegistryError("receipt-bound outcome-QC gate SHA-256 changed")
    gate = _load_json(gate_path, label="receipt-bound outcome-QC gate")
    policy_binding = authorization.get("outcome_qc_policy")
    gate_policy = gate.get("policy")
    if (
        not isinstance(policy_binding, Mapping)
        or set(policy_binding) != {"path", "sha256", "format", "policy_id", "required"}
        or policy_binding.get("required") is not True
        or policy_binding.get("format") != "thermoroute.route-a-outcome-qc-policy.v1"
        or not isinstance(gate_policy, Mapping)
        or dict(gate_policy)
        != {
            "path": policy_binding.get("path"),
            "sha256": policy_binding.get("sha256"),
            "policy_id": policy_binding.get("policy_id"),
        }
    ):
        raise ClaimRegistryError("outcome-QC gate differs from its frozen policy")
    policy_path = _inside(root, policy_binding["path"], require_file=True)
    if _sha256_file(policy_path) != policy_binding.get("sha256"):
        raise ClaimRegistryError("frozen outcome-QC policy SHA-256 changed")
    inference = protocol.get("primary_inference_contract")
    family = inference.get("confirmatory_family") if isinstance(inference, Mapping) else None
    availability = protocol.get("availability_contract")
    if (
        not isinstance(family, list)
        or len(family) != 5
        or not isinstance(availability, Mapping)
        or type(availability.get("minimum_valid_targets_per_station_horizon"))
        is not int
    ):
        raise ClaimRegistryError("bound protocol lacks the outcome-QC family")
    try:
        _outcome_qc_structure_api(root)(
            gate,
            root=root,
            policy_path=policy_path,
            protocol=protocol,
            minimum_targets=int(
                availability["minimum_valid_targets_per_station_horizon"]
            ),
        )
    except Exception as exc:
        raise ClaimRegistryError(
            "receipt-bound outcome-QC gate failed deep semantic validation"
        ) from exc
    single = gate.get("single_extreme_influence")
    leave_one = gate.get("leave_one_huc_direction")
    test_rows = statistics.get("tests")
    if (
        not isinstance(single, list)
        or not isinstance(leave_one, list)
        or not isinstance(test_rows, list)
        or len(test_rows) != 5
    ):
        raise ClaimRegistryError("outcome-QC gate lacks the five-test statistics")
    expected_test_ids = [str(row["test_id"]) for row in test_rows]
    if (
        [str(row.get("test_id", "")) for row in single if isinstance(row, Mapping)]
        != expected_test_ids
        or [str(row.get("test_id", "")) for row in leave_one if isinstance(row, Mapping)]
        != expected_test_ids
        or not all(
            isinstance(row, Mapping) and isinstance(row.get("pass"), bool)
            for row in [*single, *leave_one]
        )
    ):
        raise ClaimRegistryError("outcome-QC gate does not cover the exact five tests")
    for statistics_row, gate_row in zip(test_rows, single):
        if gate_row.get("n_reportable_stations") != statistics_row.get("n_stations"):
            raise ClaimRegistryError(
                "outcome-QC gate and formal statistics disagree on station count"
            )
        if (
            statistics_row.get("status") == "ESTIMABLE"
            and gate_row.get("primary_unfiltered_effect_c")
            != statistics_row.get("median_effect_c")
        ):
            raise ClaimRegistryError(
                "outcome-QC gate and formal statistics disagree on primary effect"
            )
    passed = bool(gate["pass"])
    statistics_binding = statistics.get("outcome_qc_gate")
    expected_statistics_binding = {
        "path": receipt_binding["path"],
        "sha256": receipt_binding["sha256"],
        "format": gate["format"],
        "status": gate["status"],
        "pass": passed,
        "directional_claims_allowed": passed,
    }
    if (
        not isinstance(statistics_binding, Mapping)
        or dict(statistics_binding) != expected_statistics_binding
    ):
        raise ClaimRegistryError("statistics outcome-QC binding or verdict changed")
    return passed


def _temporal_coverage_claim_evidence(
    *,
    root: Path,
    phase: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> None:
    """Require the receipt-bound, physically replayed nonfiltering audit."""
    receipt = phase.get("receipt")
    state = phase.get("state")
    authorization = phase.get("authorization")
    if not all(isinstance(value, Mapping) for value in (receipt, state, authorization)):
        raise ClaimRegistryError("verified POST lacks temporal-coverage evidence")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ClaimRegistryError("receipt lacks temporal-coverage artifacts")
    audit_binding = artifacts.get("temporal_coverage_audit")
    if (
        not isinstance(audit_binding, Mapping)
        or set(audit_binding) != {"path", "sha256"}
        or audit_binding.get("path") != state.get("temporal_coverage_audit")
    ):
        raise ClaimRegistryError("receipt lacks canonical temporal-coverage audit")
    expected_receipt_evidence = {
        **dict(audit_binding),
        "format": "thermoroute.route-a-temporal-coverage-audit.v1",
        "core_status": "DERIVED_CORE_REQUIRES_RECEIPT_BINDING",
        "physical_replay_verified": True,
        "source_binding_count": 11,
    }
    if receipt.get("temporal_coverage_audit") != expected_receipt_evidence:
        raise ClaimRegistryError(
            "receipt lacks verified temporal-coverage physical replay evidence"
        )
    audit_path = _inside(root, audit_binding["path"], require_file=True)
    if _sha256_file(audit_path) != audit_binding.get("sha256"):
        raise ClaimRegistryError("receipt-bound temporal-coverage audit changed")
    audit = _load_json(audit_path, label="receipt-bound temporal-coverage audit")
    stable = dict(audit)
    self_hash = stable.pop("audit_self_sha256", None)
    if (
        audit.get("format") != "thermoroute.route-a-temporal-coverage-audit.v1"
        or audit.get("status") != "DERIVED_CORE_REQUIRES_RECEIPT_BINDING"
        or not isinstance(self_hash, str)
        or self_hash
        != hashlib.sha256(_canonical_json(stable).encode("utf-8")).hexdigest()
        or audit.get("primary_statistics_unchanged") is not True
        or audit.get("primary_station_set_unchanged") is not True
        or audit.get("sensitivity_changes_primary_result_or_decision") is not False
        or audit.get("inference_computed") is not False
    ):
        raise ClaimRegistryError("temporal-coverage audit role/status changed")
    policy_binding = authorization.get("temporal_coverage_policy")
    if (
        not isinstance(policy_binding, Mapping)
        or set(policy_binding)
        != {"path", "sha256", "format", "policy_id", "status", "required"}
        or policy_binding.get("path")
        != "protocols/route_a_temporal_coverage_policy_v1.json"
        or policy_binding.get("format")
        != "thermoroute.route-a-temporal-coverage-policy.v1"
        or policy_binding.get("status") != "FROZEN_PRELABEL_OUTCOME_FREE"
        or policy_binding.get("required") is not True
    ):
        raise ClaimRegistryError("authorization lacks frozen temporal-coverage policy")
    policy_path = _inside(root, policy_binding["path"], require_file=True)
    if _sha256_file(policy_path) != policy_binding.get("sha256"):
        raise ClaimRegistryError("frozen temporal-coverage policy bytes changed")
    source_bindings = audit.get("source_bindings")
    registries = authorization.get("registries")
    if not isinstance(source_bindings, Mapping) or not isinstance(registries, Mapping):
        raise ClaimRegistryError("temporal-coverage source closure is absent")
    expected_sources = {
        "policy": {
            "path": policy_binding["path"],
            "sha256": policy_binding["sha256"],
        },
        "protocol": {
            "path": authorization["protocol"]["path"],
            "sha256": authorization["protocol"]["sha256"],
        },
        "acquisition_manifest": dict(artifacts["acquisition_manifest"]),
        "temporal_normalized_outcomes": dict(
            artifacts["temporal_normalized_outcomes"]
        ),
        "external_normalized_outcomes": dict(
            artifacts["external_normalized_outcomes"]
        ),
        "temporal_site_registry": dict(registries["development"]),
        "external_site_registry": dict(registries["external"]),
        "temporal_full_predictions": dict(artifacts["temporal_predictions"]),
        "external_full_predictions": dict(artifacts["external_predictions"]),
        "availability_registry": dict(artifacts["availability_registry"]),
        "statistics": dict(artifacts["statistics"]),
    }
    if dict(source_bindings) != expected_sources:
        raise ClaimRegistryError(
            "temporal-coverage audit source bindings differ from receipt/authorization"
        )
    try:
        _coverage_replay_api(root)(
            root=root,
            authorization=authorization,
            receipt_artifacts=artifacts,
            expected_audit=audit,
        )
    except Exception as exc:
        raise ClaimRegistryError(
            "receipt-bound temporal-coverage audit failed physical replay"
        ) from exc
    if "temporal_coverage_audit" in statistics:
        raise ClaimRegistryError("statistics creates a temporal-coverage hash cycle")
    rows = audit.get("comparison_sensitivities")
    statistic_rows = statistics.get("tests")
    if not isinstance(rows, list) or not isinstance(statistic_rows, list) or len(rows) != 5:
        raise ClaimRegistryError("temporal-coverage audit lacks the five-test family")
    by_id = {str(row.get("test_id")): row for row in statistic_rows if isinstance(row, Mapping)}
    frozen_order = [
        "equal_12cell",
        "leave_one_year_2021",
        "leave_one_year_2022",
        "leave_one_year_2023",
        "leave_one_season_DJF",
        "leave_one_season_MAM",
        "leave_one_season_JJA",
        "leave_one_season_SON",
    ]
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("test_id")) not in by_id:
            raise ClaimRegistryError("temporal-coverage comparison identity changed")
        formal = by_id[str(row["test_id"])]
        if (
            row.get("formal_statistics_status") != formal.get("status")
            or row.get("formal_median_effect_c") != formal.get("median_effect_c")
            or row.get("n_primary_reportable_stations") != formal.get("n_stations")
            or row.get(
                "prediction_derived_descriptive_effect_does_not_upgrade_inference"
            )
            is not True
        ):
            raise ClaimRegistryError(
                "temporal coverage conflates formal and descriptive effects"
            )
        if formal.get("status") != "ESTIMABLE" and row.get(
            "formal_median_effect_c"
        ) is not None:
            raise ClaimRegistryError("NOT_ESTIMABLE coverage row carries a formal effect")
        candidates = row.get("frozen_sensitivity_candidates")
        worst = row.get("frozen_worst_unfavorable_sensitivity")
        if (
            not isinstance(candidates, list)
            or [value.get("source") for value in candidates if isinstance(value, Mapping)]
            != frozen_order
            or not isinstance(worst, Mapping)
        ):
            raise ClaimRegistryError("temporal coverage omits a frozen sensitivity")
        values = [value.get("descriptive_median_effect_c") for value in candidates]
        if all(value is None for value in values):
            expected_worst = (None, None, None)
        elif any(value is None for value in values):
            raise ClaimRegistryError("temporal coverage partially omits sensitivities")
        else:
            index = max(range(8), key=lambda item: (float(values[item]), -item))
            expected_worst = (index, frozen_order[index], values[index])
        if (
            worst.get("frozen_order_index"),
            worst.get("source"),
            worst.get("descriptive_median_effect_c"),
        ) != expected_worst:
            raise ClaimRegistryError("temporal coverage worst sensitivity changed")


def _number(value: object) -> str:
    if value is None:
        return "NA"
    number = _finite_number(value, label="rendered number")
    return json.dumps(number, allow_nan=False, separators=(",", ":"))


def _result_verdict(
    row: Mapping[str, Any],
    *,
    inference_claim_eligible: bool,
    outcome_qc_claim_eligible: bool,
) -> str:
    if row["status"] != "ESTIMABLE":
        return "NOT_ESTIMABLE"
    if inference_claim_eligible is not True and outcome_qc_claim_eligible is not True:
        return "DESCRIPTIVE_ONLY_BOTH_GATES_FAILED"
    if inference_claim_eligible is not True:
        return "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED"
    if outcome_qc_claim_eligible is not True:
        return "DESCRIPTIVE_ONLY_OUTCOME_QC_GATE_FAILED"
    p_support = row["reject_at_0_05"] is True
    interval_support = row["confidence_bound_supports_margin"] is True
    if p_support != interval_support:
        return "EVIDENCE_CONFLICT_NOT_SUPPORTED"
    if not p_support:
        return "NOT_ESTABLISHED"
    return (
        "SUPERIORITY_SUPPORTED" if float(row["margin_c"]) == 0.0 else ("NONINFERIORITY_SUPPORTED")
    )


def _render_result_text(
    row: Mapping[str, Any],
    templates: Mapping[str, Any],
    *,
    inference_claim_eligible: bool,
    outcome_qc_claim_eligible: bool,
) -> tuple[str, str]:
    verdict = _result_verdict(
        row,
        inference_claim_eligible=inference_claim_eligible,
        outcome_qc_claim_eligible=outcome_qc_claim_eligible,
    )
    template = templates.get(verdict)
    if not isinstance(template, str) or not template:
        raise ClaimRegistryError(f"missing fixed result template: {verdict}")
    if verdict == "NOT_ESTIMABLE":
        values = {
            "test_id": row["test_id"],
            "candidate": row["candidate"],
            "reference": row["reference"],
            "horizon": int(row["horizon"]),
            "margin_c": _number(row["margin_c"]),
            "n_stations": int(row["n_stations"]),
            "n_clusters": int(row["n_clusters"]),
        }
    else:
        values = {
            "test_id": row["test_id"],
            "candidate": row["candidate"],
            "reference": row["reference"],
            "horizon": int(row["horizon"]),
            "margin_c": _number(row["margin_c"]),
            "effect_c": _number(row["median_effect_c"]),
            "ci_low_c": _number(row["ci_low_c"]),
            "ci_high_c": _number(row["ci_high_c"]),
            "p_raw": _number(row["p_one_sided_raw"]),
            "p_holm": _number(row["p_holm"]),
            "n_stations": int(row["n_stations"]),
            "n_clusters": int(row["n_clusters"]),
        }
    try:
        return template.format_map(values), verdict
    except (KeyError, ValueError) as exc:
        raise ClaimRegistryError(f"result template cannot render: {verdict}") from exc


def _claim_body(text: str, entry: Mapping[str, Any]) -> str:
    return f"{text}\n<!-- ROUTE_A_CLAIM_ENTRY {_canonical_json(entry)} -->"


def _claim_block(claim_id: str, body: str) -> bytes:
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return (
        f"<!-- ROUTE_A_CLAIM {claim_id} sha256={digest} -->\n{body}\n<!-- END ROUTE_A_CLAIM -->"
    ).encode("utf-8")


def _expected_claims(
    *,
    registry: Mapping[str, Any],
    phase: str,
    statistics: Mapping[str, Any] | None,
    inference_claim_eligible: bool = False,
    outcome_qc_claim_eligible: bool = False,
) -> Mapping[str, Mapping[str, Any]]:
    templates = registry["claim_templates"]
    expected: dict[str, Mapping[str, Any]] = {}
    for constraint in registry["permanent_constraints"]:
        spec = constraint["claim"]
        template = templates.get(spec["template_id"])
        if not isinstance(template, str) or not template:
            raise ClaimRegistryError("fixed limitation template is absent")
        entry = dict(spec)
        body = _claim_body(template, entry)
        expected[str(spec["claim_id"])] = {
            "spec": spec,
            "body": body,
            "block": _claim_block(str(spec["claim_id"]), body),
        }
    if phase == POST_PHASE:
        if statistics is None:
            raise ClaimRegistryError("POST claim rendering lacks verified statistics")
        by_id = {str(row["test_id"]): row for row in statistics["tests"]}
        for spec in registry["result_claim_specs"]:
            test_id = str(spec["evidence"]["test_id"])
            text, verdict = _render_result_text(
                by_id[test_id],
                templates,
                inference_claim_eligible=inference_claim_eligible,
                outcome_qc_claim_eligible=outcome_qc_claim_eligible,
            )
            entry = dict(spec)
            entry["polarity"] = verdict
            entry["template_id"] = verdict
            body = _claim_body(text, entry)
            expected[str(spec["claim_id"])] = {
                "spec": spec,
                "body": body,
                "block": _claim_block(str(spec["claim_id"]), body),
            }
    return expected


def render_result_claim_blocks(*, root: Path, registry_path: Path) -> Mapping[str, bytes]:
    """Return, but never write, deterministic POST blocks grouped by target."""
    root = root.resolve()
    registry = _load_registry(registry_path.resolve())
    if registry.get("format") != V2_FORMAT:
        raise ClaimRegistryError("result block rendering requires claim-ledger v2")
    protocol = _validate_protocol_binding(root, registry)
    phase = resolve_phase(root=root, registry=registry)
    if phase["phase"] != POST_PHASE:
        raise ClaimRegistryError("result blocks exist only after a verified opening")
    statistics, _ = _load_verified_statistics(
        root=root, registry=registry, phase=phase, protocol=protocol
    )
    authorization = phase.get("authorization")
    gate = authorization.get("inference_gate") if isinstance(authorization, Mapping) else None
    if not isinstance(gate, Mapping) or not isinstance(gate.get("claim_eligible"), bool):
        raise ClaimRegistryError("verified Route-A authorization lacks a boolean inference gate")
    outcome_qc_claim_eligible = _outcome_qc_claim_eligibility(
        root=root,
        phase=phase,
        statistics=statistics,
        protocol=protocol,
    )
    _temporal_coverage_claim_evidence(
        root=root, phase=phase, statistics=statistics
    )
    expected = _expected_claims(
        registry=registry,
        phase=POST_PHASE,
        statistics=statistics,
        inference_claim_eligible=bool(gate["claim_eligible"]),
        outcome_qc_claim_eligible=outcome_qc_claim_eligible,
    )
    return _group_result_claim_blocks(registry=registry, expected=expected)


def _group_result_claim_blocks(
    *, registry: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, bytes]:
    grouped: dict[str, list[bytes]] = defaultdict(list)
    result_ids = list(registry["required_postopen_coverage"]["claim_ids"])
    if len(result_ids) != len(set(result_ids)):
        raise ClaimRegistryError("result block coverage changed during rendering")
    for claim_id in result_ids:
        claim = expected.get(str(claim_id))
        if not isinstance(claim, Mapping):
            raise ClaimRegistryError("result block is absent from deterministic rendering")
        spec = claim["spec"]
        for target in spec["render_targets"]:
            grouped[str(target)].append(claim["block"])
    return {target: b"\n\n".join(blocks) for target, blocks in sorted(grouped.items())}


def _postopen_result_suffix(*, registry: Mapping[str, Any], rendered_blocks: bytes) -> bytes:
    transform = registry["postopen_document_transform"]
    # _load_registry validates the exact transform contract before this helper runs.
    return (
        b"\n\n"
        + str(transform["heading"]).encode("utf-8")
        + b"\n\n"
        + str(transform["preamble"]).encode("utf-8")
        + b"\n\n"
        + rendered_blocks
        + b"\n"
    )


def render_postopen_document_bytes(*, root: Path, registry_path: Path) -> Mapping[str, bytes]:
    """Return complete deterministic POST documents without writing them.

    Each returned document is the exact SHA-frozen PRE byte string plus the sole
    generated result suffix.  The function refuses to build on a hand-edited or
    otherwise non-baseline document.
    """
    root = root.resolve()
    registry = _load_registry(registry_path.resolve())
    rendered = render_result_claim_blocks(root=root, registry_path=registry_path)
    baseline = registry["preopen_document_sha256"]
    documents: dict[str, bytes] = {}
    for target, blocks in rendered.items():
        path = _inside(root, target, require_file=True)
        prefix = path.read_bytes()
        if hashlib.sha256(prefix).hexdigest() != baseline[target]:
            raise ClaimRegistryError(
                f"cannot render POST results over non-baseline document: {target}"
            )
        documents[target] = prefix + _postopen_result_suffix(
            registry=registry, rendered_blocks=blocks
        )
    return documents


def _safe_document_publication_path(root: Path, relative: str) -> Path:
    """Resolve a document target while rejecting every symlink component."""
    lexical = Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts:
        raise ClaimRegistryError(
            f"post-opening document target is not a safe relative path: {relative}"
        )
    raw = root / lexical
    current = root
    for part in Path(relative).parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise ClaimRegistryError(
                f"post-opening document target is absent: {relative}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise ClaimRegistryError(
                f"post-opening document target uses a symlink: {relative}"
            )
    resolved = raw.resolve()
    if resolved != root and root not in resolved.parents:
        raise ClaimRegistryError(
            f"post-opening document target escapes repository: {relative}"
        )
    if not resolved.is_file():
        raise ClaimRegistryError(
            f"post-opening document target is not a file: {relative}"
        )
    return resolved


@contextmanager
def _document_publication_lock(registry_path: Path):
    """Serialize cooperating publishers without creating mutable lock evidence."""
    try:
        stream = registry_path.open("rb")
    except OSError as exc:
        raise ClaimRegistryError("cannot open the claim-publication lock") from exc
    with stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise ClaimRegistryError("cannot acquire the claim-publication lock") from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _atomic_replace_document(path: Path, payload: bytes) -> None:
    """Durably replace one verified document from a same-directory temp file."""
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.route-a-post-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_postopen_documents(
    *, root: Path, registry_path: Path
) -> Mapping[str, str]:
    """Publish deterministic receipt-derived result suffixes and verify closure.

    Every target must be either the exact frozen PRE document or the exact
    already-published POST document.  This makes the operation idempotent and
    lets a later invocation finish safely if a process stopped between multiple
    document replacements.  No arbitrary manuscript edit is accepted.
    """
    root = root.resolve()
    registry_path = registry_path.resolve()
    with _document_publication_lock(registry_path):
        registry = _load_registry(registry_path)
        if registry.get("format") != V2_FORMAT:
            raise ClaimRegistryError("POST publication requires claim-ledger v2")
        rendered = render_result_claim_blocks(root=root, registry_path=registry_path)
        if not rendered:
            raise ClaimRegistryError("POST publication has no declared result target")

        baseline = registry["preopen_document_sha256"]
        expected_payloads: dict[str, bytes] = {}
        for relative in registry["required_documents"]:
            path = _safe_document_publication_path(root, str(relative))
            current = path.read_bytes()
            blocks = rendered.get(str(relative))
            if blocks is None:
                if hashlib.sha256(current).hexdigest() != baseline[relative]:
                    raise ClaimRegistryError(
                        "non-result document differs from its frozen PRE bytes: "
                        f"{relative}"
                    )
                continue
            suffix = _postopen_result_suffix(
                registry=registry, rendered_blocks=blocks
            )
            if hashlib.sha256(current).hexdigest() == baseline[relative]:
                expected_payloads[str(relative)] = current + suffix
                continue
            if current.endswith(suffix):
                prefix = current[: -len(suffix)]
                if hashlib.sha256(prefix).hexdigest() == baseline[relative]:
                    expected_payloads[str(relative)] = current
                    continue
            raise ClaimRegistryError(
                f"result document is neither exact PRE nor deterministic POST: {relative}"
            )

        if set(expected_payloads) != set(rendered):
            raise ClaimRegistryError(
                "rendered POST targets differ from the required-document registry"
            )
        for relative, expected in expected_payloads.items():
            path = _safe_document_publication_path(root, relative)
            current = path.read_bytes()
            if current != expected:
                if hashlib.sha256(current).hexdigest() != baseline[relative]:
                    raise ClaimRegistryError(
                        "result document changed after publication planning: "
                        f"{relative}"
                    )
                _atomic_replace_document(path, expected)
            if path.read_bytes() != expected:
                raise ClaimRegistryError(
                    f"post-opening document replacement did not persist: {relative}"
                )

        violations = validate_claims(
            root=root, registry_path=registry_path, require_complete=True
        )
        if violations:
            raise ClaimRegistryError(
                "published POST documents failed claim validation: "
                + "; ".join(violations)
            )
        return {
            relative: hashlib.sha256(payload).hexdigest()
            for relative, payload in sorted(expected_payloads.items())
        }


def _document_transform_violations(
    *,
    root: Path,
    registry: Mapping[str, Any],
    phase: str,
    rendered_result_blocks: Mapping[str, bytes],
) -> list[str]:
    violations: list[str] = []
    baseline = registry["preopen_document_sha256"]
    for relative in registry["required_documents"]:
        path = _inside(root, relative, require_file=True)
        payload = path.read_bytes()
        expected_prefix_sha = str(baseline[relative])
        blocks = rendered_result_blocks.get(relative)
        if phase == PRE_PHASE or blocks is None:
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != expected_prefix_sha:
                state = "PRE baseline" if phase == PRE_PHASE else "unchanged POST"
                violations.append(
                    f"DOCUMENT_INTEGRITY: {relative} differs from its {state} SHA-256"
                )
            continue
        suffix = _postopen_result_suffix(registry=registry, rendered_blocks=blocks)
        if not payload.endswith(suffix):
            violations.append(
                f"DOCUMENT_INTEGRITY: {relative} lacks the exact generated POST suffix"
            )
            continue
        prefix = payload[: -len(suffix)]
        if hashlib.sha256(prefix).hexdigest() != expected_prefix_sha:
            violations.append(
                f"DOCUMENT_INTEGRITY: {relative} POST prefix differs from the frozen PRE bytes"
            )
    return violations


def _parse_blocks(text: str) -> tuple[list[Mapping[str, Any]], str, list[str]]:
    blocks: list[Mapping[str, Any]] = []
    spans: list[tuple[int, int]] = []
    malformed: list[str] = []
    position = 0
    while True:
        start = BLOCK_START.search(text, position)
        if start is None:
            break
        end_index = text.find(BLOCK_END, start.end())
        if end_index < 0:
            malformed.append(f"unterminated block {start.group('claim_id')}")
            position = start.end()
            continue
        body = text[start.end() : end_index]
        stop = end_index + len(BLOCK_END)
        blocks.append(
            {
                "claim_id": start.group("claim_id"),
                "sha256": start.group("sha256"),
                "body": body,
                "start": start.start(),
                "stop": stop,
            }
        )
        spans.append((start.start(), stop))
        position = stop
    outside_parts = []
    cursor = 0
    for span_start, span_stop in spans:
        outside_parts.append(text[cursor:span_start])
        outside_parts.append("\n")
        cursor = span_stop
    outside_parts.append(text[cursor:])
    outside = "".join(outside_parts)
    if BLOCK_TOKEN in outside:
        malformed.append("malformed ROUTE_A_CLAIM marker")
    return blocks, outside, malformed


def _compile_lint(pattern: str, *, lint_id: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    except re.error as exc:
        raise ClaimRegistryError(f"{lint_id} has invalid lint regex: {pattern}") from exc


def _validate_v1(
    *, root: Path, registry: Mapping[str, Any], documents: list[Path], require_complete: bool
) -> list[str]:
    """Compatibility reader; v1 can lint but can never prove a pending opening."""
    violations: list[str] = []
    for claim in registry["claims"]:
        claim_id = str(claim["claim_id"])
        regexes = claim.get("forbidden_regex", [])
        if not isinstance(regexes, list) or not all(isinstance(value, str) for value in regexes):
            raise ClaimRegistryError(f"{claim_id} forbidden_regex is malformed")
        compiled = [_compile_lint(pattern, lint_id=claim_id) for pattern in regexes]
        for path in documents:
            text = path.read_text(encoding="utf-8", errors="strict")
            for pattern, regex in zip(regexes, compiled):
                match = regex.search(text)
                if match is not None:
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(f"{claim_id}: {path.relative_to(root)}:{line}: {pattern}")
        if require_complete and claim.get("status") == "PENDING_SINGLE_OPENING":
            violations.append(
                f"{claim_id}: legacy v1 artifact existence cannot establish a "
                "verified completed opening; upgrade to claim-ledger v2"
            )
    return violations


def validate_claims(
    *, root: Path, registry_path: Path, require_complete: bool = False
) -> list[str]:
    """Validate claims without accepting a phase, receipt or statistics argument."""
    root = root.resolve()
    registry = _load_registry(registry_path.resolve())
    patterns = registry.get("documents")
    if not isinstance(patterns, list) or not all(isinstance(value, str) for value in patterns):
        raise ClaimRegistryError("document registry is malformed")
    documents = _documents(root, patterns)
    if not documents:
        raise ClaimRegistryError("claim scan selected no documents")
    if registry.get("format") == V1_FORMAT:
        return _validate_v1(
            root=root,
            registry=registry,
            documents=documents,
            require_complete=require_complete,
        )

    protocol = _validate_protocol_binding(root, registry)
    phase_evidence = resolve_phase(root=root, registry=registry)
    phase = str(phase_evidence["phase"])
    if phase not in {PRE_PHASE, POST_PHASE}:
        raise ClaimRegistryError(f"Route-A phase is {INDETERMINATE_PHASE}")
    if require_complete and phase != POST_PHASE:
        return ["PHASE: --require-complete requires a verified completed receipt"]
    expected_document_relatives = set(registry["required_documents"])
    if phase == POST_PHASE:
        state = phase_evidence.get("state")
        if not isinstance(state, Mapping) or not isinstance(state.get("report"), str):
            raise ClaimRegistryError("verified POST lacks its canonical trusted report")
        expected_document_relatives.add(str(state["report"]))
    actual_document_relatives = {path.relative_to(root).as_posix() for path in documents}
    if actual_document_relatives != expected_document_relatives:
        missing = sorted(expected_document_relatives - actual_document_relatives)
        extra = sorted(actual_document_relatives - expected_document_relatives)
        raise ClaimRegistryError(
            f"canonical claim document closure changed (missing={missing}, extra={extra})"
        )
    statistics: Mapping[str, Any] | None = None
    inference_claim_eligible = False
    outcome_qc_claim_eligible = False
    if phase == POST_PHASE:
        statistics, _ = _load_verified_statistics(
            root=root,
            registry=registry,
            phase=phase_evidence,
            protocol=protocol,
        )
        authorization = phase_evidence.get("authorization")
        gate = authorization.get("inference_gate") if isinstance(authorization, Mapping) else None
        if not isinstance(gate, Mapping) or not isinstance(gate.get("claim_eligible"), bool):
            raise ClaimRegistryError(
                "verified Route-A authorization lacks a boolean inference gate"
            )
        inference_claim_eligible = bool(gate["claim_eligible"])
        outcome_qc_claim_eligible = _outcome_qc_claim_eligibility(
            root=root,
            phase=phase_evidence,
            statistics=statistics,
            protocol=protocol,
        )
        _temporal_coverage_claim_evidence(
            root=root,
            phase=phase_evidence,
            statistics=statistics,
        )
    expected = _expected_claims(
        registry=registry,
        phase=phase,
        statistics=statistics,
        inference_claim_eligible=inference_claim_eligible,
        outcome_qc_claim_eligible=outcome_qc_claim_eligible,
    )
    rendered_result_blocks = (
        _group_result_claim_blocks(registry=registry, expected=expected)
        if phase == POST_PHASE
        else {}
    )
    expected_entries = {
        str(item["claim"]["claim_id"]): item["claim"] for item in registry["permanent_constraints"]
    }
    expected_entries.update(
        {str(item["claim_id"]): item for item in registry["result_claim_specs"]}
    )
    occurrences: dict[str, list[str]] = defaultdict(list)
    outside_by_path: dict[Path, str] = {}
    violations = _document_transform_violations(
        root=root,
        registry=registry,
        phase=phase,
        rendered_result_blocks=rendered_result_blocks,
    )
    for path in documents:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="strict")
        blocks, outside, malformed = _parse_blocks(text)
        outside_by_path[path] = outside
        violations.extend(f"BLOCK: {relative}: {message}" for message in malformed)
        for block in blocks:
            claim_id = str(block["claim_id"])
            occurrence = f"{relative}:{text.count(chr(10), 0, int(block['start'])) + 1}"
            occurrences[claim_id].append(relative)
            if claim_id not in expected_entries:
                violations.append(f"BLOCK: {occurrence}: unknown claim ID {claim_id}")
                continue
            spec = expected_entries[claim_id]
            if phase not in spec["phase_allowed"]:
                violations.append(
                    f"BLOCK: {occurrence}: {claim_id} is not allowed in phase {phase}"
                )
                continue
            if relative not in spec["render_targets"]:
                violations.append(
                    f"BLOCK: {occurrence}: {claim_id} has an undeclared render target"
                )
            actual_digest = hashlib.sha256(str(block["body"]).encode("utf-8")).hexdigest()
            if block["sha256"] != actual_digest:
                violations.append(f"BLOCK: {occurrence}: {claim_id} body hash is invalid")
            expected_claim = expected.get(claim_id)
            if expected_claim is None or block["body"] != expected_claim["body"]:
                violations.append(
                    f"BLOCK: {occurrence}: {claim_id} differs from deterministic evidence rendering"
                )
    for claim_id, targets in occurrences.items():
        counts = Counter(targets)
        for target, count in counts.items():
            if count > 1:
                violations.append(
                    f"COVERAGE: {claim_id} appears {count} times in {target}; expected at most once"
                )
    if phase == POST_PHASE:
        for claim_id in registry["required_postopen_coverage"]["claim_ids"]:
            count = len(occurrences.get(str(claim_id), []))
            if count != 1:
                violations.append(
                    f"COVERAGE: POST claim {claim_id} appears {count} times; expected exactly once"
                )
    else:
        result_ids = set(registry["required_postopen_coverage"]["claim_ids"])
        for claim_id in result_ids:
            if occurrences.get(str(claim_id)):
                violations.append(f"COVERAGE: PRE phase contains result claim {claim_id}")
    for claim_id in registry["required_permanent_coverage"]["claim_ids"]:
        count = len(occurrences.get(str(claim_id), []))
        if count != 1:
            violations.append(
                f"COVERAGE: permanent limitation {claim_id} appears {count} times; "
                "expected exactly once"
            )

    # Fixed negative sentences are the only authoritative free-text exceptions.
    limitation_templates = {
        str(registry["claim_templates"][constraint["claim"]["template_id"]])
        for constraint in registry["permanent_constraints"]
    }
    lint_groups: list[tuple[str, str]] = []
    for constraint in registry["permanent_constraints"]:
        lint_groups.extend(
            (str(constraint["constraint_id"]), str(pattern)) for pattern in constraint["lint_regex"]
        )
    for lint in registry["free_text_lints"]:
        if not isinstance(lint, Mapping) or set(lint) != {"lint_id", "regex"}:
            raise ClaimRegistryError("free-text lint schema is malformed")
        lint_groups.append((str(lint["lint_id"]), str(lint["regex"])))
    compiled = [
        (lint_id, pattern, _compile_lint(pattern, lint_id=lint_id))
        for lint_id, pattern in lint_groups
    ]
    for path, outside in outside_by_path.items():
        lint_text = outside
        for template in limitation_templates:
            lint_text = lint_text.replace(template, " ")
        for lint_id, pattern, regex in compiled:
            match = regex.search(lint_text)
            if match is not None:
                line = lint_text.count("\n", 0, match.start()) + 1
                violations.append(f"LINT {lint_id}: {path.relative_to(root)}:{line}: {pattern}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--write-generated-results",
        action="store_true",
        help=(
            "after a verified opening, atomically append only the deterministic "
            "receipt-derived result suffix"
        ),
    )
    args = parser.parse_args()
    try:
        published: Mapping[str, str] = {}
        if args.write_generated_results:
            published = publish_postopen_documents(
                root=args.root,
                registry_path=args.registry,
            )
        violations = validate_claims(
            root=args.root,
            registry_path=args.registry,
            require_complete=(args.require_complete or args.write_generated_results),
        )
    except ClaimRegistryError as exc:
        print(f"claim-registry validation failed: {exc}", file=sys.stderr)
        return 2
    if violations:
        print("Route-A claim violations:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    for relative, digest in published.items():
        print(f"published {relative} sha256={digest}")
    print("Route-A claims OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
