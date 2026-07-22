"""Outcome-free inference-eligibility gate for Route A.

The Route-A point estimand is a descriptive statistic of a fixed,
availability-enriched station cohort.  HUC2 resampling and sign flipping add a
separate, assumption-conditional superpopulation interpretation.  This module
keeps those scopes distinct and makes the latter fail closed unless every
predeclared assumption, cluster-design, and null-calibration gate passes.

Only three live evidence classes are read: the sealed v1 protocol, the frozen
development station registry, and the source inventory.  Confirmation outcomes,
predictions, trained-model outputs, network resources, and caller-supplied
effect vectors are not accepted by this API.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from .repro import canonical_json, sha256_json, source_inventory


GATE_FORMAT = "thermoroute.route-a-inference-gate.v1"
AMENDMENT_FORMAT = "thermoroute.route-a-inference-amendment.v1"
AMENDMENT_SEAL_FORMAT = "thermoroute.route-a-inference-amendment-seal.v1"

BASE_PROTOCOL_RELATIVE = "protocols/route_a_confirmatory_v1.json"
BASE_PROTOCOL_SEAL_RELATIVE = "protocols/route_a_protocol_seal_v1.json"
STATION_REGISTRY_RELATIVE = "data_usgs/station_registry_v1.csv"
AMENDMENT_RELATIVE = "protocols/route_a_inference_amendment_v1.json"
AMENDMENT_SEAL_RELATIVE = "protocols/route_a_inference_amendment_seal_v1.json"
DEFAULT_GATE_RELATIVE = "outputs/prelabel/route_a_inference_gate_v1.json"

MIN_CLUSTERS = 30
MIN_EFFECTIVE_CLUSTER_FRACTION = 0.75
MAX_LARGEST_CLUSTER_SHARE = 0.25

STRUCTURAL_ASSUMPTIONS: tuple[dict[str, str], ...] = (
    {
        "assumption_id": "INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING",
        "status": "NOT_ESTABLISHED",
        "basis": (
            "The fixed HUC2 groups were not probability sampled and HUC2 is a "
            "coarse region rather than an independently sampled river-network unit."
        ),
    },
    {
        "assumption_id": "JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY",
        "status": "NOT_ESTABLISHED",
        "basis": (
            "There is no randomized sign assignment or outcome-free structural "
            "argument establishing joint sign symmetry of each complete HUC2 "
            "effect vector around the tested margin."
        ),
    },
)

NULL_SIMULATION_SCENARIOS: tuple[str, ...] = (
    "independent_gaussian_intracluster_correlation_grid",
    "independent_student_t3_heavy_tail",
    "known_cluster_size_heteroskedasticity",
    "median_zero_skewed_lognormal_gaussian_copula",
    "cluster_level_skewed_shock",
    "cross_huc_shared_factor_dependence",
)


class InferenceGateError(RuntimeError):
    """The pre-outcome inference contract is absent, stale, or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str, *, require_file: bool = True) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise InferenceGateError("inference-gate path must be a relative allowlisted path")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise InferenceGateError("inference-gate path escapes repository root")
    if require_file and not path.is_file():
        raise InferenceGateError(f"required inference-gate input is absent: {relative}")
    return path


def _require_allowlisted(path: str | Path, *, root: Path, expected: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    allowed = (root / expected).resolve()
    if resolved != allowed:
        raise InferenceGateError(
            f"inference-gate input is not allowlisted: expected {expected}"
        )
    if not resolved.is_file():
        raise InferenceGateError(f"required inference-gate input is absent: {expected}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceGateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise InferenceGateError(f"{label} is not a JSON object")
    return value


def _binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
    }


def _validate_binding(root: Path, binding: object, *, label: str) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise InferenceGateError(f"{label} binding is malformed")
    path = _inside(root, str(binding.get("path", "")))
    digest = binding.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise InferenceGateError(f"{label} binding lacks a SHA-256")
    if _sha256_file(path) != digest:
        raise InferenceGateError(f"{label} checksum changed")
    return path


def _canonical_family(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    inference = protocol.get("primary_inference_contract")
    family = inference.get("confirmatory_family") if isinstance(inference, Mapping) else None
    if not isinstance(family, list) or len(family) != 5:
        raise InferenceGateError("base protocol does not contain exactly five tests")
    output: list[dict[str, Any]] = []
    required = {
        "test_id", "candidate", "reference", "horizon", "margin_c",
        "alternative", "bootstrap_seed", "sign_flip_seed", "description",
    }
    for item in family:
        if not isinstance(item, Mapping) or set(item) != required:
            raise InferenceGateError("base confirmatory-family schema changed")
        output.append(dict(item))
    return output


def cluster_geometry(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Compute row-order-invariant HUC2 concentration diagnostics."""
    sites: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        site = str(row.get("site_no", "")).strip()
        huc = str(row.get("huc2", "")).strip()
        if not site or not huc:
            raise InferenceGateError("station registry has an empty site_no or huc2")
        if site in sites:
            raise InferenceGateError("station registry has duplicate site_no values")
        sites.add(site)
        counts[huc] += 1
    if not sites or not counts:
        raise InferenceGateError("station registry is empty")
    station_count = len(sites)
    cluster_sizes = sorted(counts.values())
    shares = [count / station_count for count in cluster_sizes]
    effective = 1.0 / sum(share * share for share in shares)
    n_clusters = len(cluster_sizes)
    mean_size = station_count / n_clusters
    size_cv = math.sqrt(
        sum((count - mean_size) ** 2 for count in cluster_sizes) / n_clusters
    ) / mean_size
    return {
        "n_stations": station_count,
        "n_clusters": n_clusters,
        "cluster_sizes_sorted": cluster_sizes,
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "cluster_size_cv": size_cv,
        "largest_cluster_share": max(shares),
        "effective_cluster_count_inverse_herfindahl": effective,
        "effective_cluster_fraction": effective / n_clusters,
    }


def load_station_geometry(registry_path: str | Path, *, root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    path = _require_allowlisted(
        registry_path, root=root_path, expected=STATION_REGISTRY_RELATIVE
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or {"site_no", "huc2"} - set(reader.fieldnames):
                raise InferenceGateError("station registry lacks site_no/huc2")
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise InferenceGateError("cannot parse the frozen station registry") from exc
    return cluster_geometry(rows)


def _validate_base_seal(
    protocol: Mapping[str, Any],
    *,
    protocol_path: Path,
    seal_path: Path,
) -> Mapping[str, Any]:
    seal = _load_json(seal_path, label="base protocol seal")
    if (
        seal.get("format") != "thermoroute.route-a-protocol-seal.v1"
        or seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
        or seal.get("protocol_id") != "route-a-confirmatory-v1"
    ):
        raise InferenceGateError("base protocol seal identity changed")
    final = seal.get("final_prelabel_protocol")
    binding = final.get("json") if isinstance(final, Mapping) else None
    if not isinstance(binding, Mapping):
        raise InferenceGateError("base protocol seal lacks its JSON binding")
    if (
        binding.get("path") != BASE_PROTOCOL_RELATIVE
        or binding.get("sha256") != _sha256_file(protocol_path)
        or protocol.get("protocol_id") != seal.get("protocol_id")
    ):
        raise InferenceGateError("base protocol bytes differ from the v1 seal")
    return seal


def inference_gate_policy() -> dict[str, Any]:
    """Return the frozen, caller-independent decision policy."""
    return {
        "estimand_scope": {
            "fixed_cohort": (
                "median paired station-RMSE difference in the fixed, "
                "availability-enriched cohort; descriptive for its reportable keys"
            ),
            "superpopulation": (
                "allowed only conditionally on independently exchangeable HUC2 "
                "sampling and joint complete-cluster-vector sign symmetry"
            ),
        },
        "cluster_thresholds": {
            "minimum_clusters": MIN_CLUSTERS,
            "minimum_effective_cluster_fraction": MIN_EFFECTIVE_CLUSTER_FRACTION,
            "maximum_largest_cluster_share_exclusive": MAX_LARGEST_CLUSTER_SHARE,
        },
        "structural_assumptions": [dict(item) for item in STRUCTURAL_ASSUMPTIONS],
        "null_simulation": {
            "role": "FALSIFICATION_ONLY_NEVER_ESTABLISHES_STRUCTURAL_ASSUMPTIONS",
            "required_before_inferential_claims": True,
            "synthetic_boundary_null_only": True,
            "post_2020_outcomes_allowed": False,
            "caller_supplied_effects_allowed": False,
            "network_allowed": False,
            "scenarios": list(NULL_SIMULATION_SCENARIOS),
        },
        "decision": {
            "all_components_must_pass": True,
            "missing_unknown_or_not_run_is_failure": True,
            "failed_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
            "failed_verdict": "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
            "p_ci_sign_flip_role_when_failed": (
                "ASSUMPTION_CONDITIONAL_SENSITIVITY_NOT_CLAIM_SUPPORT"
            ),
        },
    }


def build_inference_gate_document(
    *,
    root: str | Path,
    protocol_path: str | Path = BASE_PROTOCOL_RELATIVE,
    protocol_seal_path: str | Path = BASE_PROTOCOL_SEAL_RELATIVE,
    station_registry_path: str | Path = STATION_REGISTRY_RELATIVE,
) -> dict[str, Any]:
    """Build the deterministic gate without accepting outcomes or effect vectors."""
    root_path = Path(root).resolve()
    protocol_file = _require_allowlisted(
        protocol_path, root=root_path, expected=BASE_PROTOCOL_RELATIVE
    )
    seal_file = _require_allowlisted(
        protocol_seal_path, root=root_path, expected=BASE_PROTOCOL_SEAL_RELATIVE
    )
    registry_file = _require_allowlisted(
        station_registry_path, root=root_path, expected=STATION_REGISTRY_RELATIVE
    )
    protocol = _load_json(protocol_file, label="base protocol")
    _validate_base_seal(
        protocol, protocol_path=protocol_file, seal_path=seal_file
    )
    family = _canonical_family(protocol)
    geometry = load_station_geometry(registry_file, root=root_path)
    policy = inference_gate_policy()

    threshold_failures: list[str] = []
    if geometry["n_clusters"] < MIN_CLUSTERS:
        threshold_failures.append("SMALL_CLUSTER_COUNT_LT_30")
    if geometry["effective_cluster_fraction"] < MIN_EFFECTIVE_CLUSTER_FRACTION:
        threshold_failures.append("EFFECTIVE_CLUSTER_FRACTION_LT_0_75")
    if geometry["largest_cluster_share"] >= MAX_LARGEST_CLUSTER_SHARE:
        threshold_failures.append("DOMINANT_CLUSTER_SHARE_GE_0_25")
    structural_failures = [
        item["assumption_id"]
        for item in STRUCTURAL_ASSUMPTIONS
        if item["status"] != "ESTABLISHED"
    ]
    cluster_pass = not threshold_failures
    structural_pass = not structural_failures
    # Simulation is intentionally not run after an earlier required gate fails.
    # Missing/not-run is itself false and can never rescue the claim decision.
    null_simulation = {
        **policy["null_simulation"],
        "status": (
            "NOT_RUN_BLOCKED_BY_STRUCTURAL_OR_CLUSTER_GATE"
            if not (cluster_pass and structural_pass)
            else "NOT_IMPLEMENTED_FAIL_CLOSED"
        ),
        "pass": False,
        "outcomes_read": False,
        "network_used": False,
        "files_read_allowlist": [
            BASE_PROTOCOL_RELATIVE,
            BASE_PROTOCOL_SEAL_RELATIVE,
            STATION_REGISTRY_RELATIVE,
            "source_inventory_patterns",
        ],
    }
    blocking = [
        *[f"STRUCTURAL_ASSUMPTION_NOT_ESTABLISHED:{value}" for value in structural_failures],
        *threshold_failures,
        "NULL_SIMULATION_NOT_PASSING",
    ]
    inventory = source_inventory(root_path)
    source_sha256 = sha256_json(inventory)
    stable: dict[str, Any] = {
        "format": GATE_FORMAT,
        "status": "FAIL_CLOSED_DESCRIPTIVE_ONLY",
        "contains_confirmation_outcomes": False,
        "post_2020_outcomes_requested_or_inspected": False,
        "network_used": False,
        "inputs": {
            "base_protocol": _binding(root_path, protocol_file),
            "base_protocol_seal": _binding(root_path, seal_file),
            "station_registry": _binding(root_path, registry_file),
            "source": {
                "source_tree_sha256": source_sha256,
                "source_inventory": inventory,
            },
        },
        "confirmatory_family": {
            "count": 5,
            "sha256": sha256_json(family),
            "objects": family,
            "candidate_reference_horizon_margin_unchanged": True,
        },
        "policy": policy,
        "policy_sha256": sha256_json(policy),
        "cluster_geometry": geometry,
        "cluster_gate": {
            "pass": cluster_pass,
            "failure_codes": threshold_failures,
        },
        "structural_assumption_gate": {
            "pass": structural_pass,
            "failure_codes": structural_failures,
        },
        "null_simulation_gate": null_simulation,
        "claim_eligible": False,
        "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
        "blocking_reasons": blocking,
    }
    stable["gate_self_sha256"] = sha256_json(stable)
    return stable


def validate_inference_gate_document(
    gate_path: str | Path,
    *,
    root: str | Path,
    protocol_path: str | Path = BASE_PROTOCOL_RELATIVE,
    protocol_seal_path: str | Path = BASE_PROTOCOL_SEAL_RELATIVE,
    station_registry_path: str | Path = STATION_REGISTRY_RELATIVE,
) -> dict[str, Any]:
    """Rebuild the outcome-free gate and require byte-semantic equality."""
    root_path = Path(root).resolve()
    gate_file = _require_allowlisted(
        gate_path, root=root_path, expected=DEFAULT_GATE_RELATIVE
    )
    actual = _load_json(gate_file, label="inference-gate artifact")
    expected = build_inference_gate_document(
        root=root_path,
        protocol_path=protocol_path,
        protocol_seal_path=protocol_seal_path,
        station_registry_path=station_registry_path,
    )
    if dict(actual) != expected:
        raise InferenceGateError("inference-gate artifact is stale or tampered")
    if actual.get("claim_eligible") is not False:
        raise InferenceGateError("current Route-A inference gate did not fail closed")
    return dict(actual)


def validate_inference_amendment(
    amendment_path: str | Path,
    *,
    root: str | Path,
    protocol_path: str | Path = BASE_PROTOCOL_RELATIVE,
    protocol_seal_path: str | Path = BASE_PROTOCOL_SEAL_RELATIVE,
) -> dict[str, Any]:
    """Validate the transparent overlay without modifying the sealed v1 protocol."""
    root_path = Path(root).resolve()
    amendment_file = _require_allowlisted(
        amendment_path, root=root_path, expected=AMENDMENT_RELATIVE
    )
    protocol_file = _require_allowlisted(
        protocol_path, root=root_path, expected=BASE_PROTOCOL_RELATIVE
    )
    seal_file = _require_allowlisted(
        protocol_seal_path, root=root_path, expected=BASE_PROTOCOL_SEAL_RELATIVE
    )
    protocol = _load_json(protocol_file, label="base protocol")
    _validate_base_seal(protocol, protocol_path=protocol_file, seal_path=seal_file)
    family = _canonical_family(protocol)
    amendment = _load_json(amendment_file, label="inference amendment")
    required = {
        "format", "status", "amendment_id", "recorded_date",
        "post_2020_wtemp_requested_or_inspected", "outcome_independent",
        "base_protocol", "base_protocol_seal", "scientific_comparisons",
        "estimand_scope", "inference_scope", "decision_overlay", "lineage_contract",
    }
    if set(amendment) != required:
        raise InferenceGateError("inference amendment schema changed")
    if (
        amendment.get("format") != AMENDMENT_FORMAT
        or amendment.get("status") != "FROZEN_PRELABEL_OUTCOME_FREE"
        or amendment.get("amendment_id") != "route-a-prelabel-inference-scope-014"
        or amendment.get("post_2020_wtemp_requested_or_inspected") is not False
        or amendment.get("outcome_independent") is not True
        or amendment.get("base_protocol") != _binding(root_path, protocol_file)
        or amendment.get("base_protocol_seal") != _binding(root_path, seal_file)
    ):
        raise InferenceGateError("inference amendment identity/attestation changed")
    comparisons = amendment.get("scientific_comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != {
        "count", "confirmatory_family_sha256", "objects", "change_allowed"
    }:
        raise InferenceGateError("inference amendment comparison registry is malformed")
    if (
        comparisons.get("count") != 5
        or comparisons.get("objects") != family
        or comparisons.get("confirmatory_family_sha256") != sha256_json(family)
        or comparisons.get("change_allowed") is not False
    ):
        raise InferenceGateError("the five scientific comparisons or margins changed")
    decision = amendment.get("decision_overlay")
    if not isinstance(decision, Mapping) or decision.get(
        "gate_failure_verdict"
    ) != "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED" or decision.get(
        "supported_claim_allowed_when_gate_fails"
    ) is not False:
        raise InferenceGateError("inference amendment is not fail closed")
    lineage = amendment.get("lineage_contract")
    if not isinstance(lineage, Mapping) or lineage != {
        "base_v1_files_remain_immutable": True,
        "separate_amendment_seal_required": True,
        "seal_path": AMENDMENT_SEAL_RELATIVE,
        "amendment_commit_must_precede_seal_commit": True,
    }:
        raise InferenceGateError("inference amendment lineage contract changed")
    return dict(amendment)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments], cwd=root, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )


def build_inference_amendment_seal_document(
    *,
    root: str | Path,
    final_prelabel_commit: str,
    amendment_path: str | Path = AMENDMENT_RELATIVE,
    protocol_seal_path: str | Path = BASE_PROTOCOL_SEAL_RELATIVE,
) -> dict[str, Any]:
    """Build the second-stage lineage seal after the amendment is committed."""
    root_path = Path(root).resolve()
    amendment = validate_inference_amendment(
        amendment_path, root=root_path, protocol_seal_path=protocol_seal_path
    )
    amendment_file = (root_path / AMENDMENT_RELATIVE).resolve()
    base_seal_file = (root_path / BASE_PROTOCOL_SEAL_RELATIVE).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", final_prelabel_commit):
        raise InferenceGateError("amendment seal requires a full Git commit")
    if not (root_path / ".git").exists():
        raise InferenceGateError("amendment seal creation requires live Git history")
    exists = _git(root_path, "cat-file", "-e", f"{final_prelabel_commit}^{{commit}}")
    if exists.returncode:
        raise InferenceGateError("amendment commit is absent")
    blob = _git(root_path, "show", f"{final_prelabel_commit}:{AMENDMENT_RELATIVE}")
    if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != _sha256_file(
        amendment_file
    ):
        raise InferenceGateError("amendment commit does not contain the current bytes")
    base_seal = _load_json(base_seal_file, label="base protocol seal")
    base_commit = str(base_seal.get("final_prelabel_protocol", {}).get("commit", ""))
    ancestor = _git(root_path, "merge-base", "--is-ancestor", base_commit, final_prelabel_commit)
    if ancestor.returncode:
        raise InferenceGateError("base v1 protocol commit is not an amendment ancestor")
    return {
        "format": AMENDMENT_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "amendment_id": amendment["amendment_id"],
        "amendment": _binding(root_path, amendment_file),
        "base_protocol_seal": _binding(root_path, base_seal_file),
        "final_prelabel_commit": final_prelabel_commit,
        "history_contract": {
            "base_protocol_commit_must_be_ancestor": True,
            "amendment_blob_must_match_commit": True,
            "amendment_commit_must_be_ancestor_of_authorization": True,
            "seal_is_created_only_after_amendment_commit": True,
        },
        "prelabel_attestation": {
            "post_2020_wtemp_requested_or_inspected": False,
            "outcome_independent": True,
        },
    }


def validate_inference_amendment_seal(
    seal_path: str | Path,
    *,
    root: str | Path,
    amendment_path: str | Path = AMENDMENT_RELATIVE,
    allow_gitless_archive: bool = False,
) -> dict[str, Any]:
    """Validate separate amendment lineage; never accept a missing commit."""
    root_path = Path(root).resolve()
    seal_file = _require_allowlisted(
        seal_path, root=root_path, expected=AMENDMENT_SEAL_RELATIVE
    )
    amendment = validate_inference_amendment(amendment_path, root=root_path)
    seal = _load_json(seal_file, label="inference amendment seal")
    required = {
        "format", "status", "amendment_id", "amendment", "base_protocol_seal",
        "final_prelabel_commit", "history_contract", "prelabel_attestation",
    }
    if set(seal) != required or (
        seal.get("format") != AMENDMENT_SEAL_FORMAT
        or seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
        or seal.get("amendment_id") != amendment["amendment_id"]
    ):
        raise InferenceGateError("inference amendment seal schema/identity changed")
    amendment_file = _validate_binding(
        root_path, seal.get("amendment"), label="inference amendment"
    )
    base_seal_file = _validate_binding(
        root_path, seal.get("base_protocol_seal"), label="base protocol seal"
    )
    if (
        amendment_file != (root_path / AMENDMENT_RELATIVE).resolve()
        or base_seal_file != (root_path / BASE_PROTOCOL_SEAL_RELATIVE).resolve()
    ):
        raise InferenceGateError("inference amendment seal names noncanonical files")
    commit = str(seal.get("final_prelabel_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise InferenceGateError("inference amendment seal commit is malformed")
    expected_history = {
        "base_protocol_commit_must_be_ancestor": True,
        "amendment_blob_must_match_commit": True,
        "amendment_commit_must_be_ancestor_of_authorization": True,
        "seal_is_created_only_after_amendment_commit": True,
    }
    if seal.get("history_contract") != expected_history or seal.get(
        "prelabel_attestation"
    ) != {
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent": True,
    }:
        raise InferenceGateError("inference amendment seal contract changed")
    if (root_path / ".git").exists():
        if _git(root_path, "cat-file", "-e", f"{commit}^{{commit}}").returncode:
            raise InferenceGateError("inference amendment seal commit is absent")
        blob = _git(root_path, "show", f"{commit}:{AMENDMENT_RELATIVE}")
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != _sha256_file(
            amendment_file
        ):
            raise InferenceGateError("sealed amendment Git blob changed")
        if _git(root_path, "merge-base", "--is-ancestor", commit, "HEAD").returncode:
            raise InferenceGateError("amendment commit is not an authorization ancestor")
        base_seal = _load_json(base_seal_file, label="base protocol seal")
        final_protocol = base_seal.get("final_prelabel_protocol")
        base_commit = str(
            final_protocol.get("commit", "")
            if isinstance(final_protocol, Mapping)
            else ""
        )
        if (
            not re.fullmatch(r"[0-9a-f]{40}", base_commit)
            or _git(
                root_path, "merge-base", "--is-ancestor", base_commit, commit
            ).returncode
        ):
            raise InferenceGateError(
                "base v1 protocol commit is not an amendment ancestor"
            )
    elif not allow_gitless_archive:
        raise InferenceGateError("amendment lineage requires Git outside release replay")
    return dict(seal)


def exclusive_create_json(path: str | Path, document: Mapping[str, Any]) -> None:
    """Create one canonical JSON artifact without replacement semantics."""
    target = Path(os.path.abspath(os.fspath(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(dict(document)) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o444)
    except FileExistsError as exc:
        raise InferenceGateError(f"refusing to replace create-only artifact: {target}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
