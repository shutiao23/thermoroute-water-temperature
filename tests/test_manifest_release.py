from __future__ import annotations

import importlib.util
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPT = ROOT / "scripts" / "14_manifest.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_release.py"
ZIP_SCRIPT = ROOT / "scripts" / "deterministic_zip.py"


def _stage09b_fixture_config(
    expected_bridge: dict[str, str], *, eval_batch_size: int = 2,
) -> dict[str, object]:
    """Build the real formal Stage-09b configuration for release fixtures."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from thermoroute import config as C
        from thermoroute.development_controls import (
            DEVELOPMENT_DISCLOSURE,
            FULL_VARIABLES,
            TRAIN_CONFIG,
            architecture_template,
            assert_parameter_budgets,
            declared_arms,
            expected_member_registry,
        )
    finally:
        sys.path.pop(0)
    arms = declared_arms()
    hash_policy = (
        "canonical-sort-identity-collections-independent-of-hash-secret"
    )
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
            "threads": 1,
            "cublas_workspace_config": ":4096:8",
            "python_hash_policy": hash_policy,
            "torch_deterministic_algorithms": True,
            "tf32": False,
            "float32_matmul_precision": "highest",
        },
        "torch": {
            "num_threads": 1,
            "num_interop_threads": 1,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }
    return {
        "stage": "09b_development_controls",
        "format": "thermoroute.development-controls.v2",
        "execution_role": (
            "prelabel_relative_to_unopened_post_2020_confirmation"
        ),
        "evidence_role": "development_only_exploratory",
        "development_disclosure": DEVELOPMENT_DISCLOSURE,
        "panel_date_range": ["2006-01-01", "2020-12-31"],
        "development_evaluation_interval": list(C.SPLIT.test),
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "training_device": "cpu",
        "variables": list(FULL_VARIABLES),
        "context_length": C.CONTEXT_LENGTH,
        "horizons": list(C.HORIZONS),
        "time_split": C.SPLIT.as_dict(),
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "train_config": asdict(TRAIN_CONFIG),
        "arms": [asdict(arm) for arm in arms],
        "expected_member_registry": [
            list(member) for member in expected_member_registry(arms)
        ],
        "parameter_counts": assert_parameter_budgets(arms, n_stations=120),
        "architecture_templates": {
            arm.arm_id: architecture_template(arm, n_stations=120)
            for arm in arms
        },
        "parameter_match_tolerance_fraction": 0.02,
        "architecture_candidates_per_arm": 1,
        "historical_tuning_budget_equalized": False,
        "development_predictor_bridge": expected_bridge,
        "formal_numerical_policy": formal_policy,
        "eval_batch_size": eval_batch_size,
    }


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path) -> Path:
    files = {
        "src/thermoroute/config.py": (
            "from dataclasses import dataclass\n"
            "HORIZONS = (1, 3, 7)\n"
            "STATIONS = ('s1',)\n"
            "@dataclass(frozen=True)\n"
            "class TrainConfig:\n"
            "    seed: int = 7\n"
            "TRAIN = TrainConfig()\n"
        ),
        "scripts/demo.py": "print('fixture')\n",
        "tests/test_demo.py": "def test_demo(): assert True\n",
        "pyproject.toml": "[project]\nname='fixture'\nversion='0.0.0'\n",
        "requirements.txt": "pandas>=2\n",
        "requirements-lock.txt": "pandas==2.2.2\n",
        "README.md": "# fixture\n",
        "data/input.csv": "x\n1\n",
        "outputs/tables/result.csv": "score\n1.0\n",
    }
    for rel, payload in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    return root / "outputs" / "manifest.json"


def _write_bytes(root: Path, relative: str, payload: bytes = b"fixture\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _commit_git_fixture(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", message,
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _binding(verifier, root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    return {"path": relative, "sha256": verifier.sha256_file(path)}


@pytest.mark.parametrize(
    "alias",
    (
        "./artifacts/value.bin",
        "artifacts//value.bin",
        "artifacts/subdir/../value.bin",
    ),
)
def test_release_binding_reader_rejects_noncanonical_paths(tmp_path, alias):
    verifier = _load_script(VERIFY_SCRIPT, "verify_release_noncanonical_alias")
    artifact = _write_bytes(tmp_path, "artifacts/value.bin", b"bound bytes\n")
    binding = {"path": alias, "sha256": verifier.sha256_file(artifact)}

    with pytest.raises(ValueError, match="not canonical"):
        verifier._add_binding(
            tmp_path, {}, "fixture", binding, label="fixture binding"
        )


def test_release_binding_reader_rejects_symlink_alias(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "verify_release_symlink_alias")
    artifact = _write_bytes(tmp_path, "artifacts/value.bin", b"bound bytes\n")
    alias = tmp_path / "artifacts" / "alias.bin"
    alias.symlink_to(artifact.name)
    binding = {
        "path": "artifacts/alias.bin",
        "sha256": verifier.sha256_file(artifact),
    }

    with pytest.raises(ValueError, match="symlink"):
        verifier._add_binding(
            tmp_path, {}, "fixture", binding, label="fixture binding"
        )


def test_release_binding_reader_rejects_hardlinked_artifact(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "verify_release_hardlink")
    artifact = _write_bytes(tmp_path, "artifacts/value.bin", b"bound bytes\n")
    os.link(artifact, tmp_path / "artifacts" / "duplicate.bin")
    binding = {
        "path": "artifacts/value.bin",
        "sha256": verifier.sha256_file(artifact),
    }

    with pytest.raises(ValueError, match="hard-linked"):
        verifier._add_binding(
            tmp_path, {}, "fixture", binding, label="fixture binding"
        )


def _chronology_binding(verifier, root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        **_binding(verifier, root, relative),
        "byte_count": path.stat().st_size,
        "git_blob_oid": "a" * 40,
    }


def _fixture_confirmatory_family() -> list[dict[str, object]]:
    specifications = (
        ("H1-h1-vs-damped", "DampedPersistence", 1, 0.0, 1001, 5001),
        ("H1-h3-vs-damped", "DampedPersistence", 3, 0.0, 1003, 5003),
        ("H1-h7-vs-damped", "DampedPersistence", 7, 0.0, 1007, 5007),
        ("H2-h3-vs-lightgbm", "LightGBM", 3, 0.05, 1103, 5103),
        ("H2-h7-vs-lightgbm", "LightGBM", 7, 0.05, 1107, 5107),
    )
    return [
        {
            "test_id": test_id,
            "candidate": "ThermoRoute",
            "reference": reference,
            "horizon": horizon,
            "margin_c": margin,
            "alternative": "candidate_minus_reference_below_margin",
            "bootstrap_seed": bootstrap_seed,
            "sign_flip_seed": sign_flip_seed,
            "description": f"fixture comparison {test_id}",
        }
        for test_id, reference, horizon, margin, bootstrap_seed, sign_flip_seed
        in specifications
    ]


def _minimal_canonical_release(verifier, root: Path) -> None:
    _write_bytes(
        root,
        verifier.REPRODUCIBILITY_LOCK,
        ("fixture==1 \\\n    --hash=sha256:" + "0" * 64 + "\n").encode(),
    )
    for relative in verifier.CANONICAL_DEVELOPMENT_PATHS:
        _write_bytes(root, relative, b"{}\n" if relative.endswith(".json") else b"fixture\n")
    _write_bytes(root, "data_usgs/raw_snapshots/huc-v1/snapshot_index.json", b"{}\n")
    _write_bytes(root, "data_usgs/raw_snapshots/huc-v1/response.rdb")


def _write_protocol_seal_fixture(
    verifier,
    root: Path,
    *,
    original_commit: str = "1" * 40,
    final_commit: str = "2" * 40,
    original_markdown_sha256: str | None = None,
) -> tuple[Path, Path, Path]:
    protocol = _write_bytes(
        root,
        "protocols/route_a_confirmatory_v1.json",
        json.dumps({
            "protocol_id": "route-a-confirmatory-v1",
            "authoritative_protocol_commit": original_commit,
            "primary_inference_contract": {
                "confirmatory_family": _fixture_confirmatory_family()
            },
            "availability_contract": {
                "minimum_valid_targets_per_station_horizon": 2
            },
            "time_holdout": {
                "primary_target_start": "2021-01-01",
                "end": "2021-12-31",
            },
        }, sort_keys=True).encode() + b"\n",
    )
    markdown = _write_bytes(
        root,
        "protocols/route_a_confirmatory_protocol.md",
        b"# Final prelabel fixture protocol\n",
    )
    seal = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": "protocols/route_a_confirmatory_protocol.md",
                "sha256": original_markdown_sha256 or "3" * 64,
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": "protocols/route_a_confirmatory_v1.json",
                "sha256": verifier.sha256_file(protocol),
            },
            "markdown": {
                "path": "protocols/route_a_confirmatory_protocol.md",
                "sha256": verifier.sha256_file(markdown),
            },
        },
        "prelabel_attestation": {
            "external_timestamp_or_public_preregistration": False,
            "independent_custodian_or_worm_storage": False,
        },
    }
    seal_path = _write_bytes(
        root,
        verifier.PROTOCOL_SEAL_PATH,
        json.dumps(seal, sort_keys=True).encode() + b"\n",
    )
    return protocol, markdown, seal_path


def _write_claim_fixture_files(stage: Path) -> None:
    validator = stage / "scripts" / "26_validate_claims.py"
    validator.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "26_validate_claims.py", validator)
    registry = stage / "protocols" / "route_a_claim_registry_v1.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({
        "format": "thermoroute.route-a-claim-registry.v1",
        "documents": ["paper/main.md"],
        "claims": [{
            "claim_id": "FIXTURE",
            "status": "SUPPORTED_AFTER_OPENING",
            "forbidden_regex": ["NEVER_MATCH_THIS_FIXTURE"],
        }],
    }), encoding="utf-8")
    _write_bytes(stage, "paper/main.md", b"scoped development language\n")


def _materialize_claim_fixture(verifier, stage: Path, profile: str) -> None:
    _write_claim_fixture_files(stage)
    validator = stage / "scripts" / "26_validate_claims.py"
    registry = stage / "protocols" / "route_a_claim_registry_v1.json"
    scanned = [stage / "paper" / "main.md"]
    audit = {
        "format": "thermoroute.route-a-release-claim-audit.v1",
        "profile": profile,
        "require_complete": profile == verifier.POSTOPEN_PROFILE,
        "validator": verifier._binding_for(stage, validator),
        "registry": verifier._binding_for(stage, registry),
        "scanned_documents": [
            verifier._binding_for(stage, path) for path in scanned
        ],
        "violation_count": 0,
        "validator_stdout": "fixture static claim audit",
    }
    audit_path = _write_bytes(
        stage,
        verifier.CLAIM_AUDIT_PATH,
        json.dumps(audit, sort_keys=True).encode() + b"\n",
    )
    marker_path = stage / verifier.PROFILE_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["claim_validation"] = verifier._binding_for(stage, audit_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _write_postopen_fixture(verifier, root: Path) -> tuple[Path, dict[str, str]]:
    _minimal_canonical_release(verifier, root)
    _write_bytes(
        root,
        "data_usgs/station_registry_v1.csv",
        b"site_no,huc2\n01073319,01\n",
    )
    _write_bytes(root, "requirements-lock.txt", b"numpy==1.0\n")
    _write_protocol_seal_fixture(verifier, root)
    _write_bytes(
        root,
        "data_usgs/external.csv",
        b"site_no,huc2\n02000001,02\n",
    )
    _write_bytes(root, "data_usgs/external.lock.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidates.csv")
    _write_bytes(root, "data_usgs/candidates.provenance.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/snapshot_index.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/response.rdb")
    _write_bytes(root, "src/thermoroute/opening.py", b"# fixed\n")
    _write_bytes(root, "src/thermoroute/chronology.py", b"# chronology gate\n")
    for relative in (
        "src/thermoroute/outcome_qc.py",
        "src/thermoroute/coverage_audit.py",
        "src/thermoroute/coverage_bridge.py",
        "src/thermoroute/repro.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    coverage_policy_path = verifier.TEMPORAL_COVERAGE_POLICY_PATH
    coverage_policy_destination = root / coverage_policy_path
    coverage_policy_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / coverage_policy_path, coverage_policy_destination)
    _write_bytes(
        root, "scripts/28_freeze_prelabel_chronology.py", b"# chronology gate\n"
    )
    _write_bytes(root, "tests/test_chronology.py", b"# chronology gate\n")
    _write_bytes(root, "scripts/route_a_trusted_scorer.py", b"# fixed\n")
    _write_claim_fixture_files(root)

    protocol_binding = _binding(
        verifier, root, "protocols/route_a_confirmatory_v1.json"
    )
    protocol_seal_binding = _binding(verifier, root, verifier.PROTOCOL_SEAL_PATH)
    family = _fixture_confirmatory_family()
    family_sha256 = verifier._sha256_json(family)
    outcome_qc_policy_path = "protocols/route_a_outcome_qc_policy_v1.json"
    outcome_qc_policy = json.loads(
        (ROOT / outcome_qc_policy_path).read_text(encoding="utf-8")
    )
    outcome_qc_policy["base_protocol"] = protocol_binding
    outcome_qc_policy["confirmatory_family_sha256"] = family_sha256
    _write_bytes(
        root, outcome_qc_policy_path, json.dumps(outcome_qc_policy).encode()
    )
    amendment_path = "protocols/route_a_inference_amendment_v1.json"
    trusted_scoring_recovery_contract = json.loads(
        (ROOT / amendment_path).read_text(encoding="utf-8")
    )["trusted_scoring_recovery_contract"]
    policy_overlay = {
        **_binding(verifier, root, outcome_qc_policy_path),
        "required": True,
        "role": verifier.OUTCOME_QC_AMENDMENT_ROLE,
    }
    coverage_policy = json.loads(
        coverage_policy_destination.read_text(encoding="utf-8")
    )
    coverage_policy_overlay = {
        **_binding(verifier, root, coverage_policy_path),
        "required": True,
        "role": verifier.TEMPORAL_COVERAGE_AMENDMENT_ROLE,
    }
    amendment = {
        "format": "thermoroute.route-a-inference-amendment.v1",
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "amendment_id": "route-a-prelabel-inference-scope-014",
        "recorded_date": "2026-07-22",
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent": True,
        "base_protocol": protocol_binding,
        "base_protocol_seal": protocol_seal_binding,
        "scientific_comparisons": {
            "count": 5,
            "confirmatory_family_sha256": family_sha256,
            "objects": family,
            "change_allowed": False,
        },
        "estimand_scope": {"fixture": "fixed cohort"},
        "inference_scope": {"fixture": "assumption conditional"},
        "decision_overlay": {
            "gate_artifact": "outputs/prelabel/route_a_inference_gate_v1.json",
            "all_gate_components_must_pass": True,
            "missing_unknown_or_not_run_is_failure": True,
            "gate_failure_verdict": "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
            "supported_claim_allowed_when_gate_fails": False,
            "strong_p_value_or_favorable_interval_cannot_override_gate": True,
            "all_five_comparisons_must_still_be_rendered_exactly_once": True,
        },
        "additional_preopen_gates": {
            "outcome_qc_policy": policy_overlay,
            "temporal_coverage_policy": coverage_policy_overlay,
        },
        "trusted_scoring_recovery_contract": trusted_scoring_recovery_contract,
        "lineage_contract": {
            "base_v1_files_remain_immutable": True,
            "separate_amendment_seal_required": True,
            "seal_path": "protocols/route_a_inference_amendment_seal_v1.json",
            "amendment_commit_must_precede_seal_commit": True,
        },
    }
    _write_bytes(root, amendment_path, json.dumps(amendment).encode())
    amendment_commit = "7" * 40
    amendment_seal_path = "protocols/route_a_inference_amendment_seal_v1.json"
    amendment_seal = {
        "format": "thermoroute.route-a-inference-amendment-seal.v1",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "amendment_id": amendment["amendment_id"],
        "amendment": _binding(verifier, root, amendment_path),
        "base_protocol_seal": protocol_seal_binding,
        "final_prelabel_commit": amendment_commit,
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
    _write_bytes(root, amendment_seal_path, json.dumps(amendment_seal).encode())

    model_entries: dict[str, list[dict[str, object]]] = {}
    builtins = {"Persistence", "DampedPersistence", "Climatology"}
    for cohort in ("temporal", "external"):
        entries = []
        for model in sorted(verifier._required_model_ids(cohort)):
            if model in builtins:
                entries.append({"model_id": model, "executor": "builtin"})
                continue
            relative = f"outputs/models/{cohort}/{model}.bundle"
            _write_bytes(root, relative, f"{cohort}/{model}\n".encode())
            entries.append({
                "model_id": model,
                "executor": "frozen_fixture_bundle",
                "artifact": _binding(verifier, root, relative),
            })
        model_entries[cohort] = entries
    runtime_sha256 = "c" * 64
    model_control_paths = sorted(verifier._working_model_control_paths(root))
    source_inventory = {
        relative: verifier.sha256_file(root / relative)
        for relative in model_control_paths
        if verifier._matches_source_inventory(relative)
    }
    source_sha256 = verifier._sha256_json(source_inventory)
    inference_policy = {
        "estimand_scope": {
            "fixed_cohort": "fixture fixed cohort",
            "superpopulation": "assumption conditional only",
        },
        "cluster_thresholds": {
            "minimum_clusters": 30,
            "minimum_effective_cluster_fraction": 0.75,
            "maximum_largest_cluster_share_exclusive": 0.25,
        },
        "structural_assumptions": [
            {
                "assumption_id": "INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING",
                "status": "NOT_ESTABLISHED",
            },
            {
                "assumption_id": "JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY",
                "status": "NOT_ESTABLISHED",
            },
        ],
        "null_simulation": {
            "role": "FALSIFICATION_ONLY_NEVER_ESTABLISHES_STRUCTURAL_ASSUMPTIONS",
            "required_before_inferential_claims": True,
            "synthetic_boundary_null_only": True,
            "post_2020_outcomes_allowed": False,
            "caller_supplied_effects_allowed": False,
            "network_allowed": False,
            "scenarios": ["fixture"],
        },
        "decision": {
            "all_components_must_pass": True,
            "missing_unknown_or_not_run_is_failure": True,
            "failed_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
            "failed_verdict": "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
        },
    }
    gate_path = "outputs/prelabel/route_a_inference_gate_v1.json"
    registry_path = root / "data_usgs/station_registry_v1.csv"
    gate_geometry = verifier._independent_station_geometry(registry_path)
    threshold_failures = [
        "SMALL_CLUSTER_COUNT_LT_30",
        "DOMINANT_CLUSTER_SHARE_GE_0_25",
    ]
    structural_failures = [
        "INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING",
        "JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY",
    ]
    null_gate = {
        **inference_policy["null_simulation"],
        "status": "NOT_RUN_BLOCKED_BY_STRUCTURAL_OR_CLUSTER_GATE",
        "pass": False,
        "outcomes_read": False,
        "network_used": False,
    }
    inference_gate = {
        "format": "thermoroute.route-a-inference-gate.v1",
        "status": "FAIL_CLOSED_DESCRIPTIVE_ONLY",
        "contains_confirmation_outcomes": False,
        "post_2020_outcomes_requested_or_inspected": False,
        "network_used": False,
        "inputs": {
            "base_protocol": protocol_binding,
            "base_protocol_seal": protocol_seal_binding,
            "station_registry": _binding(
                verifier, root, "data_usgs/station_registry_v1.csv"
            ),
            "source": {
                "source_tree_sha256": source_sha256,
                "source_inventory": source_inventory,
            },
        },
        "confirmatory_family": {
            "count": 5,
            "sha256": family_sha256,
            "objects": family,
            "candidate_reference_horizon_margin_unchanged": True,
        },
        "policy": inference_policy,
        "policy_sha256": verifier._sha256_json(inference_policy),
        "cluster_geometry": gate_geometry,
        "cluster_gate": {"pass": False, "failure_codes": threshold_failures},
        "structural_assumption_gate": {
            "pass": False,
            "failure_codes": structural_failures,
        },
        "null_simulation_gate": null_gate,
        "claim_eligible": False,
        "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
        "blocking_reasons": [
            *(f"STRUCTURAL_ASSUMPTION_NOT_ESTABLISHED:{value}"
              for value in structural_failures),
            *threshold_failures,
            "NULL_SIMULATION_NOT_PASSING",
        ],
    }
    inference_gate["gate_self_sha256"] = verifier._sha256_json(inference_gate)
    _write_bytes(root, gate_path, json.dumps(inference_gate).encode())
    bridge_dependencies = {
        "frozen": "data_usgs/development_predictor_bridge_v1/frozen.parquet",
        "refreshed": "data_usgs/development_predictor_bridge_v1/refreshed.parquet",
        "report": "data_usgs/development_predictor_bridge_v1/report.json",
        "request_map": "data_usgs/development_predictor_bridge_v1/request_map.json",
        "daymet": "data_usgs/raw_snapshots/development-bridge/daymet.json",
        "gridmet": "data_usgs/raw_snapshots/development-bridge/gridmet.json",
        "gridmet_schema": "data_usgs/raw_snapshots/development-bridge/schema.json",
    }
    for relative in bridge_dependencies.values():
        _write_bytes(root, relative, b"{}\n" if relative.endswith(".json") else b"bridge\n")
    bridge_path = "data_usgs/development_predictor_bridge_v1.json"
    bridge = {
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": source_sha256,
        "panel": _binding(verifier, root, "data_usgs/panel_usgs_120v2.parquet"),
        "registry": _binding(verifier, root, "data_usgs/station_registry_v1.csv"),
        "normalized": {
            name: _binding(verifier, root, bridge_dependencies[name])
            for name in ("frozen", "refreshed")
        },
        "report": _binding(verifier, root, bridge_dependencies["report"]),
        "request_map": _binding(verifier, root, bridge_dependencies["request_map"]),
        "raw_snapshot_indexes": {
            name: _binding(verifier, root, bridge_dependencies[name])
            for name in ("daymet", "gridmet", "gridmet_schema")
        },
    }
    _write_bytes(root, bridge_path, json.dumps(bridge).encode())

    def write_preopening_gate_fixtures() -> dict[str, dict[str, str]]:
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from thermoroute.development_controls import (
                ARCHITECTURE_BUDGET_FORMAT,
                METRIC_SUMMARY_FORMAT,
                REPORT_FORMAT,
                SEMANTIC_AUDIT_FORMAT,
                architecture_budget_rows,
                budget_csv_bytes,
                declared_arms,
                recompute_metric_summary,
                recompute_paired_effect_summary,
                recompute_station_rmse,
                render_report,
                scientific_summary_document,
                summary_csv_bytes,
            )
        finally:
            sys.path.pop(0)

        expected_members = verifier._stage09b_release_members()
        controls_config = _stage09b_fixture_config(
            _binding(verifier, root, bridge_path), eval_batch_size=2,
        )
        identity_without_run = {
            "panel_sha256": bridge["panel"]["sha256"],
            "registry_sha256": bridge["registry"]["sha256"],
            "config_sha256": verifier._sha256_json(controls_config),
            "source_sha256": source_sha256,
            "runtime_sha256": runtime_sha256,
            "schema_version": "thermoroute.run.v1",
        }
        identity = {
            "run_id": verifier._sha256_json(identity_without_run)[:20],
            **identity_without_run,
        }
        run_dir = (
            f"outputs/runs/09b_development_controls/{identity['run_id']}"
        )
        matrix_audit = {
            "expected_members": 31,
            "prediction_rows": 31 * 9,
            "common_forecast_keys": 9,
            "splits": ["calib", "test", "val"],
            "reference_member": "PlainMLP-7var/seed0",
        }
        run_manifest_path = f"{run_dir}/run.json"
        _write_bytes(root, run_manifest_path, json.dumps({
            "schema_version": "thermoroute.run.v1",
            "identity": identity,
            "resolved_config": controls_config,
            "created_utc": "2026-07-22T00:00:00+00:00",
            "environment": {},
            "git": {},
            "provenance": {
                "development_only": True,
                "post_2020_outcomes_requested_or_read": False,
                "suite_pointer_written": False,
                "training_device": "cpu",
            },
        }).encode())
        arm_documents = {
            str(arm["arm_id"]): arm for arm in controls_config["arms"]
        }
        arm_features = {
            arm_id: str(document["feature_set"])
            for arm_id, document in arm_documents.items()
        }
        member_registry = []
        member_frames = {}
        base_parents = {
            "frozen_panel": identity["panel_sha256"],
            "frozen_station_registry": identity["registry_sha256"],
            "development_predictor_bridge": verifier.sha256_file(root / bridge_path),
        }
        for arm_id, seed in expected_members:
            relative = f"{run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
            checkpoint_relative = f"{run_dir}/checkpoints/{arm_id}/seed{seed}.pt"
            checkpoint = _write_bytes(
                root, checkpoint_relative,
                f"fixture checkpoint {arm_id}/seed{seed}\n".encode(),
            )
            checkpoint_sidecar_relative = checkpoint_relative + ".meta.json"
            arm_document = arm_documents[arm_id]
            arm_config = {
                **controls_config,
                "arm": arm_document,
                "seed": seed,
                "trainable_parameters": controls_config["parameter_counts"][arm_id],
            }
            expected_model_class = {
                "PlainMLP": "thermoroute.neural_baselines.PlainMLPForecaster",
                "PlainCausalTCN": (
                    "thermoroute.neural_baselines.PlainCausalTCNForecaster"
                ),
                "ThermoRoute": "thermoroute.thermoroute.ThermoRoute",
            }[str(arm_document["family"])]
            _write_bytes(root, checkpoint_sidecar_relative, json.dumps({
                "format": "thermoroute.training-checkpoint-metadata.v2",
                "checkpoint_format": "thermoroute.training-checkpoint.v3",
                "run_id": identity["run_id"],
                "epoch": 4,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": verifier.sha256_file(checkpoint),
                "resolved_config_sha256": verifier._sha256_json(arm_config),
                "extra_sha256": "e" * 64,
                "model_class": expected_model_class,
                "optimizer_class": "torch.optim.adamw.AdamW",
                "scheduler_class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "scheduler_present": True,
            }).encode())
            records = []
            for split_index, split in enumerate(("val", "calib", "test")):
                for horizon in (1, 3, 7):
                    issue_date = pd.Timestamp("2017-01-01") + pd.Timedelta(
                        days=split_index * 30 + horizon
                    )
                    truth = float(split_index + horizon / 10)
                    prediction_value = truth + float(seed + 1) / 100
                    records.append({
                        "model": arm_id,
                        "scope": "development_only_2006_2020",
                        "feature_set": arm_features[arm_id],
                        "seed": seed,
                        "site_id": "01073319",
                        "horizon": horizon,
                        "split": split,
                        "issue_date": issue_date,
                        "target_date": issue_date + pd.Timedelta(days=horizon),
                        "y_true": truth,
                        "y_pred": prediction_value,
                        "q05": prediction_value - 1.0,
                        "q50": prediction_value,
                        "q95": prediction_value + 1.0,
                        "p_exceed": 0.25,
                    })
            frame = pd.DataFrame.from_records(records)
            prediction = root / relative
            prediction.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(prediction, index=False)
            member_frames[(arm_id, seed)] = frame
            sidecar_relative = relative + ".meta.json"
            checkpoint_binding = _binding(
                verifier, root, checkpoint_relative,
            )
            checkpoint_sidecar_binding = _binding(
                verifier, root, checkpoint_sidecar_relative,
            )
            member_parents = {
                **base_parents,
                "training_checkpoint": checkpoint_binding["sha256"],
                "training_checkpoint_sidecar": checkpoint_sidecar_binding["sha256"],
            }
            training_summary = {
                "best_validation_metric": 0.25,
                "selected_epoch": 2,
                "checkpoint_final_epoch": 4,
            }
            member_extra = {
                "format": "thermoroute.development-control-arm.v2",
                "arm_id": arm_id,
                "family": arm_document["family"],
                "feature_set": arm_document["feature_set"],
                "variables": arm_document["variables"],
                "seed": seed,
                "trainable_parameters": controls_config["parameter_counts"][arm_id],
                "architecture": controls_config["architecture_templates"][arm_id],
                "training_device": "cpu",
                "station_balanced": True,
                "selection_metric": "station_macro",
                "train_config": controls_config["train_config"],
                "context_length": 32,
                "horizons": [1, 3, 7],
                "development_only": True,
                "development_evaluation_interval": [
                    "2019-01-01", "2020-12-31"
                ],
                "blind_or_confirmatory": False,
                "suite_pointer_written": False,
                "eval_batch_size": 2,
                "training_summary": training_summary,
            }
            _write_bytes(root, sidecar_relative, json.dumps({
                "schema_version": "thermoroute.artifact.v1",
                "kind": "development_control_arm_predictions",
                "artifact": prediction.name,
                "artifact_sha256": verifier.sha256_file(prediction),
                "artifact_bytes": prediction.stat().st_size,
                "content_schema": "thermoroute.predictions.v1",
                "run": identity,
                "parents": dict(sorted(member_parents.items())),
                "extra": member_extra,
                "created_utc": "2026-07-22T00:00:00+00:00",
            }).encode())
            member_registry.append({
                "arm_id": arm_id,
                "seed": seed,
                "checkpoint": checkpoint_binding,
                "checkpoint_sidecar": checkpoint_sidecar_binding,
                "prediction": _binding(verifier, root, relative),
                "prediction_sidecar": _binding(verifier, root, sidecar_relative),
            })
        final_paths = {
            "predictions": f"{run_dir}/development_controls_predictions.parquet",
            "architecture_budget": (
                f"{run_dir}/development_controls_architecture_budget.csv"
            ),
            "metric_summary": f"{run_dir}/development_controls_metric_summary.csv",
            "report": f"{run_dir}/development_controls_report.md",
            "semantic_audit": f"{run_dir}/development_controls_semantic_audit.json",
        }
        combined = pd.concat(
            [member_frames[member] for member in expected_members], ignore_index=True
        )
        combined.to_parquet(root / final_paths["predictions"], index=False)
        budget = architecture_budget_rows(
            declared_arms(), n_stations=120, train_examples=9
        )
        _write_bytes(
            root, final_paths["architecture_budget"],
            budget_csv_bytes(budget),
        )
        summary = recompute_metric_summary(member_frames)
        paired_effects = recompute_paired_effect_summary(
            recompute_station_rmse(member_frames),
            exact_common_forecast_keys_verified=True,
        )
        _write_bytes(
            root, final_paths["metric_summary"], summary_csv_bytes(summary)
        )
        _write_bytes(
            root,
            final_paths["report"],
            render_report(
                run_id=identity["run_id"],
                audit=matrix_audit,
                budget=budget,
                summary=summary,
                paired_effects=paired_effects,
            ).encode("utf-8"),
        )
        final_specs = {
            "predictions": (
                "development_controls_combined_predictions",
                "thermoroute.predictions.v1", "combined_predictions",
            ),
            "architecture_budget": (
                "development_controls_budget", ARCHITECTURE_BUDGET_FORMAT,
                "architecture_budget",
            ),
            "metric_summary": (
                "development_controls_metric_summary", METRIC_SUMMARY_FORMAT,
                "metric_summary",
            ),
            "report": (
                "development_controls_report", REPORT_FORMAT, "report",
            ),
            "semantic_audit": (
                "development_controls_semantic_audit", SEMANTIC_AUDIT_FORMAT,
                "semantic_audit",
            ),
        }
        final_parents = {
            **base_parents,
            **{
                f"arm::{entry['arm_id']}::seed{entry['seed']}::prediction": (
                    entry["prediction"]["sha256"]
                )
                for entry in member_registry
            },
            **{
                f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint": (
                    entry["checkpoint"]["sha256"]
                )
                for entry in member_registry
            },
            **{
                f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint_sidecar": (
                    entry["checkpoint_sidecar"]["sha256"]
                )
                for entry in member_registry
            },
        }

        def write_final_sidecar(name: str) -> None:
            relative = final_paths[name]
            artifact = root / relative
            kind, content_schema, role = final_specs[name]
            _write_bytes(root, relative + ".meta.json", json.dumps({
                "schema_version": "thermoroute.artifact.v1",
                "kind": kind,
                "artifact": artifact.name,
                "artifact_sha256": verifier.sha256_file(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "content_schema": content_schema,
                "run": identity,
                "parents": dict(sorted(final_parents.items())),
                "extra": verifier._stage09b_expected_final_extra(
                    matrix_audit, role=role,
                ),
                "created_utc": "2026-07-22T00:00:00+00:00",
            }).encode())

        for name in ("predictions", "architecture_budget", "metric_summary", "report"):
            write_final_sidecar(name)

        canonical_evaluation = verifier._normalise_stage09b_release_prediction(
            member_frames[expected_members[0]],
            arm_id=expected_members[0][0],
            seed=expected_members[0][1],
            feature_set=arm_features[expected_members[0][0]],
            reference=None,
        )[["split", "site_id", "horizon", "issue_date", "target_date", "y_true"]]
        canonical_evaluation_sha256 = verifier._stage09b_window_registry_digest(
            canonical_evaluation
        )
        canonical_train_sha256 = verifier._stage09b_window_registry_digest(
            canonical_evaluation
        )

        def fixture_canonical_windows(*_args, **_kwargs):
            return (
                canonical_evaluation.copy(), 9,
                canonical_evaluation_sha256, canonical_train_sha256,
                ("01073319",),
            )

        verifier._stage09b_rebuild_canonical_windows = fixture_canonical_windows
        semantic_members = []
        for entry in member_registry:
            prediction = root / entry["prediction"]["path"]
            prediction_sidecar = root / entry["prediction_sidecar"]["path"]
            checkpoint = root / entry["checkpoint"]["path"]
            checkpoint_sidecar = root / entry["checkpoint_sidecar"]["path"]
            member = (entry["arm_id"], entry["seed"])
            normalised = verifier._normalise_stage09b_release_prediction(
                member_frames[member], arm_id=entry["arm_id"], seed=entry["seed"],
                feature_set=arm_features[entry["arm_id"]],
                reference=canonical_evaluation,
            )
            semantic_members.append({
                "arm_id": entry["arm_id"],
                "seed": entry["seed"],
                "checkpoint": {
                    "sha256": verifier.sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                },
                "checkpoint_sidecar": {
                    "sha256": verifier.sha256_file(checkpoint_sidecar),
                    "bytes": checkpoint_sidecar.stat().st_size,
                },
                "prediction": {
                    "sha256": verifier.sha256_file(prediction),
                    "bytes": prediction.stat().st_size,
                },
                "prediction_sidecar": {
                    "sha256": verifier.sha256_file(prediction_sidecar),
                    "bytes": prediction_sidecar.stat().st_size,
                },
                "normalised_prediction_sha256": (
                    verifier._stage09b_prediction_content_digest(normalised)
                ),
                "best_model_state_prediction_replay_verified": True,
            })
        derived = {}
        for label, name in (
            ("architecture_budget", "architecture_budget"),
            ("combined_predictions", "predictions"),
            ("metric_summary", "metric_summary"),
            ("report", "report"),
        ):
            artifact = root / final_paths[name]
            sidecar = root / (final_paths[name] + ".meta.json")
            derived[label] = {
                "artifact": {
                    "sha256": verifier.sha256_file(artifact),
                    "bytes": artifact.stat().st_size,
                },
                "sidecar": {
                    "sha256": verifier.sha256_file(sidecar),
                    "bytes": sidecar.stat().st_size,
                },
            }
        semantic = {
            "format": "thermoroute.development-controls-semantic-audit.v3",
            "status": "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY",
            "run_id": identity["run_id"],
            "evidence_scope": "best_model_state_prediction_replay",
            "best_model_state_prediction_replay_verified": True,
            "training_replay_verified": False,
            "post_2020_outcomes_requested_or_read": False,
            "matrix_audit": matrix_audit,
            "canonical_window_registry": {
                "sha256": canonical_evaluation_sha256,
                "common_forecast_keys": 9,
                "train_examples_per_epoch": 9,
                "train_registry_sha256": canonical_train_sha256,
            },
            "scientific_summary": scientific_summary_document(paired_effects),
            "members": semantic_members,
            "derived_artifacts": derived,
        }
        semantic["semantic_audit_self_sha256"] = verifier._sha256_json(semantic)
        _write_bytes(
            root, final_paths["semantic_audit"], json.dumps(semantic).encode()
        )
        write_final_sidecar("semantic_audit")
        controls_artifacts = {
            "run_manifest": _binding(verifier, root, run_manifest_path),
            "frozen_panel_spec": _binding(
                verifier, root, "data_usgs/frozen_panel_v1.json"
            ),
            "panel": bridge["panel"],
            "registry": bridge["registry"],
            "predictor_bridge": _binding(verifier, root, bridge_path),
            **{
                name: _binding(verifier, root, relative)
                for name, relative in final_paths.items()
            },
            "prediction_sidecar": _binding(
                verifier, root, final_paths["predictions"] + ".meta.json"
            ),
            "architecture_budget_sidecar": _binding(
                verifier, root, final_paths["architecture_budget"] + ".meta.json"
            ),
            "metric_summary_sidecar": _binding(
                verifier, root, final_paths["metric_summary"] + ".meta.json"
            ),
            "report_sidecar": _binding(
                verifier, root, final_paths["report"] + ".meta.json"
            ),
            "semantic_audit_sidecar": _binding(
                verifier, root, final_paths["semantic_audit"] + ".meta.json"
            ),
        }
        controls = {
            "format": "thermoroute.stage09b-completion-receipt.v3",
            "status": "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
            "stage": "09b_development_controls",
            "run_id": identity["run_id"],
            "run_identity": identity,
            "formal_configuration": controls_config,
            "evidence_scope": "best_model_state_prediction_replay",
            "best_model_state_prediction_replay_verified": True,
            "training_replay_verified": False,
            "matrix_audit": matrix_audit,
            "member_registry": member_registry,
            "artifacts": controls_artifacts,
            "post_2020_outcomes_requested_or_read": False,
        }
        controls["receipt_self_sha256"] = verifier._sha256_json(controls)
        controls_path = "outputs/models/route_a_stage09b_completion.json"
        _write_bytes(root, controls_path, json.dumps(controls).encode())

        stage9_artifacts = {}
        for name in (
            "run_manifest", "predictions", "prediction_sidecar", "scores",
            "report", "lightgbm_selection", "thermoroute_pointer",
            "lightgbm_pointer", "components_pointer",
        ):
            relative = f"outputs/runs/stage9-fixture/{name}.json"
            _write_bytes(root, relative, b"{}\n")
            stage9_artifacts[name] = _binding(verifier, root, relative)
        stage9_identity = {**identity, "run_id": "stage9-fixture"}
        stage9 = {
            "format": "thermoroute.stage09-completion-receipt.v1",
            "status": "PASS_FORMAL_STAGE09_COMPLETE",
            "stage": "09_usgs_experiment",
            "run_id": stage9_identity["run_id"],
            "run_identity": stage9_identity,
            "formal_configuration": {"stage": "09_usgs_experiment"},
            "confirmation_outcomes_requested_or_read": False,
            "artifacts": stage9_artifacts,
        }
        stage9["receipt_self_sha256"] = verifier._sha256_json(stage9)
        stage9_path = "outputs/models/route_a_stage09_completion.json"
        _write_bytes(root, stage9_path, json.dumps(stage9).encode())
        return {
            "stage09_completion": _binding(verifier, root, stage9_path),
            "stage09b_development_controls": _binding(
                verifier, root, controls_path
            ),
        }

    preopening_gates = write_preopening_gate_fixtures()
    suite = {
        "format": "thermoroute.route-a-model-suite.v1",
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": runtime_sha256,
        "preopening_gates": preopening_gates,
        "development_contract": {
            "frozen_panel_spec": _binding(
                verifier, root, "data_usgs/frozen_panel_v1.json"
            ),
            "panel": bridge["panel"],
            "registry": bridge["registry"],
            "predictor_bridge": _binding(verifier, root, bridge_path),
            "source_sha256": source_sha256,
        },
        "cohorts": {
            cohort: {"models": entries}
            for cohort, entries in model_entries.items()
        },
    }
    suite_path = root / "data_usgs/confirmatory_model_suite_v1.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    source_tree_sha256 = source_sha256
    development_replay = {
        "format": "thermoroute.route-a-development-replay.v1",
        "status": "PASS_FULL_DEVELOPMENT_REPLAY_NO_CONFIRMATION_DATA",
        "suite": _binding(verifier, root, "data_usgs/confirmatory_model_suite_v1.json"),
        "source_tree_sha256": source_tree_sha256,
        "runtime_sha256": runtime_sha256,
        "confirmation_period_read": False,
    }
    development_replay_path = root / "outputs/model_replay/route_a_development_replay_v1.json"
    development_replay_path.parent.mkdir(parents=True, exist_ok=True)
    development_replay_path.write_text(json.dumps(development_replay), encoding="utf-8")

    cohort_tables = {}
    for cohort in ("temporal", "external"):
        relative = f"data_usgs/prelabel/{cohort}.parquet"
        _write_bytes(root, relative)
        cohort_tables[cohort] = _binding(verifier, root, relative)
    evidence = []
    for index in range(4):
        relative = f"data_usgs/raw_snapshots/met-{index}/snapshot_index.json"
        _write_bytes(root, relative, b"{}\n")
        _write_bytes(root, f"data_usgs/raw_snapshots/met-{index}/response.bin")
        evidence.append({
            "contains_outcome": False,
            "contains_outcome_labels": False,
            "artifact": _binding(verifier, root, relative),
        })
    inputs = {
        "format": "thermoroute.route-a-prelabel-inputs.v1",
        "status": "FROZEN_PRELABEL_NO_OUTCOMES",
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "post_2020_wtemp_requested_or_inspected": False,
        "cohort_tables": cohort_tables,
        "source_evidence": evidence,
    }
    inputs_path = root / "data_usgs/confirmatory_actual_inputs_v1.json"
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")

    chronology_order = {
        "model_freeze_commit": "4" * 40,
        "input_evidence_commit": "5" * 40,
        "receipt_creation_base_commit": "6" * 40,
        "strict_order_verified": True,
    }
    chronology_stable = {
        "format": verifier.CHRONOLOGY_FORMAT,
        "status": verifier.CHRONOLOGY_STATUS,
        "order": chronology_order,
        "protocol_history": {
            "seal": _chronology_binding(
                verifier, root, verifier.PROTOCOL_SEAL_PATH
            ),
            "original_commit": "1" * 40,
            "final_prelabel_commit": "2" * 40,
            "declared_git_show_bindings": [
                {
                    "role": "original_markdown",
                    "commit": "1" * 40,
                    "path": "protocols/route_a_confirmatory_protocol.md",
                    "sha256": "3" * 64,
                },
                {
                    "role": "final_json",
                    "commit": "2" * 40,
                    "path": "protocols/route_a_confirmatory_v1.json",
                    "sha256": verifier.sha256_file(
                        root / "protocols/route_a_confirmatory_v1.json"
                    ),
                },
                {
                    "role": "final_markdown",
                    "commit": "2" * 40,
                    "path": "protocols/route_a_confirmatory_protocol.md",
                    "sha256": verifier.sha256_file(
                        root / "protocols/route_a_confirmatory_protocol.md"
                    ),
                },
            ],
        },
        "paths": {
            "protocol_seal": verifier.PROTOCOL_SEAL_PATH,
            "model_suite": "data_usgs/confirmatory_model_suite_v1.json",
            "development_replay": (
                "outputs/model_replay/route_a_development_replay_v1.json"
            ),
            "candidate_table": "data_usgs/candidates.csv",
            "candidate_provenance": "data_usgs/candidates.provenance.json",
            "candidate_snapshot_index": (
                "data_usgs/candidate-raw/snapshot_index.json"
            ),
            "external_registry": "data_usgs/external.csv",
            "external_lock": "data_usgs/external.lock.json",
            "input_manifest": "data_usgs/confirmatory_actual_inputs_v1.json",
        },
        "required_gate_files_at_model_freeze": [
            _chronology_binding(verifier, root, relative)
            for relative in (
                "src/thermoroute/chronology.py",
                "src/thermoroute/outcome_qc.py",
                "scripts/28_freeze_prelabel_chronology.py",
                "tests/test_chronology.py",
                "protocols/route_a_outcome_qc_policy_v1.json",
            )
        ],
        "model_source_control_artifacts": [
            _chronology_binding(verifier, root, relative)
            for relative in model_control_paths
        ],
        "source_tree_sha256": source_tree_sha256,
        "model_freeze_artifacts": [
            _chronology_binding(verifier, root, relative)
            for relative in (
                "data_usgs/confirmatory_model_suite_v1.json",
                "outputs/model_replay/route_a_development_replay_v1.json",
            )
        ],
        "input_evidence_artifacts": [
            _chronology_binding(verifier, root, relative)
            for relative in (
                "data_usgs/candidates.csv",
                "data_usgs/candidates.provenance.json",
                "data_usgs/candidate-raw/snapshot_index.json",
                "data_usgs/external.csv",
                "data_usgs/external.lock.json",
                "data_usgs/confirmatory_actual_inputs_v1.json",
            )
        ],
        "absence_at_model_freeze": {
            "checked_paths": [
                "data_usgs/candidates.csv",
                "data_usgs/confirmatory_actual_inputs_v1.json",
            ],
            "present_paths": [],
        },
        "post_model_control_audit": {
            "protected_directories": list(verifier.PROTECTED_DIRECTORIES),
            "protected_exact_files": list(verifier.PROTECTED_EXACT_FILES),
            "protected_root_patterns": list(verifier.PROTECTED_ROOT_PATTERNS),
            "committed_touches": [],
            "worktree_changes": [],
        },
        "post_freeze_artifact_mutation_count": 0,
        "external_timestamp_or_public_preregistration": False,
        "independent_custodian_or_worm_storage": False,
        "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        "fallback_if_validation_fails": (
            "TRANSDUCTIVE_RETROSPECTIVE_EXPLORATION_"
            "CONFIRMATION_CLAIMS_PROHIBITED"
        ),
    }
    chronology = {
        **chronology_stable,
        "receipt_self_sha256": verifier._chronology_self_sha256(
            chronology_stable
        ),
    }
    chronology_path = root / verifier.CHRONOLOGY_PATH
    chronology_path.parent.mkdir(parents=True, exist_ok=True)
    chronology_path.write_text(json.dumps(chronology), encoding="utf-8")

    namespace = "b" * 24
    base = f"outputs/confirmatory/route_a_{namespace}"
    state = {
        "namespace": namespace,
        "run_directory": base,
        "work_order": f"{base}/acquisition_work_order_v1.json",
        "intent": f"{base}/opening_intent_v1.json",
        "transport_root": f"{base}/transport",
        "raw_nwis_root": f"{base}/transport/raw_nwis_v1",
        "raw_nwis_snapshot_index": (
            f"{base}/transport/raw_nwis_v1/snapshot_index.json"
        ),
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
    authorization_path = root / "data_usgs/confirmatory_opening_authorization_v1.json"
    fixed_binding = _binding(verifier, root, "src/thermoroute/opening.py")
    scorer_binding = _binding(verifier, root, "scripts/route_a_trusted_scorer.py")
    authorization = {
        "format": verifier.AUTHORIZATION_FORMAT,
        "status": "AUTHORIZED_LABELS_STILL_SEALED",
        "opening_id": "a" * 24,
        "protocol": {
            **_binding(
                verifier, root, "protocols/route_a_confirmatory_v1.json"
            ),
            "seal": _binding(verifier, root, verifier.PROTOCOL_SEAL_PATH),
            "final_prelabel_commit": "2" * 40,
            "authoritative_commit": "1" * 40,
            "authoritative_markdown_sha256": "3" * 64,
        },
        "registries": {
            "development": _binding(verifier, root, "data_usgs/station_registry_v1.csv"),
            "external": _binding(verifier, root, "data_usgs/external.csv"),
            "external_lock": _binding(verifier, root, "data_usgs/external.lock.json"),
            "development_panel_spec": _binding(
                verifier, root, "data_usgs/frozen_panel_v1.json"
            ),
            "candidate_table": _binding(verifier, root, "data_usgs/candidates.csv"),
            "candidate_provenance": _binding(
                verifier, root, "data_usgs/candidates.provenance.json"
            ),
            "candidate_snapshot_index": _binding(
                verifier, root, "data_usgs/candidate-raw/snapshot_index.json"
            ),
        },
        "model_suite": _binding(
            verifier, root, "data_usgs/confirmatory_model_suite_v1.json"
        ),
        "development_replay": _binding(
            verifier, root, "outputs/model_replay/route_a_development_replay_v1.json"
        ),
        "prelabel_chronology": {
            **_binding(verifier, root, verifier.CHRONOLOGY_PATH),
            "format": verifier.CHRONOLOGY_FORMAT,
            "status": verifier.CHRONOLOGY_STATUS,
            "order": chronology_order,
            "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        },
        "inference_amendment": {
            **_binding(verifier, root, amendment_path),
            "format": amendment["format"],
            "amendment_id": amendment["amendment_id"],
            "seal": _binding(verifier, root, amendment_seal_path),
            "final_prelabel_commit": amendment_commit,
        },
        "inference_gate": {
            **_binding(verifier, root, gate_path),
            "format": inference_gate["format"],
            "status": inference_gate["status"],
            "claim_eligible": inference_gate["claim_eligible"],
            "analysis_mode": inference_gate["analysis_mode"],
            "policy_sha256": inference_gate["policy_sha256"],
        },
        "outcome_qc_policy": {
            **_binding(verifier, root, outcome_qc_policy_path),
            "format": outcome_qc_policy["format"],
            "policy_id": outcome_qc_policy["policy_id"],
            "required": True,
        },
        "temporal_coverage_policy": {
            **_binding(verifier, root, coverage_policy_path),
            "format": coverage_policy["format"],
            "policy_id": coverage_policy["policy_id"],
            "status": coverage_policy["status"],
            "required": True,
        },
        "acquisition_plan": {
            "history_start": "2020-11-30",
            "target_start": "2021-01-01",
            "target_end": "2023-12-31",
            "nwis_parameter_codes": ["00010", "00060", "00065"],
            "nwis_statistic_code": "00003",
            "request_partition": "one frozen site_no for the complete interval",
            "no_outcome_based_site_replacement": True,
            "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
            "canonical_endpoint": "https://waterservices.usgs.gov/nwis/dv/",
            "transport": "LIVE_HTTPS_ONLY_NO_PRESEEDED_OUTCOMES",
            "maximum_response_bytes_per_request": (
                verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
            ),
        },
        "actual_inputs": _binding(
            verifier, root, "data_usgs/confirmatory_actual_inputs_v1.json"
        ),
        "runtime": {
            "format": "thermoroute.route-a-runtime.v1",
            "requirements_lock": _binding(verifier, root, "requirements-lock.txt"),
            "hashed_requirements_lock": _binding(
                verifier, root, verifier.REPRODUCIBILITY_LOCK
            ),
            "installed_version_validation": {"status": "fixture"},
            "numerical_runtime_contract": {"fixture": True},
            "runtime_sha256": runtime_sha256,
            "python_executable": {
                "realpath": str(root / "fixture-python"),
                "sha256": "d" * 64,
            },
            "golden_inference_sha256": "e" * 64,
            "formal_numerical_policy": {"status": "fixture"},
            "deterministic_child_policy": {"device": "cpu"},
        },
        "required_models": {
            "temporal": [
                "Persistence",
                "DampedPersistence",
                "Climatology",
                "LightGBM",
                "LSTM",
                "ThermoRoute",
                "DampedPriorOnly",
                "TR-noDynamicPrior",
                "TR-fixedKappa",
                "TR-noRouter",
                "TR-noMoE",
                "TR-noTCN",
                "TR-unbounded",
            ],
            "external": [
                "Persistence",
                "DampedPersistence",
                "Climatology",
                "LightGBM",
                "LSTM",
                "ThermoRoute",
            ],
        },
        "fixed_code": {
            "modules": {"opening": fixed_binding},
            "files": {"opening": fixed_binding},
            "entrypoints": {"trusted_scorer": scorer_binding},
            "sha256": "9" * 64,
        },
        "source": {
            "authorization_path": "data_usgs/confirmatory_opening_authorization_v1.json",
            "git_commit_before_authorization": "0" * 40,
            "source_tree_sha256": source_tree_sha256,
            "source_inventory": source_inventory,
        },
        "state_paths": state,
    }
    authorization["authorization_self_sha256"] = verifier._sha256_json(authorization)
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    authorization_sha = verifier.sha256_file(authorization_path)
    work_order_stable = {
        "format": verifier.ACQUISITION_WORK_ORDER_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_path": authorization["source"]["authorization_path"],
        "authorization_sha256": authorization_sha,
        "source_tree_sha256": authorization["source"]["source_tree_sha256"],
        "runtime_sha256": authorization["runtime"]["runtime_sha256"],
        "fixed_code_sha256": authorization["fixed_code"]["sha256"],
        "acquisition_plan": authorization["acquisition_plan"],
        "state_paths": state,
        "site_registries": {
            "temporal": {
                "sha256": authorization["registries"]["development"]["sha256"],
                "sites": ["01073319"],
            },
            "external": {
                "sha256": authorization["registries"]["external"]["sha256"],
                "sites": ["02000001"],
            },
        },
    }
    work_order = {
        **work_order_stable,
        "work_order_self_sha256": verifier._sha256_json(work_order_stable),
    }
    _write_bytes(root, state["work_order"], json.dumps(work_order).encode())

    trusted_validator = {"sha256": "f" * 64, "implementation": "fixture"}
    preflight = {
        "fixture": True,
        "prelabel_chronology_sha256": authorization["prelabel_chronology"][
            "sha256"
        ],
    }
    intent = {
        "format": verifier.INTENT_FORMAT,
        "status": "OPENING_STARTED_IRREVERSIBLE",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "preflight_attestation_sha256": verifier._sha256_json(preflight),
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "work_order_file_sha256": verifier.sha256_file(root / state["work_order"]),
        "fixed_code_sha256": authorization["fixed_code"]["sha256"],
        "runtime_sha256": authorization["runtime"]["runtime_sha256"],
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
        "trusted_validator": trusted_validator,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
    }
    intent["intent_self_sha256"] = verifier._sha256_json(intent)
    (root / state["intent"]).write_text(json.dumps(intent), encoding="utf-8")

    request_ledger = f"{state['transport_root']}/request_ledger_v1.json"
    attempt_index = f"{state['transport_root']}/transport_attempt_index_v1.json"
    attempts_root = f"{state['transport_root']}/transport_attempts_v1"
    raw_index = state["raw_nwis_snapshot_index"]
    request_specs = []
    for ordinal, (cohort, site) in enumerate(
        (("temporal", "01073319"), ("external", "02000001")), start=1
    ):
        request = {
            "schema_version": 1,
            "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
            "method": "GET",
            "url": verifier._expected_confirmatory_nwis_url(
                site,
                authorization["acquisition_plan"]["history_start"],
                authorization["acquisition_plan"]["target_end"],
            ),
            "headers": {},
        }
        request_specs.append({
            "ordinal": ordinal,
            "cohort": cohort,
            "site_no": site,
            "request": request,
            "request_sha256": hashlib.sha256(
                verifier._canonical_json_bytes(request)
            ).hexdigest(),
        })
    request_ids = [row["request_sha256"] for row in request_specs]
    temporal_request, external_request = request_ids
    ledger_stable = {
        "format": verifier.ACQUISITION_REQUEST_LEDGER_FORMAT,
        "status": "FROZEN_BEFORE_FIRST_HTTPS_REQUEST",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "work_order_file_sha256": verifier.sha256_file(root / state["work_order"]),
        "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
        "maximum_response_bytes_per_request": (
            verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
        "request_order": "temporal_then_external_each_site_no_ascending",
        "request_count": len(request_specs),
        "requests": request_specs,
        "station_or_request_replacement_allowed": False,
    }
    ledger = {
        **ledger_stable,
        "request_ledger_self_sha256": verifier._sha256_json(ledger_stable),
    }
    _write_bytes(root, request_ledger, json.dumps(ledger).encode())
    ledger_sha256 = verifier.sha256_file(root / request_ledger)

    attempt_partitions = [
        {
            "completed_before": [],
            "missing_before": sorted(request_ids),
            "completed_after": [temporal_request],
            "missing_after": [external_request],
            "status": "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING",
            "failure_class": "FIXTURE_PROCESS_INTERRUPTION",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:11:00+00:00",
        },
        {
            "completed_before": [temporal_request],
            "missing_before": [external_request],
            "completed_after": sorted(request_ids),
            "missing_after": [],
            "status": "ALL_LEDGER_TRANSACTIONS_COMPLETE",
            "failure_class": None,
            "started_at": "2026-01-01T00:12:00+00:00",
            "completed_at": "2026-01-01T00:21:00+00:00",
        },
    ]
    attempt_rows = []
    for number, partition in enumerate(attempt_partitions, start=1):
        mode = (
            "INITIAL_OPENING_TRANSPORT"
            if number == 1
            else "RESUME_SAME_OPENING"
        )
        start_stable = {
            "format": verifier.ACQUISITION_ATTEMPT_START_FORMAT,
            "status": "TRANSPORT_ATTEMPT_STARTED",
            "opening_id": authorization["opening_id"],
            "authorization_sha256": authorization_sha,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "request_ledger_sha256": ledger_sha256,
            "attempt_number": number,
            "mode": mode,
            "opening_count": 1,
            "completed_before_attempt_request_sha256": sorted(
                partition["completed_before"]
            ),
            "missing_at_start_request_sha256": sorted(
                partition["missing_before"]
            ),
            "response_replacement_allowed": False,
            "started_at_utc": partition["started_at"],
        }
        start = {
            **start_stable,
            "attempt_start_self_sha256": verifier._sha256_json(start_stable),
        }
        start_relative = f"{attempts_root}/attempt_{number:06d}_start.json"
        _write_bytes(root, start_relative, json.dumps(start).encode())
        result_stable = {
            "format": verifier.ACQUISITION_ATTEMPT_RESULT_FORMAT,
            "status": partition["status"],
            "opening_id": authorization["opening_id"],
            "authorization_sha256": authorization_sha,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "request_ledger_sha256": ledger_sha256,
            "attempt_number": number,
            "attempt_start_sha256": verifier.sha256_file(root / start_relative),
            "opening_count": 1,
            "completed_request_sha256": sorted(partition["completed_after"]),
            "missing_request_sha256": sorted(partition["missing_after"]),
            "failure_class": partition["failure_class"],
            "response_replacement_count": 0,
            "completed_at_utc": partition["completed_at"],
        }
        result = {
            **result_stable,
            "attempt_result_self_sha256": verifier._sha256_json(result_stable),
        }
        result_relative = f"{attempts_root}/attempt_{number:06d}_result.json"
        _write_bytes(root, result_relative, json.dumps(result).encode())
        attempt_rows.append({
            "attempt_number": number,
            "mode": mode,
            "status": partition["status"],
            "start": _binding(verifier, root, start_relative),
            "result": _binding(verifier, root, result_relative),
        })

    series_registry = {"WTEMP": [], "FLOW": [], "WLEVEL": []}
    snapshot_records = []
    request_map_rows = []
    retrieval_times = {
        temporal_request: "2026-01-01T00:10:00+00:00",
        external_request: "2026-01-01T00:20:00+00:00",
    }
    request_attempts = {temporal_request: 1, external_request: 2}
    for spec in request_specs:
        request_sha = spec["request_sha256"]
        transaction_root = (
            f"{state['raw_nwis_root']}/{verifier.CONFIRMATORY_NWIS_PROVIDER}/"
            f"{request_sha}"
        )
        response_relative = f"{transaction_root}/response.bin"
        metadata_relative = f"{transaction_root}/metadata.json"
        payload = f"# fixture NWIS response for {spec['site_no']}\n".encode()
        _write_bytes(root, response_relative, payload)
        metadata = {
            "schema_version": 1,
            "opening_id": authorization["opening_id"],
            "authorization_sha256": authorization_sha,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "request_ledger_sha256": ledger_sha256,
            "attempt_number": request_attempts[request_sha],
            "request": spec["request"],
            "request_sha256": request_sha,
            "retrieved_at_utc": retrieval_times[request_sha],
            "http_status": 200,
            "response_headers": {"Content-Type": "text/plain"},
            "final_url": spec["request"]["url"],
            "byte_count": len(payload),
            "response_sha256": hashlib.sha256(payload).hexdigest(),
            "response_file": "response.bin",
            "maximum_response_bytes_per_request": (
                verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
            ),
        }
        _write_bytes(root, metadata_relative, json.dumps(metadata).encode())
        record = {
            "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
            "request_sha256": request_sha,
            "response_sha256": metadata["response_sha256"],
            "retrieved_at_utc": metadata["retrieved_at_utc"],
            "byte_count": metadata["byte_count"],
            "attempt_number": metadata["attempt_number"],
            "request": spec["request"],
            "metadata_path": (
                f"{verifier.CONFIRMATORY_NWIS_PROVIDER}/{request_sha}/metadata.json"
            ),
            "metadata_sha256": verifier.sha256_file(root / metadata_relative),
            "response_path": (
                f"{verifier.CONFIRMATORY_NWIS_PROVIDER}/{request_sha}/response.bin"
            ),
            "series_registry": series_registry,
        }
        snapshot_records.append(record)
        request_map_rows.append({
            "cohort": spec["cohort"],
            "site_no": spec["site_no"],
            "request_sha256": request_sha,
            "response_sha256": record["response_sha256"],
            "retrieved_at_utc": record["retrieved_at_utc"],
            "byte_count": record["byte_count"],
            "attempt_number": record["attempt_number"],
            "series_registry": series_registry,
        })
    # The live transport producer keeps this staging directory after atomically
    # publishing transactions.  It is intentionally empty and may disappear
    # when the release materializer copies only evidence files.
    (root / state["raw_nwis_root"] / verifier.CONFIRMATORY_NWIS_PROVIDER / ".pending").mkdir(
        parents=True,
        exist_ok=True,
    )
    snapshot_records.sort(key=lambda row: row["request_sha256"])
    _write_bytes(root, raw_index, json.dumps({
        "schema_version": 1,
        "snapshot_count": len(snapshot_records),
        "records": snapshot_records,
    }).encode())
    request_map_rows.sort(key=lambda row: (row["cohort"], row["site_no"]))
    _write_bytes(root, state["acquisition_request_map"], json.dumps({
        "format": verifier.ACQUISITION_REQUEST_MAP_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
        "request_count": len(request_map_rows),
        "requests": request_map_rows,
    }).encode())
    attempt_index_stable = {
        "format": verifier.ACQUISITION_ATTEMPT_INDEX_FORMAT,
        "status": "ALL_LEDGER_TRANSACTIONS_COMPLETE",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "request_ledger": _binding(verifier, root, request_ledger),
        "request_count": len(request_ids),
        "attempt_count": len(attempt_rows),
        "resume_count": 1,
        "opening_count": 1,
        "response_replacement_count": 0,
        "completed_before_final_attempt_request_sha256": [temporal_request],
        "retrieval_span_utc": {
            "first": retrieval_times[temporal_request],
            "last": retrieval_times[external_request],
        },
        "attempts": attempt_rows,
    }
    attempt_index_document = {
        **attempt_index_stable,
        "attempt_index_self_sha256": verifier._sha256_json(attempt_index_stable),
    }
    _write_bytes(root, attempt_index, json.dumps(attempt_index_document).encode())
    outcome_dates = pd.date_range("2020-11-30", "2023-12-31", freq="D")
    for cohort, site in (
        ("temporal", "01073319"),
        ("external", "02000001"),
    ):
        outcome_path = root / state[f"{cohort}_outcomes"]
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "site_no": [site] * len(outcome_dates),
                "DATE": outcome_dates,
                "WTEMP": 10.0 + outcome_dates.dayofyear / 1000.0,
                "WTEMP_value_status": ["RETAINED_FINITE_VALUE"]
                * len(outcome_dates),
            }
        ).to_parquet(outcome_path, index=False)
    acquisition = {
        "format": verifier.ACQUISITION_MANIFEST_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "protocol_sha256": authorization["protocol"]["sha256"],
        "labels_state": "OPENED_ONCE",
        "site_replacement_count": 0,
        "response_replacement_count": 0,
        "history_start": authorization["acquisition_plan"]["history_start"],
        "target_start": authorization["acquisition_plan"]["target_start"],
        "target_end": authorization["acquisition_plan"]["target_end"],
        "maximum_response_bytes_per_request": (
            verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
        "transport_summary": {
            "opening_count": 1,
            "attempt_count": 2,
            "resume_count": 1,
            "completed_before_final_attempt_request_sha256": [temporal_request],
            "retrieval_span_utc": {
                "first": retrieval_times[temporal_request],
                "last": retrieval_times[external_request],
            },
        },
        "request_ledger": _binding(verifier, root, request_ledger),
        "transport_attempt_index": _binding(verifier, root, attempt_index),
        "raw_nwis_snapshot_index": _binding(verifier, root, raw_index),
        "request_map": _binding(verifier, root, state["acquisition_request_map"]),
        "normalized_outcome_tables": {
            "temporal": _binding(verifier, root, state["temporal_outcomes"]),
            "external": _binding(verifier, root, state["external_outcomes"]),
        },
        "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
    }
    (root / state["acquisition_manifest"]).write_text(
        json.dumps(acquisition), encoding="utf-8"
    )
    availability_rows: list[dict[str, object]] = []
    for cohort, site in (
        ("temporal", "01073319"),
        ("external", "02000001"),
    ):
        rows: list[dict[str, object]] = []
        for model_index, model in enumerate(
            authorization["required_models"][cohort]
        ):
            for horizon in (1, 3, 7):
                issue_dates = pd.date_range(
                    "2021-01-01",
                    pd.Timestamp("2023-12-31") - pd.Timedelta(days=horizon),
                    freq="D",
                )
                for issue in issue_dates:
                    target = issue + pd.Timedelta(days=horizon)
                    truth = 10.0 + target.dayofyear / 1000.0
                    rows.append(
                        {
                            "model": model,
                            "site_id": site,
                            "horizon": horizon,
                            "issue_date": issue,
                            "target_date": target,
                            "y_true": float(truth),
                            "y_pred": float(truth + (model_index + 1) / 100.0),
                        }
                    )
        frame = pd.DataFrame.from_records(rows).sort_values(
            ["model", "site_id", "horizon", "issue_date", "target_date"],
            kind="mergesort",
        ).reset_index(drop=True)
        prediction_path = root / state[f"{cohort}_predictions"]
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(prediction_path, index=False)
        for horizon in (1, 3, 7):
            count = len(
                pd.date_range(
                    "2021-01-01",
                    pd.Timestamp("2023-12-31") - pd.Timedelta(days=horizon),
                    freq="D",
                )
            )
            availability_rows.append(
                {
                    "cohort": cohort,
                    "site_no": site,
                    "horizon": horizon,
                    "n_valid_targets": count,
                    "reportable": True,
                }
            )
    availability_path = root / state["availability_registry"]
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(availability_rows).to_csv(
        availability_path, index=False, lineterminator="\n"
    )
    _write_bytes(root, state["outcome_quality_audit"], b"{}\n")
    prediction_rows = []
    models_by_horizon = {
        1: ("ThermoRoute", "DampedPersistence"),
        3: ("ThermoRoute", "DampedPersistence", "LightGBM"),
        7: ("ThermoRoute", "DampedPersistence", "LightGBM"),
    }
    for horizon, models in models_by_horizon.items():
        for offset in range(3):
            issue = pd.Timestamp("2021-02-01") + pd.Timedelta(days=offset)
            for model in models:
                prediction_rows.append({
                    "model": model,
                    "site_id": "01073319",
                    "horizon": horizon,
                    "issue_date": issue,
                    "target_date": issue + pd.Timedelta(days=horizon),
                    "y_true": 10.0,
                    "y_pred": 10.0 if model == "ThermoRoute" else 11.0,
                })
    outcome_predictions = pd.DataFrame.from_records(prediction_rows)
    normalized_temporal = pd.DataFrame({
        "site_no": ["01073319"] * 3,
        "DATE": pd.date_range("2021-02-01", periods=3, freq="D"),
        "WTEMP": [10.0, 11.0, 12.0],
    })
    spatial_sensitivity = {
        "comparisons": [
            {
                "test_id": row["test_id"],
                "station_weighted_median_effect_c": -1.0,
                "leave_one_huc": [{
                    "held_out_huc2": "01",
                    "effect_minus_margin_c": -1.0 - float(row["margin_c"]),
                }],
            }
            for row in family
        ]
    }
    outcome_qc_module = verifier._load_canonical_outcome_qc_module(root)
    outcome_qc_gate = outcome_qc_module.build_outcome_qc_gate_document(
        root=root,
        policy_path=root / outcome_qc_policy_path,
        protocol=json.loads(
            (root / "protocols/route_a_confirmatory_v1.json").read_text(
                encoding="utf-8"
            )
        ),
        temporal_predictions=outcome_predictions,
        normalized_temporal=normalized_temporal,
        spatial_sensitivity=spatial_sensitivity,
        minimum_targets=2,
    )
    _write_bytes(
        root, state["outcome_qc_gate"], json.dumps(outcome_qc_gate).encode()
    )
    _write_bytes(root, state["approved_target_sensitivity"], b"{}\n")
    _write_bytes(root, state["spatial_sensitivity"], b"{}\n")
    _write_bytes(root, state["probabilistic_evaluation"], b"{}\n")
    tests = [
        {
            "test_id": row["test_id"],
            "status": "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS",
            "median_effect_c": None,
            "n_stations": 1,
            "n_clusters": 1,
            "reject_at_0_05": False,
            "confidence_bound_supports_margin": False,
        }
        for row in family
    ]
    (root / state["statistics"]).write_text(json.dumps({
        "format": verifier.STATISTICS_FORMAT,
        "tests": tests,
        "outcome_qc_gate": {
            **_binding(verifier, root, state["outcome_qc_gate"]),
            "format": outcome_qc_gate["format"],
            "status": outcome_qc_gate["status"],
            "pass": True,
            "directional_claims_allowed": True,
        },
    }), encoding="utf-8")
    coverage_module = verifier._load_canonical_coverage_bridge_module(root)
    temporal_coverage_audit = (
        coverage_module.replay_temporal_coverage_from_physical_files(
            root=root,
            authorization=authorization,
        )
    )
    _write_bytes(
        root,
        state["temporal_coverage_audit"],
        json.dumps(
            temporal_coverage_audit,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n",
    )
    _write_bytes(root, state["report"], b"# trusted report\n")
    receipt_artifacts = {
        "acquisition_manifest": _binding(verifier, root, state["acquisition_manifest"]),
        "raw_nwis_snapshot_index": _binding(verifier, root, raw_index),
        "acquisition_request_map": _binding(verifier, root, state["acquisition_request_map"]),
        "temporal_normalized_outcomes": _binding(verifier, root, state["temporal_outcomes"]),
        "external_normalized_outcomes": _binding(verifier, root, state["external_outcomes"]),
        "availability_registry": _binding(verifier, root, state["availability_registry"]),
        "outcome_quality_audit": _binding(verifier, root, state["outcome_quality_audit"]),
        "outcome_qc_gate": _binding(verifier, root, state["outcome_qc_gate"]),
        "approved_target_sensitivity": _binding(
            verifier, root, state["approved_target_sensitivity"]
        ),
        "spatial_sensitivity": _binding(verifier, root, state["spatial_sensitivity"]),
        "probabilistic_evaluation": _binding(
            verifier, root, state["probabilistic_evaluation"]
        ),
        "temporal_predictions": _binding(verifier, root, state["temporal_predictions"]),
        "external_predictions": _binding(verifier, root, state["external_predictions"]),
        "statistics": _binding(verifier, root, state["statistics"]),
        "temporal_coverage_audit": _binding(
            verifier, root, state["temporal_coverage_audit"]
        ),
        "report": _binding(verifier, root, state["report"]),
    }
    release_artifacts = {
        key: {
            "format": (
                verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT
                if key == "temporal_coverage_audit"
                else "fixture-format"
            ),
            **binding,
        }
        for key, binding in receipt_artifacts.items()
    }
    release_bindings = {
        "format": "thermoroute.route-a-release-bindings.v1",
        "opening_id": authorization["opening_id"],
        "state_namespace": state["namespace"],
        "authorization": {
            "format": verifier.AUTHORIZATION_FORMAT,
            "path": authorization["source"]["authorization_path"],
            "sha256": authorization_sha,
        },
        "artifacts": release_artifacts,
        "receipt": {
            "format": verifier.RECEIPT_FORMAT,
            "path": state["receipt"],
            "external_sha256_path": state["receipt_sha256"],
        },
    }
    receipt = {
        "format": verifier.RECEIPT_FORMAT,
        "status": "OPENED_AND_SCORED_ONCE",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "intent_sha256": verifier.sha256_file(root / state["intent"]),
        "work_order_sha256": verifier.sha256_file(root / state["work_order"]),
        "preflight_attestation": preflight,
        "preflight_attestation_sha256": verifier._sha256_json(preflight),
        "fixed_code": authorization["fixed_code"],
        "authorized_runtime": authorization["runtime"],
        "completion_environment": {
            "numerical_runtime_sha256": authorization["runtime"]["runtime_sha256"]
        },
        "python_hash_seed_interpreter_effect": (
            "present_but_ignored_under_isolated_mode"
        ),
        "completed_at_utc": "2026-01-01T00:30:00+00:00",
        "opening_count": 1,
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
        "transport_recovery": acquisition["transport_summary"],
        "all_predeclared_models_reported": True,
        "reported_models": {
            cohort: sorted(values)
            for cohort, values in authorization["required_models"].items()
        },
        "trusted_validator": trusted_validator,
        "artifacts": receipt_artifacts,
        "trusted_prediction_hashes": {
            cohort: {
                "rows": 1,
                "sha256": digest,
                "schema": "thermoroute.predictions.v1",
                "ensemble_rule": (
                    "mean frozen members, then frozen CQR and horizon Platt "
                    "calibration"
                ),
            }
            for cohort, digest in (
                ("temporal", "a" * 64),
                ("external", "b" * 64),
            )
        },
        "formal_tests": tests,
        "temporal_coverage_audit": {
            **receipt_artifacts["temporal_coverage_audit"],
            "format": verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT,
            "core_status": verifier.TEMPORAL_COVERAGE_CORE_STATUS,
            "physical_replay_verified": True,
            "source_binding_count": 11,
        },
        "state_paths": state,
        "release_bindings": release_bindings,
        "intent_self_sha256": intent["intent_self_sha256"],
        "security_boundary": (
            "misoperation/replay guard for an honest filesystem owner; not a "
            "defense against an owner who can replace the interpreter or files"
        ),
    }
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    (root / state["receipt"]).write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha = verifier.sha256_file(root / state["receipt"])
    _write_bytes(
        root,
        state["receipt_sha256"],
        f"{receipt_sha}  opening_receipt_v1.json\n".encode(),
    )
    representatives = {
        "canonical_development": "data_usgs/panel_usgs_120v2.parquet",
        "authorization": "protocols/route_a_confirmatory_v1.json",
        "inference_gates": gate_path,
        "registries": "data_usgs/external.csv",
        "candidate_evidence": "data_usgs/candidates.csv",
        "model_suite": "data_usgs/confirmatory_model_suite_v1.json",
        "model_bundles": "outputs/models/temporal/LSTM.bundle",
        "prelabel_chronology": verifier.CHRONOLOGY_PATH,
        "prelabel_inputs": "data_usgs/prelabel/temporal.parquet",
        "raw_meteorology": "data_usgs/raw_snapshots/met-0/response.bin",
        "opening_intent": state["intent"],
        "raw_nwis": (
            f"{state['raw_nwis_root']}/{verifier.CONFIRMATORY_NWIS_PROVIDER}/"
            f"{temporal_request}/response.bin"
        ),
        "normalized_outcomes": state["temporal_outcomes"],
        "trusted_predictions": state["temporal_predictions"],
        "availability": state["availability_registry"],
        "sensitivity_audits": state["outcome_quality_audit"],
        "outcome_qc": state["outcome_qc_gate"],
        "probabilistic_evaluation": state["probabilistic_evaluation"],
        "statistics": state["statistics"],
        "temporal_coverage": state["temporal_coverage_audit"],
        "report": state["report"],
        "receipt": state["receipt_sha256"],
        "environment_attestations": "requirements-lock.txt",
        "reproducibility_lock": verifier.REPRODUCIBILITY_LOCK,
    }
    return authorization_path, representatives


def _refresh_postopen_coverage_evidence(
    verifier, root: Path, authorization_path: Path
) -> None:
    """Rebuild the fixture audit after a deliberately self-consistent mutation."""
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    coverage_module = verifier._load_canonical_coverage_bridge_module(root)
    audit = coverage_module.replay_temporal_coverage_from_physical_files(
        root=root,
        authorization=authorization,
    )
    audit_path = root / state["temporal_coverage_audit"]
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path = root / state["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    binding = _binding(verifier, root, state["temporal_coverage_audit"])
    receipt["artifacts"]["temporal_coverage_audit"] = binding
    released = receipt["release_bindings"]["artifacts"][
        "temporal_coverage_audit"
    ]
    released.update(binding)
    receipt["temporal_coverage_audit"] = {
        **binding,
        "format": verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT,
        "core_status": verifier.TEMPORAL_COVERAGE_CORE_STATUS,
        "physical_replay_verified": True,
        "source_binding_count": 11,
    }
    receipt.pop("receipt_self_sha256", None)
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_digest = verifier.sha256_file(receipt_path)
    (root / state["receipt_sha256"]).write_text(
        f"{receipt_digest}  opening_receipt_v1.json\n", encoding="utf-8"
    )


def _refresh_postopen_transport_evidence(
    verifier, root: Path, authorization_path: Path
) -> None:
    """Rebind a deliberately mutated transport chain through its outer receipt."""
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    acquisition_path = root / state["acquisition_manifest"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))

    ledger_relative = acquisition["request_ledger"]["path"]
    acquisition["request_ledger"] = _binding(verifier, root, ledger_relative)
    for key in ("raw_nwis_snapshot_index", "request_map"):
        relative = acquisition[key]["path"]
        acquisition[key] = _binding(verifier, root, relative)
    index_relative = acquisition["transport_attempt_index"]["path"]
    index_path = root / index_relative
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for row in index.get("attempts", []):
        for key in ("start", "result"):
            binding = row.get(key)
            if isinstance(binding, dict):
                row[key] = _binding(verifier, root, binding["path"])
    index.pop("attempt_index_self_sha256", None)
    index["attempt_index_self_sha256"] = verifier._sha256_json(index)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    acquisition["transport_attempt_index"] = _binding(
        verifier, root, index_relative
    )
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    receipt_path = root / state["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    acquisition_binding = _binding(
        verifier, root, state["acquisition_manifest"]
    )
    receipt["artifacts"]["acquisition_manifest"] = acquisition_binding
    receipt["release_bindings"]["artifacts"]["acquisition_manifest"].update(
        acquisition_binding
    )
    for acquisition_key, artifact_key in (
        ("raw_nwis_snapshot_index", "raw_nwis_snapshot_index"),
        ("request_map", "acquisition_request_map"),
    ):
        binding = dict(acquisition[acquisition_key])
        receipt["artifacts"][artifact_key] = binding
        receipt["release_bindings"]["artifacts"][artifact_key].update(binding)
    receipt["transport_recovery"] = acquisition.get("transport_summary")
    _reseal_postopen_fixture_receipt(verifier, root, state, receipt)
    _refresh_postopen_coverage_evidence(verifier, root, authorization_path)


def _reseal_postopen_fixture_receipt(
    verifier, root: Path, state: dict[str, str], receipt: dict[str, object]
) -> None:
    receipt.pop("receipt_self_sha256", None)
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path = root / state["receipt"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (root / state["receipt_sha256"]).write_text(
        f"{verifier.sha256_file(receipt_path)}  opening_receipt_v1.json\n",
        encoding="utf-8",
    )


def _validate_fixture_inference_closure(
    verifier, root: Path, authorization_path: Path
) -> dict[str, object]:
    """Rebind a mutated amendment and exercise its independent release check."""
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    amendment_binding = authorization["inference_amendment"]
    seal_path = root / amendment_binding["seal"]["path"]
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["amendment"] = _binding(verifier, root, amendment_binding["path"])
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    amendment_binding.update(_binding(verifier, root, amendment_binding["path"]))
    amendment_binding["seal"] = _binding(
        verifier, root, amendment_binding["seal"]["path"]
    )

    protocol_binding = authorization["protocol"]
    protocol_document = json.loads(
        (root / protocol_binding["path"]).read_text(encoding="utf-8")
    )
    outcome_qc_policy = json.loads(
        (root / authorization["outcome_qc_policy"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    temporal_coverage_policy = json.loads(
        (root / authorization["temporal_coverage_policy"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return verifier._validate_inference_closure(
        root,
        {},
        authorization,
        protocol_binding=protocol_binding,
        protocol_document=protocol_document,
        protocol_seal_path=root / protocol_binding["seal"]["path"],
        outcome_qc_policy=outcome_qc_policy,
        temporal_coverage_policy=temporal_coverage_policy,
    )


def _manifest_command(root: Path, manifest: Path, *extra: str):
    return [
        sys.executable,
        str(MANIFEST_SCRIPT),
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--no-git",
        *extra,
    ]


def test_manifest_binds_revision_source_config_data_and_detects_change(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    commit, tree = "a" * 40, "b" * 40
    subprocess.run(
        _manifest_command(
            tmp_path,
            manifest_path,
            "--source-git-commit",
            commit,
            "--source-git-tree",
            tree,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "thermoroute.provenance-manifest.v2"
    assert document["git"]["commit"] == commit
    assert {"@git", "@source", "@config", "@dependencies"} <= set(document["dag"])
    assert set(document["dag"]["outputs/tables/result.csv"]["parents"]) >= {
        "@git", "@source", "@config", "@dependencies", "data/input.csv",
    }

    subprocess.run(
        _manifest_command(tmp_path, manifest_path, "--check"),
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "data" / "input.csv").write_text("x\n2\n", encoding="utf-8")
    changed = subprocess.run(
        _manifest_command(tmp_path, manifest_path, "--check"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert changed.returncode == 1
    assert "CHANGED data/input.csv" in changed.stderr


def test_release_boundary_requires_contract_and_rejects_traversal(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_release_test")
    complete = set(verifier.REQUIRED_MEMBERS) | {
        "paper/main.tex", "outputs/reports/report.md",
    }
    verifier.validate_members(complete)
    with pytest.raises(ValueError, match="missing required members"):
        verifier.validate_members(complete - {"data/b1.csv"})
    with pytest.raises(ValueError, match="mixed-generation"):
        verifier.validate_members(
            complete | {"outputs/tables/usgs_stations_with_huc.csv"}
        )

    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("thermoroute/../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe archive path"):
            verifier.normalised_members(archive)


def test_archive_resource_limits_are_checked_before_extraction(tmp_path, monkeypatch):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_zip_limits_test")

    def archive_with(payloads: list[bytes]) -> Path:
        path = tmp_path / f"fixture-{len(list(tmp_path.glob('fixture-*.zip')))}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, payload in enumerate(payloads):
                info = zipfile.ZipInfo(f"thermoroute/data/{index}.bin")
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(info, payload)
        return path

    member_limited = archive_with([b"a", b"b"])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(ValueError, match="member count"):
        verifier._preflight_zip_container(member_limited)
    with zipfile.ZipFile(member_limited) as archive:
        with pytest.raises(ValueError, match="member count"):
            verifier.normalised_members(archive)

    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBERS", 10)
    member_bytes = archive_with([b"four"])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBER_BYTES", 3)
    with zipfile.ZipFile(member_bytes) as archive:
        with pytest.raises(ValueError, match="uncompressed safety limit"):
            verifier.normalised_members(archive)

    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBER_BYTES", 10)
    total_bytes = archive_with([b"abc", b"def"])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_TOTAL_BYTES", 5)
    with zipfile.ZipFile(total_bytes) as archive:
        with pytest.raises(ValueError, match="total uncompressed"):
            verifier.normalised_members(archive)

    monkeypatch.setattr(verifier, "MAX_ARCHIVE_TOTAL_BYTES", 10_000)
    compressed = archive_with([b"0" * 4096])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBER_BYTES", 10_000)
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_COMPRESSION_RATIO", 2)
    with zipfile.ZipFile(compressed) as archive:
        with pytest.raises(ValueError, match="compression ratio"):
            verifier.normalised_members(archive)


def test_archive_python_is_not_executed_before_git_source_binding(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_preexec_gate_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    _minimal_canonical_release(verifier, source)
    verifier.materialize_release_profile(source, stage, verifier.PREOPEN_PROFILE)
    sentinel = stage / "ARCHIVE_PYTHON_EXECUTED"
    _write_bytes(
        stage,
        "scripts/26_validate_claims.py",
        (
            "from pathlib import Path\n"
            "Path('ARCHIVE_PYTHON_EXECUTED').write_text('bad', encoding='utf-8')\n"
        ).encode(),
    )
    registry_path = _write_bytes(
        stage,
        "protocols/route_a_claim_registry_v1.json",
        json.dumps({"documents": ["paper/main.md"]}).encode(),
    )
    paper = _write_bytes(stage, "paper/main.md", b"preopen language\n")
    validator = stage / "scripts/26_validate_claims.py"
    with pytest.raises(ValueError, match="Git/preregistration evidence"):
        verifier.materialize_claim_audit(stage, verifier.PREOPEN_PROFILE)
    assert not sentinel.exists()
    audit = {
        "format": "thermoroute.route-a-release-claim-audit.v1",
        "profile": verifier.PREOPEN_PROFILE,
        "require_complete": False,
        "validator": verifier._binding_for(stage, validator),
        "registry": verifier._binding_for(stage, registry_path),
        "scanned_documents": [verifier._binding_for(stage, paper)],
        "violation_count": 0,
    }
    audit_path = _write_bytes(
        stage,
        verifier.CLAIM_AUDIT_PATH,
        json.dumps(audit, sort_keys=True).encode() + b"\n",
    )
    marker_path = stage / verifier.PROFILE_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["claim_validation"] = verifier._binding_for(stage, audit_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="Git/preregistration evidence"):
        verifier.verify_release_profile(stage, run_trusted_replay=True)
    assert not sentinel.exists()


def test_git_path_lifetime_walk_sees_add_delete_on_merged_side_branch(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_path_lifetime_test")
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })

    def commit(message: str) -> str:
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=root,
            env=environment,
            check=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    _write_bytes(root, "base.txt")
    commit("base")
    main_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    subprocess.run(["git", "switch", "-q", "-c", "side"], cwd=root, check=True)
    receipt_relative = "outputs/prelabel/route_a_prelabel_chronology_v1.json"
    _write_bytes(root, receipt_relative, b"side birth\n")
    side_birth = commit("side receipt birth")
    (root / receipt_relative).unlink()
    commit("side receipt deletion")
    subprocess.run(["git", "switch", "-q", main_branch], cwd=root, check=True)
    _write_bytes(root, receipt_relative, b"canonical birth\n")
    main_birth = commit("canonical receipt birth")
    subprocess.run(
        ["git", "merge", "-q", "--no-ff", "side", "-m", "merge side"],
        cwd=root,
        env=environment,
        check=True,
    )
    tip = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()

    assert set(verifier._git_path_creation_commits(root, tip, receipt_relative)) == {
        side_birth,
        main_birth,
    }


def test_release_replay_accepts_one_strict_immutable_seal_birth(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_seal_lineage_valid_test"
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _write_bytes(root, "base.txt")
    _commit_git_fixture(root, "base")
    relative = "protocols/route_a_inference_amendment_seal_v1.json"
    _write_bytes(root, "protocols/route_a_inference_amendment_v1.json", b"{}\n")
    amendment_commit = _commit_git_fixture(root, "amendment")
    payload = b'{"seal":"canonical"}\n'
    _write_bytes(root, relative, payload)
    creation = _commit_git_fixture(root, "seal")

    assert verifier._verify_unique_immutable_path_creation(
        root,
        tip="HEAD",
        predecessor=amendment_commit,
        relative=relative,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        label="inference amendment seal",
    ) == creation


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("same_commit", "existed at its required predecessor commit"),
        ("preexisting", "existed at its required predecessor commit"),
        ("add_delete_readd", "exactly one reachable Git creation"),
        ("post_create_modify", "deleted or changed after creation"),
    ),
)
def test_release_replay_rejects_adversarial_seal_histories(
    tmp_path, attack, error,
):
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_seal_lineage_{attack}_test"
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _write_bytes(root, "base.txt")
    _commit_git_fixture(root, "base")
    amendment = "protocols/route_a_inference_amendment_v1.json"
    relative = "protocols/route_a_inference_amendment_seal_v1.json"
    payload = b'{"seal":"canonical"}\n'

    if attack == "same_commit":
        _write_bytes(root, amendment, b"{}\n")
        _write_bytes(root, relative, payload)
        amendment_commit = _commit_git_fixture(root, "amendment and seal")
    elif attack == "preexisting":
        _write_bytes(root, relative, payload)
        _commit_git_fixture(root, "premature seal")
        _write_bytes(root, amendment, b"{}\n")
        amendment_commit = _commit_git_fixture(root, "later amendment")
    else:
        _write_bytes(root, amendment, b"{}\n")
        amendment_commit = _commit_git_fixture(root, "amendment")
        _write_bytes(root, relative, payload)
        _commit_git_fixture(root, "seal")
        if attack == "add_delete_readd":
            (root / relative).unlink()
            _commit_git_fixture(root, "delete seal")
            _write_bytes(root, relative, payload)
            _commit_git_fixture(root, "re-add seal")
        else:
            _write_bytes(root, relative, b'{"seal":"changed"}\n')
            _commit_git_fixture(root, "modify seal")
            _write_bytes(root, relative, payload)
            _commit_git_fixture(root, "restore seal")

    with pytest.raises(ValueError, match=error):
        verifier._verify_unique_immutable_path_creation(
            root,
            tip="HEAD",
            predecessor=amendment_commit,
            relative=relative,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            label="inference amendment seal",
        )


def test_manifest_refuses_unsealed_canonical_stage09_current_truth(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    prediction = (
        tmp_path
        / "outputs"
        / "predictions"
        / "usgs_predictions_stage9_v2.parquet"
    )
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"unsealed-stage09-bytes")
    scores = tmp_path / "outputs" / "tables" / "usgs_scores.csv"
    scores.parent.mkdir(parents=True, exist_ok=True)
    scores.write_text("horizon,site\n", encoding="utf-8")
    result = subprocess.run(
        _manifest_command(tmp_path, manifest_path),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "USGS_CURRENT_TRUTH_STALE" in result.stderr


def test_huc_verifier_replays_derived_rows_from_raw_nwis(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_huc_replay_test")
    source = ROOT / "data_usgs"
    target = tmp_path / "data_usgs"
    target.mkdir()
    for name in (
        "frozen_panel_v1.json",
        "station_registry_v1.csv",
        "huc_metadata_usgs_v1.csv",
        "huc_metadata_usgs_v1.provenance.json",
    ):
        shutil.copy2(source / name, target / name)
    shutil.copytree(
        source / "raw_snapshots" / "huc-v1",
        target / "raw_snapshots" / "huc-v1",
    )
    verifier.verify_canonical_huc_closure(tmp_path)

    # Forge a self-consistent registry/derived table/spec while retaining the
    # actual immutable NWIS response.  Digest-only closure would accept this;
    # raw replay must reject it.
    registry = (target / "station_registry_v1.csv").read_text(encoding="utf-8")
    registry = registry.replace(",1060003,1,55.7,", ",99060003,99,55.7,", 1)
    (target / "station_registry_v1.csv").write_text(registry, encoding="utf-8")
    huc = (target / "huc_metadata_usgs_v1.csv").read_text(encoding="utf-8")
    huc = huc.replace(",01060003,01,55.7", ",99060003,99,55.7", 1)
    (target / "huc_metadata_usgs_v1.csv").write_text(huc, encoding="utf-8")
    provenance_path = target / "huc_metadata_usgs_v1.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["derived_csv_sha256"] = verifier.sha256_file(
        target / "huc_metadata_usgs_v1.csv"
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    spec_path = target / "frozen_panel_v1.json"
    panel_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    station = panel_spec["station_registry"]
    station["sha256"] = verifier.sha256_file(target / "station_registry_v1.csv")
    station["huc_metadata"]["source_sha256"] = verifier.sha256_file(
        target / "huc_metadata_usgs_v1.csv"
    )
    station["huc_metadata"]["provenance_sha256"] = verifier.sha256_file(
        provenance_path
    )
    spec_path.write_text(json.dumps(panel_spec), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be replayed from raw NWIS"):
        verifier.verify_canonical_huc_closure(tmp_path)


def test_preopen_profile_is_explicit_and_rejects_any_result_or_label_path(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_preopen_profile_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    _minimal_canonical_release(verifier, source)
    document = verifier.materialize_release_profile(
        source, stage, verifier.PREOPEN_PROFILE
    )
    _materialize_claim_fixture(verifier, stage, verifier.PREOPEN_PROFILE)
    assert document["profile"] == "PREOPEN_NOT_COMPLETE"
    assert document["confirmatory_scoring_completed"] is False
    assert document["directional_claims_allowed"] is False
    assert document["supported_test_ids"] == []
    assert document["supports_route_a_confirmatory_conclusions"] is False
    assert document["labels_included"] is False
    assert "cannot support" in document["warning"]
    assert verifier.verify_release_profile(
        stage, run_trusted_replay=False
    ) == verifier.PREOPEN_PROFILE

    forbidden = _write_bytes(
        stage, "outputs/confirmatory/route_a_fake/trusted/statistics_v1.json", b"{}\n"
    )
    with pytest.raises(ValueError, match="confirmation/label/result"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    forbidden.unlink()
    labels = _write_bytes(stage, "data_usgs/labels/post2020.parquet")
    with pytest.raises(ValueError, match="confirmation/label/result"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    labels.unlink()


def test_postopen_profile_closes_every_required_category_and_missing_file_fails(
    tmp_path, monkeypatch
):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_postopen_profile_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    authorization, representatives = _write_postopen_fixture(verifier, source)
    document = verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization,
    )
    _materialize_claim_fixture(verifier, stage, verifier.POSTOPEN_PROFILE)
    assert document["profile"] == "ROUTE_A_OPENED_COMPLETE"
    assert document["confirmatory_scoring_completed"] is True
    assert document["directional_claims_allowed"] is False
    assert document["inference_gate_claim_eligible"] is False
    assert document[
        "gross_plausibility_and_aggregate_sensitivity_gate_passed"
    ] is True
    assert document["supported_test_ids"] == []
    assert document["supports_route_a_confirmatory_conclusions"] is False
    assert set(document["artifact_closure"]) == verifier.REQUIRED_POSTOPEN_CATEGORIES
    replay_calls = []
    monkeypatch.setattr(
        verifier,
        "_verify_git_history_evidence",
        lambda *_args, **_kwargs: replay_calls.append("git"),
    )
    monkeypatch.setattr(
        verifier,
        "_verify_claim_audit",
        lambda *_args, **kwargs: replay_calls.append(
            f"claims:{kwargs['execute_validator']}"
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_run_trusted_replay",
        lambda *_args, **_kwargs: replay_calls.append("trusted"),
    )
    assert verifier.verify_release_profile(
        stage, run_trusted_replay=True
    ) == verifier.POSTOPEN_PROFILE
    assert replay_calls == ["git", "claims:True", "trusted"]
    marker_path = stage / verifier.PROFILE_MARKER
    marker_bytes = marker_path.read_bytes()
    overstated = json.loads(marker_bytes)
    overstated["supports_route_a_confirmatory_conclusions"] = True
    overstated["directional_claims_allowed"] = True
    overstated["supported_test_ids"] = [
        _fixture_confirmatory_family()[0]["test_id"]
    ]
    marker_path.write_text(json.dumps(overstated), encoding="utf-8")
    with pytest.raises(ValueError, match="claim status"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    for category, relative in representatives.items():
        artifact = stage / relative
        payload = artifact.read_bytes()
        artifact.unlink()
        with pytest.raises(
            ValueError,
            match="absent|closure|missing|lacks|cannot read|identity|transport",
        ):
            verifier.verify_release_profile(stage, run_trusted_replay=False)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(payload)
        assert category in document["artifact_closure"]
    verifier.verify_release_profile(stage, run_trusted_replay=False)
    stale = _write_bytes(stage, "outputs/tables/usgs_scores_old_cohort.csv")
    with pytest.raises(ValueError, match="outside the authorization closure"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    stale.unlink()


def test_postopen_coverage_replays_even_when_full_trusted_replay_is_disabled(
    tmp_path, monkeypatch
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_mandatory_coverage_replay_test"
    )
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    authorization, _ = _write_postopen_fixture(verifier, source)
    verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization,
    )
    _materialize_claim_fixture(verifier, stage, verifier.POSTOPEN_PROFILE)

    real_module = verifier._load_canonical_coverage_bridge_module(stage)
    calls: list[str] = []

    class ReplayProxy:
        @staticmethod
        def replay_temporal_coverage_from_physical_files(**kwargs):
            calls.append("coverage")
            return real_module.replay_temporal_coverage_from_physical_files(
                **kwargs
            )

    monkeypatch.setattr(
        verifier,
        "_verify_git_history_evidence",
        lambda *_args, **_kwargs: calls.append("git"),
    )
    monkeypatch.setattr(
        verifier,
        "_load_canonical_coverage_bridge_module",
        lambda _root: ReplayProxy,
    )
    monkeypatch.setattr(
        verifier, "_verify_claim_audit", lambda *_args, **_kwargs: None
    )

    def forbidden_trusted_replay(*_args, **_kwargs):
        raise AssertionError("full trusted replay must remain disabled")

    monkeypatch.setattr(verifier, "_run_trusted_replay", forbidden_trusted_replay)
    assert (
        verifier.verify_release_profile(stage, run_trusted_replay=False)
        == verifier.POSTOPEN_PROFILE
    )
    assert calls == ["git", "coverage"]


def test_postopen_archive_code_cannot_execute_before_git_identity_check(
    tmp_path, monkeypatch
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_postopen_preexec_gate_test"
    )
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    authorization, _ = _write_postopen_fixture(verifier, source)
    verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization,
    )
    _materialize_claim_fixture(verifier, stage, verifier.POSTOPEN_PROFILE)
    sentinel = tmp_path / "ARCHIVE_COVERAGE_PYTHON_EXECUTED"
    (stage / "src/thermoroute/coverage_bridge.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('bad', encoding='utf-8')\n",
        encoding="utf-8",
    )

    def reject_git(*_args, **_kwargs):
        raise ValueError("Git identity rejected before archive import")

    monkeypatch.setattr(verifier, "_verify_git_history_evidence", reject_git)
    with pytest.raises(ValueError, match="before archive import"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    assert not sentinel.exists()


def test_postopen_authorization_requires_exact_coverage_state_registry(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_coverage_state_registry_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    for attack in ("missing", "extra"):
        attacked = tmp_path / attack
        shutil.copytree(source, attacked)
        path = attacked / authorization_path.relative_to(source)
        authorization = json.loads(path.read_text(encoding="utf-8"))
        if attack == "missing":
            authorization["state_paths"].pop("temporal_coverage_audit")
        else:
            authorization["state_paths"]["coverage_alias"] = (
                authorization["state_paths"]["temporal_coverage_audit"]
            )
        authorization.pop("authorization_self_sha256")
        authorization["authorization_self_sha256"] = verifier._sha256_json(
            authorization
        )
        path.write_text(json.dumps(authorization), encoding="utf-8")
        with pytest.raises(ValueError, match="state-path registry changed"):
            verifier._validate_authorization_structure(attacked, path)


def test_postopen_receipt_requires_exact_coverage_artifact_registry(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_coverage_receipt_registry_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    for attack in ("missing", "extra"):
        attacked = tmp_path / f"receipt-{attack}"
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        authorization = json.loads(
            attacked_authorization.read_text(encoding="utf-8")
        )
        state = authorization["state_paths"]
        receipt_path = attacked / state["receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if attack == "missing":
            receipt["artifacts"].pop("temporal_coverage_audit")
            receipt["release_bindings"]["artifacts"].pop(
                "temporal_coverage_audit"
            )
        else:
            receipt["artifacts"]["coverage_alias"] = dict(
                receipt["artifacts"]["temporal_coverage_audit"]
            )
            receipt["release_bindings"]["artifacts"]["coverage_alias"] = {
                "format": verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT,
                **receipt["artifacts"]["coverage_alias"],
            }
        _reseal_postopen_fixture_receipt(
            verifier, attacked, state, receipt
        )
        with pytest.raises(ValueError, match="artifact registry"):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_postopen_release_rejects_recovery_contract_missing_tamper_and_extra(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_recovery_contract_attacks_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    amendment_relative = "protocols/route_a_inference_amendment_v1.json"
    amendment = json.loads(
        (source / amendment_relative).read_text(encoding="utf-8")
    )
    assert (
        amendment["trusted_scoring_recovery_contract"]
        == verifier.TRUSTED_SCORING_RECOVERY_CONTRACT
    )
    _validate_fixture_inference_closure(verifier, source, authorization_path)

    for attack in ("missing", "tamper", "extra"):
        attacked = tmp_path / f"recovery-{attack}"
        shutil.copytree(source, attacked)
        attacked_amendment_path = attacked / amendment_relative
        attacked_amendment = json.loads(
            attacked_amendment_path.read_text(encoding="utf-8")
        )
        recovery = attacked_amendment["trusted_scoring_recovery_contract"]
        if attack == "missing":
            recovery.pop("maximum_frozen_request_ledgers_per_opening")
        elif attack == "tamper":
            recovery["maximum_logical_openings"] = 2
        else:
            recovery["unfrozen_recovery_extension"] = True
        attacked_amendment_path.write_text(
            json.dumps(attacked_amendment), encoding="utf-8"
        )
        with pytest.raises(
            ValueError,
            match="trusted-scoring recovery contract changed",
        ):
            _validate_fixture_inference_closure(
                verifier,
                attacked,
                attacked / authorization_path.relative_to(source),
            )


def test_fast_release_requires_exact_opening_transport_document_schemas(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_transport_top_level_schema_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    attacks = (
        ("work-order-missing", state["work_order"], "source_tree_sha256"),
        ("work-order-extra", state["work_order"], None),
        ("intent-missing", state["intent"], "started_at_utc"),
        ("intent-extra", state["intent"], None),
        ("receipt-missing", state["receipt"], "security_boundary"),
        ("receipt-extra", state["receipt"], None),
        (
            "acquisition-missing",
            state["acquisition_manifest"],
            "protocol_sha256",
        ),
        ("acquisition-extra", state["acquisition_manifest"], None),
    )
    for attack, relative, missing_key in attacks:
        attacked = tmp_path / attack
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        path = attacked / relative
        document = json.loads(path.read_text(encoding="utf-8"))
        if missing_key is None:
            document["unfrozen_schema_extension"] = True
        else:
            document.pop(missing_key)
        if relative == state["work_order"]:
            document.pop("work_order_self_sha256", None)
            document["work_order_self_sha256"] = verifier._sha256_json(document)
            path.write_text(json.dumps(document), encoding="utf-8")
        elif relative == state["intent"]:
            document.pop("intent_self_sha256", None)
            document["intent_self_sha256"] = verifier._sha256_json(document)
            path.write_text(json.dumps(document), encoding="utf-8")
        elif relative == state["receipt"]:
            _reseal_postopen_fixture_receipt(
                verifier, attacked, state, document
            )
        else:
            path.write_text(json.dumps(document), encoding="utf-8")
            _refresh_postopen_transport_evidence(
                verifier, attacked, attacked_authorization
            )
        with pytest.raises(
            ValueError,
            match="exact|schema|work-order|acquisition manifest",
        ):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_fast_release_rejects_forged_transport_chain_attacks(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_transport_chain_attacks_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    verifier.build_release_profile(
        source,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization_path,
    )

    for attack in (
        "forged",
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "hash",
        "path",
        "partition",
        "attempt-chain",
        "attempt-time",
        "time",
        "series",
        "oversize",
    ):
        attacked = tmp_path / f"transport-{attack}"
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        acquisition_path = attacked / state["acquisition_manifest"]
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        ledger_path = attacked / acquisition["request_ledger"]["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if attack == "forged":
            ledger["requests"][0]["request"]["url"] += "&forged=1"
            ledger["requests"][0]["request_sha256"] = hashlib.sha256(
                verifier._canonical_json_bytes(
                    ledger["requests"][0]["request"]
                )
            ).hexdigest()
        elif attack == "missing":
            ledger.pop("request_order")
        elif attack == "extra":
            ledger["unfrozen_ledger_extension"] = True
        elif attack == "reordered":
            ledger["requests"] = list(reversed(ledger["requests"]))
        elif attack == "duplicate":
            ledger["requests"].append(dict(ledger["requests"][0]))
            ledger["request_count"] = len(ledger["requests"])
        elif attack == "path":
            alias = (
                Path(state["acquisition_manifest"]).parent
                / "request_ledger_alias_v1.json"
            ).as_posix()
            shutil.copy2(ledger_path, attacked / alias)
            acquisition["request_ledger"] = _binding(
                verifier, attacked, alias
            )
            acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
        elif attack in {"partition", "attempt-chain", "attempt-time"}:
            index_path = attacked / acquisition["transport_attempt_index"]["path"]
            index = json.loads(index_path.read_text(encoding="utf-8"))
            row = index["attempts"][0 if attack == "partition" else 1]
            start_path = attacked / row["start"]["path"]
            start = json.loads(start_path.read_text(encoding="utf-8"))
            if attack == "partition":
                start["missing_at_start_request_sha256"].append(
                    start["missing_at_start_request_sha256"][0]
                )
            elif attack == "attempt-chain":
                all_requests = sorted(
                    item["request_sha256"] for item in ledger["requests"]
                )
                start["completed_before_attempt_request_sha256"] = []
                start["missing_at_start_request_sha256"] = all_requests
            else:
                start["started_at_utc"] = "2026-01-01T00:05:00+00:00"
            start.pop("attempt_start_self_sha256")
            start["attempt_start_self_sha256"] = verifier._sha256_json(start)
            start_path.write_text(json.dumps(start), encoding="utf-8")
        elif attack in {"time", "series", "oversize"}:
            snapshot_path = attacked / acquisition["raw_nwis_snapshot_index"]["path"]
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            record = snapshot["records"][0]
            request_map_path = attacked / acquisition["request_map"]["path"]
            request_map = json.loads(request_map_path.read_text(encoding="utf-8"))
            request_row = next(
                row for row in request_map["requests"]
                if row["request_sha256"] == record["request_sha256"]
            )
            raw_root = attacked / state["raw_nwis_root"]
            metadata_path = raw_root / record["metadata_path"]
            response_path = raw_root / record["response_path"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if attack == "time":
                forged_time = "2025-12-31T23:59:59+00:00"
                metadata["retrieved_at_utc"] = forged_time
                record["retrieved_at_utc"] = forged_time
                request_row["retrieved_at_utc"] = forged_time
            elif attack == "series":
                forged_series = {
                    "WTEMP": [{
                        "parameter_code": "00010",
                        "value_column": "FORGED_00010_00003",
                        "qualifier_column": None,
                    }],
                    "FLOW": [],
                    "WLEVEL": [],
                }
                record["series_registry"] = forged_series
                request_row["series_registry"] = forged_series
            else:
                payload = b"x" * (
                    verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES + 1
                )
                response_path.write_bytes(payload)
                metadata["byte_count"] = len(payload)
                metadata["response_sha256"] = hashlib.sha256(payload).hexdigest()
                record["byte_count"] = len(payload)
                record["response_sha256"] = metadata["response_sha256"]
                request_row["byte_count"] = len(payload)
                request_row["response_sha256"] = metadata["response_sha256"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            record["metadata_sha256"] = verifier.sha256_file(metadata_path)
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            request_map_path.write_text(json.dumps(request_map), encoding="utf-8")
        elif attack == "hash":
            ledger["request_ledger_self_sha256"] = "0" * 64

        if attack not in {
            "path", "partition", "attempt-chain", "attempt-time", "time",
            "series", "oversize", "hash",
        }:
            ledger.pop("request_ledger_self_sha256", None)
            ledger["request_ledger_self_sha256"] = verifier._sha256_json(ledger)
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        elif attack == "hash":
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        _refresh_postopen_transport_evidence(
            verifier, attacked, attacked_authorization
        )
        with pytest.raises(
            ValueError,
            match=(
                "transport|request ledger|request-ledger|attempt|partition|"
                "canonical|exact contract|raw NWIS|response"
            ),
        ):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_postopen_coverage_replay_rejects_tamper_and_path_topology_attacks(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_coverage_physical_attacks_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    for attack in (
        "audit_tamper",
        "prediction_tamper",
        "receipt_path_alias",
        "prediction_symlink",
        "prediction_hardlink",
    ):
        attacked = tmp_path / attack
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        authorization = json.loads(
            attacked_authorization.read_text(encoding="utf-8")
        )
        state = authorization["state_paths"]
        receipt_path = attacked / state["receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if attack == "audit_tamper":
            audit_path = attacked / state["temporal_coverage_audit"]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["status"] = "FORGED_PASS"
            audit.pop("audit_self_sha256")
            audit["audit_self_sha256"] = verifier._sha256_json(audit)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            binding = _binding(
                verifier, attacked, state["temporal_coverage_audit"]
            )
            receipt["artifacts"]["temporal_coverage_audit"] = binding
            receipt["release_bindings"]["artifacts"][
                "temporal_coverage_audit"
            ].update(binding)
            receipt["temporal_coverage_audit"].update(binding)
        elif attack == "prediction_tamper":
            prediction_path = attacked / state["temporal_predictions"]
            frame = pd.read_parquet(prediction_path)
            frame.loc[0, "y_pred"] = float(frame.loc[0, "y_pred"]) + 0.5
            frame.to_parquet(prediction_path, index=False)
            binding = _binding(
                verifier, attacked, state["temporal_predictions"]
            )
            receipt["artifacts"]["temporal_predictions"] = binding
            receipt["release_bindings"]["artifacts"][
                "temporal_predictions"
            ].update(binding)
        elif attack == "receipt_path_alias":
            binding = dict(receipt["artifacts"]["external_predictions"])
            receipt["artifacts"]["temporal_predictions"] = binding
            receipt["release_bindings"]["artifacts"][
                "temporal_predictions"
            ].update(binding)
        else:
            prediction_path = attacked / state["temporal_predictions"]
            backup = tmp_path / f"{attack}-target.parquet"
            shutil.copy2(prediction_path, backup)
            prediction_path.unlink()
            if attack == "prediction_symlink":
                prediction_path.symlink_to(backup)
            else:
                os.link(backup, prediction_path)
        _reseal_postopen_fixture_receipt(
            verifier, attacked, state, receipt
        )
        with pytest.raises(
            ValueError,
            match=(
                "temporal-coverage|coverage|canonical|leaves|escapes|"
                "regular|unsafe|hard-linked"
            ),
        ):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_postopen_release_rejects_self_consistent_nested_outcome_gate_forgery(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_nested_outcome_gate_attack_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]

    gate_path = source / state["outcome_qc_gate"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["single_extreme_influence"][0]["absolute_effect_change_c"] += 0.01
    gate.pop("gate_self_sha256")
    gate["gate_self_sha256"] = verifier._sha256_json(gate)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    statistics_path = source / state["statistics"]
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["outcome_qc_gate"]["sha256"] = verifier.sha256_file(gate_path)
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")

    receipt_path = source / state["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for label, path in (
        ("outcome_qc_gate", gate_path),
        ("statistics", statistics_path),
    ):
        digest = verifier.sha256_file(path)
        receipt["artifacts"][label]["sha256"] = digest
        receipt["release_bindings"]["artifacts"][label]["sha256"] = digest
    receipt.pop("receipt_self_sha256")
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_digest = verifier.sha256_file(receipt_path)
    (source / state["receipt_sha256"]).write_text(
        f"{receipt_digest}  opening_receipt_v1.json\n", encoding="utf-8"
    )
    _refresh_postopen_coverage_evidence(
        verifier, source, authorization_path
    )

    with pytest.raises(ValueError, match="outcome-QC gate semantics changed"):
        verifier.build_release_profile(
            source,
            verifier.POSTOPEN_PROFILE,
            authorization_path=authorization_path,
        )


def test_postopen_release_distinguishes_transport_resume_from_second_opening(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_transport_completion_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    acquisition_path = source / state["acquisition_manifest"]
    receipt_path = source / state["receipt"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    resumed_transport = dict(acquisition["transport_summary"])
    assert resumed_transport["opening_count"] == 1
    assert resumed_transport["attempt_count"] == 2
    assert resumed_transport["resume_count"] == 1
    verifier.build_release_profile(
        source,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization_path,
    )

    second_opening = {**resumed_transport, "opening_count": 2}
    acquisition["transport_summary"] = second_opening
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    _refresh_postopen_transport_evidence(
        verifier, source, authorization_path
    )
    with pytest.raises(ValueError, match="transport summaries differ"):
        verifier.build_release_profile(
            source,
            verifier.POSTOPEN_PROFILE,
            authorization_path=authorization_path,
        )

    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    acquisition["transport_summary"] = resumed_transport
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    _refresh_postopen_transport_evidence(
        verifier, source, authorization_path
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["same_opening_transport_resume_allowed"] = False
    _reseal_postopen_fixture_receipt(
        verifier, source, state, receipt
    )
    with pytest.raises(ValueError, match="receipt exact production schema"):
        verifier.build_release_profile(
            source,
            verifier.POSTOPEN_PROFILE,
            authorization_path=authorization_path,
        )


def test_release_verifier_requires_both_receipts_and_exact_control_members(
    tmp_path, monkeypatch,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_preopening_control_gates_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _representatives = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    suite_path = source / authorization["model_suite"]["path"]
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    development = suite["development_contract"]
    original_paired_recompute = verifier._stage09b_recompute_paired_effects
    paired_recompute_calls: list[int] = []

    def observed_paired_recompute(station_metrics):
        paired_recompute_calls.append(len(set(zip(
            station_metrics["arm_id"].astype(str),
            station_metrics["seed"].astype(int),
            strict=True,
        ))))
        return original_paired_recompute(station_metrics)

    monkeypatch.setattr(
        verifier, "_stage09b_recompute_paired_effects", observed_paired_recompute,
    )
    verifier._validate_preopening_completion_gates(
        source, {}, suite, development, suite["numerical_runtime_sha256"]
    )
    assert paired_recompute_calls == [31]

    missing = json.loads(json.dumps(suite))
    missing["preopening_gates"].pop("stage09b_development_controls")
    with pytest.raises(ValueError, match="Stage-9/09b"):
        verifier._validate_preopening_completion_gates(
            source, {}, missing, development, suite["numerical_runtime_sha256"]
        )

    controls_binding = suite["preopening_gates"]["stage09b_development_controls"]
    controls_path = source / controls_binding["path"]
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    controls["member_registry"].pop()
    stable = {
        key: value for key, value in controls.items()
        if key != "receipt_self_sha256"
    }
    controls["receipt_self_sha256"] = verifier._sha256_json(stable)
    controls_path.write_text(json.dumps(controls), encoding="utf-8")
    suite["preopening_gates"]["stage09b_development_controls"] = _binding(
        verifier, source, controls_binding["path"]
    )
    with pytest.raises(ValueError, match="matrix audit|31 members"):
        verifier._validate_preopening_completion_gates(
            source, {}, suite, development, suite["numerical_runtime_sha256"]
        )


@pytest.mark.parametrize(
    "attack",
    (
        "seed_bool", "seed_float", "horizon_bool", "horizon_float",
        "site_integer", "string_date", "timezone", "intraday",
        "y_true_bool", "y_pred_integer", "q05_integer", "q50_bool",
        "q95_integer", "probability_bool",
    ),
)
def test_independent_release_normaliser_rejects_coercion_aliases(
    attack: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_stage09b_{attack}_alias_test",
    )
    records = []
    for split_index, split in enumerate(("val", "calib", "test")):
        for horizon in (1, 3, 7):
            issue = pd.Timestamp("2017-01-01") + pd.Timedelta(days=split_index)
            records.append({
                "model": "PlainMLP-7var",
                "scope": "development_only_2006_2020",
                "feature_set": "all_7",
                "seed": 0,
                "site_id": "12345678",
                "horizon": horizon,
                "split": split,
                "issue_date": issue,
                "target_date": issue + pd.Timedelta(days=horizon),
                "y_true": 1.0,
                "y_pred": 1.1,
                "q05": 0.5,
                "q50": 1.1,
                "q95": 1.5,
                "p_exceed": 0.25,
            })
    frame = pd.DataFrame.from_records(records)
    if attack == "seed_bool":
        frame["seed"] = False
    elif attack == "seed_float":
        frame["seed"] = frame["seed"].astype("float64")
    elif attack == "horizon_bool":
        frame["horizon"] = frame["horizon"].astype(object)
        frame.loc[frame["horizon"].eq(1), "horizon"] = True
    elif attack == "horizon_float":
        frame["horizon"] = frame["horizon"].astype("float64")
    elif attack == "site_integer":
        frame["site_id"] = 12_345_678
    elif attack == "string_date":
        frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")
    elif attack == "timezone":
        frame["issue_date"] = frame["issue_date"].dt.tz_localize("UTC")
        frame["target_date"] = frame["target_date"].dt.tz_localize("UTC")
    elif attack == "intraday":
        frame["issue_date"] += pd.Timedelta(hours=1)
        frame["target_date"] += pd.Timedelta(hours=1)
    elif attack == "y_true_bool":
        frame["y_true"] = False
    elif attack == "y_pred_integer":
        frame["y_pred"] = 1
    elif attack == "q05_integer":
        frame["q05"] = 0
    elif attack == "q50_bool":
        frame["q50"] = True
    elif attack == "q95_integer":
        frame["q95"] = 2
    else:
        frame["p_exceed"] = False
    with pytest.raises(ValueError, match="Stage-09b prediction"):
        verifier._normalise_stage09b_release_prediction(
            frame,
            arm_id="PlainMLP-7var",
            seed=0,
            feature_set="all_7",
            reference=None,
        )


@pytest.mark.parametrize("attack", ("string_date", "y_true_bool", "q50_integer"))
def test_independent_release_rejects_noncanonical_prediction_arrow_types(
    tmp_path: Path, attack: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_stage09b_{attack}_arrow_test",
    )
    issue = pd.Timestamp("2017-01-01")
    frame = pd.DataFrame([{
        "model": "PlainMLP-7var",
        "scope": "development_only_2006_2020",
        "feature_set": "all_7",
        "seed": 0,
        "site_id": "12345678",
        "horizon": 1,
        "split": "test",
        "issue_date": issue,
        "target_date": issue + pd.Timedelta(days=1),
        "y_true": 1.0,
        "y_pred": 1.1,
        "q05": 0.5,
        "q50": 1.1,
        "q95": 1.5,
        "p_exceed": 0.25,
    }])
    if attack == "string_date":
        frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")
    elif attack == "y_true_bool":
        frame["y_true"] = False
    else:
        frame["q50"] = 1
    path = tmp_path / "attacked.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Arrow type"):
        verifier._stage09b_assert_prediction_arrow_schema(path)


@pytest.mark.parametrize("attack", ("split", "station", "seed", "horizon"))
def test_independent_release_paired_recompute_rejects_registry_attacks(
    attack: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_stage09b_{attack}_registry_test",
    )
    rows = [
        {
            "arm_id": arm_id,
            "seed": seed,
            "split": split,
            "horizon": horizon,
            "site_id": site,
            "forecast_keys": 2,
            "station_rmse_c": 1.0 + seed / 100,
        }
        for arm_id, seed in verifier._stage09b_release_members()
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
        for site in ("01234567", "12345678")
    ]
    station = pd.DataFrame.from_records(rows)
    if attack == "split":
        station = station.loc[~station["split"].eq("val")].copy()
    elif attack == "station":
        mask = (
            station["arm_id"].eq("ThermoRoute-ladder-07_plus_WDSP")
            & station["seed"].eq(0)
            & station["split"].eq("test")
            & station["horizon"].eq(1)
            & station["site_id"].eq("12345678")
        )
        station.loc[mask, "site_id"] = "99999999"
    elif attack == "seed":
        station = station.loc[
            ~(
                station["arm_id"].eq("PlainMLP-7var")
                & station["seed"].eq(4)
            )
        ].copy()
    else:
        station.loc[station["horizon"].eq(7), "horizon"] = 5
    with pytest.raises(ValueError, match="registry|31 members"):
        verifier._stage09b_recompute_paired_effects(station)


def test_release_lineage_hash_matches_live_opening_under_non_ascii_root(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from thermoroute.repro import sha256_json

    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_unicode_hash_test")
    source = tmp_path / "[副业]论文" / "source"
    stage = tmp_path / "[副业]论文" / "stage"
    source.mkdir(parents=True)
    stage.mkdir(parents=True)
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    stable = dict(authorization)
    claimed = stable.pop("authorization_self_sha256")

    assert claimed == sha256_json(stable)
    assert verifier._sha256_json(stable) == sha256_json(stable)
    document = verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization_path,
    )
    assert document["profile"] == verifier.POSTOPEN_PROFILE


def test_deterministic_zip_normalises_order_timestamp_modes_and_manifest_time(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_zip_shape_test")
    zipper = _load_script(ZIP_SCRIPT, "thermoroute_deterministic_zip_test")
    stage = tmp_path / "stage"
    _write_bytes(stage, "scripts/tool.py", b"print('x')\n")
    _write_bytes(stage, "data/value.txt", b"value\n")
    manifest = _write_bytes(
        stage,
        "outputs/manifest.json",
        json.dumps({"generated_utc": "2099-01-01T00:00:00+00:00"}).encode(),
    )
    revision = {
        "compute_commit": "a" * 40,
        "manuscript_commit": "b" * 40,
        "committed_document_diff": [],
    }
    claim_validation = {"path": "evidence/claims.json", "sha256": "c" * 64}
    history_evidence = {"bundle": {"path": "evidence/history.bundle"}}
    lock_binding = {
        "path": "requirements-lock-py312-hashed.txt",
        "sha256": "d" * 64,
    }
    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps({
            "profile": verifier.POSTOPEN_PROFILE,
            "authorized_worktree_dirt_policy": revision,
            "claim_validation": claim_validation,
            "git_history_evidence": history_evidence,
            "artifact_closure": {"reproducibility_lock": lock_binding},
        }).encode(),
    )
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    zipper.create_deterministic_zip(stage, first)
    first_sha = hashlib.sha256(first.read_bytes()).hexdigest()

    os.chmod(stage / "scripts/tool.py", 0o600)
    os.chmod(stage / "data/value.txt", 0o777)
    os.utime(stage / "data/value.txt", (2_000_000_000, 2_000_000_000))
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["generated_utc"] = "2100-01-01T00:00:00+00:00"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    zipper.create_deterministic_zip(stage, second)
    assert hashlib.sha256(second.read_bytes()).hexdigest() == first_sha

    with zipfile.ZipFile(second) as archive:
        members = verifier.normalised_members(archive)
        assert {"scripts/tool.py", "data/value.txt", "outputs/manifest.json"} <= members
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        modes = {
            info.filename: (info.external_attr >> 16) & 0o777
            for info in archive.infolist() if not info.is_dir()
        }
        assert modes["thermoroute/scripts/tool.py"] == 0o755
        assert modes["thermoroute/data/value.txt"] == 0o644
        archived_manifest = json.loads(
            archive.read("thermoroute/outputs/manifest.json")
        )
        assert archived_manifest["release_revision"] == revision
        assert archived_manifest["release_evidence"] == {
            "profile": verifier.POSTOPEN_PROFILE,
            "claim_validation": claim_validation,
            "git_history_evidence": history_evidence,
            "reproducibility_lock": lock_binding,
        }


def test_git_bundle_replays_sealed_protocol_after_release_relocation(
    tmp_path, monkeypatch
):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_git_evidence_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    protocol_relative = "protocols/route_a_confirmatory_protocol.md"
    protocol = _write_bytes(source, protocol_relative, b"# original fixture protocol\n")
    original_bytes = protocol.read_bytes()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", protocol_relative], cwd=source, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })
    subprocess.run(
        ["git", "commit", "-q", "-m", "original protocol"],
        cwd=source,
        env=environment,
        check=True,
    )
    original_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    protocol.write_bytes(b"# final prelabel fixture protocol\n")
    protocol_json = _write_bytes(
        source,
        "protocols/route_a_confirmatory_v1.json",
        json.dumps({
            "protocol_id": "route-a-confirmatory-v1",
            "authoritative_protocol_commit": original_commit,
        }, sort_keys=True).encode() + b"\n",
    )
    subprocess.run(
        ["git", "add", protocol_relative, "protocols/route_a_confirmatory_v1.json"],
        cwd=source, check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "final prelabel protocol"],
        cwd=source, env=environment, check=True,
    )
    final_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    seal_document = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": protocol_relative,
                "sha256": hashlib.sha256(original_bytes).hexdigest(),
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": "protocols/route_a_confirmatory_v1.json",
                "sha256": hashlib.sha256(protocol_json.read_bytes()).hexdigest(),
            },
            "markdown": {
                "path": protocol_relative,
                "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            },
        },
        "prelabel_attestation": {
            "external_timestamp_or_public_preregistration": False,
            "independent_custodian_or_worm_storage": False,
        },
    }
    seal_path = _write_bytes(
        source,
        verifier.PROTOCOL_SEAL_PATH,
        json.dumps(seal_document, sort_keys=True).encode() + b"\n",
    )
    subprocess.run(
        ["git", "add", verifier.PROTOCOL_SEAL_PATH], cwd=source, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "mechanical protocol seal"],
        cwd=source, env=environment, check=True,
    )
    compute_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    _write_bytes(source, "README.md", b"later documentation\n")
    subprocess.run(["git", "add", "README.md"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "later docs"],
        cwd=source,
        env=environment,
        check=True,
    )

    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps({"profile": verifier.PREOPEN_PROFILE}).encode(),
    )
    for source_path, relative in (
        (protocol_json, "protocols/route_a_confirmatory_v1.json"),
        (protocol, protocol_relative),
        (seal_path, verifier.PROTOCOL_SEAL_PATH),
    ):
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    evidence = verifier.materialize_git_history_evidence(
        source, stage, verifier.PREOPEN_PROFILE
    )
    assert evidence["sealed_protocol_blob"]["sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    assert evidence["final_prelabel_protocol_commit"] == final_commit
    assert evidence["external_timestamp_or_public_preregistration"] is False

    relocated = tmp_path / "a-different-absolute-path" / "release"
    shutil.copytree(stage, relocated)
    marker = json.loads(
        (relocated / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    verifier._verify_git_history_evidence(
        relocated, marker, verifier.PREOPEN_PROFILE
    )

    # This fixture predates the chronology receipt and intentionally isolates
    # protocol replay plus the compute-to-manuscript document diff.  The
    # dedicated real-Git test below exercises the complete chronology verifier.
    monkeypatch.setattr(
        verifier,
        "_verify_prelabel_chronology_from_bundle",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        verifier,
        "_verify_authorized_compute_tree_from_bundle",
        lambda **_kwargs: None,
    )

    manuscript_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    shutil.copy2(source / "README.md", relocated / "README.md")
    authorization_relative = "data_usgs/opening_authorization.json"
    _write_bytes(
        relocated,
        authorization_relative,
        json.dumps({
            "protocol": {
                "authoritative_markdown_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "final_prelabel_commit": final_commit,
                "seal": _binding(verifier, relocated, verifier.PROTOCOL_SEAL_PATH),
            },
        }).encode(),
    )
    document_binding = {
        **_binding(verifier, relocated, "README.md"),
        "bytes": (relocated / "README.md").stat().st_size,
    }
    marker["profile"] = verifier.POSTOPEN_PROFILE
    marker["authorization"] = {"path": authorization_relative}
    marker["git_history_evidence"]["profile"] = verifier.POSTOPEN_PROFILE
    marker["git_history_evidence"]["compute_commit"] = compute_commit
    marker["git_history_evidence"]["manuscript_commit"] = manuscript_commit
    marker["authorized_worktree_dirt_policy"] = {
        "committed_document_diff": [document_binding],
    }
    verifier._verify_git_history_evidence(
        relocated, marker, verifier.POSTOPEN_PROFILE
    )

    missing_binding = json.loads(json.dumps(marker))
    missing_binding["authorized_worktree_dirt_policy"][
        "committed_document_diff"
    ] = []
    with pytest.raises(ValueError, match="diff differs"):
        verifier._verify_git_history_evidence(
            relocated, missing_binding, verifier.POSTOPEN_PROFILE
        )
    extra_binding = json.loads(json.dumps(marker))
    extra_binding["authorized_worktree_dirt_policy"][
        "committed_document_diff"
    ].append({"path": "paper/extra.md", "sha256": "1" * 64, "bytes": 1})
    with pytest.raises(ValueError, match="diff differs"):
        verifier._verify_git_history_evidence(
            relocated, extra_binding, verifier.POSTOPEN_PROFILE
        )
    wrong_blob = json.loads(json.dumps(marker))
    wrong_blob["authorized_worktree_dirt_policy"]["committed_document_diff"][0][
        "sha256"
    ] = "2" * 64
    with pytest.raises(ValueError, match="manuscript blob differs"):
        verifier._verify_git_history_evidence(
            relocated, wrong_blob, verifier.POSTOPEN_PROFILE
        )

    _write_bytes(source, "src/hidden_compute_change.py", b"hidden = True\n")
    subprocess.run(["git", "add", "src/hidden_compute_change.py"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "forbidden hidden compute change"],
        cwd=source, env=environment, check=True,
    )
    (source / "src/hidden_compute_change.py").unlink()
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "hide forbidden change by reverting it"],
        cwd=source, env=environment, check=True,
    )
    hidden_stage = tmp_path / "hidden-stage"
    _write_bytes(
        hidden_stage,
        verifier.PROFILE_MARKER,
        json.dumps({"profile": verifier.PREOPEN_PROFILE}).encode(),
    )
    for source_path, relative in (
        (protocol_json, "protocols/route_a_confirmatory_v1.json"),
        (protocol, protocol_relative),
        (seal_path, verifier.PROTOCOL_SEAL_PATH),
    ):
        destination = hidden_stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    hidden_evidence = verifier.materialize_git_history_evidence(
        source, hidden_stage, verifier.PREOPEN_PROFILE
    )
    shutil.copy2(source / "README.md", hidden_stage / "README.md")
    _write_bytes(
        hidden_stage,
        authorization_relative,
        (relocated / authorization_relative).read_bytes(),
    )
    hidden_marker = json.loads(
        (hidden_stage / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    hidden_marker["profile"] = verifier.POSTOPEN_PROFILE
    hidden_marker["authorization"] = {"path": authorization_relative}
    hidden_evidence["profile"] = verifier.POSTOPEN_PROFILE
    hidden_evidence["compute_commit"] = compute_commit
    hidden_marker["git_history_evidence"] = hidden_evidence
    hidden_marker["authorized_worktree_dirt_policy"] = {
        "committed_document_diff": [{
            **_binding(verifier, hidden_stage, "README.md"),
            "bytes": (hidden_stage / "README.md").stat().st_size,
        }],
    }
    with pytest.raises(ValueError, match="forbidden compute-to-manuscript"):
        verifier._verify_git_history_evidence(
            hidden_stage, hidden_marker, verifier.POSTOPEN_PROFILE
        )

    main_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "checkout", "-q", "-b", "unrelated-fixture", compute_commit],
        cwd=source, check=True,
    )
    _write_bytes(source, "paper/unrelated.md", b"unrelated history\n")
    subprocess.run(["git", "add", "paper/unrelated.md"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "unrelated sibling"],
        cwd=source, env=environment, check=True,
    )
    unrelated_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-q", main_branch], cwd=source, check=True)
    unrelated_stage = tmp_path / "unrelated-stage"
    shutil.copytree(hidden_stage, unrelated_stage)
    unrelated_bundle = unrelated_stage / verifier.GIT_BUNDLE_PATH
    unrelated_bundle.unlink()
    subprocess.run(
        ["git", "bundle", "create", str(unrelated_bundle), "--all"],
        cwd=source, check=True, capture_output=True, text=True,
    )
    unrelated_marker = json.loads(json.dumps(hidden_marker))
    unrelated_marker["git_history_evidence"]["bundle"] = _binding(
        verifier, unrelated_stage, verifier.GIT_BUNDLE_PATH
    )
    unrelated_marker["git_history_evidence"]["compute_commit"] = unrelated_commit
    with pytest.raises(ValueError, match="compute-to-manuscript"):
        verifier._verify_git_history_evidence(
            unrelated_stage, unrelated_marker, verifier.POSTOPEN_PROFILE
        )

    (relocated / verifier.GIT_BUNDLE_PATH).unlink()
    with pytest.raises(ValueError, match="absent"):
        verifier._verify_git_history_evidence(
            relocated, marker, verifier.POSTOPEN_PROFILE
        )


def test_postopen_git_bundle_replays_real_prelabel_chronology_and_rejects_tamper(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_real_chronology_git_evidence_test"
    )
    source = tmp_path / "chronology-source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Chronology Fixture",
        "GIT_AUTHOR_EMAIL": "chronology@example.invalid",
        "GIT_COMMITTER_NAME": "Chronology Fixture",
        "GIT_COMMITTER_EMAIL": "chronology@example.invalid",
    })

    def write_json(relative: str, value: object) -> Path:
        return _write_bytes(
            source,
            relative,
            json.dumps(value, sort_keys=True).encode("utf-8") + b"\n",
        )

    def git_text(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=source,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def commit(message: str) -> str:
        subprocess.run(["git", "add", "-A"], cwd=source, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=source,
            env=environment,
            check=True,
        )
        return git_text("rev-parse", "HEAD")

    def git_blob_binding(commit_id: str, relative: str) -> dict[str, object]:
        payload = subprocess.run(
            ["git", "show", f"{commit_id}:{relative}"],
            cwd=source,
            capture_output=True,
            check=True,
        ).stdout
        oid = git_text("rev-parse", f"{commit_id}:{relative}")
        assert len(oid) == 40
        return {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "git_blob_oid": oid,
        }

    protocol_markdown_path = "protocols/route_a_confirmatory_protocol.md"
    protocol_json_path = "protocols/route_a_confirmatory_v1.json"
    original_markdown = _write_bytes(
        source, protocol_markdown_path, b"# Original preregistration\n"
    ).read_bytes()
    original_commit = commit("original preregistration")

    final_markdown = _write_bytes(
        source, protocol_markdown_path, b"# Final prelabel protocol\n"
    )
    final_protocol = write_json(
        protocol_json_path,
        {
            "protocol_id": "route-a-confirmatory-v1",
            "authoritative_protocol_commit": original_commit,
        },
    )
    final_commit = commit("final prelabel protocol")

    seal = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": protocol_markdown_path,
                "sha256": hashlib.sha256(original_markdown).hexdigest(),
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": protocol_json_path,
                "sha256": hashlib.sha256(final_protocol.read_bytes()).hexdigest(),
            },
            "markdown": {
                "path": protocol_markdown_path,
                "sha256": hashlib.sha256(final_markdown.read_bytes()).hexdigest(),
            },
        },
        "prelabel_attestation": {
            "external_timestamp_or_public_preregistration": False,
            "independent_custodian_or_worm_storage": False,
        },
    }
    write_json(verifier.PROTOCOL_SEAL_PATH, seal)
    _write_bytes(source, "pyproject.toml", b"[project]\nname='chronology-fixture'\n")
    _write_bytes(source, "requirements.txt", b"fixture>=1\n")
    _write_bytes(source, "requirements-lock.txt", b"fixture==1\n")
    _write_bytes(
        source,
        verifier.REPRODUCIBILITY_LOCK,
        b"fixture==1 --hash=sha256:" + b"0" * 64 + b"\n",
    )
    gate_paths = (
        "src/thermoroute/chronology.py",
        "src/thermoroute/outcome_qc.py",
        "scripts/28_freeze_prelabel_chronology.py",
        "tests/test_chronology.py",
        "protocols/route_a_outcome_qc_policy_v1.json",
    )
    for relative in gate_paths:
        _write_bytes(source, relative, f"# frozen gate: {relative}\n".encode())
    fixed_modules = {
        "thermoroute.opening": "src/thermoroute/opening.py",
        "thermoroute.model_suite": "src/thermoroute/model_suite.py",
        "thermoroute.frozen_inference": "src/thermoroute/frozen_inference.py",
        "thermoroute.datasets": "src/thermoroute/datasets.py",
        "thermoroute.provenance": "src/thermoroute/provenance.py",
        "thermoroute.usgs": "src/thermoroute/usgs.py",
        "thermoroute.inference_gate": "src/thermoroute/inference_gate.py",
        "thermoroute.outcome_qc": "src/thermoroute/outcome_qc.py",
        "thermoroute.quantiles": "src/thermoroute/quantiles.py",
        "thermoroute.coverage_audit": "src/thermoroute/coverage_audit.py",
        "thermoroute.coverage_bridge": "src/thermoroute/coverage_bridge.py",
        "thermoroute.repro": "src/thermoroute/repro.py",
    }
    fixed_files = {
        relative: relative
        for relative in (
            "src/thermoroute/opening_contract.py",
            "src/thermoroute/outcome_acquisition.py",
        )
    }
    fixed_entrypoints = {
        "orchestrator": "scripts/route_a_opening_orchestrator.py",
        "acquisition": "scripts/route_a_outcome_acquisition.py",
        "trusted_scorer": "scripts/route_a_trusted_scorer.py",
    }
    for relative in {
        *fixed_modules.values(), *fixed_files.values(), *fixed_entrypoints.values()
    }:
        _write_bytes(source, relative, f"# fixed code: {relative}\n".encode())
    model_suite_path = "data_usgs/confirmatory_model_suite_v1.json"
    development_replay_path = (
        "outputs/model_replay/route_a_development_replay_v1.json"
    )
    development_paths = {
        "frozen_panel_spec": "data_usgs/frozen_panel_v1.json",
        "panel": "data_usgs/panel_usgs_120v2.parquet",
        "registry": "data_usgs/station_registry_v1.csv",
    }
    for relative in development_paths.values():
        _write_bytes(source, relative, b"{}\n" if relative.endswith(".json") else b"dev\n")
    prediction_path = "outputs/development/predictions.parquet"
    prediction_sidecar = prediction_path + ".meta.json"
    _write_bytes(source, prediction_path, b"predictions\n")
    _write_bytes(source, prediction_sidecar, b"{}\n")
    lgb_model_paths = {
        head: f"outputs/models/lgb/member_h1_{head}.txt"
        for head in ("point", "q05", "q50", "q95", "event")
    }
    for head, relative in lgb_model_paths.items():
        _write_bytes(source, relative, f"{head} tree\n".encode())
    lgb_manifest_path = "outputs/models/lgb/manifest.json"
    runtime_sha256 = "e" * 64
    raw_crossing_member = {
        "1": {
            "rows": 3,
            "forecast_key_sha256": "a" * 64,
            "raw_prediction_sha256": "b" * 64,
            "q05_above_q50_count": 0,
            "q50_above_q95_count": 0,
            "any_crossing_count": 0,
            "any_crossing_rate": 0.0,
            "maximum_crossing_gap_c": 0.0,
        }
    }
    raw_crossing_audit = {
        "format": "thermoroute.raw-quantile-crossing-audit.v1",
        "scope": "development_export_rows_before_repair",
        "key_columns": [
            "site_id", "horizon", "split", "issue_date", "target_date"
        ],
        "repair_method": "median_preserving_endpoint_clip_v1",
        "members": {"seed0": raw_crossing_member},
    }
    raw_crossing_audit["audit_sha256"] = verifier._sha256_json(
        raw_crossing_audit
    )
    write_json(
        lgb_manifest_path,
        {
            "format": "thermoroute.lightgbm-bundle.v2",
            "training_device": "cpu",
            "runtime_sha256": runtime_sha256,
            "heads": ["point", "q05", "q50", "q95", "event"],
            "members": ["seed0"],
            "member_count": 1,
            "horizons": [1],
            "quantile_repair": {
                "method": "median_preserving_endpoint_clip_v1",
                "version": 1,
                "nominal_head_levels": {
                    "q05": 0.05,
                    "q50": 0.50,
                    "q95": 0.95,
                },
                "q05_operation": "minimum(raw_q05,raw_q50)",
                "q50_operation": "raw_q50_unchanged",
                "q95_operation": "maximum(raw_q95,raw_q50)",
                "nominal_median_preserved_exactly": True,
            },
            "raw_quantile_crossing_audit": raw_crossing_audit,
            "models": {
                "seed0": {
                    "1": {
                        head: {
                            "path": PurePosixPath(relative).name,
                            "sha256": verifier.sha256_file(source / relative),
                        }
                        for head, relative in lgb_model_paths.items()
                    }
                }
            },
            "development_prediction": {
                "artifact": {
                    **_binding(verifier, source, prediction_path),
                    "sidecar": _binding(verifier, source, prediction_sidecar),
                }
            },
        },
    )
    inference_amendment_path = "protocols/route_a_inference_amendment_v1.json"
    inference_amendment_seal_path = (
        "protocols/route_a_inference_amendment_seal_v1.json"
    )
    inference_gate_path = "outputs/prelabel/route_a_inference_gate_v1.json"
    write_json(inference_amendment_path, {"fixture": "outcome-free amendment"})
    amendment_commit = commit("freeze outcome-free inference amendment")
    write_json(inference_amendment_seal_path, {"fixture": "amendment seal"})
    seal_commit = commit("seal outcome-free inference amendment")
    write_json(inference_gate_path, {"fixture": "fail-closed inference gate"})
    frozen_source_inventory = {
        relative: verifier.sha256_file(source / relative)
        for relative in sorted(verifier._working_model_control_paths(source))
        if verifier._matches_source_inventory(relative)
    }
    frozen_source_sha = verifier._sha256_json(frozen_source_inventory)
    bridge_path = "data_usgs/development_predictor_bridge_v1.json"
    bridge_normalized = {
        "frozen": (
            "data_usgs/development_predictor_bridge_v1/"
            "frozen_panel_predictors_2018_2020.parquet"
        ),
        "refreshed": (
            "data_usgs/development_predictor_bridge_v1/"
            "refreshed_predictors_2018_2020.parquet"
        ),
    }
    bridge_report = "data_usgs/development_predictor_bridge_v1/bridge_report_v1.json"
    bridge_request_map = (
        "data_usgs/development_predictor_bridge_v1/source_request_map_v1.json"
    )
    for relative in (*bridge_normalized.values(), bridge_report, bridge_request_map):
        _write_bytes(source, relative, b"{}\n" if relative.endswith(".json") else b"bridge\n")
    bridge_indexes: dict[str, str] = {}
    bridge_raw_paths: set[str] = set()
    for name in ("daymet", "gridmet", "gridmet_schema"):
        index = (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            f"{name}/snapshot_index_v2.json"
        )
        metadata = str(PurePosixPath(index).parent / "metadata.json")
        response = str(PurePosixPath(index).parent / "response.bin")
        metadata_path = write_json(metadata, {})
        _write_bytes(source, response, f"{name} raw\n".encode())
        write_json(
            index,
            {
                "schema_version": 2,
                "snapshot_count": 1,
                "records": [{
                    "provider": name,
                    "request_sha256": hashlib.sha256(
                        f"{name} request".encode()
                    ).hexdigest(),
                    "metadata_path": "metadata.json",
                    "response_path": "response.bin",
                    "response_sha256": verifier.sha256_file(source / response),
                    "metadata_sha256": verifier.sha256_file(metadata_path),
                    "metadata_byte_count": metadata_path.stat().st_size,
                    "retrieved_at_utc": "2026-07-22T00:00:00+00:00",
                    "byte_count": (source / response).stat().st_size,
                    "request": {"provider": name},
                }]
            },
        )
        bridge_indexes[name] = index
        bridge_raw_paths.update({index, metadata, response})
    bridge = {
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": frozen_source_sha,
        "panel": _binding(verifier, source, development_paths["panel"]),
        "registry": _binding(verifier, source, development_paths["registry"]),
        "normalized": {
            name: _binding(verifier, source, relative)
            for name, relative in bridge_normalized.items()
        },
        "report": _binding(verifier, source, bridge_report),
        "request_map": _binding(verifier, source, bridge_request_map),
        "raw_snapshot_indexes": {
            name: _binding(verifier, source, relative)
            for name, relative in bridge_indexes.items()
        },
    }
    write_json(bridge_path, bridge)

    stage09_run_id = "stage09-fixture"
    stage09_artifacts = {
        "run_manifest": (
            f"outputs/runs/09_usgs_experiment/{stage09_run_id}/run.json"
        ),
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
    for label, relative in stage09_artifacts.items():
        _write_bytes(source, relative, f"stage09 {label}\n".encode())
    stage09_receipt_path = "outputs/models/route_a_stage09_completion.json"
    stage09_receipt = {
        "format": "thermoroute.stage09-completion-receipt.v1",
        "status": "PASS_FORMAL_STAGE09_COMPLETE",
        "stage": "09_usgs_experiment",
        "run_id": stage09_run_id,
        "run_identity": {"run_id": stage09_run_id},
        "formal_configuration": {"fixture": True},
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": {
            label: _binding(verifier, source, relative)
            for label, relative in stage09_artifacts.items()
        },
    }
    stage09_receipt["receipt_self_sha256"] = verifier._sha256_json(
        stage09_receipt
    )
    write_json(stage09_receipt_path, stage09_receipt)

    stage09b_run_id = "stage09b-fixture"
    stage09b_run_dir = (
        f"outputs/runs/09b_development_controls/{stage09b_run_id}"
    )
    stage09b_artifacts = {
        "run_manifest": f"{stage09b_run_dir}/run.json",
        "frozen_panel_spec": development_paths["frozen_panel_spec"],
        "panel": development_paths["panel"],
        "registry": development_paths["registry"],
        "predictor_bridge": bridge_path,
        "predictions": f"{stage09b_run_dir}/development_controls_predictions.parquet",
        "prediction_sidecar": (
            f"{stage09b_run_dir}/development_controls_predictions.parquet.meta.json"
        ),
        "architecture_budget": (
            f"{stage09b_run_dir}/development_controls_architecture_budget.csv"
        ),
        "architecture_budget_sidecar": (
            f"{stage09b_run_dir}/development_controls_architecture_budget.csv.meta.json"
        ),
        "metric_summary": f"{stage09b_run_dir}/development_controls_metric_summary.csv",
        "metric_summary_sidecar": (
            f"{stage09b_run_dir}/development_controls_metric_summary.csv.meta.json"
        ),
        "report": f"{stage09b_run_dir}/development_controls_report.md",
        "report_sidecar": f"{stage09b_run_dir}/development_controls_report.md.meta.json",
        "semantic_audit": f"{stage09b_run_dir}/development_controls_semantic_audit.json",
        "semantic_audit_sidecar": (
            f"{stage09b_run_dir}/development_controls_semantic_audit.json.meta.json"
        ),
    }
    for label, relative in stage09b_artifacts.items():
        if label in {
            "frozen_panel_spec", "panel", "registry", "predictor_bridge",
            "semantic_audit", "semantic_audit_sidecar",
        }:
            continue
        _write_bytes(source, relative, f"stage09b {label}\n".encode())
    stage09b_members = []
    semantic_members = []
    stage09b_member_paths: set[str] = set()
    for arm_id, seed in verifier._stage09b_release_members():
        member_path = (
            f"{stage09b_run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
        )
        member_sidecar = f"{member_path}.meta.json"
        checkpoint_path = (
            f"{stage09b_run_dir}/checkpoints/{arm_id}/seed{seed}.pt"
        )
        checkpoint_sidecar = f"{checkpoint_path}.meta.json"
        _write_bytes(source, member_path, f"{arm_id}/seed{seed}\n".encode())
        _write_bytes(source, member_sidecar, b"{}\n")
        _write_bytes(
            source,
            checkpoint_path,
            f"checkpoint:{arm_id}:seed{seed}\n".encode(),
        )
        _write_bytes(source, checkpoint_sidecar, b"{}\n")
        stage09b_member_paths.update({
            member_path,
            member_sidecar,
            checkpoint_path,
            checkpoint_sidecar,
        })
        stage09b_members.append({
            "arm_id": arm_id,
            "seed": seed,
            "prediction": _binding(verifier, source, member_path),
            "prediction_sidecar": _binding(verifier, source, member_sidecar),
            "checkpoint": _binding(verifier, source, checkpoint_path),
            "checkpoint_sidecar": _binding(
                verifier, source, checkpoint_sidecar
            ),
        })
        semantic_members.append({
            "arm_id": arm_id,
            "seed": seed,
            "prediction": {
                "sha256": verifier.sha256_file(source / member_path),
                "bytes": (source / member_path).stat().st_size,
            },
            "prediction_sidecar": {
                "sha256": verifier.sha256_file(source / member_sidecar),
                "bytes": (source / member_sidecar).stat().st_size,
            },
            "checkpoint": {
                "sha256": verifier.sha256_file(source / checkpoint_path),
                "bytes": (source / checkpoint_path).stat().st_size,
            },
            "checkpoint_sidecar": {
                "sha256": verifier.sha256_file(source / checkpoint_sidecar),
                "bytes": (source / checkpoint_sidecar).stat().st_size,
            },
            "normalised_prediction_sha256": hashlib.sha256(
                f"normalised:{arm_id}:{seed}".encode()
            ).hexdigest(),
            "best_model_state_prediction_replay_verified": True,
        })

    stage09b_matrix = {
        "expected_members": 31,
        "prediction_rows": 93,
        "common_forecast_keys": 3,
        "splits": ["calib", "test", "val"],
        "reference_member": "PlainMLP-7var/seed0",
    }

    def stage09b_descriptor(relative: str) -> dict[str, object]:
        return {
            "sha256": verifier.sha256_file(source / relative),
            "bytes": (source / relative).stat().st_size,
        }

    stage09b_paired_records = [
        {
            "comparison_family": comparison["comparison_family"],
            "comparison_id": comparison["comparison_id"],
            "candidate_arm_id": comparison["candidate_arm_id"],
            "reference_arm_id": comparison["reference_arm_id"],
            "seed": seed,
            "split": split,
            "horizon": horizon,
            "common_forecast_keys": 1,
            "stations": 1,
            "median_paired_station_rmse_difference_c": 0.0,
        }
        for comparison in verifier._stage09b_paired_comparison_registry()
        for seed in (0, 1, 2)
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
    ]
    stage09b_scientific = verifier._stage09b_scientific_summary(
        pd.DataFrame.from_records(stage09b_paired_records)
    )
    semantic_audit = {
        "format": "thermoroute.development-controls-semantic-audit.v3",
        "status": "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "run_id": stage09b_run_id,
        "evidence_scope": "best_model_state_prediction_replay",
        "training_replay_verified": False,
        "best_model_state_prediction_replay_verified": True,
        "post_2020_outcomes_requested_or_read": False,
        "matrix_audit": stage09b_matrix,
        "canonical_window_registry": {
            "sha256": "c" * 64,
            "common_forecast_keys": 3,
            "train_examples_per_epoch": 3,
            "train_registry_sha256": "d" * 64,
        },
        "scientific_summary": stage09b_scientific,
        "members": semantic_members,
        "derived_artifacts": {
            "architecture_budget": {
                "artifact": stage09b_descriptor(
                    stage09b_artifacts["architecture_budget"]
                ),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["architecture_budget_sidecar"]
                ),
            },
            "combined_predictions": {
                "artifact": stage09b_descriptor(stage09b_artifacts["predictions"]),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["prediction_sidecar"]
                ),
            },
            "metric_summary": {
                "artifact": stage09b_descriptor(
                    stage09b_artifacts["metric_summary"]
                ),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["metric_summary_sidecar"]
                ),
            },
            "report": {
                "artifact": stage09b_descriptor(stage09b_artifacts["report"]),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["report_sidecar"]
                ),
            },
        },
    }
    semantic_audit["semantic_audit_self_sha256"] = verifier._sha256_json(
        semantic_audit
    )
    write_json(stage09b_artifacts["semantic_audit"], semantic_audit)
    _write_bytes(source, stage09b_artifacts["semantic_audit_sidecar"], b"{}\n")
    stage09b_receipt_path = "outputs/models/route_a_stage09b_completion.json"
    stage09b_receipt = {
        "format": "thermoroute.stage09b-completion-receipt.v3",
        "status": "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "stage": "09b_development_controls",
        "run_id": stage09b_run_id,
        "run_identity": {"run_id": stage09b_run_id},
        "formal_configuration": {"fixture": True},
        "evidence_scope": "best_model_state_prediction_replay",
        "training_replay_verified": False,
        "best_model_state_prediction_replay_verified": True,
        "matrix_audit": stage09b_matrix,
        "member_registry": stage09b_members,
        "artifacts": {
            label: _binding(verifier, source, relative)
            for label, relative in stage09b_artifacts.items()
        },
        "post_2020_outcomes_requested_or_read": False,
    }
    stage09b_receipt["receipt_self_sha256"] = verifier._sha256_json(
        stage09b_receipt
    )
    write_json(stage09b_receipt_path, stage09b_receipt)

    suite = {
        "format": "thermoroute.route-a-model-suite.v1",
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": runtime_sha256,
        "development_contract": {
            **{
                name: _binding(verifier, source, relative)
                for name, relative in development_paths.items()
            },
            "predictor_bridge": _binding(verifier, source, bridge_path),
            "source_sha256": frozen_source_sha,
        },
        "preopening_gates": {
            "stage09_completion": _binding(
                verifier, source, stage09_receipt_path
            ),
            "stage09b_development_controls": _binding(
                verifier, source, stage09b_receipt_path
            ),
        },
        "cohorts": {
            "temporal": {
                "models": [{"model_id": "Persistence", "executor": "builtin"}]
            },
            "external": {
                "models": [{
                    "model_id": "LightGBM",
                    "executor": "lightgbm_bundle",
                    "artifact": _binding(verifier, source, lgb_manifest_path),
                }]
            },
        },
    }
    write_json(model_suite_path, suite)
    write_json(
        development_replay_path,
        {
            "format": "thermoroute.route-a-development-replay.v1",
            "suite": _binding(verifier, source, model_suite_path),
            "source_tree_sha256": frozen_source_sha,
            "runtime_sha256": runtime_sha256,
        },
    )
    model_artifact_paths = {
        model_suite_path,
        development_replay_path,
        *development_paths.values(),
        prediction_path,
        prediction_sidecar,
        lgb_manifest_path,
        *lgb_model_paths.values(),
        bridge_path,
        *bridge_normalized.values(),
        bridge_report,
        bridge_request_map,
        *bridge_raw_paths,
        stage09_receipt_path,
        *stage09_artifacts.values(),
        stage09b_receipt_path,
        *stage09b_artifacts.values(),
        *stage09b_member_paths,
    }
    model_commit = commit("freeze executable models and chronology gate")
    model_tree_paths = {
        value
        for value in git_text("ls-tree", "-r", "--name-only", model_commit).splitlines()
        if verifier._is_model_control_path(value)
    }
    model_source_inventory = {
        relative: str(git_blob_binding(model_commit, relative)["sha256"])
        for relative in sorted(model_tree_paths)
        if verifier._matches_source_inventory(relative)
    }
    source_tree_sha256 = verifier._sha256_json(model_source_inventory)
    assert source_tree_sha256 == frozen_source_sha

    input_paths = {
        "candidate_table": "data_usgs/confirmatory_candidate_sites_v1.csv",
        "candidate_provenance": (
            "data_usgs/confirmatory_candidate_sites_v1.provenance.json"
        ),
        "candidate_snapshot_index": (
            "data_usgs/raw_snapshots/confirmatory-candidates-v1/"
            "snapshot_index.json"
        ),
        "external_registry": "data_usgs/confirmatory_site_registry_v1.csv",
        "external_lock": "data_usgs/confirmatory_site_registry_v1.lock.json",
        "input_manifest": "data_usgs/confirmatory_actual_inputs_v1.json",
    }
    _write_bytes(source, input_paths["candidate_table"], b"site_no\n9\n")
    write_json(input_paths["candidate_provenance"], {})
    candidate_index = input_paths["candidate_snapshot_index"]
    candidate_metadata = str(PurePosixPath(candidate_index).parent / "record.json")
    candidate_response = str(PurePosixPath(candidate_index).parent / "response.txt")
    write_json(candidate_metadata, {})
    _write_bytes(source, candidate_response, b"candidate metadata\n")
    write_json(
        candidate_index,
        {
            "records": [{
                "metadata_path": "record.json",
                "response_path": "response.txt",
                "response_sha256": verifier.sha256_file(source / candidate_response),
            }]
        },
    )
    _write_bytes(source, input_paths["external_registry"], b"site_no,lat,lon\n9,1,2\n")
    write_json(
        input_paths["external_lock"],
        {
            "status": "REGISTRY_FROZEN_LABELS_SEALED",
            "confirmatory_registry_sha256": verifier.sha256_file(
                source / input_paths["external_registry"]
            ),
            "frozen_artifacts": {
                "development_panel_spec": _binding(
                    verifier, source, development_paths["frozen_panel_spec"]
                ),
                "candidate_table": _binding(
                    verifier, source, input_paths["candidate_table"]
                ),
                "candidate_provenance": _binding(
                    verifier, source, input_paths["candidate_provenance"]
                ),
                "candidate_snapshot_index": _binding(
                    verifier, source, candidate_index
                ),
            },
        },
    )
    temporal_table = "data_usgs/confirmatory_predictors/temporal.parquet"
    external_table = "data_usgs/confirmatory_predictors/external.parquet"
    request_map = "data_usgs/confirmatory_predictors/source_request_map_v1.json"
    met_index = (
        "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1/daymet-v1/"
        "snapshot_index.json"
    )
    met_metadata = str(PurePosixPath(met_index).parent / "record.json")
    met_response = str(PurePosixPath(met_index).parent / "response.txt")
    for relative, payload in (
        (temporal_table, b"temporal\n"),
        (external_table, b"external\n"),
        (met_response, b"meteorology\n"),
    ):
        _write_bytes(source, relative, payload)
    write_json(request_map, {})
    write_json(met_metadata, {})
    write_json(
        met_index,
        {
            "records": [{
                "metadata_path": "record.json",
                "response_path": "response.txt",
                "response_sha256": verifier.sha256_file(source / met_response),
            }]
        },
    )
    write_json(
        input_paths["input_manifest"],
        {
            "format": "thermoroute.route-a-prelabel-inputs.v1",
            "status": "FROZEN_PRELABEL_NO_OUTCOMES",
            "contains_outcome": False,
            "contains_outcome_labels": False,
            "post_2020_wtemp_requested_or_inspected": False,
            "cohort_tables": {
                "temporal": _binding(verifier, source, temporal_table),
                "external": _binding(verifier, source, external_table),
            },
            "registry_inputs": {
                "temporal": _binding(
                    verifier, source, development_paths["registry"]
                ),
                "external": _binding(
                    verifier, source, input_paths["external_registry"]
                ),
            },
            "source_evidence": [
                {
                    "evidence_type": "snapshot_index",
                    "contains_outcome": False,
                    "contains_outcome_labels": False,
                    "artifact": _binding(verifier, source, met_index),
                },
                {
                    "evidence_type": "normalized_immutable_snapshot",
                    "contains_outcome": False,
                    "contains_outcome_labels": False,
                    "artifact": _binding(verifier, source, request_map),
                },
            ],
        },
    )
    input_artifact_paths = {
        *input_paths.values(),
        development_paths["frozen_panel_spec"],
        development_paths["registry"],
        candidate_metadata,
        candidate_response,
        temporal_table,
        external_table,
        request_map,
        met_index,
        met_metadata,
        met_response,
    }
    input_commit = commit("freeze outcome-free input evidence")

    _write_bytes(
        source,
        "outputs/prelabel/chronology_receipt_base.marker",
        b"receipt creation base\n",
    )
    receipt_base_commit = commit("establish chronology receipt base")

    authorization_path = "data_usgs/confirmatory_opening_authorization_v1.json"
    chronology_order = {
        "model_freeze_commit": model_commit,
        "input_evidence_commit": input_commit,
        "receipt_creation_base_commit": receipt_base_commit,
        "strict_order_verified": True,
    }
    chronology_paths = {
        "protocol_seal": verifier.PROTOCOL_SEAL_PATH,
        "model_suite": model_suite_path,
        "development_replay": development_replay_path,
        **input_paths,
    }
    chronology = {
        "format": verifier.CHRONOLOGY_FORMAT,
        "status": verifier.CHRONOLOGY_STATUS,
        "order": chronology_order,
        "protocol_history": {
            "seal": git_blob_binding(model_commit, verifier.PROTOCOL_SEAL_PATH),
            "original_commit": original_commit,
            "final_prelabel_commit": final_commit,
            "declared_git_show_bindings": [
                {
                    "role": "original_markdown",
                    "commit": original_commit,
                    "path": protocol_markdown_path,
                    "sha256": hashlib.sha256(original_markdown).hexdigest(),
                },
                {
                    "role": "final_json",
                    "commit": final_commit,
                    "path": protocol_json_path,
                    "sha256": hashlib.sha256(final_protocol.read_bytes()).hexdigest(),
                },
                {
                    "role": "final_markdown",
                    "commit": final_commit,
                    "path": protocol_markdown_path,
                    "sha256": hashlib.sha256(final_markdown.read_bytes()).hexdigest(),
                },
            ],
        },
        "paths": chronology_paths,
        "required_gate_files_at_model_freeze": [
            git_blob_binding(model_commit, relative) for relative in gate_paths
        ],
        "model_source_control_artifacts": [
            git_blob_binding(model_commit, relative)
            for relative in sorted(model_tree_paths)
        ],
        "source_tree_sha256": source_tree_sha256,
        "model_freeze_artifacts": [
            git_blob_binding(model_commit, relative)
            for relative in sorted(model_artifact_paths)
        ],
        "input_evidence_artifacts": [
            git_blob_binding(input_commit, relative)
            for relative in sorted(input_artifact_paths)
        ],
        "absence_at_model_freeze": {
            "checked_paths": sorted(
                [*input_paths.values(), authorization_path, verifier.CHRONOLOGY_PATH]
            ),
            "present_paths": [],
        },
        "post_model_control_audit": {
            "protected_directories": list(verifier.PROTECTED_DIRECTORIES),
            "protected_exact_files": list(verifier.PROTECTED_EXACT_FILES),
            "protected_root_patterns": list(verifier.PROTECTED_ROOT_PATTERNS),
            "committed_touches": [],
            "worktree_changes": [],
        },
        "post_freeze_artifact_mutation_count": 0,
        "external_timestamp_or_public_preregistration": False,
        "independent_custodian_or_worm_storage": False,
        "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        "fallback_if_validation_fails": (
            "TRANSDUCTIVE_RETROSPECTIVE_EXPLORATION_CONFIRMATION_CLAIMS_PROHIBITED"
        ),
    }
    chronology["receipt_self_sha256"] = verifier._chronology_self_sha256(
        chronology
    )
    write_json(verifier.CHRONOLOGY_PATH, chronology)
    compute_commit = commit("freeze chronology receipt")

    def fixed_binding(relative: str) -> dict[str, str]:
        return {
            "path": relative,
            "realpath": str((source / relative).resolve()),
            "sha256": verifier.sha256_file(source / relative),
        }

    fixed_stable = {
        "modules": {
            name: fixed_binding(relative)
            for name, relative in fixed_modules.items()
        },
        "files": {
            name: fixed_binding(relative)
            for name, relative in fixed_files.items()
        },
        "entrypoints": {
            name: fixed_binding(relative)
            for name, relative in fixed_entrypoints.items()
        },
    }

    # Production authorization is create-only Git dirt after the compute
    # commit.  Keep it untracked while committing the later manuscript bytes.
    authorization = {
        "protocol": {
            "authoritative_commit": original_commit,
            "authoritative_markdown_sha256": hashlib.sha256(
                original_markdown
            ).hexdigest(),
            "final_prelabel_commit": final_commit,
            "seal": _binding(verifier, source, verifier.PROTOCOL_SEAL_PATH),
        },
        "registries": {
            "candidate_table": {"path": input_paths["candidate_table"]},
            "candidate_provenance": {"path": input_paths["candidate_provenance"]},
            "candidate_snapshot_index": {
                "path": input_paths["candidate_snapshot_index"]
            },
            "external": {"path": input_paths["external_registry"]},
            "external_lock": {"path": input_paths["external_lock"]},
        },
        "model_suite": {"path": model_suite_path},
        "development_replay": {"path": development_replay_path},
        "actual_inputs": {"path": input_paths["input_manifest"]},
        "prelabel_chronology": {
            **_binding(verifier, source, verifier.CHRONOLOGY_PATH),
            "format": verifier.CHRONOLOGY_FORMAT,
            "status": verifier.CHRONOLOGY_STATUS,
            "order": chronology_order,
            "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        },
        "inference_amendment": {
            **_binding(verifier, source, inference_amendment_path),
            "seal": _binding(verifier, source, inference_amendment_seal_path),
            "final_prelabel_commit": amendment_commit,
        },
        "inference_gate": _binding(verifier, source, inference_gate_path),
        "runtime": {
            "requirements_lock": _binding(
                verifier, source, "requirements-lock.txt"
            ),
            "hashed_requirements_lock": _binding(
                verifier, source, verifier.REPRODUCIBILITY_LOCK
            ),
        },
        "fixed_code": {
            "format": "thermoroute.route-a-fixed-code.v1",
            **fixed_stable,
            "sha256": verifier._sha256_json(fixed_stable),
        },
        "source": {
            "git_commit_before_authorization": compute_commit,
            "source_tree_sha256": source_tree_sha256,
            "source_inventory": model_source_inventory,
        },
    }
    write_json(authorization_path, authorization)

    readme = _write_bytes(source, "README.md", b"post-opening manuscript only\n")
    subprocess.run(["git", "add", "README.md"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "render post-opening manuscript"],
        cwd=source,
        env=environment,
        check=True,
    )
    manuscript_commit = git_text("rev-parse", "HEAD")
    assert len({
        original_commit,
        final_commit,
        amendment_commit,
        seal_commit,
        model_commit,
        input_commit,
        receipt_base_commit,
        compute_commit,
        manuscript_commit,
    }) == 9

    stage = tmp_path / "chronology-stage"
    shutil.copytree(source, stage, ignore=shutil.ignore_patterns(".git"))
    document_binding = {
        **_binding(verifier, stage, "README.md"),
        "bytes": readme.stat().st_size,
    }
    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps({
            "profile": verifier.POSTOPEN_PROFILE,
            "authorization": {"path": authorization_path},
            "authorized_worktree_dirt_policy": {
                "compute_commit": compute_commit,
                "manuscript_commit": manuscript_commit,
                "committed_document_diff": [document_binding],
            },
        }, sort_keys=True).encode() + b"\n",
    )

    evidence = verifier.materialize_git_history_evidence(
        source, stage, verifier.POSTOPEN_PROFILE
    )
    receipt_oid = git_text(
        "rev-parse", f"{compute_commit}:{verifier.CHRONOLOGY_PATH}"
    )
    assert evidence["prelabel_chronology"] == {
        "receipt": {
            **_binding(verifier, stage, verifier.CHRONOLOGY_PATH),
            "bytes": (stage / verifier.CHRONOLOGY_PATH).stat().st_size,
        },
        "receipt_commit": compute_commit,
        "receipt_git_blob_oid": receipt_oid,
        "order": chronology_order,
        "model_source_control_artifact_count": len(model_tree_paths),
        "model_freeze_artifact_count": len(model_artifact_paths),
        "input_evidence_artifact_count": len(input_artifact_paths),
    }

    relocated = tmp_path / "relocated" / "release"
    shutil.copytree(stage, relocated)
    marker = json.loads(
        (relocated / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    verifier._verify_git_history_evidence(
        relocated, marker, verifier.POSTOPEN_PROFILE
    )

    # Recomputing mutable archive JSON hashes after replacing executable code
    # must not defeat the immutable compute-commit blob comparison.
    attacked = tmp_path / "attacked-release"
    shutil.copytree(relocated, attacked)
    attacked_opening = attacked / "src/thermoroute/opening.py"
    attacked_opening.write_bytes(b"# attacker-replaced executable\n")
    attacked_sha = verifier.sha256_file(attacked_opening)
    attacked_chronology_path = attacked / verifier.CHRONOLOGY_PATH
    attacked_chronology = json.loads(
        attacked_chronology_path.read_text(encoding="utf-8")
    )
    for binding in attacked_chronology["model_source_control_artifacts"]:
        if binding["path"] == "src/thermoroute/opening.py":
            payload = attacked_opening.read_bytes()
            binding["sha256"] = attacked_sha
            binding["byte_count"] = len(payload)
            binding["git_blob_oid"] = hashlib.sha1(
                f"blob {len(payload)}\0".encode() + payload
            ).hexdigest()
    attacked_inventory = {
        item["path"]: item["sha256"]
        for item in attacked_chronology["model_source_control_artifacts"]
        if verifier._matches_source_inventory(item["path"])
    }
    attacked_tree_sha = verifier._sha256_json(attacked_inventory)
    attacked_chronology["source_tree_sha256"] = attacked_tree_sha
    attacked_chronology.pop("receipt_self_sha256")
    attacked_chronology["receipt_self_sha256"] = (
        verifier._chronology_self_sha256(attacked_chronology)
    )
    attacked_chronology_path.write_text(
        json.dumps(attacked_chronology, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attacked_authorization_path = attacked / authorization_path
    attacked_authorization = json.loads(
        attacked_authorization_path.read_text(encoding="utf-8")
    )
    attacked_authorization["prelabel_chronology"]["sha256"] = (
        verifier.sha256_file(attacked_chronology_path)
    )
    attacked_authorization["source"]["source_inventory"] = attacked_inventory
    attacked_authorization["source"]["source_tree_sha256"] = attacked_tree_sha
    attacked_authorization["fixed_code"]["modules"][
        "thermoroute.opening"
    ]["sha256"] = attacked_sha
    attacked_fixed_stable = {
        group: attacked_authorization["fixed_code"][group]
        for group in ("modules", "files", "entrypoints")
    }
    attacked_authorization["fixed_code"]["sha256"] = verifier._sha256_json(
        attacked_fixed_stable
    )
    attacked_authorization_path.write_text(
        json.dumps(attacked_authorization, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attacked_marker = json.loads(
        (attacked / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="compute Git blob"):
        verifier._verify_git_history_evidence(
            attacked, attacked_marker, verifier.POSTOPEN_PROFILE
        )

    added = tmp_path / "added-source-release"
    shutil.copytree(relocated, added)
    added_source = _write_bytes(
        added, "src/thermoroute/injected.py", b"INJECTED = True\n"
    )
    added_authorization_path = added / authorization_path
    added_authorization = json.loads(
        added_authorization_path.read_text(encoding="utf-8")
    )
    added_inventory = dict(added_authorization["source"]["source_inventory"])
    added_inventory["src/thermoroute/injected.py"] = verifier.sha256_file(
        added_source
    )
    added_authorization["source"]["source_inventory"] = added_inventory
    added_authorization["source"]["source_tree_sha256"] = verifier._sha256_json(
        added_inventory
    )
    added_authorization_path.write_text(
        json.dumps(added_authorization, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    added_marker = json.loads(
        (added / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="path set differs from compute Git tree"):
        verifier._verify_git_history_evidence(
            added, added_marker, verifier.POSTOPEN_PROFILE
        )

    tampered = json.loads(json.dumps(marker))
    tampered["git_history_evidence"]["prelabel_chronology"]["order"][
        "model_freeze_commit"
    ] = input_commit
    with pytest.raises(ValueError, match="chronology evidence differs from its receipt"):
        verifier._verify_git_history_evidence(
            relocated, tampered, verifier.POSTOPEN_PROFILE
        )


def test_opening_identity_is_portable_in_real_python_i_process(tmp_path):
    relocated = tmp_path / "different-absolute-root" / "release"
    shutil.copytree(ROOT / "src", relocated / "src")
    scripts = relocated / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "route_a_opening_orchestrator.py",
        "route_a_outcome_acquisition.py",
        "route_a_trusted_scorer.py",
    ):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    program = r"""
import copy
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
from thermoroute.opening import (
    OpeningContractError,
    _fixed_code_identity,
    _validate_portable_fixed_code_identity,
    _validate_portable_runtime_identity,
    sha256_json,
)

assert sys.flags.isolated
current = _fixed_code_identity(root)
frozen = copy.deepcopy(current)
for group in ("modules", "files", "entrypoints"):
    for binding in frozen[group].values():
        binding["realpath"] = "/sealed-on-another-machine/" + binding["path"]
stable = {group: frozen[group] for group in ("modules", "files", "entrypoints")}
frozen["sha256"] = sha256_json(stable)
_validate_portable_fixed_code_identity(frozen, current)

runtime = {
    "format": "fixture-runtime",
    "python_executable": {
        "invoked_path": sys.executable,
        "realpath": str(Path(sys.executable).resolve()),
        "sha256": "a" * 64,
    },
    "stable_contract": {"cpu": True},
}
frozen_runtime = copy.deepcopy(runtime)
frozen_runtime["python_executable"]["invoked_path"] = "/sealed/python"
frozen_runtime["python_executable"]["realpath"] = "/sealed/python-real"
_validate_portable_runtime_identity(frozen_runtime, runtime)

tampered = copy.deepcopy(frozen)
tampered["entrypoints"]["trusted_scorer"]["sha256"] = "0" * 64
tampered["sha256"] = sha256_json({
    group: tampered[group] for group in ("modules", "files", "entrypoints")
})
try:
    _validate_portable_fixed_code_identity(tampered, current)
except OpeningContractError:
    pass
else:
    raise AssertionError("relocation policy accepted changed trusted-scorer bytes")
print(json.dumps({"isolated": True, "relocated_root": str(root)}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(relocated)],
        cwd=relocated,
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
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status == {"isolated": True, "relocated_root": str(relocated.resolve())}


def test_release_materialization_rejects_hidden_git_index_flags(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_hidden_index_release_test"
    )
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    _minimal_canonical_release(verifier, source)
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", "fixture",
        ],
        cwd=source,
        check=True,
    )
    relative = "requirements-lock-py312-hashed.txt"
    for enable, disable in (
        ("--assume-unchanged", "--no-assume-unchanged"),
        ("--skip-worktree", "--no-skip-worktree"),
    ):
        subprocess.run(
            ["git", "update-index", enable, relative], cwd=source, check=True
        )
        with pytest.raises(
            ValueError, match="assume-unchanged/skip-worktree"
        ):
            verifier.materialize_release_profile(
                source, stage, verifier.PREOPEN_PROFILE
            )
        subprocess.run(
            ["git", "update-index", disable, relative], cwd=source, check=True
        )
    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps({"profile": verifier.PREOPEN_PROFILE}).encode(),
    )
    dirty = _write_bytes(source, "untracked-release-dirt.txt")
    with pytest.raises(ValueError, match="clean Git worktree"):
        verifier.materialize_git_history_evidence(
            source, stage, verifier.PREOPEN_PROFILE
        )
    dirty.unlink()


def test_release_git_audit_rejects_history_overlays_and_ambient_redirects(
    tmp_path, monkeypatch
):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_git_overlay_test")
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })
    _write_bytes(root, "tracked.txt", b"one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "one"], cwd=root,
        env=environment, check=True,
    )
    (root / "tracked.txt").write_bytes(b"two\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "two"], cwd=root,
        env=environment, check=True,
    )
    verifier._assert_safe_git_repository(root)

    subprocess.run(["git", "replace", "HEAD", "HEAD^"], cwd=root, check=True)
    with pytest.raises(ValueError, match="replacement refs"):
        verifier._assert_safe_git_repository(root)
    subprocess.run(["git", "replace", "-d", "HEAD"], cwd=root, check=True)

    graft = root / ".git/info/grafts"
    graft.parent.mkdir(parents=True, exist_ok=True)
    graft.write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy grafts"):
        verifier._assert_safe_git_repository(root)
    graft.unlink()

    alternates = root / ".git/objects/info/alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_text("/tmp/forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="object alternates"):
        verifier._assert_safe_git_repository(root)
    alternates.unlink()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    shallow = root / ".git/shallow"
    shallow.write_text(head + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="shallow"):
        verifier._assert_safe_git_repository(root)
    shallow.unlink()

    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    with pytest.raises(ValueError, match="ambient Git"):
        verifier._assert_safe_git_repository(root)


def test_postopen_git_dirt_allows_only_authorization_and_canonical_namespace(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_postopen_dirt_test")
    root = tmp_path / "repo"
    root.mkdir()
    tracked = _write_bytes(root, "tracked.txt")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=root,
        env=environment,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    namespace = "1" * 24
    base = f"outputs/confirmatory/route_a_{namespace}"
    state = {
        "namespace": namespace,
        "run_directory": base,
        "work_order": f"{base}/acquisition_work_order_v1.json",
        "intent": f"{base}/opening_intent_v1.json",
        "transport_root": f"{base}/transport",
        "raw_nwis_root": f"{base}/transport/raw_nwis_v1",
        "raw_nwis_snapshot_index": (
            f"{base}/transport/raw_nwis_v1/snapshot_index.json"
        ),
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
    authorization_relative = "data_usgs/confirmatory_opening_authorization_v1.json"
    authorization = {
        "format": verifier.AUTHORIZATION_FORMAT,
        "status": "AUTHORIZED_LABELS_STILL_SEALED",
        "opening_id": "2" * 24,
        "source": {
            "authorization_path": authorization_relative,
            "git_commit_before_authorization": head,
        },
        "outcome_qc_policy": {
            "path": "protocols/route_a_outcome_qc_policy_v1.json",
            "sha256": "7" * 64,
            "format": "thermoroute.route-a-outcome-qc-policy.v1",
            "policy_id": "route-a-outcome-qc-and-influence-001",
            "required": True,
        },
        "temporal_coverage_policy": {
            "path": verifier.TEMPORAL_COVERAGE_POLICY_PATH,
            "sha256": verifier.TEMPORAL_COVERAGE_POLICY_SHA256,
            "format": verifier.TEMPORAL_COVERAGE_POLICY_FORMAT,
            "policy_id": verifier.TEMPORAL_COVERAGE_POLICY_ID,
            "status": "FROZEN_PRELABEL_OUTCOME_FREE",
            "required": True,
        },
        "inference_amendment": {
            "path": "protocols/route_a_inference_amendment_v1.json",
            "sha256": "8" * 64,
            "format": "thermoroute.route-a-inference-amendment.v1",
            "amendment_id": "route-a-prelabel-inference-scope-014",
            "seal": {
                "path": "protocols/route_a_inference_amendment_seal_v1.json",
                "sha256": "9" * 64,
            },
            "final_prelabel_commit": head,
        },
        "inference_gate": {
            "path": "outputs/prelabel/route_a_inference_gate_v1.json",
            "sha256": "a" * 64,
            "format": "thermoroute.route-a-inference-gate.v1",
            "status": "FAIL_CLOSED_DESCRIPTIVE_ONLY",
            "claim_eligible": False,
            "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
            "policy_sha256": "b" * 64,
        },
        "runtime": {
            "format": "thermoroute.route-a-runtime.v1",
            "requirements_lock": {
                "path": "requirements-lock.txt",
                "sha256": "3" * 64,
            },
            "hashed_requirements_lock": {
                "path": verifier.REPRODUCIBILITY_LOCK,
                "sha256": "6" * 64,
            },
            "installed_version_validation": {},
            "numerical_runtime_contract": {},
            "runtime_sha256": "4" * 64,
            "python_executable": {},
            "golden_inference_sha256": "5" * 64,
            "formal_numerical_policy": {},
            "deterministic_child_policy": {},
        },
        "state_paths": state,
    }
    authorization["authorization_self_sha256"] = verifier._sha256_json(authorization)
    authorization_path = root / authorization_relative
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    namespace_artifact = _write_bytes(root, f"{base}/opening_receipt_v1.json")

    policy = verifier.validate_postopen_git_dirt(root, authorization_path)
    assert policy["untracked_exact"] == [authorization_relative]
    assert policy["untracked_prefixes"] == [base + "/"]

    readme = _write_bytes(root, "README.md", b"post-opening manuscript\n")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "docs after opening"],
        cwd=root, env=environment, check=True,
    )
    doc_policy = verifier.validate_postopen_git_dirt(root, authorization_path)
    assert doc_policy["compute_commit"] == head
    assert doc_policy["manuscript_commit"] != head
    assert [row["path"] for row in doc_policy["committed_document_diff"]] == [
        "README.md"
    ]
    assert readme.is_file()

    hidden = _write_bytes(root, "src/hidden_then_reverted.py", b"hidden = True\n")
    subprocess.run(["git", "add", "src/hidden_then_reverted.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "forbidden hidden source"],
        cwd=root, env=environment, check=True,
    )
    hidden.unlink()
    subprocess.run(
        ["git", "add", "-u", "--", "src/hidden_then_reverted.py"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "revert forbidden hidden source"],
        cwd=root, env=environment, check=True,
    )
    with pytest.raises(ValueError, match="outside the documentation whitelist"):
        verifier.validate_postopen_git_dirt(root, authorization_path)

    extra = _write_bytes(root, "unexpected.txt")
    with pytest.raises(ValueError, match="extra Git dirt"):
        verifier.validate_postopen_git_dirt(root, authorization_path)
    extra.unlink()
    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked worktree"):
        verifier.validate_postopen_git_dirt(root, authorization_path)
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "forbidden compute edit"],
        cwd=root, env=environment, check=True,
    )
    with pytest.raises(ValueError, match="outside the documentation whitelist"):
        verifier.validate_postopen_git_dirt(root, authorization_path)
    assert namespace_artifact.is_file()


def test_real_preopen_materialization_keeps_frozen_panel_source_metadata_closed(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_real_panel_closure_test")
    stage = tmp_path / "stage"
    stage.mkdir()
    verifier.materialize_release_profile(ROOT, stage, verifier.PREOPEN_PROFILE)
    command = [
        sys.executable,
        "-c",
        (
            "from thermoroute.evidence import FrozenPanelSpec; "
            "e=FrozenPanelSpec.load(r'"
            + str(stage / "data_usgs" / "frozen_panel_v1.json")
            + "').verify(); assert e['station_count']==120"
        ),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        command,
        cwd=stage,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
