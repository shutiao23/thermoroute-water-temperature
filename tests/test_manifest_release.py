from __future__ import annotations

import csv
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPT = ROOT / "scripts" / "14_manifest.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_release.py"
ZIP_SCRIPT = ROOT / "scripts" / "deterministic_zip.py"


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


def _binding(verifier, root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    return {"path": relative, "sha256": verifier.sha256_file(path)}


def _chronology_binding(verifier, root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        **_binding(verifier, root, relative),
        "byte_count": path.stat().st_size,
        "git_blob_oid": "a" * 40,
    }


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
    _write_bytes(root, "requirements-lock.txt", b"numpy==1.0\n")
    _write_protocol_seal_fixture(verifier, root)
    _write_bytes(root, "data_usgs/external.csv")
    _write_bytes(root, "data_usgs/external.lock.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidates.csv")
    _write_bytes(root, "data_usgs/candidates.provenance.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/snapshot_index.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/response.rdb")
    _write_bytes(root, "src/thermoroute/opening.py", b"# fixed\n")
    _write_bytes(root, "src/thermoroute/chronology.py", b"# chronology gate\n")
    _write_bytes(
        root, "scripts/28_freeze_prelabel_chronology.py", b"# chronology gate\n"
    )
    _write_bytes(root, "tests/test_chronology.py", b"# chronology gate\n")
    _write_bytes(root, "scripts/route_a_trusted_scorer.py", b"# fixed\n")
    _write_claim_fixture_files(root)

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
        identity = {
            "run_id": "stage09b-fixture",
            "panel_sha256": bridge["panel"]["sha256"],
            "registry_sha256": bridge["registry"]["sha256"],
            "config_sha256": "d" * 64,
            "source_sha256": source_sha256,
            "runtime_sha256": runtime_sha256,
            "schema_version": "thermoroute.run.v1",
        }
        expected_members = verifier._stage09b_release_members()
        controls_config = {
            "stage": "09b_development_controls",
            "training_device": "cpu",
            "panel_date_range": ["2006-01-01", "2020-12-31"],
            "blind_or_confirmatory": False,
            "suite_pointer_written": False,
            "expected_member_registry": [
                [arm, seed] for arm, seed in expected_members
            ],
            "development_predictor_bridge": _binding(verifier, root, bridge_path),
        }
        run_manifest_path = "outputs/runs/09b-fixture/run.json"
        _write_bytes(root, run_manifest_path, json.dumps({
            "identity": identity,
            "resolved_config": controls_config,
        }).encode())
        member_registry = []
        for arm_id, seed in expected_members:
            relative = f"outputs/runs/09b-fixture/members/{arm_id}/seed{seed}.parquet"
            prediction = _write_bytes(root, relative, f"{arm_id}/{seed}\n".encode())
            sidecar_relative = relative + ".meta.json"
            _write_bytes(root, sidecar_relative, json.dumps({
                "kind": "development_control_arm_predictions",
                "artifact_sha256": verifier.sha256_file(prediction),
                "run": identity,
                "extra": {
                    "arm_id": arm_id,
                    "seed": seed,
                    "training_device": "cpu",
                    "development_only": True,
                    "blind_or_confirmatory": False,
                },
            }).encode())
            member_registry.append({
                "arm_id": arm_id,
                "seed": seed,
                "prediction": _binding(verifier, root, relative),
                "prediction_sidecar": _binding(verifier, root, sidecar_relative),
            })
        final_paths = {
            "predictions": "outputs/runs/09b-fixture/controls.parquet",
            "architecture_budget": "outputs/runs/09b-fixture/budget.csv",
            "report": "outputs/runs/09b-fixture/report.md",
        }
        _write_bytes(root, final_paths["predictions"], b"combined fixture\n")
        parameters = {
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
        budget_buffer = io.StringIO()
        writer = csv.DictWriter(budget_buffer, fieldnames=(
            "arm_id", "trainable_parameters", "training_device",
            "historical_tuning_budget_equalized",
        ))
        writer.writeheader()
        for arm_id, count in parameters.items():
            writer.writerow({
                "arm_id": arm_id,
                "trainable_parameters": count,
                "training_device": "cpu",
                "historical_tuning_budget_equalized": False,
            })
        _write_bytes(
            root, final_paths["architecture_budget"],
            budget_buffer.getvalue().encode(),
        )
        _write_bytes(root, final_paths["report"], b"complete controls fixture\n")
        final_kinds = {
            "predictions": "development_controls_combined_predictions",
            "architecture_budget": "development_controls_budget",
            "report": "development_controls_report",
        }
        for name, relative in final_paths.items():
            _write_bytes(root, relative + ".meta.json", json.dumps({
                "kind": final_kinds[name],
                "artifact_sha256": verifier.sha256_file(root / relative),
                "run": identity,
                "extra": {
                    "expected_members": 31,
                    "development_only": True,
                    "blind_or_confirmatory": False,
                },
            }).encode())
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
            "report_sidecar": _binding(
                verifier, root, final_paths["report"] + ".meta.json"
            ),
        }
        controls = {
            "format": "thermoroute.stage09b-completion-receipt.v1",
            "status": "PASS_FORMAL_STAGE09B_CONTROLS_COMPLETE",
            "stage": "09b_development_controls",
            "run_id": identity["run_id"],
            "run_identity": identity,
            "formal_configuration": controls_config,
            "matrix_audit": {
                "expected_members": 31,
                "prediction_rows": 31 * 9,
                "common_forecast_keys": 9,
                "splits": ["calib", "test", "val"],
                "reference_member": "PlainMLP-7var/seed0",
            },
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
                "scripts/28_freeze_prelabel_chronology.py",
                "tests/test_chronology.py",
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
    work_order = {"format": "fixture-work-order"}
    work_order["work_order_self_sha256"] = verifier._sha256_json(work_order)
    _write_bytes(
        root, state["work_order"], json.dumps(work_order, sort_keys=True).encode()
    )
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
            cohort: sorted(verifier._required_model_ids(cohort))
            for cohort in ("temporal", "external")
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
        "trusted_validator": trusted_validator,
    }
    intent["intent_self_sha256"] = verifier._sha256_json(intent)
    (root / state["intent"]).write_text(json.dumps(intent), encoding="utf-8")

    raw_index = f"{state['raw_nwis_root']}/snapshot_index.json"
    _write_bytes(root, raw_index, b"{}\n")
    _write_bytes(root, f"{state['raw_nwis_root']}/response.rdb")
    request_ledger = f"{base}/acquisition/request_ledger_v1.json"
    attempt_index = f"{base}/acquisition/transport_attempt_index_v1.json"
    _write_bytes(root, request_ledger, b"{}\n")
    _write_bytes(root, attempt_index, b"{}\n")
    _write_bytes(root, state["acquisition_request_map"], b"{}\n")
    _write_bytes(root, state["temporal_outcomes"])
    _write_bytes(root, state["external_outcomes"])
    acquisition = {
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "labels_state": "OPENED_ONCE",
        "site_replacement_count": 0,
        "response_replacement_count": 0,
        "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
        "request_ledger": _binding(verifier, root, request_ledger),
        "transport_attempt_index": _binding(verifier, root, attempt_index),
        "raw_nwis_snapshot_index": _binding(verifier, root, raw_index),
        "request_map": _binding(verifier, root, state["acquisition_request_map"]),
        "normalized_outcome_tables": {
            "temporal": _binding(verifier, root, state["temporal_outcomes"]),
            "external": _binding(verifier, root, state["external_outcomes"]),
        },
    }
    (root / state["acquisition_manifest"]).write_text(
        json.dumps(acquisition), encoding="utf-8"
    )
    _write_bytes(root, state["availability_registry"])
    _write_bytes(root, state["outcome_quality_audit"], b"{}\n")
    _write_bytes(root, state["approved_target_sensitivity"], b"{}\n")
    _write_bytes(root, state["spatial_sensitivity"], b"{}\n")
    _write_bytes(root, state["probabilistic_evaluation"], b"{}\n")
    _write_bytes(root, state["temporal_predictions"])
    _write_bytes(root, state["external_predictions"])
    tests = [{"test_id": f"T{index}"} for index in range(1, 6)]
    (root / state["statistics"]).write_text(json.dumps({
        "format": verifier.STATISTICS_FORMAT,
        "tests": tests,
    }), encoding="utf-8")
    _write_bytes(root, state["report"], b"# trusted report\n")
    receipt_artifacts = {
        "acquisition_manifest": _binding(verifier, root, state["acquisition_manifest"]),
        "raw_nwis_snapshot_index": _binding(verifier, root, raw_index),
        "acquisition_request_map": _binding(verifier, root, state["acquisition_request_map"]),
        "temporal_normalized_outcomes": _binding(verifier, root, state["temporal_outcomes"]),
        "external_normalized_outcomes": _binding(verifier, root, state["external_outcomes"]),
        "availability_registry": _binding(verifier, root, state["availability_registry"]),
        "outcome_quality_audit": _binding(verifier, root, state["outcome_quality_audit"]),
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
        "report": _binding(verifier, root, state["report"]),
    }
    release_artifacts = {
        key: {"format": "fixture-format", **binding}
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
        "opening_count": 1,
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "all_predeclared_models_reported": True,
        "reported_models": {
            cohort: sorted(values)
            for cohort, values in authorization["required_models"].items()
        },
        "trusted_validator": trusted_validator,
        "artifacts": receipt_artifacts,
        "formal_tests": tests,
        "state_paths": state,
        "release_bindings": release_bindings,
        "intent_self_sha256": intent["intent_self_sha256"],
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
        "registries": "data_usgs/external.csv",
        "candidate_evidence": "data_usgs/candidates.csv",
        "model_suite": "data_usgs/confirmatory_model_suite_v1.json",
        "model_bundles": "outputs/models/temporal/LSTM.bundle",
        "prelabel_chronology": verifier.CHRONOLOGY_PATH,
        "prelabel_inputs": "data_usgs/prelabel/temporal.parquet",
        "raw_meteorology": "data_usgs/raw_snapshots/met-0/response.bin",
        "opening_intent": state["intent"],
        "raw_nwis": f"{state['raw_nwis_root']}/response.rdb",
        "normalized_outcomes": state["temporal_outcomes"],
        "trusted_predictions": state["temporal_predictions"],
        "availability": state["availability_registry"],
        "sensitivity_audits": state["outcome_quality_audit"],
        "probabilistic_evaluation": state["probabilistic_evaluation"],
        "statistics": state["statistics"],
        "report": state["report"],
        "receipt": state["receipt_sha256"],
        "environment_attestations": "requirements-lock.txt",
        "reproducibility_lock": verifier.REPRODUCIBILITY_LOCK,
    }
    return authorization_path, representatives


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


def test_manifest_refuses_unsealed_legacy_usgs_current_truth(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    legacy = tmp_path / "outputs" / "predictions" / "usgs_predictions_v2.parquet"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-unsealed-bytes")
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


def test_postopen_profile_closes_every_required_category_and_missing_file_fails(tmp_path):
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
    assert set(document["artifact_closure"]) == verifier.REQUIRED_POSTOPEN_CATEGORIES
    assert verifier.verify_release_profile(
        stage, run_trusted_replay=False
    ) == verifier.POSTOPEN_PROFILE

    for category, relative in representatives.items():
        artifact = stage / relative
        payload = artifact.read_bytes()
        artifact.unlink()
        with pytest.raises(ValueError, match="absent|closure|missing|lacks|cannot read"):
            verifier.verify_release_profile(stage, run_trusted_replay=False)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(payload)
        assert category in document["artifact_closure"]
    verifier.verify_release_profile(stage, run_trusted_replay=False)
    stale = _write_bytes(stage, "outputs/tables/usgs_scores_old_cohort.csv")
    with pytest.raises(ValueError, match="outside the authorization closure"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    stale.unlink()


def test_release_verifier_requires_both_receipts_and_exact_control_members(tmp_path):
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
    verifier._validate_preopening_completion_gates(
        source, {}, suite, development, suite["numerical_runtime_sha256"]
    )

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
    ].append({"path": ".zenodo.json", "sha256": "1" * 64, "bytes": 1})
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
        "scripts/28_freeze_prelabel_chronology.py",
        "tests/test_chronology.py",
    )
    for relative in gate_paths:
        _write_bytes(source, relative, f"# frozen gate: {relative}\n".encode())
    fixed_modules = {
        "thermoroute.opening": "src/thermoroute/opening.py",
        "thermoroute.model_suite": "src/thermoroute/model_suite.py",
        "thermoroute.frozen_inference": "src/thermoroute/frozen_inference.py",
        "thermoroute.datasets": "src/thermoroute/datasets.py",
        "thermoroute.usgs": "src/thermoroute/usgs.py",
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
        "panel": "data_usgs/panel.parquet",
        "registry": "data_usgs/station_registry_v1.csv",
    }
    for relative in development_paths.values():
        _write_bytes(source, relative, b"{}\n" if relative.endswith(".json") else b"dev\n")
    prediction_path = "outputs/development/predictions.parquet"
    prediction_sidecar = prediction_path + ".meta.json"
    _write_bytes(source, prediction_path, b"predictions\n")
    _write_bytes(source, prediction_sidecar, b"{}\n")
    lgb_model_path = "outputs/models/lgb/member_h1_point.txt"
    _write_bytes(source, lgb_model_path, b"tree\n")
    lgb_manifest_path = "outputs/models/lgb/manifest.json"
    runtime_sha256 = "e" * 64
    write_json(
        lgb_manifest_path,
        {
            "format": "thermoroute.lightgbm-bundle.v1",
            "training_device": "cpu",
            "runtime_sha256": runtime_sha256,
            "models": {
                "seed0": {
                    "1": {
                        "point": {
                            "path": "member_h1_point.txt",
                            "sha256": verifier.sha256_file(source / lgb_model_path),
                        }
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
            f"{name}/snapshot_index.json"
        )
        metadata = str(PurePosixPath(index).parent / "metadata.json")
        response = str(PurePosixPath(index).parent / "response.bin")
        write_json(metadata, {})
        _write_bytes(source, response, f"{name} raw\n".encode())
        write_json(
            index,
            {
                "records": [{
                    "metadata_path": "metadata.json",
                    "response_path": "response.bin",
                    "response_sha256": verifier.sha256_file(source / response),
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
        lgb_model_path,
        bridge_path,
        *bridge_normalized.values(),
        bridge_report,
        bridge_request_map,
        *bridge_raw_paths,
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
        model_commit,
        input_commit,
        receipt_base_commit,
        compute_commit,
        manuscript_commit,
    }) == 7

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
    authorization_relative = "data_usgs/confirmatory_opening_authorization_v1.json"
    authorization = {
        "format": verifier.AUTHORIZATION_FORMAT,
        "status": "AUTHORIZED_LABELS_STILL_SEALED",
        "opening_id": "2" * 24,
        "source": {
            "authorization_path": authorization_relative,
            "git_commit_before_authorization": head,
        },
            "runtime": {
                "format": "thermoroute.route-a-runtime.v1",
                "requirements_lock": {"path": "requirements-lock.txt", "sha256": "3" * 64},
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
