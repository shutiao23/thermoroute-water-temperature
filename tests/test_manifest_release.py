from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
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


def _materialize_claim_fixture(verifier, stage: Path, profile: str) -> None:
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
    verifier.materialize_claim_audit(stage, profile)


def _write_postopen_fixture(verifier, root: Path) -> tuple[Path, dict[str, str]]:
    _minimal_canonical_release(verifier, root)
    _write_bytes(root, "requirements-lock.txt", b"numpy==1.0\n")
    _write_bytes(root, "protocols/route_a.json", b"{}\n")
    _write_bytes(root, "data_usgs/external.csv")
    _write_bytes(root, "data_usgs/external.lock.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidates.csv")
    _write_bytes(root, "data_usgs/candidates.provenance.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/snapshot_index.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/response.rdb")
    _write_bytes(root, "src/thermoroute/opening.py", b"# fixed\n")
    _write_bytes(root, "scripts/route_a_trusted_scorer.py", b"# fixed\n")

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
    suite = {
        "format": "thermoroute.route-a-model-suite.v1",
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "cohorts": {
            cohort: {"models": entries}
            for cohort, entries in model_entries.items()
        },
    }
    suite_path = root / "data_usgs/confirmatory_model_suite_v1.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    source_tree_sha256 = "8" * 64
    runtime_sha256 = "c" * 64
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
        "protocol": _binding(verifier, root, "protocols/route_a.json"),
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
            "python_executable": {"realpath": "/fixture/python", "sha256": "d" * 64},
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
        },
        "state_paths": state,
    }
    authorization["authorization_self_sha256"] = verifier._sha256_json(authorization)
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    authorization_sha = verifier.sha256_file(authorization_path)

    trusted_validator = {"sha256": "f" * 64, "implementation": "fixture"}
    preflight = {"fixture": True}
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
    _write_bytes(root, state["acquisition_request_map"], b"{}\n")
    _write_bytes(root, state["temporal_outcomes"])
    _write_bytes(root, state["external_outcomes"])
    acquisition = {
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "labels_state": "OPENED_ONCE",
        "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
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
        "authorization": "protocols/route_a.json",
        "registries": "data_usgs/external.csv",
        "candidate_evidence": "data_usgs/candidates.csv",
        "model_suite": "data_usgs/confirmatory_model_suite_v1.json",
        "model_bundles": "outputs/models/temporal/LSTM.bundle",
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


def test_git_bundle_replays_sealed_protocol_after_release_relocation(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_git_evidence_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    protocol_relative = "protocols/route_a_confirmatory_protocol.md"
    protocol = _write_bytes(source, protocol_relative, b"# sealed fixture protocol\n")
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
        ["git", "commit", "-q", "-m", "seal protocol"],
        cwd=source,
        env=environment,
        check=True,
    )
    sealed_commit = subprocess.run(
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
    _write_bytes(
        stage,
        "protocols/route_a_confirmatory_v1.json",
        json.dumps({"authoritative_protocol_commit": sealed_commit}).encode(),
    )
    shutil.copy2(protocol, stage / protocol_relative)
    evidence = verifier.materialize_git_history_evidence(
        source, stage, verifier.PREOPEN_PROFILE
    )
    assert evidence["sealed_protocol_blob"]["sha256"] == hashlib.sha256(
        protocol.read_bytes()
    ).hexdigest()
    assert evidence["external_timestamp_or_public_preregistration"] is False

    relocated = tmp_path / "a-different-absolute-path" / "release"
    shutil.copytree(stage, relocated)
    marker = json.loads(
        (relocated / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    verifier._verify_git_history_evidence(
        relocated, marker, verifier.PREOPEN_PROFILE
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
                "authoritative_markdown_sha256": hashlib.sha256(
                    protocol.read_bytes()
                ).hexdigest(),
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
    marker["git_history_evidence"]["compute_commit"] = sealed_commit
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
    hidden_stage = tmp_path / "hidden-stage"
    _write_bytes(
        hidden_stage,
        verifier.PROFILE_MARKER,
        json.dumps({"profile": verifier.PREOPEN_PROFILE}).encode(),
    )
    _write_bytes(
        hidden_stage,
        "protocols/route_a_confirmatory_v1.json",
        json.dumps({"authoritative_protocol_commit": sealed_commit}).encode(),
    )
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
    hidden_evidence["compute_commit"] = sealed_commit
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
        ["git", "checkout", "-q", "-b", "unrelated-fixture", sealed_commit],
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
    with pytest.raises(ValueError, match="compute commit is not an ancestor"):
        verifier._verify_git_history_evidence(
            unrelated_stage, unrelated_marker, verifier.POSTOPEN_PROFILE
        )

    (relocated / verifier.GIT_BUNDLE_PATH).unlink()
    with pytest.raises(ValueError, match="absent"):
        verifier._verify_git_history_evidence(
            relocated, marker, verifier.POSTOPEN_PROFILE
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
