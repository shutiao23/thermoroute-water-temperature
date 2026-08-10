"""Adversarial tests for the create-only semantic registries v4 authority."""

from __future__ import annotations

import copy
import csv
import importlib
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

P = importlib.import_module("scripts.final.publish_semantic_registries_v4")
D = importlib.import_module("scripts.final.build_semantic_data_registries_v4")
C = importlib.import_module("scripts.final.build_semantic_contract_registries_v4")


def _synthetic_data_candidate(root: Path) -> tuple[Any, Any]:
    """Reuse the data builder's own synthetic evidence constructor."""

    helpers = runpy.run_path(str(ROOT / "tests/final/test_semantic_data_registries_v4.py"))
    paths, config, _raw = helpers["_make_evidence"](root)
    return D.build_twice(paths, config=config, evidence_root=root), config


def _contract_snapshot_for_data(data_candidate: Any) -> C.SourceSnapshot:
    with C.STATION_REGISTRY.open(encoding="utf-8", newline="") as stream:
        pairs = sorted((row["site_no"], row["huc2"].zfill(2)) for row in csv.DictReader(stream))
    key_manifest = json.loads(C.KEY_AUTHORITY_MANIFEST.read_bytes())
    reportable = tuple(key_manifest["primary_registry"]["reportable_site_ids"])
    bindings: dict[str, dict[str, object]] = {
        role: {
            "path": f"test/{role}",
            "sha256": C.PINNED_SHA256[role],
            "size_bytes": C.PINNED_SIZES[role],
        }
        for role in C.PINNED_SHA256
    }
    bindings.update(
        {
            "semantic_data_manifest": {
                "path": D.MANIFEST_FILENAME,
                "sha256": P._sha256(data_candidate.files[D.MANIFEST_FILENAME]),
                "size_bytes": len(data_candidate.files[D.MANIFEST_FILENAME]),
            },
            "semantic_daily_registry": {
                "path": D.DAILY_FILENAME,
                "sha256": P._sha256(data_candidate.files[D.DAILY_FILENAME]),
                "size_bytes": len(data_candidate.files[D.DAILY_FILENAME]),
            },
            "semantic_training_registry": {
                "path": D.TRAINING_FILENAME,
                "sha256": P._sha256(data_candidate.files[D.TRAINING_FILENAME]),
                "size_bytes": len(data_candidate.files[D.TRAINING_FILENAME]),
            },
        }
    )
    outputs = data_candidate.manifest["outputs"]
    semantic_outputs = {
        name: {"path": name, **dict(outputs[name])}
        for name in (D.DAILY_FILENAME, D.TRAINING_FILENAME)
    }
    return C.SourceSnapshot(
        bindings=bindings,
        stations=tuple(site for site, _huc in pairs),
        huc2_by_station=dict(pairs),
        reportable_stations=reportable,
        semantic_outputs=semantic_outputs,
        signatures={},
    )


@pytest.fixture(scope="module")
def candidate_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any, Any]:
    evidence = tmp_path_factory.mktemp("semantic-authority-evidence")
    data_candidate, data_config = _synthetic_data_candidate(evidence)
    contract_candidate = C._build_from_snapshot(_contract_snapshot_for_data(data_candidate))
    return data_candidate, contract_candidate, data_config


@pytest.fixture(scope="module")
def authority_bundle(candidate_pair: tuple[Any, Any, Any]) -> P.AuthorityBundle:
    data_candidate, contract_candidate, data_config = candidate_pair
    return P._build_authority_bundle(
        data_candidate,
        contract_candidate,
        P._capture_attestation_sources(),
        data_config=data_config,
        require_production=False,
    )


def test_authority_binds_two_manifests_all_eight_registries_and_exact_bytes(
    authority_bundle: P.AuthorityBundle,
) -> None:
    P._verify_authority_bundle(authority_bundle)
    assert set(authority_bundle.files) == set(P.ALL_FILENAMES)
    assert len(authority_bundle.files) == 11
    assert set(P.CANDIDATE_MANIFEST_FILENAMES) <= set(authority_bundle.files)
    assert len(P.DATA_REGISTRY_FILENAMES) == 2
    assert len(P.CONTRACT_REGISTRY_FILENAMES) == 6
    manifest = authority_bundle.manifest
    assert manifest["artifact_id"] == P.ARTIFACT_ID
    assert manifest["status"] == P.STATUS
    assert manifest["authority_class"] == "TIER1_SEMANTIC_DATA_AND_CONTRACT_ONLY"
    assert manifest["formal_semantic_registry_authority"] is True
    assert manifest["publication_contract"]["exact_file_count"] == 11
    assert manifest["publication_contract"]["bound_candidate_file_count"] == 10
    assert manifest["publication_contract"]["authority_manifest_self_hash_bound"] is False

    data_receipt = manifest["candidate_manifests"]["semantic_data"]
    contract_receipt = manifest["candidate_manifests"]["semantic_contract"]
    assert data_receipt["sha256"] == P._sha256(authority_bundle.files[D.MANIFEST_FILENAME])
    assert contract_receipt["sha256"] == P._sha256(authority_bundle.files[C.MANIFEST_FILENAME])
    for candidate_role, names in (
        ("semantic_data", P.DATA_REGISTRY_FILENAMES),
        ("semantic_contract", P.CONTRACT_REGISTRY_FILENAMES),
    ):
        for name in names:
            record = manifest["registries"][candidate_role][name]
            assert record["sha256"] == P._sha256(authority_bundle.files[name])
            assert record["size_bytes"] == len(authority_bundle.files[name])


def test_authority_is_not_execution_or_forcing_seal_authority(
    authority_bundle: P.AuthorityBundle,
) -> None:
    manifest = authority_bundle.manifest
    assert manifest["execution_authorized"] is False
    assert manifest["model_execution_authorized"] is False
    assert manifest["forcing_protocol_seal_bound"] is False
    assert manifest["forcing_protocol_seal_required_separately"] is True
    assert manifest["may_replace_or_amend_forcing_protocol_seal"] is False
    assert manifest["candidate_status_promoted_to_executed"] is False
    assert manifest["model_prediction_score_effect_or_result_artifacts_read"] is False
    boundary = manifest["evidence_boundary"]
    assert boundary["prediction_files_read"] is False
    assert boundary["score_tables_read"] is False
    assert boundary["model_checkpoints_read"] is False
    assert boundary["effect_or_contrast_results_read"] is False
    assert boundary["runner_imported_or_executed"] is False
    assert boundary["prediction_or_score_rows_read"] == 0
    for filename in P.CANDIDATE_MANIFEST_FILENAMES:
        candidate = P._strict_json(authority_bundle.files[filename], label=filename)
        assert candidate["status"] == "CANDIDATE_NOT_AUTHORITY"
        assert candidate["execution_authorized"] is False


def test_exact_primary_and_withdrawn_forensic_state_receipt(
    authority_bundle: P.AuthorityBundle,
) -> None:
    receipt = authority_bundle.manifest["protocol_cell_state_receipt"]
    assert receipt == {
        "primary_cell_count": 72,
        "primary_state_counts": {
            "PLANNED": 48,
            "REGISTERED": 24,
            "EXECUTED": 0,
            "WITHDRAWN": 0,
        },
        "primary_executed_cell_count": 0,
        "candidate_may_assert_executed": False,
        "withdrawn_forensic_cell_count": 432,
        "withdrawn_forensic_state": "WITHDRAWN",
        "withdrawn_forensic_included_in_primary_72": False,
        "withdrawn_forensic_eligible_as_score_or_result_evidence": False,
    }
    cell = P._strict_json(authority_bundle.files[C.CELL_FILENAME], label=C.CELL_FILENAME)
    primary = cell["protocol_declared_cell_status"]["primary_matrix"]
    assert primary["state_counts"] == receipt["primary_state_counts"]
    assert len(primary["cells"]) == 72
    assert not any(record["protocol_state"] == "EXECUTED" for record in primary["cells"])
    forensic = cell["protocol_declared_cell_status"]["withdrawn_forensic_scope"]
    assert forensic["protocol_state"] == "WITHDRAWN"
    assert forensic["logical_cell_count"] == 432


def test_builder_test_source_input_and_runtime_hashes_are_bound(
    authority_bundle: P.AuthorityBundle,
) -> None:
    manifest = authority_bundle.manifest
    code = manifest["code_bindings"]
    assert set(code["builders"]) == {
        "semantic_data_builder",
        "semantic_contract_builder",
    }
    assert set(code["publisher"]) == {"semantic_authority_publisher"}
    assert set(code["tests"]) == {
        "semantic_data_builder_tests",
        "semantic_contract_builder_tests",
        "semantic_authority_publisher_tests",
    }
    all_code_records = {
        **code["builders"],
        **code["publisher"],
        **code["tests"],
    }
    for role, record in all_code_records.items():
        source = authority_bundle.attestation_sources[role]
        assert record == source.binding()
        assert record["sha256"] == P._sha256(source.payload)
    binding_set = {
        "builders": code["builders"],
        "publisher": code["publisher"],
        "tests": code["tests"],
    }
    assert code["canonical_binding_set_sha256"] == P._canonical_document_sha256(binding_set)

    inputs = manifest["source_input_bindings"]
    assert inputs["semantic_data_candidate_inputs"]
    assert inputs["semantic_contract_candidate_inputs"]
    assert inputs["semantic_data_builder_snapshot"] == inputs["semantic_data_candidate_inputs"]
    assert inputs["semantic_contract_builder_snapshot"]
    runtime = manifest["runtime_receipts"]
    for role in (
        "authority_active_runtime",
        "semantic_data_active_runtime",
        "semantic_data_expected_runtime",
        "semantic_contract_route_a_runtime",
        "semantic_contract_route_a_packages",
    ):
        assert runtime[role]["canonical_sha256"] == P._canonical_document_sha256(
            runtime[role]["values"]
        )
    assert runtime["runtime_identity_is_execution_authority"] is False


def test_contract_must_bind_exact_rebuilt_semantic_data_bytes(
    candidate_pair: tuple[Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_candidate, contract_candidate, data_config = candidate_pair
    attacked_manifest = copy.deepcopy(contract_candidate.manifest)
    attacked_manifest["source_inputs"]["semantic_daily_registry"]["sha256"] = "0" * 64
    monkeypatch.setattr(P, "_verify_contract_candidate", lambda _bundle: attacked_manifest)
    with pytest.raises(P.SemanticAuthorityError, match="exact semantic-data bytes"):
        P._verify_candidate_cross_bindings(
            data_candidate,
            contract_candidate,
            data_config=data_config,
            require_production=False,
        )


def test_executed_state_injection_fails_closed(
    candidate_pair: tuple[Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_candidate, contract_candidate, data_config = candidate_pair
    documents = dict(contract_candidate.documents)
    cell = dict(documents[C.CELL_FILENAME])
    status_map = copy.deepcopy(cell["protocol_declared_cell_status"])
    status_map["primary_matrix"]["cells"][0]["protocol_state"] = "EXECUTED"
    status_map["primary_matrix"]["state_counts"] = {
        "PLANNED": 47,
        "REGISTERED": 24,
        "EXECUTED": 1,
        "WITHDRAWN": 0,
    }
    cell["protocol_declared_cell_status"] = status_map
    documents[C.CELL_FILENAME] = cell
    attacked = SimpleNamespace(
        files=contract_candidate.files,
        documents=documents,
        snapshot=contract_candidate.snapshot,
        manifest=contract_candidate.manifest,
    )
    monkeypatch.setattr(
        P,
        "_verify_contract_candidate",
        lambda _bundle: contract_candidate.manifest,
    )
    with pytest.raises(P.SemanticAuthorityError, match="72-cell state arithmetic"):
        P._verify_candidate_cross_bindings(
            data_candidate,
            attacked,
            data_config=data_config,
            require_production=False,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("execution_authorized", True),
        ("execution_authorized", "false"),
        ("forcing_protocol_seal_bound", True),
        ("may_replace_or_amend_forcing_protocol_seal", True),
    ],
)
def test_authority_manifest_promotion_injections_fail_closed(
    authority_bundle: P.AuthorityBundle,
    field: str,
    replacement: object,
) -> None:
    manifest = authority_bundle.manifest
    manifest[field] = replacement
    files = dict(authority_bundle.files)
    files[P.AUTHORITY_MANIFEST_FILENAME] = P._canonical_json_bytes(manifest)
    attacked = P.AuthorityBundle(
        files=files,
        data_candidate=authority_bundle.data_candidate,
        contract_candidate=authority_bundle.contract_candidate,
        attestation_sources=authority_bundle.attestation_sources,
        data_config=authority_bundle.data_config,
        require_production=False,
        _token=P._BUNDLE_TOKEN,
    )
    with pytest.raises(P.SemanticAuthorityError, match="authority manifest differs"):
        P._verify_authority_bundle(attacked)


def _patch_publication_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    authority_bundle: P.AuthorityBundle,
) -> None:
    monkeypatch.setattr(
        P,
        "_rebuild_candidates",
        lambda _work_directory: (
            authority_bundle.data_candidate,
            authority_bundle.contract_candidate,
            authority_bundle.data_config,
            (None, None),
        ),
    )
    monkeypatch.setattr(
        P,
        "_build_authority_bundle",
        lambda *_args, **_kwargs: authority_bundle,
    )
    monkeypatch.setattr(P, "_revalidate_candidate_sources", lambda *_args: None)


def test_create_only_same_directory_staged_writer_commits_exactly_once(
    tmp_path: Path,
    authority_bundle: P.AuthorityBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_publication_rebuild(monkeypatch, authority_bundle)
    destination = tmp_path / "semantic-authority"
    observed, written = P.publish_authority(destination)
    assert observed == destination.absolute()
    assert written is authority_bundle
    assert {path.name for path in destination.iterdir()} == set(P.ALL_FILENAMES)
    assert all(path.is_file() and path.stat().st_nlink == 1 for path in destination.iterdir())
    for name, payload in authority_bundle.files.items():
        assert (destination / name).read_bytes() == payload
    assert not list(tmp_path.glob(".*.semantic-authority-stage.*"))
    assert not list(tmp_path.glob(".*.semantic-authority-rebuild.*"))
    assert not list(tmp_path.glob(".*.semantic-authority-create.lock"))
    with pytest.raises(P.SemanticAuthorityError, match="already exists"):
        P.publish_authority(destination)


def test_rename_noreplace_toctou_preserves_racing_destination_and_cleans_owned_stage(
    tmp_path: Path,
    authority_bundle: P.AuthorityBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_publication_rebuild(monkeypatch, authority_bundle)
    original = P._rename_noreplace

    def race(parent_descriptor: int, source: str, destination: str) -> None:
        os.mkdir(destination, dir_fd=parent_descriptor)
        original(parent_descriptor, source, destination)

    monkeypatch.setattr(P, "_rename_noreplace", race)
    destination = tmp_path / "raced-authority"
    with pytest.raises(P.SemanticAuthorityError, match="refusing to overwrite"):
        P.publish_authority(destination)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".*.semantic-authority-stage.*"))
    assert not list(tmp_path.glob(".*.semantic-authority-rebuild.*"))
    assert not list(tmp_path.glob(".*.semantic-authority-create.lock"))


def test_destination_symlink_hardlink_and_parser_boundaries(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(P.SemanticAuthorityError, match="symlink"):
        P._validate_destination(linked_parent / "authority")

    descriptor = os.open(real_parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        first = real_parent / "first.json"
        first.write_bytes(b"{}\n")
        os.link(first, real_parent / "second.json")
        with pytest.raises(P.SemanticAuthorityError, match="hard link"):
            P._read_at(descriptor, first.name, label="hard-linked staged file")
    finally:
        os.close(descriptor)

    parser = P._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--output-dir", "/tmp/semantic-authority"])
    assert args.output_dir == Path("/tmp/semantic-authority")
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "/tmp/x", "--overwrite"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "/tmp/x", "--resume"])


def test_canonical_path_and_final_output_rebuild_workspace_are_separate() -> None:
    assert P.CANONICAL_AUTHORITY_DIRECTORY == (
        ROOT / "outputs/final/semantic_registries_v4_authority"
    )
    assert P._rebuild_parent(P.FINAL_OUTPUT_ROOT) == ROOT / "outputs"
