"""Focused adversarial tests for scratch-only semantic-contract registries v4."""

from __future__ import annotations

import copy
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S = importlib.import_module("scripts.final.build_semantic_contract_registries_v4")


def _dummy_snapshot() -> S.SourceSnapshot:
    with S.STATION_REGISTRY.open(encoding="utf-8", newline="") as stream:
        pairs = sorted((row["site_no"], row["huc2"].zfill(2)) for row in csv.DictReader(stream))
    key_manifest = json.loads(S.KEY_AUTHORITY_MANIFEST.read_bytes())
    reportable = tuple(key_manifest["primary_registry"]["reportable_site_ids"])
    bindings: dict[str, dict[str, object]] = {
        role: {
            "path": f"test/{role}",
            "sha256": S.PINNED_SHA256[role],
            "size_bytes": S.PINNED_SIZES[role],
        }
        for role in S.PINNED_SHA256
    }
    bindings.update(
        {
            "semantic_data_manifest": {
                "path": S.SEMANTIC_MANIFEST_FILENAME,
                "sha256": "0" * 64,
                "size_bytes": 10,
            },
            "semantic_daily_registry": {
                "path": S.SEMANTIC_DAILY_FILENAME,
                "sha256": "1" * 64,
                "size_bytes": 20,
            },
            "semantic_training_registry": {
                "path": S.SEMANTIC_TRAINING_FILENAME,
                "sha256": "2" * 64,
                "size_bytes": 30,
            },
        }
    )
    semantic_outputs = {
        S.SEMANTIC_DAILY_FILENAME: {
            "path": S.SEMANTIC_DAILY_FILENAME,
            "sha256": "1" * 64,
            "size_bytes": 20,
            "row_count": 788_880,
            "columns": list(S.DAILY_COLUMNS),
            "arrow_schema": S._schema_signature(S.DAILY_ARROW_SCHEMA),
            "ordered_semantic_content_sha256": "3" * 64,
            "ordered_identity_sha256": "4" * 64,
        },
        S.SEMANTIC_TRAINING_FILENAME: {
            "path": S.SEMANTIC_TRAINING_FILENAME,
            "sha256": "2" * 64,
            "size_bytes": 30,
            "row_count": 1_261_994,
            "columns": list(S.TRAINING_COLUMNS),
            "arrow_schema": S._schema_signature(S.TRAINING_ARROW_SCHEMA),
            "ordered_semantic_content_sha256": "5" * 64,
            "ordered_identity_sha256": "6" * 64,
        },
    }
    return S.SourceSnapshot(
        bindings=bindings,
        stations=tuple(site for site, _huc in pairs),
        huc2_by_station=dict(pairs),
        reportable_stations=reportable,
        semantic_outputs=semantic_outputs,
        signatures={},
    )


@pytest.fixture(scope="module")
def snapshot() -> S.SourceSnapshot:
    return _dummy_snapshot()


@pytest.fixture(scope="module")
def bundle(snapshot: S.SourceSnapshot) -> S.CandidateBundle:
    return S._build_from_snapshot(snapshot)


def _documents(bundle: S.CandidateBundle) -> dict[str, dict[str, Any]]:
    return {name: json.loads(bundle.files[name]) for name in S.REGISTRY_FILENAMES}


def test_exact_inventory_phase1_receipts_and_referential_closure(
    bundle: S.CandidateBundle,
    snapshot: S.SourceSnapshot,
) -> None:
    documents = _documents(bundle)
    S.validate_registry_documents(documents, expected_snapshot=snapshot)
    cell = documents[S.CELL_FILENAME]
    assert cell["arithmetic"]["fit_sum"] == "12 + 5280 + 40 = 5332"
    assert len(cell["logical_cells"]) == 74
    assert len(cell["variants"]) == 62
    assert len(cell["fits"]) == 5_332
    planned_lightgbm = [
        record
        for record in cell["logical_cells"]
        if record["family"] == "primary_lightgbm_f0_f3_planned"
    ]
    assert len(planned_lightgbm) == 24
    assert {record["protocol_state"] for record in planned_lightgbm} == {"PLANNED"}
    fit_logical_ids = {record["logical_cell_id"] for record in cell["fits"]}
    assert not ({record["logical_cell_id"] for record in planned_lightgbm} & fit_logical_ids)
    assert cell["independent_phase1_plan_receipts"]["neural"]["observed_sha256"] == (
        S.EXPECTED_NEURAL_PHASE1_PLAN_SHA256
    )
    assert cell["independent_phase1_plan_receipts"]["l2_u2"]["observed_sha256"] == (
        S.EXPECTED_L2_U2_PHASE1_PLAN_SHA256
    )
    assert bundle.manifest["cross_registry_receipt"]["all_fit_references_closed"]
    assert bundle.manifest["scope"]["logical_cells"] == 74
    assert bundle.manifest["scope"]["protocol_primary_cells"] == 72
    assert not bundle.manifest["scope"]["complete_v4_matrix"]
    assert not bundle.manifest["integrity_scope"]["sha256_is_scientific_evidence"]
    assert bundle.manifest["evidence_boundary"]["score_free"]
    assert bundle.manifest["evidence_boundary"]["executed_protocol_cells"] == 0


def test_protocol_declared_primary_state_map_is_72_cells_and_never_claims_execution(
    bundle: S.CandidateBundle,
) -> None:
    state_map = bundle.documents[S.CELL_FILENAME]["protocol_declared_cell_status"]
    primary = state_map["primary_matrix"]
    assert state_map["state_vocabulary"] == [
        "PLANNED",
        "REGISTERED",
        "EXECUTED",
        "WITHDRAWN",
    ]
    assert primary["protocol_logical_cell_count"] == 72
    assert primary["state_counts"] == {
        "PLANNED": 48,
        "REGISTERED": 24,
        "EXECUTED": 0,
        "WITHDRAWN": 0,
    }
    assert len(primary["cells"]) == 72
    assert {record["protocol_state"] for record in primary["cells"]} == {
        "PLANNED",
        "REGISTERED",
    }
    assert not any(record["execution_receipt_bound"] for record in primary["cells"])
    assert not any(record["score_or_result_receipt_bound"] for record in primary["cells"])
    registered = [record for record in primary["cells"] if record["protocol_state"] == "REGISTERED"]
    assert {record["architecture"] for record in registered} == {"plain_TCN"}
    assert {record["forcing"] for record in registered} == {"F0", "F3_full"}
    withdrawn = state_map["withdrawn_forensic_scope"]
    assert withdrawn["protocol_state"] == "WITHDRAWN"
    assert withdrawn["logical_cell_count"] == 432
    assert not withdrawn["included_in_primary_72"]


def test_fold_universes_pcg64_huc2_scoreability_and_15_digit_station(
    bundle: S.CandidateBundle,
) -> None:
    fold = bundle.documents[S.FOLD_FILENAME]
    assert fold["universe_contract"]["training_station_count"] == 120
    assert fold["universe_contract"]["reportable_scoring_station_count"] == 116
    assert fold["universe_contract"]["station_255534081324000_preserved"]
    assert fold["independent_phase1_fold_receipt"]["whole_region_observed_sha256"] == (
        S.EXPECTED_REGION_FOLD_SHA256
    )
    assert len(fold["folds"]) == 44
    assert len(fold["temporal_partitions"]) == 1
    temporal = fold["temporal_partitions"][0]
    assert temporal["partition_id"] == S.KNOWN_SITE_TEMPORAL_PARTITION_ID
    assert not temporal["spatial_cv_fold_applicable"]
    assert temporal["spatial_fold_id"] is None
    assert temporal["split_seed"] is None
    assert not any(record["geometry"] == "known_site_temporal" for record in fold["folds"])
    random_folds = [record for record in fold["folds"] if record["geometry"] == "random_site"]
    assert len(random_folds) == 40
    for seed in range(10):
        selected = [record for record in random_folds if record["split_seed"] == seed]
        assert [record["held_station_count"] for record in selected] == [30, 30, 31, 29]
    region = [record for record in fold["folds"] if record["geometry"] == "whole_region"]
    assert [record["held_station_count"] for record in region] == [30, 30, 31, 29]
    assert [record["scoreable_held_station_count"] for record in region] == [30, 30, 29, 27]
    assert all(record["split_seed"] is None for record in region)
    assert all(len(record["huc2"]) == 2 for record in fold["stations"])

    fits = bundle.documents[S.CELL_FILENAME]["fits"]
    temporal_fits = [record for record in fits if record["geometry"] == "known_site_temporal"]
    assert len(temporal_fits) == 12
    assert all(record["spatial_fold_id"] is None for record in temporal_fits)
    assert all(
        record["temporal_partition_id"] == S.KNOWN_SITE_TEMPORAL_PARTITION_ID
        for record in temporal_fits
    )
    spatial_fits = [record for record in fits if record["geometry"] != "known_site_temporal"]
    assert all(record["temporal_partition_id"] is None for record in spatial_fits)
    assert all(record["spatial_fold_id"] is not None for record in spatial_fits)


def test_model_input_contrast_environment_blockers_are_explicit(
    bundle: S.CandidateBundle,
) -> None:
    models = bundle.documents[S.MODEL_FILENAME]
    assert models["global_policy"]["model_fit_seed_effect_aggregation"] == (
        "mean_of_within_fit_seed_station_paired_effects"
    )
    assert models["global_policy"]["fit_seed_prediction_ensembling"] is False
    neural = [record for record in models["models"] if record["family"].startswith("neural")]
    assert all(record["active_parameter_contract"]["count"] is None for record in neural)
    assert all(record["checkpoint_contract"]["seal_blocker"] for record in neural)

    inputs = bundle.documents[S.INPUT_FILENAME]["inputs"]
    l2_u2 = next(record for record in inputs if record["local_level"] == "L2_U2")
    assert {"WTEMP_values", "WTEMP_observedness", "FLOW_values", "FLOW_observedness"} <= set(
        l2_u2["forbidden_history_channels"]
    )
    assert l2_u2["raw_projection_finiteness"]["all_station_training_meteorology_gap_dates"] == [
        "2008-12-31",
        "2012-12-31",
    ]
    assert not l2_u2["raw_projection_finiteness"][
        "raw_semantic_daily_registry_claimed_fully_finite"
    ]

    contrasts = bundle.documents[S.CONTRAST_FILENAME]
    blocked = next(
        record
        for record in contrasts["contrasts"]
        if record["contrast_id"] == "contrast:l2_u2:L2_U2_vs_L2"
    )
    assert blocked["availability"] == "BLOCKED"
    assert "same-authority L2 comparator" in blocked["blocker"]
    assert contrasts["uncertainty_and_sensitivity"]["whole_huc2_cluster_bootstrap_draws"] == 10_000
    assert contrasts["uncertainty_and_sensitivity"]["expected_sign_flip_configurations"] == 32_768
    geometry_contracts = contrasts["geometry_specific_aggregation_contracts"]
    assert geometry_contracts["random_site"]["split_seeds"] == list(range(10))
    assert geometry_contracts["random_site"]["spatial_cv_fold_indices"] == [0, 1, 2, 3]
    assert not geometry_contracts["whole_region"]["split_seed_applicable"]
    assert geometry_contracts["whole_region"]["split_seeds"] == []
    assert not geometry_contracts["known_site_temporal"]["split_seed_applicable"]
    assert geometry_contracts["known_site_temporal"]["spatial_cv_fold_indices"] == []
    geometry_penalty = next(
        record
        for record in contrasts["contrasts"]
        if record["contrast_id"]
        == "contrast:neural:geometry_penalty:whole_region_vs_random_site"
    )
    per_geometry = geometry_penalty["matched_folds"]["geometry_contracts"]
    assert per_geometry["whole_region"]["split_seeds"] == []
    assert per_geometry["random_site"]["split_seeds"] == list(range(10))
    assert "inside each random split seed" in geometry_penalty["cross_geometry_aggregation"]

    environments = bundle.documents[S.ENVIRONMENT_FILENAME]
    assert environments["runtime_boundary"]["authorized_execution_environment_id"] is None
    route_a = next(
        record
        for record in environments["environments"]
        if record["environment_id"] == "environment:route_a_candidate"
    )
    assert route_a["runtime"] == dict(S.ROUTE_A_RUNTIME)
    assert route_a["packages"] == dict(S.ROUTE_A_DIRECT_LOCK_PINS)
    assert route_a["direct_lock_pin_set_complete"]
    assert route_a["runtime_identity_enforced_at_candidate_build"]
    assert route_a["thread_policy"]["THERMOROUTE_FORMAL_THREADS"] == "1"
    assert route_a["determinism_policy"]["LightGBM_deterministic"]
    assert not route_a["determinism_policy"]["authority_verified"]
    assert not route_a["may_execute_registered_fits"]


def test_every_document_is_canonical_candidate_and_manifest_binds_exact_bytes(
    bundle: S.CandidateBundle,
) -> None:
    assert set(bundle.files) == set(S.ALL_FILENAMES)
    for name, payload in bundle.files.items():
        parsed = S._strict_json(payload, label=name)
        assert parsed["status"] == S.STATUS
        assert parsed["execution_authorized"] is False
        trust = parsed["candidate_trust_boundary"]
        assert trust == S._candidate_trust_boundary()
        assert not trust["semantic_data_candidate_accepted_as_scientific_evidence"]
        assert not trust["model_score_checkpoint_effect_or_result_artifacts_read_or_accepted"]
        assert S._canonical_json_bytes(parsed) == payload
    for name in S.REGISTRY_FILENAMES:
        output = bundle.manifest["outputs"][name]
        assert output["sha256"] == S._sha256(bundle.files[name])
        assert output["size_bytes"] == len(bundle.files[name])
        assert output["top_level_fields"] == sorted(bundle.documents[name])
    assert "semantic_contract_registry_manifest_v4_candidate.json" not in bundle.manifest["outputs"]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{"a":1}',
        b' {"a":1}\n',
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_and_noncanonical_bytes(payload: bytes) -> None:
    with pytest.raises(S.SemanticContractError, match="strict JSON|canonical JSON"):
        S._strict_json(payload, label="injected")


def test_missing_route_a_direct_runtime_distribution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_distribution: str) -> str:
        raise S.importlib.metadata.PackageNotFoundError("injected")

    monkeypatch.setattr(S.importlib.metadata, "version", missing)
    with pytest.raises(S.SemanticContractError, match="runtime distribution is unavailable"):
        S._distribution_version("lightgbm")


def _extra_field(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.CELL_FILENAME]["unexpected"] = True


def _missing_fit(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.CELL_FILENAME]["fits"].pop()


def _duplicate_fit_id(documents: dict[str, dict[str, Any]]) -> None:
    fits = documents[S.CELL_FILENAME]["fits"]
    fits[1]["fit_id"] = fits[0]["fit_id"]


def _wrong_count(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.CELL_FILENAME]["arithmetic"]["fit_count"] = 5_331


def _wrong_fold_membership(documents: dict[str, dict[str, Any]]) -> None:
    folds = documents[S.FOLD_FILENAME]["folds"]
    folds[1]["held_stations"][0], folds[1]["train_stations"][0] = (
        folds[1]["train_stations"][0],
        folds[1]["held_stations"][0],
    )


def _wrong_fit_reference(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.CELL_FILENAME]["fits"][0]["spatial_fold_id"] = "fold:absent"


def _wrong_status(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.MODEL_FILENAME]["status"] = "AUTHORITY"


def _string_boolean(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.ENVIRONMENT_FILENAME]["execution_authorized"] = "false"


def _confuse_120_and_116(documents: dict[str, dict[str, Any]]) -> None:
    excluded = next(
        record
        for record in documents[S.FOLD_FILENAME]["stations"]
        if not record["in_reportable_scoring_universe"]
    )
    excluded["in_reportable_scoring_universe"] = True


def _l2_leakage(documents: dict[str, dict[str, Any]]) -> None:
    record = next(
        item for item in documents[S.INPUT_FILENAME]["inputs"] if item["local_level"] == "L2_U2"
    )
    record["allowed_history_channels"].append("FLOW_values_and_raw_observed_masks")


def _mismatched_contrast(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.CONTRAST_FILENAME]["contrasts"][0]["matched_fit_seeds"] = [0, 1, 2, 3]


def _claim_partial_complete(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.CELL_FILENAME]["registered_scope"]["complete_v4_matrix"] = True


def _temporal_partition_as_cv_fold(documents: dict[str, dict[str, Any]]) -> None:
    temporal = documents[S.FOLD_FILENAME]["temporal_partitions"][0]
    temporal["spatial_cv_fold_applicable"] = True
    temporal["spatial_fold_id"] = "fold:fake-temporal"


def _whole_region_fake_split_seed(documents: dict[str, dict[str, Any]]) -> None:
    region = next(
        record
        for record in documents[S.FOLD_FILENAME]["folds"]
        if record["geometry"] == "whole_region"
    )
    region["split_seed"] = 0


def _falsely_claim_protocol_execution(documents: dict[str, dict[str, Any]]) -> None:
    state_map = documents[S.CELL_FILENAME]["protocol_declared_cell_status"]
    state_map["primary_matrix"]["cells"][0]["protocol_state"] = "EXECUTED"
    state_map["primary_matrix"]["state_counts"]["PLANNED"] -= 1
    state_map["primary_matrix"]["state_counts"]["EXECUTED"] = 1


def _promote_candidate_trust(documents: dict[str, dict[str, Any]]) -> None:
    documents[S.MODEL_FILENAME]["candidate_trust_boundary"][
        "semantic_data_candidate_accepted_as_scientific_evidence"
    ] = True


def _planned_lightgbm_claimed_registered(documents: dict[str, dict[str, Any]]) -> None:
    planned = next(
        record
        for record in documents[S.CELL_FILENAME]["logical_cells"]
        if record["family"] == "primary_lightgbm_f0_f3_planned"
    )
    planned["protocol_state"] = "REGISTERED"


@pytest.mark.parametrize(
    "attack",
    [
        _extra_field,
        _missing_fit,
        _duplicate_fit_id,
        _wrong_count,
        _wrong_fold_membership,
        _wrong_fit_reference,
        _wrong_status,
        _string_boolean,
        _confuse_120_and_116,
        _l2_leakage,
        _mismatched_contrast,
        _claim_partial_complete,
        _temporal_partition_as_cv_fold,
        _whole_region_fake_split_seed,
        _falsely_claim_protocol_execution,
        _promote_candidate_trust,
        _planned_lightgbm_claimed_registered,
    ],
    ids=lambda function: function.__name__,
)
def test_registry_injections_fail_closed(
    attack: Any,
    bundle: S.CandidateBundle,
    snapshot: S.SourceSnapshot,
) -> None:
    documents = _documents(bundle)
    attack(documents)
    with pytest.raises(S.SemanticContractError):
        S.validate_registry_documents(documents, expected_snapshot=snapshot)


def test_manifest_hash_size_count_and_cross_receipt_injections_fail_closed(
    bundle: S.CandidateBundle,
) -> None:
    manifest = copy.deepcopy(bundle.manifest)
    manifest["outputs"][S.CELL_FILENAME]["record_counts"]["fits"] = 5_331
    with pytest.raises(S.SemanticContractError, match="manifest differs"):
        S._verify_manifest(
            manifest,
            bundle.documents,
            {name: bundle.files[name] for name in S.REGISTRY_FILENAMES},
            bundle.snapshot,
        )
    manifest = copy.deepcopy(bundle.manifest)
    manifest["cross_registry_receipt"]["all_fit_references_closed"] = False
    with pytest.raises(S.SemanticContractError, match="manifest differs"):
        S._verify_manifest(
            manifest,
            bundle.documents,
            {name: bundle.files[name] for name in S.REGISTRY_FILENAMES},
            bundle.snapshot,
        )


def test_input_capture_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"{}\n")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises(S.SemanticContractError, match="symlink"):
        S._capture_regular(symlink, label="symlink injection", retain_payload=True)
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(S.SemanticContractError, match="hard link"):
        S._capture_regular(source, label="hardlink injection", retain_payload=True)


def test_destination_is_explicit_create_only_and_outside_outputs_final(tmp_path: Path) -> None:
    destination = tmp_path / "candidate"
    assert S._validate_destination(destination) == (destination.absolute(), tmp_path.absolute())
    destination.mkdir()
    with pytest.raises(S.SemanticContractError, match="already exists"):
        S._validate_destination(destination)
    prohibited = S.FINAL_OUTPUT_ROOT / "semantic-contract-test-must-not-exist"
    with pytest.raises(S.SemanticContractError, match="prohibited under outputs/final"):
        S._validate_destination(prohibited)
    linked_parent = tmp_path / "linked-parent"
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(S.SemanticContractError, match="symlink"):
        S._validate_destination(linked_parent / "candidate")


def test_staged_reader_rejects_hardlink(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        source = tmp_path / "one.json"
        source.write_bytes(b"{}\n")
        os.link(source, tmp_path / "two.json")
        with pytest.raises(S.SemanticContractError, match="hard link"):
            S._read_at(descriptor, source.name, label="staged hardlink")
    finally:
        os.close(descriptor)


def test_closed_writer_commits_exactly_once(
    tmp_path: Path,
    bundle: S.CandidateBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(S, "build_twice", lambda _paths, *, config: bundle)
    monkeypatch.setattr(S, "_revalidate_sources", lambda _paths, _config, _snapshot: None)
    paths = S.RegistryInputPaths.production(tmp_path / "unused-source")
    destination = tmp_path / "published-candidate"
    observed, written = S.write_candidate(
        paths,
        destination,
        config=S.BuildConfig(enforce_route_a_runtime=False),
    )
    assert observed == destination.absolute()
    assert written is bundle
    assert {path.name for path in destination.iterdir()} == set(S.ALL_FILENAMES)
    assert all(path.stat().st_nlink == 1 for path in destination.iterdir())
    with pytest.raises(S.SemanticContractError, match="already exists"):
        S.write_candidate(
            paths,
            destination,
            config=S.BuildConfig(enforce_route_a_runtime=False),
        )


def test_cli_requires_both_explicit_scratch_paths() -> None:
    parser = S._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        ["--semantic-data-dir", "/tmp/data-candidate", "--output-dir", "/tmp/contract-candidate"]
    )
    assert args.semantic_data_dir == Path("/tmp/data-candidate")
    assert args.output_dir == Path("/tmp/contract-candidate")
