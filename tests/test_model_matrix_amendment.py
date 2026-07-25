from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import thermoroute.model_matrix_amendment as MMA
from thermoroute.model_matrix_amendment import (
    AMENDMENT_FORMAT,
    AMENDMENT_ID,
    AMENDMENT_RELATIVE,
    AMENDMENT_SEAL_RELATIVE,
    AMENDMENT_SEAL_FORMAT,
    AMENDMENT_SEAL_STATUS,
    AMENDMENT_STATUS,
    GOVERNANCE_SHA256,
    GOVERNANCE_SEALS,
    HISTORY_CONTRACT,
    PLAIN_CONTROL_ALLOWED_INPUTS,
    PLAIN_CONTROL_FORBIDDEN_INPUTS,
    PROTOCOL_OBJECT_SHA256,
    ModelMatrixAmendmentError,
    build_model_matrix_amendment_seal_document,
    expected_model_matrix_amendment_document,
    validate_model_matrix_amendment,
    validate_model_matrix_amendment_seal,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_contract(tmp_path: Path) -> Path:
    for relative in (*GOVERNANCE_SHA256, AMENDMENT_RELATIVE):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path / AMENDMENT_RELATIVE


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _git_commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _copy_relative(source: Path, destination: Path, relative: str) -> None:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / relative, target)


def _document_repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative in GOVERNANCE_SHA256:
        _copy_relative(ROOT, root, relative)
    governance_commit = _git_commit(root, "immutable governance chain")
    _copy_relative(ROOT, root, AMENDMENT_RELATIVE)
    document_commit = _git_commit(root, "model matrix amendment document")
    return root, governance_commit, document_commit


def _sealed_repository(
    tmp_path: Path,
) -> tuple[Path, str, str, dict[str, object]]:
    root, _governance_commit, document_commit = _document_repository(tmp_path)
    seal = build_model_matrix_amendment_seal_document(
        root=root,
        amendment_document_commit=document_commit,
    )
    _write_json(root / AMENDMENT_SEAL_RELATIVE, seal)
    seal_commit = _git_commit(root, "separate model matrix amendment seal")
    return root, document_commit, seal_commit, seal


def _gitless_archive(source: Path, destination: Path) -> Path:
    destination.mkdir()
    for relative in (*GOVERNANCE_SHA256, AMENDMENT_RELATIVE, AMENDMENT_SEAL_RELATIVE):
        _copy_relative(source, destination, relative)
    return destination


def test_production_amendment_is_exact_outcome_free_and_has_no_future_seal_hash() -> None:
    value = validate_model_matrix_amendment(ROOT / AMENDMENT_RELATIVE, root=ROOT)

    assert value["format"] == AMENDMENT_FORMAT
    assert value["status"] == AMENDMENT_STATUS
    assert value["amendment_id"] == AMENDMENT_ID
    assert value["prelabel_attestation"] == MMA.PRELABEL_ATTESTATION
    assert value["governance_inputs"]["route_a_claim_registry_v1"] == {
        "path": "protocols/route_a_claim_registry_v1.json",
        "sha256": "bc3d489b6f2cfe945789a57990a15291b332ae63be6c8df0211aa2b6ff8b70b6",
    }
    lineage = value["lineage_contract"]
    assert lineage["seal_path"] == AMENDMENT_SEAL_RELATIVE
    assert lineage["seal_sha256_declared_in_this_document"] is False
    assert lineage["amendment_document_commit_must_precede_seal_commit"] is True


def test_stage09_matrix_is_exactly_seven_by_five() -> None:
    value = validate_model_matrix_amendment(AMENDMENT_RELATIVE, root=ROOT)
    matrix = value["stage09_architecture_control_matrix"]

    assert matrix["arm_count"] == len(matrix["control_arms"]) == 7
    assert matrix["seeds_per_arm"] == [0, 1, 2, 3, 4]
    assert matrix["seed_count_per_arm"] == 5
    assert matrix["expected_member_count"] == 7 * 5 == 35
    assert matrix["pairing"] == {
        "reference": "ThermoRoute_member_with_the_exact_same_seed",
        "same_seed_required": True,
        "exact_forecast_keys_required": True,
        "exact_y_true_required": True,
        "best_seed_selection_allowed": False,
    }
    assert matrix["reporting"]["causal_or_module_necessity_attribution_allowed"] is False
    assert matrix["reporting"]["capacity_matched_claim_allowed"] is False


def test_stage09b_matrix_freezes_45_360_and_405_contracts() -> None:
    value = validate_model_matrix_amendment(AMENDMENT_RELATIVE, root=ROOT)
    matrix = value["stage09b_development_control_matrix"]

    assert matrix["arm_count"] == len(matrix["arms"]) == 9
    assert matrix["seed_count_per_arm"] == len(matrix["seeds_per_arm"]) == 5
    assert matrix["expected_member_count"] == 9 * 5 == 45
    assert len(matrix["splits"]) == 3
    assert len(matrix["horizons"]) == 3
    assert matrix["member_summary_cell_count"] == 9 * 5 * 3 * 3 == 405
    comparisons = matrix["paired_comparisons"]
    assert matrix["paired_comparison_count"] == len(comparisons) == 8
    assert all(row["seeds"] == [0, 1, 2, 3, 4] for row in comparisons)
    assert matrix["paired_seed_count"] == 8 * 5 == 40
    assert matrix["paired_effect_cell_count"] == 8 * 5 * 3 * 3 == 360


def test_plain_controls_are_prospectively_information_matched_but_not_physics_models() -> None:
    value = validate_model_matrix_amendment(AMENDMENT_RELATIVE, root=ROOT)
    revision = value["stage09b_development_control_matrix"][
        "plain_control_information_fairness_revision"
    ]

    assert revision["affected_arms"] == ["PlainMLP-7var", "PlainCausalTCN-7var"]
    assert revision["allowed_consumed_batch_keys"] == list(
        PLAIN_CONTROL_ALLOWED_INPUTS
    )
    assert revision["explicitly_forbidden_batch_keys"] == list(
        PLAIN_CONTROL_FORBIDDEN_INPUTS
    )
    assert set(revision["allowed_consumed_batch_keys"]).isdisjoint(
        revision["explicitly_forbidden_batch_keys"]
    )
    assert revision["all_unlisted_batch_keys_forbidden_to_consume"] is True
    assert revision["common_anchor"] == "damped_prior"
    assert revision["prediction_head"] == (
        "unrestricted_residual_added_to_damped_prior"
    )
    assert revision["router_present"] is False
    assert revision["mixture_of_experts_present"] is False
    assert revision["physics_modules_present"] is False
    assert revision["bounded_residual_present"] is False
    budget = revision["parameter_budget"]
    assert budget["maximum_absolute_relative_difference"] == 0.02
    assert budget["exact_new_parameter_count_declared_here"] is False
    assert revision["old_stage09b_artifacts_eligible_for_final_freeze"] is False
    assert revision["full_retraining_and_replay_required"] is True
    assert revision["outcome_or_result_based_retuning_allowed"] is False


def test_primary_contract_is_explicitly_unchanged() -> None:
    value = validate_model_matrix_amendment(AMENDMENT_RELATIVE, root=ROOT)
    scope = value["scientific_scope"]
    boundary = value["unchanged_primary_boundary"]

    assert scope["primary_model_or_feature_contract_changed"] is False
    assert scope["stage09_control_identity_or_intervention_changed"] is False
    assert scope["stage09b_plain_control_input_and_head_definition_changed"] is True
    assert boundary["confirmatory_family_count"] == 5
    for field in (
        "formal_comparisons_changed",
        "formal_margins_changed",
        "formal_estimand_changed",
        "formal_decisions_changed",
        "multiplicity_family_changed",
        "primary_models_changed",
        "primary_feature_order_changed",
        "primary_fit_data_changed",
        "primary_frozen_hyperparameters_changed",
        "primary_forecast_key_registry_changed",
        "probabilistic_event_contract_changed",
        "confirmation_data_may_be_used_for_fit_or_selection",
    ):
        assert boundary[field] is False
    assert boundary["architecture_controls_remain_exploratory_descriptive"] is True
    assert boundary["stage09b_remains_development_only_exploratory"] is True


def test_protocol_object_hashes_are_recomputed_from_their_named_objects() -> None:
    value = validate_model_matrix_amendment(AMENDMENT_RELATIVE, root=ROOT)
    bindings = value["primary_contract_object_bindings"]

    assert set(bindings) == set(PROTOCOL_OBJECT_SHA256)
    cache: dict[str, object] = {}
    for label, (relative, pointer, expected_digest) in PROTOCOL_OBJECT_SHA256.items():
        if relative not in cache:
            cache[relative] = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        selected = cache[relative]
        for component in pointer:
            assert isinstance(selected, dict)
            selected = selected[component]
        assert _canonical_sha256(selected) == expected_digest
        assert bindings[label] == {
            "source": relative,
            "json_path": list(pointer),
            "sha256": expected_digest,
        }


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("stage09_architecture_control_matrix", "expected_member_count"), 7),
        (("stage09b_development_control_matrix", "paired_effect_cell_count"), 359),
        (
            (
                "stage09b_development_control_matrix",
                "plain_control_information_fairness_revision",
                "common_anchor",
            ),
            "clim_tgt",
        ),
        (("unchanged_primary_boundary", "formal_margins_changed"), True),
        (("lineage_contract", "seal_sha256_declared_in_this_document"), True),
    ),
)
def test_amendment_content_tampering_fails_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    amendment = _copy_contract(tmp_path)
    attacked = json.loads(amendment.read_text(encoding="utf-8"))
    selected = attacked
    for component in path[:-1]:
        selected = selected[component]
    selected[path[-1]] = value
    _write_json(amendment, attacked)

    with pytest.raises(ModelMatrixAmendmentError, match="stale/tampered"):
        validate_model_matrix_amendment(amendment, root=tmp_path)


def test_extra_schema_field_and_duplicate_json_key_fail_closed(tmp_path: Path) -> None:
    amendment = _copy_contract(tmp_path)
    attacked = json.loads(amendment.read_text(encoding="utf-8"))
    attacked["unexpected"] = True
    _write_json(amendment, attacked)
    with pytest.raises(ModelMatrixAmendmentError, match="stale/tampered"):
        validate_model_matrix_amendment(amendment, root=tmp_path)

    amendment.write_text('{"format":"a","format":"b"}\n', encoding="utf-8")
    with pytest.raises(ModelMatrixAmendmentError, match="duplicate JSON key"):
        validate_model_matrix_amendment(amendment, root=tmp_path)


def test_upstream_file_byte_tampering_fails_closed(tmp_path: Path) -> None:
    amendment = _copy_contract(tmp_path)
    upstream = tmp_path / "protocols/route_a_confirmatory_v1.json"
    upstream.write_bytes(upstream.read_bytes() + b"\n")

    with pytest.raises(ModelMatrixAmendmentError, match="governance checksum changed"):
        validate_model_matrix_amendment(amendment, root=tmp_path)


def test_object_hash_recomputation_is_independent_of_whole_file_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_contract(tmp_path)
    relative = "protocols/route_a_confirmatory_v1.json"
    upstream = tmp_path / relative
    attacked = json.loads(upstream.read_text(encoding="utf-8"))
    attacked["primary_inference_contract"]["confirmatory_family"][0][
        "margin_c"
    ] = -999.0
    _write_json(upstream, attacked)
    monkeypatch.setitem(
        MMA.GOVERNANCE_SHA256,
        relative,
        hashlib.sha256(upstream.read_bytes()).hexdigest(),
    )

    with pytest.raises(ModelMatrixAmendmentError, match="protocol object checksum changed"):
        expected_model_matrix_amendment_document(root=tmp_path)


def test_symlinks_and_non_regular_inputs_fail_closed(tmp_path: Path) -> None:
    amendment = _copy_contract(tmp_path)
    amendment.unlink()
    amendment.symlink_to(ROOT / AMENDMENT_RELATIVE)
    with pytest.raises(ModelMatrixAmendmentError, match="symlink"):
        validate_model_matrix_amendment(amendment, root=tmp_path)

    shutil.rmtree(tmp_path)
    amendment = _copy_contract(tmp_path)
    upstream = tmp_path / "protocols/route_a_inference_amendment_v2.json"
    upstream.unlink()
    upstream.mkdir()
    with pytest.raises(ModelMatrixAmendmentError, match="not a regular file"):
        validate_model_matrix_amendment(amendment, root=tmp_path)


def test_only_the_canonical_amendment_path_is_accepted(tmp_path: Path) -> None:
    _copy_contract(tmp_path)
    wrong = tmp_path / "protocols/copy.json"
    shutil.copyfile(tmp_path / AMENDMENT_RELATIVE, wrong)

    with pytest.raises(ModelMatrixAmendmentError, match="not allowlisted"):
        validate_model_matrix_amendment(wrong, root=tmp_path)
    with pytest.raises(ModelMatrixAmendmentError, match="not allowlisted"):
        validate_model_matrix_amendment(
            "protocols/../protocols/route_a_model_matrix_amendment_v1.json",
            root=tmp_path,
        )


def test_future_seal_builder_and_live_validator_accept_strict_history(
    tmp_path: Path,
) -> None:
    root, _governance, document_commit = _document_repository(tmp_path)
    seal = build_model_matrix_amendment_seal_document(
        root=root,
        amendment_document_commit=document_commit,
    )

    assert set(seal) == {
        "format",
        "status",
        "amendment_id",
        "amendment",
        "governance_seals",
        "amendment_document_commit",
        "history_contract",
        "prelabel_attestation",
    }
    assert seal["format"] == AMENDMENT_SEAL_FORMAT
    assert seal["status"] == AMENDMENT_SEAL_STATUS
    assert seal["amendment_id"] == AMENDMENT_ID
    assert seal["amendment_document_commit"] == document_commit
    assert seal["history_contract"] == HISTORY_CONTRACT
    assert seal["prelabel_attestation"] == MMA.PRELABEL_ATTESTATION
    assert seal["governance_seals"] == {
        label: {"path": relative, "sha256": digest}
        for label, (relative, digest) in GOVERNANCE_SEALS.items()
    }

    _write_json(root / AMENDMENT_SEAL_RELATIVE, seal)
    _git_commit(root, "separate seal")
    assert validate_model_matrix_amendment_seal(
        root / AMENDMENT_SEAL_RELATIVE,
        root=root,
    ) == seal


def test_optional_model_freeze_authorization_and_release_descendants_are_replayed(
    tmp_path: Path,
) -> None:
    root, _document, _seal_commit, seal = _sealed_repository(tmp_path)
    (root / "model-freeze.txt").write_text("frozen\n", encoding="utf-8")
    model_freeze = _git_commit(root, "model freeze")
    (root / "authorization.txt").write_text("authorized\n", encoding="utf-8")
    authorization = _git_commit(root, "authorization")
    (root / "release.txt").write_text("released\n", encoding="utf-8")
    release = _git_commit(root, "release")

    assert validate_model_matrix_amendment_seal(
        root / AMENDMENT_SEAL_RELATIVE,
        root=root,
        model_freeze_commit=model_freeze,
        authorization_commit=authorization,
        release_commit=release,
    ) == seal


def test_optional_model_freeze_must_strictly_follow_the_seal(
    tmp_path: Path,
) -> None:
    root, _document, seal_commit, _seal = _sealed_repository(tmp_path)

    with pytest.raises(ModelMatrixAmendmentError, match="not strict"):
        validate_model_matrix_amendment_seal(
            root / AMENDMENT_SEAL_RELATIVE,
            root=root,
            model_freeze_commit=seal_commit,
        )


def test_builder_rejects_a_preexisting_worktree_seal_path(tmp_path: Path) -> None:
    root, _governance, document_commit = _document_repository(tmp_path)
    _write_json(root / AMENDMENT_SEAL_RELATIVE, {"premature": True})

    with pytest.raises(ModelMatrixAmendmentError, match="already exists"):
        build_model_matrix_amendment_seal_document(
            root=root,
            amendment_document_commit=document_commit,
        )


def test_every_existing_governance_seal_must_strictly_precede_document(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    late_relative = GOVERNANCE_SEALS["inference_amendment_seal_v2"][0]
    for relative in GOVERNANCE_SHA256:
        if relative != late_relative:
            _copy_relative(ROOT, root, relative)
    _git_commit(root, "incomplete governance chain")
    _copy_relative(ROOT, root, late_relative)
    _copy_relative(ROOT, root, AMENDMENT_RELATIVE)
    document_commit = _git_commit(root, "forbidden same-commit governance seal")

    with pytest.raises(ModelMatrixAmendmentError, match="not strict"):
        build_model_matrix_amendment_seal_document(
            root=root,
            amendment_document_commit=document_commit,
        )


def test_same_commit_document_and_seal_is_rejected_by_lineage_replay(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative in GOVERNANCE_SHA256:
        _copy_relative(ROOT, root, relative)
    _git_commit(root, "governance")
    _copy_relative(ROOT, root, AMENDMENT_RELATIVE)
    payload = b'{"premature":"seal"}\n'
    seal_path = root / AMENDMENT_SEAL_RELATIVE
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_bytes(payload)
    same_commit = _git_commit(root, "forbidden document and seal together")
    (root / "later.txt").write_text("later\n", encoding="utf-8")
    _git_commit(root, "later tip")

    with pytest.raises(ModelMatrixAmendmentError, match="seal existed"):
        MMA._validate_model_matrix_amendment_seal_git_lineage(
            root=root,
            amendment_document_commit=same_commit,
            tip="HEAD",
            amendment_sha256=hashlib.sha256(
                (root / AMENDMENT_RELATIVE).read_bytes()
            ).hexdigest(),
            seal_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_seal_created_before_document_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative in GOVERNANCE_SHA256:
        _copy_relative(ROOT, root, relative)
    _git_commit(root, "governance")
    payload = b'{"premature":"seal"}\n'
    seal_path = root / AMENDMENT_SEAL_RELATIVE
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_bytes(payload)
    _git_commit(root, "premature seal")
    _copy_relative(ROOT, root, AMENDMENT_RELATIVE)
    document_commit = _git_commit(root, "later document")
    (root / "later.txt").write_text("later\n", encoding="utf-8")
    _git_commit(root, "later tip")

    with pytest.raises(ModelMatrixAmendmentError, match="seal existed"):
        MMA._validate_model_matrix_amendment_seal_git_lineage(
            root=root,
            amendment_document_commit=document_commit,
            tip="HEAD",
            amendment_sha256=hashlib.sha256(
                (root / AMENDMENT_RELATIVE).read_bytes()
            ).hexdigest(),
            seal_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_wrong_document_commit_blob_is_rejected_even_after_restoration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative in GOVERNANCE_SHA256:
        _copy_relative(ROOT, root, relative)
    _git_commit(root, "governance")
    attacked = json.loads((ROOT / AMENDMENT_RELATIVE).read_text(encoding="utf-8"))
    attacked["status"] = "ATTACKED_AT_DOCUMENT_COMMIT"
    _write_json(root / AMENDMENT_RELATIVE, attacked)
    wrong_document_commit = _git_commit(root, "wrong document blob")
    _copy_relative(ROOT, root, AMENDMENT_RELATIVE)
    _git_commit(root, "restore exact document")
    seal = MMA._seal_contract(root, wrong_document_commit)
    _write_json(root / AMENDMENT_SEAL_RELATIVE, seal)
    _git_commit(root, "seal after restored document")

    with pytest.raises(ModelMatrixAmendmentError, match="changed after freezing"):
        validate_model_matrix_amendment_seal(
            root / AMENDMENT_SEAL_RELATIVE,
            root=root,
        )


def test_document_delete_and_readd_is_rejected(tmp_path: Path) -> None:
    root, _governance, document_commit = _document_repository(tmp_path)
    document_path = root / AMENDMENT_RELATIVE
    original = document_path.read_bytes()
    document_path.unlink()
    _git_commit(root, "delete amendment")
    document_path.write_bytes(original)
    _git_commit(root, "re-add amendment")
    seal = MMA._seal_contract(root, document_commit)
    _write_json(root / AMENDMENT_SEAL_RELATIVE, seal)
    _git_commit(root, "seal")

    with pytest.raises(ModelMatrixAmendmentError, match="exactly one Git creation"):
        validate_model_matrix_amendment_seal(
            root / AMENDMENT_SEAL_RELATIVE,
            root=root,
        )


def test_seal_delete_and_readd_is_rejected(tmp_path: Path) -> None:
    root, _document, _seal_commit, _seal = _sealed_repository(tmp_path)
    seal_path = root / AMENDMENT_SEAL_RELATIVE
    original = seal_path.read_bytes()
    seal_path.unlink()
    _git_commit(root, "delete seal")
    seal_path.write_bytes(original)
    _git_commit(root, "re-add seal")

    with pytest.raises(ModelMatrixAmendmentError, match="exactly one Git creation"):
        validate_model_matrix_amendment_seal(seal_path, root=root)


def test_declared_document_commit_on_nonancestor_branch_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative in GOVERNANCE_SHA256:
        _copy_relative(ROOT, root, relative)
    _git_commit(root, "governance")
    main_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "switch", "-q", "-c", "side"], cwd=root, check=True)
    _copy_relative(ROOT, root, AMENDMENT_RELATIVE)
    side_document = _git_commit(root, "side-branch document")
    subprocess.run(["git", "switch", "-q", main_branch], cwd=root, check=True)
    _copy_relative(ROOT, root, AMENDMENT_RELATIVE)
    _git_commit(root, "main-branch document")
    seal = MMA._seal_contract(root, side_document)
    _write_json(root / AMENDMENT_SEAL_RELATIVE, seal)
    _git_commit(root, "main-branch seal naming side commit")

    with pytest.raises(ModelMatrixAmendmentError, match="chronology failed"):
        validate_model_matrix_amendment_seal(
            root / AMENDMENT_SEAL_RELATIVE,
            root=root,
        )


@pytest.mark.parametrize("target", ("amendment", "seal"))
def test_post_seal_modify_and_restore_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    root, _document, _seal_commit, _seal = _sealed_repository(tmp_path)
    relative = AMENDMENT_RELATIVE if target == "amendment" else AMENDMENT_SEAL_RELATIVE
    path = root / relative
    original = path.read_bytes()
    attacked = json.loads(original.decode("utf-8"))
    attacked["status"] = "TEMPORARILY_REWRITTEN"
    _write_json(path, attacked)
    _git_commit(root, f"modify {target}")
    path.write_bytes(original)
    _git_commit(root, f"restore {target}")

    with pytest.raises(ModelMatrixAmendmentError, match="changed after freezing"):
        validate_model_matrix_amendment_seal(
            root / AMENDMENT_SEAL_RELATIVE,
            root=root,
        )


def test_gitless_mode_requires_and_checks_both_outer_frozen_hashes(
    tmp_path: Path,
) -> None:
    root, _document, _seal_commit, seal = _sealed_repository(tmp_path / "source")
    archive = _gitless_archive(root, tmp_path / "archive")
    amendment_sha256 = hashlib.sha256(
        (archive / AMENDMENT_RELATIVE).read_bytes()
    ).hexdigest()
    seal_sha256 = hashlib.sha256(
        (archive / AMENDMENT_SEAL_RELATIVE).read_bytes()
    ).hexdigest()

    assert validate_model_matrix_amendment_seal(
        archive / AMENDMENT_SEAL_RELATIVE,
        root=archive,
        allow_gitless_archive=True,
        expected_amendment_sha256=amendment_sha256,
        expected_seal_sha256=seal_sha256,
    ) == seal
    with pytest.raises(ModelMatrixAmendmentError, match="requires outer-frozen"):
        validate_model_matrix_amendment_seal(
            archive / AMENDMENT_SEAL_RELATIVE,
            root=archive,
            allow_gitless_archive=True,
        )
    with pytest.raises(ModelMatrixAmendmentError, match="outer-frozen seal"):
        validate_model_matrix_amendment_seal(
            archive / AMENDMENT_SEAL_RELATIVE,
            root=archive,
            allow_gitless_archive=True,
            expected_amendment_sha256=amendment_sha256,
            expected_seal_sha256="0" * 64,
        )
    with pytest.raises(ModelMatrixAmendmentError, match="cannot prove"):
        validate_model_matrix_amendment_seal(
            archive / AMENDMENT_SEAL_RELATIVE,
            root=archive,
            allow_gitless_archive=True,
            expected_amendment_sha256=amendment_sha256,
            expected_seal_sha256=seal_sha256,
            model_freeze_commit="1" * 40,
        )


def test_future_seal_symlink_and_duplicate_key_attacks_fail_closed(
    tmp_path: Path,
) -> None:
    root, _document, _seal_commit, _seal = _sealed_repository(tmp_path / "source")
    archive = _gitless_archive(root, tmp_path / "archive")
    amendment_sha256 = hashlib.sha256(
        (archive / AMENDMENT_RELATIVE).read_bytes()
    ).hexdigest()
    seal_path = archive / AMENDMENT_SEAL_RELATIVE
    seal_path.unlink()
    seal_path.symlink_to(root / AMENDMENT_SEAL_RELATIVE)
    with pytest.raises(ModelMatrixAmendmentError, match="symlink"):
        validate_model_matrix_amendment_seal(
            seal_path,
            root=archive,
            allow_gitless_archive=True,
            expected_amendment_sha256=amendment_sha256,
            expected_seal_sha256=hashlib.sha256(
                (root / AMENDMENT_SEAL_RELATIVE).read_bytes()
            ).hexdigest(),
        )

    seal_path.unlink()
    seal_path.write_text(
        '{"format":"first","format":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ModelMatrixAmendmentError, match="duplicate JSON key"):
        validate_model_matrix_amendment_seal(
            seal_path,
            root=archive,
            allow_gitless_archive=True,
            expected_amendment_sha256=amendment_sha256,
            expected_seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
        )
