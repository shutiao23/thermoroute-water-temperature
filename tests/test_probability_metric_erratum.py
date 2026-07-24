from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from thermoroute.probability_metric_erratum import (
    BASE_PROTOCOL_RELATIVE,
    BASE_PROTOCOL_SEAL_RELATIVE,
    ERRATUM_FORMAT,
    ERRATUM_ID,
    ERRATUM_RELATIVE,
    ERRATUM_SEAL_FORMAT,
    ERRATUM_SEAL_RELATIVE,
    ERRATUM_SEAL_STATUS,
    HISTORY_CONTRACT,
    INFERENCE_AMENDMENT_RELATIVE,
    INFERENCE_AMENDMENT_SEAL_RELATIVE,
    PRELABEL_ATTESTATION,
    ProbabilityMetricErratumError,
    _validate_document_git_lineage,
    _validate_probability_metric_erratum_seal_git_lineage,
    build_probability_metric_erratum_seal_document,
    expected_probability_metric_erratum_document,
    validate_probability_metric_erratum,
    validate_probability_metric_erratum_seal,
)


ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_INPUTS = (
    BASE_PROTOCOL_RELATIVE,
    BASE_PROTOCOL_SEAL_RELATIVE,
    INFERENCE_AMENDMENT_RELATIVE,
    INFERENCE_AMENDMENT_SEAL_RELATIVE,
    ERRATUM_RELATIVE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(root / relative)}


def _copy_inputs(root: Path) -> None:
    for relative in GOVERNANCE_INPUTS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)


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


def _lineage_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "base.txt").write_text("base\n", encoding="utf-8")
    _git_commit(root, "base")
    return root


def _write_lineage_seal(root: Path, payload: bytes) -> None:
    path = root / ERRATUM_SEAL_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _complete_erratum_repository(tmp_path: Path) -> tuple[Path, str]:
    """Create a minimal valid repository through the erratum-document birth."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)

    protocol = {
        "protocol_id": "route-a-confirmatory-v1",
        "primary_inference_contract": {
            "confirmatory_family": [{"test_id": f"T{index}"} for index in range(1, 6)],
            "probabilistic_event_contract": {"role": "fixture-frozen-development-only"},
        },
    }
    _write_json(root / BASE_PROTOCOL_RELATIVE, protocol)
    base_commit = _git_commit(root, "base protocol")

    base_seal = {
        "format": "thermoroute.route-a-protocol-seal.v1",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "final_prelabel_protocol": {
            "commit": base_commit,
            "json": _binding(root, BASE_PROTOCOL_RELATIVE),
        },
    }
    _write_json(root / BASE_PROTOCOL_SEAL_RELATIVE, base_seal)
    _git_commit(root, "base protocol seal")

    amendment = {
        "format": "thermoroute.route-a-inference-amendment.v2",
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent": True,
    }
    _write_json(root / INFERENCE_AMENDMENT_RELATIVE, amendment)
    amendment_commit = _git_commit(root, "inference amendment")

    amendment_seal = {
        "format": "thermoroute.route-a-inference-amendment-seal.v2",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "amendment": _binding(root, INFERENCE_AMENDMENT_RELATIVE),
        "final_prelabel_commit": amendment_commit,
    }
    _write_json(root / INFERENCE_AMENDMENT_SEAL_RELATIVE, amendment_seal)
    _git_commit(root, "inference amendment seal")

    erratum = expected_probability_metric_erratum_document(root=root)
    _write_json(root / ERRATUM_RELATIVE, erratum)
    document_commit = _git_commit(root, "probability metric erratum")
    return root, document_commit


def _complete_sealed_erratum_repository(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root, document_commit = _complete_erratum_repository(tmp_path)
    seal = build_probability_metric_erratum_seal_document(
        root=root,
        erratum_document_commit=document_commit,
    )
    _write_json(root / ERRATUM_SEAL_RELATIVE, seal)
    _git_commit(root, "separate probability metric erratum seal")
    return root, seal


def _add_linked_worktree(root: Path, destination: Path) -> Path:
    subprocess.run(
        ["git", "worktree", "add", "-q", "--detach", str(destination), "HEAD"],
        cwd=root,
        check=True,
    )
    return destination


def _gitless_seal(root: Path, *, commit: str = "1" * 40) -> dict[str, object]:
    return {
        "format": ERRATUM_SEAL_FORMAT,
        "status": ERRATUM_SEAL_STATUS,
        "erratum_id": ERRATUM_ID,
        "erratum": _binding(root, ERRATUM_RELATIVE),
        "governance_seals": {
            "base_protocol_seal": _binding(root, BASE_PROTOCOL_SEAL_RELATIVE),
            "inference_amendment_seal": _binding(
                root,
                INFERENCE_AMENDMENT_SEAL_RELATIVE,
            ),
        },
        "erratum_document_commit": commit,
        "history_contract": HISTORY_CONTRACT,
        "prelabel_attestation": PRELABEL_ATTESTATION,
    }


def test_production_erratum_is_exact_and_outcome_free() -> None:
    document = validate_probability_metric_erratum(ROOT / ERRATUM_RELATIVE, root=ROOT)

    assert document["format"] == ERRATUM_FORMAT
    assert document["erratum_id"] == ERRATUM_ID
    assert document["prelabel_attestation"] == PRELABEL_ATTESTATION
    assert document["scientific_scope"]["confirmatory_family_count"] == 5
    assert document["scientific_scope"]["formal_comparisons_changed"] is False
    assert document["scientific_scope"]["formal_margins_changed"] is False
    assert document["scientific_scope"]["formal_decisions_changed"] is False
    corrected = document["corrected_metric_contract"]
    assert corrected["interval_coverage_and_width_source"] == ("bundle_frozen_2018_cqr_endpoints")
    assert corrected["pinball_quantile_source"] == (
        "direct_nominal_pre_cqr_ensemble_q05_q50_q95_retained_in_memory_before_cqr"
    )
    assert corrected["nominal_quantile_handling"] == {
        "source": ("direct_nominal_pre_cqr_ensemble_q05_q50_q95_retained_in_memory_before_cqr"),
        "storage": ("transient_in_memory_only_not_written_to_public_prediction_products"),
        "member_aggregation": "equal_weight_member_mean_before_cqr",
        "cqr_forward_parity": (
            "bitwise_float64_nominal_q05_minus_offset_and_nominal_q95_plus_"
            "offset_equal_stored_endpoints"
        ),
        "q50_forward_parity": "bitwise_nominal_q50_equal_stored_q50",
        "endpoint_inversion_used": False,
        "public_prediction_schema_changed": False,
    }
    assert corrected["bundle_scoring_pipeline_contracts"]["LightGBM"] == (
        "bundle_declared_median_preserving_endpoint_clip_per_member_then_"
        "equal_weight_member_mean_then_frozen_cqr"
    )
    assert corrected["pinball_aggregation"]["called_CRPS"] is False
    assert corrected["all_event_probability_metrics_use_post_Platt_probability"] is True
    unchanged = document["unchanged_contract"]
    assert unchanged["model_definition_changed"] is False
    assert unchanged["fit_data_changed"] is False
    assert unchanged["frozen_hyperparameters_changed"] is False
    assert unchanged["hyperparameter_retuning_allowed"] is False
    assert unchanged["old_pre_erratum_model_artifacts_eligible_for_final_freeze"] is False
    assert unchanged["full_retraining_and_replay_under_new_source_required"] is True
    assert unchanged["retraining_role"] == (
        "mechanical_reproduction_under_the_unchanged_frozen_contract_not_retuning"
    )
    assert unchanged["confirmation_data_may_change_or_reestimate_CQR"] is False
    assert unchanged["confirmation_data_may_change_or_reestimate_Platt"] is False
    assert unchanged["confirmation_data_may_change_or_reestimate_event_threshold"] is False
    assert unchanged["mechanical_recomputation_contract"] == {
        "required": True,
        "components": ["model_fits", "CQR", "Platt", "event_threshold"],
        "fit_inputs": "identical_frozen_development_only_fit_inputs",
        "algorithm": "identical_frozen_algorithm",
        "parameters": "identical_frozen_fitting_parameters",
        "purpose": "deterministic_mechanical_reproduction_only",
        "confirmation_data_used_for_fit": False,
        "retuning_or_adaptation_allowed": False,
    }
    implementation = document["implementation_requirements"]
    assert implementation["trusted_artifact_format"].endswith("probabilistic-evaluation.v2")
    assert implementation["trusted_artifact_path"].endswith("probabilistic_evaluation_v2.json")
    assert implementation["transient_nominal_quantiles_required_for_v2_replay"] is True
    assert implementation["bitwise_forward_cqr_parity_required"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prelabel_attestation", {**PRELABEL_ATTESTATION, "network_used": True}),
        ("status", "FROZEN_AFTER_LABEL_ACCESS"),
    ),
)
def test_erratum_rejects_attestation_and_schema_tamper(
    tmp_path: Path, field: str, value: object
) -> None:
    _copy_inputs(tmp_path)
    path = tmp_path / ERRATUM_RELATIVE
    attacked = json.loads(path.read_text(encoding="utf-8"))
    attacked[field] = value
    path.write_text(json.dumps(attacked), encoding="utf-8")

    with pytest.raises(ProbabilityMetricErratumError, match="stale|tampered"):
        validate_probability_metric_erratum(path, root=tmp_path)


def test_erratum_rejects_governance_and_metric_contract_drift(
    tmp_path: Path,
) -> None:
    _copy_inputs(tmp_path)
    path = tmp_path / ERRATUM_RELATIVE
    attacked = json.loads(path.read_text(encoding="utf-8"))
    attacked["corrected_metric_contract"]["pinball_quantile_source"] = "post_CQR_endpoints"
    path.write_text(json.dumps(attacked), encoding="utf-8")
    with pytest.raises(ProbabilityMetricErratumError, match="stale|tampered"):
        validate_probability_metric_erratum(path, root=tmp_path)

    shutil.copyfile(ROOT / ERRATUM_RELATIVE, path)
    protocol = tmp_path / BASE_PROTOCOL_RELATIVE
    protocol.write_bytes(protocol.read_bytes() + b"\n")
    with pytest.raises(ProbabilityMetricErratumError, match="protocol|seal"):
        validate_probability_metric_erratum(path, root=tmp_path)


def test_gitless_seal_validates_exact_schema_hashes_and_attestation(
    tmp_path: Path,
) -> None:
    _copy_inputs(tmp_path)
    seal = {
        "format": ERRATUM_SEAL_FORMAT,
        "status": ERRATUM_SEAL_STATUS,
        "erratum_id": ERRATUM_ID,
        "erratum": _binding(tmp_path, ERRATUM_RELATIVE),
        "governance_seals": {
            "base_protocol_seal": _binding(tmp_path, BASE_PROTOCOL_SEAL_RELATIVE),
            "inference_amendment_seal": _binding(tmp_path, INFERENCE_AMENDMENT_SEAL_RELATIVE),
        },
        "erratum_document_commit": "1" * 40,
        "history_contract": HISTORY_CONTRACT,
        "prelabel_attestation": PRELABEL_ATTESTATION,
    }
    path = tmp_path / ERRATUM_SEAL_RELATIVE
    path.write_text(json.dumps(seal), encoding="utf-8")

    assert (
        validate_probability_metric_erratum_seal(path, root=tmp_path, allow_gitless_archive=True)
        == seal
    )

    attacked = dict(seal)
    attacked["erratum_document_commit"] = "1" * 39
    path.write_text(json.dumps(attacked), encoding="utf-8")
    with pytest.raises(ProbabilityMetricErratumError, match="commit is malformed"):
        validate_probability_metric_erratum_seal(path, root=tmp_path, allow_gitless_archive=True)


def test_seal_builder_and_live_validator_accept_strict_valid_history(
    tmp_path: Path,
) -> None:
    root, document_commit = _complete_erratum_repository(tmp_path)

    seal = build_probability_metric_erratum_seal_document(
        root=root,
        erratum_document_commit=document_commit,
    )

    assert seal["erratum_document_commit"] == document_commit
    assert seal["history_contract"] == HISTORY_CONTRACT
    assert seal["history_contract"]["erratum_document_created_exactly_once"] is True
    _write_json(root / ERRATUM_SEAL_RELATIVE, seal)
    _git_commit(root, "separate probability metric erratum seal")
    assert validate_probability_metric_erratum_seal(root / ERRATUM_SEAL_RELATIVE, root=root) == seal


def test_live_validator_accepts_a_real_linked_worktree_gitfile(
    tmp_path: Path,
) -> None:
    main, seal = _complete_sealed_erratum_repository(tmp_path / "main-fixture")
    linked = _add_linked_worktree(main, tmp_path / "linked")
    marker = linked / ".git"

    assert marker.is_file() and not marker.is_symlink()
    assert marker.read_text(encoding="utf-8").startswith("gitdir: ")
    assert validate_probability_metric_erratum_seal(
        linked / ERRATUM_SEAL_RELATIVE,
        root=linked,
    ) == seal


@pytest.mark.parametrize("attack", ("root_alias", "git_marker"))
def test_seal_builder_rejects_root_and_git_marker_symlinks(
    tmp_path: Path,
    attack: str,
) -> None:
    root, document_commit = _complete_erratum_repository(tmp_path)
    attacked_root = root
    if attack == "root_alias":
        attacked_root = tmp_path / "repo-alias"
        attacked_root.symlink_to(root, target_is_directory=True)
    else:
        marker = root / ".git"
        actual = root / "actual-git"
        marker.rename(actual)
        marker.symlink_to(actual.name, target_is_directory=True)

    with pytest.raises(
        ProbabilityMetricErratumError,
        match="symlink|safe .git marker",
    ):
        build_probability_metric_erratum_seal_document(
            root=attacked_root,
            erratum_document_commit=document_commit,
        )


@pytest.mark.parametrize("attack", ("gitdir", "commondir"))
def test_live_validator_rejects_symlinked_linked_worktree_metadata(
    tmp_path: Path,
    attack: str,
) -> None:
    main, _seal = _complete_sealed_erratum_repository(tmp_path / "main-fixture")
    linked = _add_linked_worktree(main, tmp_path / "linked")
    marker = linked / ".git"
    raw_git_directory = marker.read_text(encoding="utf-8").strip()
    assert raw_git_directory.startswith("gitdir: ")
    git_directory = Path(raw_git_directory.removeprefix("gitdir: "))

    if attack == "gitdir":
        alias = tmp_path / "gitdir-alias"
        alias.symlink_to(git_directory.parent, target_is_directory=True)
        marker.write_text(
            f"gitdir: {alias / git_directory.name}\n",
            encoding="utf-8",
        )
    else:
        common = main / ".git"
        alias = tmp_path / "commondir-alias"
        alias.symlink_to(common, target_is_directory=True)
        (git_directory / "commondir").write_text(
            f"{alias}\n",
            encoding="utf-8",
        )

    with pytest.raises(ProbabilityMetricErratumError, match="symlink"):
        validate_probability_metric_erratum_seal(
            linked / ERRATUM_SEAL_RELATIVE,
            root=linked,
        )


def test_gitless_archive_does_not_discover_an_ancestor_repository(
    tmp_path: Path,
) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=outer, check=True)
    archive = outer / "archive"
    archive.mkdir()
    _copy_inputs(archive)
    seal = _gitless_seal(archive)
    _write_json(archive / ERRATUM_SEAL_RELATIVE, seal)
    _git_commit(outer, "gitless archive inside an ancestor repository")

    assert validate_probability_metric_erratum_seal(
        archive / ERRATUM_SEAL_RELATIVE,
        root=archive,
        allow_gitless_archive=True,
    ) == seal


def test_gitless_mode_rejects_an_unsafe_local_git_marker(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    _copy_inputs(archive)
    seal = _gitless_seal(archive)
    _write_json(archive / ERRATUM_SEAL_RELATIVE, seal)
    attacker_git = tmp_path / "attacker.git"
    attacker_git.mkdir()
    (archive / ".git").symlink_to(attacker_git, target_is_directory=True)

    with pytest.raises(ProbabilityMetricErratumError, match="safe .git marker"):
        validate_probability_metric_erratum_seal(
            archive / ERRATUM_SEAL_RELATIVE,
            root=archive,
            allow_gitless_archive=True,
        )


@pytest.mark.parametrize(
    ("variable", "value"),
    (
        ("GIT_DIR", "/attacker/redirected.git"),
        ("GIT_INDEX_FILE", "/attacker/redirected.index"),
        ("GIT_CONFIG_COUNT", "0"),
        ("GIT_CONFIG_KEY_0", "include.path"),
        ("GIT_CONFIG_VALUE_0", "/attacker/config"),
        ("GIT_NO_REPLACE_OBJECTS", "0"),
    ),
)
def test_seal_builder_rejects_ambient_git_redirects_and_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    root, document_commit = _complete_erratum_repository(tmp_path)
    monkeypatch.setenv(variable, value)

    with pytest.raises(ProbabilityMetricErratumError, match="ambient Git"):
        build_probability_metric_erratum_seal_document(
            root=root,
            erratum_document_commit=document_commit,
        )


def test_seal_builder_rejects_git_replacement_refs(tmp_path: Path) -> None:
    root, document_commit = _complete_erratum_repository(tmp_path)
    parent = subprocess.run(
        ["git", "rev-parse", f"{document_commit}^"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "replace", document_commit, parent], cwd=root, check=True)

    with pytest.raises(ProbabilityMetricErratumError, match="replacement refs"):
        build_probability_metric_erratum_seal_document(
            root=root,
            erratum_document_commit=document_commit,
        )


def test_seal_builder_rejects_shallow_repository(tmp_path: Path) -> None:
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    source, document_commit = _complete_erratum_repository(source_parent)
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth",
            "1",
            source.as_uri(),
            str(shallow),
        ],
        check=True,
    )

    with pytest.raises(ProbabilityMetricErratumError, match="shallow"):
        build_probability_metric_erratum_seal_document(
            root=shallow,
            erratum_document_commit=document_commit,
        )


@pytest.mark.parametrize("relative", ("info/grafts", "objects/info/alternates"))
def test_seal_builder_rejects_grafts_and_object_alternates(tmp_path: Path, relative: str) -> None:
    root, document_commit = _complete_erratum_repository(tmp_path)
    attack_path = root / ".git" / relative
    attack_path.parent.mkdir(parents=True, exist_ok=True)
    attack_path.write_text("", encoding="utf-8")

    with pytest.raises(ProbabilityMetricErratumError, match="prohibits Git"):
        build_probability_metric_erratum_seal_document(
            root=root,
            erratum_document_commit=document_commit,
        )


def test_seal_builder_rejects_nonmatching_git_top_level(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=outer, check=True)
    nested = outer / "nested"
    nested.mkdir()
    _copy_inputs(nested)
    document_commit = _git_commit(outer, "governance fixture in nested directory")
    subprocess.run(["git", "config", "core.worktree", str(outer)], cwd=outer, check=True)
    (nested / ".git").write_text("gitdir: ../.git\n", encoding="utf-8")

    with pytest.raises(ProbabilityMetricErratumError, match="exact Git top-level"):
        build_probability_metric_erratum_seal_document(
            root=nested,
            erratum_document_commit=document_commit,
        )


def test_seal_builder_rejects_preexisting_document_declaration(
    tmp_path: Path,
) -> None:
    root, _birth = _complete_erratum_repository(tmp_path)
    (root / "unrelated.txt").write_text("later\n", encoding="utf-8")
    later_commit = _git_commit(root, "unrelated later commit")

    with pytest.raises(
        ProbabilityMetricErratumError,
        match="exactly one Git creation equal to its declared",
    ):
        build_probability_metric_erratum_seal_document(
            root=root,
            erratum_document_commit=later_commit,
        )


def test_seal_builder_rejects_document_delete_and_readd(
    tmp_path: Path,
) -> None:
    root, _birth = _complete_erratum_repository(tmp_path)
    erratum_path = root / ERRATUM_RELATIVE
    original = erratum_path.read_bytes()
    erratum_path.unlink()
    _git_commit(root, "delete erratum document")
    erratum_path.write_bytes(original)
    readd_commit = _git_commit(root, "re-add identical erratum document")

    with pytest.raises(
        ProbabilityMetricErratumError,
        match="exactly one Git creation equal to its declared",
    ):
        build_probability_metric_erratum_seal_document(
            root=root,
            erratum_document_commit=readd_commit,
        )


def test_seal_builder_rejects_same_document_blob_at_later_declaration(
    tmp_path: Path,
) -> None:
    root, _birth = _complete_erratum_repository(tmp_path)
    erratum_path = root / ERRATUM_RELATIVE
    original = erratum_path.read_bytes()
    attacked = json.loads(original.decode("utf-8"))
    attacked["status"] = "TEMPORARILY_CHANGED"
    _write_json(erratum_path, attacked)
    _git_commit(root, "temporarily modify erratum document")
    erratum_path.write_bytes(original)
    restored_commit = _git_commit(root, "restore the exact original blob")

    with pytest.raises(
        ProbabilityMetricErratumError,
        match="exactly one Git creation equal to its declared",
    ):
        build_probability_metric_erratum_seal_document(
            root=root,
            erratum_document_commit=restored_commit,
        )


@pytest.mark.parametrize("symlink_component", ("document", "directory"))
def test_erratum_rejects_symlinked_governance_inputs(
    tmp_path: Path, symlink_component: str
) -> None:
    _copy_inputs(tmp_path)
    if symlink_component == "document":
        canonical = tmp_path / ERRATUM_RELATIVE
        actual = canonical.with_name("actual_erratum.json")
        canonical.rename(actual)
        canonical.symlink_to(actual.name)
    else:
        canonical_directory = tmp_path / "protocols"
        actual_directory = tmp_path / "actual_protocols"
        canonical_directory.rename(actual_directory)
        canonical_directory.symlink_to(actual_directory.name, target_is_directory=True)

    with pytest.raises(ProbabilityMetricErratumError, match="symlink"):
        validate_probability_metric_erratum(
            tmp_path / ERRATUM_RELATIVE,
            root=tmp_path,
        )


def test_seal_lineage_accepts_one_strict_immutable_birth(
    tmp_path: Path,
) -> None:
    root = _lineage_repository(tmp_path)
    erratum = root / ERRATUM_RELATIVE
    erratum.parent.mkdir(parents=True, exist_ok=True)
    erratum.write_text("{}\n", encoding="utf-8")
    document_commit = _git_commit(root, "erratum document")
    payload = b'{"seal":"canonical"}\n'
    _write_lineage_seal(root, payload)
    creation = _git_commit(root, "separate erratum seal")

    assert (
        _validate_probability_metric_erratum_seal_git_lineage(
            root=root,
            erratum_document_commit=document_commit,
            tip="HEAD",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
        == creation
    )


def test_inference_amendment_seal_must_strictly_precede_erratum_document(
    tmp_path: Path,
) -> None:
    root = _lineage_repository(tmp_path)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    base_seal = root / BASE_PROTOCOL_SEAL_RELATIVE
    base_seal.parent.mkdir(parents=True, exist_ok=True)
    base_seal.write_text(
        json.dumps({"final_prelabel_protocol": {"commit": base_commit}}),
        encoding="utf-8",
    )
    _git_commit(root, "base protocol seal")
    amendment = root / INFERENCE_AMENDMENT_RELATIVE
    amendment.write_text("{}\n", encoding="utf-8")
    amendment_commit = _git_commit(root, "inference amendment")

    amendment_seal = root / INFERENCE_AMENDMENT_SEAL_RELATIVE
    amendment_seal.write_text(
        json.dumps({"final_prelabel_commit": amendment_commit}),
        encoding="utf-8",
    )
    erratum = root / ERRATUM_RELATIVE
    erratum.write_text("{}\n", encoding="utf-8")
    document_commit = _git_commit(root, "forbidden same-commit amendment seal and erratum")

    with pytest.raises(ProbabilityMetricErratumError, match="strictly pre-erratum"):
        _validate_document_git_lineage(
            root=root,
            erratum_document_commit=document_commit,
            tip="HEAD",
            erratum_file=erratum,
        )


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("same_commit", "existed at the document commit"),
        ("preexisting", "existed at the document commit"),
        ("delete_readd", "exactly one Git creation"),
        ("modify_restore", "changed after freezing"),
    ),
)
def test_seal_lineage_rejects_adversarial_histories(
    tmp_path: Path, attack: str, error: str
) -> None:
    root = _lineage_repository(tmp_path)
    erratum = root / ERRATUM_RELATIVE
    erratum.parent.mkdir(parents=True, exist_ok=True)
    payload = b'{"seal":"canonical"}\n'

    if attack == "same_commit":
        erratum.write_text("{}\n", encoding="utf-8")
        _write_lineage_seal(root, payload)
        document_commit = _git_commit(root, "erratum and seal")
    elif attack == "preexisting":
        _write_lineage_seal(root, payload)
        _git_commit(root, "premature seal")
        erratum.write_text("{}\n", encoding="utf-8")
        document_commit = _git_commit(root, "later erratum")
    else:
        erratum.write_text("{}\n", encoding="utf-8")
        document_commit = _git_commit(root, "erratum document")
        _write_lineage_seal(root, payload)
        _git_commit(root, "seal")
        if attack == "delete_readd":
            (root / ERRATUM_SEAL_RELATIVE).unlink()
            _git_commit(root, "delete seal")
            _write_lineage_seal(root, payload)
            _git_commit(root, "re-add seal")
        else:
            _write_lineage_seal(root, b'{"seal":"changed"}\n')
            _git_commit(root, "modify seal")
            _write_lineage_seal(root, payload)
            _git_commit(root, "restore seal")

    with pytest.raises(ProbabilityMetricErratumError, match=error):
        _validate_probability_metric_erratum_seal_git_lineage(
            root=root,
            erratum_document_commit=document_commit,
            tip="HEAD",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
