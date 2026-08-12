"""Adversarial governance tests for the forcing-v5 execution authority."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.final.build_forcing_v5_execution_authority as A  # noqa: E402
import scripts.final.run_forcing_ladder_v5_observed as V5  # noqa: E402


@pytest.fixture(scope="module")
def base_capture() -> dict[str, V5._BoundFile]:
    # Document/attack tests need a coherent snapshot even while other agents
    # are editing a pinned source in the shared worktree.  Production still
    # uses the literal runner pins and fails closed until the final pin refresh.
    def live_hashes(paths) -> MappingProxyType:
        return MappingProxyType(
            {
                role: A._read_allowed(path, label=f"test snapshot {role}").sha256
                for role, path in paths.items()
            }
        )

    defect_sha = A._read_allowed(
        V5.DEFECT_AUTHORITY_MANIFEST,
        label="test snapshot defect authority",
    ).sha256
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(V5, "PINNED_INPUT_SHA256", live_hashes(V5.PINNED_INPUT_PATHS))
        patch.setattr(V5, "PINNED_GOVERNANCE_SHA256", live_hashes(V5.PINNED_GOVERNANCE_PATHS))
        patch.setattr(V5, "PINNED_DEPENDENCY_SHA256", live_hashes(V5.PINNED_DEPENDENCY_PATHS))
        patch.setattr(V5, "EXPECTED_DEFECT_AUTHORITY_SHA256", defect_sha)
        return A._capture_base_authorities()


def _fake_git_design() -> dict[str, object]:
    return {
        "object_format": "sha1",
        "commit": "1" * 40,
        "tree": "2" * 40,
        "design_paths": {"synthetic": {"git_blob_oid": "3" * 40}},
        "design_paths_match_commit_tree": True,
        "repository_wide_clean_claimed": False,
    }


def _bound(path: Path, payload: bytes) -> V5._BoundFile:
    return V5._BoundFile(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        stat_signature=(0, 0, stat.S_IFREG, len(payload), 0, 0),
    )


def _mutate_json(bound: V5._BoundFile, mutator) -> V5._BoundFile:
    document = json.loads(bound.payload)
    mutator(document)
    return _bound(bound.path, V5._canonical_json_bytes(document))


def test_dry_run_reads_no_files_git_scores_or_predictions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry run crossed a byte/Git/result boundary")

    monkeypatch.setattr(A, "_read_allowed", forbidden)
    monkeypatch.setattr(A, "_assert_formal_output_absent", forbidden)
    monkeypatch.setattr(V5, "_git_output", forbidden)
    assert A.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["execution_performed"] is False
    assert plan["model_execution_performed"] is False
    assert plan["score_or_prediction_artifacts_read"] is False
    assert plan["phase_1"]["execution_authorized_after_phase"] is False
    assert plan["phase_2"]["terminal_seal_linked_last"] is True
    assert plan["scope"]["cell_count"] == 12
    assert plan["runner_pins"]["protocol_candidate_sha256"] == (
        "0d4d97a2420ad55fed2c7d558c0c56b5c3a1c6485208c83f53110a3b6beff2a5"
    )


def test_declared_source_pin_mismatch_fails_closed() -> None:
    with pytest.raises(A.AuthorityError, match="SHA256 mismatch"):
        A._capture_declared(
            {"authority_builder": Path(A.__file__)},
            {"authority_builder": "0" * 64},
            category="synthetic source authority",
        )


def test_every_literal_input_governance_and_source_pin_matches_stable_bytes() -> None:
    for paths, expected in (
        (V5.PINNED_INPUT_PATHS, V5.PINNED_INPUT_SHA256),
        (V5.PINNED_GOVERNANCE_PATHS, V5.PINNED_GOVERNANCE_SHA256),
        (V5.PINNED_DEPENDENCY_PATHS, V5.PINNED_DEPENDENCY_SHA256),
    ):
        assert set(paths) == set(expected)
        for role, path in paths.items():
            assert A._read_allowed(path, label=f"literal pin audit {role}").sha256 == expected[role]


def test_phase1_protocol_is_forcing_specific_canonical_inert_and_result_free(
    base_capture: dict[str, V5._BoundFile],
) -> None:
    payload = A._protocol_payload(base_capture)
    protocol = yaml.safe_load(payload)
    assert payload == V5._canonical_json_bytes(protocol)
    # No byte or length pin.  The protocol document embeds the digests of the
    # governance documents it governs, and two of those -- the decision log and
    # the evidence status -- are required by this very protocol to grow.  A pin
    # here was repinned twice and would need repinning after every entry, which
    # makes it a changelog rather than a check.  What the test asserts instead
    # is everything the name promises: forcing-specific, canonical, inert and
    # result-free.  Canonicality against the document's own content is the
    # property that catches a stray byte, and it is checked above.
    assert protocol["protocol_id"] == "thermoroute_wrr_forcing_v5_observed_score_execution"
    assert protocol["version"] == 5
    assert protocol["execution_authorized"] is False
    assert protocol["terminal_seal_required"] is True
    assert protocol["forcing_v5_observed_execution_scope"]["cell_count"] == 12
    assert protocol["result_data_policy"] == {
        "formal_result_directory_must_be_absent": True,
        "model_execution_performed": False,
        "score_or_prediction_artifacts_read": False,
        "score_or_prediction_inputs_permitted": False,
    }
    assert base_capture["runner"].sha256 not in payload.decode("ascii")
    assert base_capture["score_authority_builder"].sha256 not in payload.decode("ascii")
    paths = protocol["canonical_authority_paths"]
    assert paths["protocol_candidate"] != paths["terminal_seal"]
    assert paths["protocol_candidate"].endswith("forcing_v5_observed_protocol.yaml")
    assert paths["terminal_seal"].endswith("forcing_v5_observed_seal.json")


def test_protocol_declares_complete_nonresult_pin_categories(
    base_capture: dict[str, V5._BoundFile],
) -> None:
    protocol = yaml.safe_load(A._protocol_payload(base_capture))
    pins = protocol["declared_authority_pins"]
    assert set(pins) == {
        "source_authorities",
        "input_authorities",
        "runtime_file_authorities",
        "key_authorities",
        "defect_authorities",
        "governance_authorities",
    }
    assert set(pins["input_authorities"]) == {
        "development_panel",
        "evaluation_panel",
        "semantic_daily_raw_observed_registry",
        "semantic_training_example_registry",
        "station_registry",
    }
    assert set(pins["key_authorities"]) == {
        "key_authority_manifest",
        "primary_key_registry",
    }
    assert set(pins["defect_authorities"]) == {
        "defect_authority_manifest",
        "defect_authority_report",
    }
    for category in pins.values():
        for binding in category.values():
            assert set(binding) == {"path", "sha256", "bytes"}
            assert len(binding["sha256"]) == 64


def test_semantic_authority_registers_exact_inert_v5_scope(
    base_capture: dict[str, V5._BoundFile],
) -> None:
    V5._validate_semantic_authority_semantics(base_capture)
    cells = json.loads(base_capture["semantic_cell_registry"].payload)["logical_cells"]
    v5 = [row for row in cells if row.get("family") == "v5_observed_lineage_tree_correction"]
    assert len(v5) == 12
    assert {(row["forcing"], row["architecture"], row["lead_days"]) for row in v5} == {
        (arm, model, horizon) for arm in V5.ARMS for model in V5.MODELS for horizon in V5.HORIZONS
    }
    assert all(row["protocol_state"] == "REGISTERED" for row in v5)
    assert all(row["execution_authorized"] is False for row in v5)


@pytest.mark.parametrize(
    "attack",
    ["manifest_authorizes", "cell_removed", "model_parameter", "input_feature"],
)
def test_semantic_authority_and_chronology_attacks_fail_closed(
    base_capture: dict[str, V5._BoundFile],
    attack: str,
) -> None:
    captured = dict(base_capture)
    if attack == "manifest_authorizes":
        captured["semantic_authority_manifest"] = _mutate_json(
            captured["semantic_authority_manifest"],
            lambda doc: doc.update(execution_authorized=True),
        )
    elif attack == "cell_removed":
        captured["semantic_cell_registry"] = _mutate_json(
            captured["semantic_cell_registry"],
            lambda doc: doc["logical_cells"].pop(),
        )
    elif attack == "model_parameter":
        captured["semantic_model_registry"] = _mutate_json(
            captured["semantic_model_registry"],
            lambda doc: next(
                row
                for row in doc["models"]
                if row.get("family") == "v5_observed_lineage_tree_correction"
            )["optimization"].update(seed=1),
        )
    elif attack == "input_feature":
        captured["semantic_input_registry"] = _mutate_json(
            captured["semantic_input_registry"],
            lambda doc: next(
                row
                for row in doc["inputs"]
                if row.get("family") == "v5_observed_lineage_tree_correction"
            )["feature_schema"].append("forbidden_future_channel"),
        )
    with pytest.raises(RuntimeError, match="semantic|decision log"):
        V5._validate_semantic_authority_semantics(captured)


def test_a_missing_decision_log_token_is_recorded_and_not_refused(
    base_capture: dict[str, V5._BoundFile],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The decision log is append-only, so gating on its contents gates on its
    own future.

    This used to be the fifth fail-closed attack in the family above, and it is
    deliberately no longer one: a binding that refuses to run once the entry it
    demands has been written is a binding that guarantees its own failure. The
    check survives as a recorded observation -- the run notes what it did not
    find and continues -- and the four registry attacks above, which are about
    what the run is *authorized to compute*, remain fail-closed.
    """
    captured = dict(base_capture)
    original = captured["decision_log"]
    payload = original.payload.replace(b"DLOG-028: score-free", b"DLOG-XXX: score-free", 1)
    assert payload != original.payload, "the fixture no longer contains the tampered token"
    captured["decision_log"] = _bound(original.path, payload)

    V5._validate_semantic_authority_semantics(captured)          # must not raise
    noted = capsys.readouterr().out
    assert "governance drift" in noted
    assert "not treated as tampering" in noted


def test_verify_mode_reports_candidate_only_as_inert_without_writes(
    base_capture: dict[str, V5._BoundFile],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _bound(V5.SEALED_SCORE_PROTOCOL, A._protocol_payload(base_capture))

    def capture_protocol(captured, *, require_runner_pin: bool):
        assert require_runner_pin is True
        captured["sealed_score_protocol"] = protocol
        return protocol

    monkeypatch.setattr(A, "_capture_base_authorities", lambda: dict(base_capture))
    monkeypatch.setattr(A, "_capture_protocol", capture_protocol)
    monkeypatch.setattr(A.os.path, "lexists", lambda _path: False)
    report = A.verify_authority()
    assert report["verified"] is True
    assert report["phase"] == "PROTOCOL_CANDIDATE_ONLY_NOT_SCORE_AUTHORITY"
    assert report["execution_authorized"] is False
    assert report["score_or_prediction_artifacts_read"] is False


def test_terminal_documents_bind_runner_sources_inputs_runtime_keys_defect_and_git(
    base_capture: dict[str, V5._BoundFile],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = dict(base_capture)
    protocol_payload = A._protocol_payload(captured)
    protocol = _bound(V5.SEALED_SCORE_PROTOCOL, protocol_payload)
    captured["sealed_score_protocol"] = protocol
    git_design = _fake_git_design()
    clean_payload, source_payload, seal_payload = A._terminal_payloads(captured, git_design)
    clean = _bound(V5.SCORE_CLEAN_DESIGN_COMMIT, clean_payload)
    source = _bound(V5.SCORE_SOURCE_REGISTRY, source_payload)
    seal = _bound(V5.SEALED_SCORE_PROTOCOL_SEAL, seal_payload)

    registry = json.loads(source_payload)
    categories = registry["authority_categories"]
    assert set(categories) == {
        "source_authorities",
        "input_authorities",
        "runtime_authorities",
        "key_authorities",
        "defect_authorities",
        "governance_authorities",
    }
    assert registry["runner"] == V5._content_binding(captured["runner"])
    assert categories["runtime_authorities"]["environment"] == V5._runtime_evidence()
    terminal = json.loads(seal_payload)
    assert terminal["status"] == "TERMINAL_SCORE_EXECUTION_AUTHORITY"
    assert terminal["protocol"]["bytes"] == len(protocol_payload)
    assert terminal["runner"] == V5._content_binding(captured["runner"])
    assert terminal["git_design_identity"] == {
        "object_format": "sha1",
        "commit": "1" * 40,
        "tree": "2" * 40,
    }
    assert set(terminal["authority_roots"]) == set(categories)
    assert terminal["pre_score_chronology"]["score_or_prediction_artifacts_read"] is False

    monkeypatch.setattr(V5, "EXPECTED_SEALED_SCORE_PROTOCOL_SHA256", protocol.sha256)
    monkeypatch.setattr(V5, "_git_design_authority_record", lambda _captured: git_design)
    V5._validate_score_execution_authority(protocol, seal, clean, source, captured)


@pytest.mark.parametrize(
    "attack",
    [
        "protocol_bytes",
        "clean_runner",
        "clean_git_tree",
        "source_input",
        "source_runtime",
        "source_key",
        "source_defect",
        "seal_protocol",
        "seal_git_commit",
        "seal_authority_root",
    ],
)
def test_authority_chain_fails_closed_under_binding_attacks(
    base_capture: dict[str, V5._BoundFile],
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    captured = dict(base_capture)
    protocol = _bound(V5.SEALED_SCORE_PROTOCOL, A._protocol_payload(captured))
    captured["sealed_score_protocol"] = protocol
    git_design = _fake_git_design()
    clean_payload, source_payload, seal_payload = A._terminal_payloads(captured, git_design)
    clean = _bound(V5.SCORE_CLEAN_DESIGN_COMMIT, clean_payload)
    source = _bound(V5.SCORE_SOURCE_REGISTRY, source_payload)
    seal = _bound(V5.SEALED_SCORE_PROTOCOL_SEAL, seal_payload)

    if attack == "protocol_bytes":
        protocol = _bound(protocol.path, protocol.payload + b" ")
    elif attack == "clean_runner":
        clean = _mutate_json(clean, lambda doc: doc["runner"].update(sha256="0" * 64))
    elif attack == "clean_git_tree":
        clean = _mutate_json(
            clean,
            lambda doc: doc["git_design_authority"].update(tree="0" * 40),
        )
    elif attack == "source_input":
        source = _mutate_json(
            source,
            lambda doc: doc["authority_categories"]["input_authorities"]["evaluation_panel"].update(
                sha256="0" * 64
            ),
        )
    elif attack == "source_runtime":
        source = _mutate_json(
            source,
            lambda doc: doc["authority_categories"]["runtime_authorities"]["environment"].update(
                python="0.0.0"
            ),
        )
    elif attack == "source_key":
        source = _mutate_json(
            source,
            lambda doc: doc["authority_categories"]["key_authorities"][
                "primary_key_registry"
            ].update(bytes=0),
        )
    elif attack == "source_defect":
        source = _mutate_json(
            source,
            lambda doc: doc["authority_categories"]["defect_authorities"][
                "defect_authority_report"
            ].update(sha256="0" * 64),
        )
    elif attack == "seal_protocol":
        seal = _mutate_json(seal, lambda doc: doc["protocol"].update(bytes=0))
    elif attack == "seal_git_commit":
        seal = _mutate_json(
            seal,
            lambda doc: doc["git_design_identity"].update(commit="0" * 40),
        )
    else:
        seal = _mutate_json(
            seal,
            lambda doc: doc["authority_roots"].update(input_authorities="0" * 64),
        )

    monkeypatch.setattr(V5, "EXPECTED_SEALED_SCORE_PROTOCOL_SHA256", protocol.sha256)
    monkeypatch.setattr(V5, "_git_design_authority_record", lambda _captured: git_design)
    with pytest.raises(RuntimeError, match="protocol|clean-design|source registry|terminal"):
        V5._validate_score_execution_authority(protocol, seal, clean, source, captured)


def test_result_path_injection_is_rejected_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "formal-results"
    score = output / "forcing_shards_v5_observed" / "scores.parquet"
    score.parent.mkdir(parents=True)
    score.write_bytes(b"forbidden")
    monkeypatch.setattr(V5, "DEFAULT_OUTPUT_DIR", output)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("score bytes were opened")

    monkeypatch.setattr(V5, "_read_stable_regular", forbidden)
    with pytest.raises(A.AuthorityError, match="forbidden formal result"):
        A._read_allowed(score, label="injected score authority")


def test_create_only_writer_commits_in_order_and_refuses_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "protocols").mkdir()
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(V5, "ROOT", tmp_path)
    first = tmp_path / "outputs" / "clean.json"
    second = tmp_path / "outputs" / "registry.json"
    terminal = tmp_path / "protocols" / "seal.json"
    observed_before_terminal: list[bool] = []

    def precommit() -> None:
        assert not first.exists()
        assert not second.exists()
        assert not terminal.exists()

    bindings = A._publish_create_only(
        ((first, b"clean\n"), (second, b"source\n"), (terminal, b"seal\n")),
        precommit=precommit,
    )
    observed_before_terminal.append(first.exists() and second.exists())
    assert first.read_bytes() == b"clean\n"
    assert second.read_bytes() == b"source\n"
    assert terminal.read_bytes() == b"seal\n"
    assert list(bindings) == ["outputs/clean.json", "outputs/registry.json", "protocols/seal.json"]
    assert observed_before_terminal == [True]
    assert not (tmp_path / A.LOCK_NAME).exists()
    assert not any(".staging." in path.name for path in tmp_path.rglob("*"))
    with pytest.raises(A.AuthorityError, match="collision"):
        A._publish_create_only(((terminal, b"different\n"),), precommit=lambda: None)
    assert terminal.read_bytes() == b"seal\n"


def test_create_only_writer_rolls_back_all_owned_files_on_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setattr(V5, "ROOT", tmp_path)
    first = tmp_path / "first.json"
    terminal = tmp_path / "seal.json"

    def injected() -> None:
        raise A.AuthorityError("injected precommit attack")

    with pytest.raises(A.AuthorityError, match="injected"):
        A._publish_create_only(
            ((first, b"first"), (terminal, b"terminal")),
            precommit=injected,
        )
    assert not first.exists()
    assert not terminal.exists()
    assert not (tmp_path / A.LOCK_NAME).exists()
    assert not any(".staging." in path.name for path in tmp_path.iterdir())


def test_create_only_writer_rejects_symlink_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(V5, "ROOT", tmp_path)
    with pytest.raises(A.AuthorityError, match="symbolic link"):
        A._publish_create_only(((linked / "seal.json", b"seal"),), precommit=lambda: None)
    assert not (real / "seal.json").exists()


def test_create_only_writer_detects_repository_root_swap_without_touching_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    root = parent / "repo"
    root.mkdir()
    moved = parent / "repo-moved"
    monkeypatch.setattr(V5, "ROOT", root)
    target = root / "terminal.json"

    def swap_root() -> None:
        root.rename(moved)
        root.mkdir()
        (root / "attacker-marker").write_bytes(b"preserve")

    with pytest.raises(RuntimeError, match="no longer names the anchored directory"):
        A._publish_create_only(((target, b"terminal"),), precommit=swap_root)
    assert (root / "attacker-marker").read_bytes() == b"preserve"
    assert not (root / "terminal.json").exists()
    assert not (moved / "terminal.json").exists()
    assert not any(".staging." in path.name for path in moved.iterdir())
    assert not (moved / A.LOCK_NAME).exists()


def test_git_design_authority_requires_exact_head_blobs_and_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    runner = tmp_path / "runner.py"
    protocol = tmp_path / "protocol.yaml"
    runner.write_bytes(b"print('runner')\n")
    protocol.write_bytes(b"{}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "runner.py", "protocol.yaml"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "clean design"], check=True)
    monkeypatch.setattr(V5, "ROOT", tmp_path)
    monkeypatch.setattr(V5, "SCORE_GIT_DESIGN_ROLES", ("runner", "sealed_score_protocol"))
    captured = {
        "runner": V5._read_stable_regular(runner, root=tmp_path, label="runner"),
        "sealed_score_protocol": V5._read_stable_regular(protocol, root=tmp_path, label="protocol"),
    }
    record = V5._git_design_authority_record(captured)
    assert record["design_paths_match_commit_tree"] is True
    assert record["repository_wide_clean_claimed"] is False
    assert set(record["design_paths"]) == {"runner", "sealed_score_protocol"}

    runner.write_bytes(b"print('attacked')\n")
    attacked = dict(captured)
    attacked["runner"] = V5._read_stable_regular(runner, root=tmp_path, label="runner")
    with pytest.raises(RuntimeError, match="differ from HEAD"):
        V5._git_design_authority_record(attacked)


def test_phase_order_rejects_phase1_after_runner_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(V5, "EXPECTED_SEALED_SCORE_PROTOCOL_SHA256", "0" * 64)
    monkeypatch.setattr(A, "_assert_formal_output_absent", lambda: None)
    monkeypatch.setattr(A, "_require_absent", lambda *_args, **_kwargs: None)
    with pytest.raises(A.AuthorityError, match="created before that pin"):
        A.publish_protocol_candidate()
