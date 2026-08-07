from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

import pytest

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
from thermoroute.model_matrix_amendment import model_matrix_contract_id
from thermoroute.repro import RUN_SCHEMA_VERSION, RunIdentity, sha256_file, sha256_json
import thermoroute.stage09b_precompute as P


ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = ROOT / "scripts" / "09b_development_controls.py"


def _load_production_script():
    module_name = "thermoroute_test_stage09b_production"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


DC = _load_production_script()


class _Closure:
    def __init__(self, digest: str) -> None:
        self.binding_digest = digest
        self.inventory = {"fixture": digest}

    def assert_unchanged(self) -> None:
        return None


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _formal_config(closure: _Closure) -> dict[str, Any]:
    arms = declared_arms()
    config = {
        "stage": "09b_development_controls",
        "format": "thermoroute.development-controls.v2",
        "execution_role": "prelabel_relative_to_unopened_post_2020_confirmation",
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
        "time_split": {
            split_name: list(split_dates)
            for split_name, split_dates in C.SPLIT.as_dict().items()
        },
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "train_config": asdict(TRAIN_CONFIG),
        "arms": [DC.canonical_arm_descriptor(arm) for arm in arms],
        "expected_member_registry": [
            list(member) for member in expected_member_registry(arms)
        ],
        "parameter_counts": assert_parameter_budgets(arms, n_stations=120),
        "architecture_templates": {
            arm.arm_id: architecture_template(arm, n_stations=120) for arm in arms
        },
        "parameter_match_tolerance_fraction": 0.02,
        "architecture_candidates_per_arm": 1,
        "historical_tuning_budget_equalized": False,
        "development_predictor_bridge": {"path": "fixture", "sha256": "b" * 64},
        "formal_numerical_policy": {"fixture": "single-thread"},
        "eval_batch_size": 17,
        "input_closure_sha256": closure.binding_digest,
        "input_closure_file_count": len(closure.inventory),
    }
    return config


def _identity(root: Path, config: dict[str, Any], closure: _Closure) -> RunIdentity:
    panel = root / "data_usgs/panel_usgs_120v2.parquet"
    registry = root / "data_usgs/station_registry_v1.csv"
    panel.parent.mkdir(parents=True)
    panel.write_bytes(b"fixture development panel through 2020\n")
    registry.write_bytes(b"site_no\n00000001\n")
    parts = {
        "schema_version": RUN_SCHEMA_VERSION,
        "panel_sha256": sha256_file(panel),
        "registry_sha256": sha256_file(registry),
        "config_sha256": sha256_json(config),
        "source_sha256": "c" * 64,
        "runtime_sha256": sha256_json({"fixture": "runtime"}),
        "input_closure_sha256": closure.binding_digest,
    }
    return RunIdentity(run_id=sha256_json(parts)[:20], **parts)


def _gate(root: Path) -> P.Stage09bMatrixGate:
    amendment = json.loads(
        (ROOT / "protocols/route_a_model_matrix_amendment_v1.json").read_text(
            encoding="utf-8"
        )
    )
    amendment_path = root / "protocols/route_a_model_matrix_amendment_v1.json"
    seal_path = root / "protocols/route_a_model_matrix_amendment_seal_v1.json"
    _json_write(amendment_path, amendment)
    for relative in (
        "protocols/route_a_numerical_policy_v2.json",
        "protocols/route_a_numerical_policy_amendment_v2.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            (ROOT / relative).read_bytes()
        )
    _json_write(seal_path, {"fixture": "separate prelabel seal"})

    def binding(path: Path) -> dict[str, Any]:
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    contract_id = model_matrix_contract_id(
        amendment["stage09_architecture_control_matrix"],
        amendment["stage09b_development_control_matrix"],
    )
    model_binding = {
        "format": P.MATRIX_BINDING_FORMAT,
        "amendment": binding(amendment_path),
        "seal": binding(seal_path),
        "amendment_format": amendment["format"],
        "amendment_status": amendment["status"],
        "seal_format": "thermoroute.route-a-model-matrix-amendment-seal.v1",
        "seal_status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "amendment_id": amendment["amendment_id"],
        "amendment_document_commit": "d" * 40,
        "model_matrix_contract_id": contract_id,
    }
    return P.Stage09bMatrixGate(amendment, {"fixture": "seal"}, model_binding, contract_id)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path.resolve()
    closure = _Closure("a" * 64)
    config = _formal_config(closure)
    identity = _identity(root, config, closure)
    gate = _gate(root)
    run_dir = root / "outputs/runs/09b_development_controls" / identity.run_id
    _json_write(run_dir / "run.json", {
        "identity": identity.as_dict(),
        "resolved_config": config,
    })
    monkeypatch.setattr(P, "assert_formal_numerical_policy", lambda **_kwargs: {})
    monkeypatch.setattr(P, "resolve_development_input_closure", lambda _root: closure)
    monkeypatch.setattr(P, "numerical_runtime_contract", lambda: {"fixture": "runtime"})
    monkeypatch.setattr(P, "validate_stage09b_model_matrix_gate", lambda _root: gate)
    authorization, work_orders = P.freeze_stage09b_precompute_plan(
        root=root,
        run_directory=run_dir,
        identity=identity,
        resolved_config=config,
        matrix_gate=gate,
        publication_guard=lambda: None,
    )
    return {
        "root": root,
        "closure": closure,
        "config": config,
        "identity": identity,
        "gate": gate,
        "authorization": authorization,
        "work_orders": work_orders,
    }


def _publish_all(
    fixture: dict[str, Any],
    *,
    registry_attack_member: P.Stage09bMember | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    semantics: dict[tuple[str, int], dict[str, Any]] = {}
    common_registry = "e" * 64
    for work_order in fixture["work_orders"]:
        raw = json.loads(work_order.read_text(encoding="utf-8"))
        member = P.Stage09bMember(
            raw["member"]["arm_index"], raw["member"]["arm_id"], raw["member"]["seed"]
        )
        for label, relative in raw["artifacts"].items():
            path = fixture["root"] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{member.member_id}:{label}\n".encode())
        validated = P.validate_stage09b_member_work_order(
            root=fixture["root"],
            work_order=work_order,
            expected_identity=fixture["identity"],
            expected_config=fixture["config"],
        )
        prediction = validated.artifact_paths["prediction"]
        content = sha256_file(prediction)
        semantic = {
            "prediction_content_sha256": content,
            "forecast_key_truth_sha256": (
                "f" * 64 if member == registry_attack_member else common_registry
            ),
            "prediction_rows": 27,
            "checkpoint_exact_replay_sha256": content,
        }
        P.publish_stage09b_member_receipt(
            validated,
            semantic_evidence=semantic,
            exact_replay_verified=True,
            publication_guard=lambda: None,
        )
        semantics[(member.arm_id, member.seed)] = semantic
    return semantics


def test_exact_registry_and_execution_only_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    members = P.expected_stage09b_precompute_members()
    assert len(members) == 45
    assert len({(member.arm_id, member.seed) for member in members}) == 45
    before = fixture["identity"].as_dict()
    observed: list[Path] = []
    lock = threading.Lock()

    def launch(path: Path) -> None:
        with lock:
            observed.append(path)

    P.execute_stage09b_member_work_orders(
        tuple(reversed(fixture["work_orders"])), workers=4, launch=launch
    )
    assert set(observed) == set(fixture["work_orders"])
    assert fixture["identity"].as_dict() == before
    with pytest.raises(P.Stage09bPrecomputeError, match="incomplete"):
        P.execute_stage09b_member_work_orders(
            (*fixture["work_orders"][:-1], fixture["work_orders"][0]),
            workers=2,
            launch=launch,
        )


def test_entrypoint_locks_full_member_transaction_and_excludes_parallelism_from_identity() -> None:
    tree = ast.parse(
        (ROOT / "scripts/09b_development_controls.py").read_text(encoding="utf-8")
    )
    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_authorized_member"
    )
    transaction = next(node for node in execute.body if isinstance(node, ast.With))
    context = transaction.items[0].context_expr
    assert isinstance(context, ast.Call)
    assert isinstance(context.func, ast.Name)
    assert context.func.id == "stage09b_member_execution_lock"
    calls = {
        node.func.id
        for node in ast.walk(transaction)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "train_arm_group",
        "replay_best_model_state_prediction",
        "publish_stage09b_member_receipt",
    }.issubset(calls)
    config_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "run_config" for target in node.targets)
    )
    assert isinstance(config_assignment.value, ast.Dict)
    config_keys = {
        key.value
        for key in config_assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "precompute_workers" not in config_keys
    assert "member_work_order" not in config_keys


def test_stubbed_45_member_matrix_freezes_and_replays_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    semantics = _publish_all(fixture)

    def inspect(member: P.Stage09bMember, _path: Path) -> dict[str, Any]:
        return semantics[(member.arm_id, member.seed)]

    resolved, coordinator = P.finalize_stage09b_precompute(
        root=fixture["root"],
        authorization_path=fixture["authorization"],
        semantic_inspector=inspect,
        publication_guard=lambda: None,
    )
    assert len(resolved) == 45
    receipt = json.loads(coordinator.read_text(encoding="utf-8"))
    assert receipt["matrix_audit"]["observed_members"] == 45
    assert receipt["matrix_audit"]["missing_members"] == 0
    assert receipt["matrix_audit"]["duplicate_members"] == 0
    assert receipt["matrix_audit"]["extra_members"] == 0
    assert receipt["matrix_audit"]["same_seed_paired_comparison_count"] == 8
    assert receipt["status"] == "PASS_INTERMEDIATE_45_MEMBER_SCHEDULING_CLOSURE"
    assert (
        receipt["evidence_role"]
        == "INTERMEDIATE_SCHEDULING_ONLY_NOT_FINAL_SCIENTIFIC_AUTHORITY"
    )
    assert receipt["final_scientific_authority"] is False
    assert (
        receipt["required_final_authority"]
        == "outputs/models/route_a_stage09b_completion.json"
    )
    assert (
        receipt["information_fairness"][
            "full_training_trajectory_replay_verified_here"
        ]
        is False
    )
    assert (
        receipt["scientific_equivalence"][
            "independent_checkpoint_replay_deferred_to_final_authority"
        ]
        is True
    )
    assert receipt["scientific_equivalence"]["worker_count_or_schedule_in_run_identity"] is False
    first_member_receipt = json.loads(
        (
            fixture["authorization"].parent
            / P.expected_stage09b_precompute_members()[0].receipt_relative
        ).read_text(encoding="utf-8")
    )
    assert (
        first_member_receipt["status"]
        == "COMMITTED_INTERMEDIATE_AFTER_PRODUCER_CHECKPOINT_REPLAY"
    )
    assert first_member_receipt["final_scientific_authority"] is False
    assert (
        first_member_receipt["required_final_authority"]
        == "outputs/models/route_a_stage09b_completion.json"
    )
    # Exact replay is idempotent; it never replaces the coordinator bytes.
    original = coordinator.read_bytes()
    _resolved, replayed = P.finalize_stage09b_precompute(
        root=fixture["root"],
        authorization_path=fixture["authorization"],
        semantic_inspector=inspect,
        publication_guard=lambda: None,
    )
    assert replayed.read_bytes() == original


def test_member_reuse_requires_fresh_exact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    work_order = fixture["work_orders"][0]
    raw = json.loads(work_order.read_text(encoding="utf-8"))
    for label, relative in raw["artifacts"].items():
        path = fixture["root"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(label.encode())
    validated = P.validate_stage09b_member_work_order(
        root=fixture["root"], work_order=work_order
    )
    content = sha256_file(validated.artifact_paths["prediction"])
    semantic = {
        "prediction_content_sha256": content,
        "forecast_key_truth_sha256": "e" * 64,
        "prediction_rows": 1,
        "checkpoint_exact_replay_sha256": content,
    }
    with pytest.raises(P.Stage09bPrecomputeError, match="fresh exact replay"):
        P.publish_stage09b_member_receipt(
            validated,
            semantic_evidence=semantic,
            exact_replay_verified=False,
            publication_guard=lambda: None,
        )


def test_same_member_direct_invocations_are_os_lock_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    validated = P.validate_stage09b_member_work_order(
        root=fixture["root"], work_order=fixture["work_orders"][0]
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []

    def first() -> None:
        try:
            with P.stage09b_member_execution_lock(validated):
                first_entered.set()
                if not release_first.wait(5):
                    raise AssertionError("test did not release first member lock")
        except BaseException as exc:
            errors.append(exc)

    def second() -> None:
        try:
            if not first_entered.wait(5):
                raise AssertionError("first member never acquired its lock")
            with P.stage09b_member_execution_lock(validated):
                second_entered.set()
        except BaseException as exc:
            errors.append(exc)

    thread_one = threading.Thread(target=first)
    thread_two = threading.Thread(target=second)
    thread_one.start()
    assert first_entered.wait(5)
    thread_two.start()
    assert not second_entered.wait(0.1)
    release_first.set()
    thread_one.join(5)
    thread_two.join(5)
    assert second_entered.is_set()
    assert not thread_one.is_alive() and not thread_two.is_alive()
    assert errors == []


def test_member_lock_rejects_symlinked_intermediate_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    validated = P.validate_stage09b_member_work_order(
        root=fixture["root"], work_order=fixture["work_orders"][0]
    )
    lock_parent = (
        validated.run_directory
        / P.MEMBER_LOCK_DIRECTORY
        / f"arm{validated.member.arm_index:02d}"
    )
    redirected = fixture["root"] / "attacker-lock-namespace"
    redirected.mkdir()
    lock_parent.rmdir()
    lock_parent.symlink_to(redirected, target_is_directory=True)

    with pytest.raises(P.Stage09bPrecomputeError, match="linked|unsafe"):
        with P.stage09b_member_execution_lock(validated):
            pytest.fail("a symlinked lock parent must never be acquired")

    assert list(redirected.iterdir()) == []


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b'{"format":"one","format":"two"}\n', "duplicate JSON key"),
        (b'{"value":NaN}\n', "non-finite JSON constant"),
        (b'{\n  "value": 1\n}\n', "not canonical"),
    ),
)
def test_control_documents_reject_noncanonical_json_bytes(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "control.json"
    path.write_bytes(payload)
    with pytest.raises(P.Stage09bPrecomputeError, match=message):
        P._load_json(path, label="control document", canonical=True)


def test_create_only_recovers_crash_after_link_before_unlink(tmp_path: Path) -> None:
    final = tmp_path / "receipt.json"
    payload = b'{"receipt":"exact"}\n'

    def fail_after_link(stage: str, temporary: Path, observed: Path) -> None:
        assert stage == "after_link_before_temp_unlink"
        assert observed == final
        assert temporary.parent == final.parent
        raise RuntimeError("simulated process death")

    with pytest.raises(RuntimeError, match="simulated process death"):
        P._publish_create_only(final, payload, _fault_injector=fail_after_link)

    candidates = P._create_only_temp_candidates(final)
    assert len(candidates) == 1
    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 2
    assert (final.stat().st_dev, final.stat().st_ino) == (
        candidates[0].stat().st_dev,
        candidates[0].stat().st_ino,
    )

    P._publish_create_only(final, payload)
    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1
    assert P._create_only_temp_candidates(final) == []


def test_create_only_rejects_external_hardlink(tmp_path: Path) -> None:
    final = tmp_path / "receipt.json"
    external = tmp_path / "external-hardlink"
    payload = b'{"receipt":"exact"}\n'
    P._publish_create_only(final, payload)
    os.link(final, external)

    with pytest.raises(
        P.Stage09bPrecomputeError,
        match="lacks one exact internal stale temporary",
    ):
        P._publish_create_only(final, payload)

    assert final.stat().st_nlink == 2
    assert external.read_bytes() == payload
    assert P._create_only_temp_candidates(final) == []


def test_create_only_cleans_owner_private_orphan_before_publication(
    tmp_path: Path,
) -> None:
    final = tmp_path / "receipt.json"
    payload = b'{"receipt":"exact"}\n'
    orphan = P._create_only_temp_path(final, "a" * 64)
    orphan.write_bytes(b"partial payload from interrupted write")
    orphan.chmod(0o600)
    assert orphan.stat().st_nlink == 1
    assert not final.exists()

    P._publish_create_only(final, payload)

    assert not orphan.exists()
    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1
    assert P._create_only_temp_candidates(final) == []


def test_large_create_only_artifact_recovers_link_unlink_crash(
    tmp_path: Path,
) -> None:
    final = tmp_path / "artifacts" / "combined_predictions.parquet"
    final.parent.mkdir()
    payload = (b"0123456789abcdef" * 131_073) + b"parquet-tail"
    descriptor, temporary = P.allocate_create_only_artifact_temp(final)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

    def fail_after_link(stage: str, staged: Path, observed: Path) -> None:
        assert stage == "after_link_before_temp_unlink"
        assert staged == temporary
        assert observed == final
        raise RuntimeError("simulated large-file process death")

    with pytest.raises(RuntimeError, match="simulated large-file process death"):
        P.publish_create_only_artifact_from_temp(
            temporary,
            final,
            publication_guard=lambda: None,
            _fault_injector=fail_after_link,
        )

    assert temporary.exists()
    assert final.stat().st_nlink == 2
    assert P.recover_create_only_artifact_state(final) is True
    assert final.stat().st_nlink == 1
    assert not temporary.exists()
    assert final.stat().st_size == len(payload)
    assert sha256_file(final) == hashlib.sha256(payload).hexdigest()


def test_large_create_only_artifact_rejects_external_hardlink(
    tmp_path: Path,
) -> None:
    final = tmp_path / "artifacts" / "combined_predictions.parquet"
    final.parent.mkdir()
    descriptor, temporary = P.allocate_create_only_artifact_temp(final)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(b"immutable large artifact")
        handle.flush()
        os.fsync(handle.fileno())
    P.publish_create_only_artifact_from_temp(
        temporary,
        final,
        publication_guard=lambda: None,
    )
    external = tmp_path / "attacker-hardlink"
    os.link(final, external)

    with pytest.raises(P.Stage09bPrecomputeError, match="hard-linked|external|excessive"):
        P.recover_create_only_artifact_state(final)

    assert final.stat().st_nlink == 2
    assert external.read_bytes() == b"immutable large artifact"


def test_large_create_only_artifact_cleans_unlinked_partial_temp(
    tmp_path: Path,
) -> None:
    final = tmp_path / "artifacts" / "combined_predictions.parquet"
    final.parent.mkdir()
    descriptor, temporary = P.allocate_create_only_artifact_temp(final)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(b"partial interrupted stream")
        handle.flush()
        os.fsync(handle.fileno())

    assert P.recover_create_only_artifact_state(final) is False
    assert not final.exists()
    assert not temporary.exists()


def test_exact_namespace_rejects_extra_empty_nested_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    precompute = fixture["authorization"].parent
    (precompute / "work_orders" / "arm99").mkdir()

    with pytest.raises(P.Stage09bPrecomputeError, match="directory closure"):
        P._assert_exact_namespace(precompute, coordinator_optional=True)


def test_pending_member_orphan_does_not_poison_parent_namespace_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    precompute = fixture["authorization"].parent
    member = P.expected_stage09b_precompute_members()[0]
    final = precompute / member.receipt_relative
    orphan = P._create_only_temp_path(final, "a" * 64)
    orphan.write_bytes(b"partial payload from interrupted member publication")
    orphan.chmod(0o600)

    P._assert_exact_namespace(precompute, coordinator_optional=True)
    assert orphan.exists()

    payload = b'{"receipt":"replacement"}\n'
    P._publish_create_only(final, payload)
    assert not orphan.exists()
    assert final.read_bytes() == payload


@pytest.mark.parametrize("attack", ("missing_receipt", "extra_receipt", "artifact_tamper"))
def test_partial_extra_and_tampered_member_state_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    semantics = _publish_all(fixture)
    precompute = fixture["authorization"].parent
    if attack == "missing_receipt":
        (precompute / P.expected_stage09b_precompute_members()[0].receipt_relative).unlink()
    elif attack == "extra_receipt":
        _json_write(precompute / "member_receipts/arm99/seed0.json", {"attack": True})
    else:
        first = P.expected_stage09b_precompute_members()[0]
        prediction = (
            fixture["root"] / "outputs/runs/09b_development_controls"
            / fixture["identity"].run_id / "arm_predictions" / first.arm_id
            / f"seed{first.seed}.parquet"
        )
        prediction.write_bytes(b"tampered\n")

    def inspect(member: P.Stage09bMember, _path: Path) -> dict[str, Any]:
        return semantics[(member.arm_id, member.seed)]

    with pytest.raises(P.Stage09bPrecomputeError):
        P.finalize_stage09b_precompute(
            root=fixture["root"],
            authorization_path=fixture["authorization"],
            semantic_inspector=inspect,
            publication_guard=lambda: None,
        )


def test_different_forecast_key_or_truth_registry_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    attacked = P.expected_stage09b_precompute_members()[-1]
    semantics = _publish_all(fixture, registry_attack_member=attacked)

    def inspect(member: P.Stage09bMember, _path: Path) -> dict[str, Any]:
        return semantics[(member.arm_id, member.seed)]

    with pytest.raises(P.Stage09bPrecomputeError, match="forecast keys and y_true"):
        P.finalize_stage09b_precompute(
            root=fixture["root"],
            authorization_path=fixture["authorization"],
            semantic_inspector=inspect,
            publication_guard=lambda: None,
        )


def test_plain_control_information_fairness_attack_is_rejected() -> None:
    amendment = json.loads(
        (ROOT / "protocols/route_a_model_matrix_amendment_v1.json").read_text(
            encoding="utf-8"
        )
    )
    amendment["stage09b_development_control_matrix"][
        "plain_control_information_fairness_revision"
    ]["explicitly_forbidden_batch_keys"] = ["target_date"]
    with pytest.raises(P.Stage09bPrecomputeError, match="model-matrix contract"):
        P._validate_matrix_document(amendment)


def test_work_order_collision_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    attacked = fixture["work_orders"][0]
    attacked.write_text('{"coordinated":"tamper"}\n', encoding="utf-8")
    with pytest.raises(P.Stage09bPrecomputeError, match="byte collision"):
        P.freeze_stage09b_precompute_plan(
            root=fixture["root"],
            run_directory=attacked.parents[3],
            identity=fixture["identity"],
            resolved_config=fixture["config"],
            matrix_gate=fixture["gate"],
            publication_guard=lambda: None,
        )


def test_live_config_matches_canonical_json_round_trip_without_fixture_repair() -> None:
    arms = declared_arms()
    closure = _Closure("a" * 64)
    config = _formal_config(closure)
    assert config == json.loads(json.dumps(config))
    for arm in config["arms"]:
        assert isinstance(arm["variables"], list)
        assert isinstance(arm["seeds"], list)
    assert all(
        isinstance(dates, list)
        for dates in config["time_split"].values()
    )


def test_parent_freeze_and_worker_equality_pass_on_live_list_shaped_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    for work_order in fixture["work_orders"]:
        raw = json.loads(work_order.read_text(encoding="utf-8"))
        member = P.Stage09bMember(
            raw["member"]["arm_index"], raw["member"]["arm_id"], raw["member"]["seed"]
        )
        validated = P.validate_stage09b_member_work_order(
            root=fixture["root"],
            work_order=work_order,
            expected_identity=fixture["identity"],
            expected_config=fixture["config"],
        )
        assert validated.member == member
    assert len(fixture["work_orders"]) == 45


def test_tuple_shaped_arm_descriptor_fails_closed() -> None:
    arms = declared_arms()
    closure = _Closure("a" * 64)
    config = _formal_config(closure)
    config["arms"] = [asdict(arm) for arm in arms]
    with pytest.raises(P.Stage09bPrecomputeError, match="arm/seed contract"):
        P._validate_formal_config(config)


def test_tuple_shaped_descriptor_rejected_by_plan_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    closure = _Closure("a" * 64)
    config = _formal_config(closure)
    arms = declared_arms()
    config["arms"] = [asdict(arm) for arm in arms]
    identity = _identity(root, config, closure)
    gate = _gate(root)
    run_dir = root / "outputs/runs/09b_development_controls" / identity.run_id
    _json_write(run_dir / "run.json", {
        "identity": identity.as_dict(),
        "resolved_config": config,
    })
    monkeypatch.setattr(P, "assert_formal_numerical_policy", lambda **_kwargs: {})
    monkeypatch.setattr(P, "resolve_development_input_closure", lambda _root: closure)
    monkeypatch.setattr(P, "numerical_runtime_contract", lambda: {"fixture": "runtime"})
    monkeypatch.setattr(P, "validate_stage09b_model_matrix_gate", lambda _root: gate)
    with pytest.raises(P.Stage09bPrecomputeError, match="arm/seed contract"):
        P.freeze_stage09b_precompute_plan(
            root=root,
            run_directory=run_dir,
            identity=identity,
            resolved_config=config,
            matrix_gate=gate,
            publication_guard=lambda: None,
        )
