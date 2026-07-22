from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from thermoroute.inference_gate import (
    AMENDMENT_RELATIVE,
    AMENDMENT_SEAL_RELATIVE,
    BASE_PROTOCOL_RELATIVE,
    BASE_PROTOCOL_SEAL_RELATIVE,
    InferenceGateError,
    STATION_REGISTRY_RELATIVE,
    build_inference_gate_document,
    cluster_geometry,
    exclusive_create_json,
    _validate_inference_amendment_seal_git_lineage,
    validate_inference_amendment,
    validate_inference_gate_document,
)
from thermoroute.outcome_qc import POLICY_RELATIVE as OUTCOME_QC_POLICY_RELATIVE


ROOT = Path(__file__).resolve().parents[1]


def _copy_gate_inputs(tmp_path: Path) -> None:
    for relative in (
        BASE_PROTOCOL_RELATIVE,
        BASE_PROTOCOL_SEAL_RELATIVE,
        STATION_REGISTRY_RELATIVE,
        AMENDMENT_RELATIVE,
        OUTCOME_QC_POLICY_RELATIVE,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    source = tmp_path / "src" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")


def _git_commit(root: Path, message: str) -> str:
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
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _seal_lineage_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "base.txt").write_text("base\n", encoding="utf-8")
    _git_commit(root, "base")
    return root


def _write_seal(root: Path, payload: bytes) -> None:
    path = root / "protocols" / "route_a_inference_amendment_seal_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_production_registry_geometry_is_exact_and_row_order_invariant() -> None:
    with (ROOT / STATION_REGISTRY_RELATIVE).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    first = cluster_geometry(rows)
    second = cluster_geometry(list(reversed(rows)))

    assert first == second
    assert first["n_stations"] == 120
    assert first["n_clusters"] == 15
    assert first["cluster_sizes_sorted"] == [
        2, 2, 3, 3, 4, 5, 5, 7, 8, 8, 10, 10, 13, 14, 26,
    ]
    assert first["effective_cluster_count_inverse_herfindahl"] == pytest.approx(
        9.536423841059602
    )
    assert first["effective_cluster_fraction"] == pytest.approx(
        0.6357615894039734
    )
    assert first["largest_cluster_share"] == pytest.approx(26 / 120)


def test_production_gate_fails_closed_before_any_outcome() -> None:
    document = build_inference_gate_document(root=ROOT)

    assert document["contains_confirmation_outcomes"] is False
    assert document["post_2020_outcomes_requested_or_inspected"] is False
    assert document["network_used"] is False
    assert document["status"] == "FAIL_CLOSED_DESCRIPTIVE_ONLY"
    assert document["claim_eligible"] is False
    assert document["analysis_mode"] == "FIXED_COHORT_DESCRIPTIVE_ONLY"
    assert document["cluster_gate"] == {
        "pass": False,
        "failure_codes": [
            "SMALL_CLUSTER_COUNT_LT_30",
            "EFFECTIVE_CLUSTER_FRACTION_LT_0_75",
        ],
    }
    assert document["structural_assumption_gate"]["pass"] is False
    assert document["null_simulation_gate"]["pass"] is False
    assert document["null_simulation_gate"]["status"] == (
        "NOT_RUN_BLOCKED_BY_STRUCTURAL_OR_CLUSTER_GATE"
    )


def test_gate_builder_has_no_network_dependency(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    document = build_inference_gate_document(root=ROOT)
    assert document["network_used"] is False


def test_gate_rejects_every_nonallowlisted_live_input(tmp_path) -> None:
    _copy_gate_inputs(tmp_path)
    alternative = tmp_path / "protocols" / "alternative.json"
    alternative.write_bytes((tmp_path / BASE_PROTOCOL_RELATIVE).read_bytes())

    with pytest.raises(InferenceGateError, match="not allowlisted"):
        build_inference_gate_document(root=tmp_path, protocol_path=alternative)


def test_gate_artifact_rebuild_detects_tamper_and_source_staleness(tmp_path) -> None:
    _copy_gate_inputs(tmp_path)
    document = build_inference_gate_document(root=tmp_path)
    gate = tmp_path / "outputs" / "prelabel" / "route_a_inference_gate_v1.json"
    exclusive_create_json(gate, document)
    assert validate_inference_gate_document(gate, root=tmp_path) == document

    original = gate.read_bytes()
    gate.chmod(0o644)
    attacked = json.loads(original)
    attacked["claim_eligible"] = True
    gate.write_text(json.dumps(attacked), encoding="utf-8")
    with pytest.raises(InferenceGateError, match="stale or tampered"):
        validate_inference_gate_document(gate, root=tmp_path)

    gate.write_bytes(original)
    (tmp_path / "src" / "fixture.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(InferenceGateError, match="stale or tampered"):
        validate_inference_gate_document(gate, root=tmp_path)


def test_gate_replay_rejects_missing_protocol_and_registry_drift(tmp_path) -> None:
    _copy_gate_inputs(tmp_path)
    document = build_inference_gate_document(root=tmp_path)
    gate = tmp_path / "outputs" / "prelabel" / "route_a_inference_gate_v1.json"
    exclusive_create_json(gate, document)

    with pytest.raises(InferenceGateError, match="not allowlisted"):
        validate_inference_gate_document(gate.with_name("missing.json"), root=tmp_path)

    registry = tmp_path / STATION_REGISTRY_RELATIVE
    original_registry = registry.read_bytes()
    lines = original_registry.decode("utf-8").splitlines()
    registry.write_text(
        "\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8"
    )
    with pytest.raises(InferenceGateError, match="stale or tampered"):
        validate_inference_gate_document(gate, root=tmp_path)
    registry.write_bytes(original_registry)

    protocol = tmp_path / BASE_PROTOCOL_RELATIVE
    protocol.write_bytes(protocol.read_bytes() + b"\n")
    with pytest.raises(InferenceGateError, match="differ from the v1 seal"):
        validate_inference_gate_document(gate, root=tmp_path)

def test_amendment_keeps_all_five_objects_and_margins_byte_semantic() -> None:
    amendment = validate_inference_amendment(
        ROOT / AMENDMENT_RELATIVE, root=ROOT
    )
    protocol = json.loads((ROOT / BASE_PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    family = protocol["primary_inference_contract"]["confirmatory_family"]

    assert amendment["scientific_comparisons"]["objects"] == family
    assert amendment["scientific_comparisons"]["change_allowed"] is False
    assert amendment["decision_overlay"][
        "supported_claim_allowed_when_gate_fails"
    ] is False
    assert amendment["additional_preopen_gates"]["outcome_qc_policy"][
        "required"
    ] is True
    assert amendment["additional_preopen_gates"]["outcome_qc_policy"]["role"] == (
        "predeclared_nonfiltering_gross_plausibility_and_aggregate_sensitivity_"
        "directional_reporting_gate_not_complete_outcome_quality_certification"
    )
    recovery = amendment["trusted_scoring_recovery_contract"]
    assert recovery["maximum_logical_openings"] == 1
    assert recovery["maximum_frozen_request_ledgers_per_opening"] == 1
    assert recovery["second_logical_opening_allowed"] is False
    assert recovery["exactly_once_http_delivery_claimed"] is False
    assert recovery[
        "response_received_but_transaction_not_durable_may_be_requested_again"
    ] is True
    assert recovery["durable_canonical_response_replacement_allowed"] is False
    assert recovery["raw_transport_resume_after_acquisition_manifest_allowed"] is False
    assert recovery["raw_acquisition_child_after_acquisition_manifest_allowed"] is False
    assert recovery["partial_invalid_or_noncanonical_trusted_directory"] == (
        "FAIL_CLOSED_NO_REPLACEMENT"
    )
    assert recovery["external_sha256_sidecar_without_receipt"] == "FAIL_CLOSED"
    assert amendment["lineage_contract"]["base_v1_files_remain_immutable"] is True


def test_amendment_seal_lineage_accepts_one_strict_immutable_birth(
    tmp_path: Path,
) -> None:
    root = _seal_lineage_repository(tmp_path)
    amendment = root / "protocols" / "route_a_inference_amendment_v1.json"
    amendment.parent.mkdir(parents=True, exist_ok=True)
    amendment.write_text("{}\n", encoding="utf-8")
    final_prelabel_commit = _git_commit(root, "amendment")
    payload = b'{"seal":"canonical"}\n'
    _write_seal(root, payload)
    creation = _git_commit(root, "seal")

    assert _validate_inference_amendment_seal_git_lineage(
        root=root,
        final_prelabel_commit=final_prelabel_commit,
        tip="HEAD",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    ) == creation


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("same_commit", "existed at its amendment commit"),
        ("preexisting", "existed at its amendment commit"),
        ("add_delete_readd", "exactly one reachable Git creation"),
        ("post_create_modify", "deleted or changed after creation"),
    ),
)
def test_amendment_seal_lineage_rejects_adversarial_histories(
    tmp_path: Path, attack: str, error: str,
) -> None:
    root = _seal_lineage_repository(tmp_path)
    amendment = root / "protocols" / "route_a_inference_amendment_v1.json"
    amendment.parent.mkdir(parents=True, exist_ok=True)
    payload = b'{"seal":"canonical"}\n'

    if attack == "same_commit":
        amendment.write_text("{}\n", encoding="utf-8")
        _write_seal(root, payload)
        final_prelabel_commit = _git_commit(root, "amendment and seal")
    elif attack == "preexisting":
        _write_seal(root, payload)
        _git_commit(root, "premature seal")
        amendment.write_text("{}\n", encoding="utf-8")
        final_prelabel_commit = _git_commit(root, "later amendment")
    else:
        amendment.write_text("{}\n", encoding="utf-8")
        final_prelabel_commit = _git_commit(root, "amendment")
        _write_seal(root, payload)
        _git_commit(root, "seal")
        if attack == "add_delete_readd":
            (root / AMENDMENT_SEAL_RELATIVE).unlink()
            _git_commit(root, "delete seal")
            _write_seal(root, payload)
            _git_commit(root, "re-add seal")
        else:
            _write_seal(root, b'{"seal":"changed"}\n')
            _git_commit(root, "modify seal")
            _write_seal(root, payload)
            _git_commit(root, "restore seal")

    with pytest.raises(InferenceGateError, match=error):
        _validate_inference_amendment_seal_git_lineage(
            root=root,
            final_prelabel_commit=final_prelabel_commit,
            tip="HEAD",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
