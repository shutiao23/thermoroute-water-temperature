from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import thermoroute.opening as opening  # noqa: E402
from thermoroute.provenance import canonical_json_bytes, sha256_file  # noqa: E402


def _publication_state(root: Path) -> dict[str, Path]:
    run = root / "outputs" / "confirmatory" / "route_a_fixture"
    run.mkdir(parents=True)
    trusted = run / "trusted"
    names = {
        "availability_registry": "availability_registry_v1.csv",
        "outcome_quality_audit": "outcome_quality_audit_v1.json",
        "outcome_qc_gate": "outcome_qc_gate_v1.json",
        "approved_target_sensitivity": "approved_target_sensitivity_v1.json",
        "spatial_sensitivity": "spatial_sensitivity_v1.json",
        "probabilistic_evaluation": "probabilistic_evaluation_v1.json",
        "temporal_predictions": "temporal_predictions_v1.parquet",
        "external_predictions": "external_predictions_v1.parquet",
        "statistics": "statistics_v1.json",
        "report": "report_v1.md",
    }
    return {
        "run_directory": run,
        **{key: trusted / name for key, name in names.items()},
    }


def _complete_stage(state: dict[str, Path]) -> Path:
    stage = opening._new_trusted_stage_directory(state)
    staged = opening._trusted_state_at_directory(state, stage)
    for ordinal, key in enumerate(opening._TRUSTED_STATE_KEYS):
        opening._exclusive_create_bytes(
            Path(staged[key]), f"{ordinal}:{key}\n".encode()
        )
    return stage


def test_trusted_directory_is_all_or_nothing_and_retryable_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _publication_state(tmp_path)
    stage = _complete_stage(state)
    canonical = opening._trusted_directory_from_state(state)
    real_rename = opening.os.rename

    def crash_before_rename(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic process interruption")

    monkeypatch.setattr(opening.os, "rename", crash_before_rename)
    with pytest.raises(opening.OpeningContractError, match="publication failed"):
        opening._atomic_publish_trusted_directory(stage, state)
    assert stage.is_dir()
    assert not canonical.exists()

    monkeypatch.setattr(opening.os, "rename", real_rename)
    assert opening._atomic_publish_trusted_directory(stage, state) == canonical
    assert canonical.is_dir()
    assert not stage.exists()
    opening._assert_exact_trusted_directory(canonical, state)


def test_atomic_receipt_bytes_never_expose_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "opening_receipt_v1.json"
    payload = b'{"complete":true}\n'
    real_link = opening.os.link

    def crash_before_publish(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic link interruption")

    monkeypatch.setattr(opening.os, "link", crash_before_publish)
    with pytest.raises(opening.OpeningContractError, match="directory traversal"):
        opening._atomic_create_bytes(receipt, payload)
    assert not receipt.exists()

    monkeypatch.setattr(opening.os, "link", real_link)
    opening._atomic_create_bytes(receipt, payload)
    assert receipt.read_bytes() == payload
    with pytest.raises(opening.OpeningAlreadyStarted):
        opening._atomic_create_bytes(receipt, b"replacement")
    assert receipt.read_bytes() == payload


@pytest.mark.parametrize("preexisting_nonempty", [False, True])
def test_existing_canonical_trusted_directory_is_never_replaced(
    tmp_path: Path, preexisting_nonempty: bool
) -> None:
    state = _publication_state(tmp_path)
    stage = _complete_stage(state)
    canonical = opening._trusted_directory_from_state(state)
    canonical.mkdir()
    sentinel = canonical / "do-not-replace"
    if preexisting_nonempty:
        sentinel.write_bytes(b"owner bytes")

    with pytest.raises(opening.OpeningAlreadyStarted, match="already exists"):
        opening._atomic_publish_trusted_directory(stage, state)
    assert canonical.is_dir()
    if preexisting_nonempty:
        assert sentinel.read_bytes() == b"owner bytes"
    else:
        assert not any(canonical.iterdir())
    assert stage.is_dir()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "extra", "hardlink"])
def test_trusted_stage_rejects_unsafe_or_nonexact_artifacts(
    tmp_path: Path, unsafe_kind: str
) -> None:
    state = _publication_state(tmp_path)
    stage = _complete_stage(state)
    staged = opening._trusted_state_at_directory(state, stage)
    first = Path(staged[opening._TRUSTED_STATE_KEYS[0]])
    if unsafe_kind == "symlink":
        first.unlink()
        first.symlink_to(Path(staged[opening._TRUSTED_STATE_KEYS[1]]).name)
    elif unsafe_kind == "extra":
        (stage / "unregistered.bin").write_bytes(b"extra")
    else:
        outside = tmp_path / "hardlink-source.bin"
        outside.write_bytes(b"linked bytes")
        first.unlink()
        opening.os.link(outside, first)

    with pytest.raises(opening.OpeningContractError):
        opening._atomic_publish_trusted_directory(stage, state)


def test_trusted_staging_rejects_cross_directory_or_traversal_layout(
    tmp_path: Path,
) -> None:
    state = _publication_state(tmp_path)
    broken = dict(state)
    broken["report"] = tmp_path / "outside" / "report_v1.md"
    with pytest.raises(opening.OpeningContractError, match="do not share"):
        opening._trusted_directory_from_state(broken)

    outside = tmp_path / "another-filesystem-in-principle" / "stage"
    with pytest.raises(opening.OpeningContractError, match="same-filesystem sibling"):
        opening._trusted_state_at_directory(state, outside)


def test_trusted_publication_lock_rejects_symlink_and_concurrent_process(
    tmp_path: Path,
) -> None:
    state = _publication_state(tmp_path)
    run = state["run_directory"]
    lock = run / opening._TRUSTED_PUBLICATION_LOCK
    target = run / "attacker-lock"
    target.write_bytes(b"")
    lock.symlink_to(target.name)
    with pytest.raises(opening.OpeningContractError, match="lock path is unsafe"):
        with opening._exclusive_trusted_publication_lock(state):
            pass
    lock.unlink()

    state_json = json.dumps({key: str(value) for key, value in state.items()})
    contender = "\n".join(
        [
            "import json, sys",
            f"sys.path.insert(0, {str(ROOT / 'src')!r})",
            "import thermoroute.opening as opening",
            "state = {key: __import__('pathlib').Path(value) "
            "for key, value in json.loads(sys.argv[1]).items()}",
            "try:",
            "    with opening._exclusive_trusted_publication_lock(state):",
            "        raise SystemExit(3)",
            "except opening.OpeningAlreadyStarted:",
            "    raise SystemExit(0)",
        ]
    )
    with opening._exclusive_trusted_publication_lock(state):
        result = subprocess.run(
            [sys.executable, "-c", contender, state_json],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr


def test_trusted_publication_rejects_group_or_world_writable_parent(
    tmp_path: Path,
) -> None:
    state = _publication_state(tmp_path)
    run = state["run_directory"]
    run.chmod(0o777)
    with pytest.raises(opening.OpeningContractError, match="owner-controlled"):
        with opening._exclusive_trusted_publication_lock(state):
            pass
    with pytest.raises(opening.OpeningContractError, match="owner-controlled"):
        opening._new_trusted_stage_directory(state)


def _stub_trusted_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, int]]:
    state: dict[str, Any] = _publication_state(tmp_path)
    root = tmp_path.resolve()
    run = Path(state["run_directory"])
    acquisition = run / "acquisition"
    acquisition.mkdir()
    state.update(
        {
            "namespace": "fixture",
            "intent": run / "opening_intent_v1.json",
            "work_order": run / "acquisition_work_order_v1.json",
            "acquisition_manifest": acquisition / "acquisition_manifest_v1.json",
            "receipt": run / "opening_receipt_v1.json",
            "receipt_sha256": run / "opening_receipt_v1.sha256",
        }
    )
    authorization = root / "authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    work_order = {"authorization_path": "authorization.json", "fixture": True}
    Path(state["work_order"]).write_bytes(canonical_json_bytes(work_order))
    Path(state["intent"]).write_text("{}\n", encoding="utf-8")
    Path(state["acquisition_manifest"]).write_bytes(
        canonical_json_bytes({"transport_summary": {"opening_count": 1}})
    )
    authorization_state = {
        key: (
            value.relative_to(root).as_posix()
            if isinstance(value, Path)
            else value
        )
        for key, value in state.items()
    }
    preflight = {
        "authorization": {
            "opening_id": "fixture-opening",
            "state_paths": authorization_state,
        },
        "authorization_sha256": "a" * 64,
        "state_paths": state,
        "runtime": {"runtime_sha256": "runtime"},
        "fixed_code": {"sha256": "fixed"},
    }
    calls = {"produce": 0, "validate": 0}

    monkeypatch.setattr(
        opening, "validate_authorization", lambda *_args, **_kwargs: preflight
    )
    monkeypatch.setattr(
        opening,
        "_assert_isolated_role",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        opening,
        "_expected_acquisition_work_order",
        lambda *_args, **_kwargs: work_order,
    )
    monkeypatch.setattr(
        opening,
        "_validated_intent",
        lambda **_kwargs: {"intent_self_sha256": "intent"},
    )
    monkeypatch.setattr(opening, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(opening, "assert_formal_numerical_policy", lambda: None)
    monkeypatch.setattr(
        opening,
        "environment_fingerprint",
        lambda: {"numerical_runtime_sha256": "runtime"},
    )
    monkeypatch.setattr(
        opening, "_preflight_attestation", lambda _preflight: {"fixture": True}
    )
    monkeypatch.setattr(
        opening,
        "_trusted_validator_identity",
        lambda _root: {"sha256": "validator"},
    )
    monkeypatch.setattr(
        opening,
        "_release_bindings",
        lambda **_kwargs: {"format": "fixture-release-bindings"},
    )

    def produce(**kwargs: Any) -> opening.OpeningProducts:
        calls["produce"] += 1
        output = kwargs["output_state_paths"]
        for ordinal, key in enumerate(opening._TRUSTED_STATE_KEYS):
            opening._exclusive_create_bytes(
                Path(output[key]), f"{ordinal}:{key}\n".encode()
            )
        return opening._opening_products_from_state(output)

    def validate(
        products: opening.OpeningProducts, **_kwargs: Any
    ) -> dict[str, Any]:
        calls["validate"] += 1
        artifacts = {
            key: {
                "path": Path(state[key]).relative_to(root).as_posix(),
                "sha256": sha256_file(Path(getattr(products, key))),
            }
            for key in opening._TRUSTED_STATE_KEYS
        }
        return {
            "artifacts": artifacts,
            "formal_tests": [],
            "trusted_prediction_hashes": {},
            "reported_models": {"temporal": [], "external": []},
            "all_required_models_reported": True,
        }

    monkeypatch.setattr(opening, "produce_trusted_opening_products", produce)
    monkeypatch.setattr(opening, "validate_opening_products", validate)

    def read_receipt(
        *, require_sidecar: bool = True, **_kwargs: Any
    ) -> dict[str, Any]:
        receipt_path = Path(state["receipt"])
        sidecar_path = Path(state["receipt_sha256"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if require_sidecar:
            assert sidecar_path.read_bytes() == opening._receipt_sidecar_bytes(
                receipt_path
            )
        return receipt

    monkeypatch.setattr(opening, "_read_completed_receipt", read_receipt)
    return Path(state["work_order"]), state, calls


@pytest.mark.parametrize(
    ("crash_point", "expected_produce_calls"),
    [
        ("after_stage_validation", 2),
        ("after_trusted_publish", 1),
        ("after_receipt_publish", 1),
    ],
)
def test_synthetic_crash_recovery_never_reacquires_or_replaces_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
    expected_produce_calls: int,
) -> None:
    work_order, state, calls = _stub_trusted_scorer(tmp_path, monkeypatch)
    acquisition_path = Path(state["acquisition_manifest"])
    acquisition_before = acquisition_path.read_bytes()
    crashed = {"done": False}

    def inject(point: str) -> None:
        if point == crash_point and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError(f"synthetic crash at {point}")

    monkeypatch.setattr(opening, "_trusted_publication_fault", inject)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        opening.isolated_score_and_receipt(work_order, root=tmp_path)

    receipt = opening.isolated_score_and_receipt(work_order, root=tmp_path)
    assert receipt["status"] == "OPENED_AND_SCORED_ONCE"
    assert calls["produce"] == expected_produce_calls
    assert acquisition_path.read_bytes() == acquisition_before
    assert Path(state["receipt"]).is_file()
    assert Path(state["receipt_sha256"]).read_bytes() == (
        opening._receipt_sidecar_bytes(Path(state["receipt"]))
    )
    opening._assert_exact_trusted_directory(
        opening._trusted_directory_from_state(state), state
    )


def test_resume_after_raw_completion_launches_only_trusted_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state: dict[str, Any] = _publication_state(tmp_path)
    run = Path(state["run_directory"])
    acquisition = run / "acquisition"
    acquisition.mkdir()
    state.update(
        {
            "namespace": "fixture",
            "intent": run / "opening_intent_v1.json",
            "work_order": run / "acquisition_work_order_v1.json",
            "acquisition_manifest": acquisition / "acquisition_manifest_v1.json",
            "temporal_outcomes": acquisition / "temporal_outcomes_v1.parquet",
            "external_outcomes": acquisition / "external_outcomes_v1.parquet",
            "receipt": run / "opening_receipt_v1.json",
            "receipt_sha256": run / "opening_receipt_v1.sha256",
        }
    )
    Path(state["intent"]).write_text("{}\n", encoding="utf-8")
    work_order = {"fixture": True}
    Path(state["work_order"]).write_bytes(canonical_json_bytes(work_order))
    Path(state["acquisition_manifest"]).write_text("{}\n", encoding="utf-8")
    Path(state["temporal_outcomes"]).write_bytes(b"immutable temporal labels")
    Path(state["external_outcomes"]).write_bytes(b"immutable external labels")
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    preflight = {
        "authorization": {
            "opening_id": "fixture-opening",
            "state_paths": {"namespace": "fixture"},
        },
        "state_paths": state,
    }
    roles: list[str] = []

    monkeypatch.setattr(
        opening, "validate_authorization", lambda *_args, **_kwargs: preflight
    )
    monkeypatch.setattr(
        opening, "_assert_isolated_role", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        opening, "_secure_canonical_state_paths", lambda *_args: state
    )
    monkeypatch.setattr(
        opening, "_preflight_attestation", lambda _preflight: {"fixture": True}
    )
    monkeypatch.setattr(
        opening,
        "_expected_acquisition_work_order",
        lambda *_args, **_kwargs: work_order,
    )
    monkeypatch.setattr(
        opening,
        "_trusted_validator_identity",
        lambda _root: {"fixture": True},
    )
    monkeypatch.setattr(opening, "_validated_intent", lambda **_kwargs: {})
    monkeypatch.setattr(
        opening,
        "_run_fixed_isolated_child",
        lambda **kwargs: roles.append(str(kwargs["role"])),
    )
    monkeypatch.setattr(opening, "_read_completed_receipt", lambda **_kwargs: {})

    opening.isolated_orchestrate_opening(
        authorization, root=tmp_path, resume=True
    )
    assert roles == ["trusted_scorer"]
