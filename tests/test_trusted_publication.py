from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import thermoroute.opening as opening  # noqa: E402
import thermoroute.outcome_acquisition as outcome_acquisition  # noqa: E402
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


def _acquisition_state(root: Path) -> dict[str, Path]:
    run = root / "outputs" / "confirmatory" / "route_a_acquisition_fixture"
    run.mkdir(parents=True)
    acquisition = run / "acquisition"
    return {
        "run_directory": run,
        "acquisition_request_map": acquisition / "request_map.json",
        "temporal_outcomes": acquisition / "temporal.parquet",
        "external_outcomes": acquisition / "external.parquet",
        "acquisition_manifest": acquisition / "manifest.json",
    }


def _fill_acquisition_stage(state: dict[str, Path], stage: Path) -> None:
    staged = outcome_acquisition._acquisition_state_at_directory(state, stage)
    for key in (
        "acquisition_request_map",
        "temporal_outcomes",
        "external_outcomes",
        "acquisition_manifest",
    ):
        outcome_acquisition._create_bytes(
            staged[key], f"complete:{key}\n".encode()
        )


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


def test_acquisition_bundle_is_all_or_nothing_at_directory_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _acquisition_state(tmp_path)
    stage = outcome_acquisition._new_acquisition_stage_directory(state)
    _fill_acquisition_stage(state, stage)
    canonical = outcome_acquisition._acquisition_directory(state)

    def crash(point: str) -> None:
        if point == "before_directory_rename":
            raise RuntimeError("synthetic pre-rename crash")

    monkeypatch.setattr(
        outcome_acquisition, "_acquisition_publication_fault", crash
    )
    with pytest.raises(RuntimeError, match="pre-rename crash"):
        outcome_acquisition._publish_acquisition_directory(stage, state)
    assert stage.is_dir()
    assert not canonical.exists()

    outcome_acquisition._cleanup_abandoned_acquisition_stages(state)
    assert not stage.exists()
    monkeypatch.setattr(
        outcome_acquisition,
        "_acquisition_publication_fault",
        lambda _point: None,
    )
    replacement = outcome_acquisition._new_acquisition_stage_directory(state)
    _fill_acquisition_stage(state, replacement)
    assert outcome_acquisition._publish_acquisition_directory(
        replacement, state
    ) == canonical
    outcome_acquisition._assert_exact_acquisition_directory(canonical, state)


def test_sigkill_abandoned_acquisition_stage_is_safely_cleaned(
    tmp_path: Path,
) -> None:
    state = _acquisition_state(tmp_path)
    serialized = {key: str(value) for key, value in state.items()}
    program = "\n".join([
        "import json, os, signal, sys",
        "from pathlib import Path",
        "import thermoroute.outcome_acquisition as acquisition",
        "state = {key: Path(value) for key, value in json.loads(sys.argv[1]).items()}",
        "stage = acquisition._new_acquisition_stage_directory(state)",
        "acquisition._create_bytes(stage / 'partial.json', b'prefix')",
        "os.kill(os.getpid(), signal.SIGKILL)",
    ])
    child = subprocess.run(
        [sys.executable, "-c", program, json.dumps(serialized)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    assert child.returncode == -signal.SIGKILL
    stages = list(
        Path(state["run_directory"]).glob(".acquisition-stage-v1-*")
    )
    assert len(stages) == 1
    outcome_acquisition._validate_abandoned_acquisition_stages(state)
    outcome_acquisition._cleanup_abandoned_acquisition_stages(state)
    assert not list(
        Path(state["run_directory"]).glob(".acquisition-stage-v1-*")
    )


def test_sigkill_abandoned_trusted_stage_is_safely_cleaned(
    tmp_path: Path,
) -> None:
    state = _publication_state(tmp_path)
    serialized = {key: str(value) for key, value in state.items()}
    program = "\n".join([
        "import json, os, signal, sys",
        "from pathlib import Path",
        "import thermoroute.opening as opening",
        "state = {key: Path(value) for key, value in json.loads(sys.argv[1]).items()}",
        "stage = opening._new_trusted_stage_directory(state)",
        "opening._exclusive_create_bytes(stage / 'partial.json', b'prefix')",
        "os.kill(os.getpid(), signal.SIGKILL)",
    ])
    child = subprocess.run(
        [sys.executable, "-c", program, json.dumps(serialized)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    assert child.returncode == -signal.SIGKILL
    run = Path(state["run_directory"])
    assert len(list(run.glob(".trusted-stage-v1-*"))) == 1
    assert opening._handle_abandoned_trusted_stages(state, remove=False) == 1
    assert opening._handle_abandoned_trusted_stages(state, remove=True) == 1
    assert not list(run.glob(".trusted-stage-v1-*"))


def test_unsafe_abandoned_trusted_stage_fails_closed_without_deletion(
    tmp_path: Path,
) -> None:
    state = _publication_state(tmp_path)
    stage = opening._new_trusted_stage_directory(state)
    outside = tmp_path / "outside-linked-artifact"
    outside.write_bytes(b"must not be unlinked\n")
    outside.chmod(0o444)
    os.link(outside, stage / "linked-artifact")

    with pytest.raises(
        opening.OpeningContractError, match="external hard link"
    ):
        opening._handle_abandoned_trusted_stages(state, remove=True)
    assert stage.is_dir()
    assert outside.read_bytes() == b"must not be unlinked\n"


@pytest.mark.parametrize("kind", ["acquisition", "trusted"])
def test_abandoned_stage_with_writable_file_fails_closed(
    tmp_path: Path,
    kind: str,
) -> None:
    if kind == "acquisition":
        state = _acquisition_state(tmp_path)
        stage = outcome_acquisition._new_acquisition_stage_directory(state)
        cleanup = lambda: outcome_acquisition._cleanup_abandoned_acquisition_stages(  # noqa: E731
            state
        )
        error = outcome_acquisition.OutcomeAcquisitionError
    else:
        state = _publication_state(tmp_path)
        stage = opening._new_trusted_stage_directory(state)
        cleanup = lambda: opening._handle_abandoned_trusted_stages(  # noqa: E731
            state, remove=True
        )
        error = opening.OpeningContractError
    writable = stage / "unfinished.tmp"
    writable.write_bytes(b"not yet immutable\n")
    writable.chmod(0o600)

    with pytest.raises(error, match="unsafe entry"):
        cleanup()
    assert stage.is_dir()
    assert writable.read_bytes() == b"not yet immutable\n"


@pytest.mark.parametrize(
    "kill_point",
    [
        "before_directory_rename",
        "after_directory_rename_before_hardening",
        "after_directory_hardening_before_parent_fsync",
    ],
)
def test_acquisition_directory_publication_recovers_real_sigkill_windows(
    tmp_path: Path,
    kill_point: str,
) -> None:
    state = _acquisition_state(tmp_path)
    serialized = {key: str(value) for key, value in state.items()}
    program = "\n".join([
        "import json, os, signal, sys",
        "from pathlib import Path",
        "import thermoroute.outcome_acquisition as acquisition",
        "state = {key: Path(value) for key, value in json.loads(sys.argv[1]).items()}",
        "stage = acquisition._new_acquisition_stage_directory(state)",
        "staged = acquisition._acquisition_state_at_directory(state, stage)",
        "for key in ('acquisition_request_map', 'temporal_outcomes', 'external_outcomes', 'acquisition_manifest'):",
        "    acquisition._create_bytes(staged[key], ('complete:' + key + '\\n').encode())",
        f"POINT = {kill_point!r}",
        "def kill(point):",
        "    if point == POINT:",
        "        os.kill(os.getpid(), signal.SIGKILL)",
        "acquisition._acquisition_publication_fault = kill",
        "acquisition._publish_acquisition_directory(stage, state)",
    ])
    child = subprocess.run(
        [sys.executable, "-c", program, json.dumps(serialized)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    assert child.returncode == -signal.SIGKILL
    canonical = outcome_acquisition._acquisition_directory(state)
    if kill_point == "before_directory_rename":
        assert not canonical.exists()
        outcome_acquisition._validate_abandoned_acquisition_stages(state)
        outcome_acquisition._cleanup_abandoned_acquisition_stages(state)
        assert not list(
            Path(state["run_directory"]).glob(".acquisition-stage-v1-*")
        )
        return
    assert canonical.is_dir()
    if kill_point == "after_directory_rename_before_hardening":
        with pytest.raises(outcome_acquisition.OutcomeAcquisitionError):
            outcome_acquisition._assert_exact_acquisition_directory(
                canonical, state
            )
        outcome_acquisition._assert_exact_acquisition_directory(
            canonical, state, allow_recoverable_canonical_mode=True
        )
        outcome_acquisition._harden_recoverable_acquisition_directory(state)
    outcome_acquisition._assert_exact_acquisition_directory(canonical, state)
    assert canonical.stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize(
    "kill_point",
    [
        "before_trusted_directory_rename",
        "after_trusted_directory_rename_before_hardening",
        "after_trusted_directory_hardening_before_parent_fsync",
    ],
)
def test_trusted_directory_publication_recovers_real_sigkill_windows(
    tmp_path: Path,
    kill_point: str,
) -> None:
    state = _publication_state(tmp_path)
    serialized = {key: str(value) for key, value in state.items()}
    program = "\n".join([
        "import json, os, signal, sys",
        "from pathlib import Path",
        "import thermoroute.opening as opening",
        "state = {key: Path(value) for key, value in json.loads(sys.argv[1]).items()}",
        "stage = opening._new_trusted_stage_directory(state)",
        "staged = opening._trusted_state_at_directory(state, stage)",
        "for ordinal, key in enumerate(opening._TRUSTED_STATE_KEYS):",
        "    opening._exclusive_create_bytes(staged[key], f'{ordinal}:{key}\\n'.encode())",
        f"POINT = {kill_point!r}",
        "def kill(point):",
        "    if point == POINT:",
        "        os.kill(os.getpid(), signal.SIGKILL)",
        "opening._trusted_publication_fault = kill",
        "opening._atomic_publish_trusted_directory(stage, state)",
    ])
    child = subprocess.run(
        [sys.executable, "-c", program, json.dumps(serialized)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    assert child.returncode == -signal.SIGKILL
    canonical = opening._trusted_directory_from_state(state)
    if kill_point == "before_trusted_directory_rename":
        assert not canonical.exists()
        assert opening._handle_abandoned_trusted_stages(
            state, remove=False
        ) == 1
        opening._handle_abandoned_trusted_stages(state, remove=True)
        assert not list(
            Path(state["run_directory"]).glob(".trusted-stage-v1-*")
        )
        return
    assert canonical.is_dir()
    if kill_point == "after_trusted_directory_rename_before_hardening":
        with pytest.raises(opening.OpeningContractError):
            opening._assert_exact_trusted_directory(canonical, state)
        opening._assert_exact_trusted_directory(
            canonical, state, allow_recoverable_canonical_mode=True
        )
        opening._harden_recoverable_trusted_directory(state)
    opening._assert_exact_trusted_directory(canonical, state)
    assert canonical.stat().st_mode & 0o777 == 0o555


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


@pytest.mark.parametrize(
    "final_name",
    [
        "opening_intent_v1.json",
        "acquisition_work_order_v1.json",
        "opening_receipt_v1.json",
        "opening_receipt_v1.sha256",
    ],
)
def test_authoritative_atomic_finals_reject_unknown_hardlinks(
    tmp_path: Path,
    final_name: str,
) -> None:
    final = tmp_path / final_name
    payload = f"authoritative:{final_name}\n".encode()
    opening._atomic_create_bytes(final, payload)
    attacker_link = tmp_path / f"attacker-{final_name}"
    os.link(final, attacker_link)

    with pytest.raises(
        opening.OpeningContractError, match="unknown hard link"
    ):
        opening._validate_atomic_final_file(
            final, payload, cleanup_temps=False
        )
    attacker_link.unlink()
    opening._validate_atomic_final_file(
        final, payload, cleanup_temps=False
    )


def test_raw_atomic_final_rejects_unknown_hardlink(tmp_path: Path) -> None:
    final = tmp_path / "request_ledger_v1.json"
    payload = b'{"complete":true}\n'
    outcome_acquisition._create_bytes(final, payload)
    attacker_link = tmp_path / "attacker-ledger.json"
    os.link(final, attacker_link)

    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError,
        match="unknown hard link",
    ):
        outcome_acquisition._require_immutable_atomic_final(
            final, label="request ledger"
        )
    attacker_link.unlink()
    outcome_acquisition._require_immutable_atomic_final(
        final, label="request ledger"
    )


@pytest.mark.parametrize("producer", ["opening", "acquisition"])
def test_atomic_create_does_not_delete_noncanonical_temp_name(
    tmp_path: Path,
    producer: str,
) -> None:
    final = tmp_path / "opening_intent_v1.json"
    unrelated = tmp_path / f".{final.name}.user_evidence.tmp"
    unrelated.write_bytes(b"unrelated owner data\n")
    unrelated.chmod(0o444)

    if producer == "opening":
        opening._atomic_create_bytes(final, b"authoritative\n")
    else:
        outcome_acquisition._create_bytes(final, b"authoritative\n")
    assert unrelated.read_bytes() == b"unrelated owner data\n"


@pytest.mark.parametrize(
    ("kill_point", "final_exists"),
    [
        ("after_temporary_prefix_write", False),
        ("after_final_mode_before_inode_fsync", False),
        ("before_no_replace_link", False),
        ("after_no_replace_link", True),
    ],
)
def test_real_process_death_leaves_no_final_prefix_and_temp_is_recoverable(
    tmp_path: Path,
    kill_point: str,
    final_exists: bool,
) -> None:
    target = tmp_path / "opening_intent_v1.json"
    payload = b'{"complete":true}\n'
    program = "\n".join([
        "import os, signal, sys",
        "from pathlib import Path",
        "import thermoroute.opening as opening",
        f"POINT = {kill_point!r}",
        "def kill(point, _path):",
        "    if point == POINT:",
        "        os.kill(os.getpid(), signal.SIGKILL)",
        "opening._atomic_create_fault = kill",
        f"opening._atomic_create_bytes(Path(sys.argv[1]), {payload!r})",
    ])
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child = subprocess.run(
        [sys.executable, "-c", program, str(target)],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    assert child.returncode == -signal.SIGKILL
    assert target.exists() is final_exists
    temporary_files = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(temporary_files) == 1
    if kill_point == "after_final_mode_before_inode_fsync":
        assert temporary_files[0].stat().st_mode & 0o222 == 0
    if target.exists():
        assert target.read_bytes() == payload
        with pytest.raises(opening.OpeningAlreadyStarted):
            opening._atomic_create_bytes(target, payload)
    else:
        opening._atomic_create_bytes(target, payload)
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
    with pytest.raises(opening.OpeningAlreadyStarted):
        opening._atomic_create_bytes(target, b"replacement")
    assert target.read_bytes() == payload


@pytest.mark.parametrize(
    ("kill_point", "final_exists"),
    [
        ("after_temporary_prefix_write", False),
        ("after_final_mode_before_inode_fsync", False),
        ("before_no_replace_link", False),
        ("after_no_replace_link", True),
    ],
)
def test_raw_atomic_create_survives_real_process_death_and_cleans_temp(
    tmp_path: Path,
    kill_point: str,
    final_exists: bool,
) -> None:
    target = tmp_path / "request_ledger_v1.json"
    payload = b'{"complete":true}\n'
    program = "\n".join([
        "import os, signal, sys",
        "from pathlib import Path",
        "import thermoroute.outcome_acquisition as acquisition",
        f"POINT = {kill_point!r}",
        "def kill(point, _path):",
        "    if point == POINT:",
        "        os.kill(os.getpid(), signal.SIGKILL)",
        "acquisition._atomic_create_fault = kill",
        f"acquisition._create_bytes(Path(sys.argv[1]), {payload!r})",
    ])
    child = subprocess.run(
        [sys.executable, "-c", program, str(target)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    assert child.returncode == -signal.SIGKILL
    assert target.exists() is final_exists
    temporary_files = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(temporary_files) == 1
    if kill_point == "after_final_mode_before_inode_fsync":
        assert temporary_files[0].stat().st_mode & 0o222 == 0
    if final_exists:
        assert target.read_bytes() == payload
        with pytest.raises(outcome_acquisition.OutcomeAcquisitionError):
            outcome_acquisition._create_bytes(target, payload)
    else:
        outcome_acquisition._create_bytes(target, payload)
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


@pytest.mark.parametrize(
    "kill_point",
    [
        "after_parent_directory_create_before_temp",
        "after_temporary_prefix_write",
        "after_final_mode_before_inode_fsync",
        "before_no_replace_link",
    ],
)
def test_execute_recovers_real_preintent_atomic_remnant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kill_point: str,
) -> None:
    run = tmp_path / "run"
    state: dict[str, Any] = {
        "namespace": "fixture",
        "run_directory": run,
        "intent": run / "opening_intent_v1.json",
        "work_order": run / "acquisition_work_order_v1.json",
        "receipt": run / "opening_receipt_v1.json",
    }
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    authorization_state = {
        key: (
            value.relative_to(tmp_path).as_posix()
            if isinstance(value, Path)
            else value
        )
        for key, value in state.items()
    }
    preflight = {
        "state_paths": state,
        "authorization_sha256": "a" * 64,
        "authorization": {
            "opening_id": "fixture-opening",
            "state_paths": authorization_state,
        },
        "fixed_code": {"sha256": "fixed"},
        "runtime": {"runtime_sha256": "runtime"},
    }
    work_order = {"work_order_self_sha256": "work-order"}
    intent_stable = {
        "format": opening.INTENT_FORMAT,
        "status": "OPENING_STARTED_IRREVERSIBLE",
        "opening_id": "fixture-opening",
        "authorization_sha256": "a" * 64,
        "preflight_attestation_sha256": opening.sha256_json({"fixture": True}),
        "work_order_self_sha256": "work-order",
        "work_order_file_sha256": hashlib.sha256(
            canonical_json_bytes(work_order)
        ).hexdigest(),
        "fixed_code_sha256": "fixed",
        "runtime_sha256": "runtime",
        "trusted_validator": {"fixture": True},
        "started_at_utc": "2026-07-22T00:00:00+00:00",
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
    }
    intended = {
        **intent_stable,
        "intent_self_sha256": opening.sha256_json(intent_stable),
    }
    payload = canonical_json_bytes(intended)
    program = "\n".join([
        "import os, signal, sys",
        "from pathlib import Path",
        "import thermoroute.opening as opening",
        f"POINT = {kill_point!r}",
        "def kill(point, _path):",
        "    if point == POINT:",
        "        os.kill(os.getpid(), signal.SIGKILL)",
        "opening._atomic_create_fault = kill",
        f"opening._atomic_create_bytes(Path(sys.argv[1]), {payload!r})",
    ])
    child = subprocess.run(
        [sys.executable, "-c", program, str(state["intent"])],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    assert child.returncode == -signal.SIGKILL
    assert not Path(state["intent"]).exists()
    if kill_point == "after_parent_directory_create_before_temp":
        assert run.is_dir()
        assert not list(run.iterdir())
        inspection, document = opening._inspect_or_recover_preintent_temp(
            state=state,
            preflight=preflight,
            root=tmp_path,
            work_order=work_order,
            publish_or_remove=False,
        )
        assert inspection == "EMPTY_SAFE"
        assert document is None
    else:
        assert list(run.glob(".opening_intent_v1.json.*.tmp"))

    monkeypatch.setattr(
        opening, "validate_authorization", lambda *_args, **_kwargs: preflight
    )
    monkeypatch.setattr(opening, "_assert_isolated_role", lambda **_kwargs: None)
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
        opening, "_trusted_validator_identity", lambda _root: {"fixture": True}
    )
    if kill_point == "after_parent_directory_create_before_temp":
        status = opening.inspect_same_opening_transport_resume(
            authorization, root=tmp_path
        )
        assert status["resume_phase"] == (
            "PRE_INTENT_EMPTY_DIRECTORY_RECOVERY_ON_EXECUTE"
        )

    def stop_after_atomic_state(**_kwargs: Any) -> None:
        raise RuntimeError("fixture stops before any outcome access")

    monkeypatch.setattr(opening, "_run_fixed_isolated_child", stop_after_atomic_state)
    with pytest.raises(RuntimeError, match="before any outcome access"):
        opening.isolated_orchestrate_opening(
            authorization, root=tmp_path, resume=False
        )
    assert Path(state["intent"]).is_file()
    assert Path(state["work_order"]).is_file()
    assert not list(run.glob(".opening_intent_v1.json.*.tmp"))
    if kill_point in {
        "after_final_mode_before_inode_fsync",
        "before_no_replace_link",
    }:
        assert Path(state["intent"]).read_bytes() == payload


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


@pytest.mark.parametrize(
    "unsafe_kind", ["symlink", "extra", "hardlink", "writable"]
)
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
    elif unsafe_kind == "hardlink":
        outside = tmp_path / "hardlink-source.bin"
        outside.write_bytes(b"linked bytes")
        first.unlink()
        opening.os.link(outside, first)
    else:
        first.chmod(0o644)

    with pytest.raises(opening.OpeningContractError):
        opening._atomic_publish_trusted_directory(stage, state)


def test_completed_receipt_reader_rejects_extra_trusted_file_before_replay(
    tmp_path: Path,
) -> None:
    state = _publication_state(tmp_path)
    stage = _complete_stage(state)
    canonical = opening._atomic_publish_trusted_directory(stage, state)
    canonical.chmod(0o755)
    (canonical / "unregistered.bin").write_bytes(b"extra")
    canonical.chmod(0o555)
    run = Path(state["run_directory"])
    state.update({
        "intent": run / "opening_intent_v1.json",
        "work_order": run / "acquisition_work_order_v1.json",
        "receipt": run / "opening_receipt_v1.json",
        "receipt_sha256": run / "opening_receipt_v1.sha256",
    })
    relative_state = {
        key: (
            value.relative_to(tmp_path).as_posix()
            if isinstance(value, Path)
            else value
        )
        for key, value in state.items()
    }
    document = {
        "format": opening.AUTHORIZATION_FORMAT,
        "source": {"authorization_path": "authorization.json"},
        "state_paths": relative_state,
    }
    document["authorization_self_sha256"] = opening.sha256_json(document)
    authorization = tmp_path / "authorization.json"
    authorization.write_bytes(canonical_json_bytes(document))
    with pytest.raises(
        opening.OpeningContractError,
        match="trusted artifact directory is incomplete or has extra entries",
    ):
        opening._read_completed_receipt(
            authorization_path=authorization,
            root=tmp_path,
        )


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
    Path(state["work_order"]).chmod(0o444)
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
    monkeypatch.setattr(
        outcome_acquisition,
        "_assert_exact_acquisition_directory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        outcome_acquisition,
        "_acquisition_directory_mode",
        lambda _state: 0o555,
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
    assert not list(
        Path(state["run_directory"]).glob(".trusted-stage-v1-*")
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
    Path(state["intent"]).chmod(0o444)
    work_order = {"fixture": True}
    Path(state["work_order"]).write_bytes(canonical_json_bytes(work_order))
    Path(state["work_order"]).chmod(0o444)
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
        "inspect_same_opening_transport_resume",
        lambda *_args, **_kwargs: {
            "status": "OPENING_INCOMPLETE_TRUSTED_RECOMPUTE_VALIDATED",
            "resume_phase": "TRUSTED_RECOMPUTE_NETWORK_FREE",
        },
    )
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
