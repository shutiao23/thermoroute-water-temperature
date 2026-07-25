from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import repro  # noqa: E402
from thermoroute.repro import (  # noqa: E402
    RUN_LOCK_SCHEMA_VERSION,
    RunIdentity,
    RunDirectoryLockError,
    advisory_file_lock,
    acquire_run_directory_lock,
    initialise_run_directory,
    release_run_directory_lock,
    run_directory_lock_path,
)


_HELPER = f"""
import sys
import time
sys.path.insert(0, {str(ROOT / 'src')!r})
from thermoroute.repro import RunDirectoryLockError, acquire_run_directory_lock

run_directory, run_id, mode = sys.argv[1:]
try:
    lock = acquire_run_directory_lock(run_directory, run_id=run_id)
except RunDirectoryLockError as exc:
    print(f"REJECTED:{{exc}}", flush=True)
    raise SystemExit(23)
print("LOCKED", flush=True)
if mode == "hold":
    time.sleep(120)
lock.release()
"""

_ADVISORY_WRITER = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
from thermoroute.repro import advisory_file_lock
with advisory_file_lock(sys.argv[1], exclusive=True):
    print("WRITER_LOCKED", flush=True)
"""


def _identity(run_id: str) -> RunIdentity:
    digest = "a" * 64
    return RunIdentity(
        run_id=run_id,
        panel_sha256=digest,
        registry_sha256=digest,
        config_sha256=digest,
        source_sha256=digest,
        runtime_sha256=digest,
    )


def _attempt(run_directory: Path, run_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", _HELPER, str(run_directory), run_id, "attempt"],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


def _holder(run_directory: Path, run_id: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-I", "-c", _HELPER, str(run_directory), run_id, "hold"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_same_process_reacquisition_is_idempotent_and_release_is_explicit(tmp_path):
    run_directory = tmp_path / "runs" / "same-run"
    first = acquire_run_directory_lock(run_directory, run_id="same-run")
    second = acquire_run_directory_lock(run_directory, run_id="same-run")
    assert second is first
    assert first.held
    assert release_run_directory_lock(run_directory)
    assert not first.held
    assert not release_run_directory_lock(run_directory)

    replacement = acquire_run_directory_lock(run_directory, run_id="same-run")
    assert replacement is not first
    replacement.release()


def test_initialise_locks_before_run_metadata_and_contender_fails_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(repro, "environment_fingerprint", lambda: {"fixture": True})
    monkeypatch.setattr(
        repro,
        "git_state",
        lambda _root: {"available": False, "commit": None, "dirty": None},
    )
    identity = _identity("initialised-run")
    run_directory = initialise_run_directory(
        tmp_path / "runs", identity, {"stage": "fixture"}
    )
    try:
        assert (run_directory / "run.json").is_file()
        result = _attempt(run_directory, identity.run_id)
        assert result.returncode == 23
        assert "already locked by another process" in result.stdout
        assert f'"pid":{os.getpid()}' in result.stdout

        owner = json.loads(run_directory_lock_path(run_directory).read_text())
        assert owner == {
            "format": RUN_LOCK_SCHEMA_VERSION,
            "state": "held",
            "run_id": identity.run_id,
            "pid": os.getpid(),
            "host": owner["host"],
            "started_utc": owner["started_utc"],
        }
        assert owner["host"]
        assert owner["started_utc"].endswith("+00:00")
    finally:
        release_run_directory_lock(run_directory)


def test_initialise_guard_failure_after_staging_releases_lock_and_allows_retry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(repro, "environment_fingerprint", lambda: {"fixture": True})
    monkeypatch.setattr(
        repro,
        "git_state",
        lambda _root: {"available": False, "commit": None, "dirty": None},
    )
    identity = _identity("guarded-new-run")
    runs = tmp_path / "runs"
    run_directory = runs / identity.run_id
    metadata_path = run_directory / "run.json"
    staged_payloads: list[dict[str, object]] = []

    def reject_publication() -> None:
        staged = list(run_directory.glob(f".{metadata_path.name}.*.tmp"))
        assert len(staged) == 1
        staged_payloads.append(json.loads(staged[0].read_text(encoding="utf-8")))
        raise RuntimeError("injected live-policy drift")

    with pytest.raises(RuntimeError, match="live-policy drift"):
        initialise_run_directory(
            runs,
            identity,
            {"stage": "fixture"},
            publication_guard=reject_publication,
        )

    assert staged_payloads[0]["identity"] == identity.as_dict()
    assert not metadata_path.exists()
    assert not list(run_directory.glob(f".{metadata_path.name}.*.tmp"))
    released = json.loads(
        run_directory_lock_path(run_directory).read_text(encoding="utf-8")
    )
    assert released["state"] == "released"

    retried = initialise_run_directory(
        runs,
        identity,
        {"stage": "fixture"},
        publication_guard=lambda: None,
    )
    try:
        assert retried == run_directory
        assert metadata_path.is_file()
    finally:
        release_run_directory_lock(retried)


def test_initialise_existing_manifest_acceptance_is_guarded_and_byte_preserving(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(repro, "environment_fingerprint", lambda: {"fixture": True})
    monkeypatch.setattr(
        repro,
        "git_state",
        lambda _root: {"available": False, "commit": None, "dirty": None},
    )
    identity = _identity("guarded-existing-run")
    runs = tmp_path / "runs"
    run_directory = initialise_run_directory(
        runs,
        identity,
        {"stage": "fixture"},
        publication_guard=lambda: None,
    )
    release_run_directory_lock(run_directory)
    metadata_path = run_directory / "run.json"
    original_bytes = metadata_path.read_bytes()
    observed_bytes: list[bytes] = []

    def reject_acceptance() -> None:
        observed_bytes.append(metadata_path.read_bytes())
        assert not list(run_directory.glob(f".{metadata_path.name}.*.tmp"))
        raise RuntimeError("injected cache-acceptance drift")

    with pytest.raises(RuntimeError, match="cache-acceptance drift"):
        initialise_run_directory(
            runs,
            identity,
            {"stage": "fixture"},
            publication_guard=reject_acceptance,
        )

    assert observed_bytes == [original_bytes]
    assert metadata_path.read_bytes() == original_bytes
    released = json.loads(
        run_directory_lock_path(run_directory).read_text(encoding="utf-8")
    )
    assert released["state"] == "released"


def test_sigkill_releases_os_lock_even_when_diagnostic_record_is_stale(tmp_path):
    run_directory = tmp_path / "runs" / "crash-run"
    process = _holder(run_directory, "crash-run")
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "LOCKED"
        process.kill()
        assert process.wait(timeout=20) != 0

        stale = json.loads(run_directory_lock_path(run_directory).read_text())
        assert stale["state"] == "held"
        recovered = acquire_run_directory_lock(run_directory, run_id="crash-run")
        assert recovered.held
        assert recovered.owner["pid"] == os.getpid()
        recovered.release()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=20)


def test_distinct_run_ids_can_execute_concurrently(tmp_path):
    first_directory = tmp_path / "runs" / "run-one"
    second_directory = tmp_path / "runs" / "run-two"
    first = acquire_run_directory_lock(first_directory, run_id="run-one")
    try:
        result = _attempt(second_directory, "run-two")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "LOCKED"
        with pytest.raises(ValueError, match="basename"):
            # A direct second open in this process is normally made idempotent;
            # use a subprocess assertion above for true contention.  This bad
            # identity instead verifies path/run identity cannot be aliased.
            acquire_run_directory_lock(first_directory, run_id="run-two")
    finally:
        first.release()


def test_run_lock_rejects_symlink_and_hardlink_without_touching_target(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    target = tmp_path / "unrelated.txt"
    target.write_text("KEEP ME", encoding="utf-8")
    original_mode = target.stat().st_mode

    symlink_run = runs / "symlink-run"
    symlink_lock = run_directory_lock_path(symlink_run)
    symlink_lock.symlink_to(target)
    with pytest.raises(RunDirectoryLockError, match="unsafe"):
        acquire_run_directory_lock(symlink_run, run_id="symlink-run")
    assert target.read_text(encoding="utf-8") == "KEEP ME"
    assert target.stat().st_mode == original_mode
    symlink_lock.unlink()

    hardlink_run = runs / "hardlink-run"
    hardlink_lock = run_directory_lock_path(hardlink_run)
    os.link(target, hardlink_lock)
    with pytest.raises(RunDirectoryLockError, match="unsafe"):
        acquire_run_directory_lock(hardlink_run, run_id="hardlink-run")
    assert target.read_text(encoding="utf-8") == "KEEP ME"
    assert target.stat().st_mode == original_mode


def test_initialise_rejects_run_directory_symlink_without_external_write(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(repro, "environment_fingerprint", lambda: {"fixture": True})
    monkeypatch.setattr(
        repro,
        "git_state",
        lambda _root: {"available": False, "commit": None, "dirty": None},
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runs / "escaped-run").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RunDirectoryLockError, match="symlink"):
        initialise_run_directory(
            runs, _identity("escaped-run"), {"stage": "fixture"}
        )
    assert not (outside / "run.json").exists()


def test_shared_transaction_lock_blocks_writer_until_reader_releases(tmp_path):
    lock_path = tmp_path / "stage25.transaction.lock"
    with advisory_file_lock(lock_path, exclusive=False):
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", _ADVISORY_WRITER, str(lock_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        assert process.poll() is None
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr
    assert stdout.strip() == "WRITER_LOCKED"
