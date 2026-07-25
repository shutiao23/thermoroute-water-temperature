from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess

import pytest

from thermoroute import release_acceptance as acceptance
from thermoroute.release_acceptance import (
    ReleaseAcceptanceError,
    run_release_mechanics_acceptance,
    validate_release_mechanics_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / acceptance.CORE_RELATIVE
RUNNER = ROOT / acceptance.RUNNER_RELATIVE


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _self_hash(value: dict[str, object]) -> str:
    stable = dict(value)
    stable.pop("receipt_self_sha256", None)
    payload = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fake_verifier(mode: str = "pass") -> bytes:
    # The fixture is intentionally stdlib-only.  It exits nonzero unless the
    # process itself observes every promised flag/environment/cwd condition.
    return f"""#!/usr/bin/env python3
from pathlib import Path
import json
import os
import sys

expected_environment = {{
    "PATH": os.defpath,
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TMPDIR": os.environ.get("TMPDIR"),
}}
assert sys.flags.isolated == 1
assert sys.flags.ignore_environment == 1
assert sys.flags.no_user_site == 1
assert sys.flags.safe_path
assert sys.flags.dont_write_bytecode == 1
assert dict(os.environ) == expected_environment
assert not any(
    key.startswith(("PYTHON", "GIT_", "DYLD")) or key == "LD_PRELOAD"
    for key in os.environ
)
assert len(sys.argv) == 4
archive = Path(sys.argv[1]).resolve()
assert archive.is_file()
assert sys.argv[2:] == ["--distribution", "LOCAL_EVIDENCE_ONLY"]
temporary_root = Path(os.environ["TMPDIR"]).resolve()
cwd = Path.cwd().resolve()
assert cwd == temporary_root / "fresh-cwd"
assert not any(cwd.iterdir())
assert os.getpid() != os.getppid()
mode = {mode!r}
if mode == "fail":
    print("synthetic verifier failure", file=sys.stderr)
    raise SystemExit(9)
if mode == "write_cwd":
    Path("forbidden-write.txt").write_text("bad", encoding="utf-8")
if mode == "bad_profile":
    print(f"LOCAL EVIDENCE OK [DO NOT DISTRIBUTE; WRONG_PROFILE]: {{archive}}")
else:
    print("manifest OK: 7 artifacts, source abcdef012345, DAG 11 nodes")
    print(
        "LOCAL EVIDENCE OK [DO NOT DISTRIBUTE; PREOPEN_NOT_COMPLETE]: "
        + str(archive)
    )
""".encode("utf-8")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        check=True,
    )
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _make_fixture(tmp_path: Path, *, mode: str = "pass") -> tuple[Path, Path, Path]:
    root = tmp_path / "fixture-repository"
    (root / "src/thermoroute").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(CORE, root / acceptance.CORE_RELATIVE)
    shutil.copy2(RUNNER, root / acceptance.RUNNER_RELATIVE)
    verifier = root / acceptance.VERIFIER_RELATIVE
    verifier.write_bytes(_fake_verifier(mode))
    verifier.chmod(0o755)
    (root / ".gitignore").write_text("dist/\noutputs/prelabel/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _commit(root, "fixed synthetic release verifier")

    archive = root / "dist/synthetic_PREOPEN_NOT_COMPLETE.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"synthetic archive boundary; contains no labels\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    receipt = root / acceptance.RECEIPT_RELATIVE
    return root, archive, receipt


def _produce(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    root, archive, receipt = _make_fixture(tmp_path)
    document = run_release_mechanics_acceptance(archive, root=root)
    return root, archive, receipt, document


def _rewrite_receipt(
    receipt: Path,
    document: dict[str, object],
    *,
    refresh_self_hash: bool = True,
) -> None:
    attacked = copy.deepcopy(document)
    if refresh_self_hash:
        attacked["receipt_self_sha256"] = _self_hash(attacked)
    receipt.chmod(0o644)
    receipt.write_bytes(_canonical_bytes(attacked))
    receipt.chmod(0o444)


def test_real_child_observes_isolated_flags_exact_environment_and_fresh_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, archive, receipt = _make_fixture(tmp_path)
    # These common injection channels exist in the controller but the fixture
    # verifier asserts that none reaches the child.
    for key, value in {
        "PYTHONPATH": "/attacker/python",
        "PYTHONSTARTUP": "/attacker/startup.py",
        "GIT_DIR": "/attacker/git",
        "GIT_CONFIG_COUNT": "1",
        "DYLD_INSERT_LIBRARIES": "/attacker/lib.dylib",
        "LD_PRELOAD": "/attacker/lib.so",
    }.items():
        monkeypatch.setenv(key, value)

    controller_pid = os.getpid()
    document = run_release_mechanics_acceptance(archive, root=root)

    assert receipt.is_file()
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444
    assert document["execution"]["controller_pid"] == controller_pid
    assert document["execution"]["verifier_pid"] != controller_pid
    assert document["execution"]["python_isolated"] is True
    assert document["execution"]["bytecode_disabled"] is True
    assert document["execution"]["fresh_cwd_empty_before"] is True
    assert document["execution"]["fresh_cwd_empty_after"] is True
    assert document["execution"]["temporary_root_removed_after_exit"] is True
    assert document["execution"]["environment"] == (acceptance.CHILD_ENVIRONMENT_CONTRACT)
    transcript = document["execution"]["stdout_transcript"]
    expected_stdout = (
        transcript[0]
        + "\n"
        + acceptance.PROFILE_EVIDENCE_LINE.replace("<archive>", str(archive))
        + "\n"
    ).encode("utf-8")
    assert document["execution"]["stdout_sha256"] == hashlib.sha256(expected_stdout).hexdigest()
    assert document["execution"]["stdout_bytes"] == len(expected_stdout)
    assert document["execution"]["stderr_transcript"] == []
    assert document["execution"]["stderr_sha256"] == hashlib.sha256(b"").hexdigest()
    assert document["execution"]["stderr_bytes"] == 0
    assert validate_release_mechanics_receipt(receipt, root=root) == document


def test_receipt_is_canonical_create_only_and_scoped_to_mechanics(tmp_path: Path) -> None:
    root, archive, receipt, document = _produce(tmp_path)
    parsed = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt.read_bytes() == _canonical_bytes(parsed)
    assert parsed == document
    assert document["format"] == acceptance.RECEIPT_FORMAT
    assert document["status"] == acceptance.RECEIPT_STATUS
    assert document["profile"] == "PREOPEN_NOT_COMPLETE"
    assert "MECHANICS_ACCEPTANCE_ONLY" in document["evidence_scope"]
    assert "NOT_MODEL_READINESS" in document["evidence_scope"]
    assert document["receipt_self_sha256"] == _self_hash(document)
    assert document["isolation_limitations"] == acceptance.ISOLATION_LIMITATIONS
    assert all(value is False for value in document["isolation_limitations"].values())
    assert document["label_safety"] == acceptance.LABEL_SAFETY
    assert all(value is False for value in document["label_safety"].values())

    with pytest.raises(ReleaseAcceptanceError, match="refusing to replace"):
        run_release_mechanics_acceptance(archive, root=root)


def test_resource_observation_has_exact_single_run_units(tmp_path: Path) -> None:
    _root, _archive, _receipt, document = _produce(tmp_path)
    observation = document["resource_observation"]
    assert observation["wall_time_ns"] > 0
    assert observation["user_cpu_time_ns"] >= 0
    assert observation["system_cpu_time_ns"] >= 0
    assert observation["total_cpu_time_ns"] == (
        observation["user_cpu_time_ns"] + observation["system_cpu_time_ns"]
    )
    assert observation["measurement_backend"] == acceptance.MEASUREMENT_BACKEND
    assert observation["repetitions"] == 1
    if platform.system() == "Darwin":
        assert observation["peak_rss_raw_unit"] == "bytes"
        assert observation["peak_rss_bytes"] == observation["peak_rss_raw"]
    else:
        assert platform.system() == "Linux"
        assert observation["peak_rss_raw_unit"] == "kibibytes"
        assert observation["peak_rss_bytes"] == observation["peak_rss_raw"] * 1024


@pytest.mark.parametrize("mode", ("fail", "bad_profile", "write_cwd"))
def test_failed_or_noncanonical_child_never_publishes_pass_receipt(
    tmp_path: Path,
    mode: str,
) -> None:
    root, archive, receipt = _make_fixture(tmp_path, mode=mode)
    with pytest.raises(ReleaseAcceptanceError):
        run_release_mechanics_acceptance(archive, root=root)
    assert not receipt.exists()


def test_dirty_or_uncommitted_verifier_fails_before_child_and_receipt(tmp_path: Path) -> None:
    root, archive, receipt = _make_fixture(tmp_path)
    verifier = root / acceptance.VERIFIER_RELATIVE
    verifier.write_bytes(verifier.read_bytes() + b"\n# uncommitted attack\n")
    with pytest.raises(ReleaseAcceptanceError, match="clean Git worktree"):
        run_release_mechanics_acceptance(archive, root=root)
    assert not receipt.exists()


def test_archive_must_be_canonical_dist_zip_with_exact_sidecar(tmp_path: Path) -> None:
    root, archive, receipt = _make_fixture(tmp_path)
    sidecar = Path(str(archive) + ".sha256")
    sidecar.write_text("0" * 64 + f"  {archive.name}\n", encoding="utf-8")
    with pytest.raises(ReleaseAcceptanceError, match="sidecar"):
        run_release_mechanics_acceptance(archive, root=root)
    assert not receipt.exists()

    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"outside")
    Path(str(outside) + ".sha256").write_text(
        hashlib.sha256(b"outside").hexdigest() + "  outside.zip\n",
        encoding="utf-8",
    )
    with pytest.raises(ReleaseAcceptanceError, match="outside repository"):
        run_release_mechanics_acceptance(outside, root=root)


def test_duplicate_nan_noncanonical_extra_and_self_hash_attacks_fail_closed(
    tmp_path: Path,
) -> None:
    root, _archive, receipt, document = _produce(tmp_path)
    original = receipt.read_bytes()
    attacks = (
        b'{"format":"a","format":"b"}\n',
        b'{"format":NaN}\n',
        json.dumps(document, indent=2, sort_keys=True).encode() + b"\n",
    )
    for payload in attacks:
        receipt.chmod(0o644)
        receipt.write_bytes(payload)
        receipt.chmod(0o444)
        with pytest.raises(ReleaseAcceptanceError):
            validate_release_mechanics_receipt(receipt, root=root)

    receipt.chmod(0o644)
    receipt.write_bytes(original)
    receipt.chmod(0o444)
    attacked = copy.deepcopy(document)
    attacked["unexpected"] = True
    _rewrite_receipt(receipt, attacked)
    with pytest.raises(ReleaseAcceptanceError, match="exact schema"):
        validate_release_mechanics_receipt(receipt, root=root)

    attacked = copy.deepcopy(document)
    attacked["status"] = "PASS_FORGED"
    _rewrite_receipt(receipt, attacked, refresh_self_hash=False)
    with pytest.raises(ReleaseAcceptanceError, match="self-hash"):
        validate_release_mechanics_receipt(receipt, root=root)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        ("isolation_limitations", "fresh_machine", True, "overstates"),
        ("isolation_limitations", "container", True, "overstates"),
        ("isolation_limitations", "network_isolation_enforced", True, "overstates"),
        ("isolation_limitations", "deployment_latency_claim_allowed", True, "overstates"),
        ("label_safety", "outcome_read_absence_claim_allowed", True, "overstates"),
        ("execution", "command", ["sh", "-c", "opening"], "execution contract"),
        ("execution", "stdout_sha256", "0" * 64, "canonical transcript"),
        ("execution", "stderr_bytes", 1, "canonical transcript"),
        (
            "execution",
            "stdout_transcript",
            [
                "manifest OK: 8 artifacts, source abcdef012345, DAG 11 nodes",
                acceptance.PROFILE_EVIDENCE_LINE,
            ],
            "canonical transcript",
        ),
        ("resource_observation", "total_cpu_time_ns", 0, "CPU total"),
    ),
)
def test_claim_command_and_resource_forgery_fail_closed(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    root, _archive, receipt, document = _produce(tmp_path)
    attacked = copy.deepcopy(document)
    attacked[section][key] = value
    _rewrite_receipt(receipt, attacked)
    with pytest.raises(ReleaseAcceptanceError, match=message):
        validate_release_mechanics_receipt(receipt, root=root)


@pytest.mark.parametrize("target", ("archive", "sidecar", "core", "runner", "verifier"))
def test_every_bound_repository_or_archive_byte_tamper_fails_closed(
    tmp_path: Path,
    target: str,
) -> None:
    root, archive, receipt, document = _produce(tmp_path)
    paths = {
        "archive": archive,
        "sidecar": Path(str(archive) + ".sha256"),
        "core": root / document["source"]["core"]["path"],
        "runner": root / document["source"]["runner"]["path"],
        "verifier": root / document["source"]["verifier"]["path"],
    }
    selected = paths[target]
    selected.write_bytes(selected.read_bytes() + b"tamper")
    with pytest.raises(ReleaseAcceptanceError):
        validate_release_mechanics_receipt(receipt, root=root)


def test_interpreter_and_git_identity_receipt_forgery_fail_closed(tmp_path: Path) -> None:
    root, _archive, receipt, document = _produce(tmp_path)
    attacked = copy.deepcopy(document)
    attacked["runtime"]["python_executable"]["sha256"] = "0" * 64
    _rewrite_receipt(receipt, attacked)
    with pytest.raises(ReleaseAcceptanceError, match="Python executable byte"):
        validate_release_mechanics_receipt(receipt, root=root)

    attacked = copy.deepcopy(document)
    attacked["source"]["git_tree"] = "0" * 40
    _rewrite_receipt(receipt, attacked)
    with pytest.raises(ReleaseAcceptanceError, match="Git tree"):
        validate_release_mechanics_receipt(receipt, root=root)


def test_receipt_source_commit_may_precede_receipt_only_or_unrelated_commit(
    tmp_path: Path,
) -> None:
    root, _archive, receipt, document = _produce(tmp_path)
    origin = document["source"]["git_commit"]
    (root / "notes.txt").write_text("later non-control note\n", encoding="utf-8")
    later = _commit(root, "later receipt-adjacent note")
    assert later != origin
    assert validate_release_mechanics_receipt(receipt, root=root) == document


def test_receipt_hardlink_and_archive_symlink_are_rejected(tmp_path: Path) -> None:
    root, archive, receipt, _document = _produce(tmp_path)
    duplicate = receipt.with_name("duplicate-receipt.json")
    os.link(receipt, duplicate)
    with pytest.raises(ReleaseAcceptanceError, match="non-linked regular file"):
        validate_release_mechanics_receipt(receipt, root=root)

    duplicate.unlink()
    archive_payload = archive.read_bytes()
    archive.unlink()
    target = root / "dist/archive-target.zip"
    target.write_bytes(archive_payload)
    archive.symlink_to(target.name)
    with pytest.raises(ReleaseAcceptanceError, match="symbolic link|alias"):
        validate_release_mechanics_receipt(receipt, root=root)


def test_production_controller_has_no_opening_or_arbitrary_command_surface() -> None:
    runner_text = RUNNER.read_text(encoding="utf-8")
    core_text = CORE.read_text(encoding="utf-8")
    assert "run_opening_once" not in runner_text + core_text
    assert "resume_opening_once" not in runner_text + core_text
    assert "--command" not in runner_text
    assert acceptance.NORMALISED_COMMAND == [
        "<bound-python-executable>",
        "-I",
        "-B",
        "<repository>/scripts/verify_release.py",
        "<repository>/<archive>",
        "--distribution",
        "LOCAL_EVIDENCE_ONLY",
    ]
    assert acceptance.RELEASE_PROFILE == "PREOPEN_NOT_COMPLETE"
    assert acceptance.LABEL_SAFETY["runner_invoked_opening_entrypoint"] is False
