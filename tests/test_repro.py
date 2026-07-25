from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import chronology as chronology_module  # noqa: E402
from thermoroute import opening_contract as opening_contract_module  # noqa: E402
from thermoroute import repro as repro_module  # noqa: E402
from thermoroute.repro import (  # noqa: E402
    _canonical_native_library_identities,
    advisory_file_lock,
    atomic_write_bytes,
    atomic_write_parquet,
    cache_is_valid,
    canonical_json,
    formal_numerical_policy,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    source_tree_hash,
    validate_artifact_sidecar,
)


def test_atomic_publication_guard_runs_after_staging_and_before_replace(tmp_path):
    destination = tmp_path / "authoritative.bin"
    destination.write_bytes(b"old-authoritative-bytes")
    observed: list[bytes] = []

    def reject() -> None:
        observed.extend(
            path.read_bytes()
            for path in destination.parent.glob(f".{destination.name}.*.tmp")
        )
        raise RuntimeError("injected live-policy drift")

    with pytest.raises(RuntimeError, match="live-policy drift"):
        atomic_write_bytes(
            destination,
            b"new-staged-bytes",
            publication_guard=reject,
        )
    assert observed == [b"new-staged-bytes"]
    assert destination.read_bytes() == b"old-authoritative-bytes"
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))

    parquet = tmp_path / "candidate.parquet"
    with pytest.raises(RuntimeError, match="live-policy drift"):
        atomic_write_parquet(
            pd.DataFrame({"value": [1.0]}),
            parquet,
            index=False,
            publication_guard=lambda: (_ for _ in ()).throw(
                RuntimeError("injected live-policy drift")
            ),
        )
    assert not parquet.exists()


def test_advisory_transaction_lock_rejects_symlink(tmp_path):
    target = tmp_path / "unrelated-owner-file"
    target.write_text("do not chmod or lock me", encoding="utf-8")
    original_mode = target.stat().st_mode
    link = tmp_path / "transaction.lock"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="cannot open transaction lock"):
        with advisory_file_lock(link, exclusive=True):
            raise AssertionError("symlink lock unexpectedly acquired")
    assert target.read_text(encoding="utf-8") == "do not chmod or lock me"
    assert target.stat().st_mode == original_mode


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_library(**overrides):
    library = {
        "user_api": "openmp",
        "internal_api": "openmp",
        "prefix": "libomp",
        "filepath": "/opt/runtime/lib/libomp.dylib",
        "version": "18.1.0",
        "architecture": "armv8",
        "threading_layer": None,
        "num_threads": 1,
    }
    library.update(overrides)
    return library


def test_native_threadpool_policy_accepts_only_live_single_thread_pools():
    repro_module._assert_native_threadpools_single_thread([
        {"user_api": "openmp", "num_threads": 1},
        _native_library(
            user_api="blas",
            internal_api="openblas",
            prefix="libopenblas",
        ),
    ])


@pytest.mark.parametrize(
    "value",
    (
        None,
        {},
        (),
        [],
        ["not-a-record"],
        [{}],
        [{"user_api": "blas"}],
        [{"num_threads": 1}],
        [_native_library(user_api=None)],
        [_native_library(user_api="")],
        [_native_library(user_api="unknown")],
        [_native_library(user_api="BLAS")],
        [_native_library(num_threads=None)],
        [_native_library(num_threads=True)],
        [_native_library(num_threads=False)],
        [_native_library(num_threads=1.0)],
        [_native_library(num_threads="1")],
        [_native_library(num_threads=-1)],
        [_native_library(num_threads=0)],
        [_native_library(num_threads=2)],
        [_native_library(), _native_library(num_threads=8)],
    ),
)
def test_native_threadpool_policy_fails_closed_for_unproved_state(value):
    with pytest.raises(RuntimeError, match="thread-pool|BLAS/OpenMP"):
        repro_module._assert_native_threadpools_single_thread(value)


def test_native_threadpool_inspection_error_fails_closed(monkeypatch):
    import threadpoolctl

    def fail():
        raise OSError("injected inspection failure")

    monkeypatch.setattr(threadpoolctl, "threadpool_info", fail)
    with pytest.raises(RuntimeError, match="cannot inspect") as caught:
        repro_module._loaded_native_threadpools()
    assert isinstance(caught.value.__cause__, OSError)


def test_native_threadpool_import_error_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    with pytest.raises(RuntimeError, match="cannot inspect") as caught:
        repro_module._loaded_native_threadpools()
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


def _formal_policy_fixture():
    hash_policy = "canonical-sort-identity-collections-independent-of-hash-secret"
    return {
        "thread_environment": {
            name: "1" for name in repro_module.FORMAL_THREAD_ENVIRONMENT
        },
        "cublas_workspace_config": ":4096:8",
        "python_hash_environment_declaration": "0",
        "python_hash_randomization_enabled": True,
        "python_hash_policy": hash_policy,
        "required": {
            "threads": 1,
            "cublas_workspace_config": ":4096:8",
            "python_hash_policy": hash_policy,
            "torch_deterministic_algorithms": True,
            "tf32": False,
            "float32_matmul_precision": "highest",
        },
        "torch": {
            "num_threads": 1,
            "num_interop_threads": 1,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }


def test_formal_policy_assertion_rejects_effective_native_drift(monkeypatch):
    policy = _formal_policy_fixture()
    monkeypatch.setattr(repro_module, "formal_numerical_policy", lambda: policy)
    monkeypatch.setattr(repro_module, "_FORMAL_THREADPOOL_CONTROLLER", object())
    monkeypatch.setattr(
        repro_module,
        "_loaded_native_threadpools",
        lambda: [_native_library(num_threads=2)],
    )
    with pytest.raises(RuntimeError, match="exactly one thread"):
        repro_module.assert_formal_numerical_policy()


def test_formal_policy_assertion_returns_stable_policy_without_live_snapshot(
    monkeypatch,
):
    policy = _formal_policy_fixture()
    monkeypatch.setattr(repro_module, "formal_numerical_policy", lambda: policy)
    monkeypatch.setattr(repro_module, "_FORMAL_THREADPOOL_CONTROLLER", object())
    monkeypatch.setattr(
        repro_module,
        "_loaded_native_threadpools",
        lambda: [{"user_api": "blas", "num_threads": 1}],
    )
    assert repro_module.assert_formal_numerical_policy() is policy
    assert "native_threadpools" not in policy


def test_formal_policy_assertion_requires_process_lifetime_limiter(monkeypatch):
    policy = _formal_policy_fixture()
    monkeypatch.setattr(repro_module, "formal_numerical_policy", lambda: policy)
    monkeypatch.setattr(repro_module, "_FORMAL_THREADPOOL_CONTROLLER", None)
    monkeypatch.setattr(
        repro_module,
        "_loaded_native_threadpools",
        lambda: [{"user_api": "blas", "num_threads": 1}],
    )
    with pytest.raises(RuntimeError, match="limiter is not active"):
        repro_module.assert_formal_numerical_policy()


def test_native_library_identity_folds_only_exact_duplicates():
    library = _native_library()
    result = _canonical_native_library_identities(
        [library, dict(library), {**library, "num_threads": 8}]
    )
    assert len(result) == 1
    assert "num_threads" not in result[0]


def test_native_library_identity_preserves_real_runtime_differences(tmp_path):
    first = tmp_path / "first" / "libomp.dylib"
    relocated = tmp_path / "relocated" / "libomp.dylib"
    changed = tmp_path / "changed" / "libomp.dylib"
    for path, payload in (
        (first, b"same native bytes"),
        (relocated, b"same native bytes"),
        (changed, b"different native bytes"),
    ):
        path.parent.mkdir()
        path.write_bytes(payload)
    base = _native_library(filepath=str(first))
    variants = [
        base,
        _native_library(filepath=str(relocated)),
        _native_library(filepath=str(changed)),
        _native_library(filepath=str(first), version="19.0.0"),
        _native_library(filepath=str(first), architecture="x86_64"),
        _native_library(filepath=str(first), threading_layer="pthreads"),
    ]
    identities = _canonical_native_library_identities(variants)
    assert len(identities) == len(variants) - 1
    assert all("filepath" not in identity for identity in identities)
    assert all(
        identity["binary_sha256"] is not None for identity in identities
    )


def test_native_library_identity_is_import_order_independent():
    libraries = [
        _native_library(),
        _native_library(
            user_api="blas",
            internal_api="openblas",
            prefix="libopenblas",
            filepath="/opt/runtime/lib/libopenblas.dylib",
            version="0.3.27",
            threading_layer="pthreads",
        ),
        _native_library(),
    ]
    forward = _canonical_native_library_identities(libraries)
    reverse = _canonical_native_library_identities(reversed(libraries))
    assert forward == reverse
    assert forward == sorted(forward, key=canonical_json)


def test_linux_cpu_identity_binds_model_flags_and_available_microcode():
    cpuinfo = """\
processor : 0
vendor_id : GenuineIntel
cpu family : 6
model : 154
model name : Example CPU 3.20GHz
stepping : 3
microcode : 0x42
flags : sse2 avx avx2

processor : 1
vendor_id : GenuineIntel
cpu family : 6
model : 154
model name :   Example CPU   3.20GHz
stepping : 3
microcode : 0x42
flags : avx2 sse2 avx
"""
    identity = repro_module._linux_cpu_identity(
        cpuinfo, sysfs_microcode="0x99\n"
    )
    assert identity == {
        "vendor_ids": ["GenuineIntel"],
        "models": [{
            "model name": "Example CPU 3.20GHz",
            "cpu family": "6",
            "model": "154",
            "stepping": "3",
        }],
        "isa_flag_sets": [["avx", "avx2", "sse2"]],
        "microcode_versions": ["0x42"],
    }
    assert "processor" not in canonical_json(identity)


def test_linux_cpu_identity_preserves_heterogeneous_cpu_classes():
    cpuinfo = """\
processor : 0
CPU implementer : 0x41
CPU architecture : 8
CPU part : 0xd0c
Features : fp asimd aes

processor : 1
CPU implementer : 0x41
CPU architecture : 8
CPU part : 0xd40
Features : fp asimd aes sve
"""
    identity = repro_module._linux_cpu_identity(cpuinfo)
    assert identity["vendor_ids"] == ["0x41"]
    assert len(identity["models"]) == 2
    assert {tuple(flags) for flags in identity["isa_flag_sets"]} == {
        ("aes", "asimd", "fp"),
        ("aes", "asimd", "fp", "sve"),
    }
    assert identity["microcode_versions"] is None


def test_darwin_cpu_identity_binds_apple_model_and_supported_isa_flags():
    identity = repro_module._darwin_cpu_identity("""\
machdep.cpu.brand_string: Apple M3 Max
hw.cpufamily: 12345
hw.cpusubfamily: 7
hw.optional.arm.FEAT_AES: 1
hw.optional.arm.FEAT_SME: 0
hw.logicalcpu: 16
""")
    assert identity == {
        "vendor_ids": ["Apple"],
        "models": [{
            "machdep.cpu.brand_string": "Apple M3 Max",
            "hw.cpufamily": "12345",
            "hw.cpusubfamily": "7",
        }],
        "isa_flag_sets": [["hw.optional.arm.FEAT_AES"]],
        "microcode_versions": None,
    }
    assert "logicalcpu" not in canonical_json(identity)


def test_cpu_identity_fails_closed_when_required_facts_are_absent():
    with pytest.raises(RuntimeError, match="ISA flags"):
        repro_module._linux_cpu_identity(
            "vendor_id: GenuineIntel\nmodel name: Example CPU\n"
        )
    with pytest.raises(RuntimeError, match="vendor"):
        repro_module._darwin_cpu_identity(
            "machdep.cpu.brand_string: Example CPU\nhw.optional.sse2: 1\n"
        )


def test_darwin_runtime_uses_absolute_sysctl_under_allowlisted_path(monkeypatch):
    monkeypatch.setattr(
        repro_module,
        "_stable_operating_system_identity",
        lambda: {"system": "Darwin", "product_version": "15.5"},
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "machdep.cpu.brand_string: Apple M2 Pro\n"
                "hw.cpufamily: 458787763\n"
                "hw.optional.arm.FEAT_AES: 1\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(repro_module.subprocess, "run", fake_run)
    identity = repro_module._stable_host_numerical_identity()

    assert commands[0][0] == [repro_module.DARWIN_SYSCTL_PATH, "-a"]
    assert Path(commands[0][0][0]).is_absolute()
    assert identity["cpu"]["vendor_ids"] == ["Apple"]


def test_os_identity_binds_linux_kernel_libc_and_distribution(monkeypatch):
    monkeypatch.setattr(repro_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(repro_module.platform, "release", lambda: "6.8.12")
    monkeypatch.setattr(repro_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        repro_module.platform, "libc_ver", lambda: ("glibc", "2.39")
    )
    monkeypatch.setattr(
        repro_module.platform,
        "freedesktop_os_release",
        lambda: {"ID": "example", "VERSION_ID": "24.04", "NAME": "ignored"},
    )
    identity = repro_module._stable_operating_system_identity()
    assert identity["system"] == "Linux"
    assert identity["kernel_release"] == "6.8.12"
    assert identity["process_abi"]["machine"] == "x86_64"
    assert identity["libc"] == {"implementation": "glibc", "version": "2.39"}
    assert identity["distribution"] == {"id": "example", "version_id": "24.04"}


def test_os_identity_binds_macos_product_and_kernel_versions(monkeypatch):
    monkeypatch.setattr(repro_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(repro_module.platform, "release", lambda: "24.6.0")
    monkeypatch.setattr(repro_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        repro_module.platform, "mac_ver", lambda: ("15.6", ("", "", ""), "")
    )
    identity = repro_module._stable_operating_system_identity()
    assert identity["system"] == "Darwin"
    assert identity["kernel_release"] == "24.6.0"
    assert identity["product_version"] == "15.6"
    assert identity["process_abi"]["machine"] == "arm64"


def test_os_identity_fails_closed_for_unknown_platform(monkeypatch):
    monkeypatch.setattr(repro_module.platform, "system", lambda: "UnknownOS")
    monkeypatch.setattr(repro_module.platform, "release", lambda: "1")
    monkeypatch.setattr(repro_module.platform, "machine", lambda: "example")
    with pytest.raises(RuntimeError, match="does not support OS"):
        repro_module._stable_operating_system_identity()


def test_isolated_python_uses_random_hash_secret_but_identity_hash_is_stable():
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(ROOT / 'src')!r});"
        "from thermoroute.repro import (assert_formal_numerical_policy,"
        "configure_deterministic_runtime,sha256_json);"
        "configure_deterministic_runtime();"
        "policy=assert_formal_numerical_policy(require_hash_randomization=True);"
        "print(int(policy['python_hash_randomization_enabled']));"
        "print(hash('thermoroute-hash-probe'));"
        "print(sha256_json({'members': {'zeta','alpha','beta'}}))"
    )
    observations = []
    for _ in range(4):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            text=True,
            capture_output=True,
            check=True,
            env=environment,
        )
        hash_flag, hash_value, identity_value = result.stdout.strip().splitlines()
        assert hash_flag == "1"
        observations.append((hash_value, identity_value))
    # This is the negative control: -I ignored the apparent fixed-seed
    # declaration, so ordinary hash() values are intentionally not reproducible.
    assert len({value for value, _digest in observations}) > 1
    # Formal identities remain reproducible because canonical_json sorts sets
    # and mappings instead of relying on hash-table iteration.
    assert len({digest for _value, digest in observations}) == 1
    assert formal_numerical_policy()["python_hash_policy"].startswith(
        "canonical-sort-identity-collections"
    )
    # The four isolated children above, not the unrelated parent interpreter,
    # are the contract under test.  The formal shell entrypoint likewise unsets
    # PYTHONHASHSEED before starting any evidence-bearing Python interpreter.


def _fixture(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "model.py").write_text("VALUE = 1\n")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / "requirements-lock.txt").write_text("numpy==1\n")
    panel = root / "panel.csv"
    registry = root / "registry.csv"
    panel.write_text("site,y\na,1\n")
    registry.write_text("site,site_no\na,1\n")
    return root, panel, registry


def test_run_identity_changes_with_data_config_and_source(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    first = resolve_run_identity(root=root, panel=panel, registry=registry, config={"delta": 1.0})

    panel.write_text("site,y\na,2\n")
    data_changed = resolve_run_identity(root=root, panel=panel, registry=registry,
                                        config={"delta": 1.0})
    assert data_changed.run_id != first.run_id

    panel.write_text("site,y\na,1\n")
    config_changed = resolve_run_identity(root=root, panel=panel, registry=registry,
                                          config={"delta": 1.01})
    assert config_changed.run_id != first.run_id

    (root / "src" / "model.py").write_text("VALUE = 2\n")
    source_changed = resolve_run_identity(root=root, panel=panel, registry=registry,
                                          config={"delta": 1.0})
    assert source_changed.run_id != first.run_id


def test_run_identity_automatically_tracks_host_runtime_hash(monkeypatch, tmp_path):
    root, panel, registry = _fixture(tmp_path)
    first_contract = {
        "host_numerical_identity": {"cpu": {"vendor_ids": ["Vendor A"]}}
    }
    second_contract = {
        "host_numerical_identity": {"cpu": {"vendor_ids": ["Vendor B"]}}
    }
    monkeypatch.setattr(
        repro_module, "numerical_runtime_contract", lambda: first_contract
    )
    first = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    monkeypatch.setattr(
        repro_module, "numerical_runtime_contract", lambda: second_contract
    )
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    assert first.runtime_sha256 == repro_module.sha256_json(first_contract)
    assert second.runtime_sha256 == repro_module.sha256_json(second_contract)
    assert second.runtime_sha256 != first.runtime_sha256
    assert second.run_id != first.run_id


def test_eval_batch_size_is_a_scientific_run_identity_input(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    first = resolve_run_identity(
        root=root, panel=panel, registry=registry,
        config={"stage": "09b_development_controls", "eval_batch_size": 2048},
    )
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry,
        config={"stage": "09b_development_controls", "eval_batch_size": 4096},
    )
    assert first.config_sha256 != second.config_sha256
    assert first.run_id != second.run_id


def test_protocol_is_part_of_source_and_run_identity(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    protocol = root / "protocols" / "route_a.json"
    protocol.parent.mkdir()
    protocol.write_text('{"margin":0.05}\n')
    first_source = source_tree_hash(root)
    first = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    protocol.write_text('{"margin":0.10}\n')
    assert source_tree_hash(root) != first_source
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    assert second.run_id != first.run_id


def test_shell_and_ci_entrypoints_are_part_of_source_identity(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    shell = root / "scripts" / "run_all.sh"
    workflow = root / ".github" / "workflows" / "ci.yml"
    shell.parent.mkdir(exist_ok=True)
    workflow.parent.mkdir(parents=True)
    shell.write_text("#!/usr/bin/env bash\npython scripts/train.py\n")
    workflow.write_text("jobs: {}\n")
    first = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )

    shell.write_text("#!/usr/bin/env bash\npython scripts/train.py --formal\n")
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    assert second.run_id != first.run_id

    workflow.write_text("jobs:\n  test: {}\n")
    third = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    assert third.run_id != second.run_id


def test_stage09_run_all_manifest_and_chronology_paths_are_exactly_aligned():
    manifest = _load_script_module(
        ROOT / "scripts/14_manifest.py", "manifest_stage09_path_contract_test"
    )
    expected = chronology_module.STAGE09_ARTIFACT_PATHS
    assert manifest.STAGE09_PREDICTIONS_PATH == expected["predictions"]
    assert manifest.STAGE09_SCORES_PATH == expected["scores"]

    run_all = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    release = (ROOT / "scripts" / "make_release_archive.sh").read_text(
        encoding="utf-8"
    )
    for script in (run_all, release):
        assert 'readonly THERMOROUTE_PYTHON="${THERMOROUTE_PYTHON:-python}"' in script
        assert "v != (3, 12)" in script
        assert "assert sys.version_info" not in script
        assert re.search(r"(?<![/\w])python3(?:\s|$)", script) is None
    assert "unset PYTHONHASHSEED" in run_all
    assert "export PYTHONHASHSEED=0" not in run_all
    assert "SOURCE_GIT_DIRTY=()" not in release
    cleanup = release.split("cleanup() {", 1)[1].split("\n}", 1)[0]
    assert "local status=$?" in cleanup
    assert "trap - EXIT" in cleanup
    assert 'exit "$status"' in cleanup
    cleanup_probe = subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -euo pipefail\n"
                "TMP_ROOT=$(mktemp -d)\n"
                "cleanup() {" + cleanup + "\n}\n"
                "trap cleanup EXIT\n"
                "false\n"
            ),
        ],
        check=False,
    )
    assert cleanup_probe.returncode != 0
    logical_lines = run_all.replace("\\\n", " ").splitlines()
    command = next(
        line.strip()
        for line in logical_lines
        if line.strip().startswith(
            '"$THERMOROUTE_PYTHON" scripts/09_usgs_experiment.py '
        )
    )
    arguments = shlex.split(command)
    assert "--out_report" not in arguments
    assert "--out_scores" not in arguments
    prediction_option = arguments.index("--out_predictions")
    assert arguments[prediction_option + 1] == Path(expected["predictions"]).name

    stage09_source = (ROOT / "scripts" / "09_usgs_experiment.py").read_text(
        encoding="utf-8"
    )
    for label in ("predictions", "report", "scores"):
        assert (
            f'default=Path(STAGE09_ARTIFACT_PATHS["{label}"]).name'
            in stage09_source
        )


def test_hashed_transitive_lock_is_part_of_source_identity(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    hashed_lock = root / "requirements-lock-py312-hashed.txt"
    hashed_lock.write_text("numpy==1 --hash=sha256:one\n")
    first = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    hashed_lock.write_text("numpy==1 --hash=sha256:two\n")
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    assert second.run_id != first.run_id


def test_source_identity_is_identical_across_all_freeze_and_release_chains(tmp_path):
    root, _panel, _registry = _fixture(tmp_path)
    files = {
        "scripts/freeze.py": "print('freeze')\n",
        "scripts/run.sh": "#!/usr/bin/env bash\npython scripts/freeze.py\n",
        "scripts/_archive/retired.py": "RETIRED = True\n",
        "tests/test_fixture.py": "def test_fixture(): assert True\n",
        "protocols/route_a.json": "{}\n",
        "protocols/route_a.md": "# protocol\n",
        ".github/workflows/ci.yml": "jobs: {}\n",
        ".github/workflows/audit.yaml": "jobs: {}\n",
        "requirements.txt": "numpy>=1\n",
        "requirements-lock-py312-hashed.txt": "numpy==1 --hash=sha256:one\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    manifest = _load_script_module(ROOT / "scripts/14_manifest.py", "manifest_identity_test")
    release = _load_script_module(ROOT / "scripts/verify_release.py", "release_identity_test")
    expected_patterns = repro_module.DEFAULT_SOURCE_PATTERNS
    assert chronology_module.SOURCE_INVENTORY_PATTERNS == expected_patterns
    assert opening_contract_module.SOURCE_INVENTORY_PATTERNS == expected_patterns
    assert manifest.RUN_SOURCE_PATTERNS == expected_patterns
    assert release.SOURCE_INVENTORY_PATTERNS == expected_patterns
    assert "requirements-lock*.txt" in manifest.SOURCE_PATTERNS

    git_environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Source identity test",
        "GIT_AUTHOR_EMAIL": "source-identity@example.invalid",
        "GIT_COMMITTER_NAME": "Source identity test",
        "GIT_COMMITTER_EMAIL": "source-identity@example.invalid",
    }
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)

    def commit(message: str) -> str:
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-q",
                "-m",
                message,
            ],
            cwd=root,
            check=True,
            env=git_environment,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def hashes(commit_sha: str) -> dict[str, str]:
        _bindings, chronology_hash = chronology_module._collect_model_source_control(
            root, commit_sha
        )
        opening_inventory = opening_contract_module._source_inventory(root)
        release_paths = release._working_source_inventory_paths(root)
        release_inventory = {
            relative: release.sha256_file(root / relative)
            for relative in sorted(release_paths)
        }
        return {
            "repro": repro_module.source_tree_hash(root),
            "chronology": chronology_hash,
            "opening_contract": repro_module.sha256_json(opening_inventory),
            "manifest": manifest._run_source_sha256(root),
            "release": release._sha256_json(release_inventory),
        }

    first = hashes(commit("initial source closure"))
    assert len(set(first.values())) == 1

    third_lock = root / "requirements-lock-experimental.txt"
    third_lock.write_text("numpy==1\n", encoding="utf-8")
    second = hashes(commit("add third dependency lock"))
    assert len(set(second.values())) == 1
    assert next(iter(second.values())) != next(iter(first.values()))
    assert "requirements-lock-experimental.txt" in {
        path.relative_to(root).as_posix()
        for path in manifest._iter_files(root, manifest.SOURCE_PATTERNS)
    }


def test_cache_requires_matching_sidecar_and_intact_bytes(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    identity = resolve_run_identity(root=root, panel=panel, registry=registry,
                                    config={"delta": 1.0})
    artifact = root / "runs" / "predictions.parquet"
    atomic_write_parquet(pd.DataFrame({"value": [1.0]}), artifact, index=False)
    assert not cache_is_valid(artifact, identity, schema="pred.v1")

    seal_artifact(artifact, identity, kind="predictions", schema="pred.v1")
    assert cache_is_valid(artifact, identity, schema="pred.v1")

    # A truncated/overwritten artifact can never be accepted merely because its
    # final filename exists.
    artifact.write_bytes(b"corrupt")
    assert not cache_is_valid(artifact, identity, schema="pred.v1")


def test_parent_sidecar_validation_is_exact_and_rejects_unknown_fields(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    identity = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    artifact = root / "parent.parquet"
    atomic_write_parquet(pd.DataFrame({"value": [1.0]}), artifact, index=False)
    sidecar = seal_artifact(
        artifact,
        identity,
        kind="parent",
        schema="pred.v1",
        parents={"panel": identity.panel_sha256},
    )
    assert validate_artifact_sidecar(
        artifact, identity=identity, schema="pred.v1", kind="parent"
    )["parents"] == {"panel": identity.panel_sha256}
    changed = json.loads(sidecar.read_text(encoding="utf-8"))
    changed["undeclared"] = True
    sidecar.write_text(json.dumps(changed), encoding="utf-8")
    assert not cache_is_valid(artifact, identity, schema="pred.v1")


def test_resealing_identical_lineage_preserves_exact_sidecar_bytes(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    identity = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1}
    )
    artifact = root / "predictions.parquet"
    atomic_write_parquet(pd.DataFrame({"value": [1.0]}), artifact, index=False)
    sidecar = seal_artifact(
        artifact,
        identity,
        kind="predictions",
        schema="pred.v1",
        parents={"panel": identity.panel_sha256},
        extra={"stage": 9},
    )
    first = sidecar.read_bytes()
    first_sha = sha256_file(sidecar)
    second = seal_artifact(
        artifact,
        identity,
        kind="predictions",
        schema="pred.v1",
        parents={"panel": identity.panel_sha256},
        extra={"stage": 9},
    )
    assert second == sidecar
    assert sidecar.read_bytes() == first
    assert sha256_file(sidecar) == first_sha


def test_different_run_never_hits_existing_cache(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    one = resolve_run_identity(root=root, panel=panel, registry=registry, config={"seed": 1})
    two = resolve_run_identity(root=root, panel=panel, registry=registry, config={"seed": 2})
    artifact = root / "predictions.parquet"
    atomic_write_parquet(pd.DataFrame({"value": [1]}), artifact, index=False)
    seal_artifact(artifact, one, kind="predictions", schema="pred.v1")
    assert cache_is_valid(artifact, one, schema="pred.v1")
    assert not cache_is_valid(artifact, two, schema="pred.v1")
