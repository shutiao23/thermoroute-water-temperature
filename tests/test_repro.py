from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


INPUT_CLOSURE_SHA256 = "f" * 64


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


def test_native_threadpool_policy_accepts_live_pools_within_declared_cap():
    repro_module._assert_native_threadpools_within_limit([
        {"user_api": "openmp", "num_threads": 1},
        _native_library(
            user_api="blas",
            internal_api="openblas",
            prefix="libopenblas",
        ),
    ])
    repro_module._assert_native_threadpools_within_limit(
        [
            _native_library(user_api="blas", num_threads=16),
            {"user_api": "openmp", "num_threads": 16},
        ],
        limit=16,
    )


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
    ),
)
def test_native_threadpool_policy_fails_closed_for_unproved_state(value):
    with pytest.raises(RuntimeError, match="thread-pool|BLAS/OpenMP"):
        repro_module._assert_native_threadpools_within_limit(value)


def test_native_threadpool_policy_rejects_over_cap_threads():
    with pytest.raises(RuntimeError, match="thread-pool|BLAS/OpenMP"):
        repro_module._assert_native_threadpools_within_limit(
            [_native_library(num_threads=17)],
            limit=16,
        )


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
    limit = repro_module.FORMAL_THREAD_LIMIT
    policy = _formal_policy_fixture()
    for name in repro_module.FORMAL_THREAD_ENVIRONMENT:
        policy["thread_environment"][name] = str(limit)
    policy["required"]["threads"] = limit
    policy["torch"]["num_threads"] = limit
    monkeypatch.setattr(repro_module, "formal_numerical_policy", lambda: policy)
    monkeypatch.setattr(repro_module, "_FORMAL_THREADPOOL_CONTROLLER", object())
    monkeypatch.setattr(
        repro_module,
        "_loaded_native_threadpools",
        lambda: [_native_library(num_threads=limit + 1)],
    )
    with pytest.raises(RuntimeError, match="declared cap"):
        repro_module.assert_formal_numerical_policy()


def test_formal_policy_assertion_returns_stable_policy_without_live_snapshot(
    monkeypatch,
):
    policy = _formal_policy_fixture()
    limit = repro_module.FORMAL_THREAD_LIMIT
    for name in repro_module.FORMAL_THREAD_ENVIRONMENT:
        policy["thread_environment"][name] = str(limit)
    policy["required"]["threads"] = limit
    policy["torch"]["num_threads"] = limit
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
    limit = repro_module.FORMAL_THREAD_LIMIT
    for name in repro_module.FORMAL_THREAD_ENVIRONMENT:
        policy["thread_environment"][name] = str(limit)
    policy["required"]["threads"] = limit
    policy["torch"]["num_threads"] = limit
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
    first = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"delta": 1.0},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )

    panel.write_text("site,y\na,2\n")
    data_changed = resolve_run_identity(root=root, panel=panel, registry=registry,
                                        config={"delta": 1.0},
                                        input_closure_sha256=INPUT_CLOSURE_SHA256)
    assert data_changed.run_id != first.run_id

    panel.write_text("site,y\na,1\n")
    config_changed = resolve_run_identity(root=root, panel=panel, registry=registry,
                                          config={"delta": 1.01},
                                          input_closure_sha256=INPUT_CLOSURE_SHA256)
    assert config_changed.run_id != first.run_id

    (root / "src" / "model.py").write_text("VALUE = 2\n")
    source_changed = resolve_run_identity(root=root, panel=panel, registry=registry,
                                          config={"delta": 1.0},
                                          input_closure_sha256=INPUT_CLOSURE_SHA256)
    assert source_changed.run_id != first.run_id

    closure_changed = resolve_run_identity(
        root=root,
        panel=panel,
        registry=registry,
        config={"delta": 1.0},
        input_closure_sha256="e" * 64,
    )
    assert closure_changed.run_id != source_changed.run_id


@pytest.mark.parametrize(
    "digest", ("", "a" * 63, "A" * 64, "g" * 64),
)
def test_run_identity_requires_lowercase_input_closure_sha256(
    tmp_path,
    digest,
):
    root, panel, registry = _fixture(tmp_path)
    with pytest.raises(ValueError, match="input_closure_sha256"):
        resolve_run_identity(
            root=root,
            panel=panel,
            registry=registry,
            config={"delta": 1.0},
            input_closure_sha256=digest,
        )


def test_resolve_run_identity_has_no_input_closure_default(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    with pytest.raises(TypeError, match="input_closure_sha256"):
        resolve_run_identity(  # type: ignore[call-arg]
            root=root, panel=panel, registry=registry, config={"delta": 1.0}
        )


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
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    monkeypatch.setattr(
        repro_module, "numerical_runtime_contract", lambda: second_contract
    )
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
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
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry,
        config={"stage": "09b_development_controls", "eval_batch_size": 4096},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
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
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    protocol.write_text('{"margin":0.10}\n')
    assert source_tree_hash(root) != first_source
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
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
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )

    shell.write_text("#!/usr/bin/env bash\npython scripts/train.py --formal\n")
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    assert second.run_id != first.run_id

    workflow.write_text("jobs:\n  test: {}\n")
    third = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    assert third.run_id != second.run_id


def test_hashed_transitive_lock_is_part_of_source_identity(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    hashed_lock = root / "requirements-lock-py312-hashed.txt"
    hashed_lock.write_text("numpy==1 --hash=sha256:one\n")
    first = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    hashed_lock.write_text("numpy==1 --hash=sha256:two\n")
    second = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    assert second.run_id != first.run_id


def test_cache_requires_matching_sidecar_and_intact_bytes(tmp_path):
    root, panel, registry = _fixture(tmp_path)
    identity = resolve_run_identity(root=root, panel=panel, registry=registry,
                                    config={"delta": 1.0},
                                    input_closure_sha256=INPUT_CLOSURE_SHA256)
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
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
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
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
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
    one = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 1},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    two = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"seed": 2},
        input_closure_sha256=INPUT_CLOSURE_SHA256,
    )
    artifact = root / "predictions.parquet"
    atomic_write_parquet(pd.DataFrame({"value": [1]}), artifact, index=False)
    seal_artifact(artifact, one, kind="predictions", schema="pred.v1")
    assert cache_is_valid(artifact, one, schema="pred.v1")
    assert not cache_is_valid(artifact, two, schema="pred.v1")
