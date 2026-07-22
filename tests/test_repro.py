from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import chronology as chronology_module  # noqa: E402
from thermoroute import opening_contract as opening_contract_module  # noqa: E402
from thermoroute import repro as repro_module  # noqa: E402
from thermoroute.repro import (  # noqa: E402
    _canonical_native_library_identities,
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


def test_isolated_python_uses_random_hash_secret_but_identity_hash_is_stable():
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(ROOT / 'src')!r});"
        "from thermoroute.repro import sha256_json;"
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
        hash_value, identity_value = result.stdout.strip().splitlines()
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
