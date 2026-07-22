from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
