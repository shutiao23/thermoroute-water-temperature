from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STAGE27_SPEC = importlib.util.spec_from_file_location(
    "development_replay_stage27_test",
    ROOT / "scripts" / "27_verify_development_replay.py",
)
assert STAGE27_SPEC is not None and STAGE27_SPEC.loader is not None
STAGE27 = importlib.util.module_from_spec(STAGE27_SPEC)
STAGE27_SPEC.loader.exec_module(STAGE27)

from thermoroute.development_replay import (  # noqa: E402
    DEVELOPMENT_REPLAY_IO_GUARD_FORMAT,
    DEVELOPMENT_REPLAY_FORMAT,
    DevelopmentReplayIOGuard,
    LEARNED_EXTERNAL,
    LEARNED_TEMPORAL,
    REPLAY_ALLOWED_CONFIRMATION_READ_PATHS,
    REPLAY_ENTRYPOINT,
    REPLAY_FORBIDDEN_CONFIRMATION_NAMESPACE_STEMS,
    _confirmation_read_policy_attestation,
    _execution_identity,
    _is_forbidden_confirmation_read,
    _load_suite,
    _member_seeds,
    _validate_formal_pycache_prefix,
    fresh_verify_development_replay_receipt,
    run_development_replay,
    run_guarded_development_replay,
    validate_development_replay_receipt,
    write_replay_receipt,
)
import thermoroute.development_replay as DEVELOPMENT_REPLAY  # noqa: E402


FORBIDDEN_CONFIRMATION_READ_PATHS = (
    "data_usgs/confirmatory_predictors/temporal.parquet",
    "data_usgs/confirmatory_outcomes/water_temperature.parquet",
    "data_usgs/confirmatory_candidate_sites_v1.csv",
    "data_usgs/confirmatory_site_registry_v1.csv",
    "data_usgs/confirmatory_opening_authorization_v1.json",
    "data_usgs/confirmatory_model_suite_v1.json.backup",
    "data_usgs/raw_snapshots/confirmatory-candidates-v1/response.bin",
    "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1/response.bin",
    "data_usgs/raw_snapshots/openmeteo-gfs-previous-runs-v1/response.bin",
    "outputs/confirmatory/route_a_fixture/opening_receipt_v1.json",
)
from thermoroute.model_suite import (  # noqa: E402
    DEVELOPMENT_REPLAY_MODEL_CONTRACTS,
    MANDATORY_ABLATIONS,
    STAGE9_ABLATION_SEEDS,
    ModelSuiteError,
)
from thermoroute.repro import (  # noqa: E402
    numerical_runtime_contract,
    sha256_file,
    sha256_json,
    source_tree_hash,
)


def _receipt(root: Path, suite: Path) -> dict:
    entrypoint = root / REPLAY_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("# replay fixture\n", encoding="utf-8")
    rows = []
    for cohort, models in (
        ("temporal", LEARNED_TEMPORAL),
        ("external", LEARNED_EXTERNAL),
    ):
        for model in models:
            executor, members, atol = DEVELOPMENT_REPLAY_MODEL_CONTRACTS[
                cohort
            ][model]
            rows.append({
                "cohort": cohort,
                "model": model,
                "executor": executor,
                "members": members,
                "rows": 12,
                "atol": atol,
                "max_abs_difference": 0.0,
                "status": "PASS",
            })
    read_paths = sorted([
        REPLAY_ENTRYPOINT,
        suite.relative_to(root).as_posix(),
    ])
    execution = _execution_identity(
        root=root,
        suite_path=suite,
        receipt_path=root / "receipt.json",
        entrypoint_path=entrypoint,
    )
    execution.update({
        "io_guard": {
            "format": DEVELOPMENT_REPLAY_IO_GUARD_FORMAT,
            "network_access_allowed": False,
            "subprocess_allowed": False,
            "repository_writes_allowed": False,
            "confirmation_read_policy": _confirmation_read_policy_attestation(),
            "repo_read_path_count": len(read_paths),
            "repo_read_paths_sha256": sha256_json(read_paths),
            "repo_read_paths": read_paths,
            "violations": [],
        },
        "security_boundary": "fixture honest-owner boundary",
    })
    value = {
        "format": DEVELOPMENT_REPLAY_FORMAT,
        "status": "PASS_FULL_DEVELOPMENT_REPLAY_NO_CONFIRMATION_DATA",
        "isolated_process_required": True,
        "suite": {
            "path": suite.relative_to(root).as_posix(),
            "sha256": sha256_file(suite),
        },
        "source_tree_sha256": source_tree_hash(root),
        "runtime_sha256": sha256_json(numerical_runtime_contract()),
        "suite_numerical_runtime_sha256": sha256_json(
            numerical_runtime_contract()
        ),
        "development_contract_sha256": "d" * 64,
        "replayed_splits": ["val", "calib", "test_2019_2020_development"],
        "confirmation_period_read": False,
        "builtins_validated_by_suite_contract": [
            "Climatology", "DampedPersistence", "Persistence"
        ],
        "models": rows,
        "execution_attestation": execution,
    }
    value["receipt_self_sha256"] = sha256_json(value)
    return value


def _suite(root: Path, *, source_sha256: str | None = None) -> Path:
    entrypoint = root / REPLAY_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("# replay fixture\n", encoding="utf-8")
    suite = root / REPLAY_ALLOWED_CONFIRMATION_READ_PATHS[0]
    suite.parent.mkdir(parents=True, exist_ok=True)
    suite.write_text(json.dumps({
        "fixture": True,
        "training_device": "cpu",
        "numerical_runtime_sha256": sha256_json(numerical_runtime_contract()),
        "development_contract": {
            "source_sha256": source_sha256 or source_tree_hash(root),
        },
    }), encoding="utf-8")
    return suite


def test_stage27_runtime_gate_precedes_suite_validation_and_model_loading(
    tmp_path, monkeypatch,
):
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({
        "training_device": "cpu",
        "numerical_runtime_sha256": "0" * 64,
    }), encoding="utf-8")
    called = False

    def forbidden_validation(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("suite validation must not run after runtime mismatch")

    monkeypatch.setattr(
        "thermoroute.development_replay.validate_model_suite_document",
        forbidden_validation,
    )
    with pytest.raises(ModelSuiteError, match="runtime differs"):
        _load_suite(tmp_path, suite)
    assert called is False


def test_development_replay_receipt_is_source_suite_and_model_closed(tmp_path):
    suite = _suite(tmp_path)
    receipt = tmp_path / "receipt.json"
    write_replay_receipt(receipt, _receipt(tmp_path, suite))
    validated = validate_development_replay_receipt(
        receipt, root=tmp_path, suite_path=suite
    )
    assert len(validated["models"]) == 13
    io_guard = validated["execution_attestation"]["io_guard"]
    assert REPLAY_ALLOWED_CONFIRMATION_READ_PATHS[0] in io_guard["repo_read_paths"]
    assert io_guard["confirmation_read_policy"] == (
        _confirmation_read_policy_attestation()
    )
    assert validated["execution_attestation"]["fresh_pycache_policy"] == {
        "required": True,
        "controller_created_initially_empty_prefix": True,
        "repository_local_cache_allowed": False,
        "preexisting_repository_pyc_eligible": False,
        "prefix_lifetime": "one_isolated_child",
    }

    changed = json.loads(receipt.read_text(encoding="utf-8"))
    changed["models"][0]["status"] = "FAIL"
    changed["receipt_self_sha256"] = sha256_json({
        key: value for key, value in changed.items()
        if key != "receipt_self_sha256"
    })
    receipt.chmod(0o644)
    receipt.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ModelSuiteError, match="failed row"):
        validate_development_replay_receipt(
            receipt, root=tmp_path, suite_path=suite
        )


def test_development_replay_receipt_never_overwrites_different_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    write_replay_receipt(path, {"value": 1})
    write_replay_receipt(path, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_replay_receipt(path, {"value": 2})


def test_development_replay_receipt_guard_fails_before_create_only_link(tmp_path):
    path = tmp_path / "receipt.json"
    observed_staging: list[bytes] = []

    def reject() -> None:
        observed_staging.extend(
            candidate.read_bytes()
            for candidate in tmp_path.glob(f".{path.name}.*.staging")
        )
        raise RuntimeError("injected replay receipt drift")

    with pytest.raises(RuntimeError, match="replay receipt drift"):
        write_replay_receipt(
            path,
            {"value": 1},
            publication_guard=reject,
        )
    assert observed_staging == [b'{"value":1}\n']
    assert not path.exists()
    assert not list(tmp_path.glob(f".{path.name}.*.staging"))


@pytest.mark.parametrize("forged_path", FORBIDDEN_CONFIRMATION_READ_PATHS)
def test_development_replay_receipt_rejects_confirmation_read_evidence(
    tmp_path, forged_path,
):
    suite = _suite(tmp_path)
    receipt = tmp_path / "receipt.json"
    document = _receipt(tmp_path, suite)
    paths = [*document["execution_attestation"]["io_guard"]["repo_read_paths"],
             forged_path]
    paths = sorted(paths)
    document["execution_attestation"]["io_guard"].update({
        "repo_read_paths": paths,
        "repo_read_path_count": len(paths),
        "repo_read_paths_sha256": sha256_json(paths),
    })
    document["receipt_self_sha256"] = sha256_json({
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    })
    write_replay_receipt(receipt, document)
    with pytest.raises(ModelSuiteError, match="read-path evidence"):
        validate_development_replay_receipt(
            receipt, root=tmp_path, suite_path=suite
        )


def test_development_replay_receipt_rejects_forged_confirmation_allowlist(tmp_path):
    suite = _suite(tmp_path)
    receipt = tmp_path / "receipt.json"
    document = _receipt(tmp_path, suite)
    io_guard = document["execution_attestation"]["io_guard"]
    forged_path = "data_usgs/confirmatory_outcomes/water_temperature.parquet"
    paths = sorted([*io_guard["repo_read_paths"], forged_path])
    io_guard["confirmation_read_policy"]["allowed_exact_paths"].append(forged_path)
    io_guard.update({
        "repo_read_paths": paths,
        "repo_read_path_count": len(paths),
        "repo_read_paths_sha256": sha256_json(paths),
    })
    document["receipt_self_sha256"] = sha256_json({
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    })
    write_replay_receipt(receipt, document)
    with pytest.raises(ModelSuiteError, match="I/O guard attestation"):
        validate_development_replay_receipt(
            receipt, root=tmp_path, suite_path=suite
        )


def test_development_replay_receipt_rejects_suite_source_drift(tmp_path):
    suite = _suite(tmp_path, source_sha256="0" * 64)
    receipt = tmp_path / "receipt.json"
    write_replay_receipt(receipt, _receipt(tmp_path, suite))
    with pytest.raises(ModelSuiteError, match="frozen model suite/current source"):
        validate_development_replay_receipt(
            receipt, root=tmp_path, suite_path=suite
        )


def test_development_replay_receipt_rejects_replay_source_drift(tmp_path):
    suite = _suite(tmp_path)
    receipt = tmp_path / "receipt.json"
    document = _receipt(tmp_path, suite)
    document["source_tree_sha256"] = "0" * 64
    document["receipt_self_sha256"] = sha256_json({
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    })
    write_replay_receipt(receipt, document)
    with pytest.raises(ModelSuiteError, match="frozen model suite/current source"):
        validate_development_replay_receipt(
            receipt, root=tmp_path, suite_path=suite
        )


def test_member_seed_mapping_is_explicit_and_fails_on_ambiguity():
    assert _member_seeds(
        ["seed0", "seed2"], {"seeds": [0, 2]}, model_id="ThermoRoute"
    ) == {"seed0": 0, "seed2": 2}
    assert _member_seeds(
        [f"seed{seed}" for seed in STAGE9_ABLATION_SEEDS],
        {"seeds": list(STAGE9_ABLATION_SEEDS)},
        model_id="TR-noRouter",
    ) == {f"seed{seed}": seed for seed in STAGE9_ABLATION_SEEDS}
    with pytest.raises(ModelSuiteError, match="cannot be matched"):
        _member_seeds(
            ["member-a", "member-b"], {"seeds": [0, 1]}, model_id="fixture"
        )


def test_formal_ablation_replay_contract_requires_all_five_seeds():
    temporal = DEVELOPMENT_REPLAY_MODEL_CONTRACTS["temporal"]
    assert tuple(STAGE9_ABLATION_SEEDS) == tuple(range(5))
    assert all(
        temporal[model][:2] == ("thermoroute_bundle", 5)
        for model in MANDATORY_ABLATIONS
    )


@pytest.mark.parametrize("relative", FORBIDDEN_CONFIRMATION_READ_PATHS)
def test_development_replay_io_guard_blocks_all_confirmation_namespaces(
    tmp_path, relative,
):
    forbidden = tmp_path / relative
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"sealed")
    guard = DevelopmentReplayIOGuard(tmp_path)
    with guard, pytest.raises(PermissionError, match="confirmation path"):
        forbidden.read_bytes()
    assert _is_forbidden_confirmation_read(relative)


def test_development_replay_io_guard_allows_only_exact_model_suite(tmp_path):
    suite_relative = REPLAY_ALLOWED_CONFIRMATION_READ_PATHS[0]
    suite = tmp_path / suite_relative
    suite.parent.mkdir(parents=True)
    suite.write_bytes(b"frozen suite")
    guard = DevelopmentReplayIOGuard(tmp_path)
    with guard:
        assert suite.read_bytes() == b"frozen suite"
    assert not _is_forbidden_confirmation_read(suite_relative)
    assert guard.attestation()["repo_read_paths"] == [suite_relative]
    assert guard.attestation()["confirmation_read_policy"] == {
        **_confirmation_read_policy_attestation(),
        "denied_namespace_stems": list(
            REPLAY_FORBIDDEN_CONFIRMATION_NAMESPACE_STEMS
        ),
    }


def test_development_replay_io_guard_blocks_repository_writes(tmp_path):
    writable = tmp_path / "unexpected.txt"
    guard = DevelopmentReplayIOGuard(tmp_path)
    with guard, pytest.raises(PermissionError, match="may not write"):
        writable.write_text("mutation", encoding="utf-8")
    assert not writable.exists()


def test_guarded_replay_fingerprints_runtime_before_subprocess_ban(
    tmp_path, monkeypatch,
):
    runtime = {"host_numerical_identity": {"cpu": "fixture"}}
    events = []
    replay_completed = False
    policy_observations: list[bool] = []

    def publication_guard() -> None:
        policy_observations.append(replay_completed)

    expected_guard = publication_guard

    def fingerprint():
        events.append("runtime")
        return runtime

    def replay(*, root, suite_path, runtime_contract, publication_guard):
        nonlocal replay_completed
        events.append("replay")
        assert root == tmp_path.resolve()
        assert suite_path == (tmp_path / "suite.json").resolve()
        assert runtime_contract == runtime
        assert publication_guard is expected_guard
        with pytest.raises(PermissionError, match="child processes"):
            subprocess.run(["/usr/bin/true"], check=False)
        replay_completed = True
        return {"status": "PASS", "receipt_self_sha256": "discarded"}

    monkeypatch.setattr(
        "thermoroute.development_replay._assert_formal_invocation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "thermoroute.development_replay._execution_identity",
        lambda **_kwargs: {"format": "fixture"},
    )
    monkeypatch.setattr(
        "thermoroute.development_replay.numerical_runtime_contract",
        fingerprint,
    )
    monkeypatch.setattr(
        "thermoroute.development_replay.run_development_replay",
        replay,
    )

    document = run_guarded_development_replay(
        root=tmp_path,
        suite_path=tmp_path / "suite.json",
        receipt_path=tmp_path / "receipt.json",
        entrypoint_path=tmp_path / "scripts" / "27_verify_development_replay.py",
        publication_guard=publication_guard,
    )

    assert events == ["runtime", "replay"]
    assert policy_observations[-1] is True
    assert document["status"] == "PASS"
    assert document["execution_attestation"]["io_guard"][
        "subprocess_allowed"
    ] is False


def test_run_development_replay_guard_reaches_numeric_bundle_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha256 = "s" * 64
    artifact = tmp_path / "lightgbm" / "manifest.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    suite = {
        "development_contract": {"source_sha256": source_sha256},
        "actual_feature_order": ["WTEMP"],
        "cohorts": {
            "temporal": {"models": [{
                "model_id": "LightGBM",
                "executor": "lightgbm_bundle",
                "member_count": 5,
                "artifact": {"path": artifact.relative_to(tmp_path).as_posix()},
            }]},
            "external": {"models": []},
        },
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    reconstructed = False
    guard_observations: list[bool] = []

    def publication_guard() -> None:
        guard_observations.append(reconstructed)
        if reconstructed:
            raise RuntimeError("injected replay bundle policy drift")

    expected_guard = publication_guard

    def fake_load(_path, *, publication_guard):
        nonlocal reconstructed
        assert publication_guard is expected_guard
        reconstructed = True
        publication_guard()
        raise AssertionError("the publication guard must fail closed")

    monkeypatch.setattr(DEVELOPMENT_REPLAY, "LEARNED_TEMPORAL", ("LightGBM",))
    monkeypatch.setattr(DEVELOPMENT_REPLAY, "LEARNED_EXTERNAL", ())
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "_load_suite",
        lambda *_args, **_kwargs: suite,
    )
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "source_tree_hash",
        lambda _root: source_sha256,
    )
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "_prepare_temporal",
        lambda *_args, **_kwargs: (
            object(), object(), ("fixture-site",), object(), object(), object(), {},
        ),
    )
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "_prepare_external",
        lambda *_args, **_kwargs: (object(), object(), object(), {}),
    )
    monkeypatch.setattr(DEVELOPMENT_REPLAY, "load_lightgbm_bundle", fake_load)

    with pytest.raises(RuntimeError, match="replay bundle policy drift"):
        run_development_replay(
            root=tmp_path,
            suite_path=suite_path,
            runtime_contract={"fixture": True},
            publication_guard=publication_guard,
        )
    assert guard_observations[-1] is True


def test_run_development_replay_guard_rechecks_after_receipt_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha256 = "s" * 64
    suite = {
        "development_contract": {"source_sha256": source_sha256},
        "actual_feature_order": ["WTEMP"],
        "numerical_runtime_sha256": "r" * 64,
        "cohorts": {
            "temporal": {"models": []},
            "external": {"models": []},
        },
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    receipt_hashed = False
    guard_observations: list[bool] = []
    original_sha256_json = DEVELOPMENT_REPLAY.sha256_json

    def tracked_sha256_json(value):
        nonlocal receipt_hashed
        digest = original_sha256_json(value)
        if isinstance(value, dict) and value.get("format") == DEVELOPMENT_REPLAY_FORMAT:
            receipt_hashed = True
        return digest

    def publication_guard() -> None:
        guard_observations.append(receipt_hashed)
        if receipt_hashed:
            raise RuntimeError("injected replay receipt policy drift")

    monkeypatch.setattr(DEVELOPMENT_REPLAY, "LEARNED_TEMPORAL", ())
    monkeypatch.setattr(DEVELOPMENT_REPLAY, "LEARNED_EXTERNAL", ())
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "_load_suite",
        lambda *_args, **_kwargs: suite,
    )
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "source_tree_hash",
        lambda _root: source_sha256,
    )
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "_prepare_temporal",
        lambda *_args, **_kwargs: (
            object(), object(), (), object(), object(), object(), {},
        ),
    )
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "_prepare_external",
        lambda *_args, **_kwargs: (object(), object(), object(), {}),
    )
    monkeypatch.setattr(DEVELOPMENT_REPLAY, "sha256_json", tracked_sha256_json)

    with pytest.raises(RuntimeError, match="replay receipt policy drift"):
        run_development_replay(
            root=tmp_path,
            suite_path=suite_path,
            runtime_contract={"fixture": True},
            publication_guard=publication_guard,
        )
    assert guard_observations[0] is False
    assert guard_observations[-1] is True


def test_fresh_verify_guard_rechecks_after_exact_replay_equivalence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = {"status": "PASS", "receipt_self_sha256": "fixture"}
    replay_completed = False
    guard_observations: list[bool] = []

    def publication_guard() -> None:
        guard_observations.append(replay_completed)
        if replay_completed:
            raise RuntimeError("injected fresh-verify policy drift")

    expected_guard = publication_guard

    def fake_replay(*, publication_guard, **_kwargs):
        nonlocal replay_completed
        assert publication_guard is expected_guard
        replay_completed = True
        return existing

    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "validate_development_replay_receipt",
        lambda *_args, **_kwargs: existing,
    )
    monkeypatch.setattr(
        DEVELOPMENT_REPLAY,
        "run_guarded_development_replay",
        fake_replay,
    )

    with pytest.raises(RuntimeError, match="fresh-verify policy drift"):
        fresh_verify_development_replay_receipt(
            tmp_path / "receipt.json",
            root=tmp_path,
            suite_path=tmp_path / "suite.json",
            entrypoint_path=tmp_path / "scripts" / "27_verify_development_replay.py",
            publication_guard=publication_guard,
        )
    assert guard_observations[-1] is True


def test_formal_replay_rejects_missing_or_repository_local_pycache(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    local = root / "__pycache__"
    local.mkdir()
    with pytest.raises(ModelSuiteError, match="fresh pycache"):
        _validate_formal_pycache_prefix(root, None)
    with pytest.raises(ModelSuiteError, match="repository-local"):
        _validate_formal_pycache_prefix(root, local)

    fresh = tmp_path / "fresh-pycache"
    fresh.mkdir()
    assert _validate_formal_pycache_prefix(root, fresh) == fresh.resolve()


def test_stage27_uses_distinct_initially_empty_pycache_for_each_child(
    tmp_path, monkeypatch,
):
    observed: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, **kwargs):
        prefix_argument = command[3]
        assert prefix_argument.startswith("pycache_prefix=")
        prefix = Path(prefix_argument.split("=", 1)[1])
        assert prefix.is_dir()
        assert list(prefix.iterdir()) == []
        observed.append((list(command), dict(kwargs["env"])))
        return subprocess.CompletedProcess(command, 0, "{}\n", "")

    monkeypatch.setattr(STAGE27.subprocess, "run", fake_run)
    suite = tmp_path / "suite.json"
    receipt = tmp_path / "receipt.json"
    STAGE27._run_isolated_worker(suite=suite, receipt=receipt, check=False)
    STAGE27._run_isolated_worker(suite=suite, receipt=receipt, check=True)

    assert len(observed) == 2
    prefixes = [command[3] for command, _environment in observed]
    assert prefixes[0] != prefixes[1]
    for command, environment in observed:
        assert command[1:3] == ["-I", "-X"]
        assert "-B" not in command
        assert "--_isolated-worker" in command
        assert "PYTHONPATH" not in environment
        assert "PYTHONPYCACHEPREFIX" not in environment
