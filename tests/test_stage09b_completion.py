from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.development_controls_gate import (  # noqa: E402
    DevelopmentControlsGateError,
    build_stage09b_completion_receipt,
    expected_stage09b_members,
    publish_stage09b_completion_receipt,
    validate_stage09b_completion_receipt,
)
import thermoroute.development_controls_gate as CONTROLS_GATE  # noqa: E402
from thermoroute.development_controls import (  # noqa: E402
    CanonicalWindowContract,
    normalise_window_registry,
    window_registry_digest,
)
from thermoroute.repro import (  # noqa: E402
    RUN_SCHEMA_VERSION,
    RunIdentity,
    seal_artifact,
    sha256_file,
    sha256_json,
    sidecar_path,
    source_tree_hash,
)


def _load_script(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


DC = _load_script("scripts/09b_development_controls.py", "stage09b_receipt_fixture")
STAGE24 = _load_script("scripts/24_freeze_model_suite.py", "stage24_controls_fixture")
VERIFY_RELEASE = _load_script("scripts/verify_release.py", "stage09b_release_fixture")


def _binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def test_stage09b_binding_rejects_hardlinked_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bound bytes\n")
    binding = _binding(tmp_path, artifact)
    os.link(artifact, tmp_path / "duplicate.bin")

    with pytest.raises(DevelopmentControlsGateError, match="canonical path"):
        CONTROLS_GATE._validated_binding(
            tmp_path, binding, label="fixture artifact"
        )


def _rehash(document: dict[str, Any]) -> None:
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    document["receipt_self_sha256"] = sha256_json(stable)


def _reseal_existing_artifact(path: Path) -> None:
    metadata = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
    seal_artifact(
        path,
        RunIdentity(**metadata["run"]),
        kind=metadata["kind"],
        schema=metadata["content_schema"],
        parents=metadata["parents"],
        extra=metadata["extra"],
    )


def _formal_policy() -> dict[str, Any]:
    return {
        "thread_environment": {
            name: "1" for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
        "cublas_workspace_config": ":4096:8",
        "python_hash_environment_declaration": "0",
        "python_hash_randomization_enabled": True,
        "python_hash_policy": (
            "canonical-sort-identity-collections-independent-of-hash-secret"
        ),
        "required": {
            "threads": 1,
            "cublas_workspace_config": ":4096:8",
            "python_hash_policy": (
                "canonical-sort-identity-collections-independent-of-hash-secret"
            ),
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


def _prediction(arm, seed: int, sites: list[str]) -> pd.DataFrame:
    split_dates = (
        ("val", pd.Timestamp("2016-01-02")),
        ("calib", pd.Timestamp("2018-01-02")),
        ("test", pd.Timestamp("2019-01-02")),
    )
    records = [
        (site, split, issue, horizon)
        for site in sites
        for split, issue in split_dates
        for horizon in DC.C.HORIZONS
    ]
    site = np.asarray([row[0] for row in records])
    split = np.asarray([row[1] for row in records])
    issue = pd.DatetimeIndex([row[2] for row in records])
    horizon = np.asarray([row[3] for row in records], dtype=int)
    target = issue + pd.to_timedelta(horizon, unit="D")
    truth = np.asarray(
        [float(index % 120) + float(value) / 10 for index, value in enumerate(horizon)],
        dtype=np.float32,
    )
    return DC.R.make_pred_frame(
        model=np.full(len(records), arm.arm_id),
        scope=np.full(len(records), DC.DEVELOPMENT_SCOPE),
        feature_set=np.full(len(records), arm.feature_set),
        seed=np.full(len(records), seed),
        site_id=site,
        horizon=horizon,
        split=split,
        issue_date=issue,
        target_date=target,
        y_true=truth,
        y_pred=truth + np.float32(seed / 100 + 0.1),
        q05=truth - 0.5,
        q50=truth + 0.1,
        q95=truth + 0.5,
        p_exceed=np.full(len(records), 0.25),
    )


@pytest.mark.parametrize(
    "attack",
    (
        "seed_bool", "horizon_float", "site_integer", "string_date",
        "timezone", "intraday", "y_true_bool", "q50_integer",
    ),
)
def test_gate_rejects_raw_prediction_coercion_aliases(attack: str) -> None:
    arm = DC.declared_arms()[0]
    frame = _prediction(arm, 0, ["12345678"])
    if attack == "seed_bool":
        frame["seed"] = False
    elif attack == "horizon_float":
        frame["horizon"] = frame["horizon"].astype("float64")
    elif attack == "site_integer":
        frame["site_id"] = 12_345_678
    elif attack == "string_date":
        frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")
    elif attack == "timezone":
        frame["issue_date"] = frame["issue_date"].dt.tz_localize("UTC")
        frame["target_date"] = frame["target_date"].dt.tz_localize("UTC")
    elif attack == "intraday":
        frame["issue_date"] += pd.Timedelta(hours=1)
        frame["target_date"] += pd.Timedelta(hours=1)
    elif attack == "y_true_bool":
        frame["y_true"] = False
    else:
        frame["q50"] = 1
    with pytest.raises(DevelopmentControlsGateError, match="Stage-09b member"):
        CONTROLS_GATE._assert_raw_prediction_physical_schema(frame)


@pytest.mark.parametrize("attack", ("string_date", "y_true_bool", "q50_integer"))
def test_gate_rejects_noncanonical_prediction_arrow_types(
    tmp_path: Path, attack: str,
) -> None:
    frame = _prediction(DC.declared_arms()[0], 0, ["12345678"])
    if attack == "string_date":
        frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")
    elif attack == "y_true_bool":
        frame["y_true"] = False
    else:
        frame["q50"] = 1
    path = tmp_path / "attacked.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(DevelopmentControlsGateError, match="Arrow type"):
        CONTROLS_GATE._assert_prediction_arrow_schema(path)


def _fixture_window_contract(sites: list[str]) -> CanonicalWindowContract:
    arm = DC.declared_arms()[0]
    frame = _prediction(arm, arm.seeds[0], sites)
    registry = normalise_window_registry(
        frame[["split", "site_id", "horizon", "issue_date", "target_date", "y_true"]]
    )
    return CanonicalWindowContract(
        registry=registry,
        train_examples=3_073,
        stations=tuple(sorted(sites)),
        registry_sha256=window_registry_digest(registry),
        train_registry_sha256="6" * 64,
    )


def _fixture_bridge_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "site_no": ["00000000"],
        "DATE": [pd.Timestamp("2018-01-01")],
        **{name: [1.0] for name in DC.FULL_VARIABLES if name != "WTEMP"},
    })


def _fixture_bridge_report(*_args, **_kwargs) -> dict[str, Any]:
    return {
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "interval": ["2018-01-01", "2020-12-31"],
        "outcome_values_requested_or_read": False,
    }


def _build_fixture(root: Path) -> dict[str, Any]:
    (root / "src").mkdir(parents=True)
    (root / "src" / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    data = root / "data_usgs"
    data.mkdir()
    panel = data / "panel_usgs_120v2.parquet"
    pd.DataFrame({"DATE": pd.Series(dtype="datetime64[ns]")}).to_parquet(
        panel, index=False
    )
    sites = [f"{index:08d}" for index in range(120)]
    registry = data / "station_registry_v1.csv"
    pd.DataFrame({
        "site_no": sites,
        "legacy_site_id": [f"n{index:03d}" for index in range(120)],
    }).to_csv(registry, index=False)
    frozen_spec = data / "frozen_panel_v1.json"
    frozen_spec.write_text(json.dumps({
        "schema_version": 1,
        "panel": {
            "path": panel.name,
            "sha256": sha256_file(panel),
            "date_start": "2006-01-01",
            "date_end": "2020-12-31",
            "station_count": 120,
        },
        "station_registry": {
            "path": registry.name,
            "sha256": sha256_file(registry),
            "station_count": 120,
        },
    }), encoding="utf-8")
    bridge_data = data / "development_predictor_bridge_v1"
    bridge_data.mkdir()
    report = bridge_data / "bridge_report_v1.json"
    request_map = bridge_data / "source_request_map_v1.json"
    frozen_predictors = bridge_data / "frozen_panel_predictors_2018_2020.parquet"
    refreshed_predictors = bridge_data / "refreshed_predictors_2018_2020.parquet"
    bridge_frame = pd.DataFrame({
        "site_no": [sites[0]],
        "DATE": [pd.Timestamp("2018-01-01")],
        **{name: [1.0] for name in DC.FULL_VARIABLES if name != "WTEMP"},
    })
    bridge_frame.to_parquet(frozen_predictors, index=False)
    bridge_frame.to_parquet(refreshed_predictors, index=False)
    bridge_report = {
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "interval": ["2018-01-01", "2020-12-31"],
        "outcome_values_requested_or_read": False,
    }
    report.write_text(json.dumps(bridge_report), encoding="utf-8")
    request_map.write_text(json.dumps({
        "format": "thermoroute.development-predictor-bridge-requests.v1",
        "outcome_values_requested_or_read": False,
        "interval": bridge_report["interval"],
        "request_count": 0,
        "requests": [],
        "gridmet_provider_contract": {"fixture": True},
    }), encoding="utf-8")
    raw_root = data / "raw_snapshots" / "development-predictor-bridge-v1"
    snapshots: dict[str, Path] = {}
    for label, directory in (
        ("daymet", "daymet-v1"),
        ("gridmet", "gridmet-v1"),
        ("gridmet_schema", "gridmet-schema-v1"),
    ):
        snapshot_dir = raw_root / directory
        response_dir = snapshot_dir / "provider" / ("a" * 64)
        response_dir.mkdir(parents=True)
        response = response_dir / "response.bin"
        response.write_bytes(f"{label} predictor response fixture".encode())
        metadata = response_dir / "metadata.json"
        metadata.write_text("{}\n", encoding="utf-8")
        snapshot = snapshot_dir / "snapshot_index_v2.json"
        metadata_bytes = metadata.read_bytes()
        snapshot.write_text(json.dumps({
            "schema_version": 2,
            "snapshot_count": 1,
            "records": [{
            "provider": "provider",
            "request_sha256": "a" * 64,
            "byte_count": response.stat().st_size,
            "metadata_byte_count": len(metadata_bytes),
            "metadata_sha256": sha256_file(metadata),
            "metadata_path": metadata.relative_to(snapshot_dir).as_posix(),
            "request": {"provider": "provider", "url": "https://example.com"},
            "response_path": response.relative_to(snapshot_dir).as_posix(),
            "response_sha256": sha256_file(response),
            "retrieved_at_utc": "2026-07-22T00:00:00+00:00",
        }]}), encoding="utf-8")
        snapshots[label] = snapshot
    bridge = data / "development_predictor_bridge_v1.json"
    bridge.write_text(json.dumps({
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": "b" * 64,
        "panel": _binding(root, panel),
        "registry": _binding(root, registry),
        "report": _binding(root, report),
        "request_map": _binding(root, request_map),
        "normalized": {
            "frozen": _binding(root, frozen_predictors),
            "refreshed": _binding(root, refreshed_predictors),
        },
        "raw_snapshot_indexes": {
            label: _binding(root, path) for label, path in snapshots.items()
        },
    }), encoding="utf-8")
    contract = _fixture_window_contract(sites)
    arms = DC.declared_arms()
    config = {
        "stage": "09b_development_controls",
        "format": DC.FINAL_FORMAT,
        "execution_role": "prelabel_relative_to_unopened_post_2020_confirmation",
        "evidence_role": "development_only_exploratory",
        "development_disclosure": DC.DEVELOPMENT_DISCLOSURE,
        "panel_date_range": ["2006-01-01", "2020-12-31"],
        "development_evaluation_interval": list(DC.C.SPLIT.test),
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "training_device": "cpu",
        "variables": list(DC.FULL_VARIABLES),
        "context_length": DC.C.CONTEXT_LENGTH,
        "horizons": list(DC.C.HORIZONS),
        "time_split": {
            key: list(interval) for key, interval in DC.C.SPLIT.as_dict().items()
        },
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "train_config": asdict(DC.TRAIN_CONFIG),
        "arms": [asdict(arm) for arm in arms],
        "expected_member_registry": [
            list(member) for member in DC.expected_member_registry(arms)
        ],
        "parameter_counts": DC.assert_parameter_budgets(arms, n_stations=120),
        "architecture_templates": {
            arm.arm_id: DC.architecture_template(arm, n_stations=120)
            for arm in arms
        },
        "parameter_match_tolerance_fraction": 0.02,
        "architecture_candidates_per_arm": 1,
        "historical_tuning_budget_equalized": False,
        "development_predictor_bridge": _binding(root, bridge),
        "formal_numerical_policy": _formal_policy(),
        "eval_batch_size": 2,
    }
    identity_parts = {
        "schema_version": RUN_SCHEMA_VERSION,
        "panel_sha256": sha256_file(panel),
        "registry_sha256": sha256_file(registry),
        "config_sha256": sha256_json(config),
        "source_sha256": source_tree_hash(root),
        "runtime_sha256": "5" * 64,
    }
    identity = RunIdentity(
        run_id=sha256_json(identity_parts)[:20],
        **identity_parts,
    )
    run_dir = root / "outputs" / "runs" / "09b_development_controls" / identity.run_id
    run_dir.mkdir(parents=True)
    manifest = run_dir / "run.json"
    manifest.write_text(json.dumps({
        "schema_version": RUN_SCHEMA_VERSION,
        "identity": identity.as_dict(),
        "resolved_config": config,
        "created_utc": "2026-07-22T12:00:00+00:00",
        "environment": {},
        "git": {},
        "provenance": {
            "development_only": True,
            "post_2020_outcomes_requested_or_read": False,
            "suite_pointer_written": False,
            "training_device": "cpu",
        },
    }), encoding="utf-8")
    parents = {
        "frozen_panel": identity.panel_sha256,
        "frozen_station_registry": identity.registry_sha256,
        "development_predictor_bridge": sha256_file(bridge),
    }
    members: dict[tuple[str, int], Path] = {}
    for arm in arms:
        parameters = DC.parameter_count(arm, n_stations=120)
        for seed in arm.seeds:
            path = run_dir / "arm_predictions" / arm.arm_id / f"seed{seed}.parquet"
            checkpoint = run_dir / "checkpoints" / arm.arm_id / f"seed{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(
                f"fixture checkpoint {arm.arm_id}/seed{seed}\n".encode()
            )
            checkpoint_metadata = checkpoint.with_name(
                checkpoint.name + ".meta.json"
            )
            checkpoint_metadata.write_text("{}\n", encoding="utf-8")
            frame = _prediction(arm, seed, sites)
            member_parents = {
                **parents,
                "training_checkpoint": sha256_file(checkpoint),
                "training_checkpoint_sidecar": sha256_file(checkpoint_metadata),
            }
            DC.write_arm_prediction(
                frame,
                path,
                identity=identity,
                arm=arm,
                seed=seed,
                parameters=parameters,
                n_stations=120,
                eval_batch_size=2,
                parents=member_parents,
                training_summary={
                    "best_validation_metric": 0.25,
                    "selected_epoch": 2,
                    "checkpoint_final_epoch": 4,
                },
            )
            members[(arm.arm_id, seed)] = path
    frames = {
        member: pd.read_parquet(path) for member, path in members.items()
    }
    summaries = DC.recompute_metric_summary(frames).to_dict(orient="records")
    audit = DC.validate_complete_prediction_matrix(
        frames, arms, allowed_sites=set(sites)
    )
    budget = DC.architecture_budget_rows(
        arms, n_stations=120, train_examples=3_073
    )
    predictions, budget_path, summary_path, report, semantic_audit = (
        DC.publish_final_artifacts(
        run_dir=run_dir,
        identity=identity,
        arms=arms,
        member_paths=members,
        member_parents=parents,
        audit=audit,
        budget=budget,
        summaries=summaries,
        train_examples=contract.train_examples,
        canonical_registry_sha256=contract.registry_sha256,
        canonical_train_registry_sha256=contract.train_registry_sha256,
    ))
    receipt_path = root / "outputs" / "models" / "route_a_stage09b_completion.json"
    receipt = build_stage09b_completion_receipt(
        root=root,
        run_id=identity.run_id,
        run_manifest=manifest,
        frozen_panel_spec=frozen_spec,
        panel=panel,
        registry=registry,
        predictor_bridge=bridge,
        member_paths=members,
        predictions=predictions,
        architecture_budget=budget_path,
        metric_summary=summary_path,
        report=report,
        semantic_audit=semantic_audit,
        matrix_audit=asdict(audit),
    )
    publish_stage09b_completion_receipt(receipt_path, receipt, root=root)
    return {
        "root": root,
        "receipt": receipt_path,
        "report": report,
        "budget": budget_path,
        "members": members,
        "identity": identity,
    }


@pytest.fixture(scope="module")
def stage09b_base(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("stage09b-gate-base")
    sites = [f"{index:08d}" for index in range(120)]
    originals = (
        CONTROLS_GATE.rebuild_canonical_window_contract,
        CONTROLS_GATE.frozen_bridge_slice,
        CONTROLS_GATE.compare_predictor_bridge,
        CONTROLS_GATE.replay_predictor_bridge_offline,
        CONTROLS_GATE._rebuild_member_replay_inputs,
        CONTROLS_GATE.DS.build_windows,
        CONTROLS_GATE._replay_member_best_state,
    )
    CONTROLS_GATE.rebuild_canonical_window_contract = (
        lambda **_kwargs: _fixture_window_contract(sites)
    )
    CONTROLS_GATE.frozen_bridge_slice = lambda *_args, **_kwargs: (
        _fixture_bridge_frame()
    )
    CONTROLS_GATE.compare_predictor_bridge = _fixture_bridge_report
    CONTROLS_GATE.replay_predictor_bridge_offline = lambda **_kwargs: (
        _fixture_bridge_frame()
    )
    CONTROLS_GATE._rebuild_member_replay_inputs = lambda **_kwargs: (
        pd.DataFrame(), object(), object(), {site: 1.0 for site in sites}
    )
    CONTROLS_GATE.DS.build_windows = lambda *_args, **_kwargs: object()
    CONTROLS_GATE._replay_member_best_state = (
        lambda *, arm, seed, contract, **_kwargs: (
            DC.normalise_prediction_frame(
                _prediction(arm, seed, sites), arm=arm, seed=seed,
                allowed_sites=set(sites), canonical_registry=contract.registry,
            ),
            {
                "best_validation_metric": 0.25,
                "selected_epoch": 2,
                "checkpoint_final_epoch": 4,
            },
        )
    )
    try:
        _build_fixture(root)
    finally:
        (
            CONTROLS_GATE.rebuild_canonical_window_contract,
            CONTROLS_GATE.frozen_bridge_slice,
            CONTROLS_GATE.compare_predictor_bridge,
            CONTROLS_GATE.replay_predictor_bridge_offline,
            CONTROLS_GATE._rebuild_member_replay_inputs,
            CONTROLS_GATE.DS.build_windows,
            CONTROLS_GATE._replay_member_best_state,
        ) = originals
    return root


@pytest.fixture(autouse=True)
def _scoped_data_replay_stubs(monkeypatch) -> None:
    sites = [f"{index:08d}" for index in range(120)]
    monkeypatch.setattr(
        CONTROLS_GATE,
        "rebuild_canonical_window_contract",
        lambda **_kwargs: _fixture_window_contract(sites),
    )
    monkeypatch.setattr(
        CONTROLS_GATE,
        "frozen_bridge_slice",
        lambda *_args, **_kwargs: _fixture_bridge_frame(),
    )
    monkeypatch.setattr(
        CONTROLS_GATE, "compare_predictor_bridge", _fixture_bridge_report
    )
    monkeypatch.setattr(
        CONTROLS_GATE, "replay_predictor_bridge_offline",
        lambda **_kwargs: _fixture_bridge_frame(),
    )
    monkeypatch.setattr(
        CONTROLS_GATE, "_rebuild_member_replay_inputs",
        lambda **_kwargs: (
            pd.DataFrame(), object(), object(), {site: 1.0 for site in sites}
        ),
    )
    monkeypatch.setattr(
        CONTROLS_GATE.DS, "build_windows", lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        CONTROLS_GATE,
        "_replay_member_best_state",
        lambda *, arm, seed, contract, **_kwargs: (
            DC.normalise_prediction_frame(
                _prediction(arm, seed, sites), arm=arm, seed=seed,
                allowed_sites=set(sites), canonical_registry=contract.registry,
            ),
            {
                "best_validation_metric": 0.25,
                "selected_epoch": 2,
                "checkpoint_final_epoch": 4,
            },
        ),
    )


@pytest.fixture
def fixture(stage09b_base: Path, tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "repo"
    shutil.copytree(stage09b_base, root)
    receipt = root / "outputs" / "models" / "route_a_stage09b_completion.json"
    document = json.loads(receipt.read_text(encoding="utf-8"))
    run_id = document["run_id"]
    members = {
        (entry["arm_id"], int(entry["seed"])): root / entry["prediction"]["path"]
        for entry in document["member_registry"]
    }
    return {
        "root": root,
        "receipt": receipt,
        "document": document,
        "report": root / document["artifacts"]["report"]["path"],
        "budget": root / document["artifacts"]["architecture_budget"]["path"],
        "members": members,
        "run_id": run_id,
    }


def test_stage09b_receipt_requires_exact_31_member_closure(fixture) -> None:
    receipt = validate_stage09b_completion_receipt(
        fixture["receipt"], root=fixture["root"]
    )
    assert len(expected_stage09b_members()) == 31
    assert receipt["matrix_audit"]["expected_members"] == 31
    assert receipt["matrix_audit"]["prediction_rows"] == (
        31 * receipt["matrix_audit"]["common_forecast_keys"]
    )
    assert receipt["best_model_state_prediction_replay_verified"] is True
    assert receipt["training_replay_verified"] is False
    assert all(
        set(entry) == {
            "arm_id", "seed", "checkpoint", "checkpoint_sidecar",
            "prediction", "prediction_sidecar",
        }
        for entry in receipt["member_registry"]
    )
    loaded, binding = STAGE24._load_verified_stage09b(
        fixture["receipt"], root=fixture["root"]
    )
    assert loaded["run_id"] == fixture["run_id"]
    assert binding == _binding(fixture["root"], fixture["receipt"])

    incomplete = json.loads(json.dumps(receipt))
    incomplete["member_registry"].pop()
    _rehash(incomplete)
    with pytest.raises(DevelopmentControlsGateError, match="31 members"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=incomplete
        )
    fixture["receipt"].unlink()
    with pytest.raises(STAGE24.ModelSuiteError, match="gate failed"):
        STAGE24._load_verified_stage09b(
            fixture["receipt"], root=fixture["root"]
        )


def test_failure_before_last_write_never_publishes_receipt(fixture) -> None:
    candidate = fixture["document"]
    fixture["receipt"].unlink()
    fixture["report"].write_text("injected report failure\n", encoding="utf-8")
    with pytest.raises(DevelopmentControlsGateError, match="checksum|canonical path"):
        publish_stage09b_completion_receipt(
            fixture["receipt"], candidate, root=fixture["root"]
        )
    assert not fixture["receipt"].exists()


def test_semantically_forged_budget_fails_closed(fixture) -> None:
    # A self-consistent receipt cannot bless a semantically changed architecture
    # budget, even if its file and sidecar hashes are recomputed.
    budget = pd.read_csv(fixture["budget"])
    budget.loc[0, "trainable_parameters"] += 1
    budget.to_csv(fixture["budget"], index=False)
    metadata = json.loads(sidecar_path(fixture["budget"]).read_text(encoding="utf-8"))
    seal_artifact(
        fixture["budget"],
        RunIdentity(**metadata["run"]),
        kind=metadata["kind"],
        schema=metadata["content_schema"],
        parents=metadata["parents"],
        extra=metadata["extra"],
    )
    forged = fixture["document"]
    forged["artifacts"]["architecture_budget"] = _binding(
        fixture["root"], fixture["budget"]
    )
    forged["artifacts"]["architecture_budget_sidecar"] = _binding(
        fixture["root"], sidecar_path(fixture["budget"])
    )
    _rehash(forged)
    with pytest.raises(DevelopmentControlsGateError, match="budget"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
        )


def test_self_consistent_member_with_changed_common_key_fails_closed(fixture) -> None:
    forged = fixture["document"]
    first_entry = forged["member_registry"][0]
    member = fixture["root"] / first_entry["prediction"]["path"]
    metadata_path = sidecar_path(member)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(member)
    frame.loc[0, "issue_date"] += pd.Timedelta(days=1)
    frame.loc[0, "target_date"] += pd.Timedelta(days=1)
    frame.to_parquet(member, index=False)
    seal_artifact(
        member,
        RunIdentity(**metadata["run"]),
        kind=metadata["kind"],
        schema=metadata["content_schema"],
        parents=metadata["parents"],
        extra=metadata["extra"],
    )
    first_entry["prediction"] = _binding(fixture["root"], member)
    first_entry["prediction_sidecar"] = _binding(fixture["root"], metadata_path)
    _rehash(forged)
    with pytest.raises(DevelopmentControlsGateError, match="canonical window registry"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
        )


@pytest.mark.parametrize(
    ("column", "operation"),
    (
        ("y_pred", lambda value: value + 0.05),
        ("q05", lambda value: value - 0.05),
        ("p_exceed", lambda _value: 0.75),
    ),
)
def test_self_consistent_member_scientific_output_forgery_fails_closed(
    fixture, column, operation
) -> None:
    forged = fixture["document"]
    entry = forged["member_registry"][0]
    member = fixture["root"] / entry["prediction"]["path"]
    frame = pd.read_parquet(member)
    frame[column] = frame[column].astype("float64")
    frame.loc[0, column] = operation(frame.loc[0, column])
    frame.to_parquet(member, index=False)
    _reseal_existing_artifact(member)
    entry["prediction"] = _binding(fixture["root"], member)
    entry["prediction_sidecar"] = _binding(
        fixture["root"], sidecar_path(member)
    )
    _rehash(forged)
    with pytest.raises(
        DevelopmentControlsGateError,
        match=(
            "checkpoint best_model_state|final artifact closure|summary|combined|semantic"
        ),
    ):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
        )


def test_rehashed_combined_prediction_forgery_fails_full_column_equality(fixture) -> None:
    forged = fixture["document"]
    combined = fixture["root"] / forged["artifacts"]["predictions"]["path"]
    frame = pd.read_parquet(combined)
    frame[["q50", "q95"]] = frame[["q50", "q95"]].astype("float64")
    frame.loc[0, "q50"] += 0.05
    frame.loc[0, "q95"] += 0.05
    frame.to_parquet(combined, index=False)
    _reseal_existing_artifact(combined)
    forged["artifacts"]["predictions"] = _binding(fixture["root"], combined)
    forged["artifacts"]["prediction_sidecar"] = _binding(
        fixture["root"], sidecar_path(combined)
    )
    _rehash(forged)
    with pytest.raises(DevelopmentControlsGateError, match="combined predictions differ"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
        )


def test_rehashed_metric_summary_forgery_is_recomputed_and_rejected(fixture) -> None:
    forged = fixture["document"]
    summary = fixture["root"] / forged["artifacts"]["metric_summary"]["path"]
    table = pd.read_csv(summary)
    table.loc[0, "median_station_rmse_c"] += 0.1
    table.to_csv(summary, index=False)
    _reseal_existing_artifact(summary)
    forged["artifacts"]["metric_summary"] = _binding(fixture["root"], summary)
    forged["artifacts"]["metric_summary_sidecar"] = _binding(
        fixture["root"], sidecar_path(summary)
    )
    _rehash(forged)
    with pytest.raises(DevelopmentControlsGateError, match="not prediction-derived"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
        )


def test_rehashed_paired_effect_forgery_is_recomputed_and_rejected(fixture) -> None:
    forged = fixture["document"]
    semantic = fixture["root"] / forged["artifacts"]["semantic_audit"]["path"]
    document = json.loads(semantic.read_text(encoding="utf-8"))
    paired = document["scientific_summary"]["paired_descriptive_effects"]
    paired["records"][0]["median_paired_station_rmse_difference_c"] += 0.1
    paired["records_sha256"] = hashlib.sha256(json.dumps(
        paired["records"], sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    document.pop("semantic_audit_self_sha256")
    document["semantic_audit_self_sha256"] = sha256_json(document)
    semantic.write_text(json.dumps(document), encoding="utf-8")
    _reseal_existing_artifact(semantic)
    forged["artifacts"]["semantic_audit"] = _binding(
        fixture["root"], semantic
    )
    forged["artifacts"]["semantic_audit_sidecar"] = _binding(
        fixture["root"], sidecar_path(semantic)
    )
    _rehash(forged)
    with pytest.raises(DevelopmentControlsGateError, match="semantic audit"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
        )


def test_formal_numerical_policy_rejects_each_previously_unchecked_field() -> None:
    attacks = (
        ("python_hash_environment_declaration", "ATTACK"),
        ("python_hash_randomization_enabled", False),
        ("required.cublas_workspace_config", "ATTACK"),
        ("required.float32_matmul_precision", "ATTACK"),
        ("torch.cudnn_deterministic", False),
        ("torch.cudnn_benchmark", True),
    )
    for dotted, value in attacks:
        policy = _formal_policy()
        target = policy
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        with pytest.raises(DevelopmentControlsGateError, match="numerical policy"):
            CONTROLS_GATE._validate_formal_policy(policy)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("best_validation_metric", True),
        ("best_validation_metric", float("nan")),
        ("best_validation_metric", -0.01),
        ("selected_epoch", True),
        ("checkpoint_final_epoch", None),
    ),
)
def test_member_training_summary_attacks_fail_closed(fixture, field, value) -> None:
    forged = fixture["document"]
    entry = forged["member_registry"][0]
    metadata_path = fixture["root"] / entry["prediction_sidecar"]["path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["extra"]["training_summary"][field] = value
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    entry["prediction_sidecar"] = _binding(fixture["root"], metadata_path)
    _rehash(forged)
    with pytest.raises(DevelopmentControlsGateError, match="training summary"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
        )


def test_eval_batch_size_and_checkpoint_member_path_are_exact_identity(fixture) -> None:
    changed_batch = fixture["document"]
    changed_batch["formal_configuration"]["eval_batch_size"] += 1
    _rehash(changed_batch)
    with pytest.raises(DevelopmentControlsGateError, match="identity|configuration|run"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=changed_batch
        )

    forged_path = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    forged_path["member_registry"][0]["checkpoint"] = (
        forged_path["member_registry"][1]["checkpoint"]
    )
    _rehash(forged_path)
    with pytest.raises(DevelopmentControlsGateError, match="path registry"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged_path
        )


def test_outer_release_rejects_one_site_or_noncanonical_window_inputs(
    tmp_path: Path,
) -> None:
    panel = tmp_path / "panel.parquet"
    registry = tmp_path / "registry.csv"
    spec = tmp_path / "spec.json"
    pd.DataFrame({
        "DATE": [pd.Timestamp("2019-01-01")],
        "site_id": ["legacy"],
        "WTEMP": [1.0],
    }).to_parquet(panel, index=False)
    pd.DataFrame({
        "site_no": ["01234567"], "legacy_site_id": ["legacy"],
    }).to_csv(registry, index=False)
    spec.write_text(json.dumps({
        "schema_version": 1,
        "evidence_role": "development_exploratory",
        "panel": {
            "path": panel.name, "sha256": sha256_file(panel),
            "date_start": "2019-01-01", "date_end": "2019-01-01",
            "row_count": 1, "station_count": 1,
        },
        "station_registry": {
            "path": registry.name, "sha256": sha256_file(registry),
            "station_count": 1,
        },
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="specification changed"):
        VERIFY_RELEASE._stage09b_rebuild_canonical_windows(panel, registry, spec)


def test_stage09b_gate_rejects_unbound_checkpoint_transaction_temp(fixture) -> None:
    run_dir = (
        fixture["root"] / "outputs" / "runs" / "09b_development_controls"
        / fixture["run_id"]
    )
    orphan = run_dir / "checkpoints" / ".seed0.pt.crashed.tmp"
    orphan.write_bytes(b"complete-but-unpublished checkpoint temp")
    with pytest.raises(DevelopmentControlsGateError, match="transaction temp"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"]
        )


def test_post_validation_artifact_mutation_is_rejected_on_revalidation(fixture) -> None:
    STAGE24._load_verified_stage09b(
        fixture["receipt"], root=fixture["root"]
    )
    with fixture["report"].open("a", encoding="utf-8") as handle:
        handle.write("injected after first validation\n")
    with pytest.raises(DevelopmentControlsGateError, match="checksum|canonical path"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"]
        )
