from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
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


def _binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def _rehash(document: dict[str, Any]) -> None:
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    document["receipt_self_sha256"] = sha256_json(stable)


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


def _build_fixture(root: Path) -> dict[str, Any]:
    (root / "src").mkdir(parents=True)
    (root / "src" / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    data = root / "data_usgs"
    data.mkdir()
    panel = data / "panel.parquet"
    panel.write_bytes(b"frozen development panel fixture")
    sites = [f"n{index:03d}" for index in range(120)]
    registry = data / "registry.csv"
    pd.DataFrame({
        "site_no": [f"{index:08d}" for index in range(120)],
        "legacy_site_id": sites,
    }).to_csv(registry, index=False)
    frozen_spec = data / "frozen.json"
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
    bridge = data / "bridge.json"
    bridge.write_text(json.dumps({
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": "b" * 64,
        "panel": _binding(root, panel),
        "registry": _binding(root, registry),
    }), encoding="utf-8")
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
    summaries: list[dict[str, Any]] = []
    for arm in arms:
        parameters = DC.parameter_count(arm, n_stations=120)
        for seed in arm.seeds:
            path = run_dir / "arm_predictions" / arm.arm_id / f"seed{seed}.parquet"
            frame = _prediction(arm, seed, sites)
            DC.write_arm_prediction(
                frame,
                path,
                identity=identity,
                arm=arm,
                seed=seed,
                parameters=parameters,
                n_stations=120,
                parents=parents,
                training_summary={
                    "best_validation_metric": 0.25,
                    "selected_epoch": 2,
                    "checkpoint_final_epoch": 4,
                },
            )
            members[(arm.arm_id, seed)] = path
            for split in ("val", "calib", "test"):
                for horizon in DC.C.HORIZONS:
                    summaries.append({
                        "arm_id": arm.arm_id,
                        "seed": seed,
                        "split": split,
                        "horizon": horizon,
                        "n": 120,
                        "rmse": 0.1,
                        "mae": 0.1,
                    })
    frames = {
        member: pd.read_parquet(path) for member, path in members.items()
    }
    audit = DC.validate_complete_prediction_matrix(
        frames, arms, allowed_sites=set(sites)
    )
    budget = DC.architecture_budget_rows(
        arms, n_stations=120, train_examples=3_073
    )
    predictions, budget_path, report = DC.publish_final_artifacts(
        run_dir=run_dir,
        identity=identity,
        arms=arms,
        member_paths=members,
        member_parents=parents,
        audit=audit,
        budget=budget,
        summaries=summaries,
    )
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
        report=report,
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
    _build_fixture(root)
    return root


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
    with pytest.raises(DevelopmentControlsGateError, match="common keys"):
        validate_stage09b_completion_receipt(
            fixture["receipt"], root=fixture["root"], document=forged
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
