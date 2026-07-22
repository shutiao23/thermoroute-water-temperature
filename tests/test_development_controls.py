from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import results as R  # noqa: E402
from thermoroute.development_controls import (  # noqa: E402
    DevelopmentControlsContractError,
)
from thermoroute.repro import RunIdentity, sidecar_path  # noqa: E402
from thermoroute.train import FitResult  # noqa: E402


SCRIPT = ROOT / "scripts" / "09b_development_controls.py"


def _load_script():
    module_name = "thermoroute_test_development_controls"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


DC = _load_script()


def _identity() -> RunIdentity:
    return RunIdentity(
        run_id="unit-test-development-controls",
        panel_sha256="1" * 64,
        registry_sha256="2" * 64,
        config_sha256="3" * 64,
        source_sha256="4" * 64,
        runtime_sha256="5" * 64,
    )


def _parents() -> dict[str, str]:
    return {
        "frozen_panel": "1" * 64,
        "frozen_station_registry": "2" * 64,
        "development_predictor_bridge": "6" * 64,
    }


def _tiny_predictions(arm, seed: int, *, site: str = "01234567") -> pd.DataFrame:
    split_dates = (
        ("val", "2016-01-02"),
        ("calib", "2018-01-02"),
        ("test", "2019-01-02"),
    )
    records = [
        (split, pd.Timestamp(date), horizon)
        for split, date in split_dates
        for horizon in DC.C.HORIZONS
    ]
    n = len(records)
    splits = np.asarray([record[0] for record in records])
    issue = pd.DatetimeIndex([record[1] for record in records])
    horizons = np.asarray([record[2] for record in records], dtype=int)
    target = issue + pd.to_timedelta(horizons, unit="D")
    truth = np.arange(1, n + 1, dtype=np.float32)
    return R.make_pred_frame(
        model=np.full(n, arm.arm_id),
        scope=np.full(n, DC.DEVELOPMENT_SCOPE),
        feature_set=np.full(n, arm.feature_set),
        seed=np.full(n, seed),
        site_id=np.full(n, site),
        horizon=horizons,
        split=splits,
        issue_date=issue,
        target_date=target,
        y_true=truth,
        y_pred=truth + 0.1,
        q05=truth - 0.5,
        q50=truth + 0.1,
        q95=truth + 0.5,
        p_exceed=np.full(n, 0.25),
    )


def test_declared_registry_and_parameter_budgets_are_exact() -> None:
    arms = DC.declared_arms()
    members = DC.expected_member_registry(arms)
    assert len(arms) == 9
    assert len(members) == 31
    assert len(set(members)) == 31
    assert [arm.variables for arm in arms[2:]] == [
        variables for _name, variables in DC.FEATURE_LADDER
    ]
    assert arms[0].seeds == (0, 1, 2, 3, 4)
    assert all(arm.seeds == (0, 1, 2) for arm in arms[2:])

    counts = DC.assert_parameter_budgets(arms, n_stations=120)
    assert counts["PlainMLP-7var"] == 38_545
    assert counts["PlainCausalTCN-7var"] == 38_031
    assert counts["ThermoRoute-ladder-07_plus_WDSP"] == 38_505
    assert abs(counts["PlainMLP-7var"] / 38_505 - 1) < 0.02
    assert abs(counts["PlainCausalTCN-7var"] / 38_505 - 1) < 0.02

    for arm in arms[2:]:
        model = DC.build_arm_model(arm, seed=0, n_stations=120)
        assert model.n_vars == len(arm.variables)
        assert model.router is not None and model.router.V == len(arm.variables)
        assert model.prior.eq_lin.in_features == DC._physics_count(arm.variables)
        configuration = DC.architecture_configuration(arm, seed=0, n_stations=120)
        assert configuration["input_variables"] == list(arm.variables)

    budget = DC.architecture_budget_rows(arms, n_stations=120, train_examples=3_073)
    assert set(budget["training_device"]) == {"cpu"}
    assert set(budget["selection_metric"]) == {"station_macro_rmse"}
    assert set(budget["historical_tuning_budget_equalized"]) == {False}
    assert set(budget["architecture_candidates_in_this_entrypoint"]) == {1}
    assert set(budget["maximum_optimizer_steps_per_seed"]) == {240}
    mlp_budget = budget.set_index("arm_id").loc["PlainMLP-7var"]
    tcn_budget = budget.set_index("arm_id").loc["PlainCausalTCN-7var"]
    assert mlp_budget["mlp_hidden_dim"] == 70
    assert tcn_budget["tcn_channels"] == 54
    assert json.loads(mlp_budget["architecture_configuration"])[
        "constructor_kwargs"
    ]["init_seed"] == "member_seed"

    source = SCRIPT.read_text(encoding="utf-8")
    assert "write_component_pointer" not in source
    assert "update_torch_component" not in source


def test_complete_matrix_requires_every_member_and_exact_common_keys() -> None:
    arms = (
        DC.ArmSpec("PlainMLP-test", "PlainMLP", "all_7", DC.FULL_VARIABLES, (0, 1)),
        DC.ArmSpec("PlainTCN-test", "PlainCausalTCN", "all_7", DC.FULL_VARIABLES, (0,)),
    )
    frames = {
        (arm.arm_id, seed): _tiny_predictions(arm, seed)
        for arm in arms
        for seed in arm.seeds
    }
    audit = DC.validate_complete_prediction_matrix(
        frames, arms, allowed_sites={"01234567"}
    )
    assert audit.expected_members == 3
    assert audit.common_forecast_keys == 9
    assert audit.prediction_rows == 27
    assert audit.splits == ("calib", "test", "val")

    incomplete = dict(frames)
    incomplete.pop(("PlainMLP-test", 1))
    with pytest.raises(DC.ControlExperimentError, match="incomplete"):
        DC.validate_complete_prediction_matrix(incomplete, arms)

    changed_key = {member: frame.copy() for member, frame in frames.items()}
    changed = changed_key[("PlainTCN-test", 0)]
    changed.loc[changed["split"].eq("test"), "issue_date"] += pd.Timedelta(days=1)
    changed.loc[changed["split"].eq("test"), "target_date"] += pd.Timedelta(days=1)
    with pytest.raises(DC.ControlExperimentError, match="forecast-key registry"):
        DC.validate_complete_prediction_matrix(changed_key, arms)

    changed_truth = {member: frame.copy() for member, frame in frames.items()}
    changed_truth[("PlainTCN-test", 0)].loc[
        lambda frame: frame["split"].eq("test"), "y_true"
    ] += np.float32(0.25)
    with pytest.raises(DC.ControlExperimentError, match="truth values"):
        DC.validate_complete_prediction_matrix(changed_truth, arms)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("issue_date", pd.NaT, "dates contain nulls"),
        ("q05", 99.0, "quantiles are not ordered"),
        ("q95", -99.0, "quantiles are not ordered"),
        ("p_exceed", 1.01, "outside"),
    ),
)
def test_prediction_semantics_reject_nat_crossing_and_probability(
    column: str, value: object, message: str,
) -> None:
    arm = DC.declared_arms()[0]
    frame = _tiny_predictions(arm, 0)
    frame.loc[0, column] = value
    with pytest.raises(DevelopmentControlsContractError, match=message):
        DC.normalise_prediction_frame(frame, arm=arm, seed=0)


@pytest.mark.parametrize(
    ("column", "operation"),
    (
        ("y_pred", lambda value: value + 0.05),
        ("q05", lambda value: value - 0.05),
        ("q50", lambda value: value + 0.05),
        ("q95", lambda value: value + 0.05),
        ("p_exceed", lambda _value: 0.75),
    ),
)
def test_prediction_digest_binds_every_scientific_output_column(
    column: str, operation,
) -> None:
    arm = DC.declared_arms()[0]
    baseline = DC.normalise_prediction_frame(
        _tiny_predictions(arm, 0), arm=arm, seed=0
    )
    attacked = baseline.copy()
    attacked.loc[0, column] = operation(attacked.loc[0, column])
    attacked = DC.normalise_prediction_frame(attacked, arm=arm, seed=0)
    assert DC.prediction_content_digest(attacked) != DC.prediction_content_digest(
        baseline
    )


def test_metric_recomputation_rejects_finite_inputs_with_overflowing_error() -> None:
    arm = DC.declared_arms()[0]
    frame = _tiny_predictions(arm, 0)
    frame["y_true"] = -1e308
    frame["y_pred"] = 1e308
    with pytest.raises(
        DevelopmentControlsContractError, match="overflows finite metric"
    ):
        DC.recompute_metric_summary({(arm.arm_id, 0): frame})


def test_canonical_registry_replaces_float32_equivalent_attacker_truth() -> None:
    arm = DC.declared_arms()[0]
    frame = _tiny_predictions(arm, 0)
    registry = frame[[
        "split", "site_id", "horizon", "issue_date", "target_date", "y_true"
    ]].copy()
    canonical_truth = registry["y_true"].astype("float32").astype("float64")
    frame["y_true"] = np.nextafter(canonical_truth, np.inf)
    normalised = DC.normalise_prediction_frame(
        frame, arm=arm, seed=0, canonical_registry=registry
    )
    canonical = registry.sort_values(
        ["split", "site_id", "horizon", "issue_date", "target_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    assert np.array_equal(
        normalised["y_true"].to_numpy(), canonical["y_true"].to_numpy()
    )


def test_immutable_prediction_cache_rejects_corrupt_or_stale_bytes(tmp_path: Path) -> None:
    arm = DC.declared_arms()[0]
    seed = arm.seeds[0]
    parameters = DC.parameter_count(arm, n_stations=120)
    path = tmp_path / "arm" / arm.arm_id / f"seed{seed}.parquet"
    frame = _tiny_predictions(arm, seed)
    summary = {
        "best_validation_metric": 0.25,
        "selected_epoch": 2,
        "checkpoint_final_epoch": 4,
    }
    DC.write_arm_prediction(
        frame,
        path,
        identity=_identity(),
        arm=arm,
        seed=seed,
        parameters=parameters,
        n_stations=120,
        parents=_parents(),
        training_summary=summary,
    )
    loaded = DC.read_arm_prediction(
        path,
        identity=_identity(),
        arm=arm,
        seed=seed,
        parameters=parameters,
        n_stations=120,
        parents=_parents(),
    )
    assert loaded is not None and len(loaded) == len(frame)
    with pytest.raises(DC.ControlExperimentError, match="overwrite"):
        DC.write_arm_prediction(
            frame,
            path,
            identity=_identity(),
            arm=arm,
            seed=seed,
            parameters=parameters,
            n_stations=120,
            parents=_parents(),
            training_summary=summary,
        )

    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(DC.ControlExperimentError, match="stale or corrupt"):
        DC.read_arm_prediction(
            path,
            identity=_identity(),
            arm=arm,
            seed=seed,
            parameters=parameters,
            n_stations=120,
            parents=_parents(),
        )


def test_tiny_mocked_training_runs_exact_5_5_21_matrix_and_publishes(tmp_path: Path) -> None:
    arms = DC.declared_arms()
    calls: list[tuple[str, int]] = []

    def fake_fit(model, _wd, _thresholds, **kwargs):
        built = model()
        arm = next(arm for arm in arms if arm.arm_id == kwargs["model_name"])
        seed = int(kwargs["seed"])
        calls.append((arm.arm_id, seed))
        return FitResult(
            model=built,
            pred=_tiny_predictions(arm, seed),
            best_val=0.125,
            epochs=0,
        )

    paths = []
    for _variables, arm_group in DC._group_arms_by_variables(arms):
        paths.extend(
            DC.train_arm_group(
                arm_group,
                wd=object(),
                thresholds={"01234567": 2.0},
                n_stations=120,
                identity=_identity(),
                run_config={"test": "tiny_mock"},
                run_dir=tmp_path,
                parents=_parents(),
                eval_batch_size=2,
                verbose=False,
                fit_function=fake_fit,
            )
        )
    expected = set(DC.expected_member_registry(arms))
    assert set(calls) == expected
    assert len(calls) == 5 + 5 + 21
    assert len(paths) == 31
    assert all(path.is_file() and sidecar_path(path).is_file() for path in paths)

    audit, members, summaries = DC.validate_prediction_paths(
        paths,
        arms,
        identity=_identity(),
        parents=_parents(),
        n_stations=120,
        allowed_sites={"01234567"},
    )
    assert audit.expected_members == 31
    assert audit.common_forecast_keys == 9
    budget = DC.architecture_budget_rows(arms, n_stations=120, train_examples=3)
    incomplete_members = dict(members)
    incomplete_members.pop(next(iter(incomplete_members)))
    blocked_dir = tmp_path / "blocked_incomplete_publication"
    with pytest.raises(DC.ControlExperimentError, match="exact audited member"):
        DC.publish_final_artifacts(
            run_dir=blocked_dir,
            identity=_identity(),
            arms=arms,
            member_paths=incomplete_members,
            member_parents=_parents(),
            audit=audit,
            budget=budget,
            summaries=summaries,
            train_examples=3,
            canonical_registry_sha256="0" * 64,
            canonical_train_registry_sha256="1" * 64,
        )
    assert not blocked_dir.exists()

    combined, budget_path, summary_path, report_path, semantic_audit_path = (
        DC.publish_final_artifacts(
        run_dir=tmp_path,
        identity=_identity(),
        arms=arms,
        member_paths=members,
        member_parents=_parents(),
        audit=audit,
        budget=budget,
        summaries=summaries,
        train_examples=3,
        canonical_registry_sha256="0" * 64,
        canonical_train_registry_sha256="1" * 64,
    ))
    combined_frame = pd.read_parquet(combined)
    assert len(combined_frame) == 31 * 9
    assert len(combined_frame[["model", "seed"]].drop_duplicates()) == 31
    assert len(pd.read_csv(budget_path)) == 9
    assert len(pd.read_csv(summary_path)) == 31 * 3 * 3
    report = report_path.read_text(encoding="utf-8")
    assert "not a blind or confirmatory test" in report
    assert "historical_tuning_budget_equalized" in report
    semantic = json.loads(semantic_audit_path.read_text(encoding="utf-8"))
    assert semantic["training_replay_verified"] is False
    for output in (combined, budget_path, summary_path, report_path, semantic_audit_path):
        metadata = json.loads(sidecar_path(output).read_text(encoding="utf-8"))
        assert metadata["extra"]["suite_pointer_written"] is False
    assert not any("pointer" in path.name.lower() for path in tmp_path.rglob("*"))

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("an exact cache hit must not retrain")

    cached_paths = []
    for _variables, arm_group in DC._group_arms_by_variables(arms):
        cached_paths.extend(
            DC.train_arm_group(
                arm_group,
                wd=object(),
                thresholds={"01234567": 2.0},
                n_stations=120,
                identity=_identity(),
                run_config={"test": "tiny_mock"},
                run_dir=tmp_path,
                parents=_parents(),
                eval_batch_size=2,
                verbose=False,
                fit_function=forbidden_fit,
            )
        )
    assert cached_paths == paths

    sidecar = sidecar_path(paths[0])
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["extra"]["variables"] = ["WTEMP"]
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(DC.ControlExperimentError, match="metadata changed"):
        DC.train_arm_group(
            [arms[0]],
            wd=object(),
            thresholds={"01234567": 2.0},
            n_stations=120,
            identity=_identity(),
            run_config={"test": "tiny_mock"},
            run_dir=tmp_path,
            parents=_parents(),
            eval_batch_size=2,
            verbose=False,
            fit_function=forbidden_fit,
        )


def _repo_artifact_snapshot(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_help_is_successful_and_creates_no_repository_artifact() -> None:
    run_root = ROOT / "outputs" / "runs" / "09b_development_controls"
    before = _repo_artifact_snapshot(run_root)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    after = _repo_artifact_snapshot(run_root)
    assert result.returncode == 0, result.stderr
    assert "development-only" in result.stdout.lower()
    assert before == after
