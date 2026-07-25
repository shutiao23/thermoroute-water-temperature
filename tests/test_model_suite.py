from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.checkpoint import (  # noqa: E402
    instantiate_inference_ensemble,
    neural_output_head_schema,
    save_inference_bundle,
)
from thermoroute import config as C  # noqa: E402
from thermoroute import model_suite as MODEL_SUITE  # noqa: E402
from thermoroute import results as R  # noqa: E402
from thermoroute.conformal import (  # noqa: E402
    cqr_policy_contract,
    finalise_cqr_offsets,
)
from thermoroute.evidence import FrozenPanelSpec  # noqa: E402
from thermoroute.frozen_inference import lstm_factory_from_metadata  # noqa: E402
from thermoroute.model_suite import (  # noqa: E402
    LIGHTGBM_HEADS,
    MODEL_SUITE_FORMAT,
    STAGE25_COMPLETION_RECEIPT_PATH,
    ModelSuiteError,
    build_stage25_completion_receipt,
    directory_binding,
    file_binding,
    development_prediction_binding,
    development_predictor_bridge_binding,
    freeze_model_suite,
    load_lightgbm_bundle,
    model_matrix_amendment_suite_binding,
    save_lightgbm_bundle,
    route_a_calibration_fit_contract,
    publish_stage25_completion_receipt,
    validate_model_suite_document,
    validate_model_matrix_suite_binding,
    validate_stage25_completion_receipt,
    validate_development_calibrated_head_gate,
    validate_development_prediction_binding,
    verify_lightgbm_prediction_parity,
    verify_sequence_prediction_parity,
    write_component_pointer,
    _create_json_or_require_identical,
    _learned_metadata_runtime_sha256,
)
from thermoroute.model_matrix_amendment import (  # noqa: E402
    AMENDMENT_ID,
    AMENDMENT_RELATIVE,
    AMENDMENT_SEAL_RELATIVE,
    GOVERNANCE_SHA256,
    MODEL_MATRIX_CONTRACT_FORMAT,
    MODEL_MATRIX_SUITE_BINDING_FORMAT,
    model_matrix_contract_id,
)
from thermoroute.quantiles import (  # noqa: E402
    LIGHTGBM_QUANTILE_REPAIR_METHOD,
    repair_lightgbm_quantiles,
)
from thermoroute.train import LSTMForecaster  # noqa: E402
from thermoroute.repro import (  # noqa: E402
    RunIdentity,
    seal_artifact,
    sha256_file,
    sha256_json,
    source_tree_hash,
)


def test_stage09_formal_ablation_registry_is_complete_five_seed_ensemble():
    assert MODEL_SUITE.STAGE9_ABLATION_SEEDS == tuple(C.USGS_SEEDS)
    temporal = MODEL_SUITE.DEVELOPMENT_REPLAY_MODEL_CONTRACTS["temporal"]
    assert all(
        temporal[model] == ("thermoroute_bundle", 5, 1e-5)
        for model in MODEL_SUITE.MANDATORY_ABLATIONS
    )


def _lgb_metadata(columns):
    offsets, offset_audit = finalise_cqr_offsets({
        ("__pooled__", horizon): 0.0 for horizon in (1, 3, 7)
    })
    return {
        "run_id": "fixture",
        "raw_feature_order": ["WTEMP", "FLOW"],
        "design_feature_order": list(columns),
        "horizons": [1, 3, 7],
        "station_agnostic": True,
        "uses_station_categorical": False,
        "station_categories": [],
        "preprocessing": {"fixture": True},
        "training_weighting": "equal_total_weight_per_station",
        "deterministic_training": {
            "deterministic": True, "force_col_wise": True, "n_jobs": 1,
        },
        "event_thresholds": {"__pooled__": 20.0},
        "event_calibrators": {},
        "conformal_offsets": {
            f"{site}|{horizon}": value
            for (site, horizon), value in offsets.items()
        },
        "conformal_policy": cqr_policy_contract(),
        "conformal_offset_audit": offset_audit,
        "calibration_fit_contract": route_a_calibration_fit_contract(
            external=True
        ),
        "source_sha256": "s",
        "panel_sha256": "p",
        "registry_sha256": "r",
        "config_sha256": "c",
        "runtime_sha256": "t" * 64,
        "input_closure_sha256": "f" * 64,
        "training_device": "cpu",
        "development_prediction": {"path": "predictions.parquet", "sha256": "d"},
    }


def _lgb_audit_inputs(X, horizons=(1, 3, 7), *, truth=0.0):
    issue_dates = pd.date_range("2019-01-01", periods=len(X), freq="D")
    return {
        horizon: (
            pd.DataFrame({
                "site_id": "fixture-site",
                "split": "test",
                "issue_date": issue_dates,
                "target_date": issue_dates + pd.to_timedelta(horizon, unit="D"),
                "y": truth,
            }),
            X,
        )
        for horizon in horizons
    }


def test_lightgbm_native_bundle_reconstructs_all_heads_with_prediction_parity(
    tmp_path,
    monkeypatch,
):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - X["b"] + rng.normal(scale=0.01, size=len(X))
    estimator = lgb.LGBMRegressor(n_estimators=12, num_leaves=7, verbosity=-1, n_jobs=1)
    estimator.fit(X, y)
    models = {
        f"seed{seed}": {
            horizon: {head: estimator for head in LIGHTGBM_HEADS}
            for horizon in (1, 3, 7)
        }
        for seed in range(5)
    }
    manifest = save_lightgbm_bundle(
        tmp_path / "lgb",
        models=models,
        metadata=_lgb_metadata(X.columns),
        quantile_audit_inputs=_lgb_audit_inputs(X),
        parity_inputs={horizon: X.iloc[:13] for horizon in (1, 3, 7)},
    )

    reconstructed: list[lgb.Booster] = []
    guard_observations: list[int] = []
    original_booster = MODEL_SUITE.lgb.Booster

    def tracked_booster(*args, **kwargs):
        value = original_booster(*args, **kwargs)
        reconstructed.append(value)
        return value

    def reject_reconstructed_booster() -> None:
        guard_observations.append(len(reconstructed))
        if reconstructed:
            assert isinstance(reconstructed[-1], original_booster)
            raise RuntimeError("injected LightGBM load policy drift")

    with monkeypatch.context() as context:
        context.setattr(MODEL_SUITE.lgb, "Booster", tracked_booster)
        with pytest.raises(RuntimeError, match="LightGBM load policy drift"):
            load_lightgbm_bundle(
                manifest,
                publication_guard=reject_reconstructed_booster,
            )
    assert guard_observations[0] == 0
    assert guard_observations[-1] > 0

    restored, metadata = load_lightgbm_bundle(manifest)
    assert set(restored) == {f"seed{seed}" for seed in range(5)}
    assert all(set(horizons) == {1, 3, 7} for horizons in restored.values())
    assert all(
        set(heads) == set(LIGHTGBM_HEADS)
        for horizons in restored.values() for heads in horizons.values()
    )
    assert all(
        metadata["roundtrip_parity"][f"seed{seed}"][str(horizon)][head]
        ["max_abs_difference"] == 0.0
        for seed in range(5) for horizon in (1, 3, 7) for head in LIGHTGBM_HEADS
    )
    expected = estimator.booster_.predict(X.iloc[:13], num_threads=1)
    assert np.array_equal(
        restored["seed4"][7]["event"].predict(X.iloc[:13], num_threads=1), expected
    )

    model_path = manifest.parent / "seed2_h3_q50.txt"
    model_path.write_text(model_path.read_text(encoding="utf-8") + "\n# tampered\n")
    with pytest.raises(ModelSuiteError, match="checksum"):
        load_lightgbm_bundle(manifest)


def test_lightgbm_raw_crossings_are_audited_and_nominal_q50_survives_replay(
    tmp_path,
    monkeypatch,
):
    X = pd.DataFrame({
        "a": np.arange(12.0),
        "b": np.arange(12.0) % 3,
        "c": np.ones(12),
    })

    def constant(value):
        return lgb.LGBMRegressor(
            n_estimators=2, min_child_samples=1, verbosity=-1, n_jobs=1,
        ).fit(X, np.full(len(X), value, dtype=float))

    point = constant(0.5)
    q05 = constant(3.0)
    q50 = constant(1.0)
    q95 = constant(2.0)
    event = constant(0.25)
    models = {
        f"seed{seed}": {
            horizon: {
                "point": point, "q05": q05, "q50": q50,
                "q95": q95, "event": event,
            }
            for horizon in (1, 3, 7)
        }
        for seed in range(5)
    }
    evaluation_design = _lgb_audit_inputs(X, truth=32.1)
    manifest = save_lightgbm_bundle(
        tmp_path / "crossed-lgb",
        models=models,
        metadata=_lgb_metadata(X.columns),
        quantile_audit_inputs=evaluation_design,
        parity_inputs={horizon: X.iloc[:4] for horizon in (1, 3, 7)},
    )
    _, metadata = load_lightgbm_bundle(manifest)
    assert metadata["quantile_repair"]["method"] == (
        LIGHTGBM_QUANTILE_REPAIR_METHOD
    )
    raw_audit = metadata["raw_quantile_crossing_audit"]["members"]
    for seed in range(5):
        for horizon in (1, 3, 7):
            summary = raw_audit[f"seed{seed}"][str(horizon)]
            assert summary["any_crossing_count"] == len(X)
            assert summary["any_crossing_rate"] == 1.0
            assert summary["maximum_crossing_gap_c"] == pytest.approx(2.0)

    expected_rows = []
    for seed in range(5):
        for horizon, (registry, _design) in evaluation_design.items():
            for row in registry.itertuples(index=False):
                expected_rows.append({
                    "seed": seed,
                    "site_id": row.site_id,
                    "horizon": horizon,
                    "split": row.split,
                    "issue_date": row.issue_date,
                    "target_date": row.target_date,
                    "y_true": float(np.float32(row.y)),
                    "y_pred": 0.5,
                    "q05": 1.0,
                    "q50": 1.0,
                    "q95": 2.0,
                    "p_exceed": 0.25,
                })
    difference = verify_lightgbm_prediction_parity(
        manifest,
        evaluation_design=evaluation_design,
        expected=pd.DataFrame(expected_rows),
        member_seeds={f"seed{seed}": seed for seed in range(5)},
        atol=1e-12,
    )
    assert difference == 0.0

    replay_completed = False
    guard_observations: list[bool] = []
    original_audit = MODEL_SUITE._build_lightgbm_raw_crossing_audit

    def tracked_audit(*args, **kwargs):
        nonlocal replay_completed
        value = original_audit(*args, **kwargs)
        replay_completed = True
        return value

    def reject_replayed_predictions() -> None:
        guard_observations.append(replay_completed)
        if replay_completed:
            raise RuntimeError("injected LightGBM parity policy drift")

    with monkeypatch.context() as context:
        context.setattr(
            MODEL_SUITE,
            "_build_lightgbm_raw_crossing_audit",
            tracked_audit,
        )
        with pytest.raises(RuntimeError, match="LightGBM parity policy drift"):
            verify_lightgbm_prediction_parity(
                manifest,
                evaluation_design=evaluation_design,
                expected=pd.DataFrame(expected_rows),
                member_seeds={f"seed{seed}": seed for seed in range(5)},
                atol=1e-12,
                publication_guard=reject_replayed_predictions,
            )
    assert guard_observations[0] is False
    assert guard_observations[-1] is True

    different_rows = pd.DataFrame(expected_rows)
    different_rows["y_true"] = np.nextafter(
        np.float32(32.1), np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ModelSuiteError, match="target labels differ"):
        verify_lightgbm_prediction_parity(
            manifest,
            evaluation_design=evaluation_design,
            expected=different_rows,
            member_seeds={f"seed{seed}": seed for seed in range(5)},
            atol=1e-12,
        )

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["raw_quantile_crossing_audit"]["members"]["seed0"]["1"][
        "any_crossing_rate"
    ] = 0.0
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelSuiteError, match="raw quantile audit self hash"):
        load_lightgbm_bundle(manifest)


def test_sequence_parity_guard_rechecks_after_numerical_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issue_date = pd.Timestamp("2019-01-01")
    expected = pd.DataFrame([{
        "model": "ThermoRoute",
        "scope": "temporal",
        "feature_set": "full",
        "seed": 0,
        "site_id": "fixture-site",
        "horizon": 1,
        "split": "test",
        "issue_date": issue_date,
        "target_date": issue_date + pd.Timedelta(days=1),
        "y_true": 11.0,
        "y_pred": 11.1,
        "q05": 10.0,
        "q50": 11.0,
        "q95": 12.0,
        "p_exceed": 0.2,
    }])
    replay_completed = False
    guard_observations: list[bool] = []

    def reject_after_replay() -> None:
        guard_observations.append(replay_completed)
        if replay_completed:
            raise RuntimeError("injected sequence parity policy drift")

    def fake_instantiate(
        _directory,
        *,
        model_factory,
        expected_member_count,
        device,
        publication_guard,
    ):
        assert model_factory is not None
        assert expected_member_count == 1
        assert device == "cpu"
        assert publication_guard is reject_after_replay
        publication_guard()
        return {"seed0": object()}, {}

    def fake_export(*_args, **_kwargs):
        nonlocal replay_completed
        replay_completed = True
        return expected.copy()

    monkeypatch.setattr(
        MODEL_SUITE,
        "instantiate_inference_ensemble",
        fake_instantiate,
    )
    monkeypatch.setattr("thermoroute.train._export_predictions", fake_export)

    with pytest.raises(RuntimeError, match="sequence parity policy drift"):
        verify_sequence_prediction_parity(
            "fixture-bundle",
            wd=object(),
            expected=expected,
            model_factory=lambda *_args: object(),
            member_seeds={"seed0": 0},
            publication_guard=reject_after_replay,
        )
    assert guard_observations[0] is False
    assert guard_observations[-1] is True


def test_median_preserving_repair_never_reassigns_nominal_q50():
    raw_q05 = np.array([4.0, -3.0, 5.0, 1.0])
    raw_q50 = np.array([2.0, -0.0, 3.0, 2.0], dtype="<f4")
    raw_q95 = np.array([1.0, 4.0, 2.0, 2.0])
    q05, q50, q95 = repair_lightgbm_quantiles(raw_q05, raw_q50, raw_q95)
    assert q50.dtype == raw_q50.dtype
    assert q50.tobytes() == raw_q50.tobytes()
    assert np.array_equal(q05, np.minimum(raw_q05, raw_q50))
    assert np.array_equal(q95, np.maximum(raw_q95, raw_q50))
    assert (q05 <= q50).all() and (q50 <= q95).all()


def _lstm_metadata():
    return {
        "run_id": "lstm-fixture",
        "architecture": {
            "class": "thermoroute.train.LSTMForecaster",
            "kwargs": {
                "n_vars": 2,
                "n_stations": 3,
                "d": 12,
                "layers": 1,
                "dropout": 0.0,
                "context": 5,
                "station_agnostic": False,
                "station_embed_dim": 4,
            },
        },
        "feature_order": ["WTEMP", "FLOW"],
        "horizons": [1, 3, 7],
        "station_to_index": {"a": 0, "b": 1, "c": 2},
        "preprocessing": {"fixture": True},
        "event_thresholds": {"a": 20.0},
        "event_calibrators": {},
        "conformal_offsets": {},
        "source_sha256": "s",
        "panel_sha256": "p",
        "registry_sha256": "r",
        "runtime_sha256": "t",
        "output_head_schema": neural_output_head_schema(),
    }


def _lstm_batch():
    return {
        "X": torch.randn(4, 5, 2),
        "Mask": torch.ones(4, 5, 2),
        "station": torch.tensor([0, 1, 2, 0]),
        "wtemp_t": torch.randn(4),
        "clim_t": torch.randn(4),
    }


def _predictor_bridge(root: Path) -> dict[str, str]:
    path = root / "data_usgs" / "development_predictor_bridge_v1.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "panel": file_binding(root, root / "panel.parquet"),
        "registry": file_binding(root, root / "registry.csv"),
    }), encoding="utf-8")
    return file_binding(root, path)


def _copy_model_matrix_suite_binding(root: Path) -> dict[str, Any]:
    """Create a Gitless byte/semantic fixture for suite-binding tests."""
    for relative in (*GOVERNANCE_SHA256, AMENDMENT_RELATIVE, AMENDMENT_SEAL_RELATIVE):
        source = ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return model_matrix_amendment_suite_binding(root)


def _placeholder_preopening_gates() -> dict[str, dict[str, str]]:
    return {
        "stage09_completion": {"path": "stage09", "sha256": "9" * 64},
        "stage09b_development_controls": {
            "path": "stage09b",
            "sha256": "b" * 64,
        },
        "stage16_lstm_completion": {
            "path": "stage16",
            "sha256": "1" * 64,
        },
        "stage25_external_completion": {
            "path": "stage25",
            "sha256": "2" * 64,
        },
    }


def test_model_matrix_suite_binding_is_exact_and_contract_is_independent(
    tmp_path: Path,
) -> None:
    binding = _copy_model_matrix_suite_binding(tmp_path)
    amendment = json.loads(
        (tmp_path / AMENDMENT_RELATIVE).read_text(encoding="utf-8")
    )
    expected_contract = sha256_json({
        "format": MODEL_MATRIX_CONTRACT_FORMAT,
        "stage09_architecture_control_matrix": amendment[
            "stage09_architecture_control_matrix"
        ],
        "stage09b_development_control_matrix": amendment[
            "stage09b_development_control_matrix"
        ],
    })

    assert set(binding) == {"format", "document", "seal", "contract_id"}
    assert binding["format"] == MODEL_MATRIX_SUITE_BINDING_FORMAT
    assert set(binding["document"]) == {
        "path", "sha256", "format", "status", "amendment_id",
        "amendment_document_commit",
    }
    assert set(binding["seal"]) == {"path", "sha256", "format", "status"}
    assert binding["document"]["amendment_id"] == AMENDMENT_ID
    assert binding["contract_id"] == expected_contract
    assert binding["contract_id"] == model_matrix_contract_id(
        amendment["stage09_architecture_control_matrix"],
        amendment["stage09b_development_control_matrix"],
    )
    assert validate_model_matrix_suite_binding(
        binding,
        root=tmp_path,
    ) == binding


def test_production_model_matrix_seal_strictly_precedes_live_suite_freeze_head(
) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    binding = model_matrix_amendment_suite_binding(
        ROOT,
        live_git_tip_commit=head,
    )
    assert binding["document"]["amendment_document_commit"] == (
        "e6e369a0076779aa4e053a23260f7c50d6bfa97e"
    )


@pytest.mark.parametrize(
    "attack",
    (
        "document_sha",
        "amendment_id",
        "document_commit",
        "seal_path",
        "seal_sha",
        "seal_status",
        "contract_id",
        "old_binding",
        "extra_outer",
        "extra_nested",
    ),
)
def test_model_matrix_suite_binding_rejects_every_tamper(
    tmp_path: Path,
    attack: str,
) -> None:
    binding: object = json.loads(
        json.dumps(_copy_model_matrix_suite_binding(tmp_path))
    )
    if attack == "old_binding":
        binding = None
    else:
        assert isinstance(binding, dict)
        if attack == "document_sha":
            binding["document"]["sha256"] = "0" * 64
        elif attack == "amendment_id":
            binding["document"]["amendment_id"] = "attacker-amendment"
        elif attack == "document_commit":
            binding["document"]["amendment_document_commit"] = "0" * 40
        elif attack == "seal_path":
            binding["seal"]["path"] = "protocols/attacker-seal.json"
        elif attack == "seal_sha":
            binding["seal"]["sha256"] = "0" * 64
        elif attack == "seal_status":
            binding["seal"]["status"] = "UNSEALED"
        elif attack == "contract_id":
            binding["contract_id"] = "0" * 64
        elif attack == "extra_outer":
            binding["extra"] = True
        elif attack == "extra_nested":
            binding["document"]["extra"] = True

    with pytest.raises(ModelSuiteError, match="model-matrix"):
        validate_model_matrix_suite_binding(binding, root=tmp_path)


def test_old_or_extra_field_model_suite_envelopes_are_rejected(tmp_path: Path) -> None:
    old = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": "b" * 64,
        "protocol_sha256": "protocol",
        "actual_feature_order": ["WTEMP"],
        "development_contract": {},
        "preopening_gates": {},
        "cohorts": {},
    }
    with pytest.raises(ModelSuiteError, match="top-level schema"):
        validate_model_suite_document(old, root=tmp_path)
    with pytest.raises(ModelSuiteError, match="top-level schema"):
        validate_model_suite_document(
            {
                **old,
                "model_matrix_amendment": {},
                "attacker_extra_field": True,
            },
            root=tmp_path,
        )


def test_model_suite_identity_hashes_the_complete_matrix_binding(
    tmp_path: Path,
) -> None:
    binding = _copy_model_matrix_suite_binding(tmp_path)
    stage24_path = ROOT / "scripts" / "24_freeze_model_suite.py"
    specification = importlib.util.spec_from_file_location(
        "thermoroute_stage24_matrix_identity_test",
        stage24_path,
    )
    assert specification is not None and specification.loader is not None
    stage24 = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(stage24)
    common = {
        "protocol_sha256": "a" * 64,
        "stage9": {"run_id": "stage9"},
        "stage09_completion": {"path": "stage09", "sha256": "9" * 64},
        "stage09b_completion": {"path": "stage09b", "sha256": "b" * 64},
        "stage16_completion": {"path": "stage16", "sha256": "1" * 64},
        "stage25_completion": {"path": "stage25", "sha256": "2" * 64},
        "lstm": {"run_id": "lstm"},
        "external": {"run_id": "external"},
        "features": ("WTEMP", "FLOW"),
    }
    original = stage24._model_suite_id(
        **common,
        model_matrix_amendment=binding,
    )
    for group, field in (
        ("document", "sha256"),
        ("document", "amendment_id"),
        ("document", "amendment_document_commit"),
        ("seal", "sha256"),
        ("seal", "status"),
        (None, "contract_id"),
    ):
        attacked = json.loads(json.dumps(binding))
        target = attacked if group is None else attacked[group]
        target[field] = "0" * len(str(target[field]))
        assert stage24._model_suite_id(
            **common,
            model_matrix_amendment=attacked,
        ) != original


def test_opening_registry_must_equal_its_bound_versioned_suite(
    tmp_path: Path,
) -> None:
    binding = _copy_model_matrix_suite_binding(tmp_path)
    versioned_document = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": "b" * 64,
        "protocol_sha256": "protocol",
        "actual_feature_order": ["WTEMP"],
        "development_contract": {},
        "model_matrix_amendment": binding,
        "preopening_gates": {},
        "cohorts": {},
    }
    versioned = tmp_path / "outputs/models/route_a_model_suite_fixture.json"
    versioned.parent.mkdir(parents=True)
    versioned.write_text(
        json.dumps(versioned_document, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attacked_alias = {
        **versioned_document,
        "protocol_sha256": "attacker-protocol",
        "versioned_suite": file_binding(tmp_path, versioned),
    }
    with pytest.raises(ModelSuiteError, match="differs from its versioned"):
        validate_model_suite_document(attacked_alias, root=tmp_path)


def test_freeze_publishes_one_matrix_binding_to_versioned_and_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _copy_model_matrix_suite_binding(tmp_path)
    monkeypatch.setattr(
        MODEL_SUITE,
        "_learned_metadata_runtime_sha256",
        lambda _root, _entries, *, publication_guard=None: "b" * 64,
    )
    monkeypatch.setattr(
        MODEL_SUITE,
        "validate_model_suite_document",
        lambda _document, *, root, publication_guard=None: None,
    )
    versioned = tmp_path / "outputs/models/suite-versioned.json"
    alias = tmp_path / "data_usgs/confirmatory_model_suite_v1.json"
    freeze_model_suite(
        versioned,
        tmp_path / "outputs/models/suite-current.json",
        root=tmp_path,
        protocol_sha256="a" * 64,
        temporal_entries=[],
        external_entries=[],
        actual_feature_order=("WTEMP", "FLOW"),
        development_contract={},
        model_matrix_amendment=binding,
        stage09_completion={"path": "stage09", "sha256": "9" * 64},
        stage09b_completion={"path": "stage09b", "sha256": "b" * 64},
        stage16_completion={"path": "stage16", "sha256": "1" * 64},
        stage25_completion={"path": "stage25", "sha256": "2" * 64},
        registry_alias=alias,
        publication_guard=lambda: None,
    )
    versioned_document = json.loads(versioned.read_text(encoding="utf-8"))
    alias_document = json.loads(alias.read_text(encoding="utf-8"))
    assert versioned_document["model_matrix_amendment"] == binding
    assert alias_document["model_matrix_amendment"] == binding
    versioned_binding = alias_document.pop("versioned_suite")
    assert versioned_binding == file_binding(tmp_path, versioned)
    assert alias_document == versioned_document


def test_development_predictor_bridge_is_a_required_exact_panel_registry_gate(
    tmp_path,
):
    (tmp_path / "panel.parquet").write_bytes(b"panel")
    (tmp_path / "registry.csv").write_bytes(b"registry")
    expected = _predictor_bridge(tmp_path)
    assert development_predictor_bridge_binding(
        tmp_path,
        panel_sha256=sha256_file(tmp_path / "panel.parquet"),
        registry_sha256=sha256_file(tmp_path / "registry.csv"),
    ) == expected
    with pytest.raises(ModelSuiteError, match="another panel"):
        development_predictor_bridge_binding(
            tmp_path,
            panel_sha256="0" * 64,
            registry_sha256=sha256_file(tmp_path / "registry.csv"),
        )


def test_lstm_five_seed_weights_bundle_has_exact_reconstruction_parity(tmp_path):
    metadata = _lstm_metadata()
    members = {}
    batch = _lstm_batch()
    expected = {}
    for seed in range(5):
        torch.manual_seed(seed)
        model = LSTMForecaster(**metadata["architecture"]["kwargs"])
        model.eval()
        members[f"seed{seed}"] = model
        output = model(batch)
        expected[f"seed{seed}"] = (
            output.point.detach().clone(), output.q50.detach().clone()
        )
    directory = save_inference_bundle(
        tmp_path / "lstm", members=members, metadata=metadata,
        expected_member_count=5,
    )
    restored, _ = instantiate_inference_ensemble(
        directory,
        model_factory=lambda _name, bundle: lstm_factory_from_metadata(bundle),
        expected_member_count=5,
    )
    for name, model in restored.items():
        output = model(batch)
        assert torch.equal(output.point, expected[name][0])
        assert torch.equal(output.q50, expected[name][1])
        assert not torch.equal(output.point, output.q50)


def test_suite_runtime_is_derived_from_every_learned_metadata_value(tmp_path):
    model = LSTMForecaster(**_lstm_metadata()["architecture"]["kwargs"])
    entries = []
    for index, digest in enumerate(("a" * 64, "a" * 64)):
        metadata = _lstm_metadata()
        metadata.update({"runtime_sha256": digest, "training_device": "cpu"})
        directory = save_inference_bundle(
            tmp_path / f"bundle-{index}",
            members={"seed0": model},
            metadata=metadata,
            expected_member_count=1,
        )
        entries.append({
            "model_id": f"learned-{index}",
            "executor": "lstm_bundle",
            "member_count": 1,
            "artifact": {"path": directory.relative_to(tmp_path).as_posix()},
        })
    assert _learned_metadata_runtime_sha256(tmp_path, entries) == "a" * 64

    changed = _lstm_metadata()
    changed.update({"runtime_sha256": "b" * 64, "training_device": "cpu"})
    third = save_inference_bundle(
        tmp_path / "bundle-drift",
        members={"seed0": model},
        metadata=changed,
        expected_member_count=1,
    )
    entries.append({
        "model_id": "learned-drift",
        "executor": "lstm_bundle",
        "member_count": 1,
        "artifact": {"path": third.relative_to(tmp_path).as_posix()},
    })
    with pytest.raises(ModelSuiteError, match="one numerical runtime"):
        _learned_metadata_runtime_sha256(tmp_path, entries)


def test_incomplete_suite_is_rejected_without_publishing_current_pointer(tmp_path):
    for name in ("spec.json", "panel.parquet", "registry.csv"):
        (tmp_path / name).write_bytes(name.encode("utf-8"))
    development_contract = {
        "frozen_panel_spec": file_binding(tmp_path, tmp_path / "spec.json"),
        "panel": file_binding(tmp_path, tmp_path / "panel.parquet"),
        "registry": file_binding(tmp_path, tmp_path / "registry.csv"),
        "predictor_bridge": _predictor_bridge(tmp_path),
        "source_sha256": source_tree_hash(tmp_path),
    }
    model_matrix_binding = _copy_model_matrix_suite_binding(tmp_path)
    development_contract["source_sha256"] = source_tree_hash(tmp_path)
    document = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": "b" * 64,
        "protocol_sha256": "protocol",
        "actual_feature_order": ["WTEMP", "FLOW"],
        "development_contract": development_contract,
        "model_matrix_amendment": model_matrix_binding,
        "preopening_gates": _placeholder_preopening_gates(),
        "cohorts": {
            "temporal": {"site_mode": "same_station", "models": []},
            "external": {
                "site_mode": "station_agnostic_history_dependent_new_site",
                "models": [],
            },
        },
    }
    with pytest.raises(ModelSuiteError, match="incomplete"):
        validate_model_suite_document(document, root=tmp_path)
    current = tmp_path / "current.json"
    with pytest.raises(ModelSuiteError, match="incomplete"):
        freeze_model_suite(
            tmp_path / "suite.json", current, root=tmp_path,
            protocol_sha256="protocol", temporal_entries=[], external_entries=[],
            actual_feature_order=("WTEMP", "FLOW"),
            development_contract=development_contract,
            publication_guard=lambda: None,
        )
    assert not current.exists()
    assert not (tmp_path / "suite.json").exists()


def test_model_suite_rejects_source_tree_drift_before_model_validation(tmp_path):
    for name in ("spec.json", "panel.parquet", "registry.csv"):
        (tmp_path / name).write_bytes(name.encode("utf-8"))
    source = tmp_path / "src" / "fixture.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    model_matrix_binding = _copy_model_matrix_suite_binding(tmp_path)
    document = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": "b" * 64,
        "protocol_sha256": "protocol",
        "actual_feature_order": ["WTEMP", "FLOW"],
        "model_matrix_amendment": model_matrix_binding,
        "preopening_gates": _placeholder_preopening_gates(),
        "development_contract": {
            "frozen_panel_spec": file_binding(tmp_path, tmp_path / "spec.json"),
            "panel": file_binding(tmp_path, tmp_path / "panel.parquet"),
            "registry": file_binding(tmp_path, tmp_path / "registry.csv"),
            "predictor_bridge": _predictor_bridge(tmp_path),
            "source_sha256": source_tree_hash(tmp_path),
        },
        "cohorts": {
            "temporal": {"models": []},
            "external": {"models": []},
        },
    }
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ModelSuiteError, match="differs from current source"):
        validate_model_suite_document(document, root=tmp_path)


def test_development_prediction_binding_recomputes_rows_keys_values_and_sidecar(tmp_path):
    rows = []
    for seed in (0, 1):
        for horizon in (1, 3):
            rows.append({
                "model": "Fixture", "scope": "development", "feature_set": "USGS",
                "seed": seed, "site_id": "01000001", "horizon": horizon,
                "split": "test", "issue_date": pd.Timestamp("2020-01-01"),
                "target_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=horizon),
                "y_true": 10.0, "y_pred": 10.0 + seed, "q05": 9.0,
                "q50": 10.0, "q95": 11.0, "p_exceed": 0.2,
            })
    frame = pd.DataFrame(rows)
    artifact = tmp_path / "predictions.parquet"
    frame.to_parquet(artifact)
    seal_artifact(
        artifact,
        RunIdentity(
            run_id="fixture",
            panel_sha256="a" * 64,
            registry_sha256="b" * 64,
            config_sha256="c" * 64,
            source_sha256="d" * 64,
            runtime_sha256="e" * 64,
            input_closure_sha256="f" * 64,
        ),
        kind="fixture-development-predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    binding = development_prediction_binding(
        tmp_path, artifact, frame, max_abs_difference=0.0, atol=1e-6
    )
    assert binding["prediction_columns"] == R.PRED_COLS
    validate_development_prediction_binding(
        tmp_path, binding, label="Fixture"
    )
    changed = frame.copy()
    changed.loc[0, "y_pred"] += 1.0
    changed.to_parquet(artifact)
    with pytest.raises(ModelSuiteError, match="checksum"):
        validate_development_prediction_binding(
            tmp_path, binding, label="Fixture"
        )


def _calibration_gate_frame() -> pd.DataFrame:
    rows = []
    for seed in (0, 1):
        for horizon in C.HORIZONS:
            for split, count, start in (
                ("val", 1, pd.Timestamp("2016-01-10")),
                ("calib", 100, pd.Timestamp("2018-01-01")),
                ("test", 1, pd.Timestamp("2019-01-10")),
            ):
                for index in range(count):
                    issue = start + pd.Timedelta(days=index)
                    target = issue + pd.Timedelta(days=horizon)
                    # Outcome truth is a property of (site, target date), not
                    # of the horizon/issue pair used to reach that date.
                    y_true = 10.0 + float(target.dayofyear % 20)
                    rows.append({
                        "model": "Fixture", "scope": "development",
                        "feature_set": "USGS", "seed": seed,
                        "site_id": "01000001", "horizon": horizon,
                        "split": split, "issue_date": issue,
                        "target_date": target,
                        "y_true": y_true, "y_pred": y_true + 0.1 * seed,
                        "q05": y_true - 1.0, "q50": y_true,
                        "q95": y_true + 1.0,
                        "p_exceed": 0.05 + 0.90 * (index % 100) / 99.0,
                    })
    return pd.DataFrame(rows)


def _calibration_gate_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    name: str,
) -> dict[str, object]:
    identity = RunIdentity(
        run_id="fixture", panel_sha256="a" * 64,
        registry_sha256="b" * 64, config_sha256="c" * 64,
        source_sha256="d" * 64, runtime_sha256="e" * 64,
        input_closure_sha256="f" * 64,
    )
    artifact = tmp_path / f"{name}.parquet"
    R.write_predictions(frame, artifact)
    seal_artifact(
        artifact, identity, kind="fixture-development-predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    binding = development_prediction_binding(
        tmp_path, artifact, frame, max_abs_difference=0.0, atol=1e-6
    )
    keys = [
        "model", "scope", "feature_set", "site_id", "horizon", "split",
        "issue_date", "target_date",
    ]
    ensemble = frame.groupby(keys, as_index=False, dropna=False, sort=True).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        q05=("q05", "mean"), q50=("q50", "mean"), q95=("q95", "mean"),
        p_exceed=("p_exceed", "mean"),
    )
    calibration = ensemble[ensemble.split.eq("calib")].copy()
    threshold = 19.0
    calibration["event"] = (calibration.y_true > threshold).astype(int)
    offsets, audit = MODEL_SUITE.cqr_offsets_with_audit(calibration, alpha=0.10)
    calibrators = MODEL_SUITE.P.fit_horizon_calibrators(
        calibration, probability_col="p_exceed", outcome_col="event",
        min_samples=100,
    )
    event_reference = {
        "format": "fixture-event-reference.v1",
        "fit_interval": ["2006-01-01", "2018-12-31"],
        "value": 0.25,
    }

    frozen_truth = (
        frame[["site_id", "target_date", "y_true"]]
        .drop_duplicates()
        .rename(columns={"target_date": "DATE", "y_true": "WTEMP"})
        .reset_index(drop=True)
    )

    def replay_event_definition(
        *_args: object,
        prediction_truth: pd.DataFrame | None = None,
        **_kwargs: object,
    ) -> tuple[dict[str, float], dict[str, object]]:
        assert prediction_truth is not None
        MODEL_SUITE._validate_development_truth_against_frozen_panel(
            frozen_truth, prediction_truth, label="Fixture"
        )
        return {"01000001": threshold}, dict(event_reference)

    monkeypatch.setattr(
        MODEL_SUITE,
        "_recompute_event_definition_from_frozen_panel",
        replay_event_definition,
    )
    return {
        **identity.as_dict(),
        "member_count": 2,
        "development_prediction": binding,
        "event_thresholds": {"01000001": threshold},
        "event_reference_climatology": event_reference,
        "event_calibrators": {
            str(horizon): calibrator.as_dict()
            for horizon, calibrator in sorted(calibrators.items())
        },
        "conformal_offsets": MODEL_SUITE.serialise_offsets(offsets),
        "conformal_policy": cqr_policy_contract(),
        "conformal_offset_audit": audit,
        "calibration_fit_contract": route_a_calibration_fit_contract(
            external=False
        ),
    }


def test_calibrated_head_gate_requires_every_seed_once_per_forecast_key(
    tmp_path, monkeypatch,
):
    frame = _calibration_gate_frame()
    metadata = _calibration_gate_metadata(
        tmp_path, monkeypatch, frame, "complete"
    )
    gate = validate_development_calibrated_head_gate(
        tmp_path, metadata, label="Fixture", external=False
    )
    assert gate["status"] == "PASS_NONNEGATIVE_CQR_WIDENS_ONLY"

    attacked = frame.copy()
    target = attacked[
        attacked.seed.eq(1) & attacked.split.eq("test")
        & attacked.horizon.eq(1)
    ].index[0]
    attacked.loc[target, "seed"] = 0
    attacked.loc[target, "feature_set"] = "ATTACK"
    attacked_metadata = _calibration_gate_metadata(
        tmp_path, monkeypatch, attacked, "attacked"
    )
    validate_development_prediction_binding(
        tmp_path, attacked_metadata["development_prediction"], label="Fixture"
    )
    with pytest.raises(ModelSuiteError, match="every declared seed exactly once"):
        validate_development_calibrated_head_gate(
            tmp_path, attacked_metadata, label="Fixture", external=False
        )


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("duplicate_panel_key", "invalid or duplicated"),
        ("absent_prediction_key", "absent from frozen panel"),
        ("nonfinite_panel_truth", "missing or non-finite"),
        ("float32_overflow_truth", "not finite at model precision"),
        ("changed_prediction_truth", "differs from frozen panel"),
    ),
)
def test_development_truth_binding_fails_closed_on_panel_or_truth_attack(
    attack, error,
):
    panel = pd.DataFrame({
        "DATE": [pd.Timestamp("2018-01-02"), pd.Timestamp("2018-01-03")],
        "site_id": ["01000001", "01000001"],
        "WTEMP": [12.0, 13.0],
    })
    truth = pd.DataFrame({
        "site_id": ["01000001"],
        "target_date": [pd.Timestamp("2018-01-02")],
        "y_true": [12.0],
    })
    if attack == "duplicate_panel_key":
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    elif attack == "absent_prediction_key":
        truth.loc[0, "target_date"] = pd.Timestamp("2018-01-04")
    elif attack == "nonfinite_panel_truth":
        panel.loc[0, "WTEMP"] = np.nan
    elif attack == "float32_overflow_truth":
        overflow = float(np.finfo(np.float32).max) * 2.0
        panel.loc[0, "WTEMP"] = overflow
        truth.loc[0, "y_true"] = overflow
    else:
        truth.loc[0, "y_true"] += 1e-3
    with pytest.raises(ModelSuiteError, match=error):
        MODEL_SUITE._validate_development_truth_against_frozen_panel(
            panel, truth, label="Fixture"
        )


def test_development_truth_binding_compares_at_model_float32_precision():
    panel_value = 32.123456789
    panel = pd.DataFrame({
        "DATE": [pd.Timestamp("2018-01-02")],
        "site_id": ["01000001"],
        "WTEMP": [panel_value],
    })
    truth = pd.DataFrame({
        "site_id": ["01000001"],
        "target_date": [pd.Timestamp("2018-01-02")],
        "y_true": [float(np.float32(panel_value))],
    })
    MODEL_SUITE._validate_development_truth_against_frozen_panel(
        panel, truth, label="Fixture"
    )


def test_calibration_gate_rejects_reclosed_prediction_truth_attack(
    tmp_path, monkeypatch,
):
    """Rehashing prediction and every envelope cannot invent panel truth."""
    frame = _calibration_gate_frame()
    metadata = _calibration_gate_metadata(
        tmp_path, monkeypatch, frame, "reclosed-y-true"
    )
    binding = metadata["development_prediction"]
    prediction_path = tmp_path / binding["artifact"]["path"]
    attacked = pd.read_parquet(prediction_path)
    selected = attacked[
        attacked["split"].eq("test") & attacked["horizon"].eq(1)
    ].iloc[0]
    same_outcome = (
        attacked["site_id"].eq(selected["site_id"])
        & attacked["target_date"].eq(selected["target_date"])
    )
    assert int(same_outcome.sum()) >= 2
    attacked.loc[same_outcome, "y_true"] += 0.25
    R.write_predictions(attacked, prediction_path)

    identity = RunIdentity(
        run_id=str(metadata["run_id"]),
        panel_sha256=str(metadata["panel_sha256"]),
        registry_sha256=str(metadata["registry_sha256"]),
        config_sha256=str(metadata["config_sha256"]),
        source_sha256=str(metadata["source_sha256"]),
        runtime_sha256=str(metadata["runtime_sha256"]),
        input_closure_sha256=str(metadata["input_closure_sha256"]),
    )
    seal_artifact(
        prediction_path,
        identity,
        kind="fixture-development-predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    metadata["development_prediction"] = development_prediction_binding(
        tmp_path,
        prediction_path,
        attacked,
        max_abs_difference=0.0,
        atol=1e-6,
    )

    metadata_path = tmp_path / "reclosed-y-true-metadata.json"
    weights_path = tmp_path / "reclosed-y-true-weights.pt"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    weights_path.write_bytes(b"fixture weights")
    pointer = {
        "metadata": file_binding(tmp_path, metadata_path),
        "weights": file_binding(tmp_path, weights_path),
        "prediction": file_binding(tmp_path, prediction_path),
        "prediction_sidecar": file_binding(
            tmp_path, MODEL_SUITE.sidecar_path(prediction_path)
        ),
    }
    pointer_path = tmp_path / "reclosed-y-true-pointer.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    artifacts = {
        "component_pointer": file_binding(tmp_path, pointer_path),
        "metadata": file_binding(tmp_path, metadata_path),
        "prediction": file_binding(tmp_path, prediction_path),
        "prediction_sidecar": pointer["prediction_sidecar"],
    }
    receipt = {
        "format": "fixture-stage-receipt.v1",
        "artifacts": artifacts,
        "artifact_closure_sha256": sha256_json(artifacts),
    }
    receipt["receipt_self_sha256"] = sha256_json(receipt)
    assert receipt["artifact_closure_sha256"] == sha256_json(
        receipt["artifacts"]
    )
    stable = dict(receipt)
    self_digest = stable.pop("receipt_self_sha256")
    assert self_digest == sha256_json(stable)

    with pytest.raises(ModelSuiteError, match="differs from frozen panel WTEMP"):
        validate_development_calibrated_head_gate(
            tmp_path, metadata, label="Fixture", external=False
        )


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("offset", "replayed conformal_offsets"),
        ("calibrator", "replayed event_calibrators"),
        ("constant_calibrator", "constant calibrator invariant"),
        ("threshold", "replayed event_thresholds"),
        ("event_reference", "replayed event_reference_climatology"),
    ),
)
def test_calibration_replay_rejects_reclosed_metadata_attack(
    tmp_path, monkeypatch, attack, error,
):
    """Rehashing every enclosing object cannot bless invented calibration."""
    frame = _calibration_gate_frame()
    metadata = _calibration_gate_metadata(
        tmp_path, monkeypatch, frame, f"reclosed-{attack}"
    )
    attacked = json.loads(json.dumps(metadata))
    if attack == "offset":
        raw = dict(attacked["conformal_offset_audit"]["raw_signed_offsets"])
        raw["01000001|1"] = 0.5
        deployed, audit = finalise_cqr_offsets(raw)
        attacked["conformal_offsets"] = dict(deployed)
        attacked["conformal_offset_audit"] = audit
    elif attack == "calibrator":
        attacked["event_calibrators"]["1"]["intercept"] += 0.25
    elif attack == "constant_calibrator":
        attacked["event_calibrators"]["1"] = {
            "intercept": float(MODEL_SUITE.P.logit(np.asarray([0.25]))[0]),
            "slope": 1.0,
            "constant": 0.25,
        }
    elif attack == "threshold":
        attacked["event_thresholds"]["01000001"] += 0.5
    else:
        attacked["event_reference_climatology"]["value"] += 0.1

    # Simulate an attacker who controls all mutable provenance envelopes: the
    # metadata bytes, prediction sidecar reseal, component pointer, artifact
    # closure and receipt self-hash are all internally current after the edit.
    bundle = tmp_path / f"bundle-{attack}"
    bundle.mkdir()
    metadata_path = bundle / "metadata.json"
    weights_path = bundle / "weights.pt"
    metadata_path.write_text(json.dumps(attacked), encoding="utf-8")
    weights_path.write_bytes(b"fixture weights")
    binding = attacked["development_prediction"]
    prediction_path = tmp_path / binding["artifact"]["path"]
    seal_artifact(
        prediction_path,
        RunIdentity(
            run_id=attacked["run_id"],
            panel_sha256=attacked["panel_sha256"],
            registry_sha256=attacked["registry_sha256"],
            config_sha256=attacked["config_sha256"],
            source_sha256=attacked["source_sha256"],
            runtime_sha256=attacked["runtime_sha256"],
            input_closure_sha256=attacked["input_closure_sha256"],
        ),
        kind="fixture-development-predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    binding["artifact"]["sidecar"] = file_binding(
        tmp_path, MODEL_SUITE.sidecar_path(prediction_path)
    )
    pointer = {
        "metadata": file_binding(tmp_path, metadata_path),
        "weights": file_binding(tmp_path, weights_path),
        "prediction_sidecar": dict(binding["artifact"]["sidecar"]),
    }
    pointer_path = tmp_path / f"pointer-{attack}.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    artifacts = {
        "component_pointer": file_binding(tmp_path, pointer_path),
        "metadata": file_binding(tmp_path, metadata_path),
        "prediction": file_binding(tmp_path, prediction_path),
        "prediction_sidecar": dict(binding["artifact"]["sidecar"]),
    }
    receipt = {
        "format": "fixture-stage-receipt.v1",
        "artifacts": artifacts,
        "artifact_closure_sha256": sha256_json(artifacts),
    }
    receipt["receipt_self_sha256"] = sha256_json(receipt)
    assert receipt["artifact_closure_sha256"] == sha256_json(
        receipt["artifacts"]
    )
    self_digest = receipt.pop("receipt_self_sha256")
    assert self_digest == sha256_json(receipt)

    with pytest.raises(ModelSuiteError, match=error):
        validate_development_calibrated_head_gate(
            tmp_path, attacked, label="Fixture", external=False
        )


@pytest.mark.parametrize("attack", ("drop_val", "escape_val_interval"))
def test_calibration_gate_rejects_reclosed_split_registry_attack(
    tmp_path, monkeypatch, attack,
):
    frame = _calibration_gate_frame()
    if attack == "drop_val":
        frame = frame[~frame.split.eq("val")].reset_index(drop=True)
        error = "exactly val/calib/test"
    else:
        target = frame[frame.split.eq("val")].index[0]
        frame.loc[target, "issue_date"] = pd.Timestamp("2018-01-01")
        frame.loc[target, "target_date"] = (
            pd.Timestamp("2018-01-01")
            + pd.Timedelta(days=int(frame.loc[target, "horizon"]))
        )
        error = "val rows escape"
    metadata = _calibration_gate_metadata(
        tmp_path, monkeypatch, frame, f"split-{attack}"
    )
    with pytest.raises(ModelSuiteError, match=error):
        validate_development_calibrated_head_gate(
            tmp_path, metadata, label="Fixture", external=False
        )


def test_calibration_event_replay_maps_legacy_panel_ids_to_stable_site_no():
    registry_path = ROOT / "data_usgs" / "station_registry_v1.csv"
    panel_path = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
    registry = pd.read_csv(registry_path, dtype={"site_no": "string"})
    stable_sites = set(registry["site_no"].astype(str))
    stable_panel = FrozenPanelSpec.load(
        ROOT / "data_usgs" / "frozen_panel_v1.json"
    ).load_panel(stable_site_ids=True)
    finite = stable_panel[
        np.isfinite(pd.to_numeric(stable_panel["WTEMP"], errors="coerce"))
    ].iloc[[0]]
    truth = finite[["site_id", "DATE", "WTEMP"]].rename(
        columns={"DATE": "target_date", "WTEMP": "y_true"}
    )
    thresholds, reference = (
        MODEL_SUITE._recompute_event_definition_from_frozen_panel(
            ROOT,
            {
                "panel_sha256": sha256_file(panel_path),
                "registry_sha256": sha256_file(registry_path),
            },
            selected_sites=stable_sites,
            external=True,
            label="Stable-site replay fixture",
            prediction_truth=truth,
        )
    )
    assert len(stable_sites) == 120
    assert all(not site.startswith("n") for site in stable_sites)
    assert thresholds == {"__pooled__": 23.0}
    assert reference["mode"] == "pooled_month"
    assert reference["fit_observation_count"] == 467_999


def test_frozen_registry_create_is_idempotent_but_never_overwrites(tmp_path):
    path = tmp_path / "registry.json"
    _create_json_or_require_identical(path, {"format": "fixture", "value": 1})
    original = path.read_bytes()
    _create_json_or_require_identical(path, {"format": "fixture", "value": 1})
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _create_json_or_require_identical(path, {"format": "fixture", "value": 2})
    assert path.read_bytes() == original


@pytest.mark.parametrize("attack", ("sibling_symlink", "hardlink_alias"))
def test_suite_freeze_rejects_versioned_output_link_aliases(
    tmp_path, monkeypatch, attack,
):
    destination = tmp_path / "outputs" / "models" / "suite-versioned.json"
    destination.parent.mkdir(parents=True)
    alias = destination.with_name(f"suite-{attack}.json")
    alias.write_bytes(b"attacker-controlled output\n")
    if attack == "sibling_symlink":
        destination.symlink_to(alias)
    else:
        os.link(alias, destination)
        assert destination.stat().st_nlink == 2
    monkeypatch.setattr(
        MODEL_SUITE,
        "_learned_metadata_runtime_sha256",
        lambda _root, _entries, *, publication_guard=None: "b" * 64,
    )
    monkeypatch.setattr(
        MODEL_SUITE,
        "validate_model_suite_document",
        lambda _document, *, root, publication_guard=None: None,
    )
    with pytest.raises(ModelSuiteError, match="symlink|single-link"):
        freeze_model_suite(
            destination,
            tmp_path / "outputs" / "models" / "current.json",
            root=tmp_path,
            protocol_sha256="a" * 64,
            temporal_entries=[],
            external_entries=[],
            actual_feature_order=("WTEMP", "FLOW"),
            development_contract={},
            publication_guard=lambda: None,
        )


def test_frozen_registry_rejects_identical_hardlink_alias(tmp_path):
    path = tmp_path / "registry.json"
    value = {"format": "fixture", "value": 1}
    _create_json_or_require_identical(path, value)
    os.link(path, tmp_path / "registry-hardlink-alias.json")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _create_json_or_require_identical(path, value)


def _publication_staging_files(path: Path) -> list[Path]:
    """Find transaction-private files for one intended authoritative path."""
    return sorted({
        *path.parent.glob(f".{path.name}.*.tmp"),
        *path.parent.glob(f".{path.name}.*.staging"),
    })


def test_frozen_registry_create_guard_failure_removes_durable_staging(tmp_path):
    path = tmp_path / "registry.json"
    value = {"format": "fixture", "value": 1}
    observed_staging: list[Path] = []

    def reject_staged_publication() -> None:
        staged = _publication_staging_files(path)
        if not staged:
            return
        assert len(staged) == 1
        assert json.loads(staged[0].read_text(encoding="utf-8")) == value
        assert not path.exists()
        observed_staging.extend(staged)
        raise RuntimeError("injected frozen-registry publication drift")

    with pytest.raises(RuntimeError, match="frozen-registry publication drift"):
        _create_json_or_require_identical(
            path,
            value,
            publication_guard=reject_staged_publication,
        )
    assert observed_staging
    assert not path.exists()
    assert not _publication_staging_files(path)


@pytest.mark.parametrize("rejected_boundary", ("versioned", "alias", "current"))
def test_suite_freeze_guard_failure_never_advances_authoritative_boundary(
    tmp_path,
    monkeypatch,
    rejected_boundary,
):
    versioned = tmp_path / "outputs" / "models" / "suite-versioned.json"
    alias = tmp_path / "data_usgs" / "confirmatory-suite.json"
    current = tmp_path / "outputs" / "models" / "suite-current.json"
    targets = {
        "versioned": versioned,
        "alias": alias,
        "current": current,
    }
    target = targets[rejected_boundary]
    old_pointer = b"pre-existing current pointer bytes\n"
    if rejected_boundary == "current":
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_bytes(old_pointer)

    monkeypatch.setattr(
        MODEL_SUITE,
        "_learned_metadata_runtime_sha256",
        lambda _root, _entries, *, publication_guard=None: "b" * 64,
    )
    monkeypatch.setattr(
        MODEL_SUITE,
        "validate_model_suite_document",
        lambda _document, *, root, publication_guard=None: None,
    )
    observed_staging: list[Path] = []

    def reject_target_staging() -> None:
        staged = _publication_staging_files(target)
        if not staged:
            return
        assert len(staged) == 1
        assert staged[0].stat().st_size > 0
        if rejected_boundary == "current":
            assert current.read_bytes() == old_pointer
        else:
            assert not target.exists()
        observed_staging.extend(staged)
        raise RuntimeError(f"injected {rejected_boundary} publication drift")

    with pytest.raises(
        RuntimeError,
        match=rf"{rejected_boundary} publication drift",
    ):
        freeze_model_suite(
            versioned,
            current,
            root=tmp_path,
            protocol_sha256="a" * 64,
            temporal_entries=[],
            external_entries=[],
            actual_feature_order=("WTEMP", "FLOW"),
            development_contract={},
            stage09_completion={"path": "stage09", "sha256": "c" * 64},
            stage09b_completion={"path": "stage09b", "sha256": "d" * 64},
            stage25_completion={"path": "stage25", "sha256": "e" * 64},
            registry_alias=alias,
            publication_guard=reject_target_staging,
        )

    assert observed_staging, "guard never observed the durable staging boundary"
    assert not _publication_staging_files(target)
    if rejected_boundary == "versioned":
        assert not versioned.exists()
        assert not alias.exists()
        assert not current.exists()
    elif rejected_boundary == "alias":
        assert versioned.is_file()
        assert not alias.exists()
        assert not current.exists()
    else:
        assert versioned.is_file()
        assert alias.is_file()
        assert current.read_bytes() == old_pointer


def test_lightgbm_declared_member_count_rejects_a_missing_seed(tmp_path):
    X = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) % 3,
                      "c": np.ones(30)})
    estimator = lgb.LGBMRegressor(n_estimators=3, verbosity=-1, n_jobs=1).fit(
        X, np.arange(30.0)
    )
    models = {
        f"seed{seed}": {
            horizon: {head: estimator for head in LIGHTGBM_HEADS}
            for horizon in (1, 3, 7)
        }
        for seed in range(4)
    }
    metadata = _lgb_metadata(X.columns)
    metadata["members"] = [f"seed{seed}" for seed in range(5)]
    metadata["member_count"] = 5
    with pytest.raises(ModelSuiteError, match="member registry"):
        save_lightgbm_bundle(
            tmp_path / "missing", models=models, metadata=metadata,
            quantile_audit_inputs=_lgb_audit_inputs(X),
            parity_inputs={horizon: X.iloc[:3] for horizon in (1, 3, 7)},
        )


def test_lightgbm_bundle_never_overwrites_a_content_address(tmp_path):
    X = pd.DataFrame({"a": np.arange(24.0), "b": np.arange(24.0) % 4,
                      "c": np.ones(24)})
    one = lgb.LGBMRegressor(n_estimators=3, verbosity=-1, n_jobs=1).fit(
        X, np.arange(24.0)
    )
    two = lgb.LGBMRegressor(n_estimators=3, verbosity=-1, n_jobs=1).fit(
        X, -np.arange(24.0)
    )
    def suite(estimator):
        return {
            f"seed{seed}": {
                horizon: {head: estimator for head in LIGHTGBM_HEADS}
                for horizon in (1, 3, 7)
            }
            for seed in range(5)
        }
    target = tmp_path / "immutable-lgb"
    manifest = save_lightgbm_bundle(
        target, models=suite(one), metadata=_lgb_metadata(X.columns),
        quantile_audit_inputs=_lgb_audit_inputs(X),
        parity_inputs={horizon: X.iloc[:3] for horizon in (1, 3, 7)},
    )
    original = manifest.read_bytes()
    # A byte-identical retry is a cache hit, not a rewrite.
    save_lightgbm_bundle(
        target, models=suite(one), metadata=_lgb_metadata(X.columns),
        quantile_audit_inputs=_lgb_audit_inputs(X),
        parity_inputs={horizon: X.iloc[:3] for horizon in (1, 3, 7)},
    )
    with pytest.raises(FileExistsError, match="non-identical"):
        save_lightgbm_bundle(
            target, models=suite(two), metadata=_lgb_metadata(X.columns),
            quantile_audit_inputs=_lgb_audit_inputs(X),
            parity_inputs={horizon: X.iloc[:3] for horizon in (1, 3, 7)},
        )
    assert manifest.read_bytes() == original


def _stage25_receipt_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object], Path]:
    """Create a byte-real but inference-free Stage-25 closure fixture."""
    source = tmp_path / "src" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    data = tmp_path / "data_usgs"
    data.mkdir()
    panel = data / "panel_usgs_120v2.parquet"
    registry = data / "station_registry_v1.csv"
    spec = data / "frozen_panel_v1.json"
    panel.write_bytes(b"development panel\n")
    registry.write_bytes(b"development registry\n")
    spec.write_bytes(b"frozen development spec\n")
    bridge_path = data / "development_predictor_bridge_v1.json"
    bridge_path.write_text(json.dumps({
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "panel": file_binding(tmp_path, panel),
        "registry": file_binding(tmp_path, registry),
    }), encoding="utf-8")
    bridge = file_binding(tmp_path, bridge_path)
    development_contract = {
        "frozen_panel_spec": file_binding(tmp_path, spec),
        "panel": file_binding(tmp_path, panel),
        "registry": file_binding(tmp_path, registry),
        "predictor_bridge": bridge,
        "source_sha256": source_tree_hash(tmp_path),
    }
    monkeypatch.setattr(
        MODEL_SUITE,
        "canonical_development_contract",
        lambda *_args, **_kwargs: dict(development_contract),
    )
    runtime_contract = {"fixture_numerical_runtime": "cpu-single-thread"}
    monkeypatch.setattr(
        MODEL_SUITE, "numerical_runtime_contract", lambda: runtime_contract
    )
    runtime_sha256 = sha256_json(runtime_contract)
    development_input_digest = "a" * 64
    input_closure_sha256 = MODEL_SUITE.compose_input_closure_digest({
        "development": development_input_digest,
    })

    class FixtureInputClosure:
        binding_digest = development_input_digest
        inventory = (object(),)

        @staticmethod
        def assert_unchanged() -> None:
            return None

    monkeypatch.setattr(
        MODEL_SUITE,
        "resolve_development_input_closure",
        lambda _root: FixtureInputClosure(),
    )
    threshold_contract = {
        "method": "pooled_training_empirical_quantile_v1",
        "quantile": 0.90,
        "pool_weighting": "equal_weight_per_finite_training_row",
        "station_balanced": False,
    }
    configuration = {
        "stage": "25_train_external_pooled_suite",
        "role": "prelabel_station_agnostic_development_training",
        "panel": panel.name,
        "registry": registry.name,
        "variables": list(MODEL_SUITE.STAGE9_USGS_VARIABLES),
        "horizons": list(C.HORIZONS),
        "seeds": list(C.USGS_SEEDS),
        "train_config": asdict(C.TrainConfig(batch_size=1536)),
        "preprocessing": "pooled_development_train_only",
        "station_agnostic": True,
        "lstm_validation_grid": [
            dict(value) for value in MODEL_SUITE.LSTM_VALIDATION_GRID
        ],
        "lightgbm_validation_grid": [
            dict(value)
            for value in MODEL_SUITE.STAGE9_LIGHTGBM_VALIDATION_GRID
        ],
        "event_reference_fit_interval": ["2006-01-01", "2018-12-31"],
        "event_threshold_estimator": threshold_contract,
        "post_2020_data_read": False,
        "training_device": "cpu",
        "development_predictor_bridge": bridge,
        "formal_numerical_policy": {"worker_threads": 1},
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": 1,
        "input_closure_component_count": 1,
    }
    identity_fields = {
        "schema_version": MODEL_SUITE.RUN_SCHEMA_VERSION,
        "panel_sha256": sha256_file(panel),
        "registry_sha256": sha256_file(registry),
        "config_sha256": sha256_json(configuration),
        "source_sha256": source_tree_hash(tmp_path),
        "runtime_sha256": runtime_sha256,
        "input_closure_sha256": input_closure_sha256,
    }
    run_id = sha256_json(identity_fields)[:20]
    identity = {"run_id": run_id, **identity_fields}
    run_manifest = (
        tmp_path / "outputs" / "runs" / "25_external_pooled"
        / run_id / "run.json"
    )
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(json.dumps({
        "schema_version": MODEL_SUITE.RUN_SCHEMA_VERSION,
        "identity": identity,
        "resolved_config": configuration,
        "created_utc": "2026-07-24T00:00:00+00:00",
        "environment": {},
        "git": {},
        "provenance": {
            "outcome_status": "NO_POST_2020_DATA_READ",
            "training_device": "cpu",
        },
    }), encoding="utf-8")

    prediction = (
        tmp_path / "outputs" / "predictions"
        / f"external_pooled_development_{run_id}.parquet"
    )
    frame = pd.DataFrame([{
        "model": "ThermoRoute",
        "scope": "external_pooled_development",
        "feature_set": "USGS",
        "seed": 0,
        "site_id": "01000001",
        "horizon": 1,
        "split": "test",
        "issue_date": pd.Timestamp("2020-01-01"),
        "target_date": pd.Timestamp("2020-01-02"),
        "y_true": 10.0,
        "y_pred": 10.0,
        "q05": 9.0,
        "q50": 10.0,
        "q95": 11.0,
        "p_exceed": 0.1,
    }])
    R.write_predictions(frame, prediction)
    run_identity = RunIdentity(
        run_id=run_id,
        panel_sha256=identity_fields["panel_sha256"],
        registry_sha256=identity_fields["registry_sha256"],
        config_sha256=identity_fields["config_sha256"],
        source_sha256=identity_fields["source_sha256"],
        runtime_sha256=identity_fields["runtime_sha256"],
        input_closure_sha256=identity_fields["input_closure_sha256"],
    )
    seal_artifact(
        prediction,
        run_identity,
        kind="external_pooled_development_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
        extra={"common_test_keys": 1, "post_2020_data_read": False},
    )
    prediction_artifact = {
        **file_binding(tmp_path, prediction),
        "sidecar": file_binding(
            tmp_path, MODEL_SUITE.sidecar_path(prediction)
        ),
    }

    metadata = {
        **identity,
        "training_device": "cpu",
        "development_prediction": {"artifact": prediction_artifact},
    }
    models_root = tmp_path / "outputs" / "models"
    entries: list[dict[str, object]] = []
    for model_id, executor, stem in (
        ("ThermoRoute", "thermoroute_bundle", "thermoroute"),
        ("LSTM", "lstm_bundle", "lstm"),
    ):
        directory = models_root / f"external_{stem}_bundle_{run_id}"
        directory.mkdir(parents=True)
        (directory / "weights.pt").write_bytes(f"{model_id} weights".encode())
        (directory / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        entries.append({
            "model_id": model_id,
            "executor": executor,
            "raw_feature_order": list(MODEL_SUITE.STAGE9_USGS_VARIABLES),
            "member_count": 5,
            "artifact": directory_binding(tmp_path, directory),
        })

    lightgbm_dir = models_root / f"external_lightgbm_bundle_{run_id}"
    lightgbm_dir.mkdir(parents=True)
    model_registry: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for seed in C.USGS_SEEDS:
        member = f"seed{seed}"
        model_registry[member] = {}
        for horizon in C.HORIZONS:
            model_registry[member][str(horizon)] = {}
            for head in LIGHTGBM_HEADS:
                path = lightgbm_dir / f"{member}_h{horizon}_{head}.txt"
                path.write_bytes(f"{member}/{horizon}/{head}\n".encode())
                model_registry[member][str(horizon)][head] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
    lightgbm_manifest = lightgbm_dir / "manifest.json"
    lightgbm_manifest.write_text(json.dumps({
        **metadata,
        "models": model_registry,
    }), encoding="utf-8")
    entries.append({
        "model_id": "LightGBM",
        "executor": "lightgbm_bundle",
        "raw_feature_order": list(MODEL_SUITE.STAGE9_USGS_VARIABLES),
        "member_count": 5,
        "artifact": file_binding(tmp_path, lightgbm_manifest),
    })

    def fake_entry_validator(
        root: Path,
        entry: dict[str, object],
        _feature_order: tuple[str, ...],
        *,
        external: bool,
    ) -> dict[str, object]:
        assert root == tmp_path and external is True
        artifact = entry["artifact"]
        assert isinstance(artifact, dict)
        path = tmp_path / str(artifact["path"])
        metadata_path = (
            path / "metadata.json"
            if entry["executor"] != "lightgbm_bundle" else path
        )
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        MODEL_SUITE, "_entry_artifact_valid", fake_entry_validator
    )
    components = models_root / "route_a_external_components.json"
    write_component_pointer(
        components,
        run_id=run_id,
        cohort="external",
        entries=entries,
        raw_feature_order=MODEL_SUITE.STAGE9_USGS_VARIABLES,
        development_contract=development_contract,
        development_prediction_artifact=prediction_artifact,
    )
    receipt_path = tmp_path / STAGE25_COMPLETION_RECEIPT_PATH
    document = build_stage25_completion_receipt(
        root=tmp_path,
        run_id=run_id,
        run_manifest=run_manifest,
        components_pointer=components,
    )
    return receipt_path, components, document, lightgbm_dir / "seed0_h1_point.txt"


def test_stage25_receipt_is_self_hashed_exact_and_atomically_published(
    tmp_path, monkeypatch,
):
    receipt_path, components, document, _model = _stage25_receipt_fixture(
        tmp_path, monkeypatch
    )
    assert document["status"] == "COMPLETE"
    assert document["confirmation_outcomes_requested_or_read"] is False
    artifacts = document["artifacts"]
    assert isinstance(artifacts, dict)
    assert len(artifacts["model_files"]) == 80
    assert document["artifact_closure_sha256"] == sha256_json(artifacts)
    assert not receipt_path.exists()
    publish_stage25_completion_receipt(
        receipt_path,
        document,
        root=tmp_path,
        components_pointer=components,
        publication_guard=lambda: None,
    )
    validated = validate_stage25_completion_receipt(
        receipt_path,
        root=tmp_path,
        components_pointer=components,
    )
    assert validated == document


@pytest.mark.parametrize("attack", ("sibling_symlink", "hardlink_alias"))
def test_stage25_validator_rejects_receipt_link_aliases(
    tmp_path, monkeypatch, attack,
):
    receipt_path, components, document, _model = _stage25_receipt_fixture(
        tmp_path, monkeypatch
    )
    publish_stage25_completion_receipt(
        receipt_path,
        document,
        root=tmp_path,
        components_pointer=components,
        publication_guard=lambda: None,
    )
    alias = receipt_path.with_name(f"stage25-{attack}.json")
    if attack == "sibling_symlink":
        alias.write_bytes(receipt_path.read_bytes())
        receipt_path.unlink()
        receipt_path.symlink_to(alias)
    else:
        os.link(receipt_path, alias)
        assert receipt_path.stat().st_nlink == 2
    with pytest.raises(ModelSuiteError, match="symlink|single-link"):
        validate_stage25_completion_receipt(
            receipt_path,
            root=tmp_path,
            components_pointer=components,
        )


def test_stage25_receipt_writer_guard_failure_publishes_no_marker(
    tmp_path, monkeypatch,
):
    receipt_path, components, document, _model = _stage25_receipt_fixture(
        tmp_path, monkeypatch
    )
    calls = 0

    def reject_at_atomic_boundary() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected receipt publication drift")

    with pytest.raises(RuntimeError, match="receipt publication drift"):
        publish_stage25_completion_receipt(
            receipt_path,
            document,
            root=tmp_path,
            components_pointer=components,
            publication_guard=reject_at_atomic_boundary,
        )
    assert calls == 2
    assert not receipt_path.exists()
    assert not list(receipt_path.parent.glob(f".{receipt_path.name}.*.tmp"))


def test_stage25_receipt_rejects_resealed_incomplete_closure_and_byte_tamper(
    tmp_path, monkeypatch,
):
    receipt_path, components, document, model = _stage25_receipt_fixture(
        tmp_path, monkeypatch
    )
    publish_stage25_completion_receipt(
        receipt_path,
        document,
        root=tmp_path,
        components_pointer=components,
        publication_guard=lambda: None,
    )
    attacked = json.loads(receipt_path.read_text(encoding="utf-8"))
    attacked["artifacts"]["model_files"].pop()
    attacked["artifact_closure_sha256"] = sha256_json(attacked["artifacts"])
    attacked.pop("receipt_self_sha256")
    attacked["receipt_self_sha256"] = sha256_json(attacked)
    receipt_path.write_text(json.dumps(attacked), encoding="utf-8")
    with pytest.raises(ModelSuiteError, match="exact artifact closure"):
        validate_stage25_completion_receipt(
            receipt_path, root=tmp_path, components_pointer=components
        )

    publish_stage25_completion_receipt(
        receipt_path,
        document,
        root=tmp_path,
        components_pointer=components,
        publication_guard=lambda: None,
    )
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(ModelSuiteError, match="checksum"):
        validate_stage25_completion_receipt(
            receipt_path, root=tmp_path, components_pointer=components
        )


def test_stage25_publish_leaves_incomplete_marker_when_preflight_fails(
    tmp_path, monkeypatch,
):
    receipt_path, components, document, model = _stage25_receipt_fixture(
        tmp_path, monkeypatch
    )
    marker = {
        "format": MODEL_SUITE.STAGE25_COMPLETION_FORMAT,
        "status": "INCOMPLETE",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(marker), encoding="utf-8")
    marker_bytes = receipt_path.read_bytes()
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(ModelSuiteError, match="checksum"):
        publish_stage25_completion_receipt(
            receipt_path,
            document,
            root=tmp_path,
            components_pointer=components,
            publication_guard=lambda: None,
        )
    assert receipt_path.read_bytes() == marker_bytes
    with pytest.raises(ModelSuiteError, match="schema is not exact"):
        validate_stage25_completion_receipt(
            receipt_path, root=tmp_path, components_pointer=components
        )


@pytest.mark.parametrize("tamper", ("kind", "run", "extra"))
def test_stage25_receipt_rejects_resealed_prediction_sidecar_semantics(
    tmp_path, monkeypatch, tamper,
):
    _receipt_path, components, document, _model = _stage25_receipt_fixture(
        tmp_path, monkeypatch
    )
    pointer = json.loads(components.read_text(encoding="utf-8"))
    prediction_binding = pointer["development_prediction_artifact"]
    prediction_path = tmp_path / prediction_binding["path"]
    lineage_path = MODEL_SUITE.sidecar_path(prediction_path)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if tamper == "kind":
        lineage["kind"] = "another_prediction_kind"
    elif tamper == "run":
        lineage["run"]["run_id"] = "f" * 20
    else:
        lineage["extra"] = {
            "common_test_keys": 1,
            "post_2020_data_read": True,
        }
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    prediction_binding["sidecar"] = file_binding(tmp_path, lineage_path)
    components.write_text(json.dumps(pointer), encoding="utf-8")

    run_manifest = tmp_path / document["artifacts"]["run_manifest"]["path"]
    with pytest.raises(ModelSuiteError, match="sidecar|lineage"):
        build_stage25_completion_receipt(
            root=tmp_path,
            run_id=str(document["run_id"]),
            run_manifest=run_manifest,
            components_pointer=components,
        )


def test_stage25_writer_check_and_suite_freeze_share_one_transaction_lock():
    stage25 = (
        ROOT / "scripts/25_train_external_pooled_suite.py"
    ).read_text(encoding="utf-8")
    stage24 = (ROOT / "scripts/24_freeze_model_suite.py").read_text(
        encoding="utf-8"
    )
    assert (
        "advisory_file_lock(C.STAGE25_TRANSACTION_LOCK, exclusive=True)"
        in stage25
    )
    assert (
        "advisory_file_lock(C.STAGE25_TRANSACTION_LOCK, exclusive=False)"
        in stage25
    )
    assert (
        "advisory_file_lock(C.STAGE25_TRANSACTION_LOCK, exclusive=False)"
        in stage24
    )
