from __future__ import annotations

import json
from pathlib import Path
import sys

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
from thermoroute import results as R  # noqa: E402
from thermoroute.frozen_inference import lstm_factory_from_metadata  # noqa: E402
from thermoroute.model_suite import (  # noqa: E402
    LIGHTGBM_HEADS,
    MODEL_SUITE_FORMAT,
    ModelSuiteError,
    file_binding,
    development_prediction_binding,
    development_predictor_bridge_binding,
    freeze_model_suite,
    load_lightgbm_bundle,
    save_lightgbm_bundle,
    validate_model_suite_document,
    validate_development_prediction_binding,
    verify_lightgbm_prediction_parity,
    _create_json_or_require_identical,
    _learned_metadata_runtime_sha256,
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
    source_tree_hash,
)


def _lgb_metadata(columns):
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
        "conformal_offsets": {},
        "source_sha256": "s",
        "panel_sha256": "p",
        "registry_sha256": "r",
        "config_sha256": "c",
        "runtime_sha256": "t" * 64,
        "training_device": "cpu",
        "development_prediction": {"path": "predictions.parquet", "sha256": "d"},
    }


def _lgb_audit_inputs(X, horizons=(1, 3, 7)):
    issue_dates = pd.date_range("2019-01-01", periods=len(X), freq="D")
    return {
        horizon: (
            pd.DataFrame({
                "site_id": "fixture-site",
                "split": "test",
                "issue_date": issue_dates,
                "target_date": issue_dates + pd.to_timedelta(horizon, unit="D"),
                "y": 0.0,
            }),
            X,
        )
        for horizon in horizons
    }


def test_lightgbm_native_bundle_reconstructs_all_heads_with_prediction_parity(tmp_path):
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
    evaluation_design = _lgb_audit_inputs(X)
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
                    "y_true": row.y,
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

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["raw_quantile_crossing_audit"]["members"]["seed0"]["1"][
        "any_crossing_rate"
    ] = 0.0
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelSuiteError, match="raw quantile audit self hash"):
        load_lightgbm_bundle(manifest)


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
    document = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "protocol_sha256": "protocol",
        "actual_feature_order": ["WTEMP", "FLOW"],
        "development_contract": development_contract,
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
        )
    assert not current.exists()
    assert not (tmp_path / "suite.json").exists()


def test_model_suite_rejects_source_tree_drift_before_model_validation(tmp_path):
    for name in ("spec.json", "panel.parquet", "registry.csv"):
        (tmp_path / name).write_bytes(name.encode("utf-8"))
    source = tmp_path / "src" / "fixture.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    document = {
        "format": MODEL_SUITE_FORMAT,
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "actual_feature_order": ["WTEMP", "FLOW"],
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


def test_frozen_registry_create_is_idempotent_but_never_overwrites(tmp_path):
    path = tmp_path / "registry.json"
    _create_json_or_require_identical(path, {"format": "fixture", "value": 1})
    original = path.read_bytes()
    _create_json_or_require_identical(path, {"format": "fixture", "value": 1})
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _create_json_or_require_identical(path, {"format": "fixture", "value": 2})
    assert path.read_bytes() == original


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
