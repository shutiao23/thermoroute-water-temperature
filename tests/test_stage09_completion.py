from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import model_suite as MODEL_SUITE  # noqa: E402
from thermoroute import results as R  # noqa: E402
from thermoroute.checkpoint import save_training_checkpoint  # noqa: E402
from thermoroute.model_suite import (  # noqa: E402
    ABLATION_INTERVENTIONS,
    MANDATORY_ABLATIONS,
    STAGE9_ABLATION_SEEDS,
    ModelSuiteError,
    build_stage09_completion_receipt,
    file_binding,
    publish_stage09_completion_receipt,
    validate_stage09_completion_receipt,
    write_component_pointer,
    write_stage09_completion_receipt,
)
from thermoroute.repro import (  # noqa: E402
    RUN_SCHEMA_VERSION,
    RunIdentity,
    atomic_write_json,
    seal_artifact,
    sha256_file,
    sha256_json,
    source_tree_hash,
)
from thermoroute.quantiles import (  # noqa: E402
    LIGHTGBM_QUANTILE_REPAIR_METHOD,
    RAW_QUANTILE_CROSSING_AUDIT_FORMAT,
    lightgbm_quantile_repair_contract,
)
from thermoroute.train import LSTMForecaster  # noqa: E402


def _load_script(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE09 = _load_script("scripts/09_usgs_experiment.py", "stage09_completion_test")
STAGE24 = _load_script("scripts/24_freeze_model_suite.py", "stage24_receipt_test")


class _FixtureDevelopmentInputClosure:
    binding_digest = "f" * 64
    inventory = (object(),)

    @staticmethod
    def assert_unchanged() -> None:
        return None


@pytest.fixture(autouse=True)
def _fixed_development_input_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODEL_SUITE,
        "resolve_development_input_closure",
        lambda _root: _FixtureDevelopmentInputClosure(),
    )


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _rehash_receipt(document: dict[str, Any]) -> None:
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    document["receipt_self_sha256"] = sha256_json(stable)


def _lightgbm_quantile_metadata(
    validation_selection: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "rows": 1,
        "forecast_key_sha256": "a" * 64,
        "raw_prediction_sha256": "b" * 64,
        "q05_above_q50_count": 0,
        "q50_above_q95_count": 0,
        "any_crossing_count": 0,
        "any_crossing_rate": 0.0,
        "maximum_crossing_gap_c": 0.0,
    }
    members = [f"seed{seed}" for seed in C.USGS_SEEDS]
    audit = {
        "format": RAW_QUANTILE_CROSSING_AUDIT_FORMAT,
        "scope": "development_export_rows_before_repair",
        "key_columns": [
            "site_id", "horizon", "split", "issue_date", "target_date",
        ],
        "repair_method": LIGHTGBM_QUANTILE_REPAIR_METHOD,
        "members": {
            member: {
                str(horizon): dict(summary) for horizon in C.HORIZONS
            }
            for member in members
        },
    }
    return {
        "members": members,
        "horizons": list(C.HORIZONS),
        "validation_selection": validation_selection,
        "quantile_repair": lightgbm_quantile_repair_contract(),
        "raw_quantile_crossing_audit": {
            **audit, "audit_sha256": sha256_json(audit),
        },
    }


def _fixture_prediction_frame(*, air2stream: bool) -> pd.DataFrame:
    issue = pd.Timestamp("2020-06-01")
    sites = ("site-a", "site-b")
    errors = {
        # These values exercise pandas decimals that the default fast parser
        # does not recover bit-for-bit; the formal reader must round-trip them.
        "Persistence": 0.9797254087524863,
        "DampedPersistence": 0.80,
        "Climatology": 1.20,
        MODEL_SUITE.STAGE9_LGO_MODEL: 0.75,
        "Air2stream-a4": 0.70,
        "Air2stream-a8": 0.60,
        **{
            name: 0.50 + index * 0.03
            for index, name in enumerate(MANDATORY_ABLATIONS)
        },
    }
    rows: list[dict[str, Any]] = []
    for horizon in C.HORIZONS:
        for site in sites:
            y_true = 0.0
            common = {
                "site_id": site,
                "horizon": horizon,
                "split": "test",
                "issue_date": issue,
                "target_date": issue + pd.Timedelta(days=horizon),
                "y_true": y_true,
            }
            for model in (
                "Persistence", "DampedPersistence", "Climatology",
                *MANDATORY_ABLATIONS, MODEL_SUITE.STAGE9_LGO_MODEL,
                *(MODEL_SUITE.STAGE9_AIR2STREAM_MODELS if air2stream else ()),
            ):
                if model == MODEL_SUITE.STAGE9_LGO_MODEL and site != sites[0]:
                    continue
                model_seeds = (
                    STAGE9_ABLATION_SEEDS
                    if model in MANDATORY_ABLATIONS else (0,)
                )
                for seed in model_seeds:
                    rows.append({
                        "model": model,
                        "scope": "fixture",
                        "feature_set": "USGS",
                        "seed": seed,
                        **common,
                        "y_pred": y_true + errors[model] + 0.01 * seed,
                        "q05": np.nan,
                        "q50": np.nan,
                        "q95": np.nan,
                        "p_exceed": np.nan,
                    })
            for model, base_error in (("ThermoRoute", 0.40), ("LightGBM", 0.55)):
                for seed in C.USGS_SEEDS:
                    rows.append({
                        "model": model,
                        "scope": "fixture",
                        "feature_set": "USGS",
                        "seed": seed,
                        **common,
                        "y_pred": y_true + base_error + 0.01 * seed,
                        "q05": np.nan,
                        "q50": np.nan,
                        "q95": np.nan,
                        "p_exceed": np.nan,
                    })
    return pd.DataFrame(rows, columns=R.PRED_COLS)


def _fixture_selection_frame() -> pd.DataFrame:
    params = (
        (15, 40, 0.03),
        (31, 40, 0.03),
        (63, 40, 0.03),
        (31, 80, 0.05),
    )
    rows = []
    for horizon in C.HORIZONS:
        for candidate_id, (num_leaves, min_child_samples, learning_rate) in enumerate(params):
            rows.append({
                "horizon": horizon,
                "candidate_id": candidate_id,
                "num_leaves": num_leaves,
                "min_child_samples": min_child_samples,
                "learning_rate": learning_rate,
                "val_station_macro_rmse": (
                    1.2891396501744685 + candidate_id / 10
                ),
                "best_iteration": 100 + candidate_id,
                "selected": candidate_id == 0,
                "selection_split": "2016-2017 validation",
            })
    return pd.DataFrame(
        rows, columns=MODEL_SUITE.STAGE9_LIGHTGBM_SELECTION_COLUMNS
    )


def _fixture_report(frame: pd.DataFrame, *, air2stream: bool) -> str:
    scores = MODEL_SUITE._expected_stage09_score_frame(frame)
    headline = MODEL_SUITE._expected_stage09_headline_rows(
        frame, scores, air2stream=air2stream
    )
    modules = MODEL_SUITE._expected_stage09_module_rows(frame)
    lgo = MODEL_SUITE._expected_stage09_lgo_rows(frame)
    header = MODEL_SUITE.STAGE9_AIR2STREAM_DISPLAY_NAME
    lines = [
        "# USGS large-sample experiment (2 stations, 5 seeds)",
        "",
        "_ThermoRoute = 5-seed mean. LightGBM = 5-seed mean._",
        "",
        MODEL_SUITE.stage09_air2stream_report_status(air2stream),
        "",
        f"| horizon | persist | damped | {header} | LightGBM | ThermoRoute | "
        "skill vs persist | skill vs damped | win-rate vs damped |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for horizon in C.HORIZONS:
        lines.append(
            "| " + " | ".join((str(horizon), *headline[str(horizon)])) + " |"
        )
    lines += [
        "",
        "## Random held-station warm-start diagnostic (1→1)",
        "",
        "| horizon | warm-start RMSE | persistence RMSE | warm-start skill |",
        "|---|---|---|---|",
    ]
    for horizon in C.HORIZONS:
        lines.append(
            "| " + " | ".join((str(horizon), *lgo[str(horizon)])) + " |"
        )
    lines += [
        "",
        "## Module ablations (five-seed deletion/intervention sensitivity; "
        "ensemble-mean median per-station RMSE, delta_scale=1.0)",
        "",
        "Audit: every mandatory control contains seeds 0--4 and uses the same "
        "five seeds as ThermoRoute on identical forecast keys and exact y_true. "
        "Interpretation: this is five-seed deletion/intervention sensitivity, "
        "not evidence of module necessity, causal mechanism, or capacity-matched "
        "attribution.",
        "",
        "| variant | h1 | h3 | h7 |",
        "|---|---|---|---|",
    ]
    for model in ("ThermoRoute", *MANDATORY_ABLATIONS):
        lines.append(
            "| " + " | ".join((model, *modules[model])) + " |"
        )
    return "\n".join(lines) + "\n"


def _stage09_fixture(
    root: Path,
    *,
    air2stream: bool = False,
) -> dict[str, Any]:
    source = root / "src" / "fixture.py"
    _write_bytes(source, b"VALUE = 1\n")
    panel = _write_bytes(
        root / "data_usgs" / "panel_usgs_120v2.parquet", b"panel"
    )
    registry = _write_bytes(
        root / "data_usgs" / "station_registry_v1.csv",
        b"site_no\nsite-a\nsite-b\n",
    )
    spec = _write_bytes(root / "data_usgs" / "frozen_panel_v1.json", b"{}\n")
    bridge_path = root / "data_usgs" / "development_predictor_bridge_v1.json"
    bridge = _write_bytes(
        bridge_path,
        json.dumps({
            "format": MODEL_SUITE.DEVELOPMENT_PREDICTOR_BRIDGE_FORMAT,
            "status": "PASS_EXACT_PRODUCT_BRIDGE",
            "outcome_values_requested_or_read": False,
            "panel": {"sha256": sha256_file(panel)},
            "registry": {"sha256": sha256_file(registry)},
        }).encode("utf-8"),
    )
    resolved_config = {
        "stage": "09_usgs_experiment",
        "protocol": MODEL_SUITE.STAGE9_FORMAL_PROTOCOL,
        "panel": "panel_usgs_120v2.parquet",
        "station_registry": "station_registry_v1.csv",
        "variables": list(MODEL_SUITE.STAGE9_USGS_VARIABLES),
        "horizons": list(C.HORIZONS),
        "context_length": C.CONTEXT_LENGTH,
        "seeds": len(C.USGS_SEEDS),
        "execution_role": "route_a_formal_candidate",
        "device": "cpu",
        "training_device": "cpu",
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "delta_scale": 1.0,
        "train_config": MODEL_SUITE.STAGE9_FORMAL_TRAIN_CONFIG,
        "thermoroute_seeds": list(C.USGS_SEEDS),
        "lightgbm_seeds": list(C.USGS_SEEDS),
        "ablation_seeds": list(STAGE9_ABLATION_SEEDS),
        "time_split": C.SPLIT.as_dict(),
        "ablations": True,
        "air2stream": air2stream,
        "eval_batch_size": 64,
        "development_predictor_bridge": file_binding(root, bridge),
        "lightgbm_validation_grid": [
            dict(params)
            for params in MODEL_SUITE.STAGE9_LIGHTGBM_VALIDATION_GRID
        ],
        "event_reference_fit_interval": ["2006-01-01", "2018-12-31"],
        "formal_numerical_policy": {"status": "fixture-formal"},
        "input_closure_sha256": "f" * 64,
        "input_closure_file_count": 1,
    }
    identity_parts = {
        "schema_version": RUN_SCHEMA_VERSION,
        "panel_sha256": sha256_file(panel),
        "registry_sha256": sha256_file(registry),
        "config_sha256": sha256_json(resolved_config),
        "source_sha256": source_tree_hash(root),
        "runtime_sha256": "e" * 64,
        "input_closure_sha256": "f" * 64,
    }
    identity = RunIdentity(
        run_id=sha256_json(identity_parts)[:20],
        **identity_parts,
    )
    run_manifest = (
        root
        / "outputs"
        / "runs"
        / "09_usgs_experiment"
        / identity.run_id
        / "run.json"
    )
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(json.dumps({
        "schema_version": RUN_SCHEMA_VERSION,
        "identity": identity.as_dict(),
        "provenance": {
            "evidence_role": "prelabel_route_a_model_build_development_only",
            "training_device": "cpu",
        },
        "resolved_config": resolved_config,
    }), encoding="utf-8")

    prediction_frame = _fixture_prediction_frame(air2stream=air2stream)
    predictions = (
        root / "outputs" / "predictions" / "usgs_predictions_stage9_v2.parquet"
    )
    R.write_predictions(prediction_frame, predictions)
    seal_artifact(
        predictions,
        identity,
        kind="canonical_stage9_usgs_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    scores = root / "outputs" / "tables" / "usgs_scores.csv"
    scores.parent.mkdir(parents=True)
    expected_scores = MODEL_SUITE._expected_stage09_score_frame(prediction_frame)
    scores.write_text(expected_scores.to_csv(index=False), encoding="utf-8")
    selection = (
        root / "outputs" / "tables" / "lightgbm_joint_validation_selection.csv"
    )
    selection_frame = _fixture_selection_frame()
    selection.write_text(selection_frame.to_csv(index=False), encoding="utf-8")
    report = root / "outputs" / "reports" / "usgs_experiment.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        _fixture_report(prediction_frame, air2stream=air2stream), encoding="utf-8"
    )

    feature_order = ["WTEMP", "FLOW"]
    entries = []
    primary = root / "outputs" / "models" / "thermoroute-fixture"
    primary_metadata = _write_bytes(primary / "metadata.json", b"{}\n")
    primary_weights = _write_bytes(primary / "weights.pt", b"weights")
    entries.append({
        "model_id": "ThermoRoute",
        "executor": "thermoroute_bundle",
        "raw_feature_order": feature_order,
        "member_count": 5,
        "artifact": {
            "path": primary.relative_to(root).as_posix(),
            "metadata_sha256": sha256_file(primary_metadata),
            "weights_sha256": sha256_file(primary_weights),
        },
    })
    lightgbm_manifest = _write_bytes(
        root / "outputs" / "models" / "lightgbm-fixture" / "manifest.json",
        json.dumps(
            _lightgbm_quantile_metadata(
                selection_frame.to_dict(orient="records")
            ),
            sort_keys=True,
        ).encode("utf-8"),
    )
    entries.append({
        "model_id": "LightGBM",
        "executor": "lightgbm_bundle",
        "raw_feature_order": feature_order,
        "member_count": 5,
        "artifact": file_binding(root, lightgbm_manifest),
    })
    for name in MANDATORY_ABLATIONS:
        directory = root / "outputs" / "models" / f"{name}-fixture"
        metadata = _write_bytes(directory / "metadata.json", b"{}\n")
        weights = _write_bytes(directory / "weights.pt", b"weights")
        entries.append({
            "model_id": name,
            "executor": "thermoroute_bundle",
            "raw_feature_order": feature_order,
            "member_count": len(STAGE9_ABLATION_SEEDS),
            "intervention": ABLATION_INTERVENTIONS[name],
            "artifact": {
                "path": directory.relative_to(root).as_posix(),
                "metadata_sha256": sha256_file(metadata),
                "weights_sha256": sha256_file(weights),
            },
        })
    development_contract = {
        "frozen_panel_spec": file_binding(root, spec),
        "panel": file_binding(root, panel),
        "registry": file_binding(root, registry),
        "predictor_bridge": file_binding(root, bridge),
        "source_sha256": identity.source_sha256,
    }
    components_pointer = (
        root / "outputs" / "models" / "route_a_stage9_components.json"
    )
    write_component_pointer(
        components_pointer,
        run_id=identity.run_id,
        cohort="temporal_stage9",
        entries=entries,
        raw_feature_order=feature_order,
        development_contract=development_contract,
        development_prediction_artifact={
            **file_binding(root, predictions),
            "sidecar": file_binding(root, predictions.with_name(
                predictions.name + ".meta.json"
            )),
        },
    )
    thermoroute_pointer = (
        root / "outputs" / "models" / "thermoroute_usgs_bundle.json"
    )
    atomic_write_json(thermoroute_pointer, {
        "run_id": identity.run_id,
        "bundle_path": entries[0]["artifact"]["path"],
        "member_count": 5,
        "metadata_sha256": entries[0]["artifact"]["metadata_sha256"],
        "weights_sha256": entries[0]["artifact"]["weights_sha256"],
    })
    lightgbm_pointer = root / "outputs" / "models" / "lightgbm_usgs_bundle.json"
    atomic_write_json(lightgbm_pointer, {
        "run_id": identity.run_id,
        "manifest": entries[1]["artifact"],
        "member_count": 5,
    })
    receipt_path = (
        root / "outputs" / "models" / "route_a_stage09_completion.json"
    )
    document = build_stage09_completion_receipt(
        root=root,
        run_id=identity.run_id,
        run_manifest=run_manifest,
        predictions=predictions,
        scores=scores,
        report=report,
        lightgbm_selection=selection,
        thermoroute_pointer=thermoroute_pointer,
        lightgbm_pointer=lightgbm_pointer,
        components_pointer=components_pointer,
    )
    write_stage09_completion_receipt(receipt_path, document)
    return {
        "receipt": receipt_path,
        "components": components_pointer,
        "report": report,
        "predictions": predictions,
        "scores": scores,
        "selection": selection,
        "run_manifest": run_manifest,
        "prediction_sidecar": predictions.with_name(
            predictions.name + ".meta.json"
        ),
        "thermoroute_pointer": thermoroute_pointer,
        "lightgbm_pointer": lightgbm_pointer,
        "lightgbm_manifest": lightgbm_manifest,
        "panel": panel,
        "registry": registry,
        "configuration": resolved_config,
        "identity": identity,
        "run_id": identity.run_id,
    }


def _rebind_report(root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    document["artifacts"]["report"] = file_binding(root, fixture["report"])
    _rehash_receipt(document)
    return document


def _rebind_scores(root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    document["artifacts"]["scores"] = file_binding(root, fixture["scores"])
    _rehash_receipt(document)
    return document


def _rebind_selection(
    root: Path,
    fixture: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    fixture["selection"].write_text(frame.to_csv(index=False), encoding="utf-8")
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    document["artifacts"]["lightgbm_selection"] = file_binding(
        root, fixture["selection"]
    )
    _rehash_receipt(document)
    return document


def _rebind_predictions(
    root: Path,
    fixture: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    frame.to_parquet(fixture["predictions"], index=False)
    seal_artifact(
        fixture["predictions"],
        fixture["identity"],
        kind="canonical_stage9_usgs_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    components = json.loads(fixture["components"].read_text(encoding="utf-8"))
    components["development_prediction_artifact"] = {
        **file_binding(root, fixture["predictions"]),
        "sidecar": file_binding(root, fixture["prediction_sidecar"]),
    }
    fixture["components"].write_text(json.dumps(components), encoding="utf-8")
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    for label, path in (
        ("predictions", fixture["predictions"]),
        ("prediction_sidecar", fixture["prediction_sidecar"]),
        ("components_pointer", fixture["components"]),
    ):
        document["artifacts"][label] = file_binding(root, path)
    _rehash_receipt(document)
    return document


def test_thermoroute_ensemble_summary_retains_target_date():
    issue = pd.Timestamp("2020-01-01")
    frame = pd.DataFrame([
        {
            "site_id": "site-a", "horizon": 1, "issue_date": issue,
            "target_date": issue + pd.Timedelta(days=1), "seed": seed,
            "y_pred": float(seed), "y_true": 1.0,
        }
        for seed in (0, 1)
    ])
    summary = STAGE09.thermoroute_ensemble_summary_frame(frame)
    assert "target_date" in summary.columns
    assert summary.loc[0, "y_pred"] == 0.5
    assert STAGE09.rmse_per_station(summary, 1) == {"site-a": 0.5}


def _multiseed_ablation_predictions() -> pd.DataFrame:
    issue = pd.Timestamp("2020-01-01")
    rows = []
    for model in ("ThermoRoute", *MANDATORY_ABLATIONS):
        for seed in STAGE9_ABLATION_SEEDS:
            for offset in (0, 1):
                rows.append({
                    "model": model,
                    "split": "test",
                    "seed": seed,
                    "site_id": f"site-{offset}",
                    "horizon": 1,
                    "issue_date": issue,
                    "target_date": issue + pd.Timedelta(days=1),
                    "y_true": 10.0 + offset,
                    "y_pred": 9.0 + offset + seed,
                })
    return pd.DataFrame(rows)


def test_multiseed_ablation_diagnostic_accepts_complete_paired_grid():
    frames = STAGE09.multiseed_ablation_diagnostic_frames(
        _multiseed_ablation_predictions()
    )
    assert tuple(frames) == ("ThermoRoute", *MANDATORY_ABLATIONS)
    assert set(frames["ThermoRoute"]["seed"]) == set(STAGE9_ABLATION_SEEDS)
    assert all(
        set(frames[name]["seed"]) == set(STAGE9_ABLATION_SEEDS)
        for name in MANDATORY_ABLATIONS
    )
    assert all(len(frames[name]) == 10 for name in frames)


def test_multiseed_ablation_diagnostic_rejects_missing_control_seed():
    frame = _multiseed_ablation_predictions()
    control = MANDATORY_ABLATIONS[0]
    frame = frame.loc[
        ~(frame["model"].eq(control) & frame["seed"].eq(
            STAGE9_ABLATION_SEEDS[-1]
        ))
    ].copy()
    with pytest.raises(ValueError, match=f"{control} must contain the exact seed"):
        STAGE09.multiseed_ablation_diagnostic_frames(frame)


def test_multiseed_ablation_diagnostic_rejects_same_seed_key_drift():
    frame = _multiseed_ablation_predictions()
    control = MANDATORY_ABLATIONS[1]
    row = frame.index[
        frame["model"].eq(control) & frame["seed"].eq(2)
    ][0]
    frame.loc[row, "site_id"] = "site-other"
    with pytest.raises(ValueError, match=f"{control} same-seed forecast keys differ"):
        STAGE09.multiseed_ablation_diagnostic_frames(frame)


def test_multiseed_ablation_diagnostic_rejects_same_seed_y_true_drift():
    frame = _multiseed_ablation_predictions()
    control = MANDATORY_ABLATIONS[2]
    row = frame.index[
        frame["model"].eq(control) & frame["seed"].eq(3)
    ][0]
    frame.loc[row, "y_true"] += 0.25
    with pytest.raises(ValueError, match=f"{control} same-seed y_true differs"):
        STAGE09.multiseed_ablation_diagnostic_frames(frame)


def test_air2stream_display_is_explicitly_unofficial_and_non_primary():
    assert STAGE09.AIR2STREAM_DISPLAY_NAME == (
        "Air2stream-style a4/a8 (unofficial, non-primary)"
    )
    source = (ROOT / "scripts" / "09_usgs_experiment.py").read_text(
        encoding="utf-8"
    )
    assert "help=f\"add {AIR2STREAM_DISPLAY_NAME}" in source


def test_report_failure_does_not_publish_formal_pointer_or_receipt(tmp_path):
    pointer = tmp_path / "pointer.json"
    receipt = tmp_path / "receipt.json"
    pointer.write_bytes(b"previous pointer\n")
    receipt.write_bytes(b"previous receipt\n")
    before = (pointer.read_bytes(), receipt.read_bytes())
    calls = []

    def fail_report() -> None:
        calls.append("report")
        raise OSError("injected report failure")

    def publish_pointers() -> None:
        calls.append("pointers")
        pointer.write_bytes(b"new pointer\n")

    def publish_receipt() -> Path:
        calls.append("receipt")
        receipt.write_bytes(b"new receipt\n")
        return receipt

    with pytest.raises(OSError, match="injected report failure"):
        STAGE09.complete_stage09_transaction(
            write_report=fail_report,
            validate_outputs=lambda: calls.append("validate"),
            publish_pointers=publish_pointers,
            publish_receipt=publish_receipt,
        )
    assert calls == ["report"]
    assert (pointer.read_bytes(), receipt.read_bytes()) == before


def test_successful_transaction_publishes_receipt_last_and_returns_cleanly(tmp_path):
    report = tmp_path / "report.md"
    pointer = tmp_path / "pointer.json"
    receipt = tmp_path / "receipt.json"
    calls = []

    def write_report() -> None:
        calls.append("report")
        report.write_text("complete\n", encoding="utf-8")

    def publish_pointers() -> None:
        assert report.is_file()
        calls.append("pointers")
        pointer.write_text("{}\n", encoding="utf-8")

    def validate_outputs() -> None:
        assert report.is_file()
        calls.append("validate")

    def publish_receipt() -> Path:
        assert report.is_file() and pointer.is_file()
        calls.append("receipt")
        receipt.write_text("{}\n", encoding="utf-8")
        return receipt

    result = STAGE09.complete_stage09_transaction(
        write_report=write_report,
        validate_outputs=validate_outputs,
        publish_pointers=publish_pointers,
        publish_receipt=publish_receipt,
    )
    assert result == receipt
    assert calls == ["report", "validate", "pointers", "receipt"]


def test_semantic_preflight_failure_does_not_publish_pointer_or_receipt(tmp_path):
    pointer = tmp_path / "pointer.json"
    receipt = tmp_path / "receipt.json"
    calls = []

    def fail_validation() -> None:
        calls.append("validate")
        raise ModelSuiteError("injected semantic failure")

    with pytest.raises(ModelSuiteError, match="injected semantic failure"):
        STAGE09.complete_stage09_transaction(
            write_report=lambda: calls.append("report"),
            validate_outputs=fail_validation,
            publish_pointers=lambda: pointer.write_text("new", encoding="utf-8"),
            publish_receipt=lambda: receipt,
        )
    assert calls == ["report", "validate"]
    assert not pointer.exists()
    assert not receipt.exists()


def test_semantic_validation_failure_does_not_replace_completion_receipt(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    before = fixture["receipt"].read_bytes()
    document = json.loads(before)
    fixture["report"].write_text("incomplete report\n", encoding="utf-8")
    document["artifacts"]["report"] = file_binding(tmp_path, fixture["report"])
    _rehash_receipt(document)

    with pytest.raises(ModelSuiteError, match="report is incomplete"):
        publish_stage09_completion_receipt(
            fixture["receipt"],
            document,
            root=tmp_path,
            stage9_pointer=fixture["components"],
            publication_guard=lambda: None,
        )
    assert fixture["receipt"].read_bytes() == before


def test_stage09_receipt_writer_guard_failure_preserves_authoritative_bytes(
    tmp_path,
):
    fixture = _stage09_fixture(tmp_path)
    before = fixture["receipt"].read_bytes()
    document = json.loads(before)
    calls = 0

    def reject_at_atomic_boundary() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected receipt publication drift")

    with pytest.raises(RuntimeError, match="receipt publication drift"):
        publish_stage09_completion_receipt(
            fixture["receipt"],
            document,
            root=tmp_path,
            stage9_pointer=fixture["components"],
            publication_guard=reject_at_atomic_boundary,
        )
    assert calls == 2
    assert fixture["receipt"].read_bytes() == before
    assert not list(
        fixture["receipt"].parent.glob(f".{fixture['receipt'].name}.*.tmp")
    )


def test_stage09_receipt_rejects_single_seed_ablation_report_claim(
    tmp_path,
):
    fixture = _stage09_fixture(tmp_path)
    before = fixture["receipt"].read_bytes()
    document = json.loads(before)
    report_text = fixture["report"].read_text(encoding="utf-8")
    report_text = report_text.replace(
        "## Module ablations (five-seed deletion/intervention sensitivity; "
        "ensemble-mean median per-station RMSE, delta_scale=1.0)",
        "## Module ablations (single-seed diagnostic; seed0-vs-seed0)",
    ).replace(
        "Audit: every mandatory control contains seeds 0--4 and uses the same "
        "five seeds as ThermoRoute on identical forecast keys and exact y_true. "
        "Interpretation: this is five-seed deletion/intervention sensitivity, "
        "not evidence of module necessity, causal mechanism, or capacity-matched "
        "attribution.",
        "Audit: seed0-only diagnostic.",
    )
    fixture["report"].write_text(report_text, encoding="utf-8")
    document["artifacts"]["report"] = file_binding(tmp_path, fixture["report"])
    _rehash_receipt(document)

    with pytest.raises(ModelSuiteError, match="five-seed ablation contract"):
        publish_stage09_completion_receipt(
            fixture["receipt"],
            document,
            root=tmp_path,
            stage9_pointer=fixture["components"],
            publication_guard=lambda: None,
        )
    assert fixture["receipt"].read_bytes() == before


@pytest.mark.parametrize(
    "required_disclosure",
    (
        "every mandatory control contains seeds 0--4",
        (
            "not evidence of module necessity, causal mechanism, or "
            "capacity-matched attribution"
        ),
    ),
)
def test_stage09_receipt_requires_five_seed_and_noncausal_disclosure(
    tmp_path,
    required_disclosure,
):
    fixture = _stage09_fixture(tmp_path)
    before = fixture["receipt"].read_bytes()
    document = json.loads(before)
    report_text = fixture["report"].read_text(encoding="utf-8")
    assert required_disclosure in report_text
    fixture["report"].write_text(
        report_text.replace(required_disclosure, "disclosure removed"),
        encoding="utf-8",
    )
    document["artifacts"]["report"] = file_binding(tmp_path, fixture["report"])
    _rehash_receipt(document)

    with pytest.raises(ModelSuiteError, match="five-seed ablation contract"):
        publish_stage09_completion_receipt(
            fixture["receipt"],
            document,
            root=tmp_path,
            stage9_pointer=fixture["components"],
            publication_guard=lambda: None,
        )
    assert fixture["receipt"].read_bytes() == before


def test_stage09_receipt_roundtrip_and_stage24_gate(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    receipt = validate_stage09_completion_receipt(
        fixture["receipt"], root=tmp_path, stage9_pointer=fixture["components"]
    )
    assert receipt["run_id"] == fixture["run_id"]
    stage9, receipt_binding = STAGE24._load_verified_stage9(
        fixture["components"], fixture["receipt"], root=tmp_path
    )
    assert stage9["run_id"] == fixture["run_id"]
    assert receipt_binding == file_binding(tmp_path, fixture["receipt"])


def test_stage09_receipt_rejects_control_bundle_missing_a_seed(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    components = json.loads(fixture["components"].read_text(encoding="utf-8"))
    control = next(
        entry for entry in components["models"]
        if entry["model_id"] == MANDATORY_ABLATIONS[0]
    )
    control["member_count"] = len(STAGE9_ABLATION_SEEDS) - 1
    atomic_write_json(fixture["components"], components)
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    document["artifacts"]["components_pointer"] = file_binding(
        tmp_path, fixture["components"]
    )
    _rehash_receipt(document)

    with pytest.raises(ModelSuiteError, match="component registry changed"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_score_reader_preserves_shortest_decimal_roundtrip(tmp_path):
    tokens = (
        "0.22633884081490238",
        "0.20911275613220767",
        "0.21833853837531267",
        "0.24576947854503056",
    )
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "horizon,site,rmse_persist,rmse_damped,rmse_thermo\n"
        f"1,02334430,{tokens[0]},{tokens[1]},{tokens[3]}\n"
        f"1,09380000,{tokens[0]},{tokens[2]},{tokens[3]}\n",
        encoding="utf-8",
    )

    frame = MODEL_SUITE._read_stage09_score_frame(scores)
    actual = frame[
        ["rmse_persist", "rmse_damped", "rmse_thermo"]
    ].to_numpy(float)
    expected = np.array([
        [float(tokens[0]), float(tokens[1]), float(tokens[3])],
        [float(tokens[0]), float(tokens[2]), float(tokens[3])],
    ])

    np.testing.assert_array_max_ulp(actual, expected, maxulp=0)
    assert frame["site"].tolist() == ["02334430", "09380000"]


def test_stage09_selection_reader_preserves_bound_validation_metrics(tmp_path):
    tokens = ("1.2891396527254995", "1.6761344986272013")
    selection = tmp_path / "selection.csv"
    selection.write_text(
        "horizon,val_station_macro_rmse\n"
        f"1,{tokens[0]}\n"
        f"3,{tokens[1]}\n",
        encoding="utf-8",
    )

    frame = MODEL_SUITE._read_stage09_lightgbm_selection_frame(selection)
    actual = frame["val_station_macro_rmse"].to_numpy(float)
    expected = np.array([float(token) for token in tokens])

    np.testing.assert_array_max_ulp(actual, expected, maxulp=0)


def test_stage09_receipt_rejects_adjacent_float_score_substitution(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    scores = MODEL_SUITE._read_stage09_score_frame(fixture["scores"])
    scores.loc[0, "rmse_persist"] = np.nextafter(
        float(scores.loc[0, "rmse_persist"]), np.inf
    )
    fixture["scores"].write_text(
        scores.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ),
        encoding="utf-8",
    )
    document = _rebind_scores(tmp_path, fixture)

    with pytest.raises(ModelSuiteError, match="bound predictions"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_rejects_adjacent_float_selection_substitution(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    selection = MODEL_SUITE._read_stage09_lightgbm_selection_frame(
        fixture["selection"]
    )
    selection.loc[0, "val_station_macro_rmse"] = np.nextafter(
        float(selection.loc[0, "val_station_macro_rmse"]), np.inf
    )
    document = _rebind_selection(tmp_path, fixture, selection)

    with pytest.raises(ModelSuiteError, match="bound bundle manifest"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_rejects_shuffled_lightgbm_grid_rows(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    selection = MODEL_SUITE._read_stage09_lightgbm_selection_frame(
        fixture["selection"]
    )
    selection.iloc[[0, 1]] = selection.iloc[[1, 0]].to_numpy()
    document = _rebind_selection(tmp_path, fixture, selection)

    with pytest.raises(ModelSuiteError, match="row order"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_rejects_changed_lightgbm_grid_parameters(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    selection = MODEL_SUITE._read_stage09_lightgbm_selection_frame(
        fixture["selection"]
    )
    selection.loc[selection["candidate_id"].eq(0), "num_leaves"] = 16
    document = _rebind_selection(tmp_path, fixture, selection)

    with pytest.raises(ModelSuiteError, match="frozen grid"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_rejects_non_argmin_lightgbm_selection(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    selection = MODEL_SUITE._read_stage09_lightgbm_selection_frame(
        fixture["selection"]
    )
    horizon = C.HORIZONS[0]
    current = selection["horizon"].eq(horizon)
    selection.loc[current, "selected"] = False
    selection.loc[
        current & selection["candidate_id"].eq(1), "selected"
    ] = True
    document = _rebind_selection(tmp_path, fixture, selection)

    with pytest.raises(ModelSuiteError, match="deterministic validation argmin"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_cross_checks_selection_against_bound_bundle(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    selection = MODEL_SUITE._read_stage09_lightgbm_selection_frame(
        fixture["selection"]
    )
    selection.loc[0, "val_station_macro_rmse"] += 0.001
    document = _rebind_selection(tmp_path, fixture, selection)

    with pytest.raises(ModelSuiteError, match="bound bundle manifest"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_rejects_negative_lightgbm_validation_rmse(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    selection = MODEL_SUITE._read_stage09_lightgbm_selection_frame(
        fixture["selection"]
    )
    selection.loc[0, "val_station_macro_rmse"] = -0.1
    document = _rebind_selection(tmp_path, fixture, selection)

    with pytest.raises(ModelSuiteError, match="selection values are invalid"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_binds_canonical_panel_bytes(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    fixture["panel"].write_bytes(b"different panel bytes")

    with pytest.raises(ModelSuiteError, match="canonical panel or station registry"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
        )


@pytest.mark.parametrize(("field", "bad_value"), [
    ("protocol", "forged-protocol"),
    ("panel", "other.parquet"),
    ("station_registry", "other.csv"),
    ("variables", ["WTEMP", "FLOW"]),
    ("horizons", [1, 3]),
    ("context_length", C.CONTEXT_LENGTH + 1),
    ("seeds", len(C.USGS_SEEDS) - 1),
    ("ablation_seeds", list(STAGE9_ABLATION_SEEDS[:-1])),
    ("selection_metric", "micro"),
    ("air2stream", True),
    ("device", "mps"),
])
def test_stage09_manifest_rejects_reidentified_noncanonical_config(
    tmp_path,
    field,
    bad_value,
):
    fixture = _stage09_fixture(tmp_path)
    manifest = json.loads(fixture["run_manifest"].read_text(encoding="utf-8"))
    manifest["resolved_config"][field] = bad_value
    identity = manifest["identity"]
    identity["config_sha256"] = sha256_json(manifest["resolved_config"])
    identity_parts = {
        "schema_version": identity["schema_version"],
        **{
            name: identity[name]
            for name in (
                    "panel_sha256", "registry_sha256", "config_sha256",
                    "source_sha256", "runtime_sha256",
                    "input_closure_sha256",
            )
        },
    }
    identity["run_id"] = sha256_json(identity_parts)[:20]
    forged_manifest = (
        tmp_path
        / "outputs"
        / "runs"
        / "09_usgs_experiment"
        / identity["run_id"]
        / "run.json"
    )
    forged_manifest.parent.mkdir(parents=True)
    forged_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelSuiteError, match="formal configuration"):
        MODEL_SUITE._load_formal_stage09_manifest(
            forged_manifest,
            root=tmp_path,
            run_id=identity["run_id"],
        )


def test_stage09_receipt_rejects_substituted_lgo_held_site(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    frame = pd.read_parquet(fixture["predictions"])
    lgo = frame["model"].eq(MODEL_SUITE.STAGE9_LGO_MODEL)
    frame.loc[lgo, "site_id"] = "site-b"
    frame.loc[lgo, "y_true"] += 1.0
    frame.loc[lgo, "y_pred"] += 1.0
    document = _rebind_predictions(tmp_path, fixture, frame)

    with pytest.raises(ModelSuiteError, match="frozen held-site registry"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_rejects_missing_raw_quantile_audit(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    manifest = json.loads(
        fixture["lightgbm_manifest"].read_text(encoding="utf-8")
    )
    manifest.pop("raw_quantile_crossing_audit")
    fixture["lightgbm_manifest"].write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    new_binding = file_binding(tmp_path, fixture["lightgbm_manifest"])

    components = json.loads(fixture["components"].read_text(encoding="utf-8"))
    lightgbm_entry = next(
        entry for entry in components["models"]
        if entry["model_id"] == "LightGBM"
    )
    lightgbm_entry["artifact"] = new_binding
    fixture["components"].write_text(
        json.dumps(components), encoding="utf-8"
    )
    pointer = json.loads(
        fixture["lightgbm_pointer"].read_text(encoding="utf-8")
    )
    pointer["manifest"] = new_binding
    fixture["lightgbm_pointer"].write_text(
        json.dumps(pointer), encoding="utf-8"
    )
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt["artifacts"]["components_pointer"] = file_binding(
        tmp_path, fixture["components"]
    )
    receipt["artifacts"]["lightgbm_pointer"] = file_binding(
        tmp_path, fixture["lightgbm_pointer"]
    )
    _rehash_receipt(receipt)
    with pytest.raises(ModelSuiteError, match="raw quantile audit schema"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=receipt,
        )


def test_stage09_receipt_rejects_manifest_config_hash_or_run_id_forgery(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    manifest = json.loads(fixture["run_manifest"].read_text(encoding="utf-8"))
    manifest["resolved_config"]["eval_batch_size"] += 1
    fixture["run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    document["artifacts"]["run_manifest"] = file_binding(
        tmp_path, fixture["run_manifest"]
    )
    _rehash_receipt(document)
    with pytest.raises(ModelSuiteError, match="identity is malformed"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )

    fixture = _stage09_fixture(tmp_path / "run-id-forgery")
    manifest = json.loads(fixture["run_manifest"].read_text(encoding="utf-8"))
    manifest["identity"]["run_id"] = "0" * 20
    fixture["run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    document["artifacts"]["run_manifest"] = file_binding(
        tmp_path / "run-id-forgery", fixture["run_manifest"]
    )
    _rehash_receipt(document)
    with pytest.raises(ModelSuiteError, match="run id is not derived"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path / "run-id-forgery",
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_stage09_receipt_requires_full_prediction_sidecar_identity(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    sidecar = json.loads(
        fixture["prediction_sidecar"].read_text(encoding="utf-8")
    )
    sidecar["run"]["panel_sha256"] = "f" * 64
    fixture["prediction_sidecar"].write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    document["artifacts"]["prediction_sidecar"] = file_binding(
        tmp_path, fixture["prediction_sidecar"]
    )
    _rehash_receipt(document)
    with pytest.raises(ModelSuiteError, match="prediction binding differs"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
            document=document,
        )


def test_suite_identity_and_frozen_document_bind_all_four_completion_receipts(
    tmp_path, monkeypatch,
):
    receipt = _write_bytes(tmp_path / "outputs" / "receipt.json", b"receipt\n")
    gate = file_binding(tmp_path, receipt)
    controls_receipt = _write_bytes(
        tmp_path / "outputs" / "controls-receipt.json", b"controls receipt\n"
    )
    controls_gate = file_binding(tmp_path, controls_receipt)
    stage16_receipt = _write_bytes(
        tmp_path / "outputs" / "stage16-receipt.json", b"stage16 receipt\n"
    )
    stage16_gate = file_binding(tmp_path, stage16_receipt)
    stage25_receipt = _write_bytes(
        tmp_path / "outputs" / "stage25-receipt.json", b"stage25 receipt\n"
    )
    stage25_gate = file_binding(tmp_path, stage25_receipt)
    common = {
        "protocol_sha256": "a" * 64,
        "stage9": {"run_id": "stage9"},
        "lstm": {"run_id": "lstm"},
        "external": {"run_id": "external"},
        "features": ("WTEMP", "FLOW"),
    }
    first_id = STAGE24._model_suite_id(
        **common,
        stage09_completion=gate,
        stage09b_completion=controls_gate,
        stage16_completion=stage16_gate,
        stage25_completion=stage25_gate,
    )
    second_id = STAGE24._model_suite_id(
        **common,
        stage09_completion={**gate, "sha256": "f" * 64},
        stage09b_completion=controls_gate,
        stage16_completion=stage16_gate,
        stage25_completion=stage25_gate,
    )
    assert first_id != second_id
    third_id = STAGE24._model_suite_id(
        **common,
        stage09_completion=gate,
        stage09b_completion={**controls_gate, "sha256": "e" * 64},
        stage16_completion=stage16_gate,
        stage25_completion=stage25_gate,
    )
    assert first_id != third_id
    fourth_id = STAGE24._model_suite_id(
        **common,
        stage09_completion=gate,
        stage09b_completion=controls_gate,
        stage16_completion={**stage16_gate, "sha256": "d" * 64},
        stage25_completion=stage25_gate,
    )
    assert first_id != fourth_id
    fifth_id = STAGE24._model_suite_id(
        **common,
        stage09_completion=gate,
        stage09b_completion=controls_gate,
        stage16_completion=stage16_gate,
        stage25_completion={**stage25_gate, "sha256": "d" * 64},
    )
    assert first_id != fifth_id

    monkeypatch.setattr(
        MODEL_SUITE, "_learned_metadata_runtime_sha256",
        lambda _root, _entries, *, publication_guard=None: "b" * 64,
    )
    monkeypatch.setattr(
        MODEL_SUITE, "validate_model_suite_document",
        lambda _document, *, root, publication_guard=None: None,
    )
    destination = tmp_path / "outputs" / "suite.json"
    MODEL_SUITE.freeze_model_suite(
        destination,
        tmp_path / "outputs" / "current.json",
        root=tmp_path,
        protocol_sha256="a" * 64,
        temporal_entries=[],
        external_entries=[],
        actual_feature_order=("WTEMP", "FLOW"),
        development_contract={},
        stage09_completion=gate,
        stage09b_completion=controls_gate,
        stage16_completion=stage16_gate,
        stage25_completion=stage25_gate,
        publication_guard=lambda: None,
    )
    frozen = json.loads(destination.read_text(encoding="utf-8"))
    assert frozen["preopening_gates"] == {
        "stage09_completion": gate,
        "stage09b_development_controls": controls_gate,
        "stage16_lstm_completion": stage16_gate,
        "stage25_external_completion": stage25_gate,
    }


def test_frozen_suite_rejects_entries_from_another_stage09_closure(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    stage9 = MODEL_SUITE.load_component_pointer(fixture["components"])
    temporal_entries = [dict(entry) for entry in stage9["models"]]
    mismatched = json.loads(json.dumps(temporal_entries))
    thermoroute = next(
        entry for entry in mismatched if entry["model_id"] == "ThermoRoute"
    )
    thermoroute["artifact"]["weights_sha256"] = "f" * 64

    with pytest.raises(ModelSuiteError, match="differ from its completion receipt"):
        MODEL_SUITE._validate_stage09_suite_alignment(
            stage9,
            mismatched,
            stage9["development_contract"],
        )

    changed_contract = {
        **stage9["development_contract"],
        "source_sha256": "f" * 64,
    }
    with pytest.raises(ModelSuiteError, match="differ from its completion receipt"):
        MODEL_SUITE._validate_stage09_suite_alignment(
            stage9,
            temporal_entries,
            changed_contract,
        )


def test_stage24_fails_closed_on_missing_or_stale_stage09_receipt(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    fixture["report"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ModelSuiteError, match="checksum|canonical path"):
        STAGE24._load_verified_stage9(
            fixture["components"], fixture["receipt"], root=tmp_path
        )
    fixture["receipt"].unlink()
    with pytest.raises(ModelSuiteError, match="absent or invalid"):
        STAGE24._load_verified_stage9(
            fixture["components"], fixture["receipt"], root=tmp_path
        )


def test_stage24_rejects_changed_or_substituted_stage09_pointer(tmp_path):
    fixture = _stage09_fixture(tmp_path)
    substituted = fixture["components"].with_name("other-components.json")
    substituted.write_bytes(fixture["components"].read_bytes())
    with pytest.raises(ModelSuiteError, match="binds another component pointer"):
        STAGE24._load_verified_stage9(
            substituted, fixture["receipt"], root=tmp_path
        )

    fixture["components"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelSuiteError, match="checksum|canonical path"):
        STAGE24._load_verified_stage9(
            fixture["components"], fixture["receipt"], root=tmp_path
        )


@pytest.mark.parametrize("attack", ("sibling_symlink", "hardlink_alias"))
def test_stage09_validator_rejects_receipt_link_aliases(tmp_path, attack):
    fixture = _stage09_fixture(tmp_path)
    alias = fixture["receipt"].with_name(f"stage09-{attack}.json")
    if attack == "sibling_symlink":
        alias.write_bytes(fixture["receipt"].read_bytes())
        fixture["receipt"].unlink()
        fixture["receipt"].symlink_to(alias)
    else:
        os.link(fixture["receipt"], alias)
        assert fixture["receipt"].stat().st_nlink == 2
    with pytest.raises(ModelSuiteError, match="symlink|single-link"):
        validate_stage09_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            stage9_pointer=fixture["components"],
        )


def test_stage24_rejects_noncanonical_protocol_before_loading_or_publication(
    tmp_path, monkeypatch,
):
    alternate = _write_bytes(tmp_path / "alternate-protocol.json", b"{}\n")
    monkeypatch.setattr(STAGE24, "_assert_stage24_policy", lambda: object())
    monkeypatch.setattr(
        STAGE24,
        "_load_verified_stage9",
        lambda *_args, **_kwargs: pytest.fail(
            "Stage 24 loaded model authority before rejecting protocol"
        ),
    )
    monkeypatch.setattr(
        STAGE24,
        "freeze_model_suite",
        lambda *_args, **_kwargs: pytest.fail(
            "Stage 24 published a suite for a noncanonical protocol"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["24_freeze_model_suite.py", "--protocol", str(alternate)],
    )
    with pytest.raises(ModelSuiteError, match="protocol path is not canonical"):
        STAGE24._run()


def _stage16_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Create a byte-real Stage09→Stage16 closure without expensive inference."""
    stage09 = _stage09_fixture(root)
    parent = stage09["predictions"]
    parent_frame = pd.read_parquet(parent)
    sites = tuple(sorted(parent_frame["site_id"].astype(str).unique()))
    monkeypatch.setattr(MODEL_SUITE.C, "STATIONS", sites)
    runtime_contract = {"fixture_numerical_runtime": "cpu-single-thread"}
    monkeypatch.setattr(
        MODEL_SUITE, "numerical_runtime_contract", lambda: runtime_contract
    )
    fixture_thresholds = {site: 1.0 for site in C.STATIONS}
    fixture_threshold_registry = {
        site: fixture_thresholds[site] for site in sorted(fixture_thresholds)
    }
    fixture_threshold_contract = {
        "target": "WTEMP",
        "fit_split": "canonical development train mask",
        "scope": "station-specific",
        "estimator": "pandas Series.quantile(q=0.90, interpolation=linear)",
        "quantile": 0.90,
        "registry": fixture_threshold_registry,
        "registry_sha256": sha256_json(fixture_threshold_registry),
    }
    monkeypatch.setattr(
        MODEL_SUITE,
        "_stage16_replay_inputs",
        lambda _root, *, build_windows: (
            object() if build_windows else None,
            dict(fixture_thresholds),
            dict(fixture_threshold_contract),
        ),
    )
    monkeypatch.setattr(
        MODEL_SUITE,
        "_verify_stage16_candidate_checkpoint",
        lambda **_kwargs: 0.0,
    )
    input_closure_sha256 = MODEL_SUITE.compose_input_closure_digest({
        "development": stage09["identity"].input_closure_sha256,
        "stage09_parent_prediction": sha256_file(parent),
        "stage09_parent_sidecar": sha256_file(MODEL_SUITE.sidecar_path(parent)),
        "stage09_completion_receipt": sha256_file(stage09["receipt"]),
        "stage09_components": sha256_file(stage09["components"]),
    })
    configuration = {
        "stage": "16_lstm_baseline_insample",
        "role": "final_route_a_development_predictions",
        "parent_sha256": sha256_file(parent),
        "models": list(MODEL_SUITE.ROUTE_A_PRIMARY_MODELS),
        "seeds": list(C.USGS_SEEDS),
        "variables": list(MODEL_SUITE.STAGE9_USGS_VARIABLES),
        "horizons": list(C.HORIZONS),
        "context_length": C.CONTEXT_LENGTH,
        "station_embedding": True,
        "station_balanced": True,
        "selection_metric": "station_macro",
        "validation_grid": [
            dict(value) for value in MODEL_SUITE.LSTM_VALIDATION_GRID
        ],
        "validation_selection_seed": C.USGS_SEEDS[0],
        "validation_selection_split": "2016-2017 only",
        "event_reference_fit_interval": ["2006-01-01", "2018-12-31"],
        "train_config": MODEL_SUITE.asdict(C.TrainConfig(batch_size=1536)),
        "training_device": "cpu",
        "formal_numerical_policy": {"worker_threads": 1},
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": (
            stage09["configuration"]["input_closure_file_count"] + 4
        ),
    }
    identity_fields = {
        "schema_version": RUN_SCHEMA_VERSION,
        "panel_sha256": sha256_file(stage09["panel"]),
        "registry_sha256": sha256_file(stage09["registry"]),
        "config_sha256": sha256_json(configuration),
        "source_sha256": source_tree_hash(root),
        "runtime_sha256": sha256_json(runtime_contract),
        "input_closure_sha256": input_closure_sha256,
    }
    identity = RunIdentity(
        run_id=sha256_json(identity_fields)[:20], **identity_fields
    )
    run_manifest = (
        root / "outputs" / "runs" / "16_lstm_baseline"
        / identity.run_id / "run.json"
    )
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(json.dumps({
        "schema_version": RUN_SCHEMA_VERSION,
        "identity": identity.as_dict(),
        "resolved_config": configuration,
        "created_utc": "2026-07-25T00:00:00+00:00",
        "environment": {},
        "git": {},
        "provenance": {
            "evidence_role": "prelabel_route_a_model_build_development_only",
            "training_device": "cpu",
        },
    }), encoding="utf-8")

    raw_frames: list[pd.DataFrame] = []
    for seed_index, seed in enumerate(C.USGS_SEEDS):
        rows: list[dict[str, Any]] = []
        for split, issue in (
            ("val", pd.Timestamp("2017-06-01")),
            ("calib", pd.Timestamp("2018-06-01")),
            ("test", pd.Timestamp("2020-06-01")),
        ):
            for site in sites:
                for horizon in C.HORIZONS:
                    y_true = 0.0
                    point = 0.25 + 0.01 * seed_index
                    rows.append({
                        "model": "LSTM",
                        "scope": "joint_usgs",
                        "feature_set": "USGS",
                        "seed": seed,
                        "site_id": site,
                        "horizon": horizon,
                        "split": split,
                        "issue_date": issue,
                        "target_date": issue + pd.Timedelta(days=horizon),
                        "y_true": y_true,
                        "y_pred": point,
                        "q05": point - 0.5,
                        "q50": point,
                        "q95": point + 0.5,
                        "p_exceed": 0.2,
                    })
        frame = pd.DataFrame(rows, columns=R.PRED_COLS)
        path = (
            run_manifest.parent / "predictions" / f"seed{seed}.parquet"
        )
        R.write_predictions(frame, path)
        seal_artifact(
            path,
            identity,
            kind="lstm_seed_predictions",
            schema=R.PREDICTION_SCHEMA_VERSION,
        )
        raw_frames.append(frame)

    raw_lstm = pd.concat(raw_frames, ignore_index=True)
    candidate = pd.concat([parent_frame, raw_lstm], ignore_index=True)
    final, audit = MODEL_SUITE.enforce_common_forecast_keys(
        candidate, MODEL_SUITE.ROUTE_A_PRIMARY_MODELS, split="test"
    )
    prediction = root / MODEL_SUITE.STAGE16_DEVELOPMENT_PREDICTION_PATH
    R.write_predictions(final, prediction)
    seal_artifact(
        prediction,
        identity,
        kind="final_route_a_development_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents={parent.name: sha256_file(parent)},
        extra={
            "parent_run_id": stage09["run_id"],
            "primary_models": MODEL_SUITE.ROUTE_A_PRIMARY_MODELS,
            "primary_common_test_keys": audit.common_unique,
            "dropped_primary_rows": audit.dropped_rows,
            "lstm_validation_rows": int(raw_lstm["split"].eq("val").sum()),
            "lstm_calibration_rows": int(
                raw_lstm["split"].eq("calib").sum()
            ),
        },
    )

    selection_rows: list[dict[str, Any]] = []
    metrics = (0.30, 0.20, 0.40)
    for candidate_id, candidate_config in enumerate(
        MODEL_SUITE.LSTM_VALIDATION_GRID
    ):
        selection_rows.append({
            "candidate_id": candidate_id,
            **candidate_config,
            "val_station_macro_rmse": metrics[candidate_id],
            "selected": candidate_id == 1,
            "selection_split": "2016-2017 validation",
        })
    selection = root / MODEL_SUITE.STAGE16_SELECTION_PATH
    selection.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(selection, index=False)
    for candidate_id, candidate_config in enumerate(
        MODEL_SUITE.LSTM_VALIDATION_GRID
    ):
        metric = metrics[candidate_id]
        candidate_rows: list[dict[str, Any]] = []
        issue = pd.Timestamp("2017-06-01")
        for site in sites:
            for horizon in C.HORIZONS:
                candidate_rows.append({
                    "model": f"LSTM-grid-{candidate_id}",
                    "scope": "validation_selection",
                    "feature_set": "USGS",
                    "seed": C.USGS_SEEDS[0],
                    "site_id": site,
                    "horizon": horizon,
                    "split": "val",
                    "issue_date": issue,
                    "target_date": issue + pd.Timedelta(days=horizon),
                    "y_true": 0.0,
                    "y_pred": metric,
                    "q05": metric - 0.5,
                    "q50": metric,
                    "q95": metric + 0.5,
                    "p_exceed": 0.2,
                })
        candidate_frame = pd.DataFrame(candidate_rows, columns=R.PRED_COLS)
        candidate_path = (
            run_manifest.parent / "selection"
            / f"candidate{candidate_id}.parquet"
        )
        R.write_predictions(candidate_frame, candidate_path)
        seal_artifact(
            candidate_path,
            identity,
            kind="lstm_validation_candidate_predictions",
            schema=R.PREDICTION_SCHEMA_VERSION,
            extra={
                "candidate_id": candidate_id,
                "candidate": candidate_config,
                "selection_split": "2016-2017 validation",
            },
        )
        checkpoint_config = {
            **configuration,
            "candidate_id": candidate_id,
            "candidate": dict(candidate_config),
        }
        checkpoint = (
            run_manifest.parent / "selection" / f"candidate{candidate_id}.pt"
        )
        model = LSTMForecaster(
            n_vars=len(MODEL_SUITE.STAGE9_USGS_VARIABLES),
            n_stations=len(sites),
            context=C.CONTEXT_LENGTH,
            station_agnostic=False,
            **dict(candidate_config),
        )
        train_config = C.TrainConfig(batch_size=1536)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=4
        )
        scheduler.step(float(metric))
        training_rng = np.random.default_rng(C.USGS_SEEDS[0])
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=train_config.max_epochs - 1,
            best_epoch=0,
            best_metric=float(metric),
            best_model_state=model.state_dict(),
            run_id=identity.run_id,
            resolved_config=checkpoint_config,
            extra={
                "bad_epochs": 0,
                "train_rng_state": training_rng.bit_generator.state,
            },
        )

    stage09_pointer = MODEL_SUITE.load_component_pointer(stage09["components"])
    development_contract = stage09_pointer["development_contract"]
    monkeypatch.setattr(
        MODEL_SUITE,
        "canonical_development_contract",
        lambda *_args, **_kwargs: dict(development_contract),
    )
    selected = dict(MODEL_SUITE.LSTM_VALIDATION_GRID[1])
    lstm_rows = final[final["model"].eq("LSTM")]
    metadata: dict[str, Any] = {
        **identity.as_dict(),
        "training_device": "cpu",
        "members": [f"seed{seed}" for seed in C.USGS_SEEDS],
        "architecture": {
            "class": "thermoroute.train.LSTMForecaster",
            "kwargs": {
                "n_vars": len(MODEL_SUITE.STAGE9_USGS_VARIABLES),
                "n_stations": len(sites),
                "context": C.CONTEXT_LENGTH,
                "station_agnostic": False,
                **selected,
            },
            "train_config": MODEL_SUITE.asdict(
                C.TrainConfig(batch_size=1536)
            ),
        },
        "station_to_index": {
            site: index for index, site in enumerate(sites)
        },
        "development_prediction": MODEL_SUITE.development_prediction_binding(
            root,
            prediction,
            lstm_rows,
            max_abs_difference=0.0,
            atol=1e-5,
        ),
    }
    bundle = (
        root / "outputs" / "models" / f"lstm_usgs_bundle_{identity.run_id}"
    )
    bundle.mkdir(parents=True)
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    (bundle / "weights.pt").write_bytes(b"five-member-weights")

    def fake_entry_validator(
        checked_root: Path,
        entry: dict[str, Any],
        feature_order: tuple[str, ...],
        *,
        external: bool,
        publication_guard=None,
    ) -> dict[str, Any]:
        assert checked_root == root
        assert entry["model_id"] == "LSTM"
        assert feature_order == MODEL_SUITE.STAGE9_USGS_VARIABLES
        assert external is False
        if publication_guard is not None:
            publication_guard()
        return dict(metadata)

    monkeypatch.setattr(
        MODEL_SUITE, "_entry_artifact_valid", fake_entry_validator
    )
    def fake_parity(**kwargs) -> dict[str, Any]:
        input_closure = MODEL_SUITE._stage16_parity_input_closure(
            root=kwargs["root"],
            bundle=kwargs["bundle"],
            metadata=kwargs["metadata"],
        )
        return {
            "format": MODEL_SUITE.STAGE16_PARITY_AUDIT_FORMAT,
            "status": "PASS_FIVE_MEMBER_VAL_CALIB_TEST_REPLAY",
            "members": [f"seed{seed}" for seed in C.USGS_SEEDS],
            "splits": ["val", "calib", "test"],
            "atol": 1e-5,
            "max_abs_difference": 0.0,
            "input_closure": input_closure,
            "input_closure_sha256": sha256_json(input_closure),
        }
    monkeypatch.setattr(
        MODEL_SUITE,
        "_verify_stage16_final_bundle_parity",
        fake_parity,
    )
    entry = {
        "model_id": "LSTM",
        "executor": "lstm_bundle",
        "raw_feature_order": list(MODEL_SUITE.STAGE9_USGS_VARIABLES),
        "member_count": 5,
        "artifact": MODEL_SUITE.directory_binding(root, bundle),
    }
    components = root / MODEL_SUITE.STAGE16_COMPONENT_POINTER_PATH
    write_component_pointer(
        components,
        run_id=identity.run_id,
        cohort="temporal_lstm",
        entries=[entry],
        raw_feature_order=MODEL_SUITE.STAGE9_USGS_VARIABLES,
        development_contract=development_contract,
        development_prediction_artifact={
            **file_binding(root, prediction),
            "sidecar": file_binding(root, MODEL_SUITE.sidecar_path(prediction)),
        },
    )
    shortcut = root / MODEL_SUITE.STAGE16_SHORTCUT_POINTER_PATH
    atomic_write_json(shortcut, {
        "run_id": identity.run_id,
        "bundle_path": bundle.relative_to(root).as_posix(),
        "member_count": 5,
        "metadata_sha256": sha256_file(bundle / "metadata.json"),
        "weights_sha256": sha256_file(bundle / "weights.pt"),
    })
    receipt_path = root / MODEL_SUITE.STAGE16_COMPLETION_RECEIPT_PATH
    document = MODEL_SUITE.build_stage16_completion_receipt(
        root=root,
        run_id=identity.run_id,
        run_manifest=run_manifest,
        stage09_receipt=stage09["receipt"],
        selection=selection,
        components_pointer=components,
        publication_guard=lambda: None,
    )
    return {
        "stage09": stage09,
        "identity": identity,
        "run_manifest": run_manifest,
        "selection": selection,
        "prediction": prediction,
        "bundle": bundle,
        "components": components,
        "shortcut": shortcut,
        "receipt": receipt_path,
        "document": document,
        "metadata": metadata,
    }


def _rehash_stage16(document: dict[str, Any]) -> None:
    document["artifact_closure_sha256"] = sha256_json(document["artifacts"])
    _rehash_receipt(document)


def test_stage16_validation_winner_is_tolerance_aware_for_near_ties():
    assert MODEL_SUITE.stage16_validation_winner(
        [0.200005, 0.200000, 0.40]
    ) == 0
    assert MODEL_SUITE.stage16_validation_winner(
        [0.20002, 0.200000, 0.40]
    ) == 1
    with pytest.raises(ModelSuiteError, match="validation winners differ"):
        MODEL_SUITE._stage16_consistent_validation_winner(
            [0.200019, 0.200000, 0.40],
            [0.2000095, 0.2000095, 0.40],
            [0.2000095, 0.2000095, 0.40],
        )


def test_stage16_receipt_closes_parent_grid_v2_bundle_and_pointers(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    document = fixture["document"]
    assert document["status"] == MODEL_SUITE.STAGE16_COMPLETION_STATUS
    assert document["parent_stage09_run_id"] == fixture["stage09"]["run_id"]
    assert len(document["artifacts"]["model_files"]) == 2
    assert len(document["artifacts"]["lstm_seed_prediction_files"]) == 10
    assert len(document["artifacts"]["selection_candidate_files"]) == 12
    assert document["selection_audit"]["status"] == (
        "PASS_BEST_STATE_REPLAY_AND_VALIDATION_SELECTION_PARITY"
    )
    threshold_contract = document["selection_audit"]["input_closure"][
        "event_threshold_contract"
    ]
    assert threshold_contract["registry_sha256"] == sha256_json(
        threshold_contract["registry"]
    )
    MODEL_SUITE.publish_stage16_completion_receipt(
        fixture["receipt"],
        document,
        root=tmp_path,
        components_pointer=fixture["components"],
        publication_guard=lambda: None,
    )
    validated = MODEL_SUITE.validate_stage16_completion_receipt(
        fixture["receipt"],
        root=tmp_path,
        components_pointer=fixture["components"],
        publication_guard=lambda: None,
    )
    assert validated == document
    assert MODEL_SUITE.stage16_completion_gate_binding(
        fixture["receipt"],
        root=tmp_path,
        components_pointer=fixture["components"],
        publication_guard=lambda: None,
    ) == file_binding(tmp_path, fixture["receipt"])


@pytest.mark.parametrize("mutation", ("alternate_grid", "selection_tamper"))
def test_stage16_receipt_rejects_grid_and_selection_tampering(
    tmp_path, monkeypatch, mutation,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    frame = pd.read_csv(fixture["selection"])
    if mutation == "alternate_grid":
        frame.loc[0, "d"] = 65
    else:
        frame["selected"] = [True, False, False]
    frame.to_csv(fixture["selection"], index=False)
    document = json.loads(json.dumps(fixture["document"]))
    document["artifacts"]["lstm_validation_selection"] = file_binding(
        tmp_path, fixture["selection"]
    )
    _rehash_stage16(document)
    with pytest.raises(
        ModelSuiteError, match="frozen grid|winner is not deterministic"
    ):
        MODEL_SUITE.validate_stage16_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            components_pointer=fixture["components"],
            document=document,
            publication_guard=lambda: None,
        )


def test_stage16_receipt_rejects_crafted_component_pointer(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    pointer = json.loads(fixture["components"].read_text(encoding="utf-8"))
    pointer["models"][0]["member_count"] = 4
    atomic_write_json(fixture["components"], pointer)
    document = json.loads(json.dumps(fixture["document"]))
    document["artifacts"]["components_pointer"] = file_binding(
        tmp_path, fixture["components"]
    )
    _rehash_stage16(document)
    with pytest.raises(ModelSuiteError, match="component entry is malformed"):
        MODEL_SUITE.validate_stage16_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            components_pointer=fixture["components"],
            document=document,
            publication_guard=lambda: None,
        )


def test_stage16_receipt_rejects_permuted_station_embedding_index(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    sites = list(fixture["metadata"]["station_to_index"])
    assert len(sites) >= 2
    fixture["metadata"]["station_to_index"] = {
        sites[0]: 1,
        sites[1]: 0,
        **{
            site: index for index, site in enumerate(sites[2:], start=2)
        },
    }
    with pytest.raises(ModelSuiteError, match="bundle metadata differs"):
        MODEL_SUITE.build_stage16_completion_receipt(
            root=tmp_path,
            run_id=fixture["identity"].run_id,
            run_manifest=fixture["run_manifest"],
            stage09_receipt=fixture["stage09"]["receipt"],
            selection=fixture["selection"],
            components_pointer=fixture["components"],
            publication_guard=lambda: None,
        )


def test_stage16_receipt_rejects_numeric_strings_in_prediction_parquet(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    candidate_binding = fixture["document"]["artifacts"][
        "selection_candidate_files"
    ][0]
    candidate = tmp_path / candidate_binding["path"]
    frame = pd.read_parquet(candidate)
    frame["y_pred"] = frame["y_pred"].map(lambda value: format(value, ".17g"))
    frame.to_parquet(candidate, index=False)
    seal_artifact(
        candidate,
        fixture["identity"],
        kind="lstm_validation_candidate_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents={},
        extra={
            "candidate_id": 0,
            "candidate": dict(MODEL_SUITE.LSTM_VALIDATION_GRID[0]),
            "selection_split": "2016-2017 validation",
        },
    )
    with pytest.raises(ModelSuiteError, match="y_pred Arrow type changed"):
        MODEL_SUITE.build_stage16_completion_receipt(
            root=tmp_path,
            run_id=fixture["identity"].run_id,
            run_manifest=fixture["run_manifest"],
            stage09_receipt=fixture["stage09"]["receipt"],
            selection=fixture["selection"],
            components_pointer=fixture["components"],
            publication_guard=lambda: None,
        )


def test_stage16_receipt_rejects_modified_non_lstm_v2_row(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    frame = pd.read_parquet(fixture["prediction"])
    row = frame.index[frame["model"].eq("Persistence")][0]
    frame.loc[row, "y_pred"] += 1.0
    R.write_predictions(frame, fixture["prediction"])
    seal_artifact(
        fixture["prediction"],
        fixture["identity"],
        kind="final_route_a_development_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents={
            fixture["stage09"]["predictions"].name:
            sha256_file(fixture["stage09"]["predictions"])
        },
        extra={
            "parent_run_id": fixture["stage09"]["run_id"],
            "primary_models": MODEL_SUITE.ROUTE_A_PRIMARY_MODELS,
            "primary_common_test_keys": 6,
            "dropped_primary_rows": 0,
            "lstm_validation_rows": 30,
            "lstm_calibration_rows": 30,
        },
    )
    # A crafted attacker can update every shallow binding.  The independent
    # Stage09 + seed-cache derivation must still reject the changed value.
    with pytest.raises(ModelSuiteError, match="exact Stage-9-plus-LSTM derivation"):
        MODEL_SUITE.build_stage16_completion_receipt(
            root=tmp_path,
            run_id=fixture["identity"].run_id,
            run_manifest=fixture["run_manifest"],
            stage09_receipt=fixture["stage09"]["receipt"],
            selection=fixture["selection"],
            components_pointer=fixture["components"],
            publication_guard=lambda: None,
        )


def test_stage16_receipt_rejects_crafted_best_state_replay_claim(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    document = json.loads(json.dumps(fixture["document"]))
    document["selection_audit"]["candidates"][1][
        "best_state_max_abs_difference"
    ] = 2e-5
    _rehash_stage16(document)
    with pytest.raises(
        ModelSuiteError,
        match="validation-selection candidate audit changed",
    ):
        MODEL_SUITE.validate_stage16_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            components_pointer=fixture["components"],
            document=document,
            publication_guard=lambda: None,
        )


@pytest.mark.parametrize(
    "mutation", ("optimizer_class", "nonterminal", "training_rng")
)
def test_stage16_rejects_malformed_candidate_checkpoint_payload(
    tmp_path, monkeypatch, mutation,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    binding = fixture["document"]["artifacts"]["selection_candidate_files"][2]
    checkpoint = tmp_path / binding["path"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if mutation == "optimizer_class":
        payload["optimizer_class"] = "torch.optim.adam.Adam"
    elif mutation == "nonterminal":
        payload["epoch"] = 1
        payload["best_epoch"] = 0
    else:
        extra = json.loads(payload["extra_json"])
        extra["train_rng_state"] = {}
        payload["extra_json"] = MODEL_SUITE.canonical_json(extra)
        payload["extra_sha256"] = sha256_json(extra)
    with pytest.raises(ModelSuiteError, match="checkpoint payload is invalid"):
        MODEL_SUITE._validate_stage16_candidate_checkpoint_payload(
            candidate_id=0,
            candidate_config=MODEL_SUITE.LSTM_VALIDATION_GRID[0],
            checkpoint_payload=payload,
            run_id=fixture["identity"].run_id,
            expected_config_json=payload["resolved_config_json"],
        )


@pytest.mark.parametrize("attack", ("shortcut_symlink", "bundle_symlink_dir"))
def test_stage16_receipt_rejects_symlink_aliases(
    tmp_path, monkeypatch, attack,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    if attack == "shortcut_symlink":
        alias = fixture["shortcut"].with_name("shortcut-alias.json")
        alias.write_bytes(fixture["shortcut"].read_bytes())
        fixture["shortcut"].unlink()
        fixture["shortcut"].symlink_to(alias)
        message = "shortcut pointer is a symlink"
    else:
        (fixture["bundle"] / "extra-directory").symlink_to(
            tmp_path / "data_usgs", target_is_directory=True
        )
        message = "bundle file closure changed"
    with pytest.raises(ModelSuiteError, match=message):
        MODEL_SUITE.build_stage16_completion_receipt(
            root=tmp_path,
            run_id=fixture["identity"].run_id,
            run_manifest=fixture["run_manifest"],
            stage09_receipt=fixture["stage09"]["receipt"],
            selection=fixture["selection"],
            components_pointer=fixture["components"],
            publication_guard=lambda: None,
        )


def test_stage16_receipt_rejects_hardlink_alias_of_authority_file(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    alias = fixture["shortcut"].with_name("shortcut-hardlink-alias.json")
    os.link(fixture["shortcut"], alias)
    assert fixture["shortcut"].stat().st_nlink == 2
    with pytest.raises(ModelSuiteError, match="canonical single-link file"):
        MODEL_SUITE.build_stage16_completion_receipt(
            root=tmp_path,
            run_id=fixture["identity"].run_id,
            run_manifest=fixture["run_manifest"],
            stage09_receipt=fixture["stage09"]["receipt"],
            selection=fixture["selection"],
            components_pointer=fixture["components"],
            publication_guard=lambda: None,
        )


def test_stage16_snapshot_rejects_atomic_replace_during_read(
    tmp_path, monkeypatch,
):
    path = _write_bytes(tmp_path / "candidate.pt", b"A" * (2 << 20))
    replacement = _write_bytes(tmp_path / "replacement.pt", b"B" * (2 << 20))
    real_read = os.read
    replaced = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, count)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(MODEL_SUITE.os, "read", racing_read)
    with pytest.raises(ModelSuiteError, match="changed while read"):
        MODEL_SUITE._stage16_file_snapshot(path, label="race fixture")
    assert replaced


def test_stage16_validator_rejects_canonical_receipt_symlink(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    MODEL_SUITE.publish_stage16_completion_receipt(
        fixture["receipt"],
        fixture["document"],
        root=tmp_path,
        components_pointer=fixture["components"],
        publication_guard=lambda: None,
    )
    alias = fixture["receipt"].with_name("stage16-receipt-alias.json")
    alias.write_bytes(fixture["receipt"].read_bytes())
    fixture["receipt"].unlink()
    fixture["receipt"].symlink_to(alias)
    with pytest.raises(ModelSuiteError, match="completion receipt path uses a symlink"):
        MODEL_SUITE.validate_stage16_completion_receipt(
            fixture["receipt"],
            root=tmp_path,
            components_pointer=fixture["components"],
            publication_guard=lambda: None,
        )


def test_stage24_replays_resealed_stage16_candidate_best_state(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)
    MODEL_SUITE.publish_stage16_completion_receipt(
        fixture["receipt"],
        fixture["document"],
        root=tmp_path,
        components_pointer=fixture["components"],
        publication_guard=lambda: None,
    )
    document = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    candidate_files = document["artifacts"]["selection_candidate_files"]
    checkpoint = tmp_path / candidate_files[2]["path"]
    checkpoint_sidecar = tmp_path / candidate_files[3]["path"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mutated_key = next(iter(payload["best_model_state"]))
    mutated_tensor = payload["best_model_state"][mutated_key].clone()
    if mutated_tensor.is_floating_point():
        mutated_tensor.reshape(-1)[0] += 1.0
    else:
        mutated_tensor.reshape(-1)[0] += 1
    payload["best_model_state"][mutated_key] = mutated_tensor
    torch.save(payload, checkpoint)
    metadata = json.loads(checkpoint_sidecar.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = sha256_file(checkpoint)
    metadata["checkpoint_bytes"] = checkpoint.stat().st_size
    atomic_write_json(checkpoint_sidecar, metadata)
    candidate_files[2] = file_binding(tmp_path, checkpoint)
    candidate_files[3] = file_binding(tmp_path, checkpoint_sidecar)
    selection_inputs = document["selection_audit"]["input_closure"]
    selection_inputs["candidate_files"] = candidate_files
    document["selection_audit"]["input_closure_sha256"] = sha256_json(
        selection_inputs
    )
    _rehash_stage16(document)
    atomic_write_json(fixture["receipt"], document)

    replayed: list[int] = []

    def reject_resealed_state(**kwargs) -> float:
        replayed.append(int(kwargs["candidate_id"]))
        observed = kwargs["checkpoint_payload"]["best_model_state"][mutated_key]
        if torch.equal(observed, mutated_tensor):
            raise ModelSuiteError("resealed candidate best state does not replay")
        return 0.0

    monkeypatch.setattr(
        MODEL_SUITE, "_verify_stage16_candidate_checkpoint", reject_resealed_state
    )
    monkeypatch.setattr(STAGE24, "_assert_stage24_policy", lambda: object())
    with pytest.raises(ModelSuiteError, match="best state does not replay"):
        STAGE24._load_verified_stage16(
            fixture["components"], fixture["receipt"], root=tmp_path
        )
    assert replayed == [0]


def test_stage16_receipt_atomic_guard_failure_leaves_no_authority(
    tmp_path, monkeypatch,
):
    fixture = _stage16_fixture(tmp_path, monkeypatch)

    def reject_at_atomic_boundary() -> None:
        raise RuntimeError("injected Stage-16 receipt publication drift")

    with pytest.raises(RuntimeError, match="publication drift"):
        MODEL_SUITE.write_stage16_completion_receipt(
            fixture["receipt"],
            fixture["document"],
            publication_guard=reject_at_atomic_boundary,
        )
    assert not fixture["receipt"].exists()
    assert not list(
        fixture["receipt"].parent.glob(
            f".{fixture['receipt'].name}.*.tmp"
        )
    )
