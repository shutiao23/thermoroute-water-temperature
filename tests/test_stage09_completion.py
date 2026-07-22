from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import model_suite as MODEL_SUITE  # noqa: E402
from thermoroute.model_suite import (  # noqa: E402
    ABLATION_INTERVENTIONS,
    MANDATORY_ABLATIONS,
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


def _load_script(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE09 = _load_script("scripts/09_usgs_experiment.py", "stage09_completion_test")
STAGE24 = _load_script("scripts/24_freeze_model_suite.py", "stage24_receipt_test")


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


def _lightgbm_quantile_metadata() -> dict[str, Any]:
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
        "quantile_repair": lightgbm_quantile_repair_contract(),
        "raw_quantile_crossing_audit": {
            **audit, "audit_sha256": sha256_json(audit),
        },
    }


def _stage09_fixture(root: Path) -> dict[str, Any]:
    source = root / "src" / "fixture.py"
    _write_bytes(source, b"VALUE = 1\n")
    panel = _write_bytes(root / "data_usgs" / "panel.parquet", b"panel")
    registry = _write_bytes(root / "data_usgs" / "registry.csv", b"registry")
    spec = _write_bytes(root / "data_usgs" / "spec.json", b"{}\n")
    bridge = _write_bytes(root / "data_usgs" / "bridge.json", b"{}\n")
    resolved_config = {
        "stage": "09_usgs_experiment",
        "execution_role": "route_a_formal_candidate",
        "training_device": "cpu",
        "station_sampling": "balanced",
        "delta_scale": 1.0,
        "thermoroute_seeds": list(C.USGS_SEEDS),
        "lightgbm_seeds": list(C.USGS_SEEDS),
        "ablations": True,
        "air2stream": True,
        "eval_batch_size": 64,
    }
    identity_parts = {
        "schema_version": RUN_SCHEMA_VERSION,
        "panel_sha256": sha256_file(panel),
        "registry_sha256": sha256_file(registry),
        "config_sha256": sha256_json(resolved_config),
        "source_sha256": source_tree_hash(root),
        "runtime_sha256": "e" * 64,
    }
    identity = RunIdentity(
        run_id=sha256_json(identity_parts)[:20],
        **identity_parts,
    )
    run_manifest = root / "outputs" / "runs" / "stage09-fixture" / "run.json"
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(json.dumps({
        "schema_version": "thermoroute.run.v1",
        "identity": identity.as_dict(),
        "provenance": {
            "evidence_role": "prelabel_route_a_model_build_development_only",
            "training_device": "cpu",
        },
        "resolved_config": resolved_config,
    }), encoding="utf-8")

    predictions = _write_bytes(
        root / "outputs" / "predictions" / "stage09.parquet", b"predictions"
    )
    seal_artifact(
        predictions,
        identity,
        kind="canonical_stage9_usgs_predictions",
        schema="fixture.predictions.v1",
    )
    scores = root / "outputs" / "tables" / "scores.csv"
    scores.parent.mkdir(parents=True)
    scores.write_text(
        "horizon,site,rmse_persist,rmse_damped,rmse_thermo\n"
        "1,site-a,1.0,0.9,0.8\n"
        "3,site-a,1.1,1.0,0.9\n"
        "7,site-a,1.2,1.1,1.0\n",
        encoding="utf-8",
    )
    selection = root / "outputs" / "tables" / "selection.csv"
    selection.write_text(
        "horizon,candidate_id,val_station_macro_rmse,selected\n"
        + "".join(
            f"{horizon},{candidate},{0.8 + candidate / 10:.1f},"
            f"{candidate == 0}\n"
            for horizon in C.HORIZONS
            for candidate in range(4)
        ),
        encoding="utf-8",
    )
    report = root / "outputs" / "reports" / "stage09.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        "# USGS large-sample experiment\n\n"
        "| Air2stream-style a4/a8 (unofficial, non-primary) |\n\n"
        "## Random held-station warm-start diagnostic\n\n"
        "## Module ablations (single-seed functionality/intervention "
        "diagnostic; seed0-vs-seed0)\n\n"
        "Audit: every mandatory control is exact seed=0 and is paired with "
        "ThermoRoute seed=0 on identical forecast keys and exact y_true. "
        "Interpretation: not evidence of module necessity, causal mechanism, "
        "or cross-seed stability.\n\n"
        "| ThermoRoute | 0.7 | 0.8 | 0.9 |\n"
        + "\n".join(f"| {name} | 0.8 | 0.9 | 1.0 |" for name in MANDATORY_ABLATIONS)
        + "\n",
        encoding="utf-8",
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
        json.dumps(_lightgbm_quantile_metadata()).encode("utf-8"),
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
            "member_count": 1,
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
    components_pointer = root / "outputs" / "models" / "stage09-components.json"
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
    thermoroute_pointer = root / "outputs" / "models" / "tr-current.json"
    atomic_write_json(thermoroute_pointer, {
        "run_id": identity.run_id,
        "bundle_path": entries[0]["artifact"]["path"],
        "member_count": 5,
        "metadata_sha256": entries[0]["artifact"]["metadata_sha256"],
        "weights_sha256": entries[0]["artifact"]["weights_sha256"],
    })
    lightgbm_pointer = root / "outputs" / "models" / "lgb-current.json"
    atomic_write_json(lightgbm_pointer, {
        "run_id": identity.run_id,
        "manifest": entries[1]["artifact"],
        "member_count": 5,
    })
    receipt_path = root / "outputs" / "models" / "stage09-receipt.json"
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
        "run_manifest": run_manifest,
        "prediction_sidecar": predictions.with_name(
            predictions.name + ".meta.json"
        ),
        "thermoroute_pointer": thermoroute_pointer,
        "lightgbm_pointer": lightgbm_pointer,
        "lightgbm_manifest": lightgbm_manifest,
        "run_id": identity.run_id,
    }


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


def _seed0_ablation_predictions() -> pd.DataFrame:
    issue = pd.Timestamp("2020-01-01")
    rows = []
    for model in ("ThermoRoute", *MANDATORY_ABLATIONS):
        seeds = (0, 1) if model == "ThermoRoute" else (0,)
        for seed in seeds:
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


def test_seed0_ablation_diagnostic_is_paired_not_ensemble_mean():
    frames = STAGE09.seed0_ablation_diagnostic_frames(
        _seed0_ablation_predictions()
    )
    assert tuple(frames) == ("ThermoRoute", *MANDATORY_ABLATIONS)
    assert set(frames["ThermoRoute"]["seed"]) == {0}
    assert frames["ThermoRoute"]["y_pred"].tolist() == [9.0, 10.0]
    assert all(set(frames[name]["seed"]) == {0} for name in MANDATORY_ABLATIONS)


def test_seed0_ablation_diagnostic_rejects_nonzero_control_seed():
    frame = _seed0_ablation_predictions()
    control = MANDATORY_ABLATIONS[0]
    frame.loc[frame["model"].eq(control), "seed"] = 1
    with pytest.raises(ValueError, match=f"{control} must contain exact seed=0"):
        STAGE09.seed0_ablation_diagnostic_frames(frame)


def test_seed0_ablation_diagnostic_rejects_different_control_keys():
    frame = _seed0_ablation_predictions()
    control = MANDATORY_ABLATIONS[1]
    row = frame.index[frame["model"].eq(control)][0]
    frame.loc[row, "target_date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match=f"{control} seed=0 forecast keys differ"):
        STAGE09.seed0_ablation_diagnostic_frames(frame)


def test_seed0_ablation_diagnostic_rejects_different_y_true():
    frame = _seed0_ablation_predictions()
    control = MANDATORY_ABLATIONS[2]
    row = frame.index[frame["model"].eq(control)][0]
    frame.loc[row, "y_true"] += 0.25
    with pytest.raises(ValueError, match=f"{control} seed=0 y_true differs"):
        STAGE09.seed0_ablation_diagnostic_frames(frame)


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
        )
    assert fixture["receipt"].read_bytes() == before


def test_stage09_receipt_rejects_old_ensemble_vs_single_seed_ablation_report(
    tmp_path,
):
    fixture = _stage09_fixture(tmp_path)
    before = fixture["receipt"].read_bytes()
    document = json.loads(before)
    report_text = fixture["report"].read_text(encoding="utf-8")
    report_text = report_text.replace(
        "## Module ablations (single-seed functionality/intervention "
        "diagnostic; seed0-vs-seed0)",
        "## Module ablations (median per-station RMSE)",
    ).replace(
        "Audit: every mandatory control is exact seed=0 and is paired with "
        "ThermoRoute seed=0 on identical forecast keys and exact y_true. "
        "Interpretation: not evidence of module necessity, causal mechanism, "
        "or cross-seed stability.\n\n",
        "",
    )
    fixture["report"].write_text(report_text, encoding="utf-8")
    document["artifacts"]["report"] = file_binding(tmp_path, fixture["report"])
    _rehash_receipt(document)

    with pytest.raises(ModelSuiteError, match="seed0 ablation diagnostic contract"):
        publish_stage09_completion_receipt(
            fixture["receipt"],
            document,
            root=tmp_path,
            stage9_pointer=fixture["components"],
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


def test_suite_identity_and_frozen_document_bind_both_completion_receipts(
    tmp_path, monkeypatch,
):
    receipt = _write_bytes(tmp_path / "outputs" / "receipt.json", b"receipt\n")
    gate = file_binding(tmp_path, receipt)
    controls_receipt = _write_bytes(
        tmp_path / "outputs" / "controls-receipt.json", b"controls receipt\n"
    )
    controls_gate = file_binding(tmp_path, controls_receipt)
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
    )
    second_id = STAGE24._model_suite_id(
        **common,
        stage09_completion={**gate, "sha256": "f" * 64},
        stage09b_completion=controls_gate,
    )
    assert first_id != second_id
    third_id = STAGE24._model_suite_id(
        **common,
        stage09_completion=gate,
        stage09b_completion={**controls_gate, "sha256": "e" * 64},
    )
    assert first_id != third_id

    monkeypatch.setattr(
        MODEL_SUITE, "_learned_metadata_runtime_sha256",
        lambda _root, _entries: "b" * 64,
    )
    monkeypatch.setattr(
        MODEL_SUITE, "validate_model_suite_document",
        lambda _document, *, root: None,
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
    )
    frozen = json.loads(destination.read_text(encoding="utf-8"))
    assert frozen["preopening_gates"] == {
        "stage09_completion": gate,
        "stage09b_development_controls": controls_gate,
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
