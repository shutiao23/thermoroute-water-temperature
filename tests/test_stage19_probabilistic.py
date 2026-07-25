from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from thermoroute import conformal
from thermoroute import results
from thermoroute.model_suite import route_a_calibration_fit_contract
from thermoroute.probability import EVENT_REFERENCE_FORMAT
from thermoroute.repro import (
    RunIdentity,
    seal_artifact,
    sha256_file,
    sha256_json,
    sidecar_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_stage19():
    path = ROOT / "scripts" / "19_probabilistic.py"
    spec = importlib.util.spec_from_file_location("stage19_probabilistic_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE19 = _load_stage19()


def test_stage19_import_activates_live_formal_numerical_policy():
    policy = STAGE19.assert_formal_numerical_policy()
    assert set(policy["thread_environment"].values()) == {"1"}
    assert policy["cublas_workspace_config"] == ":4096:8"


def _event_reference(sites: tuple[str, ...]) -> dict[str, object]:
    station_probability = {site: 0.5 for site in sites}
    station_month = {
        f"{site}|{month}": 0.5 for site in sites for month in range(1, 13)
    }
    return {
        "format": EVENT_REFERENCE_FORMAT,
        "mode": "station_month",
        "threshold_scope": "station_development_train_q90",
        "fit_interval": ["2006-01-01", "2018-12-31"],
        "smoothing": 2.0,
        "global_probability": 0.5,
        "station_probability": station_probability,
        "station_month_probability": station_month,
        "fit_observation_count": 1000,
    }


def _metadata(sites: tuple[str, ...], *, offset: float = 1.0) -> dict[str, object]:
    raw_offsets = {
        (site, horizon): offset
        for site in sites
        for horizon in STAGE19.HORIZONS
    }
    deployed, offset_audit = conformal.finalise_cqr_offsets(raw_offsets)
    serialized = {
        f"{site}|{horizon}": value
        for (site, horizon), value in deployed.items()
    }
    return {
        "horizons": list(STAGE19.HORIZONS),
        "event_thresholds": {site: 10.0 for site in sites},
        "event_reference_climatology": _event_reference(sites),
        "event_calibrators": {
            str(horizon): {"intercept": 0.0, "slope": 0.0, "constant": 0.5}
            for horizon in STAGE19.HORIZONS
        },
        "conformal_offsets": serialized,
        "conformal_policy": conformal.cqr_policy_contract(),
        "conformal_offset_audit": offset_audit,
        "calibration_fit_contract": route_a_calibration_fit_contract(external=False),
    }


def _ensemble_frames(
    sites_and_counts: tuple[tuple[str, int], ...] = (("01000001", 100),),
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for model in STAGE19.PROB_MODELS:
        rows: list[dict[str, object]] = []
        for horizon in STAGE19.HORIZONS:
            for site, count in sites_and_counts:
                for index, issue_date in enumerate(
                    pd.date_range("2019-01-01", periods=count, freq="D")
                ):
                    rows.append({
                        "site_id": site,
                        "horizon": horizon,
                        "issue_date": issue_date,
                        "target_date": issue_date + pd.Timedelta(days=horizon),
                        "y_true": 0.0,
                        "y_pred": 1.0,
                        "q05": 0.5,
                        "q50": 1.0,
                        "q95": 1.5,
                        "p_exceed": 0.2 + 0.0 * index,
                    })
        frames[model] = pd.DataFrame(rows)
    return frames


def test_station_balanced_weights_are_not_row_micro_average():
    sites = np.array(["short", "long", "long", "long"])
    losses = np.array([10.0, 0.0, 0.0, 0.0])
    weights = STAGE19.station_balanced_weights(sites)
    assert STAGE19._weighted_mean(losses, weights) == pytest.approx(5.0)
    assert losses.mean() == pytest.approx(2.5)
    totals = pd.DataFrame({"site": sites, "weight": weights}).groupby("site").weight.sum()
    assert np.array_equal(totals.to_numpy(), np.array([0.5, 0.5]))


def test_bundle_event_reference_must_cover_exact_2006_2018_interval():
    sites = ("01000001",)
    metadata = {model: _metadata(sites) for model in STAGE19.PROB_MODELS}
    audit = STAGE19.validate_probability_metadata(metadata, sites=sites)
    assert audit["event_reference_fit_interval"] == ["2006-01-01", "2018-12-31"]
    assert audit["fitting_performed_by_stage19"] is False

    changed = {model: _metadata(sites) for model in STAGE19.PROB_MODELS}
    changed["LSTM"]["event_reference_climatology"]["fit_interval"] = [
        "2006-01-01", "2015-12-31"
    ]
    with pytest.raises(STAGE19.ProbabilityContractError, match="2006--2018"):
        STAGE19.validate_probability_metadata(changed, sites=sites)


@pytest.mark.parametrize(
    ("field", "value"),
    (("slope", 1.0), ("intercept", 0.25)),
)
def test_constant_platt_registry_requires_exact_constant_branch(field, value):
    sites = ("01000001",)
    metadata = {model: _metadata(sites) for model in STAGE19.PROB_MODELS}
    metadata["ThermoRoute"]["event_calibrators"]["1"][field] = value
    with pytest.raises(
        STAGE19.ProbabilityContractError,
        match="zero slope and logit intercept",
    ):
        STAGE19.validate_probability_metadata(metadata, sites=sites)


def test_raw_three_quantile_score_is_unscaled_and_cqr_only_changes_endpoints():
    sites = ("01000001",)
    metadata = {model: _metadata(sites, offset=1.0) for model in STAGE19.PROB_MODELS}
    audit = STAGE19.validate_probability_metadata(metadata, sites=sites)
    scores, _reliability, application = STAGE19.evaluate_probability_models(
        _ensemble_frames(), metadata, audit
    )
    row = scores[
        scores.model.eq("ThermoRoute") & scores.horizon.eq(1)
    ].iloc[0]
    # For y=0 and q=(.5,1,1.5), pinballs are .475, .50, .075; mean=.35.
    assert row.RAW_PINBALL_Q05 == pytest.approx(0.475)
    assert row.RAW_PINBALL_Q50 == pytest.approx(0.50)
    assert row.RAW_PINBALL_Q95 == pytest.approx(0.075)
    assert row.THREE_QUANTILE_SCORE == pytest.approx(0.35)
    assert row.THREE_QUANTILE_SCORE != pytest.approx(1.00)
    # Frozen offset 1 changes [.5,1.5] into [-.5,2.5].
    assert row.PICP == pytest.approx(1.0)
    assert row.MPIW == pytest.approx(3.0)
    assert application["calibration_application"]["ThermoRoute"][
        "evaluation_time_quantile_repair_applied"
    ] is False


def test_strict_ensemble_registry_rejects_exact_truth_or_head_drift():
    rows: list[dict[str, object]] = []
    for model in STAGE19.PROB_MODELS:
        for seed in (0, 1):
            rows.append({
                "model": model,
                "seed": seed,
                "split": "test",
                "site_id": "01000001",
                "horizon": 1,
                "issue_date": pd.Timestamp("2019-01-01"),
                "target_date": pd.Timestamp("2019-01-02"),
                "y_true": 1.0,
                "y_pred": 1.0,
                "q05": 0.0,
                "q50": 1.0,
                "q95": 2.0,
                "p_exceed": 0.5,
            })
    frame = pd.DataFrame(rows)
    frame.loc[(frame.model.eq("LSTM")) & (frame.seed.eq(0)), "y_true"] = np.nextafter(
        1.0, 2.0
    )
    with pytest.raises(STAGE19.ProbabilityContractError, match="exact y_true"):
        STAGE19.strict_ensemble_frames(
            frame,
            STAGE19.PROB_MODELS,
            require_probability_heads=True,
            expected_seeds={model: (0, 1) for model in STAGE19.PROB_MODELS},
        )

    frame.loc[frame.model.eq("LSTM"), "y_true"] = 1.0
    frame.loc[(frame.model.eq("LightGBM")) & (frame.seed.eq(1)), "q95"] = np.nan
    with pytest.raises(STAGE19.ProbabilityContractError, match="missing probabilistic heads"):
        STAGE19.strict_ensemble_frames(
            frame,
            STAGE19.PROB_MODELS,
            require_probability_heads=True,
            expected_seeds={model: (0, 1) for model in STAGE19.PROB_MODELS},
        )


def test_strict_ensemble_registry_rejects_wrong_seed_with_same_count():
    rows: list[dict[str, object]] = []
    for model in STAGE19.PROB_MODELS:
        seeds = (0, 9) if model == "LightGBM" else (0, 1)
        for seed in seeds:
            rows.append({
                "model": model,
                "seed": seed,
                "split": "test",
                "site_id": "01000001",
                "horizon": 1,
                "issue_date": pd.Timestamp("2019-01-01"),
                "target_date": pd.Timestamp("2019-01-02"),
                "y_true": 1.0,
                "y_pred": 1.0,
                "q05": 0.0,
                "q50": 1.0,
                "q95": 2.0,
                "p_exceed": 0.5,
            })
    with pytest.raises(STAGE19.ProbabilityContractError, match="seed registry"):
        STAGE19.strict_ensemble_frames(
            pd.DataFrame(rows),
            STAGE19.PROB_MODELS,
            require_probability_heads=True,
            expected_seeds={model: (0, 1) for model in STAGE19.PROB_MODELS},
        )


@pytest.mark.parametrize(
    ("column", "value", "diagnostic"),
    (
        ("q95", 0.0, "empty interval"),
        ("p_exceed", -0.1, "outside \\[0, 1\\]"),
        ("p_exceed", 1.1, "outside \\[0, 1\\]"),
    ),
)
def test_strict_ensemble_rejects_empty_interval_and_probability_range(
    column, value, diagnostic
):
    frame = _ensemble_frames()["ThermoRoute"].copy()
    frame["model"] = "ThermoRoute"
    frame["seed"] = 0
    frame["split"] = "test"
    frame[column] = value
    with pytest.raises(STAGE19.ProbabilityContractError, match=diagnostic):
        STAGE19.strict_ensemble_frames(
            frame,
            ("ThermoRoute",),
            require_probability_heads=True,
            expected_seeds={"ThermoRoute": (0,)},
        )


def _receipt_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, Path]]:
    panel = tmp_path / "panel.parquet"
    registry = tmp_path / "registry.csv"
    protocol = tmp_path / "protocol.json"
    stage16_receipt = tmp_path / "stage16-receipt.json"
    stage9 = tmp_path / "stage9.json"
    lstm = tmp_path / "lstm.json"
    prediction = tmp_path / "predictions.parquet"
    for path, payload in (
        (panel, b"panel"),
        (registry, b"registry"),
        (protocol, b"protocol"),
        (stage16_receipt, b"stage16 receipt"),
        (stage9, b"stage9"),
        (lstm, b"lstm"),
        (prediction, b"predictions"),
    ):
        path.write_bytes(payload)

    frozen_identity = RunIdentity(
        run_id="frozen-input",
        panel_sha256=sha256_file(panel),
        registry_sha256=sha256_file(registry),
        config_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256="3" * 64,
        input_closure_sha256="4" * 64,
    )
    seal_artifact(
        prediction,
        frozen_identity,
        kind="final_route_a_development_predictions",
        schema=results.PREDICTION_SCHEMA_VERSION,
        parents={"fixture_parent": "4" * 64},
    )

    model_metadata: dict[str, Path] = {}
    for model in STAGE19.PROB_MODELS:
        path = tmp_path / f"{model}.metadata.json"
        path.write_text(f"{model}\n", encoding="utf-8")
        model_metadata[model] = path

    def binding(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(path),
        }

    inputs = {
        "prediction": {
            "artifact": binding(prediction),
            "lineage_sidecar": binding(sidecar_path(prediction)),
        },
        "stage16_completion_receipt": binding(stage16_receipt),
        "panel": binding(panel),
        "registry": binding(registry),
        "protocol": binding(protocol),
        "component_pointers": {"stage9": binding(stage9), "lstm": binding(lstm)},
        "bundle_metadata": {
            model: binding(path) for model, path in model_metadata.items()
        },
        "bundle_files": {
            model: [binding(path)] for model, path in model_metadata.items()
        },
    }
    input_components = {
        "panel": inputs["panel"]["sha256"],
        "registry": inputs["registry"]["sha256"],
        "prediction": inputs["prediction"]["artifact"]["sha256"],
        "prediction_lineage": inputs["prediction"]["lineage_sidecar"]["sha256"],
        "stage16_completion_receipt": inputs["stage16_completion_receipt"][
            "sha256"
        ],
        "stage9_components": inputs["component_pointers"]["stage9"]["sha256"],
        "lstm_components": inputs["component_pointers"]["lstm"]["sha256"],
        "protocol": inputs["protocol"]["sha256"],
        **{
            f"model_file:{binding['path']}": binding["sha256"]
            for model in STAGE19.PROB_MODELS
            for binding in inputs["bundle_files"][model]
        },
    }
    stage19_identity = RunIdentity(
        run_id="stage19-fixture",
        panel_sha256=frozen_identity.panel_sha256,
        registry_sha256=frozen_identity.registry_sha256,
        config_sha256="5" * 64,
        source_sha256=frozen_identity.source_sha256,
        runtime_sha256=frozen_identity.runtime_sha256,
        input_closure_sha256=STAGE19.compose_input_closure_digest(
            input_components
        ),
    )
    parents = {
        "prediction": inputs["prediction"]["artifact"]["sha256"],
        "prediction_lineage": inputs["prediction"]["lineage_sidecar"]["sha256"],
        "stage16_completion_receipt": inputs["stage16_completion_receipt"][
            "sha256"
        ],
        "stage9_components": inputs["component_pointers"]["stage9"]["sha256"],
        "lstm_components": inputs["component_pointers"]["lstm"]["sha256"],
        "protocol": inputs["protocol"]["sha256"],
        **{
            f"{model}_bundle_metadata": inputs["bundle_metadata"][model]["sha256"]
            for model in STAGE19.PROB_MODELS
        },
    }
    receipt = tmp_path / "receipt.json"
    outputs: dict[str, Path] = {}
    artifact_bindings: dict[str, object] = {}
    for label, schema in STAGE19.OUTPUT_CONTENT_SCHEMAS.items():
        path = tmp_path / f"{label}.out"
        path.write_text(f"{label}\n", encoding="utf-8")
        seal_artifact(
            path,
            stage19_identity,
            kind=f"route_a_development_{label}",
            schema=schema,
            parents=parents,
            extra={"development_only": True, "receipt_path": "receipt.json"},
        )
        outputs[label] = path
        artifact_bindings[label] = {
            "artifact": binding(path),
            "lineage_sidecar": binding(sidecar_path(path)),
        }
    contract = {"fixture": True}
    document: dict[str, object] = {
        "format": STAGE19.RECEIPT_FORMAT,
        "status": "PASS",
        "scientific_role": "previously_inspected_development_evaluation_2019_2020",
        "inference_computed": False,
        "run_identity": stage19_identity.as_dict(),
        "lineage": {
            "source_sha256": stage19_identity.source_sha256,
            "panel_sha256": stage19_identity.panel_sha256,
            "registry_sha256": stage19_identity.registry_sha256,
            "execution_runtime_sha256": stage19_identity.runtime_sha256,
            "frozen_bundle_runtime_sha256": stage19_identity.runtime_sha256,
        },
        "inputs": inputs,
        "contract": contract,
        "contract_sha256": sha256_json(contract),
        "event_reference_sha256": "6" * 64,
        "probability_registry_audit": {},
        "point_registry_audit": {},
        "artifacts": artifact_bindings,
    }
    STAGE19._write_self_hashed_receipt(receipt, document)
    return receipt, document, outputs


def test_self_hashed_receipt_binds_lineage_and_detects_tamper(tmp_path):
    receipt, document, outputs = _receipt_fixture(tmp_path)
    validated = STAGE19.validate_probability_receipt(
        receipt,
        root=tmp_path,
        enforce_current_source_and_runtime=False,
        enforce_canonical_paths=False,
    )
    assert validated["receipt_self_sha256"] == sha256_json(document)

    output = outputs["probability_scores"]
    original_output = output.read_bytes()
    output.write_bytes(b"score\n999\n")
    with pytest.raises(STAGE19.ProbabilityContractError, match="binding changed"):
        STAGE19.validate_probability_receipt(
            receipt,
            root=tmp_path,
            enforce_current_source_and_runtime=False,
            enforce_canonical_paths=False,
        )

    output.write_bytes(original_output)
    altered = json.loads(receipt.read_text(encoding="utf-8"))
    altered["status"] = "PASS_BUT_CHANGED"
    receipt.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(STAGE19.ProbabilityContractError, match="format/status"):
        STAGE19.validate_probability_receipt(
            receipt,
            root=tmp_path,
            enforce_current_source_and_runtime=False,
            enforce_canonical_paths=False,
        )


def test_receipt_cannot_drop_artifact_and_reseal_public_self_hash(tmp_path):
    receipt, _document, _outputs = _receipt_fixture(tmp_path)
    changed = json.loads(receipt.read_text(encoding="utf-8"))
    changed["artifacts"].pop("calibration_audit")
    changed.pop("receipt_self_sha256")
    changed["receipt_self_sha256"] = sha256_json(changed)
    receipt.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(STAGE19.ProbabilityContractError, match="artifact closure"):
        STAGE19.validate_probability_receipt(
            receipt,
            root=tmp_path,
            enforce_current_source_and_runtime=False,
            enforce_canonical_paths=False,
        )


def test_receipt_cannot_drop_stage16_gate_and_reseal_public_self_hash(tmp_path):
    receipt, _document, _outputs = _receipt_fixture(tmp_path)
    changed = json.loads(receipt.read_text(encoding="utf-8"))
    changed["inputs"].pop("stage16_completion_receipt")
    changed.pop("receipt_self_sha256")
    changed["receipt_self_sha256"] = sha256_json(changed)
    receipt.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(STAGE19.ProbabilityContractError, match="input closure"):
        STAGE19.validate_probability_receipt(
            receipt,
            root=tmp_path,
            enforce_current_source_and_runtime=False,
            enforce_canonical_paths=False,
        )


def test_receipt_cannot_replace_input_closure_and_reseal_public_self_hash(tmp_path):
    receipt, _document, _outputs = _receipt_fixture(tmp_path)
    changed = json.loads(receipt.read_text(encoding="utf-8"))
    changed["run_identity"]["input_closure_sha256"] = "0" * 64
    changed.pop("receipt_self_sha256")
    changed["receipt_self_sha256"] = sha256_json(changed)
    receipt.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(STAGE19.ProbabilityContractError, match="bindings disagree"):
        STAGE19.validate_probability_receipt(
            receipt,
            root=tmp_path,
            enforce_current_source_and_runtime=False,
            enforce_canonical_paths=False,
        )


def test_stage19_main_uses_stage16_then_stage19_lock_order(monkeypatch, tmp_path):
    events: list[object] = []
    stage16_lock = tmp_path / "stage16.lock"
    stage19_lock = tmp_path / "stage19.lock"
    monkeypatch.setattr(STAGE19.C, "STAGE16_TRANSACTION_LOCK", stage16_lock)
    monkeypatch.setattr(STAGE19.C, "STAGE19_TRANSACTION_LOCK", stage19_lock)
    args = SimpleNamespace(check=False)
    monkeypatch.setattr(STAGE19, "_parse_args", lambda: args)

    @contextmanager
    def fake_lock(path, *, exclusive):
        events.append(("enter", path, exclusive))
        yield
        events.append(("exit", path, exclusive))

    def fake_gate():
        events.append("gate")
        return {"artifacts": {}}, {"path": "stage16", "sha256": "a" * 64}

    def fake_run(run_args, *, stage16_document, stage16_binding):
        events.append(("run", run_args, stage16_document, stage16_binding))

    monkeypatch.setattr(STAGE19, "advisory_file_lock", fake_lock)
    monkeypatch.setattr(STAGE19, "_validated_stage16_gate_under_lock", fake_gate)
    monkeypatch.setattr(STAGE19, "_run", fake_run)
    STAGE19.main()

    assert events == [
        ("enter", stage16_lock, False),
        "gate",
        ("enter", stage19_lock, True),
        (
            "run",
            args,
            {"artifacts": {}},
            {"path": "stage16", "sha256": "a" * 64},
        ),
        ("exit", stage19_lock, True),
        ("exit", stage16_lock, False),
    ]


def test_stage19_gate_helper_fails_closed_on_missing_stage16(monkeypatch):
    def missing_gate(*_args, **_kwargs):
        raise STAGE19.ModelSuiteError("missing")

    monkeypatch.setattr(
        STAGE19, "validate_stage16_completion_receipt", missing_gate
    )
    with pytest.raises(
        STAGE19.ProbabilityContractError,
        match="Stage-16 completion gate is absent, stale, or malformed",
    ):
        STAGE19._validated_stage16_gate_under_lock()


def test_live_guard_failure_preserves_artifact_and_cannot_publish_pass(
    monkeypatch, tmp_path,
):
    receipt = tmp_path / "probabilistic_evaluation_v2.json"
    receipt.write_text('{"status":"PASS"}\n', encoding="utf-8")
    artifacts = {
        label: tmp_path / f"{label}.artifact"
        for label in STAGE19.OUTPUT_CONTENT_SCHEMAS
    }
    for label, path in artifacts.items():
        path.write_bytes(f"old:{label}\n".encode("utf-8"))
    original_artifacts = {
        label: path.read_bytes() for label, path in artifacts.items()
    }
    calls = 0

    def fail_after_incomplete():
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("simulated live native pool drift")
        return {"live": "formal"}

    monkeypatch.setattr(
        STAGE19, "assert_formal_numerical_policy", fail_after_incomplete
    )
    with pytest.raises(RuntimeError, match="live native pool drift"):
        STAGE19._publish_stage19_outputs(
            receipt_path=receipt,
            artifacts=artifacts,
            probability_scores_payload=b"new probability scores\n",
            point_scores_payload=b"new point scores\n",
            calibration_audit={"new": True},
            reliability_figure_payload=b"new figure\n",
            report_payload=b"new report\n",
        )
    # The first guarded boundary revokes the old PASS.  The second guard runs
    # after the candidate bytes are durable but before canonical replacement.
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "INCOMPLETE"
    assert {
        label: path.read_bytes() for label, path in artifacts.items()
    } == original_artifacts
    assert not list(tmp_path.glob(".*.tmp"))

    with pytest.raises(RuntimeError, match="live native pool drift"):
        STAGE19._write_self_hashed_receipt(
            receipt,
            {"format": STAGE19.RECEIPT_FORMAT, "status": "PASS"},
        )
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "INCOMPLETE"


def test_receipt_validation_fails_closed_when_live_policy_drifts(
    monkeypatch, tmp_path,
):
    receipt, _document, _outputs = _receipt_fixture(tmp_path)

    def reject_live_policy():
        raise RuntimeError("simulated live policy rejection")

    monkeypatch.setattr(
        STAGE19, "assert_formal_numerical_policy", reject_live_policy
    )
    with pytest.raises(RuntimeError, match="live policy rejection"):
        STAGE19.validate_probability_receipt(
            receipt,
            root=tmp_path,
            enforce_current_source_and_runtime=False,
            enforce_canonical_paths=False,
        )


def test_formal_receipt_rejects_alternate_output_namespace(tmp_path):
    receipt, _document, _outputs = _receipt_fixture(tmp_path)
    with pytest.raises(STAGE19.ProbabilityContractError, match="canonical path"):
        STAGE19.validate_probability_receipt(
            receipt,
            root=tmp_path,
            enforce_current_source_and_runtime=False,
        )
