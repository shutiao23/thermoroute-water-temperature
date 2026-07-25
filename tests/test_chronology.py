from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.chronology import (  # noqa: E402
    ChronologyError,
    DEFAULT_MODEL_MATRIX_AMENDMENT,
    DEFAULT_MODEL_MATRIX_AMENDMENT_SEAL,
    MODEL_MATRIX_AMENDMENT_FORMAT,
    MODEL_MATRIX_AMENDMENT_ID,
    MODEL_MATRIX_AMENDMENT_STATUS,
    MODEL_MATRIX_DOCUMENT_LINEAGE_CONTRACT,
    MODEL_MATRIX_PRELABEL_ATTESTATION,
    MODEL_MATRIX_SEAL_FORMAT,
    MODEL_MATRIX_SEAL_HISTORY_CONTRACT,
    MODEL_MATRIX_SEAL_STATUS,
    REQUIRED_GATE_PATHS,
    STAGE09_ARTIFACT_PATHS,
    STAGE09B_MEMBERS,
    _stage09b_scientific_comparison_registry,
    freeze_prelabel_chronology,
    validate_prelabel_chronology,
)
from thermoroute.checkpoint import neural_output_head_schema  # noqa: E402
from thermoroute.repro import source_tree_hash  # noqa: E402


def _run(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write(root: Path, relative: str, payload: bytes | str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    path.write_bytes(payload)
    return path


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _repro_sha(value: dict[str, Any]) -> str:
    return _sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _compose_input_closure(components: dict[str, str]) -> str:
    document = {
        "format": "thermoroute.composed-input-closure.v1",
        "components": [
            {"name": name, "sha256": digest}
            for name, digest in sorted(components.items())
        ],
    }
    return _sha(
        (json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n").encode("utf-8")
    )


def _fixture_run_identity(
    configuration: dict[str, Any],
    *,
    panel_sha256: str,
    registry_sha256: str,
    input_closure_sha256: str,
) -> dict[str, Any]:
    fields = {
        "panel_sha256": panel_sha256,
        "registry_sha256": registry_sha256,
        "config_sha256": _repro_sha(configuration),
        "source_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "input_closure_sha256": input_closure_sha256,
        "schema_version": "thermoroute.run.v2",
    }
    return {"run_id": _repro_sha(fields)[:20], **fields}


def _file_sha(root: Path, relative: str) -> str:
    return _sha((root / relative).read_bytes())


def _binding(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _file_sha(root, relative)}


def _stage09b_scientific_summary() -> dict[str, Any]:
    comparisons = _stage09b_scientific_comparison_registry()
    records = [
        {
            "comparison_family": comparison["comparison_family"],
            "comparison_id": comparison["comparison_id"],
            "candidate_arm_id": comparison["candidate_arm_id"],
            "reference_arm_id": comparison["reference_arm_id"],
            "seed": seed,
            "split": split,
            "horizon": horizon,
            "common_forecast_keys": 1,
            "stations": 1,
            "median_paired_station_rmse_difference_c": 0.0,
        }
        for comparison in comparisons
        for seed in comparison["seeds"]
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
    ]
    records_sha = _sha(json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))
    ladder = (
        ("01_WTEMP", ["WTEMP"]),
        ("02_plus_FLOW", ["WTEMP", "FLOW"]),
        ("03_plus_TEMP", ["WTEMP", "FLOW", "TEMP"]),
        ("04_plus_PRCP", ["WTEMP", "FLOW", "TEMP", "PRCP"]),
        ("05_plus_RHMEAN", ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN"]),
        ("06_plus_DH", ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH"]),
        (
            "07_plus_WDSP",
            ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"],
        ),
    )
    return {
        "format": "thermoroute.development-controls-scientific-summary.v1",
        "metric_summary_format": (
            "thermoroute.development-controls-metric-summary.v2"
        ),
        "primary_member_estimand": {
            "name": "median_across_stations_of_within_station_rmse_c",
            "column": "median_station_rmse_c",
            "unit": "degree_Celsius",
            "aggregation": "median_of_within_station_RMSE",
            "station_weighting": "one_station_one_value",
        },
        "secondary_member_estimands": {
            "micro_rmse_c": {
                "role": "secondary_not_primary_estimand",
                "aggregation": "RMSE_over_all_forecast_keys",
            },
            "micro_mae_c": {
                "role": "secondary_not_primary_estimand",
                "aggregation": "MAE_over_all_forecast_keys",
            },
        },
        "paired_descriptive_effects": {
            "estimand": (
                "median_across_stations_of_candidate_rmse_minus_reference_rmse_c"
            ),
            "effect_convention": "candidate_minus_reference",
            "negative_favours": "candidate",
            "same_seed": True,
            "exact_common_forecast_keys_verified": True,
            "comparison_registry": comparisons,
            "feature_ladder_order": [
                {"rung": rung, "variables": variables} for rung, variables in ladder
            ],
            "feature_ladder_fixed_order_path_dependent": True,
            "independent_feature_contribution_claimed": False,
            "causal_effect_claimed": False,
            "records_sha256": records_sha,
            "records": records,
        },
    }


def _commit(root: Path, message: str) -> str:
    _run(root, "add", "-A")
    _run(root, "commit", "-m", message)
    return _run(root, "rev-parse", "HEAD")


def _snapshot(
    root: Path,
    index_path: str,
    *,
    payload: bytes,
) -> None:
    base = Path(index_path).parent
    metadata = (base / "provider" / "request" / "metadata.json").as_posix()
    response = (base / "provider" / "request" / "response.bin").as_posix()
    metadata_payload = _json_bytes({"status": 200})
    _write(root, metadata, metadata_payload)
    _write(root, response, payload)
    _write(
        root,
        index_path,
        _json_bytes(
            {
                "schema_version": 2,
                "snapshot_count": 1,
                "records": [
                    {
                        "provider": "fixture-provider",
                        "request_sha256": _sha(b"fixture-request"),
                        "metadata_path": str(Path(metadata).relative_to(base)),
                        "response_path": str(Path(response).relative_to(base)),
                        "response_sha256": _sha(payload),
                        "metadata_sha256": _sha(metadata_payload),
                        "metadata_byte_count": len(metadata_payload),
                        "retrieved_at_utc": "2020-01-01T00:00:00+00:00",
                        "byte_count": len(payload),
                        "request": {"provider": "fixture-provider"},
                    }
                ],
            }
        ),
    )


def _seed_stage16_completion(
    root: Path,
    *,
    stage09_path: str,
    stage09_run_id: str,
    attack: str | None,
) -> dict[str, Any]:
    """Create a lightweight but byte-complete formal Stage-16 fixture."""
    stage09_receipt = json.loads((root / stage09_path).read_text(encoding="utf-8"))
    stage09_configuration = stage09_receipt["formal_configuration"]
    input_closure_sha256 = _compose_input_closure({
        "development": stage09_receipt["run_identity"][
            "input_closure_sha256"
        ],
        "stage09_parent_prediction": _file_sha(
            root, "outputs/predictions/usgs_predictions_stage9_v2.parquet"
        ),
        "stage09_parent_sidecar": _file_sha(
            root,
            "outputs/predictions/usgs_predictions_stage9_v2.parquet.meta.json",
        ),
        "stage09_completion_receipt": _file_sha(root, stage09_path),
        "stage09_components": _file_sha(
            root, "outputs/models/route_a_stage9_components.json"
        ),
    })
    configuration = {
        "fixture": True,
        "training_device": "cpu",
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": (
            stage09_configuration["input_closure_file_count"] + 4
        ),
    }
    identity = _fixture_run_identity(
        configuration,
        panel_sha256=_file_sha(
            root, "data_usgs/panel_usgs_120v2.parquet"
        ),
        registry_sha256=_file_sha(
            root, "data_usgs/station_registry_v1.csv"
        ),
        input_closure_sha256=input_closure_sha256,
    )
    run_id = identity["run_id"]
    run_dir = f"outputs/runs/16_lstm_baseline/{run_id}"
    run_manifest = f"{run_dir}/run.json"
    selection = "outputs/tables/lstm_validation_selection.csv"
    development = "outputs/predictions/usgs_predictions_v2.parquet"
    development_sidecar = f"{development}.meta.json"
    bundle = f"outputs/models/lstm_usgs_bundle_{run_id}"
    bundle_metadata = f"{bundle}/metadata.json"
    bundle_weights = f"{bundle}/weights.pt"
    shortcut = "outputs/models/lstm_usgs_bundle.json"
    components = "outputs/models/route_a_lstm_components.json"
    for relative, payload in (
        (run_manifest, "{}\n"),
        (selection, "candidate_id,val_station_macro_rmse,selected\n1,0.2,true\n"),
        (development, "stage16 development predictions\n"),
        (development_sidecar, "{}\n"),
        (bundle_metadata, "{}\n"),
        (bundle_weights, "five-member weights\n"),
        (shortcut, "{}\n"),
        (components, "{}\n"),
    ):
        _write(root, relative, payload)

    seed_paths = [
        path
        for seed in range(5)
        for path in (
            f"{run_dir}/predictions/seed{seed}.parquet",
            f"{run_dir}/predictions/seed{seed}.parquet.meta.json",
        )
    ]
    candidate_paths = [
        path
        for candidate_id in range(3)
        for path in (
            f"{run_dir}/selection/candidate{candidate_id}.parquet",
            f"{run_dir}/selection/candidate{candidate_id}.parquet.meta.json",
            f"{run_dir}/selection/candidate{candidate_id}.pt",
            f"{run_dir}/selection/candidate{candidate_id}.pt.meta.json",
        )
    ]
    for relative in seed_paths + candidate_paths:
        _write(root, relative, f"fixture:{relative}\n")

    artifacts: dict[str, Any] = {
        "run_manifest": _binding(root, run_manifest),
        "stage09_completion_receipt": _binding(root, stage09_path),
        "stage09_parent_predictions": _binding(
            root, "outputs/predictions/usgs_predictions_stage9_v2.parquet"
        ),
        "stage09_parent_prediction_sidecar": _binding(
            root,
            "outputs/predictions/usgs_predictions_stage9_v2.parquet.meta.json",
        ),
        "lstm_validation_selection": _binding(root, selection),
        "development_predictions": _binding(root, development),
        "development_prediction_sidecar": _binding(root, development_sidecar),
        "model_files": [
            _binding(root, bundle_metadata),
            _binding(root, bundle_weights),
        ],
        "lstm_seed_prediction_files": [
            _binding(root, relative) for relative in seed_paths
        ],
        "selection_candidate_files": [
            _binding(root, relative) for relative in candidate_paths
        ],
        "shortcut_pointer": _binding(root, shortcut),
        "components_pointer": _binding(root, components),
    }
    threshold_registry = {"fixture-monitoring-site": 1.0}
    threshold_contract = {
        "target": "WTEMP",
        "fit_split": "canonical development train mask",
        "scope": "station-specific",
        "estimator": "pandas Series.quantile(q=0.90, interpolation=linear)",
        "quantile": 0.90,
        "registry": threshold_registry,
        "registry_sha256": _repro_sha(threshold_registry),
    }
    selection_input_closure = {
        "run_manifest": artifacts["run_manifest"],
        "panel": _binding(root, "data_usgs/panel_usgs_120v2.parquet"),
        "frozen_panel_spec": _binding(root, "data_usgs/frozen_panel_v1.json"),
        "station_registry": _binding(root, "data_usgs/station_registry_v1.csv"),
        "selection": artifacts["lstm_validation_selection"],
        "candidate_files": artifacts["selection_candidate_files"],
        "event_threshold_contract": threshold_contract,
    }
    metrics = (0.30, 0.20, 0.40)
    selection_audit = {
        "format": "thermoroute.stage16-selection-audit.v1",
        "status": "PASS_BEST_STATE_REPLAY_AND_VALIDATION_SELECTION_PARITY",
        "metric": "mean_station_rmse_across_all_horizons",
        "selection_split": "2016-2017 validation",
        "metric_atol": 1e-5,
        "replay_atol": 1e-5,
        "winner_candidate_id": 1,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "recomputed_val_station_macro_rmse": metric,
                "reported_val_station_macro_rmse": metric,
                "checkpoint_best_metric": metric,
                "best_state_max_abs_difference": 0.0,
                "selected": candidate_id == 1,
            }
            for candidate_id, metric in enumerate(metrics)
        ],
        "input_closure": selection_input_closure,
        "input_closure_sha256": _repro_sha(selection_input_closure),
    }
    development_prediction = {
        "artifact": {
            **artifacts["development_predictions"],
            "sidecar": artifacts["development_prediction_sidecar"],
        },
        "rows": 1,
        "selection": {"model": "LSTM", "seeds": list(range(5))},
        "forecast_key_columns": [
            "site_id", "horizon", "issue_date", "target_date",
        ],
        "prediction_columns": ["model", "site_id", "y_true", "y_pred"],
        "forecast_key_registry_sha256": "c" * 64,
        "prediction_sha256": "d" * 64,
        "max_abs_difference": 0.0,
        "atol": 1e-5,
    }
    parity_input_closure = {
        "panel": _binding(root, "data_usgs/panel_usgs_120v2.parquet"),
        "frozen_panel_spec": _binding(root, "data_usgs/frozen_panel_v1.json"),
        "station_registry": _binding(root, "data_usgs/station_registry_v1.csv"),
        "bundle_metadata": artifacts["model_files"][0],
        "bundle_weights": artifacts["model_files"][1],
        "development_prediction": development_prediction,
    }
    parity_audit = {
        "format": "thermoroute.stage16-bundle-parity.v1",
        "status": "PASS_FIVE_MEMBER_VAL_CALIB_TEST_REPLAY",
        "members": [f"seed{seed}" for seed in range(5)],
        "splits": ["val", "calib", "test"],
        "atol": 1e-5,
        "max_abs_difference": 0.0,
        "input_closure": parity_input_closure,
        "input_closure_sha256": _repro_sha(parity_input_closure),
    }
    if attack == "missing_seed_binding":
        artifacts["lstm_seed_prediction_files"].pop()
    elif attack == "tampered_selection_audit":
        selection_audit["candidates"][1]["best_state_max_abs_difference"] = 0.5
    elif attack == "three_view_winner_disagreement":
        selection_audit["winner_candidate_id"] = 0
        reported = (0.300000, 0.299991, 0.400000)
        replayed = (0.300002, 0.299990, 0.400000)
        for candidate_id, candidate in enumerate(selection_audit["candidates"]):
            candidate["reported_val_station_macro_rmse"] = reported[candidate_id]
            candidate["recomputed_val_station_macro_rmse"] = replayed[candidate_id]
            candidate["checkpoint_best_metric"] = replayed[candidate_id]
            candidate["selected"] = candidate_id == 0
    elif attack not in {None, "missing_gate"}:
        raise AssertionError(f"unknown Stage-16 fixture attack: {attack}")

    receipt = {
        "format": "thermoroute.stage16-completion-receipt.v1",
        "status": "PASS_FORMAL_STAGE16_COMPLETE",
        "stage": "16_lstm_baseline_insample",
        "run_id": run_id,
        "parent_stage09_run_id": stage09_run_id,
        "run_identity": identity,
        "formal_configuration": configuration,
        "training_device": "cpu",
        "confirmation_outcomes_requested_or_read": False,
        "selection_audit": selection_audit,
        "bundle_prediction_parity": parity_audit,
        "artifacts": artifacts,
        "artifact_closure_sha256": _repro_sha(artifacts),
    }
    receipt["receipt_self_sha256"] = _repro_sha(receipt)
    receipt_path = "outputs/models/route_a_stage16_completion.json"
    _write(root, receipt_path, _json_bytes(receipt))
    return {"path": receipt_path, "run_id": run_id, "artifacts": artifacts}


def _seed_model_matrix_governance(
    root: Path,
    *,
    attack: str | None = None,
) -> dict[str, str]:
    amendment = {
        "format": MODEL_MATRIX_AMENDMENT_FORMAT,
        "status": MODEL_MATRIX_AMENDMENT_STATUS,
        "amendment_id": MODEL_MATRIX_AMENDMENT_ID,
        "recorded_date": "2026-07-25",
        "governance_inputs": {},
        "primary_contract_object_bindings": {},
        "scientific_scope": {},
        "stage09_architecture_control_matrix": {"fixture": "stage09"},
        "stage09b_development_control_matrix": {"fixture": "stage09b"},
        "unchanged_primary_boundary": {},
        "lineage_contract": MODEL_MATRIX_DOCUMENT_LINEAGE_CONTRACT,
        "prelabel_attestation": MODEL_MATRIX_PRELABEL_ATTESTATION,
    }
    _write(root, DEFAULT_MODEL_MATRIX_AMENDMENT, _json_bytes(amendment))
    document_commit = _commit(root, "freeze fixture model-matrix amendment")
    seal = {
        "format": MODEL_MATRIX_SEAL_FORMAT,
        "status": MODEL_MATRIX_SEAL_STATUS,
        "amendment_id": MODEL_MATRIX_AMENDMENT_ID,
        "amendment": {
            "path": DEFAULT_MODEL_MATRIX_AMENDMENT,
            "sha256": _file_sha(root, DEFAULT_MODEL_MATRIX_AMENDMENT),
        },
        "amendment_document_commit": (
            "0" * 40 if attack == "wrong_document_commit" else document_commit
        ),
        "governance_seals": {},
        "history_contract": MODEL_MATRIX_SEAL_HISTORY_CONTRACT,
        "prelabel_attestation": MODEL_MATRIX_PRELABEL_ATTESTATION,
    }
    _write(root, DEFAULT_MODEL_MATRIX_AMENDMENT_SEAL, _json_bytes(seal))
    if attack != "seal_at_model_freeze":
        seal_commit = _commit(root, "seal fixture model-matrix amendment")
    else:
        seal_commit = ""
    if attack == "rewrite_document_after_seal":
        amendment["scientific_scope"] = {"changed_after_seal": True}
        _write(root, DEFAULT_MODEL_MATRIX_AMENDMENT, _json_bytes(amendment))
    return {
        "document_commit": document_commit,
        "seal_commit": seal_commit,
    }


def _seed_model_commit(
    root: Path,
    *,
    original_commit: str,
    final_commit: str,
    matrix_document_commit: str,
    matrix_attack: str | None,
    leak_before_model: bool,
    lightgbm_bundle_format: str = "thermoroute.lightgbm-bundle.v2",
    stage16_attack: str | None = None,
) -> str:
    original_markdown = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{original_commit}:protocols/route_a_confirmatory_protocol.md",
        ],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    final_json = (root / "protocols/route_a_confirmatory_v1.json").read_bytes()
    final_markdown = (root / "protocols/route_a_confirmatory_protocol.md").read_bytes()
    seal = {
        "format": "thermoroute.route-a-protocol-seal.v1",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": "protocols/route_a_confirmatory_protocol.md",
                "sha256": _sha(original_markdown),
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": "protocols/route_a_confirmatory_v1.json",
                "sha256": _sha(final_json),
            },
            "markdown": {
                "path": "protocols/route_a_confirmatory_protocol.md",
                "sha256": _sha(final_markdown),
            },
        },
    }
    _write(root, "protocols/route_a_protocol_seal_v1.json", _json_bytes(seal))
    for path in REQUIRED_GATE_PATHS:
        if not (root / path).exists():
            _write(root, path, f"# frozen gate fixture: {path}\n")

    _write(root, "data_usgs/frozen_panel_v1.json", "{}\n")
    _write(root, "data_usgs/panel_usgs_120v2.parquet", b"development-panel")
    _write(root, "data_usgs/station_registry_v1.csv", "site_no,lat,lon\n1,1,2\n")
    _write(root, "outputs/development/predictions.parquet", b"predictions")
    _write(root, "outputs/development/predictions.parquet.meta.json", "{}\n")
    prediction = {
        "artifact": {
            **_binding(root, "outputs/development/predictions.parquet"),
            "sidecar": _binding(
                root, "outputs/development/predictions.parquet.meta.json"
            ),
        }
    }

    weights = b"safe-weights"
    _write(root, "outputs/models/torch/weights.pt", weights)
    torch_metadata = {
        "format": "thermoroute.inference-bundle.v2",
        "weights_sha256": _sha(weights),
        "output_head_schema": neural_output_head_schema(),
        "development_prediction": prediction,
    }
    _write(
        root,
        "outputs/models/torch/metadata.json",
        _json_bytes(torch_metadata),
    )
    model_text = b"tree\n"
    _write(root, "outputs/models/lgb/member_h1_point.txt", model_text)
    lgb_manifest = {
        "format": lightgbm_bundle_format,
        "models": {
            "seed0": {
                "1": {
                    "point": {
                        "path": "member_h1_point.txt",
                        "sha256": _sha(model_text),
                    }
                }
            }
        },
        "development_prediction": prediction,
    }
    _write(root, "outputs/models/lgb/manifest.json", _json_bytes(lgb_manifest))

    # Freeze a predictor bridge made by a deliberately different source tree.
    # Its own source identity is valid but is not the later training identity.
    bridge_normalized = {}
    for name in ("frozen", "refreshed"):
        relative = f"data_usgs/bridge/{name}.parquet"
        _write(root, relative, f"{name} predictors".encode())
        bridge_normalized[name] = _binding(root, relative)
    bridge_indexes = {}
    for name in ("daymet", "gridmet", "gridmet_schema"):
        relative = f"data_usgs/bridge/{name}/snapshot_index_v2.json"
        _snapshot(root, relative, payload=f"{name} response".encode())
        bridge_indexes[name] = _binding(root, relative)
    _write(root, "data_usgs/bridge/report.json", "{}\n")
    _write(root, "data_usgs/bridge/request_map.json", "{}\n")
    bridge = {
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": "b" * 64,
        "panel": _binding(root, "data_usgs/panel_usgs_120v2.parquet"),
        "registry": _binding(root, "data_usgs/station_registry_v1.csv"),
        "normalized": bridge_normalized,
        "raw_snapshot_indexes": bridge_indexes,
        "report": _binding(root, "data_usgs/bridge/report.json"),
        "request_map": _binding(root, "data_usgs/bridge/request_map.json"),
    }
    bridge_path = "data_usgs/development_predictor_bridge_v1.json"
    _write(root, bridge_path, _json_bytes(bridge))

    development_input_closure_sha256 = "c" * 64
    stage9_configuration = {
        "fixture": True,
        "input_closure_sha256": development_input_closure_sha256,
        "input_closure_file_count": 1,
    }
    stage9_identity = _fixture_run_identity(
        stage9_configuration,
        panel_sha256=_file_sha(
            root, "data_usgs/panel_usgs_120v2.parquet"
        ),
        registry_sha256=_file_sha(
            root, "data_usgs/station_registry_v1.csv"
        ),
        input_closure_sha256=development_input_closure_sha256,
    )
    stage9_run_id = stage9_identity["run_id"]
    stage9_run_manifest = (
        f"outputs/runs/09_usgs_experiment/{stage9_run_id}/run.json"
    )
    _write(root, stage9_run_manifest, _json_bytes({
        "schema_version": "thermoroute.run.v2",
        "identity": stage9_identity,
        "resolved_config": stage9_configuration,
    }))
    for label, relative in STAGE09_ARTIFACT_PATHS.items():
        _write(root, relative, f"stage09 {label}\n")
    stage9 = {
        "format": "thermoroute.stage09-completion-receipt.v1",
        "status": "PASS_FORMAL_STAGE09_COMPLETE",
        "stage": "09_usgs_experiment",
        "run_id": stage9_run_id,
        "run_identity": stage9_identity,
        "formal_configuration": stage9_configuration,
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": {
            "run_manifest": _binding(root, stage9_run_manifest),
            **{
                label: _binding(root, relative)
                for label, relative in STAGE09_ARTIFACT_PATHS.items()
            },
        },
    }
    stage9["receipt_self_sha256"] = _repro_sha(stage9)
    stage9_path = "outputs/models/route_a_stage09_completion.json"
    _write(root, stage9_path, _json_bytes(stage9))
    stage16 = _seed_stage16_completion(
        root,
        stage09_path=stage9_path,
        stage09_run_id=stage9_run_id,
        attack=stage16_attack,
    )

    stage09b_configuration = {
        "fixture": True,
        "input_closure_sha256": development_input_closure_sha256,
        "input_closure_file_count": 1,
    }
    stage09b_identity = _fixture_run_identity(
        stage09b_configuration,
        panel_sha256=_file_sha(
            root, "data_usgs/panel_usgs_120v2.parquet"
        ),
        registry_sha256=_file_sha(
            root, "data_usgs/station_registry_v1.csv"
        ),
        input_closure_sha256=development_input_closure_sha256,
    )
    stage09b_run_id = stage09b_identity["run_id"]
    stage09b_run_dir = (
        f"outputs/runs/09b_development_controls/{stage09b_run_id}"
    )
    _write(root, f"{stage09b_run_dir}/run.json", _json_bytes({
        "schema_version": "thermoroute.run.v2",
        "identity": stage09b_identity,
        "resolved_config": stage09b_configuration,
    }))
    members = []
    semantic_members = []
    for arm_id, seed in STAGE09B_MEMBERS:
        prediction_path = (
            f"{stage09b_run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
        )
        prediction_sidecar = f"{prediction_path}.meta.json"
        checkpoint_path = (
            f"{stage09b_run_dir}/checkpoints/{arm_id}/seed{seed}.pt"
        )
        checkpoint_sidecar = f"{checkpoint_path}.meta.json"
        _write(root, prediction_path, f"{arm_id}/seed{seed}".encode())
        _write(root, prediction_sidecar, "{}\n")
        _write(root, checkpoint_path, f"checkpoint:{arm_id}/seed{seed}".encode())
        _write(root, checkpoint_sidecar, "{}\n")
        members.append({
            "arm_id": arm_id,
            "seed": seed,
            "checkpoint": _binding(root, checkpoint_path),
            "checkpoint_sidecar": _binding(root, checkpoint_sidecar),
            "prediction": _binding(root, prediction_path),
            "prediction_sidecar": _binding(root, prediction_sidecar),
        })
        semantic_members.append({
            "arm_id": arm_id,
            "seed": seed,
            "prediction": {
                "sha256": _file_sha(root, prediction_path),
                "bytes": (root / prediction_path).stat().st_size,
            },
            "prediction_sidecar": {
                "sha256": _file_sha(root, prediction_sidecar),
                "bytes": (root / prediction_sidecar).stat().st_size,
            },
            "checkpoint": {
                "sha256": _file_sha(root, checkpoint_path),
                "bytes": (root / checkpoint_path).stat().st_size,
            },
            "checkpoint_sidecar": {
                "sha256": _file_sha(root, checkpoint_sidecar),
                "bytes": (root / checkpoint_sidecar).stat().st_size,
            },
            "normalised_prediction_sha256": _sha(
                f"normalised:{arm_id}:{seed}".encode()
            ),
            "best_model_state_prediction_replay_verified": True,
        })
    final_paths = {
        "predictions": f"{stage09b_run_dir}/development_controls_predictions.parquet",
        "prediction_sidecar": (
            f"{stage09b_run_dir}/development_controls_predictions.parquet.meta.json"
        ),
        "architecture_budget": (
            f"{stage09b_run_dir}/development_controls_architecture_budget.csv"
        ),
        "architecture_budget_sidecar": (
            f"{stage09b_run_dir}/development_controls_architecture_budget.csv.meta.json"
        ),
        "metric_summary": (
            f"{stage09b_run_dir}/development_controls_metric_summary.csv"
        ),
        "metric_summary_sidecar": (
            f"{stage09b_run_dir}/development_controls_metric_summary.csv.meta.json"
        ),
        "report": f"{stage09b_run_dir}/development_controls_report.md",
        "report_sidecar": (
            f"{stage09b_run_dir}/development_controls_report.md.meta.json"
        ),
    }
    for label, relative in final_paths.items():
        _write(root, relative, f"stage09b {label}\n")

    matrix_audit = {
        "expected_members": len(STAGE09B_MEMBERS),
        "prediction_rows": len(STAGE09B_MEMBERS) * 3,
        "common_forecast_keys": 3,
        "splits": ["calib", "test", "val"],
        "reference_member": "PlainMLP-7var/seed0",
    }
    descriptor = lambda relative: {  # noqa: E731 - compact fixture helper
        "sha256": _file_sha(root, relative),
        "bytes": (root / relative).stat().st_size,
    }
    semantic = {
        "format": "thermoroute.development-controls-semantic-audit.v3",
        "status": "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "run_id": stage09b_run_id,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
        "post_2020_outcomes_requested_or_read": False,
        "matrix_audit": matrix_audit,
        "canonical_window_registry": {
            "sha256": "c" * 64,
            "common_forecast_keys": 3,
            "train_examples_per_epoch": 3,
            "train_registry_sha256": "d" * 64,
        },
        "scientific_summary": _stage09b_scientific_summary(),
        "members": semantic_members,
        "derived_artifacts": {
            "architecture_budget": {
                "artifact": descriptor(final_paths["architecture_budget"]),
                "sidecar": descriptor(final_paths["architecture_budget_sidecar"]),
            },
            "combined_predictions": {
                "artifact": descriptor(final_paths["predictions"]),
                "sidecar": descriptor(final_paths["prediction_sidecar"]),
            },
            "metric_summary": {
                "artifact": descriptor(final_paths["metric_summary"]),
                "sidecar": descriptor(final_paths["metric_summary_sidecar"]),
            },
            "report": {
                "artifact": descriptor(final_paths["report"]),
                "sidecar": descriptor(final_paths["report_sidecar"]),
            },
        },
    }
    semantic["semantic_audit_self_sha256"] = _repro_sha(semantic)
    semantic_path = f"{stage09b_run_dir}/development_controls_semantic_audit.json"
    _write(root, semantic_path, _json_bytes(semantic))
    _write(root, f"{semantic_path}.meta.json", "{}\n")
    stage09b = {
        "format": "thermoroute.stage09b-completion-receipt.v3",
        "status": "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "stage": "09b_development_controls",
        "run_id": stage09b_run_id,
        "run_identity": stage09b_identity,
        "formal_configuration": stage09b_configuration,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
        "matrix_audit": matrix_audit,
        "member_registry": members,
        "artifacts": {
            "run_manifest": _binding(root, f"{stage09b_run_dir}/run.json"),
            "frozen_panel_spec": _binding(root, "data_usgs/frozen_panel_v1.json"),
            "panel": _binding(root, "data_usgs/panel_usgs_120v2.parquet"),
            "registry": _binding(root, "data_usgs/station_registry_v1.csv"),
            "predictor_bridge": _binding(root, bridge_path),
            **{
                label: _binding(root, relative)
                for label, relative in final_paths.items()
            },
            "semantic_audit": _binding(root, semantic_path),
            "semantic_audit_sidecar": _binding(root, f"{semantic_path}.meta.json"),
        },
        "post_2020_outcomes_requested_or_read": False,
    }
    stage09b["receipt_self_sha256"] = _repro_sha(stage09b)
    stage09b_path = "outputs/models/route_a_stage09b_completion.json"
    _write(root, stage09b_path, _json_bytes(stage09b))

    stage25_input_closure_sha256 = _compose_input_closure({
        "development": development_input_closure_sha256,
    })
    stage25_configuration = {
        "fixture": True,
        "input_closure_sha256": stage25_input_closure_sha256,
        "input_closure_file_count": 1,
        "input_closure_component_count": 1,
    }
    stage25_identity = _fixture_run_identity(
        stage25_configuration,
        panel_sha256=_file_sha(
            root, "data_usgs/panel_usgs_120v2.parquet"
        ),
        registry_sha256=_file_sha(
            root, "data_usgs/station_registry_v1.csv"
        ),
        input_closure_sha256=stage25_input_closure_sha256,
    )
    stage25_run_id = stage25_identity["run_id"]
    stage25_run_manifest = (
        f"outputs/runs/25_external_pooled/{stage25_run_id}/run.json"
    )
    stage25_pointer = "outputs/models/route_a_external_components.json"
    stage25_predictions = (
        "outputs/predictions/"
        f"external_pooled_development_{stage25_run_id}.parquet"
    )
    stage25_prediction_sidecar = f"{stage25_predictions}.meta.json"
    for relative, payload in (
        (stage25_run_manifest, _json_bytes({
            "schema_version": "thermoroute.run.v2",
            "identity": stage25_identity,
            "resolved_config": stage25_configuration,
        })),
        (stage25_pointer, "{}\n"),
        (stage25_predictions, "stage25 development predictions\n"),
        (stage25_prediction_sidecar, "{}\n"),
    ):
        _write(root, relative, payload)
    stage25_model_paths = [
        f"outputs/models/stage25-fixture/model_{index:03d}.bin"
        for index in range(80)
    ]
    for index, relative in enumerate(stage25_model_paths):
        _write(root, relative, f"model-{index}\n")
    stage25_artifacts = {
        "run_manifest": _binding(root, stage25_run_manifest),
        "components_pointer": _binding(root, stage25_pointer),
        "development_predictions": _binding(root, stage25_predictions),
        "development_prediction_sidecar": _binding(
            root, stage25_prediction_sidecar
        ),
        "frozen_panel_spec": _binding(root, "data_usgs/frozen_panel_v1.json"),
        "development_panel": _binding(
            root, "data_usgs/panel_usgs_120v2.parquet"
        ),
        "station_registry": _binding(root, "data_usgs/station_registry_v1.csv"),
        "development_predictor_bridge": _binding(root, bridge_path),
        "model_files": [
            _binding(root, relative) for relative in stage25_model_paths
        ],
    }
    stage25 = {
        "format": "thermoroute.stage25-completion-receipt.v1",
        "status": "COMPLETE",
        "stage": "25_train_external_pooled_suite",
        "run_id": stage25_run_id,
        "run_identity": stage25_identity,
        "formal_configuration": stage25_configuration,
        "training_device": "cpu",
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": stage25_artifacts,
        "artifact_closure_sha256": _repro_sha(stage25_artifacts),
    }
    stage25["receipt_self_sha256"] = _repro_sha(stage25)
    stage25_path = "outputs/models/route_a_stage25_completion.json"
    _write(root, stage25_path, _json_bytes(stage25))

    # This is computed only after every source/protocol/control fixture byte is
    # present.  The frozen suite and replay must agree with the exact source
    # inventory committed alongside the model artifacts.
    frozen_source_sha256 = source_tree_hash(root)

    suite: dict[str, Any] = {
        "format": "thermoroute.route-a-model-suite.v1",
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "model_matrix_amendment": {
            "format": "thermoroute.route-a-model-matrix-suite-binding.v1",
            "document": {
                "path": DEFAULT_MODEL_MATRIX_AMENDMENT,
                "sha256": _file_sha(root, DEFAULT_MODEL_MATRIX_AMENDMENT),
                "format": MODEL_MATRIX_AMENDMENT_FORMAT,
                "status": MODEL_MATRIX_AMENDMENT_STATUS,
                "amendment_id": MODEL_MATRIX_AMENDMENT_ID,
                "amendment_document_commit": matrix_document_commit,
            },
            "seal": {
                "path": DEFAULT_MODEL_MATRIX_AMENDMENT_SEAL,
                "sha256": _file_sha(root, DEFAULT_MODEL_MATRIX_AMENDMENT_SEAL),
                "format": MODEL_MATRIX_SEAL_FORMAT,
                "status": MODEL_MATRIX_SEAL_STATUS,
            },
            "contract_id": _repro_sha(
                {
                    "format": "thermoroute.route-a-model-matrix-contract.v1",
                    "stage09_architecture_control_matrix": {
                        "fixture": "stage09"
                    },
                    "stage09b_development_control_matrix": {
                        "fixture": "stage09b"
                    },
                }
            ),
        },
        "development_contract": {
            "frozen_panel_spec": _binding(root, "data_usgs/frozen_panel_v1.json"),
            "panel": _binding(root, "data_usgs/panel_usgs_120v2.parquet"),
            "registry": _binding(root, "data_usgs/station_registry_v1.csv"),
            "predictor_bridge": _binding(root, bridge_path),
            "source_sha256": frozen_source_sha256,
        },
        "preopening_gates": {
            "stage09_completion": _binding(root, stage9_path),
            **(
                {}
                if stage16_attack == "missing_gate"
                else {"stage16_lstm_completion": _binding(root, stage16["path"])}
            ),
            "stage09b_development_controls": _binding(root, stage09b_path),
            "stage25_external_completion": _binding(root, stage25_path),
        },
        "cohorts": {
            "temporal": {
                "models": [
                    {"model_id": "Persistence", "executor": "builtin"},
                    {
                        "model_id": "ThermoRoute",
                        "executor": "thermoroute_bundle",
                        "artifact": {
                            "path": "outputs/models/torch",
                            "metadata_sha256": _file_sha(
                                root, "outputs/models/torch/metadata.json"
                            ),
                            "weights_sha256": _sha(weights),
                        },
                    },
                ]
            },
            "external": {
                "models": [
                    {
                        "model_id": "LightGBM",
                        "executor": "lightgbm_bundle",
                        "artifact": _binding(root, "outputs/models/lgb/manifest.json"),
                    }
                ]
            },
        },
    }
    if matrix_attack == "suite_contract_id":
        suite["model_matrix_amendment"]["contract_id"] = "f" * 64
    suite_path = "data_usgs/confirmatory_model_suite_v1.json"
    _write(root, suite_path, _json_bytes(suite))
    replay = {
        "format": "thermoroute.route-a-development-replay.v1",
        "suite": _binding(root, suite_path),
        "source_tree_sha256": frozen_source_sha256,
    }
    replay["receipt_self_sha256"] = _repro_sha(replay)
    _write(
        root,
        "outputs/model_replay/route_a_development_replay_v1.json",
        _json_bytes(replay),
    )
    if leak_before_model:
        _write(root, "data_usgs/confirmatory_candidate_sites_v1.csv", "site_no\n9\n")
    return _commit(root, "freeze model suite and chronology implementation")


def _seed_evidence_commit(root: Path, *, candidate_already_exists: bool) -> str:
    candidate_table = "data_usgs/confirmatory_candidate_sites_v1.csv"
    if not candidate_already_exists:
        _write(root, candidate_table, "site_no\n9\n")
    candidate_provenance = "data_usgs/confirmatory_candidate_sites_v1.provenance.json"
    _write(root, candidate_provenance, "{}\n")
    candidate_index = (
        "data_usgs/raw_snapshots/confirmatory-candidates-v1/snapshot_index.json"
    )
    _snapshot(root, candidate_index, payload=b"candidate metadata")
    external_registry = "data_usgs/confirmatory_site_registry_v1.csv"
    _write(root, external_registry, "site_no,lat,lon\n9,3,4\n")
    external_lock = "data_usgs/confirmatory_site_registry_v1.lock.json"
    lock = {
        "schema_version": 1,
        "status": "REGISTRY_FROZEN_LABELS_SEALED",
        "confirmatory_registry_sha256": _file_sha(root, external_registry),
        "frozen_artifacts": {
            "development_panel_spec": _binding(root, "data_usgs/frozen_panel_v1.json"),
            "candidate_table": _binding(root, candidate_table),
            "candidate_provenance": _binding(root, candidate_provenance),
            "candidate_snapshot_index": _binding(root, candidate_index),
        },
    }
    _write(root, external_lock, _json_bytes(lock))

    temporal_table = (
        "data_usgs/confirmatory_predictors/historical-retrospective-v1/"
        "temporal_retrospective_meteorology_v1.parquet"
    )
    external_table = (
        "data_usgs/confirmatory_predictors/historical-retrospective-v1/"
        "external_retrospective_meteorology_v1.parquet"
    )
    request_map = (
        "data_usgs/confirmatory_predictors/historical-retrospective-v1/"
        "source_request_map_v1.json"
    )
    _write(root, temporal_table, b"temporal met")
    _write(root, external_table, b"external met")
    _write(root, request_map, "{}\n")
    met_index = (
        "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1/daymet-v1/"
        "snapshot_index.json"
    )
    _snapshot(root, met_index, payload=b"meteorology")
    manifest = {
        "format": "thermoroute.route-a-prelabel-inputs.v1",
        "status": "FROZEN_PRELABEL_NO_OUTCOMES",
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "post_2020_wtemp_requested_or_inspected": False,
        "cohort_tables": {
            "temporal": _binding(root, temporal_table),
            "external": _binding(root, external_table),
        },
        "registry_inputs": {
            "temporal": {
                **_binding(root, "data_usgs/station_registry_v1.csv"),
                "columns_read": ["site_no", "lat", "lon"],
            },
            "external": {
                **_binding(root, external_registry),
                "columns_read": ["site_no", "lat", "lon"],
            },
        },
        "source_evidence": [
            {
                "evidence_type": "snapshot_index",
                "contains_outcome": False,
                "contains_outcome_labels": False,
                "artifact": _binding(root, met_index),
            },
            {
                "evidence_type": "normalized_immutable_snapshot",
                "contains_outcome": False,
                "contains_outcome_labels": False,
                "artifact": _binding(root, request_map),
            },
        ],
    }
    _write(
        root,
        "data_usgs/confirmatory_actual_inputs_v1.json",
        _json_bytes(manifest),
    )
    return _commit(root, "freeze label-free candidate and input evidence")


def _repository(
    tmp_path: Path,
    *,
    leak_before_model: bool = False,
    source_change: bool = False,
    creation_base: bool = True,
    lightgbm_bundle_format: str = "thermoroute.lightgbm-bundle.v2",
    stage16_attack: str | None = None,
    matrix_attack: str | None = None,
) -> dict[str, Any]:
    root = tmp_path / "repo"
    root.mkdir()
    _run(root, "init", "-q")
    _run(root, "config", "user.email", "route-a@example.invalid")
    _run(root, "config", "user.name", "Route A test")
    _write(root, "protocols/route_a_confirmatory_protocol.md", "original protocol\n")
    original = _commit(root, "original preregistration")
    _write(root, "protocols/route_a_confirmatory_protocol.md", "final protocol\n")
    _write(root, "protocols/route_a_confirmatory_v1.json", "{\"schema_version\": 1}\n")
    final = _commit(root, "final prelabel protocol")
    matrix = _seed_model_matrix_governance(root, attack=matrix_attack)
    model = _seed_model_commit(
        root,
        original_commit=original,
        final_commit=final,
        matrix_document_commit=matrix["document_commit"],
        matrix_attack=matrix_attack,
        leak_before_model=leak_before_model,
        lightgbm_bundle_format=lightgbm_bundle_format,
        stage16_attack=stage16_attack,
    )
    evidence = _seed_evidence_commit(
        root, candidate_already_exists=leak_before_model
    )
    marker = evidence
    if creation_base:
        _write(root, "notes/chronology-marker.txt", "evidence committed\n")
        marker = _commit(root, "mark evidence ready for chronology receipt")
    if source_change:
        _write(root, "src/thermoroute/post_freeze_change.py", "changed = True\n")
        marker = _commit(root, "forbidden post-freeze source change")
    return {
        "root": root,
        "original": original,
        "final": final,
        "matrix_document": matrix["document_commit"],
        "matrix_seal": matrix["seal_commit"],
        "model": model,
        "evidence": evidence,
        "marker": marker,
        "receipt": root / "outputs/prelabel/route_a_prelabel_chronology_v1.json",
    }


def _freeze(state: dict[str, Any]) -> dict[str, Any]:
    return freeze_prelabel_chronology(
        state["receipt"],
        root=state["root"],
        model_freeze_commit=state["model"],
        input_evidence_commit=state["evidence"],
    )


def _publish_receipt(state: dict[str, Any]) -> str:
    return _commit(state["root"], "publish immutable chronology receipt")


def test_chronology_freezes_and_replays_every_git_bound_artifact(tmp_path):
    state = _repository(tmp_path)
    document = _freeze(state)
    _publish_receipt(state)
    assert document["status"] == "PASS_REPOSITORY_INTERNAL_PRELABEL_ORDER"
    assert document["order"]["model_freeze_commit"] == state["model"]
    assert document["order"]["input_evidence_commit"] == state["evidence"]
    assert document["model_matrix_history"]["amendment_document_commit"] == state[
        "matrix_document"
    ]
    assert document["model_matrix_history"]["seal_commit"] == state["matrix_seal"]
    assert document["model_matrix_history"]["model_freeze_commit"] == state["model"]
    assert document["model_matrix_history"]["strict_order_verified"] is True
    assert len(document["model_freeze_artifacts"]) >= 10
    assert len(document["input_evidence_artifacts"]) >= 15
    assert document["external_timestamp_or_public_preregistration"] is False
    stage16 = json.loads(
        (state["root"] / "outputs/models/route_a_stage16_completion.json")
        .read_text(encoding="utf-8")
    )
    stage16_paths = {
        str(binding["path"])
        for value in stage16["artifacts"].values()
        for binding in (value if isinstance(value, list) else [value])
    }
    chronology_paths = {
        str(binding["path"]) for binding in document["model_freeze_artifacts"]
    }
    assert stage16_paths <= chronology_paths
    assert len(stage16["artifacts"]["model_files"]) == 2
    assert len(stage16["artifacts"]["lstm_seed_prediction_files"]) == 10
    assert len(stage16["artifacts"]["selection_candidate_files"]) == 12
    assert validate_prelabel_chronology(
        state["receipt"], root=state["root"]
    ) == document


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("seal_at_model_freeze", "amendment seal < model freeze"),
        ("wrong_document_commit", "model-matrix seal contract changed"),
        ("rewrite_document_after_seal", "working-tree bytes differ|changed after"),
        ("suite_contract_id", "model suite and model-matrix history differ"),
    ),
)
def test_chronology_rejects_broken_model_matrix_lineage(
    tmp_path, attack, message,
):
    state = _repository(tmp_path, matrix_attack=attack)
    with pytest.raises(ChronologyError, match=message):
        _freeze(state)


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("missing_gate", "exact Stage-09/09b/16/25 completion gates"),
        ("missing_seed_binding", "wrong cardinality"),
        ("tampered_selection_audit", "candidate best-state replay changed"),
        (
            "three_view_winner_disagreement",
            "three validation metric views do not select the same",
        ),
    ),
)
def test_chronology_rejects_missing_or_resealed_stage16_gate(
    tmp_path, attack, message,
):
    state = _repository(tmp_path, stage16_attack=attack)
    with pytest.raises(ChronologyError, match=message):
        _freeze(state)


def test_chronology_requires_post_evidence_outcome_free_creation_base(tmp_path):
    state = _repository(tmp_path, creation_base=False)
    assert _run(state["root"], "rev-parse", "HEAD") == state["evidence"]
    with pytest.raises(
        ChronologyError,
        match="input evidence < chronology receipt creation base",
    ):
        _freeze(state)

    gate = {
        "format": "thermoroute.route-a-inference-gate.v1",
        "status": "FAIL_CLOSED_DESCRIPTIVE_ONLY",
        "contains_confirmation_outcomes": False,
        "post_2020_outcomes_requested_or_inspected": False,
    }
    _write(
        state["root"],
        "outputs/prelabel/route_a_inference_gate_v1.json",
        _json_bytes(gate),
    )
    gate_commit = _commit(
        state["root"], "freeze outcome-free inference gate as receipt base"
    )
    document = _freeze(state)
    assert document["order"]["receipt_creation_base_commit"] == gate_commit
    _publish_receipt(state)
    assert validate_prelabel_chronology(
        state["receipt"], root=state["root"]
    ) == document


@pytest.mark.parametrize(
    ("bundle_format", "accepted"),
    (
        ("thermoroute.lightgbm-bundle.v2", True),
        ("thermoroute.lightgbm-bundle.v1", False),
    ),
)
def test_chronology_requires_lightgbm_bundle_v2(
    tmp_path, bundle_format, accepted
):
    state = _repository(tmp_path, lightgbm_bundle_format=bundle_format)
    if accepted:
        document = _freeze(state)
        assert document["status"] == "PASS_REPOSITORY_INTERNAL_PRELABEL_ORDER"
    else:
        with pytest.raises(
            ChronologyError, match="unsupported LightGBM bundle in model suite"
        ):
            _freeze(state)


def test_gitless_archive_replays_current_chronology_bound_bytes(tmp_path):
    state = _repository(tmp_path)
    document = _freeze(state)
    _publish_receipt(state)
    archive = tmp_path / "archive"
    shutil.copytree(state["root"], archive, ignore=shutil.ignore_patterns(".git"))
    receipt = archive / "outputs/prelabel/route_a_prelabel_chronology_v1.json"
    assert validate_prelabel_chronology(
        receipt, root=archive, allow_gitless_archive=True
    ) == document

    target = archive / document["model_freeze_artifacts"][0]["path"]
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ChronologyError, match="archive bytes differ"):
        validate_prelabel_chronology(
            receipt, root=archive, allow_gitless_archive=True
        )


def test_chronology_rejects_worktree_artifact_tamper(tmp_path):
    state = _repository(tmp_path)
    _freeze(state)
    _publish_receipt(state)
    target = state["root"] / "data_usgs/confirmatory_actual_inputs_v1.json"
    target.chmod(0o644)
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ChronologyError, match="working-tree bytes differ"):
        validate_prelabel_chronology(state["receipt"], root=state["root"])


def test_chronology_rejects_non_strict_or_reversed_ancestry(tmp_path):
    state = _repository(tmp_path)
    with pytest.raises(ChronologyError, match="strict Git order"):
        freeze_prelabel_chronology(
            state["receipt"],
            root=state["root"],
            model_freeze_commit=state["evidence"],
            input_evidence_commit=state["model"],
        )


def test_chronology_receipt_must_be_absent_at_creation_base(tmp_path):
    state = _repository(tmp_path)
    relative = state["receipt"].relative_to(state["root"]).as_posix()
    _write(state["root"], relative, "{}\n")
    _commit(state["root"], "premature chronology receipt")
    state["receipt"].unlink()
    with pytest.raises(ChronologyError, match="declared creation base"):
        _freeze(state)


def test_chronology_rejects_uncommitted_or_rewritten_receipt(tmp_path):
    state = _repository(tmp_path)
    _freeze(state)
    original = state["receipt"].read_bytes()
    with pytest.raises(ChronologyError, match="creation base < committed receipt"):
        validate_prelabel_chronology(state["receipt"], root=state["root"])

    state["receipt"].chmod(0o644)
    state["receipt"].write_text("{}\n", encoding="utf-8")
    _commit(state["root"], "publish placeholder chronology receipt")
    state["receipt"].write_bytes(original)
    _commit(state["root"], "rewrite chronology receipt after publication")
    with pytest.raises(ChronologyError, match="added exactly once"):
        validate_prelabel_chronology(state["receipt"], root=state["root"])


def test_chronology_rejects_add_delete_hidden_on_merged_side_branch(tmp_path):
    state = _repository(tmp_path)
    main_branch = _run(state["root"], "branch", "--show-current")
    _run(state["root"], "switch", "-c", "receipt-history-attack")
    relative = state["receipt"].relative_to(state["root"]).as_posix()
    _write(state["root"], relative, "{}\n")
    _commit(state["root"], "side branch adds fake receipt")
    state["receipt"].unlink()
    _commit(state["root"], "side branch deletes fake receipt")
    _run(state["root"], "switch", main_branch)
    _freeze(state)
    _publish_receipt(state)
    _run(
        state["root"],
        "merge",
        "--no-ff",
        "-s",
        "ours",
        "-m",
        "merge hidden receipt history",
        "receipt-history-attack",
    )
    with pytest.raises(ChronologyError, match="added exactly once"):
        validate_prelabel_chronology(state["receipt"], root=state["root"])


def test_chronology_rejects_candidate_artifact_present_at_model_freeze(tmp_path):
    state = _repository(tmp_path, leak_before_model=True)
    with pytest.raises(ChronologyError, match="existed at model freeze"):
        _freeze(state)


def test_chronology_rejects_any_post_model_source_change(tmp_path):
    state = _repository(tmp_path, source_change=True)
    with pytest.raises(
        ChronologyError,
        match="source/control path changed|working source/control path set differs",
    ):
        _freeze(state)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_chronology_rejects_hidden_git_index_flags(tmp_path, flag):
    state = _repository(tmp_path)
    relative = "src/thermoroute/chronology.py"
    _run(state["root"], "update-index", flag, relative)
    _write(state["root"], relative, "# hidden source mutation\n")
    with pytest.raises(
        ChronologyError,
        match="assume-unchanged/skip-worktree flags are prohibited",
    ):
        _freeze(state)


def test_chronology_rejects_ignored_untracked_source(tmp_path):
    state = _repository(tmp_path)
    _write(state["root"], ".git/info/exclude", "src/ignored_attack.py\n")
    _write(state["root"], "src/ignored_attack.py", "ATTACK = True\n")
    with pytest.raises(
        ChronologyError,
        match="working source/control path set differs",
    ):
        _freeze(state)


def test_chronology_rejects_timestamp_valid_compiled_python(tmp_path):
    state = _repository(tmp_path)
    source = state["root"] / "src/thermoroute/chronology.py"
    cache = importlib.util.cache_from_source(str(source))
    py_compile.compile(str(source), cfile=cache, doraise=True)
    with pytest.raises(ChronologyError, match="compiled Python cache is prohibited"):
        _freeze(state)


def test_canonical_run_disables_bytecode_before_first_python_call():
    """The standard pipeline must not manufacture its own chronology blocker."""
    script = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    clear_prefix = script.index("unset PYTHONPYCACHEPREFIX")
    disable = script.index("export PYTHONDONTWRITEBYTECODE=1")
    first_python = script.index('"$THERMOROUTE_PYTHON"')
    assert clear_prefix < disable < first_python


def test_chronology_rejects_git_replace_refs(tmp_path):
    state = _repository(tmp_path)
    _run(state["root"], "replace", state["model"], state["evidence"])
    with pytest.raises(ChronologyError, match="Git replace refs are prohibited"):
        _freeze(state)


def test_chronology_rejects_legacy_grafts(tmp_path):
    state = _repository(tmp_path)
    relative = _run(state["root"], "rev-parse", "--git-path", "info/grafts")
    grafts = Path(relative)
    if not grafts.is_absolute():
        grafts = state["root"] / grafts
    grafts.parent.mkdir(parents=True, exist_ok=True)
    grafts.write_text(f"{state['marker']} {state['model']}\n", encoding="utf-8")
    with pytest.raises(ChronologyError, match="Git legacy grafts are prohibited"):
        _freeze(state)


def test_chronology_rejects_shallow_repository(tmp_path):
    state = _repository(tmp_path)
    relative = _run(state["root"], "rev-parse", "--git-path", "shallow")
    shallow = Path(relative)
    if not shallow.is_absolute():
        shallow = state["root"] / shallow
    shallow.parent.mkdir(parents=True, exist_ok=True)
    shallow.write_text(f"{state['marker']}\n", encoding="ascii")
    with pytest.raises(ChronologyError, match="shallow Git repository"):
        _freeze(state)


def test_chronology_rejects_ambient_git_repository_override(tmp_path, monkeypatch):
    state = _repository(tmp_path)
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    with pytest.raises(ChronologyError, match="ambient Git.*override is prohibited"):
        _freeze(state)


def test_chronology_fails_without_git_repository(tmp_path):
    with pytest.raises(ChronologyError, match="Git command failed"):
        freeze_prelabel_chronology(
            tmp_path / "receipt.json",
            root=tmp_path,
            model_freeze_commit="HEAD",
            input_evidence_commit="HEAD",
        )
