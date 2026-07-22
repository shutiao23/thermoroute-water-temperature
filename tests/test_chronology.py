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
    STAGE09_ARTIFACT_PATHS,
    STAGE09B_MEMBERS,
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


def _file_sha(root: Path, relative: str) -> str:
    return _sha((root / relative).read_bytes())


def _binding(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _file_sha(root, relative)}


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
    _write(root, metadata, _json_bytes({"status": 200}))
    _write(root, response, payload)
    _write(
        root,
        index_path,
        _json_bytes(
            {
                "schema_version": 1,
                "snapshot_count": 1,
                "records": [
                    {
                        "metadata_path": str(Path(metadata).relative_to(base)),
                        "response_path": str(Path(response).relative_to(base)),
                        "response_sha256": _sha(payload),
                    }
                ],
            }
        ),
    )


def _seed_model_commit(
    root: Path,
    *,
    original_commit: str,
    final_commit: str,
    leak_before_model: bool,
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
    for path in (
        "src/thermoroute/chronology.py",
        "src/thermoroute/outcome_qc.py",
        "scripts/28_freeze_prelabel_chronology.py",
        "tests/test_chronology.py",
        "protocols/route_a_outcome_qc_policy_v1.json",
    ):
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
        "format": "thermoroute.lightgbm-bundle.v1",
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
        relative = f"data_usgs/bridge/{name}/snapshot_index.json"
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

    stage9_run_id = "stage09-fixture"
    stage9_run_manifest = (
        f"outputs/runs/09_usgs_experiment/{stage9_run_id}/run.json"
    )
    _write(root, stage9_run_manifest, "{}\n")
    for label, relative in STAGE09_ARTIFACT_PATHS.items():
        _write(root, relative, f"stage09 {label}\n")
    stage9 = {
        "format": "thermoroute.stage09-completion-receipt.v1",
        "status": "PASS_FORMAL_STAGE09_COMPLETE",
        "stage": "09_usgs_experiment",
        "run_id": stage9_run_id,
        "run_identity": {"run_id": stage9_run_id},
        "formal_configuration": {"fixture": True},
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

    stage09b_run_id = "stage09b-fixture"
    stage09b_run_dir = (
        f"outputs/runs/09b_development_controls/{stage09b_run_id}"
    )
    _write(root, f"{stage09b_run_dir}/run.json", "{}\n")
    members = []
    semantic_members = []
    for arm_id, seed in STAGE09B_MEMBERS:
        prediction_path = (
            f"{stage09b_run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
        )
        prediction_sidecar = f"{prediction_path}.meta.json"
        _write(root, prediction_path, f"{arm_id}/seed{seed}".encode())
        _write(root, prediction_sidecar, "{}\n")
        members.append({
            "arm_id": arm_id,
            "seed": seed,
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
            "normalised_prediction_sha256": _sha(
                f"normalised:{arm_id}:{seed}".encode()
            ),
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
        "expected_members": 31,
        "prediction_rows": 93,
        "common_forecast_keys": 3,
        "splits": ["calib", "test", "val"],
        "reference_member": "PlainMLP-7var/seed0",
    }
    descriptor = lambda relative: {  # noqa: E731 - compact fixture helper
        "sha256": _file_sha(root, relative),
        "bytes": (root / relative).stat().st_size,
    }
    semantic = {
        "format": "thermoroute.development-controls-semantic-audit.v1",
        "status": "PASS_PREDICTION_ARTIFACT_CLOSURE",
        "run_id": stage09b_run_id,
        "evidence_scope": "prediction_artifact_closure",
        "training_replay_verified": False,
        "post_2020_outcomes_requested_or_read": False,
        "matrix_audit": matrix_audit,
        "canonical_window_registry": {
            "sha256": "c" * 64,
            "common_forecast_keys": 3,
            "train_examples_per_epoch": 3,
            "train_registry_sha256": "d" * 64,
        },
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
        "format": "thermoroute.stage09b-completion-receipt.v2",
        "status": "PASS_STAGE09B_PREDICTION_ARTIFACT_CLOSURE",
        "stage": "09b_development_controls",
        "run_id": stage09b_run_id,
        "run_identity": {"run_id": stage09b_run_id},
        "formal_configuration": {"fixture": True},
        "evidence_scope": "prediction_artifact_closure",
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

    # This is computed only after every source/protocol/control fixture byte is
    # present.  The frozen suite and replay must agree with the exact source
    # inventory committed alongside the model artifacts.
    frozen_source_sha256 = source_tree_hash(root)

    suite = {
        "format": "thermoroute.route-a-model-suite.v1",
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "development_contract": {
            "frozen_panel_spec": _binding(root, "data_usgs/frozen_panel_v1.json"),
            "panel": _binding(root, "data_usgs/panel_usgs_120v2.parquet"),
            "registry": _binding(root, "data_usgs/station_registry_v1.csv"),
            "predictor_bridge": _binding(root, bridge_path),
            "source_sha256": frozen_source_sha256,
        },
        "preopening_gates": {
            "stage09_completion": _binding(root, stage9_path),
            "stage09b_development_controls": _binding(root, stage09b_path),
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
    model = _seed_model_commit(
        root,
        original_commit=original,
        final_commit=final,
        leak_before_model=leak_before_model,
    )
    evidence = _seed_evidence_commit(
        root, candidate_already_exists=leak_before_model
    )
    _write(root, "notes/chronology-marker.txt", "evidence committed\n")
    marker = _commit(root, "mark evidence ready for chronology receipt")
    if source_change:
        _write(root, "src/thermoroute/post_freeze_change.py", "changed = True\n")
        marker = _commit(root, "forbidden post-freeze source change")
    return {
        "root": root,
        "original": original,
        "final": final,
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
    assert len(document["model_freeze_artifacts"]) >= 10
    assert len(document["input_evidence_artifacts"]) >= 15
    assert document["external_timestamp_or_public_preregistration"] is False
    assert validate_prelabel_chronology(
        state["receipt"], root=state["root"]
    ) == document


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
