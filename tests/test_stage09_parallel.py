from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile
import threading
import time
from typing import Any, Mapping

import pytest

import thermoroute.stage09_parallel as STAGE09_PARALLEL

from thermoroute.model_matrix_amendment import (
    AMENDMENT_FORMAT,
    AMENDMENT_ID,
    AMENDMENT_RELATIVE,
    AMENDMENT_SEAL_FORMAT,
    AMENDMENT_SEAL_RELATIVE,
    AMENDMENT_SEAL_STATUS,
    AMENDMENT_STATUS,
    STAGE09_CONTROLS,
    model_matrix_contract_id,
)
from thermoroute.repro import (
    RUN_SCHEMA_VERSION,
    RunIdentity,
    configure_deterministic_runtime,
    numerical_runtime_contract,
    sha256_json,
    source_tree_hash,
)
from thermoroute.stage09_parallel import (
    CONTROL_MEMBER_COUNT,
    COORDINATOR_RECEIPT_FORMAT,
    ControlMember,
    DEFAULT_CONTROL_WORKERS,
    MAX_CONTROL_WORKERS,
    ModelMatrixGate,
    RECOMMENDED_MAX_CONTROL_WORKERS,
    SEMANTIC_VALIDATION_FORMAT,
    Stage09ParallelError,
    ValidatedWorkOrder,
    execute_work_orders_bounded,
    expected_control_members,
    freeze_control_matrix_receipt,
    freeze_control_plan,
    materialize_member_resume_checkpoint,
    publish_control_member,
    validate_member_archive,
    validate_work_order_structure,
)


ROOT = Path(__file__).resolve().parents[1]
INTERVENTIONS: dict[str, dict[str, Any]] = {
    "DampedPriorOnly": {"use_prior": False, "residual_model": False},
    "TR-noDynamicPrior": {"use_prior": False},
    "TR-fixedKappa": {"fixed_kappa": True},
    "TR-noRouter": {"use_router": False},
    "TR-noMoE": {"use_moe": False},
    "TR-noTCN": {"use_tcn": False},
    "TR-unbounded": {"delta_scale": None},
}


class _UnchangedClosure:
    @staticmethod
    def assert_unchanged() -> None:
        return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _repository_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _binding(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _formal_config() -> dict[str, Any]:
    train_config = {
        "d_model": 40,
        "encoder_blocks": 2,
        "kernel_size": 3,
        "dropout": 0.15,
        "n_experts": 3,
        "station_embed_dim": 8,
        "lr": 0.002,
        "weight_decay": 0.0001,
        "batch_size": 1536,
        "max_epochs": 80,
        "patience": 12,
        "grad_clip": 1.0,
        "lambda_event": 0.3,
        "lambda_residual": 0.01,
        "lambda_crossing": 1.0,
        "temperature_loss_scale": 1.0,
    }
    return {
        "stage": "09_usgs_experiment",
        "protocol": "route_a_strict_v1_balanced_delta1",
        "panel": "panel_usgs_120v2.parquet",
        "station_registry": "station_registry_v1.csv",
        "variables": ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"],
        "horizons": [1, 3, 7],
        "context_length": 32,
        "seeds": 5,
        "time_split": {},
        "train_config": train_config,
        "thermoroute_seeds": [0, 1, 2, 3, 4],
        "lightgbm_seeds": [0, 1, 2, 3, 4],
        "ablation_seeds": [0, 1, 2, 3, 4],
        "delta_scale": 1.0,
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "ablations": True,
        "air2stream": False,
        "device": "cpu",
        "training_device": "cpu",
        "execution_role": "route_a_formal_candidate",
        "development_predictor_bridge": {"sha256": "b" * 64},
        "eval_batch_size": 4096,
        "lightgbm_validation_grid": [],
        "event_reference_fit_interval": ["2006-01-01", "2018-12-31"],
        "formal_numerical_policy": {"threads": 1},
        "input_closure_sha256": "a" * 64,
        "input_closure_file_count": 1,
    }


def _fixture_repository(tmp_path: Path) -> tuple[
    Path,
    RunIdentity,
    ModelMatrixGate,
    Path,
    tuple[Path, ...],
]:
    for relative in (AMENDMENT_RELATIVE, AMENDMENT_SEAL_RELATIVE):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    amendment = json.loads((tmp_path / AMENDMENT_RELATIVE).read_text())
    seal = json.loads((tmp_path / AMENDMENT_SEAL_RELATIVE).read_text())
    contract_id = model_matrix_contract_id(
        amendment["stage09_architecture_control_matrix"],
        amendment["stage09b_development_control_matrix"],
    )
    gate = ModelMatrixGate(
        amendment=amendment,
        seal=seal,
        binding={
            "format": "thermoroute.stage09-control-model-matrix-binding.v1",
            "amendment": _binding(tmp_path, AMENDMENT_RELATIVE),
            "seal": _binding(tmp_path, AMENDMENT_SEAL_RELATIVE),
            "amendment_format": AMENDMENT_FORMAT,
            "amendment_status": AMENDMENT_STATUS,
            "seal_format": AMENDMENT_SEAL_FORMAT,
            "seal_status": AMENDMENT_SEAL_STATUS,
            "amendment_id": AMENDMENT_ID,
            "amendment_document_commit": seal["amendment_document_commit"],
            "model_matrix_contract_id": contract_id,
        },
        contract_id=contract_id,
    )
    configure_deterministic_runtime()
    config = _formal_config()
    identity_parts = {
        "panel_sha256": "1" * 64,
        "registry_sha256": "2" * 64,
        "config_sha256": sha256_json(config),
        "source_sha256": source_tree_hash(tmp_path),
        "runtime_sha256": sha256_json(numerical_runtime_contract()),
        "input_closure_sha256": "a" * 64,
        "schema_version": RUN_SCHEMA_VERSION,
    }
    identity = RunIdentity(
        run_id=sha256_json(identity_parts)[:20],
        **identity_parts,
    )
    run_dir = (
        tmp_path
        / "outputs"
        / "runs"
        / "09_usgs_experiment"
        / identity.run_id
    )
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        _repository_json({"identity": identity.as_dict()}), encoding="utf-8"
    )
    authorization, work_orders = freeze_control_plan(
        root=tmp_path,
        run_directory=run_dir,
        identity=identity,
        resolved_config=config,
        matrix_gate=gate,
        interventions=INTERVENTIONS,
        publication_guard=lambda: None,
    )
    return tmp_path, identity, gate, authorization, work_orders


def _validated(
    root: Path,
    identity: RunIdentity,
    gate: ModelMatrixGate,
    work_order: Path,
) -> ValidatedWorkOrder:
    document, authorization, observed = validate_work_order_structure(
        root=root, work_order_path=work_order
    )
    assert observed == identity
    return ValidatedWorkOrder(
        root=root,
        path=work_order,
        document=document,
        authorization=authorization,
        identity=identity,
        input_closure=_UnchangedClosure(),
        matrix_gate=gate,
    )


def _stub_producer(member: ControlMember):
    def produce(payload: Path) -> dict[str, Any]:
        value = f"{member.arm_id}|{member.seed}\n".encode()
        (payload / "predictions.parquet").write_bytes(value)
        (payload / "predictions.parquet.meta.json").write_bytes(
            hashlib.sha256(value).hexdigest().encode()
        )
        bundle = payload / "bundle"
        bundle.mkdir()
        (bundle / "metadata.json").write_bytes(value[::-1])
        (bundle / "weights.pt").write_bytes(value * 2)
        (payload / "training_checkpoint.pt").write_bytes(value * 3)
        (payload / "training_checkpoint.pt.meta.json").write_bytes(value * 4)
        return {
            "best_epoch_index": 2,
            "best_selection_metric_value": float(
                member.arm_index + member.seed / 10
            ),
        }

    return produce


def _stub_semantic_validator(
    payload: Path, validated: ValidatedWorkOrder
) -> dict[str, Any]:
    """Model a source-bound semantic loader for transaction unit tests."""
    member = validated.member
    value = f"{member.arm_id}|{member.seed}\n".encode()
    expected = {
        "predictions.parquet": value,
        "predictions.parquet.meta.json": hashlib.sha256(value).hexdigest().encode(),
        "bundle/metadata.json": value[::-1],
        "bundle/weights.pt": value * 2,
        "training_checkpoint.pt": value * 3,
        "training_checkpoint.pt.meta.json": value * 4,
    }
    for relative, content in expected.items():
        if (payload / relative).read_bytes() != content:
            raise Stage09ParallelError(
                f"test semantic payload changed: {relative}"
            )
    def digest(relative: str) -> str:
        return _sha(payload / relative)
    metric = float(member.arm_index + member.seed / 10)
    reference = (
        validated.run_directory
        / "predictions"
        / f"thermoroute_seed{member.seed}.parquet"
    )
    reference_sidecar = reference.with_name(reference.name + ".meta.json")
    reference_sha256 = _sha(reference) if reference.is_file() else "a" * 64
    reference_sidecar_sha256 = (
        _sha(reference_sidecar) if reference_sidecar.is_file() else "b" * 64
    )
    return {
        "format": SEMANTIC_VALIDATION_FORMAT,
        "model_id": member.arm_id,
        "seed": member.seed,
        "scope": "ablation_usgs",
        "feature_set": "USGS",
        "prediction_schema": "thermoroute.predictions.v1",
        "prediction_artifact_sha256": digest("predictions.parquet"),
        "prediction_sidecar_sha256": digest(
            "predictions.parquet.meta.json"
        ),
        "bundle_metadata_sha256": digest("bundle/metadata.json"),
        "bundle_weights_sha256": digest("bundle/weights.pt"),
        "bundle_member_id": f"seed{member.seed}",
        "checkpoint_payload_sha256": digest("training_checkpoint.pt"),
        "checkpoint_sidecar_sha256": digest(
            "training_checkpoint.pt.meta.json"
        ),
        "checkpoint_best_epoch_index": 2,
        "checkpoint_best_selection_metric_value": metric,
        "same_seed_reference_prediction_sha256": reference_sha256,
        "same_seed_reference_sidecar_sha256": reference_sidecar_sha256,
        "forecast_record_count": 1,
        "forecast_key_sha256": "c" * 64,
        "truth_float32_sha256": "d" * 64,
        "prediction_sidecar_run_identity_exact": True,
        "bundle_metadata_identity_exact": True,
        "bundle_member_registry_exact": True,
        "bundle_intervention_exact": True,
        "bundle_weights_safely_loaded": True,
        "checkpoint_weights_safely_loaded": True,
        "bundle_matches_checkpoint_best_state_exact": True,
        "same_seed_forecast_keys_exact": True,
        "same_seed_split_exact": True,
        "same_seed_y_true_float32_exact": True,
        "full_prediction_replay_equivalence_proved": True,
        "full_prediction_replay_max_abs_difference": 0.0,
    }


def _rewrite_archive_receipt(
    archive_path: Path,
    transform: Any,
) -> None:
    with tarfile.open(archive_path, mode="r:") as archive:
        payloads = {
            item.name: archive.extractfile(item).read()
            for item in archive.getmembers()
        }
    payloads["receipt.json"] = transform(payloads["receipt.json"])
    candidate = archive_path.with_name("tampered-candidate.member.tar")
    with tarfile.open(candidate, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name, content in sorted(payloads.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    candidate.replace(archive_path)


def _publish_all(
    root: Path,
    identity: RunIdentity,
    gate: ModelMatrixGate,
    work_orders: tuple[Path, ...],
) -> None:
    for member, work_order in zip(
        expected_control_members(), work_orders, strict=True
    ):
        publish_control_member(
            validated=_validated(root, identity, gate, work_order),
            producer=_stub_producer(member),
            semantic_validator=_stub_semantic_validator,
        )


def _thermoroute_references(
    root: Path, identity: RunIdentity
) -> dict[int, dict[str, Path]]:
    run_dir = (
        root
        / "outputs"
        / "runs"
        / "09_usgs_experiment"
        / identity.run_id
    )
    result: dict[int, dict[str, Path]] = {}
    for seed in range(5):
        prediction = run_dir / "predictions" / f"thermoroute_seed{seed}.parquet"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_bytes(f"ThermoRoute seed{seed}\n".encode())
        sidecar = prediction.with_name(prediction.name + ".meta.json")
        sidecar.write_text(
            _repository_json(
                {
                    "run": identity.as_dict(),
                    "artifact": prediction.name,
                    "artifact_sha256": _sha(prediction),
                    "artifact_bytes": prediction.stat().st_size,
                    "kind": "thermoroute_seed_predictions",
                }
            ),
            encoding="utf-8",
        )
        bundle = run_dir / "member_bundles" / f"seed{seed}"
        bundle.mkdir(parents=True)
        weights = bundle / "weights.pt"
        weights.write_bytes(f"weights seed{seed}\n".encode())
        (bundle / "metadata.json").write_text(
            _repository_json(
                {
                    "run_id": identity.run_id,
                    "source_sha256": identity.source_sha256,
                    "panel_sha256": identity.panel_sha256,
                    "registry_sha256": identity.registry_sha256,
                    "runtime_sha256": identity.runtime_sha256,
                    "input_closure_sha256": identity.input_closure_sha256,
                    "members": [f"seed{seed}"],
                    "member_count": 1,
                    "weights_sha256": _sha(weights),
                }
            ),
            encoding="utf-8",
        )
        result[seed] = {"prediction": prediction, "bundle": bundle}
    return result


def _load_stage09_script_module():
    import importlib.util

    path = ROOT / "scripts" / "09_usgs_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "stage09_semantic_validation_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_semantic_payload(
    root: Path,
    identity: RunIdentity,
    gate: ModelMatrixGate,
    work_order: Path,
) -> tuple[ValidatedWorkOrder, Path]:
    import numpy as np
    import pandas as pd
    import torch

    from thermoroute import config as C
    from thermoroute import results as R
    from thermoroute.checkpoint import (
        CHECKPOINT_METADATA_VERSION,
        CHECKPOINT_VERSION,
        neural_output_head_schema,
        save_inference_bundle,
    )
    from thermoroute.repro import seal_artifact

    validated = _validated(root, identity, gate, work_order)
    member = validated.member
    payload = (
        validated.run_directory
        / ".stage09-control-member-staging"
        / "real-semantic-payload"
    )
    payload.mkdir()
    issue = pd.to_datetime(["2016-01-01"] * 3)
    horizon = np.asarray([1, 3, 7], dtype=np.int64)
    target = issue + pd.to_timedelta(horizon, unit="D")
    truth = np.asarray([1.25, 2.5, 3.75], dtype=np.float64)

    def frame(model: str, scope: str, prediction: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": [model] * 3,
                "scope": [scope] * 3,
                "feature_set": ["USGS"] * 3,
                "seed": np.asarray([member.seed] * 3, dtype=np.int64),
                "site_id": ["ordinary-monitoring-site"] * 3,
                "horizon": horizon,
                "split": ["val"] * 3,
                "issue_date": issue,
                "target_date": target,
                "y_true": truth,
                "y_pred": np.asarray([prediction] * 3, dtype=np.float64),
                "q05": np.asarray([prediction - 1.0] * 3),
                "q50": np.asarray([prediction] * 3),
                "q95": np.asarray([prediction + 1.0] * 3),
                "p_exceed": np.asarray([0.5] * 3),
            }
        )[R.PRED_COLS]

    prediction = payload / "predictions.parquet"
    R.write_predictions(frame(member.arm_id, "ablation_usgs", 2.0), prediction)
    seal_artifact(
        prediction,
        identity,
        kind="thermoroute_ablation_seed_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    reference = (
        validated.run_directory
        / "predictions"
        / f"thermoroute_seed{member.seed}.parquet"
    )
    reference.parent.mkdir()
    R.write_predictions(frame("ThermoRoute", "joint_usgs", 2.25), reference)
    seal_artifact(
        reference,
        identity,
        kind="thermoroute_seed_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )

    scientific = validated.document["scientific_member_config"]
    assert isinstance(scientific, Mapping)
    intervention = scientific["model_kwargs_intervention"]
    assert isinstance(intervention, Mapping)
    kwargs: dict[str, Any] = {
        "n_vars": 7,
        "n_stations": len(C.STATIONS),
        "n_phys": 4,
        "station_agnostic": False,
        "use_prior": True,
        "use_router": True,
        "use_moe": True,
        "sparse_router": True,
        "fixed_kappa": False,
        "delta_scale": 1.0,
        "use_tcn": True,
        "residual_model": True,
        "safety_anchor": "damped",
        "use_wlevel": False,
    }
    kwargs.update(intervention)
    bundle_metadata = {
        "run_id": identity.run_id,
        "architecture": {
            "class": "thermoroute.thermoroute.ThermoRoute",
            "kwargs": kwargs,
            "train_config": scientific["train_config"],
        },
        "feature_order": ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"],
        "horizons": [1, 3, 7],
        "station_to_index": {
            station: index for index, station in enumerate(C.STATIONS)
        },
        "preprocessing": {},
        "event_thresholds": {},
        "event_calibrators": {},
        "conformal_offsets": {},
        "source_sha256": identity.source_sha256,
        "panel_sha256": identity.panel_sha256,
        "registry_sha256": identity.registry_sha256,
        "config_sha256": identity.config_sha256,
        "runtime_sha256": identity.runtime_sha256,
        "input_closure_sha256": identity.input_closure_sha256,
        "training_device": "cpu",
        "output_head_schema": neural_output_head_schema(),
    }
    save_inference_bundle(
        payload / "bundle",
        members={f"seed{member.seed}": {"weight": torch.ones(1)}},
        metadata=bundle_metadata,
        expected_member_count=1,
    )

    resolved = validated.authorization["resolved_config"]
    assert isinstance(resolved, Mapping)
    model_kwargs = dict(intervention)
    model_kwargs.setdefault("delta_scale", resolved["delta_scale"])
    checkpoint_config = {
        **dict(resolved),
        "arm": member.arm_id,
        "seed": member.seed,
        "model_kwargs": model_kwargs,
    }
    resolved_json = json.dumps(
        checkpoint_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    extra_json = "{}"
    checkpoint_value = {
        "format": CHECKPOINT_VERSION,
        "run_id": identity.run_id,
        "resolved_config_json": resolved_json,
        "resolved_config_sha256": hashlib.sha256(
            resolved_json.encode()
        ).hexdigest(),
        "extra_json": extra_json,
        "extra_sha256": hashlib.sha256(extra_json.encode()).hexdigest(),
        "epoch": 2,
        "best_epoch": 2,
        "best_metric": 0.125,
        "model_class": "thermoroute.thermoroute.ThermoRoute",
        "optimizer_class": "torch.optim.adamw.AdamW",
        "scheduler_class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "model_state": {"weight": torch.ones(1)},
        "best_model_state": {"weight": torch.ones(1)},
        "optimizer_state": {},
        "scheduler_present": True,
        "scheduler_state": {},
        "rng_state": {},
    }
    checkpoint = payload / "training_checkpoint.pt"
    torch.save(checkpoint_value, checkpoint)
    checkpoint_sidecar = {
        "format": CHECKPOINT_METADATA_VERSION,
        "checkpoint_format": CHECKPOINT_VERSION,
        "run_id": identity.run_id,
        "epoch": 2,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": _sha(checkpoint),
        "resolved_config_sha256": checkpoint_value["resolved_config_sha256"],
        "extra_sha256": checkpoint_value["extra_sha256"],
        "model_class": checkpoint_value["model_class"],
        "optimizer_class": checkpoint_value["optimizer_class"],
        "scheduler_class": checkpoint_value["scheduler_class"],
        "scheduler_present": True,
    }
    (payload / "training_checkpoint.pt.meta.json").write_text(
        _repository_json(checkpoint_sidecar), encoding="utf-8"
    )
    return validated, payload


def test_control_plan_is_exact_7x5_and_worker_width_is_not_scientific_identity(
    tmp_path: Path,
) -> None:
    root, _identity, _gate, authorization, work_orders = _fixture_repository(
        tmp_path
    )
    document = json.loads(authorization.read_text())
    assert len(work_orders) == CONTROL_MEMBER_COUNT == 35
    assert [entry["member_id"] for entry in document["matrix"]] == [
        member.member_id for member in expected_control_members()
    ]
    assert document["matrix_audit"] == {
        "arm_count": 7,
        "seed_count_per_arm": 5,
        "expected_member_count": 35,
        "arm_major_seed_minor_order": True,
        "same_seed_pairing_with_thermoroute": True,
        "missing_duplicate_or_extra_member_allowed": False,
    }
    assert "control_workers" not in document
    assert "max_workers" not in document
    assert document["run_identity"]["config_sha256"] == sha256_json(
        document["resolved_config"]
    )
    assert all(path.is_relative_to(root) for path in work_orders)
    run_directory = authorization.parent.parent
    assert (run_directory / ".stage09-control-member-locks").is_dir()
    assert (run_directory / ".stage09-control-member-staging").is_dir()
    assert DEFAULT_CONTROL_WORKERS == RECOMMENDED_MAX_CONTROL_WORKERS == 2
    assert MAX_CONTROL_WORKERS >= RECOMMENDED_MAX_CONTROL_WORKERS


def test_production_semantic_loader_rejects_text_poison_and_truth_mismatch(
    tmp_path: Path,
) -> None:
    import pandas as pd

    from thermoroute import results as R
    from thermoroute.repro import seal_artifact

    script = _load_stage09_script_module()
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        clean_root
    )
    validated, payload = _real_semantic_payload(
        root, identity, gate, work_orders[0]
    )
    semantic = script._validate_stage09_control_payload_semantics(
        payload,
        validated,
        prediction_replay=lambda _bundle, _control, _validated: 0.0,
    )
    assert semantic["same_seed_forecast_keys_exact"] is True
    assert semantic["same_seed_y_true_float32_exact"] is True
    assert semantic["bundle_weights_safely_loaded"] is True
    assert semantic["checkpoint_weights_safely_loaded"] is True
    assert semantic["full_prediction_replay_equivalence_proved"] is True
    assert semantic["full_prediction_replay_max_abs_difference"] == 0.0

    for case_index, poisoned_relative in enumerate(
        ("predictions.parquet", "bundle/metadata.json", "bundle/weights.pt")
    ):
        case_root = tmp_path / f"text-poison-{case_index}"
        case_root.mkdir()
        root, identity, gate, _authorization, work_orders = (
            _fixture_repository(case_root)
        )
        validated, payload = _real_semantic_payload(
            root, identity, gate, work_orders[0]
        )
        (payload / poisoned_relative).write_text(
            "arbitrary attacker-controlled text", encoding="utf-8"
        )
        with pytest.raises(Stage09ParallelError):
            script._validate_stage09_control_payload_semantics(
                payload,
                validated,
                prediction_replay=lambda _bundle, _control, _validated: 0.0,
            )

    truth_root = tmp_path / "truth-poison"
    truth_root.mkdir()
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        truth_root
    )
    validated, payload = _real_semantic_payload(
        root, identity, gate, work_orders[0]
    )
    reference = (
        validated.run_directory
        / "predictions"
        / f"thermoroute_seed{validated.member.seed}.parquet"
    )
    changed = pd.read_parquet(reference)
    changed.loc[0, "y_true"] += 1.0
    R.write_predictions(changed, reference)
    seal_artifact(
        reference,
        identity,
        kind="thermoroute_seed_predictions",
        schema=R.PREDICTION_SCHEMA_VERSION,
    )
    with pytest.raises(Stage09ParallelError, match="keys/truth"):
        script._validate_stage09_control_payload_semantics(
            payload,
            validated,
            prediction_replay=lambda _bundle, _control, _validated: 0.0,
        )


def test_production_semantic_loader_rejects_finite_wrong_bundle_state(
    tmp_path: Path,
) -> None:
    import torch

    script = _load_stage09_script_module()
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    validated, payload = _real_semantic_payload(
        root, identity, gate, work_orders[0]
    )
    weights_path = payload / "bundle" / "weights.pt"
    metadata_path = payload / "bundle" / "metadata.json"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    member_name = f"seed{validated.member.seed}"
    weights[member_name]["weight"] = torch.full((1,), 2.0)
    torch.save(weights, weights_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["weights_sha256"] = _sha(weights_path)
    metadata_path.write_text(_repository_json(metadata), encoding="utf-8")

    with pytest.raises(Stage09ParallelError, match="checkpoint best state"):
        script._validate_stage09_control_payload_semantics(
            payload,
            validated,
            prediction_replay=lambda _bundle, _control, _validated: 0.0,
        )


def test_fixed_resume_checkpoint_survives_interrupt_and_cleans_after_commit(
    tmp_path: Path,
) -> None:
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    member = expected_control_members()[0]
    validated = _validated(root, identity, gate, work_orders[0])
    checkpoint = validated.resume_checkpoint_path
    sidecar = checkpoint.with_name(checkpoint.name + ".meta.json")
    value = f"{member.arm_id}|{member.seed}\n".encode()

    def interrupted(_payload: Path) -> dict[str, Any]:
        checkpoint.write_bytes(value * 3)
        sidecar.write_bytes(value * 4)
        checkpoint.chmod(0o600)
        sidecar.chmod(0o600)
        raise KeyboardInterrupt("injected ordinary interruption")

    with pytest.raises(KeyboardInterrupt, match="ordinary interruption"):
        publish_control_member(
            validated=validated,
            producer=interrupted,
            semantic_validator=_stub_semantic_validator,
        )
    assert checkpoint.read_bytes() == value * 3
    assert sidecar.read_bytes() == value * 4
    assert checkpoint.parent.name == validated.document["work_order_self_sha256"]

    def resumed(payload: Path) -> dict[str, Any]:
        assert checkpoint.read_bytes() == value * 3
        result = _stub_producer(member)(payload)
        (payload / "training_checkpoint.pt").chmod(0o600)
        (payload / "training_checkpoint.pt.meta.json").chmod(0o600)
        materialize_member_resume_checkpoint(
            validated=validated,
            payload=payload,
        )
        return result

    archive, reused = publish_control_member(
        validated=validated,
        producer=resumed,
        semantic_validator=_stub_semantic_validator,
    )
    assert archive.is_file()
    assert reused is False
    assert not checkpoint.parent.exists()


def test_fixed_member_checkpoint_resumes_bitwise_and_rejects_wrong_config(
    tmp_path: Path,
) -> None:
    from dataclasses import replace
    import importlib.util
    import torch

    from thermoroute import config as C
    from thermoroute.train import fit_model

    fixture_spec = importlib.util.spec_from_file_location(
        "stage09_training_resume_fixture",
        ROOT / "tests" / "test_training_resume.py",
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)

    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    validated = _validated(root, identity, gate, work_orders[0])
    checkpoint = validated.resume_checkpoint_path
    cfg = replace(
        C.TRAIN,
        batch_size=3,
        max_epochs=5,
        patience=20,
        dropout=0.0,
    )
    thresholds = {"s0": 0.0, "s1": 0.0}
    resume_config = {"fixture": "stage09-member-resume", "epochs": 5}
    previous_stations = C.STATIONS
    C.STATIONS = ("s0", "s1")
    try:
        uninterrupted = fit_model(
            fixture._TinyForecaster,
            fixture._TinyWindows(),
            thresholds,
            cfg=cfg,
            seed=13,
            eval_batch_size=2,
            model_name="fixture",
        )

        def stop_after_two_epochs(_payload: Path) -> dict[str, Any]:
            fit_model(
                fixture._TinyForecaster,
                fixture._TinyWindows(),
                thresholds,
                cfg=cfg,
                seed=13,
                eval_batch_size=2,
                model_name="fixture",
                checkpoint_path=checkpoint,
                run_id=identity.run_id,
                resolved_config=resume_config,
                stop_after_epoch=1,
            )
            raise RuntimeError("injected epoch interruption")

        with pytest.raises(RuntimeError, match="epoch interruption"):
            publish_control_member(
                validated=validated,
                producer=stop_after_two_epochs,
                semantic_validator=_stub_semantic_validator,
            )
        assert checkpoint.is_file()
        assert checkpoint.with_name(checkpoint.name + ".meta.json").is_file()

        def wrong_config(_payload: Path) -> dict[str, Any]:
            fit_model(
                fixture._TinyForecaster,
                fixture._TinyWindows(),
                thresholds,
                cfg=cfg,
                seed=13,
                eval_batch_size=2,
                model_name="fixture",
                checkpoint_path=checkpoint,
                run_id=identity.run_id,
                resolved_config={**resume_config, "epochs": 6},
            )
            raise AssertionError("wrong-config checkpoint was accepted")

        with pytest.raises(ValueError, match="config|resolved"):
            publish_control_member(
                validated=validated,
                producer=wrong_config,
                semantic_validator=_stub_semantic_validator,
            )

        resumed_result: dict[str, Any] = {}

        def resume_to_completion(_payload: Path) -> dict[str, Any]:
            resumed_result["value"] = fit_model(
                fixture._TinyForecaster,
                fixture._TinyWindows(),
                thresholds,
                cfg=cfg,
                seed=13,
                eval_batch_size=2,
                model_name="fixture",
                checkpoint_path=checkpoint,
                run_id=identity.run_id,
                resolved_config=resume_config,
            )
            raise RuntimeError("resume equivalence captured")

        with pytest.raises(RuntimeError, match="equivalence captured"):
            publish_control_member(
                validated=validated,
                producer=resume_to_completion,
                semantic_validator=_stub_semantic_validator,
            )
        resumed = resumed_result["value"]
        assert uninterrupted.best_val == resumed.best_val
        for name, value in uninterrupted.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[name]), name
        sort_key = ["split", "site_id", "horizon", "issue_date"]
        assert uninterrupted.pred.sort_values(sort_key).reset_index(
            drop=True
        ).equals(resumed.pred.sort_values(sort_key).reset_index(drop=True))
    finally:
        C.STATIONS = previous_stations


def test_safe_directory_accepts_concurrent_first_create_only_after_revalidation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "concurrent-private-directory"
    barrier = threading.Barrier(16)

    def create(_index: int) -> Path:
        barrier.wait()
        return STAGE09_PARALLEL._safe_directory(target, parent=tmp_path)

    with ThreadPoolExecutor(max_workers=16) as executor:
        observed = tuple(executor.map(create, range(16)))
    assert observed == (target,) * 16
    assert target.is_dir() and not target.is_symlink()
    assert target.stat().st_mode & 0o077 == 0


def test_create_only_file_is_complete_and_idempotent_under_concurrent_start(
    tmp_path: Path,
) -> None:
    target = tmp_path / "concurrent-authority.json"
    content = (b'{"complete":true,"payload":"' + b"x" * 100_000 + b'"}\n')
    barrier = threading.Barrier(16)

    def publish(_index: int) -> None:
        barrier.wait()
        STAGE09_PARALLEL._write_create_only(target, content)

    with ThreadPoolExecutor(max_workers=16) as executor:
        tuple(executor.map(publish, range(16)))
    assert target.read_bytes() == content
    assert target.stat().st_nlink == 1
    assert target.stat().st_mode & 0o077 == 0


def test_atomic_publication_fault_before_rename_leaves_no_public_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    member = expected_control_members()[0]
    validated = _validated(root, identity, gate, work_orders[0])
    archive = (
        validated.run_directory
        / "stage09_control_precompute_v1"
        / member.archive_relative
    )

    def crash_before_commit(_source: Path, _destination: Path) -> None:
        raise RuntimeError("injected pre-rename crash")

    monkeypatch.setattr(
        STAGE09_PARALLEL,
        "_atomic_exclusive_rename",
        crash_before_commit,
    )
    with pytest.raises(RuntimeError, match="injected pre-rename crash"):
        publish_control_member(
            validated=validated,
            producer=_stub_producer(member),
            semantic_validator=_stub_semantic_validator,
        )
    assert not archive.exists()
    staging_root = validated.run_directory / ".stage09-control-member-staging"
    assert tuple(staging_root.iterdir()) == ()


def test_external_hard_link_at_final_name_is_rejected_without_training(
    tmp_path: Path,
) -> None:
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    member = expected_control_members()[0]
    validated = _validated(root, identity, gate, work_orders[0])
    archive = (
        validated.run_directory
        / "stage09_control_precompute_v1"
        / member.archive_relative
    )
    external = root / "external-untrusted.member.tar"
    external.write_bytes(b"untrusted")
    archive.hardlink_to(external)
    calls = 0

    def forbidden(_payload: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "best_epoch_index": 0,
            "best_selection_metric_value": 0.0,
        }

    with pytest.raises(Stage09ParallelError, match="unsafe|collision"):
        publish_control_member(
            validated=validated,
            producer=forbidden,
            semantic_validator=_stub_semantic_validator,
        )
    assert calls == 0


@pytest.mark.parametrize(
    "poisoned_relative",
    ("predictions.parquet", "bundle/metadata.json", "bundle/weights.pt"),
)
def test_semantic_poison_cannot_reach_archive_commit(
    tmp_path: Path,
    poisoned_relative: str,
) -> None:
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    member = expected_control_members()[0]
    validated = _validated(root, identity, gate, work_orders[0])
    archive = (
        validated.run_directory
        / "stage09_control_precompute_v1"
        / member.archive_relative
    )

    def poisoned(payload: Path) -> dict[str, Any]:
        result = _stub_producer(member)(payload)
        (payload / poisoned_relative).write_text(
            "arbitrary attacker-controlled text", encoding="utf-8"
        )
        return result

    with pytest.raises(Stage09ParallelError, match="semantic payload changed"):
        publish_control_member(
            validated=validated,
            producer=poisoned,
            semantic_validator=_stub_semantic_validator,
        )
    assert not archive.exists()


def test_valid_member_is_create_only_and_exact_retry_reuses_without_training(
    tmp_path: Path,
) -> None:
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    member = expected_control_members()[0]
    validated = _validated(root, identity, gate, work_orders[0])
    archive, reused = publish_control_member(
        validated=validated,
        producer=_stub_producer(member),
        semantic_validator=_stub_semantic_validator,
    )
    assert reused is False
    before = archive.read_bytes()
    calls = 0

    def forbidden(_payload: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("verified retry must not retrain")

    same, reused = publish_control_member(
        validated=validated,
        producer=forbidden,
        semantic_validator=_stub_semantic_validator,
    )
    assert same == archive
    assert reused is True
    assert calls == 0
    assert archive.read_bytes() == before
    receipt = validate_member_archive(
        root=root,
        work_order_path=work_orders[0],
    )
    assert receipt["member"] == member.as_dict()
    assert receipt["run_identity"] == identity.as_dict()
    assert receipt["result_metadata"] == {
        "best_epoch_index": 2,
        "best_selection_metric_value": 0.0,
    }
    assert receipt["semantic_validation"]["checkpoint_best_epoch_index"] == 2
    assert receipt["execution_attestation"][
        "worker_count_recorded_in_scientific_identity"
    ] is False


@pytest.mark.parametrize(
    "collision",
    [
        "partial_file",
        "directory",
        "tampered_tar",
        "trailing_tar_bytes",
        "hardlink_alias",
    ],
)
def test_partial_collision_or_tamper_fails_closed_without_retraining(
    tmp_path: Path,
    collision: str,
) -> None:
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    member = expected_control_members()[0]
    validated = _validated(root, identity, gate, work_orders[0])
    archive = (
        validated.run_directory
        / "stage09_control_precompute_v1"
        / member.archive_relative
    )
    if collision == "directory":
        archive.mkdir(parents=True)
    elif collision == "partial_file":
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"partial")
    else:
        publish_control_member(
            validated=validated,
            producer=_stub_producer(member),
            semantic_validator=_stub_semantic_validator,
        )
        if collision == "tampered_tar":
            content = bytearray(archive.read_bytes())
            content[600] ^= 0x01
            archive.write_bytes(content)
        elif collision == "trailing_tar_bytes":
            archive.write_bytes(archive.read_bytes() + b"trailing-tamper")
        else:
            (archive.parent / "external-alias.member.tar").hardlink_to(archive)
    calls = 0

    def forbidden(_payload: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "best_epoch_index": 0,
            "best_selection_metric_value": 0.0,
        }

    with pytest.raises(Stage09ParallelError):
        publish_control_member(
            validated=validated,
            producer=forbidden,
            semantic_validator=_stub_semantic_validator,
        )
    assert calls == 0


def test_coordinator_proves_exact_matrix_and_rejects_missing_or_extra(
    tmp_path: Path,
) -> None:
    root, identity, gate, authorization, work_orders = _fixture_repository(
        tmp_path
    )
    references = _thermoroute_references(root, identity)
    _publish_all(root, identity, gate, work_orders)
    receipt_path = freeze_control_matrix_receipt(
        root=root,
        authorization_path=authorization,
        work_orders=work_orders,
        thermoroute_references=references,
        semantic_validator=_stub_semantic_validator,
        publication_guard=lambda: None,
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["format"] == COORDINATOR_RECEIPT_FORMAT
    assert receipt["status"] == "EXACT_35_MEMBER_MATRIX_COMPLETE"
    assert len(receipt["member_registry"]) == 35
    assert [
        item["seed"] for item in receipt["same_seed_thermoroute_references"]
    ] == [0, 1, 2, 3, 4]
    assert receipt["matrix_audit"]["seeds_per_arm"] == [0, 1, 2, 3, 4]
    assert receipt["matrix_audit"]["same_seed_reference_artifacts_bound"] is True
    assert receipt["matrix_audit"][
        "same_seed_forecast_keys_and_y_true_exactly_verified"
    ] is True
    assert receipt["matrix_audit"][
        "full_prediction_replay_equivalence_proved"
    ] is True
    assert receipt["serial_semantics_contract"] == {
        "same_member_algorithm_config_seed_and_epoch_policy": True,
        "member_results_sorted_before_aggregation": True,
        "ensemble_uses_equal_weight_across_exact_five_seeds": True,
        "parallelism_changes_only_wall_clock_scheduling": True,
        "parallelism_enters_scientific_identity": False,
        "best_seed_selection_allowed": False,
    }

    first_archive = root / receipt["member_registry"][0]["archive"]["path"]
    held = first_archive.with_suffix(".held")
    first_archive.rename(held)
    with pytest.raises(Stage09ParallelError, match="missing"):
        freeze_control_matrix_receipt(
            root=root,
            authorization_path=authorization,
            work_orders=work_orders,
            thermoroute_references=references,
            semantic_validator=_stub_semantic_validator,
            publication_guard=lambda: None,
        )
    held.rename(first_archive)
    extra = first_archive.parent / "seed99.member.tar"
    extra.write_bytes(first_archive.read_bytes())
    with pytest.raises(Stage09ParallelError, match="extra"):
        freeze_control_matrix_receipt(
            root=root,
            authorization_path=authorization,
            work_orders=work_orders,
            thermoroute_references=references,
            semantic_validator=_stub_semantic_validator,
            publication_guard=lambda: None,
        )
    extra.unlink()
    orphan_root = (
        authorization.parent.parent
        / ".stage09-control-member-resume"
        / "interrupted-member"
    )
    orphan_root.mkdir()
    (orphan_root / "training_checkpoint.pt").write_bytes(b"interrupted")
    with pytest.raises(Stage09ParallelError, match="resume.*quiescent"):
        freeze_control_matrix_receipt(
            root=root,
            authorization_path=authorization,
            work_orders=work_orders,
            thermoroute_references=references,
            semantic_validator=_stub_semantic_validator,
            publication_guard=lambda: None,
        )
    (orphan_root / "training_checkpoint.pt").unlink()
    orphan_root.rmdir()
    reference_weights = references[0]["bundle"] / "weights.pt"
    reference_weights.write_bytes(reference_weights.read_bytes() + b"tampered")
    with pytest.raises(Stage09ParallelError, match="reference|lineage|bundle"):
        freeze_control_matrix_receipt(
            root=root,
            authorization_path=authorization,
            work_orders=work_orders,
            thermoroute_references=references,
            semantic_validator=_stub_semantic_validator,
            publication_guard=lambda: None,
        )


def test_coordinator_rejects_noncanonical_same_run_reference_path(
    tmp_path: Path,
) -> None:
    root, identity, gate, authorization, work_orders = _fixture_repository(
        tmp_path
    )
    references = _thermoroute_references(root, identity)
    _publish_all(root, identity, gate, work_orders)
    alternate = (
        root
        / "outputs"
        / "runs"
        / "09_usgs_experiment"
        / identity.run_id
        / "predictions"
        / "alternate_seed0.parquet"
    )
    alternate.write_bytes(references[0]["prediction"].read_bytes())
    altered = {seed: dict(value) for seed, value in references.items()}
    altered[0]["prediction"] = alternate
    with pytest.raises(Stage09ParallelError, match="prediction path changed"):
        freeze_control_matrix_receipt(
            root=root,
            authorization_path=authorization,
            work_orders=work_orders,
            thermoroute_references=altered,
            semantic_validator=_stub_semantic_validator,
            publication_guard=lambda: None,
        )


def test_scheduler_enforces_upper_bound_and_gates_before_launch() -> None:
    work_orders = tuple(Path(f"/tmp/work-{index}.json") for index in range(35))
    lock = threading.Lock()
    active = 0
    maximum = 0
    gate_calls = 0

    def guard() -> None:
        nonlocal gate_calls
        with lock:
            gate_calls += 1

    def launch(_path: Path) -> int:
        nonlocal active, maximum
        with lock:
            assert gate_calls >= 1
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return 0

    execute_work_orders_bounded(
        work_orders,
        max_workers=3,
        launch=launch,
        publication_guard=guard,
    )
    assert 1 < maximum <= 3
    assert gate_calls == 2
    with pytest.raises(Stage09ParallelError, match="must be"):
        execute_work_orders_bounded(
            work_orders,
            max_workers=MAX_CONTROL_WORKERS + 1,
            launch=launch,
            publication_guard=guard,
        )


def test_scheduler_stops_replenishing_and_terminates_on_first_failure() -> None:
    work_orders = tuple(Path(f"/tmp/work-{index}.json") for index in range(35))
    release = threading.Event()
    lock = threading.Lock()
    launched: list[Path] = []
    terminate_calls = 0

    def launch(path: Path) -> int:
        with lock:
            launched.append(path)
        if path == work_orders[0]:
            return 7
        release.wait(timeout=5.0)
        return 0

    def terminate() -> None:
        nonlocal terminate_calls
        terminate_calls += 1
        release.set()

    with pytest.raises(Stage09ParallelError, match="worker failed"):
        execute_work_orders_bounded(
            work_orders,
            max_workers=3,
            launch=launch,
            terminate=terminate,
            publication_guard=lambda: None,
        )
    assert 1 <= len(launched) <= 3
    assert terminate_calls == 1


def test_work_order_or_authority_coordinated_edit_is_rejected(tmp_path: Path) -> None:
    root, _identity, _gate, authorization, work_orders = _fixture_repository(
        tmp_path
    )
    document = json.loads(work_orders[0].read_text())
    document["scientific_member_config"]["seed"] = 4
    stable = {
        key: value
        for key, value in document.items()
        if key != "work_order_self_sha256"
    }
    document["work_order_self_sha256"] = sha256_json(stable)
    work_orders[0].write_text(_canonical_json(document), encoding="utf-8")
    with pytest.raises(Stage09ParallelError, match="scientific member config"):
        validate_work_order_structure(root=root, work_order_path=work_orders[0])

    # A separately edited authority cannot be made authoritative merely by
    # resealing its own JSON; every work order carries the original file hash.
    authority = json.loads(authorization.read_text())
    authority["resolved_config"]["train_config"]["max_epochs"] = 79
    stable_authority = {
        key: value
        for key, value in authority.items()
        if key != "authorization_self_sha256"
    }
    authority["authorization_self_sha256"] = sha256_json(stable_authority)
    authorization.write_text(_canonical_json(authority), encoding="utf-8")
    with pytest.raises(Stage09ParallelError):
        validate_work_order_structure(root=root, work_order_path=work_orders[1])


def test_structural_validator_recomputes_run_id_content_address(
    tmp_path: Path,
) -> None:
    root, identity, _gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    forged = json.loads(work_orders[0].read_text())
    forged_run_id = "0" * 20
    assert forged_run_id != identity.run_id
    forged["run_identity"]["run_id"] = forged_run_id
    stable = {
        key: value
        for key, value in forged.items()
        if key != "work_order_self_sha256"
    }
    forged["work_order_self_sha256"] = sha256_json(stable)
    work_orders[0].write_text(_canonical_json(forged), encoding="utf-8")
    with pytest.raises(Stage09ParallelError, match="content address"):
        validate_work_order_structure(root=root, work_order_path=work_orders[0])


@pytest.mark.parametrize("attack", ["duplicate_key", "nonfinite", "pretty_json"])
def test_work_order_requires_one_exact_canonical_finite_json_line(
    tmp_path: Path,
    attack: str,
) -> None:
    root, _identity, _gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    path = work_orders[0]
    raw = path.read_bytes()
    if attack == "duplicate_key":
        raw = raw.replace(
            b"{",
            b'{"format":"thermoroute.stage09-control-work-order.v1",',
            1,
        )
        expected = "duplicate JSON key"
    elif attack == "nonfinite":
        marker = b'"model_matrix_contract_id":"'
        start = raw.index(marker) + len(marker)
        end = raw.index(b'"', start)
        raw = raw[: start - 1] + b"NaN" + raw[end + 1 :]
        expected = "nonfinite JSON value"
    else:
        raw = (json.dumps(json.loads(raw), indent=2) + "\n").encode()
        expected = "exact canonical JSON line"
    path.write_bytes(raw)
    with pytest.raises(Stage09ParallelError, match=expected):
        validate_work_order_structure(root=root, work_order_path=path)


@pytest.mark.parametrize("attack", ["duplicate_key", "nonfinite"])
def test_member_receipt_rejects_duplicate_or_nonfinite_json(
    tmp_path: Path,
    attack: str,
) -> None:
    root, identity, gate, _authorization, work_orders = _fixture_repository(
        tmp_path
    )
    member = expected_control_members()[0]
    archive_path, _reused = publish_control_member(
        validated=_validated(root, identity, gate, work_orders[0]),
        producer=_stub_producer(member),
        semantic_validator=_stub_semantic_validator,
    )
    if attack == "duplicate_key":
        _rewrite_archive_receipt(
            archive_path,
            lambda raw: raw.replace(
                b"{",
                b'{"status":"COMPLETE_AND_CREATE_ONLY",',
                1,
            ),
        )
        expected = "duplicate JSON key"
    else:
        _rewrite_archive_receipt(
            archive_path,
            lambda raw: raw.replace(
                b'"best_selection_metric_value":0.0',
                b'"best_selection_metric_value":NaN',
                1,
            ),
        )
        expected = "nonfinite JSON value"
    with pytest.raises(Stage09ParallelError, match=expected):
        validate_member_archive(root=root, work_order_path=work_orders[0])


def test_frozen_registry_names_match_amendment_order() -> None:
    members = expected_control_members()
    assert tuple(dict.fromkeys(member.arm_id for member in members)) == tuple(
        STAGE09_CONTROLS
    )
    assert [member.seed for member in members[:5]] == [0, 1, 2, 3, 4]
    assert len({(member.arm_id, member.seed) for member in members}) == 35
