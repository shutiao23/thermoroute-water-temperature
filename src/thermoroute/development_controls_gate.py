"""Fail-closed Stage-09b best-model-state prediction-replay gate.

The gate replays data preparation and window construction from the frozen
2006--2020 panel.  It then validates every semantic prediction field, derives
metrics again, regenerates the report byte-for-byte, and proves that the
combined Parquet is a full-column copy of all 31 immutable member artifacts.
Each member prediction is also regenerated from the safely loaded checkpoint
``best_model_state``.  The gate makes no full training-trajectory replay claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch

from . import config as C
from . import data as D
from . import datasets as DS
from . import features as F
from . import results as R
from .checkpoint import checkpoint_sidecar_path, load_training_checkpoint
from .development_controls import (
    DEVELOPMENT_DISCLOSURE,
    FULL_VARIABLES,
    TRAIN_CONFIG,
    ArmSpec,
    CanonicalWindowContract,
    architecture_budget_rows,
    architecture_configuration,
    architecture_template,
    assert_parameter_budgets,
    budget_csv_bytes,
    build_arm_model,
    declared_arms,
    expected_member_registry,
    normalise_prediction_frame,
    prediction_content_digest,
    rebuild_canonical_window_contract,
    recompute_metric_summary,
    render_report,
    summary_csv_bytes,
)
from .predictor_bridge import (
    assert_exact_predictor_table,
    compare_predictor_bridge,
    frozen_bridge_slice,
    replay_predictor_bridge_offline,
)
from .repro import (
    RUN_SCHEMA_VERSION,
    RunIdentity,
    atomic_write_json,
    sha256_file,
    sha256_json,
    sidecar_path,
    source_tree_hash,
    validate_artifact_sidecar,
)
from .train import export_predictions


STAGE09B_COMPLETION_FORMAT = "thermoroute.stage09b-completion-receipt.v3"
STAGE09B_COMPLETION_STATUS = "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY"
STAGE09B_COMPLETION_RECEIPT_PATH = "outputs/models/route_a_stage09b_completion.json"
STAGE09B_STAGE = "09b_development_controls"
STAGE09B_FINAL_FORMAT = "thermoroute.development-controls.v2"
STAGE09B_MEMBER_PREDICTION_KIND = "development_control_arm_predictions"
STAGE09B_MEMBER_EXTRA_FORMAT = "thermoroute.development-control-arm.v2"
STAGE09B_FINAL_PREDICTION_KIND = "development_controls_combined_predictions"
STAGE09B_SUMMARY_KIND = "development_controls_metric_summary"
STAGE09B_SEMANTIC_AUDIT_KIND = "development_controls_semantic_audit"
STAGE09B_SEMANTIC_AUDIT_FORMAT = "thermoroute.development-controls-semantic-audit.v2"
STAGE09B_FINAL_ARTIFACTS = (
    "run_manifest", "frozen_panel_spec", "panel", "registry", "predictor_bridge",
    "predictions", "prediction_sidecar", "architecture_budget",
    "architecture_budget_sidecar", "metric_summary", "metric_summary_sidecar",
    "report", "report_sidecar", "semantic_audit", "semantic_audit_sidecar",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_DATA_PATHS = {
    "frozen_panel_spec": "data_usgs/frozen_panel_v1.json",
    "panel": "data_usgs/panel_usgs_120v2.parquet",
    "registry": "data_usgs/station_registry_v1.csv",
    "predictor_bridge": "data_usgs/development_predictor_bridge_v1.json",
}


class DevelopmentControlsGateError(ValueError):
    """The Stage-09b closure is absent, stale, incomplete, or inconsistent."""


def expected_stage09b_members() -> tuple[tuple[str, int], ...]:
    members = expected_member_registry()
    if len(members) != 31:
        raise DevelopmentControlsGateError("Stage-09b member contract is not 31 members")
    return members


def _relative(root: Path, path: str | Path) -> str:
    root, resolved = root.resolve(), Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise DevelopmentControlsGateError("Stage-09b artifact escapes repository")
    return resolved.relative_to(root).as_posix()


def _file_binding(root: Path, path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise DevelopmentControlsGateError(f"Stage-09b artifact is absent: {resolved}")
    return {"path": _relative(root, resolved), "sha256": sha256_file(resolved)}


def _validated_binding(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise DevelopmentControlsGateError(f"{label} binding is malformed")
    raw = value.get("path")
    digest = value.get("sha256")
    if (
        not isinstance(raw, str) or Path(raw).is_absolute()
        or not isinstance(digest, str) or _HEX64.fullmatch(digest) is None
    ):
        raise DevelopmentControlsGateError(f"{label} binding is malformed")
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise DevelopmentControlsGateError(f"{label} path escapes repository")
    if (
        not path.is_file()
        or path.stat().st_nlink != 1
        or dict(value) != _file_binding(root, path)
    ):
        raise DevelopmentControlsGateError(f"{label} checksum or canonical path changed")
    return path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentControlsGateError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise DevelopmentControlsGateError(f"{label} is absent or malformed")
    return value


def _validate_formal_policy(value: object) -> None:
    if not isinstance(value, Mapping):
        raise DevelopmentControlsGateError("Stage-09b numerical policy is malformed")
    threads = value.get("thread_environment")
    required = value.get("required")
    torch_policy = value.get("torch")
    names = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    expected_required = {
        "threads": 1,
        "cublas_workspace_config": ":4096:8",
        "python_hash_policy": (
            "canonical-sort-identity-collections-independent-of-hash-secret"
        ),
        "torch_deterministic_algorithms": True,
        "tf32": False,
        "float32_matmul_precision": "highest",
    }
    expected_torch = {
        "num_threads": 1,
        "num_interop_threads": 1,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if (
        set(value)
        != {
            "thread_environment", "cublas_workspace_config",
            "python_hash_environment_declaration",
            "python_hash_randomization_enabled", "python_hash_policy",
            "required", "torch",
        }
        or not isinstance(threads, Mapping)
        or set(threads) != set(names)
        or any(threads.get(name) != "1" for name in names)
        or value.get("cublas_workspace_config") != ":4096:8"
        or value.get("python_hash_environment_declaration") != "0"
        or value.get("python_hash_randomization_enabled") is not True
        or value.get("python_hash_policy")
        != "canonical-sort-identity-collections-independent-of-hash-secret"
        or not isinstance(required, Mapping)
        or dict(required) != expected_required
        or not isinstance(torch_policy, Mapping)
        or dict(torch_policy) != expected_torch
    ):
        raise DevelopmentControlsGateError("Stage-09b numerical policy is not formal CPU policy")


def _arm_documents(arms: Sequence[ArmSpec]) -> list[dict[str, Any]]:
    return json.loads(json.dumps([asdict(arm) for arm in arms]))


def _validate_formal_configuration(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DevelopmentControlsGateError("Stage-09b formal configuration is malformed")
    arms = declared_arms()
    counts = assert_parameter_budgets(arms, n_stations=120)
    expected_keys = {
        "stage", "format", "execution_role", "evidence_role",
        "development_disclosure", "panel_date_range",
        "development_evaluation_interval", "blind_or_confirmatory",
        "suite_pointer_written", "training_device", "variables",
        "context_length", "horizons", "time_split", "station_sampling",
        "selection_metric", "train_config", "arms", "expected_member_registry",
        "parameter_counts", "architecture_templates",
        "parameter_match_tolerance_fraction", "architecture_candidates_per_arm",
        "historical_tuning_budget_equalized", "development_predictor_bridge",
        "formal_numerical_policy", "eval_batch_size",
    }
    expected_templates = {
        arm.arm_id: architecture_template(arm, n_stations=120) for arm in arms
    }
    bridge = value.get("development_predictor_bridge")
    if (
        set(value) != expected_keys
        or value.get("stage") != STAGE09B_STAGE
        or value.get("format") != STAGE09B_FINAL_FORMAT
        or value.get("execution_role")
        != "prelabel_relative_to_unopened_post_2020_confirmation"
        or value.get("evidence_role") != "development_only_exploratory"
        or value.get("development_disclosure") != DEVELOPMENT_DISCLOSURE
        or value.get("panel_date_range") != ["2006-01-01", "2020-12-31"]
        or value.get("development_evaluation_interval") != list(C.SPLIT.test)
        or value.get("blind_or_confirmatory") is not False
        or value.get("suite_pointer_written") is not False
        or value.get("training_device") != "cpu"
        or value.get("variables") != list(FULL_VARIABLES)
        or value.get("context_length") != C.CONTEXT_LENGTH
        or value.get("horizons") != list(C.HORIZONS)
        or value.get("time_split")
        != {key: list(interval) for key, interval in C.SPLIT.as_dict().items()}
        or value.get("station_sampling") != "balanced"
        or value.get("selection_metric") != "station_macro"
        or value.get("train_config") != asdict(TRAIN_CONFIG)
        or value.get("arms") != _arm_documents(arms)
        or value.get("expected_member_registry")
        != [[arm_id, seed] for arm_id, seed in expected_stage09b_members()]
        or value.get("parameter_counts") != counts
        or value.get("architecture_templates") != expected_templates
        or value.get("parameter_match_tolerance_fraction") != 0.02
        or value.get("architecture_candidates_per_arm") != 1
        or value.get("historical_tuning_budget_equalized") is not False
        or type(value.get("eval_batch_size")) is not int
        or value["eval_batch_size"] < 1
        or not isinstance(bridge, Mapping)
        or set(bridge) != {"path", "sha256"}
    ):
        raise DevelopmentControlsGateError("Stage-09b is not the exact formal 31-member run")
    _validate_formal_policy(value.get("formal_numerical_policy"))
    return value


def _validate_run_manifest(
    path: Path, *, root: Path, run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(path, label="Stage-09b run manifest")
    config = _validate_formal_configuration(manifest.get("resolved_config"))
    identity = manifest.get("identity")
    identity_keys = {
        "run_id", "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "schema_version",
    }
    digests = (
        "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256",
    )
    if (
        set(manifest) != {
            "schema_version", "identity", "resolved_config", "created_utc",
            "environment", "git", "provenance",
        }
        or manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or not isinstance(identity, dict) or set(identity) != identity_keys
        or identity.get("schema_version") != RUN_SCHEMA_VERSION
        or identity.get("run_id") != run_id
        or any(
            not isinstance(identity.get(field), str)
            or _HEX64.fullmatch(str(identity[field])) is None for field in digests
        )
        or identity.get("config_sha256") != sha256_json(config)
        or identity.get("source_sha256") != source_tree_hash(root)
    ):
        raise DevelopmentControlsGateError("Stage-09b receipt is stale for the run/source")
    identity_parts = {
        "schema_version": identity["schema_version"],
        **{field: identity[field] for field in digests},
    }
    if identity["run_id"] != sha256_json(identity_parts)[:20]:
        raise DevelopmentControlsGateError("Stage-09b run id is not content-addressed")
    try:
        created = datetime.fromisoformat(str(manifest["created_utc"]))
    except ValueError as exc:
        raise DevelopmentControlsGateError("Stage-09b timestamp is malformed") from exc
    provenance = manifest.get("provenance")
    if (
        created.tzinfo is None or created.utcoffset() is None
        or not isinstance(provenance, Mapping)
        or provenance.get("development_only") is not True
        or provenance.get("post_2020_outcomes_requested_or_read") is not False
        or provenance.get("suite_pointer_written") is not False
        or provenance.get("training_device") != "cpu"
    ):
        raise DevelopmentControlsGateError("Stage-09b development provenance changed")
    expected = (root / "outputs" / "runs" / STAGE09B_STAGE / run_id / "run.json").resolve()
    if path.resolve() != expected:
        raise DevelopmentControlsGateError("Stage-09b run manifest path is noncanonical")
    return dict(identity), config


def _validate_bridge_binding(root: Path, value: object, *, label: str) -> Path:
    return _validated_binding(root, value, label=f"predictor bridge {label}")


def _validate_snapshot_index(root: Path, path: Path) -> None:
    index = _load_json(path, label="predictor bridge snapshot index")
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise DevelopmentControlsGateError("predictor bridge snapshot index is empty")
    for record in records:
        if not isinstance(record, Mapping):
            raise DevelopmentControlsGateError("predictor bridge snapshot record is malformed")
        response_raw = record.get("response_path")
        response_sha = record.get("response_sha256")
        metadata_raw = record.get("metadata_path")
        if (
            not isinstance(response_raw, str) or Path(response_raw).is_absolute()
            or not isinstance(metadata_raw, str) or Path(metadata_raw).is_absolute()
            or not isinstance(response_sha, str) or _HEX64.fullmatch(response_sha) is None
            or type(record.get("byte_count")) is not int
        ):
            raise DevelopmentControlsGateError("predictor bridge snapshot record is malformed")
        response = (path.parent / response_raw).resolve()
        metadata = (path.parent / metadata_raw).resolve()
        if (
            root not in response.parents or root not in metadata.parents
            or not response.is_file() or not metadata.is_file()
            or response.stat().st_size != record["byte_count"]
            or sha256_file(response) != response_sha
        ):
            raise DevelopmentControlsGateError("predictor bridge nested snapshot changed")


def _validate_data_contract(
    *, root: Path, paths: Mapping[str, Path], identity: Mapping[str, Any],
) -> CanonicalWindowContract:
    if any(
        paths[label].resolve() != (root / relative).resolve()
        for label, relative in _CANONICAL_DATA_PATHS.items()
    ):
        raise DevelopmentControlsGateError(
            "Stage-09b development data leave the canonical namespace"
        )
    spec = _load_json(paths["frozen_panel_spec"], label="frozen panel specification")
    panel_spec = spec.get("panel")
    registry_spec = spec.get("station_registry")
    if (
        spec.get("schema_version") != 1
        or not isinstance(panel_spec, Mapping) or not isinstance(registry_spec, Mapping)
        or panel_spec.get("date_start") != "2006-01-01"
        or panel_spec.get("date_end") != "2020-12-31"
        or panel_spec.get("station_count") != 120
        or registry_spec.get("station_count") != 120
        or panel_spec.get("sha256") != identity.get("panel_sha256")
        or registry_spec.get("sha256") != identity.get("registry_sha256")
        or (paths["frozen_panel_spec"].parent / str(panel_spec.get("path"))).resolve()
        != paths["panel"]
        or (paths["frozen_panel_spec"].parent / str(registry_spec.get("path"))).resolve()
        != paths["registry"]
        or sha256_file(paths["panel"]) != identity.get("panel_sha256")
        or sha256_file(paths["registry"]) != identity.get("registry_sha256")
    ):
        raise DevelopmentControlsGateError("Stage-09b frozen panel/registry changed")
    try:
        registry = pd.read_csv(paths["registry"], dtype={"site_no": "string"})
    except Exception as exc:
        raise DevelopmentControlsGateError("Stage-09b registry cannot be read") from exc
    sites = tuple(sorted(registry.get("site_no", pd.Series(dtype="string")).dropna().astype(str)))
    if (
        len(registry) != 120 or len(set(sites)) != 120
        or any(not site.isdigit() or not 8 <= len(site) <= 15 for site in sites)
    ):
        raise DevelopmentControlsGateError("Stage-09b registry lacks stable site_no keys")
    bridge = _load_json(paths["predictor_bridge"], label="development predictor bridge")
    if (
        bridge.get("format") != "thermoroute.development-predictor-bridge.v1"
        or bridge.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or bridge.get("outcome_values_requested_or_read") is not False
        or not isinstance(bridge.get("source_tree_sha256"), str)
        or _HEX64.fullmatch(bridge["source_tree_sha256"]) is None
        or bridge.get("panel") != _file_binding(root, paths["panel"])
        or bridge.get("registry") != _file_binding(root, paths["registry"])
    ):
        raise DevelopmentControlsGateError("Stage-09b predictor bridge is stale")
    report_path = _validate_bridge_binding(root, bridge.get("report"), label="report")
    request_map_path = _validate_bridge_binding(
        root, bridge.get("request_map"), label="request_map"
    )
    normalized = bridge.get("normalized")
    raw_indexes = bridge.get("raw_snapshot_indexes")
    if not isinstance(normalized, Mapping) or set(normalized) != {"frozen", "refreshed"}:
        raise DevelopmentControlsGateError("predictor bridge normalized products are incomplete")
    expected_bridge_paths = {
        "frozen": (
            "data_usgs/development_predictor_bridge_v1/"
            "frozen_panel_predictors_2018_2020.parquet"
        ),
        "refreshed": (
            "data_usgs/development_predictor_bridge_v1/"
            "refreshed_predictors_2018_2020.parquet"
        ),
    }
    normalised_paths: dict[str, Path] = {}
    for label, binding in normalized.items():
        normalised_paths[label] = _validate_bridge_binding(
            root, binding, label=f"normalized/{label}"
        )
        if normalised_paths[label] != (root / expected_bridge_paths[label]).resolve():
            raise DevelopmentControlsGateError(
                "predictor bridge normalized path is noncanonical"
            )
    expected_raw_paths = {
        "daymet": (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "daymet-v1/snapshot_index_v2.json"
        ),
        "gridmet": (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-v1/snapshot_index_v2.json"
        ),
        "gridmet_schema": (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-schema-v1/snapshot_index_v2.json"
        ),
    }
    if not isinstance(raw_indexes, Mapping) or set(raw_indexes) != set(expected_raw_paths):
        raise DevelopmentControlsGateError("predictor bridge raw indexes are incomplete")
    for label, binding in raw_indexes.items():
        index_path = _validate_bridge_binding(root, binding, label=f"raw/{label}")
        if index_path != (root / expected_raw_paths[label]).resolve():
            raise DevelopmentControlsGateError("predictor bridge raw index is noncanonical")
        _validate_snapshot_index(root, index_path)
    if report_path != (
        root / "data_usgs/development_predictor_bridge_v1/bridge_report_v1.json"
    ).resolve() or request_map_path != (
        root / "data_usgs/development_predictor_bridge_v1/source_request_map_v1.json"
    ).resolve():
        raise DevelopmentControlsGateError("predictor bridge report/request path changed")
    try:
        panel = pd.read_parquet(paths["panel"])
        stable_registry = pd.read_csv(
            paths["registry"],
            dtype={"site_no": "string", "legacy_site_id": "string"},
            keep_default_na=False,
        )
        expected_frozen = frozen_bridge_slice(panel, stable_registry)
        frozen = pd.read_parquet(normalised_paths["frozen"])
        stored_refreshed = pd.read_parquet(normalised_paths["refreshed"])
        refreshed = replay_predictor_bridge_offline(
            registry_path=paths["registry"],
            request_map_path=request_map_path,
            daymet_index_path=(root / expected_raw_paths["daymet"]),
            gridmet_index_path=(root / expected_raw_paths["gridmet"]),
            gridmet_schema_index_path=(root / expected_raw_paths["gridmet_schema"]),
            expected_sites=120,
        )
        assert_exact_predictor_table(
            refreshed, stored_refreshed, label="stored refreshed predictor table"
        )
        frozen_attestation = compare_predictor_bridge(expected_frozen, frozen)
        report = compare_predictor_bridge(frozen, refreshed)
    except Exception as exc:
        raise DevelopmentControlsGateError("predictor bridge replay failed") from exc
    if frozen_attestation.get("status") != "PASS_EXACT_PRODUCT_BRIDGE":
        raise DevelopmentControlsGateError("normalized frozen predictors differ from panel")
    stored_report = _load_json(report_path, label="predictor bridge report")
    request_map = _load_json(request_map_path, label="predictor bridge request map")
    if stored_report != report:
        raise DevelopmentControlsGateError("predictor bridge report is not replay-derived")
    if (
        request_map.get("format")
        != "thermoroute.development-predictor-bridge-requests.v1"
        or request_map.get("outcome_values_requested_or_read") is not False
        or request_map.get("interval") != report.get("interval")
        or type(request_map.get("request_count")) is not int
        or not isinstance(request_map.get("requests"), list)
        or request_map.get("request_count") != len(request_map["requests"])
        or not isinstance(request_map.get("gridmet_provider_contract"), Mapping)
    ):
        raise DevelopmentControlsGateError("predictor bridge request map changed")
    try:
        contract = rebuild_canonical_window_contract(
            panel_path=paths["panel"], frozen_spec_path=paths["frozen_panel_spec"],
        )
    except Exception as exc:
        raise DevelopmentControlsGateError("canonical window registry replay failed") from exc
    if contract.stations != sites:
        raise DevelopmentControlsGateError("canonical panel does not decode to stable site_no")
    return contract


def _expected_member_extra(
    arm: ArmSpec, *, seed: int, n_stations: int, eval_batch_size: int,
) -> dict[str, Any]:
    return {
        "format": STAGE09B_MEMBER_EXTRA_FORMAT,
        "arm_id": arm.arm_id,
        "family": arm.family,
        "feature_set": arm.feature_set,
        "variables": list(arm.variables),
        "seed": seed,
        "trainable_parameters": architecture_configuration(
            arm, seed=seed, n_stations=n_stations,
        )["trainable_parameters"],
        "architecture": architecture_configuration(
            arm, seed=seed, n_stations=n_stations,
        ),
        "training_device": "cpu",
        "station_balanced": True,
        "selection_metric": "station_macro",
        "train_config": asdict(TRAIN_CONFIG),
        "context_length": C.CONTEXT_LENGTH,
        "horizons": list(C.HORIZONS),
        "development_only": True,
        "development_evaluation_interval": list(C.SPLIT.test),
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "eval_batch_size": int(eval_batch_size),
    }


def _validate_training_summary(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "best_validation_metric", "selected_epoch", "checkpoint_final_epoch",
    }:
        raise DevelopmentControlsGateError("Stage-09b training summary is malformed")
    best, selected, final = (
        value["best_validation_metric"], value["selected_epoch"],
        value["checkpoint_final_epoch"],
    )
    if (
        type(best) not in {int, float} or not math.isfinite(float(best))
        or float(best) < 0.0
        or type(selected) is not int or selected < 0
        or type(final) is not int or final < selected
    ):
        raise DevelopmentControlsGateError("Stage-09b training summary is malformed")


def _replay_member_best_state(
    *, checkpoint: Path, arm: ArmSpec, seed: int, wd: DS.WindowedData,
    thresholds: dict[str, float], identity: Mapping[str, Any],
    config: Mapping[str, Any], contract: CanonicalWindowContract,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Safely reproduce one member from checkpoint ``best_model_state``."""
    model = build_arm_model(arm, seed=seed, n_stations=len(contract.stations)).to("cpu")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=TRAIN_CONFIG.lr, weight_decay=TRAIN_CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4,
    )
    arm_config = {
        **dict(config),
        "arm": asdict(arm),
        "seed": int(seed),
        "trainable_parameters": architecture_configuration(
            arm, seed=seed, n_stations=len(contract.stations)
        )["trainable_parameters"],
    }
    try:
        resumed = load_training_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
            expected_run_id=str(identity["run_id"]),
            expected_resolved_config=arm_config, map_location="cpu",
            recover_missing_sidecar=False,
        )
        if resumed.best_model_state is None:
            raise ValueError("checkpoint lacks best_model_state")
        model.load_state_dict(resumed.best_model_state)
        frame = export_predictions(
            model, wd, thresholds, torch.device("cpu"), arm.arm_id,
            "development_only_2006_2020", arm.feature_set, seed,
            batch_size=int(config["eval_batch_size"]),
            splits=("val", "calib", "test"),
        )
        frame["model"] = arm.arm_id
        frame["scope"] = "development_only_2006_2020"
        frame["feature_set"] = arm.feature_set
        frame["seed"] = int(seed)
        normalised = normalise_prediction_frame(
            frame, arm=arm, seed=seed, allowed_sites=set(contract.stations),
            canonical_registry=contract.registry,
        )
    except Exception as exc:
        raise DevelopmentControlsGateError(
            f"Stage-09b {arm.arm_id}/seed{seed} best-state replay failed"
        ) from exc
    return normalised, {
        "best_validation_metric": float(resumed.best_metric),
        "selected_epoch": int(resumed.best_epoch),
        "checkpoint_final_epoch": int(resumed.epoch),
    }


def _rebuild_member_replay_inputs(
    *, panel_path: Path, frozen_spec_path: Path, stations: Sequence[str],
) -> tuple[pd.DataFrame, D.SplitMasks, F.HarmonicClimatology, dict[str, float]]:
    """Rebuild all train-fit inputs used by checkpoint best-state export."""
    bundle = D.prepare_dataset_from_panel(
        str(panel_path), frozen_spec=frozen_spec_path, stable_site_ids=True,
    )
    panel_raw = bundle["panel_raw"]
    panel = bundle["panel"]
    masks = bundle["masks"]
    if not isinstance(panel_raw, pd.DataFrame) or not isinstance(panel, pd.DataFrame):
        raise TypeError("panel preparation returned invalid tables")
    if not isinstance(masks, D.SplitMasks):
        raise TypeError("panel preparation returned invalid split masks")
    climatology = F.HarmonicClimatology.fit(panel_raw, masks.train)
    thresholds = {
        site: float(
            panel_raw.loc[
                masks.train & panel_raw["site_id"].astype(str).eq(site), "WTEMP"
            ].quantile(C.EXCEEDANCE_QUANTILE)
        )
        for site in stations
    }
    if any(not math.isfinite(value) for value in thresholds.values()):
        raise ValueError("non-finite train threshold")
    return panel, masks, climatology, thresholds


def _validate_member_predictions(
    *, root: Path, member_registry: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any], expected_parents: Mapping[str, str],
    contract: CanonicalWindowContract, config: Mapping[str, Any],
    panel_path: Path, frozen_spec_path: Path,
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], str], pd.DataFrame]:
    expected = expected_stage09b_members()
    arms = {arm.arm_id: arm for arm in declared_arms()}
    observed: list[tuple[str, int]] = []
    counts: dict[tuple[str, int], int] = {}
    digests: dict[tuple[str, int], str] = {}
    summaries: list[pd.DataFrame] = []
    run_identity = RunIdentity(**identity)
    allowed_sites = set(contract.stations)
    try:
        panel, masks, climatology, thresholds = _rebuild_member_replay_inputs(
            panel_path=panel_path, frozen_spec_path=frozen_spec_path,
            stations=contract.stations,
        )
    except Exception as exc:
        raise DevelopmentControlsGateError("Stage-09b replay inputs cannot be rebuilt") from exc
    active_variables: tuple[str, ...] | None = None
    active_windows: DS.WindowedData | None = None
    for entry in member_registry:
        if not isinstance(entry, Mapping) or set(entry) != {
            "arm_id", "seed", "checkpoint", "checkpoint_sidecar",
            "prediction", "prediction_sidecar",
        }:
            raise DevelopmentControlsGateError("Stage-09b member receipt schema changed")
        arm_id, seed = entry.get("arm_id"), entry.get("seed")
        if not isinstance(arm_id, str) or type(seed) is not int or arm_id not in arms:
            raise DevelopmentControlsGateError("Stage-09b member identity is malformed")
        member = (arm_id, seed)
        observed.append(member)
        prediction = _validated_binding(root, entry["prediction"], label=f"member {member}")
        metadata_path = _validated_binding(
            root, entry["prediction_sidecar"], label=f"member sidecar {member}",
        )
        checkpoint = _validated_binding(
            root, entry["checkpoint"], label=f"member checkpoint {member}",
        )
        checkpoint_metadata = _validated_binding(
            root, entry["checkpoint_sidecar"],
            label=f"member checkpoint sidecar {member}",
        )
        expected_path = (
            root / "outputs" / "runs" / STAGE09B_STAGE / identity["run_id"]
            / "arm_predictions" / arm_id / f"seed{seed}.parquet"
        ).resolve()
        expected_checkpoint = (
            root / "outputs" / "runs" / STAGE09B_STAGE / identity["run_id"]
            / "checkpoints" / arm_id / f"seed{seed}.pt"
        ).resolve()
        if (
            prediction != expected_path
            or metadata_path != sidecar_path(prediction).resolve()
            or checkpoint != expected_checkpoint
            or checkpoint_metadata != checkpoint_sidecar_path(checkpoint).resolve()
        ):
            raise DevelopmentControlsGateError("Stage-09b member path registry changed")
        try:
            metadata = validate_artifact_sidecar(
                prediction, identity=run_identity, schema=R.PREDICTION_SCHEMA_VERSION,
                kind=STAGE09B_MEMBER_PREDICTION_KIND,
            )
        except (OSError, ValueError) as exc:
            raise DevelopmentControlsGateError("Stage-09b member sidecar is invalid") from exc
        static = _expected_member_extra(
            arms[arm_id], seed=seed, n_stations=120,
            eval_batch_size=int(config["eval_batch_size"]),
        )
        extra = metadata.get("extra")
        member_parents = {
            **dict(expected_parents),
            "training_checkpoint": sha256_file(checkpoint),
            "training_checkpoint_sidecar": sha256_file(checkpoint_metadata),
        }
        if (
            metadata.get("parents") != dict(sorted(member_parents.items()))
            or not isinstance(extra, dict)
            or set(extra) != {*static, "training_summary"}
            or any(extra.get(key) != value for key, value in static.items())
        ):
            raise DevelopmentControlsGateError("Stage-09b member architecture/sidecar changed")
        _validate_training_summary(extra["training_summary"])
        if active_variables != arms[arm_id].variables:
            try:
                active_windows = DS.build_windows(
                    panel, masks, climatology, context=C.CONTEXT_LENGTH,
                    horizons=C.HORIZONS, variables=arms[arm_id].variables,
                    require_observed_target=True,
                )
            except Exception as exc:
                raise DevelopmentControlsGateError(
                    "Stage-09b member windows cannot be rebuilt"
                ) from exc
            active_variables = arms[arm_id].variables
        assert active_windows is not None
        replayed, replay_summary = _replay_member_best_state(
            checkpoint=checkpoint, arm=arms[arm_id], seed=seed,
            wd=active_windows, thresholds=thresholds, identity=identity,
            config=config, contract=contract,
        )
        if dict(extra["training_summary"]) != replay_summary:
            raise DevelopmentControlsGateError(
                "Stage-09b training summary differs from checkpoint"
            )
        try:
            if pq.ParquetFile(prediction).schema_arrow.names != R.PRED_COLS:
                raise DevelopmentControlsGateError("Stage-09b member schema changed")
            frame = pd.read_parquet(prediction, columns=R.PRED_COLS)
            normalised = normalise_prediction_frame(
                frame, arm=arms[arm_id], seed=seed, allowed_sites=allowed_sites,
                canonical_registry=contract.registry,
            )
        except DevelopmentControlsGateError:
            raise
        except Exception as exc:
            raise DevelopmentControlsGateError(
                f"Stage-09b member semantic validation failed: {exc}"
            ) from exc
        counts[member] = len(normalised)
        digests[member] = prediction_content_digest(normalised)
        if digests[member] != prediction_content_digest(replayed):
            raise DevelopmentControlsGateError(
                "Stage-09b prediction differs from checkpoint best_model_state"
            )
        summaries.append(recompute_metric_summary({member: normalised}))
    if tuple(observed) != expected:
        raise DevelopmentControlsGateError("Stage-09b receipt does not bind exactly 31 members")
    summary = pd.concat(summaries, ignore_index=True).sort_values(
        ["arm_id", "seed", "split", "horizon"], kind="mergesort"
    ).reset_index(drop=True)
    return counts, digests, summary


def _expected_final_extra(audit: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    return {
        "format": STAGE09B_FINAL_FORMAT,
        "artifact_role": role,
        "expected_members": audit["expected_members"],
        "prediction_rows": audit["prediction_rows"],
        "common_forecast_keys_per_member": audit["common_forecast_keys"],
        "splits": audit["splits"],
        "reference_member": audit["reference_member"],
        "development_only": True,
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
    }


def _validate_combined_predictions(
    path: Path, *, root: Path, member_registry: Sequence[Mapping[str, Any]],
) -> None:
    import pyarrow as pa

    expected = expected_stage09b_members()
    try:
        combined = pq.ParquetFile(path)
        if combined.schema_arrow.names != R.PRED_COLS:
            raise DevelopmentControlsGateError("Stage-09b combined schema changed")
        iterator = iter(combined.iter_batches(columns=R.PRED_COLS, batch_size=65_536))
        current: Any | None = None
        current_offset = 0

        def take(rows: int) -> Any:
            nonlocal current, current_offset
            pieces: list[Any] = []
            remaining = rows
            while remaining:
                if current is None or current_offset == current.num_rows:
                    try:
                        current = next(iterator)
                    except StopIteration as exc:
                        raise DevelopmentControlsGateError(
                            "Stage-09b combined predictions end early"
                        ) from exc
                    current_offset = 0
                count = min(remaining, current.num_rows - current_offset)
                pieces.append(current.slice(current_offset, count))
                current_offset += count
                remaining -= count
            return pa.Table.from_batches(pieces).combine_chunks()

        if len(member_registry) != len(expected):
            raise DevelopmentControlsGateError("Stage-09b combined member registry changed")
        for entry, member in zip(member_registry, expected, strict=True):
            if (entry.get("arm_id"), entry.get("seed")) != member:
                raise DevelopmentControlsGateError("Stage-09b combined member order changed")
            source = _validated_binding(
                root, entry["prediction"], label=f"combined source member {member}",
            )
            member_file = pq.ParquetFile(source)
            if member_file.schema_arrow.names != R.PRED_COLS:
                raise DevelopmentControlsGateError("Stage-09b member schema changed")
            for batch in member_file.iter_batches(columns=R.PRED_COLS, batch_size=65_536):
                expected_table = pa.Table.from_batches([batch]).combine_chunks()
                if not take(batch.num_rows).equals(expected_table, check_metadata=False):
                    raise DevelopmentControlsGateError(
                        "Stage-09b combined predictions differ in a prediction column"
                    )
        if current is not None and current_offset < current.num_rows:
            raise DevelopmentControlsGateError("Stage-09b combined predictions have extras")
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise DevelopmentControlsGateError("Stage-09b combined predictions have extras")
    except DevelopmentControlsGateError:
        raise
    except Exception as exc:
        raise DevelopmentControlsGateError("Stage-09b combined predictions are unreadable") from exc


def _descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _assert_no_transaction_temps(run_dir: Path) -> None:
    if not run_dir.is_dir():
        raise DevelopmentControlsGateError("Stage-09b run directory is absent")
    for path in run_dir.rglob("*"):
        name = path.name
        if (
            (name.startswith(".") and name.endswith(".tmp"))
            or name.endswith(".recovery-probe")
        ):
            raise DevelopmentControlsGateError(
                f"Stage-09b run retains an unbound transaction temp: {path}"
            )


def _expected_semantic_audit(
    *, run_id: str, audit: Mapping[str, Any], contract: CanonicalWindowContract,
    member_registry: Sequence[Mapping[str, Any]], member_digests: Mapping[tuple[str, int], str],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": STAGE09B_SEMANTIC_AUDIT_FORMAT,
        "status": "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "run_id": run_id,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
        "post_2020_outcomes_requested_or_read": False,
        "matrix_audit": dict(audit),
        "canonical_window_registry": {
            "sha256": contract.registry_sha256,
            "common_forecast_keys": len(contract.registry),
            "train_examples_per_epoch": contract.train_examples,
            "train_registry_sha256": contract.train_registry_sha256,
        },
        "members": [
            {
                "arm_id": str(entry["arm_id"]),
                "seed": int(entry["seed"]),
                "checkpoint": _descriptor(paths_from_entry[2]),
                "checkpoint_sidecar": _descriptor(paths_from_entry[3]),
                "prediction": _descriptor(paths_from_entry[0]),
                "prediction_sidecar": _descriptor(paths_from_entry[1]),
                "normalised_prediction_sha256": member_digests[
                    (str(entry["arm_id"]), int(entry["seed"]))
                ],
                "best_model_state_prediction_replay_verified": True,
            }
            for entry in member_registry
            for paths_from_entry in [(
                Path(paths["run_manifest"]).parents[0]
                / "arm_predictions" / str(entry["arm_id"])
                / f"seed{int(entry['seed'])}.parquet",
                sidecar_path(
                    Path(paths["run_manifest"]).parents[0]
                    / "arm_predictions" / str(entry["arm_id"])
                    / f"seed{int(entry['seed'])}.parquet"
                ),
                Path(paths["run_manifest"]).parents[0]
                / "checkpoints" / str(entry["arm_id"])
                / f"seed{int(entry['seed'])}.pt",
                checkpoint_sidecar_path(
                    Path(paths["run_manifest"]).parents[0]
                    / "checkpoints" / str(entry["arm_id"])
                    / f"seed{int(entry['seed'])}.pt"
                ),
            )]
        ],
        "derived_artifacts": {
            "architecture_budget": {
                "artifact": _descriptor(paths["architecture_budget"]),
                "sidecar": _descriptor(paths["architecture_budget_sidecar"]),
            },
            "combined_predictions": {
                "artifact": _descriptor(paths["predictions"]),
                "sidecar": _descriptor(paths["prediction_sidecar"]),
            },
            "metric_summary": {
                "artifact": _descriptor(paths["metric_summary"]),
                "sidecar": _descriptor(paths["metric_summary_sidecar"]),
            },
            "report": {
                "artifact": _descriptor(paths["report"]),
                "sidecar": _descriptor(paths["report_sidecar"]),
            },
        },
    }
    document["semantic_audit_self_sha256"] = sha256_json(document)
    return document


def build_stage09b_completion_receipt(
    *, root: str | Path, run_id: str, run_manifest: str | Path,
    frozen_panel_spec: str | Path, panel: str | Path, registry: str | Path,
    predictor_bridge: str | Path,
    member_paths: Mapping[tuple[str, int], str | Path],
    predictions: str | Path, architecture_budget: str | Path,
    metric_summary: str | Path, report: str | Path, semantic_audit: str | Path,
    matrix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = _load_json(Path(run_manifest), label="Stage-09b run manifest")
    identity, config = manifest.get("identity"), manifest.get("resolved_config")
    if not isinstance(identity, dict) or not isinstance(config, dict):
        raise DevelopmentControlsGateError("Stage-09b run manifest is malformed")
    if identity.get("run_id") != run_id:
        raise DevelopmentControlsGateError("Stage-09b run id differs from manifest")
    expected = expected_stage09b_members()
    if set(member_paths) != set(expected) or len(member_paths) != len(expected):
        raise DevelopmentControlsGateError("Stage-09b receipt requires exactly 31 members")
    final = {
        "run_manifest": Path(run_manifest),
        "frozen_panel_spec": Path(frozen_panel_spec),
        "panel": Path(panel),
        "registry": Path(registry),
        "predictor_bridge": Path(predictor_bridge),
        "predictions": Path(predictions),
        "prediction_sidecar": sidecar_path(predictions),
        "architecture_budget": Path(architecture_budget),
        "architecture_budget_sidecar": sidecar_path(architecture_budget),
        "metric_summary": Path(metric_summary),
        "metric_summary_sidecar": sidecar_path(metric_summary),
        "report": Path(report),
        "report_sidecar": sidecar_path(report),
        "semantic_audit": Path(semantic_audit),
        "semantic_audit_sidecar": sidecar_path(semantic_audit),
    }
    document: dict[str, Any] = {
        "format": STAGE09B_COMPLETION_FORMAT,
        "status": STAGE09B_COMPLETION_STATUS,
        "stage": STAGE09B_STAGE,
        "run_id": run_id,
        "run_identity": identity,
        "formal_configuration": config,
        "evidence_scope": "best_model_state_prediction_replay",
        "best_model_state_prediction_replay_verified": True,
        "training_replay_verified": False,
        "matrix_audit": json.loads(json.dumps(dict(matrix_audit), sort_keys=True, allow_nan=False)),
        "member_registry": [
            {
                "arm_id": arm_id, "seed": seed,
                "checkpoint": _file_binding(
                    root,
                    Path(member_paths[(arm_id, seed)]).parents[2]
                    / "checkpoints" / arm_id / f"seed{seed}.pt",
                ),
                "checkpoint_sidecar": _file_binding(
                    root,
                    checkpoint_sidecar_path(
                        Path(member_paths[(arm_id, seed)]).parents[2]
                        / "checkpoints" / arm_id / f"seed{seed}.pt"
                    ),
                ),
                "prediction": _file_binding(root, member_paths[(arm_id, seed)]),
                "prediction_sidecar": _file_binding(
                    root, sidecar_path(member_paths[(arm_id, seed)]),
                ),
            }
            for arm_id, seed in expected
        ],
        "artifacts": {label: _file_binding(root, path) for label, path in final.items()},
        "post_2020_outcomes_requested_or_read": False,
    }
    document["receipt_self_sha256"] = sha256_json(document)
    return document


def validate_stage09b_completion_receipt(
    receipt_path: str | Path, *, root: str | Path,
    document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    receipt_path = Path(receipt_path).resolve()
    if receipt_path != (root / STAGE09B_COMPLETION_RECEIPT_PATH).resolve():
        raise DevelopmentControlsGateError("Stage-09b receipt path is noncanonical")
    receipt = _load_json(receipt_path, label="Stage-09b receipt") if document is None else dict(document)
    expected_keys = {
        "format", "status", "stage", "run_id", "run_identity",
        "formal_configuration", "evidence_scope", "training_replay_verified",
        "best_model_state_prediction_replay_verified",
        "matrix_audit", "member_registry", "artifacts",
        "post_2020_outcomes_requested_or_read", "receipt_self_sha256",
    }
    stable = {key: value for key, value in receipt.items() if key != "receipt_self_sha256"}
    run_id = receipt.get("run_id")
    if (
        set(receipt) != expected_keys
        or receipt.get("receipt_self_sha256") != sha256_json(stable)
        or receipt.get("format") != STAGE09B_COMPLETION_FORMAT
        or receipt.get("status") != STAGE09B_COMPLETION_STATUS
        or receipt.get("stage") != STAGE09B_STAGE
        or not isinstance(run_id, str) or not run_id
        or receipt.get("evidence_scope") != "best_model_state_prediction_replay"
        or receipt.get("training_replay_verified") is not False
        or receipt.get("best_model_state_prediction_replay_verified") is not True
        or receipt.get("post_2020_outcomes_requested_or_read") is not False
    ):
        raise DevelopmentControlsGateError("Stage-09b receipt is not an exact closure PASS")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(STAGE09B_FINAL_ARTIFACTS):
        raise DevelopmentControlsGateError("Stage-09b final artifact registry is incomplete")
    paths = {
        label: _validated_binding(root, artifacts[label], label=f"Stage-09b {label}")
        for label in STAGE09B_FINAL_ARTIFACTS
    }
    expected_run_dir = (
        root / "outputs" / "runs" / STAGE09B_STAGE / str(run_id)
    ).resolve()
    _assert_no_transaction_temps(expected_run_dir)
    expected_final_paths = {
        "run_manifest": expected_run_dir / "run.json",
        "predictions": expected_run_dir / "development_controls_predictions.parquet",
        "architecture_budget": (
            expected_run_dir / "development_controls_architecture_budget.csv"
        ),
        "metric_summary": expected_run_dir / "development_controls_metric_summary.csv",
        "report": expected_run_dir / "development_controls_report.md",
        "semantic_audit": expected_run_dir / "development_controls_semantic_audit.json",
        **{
            label: (root / relative).resolve()
            for label, relative in _CANONICAL_DATA_PATHS.items()
        },
    }
    if any(paths[label] != expected for label, expected in expected_final_paths.items()):
        raise DevelopmentControlsGateError("Stage-09b final artifact path is noncanonical")
    for artifact_label, sidecar_label in (
        ("predictions", "prediction_sidecar"),
        ("architecture_budget", "architecture_budget_sidecar"),
        ("metric_summary", "metric_summary_sidecar"),
        ("report", "report_sidecar"),
        ("semantic_audit", "semantic_audit_sidecar"),
    ):
        if paths[sidecar_label] != sidecar_path(paths[artifact_label]).resolve():
            raise DevelopmentControlsGateError("Stage-09b final sidecar alignment changed")
    identity, config = _validate_run_manifest(paths["run_manifest"], root=root, run_id=run_id)
    if receipt.get("run_identity") != identity or receipt.get("formal_configuration") != config:
        raise DevelopmentControlsGateError("Stage-09b run identity/configuration changed")
    if config["development_predictor_bridge"] != _file_binding(root, paths["predictor_bridge"]):
        raise DevelopmentControlsGateError("Stage-09b config binds another predictor bridge")
    contract = _validate_data_contract(root=root, paths=paths, identity=identity)
    members = receipt.get("member_registry")
    if not isinstance(members, list) or len(members) != 31:
        raise DevelopmentControlsGateError("Stage-09b receipt does not contain 31 members")
    parents = {
        "frozen_panel": identity["panel_sha256"],
        "frozen_station_registry": identity["registry_sha256"],
        "development_predictor_bridge": sha256_file(paths["predictor_bridge"]),
    }
    counts, member_digests, summary = _validate_member_predictions(
        root=root, member_registry=members, identity=identity,
        expected_parents=parents, contract=contract, config=config,
        panel_path=paths["panel"], frozen_spec_path=paths["frozen_panel_spec"],
    )
    audit = receipt.get("matrix_audit")
    expected_members = expected_stage09b_members()
    if (
        not isinstance(audit, Mapping)
        or set(audit) != {
            "expected_members", "prediction_rows", "common_forecast_keys",
            "splits", "reference_member",
        }
        or audit.get("expected_members") != 31
        or audit.get("common_forecast_keys") != len(contract.registry)
        or audit.get("prediction_rows") != len(contract.registry) * 31
        or audit.get("prediction_rows") != sum(counts.values())
        or audit.get("splits") != ["calib", "test", "val"]
        or audit.get("reference_member")
        != f"{expected_members[0][0]}/seed{expected_members[0][1]}"
    ):
        raise DevelopmentControlsGateError("Stage-09b matrix audit is incomplete or stale")
    final_parents = {
        **parents,
        **{
            f"arm::{entry['arm_id']}::seed{entry['seed']}::prediction": (
                entry["prediction"]["sha256"]
            )
            for entry in members
        },
        **{
            f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint": (
                entry["checkpoint"]["sha256"]
            )
            for entry in members
        },
        **{
            f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint_sidecar": (
                entry["checkpoint_sidecar"]["sha256"]
            )
            for entry in members
        },
    }
    specs = (
        ("predictions", STAGE09B_FINAL_PREDICTION_KIND, R.PREDICTION_SCHEMA_VERSION, "combined_predictions"),
        ("architecture_budget", "development_controls_budget", "text/csv", "architecture_budget"),
        ("metric_summary", STAGE09B_SUMMARY_KIND, "text/csv", "metric_summary"),
        ("report", "development_controls_report", "text/markdown", "report"),
        ("semantic_audit", STAGE09B_SEMANTIC_AUDIT_KIND, "application/json", "semantic_audit"),
    )
    run_identity = RunIdentity(**identity)
    for label, kind, schema, role in specs:
        try:
            metadata = validate_artifact_sidecar(
                paths[label], identity=run_identity, schema=schema, kind=kind,
            )
        except (OSError, ValueError) as exc:
            raise DevelopmentControlsGateError("Stage-09b final sidecar is invalid") from exc
        if (
            metadata.get("parents") != dict(sorted(final_parents.items()))
            or metadata.get("extra") != _expected_final_extra(audit, role=role)
        ):
            raise DevelopmentControlsGateError("Stage-09b final artifact closure changed")
    expected_budget = architecture_budget_rows(
        declared_arms(), n_stations=120, train_examples=contract.train_examples,
    )
    if paths["architecture_budget"].read_bytes() != budget_csv_bytes(expected_budget):
        raise DevelopmentControlsGateError("Stage-09b architecture/optimizer budget changed")
    if paths["metric_summary"].read_bytes() != summary_csv_bytes(summary):
        raise DevelopmentControlsGateError("Stage-09b metric summary is not prediction-derived")
    expected_report = render_report(
        run_id=run_id, audit=audit, budget=expected_budget, summary=summary,
    ).encode("utf-8")
    if paths["report"].read_bytes() != expected_report:
        raise DevelopmentControlsGateError("Stage-09b report is not summary-derived")
    _validate_combined_predictions(
        paths["predictions"], root=root, member_registry=members,
    )
    expected_semantic = _expected_semantic_audit(
        run_id=run_id, audit=audit, contract=contract,
        member_registry=members, member_digests=member_digests, paths=paths,
    )
    semantic = _load_json(paths["semantic_audit"], label="Stage-09b semantic audit")
    if semantic != expected_semantic:
        raise DevelopmentControlsGateError("Stage-09b semantic audit is stale or forged")
    return receipt


def write_stage09b_completion_receipt(path: str | Path, document: Mapping[str, Any]) -> Path:
    stable = {key: value for key, value in document.items() if key != "receipt_self_sha256"}
    if document.get("receipt_self_sha256") != sha256_json(stable):
        raise DevelopmentControlsGateError("Stage-09b receipt self hash is invalid")
    destination = Path(path)
    atomic_write_json(destination, dict(document))
    return destination


def publish_stage09b_completion_receipt(
    receipt_path: str | Path, document: Mapping[str, Any], *, root: str | Path,
) -> Path:
    validate_stage09b_completion_receipt(receipt_path, root=root, document=document)
    destination = write_stage09b_completion_receipt(receipt_path, document)
    validate_stage09b_completion_receipt(destination, root=root)
    return destination


def stage09b_completion_gate_binding(
    receipt_path: str | Path, *, root: str | Path,
) -> dict[str, str]:
    validate_stage09b_completion_receipt(receipt_path, root=root)
    return _file_binding(Path(root).resolve(), receipt_path)
