"""Fail-closed Stage-09b prediction-artifact completion gate.

The gate replays data preparation and window construction from the frozen
2006--2020 panel.  It then validates every semantic prediction field, derives
metrics again, regenerates the report byte-for-byte, and proves that the
combined Parquet is a full-column copy of all 31 immutable member artifacts.
It deliberately makes no checkpoint-backed training-replay claim.
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

from . import config as C
from . import results as R
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
    compare_predictor_bridge,
    frozen_bridge_slice,
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


STAGE09B_COMPLETION_FORMAT = "thermoroute.stage09b-completion-receipt.v2"
STAGE09B_COMPLETION_STATUS = "PASS_STAGE09B_PREDICTION_ARTIFACT_CLOSURE"
STAGE09B_COMPLETION_RECEIPT_PATH = "outputs/models/route_a_stage09b_completion.json"
STAGE09B_STAGE = "09b_development_controls"
STAGE09B_FINAL_FORMAT = "thermoroute.development-controls.v2"
STAGE09B_MEMBER_PREDICTION_KIND = "development_control_arm_predictions"
STAGE09B_MEMBER_EXTRA_FORMAT = "thermoroute.development-control-arm.v1"
STAGE09B_FINAL_PREDICTION_KIND = "development_controls_combined_predictions"
STAGE09B_SUMMARY_KIND = "development_controls_metric_summary"
STAGE09B_SEMANTIC_AUDIT_KIND = "development_controls_semantic_audit"
STAGE09B_SEMANTIC_AUDIT_FORMAT = "thermoroute.development-controls-semantic-audit.v1"
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
    if not path.is_file() or dict(value) != _file_binding(root, path):
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
        or value.get("python_hash_randomization_enabled") is not False
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
        "formal_numerical_policy",
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
            "daymet-v1/snapshot_index.json"
        ),
        "gridmet": (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-v1/snapshot_index.json"
        ),
        "gridmet_schema": (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-schema-v1/snapshot_index.json"
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
        refreshed = pd.read_parquet(normalised_paths["refreshed"])
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
    arm: ArmSpec, *, seed: int, n_stations: int,
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
        not isinstance(best, (int, float)) or not math.isfinite(float(best))
        or type(selected) is not int or selected < 0
        or (final is not None and (type(final) is not int or final < selected))
    ):
        raise DevelopmentControlsGateError("Stage-09b training summary is malformed")


def _validate_member_predictions(
    *, root: Path, member_registry: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any], expected_parents: Mapping[str, str],
    contract: CanonicalWindowContract,
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], str], pd.DataFrame]:
    expected = expected_stage09b_members()
    arms = {arm.arm_id: arm for arm in declared_arms()}
    observed: list[tuple[str, int]] = []
    counts: dict[tuple[str, int], int] = {}
    digests: dict[tuple[str, int], str] = {}
    summaries: list[pd.DataFrame] = []
    run_identity = RunIdentity(**identity)
    allowed_sites = set(contract.stations)
    for entry in member_registry:
        if not isinstance(entry, Mapping) or set(entry) != {
            "arm_id", "seed", "prediction", "prediction_sidecar",
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
        expected_path = (
            root / "outputs" / "runs" / STAGE09B_STAGE / identity["run_id"]
            / "arm_predictions" / arm_id / f"seed{seed}.parquet"
        ).resolve()
        if prediction != expected_path or metadata_path != sidecar_path(prediction).resolve():
            raise DevelopmentControlsGateError("Stage-09b member path registry changed")
        try:
            metadata = validate_artifact_sidecar(
                prediction, identity=run_identity, schema=R.PREDICTION_SCHEMA_VERSION,
                kind=STAGE09B_MEMBER_PREDICTION_KIND,
            )
        except (OSError, ValueError) as exc:
            raise DevelopmentControlsGateError("Stage-09b member sidecar is invalid") from exc
        static = _expected_member_extra(arms[arm_id], seed=seed, n_stations=120)
        extra = metadata.get("extra")
        if (
            metadata.get("parents") != dict(sorted(expected_parents.items()))
            or not isinstance(extra, dict)
            or set(extra) != {*static, "training_summary"}
            or any(extra.get(key) != value for key, value in static.items())
        ):
            raise DevelopmentControlsGateError("Stage-09b member architecture/sidecar changed")
        _validate_training_summary(extra["training_summary"])
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
        "evidence_scope": "prediction_artifact_closure",
        "training_replay_verified": False,
    }


def _validate_combined_predictions(
    path: Path, *, contract: CanonicalWindowContract,
    member_digests: Mapping[tuple[str, int], str],
) -> None:
    expected = expected_stage09b_members()
    arms = {arm.arm_id: arm for arm in declared_arms()}
    member_index = 0
    current_member: tuple[str, int] | None = None
    current_frames: list[pd.DataFrame] = []

    def finish_member() -> None:
        nonlocal member_index, current_member, current_frames
        if current_member is None:
            return
        if member_index >= len(expected) or current_member != expected[member_index]:
            raise DevelopmentControlsGateError("Stage-09b combined member order changed")
        frame = pd.concat(current_frames, ignore_index=True)
        try:
            normalised = normalise_prediction_frame(
                frame, arm=arms[current_member[0]], seed=current_member[1],
                allowed_sites=set(contract.stations),
                canonical_registry=contract.registry,
            )
        except Exception as exc:
            raise DevelopmentControlsGateError("combined prediction semantics changed") from exc
        if prediction_content_digest(normalised) != member_digests[current_member]:
            raise DevelopmentControlsGateError(
                "Stage-09b combined predictions differ in a prediction column"
            )
        member_index += 1
        current_member = None
        current_frames = []

    try:
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow.names != R.PRED_COLS:
            raise DevelopmentControlsGateError("Stage-09b combined schema changed")
        for batch in parquet.iter_batches(columns=R.PRED_COLS, batch_size=65_536):
            frame = batch.to_pandas()
            models = frame["model"].astype(str).to_numpy()
            seeds = pd.to_numeric(frame["seed"], errors="raise").astype(int).to_numpy()
            starts = [0]
            starts.extend(
                index for index in range(1, len(frame))
                if models[index] != models[index - 1] or seeds[index] != seeds[index - 1]
            )
            starts.append(len(frame))
            for start, stop in zip(starts[:-1], starts[1:], strict=True):
                member = (str(models[start]), int(seeds[start]))
                if current_member is not None and member != current_member:
                    finish_member()
                if current_member is None:
                    current_member = member
                current_frames.append(frame.iloc[start:stop].copy())
        finish_member()
    except DevelopmentControlsGateError:
        raise
    except Exception as exc:
        raise DevelopmentControlsGateError("Stage-09b combined predictions are unreadable") from exc
    if member_index != len(expected):
        raise DevelopmentControlsGateError("Stage-09b combined predictions omit members")


def _descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _expected_semantic_audit(
    *, run_id: str, audit: Mapping[str, Any], contract: CanonicalWindowContract,
    member_registry: Sequence[Mapping[str, Any]], member_digests: Mapping[tuple[str, int], str],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": STAGE09B_SEMANTIC_AUDIT_FORMAT,
        "status": "PASS_PREDICTION_ARTIFACT_CLOSURE",
        "run_id": run_id,
        "evidence_scope": "prediction_artifact_closure",
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
                "prediction": _descriptor(paths_from_entry[0]),
                "prediction_sidecar": _descriptor(paths_from_entry[1]),
                "normalised_prediction_sha256": member_digests[
                    (str(entry["arm_id"]), int(entry["seed"]))
                ],
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
        "evidence_scope": "prediction_artifact_closure",
        "training_replay_verified": False,
        "matrix_audit": json.loads(json.dumps(dict(matrix_audit), sort_keys=True, allow_nan=False)),
        "member_registry": [
            {
                "arm_id": arm_id, "seed": seed,
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
        or receipt.get("evidence_scope") != "prediction_artifact_closure"
        or receipt.get("training_replay_verified") is not False
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
        expected_parents=parents, contract=contract,
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
            f"arm::{entry['arm_id']}::seed{entry['seed']}": entry["prediction"]["sha256"]
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
        paths["predictions"], contract=contract, member_digests=member_digests,
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
