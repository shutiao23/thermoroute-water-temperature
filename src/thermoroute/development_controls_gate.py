"""Fail-closed completion gate for the development-only matched controls.

This module has no acquisition path.  It validates only the frozen 2006--2020
development artifacts produced by :mod:`scripts/09b_development_controls.py`.
The final receipt binds the run identity, exact 31-member matrix, common
forecast-key closure, architecture budget, combined predictions, report, and
every lineage sidecar.  A receipt is therefore an admission record, not a
success-shaped marker file.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
from . import results as R
from .registry import FORECAST_KEY, targets_match_at_model_precision
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


STAGE09B_COMPLETION_FORMAT = "thermoroute.stage09b-completion-receipt.v1"
STAGE09B_COMPLETION_STATUS = "PASS_FORMAL_STAGE09B_CONTROLS_COMPLETE"
STAGE09B_COMPLETION_RECEIPT_PATH = (
    "outputs/models/route_a_stage09b_completion.json"
)
STAGE09B_STAGE = "09b_development_controls"
STAGE09B_FINAL_FORMAT = "thermoroute.development-controls.v1"
STAGE09B_DEVELOPMENT_SCOPE = "development_only_2006_2020"
STAGE09B_DEVELOPMENT_DISCLOSURE = (
    "2019-2020 outcomes were already inspected during development; this is "
    "exploratory development evidence, not a blind or confirmatory test."
)
STAGE09B_FINAL_PREDICTION_KIND = "development_controls_combined_predictions"
STAGE09B_MEMBER_PREDICTION_KIND = "development_control_arm_predictions"
STAGE09B_MEMBER_EXTRA_FORMAT = "thermoroute.development-control-arm.v1"
STAGE09B_FINAL_ARTIFACTS = (
    "run_manifest",
    "frozen_panel_spec",
    "panel",
    "registry",
    "predictor_bridge",
    "predictions",
    "prediction_sidecar",
    "architecture_budget",
    "architecture_budget_sidecar",
    "report",
    "report_sidecar",
)
STAGE09B_FULL_VARIABLES = (
    "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP",
)
_LADDER = (
    ("01_WTEMP", ("WTEMP",)),
    ("02_plus_FLOW", ("WTEMP", "FLOW")),
    ("03_plus_TEMP", ("WTEMP", "FLOW", "TEMP")),
    ("04_plus_PRCP", ("WTEMP", "FLOW", "TEMP", "PRCP")),
    ("05_plus_RHMEAN", ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN")),
    ("06_plus_DH", ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH")),
    ("07_plus_WDSP", STAGE09B_FULL_VARIABLES),
)
STAGE09B_ARMS: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "PlainMLP-7var",
        "family": "PlainMLP",
        "feature_set": "all_7_variables",
        "variables": list(STAGE09B_FULL_VARIABLES),
        "seeds": list(C.USGS_SEEDS),
    },
    {
        "arm_id": "PlainCausalTCN-7var",
        "family": "PlainCausalTCN",
        "feature_set": "all_7_variables",
        "variables": list(STAGE09B_FULL_VARIABLES),
        "seeds": list(C.USGS_SEEDS),
    },
    *(
        {
            "arm_id": f"ThermoRoute-ladder-{rung}",
            "family": "ThermoRoute",
            "feature_set": f"feature_ladder_{rung}",
            "variables": list(variables),
            "seeds": list(C.USGS_SEEDS[:3]),
        }
        for rung, variables in _LADDER
    ),
)
STAGE09B_PARAMETER_COUNTS = {
    "PlainMLP-7var": 38_545,
    "PlainCausalTCN-7var": 38_031,
    "ThermoRoute-ladder-01_WTEMP": 37_775,
    "ThermoRoute-ladder-02_plus_FLOW": 37_896,
    "ThermoRoute-ladder-03_plus_TEMP": 38_018,
    "ThermoRoute-ladder-04_plus_PRCP": 38_139,
    "ThermoRoute-ladder-05_plus_RHMEAN": 38_261,
    "ThermoRoute-ladder-06_plus_DH": 38_383,
    "ThermoRoute-ladder-07_plus_WDSP": 38_505,
}
STAGE09B_REFERENCE_PARAMETERS = 38_505
STAGE09B_TRAIN_CONFIG = asdict(C.TrainConfig(batch_size=1536))


class DevelopmentControlsGateError(ValueError):
    """The Stage-09b closure is absent, stale, incomplete, or inconsistent."""


def expected_stage09b_members() -> tuple[tuple[str, int], ...]:
    """Return the exact 5 + 5 + 21 member registry in publication order."""
    members = tuple(
        (str(arm["arm_id"]), int(seed))
        for arm in STAGE09B_ARMS
        for seed in arm["seeds"]
    )
    if len(members) != 31 or len(set(members)) != 31:
        raise DevelopmentControlsGateError("Stage-09b member contract is invalid")
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
    if not isinstance(raw, str) or Path(raw).is_absolute():
        raise DevelopmentControlsGateError(f"{label} path is not repository-relative")
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
    if (
        not isinstance(threads, Mapping)
        or any(threads.get(name) != "1" for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        ))
        or value.get("cublas_workspace_config") != ":4096:8"
        or value.get("python_hash_policy")
        != "canonical-sort-identity-collections-independent-of-hash-secret"
        or not isinstance(required, Mapping)
        or required.get("threads") != 1
        or required.get("torch_deterministic_algorithms") is not True
        or required.get("tf32") is not False
        or not isinstance(torch_policy, Mapping)
        or torch_policy.get("num_threads") != 1
        or torch_policy.get("num_interop_threads") != 1
        or torch_policy.get("deterministic_algorithms") is not True
        or torch_policy.get("cuda_matmul_allow_tf32") is not False
        or torch_policy.get("cudnn_allow_tf32") is not False
        or torch_policy.get("float32_matmul_precision") != "highest"
    ):
        raise DevelopmentControlsGateError("Stage-09b numerical policy is not formal CPU policy")


def _validate_formal_configuration(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DevelopmentControlsGateError("Stage-09b formal configuration is malformed")
    expected_keys = {
        "stage", "format", "execution_role", "evidence_role",
        "development_disclosure", "panel_date_range",
        "development_evaluation_interval", "blind_or_confirmatory",
        "suite_pointer_written", "training_device", "variables",
        "context_length", "horizons", "time_split", "station_sampling",
        "selection_metric", "train_config", "arms",
        "expected_member_registry", "parameter_counts",
        "architecture_templates", "parameter_match_tolerance_fraction",
        "architecture_candidates_per_arm", "historical_tuning_budget_equalized",
        "development_predictor_bridge", "formal_numerical_policy",
    }
    expected_registry = [[arm, seed] for arm, seed in expected_stage09b_members()]
    if (
        set(value) != expected_keys
        or value.get("stage") != STAGE09B_STAGE
        or value.get("format") != STAGE09B_FINAL_FORMAT
        or value.get("execution_role")
        != "prelabel_relative_to_unopened_post_2020_confirmation"
        or value.get("evidence_role") != "development_only_exploratory"
        or value.get("development_disclosure") != STAGE09B_DEVELOPMENT_DISCLOSURE
        or value.get("panel_date_range") != ["2006-01-01", "2020-12-31"]
        or value.get("development_evaluation_interval") != list(C.SPLIT.test)
        or value.get("blind_or_confirmatory") is not False
        or value.get("suite_pointer_written") is not False
        or value.get("training_device") != "cpu"
        or value.get("variables") != list(STAGE09B_FULL_VARIABLES)
        or value.get("context_length") != C.CONTEXT_LENGTH
        or value.get("horizons") != list(C.HORIZONS)
        or value.get("time_split") != {
            key: list(interval) for key, interval in C.SPLIT.as_dict().items()
        }
        or value.get("station_sampling") != "balanced"
        or value.get("selection_metric") != "station_macro"
        or value.get("train_config") != STAGE09B_TRAIN_CONFIG
        or value.get("arms") != list(STAGE09B_ARMS)
        or value.get("expected_member_registry") != expected_registry
        or value.get("parameter_counts") != STAGE09B_PARAMETER_COUNTS
        or value.get("parameter_match_tolerance_fraction") != 0.02
        or value.get("architecture_candidates_per_arm") != 1
        or value.get("historical_tuning_budget_equalized") is not False
    ):
        raise DevelopmentControlsGateError("Stage-09b is not the exact formal 31-member run")
    templates = value.get("architecture_templates")
    if not isinstance(templates, Mapping) or set(templates) != set(
        STAGE09B_PARAMETER_COUNTS
    ):
        raise DevelopmentControlsGateError("Stage-09b architecture template registry changed")
    for arm_id, template in templates.items():
        if (
            not isinstance(template, Mapping)
            or template.get("initialization_seed_policy")
            != "exact declared member seed"
            or template.get("trainable_parameters")
            != STAGE09B_PARAMETER_COUNTS[arm_id]
        ):
            raise DevelopmentControlsGateError("Stage-09b architecture template changed")
    bridge = value.get("development_predictor_bridge")
    if not isinstance(bridge, Mapping) or set(bridge) != {"path", "sha256"}:
        raise DevelopmentControlsGateError("Stage-09b predictor bridge binding is malformed")
    _validate_formal_policy(value.get("formal_numerical_policy"))
    return value


def _validate_run_manifest(
    path: Path, *, root: Path, run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(path, label="Stage-09b run manifest")
    expected_manifest_keys = {
        "schema_version", "identity", "resolved_config", "created_utc",
        "environment", "git", "provenance",
    }
    identity = manifest.get("identity")
    identity_keys = {
        "run_id", "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "schema_version",
    }
    digests = (
        "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256",
    )
    config = _validate_formal_configuration(manifest.get("resolved_config"))
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or not isinstance(identity, dict)
        or set(identity) != identity_keys
        or identity.get("schema_version") != RUN_SCHEMA_VERSION
        or identity.get("run_id") != run_id
        or any(
            not isinstance(identity.get(field), str)
            or len(str(identity[field])) != 64
            for field in digests
        )
        or identity.get("config_sha256") != sha256_json(config)
        or identity.get("source_sha256") != source_tree_hash(root)
    ):
        raise DevelopmentControlsGateError(
            "Stage-09b completion receipt is stale for the run or current source"
        )
    identity_parts = {
        "schema_version": identity["schema_version"],
        **{field: identity[field] for field in digests},
    }
    if identity["run_id"] != sha256_json(identity_parts)[:20]:
        raise DevelopmentControlsGateError("Stage-09b run id is not content-addressed")
    try:
        created = datetime.fromisoformat(str(manifest["created_utc"]))
    except ValueError as exc:
        raise DevelopmentControlsGateError("Stage-09b run timestamp is malformed") from exc
    provenance = manifest.get("provenance")
    if (
        created.tzinfo is None
        or created.utcoffset() is None
        or not isinstance(provenance, Mapping)
        or provenance.get("development_only") is not True
        or provenance.get("post_2020_outcomes_requested_or_read") is not False
        or provenance.get("suite_pointer_written") is not False
        or provenance.get("training_device") != "cpu"
    ):
        raise DevelopmentControlsGateError("Stage-09b development-only provenance changed")
    expected_manifest = (
        root / "outputs" / "runs" / STAGE09B_STAGE / run_id / "run.json"
    ).resolve()
    if path.resolve() != expected_manifest:
        raise DevelopmentControlsGateError("Stage-09b receipt binds a noncanonical run manifest")
    return dict(identity), config


def _validate_data_contract(
    *, root: Path, paths: Mapping[str, Path], identity: Mapping[str, Any],
) -> set[str]:
    spec = _load_json(paths["frozen_panel_spec"], label="frozen panel specification")
    panel_spec = spec.get("panel")
    registry_spec = spec.get("station_registry")
    if (
        spec.get("schema_version") != 1
        or not isinstance(panel_spec, Mapping)
        or not isinstance(registry_spec, Mapping)
        or panel_spec.get("date_start") != "2006-01-01"
        or panel_spec.get("date_end") != "2020-12-31"
        or panel_spec.get("station_count") != 120
        or registry_spec.get("station_count") != 120
        or panel_spec.get("sha256") != identity.get("panel_sha256")
        or registry_spec.get("sha256") != identity.get("registry_sha256")
        or (paths["frozen_panel_spec"].parent / str(panel_spec.get("path"))).resolve()
        != paths["panel"]
        or (
            paths["frozen_panel_spec"].parent / str(registry_spec.get("path"))
        ).resolve() != paths["registry"]
        or sha256_file(paths["panel"]) != identity.get("panel_sha256")
        or sha256_file(paths["registry"]) != identity.get("registry_sha256")
    ):
        raise DevelopmentControlsGateError("Stage-09b frozen panel/registry contract changed")
    bridge = _load_json(paths["predictor_bridge"], label="development predictor bridge")
    if (
        bridge.get("format") != "thermoroute.development-predictor-bridge.v1"
        or bridge.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or bridge.get("outcome_values_requested_or_read") is not False
        or not isinstance(bridge.get("source_tree_sha256"), str)
        or len(bridge["source_tree_sha256"]) != 64
        or bridge.get("panel") != _file_binding(root, paths["panel"])
        or bridge.get("registry") != _file_binding(root, paths["registry"])
    ):
        raise DevelopmentControlsGateError("Stage-09b predictor bridge is stale")
    try:
        registry = pd.read_csv(
            paths["registry"],
            dtype={"site_no": "string", "legacy_site_id": "string"},
        )
    except Exception as exc:
        raise DevelopmentControlsGateError("Stage-09b registry cannot be read") from exc
    if "site_no" not in registry or "legacy_site_id" not in registry or len(registry) != 120:
        raise DevelopmentControlsGateError("Stage-09b registry is not the exact 120-site cohort")
    if registry["site_no"].astype(str).nunique() != 120:
        raise DevelopmentControlsGateError("Stage-09b registry primary keys are invalid")
    sites = set(registry["legacy_site_id"].astype(str))
    if len(sites) != 120 or any(not site for site in sites):
        raise DevelopmentControlsGateError("Stage-09b registry site ids are invalid")
    return sites


def _expected_member_extra(
    config: Mapping[str, Any], arm: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    template = deepcopy(config["architecture_templates"][arm["arm_id"]])
    template.pop("initialization_seed_policy", None)
    constructor = template.get("constructor_kwargs")
    if isinstance(constructor, dict) and constructor.get("init_seed") == "member_seed":
        constructor["init_seed"] = seed
    if template.get("initialization_seed") == "member_seed":
        template["initialization_seed"] = seed
    return {
        "format": STAGE09B_MEMBER_EXTRA_FORMAT,
        "arm_id": arm["arm_id"],
        "family": arm["family"],
        "feature_set": arm["feature_set"],
        "variables": arm["variables"],
        "seed": seed,
        "trainable_parameters": STAGE09B_PARAMETER_COUNTS[arm["arm_id"]],
        "architecture": template,
        "training_device": "cpu",
        "station_balanced": True,
        "selection_metric": "station_macro",
        "train_config": STAGE09B_TRAIN_CONFIG,
        "context_length": C.CONTEXT_LENGTH,
        "horizons": list(C.HORIZONS),
        "development_only": True,
        "development_evaluation_interval": list(C.SPLIT.test),
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
    }


def _normalised_key_truth(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["split", *FORECAST_KEY, "y_true"]
    if list(frame.columns) != columns:
        frame = frame.loc[:, columns].copy()
    else:
        frame = frame.copy()
    frame["site_id"] = frame["site_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="raise").astype("int64")
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    key = ["split", *FORECAST_KEY]
    if frame.duplicated(key).any():
        raise DevelopmentControlsGateError("Stage-09b prediction has duplicate forecast keys")
    if (
        set(frame["split"]) != {"val", "calib", "test"}
        or set(frame["horizon"]) != set(C.HORIZONS)
        or frame["target_date"].max() > pd.Timestamp("2020-12-31")
        or frame["issue_date"].min() < pd.Timestamp("2016-01-01")
        or not np.isfinite(pd.to_numeric(frame["y_true"], errors="coerce")).all()
    ):
        raise DevelopmentControlsGateError("Stage-09b prediction escaped development scope")
    for split, (lower, upper) in C.SPLIT.as_dict().items():
        if split == "train":
            continue
        selected = frame["split"].eq(split)
        if (
            not selected.any()
            or (frame.loc[selected, "issue_date"] < pd.Timestamp(lower)).any()
            or (frame.loc[selected, "target_date"] > pd.Timestamp(upper)).any()
        ):
            raise DevelopmentControlsGateError(f"Stage-09b {split} forecast keys changed")
    return frame.sort_values(key, kind="mergesort").reset_index(drop=True)


def _validate_member_predictions(
    *,
    root: Path,
    member_registry: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    config: Mapping[str, Any],
    expected_parents: Mapping[str, str],
    allowed_sites: set[str],
) -> tuple[pd.DataFrame, dict[tuple[str, int], int]]:
    expected_members = expected_stage09b_members()
    arm_by_id = {str(arm["arm_id"]): arm for arm in STAGE09B_ARMS}
    observed: list[tuple[str, int]] = []
    reference: pd.DataFrame | None = None
    counts: dict[tuple[str, int], int] = {}
    run_identity = RunIdentity(**identity)
    for index, entry in enumerate(member_registry):
        if not isinstance(entry, Mapping) or set(entry) != {
            "arm_id", "seed", "prediction", "prediction_sidecar",
        }:
            raise DevelopmentControlsGateError("Stage-09b member receipt schema is not exact")
        arm_id, seed = entry.get("arm_id"), entry.get("seed")
        if not isinstance(arm_id, str) or type(seed) is not int:
            raise DevelopmentControlsGateError("Stage-09b member identity is malformed")
        member = (arm_id, seed)
        observed.append(member)
        if member not in expected_members or arm_id not in arm_by_id:
            raise DevelopmentControlsGateError("Stage-09b member registry contains an extra member")
        prediction = _validated_binding(
            root, entry["prediction"], label=f"Stage-09b member {arm_id}/seed{seed}"
        )
        metadata_path = _validated_binding(
            root,
            entry["prediction_sidecar"],
            label=f"Stage-09b member sidecar {arm_id}/seed{seed}",
        )
        expected_path = (
            root / "outputs" / "runs" / STAGE09B_STAGE / identity["run_id"]
            / "arm_predictions" / arm_id / f"seed{seed}.parquet"
        ).resolve()
        if prediction != expected_path or metadata_path != sidecar_path(prediction).resolve():
            raise DevelopmentControlsGateError("Stage-09b member path registry changed")
        try:
            metadata = validate_artifact_sidecar(
                prediction,
                identity=run_identity,
                schema=R.PREDICTION_SCHEMA_VERSION,
                kind=STAGE09B_MEMBER_PREDICTION_KIND,
            )
        except (OSError, ValueError) as exc:
            raise DevelopmentControlsGateError("Stage-09b member sidecar is invalid") from exc
        if metadata["parents"] != dict(sorted(expected_parents.items())):
            raise DevelopmentControlsGateError("Stage-09b member parent lineage changed")
        extra = metadata.get("extra")
        expected_extra = _expected_member_extra(config, arm_by_id.get(arm_id, {}), seed=seed)
        if not isinstance(extra, dict) or set(extra) != {*expected_extra, "training_summary"}:
            raise DevelopmentControlsGateError("Stage-09b member metadata schema changed")
        if any(extra.get(key) != value for key, value in expected_extra.items()):
            raise DevelopmentControlsGateError("Stage-09b member architecture or budget changed")
        summary = extra.get("training_summary")
        if (
            not isinstance(summary, Mapping)
            or set(summary) != {
                "best_validation_metric", "selected_epoch", "checkpoint_final_epoch",
            }
            or not isinstance(summary.get("best_validation_metric"), (int, float))
            or not math.isfinite(float(summary["best_validation_metric"]))
            or type(summary.get("selected_epoch")) is not int
            or summary["selected_epoch"] < 0
            or (
                summary.get("checkpoint_final_epoch") is not None
                and (
                    type(summary["checkpoint_final_epoch"]) is not int
                    or summary["checkpoint_final_epoch"] < summary["selected_epoch"]
                )
            )
        ):
            raise DevelopmentControlsGateError("Stage-09b training summary changed")
        try:
            schema_names = pq.ParquetFile(prediction).schema_arrow.names
            if schema_names != R.PRED_COLS:
                raise DevelopmentControlsGateError("Stage-09b member prediction schema changed")
            frame = pd.read_parquet(
                prediction, columns=["split", *FORECAST_KEY, "y_true"]
            )
        except DevelopmentControlsGateError:
            raise
        except Exception as exc:
            raise DevelopmentControlsGateError("Stage-09b member prediction is unreadable") from exc
        if set(frame["site_id"].astype(str)) != allowed_sites:
            raise DevelopmentControlsGateError("Stage-09b member station registry changed")
        current = _normalised_key_truth(frame)
        if reference is None:
            reference = current
        else:
            keys = ["split", *FORECAST_KEY]
            if (
                not current[keys].equals(reference[keys])
                or not targets_match_at_model_precision(
                    current["y_true"], reference["y_true"]
                )
            ):
                raise DevelopmentControlsGateError(
                    "Stage-09b members do not share exact common keys and truth"
                )
        counts[member] = len(current)
    if tuple(observed) != expected_members:
        raise DevelopmentControlsGateError("Stage-09b receipt does not bind exactly 31 members")
    assert reference is not None
    return reference, counts


def _validate_budget(
    path: Path, *, config: Mapping[str, Any], common_keys: int,
) -> None:
    try:
        budget = pd.read_csv(path)
    except Exception as exc:
        raise DevelopmentControlsGateError("Stage-09b architecture budget is unreadable") from exc
    expected_columns = [
        "arm_id", "family", "feature_set", "variables", "variable_count",
        "seed_count", "seeds", "trainable_parameters",
        "thermoroute_full_reference_parameters",
        "parameter_difference_from_full_thermoroute",
        "parameter_ratio_to_full_thermoroute",
        "matched_within_2pct_of_full_thermoroute", "context_length", "horizons",
        "optimizer", "learning_rate", "weight_decay", "batch_size", "max_epochs",
        "early_stopping_patience", "selection_metric", "station_sampling",
        "train_examples_per_epoch", "maximum_optimizer_steps_per_seed",
        "architecture_candidates_in_this_entrypoint", "architecture_configuration",
        "mlp_hidden_dim", "mlp_depth", "tcn_channels", "tcn_blocks",
        "tcn_kernel_size", "thermoroute_d_model",
        "historical_tuning_budget_equalized", "training_device", "evidence_role",
    ]
    if list(budget.columns) != expected_columns or len(budget) != len(STAGE09B_ARMS):
        raise DevelopmentControlsGateError("Stage-09b architecture budget schema changed")
    if list(budget["arm_id"].astype(str)) != [str(arm["arm_id"]) for arm in STAGE09B_ARMS]:
        raise DevelopmentControlsGateError("Stage-09b architecture budget omits an arm")
    train_examples = pd.to_numeric(budget["train_examples_per_epoch"], errors="coerce")
    if train_examples.isna().any() or train_examples.nunique() != 1 or train_examples.iloc[0] < 1:
        raise DevelopmentControlsGateError("Stage-09b training-example budget changed")
    expected_steps = math.ceil(int(train_examples.iloc[0]) / STAGE09B_TRAIN_CONFIG["batch_size"])
    expected_steps *= STAGE09B_TRAIN_CONFIG["max_epochs"]
    for arm, row in zip(STAGE09B_ARMS, budget.to_dict(orient="records"), strict=True):
        arm_id = str(arm["arm_id"])
        parameters = STAGE09B_PARAMETER_COUNTS[arm_id]
        if (
            row["family"] != arm["family"]
            or row["feature_set"] != arm["feature_set"]
            or row["variables"] != "+".join(arm["variables"])
            or int(row["variable_count"]) != len(arm["variables"])
            or int(row["seed_count"]) != len(arm["seeds"])
            or str(row["seeds"]) != ",".join(str(seed) for seed in arm["seeds"])
            or int(row["trainable_parameters"]) != parameters
            or int(row["thermoroute_full_reference_parameters"])
            != STAGE09B_REFERENCE_PARAMETERS
            or int(row["parameter_difference_from_full_thermoroute"])
            != parameters - STAGE09B_REFERENCE_PARAMETERS
            or not np.isclose(
                float(row["parameter_ratio_to_full_thermoroute"]),
                parameters / STAGE09B_REFERENCE_PARAMETERS,
                rtol=1e-12,
                atol=1e-12,
            )
            or int(row["context_length"]) != C.CONTEXT_LENGTH
            or str(row["horizons"]) != ",".join(str(value) for value in C.HORIZONS)
            or row["optimizer"] != "torch.optim.AdamW"
            or float(row["learning_rate"]) != STAGE09B_TRAIN_CONFIG["lr"]
            or float(row["weight_decay"]) != STAGE09B_TRAIN_CONFIG["weight_decay"]
            or int(row["batch_size"]) != STAGE09B_TRAIN_CONFIG["batch_size"]
            or int(row["max_epochs"]) != STAGE09B_TRAIN_CONFIG["max_epochs"]
            or int(row["early_stopping_patience"]) != STAGE09B_TRAIN_CONFIG["patience"]
            or row["selection_metric"] != "station_macro_rmse"
            or row["station_sampling"] != "equal_station_fixed_size_bootstrap"
            or int(row["maximum_optimizer_steps_per_seed"]) != expected_steps
            or int(row["architecture_candidates_in_this_entrypoint"]) != 1
            or json.loads(str(row["architecture_configuration"]))
            != config["architecture_templates"][arm_id]
            or bool(row["historical_tuning_budget_equalized"]) is not False
            or row["training_device"] != "cpu"
            or row["evidence_role"] != "development_only_exploratory"
        ):
            raise DevelopmentControlsGateError("Stage-09b architecture/optimizer budget changed")
    if common_keys < 1:
        raise DevelopmentControlsGateError("Stage-09b common-key budget is empty")


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
    }


def _validate_combined_predictions(
    path: Path,
    *,
    reference: pd.DataFrame,
    expected_members: Sequence[tuple[str, int]],
) -> None:
    columns = ["model", "seed", "split", *FORECAST_KEY, "y_true"]
    try:
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow.names != R.PRED_COLS:
            raise DevelopmentControlsGateError("Stage-09b combined prediction schema changed")
        batches = parquet.iter_batches(columns=columns, batch_size=65_536)
        member_index = 0
        current_member: tuple[str, int] | None = None
        current_frames: list[pd.DataFrame] = []

        def finish_member() -> None:
            nonlocal member_index, current_member, current_frames
            if current_member is None:
                return
            if member_index >= len(expected_members) or current_member != expected_members[member_index]:
                raise DevelopmentControlsGateError("Stage-09b combined member order changed")
            frame = pd.concat(current_frames, ignore_index=True)
            normalized = _normalised_key_truth(frame[["split", *FORECAST_KEY, "y_true"]])
            keys = ["split", *FORECAST_KEY]
            if (
                not normalized[keys].equals(reference[keys])
                or not targets_match_at_model_precision(
                    normalized["y_true"], reference["y_true"]
                )
            ):
                raise DevelopmentControlsGateError(
                    "Stage-09b combined predictions differ from member closure"
                )
            member_index += 1
            current_member = None
            current_frames = []

        for batch in batches:
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
    if member_index != len(expected_members):
        raise DevelopmentControlsGateError("Stage-09b combined predictions omit members")


def build_stage09b_completion_receipt(
    *,
    root: str | Path,
    run_id: str,
    run_manifest: str | Path,
    frozen_panel_spec: str | Path,
    panel: str | Path,
    registry: str | Path,
    predictor_bridge: str | Path,
    member_paths: Mapping[tuple[str, int], str | Path],
    predictions: str | Path,
    architecture_budget: str | Path,
    report: str | Path,
    matrix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a content-bound candidate receipt after every output exists."""
    root = Path(root).resolve()
    manifest = _load_json(Path(run_manifest), label="Stage-09b run manifest")
    identity = manifest.get("identity")
    config = manifest.get("resolved_config")
    if not isinstance(identity, dict) or not isinstance(config, dict):
        raise DevelopmentControlsGateError("Stage-09b run manifest is malformed")
    if identity.get("run_id") != run_id:
        raise DevelopmentControlsGateError("Stage-09b receipt run id differs from manifest")
    expected = expected_stage09b_members()
    if set(member_paths) != set(expected) or len(member_paths) != len(expected):
        raise DevelopmentControlsGateError("Stage-09b receipt requires exactly 31 members")
    predictions = Path(predictions).resolve()
    architecture_budget = Path(architecture_budget).resolve()
    report = Path(report).resolve()
    document: dict[str, Any] = {
        "format": STAGE09B_COMPLETION_FORMAT,
        "status": STAGE09B_COMPLETION_STATUS,
        "stage": STAGE09B_STAGE,
        "run_id": run_id,
        "run_identity": identity,
        "formal_configuration": config,
        "matrix_audit": json.loads(
            json.dumps(dict(matrix_audit), sort_keys=True, allow_nan=False)
        ),
        "member_registry": [
            {
                "arm_id": arm_id,
                "seed": seed,
                "prediction": _file_binding(root, member_paths[(arm_id, seed)]),
                "prediction_sidecar": _file_binding(
                    root, sidecar_path(member_paths[(arm_id, seed)])
                ),
            }
            for arm_id, seed in expected
        ],
        "artifacts": {
            "run_manifest": _file_binding(root, run_manifest),
            "frozen_panel_spec": _file_binding(root, frozen_panel_spec),
            "panel": _file_binding(root, panel),
            "registry": _file_binding(root, registry),
            "predictor_bridge": _file_binding(root, predictor_bridge),
            "predictions": _file_binding(root, predictions),
            "prediction_sidecar": _file_binding(root, sidecar_path(predictions)),
            "architecture_budget": _file_binding(root, architecture_budget),
            "architecture_budget_sidecar": _file_binding(
                root, sidecar_path(architecture_budget)
            ),
            "report": _file_binding(root, report),
            "report_sidecar": _file_binding(root, sidecar_path(report)),
        },
        "post_2020_outcomes_requested_or_read": False,
    }
    document["receipt_self_sha256"] = sha256_json(document)
    return document


def validate_stage09b_completion_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path,
    document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deeply validate a published or not-yet-published Stage-09b receipt."""
    root = Path(root).resolve()
    receipt_path = Path(receipt_path).resolve()
    if receipt_path != root and root not in receipt_path.parents:
        raise DevelopmentControlsGateError("Stage-09b receipt escapes repository")
    receipt = (
        _load_json(receipt_path, label="Stage-09b completion receipt")
        if document is None else dict(document)
    )
    expected_keys = {
        "format", "status", "stage", "run_id", "run_identity",
        "formal_configuration", "matrix_audit", "member_registry", "artifacts",
        "post_2020_outcomes_requested_or_read", "receipt_self_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise DevelopmentControlsGateError("Stage-09b receipt schema is not exact")
    stable = {key: value for key, value in receipt.items() if key != "receipt_self_sha256"}
    if receipt.get("receipt_self_sha256") != sha256_json(stable):
        raise DevelopmentControlsGateError("Stage-09b receipt self hash changed")
    run_id = receipt.get("run_id")
    if (
        receipt.get("format") != STAGE09B_COMPLETION_FORMAT
        or receipt.get("status") != STAGE09B_COMPLETION_STATUS
        or receipt.get("stage") != STAGE09B_STAGE
        or not isinstance(run_id, str)
        or not run_id
        or receipt.get("post_2020_outcomes_requested_or_read") is not False
    ):
        raise DevelopmentControlsGateError("Stage-09b receipt is not a formal PASS")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        STAGE09B_FINAL_ARTIFACTS
    ):
        raise DevelopmentControlsGateError("Stage-09b final artifact registry is incomplete")
    paths = {
        label: _validated_binding(root, artifacts[label], label=f"Stage-09b {label}")
        for label in STAGE09B_FINAL_ARTIFACTS
    }
    identity, config = _validate_run_manifest(
        paths["run_manifest"], root=root, run_id=run_id
    )
    if receipt.get("run_identity") != identity or receipt.get("formal_configuration") != config:
        raise DevelopmentControlsGateError("Stage-09b receipt run identity/configuration changed")
    if artifacts["prediction_sidecar"] != _file_binding(
        root, sidecar_path(paths["predictions"])
    ) or artifacts["architecture_budget_sidecar"] != _file_binding(
        root, sidecar_path(paths["architecture_budget"])
    ) or artifacts["report_sidecar"] != _file_binding(root, sidecar_path(paths["report"])):
        raise DevelopmentControlsGateError("Stage-09b final sidecar alignment changed")
    bridge_binding = _file_binding(root, paths["predictor_bridge"])
    if config.get("development_predictor_bridge") != bridge_binding:
        raise DevelopmentControlsGateError("Stage-09b run config binds another predictor bridge")
    allowed_sites = _validate_data_contract(root=root, paths=paths, identity=identity)
    member_registry = receipt.get("member_registry")
    if not isinstance(member_registry, list) or len(member_registry) != 31:
        raise DevelopmentControlsGateError("Stage-09b receipt does not contain 31 members")
    parents = {
        "frozen_panel": identity["panel_sha256"],
        "frozen_station_registry": identity["registry_sha256"],
        "development_predictor_bridge": bridge_binding["sha256"],
    }
    reference, member_counts = _validate_member_predictions(
        root=root,
        member_registry=member_registry,
        identity=identity,
        config=config,
        expected_parents=parents,
        allowed_sites=allowed_sites,
    )
    audit = receipt.get("matrix_audit")
    expected_audit_keys = {
        "expected_members", "prediction_rows", "common_forecast_keys",
        "splits", "reference_member",
    }
    expected_members = expected_stage09b_members()
    if (
        not isinstance(audit, Mapping)
        or set(audit) != expected_audit_keys
        or audit.get("expected_members") != 31
        or audit.get("common_forecast_keys") != len(reference)
        or audit.get("prediction_rows") != len(reference) * 31
        or audit.get("prediction_rows") != sum(member_counts.values())
        or audit.get("splits") != ["calib", "test", "val"]
        or audit.get("reference_member") != f"{expected_members[0][0]}/seed{expected_members[0][1]}"
    ):
        raise DevelopmentControlsGateError("Stage-09b matrix audit is incomplete or stale")
    final_parents = {
        **parents,
        **{
            f"arm::{entry['arm_id']}::seed{entry['seed']}": entry["prediction"]["sha256"]
            for entry in member_registry
        },
    }
    run_identity = RunIdentity(**identity)
    final_specs = (
        (
            paths["predictions"], STAGE09B_FINAL_PREDICTION_KIND,
            R.PREDICTION_SCHEMA_VERSION, "combined_predictions",
        ),
        (paths["architecture_budget"], "development_controls_budget", "text/csv", "architecture_budget"),
        (paths["report"], "development_controls_report", "text/markdown", "report"),
    )
    for path, kind, schema, role in final_specs:
        try:
            metadata = validate_artifact_sidecar(
                path, identity=run_identity, schema=schema, kind=kind
            )
        except (OSError, ValueError) as exc:
            raise DevelopmentControlsGateError("Stage-09b final sidecar is invalid") from exc
        if (
            metadata["parents"] != dict(sorted(final_parents.items()))
            or metadata["extra"] != _expected_final_extra(audit, role=role)
        ):
            raise DevelopmentControlsGateError("Stage-09b final artifact closure changed")
    _validate_budget(
        paths["architecture_budget"],
        config=config,
        common_keys=len(reference),
    )
    try:
        report = paths["report"].read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DevelopmentControlsGateError("Stage-09b report is unreadable") from exc
    required_report_text = (
        f"Run ID: `{run_id}`",
        "Status: **COMPLETE DEVELOPMENT MATRIX**",
        "not a blind or confirmatory test",
        f"Exact member count: {len(expected_members)}",
        f"Common forecast keys per member:\n{len(reference)}",
        "historical_tuning_budget_equalized",
    )
    if any(text not in report for text in required_report_text):
        raise DevelopmentControlsGateError("Stage-09b report is incomplete")
    _validate_combined_predictions(
        paths["predictions"], reference=reference, expected_members=expected_members
    )
    return receipt


def write_stage09b_completion_receipt(
    path: str | Path, document: Mapping[str, Any],
) -> Path:
    """Atomically publish a prevalidated receipt as the transaction's last write."""
    stable = {key: value for key, value in document.items() if key != "receipt_self_sha256"}
    if document.get("receipt_self_sha256") != sha256_json(stable):
        raise DevelopmentControlsGateError("Stage-09b receipt self hash is invalid")
    destination = Path(path)
    atomic_write_json(destination, dict(document))
    return destination


def publish_stage09b_completion_receipt(
    receipt_path: str | Path,
    document: Mapping[str, Any],
    *,
    root: str | Path,
) -> Path:
    """Deep-validate the full closure, then atomically publish its PASS receipt."""
    validate_stage09b_completion_receipt(
        receipt_path, root=root, document=document
    )
    destination = write_stage09b_completion_receipt(receipt_path, document)
    validate_stage09b_completion_receipt(destination, root=root)
    return destination


def stage09b_completion_gate_binding(
    receipt_path: str | Path, *, root: str | Path,
) -> dict[str, str]:
    """Validate Stage-09b and return the exact path+SHA frozen downstream."""
    validate_stage09b_completion_receipt(receipt_path, root=root)
    return _file_binding(Path(root).resolve(), receipt_path)
